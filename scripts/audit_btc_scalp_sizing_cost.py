#!/usr/bin/env python3
"""BTC scalp 出场可达性 / 仓位（Kelly）/ 高杠杆强平缓冲 / 实盘成本 的 R 空间诊断。

输入（全部本地文件，只读）：
  --engine-events  引擎交易（audit_btc_scalp_entry_funnel.py --events-out 的 JSONL，含 net_r / mfe_r / stop_pct）
  --path-events    可选：live 语义路径回放的同格式 JSONL（含执行层门）
  --live-trades    trades_audit.csv（scripts/audit_btc_scalp_live_history.py 产出）
  --okx            okx_history.json（scripts/dump_okx_btc_history.py 产出）

输出：
  1. MFE 可达性表：曾到 1R/2R/3R/4R/5R 的比例，与实盘止盈倍数下的理论触达率；
  2. Kelly：每笔净 R 的均值 / 标准差、近似 Kelly（mean/var）与数值最优 Kelly（max E[log(1+fR)]），
     bootstrap 区间；半 Kelly；与实盘 risk_per_trade 对比；
  3. 高杠杆保护：止损宽度分布、>2% 止损占比、10/15/20x 下强平缓冲（= 100/杠杆 - 维持保证金率 - 止损%）；
  4. 实盘成本：每笔手续费 R（按交易所实际持仓量）、手续费占毛盈亏比例。

用法::

    python3 scripts/audit_btc_scalp_sizing_cost.py \\
        --engine-events var/tmp/btc_headroom/engine_trades_2023.jsonl \\
        --path-events var/tmp/btc_headroom/ab/path_base.jsonl \\
        --live-trades var/tokyo_audit/btc_scalp_20261013/trades_audit.csv \\
        --okx var/tokyo_audit/btc_scalp_20261013/okx_history.json \\
        --output var/tmp/btc_headroom/sizing_cost.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

BTC_SWAP_CONTRACT_BTC = 0.01  # OKX BTC-USDT-SWAP ctVal


# ---------------------------------------------------------------- 纯函数


def mfe_reach_table(mfe_r: Sequence[float], levels: Sequence[float] = (0.5, 1, 2, 3, 4, 5)) -> dict[str, float]:
    """每个 R 档位被 MFE 触及的比例（%）。"""
    values = np.asarray([v for v in mfe_r if v is not None and not math.isnan(float(v))], dtype=float)
    if len(values) == 0:
        return {f"{level:g}R": 0.0 for level in levels}
    return {f"{level:g}R": round(float((values >= level).mean() * 100.0), 1) for level in levels}


def kelly_fraction_approx(r_values: Sequence[float]) -> float | None:
    """小 edge 近似：f* ≈ mean / var（以「每 1R 占资金的比例」为单位）。"""
    values = np.asarray(r_values, dtype=float)
    if len(values) < 2:
        return None
    var = float(values.var(ddof=1))
    if var <= 0:
        return None
    return float(values.mean() / var)


def kelly_fraction_numeric(r_values: Sequence[float], f_max: float = 1.0, steps: int = 2001) -> float:
    """数值求解 argmax_f mean(log(1 + f * R))，f ∈ [0, f_max]，且要求 1 + f * min(R) > 0。

    R 是以「初始止损距离」为单位的每笔净收益，所以 f 就是每笔风险占资金的比例。
    期望为负时返回 0。
    """
    values = np.asarray(r_values, dtype=float)
    if len(values) == 0 or values.mean() <= 0:
        return 0.0
    worst = float(values.min())
    upper = f_max if worst >= 0 else min(f_max, 0.999 / abs(worst))
    grid = np.linspace(0.0, upper, int(steps))
    growth = np.array([np.mean(np.log1p(f * values)) for f in grid])
    return float(grid[int(np.argmax(growth))])


def log_growth_per_trade(r_values: Sequence[float], fraction: float) -> float:
    values = np.asarray(r_values, dtype=float)
    if len(values) == 0:
        return 0.0
    if 1.0 + fraction * float(values.min()) <= 0:
        return float("-inf")
    return float(np.mean(np.log1p(fraction * values)))


def bootstrap_kelly(r_values: Sequence[float], n_boot: int = 2000, seed: int = 20261013,
                    alpha: float = 0.10) -> tuple[float, float]:
    values = np.asarray(r_values, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(int(n_boot)):
        sample = values[rng.integers(0, len(values), size=len(values))]
        estimates.append(kelly_fraction_numeric(sample, steps=401))
    lo, hi = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return round(float(lo), 4), round(float(hi), 4)


def liquidation_buffer_pct(stop_pct: float, leverage: float, maintenance_margin_pct: float = 0.5) -> float:
    """逐仓近似：强平价距入场 = 1/杠杆 - 维持保证金率；缓冲 = 该距离 - 止损距离（单位 %）。"""
    return 100.0 / float(leverage) - float(maintenance_margin_pct) - float(stop_pct)


def fee_r(fee_usdt: float, contracts: float, entry: float, stop: float, ct_val: float = BTC_SWAP_CONTRACT_BTC) -> float | None:
    """手续费折算成 R：|fee| / (实际持仓 BTC × 初始止损距离)。"""
    risk = abs(float(contracts) * ct_val * (float(entry) - float(stop)))
    if risk <= 0:
        return None
    return abs(float(fee_usdt)) / risk


# ---------------------------------------------------------------- 汇总


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def r_block(events: list[dict[str, Any]], live_risk: Sequence[float]) -> dict[str, Any]:
    net = [float(e["net_r"]) for e in events if e.get("net_r") is not None]
    mfe = [float(e["mfe_r"]) for e in events if e.get("mfe_r") is not None]
    stops = [float(e["stop_pct"]) for e in events if e.get("stop_pct") is not None]
    exits = pd.Series([str(e.get("exit_reason") or "") for e in events]).value_counts().to_dict()
    full = kelly_fraction_numeric(net)
    out: dict[str, Any] = {
        "trades": len(net),
        "net_r_mean": round(float(np.mean(net)), 4),
        "net_r_std": round(float(np.std(net, ddof=1)), 4),
        "net_r_median": round(float(np.median(net)), 4),
        "exit_reasons": exits,
        "mfe_reach_pct": mfe_reach_table(mfe),
        "mfe_r_median": round(float(np.median(mfe)), 3),
        "kelly_approx": round(kelly_fraction_approx(net) or 0.0, 4),
        "kelly_numeric": round(full, 4),
        "kelly_numeric_ci90": bootstrap_kelly(net),
        "half_kelly": round(full / 2.0, 4),
        "log_growth_per_trade": {
            f"f={f:g}": round(log_growth_per_trade(net, f), 5) for f in sorted({*live_risk, round(full, 4), round(full / 2, 4)})
        },
        "stop_pct": {
            "median": round(float(np.median(stops)), 3),
            "p90": round(float(np.quantile(stops, 0.9)), 3),
            "max": round(float(np.max(stops)), 3),
            "share_gt_2pct": round(float((np.asarray(stops) > 2.0).mean() * 100.0), 1),
        },
        "liquidation_buffer_pct": {
            f"{lev}x": {
                "min": round(min(liquidation_buffer_pct(s, lev) for s in stops), 3),
                "share_lt_1.2": round(float(np.mean([liquidation_buffer_pct(s, lev) < 1.2 for s in stops]) * 100.0), 1),
            }
            for lev in (10, 15, 20)
        },
        # 实盘 notional = min(0.8 × 杠杆 × 资金, 资金 × risk / 止损%)：被 0.8×杠杆 上限截断的占比
        "notional_capped_share_pct": {
            f"risk={risk:g}": {
                f"{lev}x": round(float(np.mean([risk / (s / 100.0) > 0.8 * lev for s in stops]) * 100.0), 1)
                for lev in (10, 15, 20)
            }
            for risk in (0.02, 0.035, 0.05, 0.12)
        },
    }
    return out


def live_cost_block(trades_csv: Path, okx_json: Path) -> dict[str, Any]:
    trades = pd.read_csv(trades_csv, parse_dates=["open_ts", "close_ts"])
    payload = json.loads(okx_json.read_text())
    pos = pd.DataFrame(payload["positions_history"])
    for col in ("uTime", "cTime"):
        pos[col] = pd.to_datetime(pos[col].astype("int64"), unit="ms", utc=True)
    for col in ("closeTotalPos", "fee", "fundingFee", "pnl", "realizedPnl", "openAvgPx"):
        pos[col] = pd.to_numeric(pos[col], errors="coerce")
    rows = []
    for _, t in trades.iterrows():
        if not bool(t.get("okx_matched")) or pd.isna(t.get("stop")):
            continue
        delta = (pos["uTime"] - t["close_ts"]).abs()
        hit = pos.loc[delta.idxmin()]
        if delta.min() > pd.Timedelta(minutes=20):
            continue
        fr = fee_r(hit["fee"], hit["closeTotalPos"], hit["openAvgPx"], t["stop"])
        rows.append({
            "open_ts": str(t["open_ts"]), "fee_usdt": float(hit["fee"]), "funding_usdt": float(hit["fundingFee"]),
            "gross_pnl": float(hit["pnl"]), "net_realized": float(hit["realizedPnl"]), "fee_r": fr,
            "notional_usdt": float(hit["closeTotalPos"]) * BTC_SWAP_CONTRACT_BTC * float(hit["openAvgPx"]),
            "realized_R": float(t["realized_R"]) if not pd.isna(t["realized_R"]) else None,
            "mfe_R": float(t["mfe_R"]) if not pd.isna(t["mfe_R"]) else None,
        })
    df = pd.DataFrame(rows)
    gross = float(df["gross_pnl"].sum())
    fees = float(df["fee_usdt"].sum())
    return {
        "bot_trades_matched": int(len(df)),
        "fee_r_mean": round(float(df["fee_r"].mean()), 4),
        "fee_r_median": round(float(df["fee_r"].median()), 4),
        "fees_usdt": round(fees, 2),
        "funding_usdt": round(float(df["funding_usdt"].sum()), 2),
        "gross_pnl_usdt": round(gross, 2),
        "net_realized_usdt": round(float(df["net_realized"].sum()), 2),
        "fee_share_of_abs_gross_pct": round(abs(fees) / abs(gross) * 100.0, 1) if gross else None,
        "fee_bps_of_notional_roundtrip": round(abs(fees) / float(df["notional_usdt"].sum()) * 1e4, 2),
        "mfe_reach_pct": mfe_reach_table([v for v in df["mfe_R"] if v is not None]),
        "realized_R_sum": round(float(pd.to_numeric(df["realized_R"]).sum()), 3),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine-events", required=True)
    parser.add_argument("--path-events", default="")
    parser.add_argument("--live-trades", required=True)
    parser.add_argument("--okx", required=True)
    parser.add_argument("--live-risk", default="0.035,0.05,0.12", help="对比用的 risk_per_trade（逗号分隔）")
    parser.add_argument("--output", default="var/tmp/btc_headroom/sizing_cost.json")
    args = parser.parse_args()

    live_risk = [float(x) for x in args.live_risk.split(",") if x.strip()]
    engine_events = load_jsonl(Path(args.engine_events))
    result: dict[str, Any] = {
        "engine": r_block(engine_events, live_risk),
        "engine_since_2025": r_block([e for e in engine_events if e["entry_time"] >= "2025-01-01"], live_risk),
    }
    # 实盘按 regime 方向性风险下注（bull_strong 多头 12%），所以分 regime 看各自的 Kelly
    regimes = sorted({str(e.get("risk_regime") or "") for e in engine_events if e.get("risk_regime")})
    for regime in regimes:
        for direction in ("BULL", "BEAR"):
            subset = [e for e in engine_events if e.get("risk_regime") == regime and e.get("direction") == direction]
            if len(subset) >= 10:
                result[f"engine_{regime}_{direction.lower()}"] = r_block(subset, live_risk)
    if args.path_events:
        path_events = load_jsonl(Path(args.path_events))
        result["path"] = r_block(path_events, live_risk)
        result["path_since_2025"] = r_block([e for e in path_events if e["entry_time"] >= "2025-01-01"], live_risk)
    result["live"] = live_cost_block(Path(args.live_trades), Path(args.okx))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
    for name in [k for k in result if k != "live"]:
        block = result[name]
        print(f"== {name}: n={block['trades']} 净R均值={block['net_r_mean']} sd={block['net_r_std']} "
              f"Kelly(近似)={block['kelly_approx']} Kelly(数值)={block['kelly_numeric']} CI90={block['kelly_numeric_ci90']} "
              f"半Kelly={block['half_kelly']}")
        print(f"   MFE 触达%={block['mfe_reach_pct']} 中位 MFE={block['mfe_r_median']}R  出场={block['exit_reasons']}")
        print(f"   止损%={block['stop_pct']}  强平缓冲={block['liquidation_buffer_pct']}")
        print(f"   log 增长/笔={block['log_growth_per_trade']}")
    live = result["live"]
    print(f"== live: n={live['bot_trades_matched']} 手续费R 均值={live['fee_r_mean']} 中位={live['fee_r_median']} "
          f"手续费={live['fees_usdt']} 毛盈亏={live['gross_pnl_usdt']} 手续费/|毛|={live['fee_share_of_abs_gross_pct']}% "
          f"往返费率={live['fee_bps_of_notional_roundtrip']}bp  MFE 触达%={live['mfe_reach_pct']}")
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
