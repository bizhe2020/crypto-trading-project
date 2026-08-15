"""GOOGL 高倍合约策略 — 信号层回测 + 杠杆敏感性。

用法:
    python scripts/backtest_googl_high_leverage.py \
        --prices-csv /path/to/价值投资project/data/prices.csv \
        --holdings-csv /path/to/价值投资project/data/berkshire_13f_holdings.csv
    python scripts/backtest_googl_high_leverage.py --sweep   # 参数扫描

报告内容:
    - 两段式信号层回测（2007-2026）：累计收益、maxDD、交易次数、conviction 段在市率
    - 杠杆应用后权益（offense/base/defense 三档）：注意杠杆回撤与单日爆仓风险
    - buy&hold GOOGL 对比
    - --sweep: pre/conv 段参数扫描（trailing、max_hold、入场窗）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.scan_googl_daily_signal import (  # noqa: E402
    CONV_FAST_WINDOW,
    CONV_MAX_HOLD_DAYS,
    CONV_TRAILING_DRAWDOWN_PCT,
    DEFAULT_INITIAL_CAPITAL,
    PRE_FAST_WINDOW,
    PRE_MAX_HOLD_DAYS,
    PRE_SLOW_WINDOW,
    PRE_TRAILING_DRAWDOWN_PCT,
    run_googl_signal,
    summarize,
)

LIQUIDATION_DAY_COUNT = 0


def apply_leverage(path: pd.DataFrame, leverage_map: dict[str, float], *, stop_loss_pct: float | None = None) -> pd.DataFrame:
    """把信号层日收益按 leverage_tier 放大，生成杠杆权益曲线。

    daily_return 是持有期的 GOOGL 日收益（已含换仓成本）。杠杆日收益 = 日收益 × 杠杆。
    单日亏损超过 1/杠杆 视为爆仓（clamp 到 -100%）。

    stop_loss_pct 非空时启用 stop-protected 模拟：单日杠杆亏损 clamp 到
    -stop_loss_pct × 杠杆，近似执行层 4% 交易所条件单止损。无止损时是
    恒定满杠杆的 worst-case（会真实清零）。
    """
    global LIQUIDATION_DAY_COUNT
    frame = path.copy()
    leverage = frame["leverage_tier"].map(leverage_map).fillna(0.0).astype(float)
    lev_daily = frame["daily_return"].astype(float) * leverage
    if stop_loss_pct and stop_loss_pct > 0:
        floor = -(float(stop_loss_pct) / 100.0) * leverage
        lev_daily = lev_daily.clip(lower=floor)
    lev_daily = lev_daily.clip(lower=-1.0)
    LIQUIDATION_DAY_COUNT = int((lev_daily <= -1.0).sum())
    frame["leveraged_daily"] = lev_daily
    frame["leveraged_capital"] = (1.0 + lev_daily).cumprod() * float(DEFAULT_INITIAL_CAPITAL)
    return frame


def stats_from_capital(capital: pd.Series) -> dict[str, Any]:
    if capital.empty:
        return {}
    peak = capital.cummax()
    mdd = ((peak - capital) / peak.replace(0, pd.NA) * 100.0).max(skipna=True) or 0.0
    ret = (capital.iloc[-1] / capital.iloc[0] - 1.0) * 100.0
    years = max((capital.index[-1] - capital.index[0]).days / 365.25, 0.01)
    cagr = ((capital.iloc[-1] / capital.iloc[0]) ** (1.0 / years) - 1.0) * 100.0 if capital.iloc[0] > 0 else 0.0
    return {"return_pct": round(float(ret), 2), "cagr_pct": round(float(cagr), 2), "max_dd_pct": round(float(mdd), 2)}


def yearly_returns(path: pd.DataFrame, capital_col: str = "capital") -> pd.DataFrame:
    cap = path.set_index("date")[capital_col].astype(float)
    yearly = cap.resample("YE").last()
    prev = cap.iloc[0]
    out: dict[str, float] = {}
    for ts, value in yearly.items():
        out[str(ts.year)] = round((value / prev - 1.0) * 100.0, 1)
        prev = value
    return pd.Series(out, name="ret_pct")


def run_backtest(
    prices_csv: Path,
    holdings_csv: Path,
    *,
    stop_loss_pct: float = 4.0,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = run_googl_signal(prices_csv, holdings_csv, **kwargs)
    path = payload["path"]
    signal_stats = summarize(path)

    # 无杠杆信号层
    signal_stats["cagr_pct"] = stats_from_capital(path.set_index("date")["capital"])["cagr_pct"]

    # buy&hold 对比（从同一 frame）
    frame = payload["frame"]
    bh = frame.set_index("date")["googl_close"]
    bh_cap = bh / bh.iloc[0] * float(DEFAULT_INITIAL_CAPITAL)
    bh_stats = stats_from_capital(bh_cap)

    # 杠杆应用（worst-case：无执行止损）
    leverage_map = {"offense": 15.0, "base": 10.0, "defense": 5.0, "flat": 0.0}
    lev_path = apply_leverage(path, leverage_map)
    lev_stats = stats_from_capital(lev_path.set_index("date")["leveraged_capital"])
    # stop-protected：近似执行层 stop_loss_pct 条件单止损
    lev_stop_path = apply_leverage(path, leverage_map, stop_loss_pct=stop_loss_pct)
    lev_stop_stats = stats_from_capital(lev_stop_path.set_index("date")["leveraged_capital"])

    return {
        "signal": signal_stats,
        "leveraged_15_10_5_worst_case": lev_stats,
        "leveraged_15_10_5_stop_protected": lev_stop_stats,
        "stop_loss_pct": stop_loss_pct,
        "liquidation_days_worst_case": LIQUIDATION_DAY_COUNT,
        "caveats": (
            "杠杆模拟是恒定满杠杆的 worst-case 上界：每个在市日都满额暴露，"
            "stop-protected 仅把单日亏损 clamp 到 stop_loss_pct×杠杆，但次日又满仓重暴露，"
            "不模拟执行层'止损后保持 flat 等信号重入场'的行为。真实高倍系统在 4h 执行层"
            "用 trailing stop + 交易所条件单 + shadow gate 保护，把单笔亏损限制在 ~4% 并暂停重入。"
            "因此杠杆列是理论下界，不是策略预期表现。4h 执行层回测（GOOGL-USDT-SWAP 2026-03 后）是第二阶段。"
        ),
        "buy_hold": bh_stats,
        "conviction_start": str(payload["conviction_start"]) if payload["conviction_start"] is not None else None,
        "yearly_signal": yearly_returns(path, "capital").to_dict(),
        "yearly_lev": yearly_returns(lev_path, "leveraged_capital").to_dict(),
        "conviction_period": _conviction_period_stats(path),
    }


def _conviction_period_stats(path: pd.DataFrame) -> dict[str, Any]:
    seg = path[path["berkshire_conviction"].astype(bool)]
    if seg.empty:
        return {}
    in_market = seg["position"].eq("GOOGL")
    cap = seg.set_index("date")["capital"]
    peak = cap.cummax()
    mdd = ((peak - cap) / peak.replace(0, pd.NA) * 100.0).max(skipna=True) or 0.0
    return {
        "start": str(seg["date"].iloc[0].date()),
        "end": str(seg["date"].iloc[-1].date()),
        "googl_hold_return_pct": round((cap.iloc[-1] / cap.iloc[0] - 1.0) * 100.0, 2),
        "max_dd_pct": round(float(mdd), 2),
        "in_market_days": int(in_market.sum()),
        "total_days": int(len(seg)),
        "in_market_pct": round(in_market.mean() * 100.0, 1),
    }


def sweep(prices_csv: Path, holdings_csv: Path) -> None:
    rows: list[dict[str, Any]] = []
    for pre_dd in (12.0, 15.0, 20.0):
        for pre_mh in (60, 90):
            for conv_mh in (0, 90):
                try:
                    payload = run_googl_signal(
                        prices_csv,
                        holdings_csv,
                        pre_trailing_drawdown_pct=pre_dd,
                        pre_max_hold_days=pre_mh,
                        conv_max_hold_days=conv_mh,
                    )
                    s = summarize(payload["path"])
                    rows.append(
                        {
                            "pre_dd": pre_dd,
                            "pre_mh": pre_mh,
                            "conv_mh": conv_mh,
                            "ret_pct": s["total_return_pct"],
                            "max_dd_pct": s["max_drawdown_pct"],
                            "trades": s["trades"],
                            "conv_in_pct": s["conviction_in_market_pct"],
                        }
                    )
                except Exception as exc:
                    rows.append({"pre_dd": pre_dd, "pre_mh": pre_mh, "conv_mh": conv_mh, "error": str(exc)})
    table = pd.DataFrame(rows).sort_values("ret_pct", ascending=False)
    print(table.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="GOOGL 高倍策略回测")
    parser.add_argument("--prices-csv", required=True)
    parser.add_argument("--holdings-csv", required=True)
    parser.add_argument("--sweep", action="store_true", help="运行参数扫描")
    parser.add_argument("--out", default=None, help="回测 JSON 输出路径")
    args = parser.parse_args()

    if args.sweep:
        sweep(Path(args.prices_csv), Path(args.holdings_csv))
        return

    report = run_backtest(Path(args.prices_csv), Path(args.holdings_csv))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print("backtest written to:", out_path)


if __name__ == "__main__":
    main()
