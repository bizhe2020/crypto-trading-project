#!/usr/bin/env python3
"""GOOGL-USDT-SWAP 4h 执行层回测（日线信号 + 4h 执行保护）。

这是日线信号层（scan_googl_daily_signal.py）之后的关键验证：日频模型看不到
日内止损，无法判断 10x 是否存活。本脚本在 4h bar 粒度模拟真实执行层：
4h trailing stop + fee/slippage×杠杆 + funding + shadow gate。

用法:
    python scripts/replay_googl_usdt_4h.py \\
        --config config/config.paper.googl-high-leverage-runtime.json \\
        --okx-4h data/okx/futures/GOOGL_USDT_USDT-4h-futures.feather
    python scripts/replay_googl_usdt_4h.py --sweep    # 杠杆结构扫描

模型（镜像 scripts/replay_qqq_usdt_10x.py，加入 shadow gate）:
    - 4h bar：日线信号 position==GOOGL → allow_long；按 leverage_tier 施加杠杆
    - 4h trailing stop：close 跌破持仓峰值×(1-stop_loss_pct) → 平仓
    - fee + slippage × 杠杆（进出各一次）
    - funding：8h 结算费率 × 杠杆（有 funding 数据时；无则跳过）
    - shadow gate：权益自峰值回撤 ≥ equity_dd_stop_pct → 冷却
      equity_dd_cooldown_bars 根 4h bar 禁入场；stop 离场后需 reentry_clear_bars
      根连续 allow bar 才允许重入

数据:
    GOOGL-USDT-SWAP 2026-03-04 上线。用户将捞取更久历史填充
    data/okx/futures/GOOGL_USDT_USDT-4h-futures.feather（列 date/open/high/low/close）
    或 CSV（ts,o,h,l,c 或 date,open,high,low,close）。文件缺失时给出明确提示。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "config" / "config.paper.googl-high-leverage-runtime.json"
DEFAULT_SIGNAL = ROOT / "var" / "runtime" / "googl" / "googl_daily_signal.csv"
DEFAULT_OKX_4H = ROOT / "data" / "okx" / "futures" / "GOOGL_USDT_USDT-4h-futures.feather"
DEFAULT_FUNDING = ROOT / "data" / "okx" / "futures" / "GOOGL_USDT_USDT-8h-funding_rate.feather"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "googl_usdt_4h_execution_replay.json"


def load_okx_4h(path: Path) -> pd.DataFrame:
    """Load GOOGL-USDT-SWAP 4h candles. 支持 feather（date/open/high/low/close）
    或 CSV（date,open,high,low,close / timestamp,o,h,l,c）。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"GOOGL 4h 数据不存在: {path}\n"
            "请先捞取 GOOGL-USDT-SWAP 历史 K 线（OKX marketHistoryCandles, bar=4H）写入该路径，"
            "列 date/open/high/low/close（feather）或 date,open,high,low,close（CSV）。"
        )
    if path.suffix in {".feather", ".ftr"}:
        df = pd.read_feather(path).copy()
    else:
        df = pd.read_csv(path)
    df = df.copy()
    if "date" not in df.columns:
        for src in ("datetime", "timestamp", "ts", "time"):
            if src in df.columns:
                if src in {"timestamp", "ts"}:
                    df["date"] = pd.to_datetime(df[src], unit="ms", utc=True)
                else:
                    df["date"] = pd.to_datetime(df[src], utc=True)
                break
        else:
            raise ValueError(f"4h 数据缺少 date/datetime/timestamp 列: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    rename = {"o": "open", "h": "high", "l": "low", "c": "close"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"4h 数据缺少 {col} 列: {list(df.columns)}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    return df


def load_funding(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_feather(path).copy() if path.suffix in {".feather", ".ftr"} else pd.read_csv(path)
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    rate_col = "fundingRate" if "fundingRate" in df.columns else (
        "funding_rate" if "funding_rate" in df.columns else "rate"
    )
    df["funding_rate_value"] = pd.to_numeric(df[rate_col], errors="coerce").fillna(0.0)
    return df[["date", "funding_rate_value"]].sort_values("date").reset_index(drop=True)


def attach_googl_daily_state(
    okx_4h: pd.DataFrame,
    signal_path: pd.DataFrame,
    *,
    trim_to_signal_end: bool = True,
) -> pd.DataFrame:
    """把日线 GOOGL 信号（position=GOOGL/FLAT + leverage_tier）附着到 4h bars。"""
    daily = signal_path[["date", "position", "leverage_tier", "target_leverage"]].copy()
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    daily = daily.sort_values("date").reset_index(drop=True)
    merged = pd.merge_asof(
        okx_4h.sort_values("date"),
        daily,
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    if trim_to_signal_end:
        merged = merged[merged["date"] <= daily["date"].max()].copy()
    merged["allow_long"] = merged["position"].eq("GOOGL")
    return merged.reset_index(drop=True)


def is_funding_settlement_bar(bar_date: Any, funding_event_time: Any) -> bool:
    if pd.isna(funding_event_time):
        return False
    bar_ts = pd.Timestamp(bar_date)
    event_ts = pd.Timestamp(funding_event_time)
    bar_ts = bar_ts.tz_localize("UTC") if bar_ts.tzinfo is None else bar_ts.tz_convert("UTC")
    event_ts = event_ts.tz_localize("UTC") if event_ts.tzinfo is None else event_ts.tz_convert("UTC")
    return bar_ts == event_ts


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0, pd.NA) * 100.0
    return float(dd.max(skipna=True) or 0.0)


def run_googl_4h_replay(
    bars: pd.DataFrame,
    funding: pd.DataFrame | None,
    *,
    leverage_tiers: dict[str, float],
    stop_loss_pct: float,
    taker_fee_rate: float,
    slippage_bps: float,
    initial_capital: float,
    equity_dd_stop_pct: float = 0.0,
    equity_dd_cooldown_bars: int = 0,
    reentry_rule: str = "clear",
    reentry_clear_bars: int = 0,
    include_funding: bool = True,
) -> dict[str, Any]:
    """4h 执行层回测主函数。

    equity_dd_stop_pct / equity_dd_cooldown_bars 对应 shadow gate：
    权益自峰值回撤 ≥ stop → 冷却 cooldown 根 4h bar 禁入场。
    reentry_rule=="clear" 时，stop 离场后需 reentry_clear_bars 根连续 allow 才重入。
    """
    merged = bars.copy()
    if funding is not None and include_funding:
        merged = pd.merge_asof(
            merged.sort_values("date"),
            funding.sort_values("date"),
            on="date",
            direction="backward",
            allow_exact_matches=True,
        )
        merged["funding_rate_value"] = merged["funding_rate_value"].fillna(0.0)
        merged["funding_event_time"] = merged["date"]
        funding_available = True
    else:
        merged["funding_rate_value"] = 0.0
        merged["funding_event_time"] = pd.NaT
        funding_available = False

    capital = float(initial_capital)
    equity_peak = capital
    holding = False
    stop_price = 0.0
    entry_price = 0.0
    entry_leverage = 0.0
    gate_cooldown_left = 0
    stopped_after_stop = False
    clear_streak = 0
    gate_events: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    current_trade: dict[str, Any] | None = None
    prev_allow = False

    per_side_cost = float(taker_fee_rate) + float(slippage_bps) / 10000.0

    for row in merged.itertuples(index=False):
        start_capital = capital
        allow_now = bool(row.allow_long)
        entered_today = False
        exited_today = False
        fee_cost = 0.0
        funding_cost = 0.0
        funding_settled = False
        stop_hit = False
        gate_blocked = False

        # --- shadow gate：cooldown / clear_streak 计数 ---
        if gate_cooldown_left > 0:
            gate_cooldown_left -= 1
        if allow_now:
            clear_streak += 1
        else:
            clear_streak = 0
            stopped_after_stop = False
            gate_cooldown_left = 0

        entry_allowed = (
            allow_now and not holding and gate_cooldown_left <= 0
            and not (stopped_after_stop and reentry_rule == "clear" and clear_streak < int(reentry_clear_bars))
        )
        if allow_now and not holding and not entry_allowed:
            gate_blocked = True
            gate_events.append(
                {
                    "date": str(pd.Timestamp(row.date)),
                    "event": "entry_blocked",
                    "reason": "gate_cooldown" if gate_cooldown_left > 0 else "reentry_clear_bars",
                }
            )

        # --- 信号翻 FLAT → 平仓 ---
        if holding and not allow_now:
            fee_cost = per_side_cost
            capital *= 1.0 - fee_cost * float(current_trade["leverage"]) if current_trade else 1.0
            holding = False
            exited_today = True
            if current_trade is not None:
                trades.append(
                    {
                        "entry_date": current_trade["entry_date"],
                        "exit_date": str(pd.Timestamp(row.date)),
                        "exit_reason": "signal_flat",
                        "leverage": float(current_trade["leverage"]),
                        "trade_return_pct": round((capital / float(current_trade["entry_capital"]) - 1.0) * 100.0, 2),
                    }
                )
            current_trade = None
            stop_price = 0.0
            entry_price = 0.0

        # --- 入场（信号允许 + 门未拦截） ---
        if entry_allowed and not holding and not prev_allow:
            tier = str(row.leverage_tier) if hasattr(row, "leverage_tier") else "base"
            lev = float(leverage_tiers.get(tier, 0.0))
            if lev > 0:
                fee_cost = per_side_cost
                capital *= 1.0 - fee_cost * lev
                holding = True
                entered_today = True
                entry_price = float(row.open)
                entry_leverage = lev
                stop_price = entry_price * (1.0 - float(stop_loss_pct) / 100.0)
                current_trade = {
                    "entry_date": str(pd.Timestamp(row.date)),
                    "entry_capital": capital,
                    "leverage": lev,
                }

        # --- 持仓中的 4h 路径 ---
        if holding:
            open_price = float(row.open)
            high_price = float(row.high)
            low_price = float(row.low)
            close_price = float(row.close)
            lev = entry_leverage
            if low_price <= stop_price:
                # 4h trailing stop 触发（交易所条件单兜底）
                stop_hit = True
                exit_price = stop_price
                bar_ret = exit_price / open_price - 1.0 if open_price > 0 else 0.0
                capital *= 1.0 + lev * bar_ret
                capital *= 1.0 - per_side_cost * lev
                holding = False
                exited_today = True
                stopped_after_stop = True
                if current_trade is not None:
                    trades.append(
                        {
                            "entry_date": current_trade["entry_date"],
                            "exit_date": str(pd.Timestamp(row.date)),
                            "exit_reason": "trailing_stop",
                            "leverage": lev,
                            "trade_return_pct": round((capital / float(current_trade["entry_capital"]) - 1.0) * 100.0, 2),
                        }
                    )
                current_trade = None
                stop_price = 0.0
                entry_price = 0.0
            else:
                bar_ret = close_price / open_price - 1.0 if open_price > 0 else 0.0
                capital *= 1.0 + lev * bar_ret
                if funding_available:
                    funding_settled = is_funding_settlement_bar(row.date, row.funding_event_time)
                    if funding_settled:
                        funding_cost = float(row.funding_rate_value) * lev
                        capital *= 1.0 - funding_cost
                stop_price = max(stop_price, close_price * (1.0 - float(stop_loss_pct) / 100.0))

        # --- shadow gate：权益回撤 → 冷却 ---
        equity_peak = max(equity_peak, capital)
        if equity_dd_stop_pct > 0 and equity_peak > 0:
            dd_pct = (equity_peak - capital) / equity_peak * 100.0
            if dd_pct >= equity_dd_stop_pct:
                if equity_dd_cooldown_bars > 0:
                    gate_cooldown_left = max(gate_cooldown_left, int(equity_dd_cooldown_bars))
                    gate_events.append(
                        {
                            "date": str(pd.Timestamp(row.date)),
                            "event": "equity_dd_gate",
                            "drawdown_pct": round(dd_pct, 2),
                            "cooldown_bars": int(equity_dd_cooldown_bars),
                        }
                    )
                equity_peak = capital  # 重置峰值，避免反复触发

        rows.append(
            {
                "date": pd.Timestamp(row.date),
                "holding": holding,
                "allow_long": allow_now,
                "entered_today": entered_today,
                "exited_today": exited_today,
                "stop_hit": stop_hit,
                "gate_blocked": gate_blocked,
                "gate_cooldown_left": int(gate_cooldown_left),
                "capital": float(capital),
                "equity_peak": float(equity_peak),
                "daily_return": capital / start_capital - 1.0 if start_capital > 0 else 0.0,
                "funding_rate_value": float(row.funding_rate_value),
                "funding_settled": bool(funding_settled),
                "fee_cost": float(fee_cost),
                "funding_cost": float(funding_cost),
            }
        )
        prev_allow = allow_now

    path = pd.DataFrame(rows)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_date", "exit_date", "exit_reason", "leverage", "trade_return_pct"]
    )
    final_cap = float(path.iloc[-1]["capital"]) if not path.empty else initial_capital
    total_ret = (final_cap / float(initial_capital) - 1.0) * 100.0
    years = max((path["date"].iloc[-1] - path["date"].iloc[0]).days / 365.25, 0.01) if not path.empty else 1.0
    cagr = (final_cap / float(initial_capital)) ** (1.0 / years) - 1.0 if final_cap > 0 else -1.0

    return {
        "summary": {
            "total_return_pct": round(total_ret, 2),
            "cagr_pct": round(cagr * 100.0, 2) if cagr > -1.0 else None,
            "max_drawdown_pct": round(max_drawdown_pct(path["capital"]), 2) if not path.empty else 0.0,
            "bars": int(len(path)),
            "invested_bars": int(path["holding"].sum()) if not path.empty else 0,
            "trades": int(len(trades_df)),
            "win_rate_pct": round(float((trades_df["trade_return_pct"] > 0).mean() * 100.0), 2) if not trades_df.empty else 0.0,
            "avg_trade_return_pct": round(float(trades_df["trade_return_pct"].mean()), 2) if not trades_df.empty else 0.0,
            "total_funding_cost_pct_est": round(float(path["funding_cost"].sum() * 100.0), 2) if not path.empty else 0.0,
            "gate_events": len(gate_events),
            "equity_dd_gate_events": int(sum(1 for e in gate_events if e["event"] == "equity_dd_gate")),
            "start": str(path["date"].iloc[0]) if not path.empty else None,
            "end": str(path["date"].iloc[-1]) if not path.empty else None,
        },
        "trades": trades_df.to_dict("records"),
        "gate_events": gate_events,
    }


def sweep(merged_bars: pd.DataFrame, funding: pd.DataFrame | None, config: dict[str, Any]) -> None:
    """杠杆结构扫描：找出可行区间。"""
    base = config.get("base_leverage", 10.0)
    offense = config.get("offense_leverage", 15.0)
    defense = config.get("defense_leverage", 5.0)
    print(f"=== 杠杆结构扫描（基准 base {base}x / offense {offense}x / defense {defense}x）===")
    rows: list[tuple[float, float, float, dict[str, float]]] = []
    for mult in (0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        tiers = {
            "offense": offense * mult,
            "base": base * mult,
            "defense": defense * mult,
            "flat": 0.0,
        }
        result = run_googl_4h_replay(
            merged_bars, funding,
            leverage_tiers=tiers,
            stop_loss_pct=float(config.get("stop_loss_pct", 4.0)),
            taker_fee_rate=float(config.get("taker_fee_rate", 0.0005)),
            slippage_bps=float(config.get("slippage_bps", 5.0)),
            initial_capital=float(config.get("initial_capital", 1000.0)),
            equity_dd_stop_pct=15.0,
            equity_dd_cooldown_bars=20,
            reentry_rule="clear",
            reentry_clear_bars=2,
        )
        s = result["summary"]
        rows.append((s["total_return_pct"], s["max_drawdown_pct"], mult, tiers))
    rows.sort(key=lambda x: x[0], reverse=True)
    print(f"{'倍率':>5} {'base':>5} {'offense':>7} {'收益%':>12} {'maxDD%':>8} {'交易':>5} {'胜率%':>6}")
    for ret, mdd, mult, tiers in rows:
        print(
            f"{mult:>5.2f} {tiers['base']:>5.1f} {tiers['offense']:>7.1f} "
            f"{ret:>12.1f} {mdd:>8.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="GOOGL-USDT-SWAP 4h 执行层回测")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--signal", default=str(DEFAULT_SIGNAL))
    parser.add_argument("--okx-4h", default=str(DEFAULT_OKX_4H))
    parser.add_argument("--funding", default=str(DEFAULT_FUNDING))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--leverage", type=float, default=None, help="覆盖全部杠杆档（单档）")
    parser.add_argument("--stop-loss-pct", type=float, default=None)
    parser.add_argument("--sweep", action="store_true", help="杠杆结构扫描")
    parser.add_argument("--no-funding", action="store_true", help="忽略 funding 成本")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"config 不存在: {config_path}")
    config = json.loads(config_path.read_text())

    bars = load_okx_4h(Path(args.okx_4h))
    signal_path = pd.read_csv(Path(args.signal))
    merged = attach_googl_daily_state(bars, signal_path)
    if merged.empty:
        raise SystemExit("4h 数据与日线信号无重叠时段。")
    funding = load_funding(Path(args.funding))

    if args.sweep:
        sweep(merged, funding, config)
        return

    if args.leverage:
        tiers = {"offense": args.leverage, "base": args.leverage, "defense": args.leverage, "flat": 0.0}
    else:
        tiers = {
            "offense": float(config.get("offense_leverage", 15.0)),
            "base": float(config.get("base_leverage", 10.0)),
            "defense": float(config.get("defense_leverage", 5.0)),
            "flat": 0.0,
        }
    result = run_googl_4h_replay(
        merged, funding,
        leverage_tiers=tiers,
        stop_loss_pct=float(args.stop_loss_pct if args.stop_loss_pct is not None else config.get("stop_loss_pct", 4.0)),
        taker_fee_rate=float(config.get("taker_fee_rate", 0.0005)),
        slippage_bps=float(config.get("slippage_bps", 5.0)),
        initial_capital=float(config.get("initial_capital", 1000.0)),
        equity_dd_stop_pct=15.0,
        equity_dd_cooldown_bars=20,
        reentry_rule="clear",
        reentry_clear_bars=2,
        include_funding=not args.no_funding,
    )
    payload = {
        "config": {
            "signal_source": str(Path(args.signal)),
            "frozen_label": config.get("frozen_label"),
            "leverage_tiers": tiers,
            "stop_loss_pct": float(args.stop_loss_pct if args.stop_loss_pct is not None else config.get("stop_loss_pct", 4.0)),
            "taker_fee_rate": float(config.get("taker_fee_rate", 0.0005)),
            "slippage_bps": float(config.get("slippage_bps", 5.0)),
            "shadow_gate": {
                "equity_dd_stop_pct": 15.0,
                "equity_dd_cooldown_bars": 20,
                "reentry_rule": "clear",
                "reentry_clear_bars": 2,
            },
        },
        **result,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("replay written to:", out)


if __name__ == "__main__":
    main()
