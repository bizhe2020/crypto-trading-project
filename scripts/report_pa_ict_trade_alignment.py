#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_pa_ict_liquidity_features import (  # noqa: E402
    LiquidityEvent,
    load_config,
    load_dataframe,
    scan_events,
    symbol_file_prefix,
)
from strategy.scalp_robust_v2_core import ScalpRobustEngine, dataframe_to_candles  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "var" / "pa_ict_liquidity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align current strategy trades with PA/ICT sweep-MSS-FVG/OTE feature events."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "config.paper.high-leverage-structure.json"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--analysis-start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--support-lookback-bars", type=int, default=40)
    parser.add_argument("--require-confirmed-retest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--swing-n", type=int, default=3)
    parser.add_argument("--swing-lookback", type=int, default=80)
    parser.add_argument("--liquidity-lookback-bars", type=int, default=192)
    parser.add_argument("--mss-lookahead-bars", type=int, default=24)
    parser.add_argument("--fvg-lookback-bars", type=int, default=8)
    parser.add_argument("--entry-lookahead-bars", type=int, default=40)
    parser.add_argument("--outcome-lookahead-bars", type=int, default=96)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--min-body-atr", type=float, default=0.7)
    parser.add_argument("--min-range-atr", type=float, default=1.1)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.05)
    parser.add_argument("--target-rr", type=float, default=2.0)
    parser.add_argument("--include-trades", action="store_true")
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args()


def parse_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def load_market_data(config: Any, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = parse_timestamp(args.start_date)
    end = parse_timestamp(args.end_date) if args.end_date else None
    data_root = Path(args.data_root or config.data_root)
    prefix = symbol_file_prefix(config.symbol)
    data_15m = data_root / f"{prefix}-{config.timeframe}-futures.feather"
    data_4h = data_root / f"{prefix}-{config.informative_timeframe}-futures.feather"
    df15 = load_dataframe(data_15m, start, end)
    df4 = load_dataframe(data_4h, pd.Timestamp("1970-01-01", tz="UTC"), end)
    return df15, df4


def run_strategy(config: Any, df15: pd.DataFrame, df4: pd.DataFrame, start_date: str) -> tuple[ScalpRobustEngine, dict[str, Any]]:
    engine = ScalpRobustEngine.from_candles(
        dataframe_to_candles(df4),
        dataframe_to_candles(df15),
        config.to_scalp_strategy_config(),
    )
    metrics = engine.run_backtest(start_date=start_date)
    return engine, metrics


def retest_zone_type(event: LiquidityEvent) -> str | None:
    retest = event.retest
    if retest is None:
        return None
    if retest.fvg_touched and retest.ote_touched:
        return "fvg_and_ote"
    if retest.fvg_touched:
        return "fvg_only"
    if retest.ote_touched:
        return "ote_only"
    return "none"


def event_support_idx(event: LiquidityEvent, require_confirmed_retest: bool) -> int | None:
    if require_confirmed_retest:
        if event.retest is None or not event.retest.confirmed:
            return None
        return event.retest.idx
    if event.retest is not None:
        return event.retest.idx
    return event.mss_idx


def match_event(
    trade: dict[str, Any],
    events_by_direction: dict[str, list[LiquidityEvent]],
    *,
    lookback_bars: int,
    require_confirmed_retest: bool,
) -> tuple[LiquidityEvent | None, int | None]:
    entry_idx = trade.get("entry_idx")
    direction = trade.get("direction")
    if entry_idx is None or direction not in events_by_direction:
        return None, None
    candidates: list[tuple[int, int, LiquidityEvent]] = []
    for event in events_by_direction[direction]:
        support_idx = event_support_idx(event, require_confirmed_retest)
        if support_idx is None:
            continue
        lag = int(entry_idx) - int(support_idx)
        if 0 <= lag <= lookback_bars:
            priority = {"fvg_and_ote": 0, "fvg_only": 1, "ote_only": 2, "none": 3}.get(retest_zone_type(event) or "none", 3)
            candidates.append((lag, priority, event))
    if not candidates:
        return None, None
    lag, _, event = sorted(candidates, key=lambda item: (item[0], item[1], -item[2].sweep_idx))[0]
    return event, lag


def profit_factor(trades: list[dict[str, Any]]) -> float:
    gross_profit = sum(float(trade.get("pnl", 0.0) or 0.0) for trade in trades if float(trade.get("pnl", 0.0) or 0.0) > 0)
    gross_loss = abs(sum(float(trade.get("pnl", 0.0) or 0.0) for trade in trades if float(trade.get("pnl", 0.0) or 0.0) <= 0))
    return round(gross_profit / gross_loss, 3) if gross_loss > 0 else 0.0


def summarize_trades(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "pnl_sum": 0.0,
            "profit_factor": 0.0,
            "avg_rr": 0.0,
            "avg_pnl_pct": 0.0,
        }
    wins = [trade for trade in trades if float(trade.get("pnl", 0.0) or 0.0) > 0]
    pnl_sum = sum(float(trade.get("pnl", 0.0) or 0.0) for trade in trades)
    rr_values = [float(trade.get("rr_ratio", 0.0) or 0.0) for trade in trades]
    pnl_pct_values = [float(trade.get("pnl_pct", 0.0) or 0.0) for trade in trades]
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(trades) - len(wins),
        "win_rate": round(len(wins) / len(trades) * 100.0, 2),
        "pnl_sum": round(pnl_sum, 2),
        "profit_factor": profit_factor(trades),
        "avg_rr": round(sum(rr_values) / len(rr_values), 3) if rr_values else 0.0,
        "avg_pnl_pct": round(sum(pnl_pct_values) / len(pnl_pct_values) * 100.0, 3) if pnl_pct_values else 0.0,
    }


def group_summary(trades: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        grouped.setdefault(str(trade.get(key) or "none"), []).append(trade)
    return {name: summarize_trades(items) for name, items in sorted(grouped.items())}


def compact_trade(trade: dict[str, Any], event: LiquidityEvent | None, lag: int | None) -> dict[str, Any]:
    entry_idx = trade.get("entry_idx")
    pressure_target_update_idx = trade.get("pressure_target_update_idx")
    pressure_touch_lock_update_idx = trade.get("pressure_touch_lock_update_idx")
    entry_price = float(trade.get("entry_price", 0.0) or 0.0)
    initial_stop = trade.get("initial_stop_price")
    pressure_target_level = trade.get("pressure_target_level")
    stop_distance_pct = None
    if entry_price > 0 and initial_stop is not None:
        stop_distance_pct = abs(entry_price - float(initial_stop)) / entry_price * 100.0
    pressure_target_distance_pct = None
    if entry_price > 0 and pressure_target_level is not None:
        level = float(pressure_target_level)
        if trade.get("direction") == "BEAR":
            pressure_target_distance_pct = (entry_price - level) / entry_price * 100.0
        else:
            pressure_target_distance_pct = (level - entry_price) / entry_price * 100.0

    payload = {
        "entry_time": trade.get("entry_time"),
        "exit_time": trade.get("exit_time"),
        "direction": trade.get("direction"),
        "pnl": trade.get("pnl"),
        "pnl_pct": trade.get("pnl_pct"),
        "rr_ratio": trade.get("rr_ratio"),
        "exit_reason": trade.get("exit_reason"),
        "entry_idx": trade.get("entry_idx"),
        "exit_idx": trade.get("exit_idx"),
        "regime_label": trade.get("regime_label"),
        "risk_regime": trade.get("risk_regime"),
        "trail_style": trade.get("trail_style"),
        "time_based_trailing_enabled": trade.get("time_based_trailing_enabled"),
        "auto_tit_reason": trade.get("auto_tit_reason"),
        "initial_stop_price": trade.get("initial_stop_price"),
        "stop_distance_pct": round(stop_distance_pct, 4) if stop_distance_pct is not None else None,
        "pressure_target_applied": trade.get("pressure_target_applied"),
        "pressure_target_source": trade.get("pressure_target_source"),
        "pressure_target_level": trade.get("pressure_target_level"),
        "pressure_target_rr": trade.get("pressure_target_rr"),
        "pressure_target_min_rr": trade.get("pressure_target_min_rr"),
        "pressure_target_dynamic_reason": trade.get("pressure_target_dynamic_reason"),
        "pressure_target_distance_pct": round(pressure_target_distance_pct, 4)
        if pressure_target_distance_pct is not None
        else None,
        "pressure_target_update_lag_bars": int(pressure_target_update_idx) - int(entry_idx)
        if pressure_target_update_idx is not None and entry_idx is not None
        else None,
        "pressure_touch_lock_applied": trade.get("pressure_touch_lock_applied"),
        "pressure_touch_lock_source": trade.get("pressure_touch_lock_source"),
        "pressure_touch_lock_level": trade.get("pressure_touch_lock_level"),
        "pressure_touch_lock_rr": trade.get("pressure_touch_lock_rr"),
        "pressure_touch_lock_update_lag_bars": int(pressure_touch_lock_update_idx) - int(entry_idx)
        if pressure_touch_lock_update_idx is not None and entry_idx is not None
        else None,
        "time_stop_exit": trade.get("exit_reason") == "time_stop_exit",
        "pa_ict_supported": event is not None,
        "pa_ict_lag_bars": lag,
    }
    if event is not None:
        payload.update(
            {
                "pa_ict_sweep_time": event.sweep_time,
                "pa_ict_mss_time": event.mss_time,
                "pa_ict_retest_time": event.retest.timestamp if event.retest else None,
                "pa_ict_time_bucket": event.time_bucket,
                "pa_ict_zone_type": retest_zone_type(event),
                "pa_ict_status": event.status,
            }
        )
    return payload


def output_path_for(
    output_dir: Path,
    start_date: str,
    analysis_start: str | None,
    strict: bool,
    support_lookback_bars: int,
    require_confirmed_retest: bool,
) -> Path:
    suffix = "_strict" if strict else ""
    mode = "confirmed" if require_confirmed_retest else "mss_or_retest"
    analysis = analysis_start or start_date
    return output_dir / (
        f"pa_ict_trade_alignment_{start_date}_analysis_{analysis}"
        f"_lb{support_lookback_bars}_{mode}{suffix}.json"
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    df15, df4 = load_market_data(config, args)
    candles = dataframe_to_candles(df15)
    events = scan_events(candles, args)
    events_by_direction: dict[str, list[LiquidityEvent]] = {"BULL": [], "BEAR": []}
    for event in events:
        events_by_direction.setdefault(event.direction, []).append(event)

    engine, metrics = run_strategy(config, df15, df4, args.start_date)
    trades = [asdict(trade) for trade in engine.trades]
    analysis_start = parse_timestamp(args.analysis_start_date) if args.analysis_start_date else None
    if analysis_start is not None:
        trades = [
            trade
            for trade in trades
            if parse_timestamp(str(trade.get("entry_time"))) >= analysis_start
        ]

    aligned: list[dict[str, Any]] = []
    for trade in trades:
        event, lag = match_event(
            trade,
            events_by_direction,
            lookback_bars=args.support_lookback_bars,
            require_confirmed_retest=bool(args.require_confirmed_retest),
        )
        row = compact_trade(trade, event, lag)
        aligned.append(row)

    supported = [trade for trade in aligned if trade["pa_ict_supported"]]
    unsupported = [trade for trade in aligned if not trade["pa_ict_supported"]]
    strict = args.min_body_atr >= 1.0 or args.min_range_atr >= 1.6
    report = {
        "config": str(config_path),
        "data": {
            "rows_15m": len(df15),
            "rows_4h": len(df4),
            "start": str(df15["date"].iloc[0]) if not df15.empty else None,
            "end": str(df15["date"].iloc[-1]) if not df15.empty else None,
        },
        "parameters": {
            "start_date": args.start_date,
            "analysis_start_date": args.analysis_start_date,
            "support_lookback_bars": args.support_lookback_bars,
            "require_confirmed_retest": args.require_confirmed_retest,
            "min_body_atr": args.min_body_atr,
            "min_range_atr": args.min_range_atr,
            "target_rr": args.target_rr,
        },
        "backtest_metrics": {
            "total_return_pct": metrics.get("total_return_pct"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            "total_trades": metrics.get("total_trades"),
            "win_rate": metrics.get("win_rate"),
            "profit_factor": metrics.get("profit_factor"),
        },
        "alignment": {
            "all": summarize_trades(aligned),
            "supported": summarize_trades(supported),
            "unsupported": summarize_trades(unsupported),
            "support_rate_pct": round(len(supported) / len(aligned) * 100.0, 2) if aligned else 0.0,
            "by_zone_type": group_summary(supported, "pa_ict_zone_type"),
            "by_time_bucket": group_summary(supported, "pa_ict_time_bucket"),
            "by_direction": group_summary(aligned, "direction"),
            "by_exit_reason": group_summary(aligned, "exit_reason"),
            "by_regime_label": group_summary(aligned, "regime_label"),
            "by_risk_regime": group_summary(aligned, "risk_regime"),
            "by_trail_style": group_summary(aligned, "trail_style"),
            "by_pressure_target_applied": group_summary(aligned, "pressure_target_applied"),
            "by_pressure_target_source": group_summary(aligned, "pressure_target_source"),
            "by_pressure_touch_lock_applied": group_summary(aligned, "pressure_touch_lock_applied"),
            "by_time_based_trailing_enabled": group_summary(aligned, "time_based_trailing_enabled"),
            "by_time_stop_exit": group_summary(aligned, "time_stop_exit"),
        },
        "examples": {
            "supported_recent": supported[-10:],
            "unsupported_recent": unsupported[-10:],
        },
    }
    if args.include_trades:
        report["trades"] = aligned

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_path_for(
        output_dir,
        args.start_date,
        args.analysis_start_date,
        strict,
        args.support_lookback_bars,
        bool(args.require_confirmed_retest),
    )
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output_path)
    print(json.dumps(report["alignment"], ensure_ascii=False, indent=2))
    if args.stdout:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
