#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.report_pa_ict_liquidity_features import LiquidityEvent, scan_events  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_pa_ict_shadow_quality_overlay import (  # noqa: E402
    add_strategy_trade_fields,
    apply_best_pressure_params,
    load_best_params,
    promoted_fixed_params,
    promoted_shadow_params,
)
from scripts.scan_shadow_on_fixed_high_leverage import replay_shadow_events  # noqa: E402
from strategy.scalp_robust_v2_core import Candle, dataframe_to_candles  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "htf_context" / "htf_pa_ict_context_report.json"
DEFAULT_DATA_15M = ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"
DEFAULT_DATA_4H = ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report 4H/1D PA/ICT context alignment for promoted high-leverage shadow events."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--best-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument("--stdout", action="store_true")

    parser.add_argument("--swing-n", type=int, default=2)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--min-body-atr", type=float, default=0.7)
    parser.add_argument("--min-range-atr", type=float, default=1.1)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.05)
    parser.add_argument("--target-rr", type=float, default=2.0)
    parser.add_argument("--require-mss", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--h4-swing-lookback", type=int, default=30)
    parser.add_argument("--h4-liquidity-lookback-bars", type=int, default=180)
    parser.add_argument("--h4-mss-lookahead-bars", type=int, default=12)
    parser.add_argument("--h4-fvg-lookback-bars", type=int, default=6)
    parser.add_argument("--h4-entry-lookahead-bars", type=int, default=18)
    parser.add_argument("--h4-outcome-lookahead-bars", type=int, default=36)
    parser.add_argument("--h4-context-ttl-bars", type=int, default=42)

    parser.add_argument("--d1-swing-lookback", type=int, default=20)
    parser.add_argument("--d1-liquidity-lookback-bars", type=int, default=90)
    parser.add_argument("--d1-mss-lookahead-bars", type=int, default=5)
    parser.add_argument("--d1-fvg-lookback-bars", type=int, default=4)
    parser.add_argument("--d1-entry-lookahead-bars", type=int, default=10)
    parser.add_argument("--d1-outcome-lookahead-bars", type=int, default=20)
    parser.add_argument("--d1-context-ttl-bars", type=int, default=14)
    return parser.parse_args()


def ts_text(candle: Candle) -> str:
    return datetime.fromtimestamp(candle.ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_feather(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    elif "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        raise ValueError(f"Unsupported dataframe format: {path}")
    return df.sort_values("date").reset_index(drop=True)


def daily_from_4h(df4: pd.DataFrame) -> pd.DataFrame:
    df = df4.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    daily = (
        df.set_index("date")
        .resample("1D", label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )
    return daily


def namespace_for(args: argparse.Namespace, timeframe: str) -> argparse.Namespace:
    if timeframe == "4h":
        return argparse.Namespace(
            swing_n=args.swing_n,
            swing_lookback=args.h4_swing_lookback,
            liquidity_lookback_bars=args.h4_liquidity_lookback_bars,
            mss_lookahead_bars=args.h4_mss_lookahead_bars,
            fvg_lookback_bars=args.h4_fvg_lookback_bars,
            entry_lookahead_bars=args.h4_entry_lookahead_bars,
            outcome_lookahead_bars=args.h4_outcome_lookahead_bars,
            atr_period=args.atr_period,
            min_body_atr=args.min_body_atr,
            min_range_atr=args.min_range_atr,
            stop_buffer_atr=args.stop_buffer_atr,
            target_rr=args.target_rr,
            allow_incomplete_tail=getattr(args, "allow_incomplete_tail", False),
        )
    return argparse.Namespace(
        swing_n=args.swing_n,
        swing_lookback=args.d1_swing_lookback,
        liquidity_lookback_bars=args.d1_liquidity_lookback_bars,
        mss_lookahead_bars=args.d1_mss_lookahead_bars,
        fvg_lookback_bars=args.d1_fvg_lookback_bars,
        entry_lookahead_bars=args.d1_entry_lookahead_bars,
        outcome_lookahead_bars=args.d1_outcome_lookahead_bars,
        atr_period=args.atr_period,
        min_body_atr=args.min_body_atr,
        min_range_atr=args.min_range_atr,
        stop_buffer_atr=args.stop_buffer_atr,
        target_rr=args.target_rr,
        allow_incomplete_tail=getattr(args, "allow_incomplete_tail", False),
    )


def event_anchor_idx(event: LiquidityEvent, require_mss: bool, entry_idx: int | None = None) -> int | None:
    if event.retest is not None and (entry_idx is None or int(event.retest.idx) <= entry_idx):
        return int(event.retest.idx)
    if event.mss_idx is not None and (entry_idx is None or int(event.mss_idx) <= entry_idx):
        return int(event.mss_idx)
    if require_mss:
        return None
    return int(event.sweep_idx)


def event_zone(event: LiquidityEvent) -> str:
    if event.retest is None:
        return "none"
    if event.retest.fvg_touched and event.retest.ote_touched:
        return "fvg_and_ote"
    if event.retest.fvg_touched:
        return "fvg_only"
    if event.retest.ote_touched:
        return "ote_only"
    return "none"


def active_context_for_entry(
    events: list[LiquidityEvent],
    candles: list[Candle],
    entry_time: pd.Timestamp,
    ttl_bars: int,
    *,
    trade_direction: str,
    require_mss: bool,
) -> dict[str, Any]:
    ts_values = [candle.ts for candle in candles]
    entry_ts = entry_time.timestamp()
    entry_idx = bisect.bisect_right(ts_values, entry_ts) - 1
    if entry_idx < 0:
        return {"state": "none", "alignment": "none"}

    best: tuple[int, LiquidityEvent] | None = None
    for event in events:
        anchor = event_anchor_idx(event, require_mss=require_mss, entry_idx=entry_idx)
        if anchor is None:
            continue
        age = entry_idx - anchor
        if 0 <= age <= ttl_bars and (best is None or anchor > best[0]):
            best = (anchor, event)

    if best is None:
        return {"state": "none", "alignment": "none", "entry_idx": entry_idx}

    anchor, event = best
    alignment = "aligned" if event.direction == trade_direction else "opposed"
    return {
        "state": "bullish" if event.direction == "BULL" else "bearish",
        "alignment": alignment,
        "entry_idx": entry_idx,
        "anchor_idx": anchor,
        "age_bars": entry_idx - anchor,
        "event_status": event.status,
        "event_zone": event_zone(event),
        "event_sweep_time": event.sweep_time,
        "event_mss_time": event.mss_time,
        "event_retest_time": event.retest.timestamp if event.retest else None,
        "event_sweep_distance_pct": round(float(event.sweep_distance_pct), 4),
    }


def profit_factor(returns: list[float]) -> float:
    gross_profit = sum(value for value in returns if value > 0)
    gross_loss = abs(sum(value for value in returns if value <= 0))
    return round(gross_profit / gross_loss, 3) if gross_loss > 0 else 0.0


def summarize_events(events: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    if not events:
        return {
            "trades": 0,
            "return_sum_pct": 0.0,
            "compounded_return_pct": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_return_pct": 0.0,
        }
    capital = initial_capital
    returns = [float(event.get("return", 0.0) or 0.0) for event in events]
    for value in returns:
        capital = max(0.0, capital * (1.0 + value))
    wins = sum(1 for value in returns if value > 0)
    return_sum = sum(returns) * 100.0
    return {
        "trades": len(events),
        "return_sum_pct": round(return_sum, 2),
        "compounded_return_pct": round((capital - initial_capital) / initial_capital * 100.0, 2),
        "win_rate": round(wins / len(events) * 100.0, 2),
        "profit_factor": profit_factor(returns),
        "avg_return_pct": round(return_sum / len(events), 4),
    }


def group_summary(events: list[dict[str, Any]], initial_capital: float, key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(str(event.get(key) or "none"), []).append(event)
    return {name: summarize_events(items, initial_capital) for name, items in sorted(grouped.items())}


def window_start_times(events: list[dict[str, Any]]) -> dict[str, pd.Timestamp]:
    if not events:
        return {}
    exits = [pd.Timestamp(event["exit_time"]).tz_convert("UTC") for event in events]
    end = max(exits)
    return {
        "current_year": pd.Timestamp(f"{end.year}-01-01", tz="UTC"),
        "last_60d": end - pd.Timedelta(days=60),
        "last_30d": end - pd.Timedelta(days=30),
    }


def window_report(events: list[dict[str, Any]], initial_capital: float, start: pd.Timestamp) -> dict[str, Any]:
    selected = [
        event
        for event in events
        if pd.Timestamp(event["entry_time"]).tz_convert("UTC") >= start
    ]
    return {
        "baseline": summarize_events(selected, initial_capital),
        "by_h4_alignment": group_summary(selected, initial_capital, "htf_h4_alignment"),
        "by_d1_alignment": group_summary(selected, initial_capital, "htf_d1_alignment"),
        "by_combo_alignment": group_summary(selected, initial_capital, "htf_combo_alignment"),
    }


def build_promoted_shadow_events(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    best_payload = load_best_params(Path(args.best_params))
    payload = load_config_payload(Path(args.config))
    payload, pressure_params = apply_best_pressure_params(payload, args.pressure_params)
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=payload.get("regime_switcher_thresholds"),
    )
    metrics, engine = run_engine(payload, prepared, args.start_date)
    trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
    initial_capital = float(metrics.get("initial_capital", 1000.0) or 1000.0)
    fixed = expansion_overlay(trades, initial_capital, promoted_fixed_params(best_payload), include_events=True)
    fixed_events = add_strategy_trade_fields(fixed["events"], trades)
    shadow_params = promoted_shadow_params(best_payload)
    shadow = replay_shadow_events(
        fixed_events,
        initial_capital,
        daily_loss_stop_pct=float(shadow_params["daily_loss_stop_pct"]),
        equity_drawdown_stop_pct=float(shadow_params["equity_drawdown_stop_pct"]),
        consecutive_loss_stop=int(shadow_params["consecutive_loss_stop"]),
        equity_drawdown_cooldown_days=int(shadow_params["equity_drawdown_cooldown_days"]),
    )
    metadata = {
        "config": str(Path(args.config).resolve()),
        "best_params": str(Path(args.best_params).resolve()),
        "pressure_params_path": None if str(args.pressure_params).lower() == "none" else str(Path(args.pressure_params).resolve()),
        "pressure_params": pressure_params,
        "data": {
            "start": str(prepared.start),
            "end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
        },
        "engine": metrics,
        "fixed_structure_overlay": {key: value for key, value in fixed.items() if key != "events"},
        "shadow": {key: value for key, value in shadow.items() if key != "events"},
        "shadow_params": shadow_params,
    }
    return shadow["events"], metadata, initial_capital


def add_htf_contexts(
    events: list[dict[str, Any]],
    h4_events: list[LiquidityEvent],
    h4_candles: list[Candle],
    d1_events: list[LiquidityEvent],
    d1_candles: list[Candle],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events:
        entry_time = pd.Timestamp(event["entry_time"]).tz_convert("UTC")
        direction = str(event.get("direction") or "")
        h4 = active_context_for_entry(
            h4_events,
            h4_candles,
            entry_time,
            args.h4_context_ttl_bars,
            trade_direction=direction,
            require_mss=bool(args.require_mss),
        )
        d1 = active_context_for_entry(
            d1_events,
            d1_candles,
            entry_time,
            args.d1_context_ttl_bars,
            trade_direction=direction,
            require_mss=bool(args.require_mss),
        )
        enriched = dict(event)
        for prefix, context in (("h4", h4), ("d1", d1)):
            for key, value in context.items():
                enriched[f"htf_{prefix}_{key}"] = value
        enriched["htf_combo_alignment"] = f"{h4.get('alignment', 'none')}+{d1.get('alignment', 'none')}"
        enriched["htf_combo_state"] = f"{h4.get('state', 'none')}+{d1.get('state', 'none')}"
        out.append(enriched)
    return out


def main() -> None:
    args = parse_args()
    shadow_events, metadata, initial_capital = build_promoted_shadow_events(args)
    df4 = load_df(Path(args.data_4h))
    h4_candles = dataframe_to_candles(df4)
    d1_df = daily_from_4h(df4)
    d1_candles = dataframe_to_candles(d1_df)
    h4_events = scan_events(h4_candles, namespace_for(args, "4h"))
    d1_events = scan_events(d1_candles, namespace_for(args, "1d"))
    enriched_events = add_htf_contexts(shadow_events, h4_events, h4_candles, d1_events, d1_candles, args)
    windows = {
        name: window_report(enriched_events, initial_capital, start)
        for name, start in window_start_times(enriched_events).items()
    }

    report: dict[str, Any] = {
        "metadata": metadata,
        "parameters": {
            "start_date": args.start_date,
            "require_mss": args.require_mss,
            "h4": {
                "swing_lookback": args.h4_swing_lookback,
                "liquidity_lookback_bars": args.h4_liquidity_lookback_bars,
                "mss_lookahead_bars": args.h4_mss_lookahead_bars,
                "context_ttl_bars": args.h4_context_ttl_bars,
            },
            "d1": {
                "swing_lookback": args.d1_swing_lookback,
                "liquidity_lookback_bars": args.d1_liquidity_lookback_bars,
                "mss_lookahead_bars": args.d1_mss_lookahead_bars,
                "context_ttl_bars": args.d1_context_ttl_bars,
            },
        },
        "htf_event_counts": {
            "h4": len(h4_events),
            "d1": len(d1_events),
            "h4_by_status": {status: sum(1 for event in h4_events if event.status == status) for status in sorted({event.status for event in h4_events})},
            "d1_by_status": {status: sum(1 for event in d1_events if event.status == status) for status in sorted({event.status for event in d1_events})},
        },
        "baseline": summarize_events(enriched_events, initial_capital),
        "by_h4_alignment": group_summary(enriched_events, initial_capital, "htf_h4_alignment"),
        "by_d1_alignment": group_summary(enriched_events, initial_capital, "htf_d1_alignment"),
        "by_combo_alignment": group_summary(enriched_events, initial_capital, "htf_combo_alignment"),
        "by_h4_state": group_summary(enriched_events, initial_capital, "htf_h4_state"),
        "by_d1_state": group_summary(enriched_events, initial_capital, "htf_d1_state"),
        "windows": windows,
        "recent_examples": enriched_events[-10:],
    }
    if args.include_events:
        report["events"] = enriched_events
        report["h4_events"] = [asdict(event) for event in h4_events]
        report["d1_events"] = [asdict(event) for event in d1_events]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    compact = {
        "baseline": report["baseline"],
        "htf_event_counts": report["htf_event_counts"],
        "by_h4_alignment": report["by_h4_alignment"],
        "by_d1_alignment": report["by_d1_alignment"],
        "by_combo_alignment": report["by_combo_alignment"],
        "windows": {
            name: {
                "baseline": value["baseline"],
                "by_h4_alignment": value["by_h4_alignment"],
                "by_d1_alignment": value["by_d1_alignment"],
                "by_combo_alignment": value["by_combo_alignment"],
            }
            for name, value in windows.items()
        },
    }
    print(output)
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    if args.stdout:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
