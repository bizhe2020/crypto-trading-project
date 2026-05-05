#!/usr/bin/env python3
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

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import _high_leverage_trade_diagnostics, load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.report_smc_trade_context import daily_candles_from_4h  # noqa: E402
from scripts.reproduce_reverse_short_overlay_candidates import clean_for_json, event_return_stats  # noqa: E402
from scripts.reproduce_smc_short_only_v1_10x import strategy_args as smc_short_v1_strategy_args  # noqa: E402
from scripts.research_reverse_short_from_failed_longs import (  # noqa: E402
    add_combo_deltas,
    add_standard_windows,
    build_combo_results,
    compact_combo_result,
    compact_result,
    event_stream_summary,
    replay_non_overlapping,
    selected_by,
    simulate_short_trade,
    standard_event_summary,
    standard_reverse_short_event,
    standard_sota_event,
)
from scripts.research_smc_standalone_v1 import (  # noqa: E402
    apply_max_open_positions,
    build_event_scan_args,
    scan_events,
    trade_rows_for_events,
)
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "stable_reverse_short_plus_smc_short_combo.json"


SMC_CASES: dict[str, dict[str, Any]] = {
    "v1_base_other_10x": {
        "target_rr": 2.0,
        "allowed_time_buckets": "other",
        "swing_n": 3,
        "min_body_atr": 0.7,
        "min_range_atr": 1.1,
        "entry_lookahead_bars": 40,
        "max_open_positions": 1,
        "max_mss_lag_bars": 15,
        "leverage": 10.0,
        "position_size_pct": 1.0,
    },
    "v1_aggressive_maxlag9_10x": {
        "target_rr": 2.0,
        "allowed_time_buckets": "other+asia_evening_ny+ny_am_killzone",
        "swing_n": 3,
        "min_body_atr": 0.7,
        "min_range_atr": 1.1,
        "entry_lookahead_bars": 40,
        "max_open_positions": 1,
        "max_mss_lag_bars": 9,
        "leverage": 10.0,
        "position_size_pct": 1.0,
    },
    "v2_medium_dispbody05_otherlag4_10x": {
        "target_rr": 2.0,
        "allowed_time_buckets": "other+asia_evening_ny+ny_am_killzone",
        "swing_n": 2,
        "min_body_atr": 0.7,
        "min_range_atr": 1.1,
        "entry_lookahead_bars": 40,
        "max_open_positions": 1,
        "max_mss_lag_bars": 15,
        "min_displacement_body_atr": 0.5,
        "other_min_mss_lag_bars": 4,
        "leverage": 10.0,
        "position_size_pct": 1.0,
    },
    "v3_lag4_9_10x": {
        "target_rr": 2.0,
        "allowed_time_buckets": "other+asia_evening_ny+ny_am_killzone",
        "swing_n": 2,
        "min_body_atr": 0.5,
        "min_range_atr": 0.9,
        "entry_lookahead_bars": 72,
        "max_open_positions": 1,
        "max_mss_lag_bars": 24,
        "min_displacement_body_atr": 0.3,
        "global_min_mss_lag_bars": 4,
        "global_max_mss_lag_bars": 9,
        "leverage": 10.0,
        "position_size_pct": 1.0,
    },
}


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine Stable reverse-short overlay with SMC short-only events in one single-slot stream.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--smc-cases", default="v1_base_other_10x,v1_aggressive_maxlag9_10x,v2_medium_dispbody05_otherlag4_10x,v3_lag4_9_10x")
    parser.add_argument("--smc-allocation-values", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--stable-allocation", type=float, default=1.0)
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=0)
    parser.add_argument("--sample-trades", type=int, default=20)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def normalize_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def roundtrip_cost_rate(taker_fee_rate: float, slippage_bps: float) -> float:
    return 2.0 * float(taker_fee_rate) + 2.0 * float(slippage_bps) / 10_000.0


def leveraged_net_return(
    *,
    signal_return_pct: float,
    leverage: float,
    position_size_pct: float,
    allocation: float = 1.0,
    taker_fee_rate: float,
    slippage_bps: float,
) -> dict[str, float]:
    gross_unit_return = float(signal_return_pct) / 100.0
    cost = roundtrip_cost_rate(taker_fee_rate, slippage_bps)
    net_unit_return = gross_unit_return - cost
    account_return = net_unit_return * float(leverage) * float(position_size_pct) * float(allocation)
    return {
        "gross_unit_return": gross_unit_return,
        "roundtrip_cost": cost,
        "net_unit_return": net_unit_return,
        "account_return": account_return,
        "gross_unit_return_pct": gross_unit_return * 100.0,
        "roundtrip_cost_pct": cost * 100.0,
        "net_unit_return_pct": net_unit_return * 100.0,
        "account_return_pct": account_return * 100.0,
    }


def overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return int(a["entry_idx"]) < int(b["exit_idx"]) and int(a["exit_idx"]) > int(b["entry_idx"])


def filtered_base_priority_overlay(
    base_events: list[dict[str, Any]],
    overlay_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    accepted: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for event in sorted(overlay_events, key=lambda item: (int(item["entry_idx"]), int(item["exit_idx"]))):
        if any(overlaps(event, base_event) for base_event in base_events):
            event_type = str(event.get("event_type") or "unknown")
            skipped[event_type] = skipped.get(event_type, 0) + 1
            continue
        accepted.append(event)
    return accepted, skipped


def replay_overlay_priority_single_slot(events: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    priority = {
        "stable_reverse_short": 0,
        "smc_short": 1,
        "sota_long": 2,
    }
    ordered = sorted(
        events,
        key=lambda item: (
            int(item.get("entry_idx", 0) or 0),
            priority.get(str(item.get("event_type") or ""), 9),
            int(item.get("exit_idx", 0) or 0),
        ),
    )
    accepted: list[dict[str, Any]] = []
    skipped_by_type: dict[str, int] = {}
    last_exit_idx = -1
    for event in ordered:
        entry_idx = int(event.get("entry_idx", 0) or 0)
        if entry_idx < last_exit_idx:
            event_type = str(event.get("event_type") or "unknown")
            skipped_by_type[event_type] = skipped_by_type.get(event_type, 0) + 1
            continue
        accepted.append(event)
        last_exit_idx = max(last_exit_idx, int(event.get("exit_idx", entry_idx) or entry_idx))
    result = standard_event_summary(accepted, initial_capital, "entry_idx")
    result["skipped_by_type"] = skipped_by_type
    return result


def replay_base_priority_smc_first(
    base_events: list[dict[str, Any]],
    stable_events: list[dict[str, Any]],
    smc_events: list[dict[str, Any]],
    initial_capital: float,
    data_end: pd.Timestamp,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    smc_filtered, smc_skipped = filtered_base_priority_overlay(base_events, smc_events)
    stable_filtered, stable_skipped = filtered_base_priority_overlay(base_events + smc_filtered, stable_events)
    result = standard_event_summary(base_events + smc_filtered + stable_filtered, initial_capital, "entry_idx")
    result = add_standard_windows(result, initial_capital, data_end, "entry_idx")
    result = add_combo_deltas(result, baseline)
    result["combo_mode"] = "base_priority_smc_first"
    result["base_priority_overlay_skipped"] = {
        "smc_short": smc_skipped.get("smc_short", 0),
        "stable_reverse_short": stable_skipped.get("stable_reverse_short", 0),
    }
    return result


def replay_base_priority_stable_first(
    base_events: list[dict[str, Any]],
    stable_events: list[dict[str, Any]],
    smc_events: list[dict[str, Any]],
    initial_capital: float,
    data_end: pd.Timestamp,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    stable_filtered, stable_skipped = filtered_base_priority_overlay(base_events, stable_events)
    smc_filtered, smc_skipped = filtered_base_priority_overlay(base_events + stable_filtered, smc_events)
    result = standard_event_summary(base_events + stable_filtered + smc_filtered, initial_capital, "entry_idx")
    result = add_standard_windows(result, initial_capital, data_end, "entry_idx")
    result = add_combo_deltas(result, baseline)
    result["combo_mode"] = "base_priority_stable_first"
    result["base_priority_overlay_skipped"] = {
        "stable_reverse_short": stable_skipped.get("stable_reverse_short", 0),
        "smc_short": smc_skipped.get("smc_short", 0),
    }
    return result


def smc_case_namespace(args: argparse.Namespace, case_params: dict[str, Any]) -> argparse.Namespace:
    defaults = {
        "target_rr": 2.0,
        "allowed_time_buckets": "other",
        "swing_n": 3,
        "min_body_atr": 0.7,
        "min_range_atr": 1.1,
        "entry_lookahead_bars": 40,
        "max_open_positions": 1,
        "min_displacement_body_atr": 0.0,
        "min_displacement_range_atr": 0.0,
        "max_mss_lag_bars": 15,
        "global_min_mss_lag_bars": 0,
        "global_max_mss_lag_bars": 0,
        "ny_max_mss_lag_bars": 0,
        "other_min_mss_lag_bars": 0,
        "drop_asia_session": False,
        "leverage": 10.0,
        "position_size_pct": 1.0,
        "maintenance_margin_pct": 0.5,
        "min_liq_buffer_pct": 1.2,
        "initial_capital": 1000.0,
        "output": "",
    }
    merged = defaults | case_params
    merged["data_15m"] = args.data_15m
    merged["data_4h"] = args.data_4h
    merged["start_date"] = args.start_date
    return argparse.Namespace(**merged)


def build_smc_events(
    case_name: str,
    case_params: dict[str, Any],
    args: argparse.Namespace,
    prepared: Any,
    daily: list[Any],
    h4_highs: list[int],
    h4_lows: list[int],
    d1_highs: list[int],
    d1_lows: list[int],
    allocation: float,
    *,
    taker_fee_rate: float,
    slippage_bps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_args = smc_case_namespace(args, case_params)
    smc_args = smc_short_v1_strategy_args(case_args)
    rows = trade_rows_for_events(
        scan_events(prepared.c15m, build_event_scan_args(smc_args)),
        prepared,
        daily,
        h4_highs,
        h4_lows,
        d1_highs,
        d1_lows,
        smc_args,
    )
    if int(getattr(case_args, "global_min_mss_lag_bars", 0)) > 0:
        floor = int(case_args.global_min_mss_lag_bars)
        rows = [row for row in rows if row["mss_lag_bars"] is None or int(row["mss_lag_bars"]) >= floor]
    if int(getattr(case_args, "global_max_mss_lag_bars", 0)) > 0:
        ceiling = int(case_args.global_max_mss_lag_bars)
        rows = [row for row in rows if row["mss_lag_bars"] is None or int(row["mss_lag_bars"]) <= ceiling]
    if int(getattr(case_args, "ny_max_mss_lag_bars", 0)) > 0:
        ny_limit = int(case_args.ny_max_mss_lag_bars)
        rows = [
            row for row in rows
            if row["time_bucket"] != "ny_am_killzone"
            or row["mss_lag_bars"] is None
            or int(row["mss_lag_bars"]) <= ny_limit
        ]
    if int(getattr(case_args, "other_min_mss_lag_bars", 0)) > 0:
        other_floor = int(case_args.other_min_mss_lag_bars)
        rows = [
            row for row in rows
            if row["time_bucket"] != "other"
            or row["mss_lag_bars"] is None
            or int(row["mss_lag_bars"]) >= other_floor
        ]
    if bool(getattr(case_args, "drop_asia_session", False)):
        rows = [row for row in rows if row["time_bucket"] != "asia_evening_ny"]
    raw_trades = len(rows)
    rows, slot_skipped = apply_max_open_positions(rows, int(smc_args.max_open_positions))

    capital = float(case_args.initial_capital)
    accepted_rows: list[dict[str, Any]] = []
    guard_skipped = 0
    failures: dict[str, int] = {}
    for row in rows:
        trade = pd.Series(
            {
                "entry_time": row["entry_time"],
                "direction": row["direction"],
                "entry_price": row["entry_price"],
                "initial_stop_price": row["stop_price"],
                "notional": capital * float(case_args.leverage) * float(case_args.position_size_pct),
            }
        )
        diagnostics = _high_leverage_trade_diagnostics(
            trade,
            capital=capital,
            leverage=float(case_args.leverage),
            maintenance_margin_pct=float(case_args.maintenance_margin_pct),
        )
        if float(diagnostics["liquidation_buffer_pct"]) < float(case_args.min_liq_buffer_pct):
            failures["liquidation_buffer_too_small"] = failures.get("liquidation_buffer_too_small", 0) + 1
            guard_skipped += 1
            continue
        return_model = leveraged_net_return(
            signal_return_pct=float(row["signal_return_pct"] or 0.0),
            leverage=float(case_args.leverage),
            position_size_pct=float(case_args.position_size_pct),
            allocation=float(allocation),
            taker_fee_rate=float(taker_fee_rate),
            slippage_bps=float(slippage_bps),
        )
        leveraged_return = float(return_model["account_return"])
        capital *= 1.0 + leveraged_return
        accepted_rows.append(row | {"leveraged_return": leveraged_return, "return_model": return_model})

    events: list[dict[str, Any]] = []
    for row in accepted_rows:
        return_value = float(row["leveraged_return"])
        return_model = row["return_model"]
        events.append(
            {
                "event_type": "smc_short",
                "entry_idx": int(row["entry_idx"]),
                "exit_idx": int(row["exit_idx"]),
                "entry_time": str(normalize_ts(row["entry_time"])),
                "exit_time": str(normalize_ts(row["exit_time"])),
                "direction": "BEAR",
                "return": return_value,
                "return_pct": round(return_value * 100.0, 4),
                "exit_reason": str(row.get("outcome") or row.get("status") or "unknown"),
                "smc_case": case_name,
                "smc_allocation": float(allocation),
                "smc_rr_result": float(row.get("rr_result", 0.0) or 0.0),
                "smc_signal_return_pct": round(float(row.get("signal_return_pct", 0.0) or 0.0), 4),
                "smc_roundtrip_cost_pct": round(float(return_model["roundtrip_cost_pct"]), 4),
                "smc_unit_return_pct": round(float(return_model["net_unit_return_pct"]), 4),
                "smc_taker_fee_rate": float(taker_fee_rate),
                "smc_slippage_bps": float(slippage_bps),
                "smc_time_bucket": str(row.get("time_bucket") or ""),
                "smc_mss_lag_bars": row.get("mss_lag_bars"),
            }
        )
    summary = {
        "raw_trades": raw_trades,
        "slot_trades": len(rows),
        "slot_skipped": slot_skipped,
        "guard_skipped": guard_skipped,
        "failures": failures,
        "accepted_trades": len(events),
        "allocation": allocation,
        "case_params": case_params,
        "fee_model": {
            "taker_fee_rate": float(taker_fee_rate),
            "slippage_bps": float(slippage_bps),
            "roundtrip_cost_pct": round(roundtrip_cost_rate(float(taker_fee_rate), float(slippage_bps)) * 100.0, 4),
        },
        "standalone_event_stats": event_return_stats(events, float(case_args.initial_capital)),
    }
    return events, summary


def build_stable_events(payload: dict[str, Any], prepared: Any, shadow_events: list[dict[str, Any]], allocation: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reverse_candidates = []
    selected_count = 0
    for event in shadow_events:
        if not selected_by(event, "guarded_weak_loss", 1):
            continue
        selected_count += 1
        simulated = simulate_short_trade(
            event=event,
            candles=prepared.c15m,
            trigger_mode="stop_loss_reversal",
            target_rr=2.75,
            max_hold_bars=80,
            leverage=5.0,
            stop_multiplier=1.1,
            max_short_stop_pct=1.75,
            virtual_invalidation_rr=None,
            virtual_invalidation_lookahead_bars=None,
            taker_fee_rate=float(payload.get("taker_fee_rate", 0.0005) or 0.0),
            slippage_bps=float(payload.get("slippage_bps", 0.0) or 0.0),
        )
        if simulated is not None:
            reverse_candidates.append(simulated)
    reverse_only = replay_non_overlapping(reverse_candidates, 1000.0)
    events = []
    for event in reverse_only["events"]:
        converted = standard_reverse_short_event(event, allocation)
        converted["event_type"] = "stable_reverse_short"
        events.append(converted)
    return events, {
        "selector_matches": selected_count,
        "simulated_candidates": len(reverse_candidates),
        "accepted_trades": len(events),
        "skipped_overlap": reverse_only.get("skipped_overlap", 0),
        "params": {
            "selector": "guarded_weak_loss",
            "target_rr": 2.75,
            "max_hold_bars": 80,
            "leverage": 5.0,
            "stop_multiplier": 1.1,
            "max_short_stop_pct": 1.75,
            "allocation": allocation,
        },
        "standalone_event_stats": event_return_stats(events, 1000.0),
    }


def overlay_stats(result: dict[str, Any], initial_capital: float, data_end: pd.Timestamp) -> dict[str, Any]:
    events = result.get("events", [])
    starts = {
        "current_year": pd.Timestamp(f"{data_end.year}-01-01", tz="UTC"),
        "last_60d": data_end - pd.Timedelta(days=60),
        "last_30d": data_end - pd.Timedelta(days=30),
    }
    out: dict[str, Any] = {}
    for event_type in ("stable_reverse_short", "smc_short"):
        typed = [event for event in events if str(event.get("event_type") or "") == event_type]
        out[event_type] = {
            "overall": event_return_stats(typed, initial_capital),
            "windows": {
                name: event_return_stats(
                    [event for event in typed if normalize_ts(event["entry_time"]) >= start],
                    initial_capital,
                )
                for name, start in starts.items()
            },
        }
    return out


def bucket_event_stats(events: list[dict[str, Any]], initial_capital: float, freq: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        ts = normalize_ts(event["entry_time"])
        key = str(ts.year) if freq == "year" else f"{ts.year:04d}-{ts.month:02d}"
        buckets.setdefault(key, []).append(event)
    return {
        key: event_return_stats(bucket, initial_capital)
        for key, bucket in sorted(buckets.items())
    }


def event_spacing_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(events, key=lambda item: normalize_ts(item["entry_time"]))
    if len(ordered) < 2:
        return {
            "min_gap_hours": None,
            "median_gap_hours": None,
            "max_gap_hours": None,
        }
    gaps = [
        (normalize_ts(ordered[idx]["entry_time"]) - normalize_ts(ordered[idx - 1]["entry_time"])).total_seconds() / 3600.0
        for idx in range(1, len(ordered))
    ]
    gaps_sorted = sorted(gaps)
    median = gaps_sorted[len(gaps_sorted) // 2] if len(gaps_sorted) % 2 else (
        gaps_sorted[len(gaps_sorted) // 2 - 1] + gaps_sorted[len(gaps_sorted) // 2]
    ) / 2.0
    return {
        "min_gap_hours": round(min(gaps), 4),
        "median_gap_hours": round(median, 4),
        "max_gap_hours": round(max(gaps), 4),
    }


def concentration_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    returns = sorted((float(event.get("return", 0.0) or 0.0) for event in events), reverse=True)
    positive = [value for value in returns if value > 0]
    total_positive = sum(positive)
    total_abs = sum(abs(value) for value in returns)
    return {
        "best_trade_return_pct": round((max(returns) if returns else 0.0) * 100.0, 4),
        "worst_trade_return_pct": round((min(returns) if returns else 0.0) * 100.0, 4),
        "top_1_share_of_positive_pct": round((positive[0] / total_positive * 100.0), 2) if total_positive > 0 and positive else 0.0,
        "top_3_share_of_positive_pct": round((sum(positive[:3]) / total_positive * 100.0), 2) if total_positive > 0 and positive else 0.0,
        "top_3_share_of_abs_return_pct": round((sum(abs(value) for value in returns[:3]) / total_abs * 100.0), 2) if total_abs > 0 else 0.0,
    }


def live_feasibility_audit(result: dict[str, Any], initial_capital: float) -> dict[str, Any]:
    events = result.get("events", [])
    by_type: dict[str, Any] = {}
    for event_type in ("sota_long", "stable_reverse_short", "smc_short"):
        typed = [event for event in events if str(event.get("event_type") or "") == event_type]
        by_type[event_type] = {
            "overall": event_return_stats(typed, initial_capital),
            "yearly": bucket_event_stats(typed, initial_capital, "year"),
            "monthly": bucket_event_stats(typed, initial_capital, "month"),
            "spacing": event_spacing_stats(typed),
            "concentration": concentration_stats(typed),
        }
    all_months = bucket_event_stats(events, initial_capital, "month")
    bad_months = {
        key: value
        for key, value in all_months.items()
        if float(value.get("compounded_return_pct", 0.0) or 0.0) < 0.0
    }
    return {
        "by_event_type": by_type,
        "all_events_monthly": all_months,
        "negative_months": bad_months,
        "negative_month_count": len(bad_months),
        "all_event_spacing": event_spacing_stats(events),
        "all_event_concentration": concentration_stats(events),
    }


def compact_with_overlay_stats(result: dict[str, Any], sample_trades: int, initial_capital: float, data_end: pd.Timestamp) -> dict[str, Any]:
    compacted = compact_combo_result(result, sample_trades)
    compacted["overlay_attribution"] = overlay_stats(result, initial_capital, data_end)
    compacted["live_feasibility_audit"] = live_feasibility_audit(result, initial_capital)
    return compacted


def main() -> None:
    args = parse_args()
    base_payload = load_config_payload(Path(args.config))
    payload, pressure_params = apply_pressure_params(base_payload, Path(args.pressure_params))
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=payload.get("regime_switcher_thresholds"),
    )
    metrics, engine = run_engine(payload, prepared, args.start_date)
    trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
    initial_capital = float(metrics.get("initial_capital", 1000.0))
    fixed = expansion_overlay(trades, initial_capital, FIXED_STRUCTURE_PARAMS, include_events=True)
    shadow = replay_shadow_events(
        fixed["events"],
        initial_capital,
        daily_loss_stop_pct=float(args.daily_loss_stop_pct),
        equity_drawdown_stop_pct=float(args.equity_drawdown_stop_pct),
        consecutive_loss_stop=int(args.consecutive_loss_stop),
        equity_drawdown_cooldown_days=int(args.equity_drawdown_cooldown_days),
    )
    shadow_events = shadow["events"]
    base_shadow_summary = event_stream_summary(shadow_events, initial_capital, prepared.end)
    base_events = [standard_sota_event(event) for event in shadow_events]
    stable_events, stable_summary = build_stable_events(payload, prepared, shadow_events, float(args.stable_allocation))
    stable_only = next(
        item for item in build_combo_results(
            base_events,
            stable_events,
            initial_capital,
            prepared.end,
            base_shadow_summary,
        )
        if str(item["combo_mode"]) == "base_priority_single_slot"
    )

    daily = daily_candles_from_4h(prepared.c4h)
    h4_highs, h4_lows = precompute_swings(prepared.c4h, n=2, lookback=80)
    d1_highs, d1_lows = precompute_swings(daily, n=2, lookback=20)

    candidates: list[dict[str, Any]] = []
    for case_name in parse_str_list(args.smc_cases):
        if case_name not in SMC_CASES:
            raise ValueError(f"Unsupported SMC case: {case_name}")
        for allocation in parse_float_list(args.smc_allocation_values):
            smc_events, smc_summary = build_smc_events(
                case_name,
                SMC_CASES[case_name],
                args,
                prepared,
                daily,
                h4_highs,
                h4_lows,
                d1_highs,
                d1_lows,
                allocation,
                taker_fee_rate=float(payload.get("taker_fee_rate", 0.0005) or 0.0),
                slippage_bps=float(payload.get("slippage_bps", 0.0) or 0.0),
            )
            smc_only = next(
                item for item in build_combo_results(
                    base_events,
                    smc_events,
                    initial_capital,
                    prepared.end,
                    base_shadow_summary,
                )
                if str(item["combo_mode"]) == "base_priority_single_slot"
            )
            stable_first = replay_base_priority_stable_first(
                base_events,
                stable_events,
                smc_events,
                initial_capital,
                prepared.end,
                base_shadow_summary,
            )
            smc_first = replay_base_priority_smc_first(
                base_events,
                stable_events,
                smc_events,
                initial_capital,
                prepared.end,
                base_shadow_summary,
            )
            overlay_priority = replay_overlay_priority_single_slot(
                base_events + stable_events + smc_events,
                initial_capital,
            )
            overlay_priority = add_standard_windows(overlay_priority, initial_capital, prepared.end, "entry_idx")
            overlay_priority = add_combo_deltas(overlay_priority, base_shadow_summary)
            overlay_priority["combo_mode"] = "overlay_priority_stable_smc"

            for result in (smc_only, stable_first, smc_first, overlay_priority):
                result["params"] = {
                    "smc_case": case_name,
                    "smc_allocation": allocation,
                    "stable_allocation": float(args.stable_allocation),
                    "combo_mode": result["combo_mode"],
                }
                result["smc_summary"] = smc_summary
                result["stable_summary"] = stable_summary
                delta = result.get("delta_vs_shadow_sota", {})
                year_delta = result.get("window_deltas_vs_shadow_sota", {}).get("current_year", {})
                dd_penalty = max(0.0, float(delta.get("max_drawdown_pct", 0.0) or 0.0)) * 10000.0
                result["merge_score"] = round(
                    float(delta.get("total_return_pct", 0.0) or 0.0)
                    + float(year_delta.get("total_return_pct", 0.0) or 0.0) * 250.0
                    - dd_penalty,
                    4,
                )
                candidates.append(result)

    candidates.sort(key=lambda item: float(item["merge_score"]), reverse=True)
    top_payload = [
        compact_with_overlay_stats(item, int(args.sample_trades), initial_capital, prepared.end)
        for item in candidates[: int(args.top)]
    ]
    if top_payload and candidates:
        top_payload[0]["all_events"] = candidates[0].get("events", [])

    report = {
        "metadata": {
            "config": str(Path(args.config).resolve()),
            "pressure_params": str(Path(args.pressure_params).resolve()),
            "pressure_params_applied": pressure_params,
            "start_date": args.start_date,
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
        },
        "baseline_shadow_sota": compact_result(base_shadow_summary, 0),
        "stable_only": compact_with_overlay_stats(stable_only, int(args.sample_trades), initial_capital, prepared.end),
        "stable_summary": stable_summary,
        "top": top_payload,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    print(output)
    base = report["baseline_shadow_sota"]
    stable = report["stable_only"]
    print(f"Baseline full={base['total_return_pct']:.2f}%/{base['max_drawdown_pct']:.2f}% 2026={base['windows']['current_year']['total_return_pct']:.2f}%")
    print(f"Stable only full={stable['total_return_pct']:.2f}%/{stable['max_drawdown_pct']:.2f}% 2026={stable['windows']['current_year']['total_return_pct']:.2f}%")
    for idx, item in enumerate(report["top"][:10], start=1):
        params = item["params"]
        delta = item["delta_vs_shadow_sota"]
        year_delta = item["window_deltas_vs_shadow_sota"]["current_year"]
        counts = item.get("event_type_counts", {})
        smc_attr = item["overlay_attribution"]["smc_short"]["overall"]
        print(
            f"{idx:02d} mode={params['combo_mode']} case={params['smc_case']} alloc={params['smc_allocation']:.2f} "
            f"full={item['total_return_pct']:.2f}%/{item['max_drawdown_pct']:.2f}% "
            f"delta={delta['total_return_pct']:.2f}%/{delta['max_drawdown_pct']:+.2f}dd "
            f"2026={item['windows']['current_year']['total_return_pct']:.2f}% yD={year_delta['total_return_pct']:.2f}% "
            f"counts={counts} smc_comp={smc_attr['compounded_return_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
