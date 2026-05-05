#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from scripts.live_readiness_report import _high_leverage_trade_diagnostics
from scripts.live_shadow_utils import add_combo_deltas, add_standard_windows, event_return_stats, standard_event_summary
from scripts.reproduce_smc_short_only_v1_10x import strategy_args as smc_short_v1_strategy_args
from scripts.research_smc_standalone_v1 import (
    apply_max_open_positions,
    build_event_scan_args,
    scan_events,
    trade_rows_for_events,
)


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


def filtered_base_priority_overlay(
    base_events: list[dict[str, Any]],
    overlay_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    accepted: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for event in sorted(overlay_events, key=lambda item: (int(item["entry_idx"]), int(item["exit_idx"]))):
        entry_idx = int(event["entry_idx"])
        exit_idx = int(event["exit_idx"])
        overlaps = any(entry_idx < int(base["exit_idx"]) and exit_idx > int(base["entry_idx"]) for base in base_events)
        if overlaps:
            event_type = str(event.get("event_type") or "unknown")
            skipped[event_type] = skipped.get(event_type, 0) + 1
            continue
        accepted.append(event)
    return accepted, skipped


def replay_base_priority_sota_first(
    base_events: list[dict[str, Any]],
    smc_events: list[dict[str, Any]],
    initial_capital: float,
    data_end: pd.Timestamp,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    smc_filtered, smc_skipped = filtered_base_priority_overlay(base_events, smc_events)
    result = standard_event_summary(base_events + smc_filtered, initial_capital, "entry_idx")
    result = add_standard_windows(result, initial_capital, data_end, "entry_idx")
    result = add_combo_deltas(result, baseline)
    result["combo_mode"] = "base_priority_sota_first"
    result["base_priority_overlay_skipped"] = {
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
    event_types = sorted({str(event.get("event_type") or "unknown") for event in events})
    by_type: dict[str, Any] = {}
    for event_type in event_types:
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
