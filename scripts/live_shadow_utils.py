#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

import pandas as pd

from scripts.live_readiness_report import max_drawdown_from_capitals, trade_return_sharpe


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_for_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def event_timestamp(event: dict[str, Any], key: str) -> pd.Timestamp:
    ts = pd.Timestamp(event[key])
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def pct(value: float) -> float:
    return round(value * 100.0, 4)


def compounded_return_pct(events: list[dict[str, Any]], initial_capital: float) -> float:
    capital = initial_capital
    for event in sorted(events, key=lambda item: (int(item.get("entry_idx", 0) or 0), int(item.get("exit_idx", 0) or 0))):
        capital = max(0.0, capital * (1.0 + float(event.get("return", 0.0) or 0.0)))
    return round((capital - initial_capital) / initial_capital * 100.0, 4)


def event_return_stats(events: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    returns = [float(event.get("return", 0.0) or 0.0) for event in events]
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    ordered_returns = sorted(returns, reverse=True)
    return {
        "trades": len(events),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(returns) * 100.0, 2) if returns else 0.0,
        "sum_return_pct": round(sum(returns) * 100.0, 4),
        "compounded_return_pct": compounded_return_pct(events, initial_capital),
        "avg_return_pct": round(sum(returns) / len(returns) * 100.0, 4) if returns else 0.0,
        "best_return_pct": round(max(returns) * 100.0, 4) if returns else 0.0,
        "worst_return_pct": round(min(returns) * 100.0, 4) if returns else 0.0,
        "top_3_sum_return_pct": round(sum(ordered_returns[:3]) * 100.0, 4) if returns else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "exit_counts": {
            reason: sum(1 for event in events if str(event.get("exit_reason") or "unknown") == reason)
            for reason in sorted({str(event.get("exit_reason") or "unknown") for event in events})
        },
    }


def replay_non_overlapping(trades: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    ordered = sorted(trades, key=lambda item: (int(item["entry_idx"]), int(item["exit_idx"])))
    capital = initial_capital
    capitals: list[float] = []
    returns: list[float] = []
    accepted: list[dict[str, Any]] = []
    skipped_overlap = 0
    last_exit_idx = -1
    exit_counts: dict[str, int] = {}
    gross_profit = 0.0
    gross_loss = 0.0

    for trade in ordered:
        if int(trade["entry_idx"]) <= last_exit_idx:
            skipped_overlap += 1
            continue
        trade_return = float(trade["return"])
        capital = max(0.0, capital * (1.0 + trade_return))
        returns.append(trade_return)
        capitals.append(capital)
        accepted_trade = dict(trade)
        accepted_trade["capital"] = round(capital, 2)
        accepted.append(accepted_trade)
        last_exit_idx = int(trade["exit_idx"])
        exit_reason = str(trade["exit_reason"])
        exit_counts[exit_reason] = exit_counts.get(exit_reason, 0) + 1
        if trade_return > 0:
            gross_profit += trade_return
        else:
            gross_loss += abs(trade_return)

    wins = sum(1 for value in returns if value > 0)
    losses = len(returns) - wins
    return {
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100.0, 2),
        "final_capital": round(capital, 2),
        "sharpe_ratio": round(trade_return_sharpe(returns), 3),
        "max_drawdown_pct": round(max_drawdown_from_capitals(capitals, initial_capital), 2),
        "trades": len(accepted),
        "raw_candidates": len(ordered),
        "skipped_overlap": skipped_overlap,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wins / len(returns) * 100.0, 2) if returns else 0.0,
        "avg_return_pct": round(sum(returns) / len(returns) * 100.0, 4) if returns else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "exit_counts": exit_counts,
        "events": accepted,
    }


def replay_window(events: list[dict[str, Any]], initial_capital: float, start: pd.Timestamp) -> dict[str, Any]:
    selected = [event for event in events if event_timestamp(event, "entry_time") >= start]
    return {key: value for key, value in replay_non_overlapping(selected, initial_capital).items() if key != "events"}


def add_windows(result: dict[str, Any], initial_capital: float, data_end: pd.Timestamp) -> dict[str, Any]:
    events = result["events"]
    starts = {
        "current_year": pd.Timestamp(f"{data_end.year}-01-01", tz="UTC"),
        "last_60d": data_end - pd.Timedelta(days=60),
        "last_30d": data_end - pd.Timedelta(days=30),
    }
    result["windows"] = {name: replay_window(events, initial_capital, start) for name, start in starts.items()}
    return result


def event_stream_summary(events: list[dict[str, Any]], initial_capital: float, data_end: pd.Timestamp) -> dict[str, Any]:
    trades = []
    for event in events:
        copied = dict(event)
        copied["entry_idx"] = int(copied.get("entry_idx") or 0)
        copied["exit_idx"] = int(copied.get("exit_idx") or copied["entry_idx"])
        copied["return_pct"] = pct(float(copied.get("return", 0.0) or 0.0))
        trades.append(copied)
    result = replay_non_overlapping(trades, initial_capital)
    return add_windows(result, initial_capital, data_end)


def compact_result(result: dict[str, Any], sample_trades: int) -> dict[str, Any]:
    payload = {key: value for key, value in result.items() if key != "events"}
    payload["sample_events"] = result.get("events", [])[:sample_trades]
    return payload


def quality_snapshot(event: dict[str, Any]) -> dict[str, Any]:
    try:
        from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS

        momentum_min = float(FIXED_STRUCTURE_PARAMS["failed_breakout_guard_min_momentum_pct"])
        ema_gap_min = float(FIXED_STRUCTURE_PARAMS["failed_breakout_guard_min_ema_gap_pct"])
        adx_min = float(FIXED_STRUCTURE_PARAMS["failed_breakout_guard_min_adx"])
    except Exception:
        momentum_min = 6.0
        ema_gap_min = 2.0
        adx_min = 38.0
    direction = str(event.get("direction") or "")
    sign = 1.0 if direction == "BULL" else -1.0
    momentum_pct = float(event.get("feature_momentum", 0.0) or 0.0) * 100.0 * sign
    ema_gap_pct = float(event.get("feature_ema_gap", 0.0) or 0.0) * 100.0 * sign
    adx = float(event.get("feature_adx", 0.0) or 0.0)
    structure_ok = (
        bool(event.get("feature_bullish_structure"))
        if direction == "BULL"
        else bool(event.get("feature_bearish_structure"))
    )
    checks = {
        "momentum": momentum_pct >= momentum_min,
        "ema_gap": ema_gap_pct >= ema_gap_min,
        "adx": adx >= adx_min,
        "structure": structure_ok,
    }
    return {"quality_score": sum(1 for passed in checks.values() if passed), "checks": checks}


def standard_sota_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": "sota_long",
        "entry_idx": int(event.get("entry_idx") or 0),
        "exit_idx": int(event.get("exit_idx") or event.get("entry_idx") or 0),
        "entry_time": str(event.get("entry_time")),
        "exit_time": str(event.get("exit_time")),
        "direction": event.get("direction"),
        "return": float(event.get("return", 0.0) or 0.0),
        "return_pct": pct(float(event.get("return", 0.0) or 0.0)),
        "exit_reason": event.get("exit_reason"),
        "source_effective_leverage": event.get("effective_leverage"),
        "source_failed_breakout_guard_applied": bool(event.get("failed_breakout_guard_applied")),
        "source_quality_score": quality_snapshot(event)["quality_score"],
        "regime_label": event.get("regime_label"),
        "feature_adx": float(event.get("feature_adx", 0.0) or 0.0),
        "feature_momentum": float(event.get("feature_momentum", 0.0) or 0.0),
        "feature_ema_gap": float(event.get("feature_ema_gap", 0.0) or 0.0),
        "feature_bullish_structure": bool(event.get("feature_bullish_structure", False)),
        "feature_bearish_structure": bool(event.get("feature_bearish_structure", False)),
    }


def standard_event_summary(
    events: list[dict[str, Any]],
    initial_capital: float,
    order_key: str,
) -> dict[str, Any]:
    ordered = sorted(events, key=lambda item: (int(item.get(order_key, 0) or 0), int(item.get("exit_idx", 0) or 0)))
    capital = initial_capital
    capitals: list[float] = []
    returns: list[float] = []
    accepted: list[dict[str, Any]] = []
    event_type_counts: dict[str, int] = {}
    exit_counts: dict[str, int] = {}
    gross_profit = 0.0
    gross_loss = 0.0
    for event in ordered:
        trade_return = float(event.get("return", 0.0) or 0.0)
        capital = max(0.0, capital * (1.0 + trade_return))
        capitals.append(capital)
        returns.append(trade_return)
        accepted_event = dict(event)
        accepted_event["capital"] = round(capital, 2)
        accepted.append(accepted_event)
        event_type = str(event.get("event_type") or "unknown")
        exit_reason = str(event.get("exit_reason") or "unknown")
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        exit_counts[exit_reason] = exit_counts.get(exit_reason, 0) + 1
        if trade_return > 0:
            gross_profit += trade_return
        else:
            gross_loss += abs(trade_return)

    wins = sum(1 for value in returns if value > 0)
    losses = len(returns) - wins
    return {
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100.0, 2),
        "final_capital": round(capital, 2),
        "sharpe_ratio": round(trade_return_sharpe(returns), 3),
        "max_drawdown_pct": round(max_drawdown_from_capitals(capitals, initial_capital), 2),
        "trades": len(accepted),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wins / len(returns) * 100.0, 2) if returns else 0.0,
        "avg_return_pct": round(sum(returns) / len(returns) * 100.0, 4) if returns else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "event_type_counts": event_type_counts,
        "exit_counts": exit_counts,
        "events": accepted,
    }


def add_standard_windows(
    result: dict[str, Any],
    initial_capital: float,
    data_end: pd.Timestamp,
    order_key: str,
) -> dict[str, Any]:
    events = result["events"]
    starts = {
        "current_year": pd.Timestamp(f"{data_end.year}-01-01", tz="UTC"),
        "last_60d": data_end - pd.Timedelta(days=60),
        "last_30d": data_end - pd.Timedelta(days=30),
    }
    result["windows"] = {
        name: {
            key: value
            for key, value in standard_event_summary(
                [event for event in events if event_timestamp(event, "entry_time") >= start],
                initial_capital,
                order_key,
            ).items()
            if key != "events"
        }
        for name, start in starts.items()
    }
    return result


def add_combo_deltas(result: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    result["delta_vs_shadow_sota"] = {
        "total_return_pct": round(float(result.get("total_return_pct", 0.0)) - float(baseline.get("total_return_pct", 0.0)), 4),
        "max_drawdown_pct": round(float(result.get("max_drawdown_pct", 0.0)) - float(baseline.get("max_drawdown_pct", 0.0)), 4),
    }
    result["window_deltas_vs_shadow_sota"] = {}
    for name, window in result.get("windows", {}).items():
        base_window = baseline.get("windows", {}).get(name, {})
        result["window_deltas_vs_shadow_sota"][name] = {
            "total_return_pct": round(float(window.get("total_return_pct", 0.0)) - float(base_window.get("total_return_pct", 0.0)), 4),
            "max_drawdown_pct": round(float(window.get("max_drawdown_pct", 0.0)) - float(base_window.get("max_drawdown_pct", 0.0)), 4),
        }
    return result


def compact_combo_result(result: dict[str, Any], sample_trades: int) -> dict[str, Any]:
    payload = {key: value for key, value in result.items() if key != "events"}
    payload["sample_events"] = result.get("events", [])[:sample_trades]
    return payload
