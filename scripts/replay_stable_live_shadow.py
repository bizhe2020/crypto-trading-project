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

from scripts.backtest_config_report import DEFAULT_DATA_15M, DEFAULT_DATA_4H, load_config_payload  # noqa: E402
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, max_drawdown_from_capitals, run_engine, trade_dataframe, trade_return_sharpe  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402
from strategy.sota_overlay_state import OverlayCandidate, replay_single_position_events  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "stable_live_shadow_replay.json"
DEFAULT_PAPER_LOG = ROOT / "var" / "high_leverage_expansion" / "stable_live_shadow_paper_decisions.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay SOTA + Stable reverse-short in chronological single-position live-shadow mode.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=0)
    parser.add_argument("--stable-allocation", type=float, default=1.0)
    parser.add_argument("--stable-target-rr", type=float, default=2.75)
    parser.add_argument("--stable-max-hold-bars", type=int, default=40)
    parser.add_argument("--stable-leverage", type=float, default=5.0)
    parser.add_argument("--stable-stop-multiplier", type=float, default=1.0)
    parser.add_argument("--stable-max-short-stop-pct", type=float, default=1.75)
    parser.add_argument("--sample-trades", type=int, default=40)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--paper-log-output", default=str(DEFAULT_PAPER_LOG))
    return parser.parse_args()


def candle_time(candle: Any) -> str:
    return str(pd.Timestamp(float(candle.ts), unit="s", tz="UTC"))


def event_timestamp(event: dict[str, Any], key: str) -> pd.Timestamp:
    return pd.Timestamp(event[key]).tz_convert("UTC")


def pct(value: float) -> float:
    return round(value * 100.0, 4)


def quality_snapshot(event: dict[str, Any]) -> dict[str, Any]:
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
        "momentum": momentum_pct >= float(FIXED_STRUCTURE_PARAMS["failed_breakout_guard_min_momentum_pct"]),
        "ema_gap": ema_gap_pct >= float(FIXED_STRUCTURE_PARAMS["failed_breakout_guard_min_ema_gap_pct"]),
        "adx": adx >= float(FIXED_STRUCTURE_PARAMS["failed_breakout_guard_min_adx"]),
        "structure": structure_ok,
    }
    return {
        "quality_score": sum(1 for passed in checks.values() if passed),
        "directional_momentum_pct": round(momentum_pct, 6),
        "directional_ema_gap_pct": round(ema_gap_pct, 6),
        "adx": round(adx, 6),
        "checks": checks,
    }


def selected_by(event: dict[str, Any], selector: str, max_quality_score: int) -> bool:
    direction = str(event.get("direction") or "")
    exit_reason = str(event.get("exit_reason") or "")
    if selector != "guarded_weak_loss":
        raise ValueError(f"Unsupported selector: {selector}")
    return (
        direction == "BULL"
        and str(event.get("regime_label") or "") == "high_growth"
        and str(event.get("risk_mode") or "") == "offense"
        and exit_reason == "stop_loss"
        and float(event.get("return", 0.0) or 0.0) < 0.0
        and bool(event.get("failed_breakout_guard_applied"))
        and int(quality_snapshot(event)["quality_score"]) <= max_quality_score
    )


def short_entry_for_event(event: dict[str, Any], candles: list[Any]) -> tuple[int, float, str] | None:
    if str(event.get("exit_reason") or "") != "stop_loss":
        return None
    entry_idx = event.get("exit_idx")
    entry_price = float(event.get("exit_price", 0.0) or 0.0)
    entry_time = str(event.get("exit_time") or "")
    if entry_idx is None or entry_price <= 0:
        return None
    return int(entry_idx), entry_price, entry_time


def simulate_short_trade(
    *,
    event: dict[str, Any],
    candles: list[Any],
    target_rr: float,
    max_hold_bars: int,
    leverage: float,
    stop_multiplier: float,
    max_short_stop_pct: float,
    taker_fee_rate: float,
    slippage_bps: float,
) -> dict[str, Any] | None:
    entry = short_entry_for_event(event, candles)
    if entry is None:
        return None
    entry_idx, entry_price, entry_time = entry
    if entry_idx + 1 >= len(candles):
        return None

    source_stop_pct = float(event.get("stop_distance_pct", 0.0) or 0.0) / 100.0
    stop_pct = source_stop_pct * stop_multiplier
    if stop_pct <= 0 or stop_pct * 100.0 > max_short_stop_pct:
        return None
    target_price = entry_price * (1.0 - stop_pct * target_rr)
    stop_price = entry_price * (1.0 + stop_pct)
    if target_price <= 0 or stop_price <= entry_price:
        return None

    start = entry_idx + 1
    end = min(len(candles) - 1, entry_idx + max(1, max_hold_bars))
    if start > end:
        return None

    exit_idx = end
    exit_price = float(candles[end].c)
    exit_reason = "time_exit"
    mfe = 0.0
    mae = 0.0
    risk_price = entry_price * stop_pct
    for idx in range(start, end + 1):
        candle = candles[idx]
        high = float(candle.h)
        low = float(candle.l)
        mfe = max(mfe, entry_price - low)
        mae = max(mae, high - entry_price)
        if high >= stop_price:
            exit_idx = idx
            exit_price = stop_price
            exit_reason = "stop_loss"
            break
        if low <= target_price:
            exit_idx = idx
            exit_price = target_price
            exit_reason = "target_rr"
            break

    gross_unit_return = (entry_price - exit_price) / entry_price
    roundtrip_cost = 2.0 * float(taker_fee_rate) + 2.0 * float(slippage_bps) / 10000.0
    unit_return = gross_unit_return - roundtrip_cost
    trade_return = unit_return * leverage
    source_quality = quality_snapshot(event)
    return {
        "source_entry_time": event.get("entry_time"),
        "source_exit_time": event.get("exit_time"),
        "source_return_pct": pct(float(event.get("return", 0.0) or 0.0)),
        "source_exit_reason": event.get("exit_reason"),
        "source_effective_leverage": event.get("effective_leverage"),
        "source_failed_breakout_guard_applied": bool(event.get("failed_breakout_guard_applied")),
        "source_quality_score": source_quality["quality_score"],
        "source_quality": source_quality,
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "entry_time": entry_time,
        "exit_time": candle_time(candles[exit_idx]),
        "entry_price": round(entry_price, 6),
        "exit_price": round(exit_price, 6),
        "stop_price": round(stop_price, 6),
        "target_price": round(target_price, 6),
        "stop_distance_pct": round(stop_pct * 100.0, 6),
        "target_rr": target_rr,
        "max_hold_bars": max_hold_bars,
        "leverage": leverage,
        "stop_multiplier": stop_multiplier,
        "exit_reason": exit_reason,
        "gross_unit_return_pct": pct(gross_unit_return),
        "unit_return_pct": pct(unit_return),
        "roundtrip_cost_pct": pct(roundtrip_cost),
        "return": trade_return,
        "return_pct": pct(trade_return),
        "mfe_rr": round(mfe / risk_price, 4) if risk_price > 0 else 0.0,
        "mae_rr": round(mae / risk_price, 4) if risk_price > 0 else 0.0,
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
    }


def standard_reverse_short_event(event: dict[str, Any], overlay_allocation: float) -> dict[str, Any]:
    copied = dict(event)
    raw_return = float(copied.get("return", 0.0) or 0.0)
    copied["event_type"] = "stable_reverse_short"
    copied["raw_return"] = raw_return
    copied["raw_return_pct"] = pct(raw_return)
    copied["overlay_allocation"] = overlay_allocation
    copied["return"] = raw_return * overlay_allocation
    copied["return_pct"] = pct(float(copied["return"]))
    copied["direction"] = "BEAR"
    return copied


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


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    if pd.isna(value):
        return None
    return value


def decision_counts(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    by_decision: dict[str, int] = {}
    by_reject_reason: dict[str, int] = {}
    by_event_type_decision: dict[str, dict[str, int]] = {}
    by_paper_tag: dict[str, int] = {}
    for decision in decisions:
        action = str(decision.get("decision") or "unknown")
        event_type = str(decision.get("event_type") or "unknown")
        paper_tag = str(decision.get("paper_tag") or "untagged")
        by_decision[action] = by_decision.get(action, 0) + 1
        by_paper_tag[paper_tag] = by_paper_tag.get(paper_tag, 0) + 1
        by_event_type_decision.setdefault(event_type, {})
        by_event_type_decision[event_type][action] = by_event_type_decision[event_type].get(action, 0) + 1
        if action == "rejected":
            reason = str(decision.get("reason") or "unknown")
            by_reject_reason[reason] = by_reject_reason.get(reason, 0) + 1
    return {
        "by_decision": by_decision,
        "by_reject_reason": by_reject_reason,
        "by_event_type_decision": by_event_type_decision,
        "by_paper_tag": by_paper_tag,
    }


def stable_preempted_sota_summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    items = [decision for decision in decisions if decision.get("paper_tag") == "stable_preempted_sota"]
    return {
        "count": len(items),
        "sota_return_sum_pct": round(sum(float(item.get("return_pct", 0.0) or 0.0) for item in items), 4),
        "positive_sota_blocked": sum(1 for item in items if float(item.get("return_pct", 0.0) or 0.0) > 0.0),
        "negative_sota_blocked": sum(1 for item in items if float(item.get("return_pct", 0.0) or 0.0) <= 0.0),
        "items": items,
    }


def live_feasibility_audit(result: dict[str, Any], initial_capital: float) -> dict[str, Any]:
    windows = result.get("windows", {})
    year = windows.get("current_year", {})
    recent = windows.get("last_60d", {})
    return {
        "initial_capital": initial_capital,
        "total_return_pct": result.get("total_return_pct"),
        "max_drawdown_pct": result.get("max_drawdown_pct"),
        "current_year_return_pct": year.get("total_return_pct"),
        "current_year_max_drawdown_pct": year.get("max_drawdown_pct"),
        "last_60d_return_pct": recent.get("total_return_pct"),
        "last_60d_max_drawdown_pct": recent.get("max_drawdown_pct"),
        "accepted_trades": result.get("trades"),
    }


def replay_base_priority_stable_first(
    base_events: list[dict[str, Any]],
    stable_events: list[dict[str, Any]],
    initial_capital: float,
    data_end: pd.Timestamp,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    stable_candidates = [to_candidate(event) for event in stable_events]
    sota_candidates = [to_candidate(event) for event in base_events]
    accepted, _decisions = replay_single_position_events(stable_candidates + sota_candidates)
    accepted_events = [candidate.metadata["event"] for candidate in accepted]
    result = standard_event_summary(accepted_events, initial_capital, "entry_idx")
    result = add_standard_windows(result, initial_capital, data_end, "entry_idx")
    result = add_combo_deltas(result, baseline)
    result["combo_mode"] = "base_priority_stable_first"
    return result


def to_candidate(event: dict[str, Any]) -> OverlayCandidate:
    return OverlayCandidate(
        event_type=str(event.get("event_type") or "unknown"),
        direction=event.get("direction"),
        entry_idx=int(event.get("entry_idx", 0) or 0),
        exit_idx=int(event.get("exit_idx", event.get("entry_idx", 0)) or 0),
        entry_time=str(event.get("entry_time") or ""),
        exit_time=str(event.get("exit_time") or ""),
        return_rate=float(event.get("return", 0.0) or 0.0),
        metadata={"event": dict(event)},
    )


def write_paper_log(path: Path, decisions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(clean_for_json(decision), ensure_ascii=False, allow_nan=False) + "\n")


def build_stable_events(
    payload: dict[str, Any],
    prepared: Any,
    shadow_events: list[dict[str, Any]],
    *,
    allocation: float,
    target_rr: float,
    max_hold_bars: int,
    leverage: float,
    stop_multiplier: float,
    max_short_stop_pct: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reverse_candidates = []
    selected_count = 0
    for event in shadow_events:
        if not selected_by(event, "guarded_weak_loss", 1):
            continue
        selected_count += 1
        simulated = simulate_short_trade(
            event=event,
            candles=prepared.c15m,
            target_rr=target_rr,
            max_hold_bars=max_hold_bars,
            leverage=leverage,
            stop_multiplier=stop_multiplier,
            max_short_stop_pct=max_short_stop_pct,
            taker_fee_rate=float(payload.get("taker_fee_rate", 0.0005) or 0.0),
            slippage_bps=float(payload.get("slippage_bps", 0.0) or 0.0),
        )
        if simulated is not None:
            reverse_candidates.append(simulated)
    reverse_only = replay_non_overlapping(reverse_candidates, 1000.0)
    events = []
    for event in reverse_only["events"]:
        events.append(standard_reverse_short_event(event, allocation))
    return events, {
        "selector_matches": selected_count,
        "simulated_candidates": len(reverse_candidates),
        "accepted_trades": len(events),
        "reverse_only_skipped_overlap": reverse_only.get("skipped_overlap", 0),
        "params": {
            "selector": "guarded_weak_loss",
            "target_rr": target_rr,
            "max_hold_bars": max_hold_bars,
            "leverage": leverage,
            "stop_multiplier": stop_multiplier,
            "max_short_stop_pct": max_short_stop_pct,
            "allocation": allocation,
        },
    }


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
    stable_events, stable_summary = build_stable_events(
        payload,
        prepared,
        shadow_events,
        allocation=float(args.stable_allocation),
        target_rr=float(args.stable_target_rr),
        max_hold_bars=int(args.stable_max_hold_bars),
        leverage=float(args.stable_leverage),
        stop_multiplier=float(args.stable_stop_multiplier),
        max_short_stop_pct=float(args.stable_max_short_stop_pct),
    )

    reference = replay_base_priority_stable_first(
        base_events,
        stable_events,
        initial_capital,
        prepared.end,
        base_shadow_summary,
    )
    live_candidates = [to_candidate(event) for event in (base_events + stable_events)]
    accepted, decisions = replay_single_position_events(live_candidates)
    live_events = [candidate.metadata["event"] for candidate in accepted]
    live = standard_event_summary(live_events, initial_capital, "entry_idx")
    live = add_standard_windows(live, initial_capital, prepared.end, "entry_idx")
    live = add_combo_deltas(live, base_shadow_summary)
    live["combo_mode"] = "live_shadow_chronological"
    live["decision_counts"] = decision_counts(decisions)
    live["live_feasibility_audit"] = live_feasibility_audit(live, initial_capital)

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
            "stable_params": stable_summary["params"],
            "paper_log_output": str(Path(args.paper_log_output).resolve()),
        },
        "baseline_shadow_sota": {key: value for key, value in base_shadow_summary.items() if key != "events"},
        "candidate_generation": {
            "sota_candidates": len(base_events),
            "stable_candidates": len(stable_events),
            "stable_summary": stable_summary,
        },
        "reference_base_priority_stable_first": compact_combo_result(reference, int(args.sample_trades)),
        "live_shadow": compact_combo_result(live, int(args.sample_trades)),
        "stable_preempted_sota": stable_preempted_sota_summary(decisions),
        "decisions": decisions,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_paper_log(Path(args.paper_log_output), decisions)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    print(output)
    base = report["baseline_shadow_sota"]
    ref = report["reference_base_priority_stable_first"]
    live_payload = report["live_shadow"]
    print(f"Baseline full={base['total_return_pct']:.2f}%/{base['max_drawdown_pct']:.2f}% 2026={base['windows']['current_year']['total_return_pct']:.2f}%")
    print(f"Reference base-priority full={ref['total_return_pct']:.2f}%/{ref['max_drawdown_pct']:.2f}% 2026={ref['windows']['current_year']['total_return_pct']:.2f}%")
    print(f"Live-shadow full={live_payload['total_return_pct']:.2f}%/{live_payload['max_drawdown_pct']:.2f}% 2026={live_payload['windows']['current_year']['total_return_pct']:.2f}%")
    print(f"Decisions={live_payload['decision_counts']}")
    preempted = report["stable_preempted_sota"]
    print(
        f"Stable preempted SOTA: count={preempted['count']} "
        f"sota_return_sum={preempted['sota_return_sum_pct']:.2f}% "
        f"positive={preempted['positive_sota_blocked']} negative={preempted['negative_sota_blocked']}"
    )
    print(f"Paper log={Path(args.paper_log_output)}")


if __name__ == "__main__":
    main()
