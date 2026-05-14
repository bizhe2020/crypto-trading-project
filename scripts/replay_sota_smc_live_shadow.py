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

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.confirmed_multiframe_score_utils import (  # noqa: E402
    align_confirmed_mapping,
    passes_score_gate,
    resample_confirmed_1h,
    score_snapshot,
)
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.live_shadow_utils import (  # noqa: E402
    add_combo_deltas,
    add_standard_windows,
    clean_for_json,
    compact_combo_result,
    compact_result,
    event_stream_summary,
    standard_event_summary,
    standard_sota_event,
)
from scripts.score_bucket_sizing_utils import apply_score_bucket_sizing_to_events  # noqa: E402
from scripts.sota_long_filters import apply_sota_structure_gate  # noqa: E402
from scripts.report_smc_trade_context import daily_candles_from_4h  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402
from scripts.smc_live_utils import SMC_CASES, build_smc_events, leveraged_net_return, live_feasibility_audit, replay_base_priority_sota_first  # noqa: E402
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "sota_smc_live_shadow_replay.json"
DEFAULT_PAPER_LOG = ROOT / "var" / "high_leverage_expansion" / "sota_smc_live_shadow_paper_decisions.jsonl"
RR_MODE_CHOICES = ("close", "extreme")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live-shadow chronological replay for SOTA long + SMC short.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument(
        "--informative-asof-from-15m",
        action="store_true",
        help="Use primary-candle as-of 4h state to match live evaluation instead of finalized 4h candles.",
    )
    parser.add_argument(
        "--confirmed-4h-only",
        action="store_true",
        help="Use only the previous fully closed 4h candle state for each primary candle.",
    )
    parser.add_argument(
        "--replay-sync-entry-to-signal-price",
        action="store_true",
        help="Sync replay entry execution price back to signal entry price after open to emulate live exchange fill reconciliation.",
    )
    parser.add_argument("--smc-case", default="v2_medium_dispbody05_otherlag4_10x", choices=sorted(SMC_CASES))
    parser.add_argument("--smc-allocation", type=float, default=1.0)
    parser.add_argument("--enable-gap-smc-short", action="store_true")
    parser.add_argument("--gap-smc-case", default="gap_expansion_21d_other_3x", choices=sorted(SMC_CASES))
    parser.add_argument("--gap-smc-min-flat-days", type=float, default=21.0)
    parser.add_argument("--gap-smc-leverage", type=float, default=3.0)
    parser.add_argument("--gap-smc-max-stop-distance-pct", type=float, default=1.5)
    parser.add_argument("--enable-sota-score-gate", action="store_true")
    parser.add_argument("--sota-score-net-min", type=int, default=3)
    parser.add_argument("--sota-score-bull-min", type=int, default=8)
    parser.add_argument("--sota-score-bear-max", type=int, default=6)
    parser.add_argument("--sota-score-conflict-mode", default="any", choices=("any", "conflict", "clean"))
    parser.add_argument("--require-non-bearish-structure-for-long", action="store_true")
    parser.add_argument("--enable-long-score-bucket-sizing", action="store_true")
    parser.add_argument(
        "--long-score-bucket-sizing-rules-json",
        default="",
        help="Optional JSON array/dict for long score bucket sizing rules.",
    )
    parser.add_argument("--enable-sota-soft-stop-recovery-overlay", action="store_true")
    parser.add_argument("--sota-soft-stop-net-min", type=int, default=None)
    parser.add_argument("--sota-soft-stop-bear-max", type=int, default=None)
    parser.add_argument("--sota-soft-stop-max-leverage", type=float, default=None)
    parser.add_argument("--sota-soft-stop-buffer-r", type=float, default=None)
    parser.add_argument("--sota-soft-stop-target-rr", type=float, default=None)
    parser.add_argument("--sota-soft-stop-max-extension-bars", type=int, default=None)
    parser.add_argument("--sota-soft-stop-exclude-score-buckets", default=None)
    parser.add_argument("--stage-trigger-rr-mode", default="close", choices=RR_MODE_CHOICES)
    parser.add_argument("--time-trailing-rr-mode", default="extreme", choices=RR_MODE_CHOICES)
    parser.add_argument("--atr-activation-rr-mode", default="extreme", choices=RR_MODE_CHOICES)
    parser.add_argument("--enable-auto-time-based-trailing", default=None)
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=12.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=4)
    parser.add_argument("--sample-trades", type=int, default=40)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--paper-log-output", default=str(DEFAULT_PAPER_LOG))
    return parser.parse_args()


def apply_trailing_rr_modes(
    payload: dict[str, Any],
    *,
    stage_trigger_rr_mode: str,
    time_trailing_rr_mode: str,
    atr_activation_rr_mode: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    updated = dict(payload)
    updated["stage_trigger_rr_mode"] = stage_trigger_rr_mode
    updated["time_trailing_rr_mode"] = time_trailing_rr_mode
    updated["atr_activation_rr_mode"] = atr_activation_rr_mode
    return updated, {
        "stage_trigger_rr_mode": stage_trigger_rr_mode,
        "time_trailing_rr_mode": time_trailing_rr_mode,
        "atr_activation_rr_mode": atr_activation_rr_mode,
    }


def _parse_optional_bool(raw: Any) -> bool | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid bool value: {raw}")


def apply_auto_tit_override(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(payload)
    enable_auto_tit = _parse_optional_bool(args.enable_auto_time_based_trailing)
    if enable_auto_tit is not None:
        updated["enable_auto_time_based_trailing"] = enable_auto_tit
    return updated


def event_key(event: dict[str, Any]) -> str:
    return f"{event.get('event_type')}|{event.get('entry_idx')}|{event.get('exit_idx')}"


def sota_event_key(event: dict[str, Any]) -> str:
    return f"sota_long|{int(event.get('entry_idx', 0) or 0)}|{int(event.get('exit_idx', 0) or 0)}"


def timestamp_for_candle(candle: Any) -> str:
    return str(pd.Timestamp(candle.ts, unit="s", tz="UTC"))


def score_bucket_name(event: dict[str, Any]) -> str:
    decision = event.get("long_score_bucket_sizing")
    if not isinstance(decision, dict):
        return ""
    rule = decision.get("rule")
    if not isinstance(rule, dict):
        return ""
    return str(rule.get("name") or "")


def stop_zone(direction: str, entry_price: float, final_stop: float) -> str:
    if entry_price <= 0.0 or final_stop <= 0.0:
        return "unknown"
    if direction == "BULL":
        if final_stop > entry_price:
            return "profit_lock"
        if abs(final_stop - entry_price) / entry_price <= 0.0005:
            return "breakeven"
        return "loss_stop"
    if final_stop < entry_price:
        return "profit_lock"
    if abs(final_stop - entry_price) / entry_price <= 0.0005:
        return "breakeven"
    return "loss_stop"


def gross_unit_return(entry_price: float, exit_price: float, direction: str) -> float:
    if entry_price <= 0.0 or exit_price <= 0.0:
        return 0.0
    if direction == "BULL":
        return exit_price / entry_price - 1.0
    return entry_price / exit_price - 1.0


def inferred_roundtrip_cost(event: dict[str, Any]) -> float:
    leverage = float(event.get("source_effective_leverage", 0.0) or 0.0)
    entry_price = float(event.get("entry_price", 0.0) or 0.0)
    final_stop = float(event.get("final_stop_price", 0.0) or event.get("exit_price", 0.0) or 0.0)
    direction = str(event.get("direction") or "BULL")
    if leverage <= 0.0 or entry_price <= 0.0 or final_stop <= 0.0:
        return 0.002
    unit_realized = float(event.get("return", 0.0) or 0.0) / leverage
    gross = gross_unit_return(entry_price, final_stop, direction)
    return min(max(gross - unit_realized, 0.0), 0.01)


def leveraged_return(event: dict[str, Any], exit_price: float) -> float:
    leverage = float(event.get("source_effective_leverage", 0.0) or 0.0)
    entry_price = float(event.get("entry_price", 0.0) or 0.0)
    direction = str(event.get("direction") or "BULL")
    if leverage <= 0.0 or entry_price <= 0.0 or exit_price <= 0.0:
        return float(event.get("return", 0.0) or 0.0)
    return (gross_unit_return(entry_price, exit_price, direction) - inferred_roundtrip_cost(event)) * leverage


def _excluded_soft_stop_buckets(raw: str) -> set[str]:
    return {item.strip() for item in str(raw or "").split(",") if item.strip()}


def soft_stop_gate_matches(event: dict[str, Any], args: argparse.Namespace) -> bool:
    if str(event.get("event_type") or "") != "sota_long":
        return False
    if str(event.get("direction") or "") != "BULL":
        return False
    if str(event.get("exit_reason") or "") != "stop_loss":
        return False
    if float(event.get("return", 0.0) or 0.0) > 0.0:
        return False
    if float(event.get("source_effective_leverage", 0.0) or 0.0) > float(args.sota_soft_stop_max_leverage):
        return False
    if int(event.get("net_score", 0) or 0) < int(args.sota_soft_stop_net_min):
        return False
    if int(event.get("bear_total", 0) or 0) > int(args.sota_soft_stop_bear_max):
        return False
    if score_bucket_name(event) in _excluded_soft_stop_buckets(str(args.sota_soft_stop_exclude_score_buckets)):
        return False

    entry_price = float(event.get("entry_price", 0.0) or 0.0)
    final_stop = float(event.get("final_stop_price", 0.0) or event.get("exit_price", 0.0) or 0.0)
    return stop_zone(str(event.get("direction") or ""), entry_price, final_stop) in {"loss_stop", "breakeven"}


def simulate_sota_soft_stop(
    event: dict[str, Any],
    candles: list[Any],
    *,
    buffer_r: float,
    target_rr: float,
    max_extension_bars: int,
) -> dict[str, Any]:
    direction = str(event.get("direction") or "BULL")
    if direction != "BULL":
        return dict(event)

    entry_idx = int(event.get("entry_idx", 0) or 0)
    original_exit_idx = int(event.get("exit_idx", entry_idx) or entry_idx)
    start_idx = original_exit_idx
    end_idx = min(original_exit_idx + int(max_extension_bars), len(candles) - 1)
    if start_idx > end_idx or start_idx >= len(candles):
        return dict(event)

    entry_price = float(event.get("entry_price", 0.0) or 0.0)
    initial_stop = float(event.get("initial_stop_price", 0.0) or 0.0)
    risk_price = abs(entry_price - initial_stop)
    if risk_price <= 0.0:
        stop_distance_pct = float(event.get("stop_distance_pct", 0.0) or 0.0)
        risk_price = abs(entry_price) * stop_distance_pct / 100.0
    if entry_price <= 0.0 or risk_price <= 0.0:
        return dict(event)

    soft_stop_price = entry_price - (1.0 + float(buffer_r)) * risk_price
    target_price = entry_price + float(target_rr) * risk_price
    selected_idx = end_idx
    selected_price = float(candles[end_idx].c)
    selected_reason = "soft_stop_timeout"

    for idx in range(start_idx, end_idx + 1):
        candle = candles[idx]
        if float(candle.l) <= soft_stop_price:
            selected_idx = idx
            selected_price = soft_stop_price
            selected_reason = "soft_stop_extra_stop"
            break
        if idx == original_exit_idx:
            continue
        if float(candle.h) >= target_price:
            selected_idx = idx
            selected_price = target_price
            selected_reason = f"soft_stop_target_{float(target_rr):g}r"
            break

    updated = dict(event)
    old_return = float(event.get("return", 0.0) or 0.0)
    new_return = leveraged_return(event, selected_price)
    updated["exit_idx"] = selected_idx
    updated["exit_time"] = timestamp_for_candle(candles[selected_idx])
    updated["return"] = new_return
    updated["return_pct"] = round(new_return * 100.0, 4)
    updated["exit_reason"] = selected_reason
    updated["sota_soft_stop_recovery_overlay"] = {
        "applied": True,
        "buffer_r": float(buffer_r),
        "target_rr": float(target_rr),
        "max_extension_bars": int(max_extension_bars),
        "original_exit_idx": original_exit_idx,
        "original_exit_time": event.get("exit_time"),
        "original_return_pct": round(old_return * 100.0, 4),
        "new_exit_idx": selected_idx,
        "new_exit_time": updated["exit_time"],
        "new_return_pct": round(new_return * 100.0, 4),
        "risk_price": round(risk_price, 6),
        "soft_stop_price": round(soft_stop_price, 6),
        "target_price": round(target_price, 6),
        "exit_price": round(selected_price, 6),
        "improved": new_return > old_return,
    }
    return updated


def apply_sota_soft_stop_recovery_overlay(
    events: list[dict[str, Any]],
    full_sota_by_key: dict[str, dict[str, Any]],
    candles: list[Any],
    args: argparse.Namespace,
    *,
    enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rule = {
        "enabled": bool(enabled),
        "net_min": int(args.sota_soft_stop_net_min),
        "bear_max": int(args.sota_soft_stop_bear_max),
        "max_leverage": float(args.sota_soft_stop_max_leverage),
        "buffer_r": float(args.sota_soft_stop_buffer_r),
        "target_rr": float(args.sota_soft_stop_target_rr),
        "max_extension_bars": int(args.sota_soft_stop_max_extension_bars),
        "exclude_score_buckets": sorted(_excluded_soft_stop_buckets(str(args.sota_soft_stop_exclude_score_buckets))),
        "stop_zones": ["loss_stop", "breakeven"],
    }
    diagnostics: dict[str, Any] = {
        "enabled": bool(enabled),
        "rule": rule,
        "selected": 0,
        "improved": 0,
        "worsened_or_equal": 0,
        "missing_full_event": 0,
        "exit_reasons": {},
        "samples": [],
    }
    if not enabled:
        return events, diagnostics

    adjusted: list[dict[str, Any]] = []
    exit_reasons: dict[str, int] = {}
    for event in events:
        if str(event.get("event_type") or "") != "sota_long":
            adjusted.append(dict(event))
            continue
        full = full_sota_by_key.get(sota_event_key(event))
        if full is None:
            diagnostics["missing_full_event"] += 1
            adjusted.append(dict(event))
            continue
        merged = dict(full)
        merged.update(event)
        merged["event_type"] = "sota_long"
        if not soft_stop_gate_matches(merged, args):
            adjusted.append(dict(event))
            continue

        updated = simulate_sota_soft_stop(
            merged,
            candles,
            buffer_r=float(args.sota_soft_stop_buffer_r),
            target_rr=float(args.sota_soft_stop_target_rr),
            max_extension_bars=int(args.sota_soft_stop_max_extension_bars),
        )
        diagnostics["selected"] += 1
        exit_reason = str(updated.get("exit_reason") or "unknown")
        exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
        if float(updated.get("return", 0.0) or 0.0) > float(event.get("return", 0.0) or 0.0):
            diagnostics["improved"] += 1
        else:
            diagnostics["worsened_or_equal"] += 1
        if len(diagnostics["samples"]) < 12:
            overlay = updated.get("sota_soft_stop_recovery_overlay") or {}
            diagnostics["samples"].append(
                {
                    "event_key": sota_event_key(event),
                    "entry_time": event.get("entry_time"),
                    "old_exit_time": event.get("exit_time"),
                    "new_exit_time": updated.get("exit_time"),
                    "old_return_pct": overlay.get("original_return_pct"),
                    "new_return_pct": overlay.get("new_return_pct"),
                    "new_exit_reason": updated.get("exit_reason"),
                    "net_score": merged.get("net_score"),
                    "bull_total": merged.get("bull_total"),
                    "bear_total": merged.get("bear_total"),
                    "regime_label": merged.get("regime_label"),
                    "source_effective_leverage": merged.get("source_effective_leverage"),
                }
            )
        adjusted.append(updated)

    diagnostics["exit_reasons"] = exit_reasons
    return adjusted, diagnostics


def priority_value(event: dict[str, Any]) -> int:
    priority = {
        "sota_long": 0,
        "smc_short": 1,
        "gap_smc_short_expansion": 2,
    }
    return priority.get(str(event.get("event_type") or ""), 9)


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


def replay_live_shadow(
    candidates: list[dict[str, Any]],
    initial_capital: float,
    data_end: pd.Timestamp,
    baseline: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            int(item.get("entry_idx", 0) or 0),
            priority_value(item),
            int(item.get("exit_idx", 0) or 0),
        ),
    )
    accepted: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    active_until_idx = -1
    active_event: dict[str, Any] | None = None

    for event in ordered:
        entry_idx = int(event.get("entry_idx", 0) or 0)
        decision: dict[str, Any] = {
            "event_key": event_key(event),
            "event_type": event.get("event_type"),
            "entry_idx": entry_idx,
            "exit_idx": int(event.get("exit_idx", entry_idx) or entry_idx),
            "entry_time": event.get("entry_time"),
            "exit_time": event.get("exit_time"),
            "direction": event.get("direction"),
            "return_pct": round(float(event.get("return", 0.0) or 0.0) * 100.0, 4),
        }
        if active_event is not None and entry_idx < active_until_idx:
            decision |= {
                "decision": "rejected",
                "reason": "position_lock_open",
                "blocking_event_key": event_key(active_event),
                "blocking_event_type": active_event.get("event_type"),
                "blocking_exit_idx": int(active_event.get("exit_idx", active_until_idx) or active_until_idx),
                "blocking_exit_time": active_event.get("exit_time"),
            }
            decisions.append(decision)
            continue

        accepted.append(event)
        active_event = event
        active_until_idx = max(entry_idx, int(event.get("exit_idx", entry_idx) or entry_idx))
        decision |= {
            "decision": "accepted",
            "reason": "priority_available",
            "paper_tag": f"accepted_{event.get('event_type')}",
        }
        decisions.append(decision)

    result = standard_event_summary(accepted, initial_capital, "entry_idx")
    result = add_standard_windows(result, initial_capital, data_end, "entry_idx")
    result = add_combo_deltas(result, baseline)
    result["combo_mode"] = "live_shadow_chronological"
    result["decision_counts"] = decision_counts(decisions)
    result["live_feasibility_audit"] = live_feasibility_audit(result, initial_capital)
    return result, decisions


def compact_live_result(result: dict[str, Any], sample_trades: int) -> dict[str, Any]:
    payload = compact_combo_result(result, sample_trades)
    payload["events"] = result.get("events", [])
    payload["decision_counts"] = result.get("decision_counts", {})
    payload["live_feasibility_audit"] = result.get("live_feasibility_audit", {})
    return payload


def compact_combo_with_events(result: dict[str, Any], sample_trades: int) -> dict[str, Any]:
    payload = compact_combo_result(result, sample_trades)
    payload["events"] = result.get("events", [])
    return payload


def _normalize_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _row_in_gap(row: dict[str, Any], gaps: list[dict[str, Any]], min_gap_days: float) -> dict[str, Any] | None:
    ts = _normalize_ts(row["entry_time"])
    for gap in gaps:
        start = _normalize_ts(gap["gap_start"])
        if start <= ts < _normalize_ts(gap["gap_end"]):
            elapsed = (ts - start).total_seconds() / 86400.0
            if elapsed >= float(min_gap_days):
                return {**gap, "gap_elapsed_days_at_entry": round(elapsed, 4)}
    return None


def _select_gap_smc_rows(
    rows: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    *,
    min_gap_days: float,
    max_stop_distance_pct: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_gap_keys: set[tuple[str, str]] = set()
    for row in sorted(rows, key=lambda item: _normalize_ts(item["entry_time"])):
        if str(row.get("direction") or "") != "BEAR":
            continue
        gap = _row_in_gap(row, gaps, min_gap_days)
        if gap is None:
            continue
        stop_distance = float(row.get("stop_distance_pct", 0.0) or 0.0)
        if float(max_stop_distance_pct) > 0.0 and stop_distance > float(max_stop_distance_pct):
            continue
        gap_key = (str(gap["gap_start"]), str(gap["gap_end"]))
        if gap_key in used_gap_keys:
            continue
        selected.append(
            {
                **row,
                "gap_start": gap["gap_start"],
                "gap_end": gap["gap_end"],
                "gap_bucket": gap.get("bucket"),
                "gap_days": gap.get("gap_days"),
                "gap_elapsed_days_at_entry": gap.get("gap_elapsed_days_at_entry"),
            }
        )
        used_gap_keys.add(gap_key)
    return selected


def _gap_smc_row_to_event(row: dict[str, Any], *, leverage: float, variant: dict[str, Any], taker_fee_rate: float, slippage_bps: float) -> dict[str, Any]:
    model = leveraged_net_return(
        signal_return_pct=float(row.get("signal_return_pct", 0.0) or 0.0),
        leverage=float(leverage),
        position_size_pct=1.0,
        allocation=1.0,
        taker_fee_rate=float(taker_fee_rate),
        slippage_bps=float(slippage_bps),
    )
    return {
        "event_type": "gap_smc_short_expansion",
        "entry_idx": int(row["entry_idx"]),
        "exit_idx": int(row["exit_idx"]),
        "entry_time": str(_normalize_ts(row["entry_time"])),
        "exit_time": str(_normalize_ts(row["exit_time"])),
        "direction": "BEAR",
        "return": float(model["account_return"]),
        "return_pct": round(float(model["account_return"]) * 100.0, 4),
        "exit_reason": str(row.get("outcome") or row.get("status") or "unknown"),
        "source_effective_leverage": float(leverage),
        "smc_signal_return_pct": round(float(row.get("signal_return_pct", 0.0) or 0.0), 4),
        "smc_unit_return_pct": round(float(model["net_unit_return_pct"]), 4),
        "smc_time_bucket": str(row.get("time_bucket") or ""),
        "smc_mss_lag_bars": row.get("mss_lag_bars"),
        "smc_stop_distance_pct": row.get("stop_distance_pct"),
        "gap_start": row.get("gap_start"),
        "gap_end": row.get("gap_end"),
        "gap_bucket": row.get("gap_bucket"),
        "gap_days": row.get("gap_days"),
        "gap_elapsed_days_at_entry": row.get("gap_elapsed_days_at_entry"),
        "gap_smc_short_expansion": {"variant": variant},
    }


def build_gap_smc_events(
    args: argparse.Namespace,
    prepared: Any,
    daily: list[Any],
    h4_highs: list[int],
    h4_lows: list[int],
    d1_highs: list[int],
    d1_lows: list[int],
    reference_events: list[dict[str, Any]],
    *,
    taker_fee_rate: float,
    slippage_bps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not bool(args.enable_gap_smc_short):
        return [], {"enabled": False, "accepted_trades": 0}
    from scripts.report_high_value_frequency_gaps import build_gap_rows
    from scripts.smc_live_utils import smc_case_namespace
    from scripts.research_smc_standalone_v1 import apply_max_open_positions, build_event_scan_args, scan_events, trade_rows_for_events

    case_params = dict(SMC_CASES[str(args.gap_smc_case)])
    case_args = smc_case_namespace(args, case_params)
    case_args.leverage = float(args.gap_smc_leverage)
    case_args.position_size_pct = 1.0
    for key, value in {
        "swing_lookback": 80,
        "liquidity_lookback_bars": 192,
        "mss_lookahead_bars": 24,
        "fvg_lookback_bars": 8,
        "outcome_lookahead_bars": 96,
        "atr_period": 14,
        "stop_buffer_atr": 0.05,
        "require_ote_touch": False,
        "bull_min_displacement_body_atr": 0.0,
        "bull_max_displacement_body_atr": 0.0,
        "bull_min_displacement_range_atr": 0.0,
        "bull_max_displacement_range_atr": 0.0,
        "min_fvg_size_pct": 0.0,
        "max_fvg_fill_pct": 0.0,
        "bear_min_sweep_distance_pct": 0.0,
        "bear_require_fvg_touch": False,
        "bear_min_fvg_size_pct": 0.0,
        "position_risk_fraction": 1.0,
    }.items():
        if not hasattr(case_args, key):
            setattr(case_args, key, value)
    raw_events = scan_events(prepared.c15m, build_event_scan_args(case_args))
    rows = trade_rows_for_events(raw_events, prepared, daily, h4_highs, h4_lows, d1_highs, d1_lows, case_args)
    rows, slot_skipped = apply_max_open_positions(rows, int(case_args.max_open_positions))
    gaps = build_gap_rows(sorted(reference_events, key=lambda item: int(item.get("entry_idx", 0) or 0)), 0.0)
    selected_rows = _select_gap_smc_rows(
        rows,
        gaps,
        min_gap_days=float(args.gap_smc_min_flat_days),
        max_stop_distance_pct=float(args.gap_smc_max_stop_distance_pct),
    )
    variant = {
        "case": str(args.gap_smc_case),
        "min_gap_days": float(args.gap_smc_min_flat_days),
        "candidate_time_buckets": str(case_params.get("allowed_time_buckets", "other")),
        "require_h4_bias_align": bool(case_params.get("require_h4_bias_align", False)),
        "require_confirmed_retest": bool(case_params.get("require_confirmed_retest", False)),
        "require_fvg_touch": bool(case_params.get("require_fvg_touch", True)),
        "allow_ote_only": bool(case_params.get("allow_ote_only", False)),
        "min_displacement_body_atr": float(case_params.get("min_displacement_body_atr", 0.5)),
        "max_mss_lag_bars": int(case_params.get("max_mss_lag_bars", 15)),
        "max_stop_distance_pct": float(args.gap_smc_max_stop_distance_pct),
        "min_signal_return_pct": -999.0,
        "leverage": float(args.gap_smc_leverage),
        "per_gap_limit": 1,
    }
    gap_events = [
        _gap_smc_row_to_event(
            row,
            leverage=float(args.gap_smc_leverage),
            variant=variant,
            taker_fee_rate=float(taker_fee_rate),
            slippage_bps=float(slippage_bps),
        )
        for row in selected_rows
    ]
    for event in gap_events:
        event["event_type"] = "gap_smc_short_expansion"
        event["smc_case"] = str(args.gap_smc_case)
        event["smc_taker_fee_rate"] = float(taker_fee_rate)
        event["smc_slippage_bps"] = float(slippage_bps)
    return gap_events, {
        "enabled": True,
        "case": str(args.gap_smc_case),
        "min_flat_days": float(args.gap_smc_min_flat_days),
        "max_stop_distance_pct": float(args.gap_smc_max_stop_distance_pct),
        "rows_after_filters": len(rows),
        "slot_skipped": int(slot_skipped),
        "accepted_trades": len(gap_events),
        "standalone_event_stats": event_stream_summary(gap_events, float(args.smc_allocation), prepared.end),
    }


def reference_gap(reference: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    reference_keys = {event_key(event) for event in reference.get("events", [])}
    live_keys = {event_key(event) for event in live.get("events", [])}
    return {
        "reference_total_return_pct": reference.get("total_return_pct"),
        "live_total_return_pct": live.get("total_return_pct"),
        "return_gap_pct": round(float(live.get("total_return_pct", 0.0) or 0.0) - float(reference.get("total_return_pct", 0.0) or 0.0), 4),
        "reference_max_drawdown_pct": reference.get("max_drawdown_pct"),
        "live_max_drawdown_pct": live.get("max_drawdown_pct"),
        "dd_gap_pct": round(float(live.get("max_drawdown_pct", 0.0) or 0.0) - float(reference.get("max_drawdown_pct", 0.0) or 0.0), 4),
        "accepted_only_in_reference": sorted(reference_keys - live_keys),
        "accepted_only_in_live": sorted(live_keys - reference_keys),
    }


def write_paper_log(path: Path, decisions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(clean_for_json(decision), ensure_ascii=False, allow_nan=False) + "\n")


def parse_score_bucket_rules(raw: str) -> Any:
    if not raw:
        return None
    return json.loads(raw)


def resolve_sota_soft_stop_args(args: argparse.Namespace, payload: dict[str, Any]) -> tuple[bool, argparse.Namespace]:
    enabled = bool(args.enable_sota_soft_stop_recovery_overlay) or bool(
        payload.get("enable_sota_soft_stop_recovery_overlay_live", False)
    )
    args.sota_soft_stop_net_min = int(
        args.sota_soft_stop_net_min
        if args.sota_soft_stop_net_min is not None
        else payload.get("sota_soft_stop_net_min", 15)
    )
    args.sota_soft_stop_bear_max = int(
        args.sota_soft_stop_bear_max
        if args.sota_soft_stop_bear_max is not None
        else payload.get("sota_soft_stop_bear_max", 0)
    )
    args.sota_soft_stop_max_leverage = float(
        args.sota_soft_stop_max_leverage
        if args.sota_soft_stop_max_leverage is not None
        else payload.get("sota_soft_stop_max_leverage", 2.0)
    )
    args.sota_soft_stop_buffer_r = float(
        args.sota_soft_stop_buffer_r
        if args.sota_soft_stop_buffer_r is not None
        else payload.get("sota_soft_stop_buffer_r", 1.0)
    )
    args.sota_soft_stop_target_rr = float(
        args.sota_soft_stop_target_rr
        if args.sota_soft_stop_target_rr is not None
        else payload.get("sota_soft_stop_target_rr", 0.0)
    )
    args.sota_soft_stop_max_extension_bars = int(
        args.sota_soft_stop_max_extension_bars
        if args.sota_soft_stop_max_extension_bars is not None
        else payload.get("sota_soft_stop_max_extension_bars", 4)
    )
    raw_excluded = (
        args.sota_soft_stop_exclude_score_buckets
        if args.sota_soft_stop_exclude_score_buckets is not None
        else payload.get("sota_soft_stop_exclude_score_buckets", "bear_total_6_20x_boost")
    )
    if isinstance(raw_excluded, list):
        args.sota_soft_stop_exclude_score_buckets = ",".join(str(item) for item in raw_excluded)
    else:
        args.sota_soft_stop_exclude_score_buckets = str(raw_excluded or "")
    return enabled, args


def apply_sota_score_gate(
    prepared: Any,
    sota_events: list[dict[str, Any]],
    *,
    enabled: bool,
    net_min: int,
    bull_min: int,
    bear_max: int,
    conflict_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not enabled:
        return sota_events, {
            "enabled": False,
            "rule": {
                "net_min": int(net_min),
                "bull_min": int(bull_min),
                "bear_max": int(bear_max),
                "conflict_mode": str(conflict_mode),
            },
            "original_candidates": len(sota_events),
            "filtered_candidates": len(sota_events),
            "removed_candidates": 0,
        }

    c1h = resample_confirmed_1h(prepared.c15m)
    mapping_1h = align_confirmed_mapping(c1h, prepared.c15m)
    filtered: list[dict[str, Any]] = []
    removed = 0
    for event in sota_events:
        entry_idx = int(event.get("entry_idx", 0) or 0)
        snapshot = score_snapshot(prepared, c1h, mapping_1h, entry_idx)
        enriched = dict(event)
        enriched.update(asdict(snapshot))
        if passes_score_gate(
            enriched,
            net_min=int(net_min),
            bull_min=int(bull_min),
            bear_max=int(bear_max),
            conflict_mode=str(conflict_mode),
        ):
            filtered.append(enriched)
        else:
            removed += 1
    return filtered, {
        "enabled": True,
        "rule": {
            "net_min": int(net_min),
            "bull_min": int(bull_min),
            "bear_max": int(bear_max),
            "conflict_mode": str(conflict_mode),
        },
        "original_candidates": len(sota_events),
        "filtered_candidates": len(filtered),
        "removed_candidates": removed,
        "candles_1h": len(c1h),
    }


def main() -> None:
    args = parse_args()
    base_payload = load_config_payload(Path(args.config))
    sota_soft_stop_enabled, args = resolve_sota_soft_stop_args(args, base_payload)
    payload, pressure_params = apply_pressure_params(base_payload, Path(args.pressure_params))
    payload, trailing_rr_modes = apply_trailing_rr_modes(
        payload,
        stage_trigger_rr_mode=str(args.stage_trigger_rr_mode),
        time_trailing_rr_mode=str(args.time_trailing_rr_mode),
        atr_activation_rr_mode=str(args.atr_activation_rr_mode),
    )
    payload = apply_auto_tit_override(payload, args)
    payload["replay_sync_entry_to_signal_price"] = bool(args.replay_sync_entry_to_signal_price)
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=payload.get("regime_switcher_thresholds"),
        informative_asof_from_15m=bool(args.informative_asof_from_15m),
        confirmed_4h_only=bool(args.confirmed_4h_only),
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
    full_sota_by_key = {sota_event_key(event): dict(event) for event in shadow_events}
    base_shadow_summary = event_stream_summary(shadow_events, initial_capital, prepared.end)
    raw_base_events = [standard_sota_event(event) for event in shadow_events]
    base_events, sota_score_gate = apply_sota_score_gate(
        prepared,
        raw_base_events,
        enabled=bool(args.enable_sota_score_gate),
        net_min=int(args.sota_score_net_min),
        bull_min=int(args.sota_score_bull_min),
        bear_max=int(args.sota_score_bear_max),
        conflict_mode=str(args.sota_score_conflict_mode),
    )
    base_events, sota_structure_gate = apply_sota_structure_gate(
        base_events,
        enabled=bool(args.require_non_bearish_structure_for_long),
    )
    base_events, long_score_bucket_sizing = apply_score_bucket_sizing_to_events(
        base_events,
        enabled=bool(args.enable_long_score_bucket_sizing),
        rules=parse_score_bucket_rules(str(args.long_score_bucket_sizing_rules_json)),
    )
    base_events, sota_soft_stop_recovery_overlay = apply_sota_soft_stop_recovery_overlay(
        base_events,
        full_sota_by_key,
        prepared.c15m,
        args,
        enabled=bool(sota_soft_stop_enabled),
    )
    gated_shadow_summary = event_stream_summary(base_events, initial_capital, prepared.end)

    daily = daily_candles_from_4h(prepared.c4h)
    h4_highs, h4_lows = precompute_swings(prepared.c4h, n=2, lookback=80)
    d1_highs, d1_lows = precompute_swings(daily, n=2, lookback=20)
    smc_events, smc_summary = build_smc_events(
        args.smc_case,
        SMC_CASES[str(args.smc_case)],
        args,
        prepared,
        daily,
        h4_highs,
        h4_lows,
        d1_highs,
        d1_lows,
        float(args.smc_allocation),
        taker_fee_rate=float(payload.get("taker_fee_rate", 0.0005) or 0.0),
        slippage_bps=float(payload.get("slippage_bps", 0.0) or 0.0),
    )
    reference = replay_base_priority_sota_first(
        base_events,
        smc_events,
        initial_capital,
        prepared.end,
        gated_shadow_summary,
    )
    gap_smc_events, gap_smc_summary = build_gap_smc_events(
        args,
        prepared,
        daily,
        h4_highs,
        h4_lows,
        d1_highs,
        d1_lows,
        list(reference.get("events", [])),
        taker_fee_rate=float(payload.get("taker_fee_rate", 0.0005) or 0.0),
        slippage_bps=float(payload.get("slippage_bps", 0.0) or 0.0),
    )
    live, decisions = replay_live_shadow(
        base_events + smc_events + gap_smc_events,
        initial_capital,
        prepared.end,
        gated_shadow_summary,
    )

    report = {
        "metadata": {
            "config": str(Path(args.config).resolve()),
            "pressure_params": str(Path(args.pressure_params).resolve()),
            "pressure_params_applied": pressure_params,
            "trailing_rr_modes": trailing_rr_modes,
            "start_date": args.start_date,
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
            "informative_asof_from_15m": bool(args.informative_asof_from_15m),
            "confirmed_4h_only": bool(args.confirmed_4h_only),
            "replay_sync_entry_to_signal_price": bool(args.replay_sync_entry_to_signal_price),
            "smc_case": args.smc_case,
            "smc_allocation": args.smc_allocation,
            "gap_smc_short": gap_smc_summary,
            "sota_score_gate": sota_score_gate,
            "sota_structure_gate": sota_structure_gate,
            "long_score_bucket_sizing": long_score_bucket_sizing,
            "sota_soft_stop_recovery_overlay": sota_soft_stop_recovery_overlay,
            "paper_log_output": str(Path(args.paper_log_output).resolve()),
        },
        "baseline_shadow_sota": compact_result(base_shadow_summary, 0),
        "gated_shadow_sota": compact_result(gated_shadow_summary, 0),
        "candidate_generation": {
            "raw_sota_candidates": len(raw_base_events),
            "sota_candidates": len(base_events),
            "smc_candidates": len(smc_events),
            "gap_smc_short_candidates": len(gap_smc_events),
            "sota_score_gate": sota_score_gate,
            "sota_structure_gate": sota_structure_gate,
            "long_score_bucket_sizing": long_score_bucket_sizing,
            "sota_soft_stop_recovery_overlay": sota_soft_stop_recovery_overlay,
            "smc_summary": smc_summary,
            "gap_smc_summary": gap_smc_summary,
        },
        "reference_base_priority_sota_first": compact_combo_with_events(reference, int(args.sample_trades)),
        "live_shadow": compact_live_result(live, int(args.sample_trades)),
        "reference_gap": reference_gap(reference, live),
        "decisions": decisions,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_paper_log(Path(args.paper_log_output), decisions)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    print(output)
    base = report["baseline_shadow_sota"]
    ref = report["reference_base_priority_sota_first"]
    live_payload = report["live_shadow"]
    gap = report["reference_gap"]
    print(f"Baseline full={base['total_return_pct']:.2f}%/{base['max_drawdown_pct']:.2f}% 2026={base['windows']['current_year']['total_return_pct']:.2f}%")
    print(f"Reference base-priority full={ref['total_return_pct']:.2f}%/{ref['max_drawdown_pct']:.2f}% 2026={ref['windows']['current_year']['total_return_pct']:.2f}%")
    print(f"Live-shadow full={live_payload['total_return_pct']:.2f}%/{live_payload['max_drawdown_pct']:.2f}% 2026={live_payload['windows']['current_year']['total_return_pct']:.2f}%")
    print(f"Live gap vs reference: return={gap['return_gap_pct']:.2f}% dd={gap['dd_gap_pct']:+.2f}")
    print(f"Decisions={live_payload['decision_counts']}")
    print(f"Paper log={Path(args.paper_log_output)}")


if __name__ == "__main__":
    main()
