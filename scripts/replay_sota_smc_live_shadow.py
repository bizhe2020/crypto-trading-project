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
from scripts.report_smc_trade_context import daily_candles_from_4h  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402
from scripts.smc_live_utils import SMC_CASES, build_smc_events, live_feasibility_audit, replay_base_priority_sota_first  # noqa: E402
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
    parser.add_argument("--enable-sota-score-gate", action="store_true")
    parser.add_argument("--sota-score-net-min", type=int, default=3)
    parser.add_argument("--sota-score-bull-min", type=int, default=8)
    parser.add_argument("--sota-score-bear-max", type=int, default=6)
    parser.add_argument("--sota-score-conflict-mode", default="any", choices=("any", "conflict", "clean"))
    parser.add_argument("--stage-trigger-rr-mode", default="close", choices=RR_MODE_CHOICES)
    parser.add_argument("--time-trailing-rr-mode", default="extreme", choices=RR_MODE_CHOICES)
    parser.add_argument("--atr-activation-rr-mode", default="extreme", choices=RR_MODE_CHOICES)
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


def event_key(event: dict[str, Any]) -> str:
    return f"{event.get('event_type')}|{event.get('entry_idx')}|{event.get('exit_idx')}"


def priority_value(event: dict[str, Any]) -> int:
    priority = {
        "sota_long": 0,
        "smc_short": 1,
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
    payload, pressure_params = apply_pressure_params(base_payload, Path(args.pressure_params))
    payload, trailing_rr_modes = apply_trailing_rr_modes(
        payload,
        stage_trigger_rr_mode=str(args.stage_trigger_rr_mode),
        time_trailing_rr_mode=str(args.time_trailing_rr_mode),
        atr_activation_rr_mode=str(args.atr_activation_rr_mode),
    )
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
    live, decisions = replay_live_shadow(
        base_events + smc_events,
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
            "sota_score_gate": sota_score_gate,
            "paper_log_output": str(Path(args.paper_log_output).resolve()),
        },
        "baseline_shadow_sota": compact_result(base_shadow_summary, 0),
        "gated_shadow_sota": compact_result(gated_shadow_summary, 0),
        "candidate_generation": {
            "raw_sota_candidates": len(raw_base_events),
            "sota_candidates": len(base_events),
            "smc_candidates": len(smc_events),
            "sota_score_gate": sota_score_gate,
            "smc_summary": smc_summary,
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
