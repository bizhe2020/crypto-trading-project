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
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.replay_stable_live_shadow import (  # noqa: E402
    add_combo_deltas,
    add_standard_windows,
    build_stable_events,
    clean_for_json,
    compact_combo_result,
    decision_counts,
    event_stream_summary,
    live_feasibility_audit,
    stable_preempted_sota_summary,
    standard_event_summary,
    standard_sota_event,
    to_candidate,
    write_paper_log,
)
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402
from scripts.smc_short_event_builder import FORMAL_SMC_CASE_NAMES, SMC_CASES, build_smc_events, daily_candles_from_4h  # noqa: E402
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402
from strategy.sota_overlay_state import replay_single_position_events  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "stable_smc_live_shadow_replay.json"
DEFAULT_PAPER_LOG = ROOT / "var" / "high_leverage_expansion" / "stable_smc_live_shadow_paper_decisions.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay SOTA + Stable reverse-short + SMC short in chronological single-position live-shadow mode.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=0)
    parser.add_argument("--smc-case", default=None, choices=sorted(SMC_CASES), help="Back-compat single SMC case override.")
    parser.add_argument("--smc-cases", default="formal", help="Comma-separated SMC cases, or 'formal'/'all'. Ignored when --smc-case is set.")
    parser.add_argument("--smc-allocation", type=float, default=None, help="Back-compat single SMC allocation override.")
    parser.add_argument("--smc-allocation-values", default="1.0", help="Comma-separated allocation values. Ignored when --smc-allocation is set.")
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


def overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_entry = int(a.get("entry_idx", 0) or 0)
    a_exit = int(a.get("exit_idx", a_entry) or a_entry)
    b_entry = int(b.get("entry_idx", 0) or 0)
    b_exit = int(b.get("exit_idx", b_entry) or b_entry)
    return a_entry < b_exit and a_exit > b_entry


def filter_overlay_for_blockers(
    blockers: list[dict[str, Any]],
    overlay_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    accepted: list[dict[str, Any]] = []
    skipped = 0
    for event in sorted(overlay_events, key=lambda item: (int(item.get("entry_idx", 0) or 0), int(item.get("exit_idx", 0) or 0))):
        if any(overlaps(event, blocker) for blocker in blockers):
            skipped += 1
            continue
        accepted.append(event)
    return accepted, skipped


def replay_base_priority_stable_smc(
    base_events: list[dict[str, Any]],
    stable_events: list[dict[str, Any]],
    smc_events: list[dict[str, Any]],
    initial_capital: float,
    data_end: pd.Timestamp,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    stable_filtered, stable_skipped = filter_overlay_for_blockers(base_events, stable_events)
    smc_filtered, smc_skipped = filter_overlay_for_blockers(base_events + stable_filtered, smc_events)
    result = standard_event_summary(base_events + stable_filtered + smc_filtered, initial_capital, "entry_idx")
    result = add_standard_windows(result, initial_capital, data_end, "entry_idx")
    result = add_combo_deltas(result, baseline)
    result["combo_mode"] = "base_priority_stable_smc"
    result["base_priority_overlay_skipped"] = {
        "stable_reverse_short": stable_skipped,
        "smc_short": smc_skipped,
    }
    return result


def reference_gap(reference: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_total_return_pct": reference.get("total_return_pct"),
        "live_total_return_pct": live.get("total_return_pct"),
        "return_gap_pct": round(float(live.get("total_return_pct", 0.0) or 0.0) - float(reference.get("total_return_pct", 0.0) or 0.0), 4),
        "reference_max_drawdown_pct": reference.get("max_drawdown_pct"),
        "live_max_drawdown_pct": live.get("max_drawdown_pct"),
        "dd_gap_pct": round(float(live.get("max_drawdown_pct", 0.0) or 0.0) - float(reference.get("max_drawdown_pct", 0.0) or 0.0), 4),
    }


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def selected_smc_case_names(args: argparse.Namespace) -> list[str]:
    if args.smc_case:
        return [str(args.smc_case)]
    raw = str(args.smc_cases or "formal").strip()
    lowered = raw.lower()
    if lowered == "formal":
        return list(FORMAL_SMC_CASE_NAMES)
    if lowered == "all":
        return sorted(SMC_CASES)
    names = parse_csv(raw)
    unknown = [name for name in names if name not in SMC_CASES]
    if unknown:
        raise ValueError(f"Unknown SMC cases: {unknown}. Available: {sorted(SMC_CASES)}")
    if not names:
        raise ValueError("No SMC cases selected.")
    return names


def selected_smc_allocations(args: argparse.Namespace) -> list[float]:
    if args.smc_allocation is not None:
        return [float(args.smc_allocation)]
    values = [float(item) for item in parse_csv(str(args.smc_allocation_values or "1.0"))]
    if not values:
        raise ValueError("No SMC allocation values selected.")
    return values


def current_year_return_pct(result: dict[str, Any]) -> float:
    window = result.get("windows", {}).get("current_year", {})
    return float(window.get("total_return_pct", 0.0) or 0.0)


def live_candidate_score(live: dict[str, Any], baseline: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    live_return = float(live.get("total_return_pct", 0.0) or 0.0)
    live_dd = float(live.get("max_drawdown_pct", 0.0) or 0.0)
    baseline_dd = float(baseline.get("max_drawdown_pct", 0.0) or 0.0)
    year_return = current_year_return_pct(live)
    dd_penalty = max(0.0, live_dd - baseline_dd) * 10000.0
    score = live_return + year_return * 250.0 - dd_penalty
    return round(score, 4), {
        "live_total_return_pct": round(live_return, 4),
        "live_current_year_return_pct": round(year_return, 4),
        "live_max_drawdown_pct": round(live_dd, 4),
        "baseline_max_drawdown_pct": round(baseline_dd, 4),
        "dd_penalty": round(dd_penalty, 4),
        "score_formula": "live_total_return_pct + current_year_return_pct * 250 - max(0, live_dd - baseline_dd) * 10000",
    }


def build_live_shadow_result(
    events: list[dict[str, Any]],
    initial_capital: float,
    data_end: pd.Timestamp,
    baseline: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    accepted, decisions = replay_single_position_events([to_candidate(event) for event in events])
    live_events = [candidate.metadata["event"] for candidate in accepted]
    live = standard_event_summary(live_events, initial_capital, "entry_idx")
    live = add_standard_windows(live, initial_capital, data_end, "entry_idx")
    live = add_combo_deltas(live, baseline)
    live["combo_mode"] = "live_shadow_chronological"
    live["decision_counts"] = decision_counts(decisions)
    live["live_feasibility_audit"] = live_feasibility_audit(live, initial_capital)
    return live, decisions


def evaluate_smc_candidate(
    *,
    case_name: str,
    allocation: float,
    args: argparse.Namespace,
    payload: dict[str, Any],
    prepared: Any,
    daily: list[Any],
    h4_highs: list[int],
    h4_lows: list[int],
    d1_highs: list[int],
    d1_lows: list[int],
    base_events: list[dict[str, Any]],
    stable_events: list[dict[str, Any]],
    stable_summary: dict[str, Any],
    initial_capital: float,
    baseline: dict[str, Any],
) -> dict[str, Any]:
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
    reference = replay_base_priority_stable_smc(
        base_events,
        stable_events,
        smc_events,
        initial_capital,
        prepared.end,
        baseline,
    )
    live, decisions = build_live_shadow_result(
        base_events + stable_events + smc_events,
        initial_capital,
        prepared.end,
        baseline,
    )
    score, score_inputs = live_candidate_score(live, baseline)
    return {
        "smc_case": case_name,
        "smc_allocation": float(allocation),
        "score": score,
        "score_inputs": score_inputs,
        "candidate_generation": {
            "sota_candidates": len(base_events),
            "stable_candidates": len(stable_events),
            "smc_candidates": len(smc_events),
            "stable_summary": stable_summary,
            "smc_summary": smc_summary,
        },
        "reference_base_priority_stable_smc": compact_combo_result(reference, int(args.sample_trades)),
        "live_shadow": compact_combo_result(live, int(args.sample_trades)),
        "reference_gap": reference_gap(reference, live),
        "stable_preempted_sota": stable_preempted_sota_summary(decisions),
        "decisions": decisions,
    }


def compact_candidate_result(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "smc_case": candidate["smc_case"],
        "smc_allocation": candidate["smc_allocation"],
        "score": candidate["score"],
        "score_inputs": candidate["score_inputs"],
        "candidate_generation": {
            "sota_candidates": candidate["candidate_generation"]["sota_candidates"],
            "stable_candidates": candidate["candidate_generation"]["stable_candidates"],
            "smc_candidates": candidate["candidate_generation"]["smc_candidates"],
            "smc_summary": candidate["candidate_generation"]["smc_summary"],
        },
        "reference_base_priority_stable_smc": candidate["reference_base_priority_stable_smc"],
        "live_shadow": candidate["live_shadow"],
        "reference_gap": candidate["reference_gap"],
        "stable_preempted_sota": candidate["stable_preempted_sota"],
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

    daily = daily_candles_from_4h(prepared.c4h)
    h4_highs, h4_lows = precompute_swings(prepared.c4h, n=2, lookback=80)
    d1_highs, d1_lows = precompute_swings(daily, n=2, lookback=20)
    smc_case_names = selected_smc_case_names(args)
    smc_allocations = selected_smc_allocations(args)
    candidates: list[dict[str, Any]] = []
    for case_name in smc_case_names:
        for allocation in smc_allocations:
            candidates.append(
                evaluate_smc_candidate(
                    case_name=case_name,
                    allocation=float(allocation),
                    args=args,
                    payload=payload,
                    prepared=prepared,
                    daily=daily,
                    h4_highs=h4_highs,
                    h4_lows=h4_lows,
                    d1_highs=d1_highs,
                    d1_lows=d1_lows,
                    base_events=base_events,
                    stable_events=stable_events,
                    stable_summary=stable_summary,
                    initial_capital=initial_capital,
                    baseline=base_shadow_summary,
                )
            )
    candidates.sort(
        key=lambda item: (
            float(item["score"]),
            float(item["live_shadow"].get("total_return_pct", 0.0) or 0.0),
        ),
        reverse=True,
    )
    selected = candidates[0]

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
            "smc_cases": smc_case_names,
            "smc_allocation_values": smc_allocations,
            "selected_smc_case": selected["smc_case"],
            "selected_smc_allocation": selected["smc_allocation"],
            "smc_case": selected["smc_case"],
            "smc_allocation": selected["smc_allocation"],
            "stable_params": stable_summary["params"],
            "paper_log_output": str(Path(args.paper_log_output).resolve()),
        },
        "baseline_shadow_sota": {key: value for key, value in base_shadow_summary.items() if key != "events"},
        "selected_candidate": {
            "smc_case": selected["smc_case"],
            "smc_allocation": selected["smc_allocation"],
            "score": selected["score"],
            "score_inputs": selected["score_inputs"],
        },
        "candidate_results": [compact_candidate_result(candidate) for candidate in candidates],
        "candidate_generation": selected["candidate_generation"],
        "reference_base_priority_stable_smc": selected["reference_base_priority_stable_smc"],
        "live_shadow": selected["live_shadow"],
        "reference_gap": selected["reference_gap"],
        "stable_preempted_sota": selected["stable_preempted_sota"],
        "decisions": selected["decisions"],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_paper_log(Path(args.paper_log_output), selected["decisions"])
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    print(output)
    base = report["baseline_shadow_sota"]
    ref = report["reference_base_priority_stable_smc"]
    live_payload = report["live_shadow"]
    selected_payload = report["selected_candidate"]
    print(
        f"Selected SMC={selected_payload['smc_case']} "
        f"allocation={selected_payload['smc_allocation']:.2f} score={selected_payload['score']:.2f}"
    )
    print(f"Baseline full={base['total_return_pct']:.2f}%/{base['max_drawdown_pct']:.2f}% 2026={base['windows']['current_year']['total_return_pct']:.2f}%")
    print(f"Reference base-priority full={ref['total_return_pct']:.2f}%/{ref['max_drawdown_pct']:.2f}% 2026={ref['windows']['current_year']['total_return_pct']:.2f}%")
    print(f"Live-shadow full={live_payload['total_return_pct']:.2f}%/{live_payload['max_drawdown_pct']:.2f}% 2026={live_payload['windows']['current_year']['total_return_pct']:.2f}%")
    print(f"Live gap vs reference: return={report['reference_gap']['return_gap_pct']:.2f}% dd={report['reference_gap']['dd_gap_pct']:+.2f}")
    print(f"Decisions={live_payload['decision_counts']}")
    for rank, candidate in enumerate(report["candidate_results"][: min(5, len(report["candidate_results"]))], start=1):
        candidate_live = candidate["live_shadow"]
        print(
            f"#{rank} {candidate['smc_case']} alloc={candidate['smc_allocation']:.2f} "
            f"score={candidate['score']:.2f} "
            f"full={candidate_live['total_return_pct']:.2f}%/{candidate_live['max_drawdown_pct']:.2f}% "
            f"2026={candidate_live['windows']['current_year']['total_return_pct']:.2f}% "
            f"trades={candidate_live['trades']}"
        )
    preempted = report["stable_preempted_sota"]
    print(
        f"Stable preempted SOTA: count={preempted['count']} "
        f"sota_return_sum={preempted['sota_return_sum_pct']:.2f}% "
        f"positive={preempted['positive_sota_blocked']} negative={preempted['negative_sota_blocked']}"
    )
    print(f"Paper log={Path(args.paper_log_output)}")


if __name__ == "__main__":
    main()
