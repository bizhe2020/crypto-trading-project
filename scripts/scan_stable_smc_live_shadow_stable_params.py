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
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.replay_stable_smc_live_shadow import apply_trailing_rr_modes, replay_live_shadow  # noqa: E402
from scripts.report_smc_trade_context import daily_candles_from_4h  # noqa: E402
from scripts.reproduce_reverse_short_overlay_candidates import clean_for_json  # noqa: E402
from scripts.research_reverse_short_from_failed_longs import (  # noqa: E402
    add_combo_deltas,
    add_standard_windows,
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
from scripts.research_stable_reverse_short_plus_smc_short import (  # noqa: E402
    SMC_CASES,
    build_smc_events,
)
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "stable_smc_live_shadow_stable_param_scan.json"
RR_MODE_CHOICES = ("close", "extreme")


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Stable reverse-short params under chronological Stable+SMC live-shadow replay.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--smc-case", default="v2_medium_dispbody05_otherlag4_10x", choices=sorted(SMC_CASES))
    parser.add_argument("--smc-allocation", type=float, default=1.0)
    parser.add_argument("--stable-allocation", type=float, default=1.0)
    parser.add_argument("--stable-selector", default="guarded_weak_loss")
    parser.add_argument("--target-rr-values", default="2.25,2.5,2.75,3.0")
    parser.add_argument("--max-hold-bars-values", default="24,32,40,48,56,64,72,80")
    parser.add_argument("--leverage-values", default="5.0")
    parser.add_argument("--stop-multiplier-values", default="1.0,1.1,1.2,1.25")
    parser.add_argument("--max-short-stop-pct-values", default="1.5,1.75,2.0")
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=0)
    parser.add_argument("--stage-trigger-rr-mode", default="close", choices=RR_MODE_CHOICES)
    parser.add_argument("--time-trailing-rr-mode", default="close", choices=RR_MODE_CHOICES)
    parser.add_argument("--atr-activation-rr-mode", default="close", choices=RR_MODE_CHOICES)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--sample-trades", type=int, default=20)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def build_stable_events_for_params(
    payload: dict[str, Any],
    prepared: Any,
    shadow_events: list[dict[str, Any]],
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
            trigger_mode="stop_loss_reversal",
            target_rr=target_rr,
            max_hold_bars=max_hold_bars,
            leverage=leverage,
            stop_multiplier=stop_multiplier,
            max_short_stop_pct=max_short_stop_pct,
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


def score_live_candidate(result: dict[str, Any], reference_full: float) -> float:
    delta = result.get("delta_vs_shadow_sota", {})
    year_delta = result.get("window_deltas_vs_shadow_sota", {}).get("current_year", {})
    decisions = result.get("decision_counts", {})
    rejected_sota = decisions.get("by_event_type_decision", {}).get("sota_long", {}).get("rejected", 0)
    dd_delta = float(delta.get("max_drawdown_pct", 0.0) or 0.0)
    full = float(result.get("total_return_pct", 0.0) or 0.0)
    gap_to_reference = max(0.0, float(reference_full) - full)
    return round(
        full
        + float(year_delta.get("total_return_pct", 0.0) or 0.0) * 200.0
        - max(0.0, dd_delta) * 50000.0
        - int(rejected_sota) * 15000.0
        - gap_to_reference * 0.05,
        4,
    )


def reference_base_priority(
    base_events: list[dict[str, Any]],
    stable_events: list[dict[str, Any]],
    smc_events: list[dict[str, Any]],
    initial_capital: float,
    data_end: pd.Timestamp,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    # Hindsight reference: base/SOTA remains priority against overlays.
    filtered_stable = [
        event for event in stable_events
        if not any(int(event["entry_idx"]) < int(base["exit_idx"]) and int(event["exit_idx"]) > int(base["entry_idx"]) for base in base_events)
    ]
    filtered_smc = [
        event for event in smc_events
        if not any(int(event["entry_idx"]) < int(base["exit_idx"]) and int(event["exit_idx"]) > int(base["entry_idx"]) for base in base_events + filtered_stable)
    ]
    result = standard_event_summary(base_events + filtered_stable + filtered_smc, initial_capital, "entry_idx")
    result = add_standard_windows(result, initial_capital, data_end, "entry_idx")
    result = add_combo_deltas(result, baseline)
    result["combo_mode"] = "reference_base_priority"
    result["event_type_counts"] = result.get("event_type_counts", {})
    return result


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

    results: list[dict[str, Any]] = []
    target_rr_values = parse_float_list(args.target_rr_values)
    max_hold_bars_values = parse_int_list(args.max_hold_bars_values)
    leverage_values = parse_float_list(args.leverage_values)
    stop_multiplier_values = parse_float_list(args.stop_multiplier_values)
    max_short_stop_pct_values = parse_float_list(args.max_short_stop_pct_values)

    for target_rr in target_rr_values:
        for max_hold_bars in max_hold_bars_values:
            for leverage in leverage_values:
                for stop_multiplier in stop_multiplier_values:
                    for max_short_stop_pct in max_short_stop_pct_values:
                        stable_events, stable_summary = build_stable_events_for_params(
                            payload,
                            prepared,
                            shadow_events,
                            float(args.stable_allocation),
                            target_rr,
                            max_hold_bars,
                            leverage,
                            stop_multiplier,
                            max_short_stop_pct,
                            selector=str(args.stable_selector),
                        )
                        reference = reference_base_priority(
                            base_events,
                            stable_events,
                            smc_events,
                            initial_capital,
                            prepared.end,
                            base_shadow_summary,
                        )
                        live, decisions = replay_live_shadow(
                            base_events + stable_events + smc_events,
                            initial_capital,
                            prepared.end,
                            base_shadow_summary,
                        )
                        live["params"] = {
                            **stable_summary["params"],
                            "smc_case": args.smc_case,
                            "smc_allocation": float(args.smc_allocation),
                            "combo_mode": "live_shadow_chronological",
                            **trailing_rr_modes,
                        }
                        live["stable_summary"] = stable_summary
                        live["smc_summary"] = smc_summary
                        live["reference_base_priority"] = {
                            "total_return_pct": reference["total_return_pct"],
                            "max_drawdown_pct": reference["max_drawdown_pct"],
                            "current_year_return_pct": reference["windows"]["current_year"]["total_return_pct"],
                            "event_type_counts": reference.get("event_type_counts", {}),
                        }
                        live["reference_gap"] = {
                            "return_gap_pct": round(float(live["total_return_pct"]) - float(reference["total_return_pct"]), 4),
                            "dd_gap_pct": round(float(live["max_drawdown_pct"]) - float(reference["max_drawdown_pct"]), 4),
                        }
                        live["score"] = score_live_candidate(live, float(reference["total_return_pct"]))
                        live["decision_sample"] = decisions[: int(args.sample_trades)]
                        results.append(live)

    results.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    by_return = sorted(results, key=lambda item: float(item.get("total_return_pct", 0.0) or 0.0), reverse=True)
    no_sota_reject = [
        item for item in by_return
        if int(item.get("decision_counts", {}).get("by_event_type_decision", {}).get("sota_long", {}).get("rejected", 0) or 0) == 0
    ]
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
        },
        "baseline_shadow_sota": compact_result(base_shadow_summary, 0),
        "experiment": {
            "smc_case": args.smc_case,
            "smc_allocation": args.smc_allocation,
            "smc_summary": smc_summary,
            "target_rr_values": target_rr_values,
            "max_hold_bars_values": max_hold_bars_values,
            "leverage_values": leverage_values,
            "stop_multiplier_values": stop_multiplier_values,
            "max_short_stop_pct_values": max_short_stop_pct_values,
            "candidate_count": len(results),
            "no_sota_reject_count": len(no_sota_reject),
        },
        "top_by_score": [compact_combo_result(item, int(args.sample_trades)) | {
            "params": item["params"],
            "stable_summary": item["stable_summary"],
            "reference_base_priority": item["reference_base_priority"],
            "reference_gap": item["reference_gap"],
            "decision_counts": item["decision_counts"],
            "score": item["score"],
        } for item in results[: int(args.top)]],
        "top_by_return": [compact_combo_result(item, int(args.sample_trades)) | {
            "params": item["params"],
            "stable_summary": item["stable_summary"],
            "reference_base_priority": item["reference_base_priority"],
            "reference_gap": item["reference_gap"],
            "decision_counts": item["decision_counts"],
            "score": item["score"],
        } for item in by_return[: int(args.top)]],
        "top_no_sota_reject": [compact_combo_result(item, int(args.sample_trades)) | {
            "params": item["params"],
            "stable_summary": item["stable_summary"],
            "reference_base_priority": item["reference_base_priority"],
            "reference_gap": item["reference_gap"],
            "decision_counts": item["decision_counts"],
            "score": item["score"],
        } for item in no_sota_reject[: int(args.top)]],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    print(output)
    base = report["baseline_shadow_sota"]
    print(f"Baseline full={base['total_return_pct']:.2f}%/{base['max_drawdown_pct']:.2f}% 2026={base['windows']['current_year']['total_return_pct']:.2f}%")
    print("Top live-shadow candidates by return:")
    for idx, item in enumerate(report["top_by_return"][:10], start=1):
        params = item["params"]
        year = item["windows"]["current_year"]
        decisions = item["decision_counts"].get("by_event_type_decision", {})
        rejected_sota = decisions.get("sota_long", {}).get("rejected", 0)
        print(
            f"{idx:02d} full={item['total_return_pct']:.2f}%/{item['max_drawdown_pct']:.2f}% "
            f"2026={year['total_return_pct']:.2f}% reject_sota={rejected_sota} "
            f"stable={item.get('event_type_counts', {}).get('stable_reverse_short', 0)} "
            f"smc={item.get('event_type_counts', {}).get('smc_short', 0)} "
            f"rr={params['target_rr']} hold={params['max_hold_bars']} lev={params['leverage']} "
            f"sm={params['stop_multiplier']} cap={params['max_short_stop_pct']} "
            f"gap={item['reference_gap']['return_gap_pct']:.2f}%"
        )
    if report["top_no_sota_reject"]:
        print("Best no-SOTA-reject candidate:")
        item = report["top_no_sota_reject"][0]
        params = item["params"]
        print(
            f"  full={item['total_return_pct']:.2f}%/{item['max_drawdown_pct']:.2f}% "
            f"2026={item['windows']['current_year']['total_return_pct']:.2f}% "
            f"rr={params['target_rr']} hold={params['max_hold_bars']} sm={params['stop_multiplier']} cap={params['max_short_stop_pct']}"
        )


if __name__ == "__main__":
    main()
