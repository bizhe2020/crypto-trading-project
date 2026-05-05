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
from scripts.reproduce_reverse_short_overlay_candidates import (  # noqa: E402
    acceptance_gate,
    clean_for_json,
    overlay_attribution,
    source_funnel,
)
from scripts.research_reverse_short_from_failed_longs import (  # noqa: E402
    add_windows,
    build_combo_results,
    compact_combo_result,
    compact_result,
    event_stream_summary,
    replay_non_overlapping,
    selected_by,
    simulate_short_trade,
    standard_reverse_short_event,
    standard_sota_event,
)
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "reverse_short_broader_2026_gate_scan.json"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Focused Broader reverse-short scan that requires at least two accepted 2026 overlay shorts."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--target-rr-values", default="0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--max-hold-bars-values", default="4,6,8,10,12,16,20,24,32")
    parser.add_argument("--leverage-values", default="4.0,5.0,6.0")
    parser.add_argument("--stop-multiplier-values", default="0.75,1.0,1.25")
    parser.add_argument("--max-short-stop-pct-values", default="1.0,1.25,1.5,1.75,2.0")
    parser.add_argument("--overlay-allocation-values", default="0.5,0.75,1.0")
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=0)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--sample-trades", type=int, default=30)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def score_broader_candidate(combo: dict[str, Any]) -> float:
    delta = combo.get("delta_vs_shadow_sota", {})
    window_delta = combo.get("window_deltas_vs_shadow_sota", {})
    year_delta = window_delta.get("current_year", {})
    last_60d_delta = window_delta.get("last_60d", {})
    accepted_short = combo.get("overlay_attribution", {}).get("accepted_overlay", {})
    year_short = combo.get("overlay_attribution", {}).get("windows", {}).get("current_year", {})
    max_dd_delta = max(0.0, float(delta.get("max_drawdown_pct", 0.0) or 0.0))
    single_trade_penalty = 0.0 if int(year_short.get("trades", 0) or 0) > 1 else 10000.0
    return round(
        float(delta.get("total_return_pct", 0.0) or 0.0)
        + float(year_delta.get("total_return_pct", 0.0) or 0.0) * 220.0
        + float(last_60d_delta.get("total_return_pct", 0.0) or 0.0) * 80.0
        + int(year_short.get("trades", 0) or 0) * 1500.0
        + int(accepted_short.get("trades", 0) or 0) * 80.0
        - max_dd_delta * 5000.0
        - single_trade_penalty,
        4,
    )


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
    standard_base_events = [standard_sota_event(event) for event in shadow_events]

    target_rr_values = parse_float_list(args.target_rr_values)
    max_hold_bars_values = parse_int_list(args.max_hold_bars_values)
    leverage_values = parse_float_list(args.leverage_values)
    stop_multiplier_values = parse_float_list(args.stop_multiplier_values)
    max_short_stop_pct_values = parse_float_list(args.max_short_stop_pct_values)
    overlay_allocation_values = parse_float_list(args.overlay_allocation_values)

    selector = "bull_high_growth_offense_loss"
    trigger_mode = "stop_loss_reversal"
    selected_events = [event for event in shadow_events if selected_by(event, selector, 1)]
    simulated_cache: dict[tuple[float, int, float, float, float], list[dict[str, Any]]] = {}
    combos: list[dict[str, Any]] = []

    taker_fee_rate = float(payload.get("taker_fee_rate", 0.0005) or 0.0)
    slippage_bps = float(payload.get("slippage_bps", 0.0) or 0.0)

    for target_rr in target_rr_values:
        for max_hold_bars in max_hold_bars_values:
            for leverage in leverage_values:
                for stop_multiplier in stop_multiplier_values:
                    for max_short_stop_pct in max_short_stop_pct_values:
                        cache_key = (target_rr, max_hold_bars, leverage, stop_multiplier, max_short_stop_pct)
                        reverse_candidates = [
                            trade for trade in (
                                simulate_short_trade(
                                    event=event,
                                    candles=prepared.c15m,
                                    trigger_mode=trigger_mode,
                                    target_rr=target_rr,
                                    max_hold_bars=max_hold_bars,
                                    leverage=leverage,
                                    stop_multiplier=stop_multiplier,
                                    max_short_stop_pct=max_short_stop_pct,
                                    virtual_invalidation_rr=None,
                                    virtual_invalidation_lookahead_bars=None,
                                    taker_fee_rate=taker_fee_rate,
                                    slippage_bps=slippage_bps,
                                )
                                for event in selected_events
                            )
                            if trade is not None
                        ]
                        simulated_cache[cache_key] = reverse_candidates
                        reverse_only = replay_non_overlapping(reverse_candidates, initial_capital)
                        reverse_only = add_windows(reverse_only, initial_capital, prepared.end)
                        reverse_only["params"] = {
                            "source_stream": "shadow",
                            "selector": selector,
                            "trigger_mode": trigger_mode,
                            "target_rr": target_rr,
                            "max_hold_bars": max_hold_bars,
                            "leverage": leverage,
                            "stop_multiplier": stop_multiplier,
                            "max_short_stop_pct": max_short_stop_pct,
                        }
                        for overlay_allocation in overlay_allocation_values:
                            standard_overlay_events = [
                                standard_reverse_short_event(event, overlay_allocation)
                                for event in reverse_only["events"]
                            ]
                            combo = next(
                                item for item in build_combo_results(
                                    standard_base_events,
                                    standard_overlay_events,
                                    initial_capital,
                                    prepared.end,
                                    base_shadow_summary,
                                )
                                if str(item["combo_mode"]) == "base_priority_single_slot"
                            )
                            combo["params"] = reverse_only["params"] | {
                                "overlay_allocation": overlay_allocation,
                                "combo_mode": "base_priority_single_slot",
                            }
                            combo["overlay_attribution"] = overlay_attribution(combo, initial_capital, prepared.end)
                            combo["source_funnel"] = source_funnel(
                                shadow_events,
                                len(selected_events),
                                reverse_candidates,
                                reverse_only,
                                combo,
                            )
                            combo["acceptance_gate"] = acceptance_gate("broader", combo, base_shadow_summary)
                            combo["broader_score"] = score_broader_candidate(combo)
                            combos.append(combo)

    gate_passed = [item for item in combos if bool(item.get("acceptance_gate", {}).get("passes"))]
    dd_valid = [
        item for item in combos
        if float(item.get("delta_vs_shadow_sota", {}).get("max_drawdown_pct", 0.0) or 0.0) <= 1.0
    ]
    combos.sort(key=lambda item: float(item.get("broader_score", 0.0) or 0.0), reverse=True)
    gate_passed.sort(
        key=lambda item: (
            float(item.get("total_return_pct", 0.0) or 0.0),
            float(item.get("window_deltas_vs_shadow_sota", {}).get("current_year", {}).get("total_return_pct", 0.0) or 0.0),
        ),
        reverse=True,
    )
    dd_valid.sort(key=lambda item: float(item.get("broader_score", 0.0) or 0.0), reverse=True)

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
        "experiment": {
            "selector": selector,
            "trigger_mode": trigger_mode,
            "target_rr_values": target_rr_values,
            "max_hold_bars_values": max_hold_bars_values,
            "leverage_values": leverage_values,
            "stop_multiplier_values": stop_multiplier_values,
            "max_short_stop_pct_values": max_short_stop_pct_values,
            "overlay_allocation_values": overlay_allocation_values,
            "selected_source_events": len(selected_events),
            "combo_candidate_count": len(combos),
            "gate_passed_count": len(gate_passed),
            "dd_valid_count": len(dd_valid),
        },
        "gate_passed_top": [
            compact_combo_result(item, int(args.sample_trades)) for item in gate_passed[: int(args.top)]
        ],
        "dd_valid_top": [
            compact_combo_result(item, int(args.sample_trades)) for item in dd_valid[: int(args.top)]
        ],
        "overall_top": [
            compact_combo_result(item, int(args.sample_trades)) for item in combos[: int(args.top)]
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    print(output)
    print("Baseline shadow SOTA:")
    base = report["baseline_shadow_sota"]
    print(f"  full={base['total_return_pct']:.2f}%/{base['max_drawdown_pct']:.2f}% 2026={base['windows']['current_year']['total_return_pct']:.2f}% trades={base['trades']}")
    print(f"Gate passed: {len(gate_passed)} / {len(combos)}")
    for idx, item in enumerate(gate_passed[:10], start=1):
        params = item["params"]
        delta = item["delta_vs_shadow_sota"]
        year = item["window_deltas_vs_shadow_sota"]["current_year"]
        short_year = item["overlay_attribution"]["windows"]["current_year"]
        print(
            f"{idx:02d} full={item['total_return_pct']:.2f}%/{item['max_drawdown_pct']:.2f}% "
            f"delta={delta['total_return_pct']:.2f}%/{delta['max_drawdown_pct']:+.2f}dd "
            f"2026d={year['total_return_pct']:.2f}% "
            f"2026_short_trades={short_year['trades']} 2026_short={short_year['compounded_return_pct']:.2f}% "
            f"shorts={item['event_type_counts'].get('reverse_short', 0)} "
            f"rr={params['target_rr']} hold={params['max_hold_bars']} lev={params['leverage']} "
            f"sm={params['stop_multiplier']} cap={params['max_short_stop_pct']} alloc={params['overlay_allocation']}"
        )


if __name__ == "__main__":
    main()
