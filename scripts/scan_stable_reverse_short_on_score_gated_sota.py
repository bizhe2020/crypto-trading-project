#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
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
from scripts.replay_stable_smc_live_shadow import (  # noqa: E402
    apply_sota_score_gate,
    build_stable_events_for_params,
    compact_combo_with_events,
    compact_live_result,
    replay_live_shadow,
)
from scripts.report_smc_trade_context import daily_candles_from_4h  # noqa: E402
from scripts.reproduce_reverse_short_overlay_candidates import clean_for_json  # noqa: E402
from scripts.research_reverse_short_from_failed_longs import compact_result, event_stream_summary, parse_float_list, parse_int_list, parse_str_list, standard_sota_event  # noqa: E402
from scripts.research_stable_reverse_short_plus_smc_short import SMC_CASES, build_smc_events, replay_base_priority_stable_first  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "stable_reverse_short_score_gated_sota_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Stable reverse-short parameters on a score-gated SOTA source stream.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--confirmed-4h-only", action="store_true")
    parser.add_argument("--informative-asof-from-15m", action="store_true")
    parser.add_argument("--replay-sync-entry-to-signal-price", action="store_true")
    parser.add_argument("--sota-score-net-min", type=int, default=2)
    parser.add_argument("--sota-score-bull-min", type=int, default=8)
    parser.add_argument("--sota-score-bear-max", type=int, default=6)
    parser.add_argument("--sota-score-conflict-mode", default="any", choices=("any", "conflict", "clean"))
    parser.add_argument("--stable-selectors", default="guarded_weak_loss,weak_quality_loss,bull_high_growth_offense_loss,all_long_stop_loss_loss,trailing_atr_profit_reverse,trailing_stop_profit_reverse")
    parser.add_argument("--stable-target-rr-values", default="1.0,1.5,2.0,2.5,2.75,3.0")
    parser.add_argument("--stable-max-hold-bars-values", default="16,24,32,40,56,64")
    parser.add_argument("--stable-leverage-values", default="2.0,3.0,4.0,5.0")
    parser.add_argument("--stable-stop-multiplier-values", default="0.75,1.0,1.25")
    parser.add_argument("--stable-max-short-stop-pct-values", default="1.25,1.5,1.75,2.0")
    parser.add_argument("--stable-allocation-values", default="0.5,0.75,1.0")
    parser.add_argument("--stable-max-quality-score", type=int, default=1)
    parser.add_argument("--smc-case", default="v2_medium_dispbody05_otherlag4_10x", choices=sorted(SMC_CASES))
    parser.add_argument("--smc-allocation", type=float, default=1.0)
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=0)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--sample-trades", type=int, default=20)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def score_candidate(result: dict[str, Any], baseline: dict[str, Any]) -> float:
    delta = result.get("delta_vs_shadow_sota", {})
    year_delta = result.get("window_deltas_vs_shadow_sota", {}).get("current_year", {})
    recent_60d = result.get("windows", {}).get("last_60d", {})
    dd_increase = max(0.0, float(delta.get("max_drawdown_pct", 0.0) or 0.0))
    return round(
        float(result.get("total_return_pct", 0.0) or 0.0)
        + float(delta.get("total_return_pct", 0.0) or 0.0) * 0.5
        + float(year_delta.get("total_return_pct", 0.0) or 0.0) * 80.0
        + float(recent_60d.get("total_return_pct", 0.0) or 0.0) * 20.0
        - dd_increase * 150.0
        - max(0.0, float(result.get("max_drawdown_pct", 0.0) or 0.0) - float(baseline.get("max_drawdown_pct", 0.0) or 0.0)) * 100.0,
        4,
    )


def main() -> None:
    args = parse_args()
    base_payload = load_config_payload(Path(args.config))
    payload, pressure_params = apply_pressure_params(base_payload, Path(args.pressure_params))
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
    initial_capital = float(metrics.get("initial_capital", 1000.0) or 1000.0)
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
    raw_sota_events = [standard_sota_event(event) for event in shadow_events]
    gated_sota_events, sota_gate_summary = apply_sota_score_gate(
        prepared,
        raw_sota_events,
        enabled=True,
        net_min=int(args.sota_score_net_min),
        bull_min=int(args.sota_score_bull_min),
        bear_max=int(args.sota_score_bear_max),
        conflict_mode=str(args.sota_score_conflict_mode),
    )
    gated_keys = {(int(event["entry_idx"]), int(event["exit_idx"])) for event in gated_sota_events}
    gated_source_events = [
        event for event in shadow_events
        if (int(event.get("entry_idx", 0) or 0), int(event.get("exit_idx", event.get("entry_idx", 0)) or 0)) in gated_keys
    ]
    raw_baseline = event_stream_summary(shadow_events, initial_capital, prepared.end)
    gated_baseline = event_stream_summary(gated_sota_events, initial_capital, prepared.end)

    daily = daily_candles_from_4h(prepared.c4h)
    h4_highs, h4_lows = precompute_swings(prepared.c4h, n=2, lookback=80)
    d1_highs, d1_lows = precompute_swings(daily, n=2, lookback=20)
    smc_events, smc_summary = build_smc_events(
        str(args.smc_case),
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
    for selector, target_rr, max_hold_bars, leverage, stop_multiplier, max_short_stop_pct, allocation in itertools.product(
        parse_str_list(args.stable_selectors),
        parse_float_list(args.stable_target_rr_values),
        parse_int_list(args.stable_max_hold_bars_values),
        parse_float_list(args.stable_leverage_values),
        parse_float_list(args.stable_stop_multiplier_values),
        parse_float_list(args.stable_max_short_stop_pct_values),
        parse_float_list(args.stable_allocation_values),
    ):
        stable_events, stable_summary = build_stable_events_for_params(
            payload,
            prepared,
            gated_source_events,
            float(allocation),
            float(target_rr),
            int(max_hold_bars),
            float(leverage),
            float(stop_multiplier),
            float(max_short_stop_pct),
            selector=str(selector),
            max_quality_score=int(args.stable_max_quality_score),
        )
        reference = replay_base_priority_stable_first(
            gated_sota_events,
            stable_events,
            smc_events,
            initial_capital,
            prepared.end,
            gated_baseline,
        )
        live, decisions = replay_live_shadow(
            gated_sota_events + stable_events + smc_events,
            initial_capital,
            prepared.end,
            gated_baseline,
        )
        live["params"] = {
            **stable_summary["params"],
            "combo_mode": "live_shadow_chronological",
            "smc_case": str(args.smc_case),
            "smc_allocation": float(args.smc_allocation),
        }
        live["stable_summary"] = stable_summary
        live["smc_summary"] = smc_summary
        live["reference_return_pct"] = reference.get("total_return_pct")
        live["reference_max_drawdown_pct"] = reference.get("max_drawdown_pct")
        live["decision_counts"] = live.get("decision_counts", {})
        live["scan_score"] = score_candidate(live, gated_baseline)
        live["decisions"] = decisions[: int(args.sample_trades)]
        results.append(live)

    results.sort(key=lambda item: float(item["scan_score"]), reverse=True)
    top = []
    for result in results[: int(args.top)]:
        compacted = compact_live_result(result, int(args.sample_trades))
        compacted["params"] = result.get("params", {})
        compacted["stable_summary"] = result.get("stable_summary", {})
        compacted["smc_summary"] = result.get("smc_summary", {})
        compacted["reference_return_pct"] = result.get("reference_return_pct")
        compacted["reference_max_drawdown_pct"] = result.get("reference_max_drawdown_pct")
        compacted["scan_score"] = result.get("scan_score")
        top.append(compacted)

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
            "confirmed_4h_only": bool(args.confirmed_4h_only),
            "informative_asof_from_15m": bool(args.informative_asof_from_15m),
            "replay_sync_entry_to_signal_price": bool(args.replay_sync_entry_to_signal_price),
            "sota_score_gate": sota_gate_summary,
            "smc_case": str(args.smc_case),
            "smc_allocation": float(args.smc_allocation),
        },
        "baseline_shadow_sota": compact_result(raw_baseline, 0),
        "gated_shadow_sota": compact_result(gated_baseline, 0),
        "candidate_generation": {
            "raw_sota_candidates": len(raw_sota_events),
            "gated_sota_candidates": len(gated_sota_events),
            "gated_stable_source_events": len(gated_source_events),
            "smc_candidates": len(smc_events),
            "scan_candidates": len(results),
            "smc_summary": smc_summary,
        },
        "top": top,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    print(output)
    base = report["gated_shadow_sota"]
    print(
        f"Gated SOTA baseline={base['total_return_pct']:.2f}%/{base['max_drawdown_pct']:.2f}% "
        f"2026={base['windows']['current_year']['total_return_pct']:.2f}%"
    )
    for idx, item in enumerate(top[:10], start=1):
        params = item["params"]
        year = item["windows"]["current_year"]
        delta = item.get("delta_vs_shadow_sota", {})
        print(
            f"{idx:02d} score={item['scan_score']:.2f} full={item['total_return_pct']:.2f}%/"
            f"{item['max_drawdown_pct']:.2f}% delta={delta.get('total_return_pct', 0.0):.2f}%/"
            f"{delta.get('max_drawdown_pct', 0.0):+.2f}dd 2026={year['total_return_pct']:.2f}% "
            f"stable={item['event_type_counts'].get('stable_reverse_short', 0)} "
            f"params={params}"
        )


if __name__ == "__main__":
    main()
