#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.live_shadow_utils import (  # noqa: E402
    add_combo_deltas,
    add_standard_windows,
    clean_for_json,
    compact_combo_result,
    parse_float_list,
    parse_int_list,
    standard_event_summary,
    standard_sota_event,
)
from scripts.replay_sota_smc_live_shadow import (  # noqa: E402
    apply_sota_score_gate,
    apply_sota_structure_gate,
    apply_trailing_rr_modes,
    build_gap_smc_events,
    parse_score_bucket_rules,
    replay_live_shadow,
)
from scripts.report_smc_trade_context import daily_candles_from_4h  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402
from scripts.score_bucket_sizing_utils import apply_score_bucket_sizing_to_events  # noqa: E402
from scripts.smc_live_utils import SMC_CASES, build_smc_events, replay_base_priority_sota_first  # noqa: E402
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "live_shadow_risk_gate_scan.json"
RR_MODE_CHOICES = ("close", "extreme")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan shadow risk-gate params under SOTA+SMC live-shadow replay.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--stage-trigger-rr-mode", default="close", choices=RR_MODE_CHOICES)
    parser.add_argument("--time-trailing-rr-mode", default="extreme", choices=RR_MODE_CHOICES)
    parser.add_argument("--atr-activation-rr-mode", default="extreme", choices=RR_MODE_CHOICES)
    parser.add_argument("--smc-case", default="v2_medium_dispbody05_otherlag4_10x", choices=sorted(SMC_CASES))
    parser.add_argument("--smc-allocation", type=float, default=1.0)
    parser.add_argument("--sota-score-net-min", type=int, default=3)
    parser.add_argument("--sota-score-bull-min", type=int, default=8)
    parser.add_argument("--sota-score-bear-max", type=int, default=6)
    parser.add_argument("--sota-score-conflict-mode", default="any", choices=("any", "conflict", "clean"))
    parser.add_argument("--require-non-bearish-structure-for-long", action="store_true")
    parser.add_argument("--enable-gap-smc-short", action="store_true")
    parser.add_argument("--gap-smc-case", default="gap_expansion_21d_other_3x", choices=sorted(SMC_CASES))
    parser.add_argument("--gap-smc-min-flat-days", type=float, default=21.0)
    parser.add_argument("--gap-smc-leverage", type=float, default=3.0)
    parser.add_argument("--gap-smc-max-stop-distance-pct", type=float, default=1.5)
    parser.add_argument("--enable-long-score-bucket-sizing", action="store_true")
    parser.add_argument(
        "--long-score-bucket-sizing-rules-json",
        default="",
        help="Optional JSON array/dict for long score bucket sizing rules.",
    )
    parser.add_argument("--daily-loss-values", default="0,4,6,8,10,12")
    parser.add_argument("--equity-dd-values", default="0,12,15,18,21,25,30")
    parser.add_argument("--equity-cooldown-values", default="0,1,2,4,6,10")
    parser.add_argument("--loss-streak-values", default="0,3,4,5,6")
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--sample-trades", type=int, default=0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def compact(result: dict[str, Any], sample_trades: int) -> dict[str, Any]:
    payload = compact_combo_result(result, sample_trades)
    payload["decision_counts"] = result.get("decision_counts", {})
    return payload


def event_key(event: dict[str, Any]) -> str:
    return f"{event.get('event_type')}|{int(event.get('entry_idx', 0) or 0)}|{int(event.get('exit_idx', 0) or 0)}"


def sort_by_return(item: dict[str, Any]) -> tuple[float, float, float]:
    live = item["live_shadow"]
    return (
        float(live.get("total_return_pct", 0.0) or 0.0),
        -float(live.get("max_drawdown_pct", 0.0) or 0.0),
        float(live.get("windows", {}).get("current_year", {}).get("total_return_pct", 0.0) or 0.0),
    )


def sort_by_2026(item: dict[str, Any]) -> tuple[float, float, float]:
    live = item["live_shadow"]
    return (
        float(live.get("windows", {}).get("current_year", {}).get("total_return_pct", 0.0) or 0.0),
        float(live.get("total_return_pct", 0.0) or 0.0),
        -float(live.get("max_drawdown_pct", 0.0) or 0.0),
    )


def main() -> None:
    args = parse_args()
    base_payload = load_config_payload(Path(args.config))
    if bool(base_payload.get("enable_sota_score_gate_live", False)):
        args.sota_score_net_min = int(base_payload.get("sota_score_net_min", args.sota_score_net_min) or args.sota_score_net_min)
        args.sota_score_bull_min = int(base_payload.get("sota_score_bull_min", args.sota_score_bull_min) or args.sota_score_bull_min)
        args.sota_score_bear_max = int(base_payload.get("sota_score_bear_max", args.sota_score_bear_max) or args.sota_score_bear_max)
        args.sota_score_conflict_mode = str(base_payload.get("sota_score_conflict_mode", args.sota_score_conflict_mode) or args.sota_score_conflict_mode)
    if bool(base_payload.get("require_non_bearish_structure_for_long_live", False)):
        args.require_non_bearish_structure_for_long = True
    if bool(base_payload.get("enable_long_score_bucket_sizing_live", False)) and not bool(args.enable_long_score_bucket_sizing):
        args.enable_long_score_bucket_sizing = True
        if not str(args.long_score_bucket_sizing_rules_json or "").strip():
            args.long_score_bucket_sizing_rules_json = json.dumps(
                base_payload.get("long_score_bucket_sizing_rules", []),
                ensure_ascii=False,
            )
    if bool(base_payload.get("enable_gap_smc_short_live", False)) and not bool(args.enable_gap_smc_short):
        args.enable_gap_smc_short = True
        args.gap_smc_case = str(base_payload.get("gap_smc_short_case", args.gap_smc_case) or args.gap_smc_case)
        args.gap_smc_min_flat_days = float(
            base_payload.get("gap_smc_short_min_flat_days", args.gap_smc_min_flat_days) or args.gap_smc_min_flat_days
        )
        args.gap_smc_leverage = float(base_payload.get("gap_smc_short_leverage", args.gap_smc_leverage) or args.gap_smc_leverage)
    payload, pressure_params = apply_pressure_params(base_payload, Path(args.pressure_params))
    payload, trailing_rr_modes = apply_trailing_rr_modes(
        payload,
        stage_trigger_rr_mode=str(args.stage_trigger_rr_mode),
        time_trailing_rr_mode=str(args.time_trailing_rr_mode),
        atr_activation_rr_mode=str(args.atr_activation_rr_mode),
    )
    payload["replay_sync_entry_to_signal_price"] = True
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=payload.get("regime_switcher_thresholds"),
        confirmed_4h_only=True,
    )
    metrics, engine = run_engine(payload, prepared, args.start_date)
    trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
    initial_capital = float(metrics.get("initial_capital", 1000.0) or 1000.0)
    fixed = expansion_overlay(trades, initial_capital, FIXED_STRUCTURE_PARAMS, include_events=True)
    fixed_events = fixed["events"]
    all_raw_sota = [standard_sota_event(event) for event in fixed_events]
    all_gated_sota, all_score_gate = apply_sota_score_gate(
        prepared,
        all_raw_sota,
        enabled=True,
        net_min=int(args.sota_score_net_min),
        bull_min=int(args.sota_score_bull_min),
        bear_max=int(args.sota_score_bear_max),
        conflict_mode=str(args.sota_score_conflict_mode),
    )
    all_gated_sota, sota_structure_gate = apply_sota_structure_gate(
        all_gated_sota,
        enabled=bool(args.require_non_bearish_structure_for_long),
    )
    all_gated_sota, long_score_bucket_sizing = apply_score_bucket_sizing_to_events(
        all_gated_sota,
        enabled=bool(args.enable_long_score_bucket_sizing),
        rules=parse_score_bucket_rules(str(args.long_score_bucket_sizing_rules_json)),
    )
    gated_sota_by_key = {event_key(event): event for event in all_gated_sota}

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

    candidates: list[dict[str, Any]] = []
    for daily_loss, equity_dd, cooldown, loss_streak in product(
        parse_float_list(args.daily_loss_values),
        parse_float_list(args.equity_dd_values),
        parse_int_list(args.equity_cooldown_values),
        parse_int_list(args.loss_streak_values),
    ):
        if equity_dd <= 0 and cooldown > 0:
            continue
        if equity_dd > 0 and cooldown <= 0:
            continue
        shadow = replay_shadow_events(
            fixed_events,
            initial_capital,
            daily_loss_stop_pct=float(daily_loss),
            equity_drawdown_stop_pct=float(equity_dd),
            consecutive_loss_stop=int(loss_streak),
            equity_drawdown_cooldown_days=int(cooldown),
        )
        raw_sota = [standard_sota_event(event) for event in shadow["events"]]
        base_events = [gated_sota_by_key[key] for event in raw_sota if (key := event_key(event)) in gated_sota_by_key]
        score_gate = {
            **all_score_gate,
            "structure_gate": sota_structure_gate,
            "original_candidates": len(raw_sota),
            "filtered_candidates": len(base_events),
            "removed_candidates": len(raw_sota) - len(base_events),
            "cached_total_candidates": len(all_raw_sota),
            "cached_filtered_candidates": len(all_gated_sota),
        }
        baseline = standard_event_summary(base_events, initial_capital, "entry_idx")
        baseline = add_standard_windows(baseline, initial_capital, prepared.end, "entry_idx")
        reference = replay_base_priority_sota_first(
            base_events,
            smc_events,
            initial_capital,
            prepared.end,
            baseline,
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
        live, decisions = replay_live_shadow(base_events + smc_events + gap_smc_events, initial_capital, prepared.end, baseline)
        live["shadow_risk_gate"] = {
            "daily_loss_stop_pct": float(daily_loss),
            "equity_drawdown_stop_pct": float(equity_dd),
            "equity_drawdown_cooldown_days": int(cooldown),
            "consecutive_loss_stop": int(loss_streak),
        }
        live = add_combo_deltas(live, baseline)
        candidates.append(
            {
                "params": live["shadow_risk_gate"],
                "raw_sota_candidates": len(raw_sota),
                "sota_score_gate": score_gate,
                "long_score_bucket_sizing": long_score_bucket_sizing,
                "smc_candidates": len(smc_events),
                "gap_smc_short_candidates": len(gap_smc_events),
                "gap_smc_summary": gap_smc_summary,
                "shadow_trigger_counts": shadow.get("trigger_counts", {}),
                "shadow_skipped_trades": int(shadow.get("skipped_trades", 0) or 0),
                "live_shadow": compact(live, int(args.sample_trades)),
                "decision_counts": live.get("decision_counts", {}),
                "decisions_rejected": int((live.get("decision_counts", {}) or {}).get("by_decision", {}).get("rejected", 0) or 0),
                "decision_count_total": len(decisions),
            }
        )

    ranked_by_return = sorted(candidates, key=sort_by_return, reverse=True)
    ranked_by_2026 = sorted(candidates, key=sort_by_2026, reverse=True)
    report = {
        "metadata": {
            "config": str(Path(args.config).resolve()),
            "pressure_params": str(Path(args.pressure_params).resolve()),
            "pressure_params_applied": pressure_params,
            "trailing_rr_modes": trailing_rr_modes,
            "start_date": args.start_date,
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "confirmed_4h_only": True,
            "replay_sync_entry_to_signal_price": True,
            "sota_score_gate": {
                "net_min": int(args.sota_score_net_min),
                "bull_min": int(args.sota_score_bull_min),
                "bear_max": int(args.sota_score_bear_max),
                "conflict_mode": str(args.sota_score_conflict_mode),
            },
            "long_score_bucket_sizing": {
                "enabled": bool(args.enable_long_score_bucket_sizing),
                "rules": parse_score_bucket_rules(str(args.long_score_bucket_sizing_rules_json)),
            },
            "smc_case": str(args.smc_case),
            "smc_allocation": float(args.smc_allocation),
            "smc_summary": smc_summary,
            "gap_smc_short": {
                "enabled": bool(args.enable_gap_smc_short),
                "case": str(args.gap_smc_case),
                "min_flat_days": float(args.gap_smc_min_flat_days),
                "max_stop_distance_pct": float(args.gap_smc_max_stop_distance_pct),
                "leverage": float(args.gap_smc_leverage),
            },
            "candidate_count": len(candidates),
        },
        "top_by_return": ranked_by_return[: int(args.top_n)],
        "top_by_2026": ranked_by_2026[: int(args.top_n)],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output)
    for idx, item in enumerate(ranked_by_return[:8], start=1):
        live = item["live_shadow"]
        year = live.get("windows", {}).get("current_year", {})
        print(
            f"{idx:02d} full={float(live['total_return_pct']):.2f}%/{float(live['max_drawdown_pct']):.2f}% "
            f"2026={float(year.get('total_return_pct', 0.0) or 0.0):.2f}% "
            f"skipped={item['shadow_skipped_trades']} params={item['params']}"
        )


if __name__ == "__main__":
    main()
