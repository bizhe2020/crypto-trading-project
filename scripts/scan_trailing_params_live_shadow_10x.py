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
from scripts.live_shadow_utils import (  # noqa: E402
    add_combo_deltas,
    add_standard_windows,
    clean_for_json,
    parse_float_list,
    standard_event_summary,
    standard_sota_event,
)
from scripts.scan_dynamic_sizing_live_shadow_10x import (  # noqa: E402
    PARAM_KEYS,
    apply_cached_score_gate,
    compact,
    dynamic_params_from_base,
    summarize_overlay,
)
from scripts.report_smc_trade_context import daily_candles_from_4h  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402
from scripts.smc_live_utils import SMC_CASES, build_smc_events  # noqa: E402
from scripts.replay_sota_smc_live_shadow import replay_live_shadow  # noqa: E402
from scripts.confirmed_multiframe_score_utils import align_confirmed_mapping, resample_confirmed_1h  # noqa: E402
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "trailing_params_live_shadow_10x_scan.json"
RR_MODE_CHOICES = ("close", "extreme")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Systematic trailing scan on the current best 10x conservative live-shadow strategy.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--smc-case", default="v2_medium_dispbody05_otherlag4_10x", choices=sorted(SMC_CASES))
    parser.add_argument("--smc-allocation", type=float, default=1.0)
    parser.add_argument("--sota-score-net-min", type=int, default=3)
    parser.add_argument("--sota-score-bull-min", type=int, default=8)
    parser.add_argument("--sota-score-bear-max", type=int, default=6)
    parser.add_argument("--sota-score-conflict-mode", default="any", choices=("any", "conflict", "clean"))
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=12.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=4)
    parser.add_argument("--stage-trigger-rr-modes", default="close")
    parser.add_argument("--time-trailing-rr-modes", default="extreme")
    parser.add_argument("--atr-activation-rr-modes", default="extreme")
    parser.add_argument("--atr-activation-rr-values", default="2.06")
    parser.add_argument("--atr-loose-multiplier-values", default="2.7")
    parser.add_argument("--atr-normal-multiplier-values", default="2.25")
    parser.add_argument("--atr-tight-multiplier-values", default="1.8")
    parser.add_argument("--enable-auto-time-based-trailing-values", default="true")
    parser.add_argument("--auto-tit-loss-streak-values", default="1")
    parser.add_argument("--auto-tit-atr-ratio-max-values", default="1.1")
    parser.add_argument("--T1-values", default="10")
    parser.add_argument("--T2-values", default="20")
    parser.add_argument("--T-max-values", default="144")
    parser.add_argument("--S0-trigger-rr-values", default="0.5")
    parser.add_argument("--S1-trigger-rr-values", default="0.8")
    parser.add_argument("--S3-trigger-rr-values", default="3.0")
    parser.add_argument("--S4-close-rr-values", default="0.8")
    parser.add_argument("--pressure-lock-rr-values", default="0.4")
    parser.add_argument("--pressure-atr-multiplier-values", default="3.0")
    parser.add_argument("--pressure-min-rr-values", default="2.0")
    parser.add_argument("--pressure-proximity-pct-values", default="0.15")
    parser.add_argument("--pressure-target-min-rr-values", default="1.25")
    parser.add_argument("--pressure-target-buffer-pct-values", default="0.03")
    parser.add_argument("--pressure-touch-lock-enabled-values", default="true")
    parser.add_argument("--pressure-touch-lock-min-rr-values", default="1.0")
    parser.add_argument("--pressure-touch-lock-buffer-pct-values", default="0.03")
    parser.add_argument("--pressure-touch-lock-atr-multiplier-values", default="0.0")
    parser.add_argument("--top-n", type=int, default=60)
    parser.add_argument("--sample-trades", type=int, default=0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def parse_bool_list(value: str) -> list[bool]:
    items: list[bool] = []
    for raw in value.split(","):
        text = raw.strip().lower()
        if not text:
            continue
        if text in {"1", "true", "yes", "on"}:
            items.append(True)
        elif text in {"0", "false", "no", "off"}:
            items.append(False)
        else:
            raise ValueError(f"invalid bool: {raw}")
    return items


def parse_mode_list(value: str) -> list[str]:
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in items if item not in RR_MODE_CHOICES]
    if invalid:
        raise ValueError(f"invalid RR mode(s): {invalid}")
    return items


def trailing_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    combos = itertools.product(
        parse_mode_list(args.stage_trigger_rr_modes),
        parse_mode_list(args.time_trailing_rr_modes),
        parse_mode_list(args.atr_activation_rr_modes),
        parse_float_list(args.atr_activation_rr_values),
        parse_float_list(args.atr_loose_multiplier_values),
        parse_float_list(args.atr_normal_multiplier_values),
        parse_float_list(args.atr_tight_multiplier_values),
        parse_bool_list(args.enable_auto_time_based_trailing_values),
        [int(x) for x in args.auto_tit_loss_streak_values.split(",") if x.strip()],
        parse_float_list(args.auto_tit_atr_ratio_max_values),
        [int(x) for x in args.T1_values.split(",") if x.strip()],
        [int(x) for x in args.T2_values.split(",") if x.strip()],
        [int(x) for x in args.T_max_values.split(",") if x.strip()],
        parse_float_list(args.S0_trigger_rr_values),
        parse_float_list(args.S1_trigger_rr_values),
        parse_float_list(args.S3_trigger_rr_values),
        parse_float_list(args.S4_close_rr_values),
        parse_float_list(args.pressure_lock_rr_values),
        parse_float_list(args.pressure_atr_multiplier_values),
        parse_float_list(args.pressure_min_rr_values),
        parse_float_list(args.pressure_proximity_pct_values),
        parse_float_list(args.pressure_target_min_rr_values),
        parse_float_list(args.pressure_target_buffer_pct_values),
        parse_bool_list(args.pressure_touch_lock_enabled_values),
        parse_float_list(args.pressure_touch_lock_min_rr_values),
        parse_float_list(args.pressure_touch_lock_buffer_pct_values),
        parse_float_list(args.pressure_touch_lock_atr_multiplier_values),
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for combo in combos:
        (
            stage_rr_mode,
            time_rr_mode,
            atr_rr_mode,
            atr_activation_rr,
            atr_loose_multiplier,
            atr_normal_multiplier,
            atr_tight_multiplier,
            enable_auto_tit,
            auto_tit_loss_streak,
            auto_tit_atr_ratio_max,
            T1,
            T2,
            T_max,
            S0_trigger_rr,
            S1_trigger_rr,
            S3_trigger_rr,
            S4_close_rr,
            pressure_lock_rr,
            pressure_atr_multiplier,
            pressure_min_rr,
            pressure_proximity_pct,
            pressure_target_min_rr,
            pressure_target_buffer_pct,
            pressure_touch_lock_enabled,
            pressure_touch_lock_min_rr,
            pressure_touch_lock_buffer_pct,
            pressure_touch_lock_atr_multiplier,
        ) = combo
        if T1 >= T2 or T2 >= T_max:
            continue
        if S0_trigger_rr > S1_trigger_rr or S1_trigger_rr > S3_trigger_rr:
            continue
        payload = {
            "stage_trigger_rr_mode": stage_rr_mode,
            "time_trailing_rr_mode": time_rr_mode,
            "atr_activation_rr_mode": atr_rr_mode,
            "atr_activation_rr": float(atr_activation_rr),
            "atr_loose_multiplier": float(atr_loose_multiplier),
            "atr_normal_multiplier": float(atr_normal_multiplier),
            "atr_tight_multiplier": float(atr_tight_multiplier),
            "enable_auto_time_based_trailing": bool(enable_auto_tit),
            "auto_tit_loss_streak": int(auto_tit_loss_streak),
            "auto_tit_atr_ratio_max": float(auto_tit_atr_ratio_max),
            "T1": int(T1),
            "T2": int(T2),
            "T_max": int(T_max),
            "S0_trigger_rr": float(S0_trigger_rr),
            "S1_trigger_rr": float(S1_trigger_rr),
            "S3_trigger_rr": float(S3_trigger_rr),
            "S4_close_rr": float(S4_close_rr),
            "pressure_lock_rr": float(pressure_lock_rr),
            "pressure_atr_multiplier": float(pressure_atr_multiplier),
            "pressure_min_rr": float(pressure_min_rr),
            "pressure_proximity_pct": float(pressure_proximity_pct),
            "pressure_target_min_rr": float(pressure_target_min_rr),
            "pressure_target_buffer_pct": float(pressure_target_buffer_pct),
            "pressure_touch_lock_enabled": bool(pressure_touch_lock_enabled),
            "pressure_touch_lock_min_rr": float(pressure_touch_lock_min_rr),
            "pressure_touch_lock_buffer_pct": float(pressure_touch_lock_buffer_pct),
            "pressure_touch_lock_atr_multiplier": float(pressure_touch_lock_atr_multiplier),
        }
        key = json.dumps(payload, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(payload)
    return out


def sort_by_2026(item: dict[str, Any]) -> tuple[float, float, float, float]:
    live = item["live_shadow"]
    current_year = live.get("windows", {}).get("current_year", {})
    return (
        float(current_year.get("total_return_pct", 0.0) or 0.0),
        float(live.get("total_return_pct", 0.0) or 0.0),
        -float(live.get("max_drawdown_pct", 0.0) or 0.0),
        -float(current_year.get("max_drawdown_pct", 0.0) or 0.0),
    )


def sort_by_return(item: dict[str, Any]) -> tuple[float, float, float]:
    live = item["live_shadow"]
    current_year = live.get("windows", {}).get("current_year", {})
    return (
        float(live.get("total_return_pct", 0.0) or 0.0),
        -float(live.get("max_drawdown_pct", 0.0) or 0.0),
        float(current_year.get("total_return_pct", 0.0) or 0.0),
    )


def main() -> None:
    args = parse_args()
    base_payload = load_config_payload(Path(args.config))
    payload, pressure_params = apply_pressure_params(base_payload, Path(args.pressure_params))
    # Lock to the conservative dynamic sizing promoted previously.
    conservative_dynamic = dynamic_params_from_base(
        base=FIXED_STRUCTURE_PARAMS,
        base_leverage=4.0,
        high_growth_leverage=7.5,
        tight_stop_leverage=7.5,
        max_effective_leverage=7.5,
        defense_leverage=2.0,
        drawdown_leverage=2.0,
        unhealthy_leverage=2.0,
        failed_breakout_guard_leverage=1.5,
    )
    payload.update(
        {
            "dynamic_base_leverage": conservative_dynamic["base_leverage"],
            "dynamic_high_growth_leverage": conservative_dynamic["high_growth_leverage"],
            "dynamic_tight_stop_leverage": conservative_dynamic["tight_stop_leverage"],
            "dynamic_recovery_leverage": conservative_dynamic["recovery_leverage"],
            "dynamic_drawdown_leverage": conservative_dynamic["drawdown_leverage"],
            "dynamic_unhealthy_leverage": conservative_dynamic["unhealthy_leverage"],
            "dynamic_defense_leverage": conservative_dynamic["defense_leverage"],
            "dynamic_max_effective_leverage": conservative_dynamic["max_effective_leverage"],
            "dynamic_failed_breakout_guard_leverage": conservative_dynamic["failed_breakout_guard_leverage"],
        }
    )
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=payload.get("regime_switcher_thresholds"),
        confirmed_4h_only=True,
    )

    daily = daily_candles_from_4h(prepared.c4h)
    h4_highs, h4_lows = precompute_swings(prepared.c4h, n=2, lookback=80)
    d1_highs, d1_lows = precompute_swings(daily, n=2, lookback=20)
    c1h = resample_confirmed_1h(prepared.c15m)
    mapping_1h = align_confirmed_mapping(c1h, prepared.c15m)
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
    score_cache: dict[int, dict[str, Any]] = {}

    grid = trailing_grid(args)
    candidates: list[dict[str, Any]] = []
    for idx, trailing_params in enumerate(grid, start=1):
        if idx == 1 or idx % 100 == 0:
            print(f"[{idx}/{len(grid)}] scanning trailing params", flush=True)
        candidate_payload = dict(payload)
        candidate_payload.update(trailing_params)
        candidate_payload["replay_sync_entry_to_signal_price"] = True
        candidate_payload["confirmed_4h_only"] = True
        metrics, engine = run_engine(candidate_payload, prepared, args.start_date)
        trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
        initial_capital = float(metrics.get("initial_capital", 1000.0) or 1000.0)
        overlay = expansion_overlay(trades, initial_capital, FIXED_STRUCTURE_PARAMS, include_events=True)
        shadow = replay_shadow_events(
            overlay["events"],
            initial_capital,
            daily_loss_stop_pct=float(args.daily_loss_stop_pct),
            equity_drawdown_stop_pct=float(args.equity_drawdown_stop_pct),
            consecutive_loss_stop=int(args.consecutive_loss_stop),
            equity_drawdown_cooldown_days=int(args.equity_drawdown_cooldown_days),
        )
        raw_sota = [standard_sota_event(event) for event in shadow["events"]]
        base_events, score_gate = apply_cached_score_gate(
            prepared,
            c1h,
            mapping_1h,
            score_cache,
            raw_sota,
            net_min=int(args.sota_score_net_min),
            bull_min=int(args.sota_score_bull_min),
            bear_max=int(args.sota_score_bear_max),
            conflict_mode=str(args.sota_score_conflict_mode),
        )
        baseline = standard_event_summary(base_events, initial_capital, "entry_idx")
        baseline = add_standard_windows(baseline, initial_capital, prepared.end, "entry_idx")
        live, decisions = replay_live_shadow(base_events + smc_events, initial_capital, prepared.end, baseline)
        live = add_combo_deltas(live, baseline)
        candidate = {
                "trailing_params": trailing_params,
                "dynamic_params": {key: conservative_dynamic.get(key) for key in PARAM_KEYS},
                "pressure_params_base": {
                    "pressure_min_rr": candidate_payload.get("pressure_min_rr"),
                    "pressure_lock_rr": candidate_payload.get("pressure_lock_rr"),
                    "pressure_atr_multiplier": candidate_payload.get("pressure_atr_multiplier"),
                    "pressure_proximity_pct": candidate_payload.get("pressure_proximity_pct"),
                    "pressure_target_min_rr": candidate_payload.get("pressure_target_min_rr"),
                    "pressure_target_buffer_pct": candidate_payload.get("pressure_target_buffer_pct"),
                    "pressure_touch_lock_enabled": candidate_payload.get("pressure_touch_lock_enabled"),
                    "pressure_touch_lock_min_rr": candidate_payload.get("pressure_touch_lock_min_rr"),
                    "pressure_touch_lock_buffer_pct": candidate_payload.get("pressure_touch_lock_buffer_pct"),
                    "pressure_touch_lock_atr_multiplier": candidate_payload.get("pressure_touch_lock_atr_multiplier"),
                },
                "engine_metrics": {
                    "total_return_pct": round(float(metrics.get("total_return_pct", 0.0) or 0.0), 2),
                    "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0) or 0.0), 2),
                    "total_trades": int(metrics.get("total_trades", 0) or 0),
                },
                "dynamic_overlay": summarize_overlay(overlay),
                "sota_score_gate": score_gate,
                "smc_summary": smc_summary,
                "live_shadow": compact(live, int(args.sample_trades)),
                "decision_count_total": len(decisions),
            }
        candidates.append(candidate)
        current_year = candidate["live_shadow"].get("windows", {}).get("current_year", {})
        print(
            f"[{idx}/{len(grid)}] done "
            f"2026={float(current_year.get('total_return_pct', 0.0) or 0.0):.2f}%/"
            f"{float(current_year.get('max_drawdown_pct', 0.0) or 0.0):.2f}% "
            f"full={float(candidate['live_shadow'].get('total_return_pct', 0.0) or 0.0):.2f}%/"
            f"{float(candidate['live_shadow'].get('max_drawdown_pct', 0.0) or 0.0):.2f}%",
            flush=True,
        )

    ranked_by_2026 = sorted(candidates, key=sort_by_2026, reverse=True)
    ranked_by_return = sorted(candidates, key=sort_by_return, reverse=True)
    report = {
        "metadata": {
            "config": str(Path(args.config).resolve()),
            "pressure_params": str(Path(args.pressure_params).resolve()),
            "pressure_params_applied": pressure_params,
            "dynamic_params_locked": {key: conservative_dynamic.get(key) for key in PARAM_KEYS},
            "start_date": str(args.start_date),
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "candidate_count": len(candidates),
        },
        "top_by_2026": ranked_by_2026[: int(args.top_n)],
        "top_by_return": ranked_by_return[: int(args.top_n)],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output)
    for idx, item in enumerate(ranked_by_2026[:8], start=1):
        live = item["live_shadow"]
        year = live.get("windows", {}).get("current_year", {})
        print(
            f"{idx:02d} 2026={float(year.get('total_return_pct', 0.0) or 0.0):.2f}%/"
            f"{float(year.get('max_drawdown_pct', 0.0) or 0.0):.2f}% "
            f"full={float(live.get('total_return_pct', 0.0) or 0.0):.2f}%/"
            f"{float(live.get('max_drawdown_pct', 0.0) or 0.0):.2f}% "
            f"params={item['trailing_params']}"
        )


if __name__ == "__main__":
    main()
