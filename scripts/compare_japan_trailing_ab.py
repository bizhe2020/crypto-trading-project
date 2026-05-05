#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.report_smc_trade_context import daily_candles_from_4h  # noqa: E402
from scripts.replay_sota_smc_live_shadow import (  # noqa: E402
    apply_trailing_rr_modes,
    compact_combo_with_events,
    compact_live_result,
    decision_counts,
    replay_live_shadow,
)
from scripts.live_shadow_utils import (  # noqa: E402
    compact_result,
    clean_for_json,
    event_stream_summary,
    standard_sota_event,
)
from scripts.smc_live_utils import (  # noqa: E402
    SMC_CASES,
    build_smc_events,
    live_feasibility_audit,
    replay_base_priority_sota_first,
)
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "japan_trailing_ab.json"
RR_MODE_CHOICES = ("close", "extreme")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare replay with current Japan-style trailing enabled vs disabled under the same SOTA/SMC live-shadow pipeline."
    )
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
        "--replay-sync-entry-to-signal-price",
        action="store_true",
        help="Sync replay entry execution price back to signal entry price after modeled open slippage.",
    )
    parser.add_argument("--smc-case", default="v2_medium_dispbody05_otherlag4_10x", choices=sorted(SMC_CASES))
    parser.add_argument("--smc-allocation", type=float, default=1.0)
    parser.add_argument("--stage-trigger-rr-mode", default="close", choices=RR_MODE_CHOICES)
    parser.add_argument("--time-trailing-rr-mode", default="close", choices=RR_MODE_CHOICES)
    parser.add_argument("--atr-activation-rr-mode", default="close", choices=RR_MODE_CHOICES)
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=0)
    parser.add_argument("--sample-trades", type=int, default=20)
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def trailing_off_payload(payload: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(payload)
    updated["enable_stage_trailing"] = False
    updated["enable_atr_trailing"] = False
    updated["enable_time_based_trailing"] = False
    updated["enable_auto_time_based_trailing"] = False
    updated["enable_pressure_level_trailing"] = False
    updated["enable_target_rr_cap"] = False
    updated["pressure_enable_target_cap"] = False
    updated["pressure_touch_lock_enabled"] = False
    return updated


def stage_only_payload(payload: dict[str, Any]) -> dict[str, Any]:
    updated = trailing_off_payload(payload)
    updated["enable_stage_trailing"] = True
    return updated


def stage_target_cap_payload(payload: dict[str, Any]) -> dict[str, Any]:
    updated = stage_only_payload(payload)
    updated["enable_target_rr_cap"] = bool(payload.get("enable_target_rr_cap", False))
    updated["loose_target_rr_cap"] = payload.get("loose_target_rr_cap")
    updated["normal_target_rr_cap"] = payload.get("normal_target_rr_cap")
    updated["tight_target_rr_cap"] = payload.get("tight_target_rr_cap")
    return updated


def stage_target_atr_payload(payload: dict[str, Any]) -> dict[str, Any]:
    updated = stage_target_cap_payload(payload)
    updated["enable_atr_trailing"] = bool(payload.get("enable_atr_trailing", False))
    updated["atr_regime_filter"] = payload.get("atr_regime_filter", "all")
    updated["atr_period"] = payload.get("atr_period", 14)
    updated["atr_activation_rr"] = payload.get("atr_activation_rr", 2.0)
    updated["atr_activation_rr_mode"] = payload.get("atr_activation_rr_mode", "close")
    updated["atr_loose_multiplier"] = payload.get("atr_loose_multiplier", 2.7)
    updated["atr_normal_multiplier"] = payload.get("atr_normal_multiplier", 2.25)
    updated["atr_tight_multiplier"] = payload.get("atr_tight_multiplier", 1.8)
    return updated


def stage_target_atr_time_payload(payload: dict[str, Any]) -> dict[str, Any]:
    updated = stage_target_atr_payload(payload)
    updated["enable_time_based_trailing"] = bool(payload.get("enable_time_based_trailing", False))
    updated["enable_auto_time_based_trailing"] = bool(payload.get("enable_auto_time_based_trailing", False))
    return updated


def pressure_only_payload(payload: dict[str, Any]) -> dict[str, Any]:
    updated = trailing_off_payload(payload)
    updated["enable_pressure_level_trailing"] = bool(payload.get("enable_pressure_level_trailing", False))
    updated["pressure_enable_target_cap"] = bool(payload.get("pressure_enable_target_cap", False))
    updated["pressure_touch_lock_enabled"] = bool(payload.get("pressure_touch_lock_enabled", False))
    return updated


def ablation_payloads(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "stage_only": stage_only_payload(payload),
        "stage_target_cap": stage_target_cap_payload(payload),
        "stage_target_atr": stage_target_atr_payload(payload),
        "stage_target_atr_time": stage_target_atr_time_payload(payload),
        "pressure_only": pressure_only_payload(payload),
    }


def compact_engine_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_return_pct": round(float(metrics.get("total_return_pct", 0.0) or 0.0), 2),
        "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0) or 0.0), 2),
        "profit_factor": round(float(metrics.get("profit_factor", 0.0) or 0.0), 4),
        "win_rate": round(float(metrics.get("win_rate", 0.0) or 0.0), 2),
        "target_hit_rate": round(float(metrics.get("target_hit_rate", 0.0) or 0.0), 2),
        "total_trades": int(metrics.get("total_trades", 0) or 0),
        "exit_reasons": metrics.get("exit_reasons", {}),
    }


def pick_window(summary: dict[str, Any], name: str) -> dict[str, Any]:
    window = summary.get("windows", {}).get(name, {})
    return {
        "total_return_pct": round(float(window.get("total_return_pct", 0.0) or 0.0), 2),
        "max_drawdown_pct": round(float(window.get("max_drawdown_pct", 0.0) or 0.0), 2),
        "profit_factor": round(float(window.get("profit_factor", 0.0) or 0.0), 4),
        "trades": int(window.get("trades", 0) or 0),
        "win_rate_pct": round(float(window.get("win_rate_pct", 0.0) or 0.0), 2),
    }


def compare_blocks(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_return_pct": round(float(left.get("total_return_pct", 0.0) or 0.0) - float(right.get("total_return_pct", 0.0) or 0.0), 2),
        "max_drawdown_pct": round(float(left.get("max_drawdown_pct", 0.0) or 0.0) - float(right.get("max_drawdown_pct", 0.0) or 0.0), 2),
        "profit_factor": round(float(left.get("profit_factor", 0.0) or 0.0) - float(right.get("profit_factor", 0.0) or 0.0), 4),
        "trades": int(left.get("trades", left.get("total_trades", 0)) or 0) - int(right.get("trades", right.get("total_trades", 0)) or 0),
        "win_rate_pct": round(
            float(left.get("win_rate_pct", left.get("win_rate", 0.0)) or 0.0)
            - float(right.get("win_rate_pct", right.get("win_rate", 0.0)) or 0.0),
            2,
        ),
    }


def build_static_smc_bundle(
    payload: dict[str, Any],
    args: argparse.Namespace,
    prepared: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    daily = daily_candles_from_4h(prepared.c4h)
    h4_highs, h4_lows = precompute_swings(prepared.c4h, n=2, lookback=80)
    d1_highs, d1_lows = precompute_swings(daily, n=2, lookback=20)
    return build_smc_events(
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


def variant_report(
    name: str,
    payload: dict[str, Any],
    args: argparse.Namespace,
    prepared: Any,
    smc_bundle: tuple[list[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
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
    base_shadow_summary = event_stream_summary(shadow_events, initial_capital, prepared.end)
    base_events = [standard_sota_event(event) for event in shadow_events]

    smc_events, smc_summary = smc_bundle

    reference = replay_base_priority_sota_first(
        base_events,
        smc_events,
        initial_capital,
        prepared.end,
        base_shadow_summary,
    )
    live, decisions = replay_live_shadow(
        base_events + smc_events,
        initial_capital,
        prepared.end,
        base_shadow_summary,
    )

    base_engine_summary = compact_engine_metrics(metrics)
    base_shadow_compact = compact_result(base_shadow_summary, int(args.sample_trades))
    reference_compact = compact_combo_with_events(reference, int(args.sample_trades))
    live_compact = compact_live_result(live, int(args.sample_trades))

    return {
        "variant": name,
        "config_flags": {
            "enable_stage_trailing": bool(payload.get("enable_stage_trailing", True)),
            "enable_atr_trailing": bool(payload.get("enable_atr_trailing", False)),
            "enable_time_based_trailing": bool(payload.get("enable_time_based_trailing", False)),
            "enable_auto_time_based_trailing": bool(payload.get("enable_auto_time_based_trailing", False)),
            "enable_pressure_level_trailing": bool(payload.get("enable_pressure_level_trailing", False)),
            "enable_target_rr_cap": bool(payload.get("enable_target_rr_cap", False)),
            "pressure_enable_target_cap": bool(payload.get("pressure_enable_target_cap", False)),
            "pressure_touch_lock_enabled": bool(payload.get("pressure_touch_lock_enabled", False)),
        },
        "base_engine": base_engine_summary,
        "baseline_shadow_sota": base_shadow_compact,
        "candidate_generation": {
            "sota_candidates": len(base_events),
            "smc_candidates": len(smc_events),
            "smc_summary": smc_summary,
        },
        "reference_base_priority_sota_first": reference_compact,
        "live_shadow": live_compact,
        "decision_counts": decision_counts(decisions),
        "live_feasibility_audit": live_feasibility_audit(live, initial_capital),
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
    )

    smc_bundle = build_static_smc_bundle(payload, args, prepared)
    trailing_on = variant_report("trailing_on", payload, args, prepared, smc_bundle)
    trailing_off = variant_report("trailing_off", trailing_off_payload(payload), args, prepared, smc_bundle)

    variants = {
        "trailing_on": trailing_on,
        "trailing_off": trailing_off,
    }
    if args.include_ablations:
        for name, variant_payload in ablation_payloads(payload).items():
            variants[name] = variant_report(name, variant_payload, args, prepared, smc_bundle)

    comparison = {
        "base_engine": compare_blocks(trailing_on["base_engine"], trailing_off["base_engine"]),
        "baseline_shadow_sota": compare_blocks(
            trailing_on["baseline_shadow_sota"],
            trailing_off["baseline_shadow_sota"],
        ),
        "reference_base_priority_sota_first": compare_blocks(
            trailing_on["reference_base_priority_sota_first"],
            trailing_off["reference_base_priority_sota_first"],
        ),
        "live_shadow": compare_blocks(
            trailing_on["live_shadow"],
            trailing_off["live_shadow"],
        ),
        "windows": {
            "baseline_shadow_sota": {
                "current_year": compare_blocks(
                    pick_window(trailing_on["baseline_shadow_sota"], "current_year"),
                    pick_window(trailing_off["baseline_shadow_sota"], "current_year"),
                ),
                "last_60d": compare_blocks(
                    pick_window(trailing_on["baseline_shadow_sota"], "last_60d"),
                    pick_window(trailing_off["baseline_shadow_sota"], "last_60d"),
                ),
                "last_30d": compare_blocks(
                    pick_window(trailing_on["baseline_shadow_sota"], "last_30d"),
                    pick_window(trailing_off["baseline_shadow_sota"], "last_30d"),
                ),
            },
            "live_shadow": {
                "current_year": compare_blocks(
                    pick_window(trailing_on["live_shadow"], "current_year"),
                    pick_window(trailing_off["live_shadow"], "current_year"),
                ),
                "last_60d": compare_blocks(
                    pick_window(trailing_on["live_shadow"], "last_60d"),
                    pick_window(trailing_off["live_shadow"], "last_60d"),
                ),
                "last_30d": compare_blocks(
                    pick_window(trailing_on["live_shadow"], "last_30d"),
                    pick_window(trailing_off["live_shadow"], "last_30d"),
                ),
            },
        },
    }
    ablation_comparison = {
        name: {
            "live_shadow_vs_trailing_off": compare_blocks(block["live_shadow"], trailing_off["live_shadow"]),
            "live_shadow_vs_trailing_on": compare_blocks(block["live_shadow"], trailing_on["live_shadow"]),
            "baseline_shadow_sota_vs_trailing_off": compare_blocks(block["baseline_shadow_sota"], trailing_off["baseline_shadow_sota"]),
            "current_year_live_vs_trailing_off": compare_blocks(
                pick_window(block["live_shadow"], "current_year"),
                pick_window(trailing_off["live_shadow"], "current_year"),
            ),
            "current_year_live_vs_trailing_on": compare_blocks(
                pick_window(block["live_shadow"], "current_year"),
                pick_window(trailing_on["live_shadow"], "current_year"),
            ),
        }
        for name, block in variants.items()
        if name not in {"trailing_on", "trailing_off"}
    }

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
            "replay_sync_entry_to_signal_price": bool(args.replay_sync_entry_to_signal_price),
            "smc_case": args.smc_case,
            "shadow_gate": {
                "daily_loss_stop_pct": args.daily_loss_stop_pct,
                "equity_drawdown_stop_pct": args.equity_drawdown_stop_pct,
                "equity_drawdown_cooldown_days": args.equity_drawdown_cooldown_days,
                "consecutive_loss_stop": args.consecutive_loss_stop,
            },
        },
        "variants": variants,
        "trailing_on": trailing_on,
        "trailing_off": trailing_off,
        "comparison_trailing_on_minus_off": comparison,
        "ablation_comparison": ablation_comparison,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    print(output)
    for label, block in variants.items():
        base_engine = block["base_engine"]
        shadow = block["baseline_shadow_sota"]
        live = block["live_shadow"]
        print(
            f"{label}: "
            f"engine={base_engine['total_return_pct']:.2f}%/{base_engine['max_drawdown_pct']:.2f}% "
            f"shadow={shadow['total_return_pct']:.2f}%/{shadow['max_drawdown_pct']:.2f}% "
            f"live={live['total_return_pct']:.2f}%/{live['max_drawdown_pct']:.2f}% "
            f"2026_live={pick_window(live, 'current_year')['total_return_pct']:.2f}%"
        )
    delta = comparison["live_shadow"]
    delta_2026 = comparison["windows"]["live_shadow"]["current_year"]
    print(
        "live_shadow delta on-off: "
        f"full_return={delta['total_return_pct']:+.2f}% "
        f"full_dd={delta['max_drawdown_pct']:+.2f}% "
        f"2026_return={delta_2026['total_return_pct']:+.2f}% "
        f"2026_dd={delta_2026['max_drawdown_pct']:+.2f}%"
    )


if __name__ == "__main__":
    main()
