#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_config_report import DEFAULT_DATA_15M, DEFAULT_DATA_4H, load_config_payload  # noqa: E402
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay, parse_float_list, parse_int_list, parse_str_list  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, add_windows, replay_shadow_events  # noqa: E402


def parse_bool_list(value: str) -> list[bool]:
    values: list[bool] = []
    for item in parse_str_list(value):
        normalized = item.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            values.append(True)
        elif normalized in {"0", "false", "no", "off"}:
            values.append(False)
        else:
            raise ValueError(f"Invalid boolean value: {item}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint trailing optimization scan for high-branch SOTA.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=48)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--output", default=str(ROOT / "var" / "high_leverage_expansion" / "joint_trailing_optimization_scan.json"))

    parser.add_argument("--atr-activation-rr-values", default="1.8,2.06,2.3")
    parser.add_argument("--atr-loose-multiplier-values", default="2.4,2.7,3.0")
    parser.add_argument("--atr-normal-multiplier-values", default="2.0,2.25,2.5")
    parser.add_argument("--atr-regime-filter-values", default="tight_style_off,all")

    parser.add_argument("--auto-tit-mode-values", default="loss_streak,health")
    parser.add_argument("--auto-tit-loss-streak-values", default="1,2")
    parser.add_argument("--auto-tit-atr-ratio-max-values", default="1.0,1.1,1.2")
    parser.add_argument("--auto-tit-trail-style-sets", default="loose,loose+normal")
    parser.add_argument("--auto-tit-regime-label-sets", default="high_growth,high_growth+normal")

    parser.add_argument("--normal-target-rr-cap-values", default="3.5,3.75,4.0")
    parser.add_argument("--tight-target-rr-cap-values", default="2.25,2.5,2.75")
    parser.add_argument("--loose-target-rr-cap-values", default="4.5,5.0,5.5")

    parser.add_argument("--pressure-min-rr-values", default="2.0")
    parser.add_argument("--pressure-lock-rr-values", default="0.3,0.4,0.5")
    parser.add_argument("--pressure-atr-multiplier-values", default="2.5,3.0")
    parser.add_argument("--pressure-proximity-pct-values", default="0.15,0.2")
    parser.add_argument("--pressure-target-min-rr-values", default="1.1,1.25,1.4")
    parser.add_argument("--pressure-target-buffer-pct-values", default="0.02,0.03")
    parser.add_argument("--pressure-touch-lock-min-rr-values", default="0.8,1.0,1.2")
    parser.add_argument("--pressure-touch-lock-buffer-pct-values", default="0.02,0.03")
    parser.add_argument("--pressure-touch-lock-requires-touch-values", default="false")
    parser.add_argument("--pressure-regime-label-sets", default="flat")
    return parser.parse_args()


def baseline_trailing_params(payload: dict[str, Any], pressure_defaults: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "enable_atr_trailing",
        "atr_activation_rr",
        "atr_loose_multiplier",
        "atr_normal_multiplier",
        "atr_regime_filter",
        "enable_time_based_trailing",
        "enable_auto_time_based_trailing",
        "auto_tit_mode",
        "auto_tit_loss_streak",
        "auto_tit_atr_ratio_max",
        "auto_tit_trail_styles",
        "auto_tit_regime_labels",
        "enable_target_rr_cap",
        "normal_target_rr_cap",
        "tight_target_rr_cap",
        "loose_target_rr_cap",
    }
    baseline = {key: deepcopy(payload.get(key)) for key in keys}
    baseline.update(deepcopy(pressure_defaults))
    return baseline


def add_candidate(candidates: list[dict[str, Any]], seen: set[str], candidate: dict[str, Any]) -> None:
    key = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
    if key in seen:
        return
    seen.add(key)
    candidates.append(candidate)


def build_candidate_params(args: argparse.Namespace, payload: dict[str, Any], pressure_defaults: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    baseline = baseline_trailing_params(payload, pressure_defaults)
    add_candidate(candidates, seen, baseline)

    scalar_options: dict[str, list[Any]] = {
        "atr_activation_rr": parse_float_list(args.atr_activation_rr_values),
        "atr_loose_multiplier": parse_float_list(args.atr_loose_multiplier_values),
        "atr_normal_multiplier": parse_float_list(args.atr_normal_multiplier_values),
        "atr_regime_filter": parse_str_list(args.atr_regime_filter_values),
        "auto_tit_mode": parse_str_list(args.auto_tit_mode_values),
        "auto_tit_loss_streak": parse_int_list(args.auto_tit_loss_streak_values),
        "auto_tit_atr_ratio_max": parse_float_list(args.auto_tit_atr_ratio_max_values),
        "normal_target_rr_cap": parse_float_list(args.normal_target_rr_cap_values),
        "tight_target_rr_cap": parse_float_list(args.tight_target_rr_cap_values),
        "loose_target_rr_cap": parse_float_list(args.loose_target_rr_cap_values),
        "pressure_min_rr": parse_float_list(args.pressure_min_rr_values),
        "pressure_lock_rr": parse_float_list(args.pressure_lock_rr_values),
        "pressure_atr_multiplier": parse_float_list(args.pressure_atr_multiplier_values),
        "pressure_proximity_pct": parse_float_list(args.pressure_proximity_pct_values),
        "pressure_target_min_rr": parse_float_list(args.pressure_target_min_rr_values),
        "pressure_target_buffer_pct": parse_float_list(args.pressure_target_buffer_pct_values),
        "pressure_touch_lock_min_rr": parse_float_list(args.pressure_touch_lock_min_rr_values),
        "pressure_touch_lock_buffer_pct": parse_float_list(args.pressure_touch_lock_buffer_pct_values),
        "pressure_touch_lock_requires_touch": parse_bool_list(args.pressure_touch_lock_requires_touch_values),
    }
    list_options: dict[str, list[list[str]]] = {
        "auto_tit_trail_styles": [item.split("+") for item in parse_str_list(args.auto_tit_trail_style_sets)],
        "auto_tit_regime_labels": [item.split("+") for item in parse_str_list(args.auto_tit_regime_label_sets)],
        "pressure_regime_labels": [item.split("+") for item in parse_str_list(args.pressure_regime_label_sets)],
    }

    for key, values in scalar_options.items():
        for value in values:
            candidate = deepcopy(baseline)
            candidate[key] = value
            add_candidate(candidates, seen, candidate)
            if len(candidates) >= int(args.candidate_limit):
                return candidates
    for key, values in list_options.items():
        for value in values:
            candidate = deepcopy(baseline)
            candidate[key] = value
            add_candidate(candidates, seen, candidate)
            if len(candidates) >= int(args.candidate_limit):
                return candidates

    combo_groups = [
        (
            ("atr_activation_rr", parse_float_list(args.atr_activation_rr_values)),
            ("atr_loose_multiplier", parse_float_list(args.atr_loose_multiplier_values)),
            ("atr_normal_multiplier", parse_float_list(args.atr_normal_multiplier_values)),
        ),
        (
            ("auto_tit_mode", parse_str_list(args.auto_tit_mode_values)),
            ("auto_tit_loss_streak", parse_int_list(args.auto_tit_loss_streak_values)),
            ("auto_tit_atr_ratio_max", parse_float_list(args.auto_tit_atr_ratio_max_values)),
        ),
        (
            ("pressure_lock_rr", parse_float_list(args.pressure_lock_rr_values)),
            ("pressure_target_min_rr", parse_float_list(args.pressure_target_min_rr_values)),
            ("pressure_touch_lock_min_rr", parse_float_list(args.pressure_touch_lock_min_rr_values)),
        ),
        (
            ("normal_target_rr_cap", parse_float_list(args.normal_target_rr_cap_values)),
            ("tight_target_rr_cap", parse_float_list(args.tight_target_rr_cap_values)),
            ("loose_target_rr_cap", parse_float_list(args.loose_target_rr_cap_values)),
        ),
    ]
    for group in combo_groups:
        keys = [item[0] for item in group]
        values = [item[1] for item in group]
        for combo in itertools.product(*values):
            candidate = deepcopy(baseline)
            for key, value in zip(keys, combo):
                candidate[key] = value
            add_candidate(candidates, seen, candidate)
            if len(candidates) >= int(args.candidate_limit):
                return candidates
    return candidates


def evaluate_candidate(
    payload: dict[str, Any],
    prepared: Any,
    start_date: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics, engine = run_engine(payload, prepared, start_date)
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
    shadow = add_windows(shadow, initial_capital)
    year = shadow.get("windows", {}).get("current_year", {})
    recent_60d = shadow.get("windows", {}).get("last_60d", {})
    recent_30d = shadow.get("windows", {}).get("last_30d", {})
    score = round(
        float(shadow["total_return_pct"])
        + float(year.get("total_return_pct", 0.0)) * 180.0
        + float(recent_60d.get("total_return_pct", 0.0)) * 80.0
        + float(recent_30d.get("total_return_pct", 0.0)) * 40.0
        - float(shadow["max_drawdown_pct"]) * 40.0
        - float(year.get("max_drawdown_pct", 0.0)) * 45.0,
        4,
    )
    return {
        "score": score,
        "engine": {
            "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 2),
            "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0)), 2),
            "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
            "total_trades": int(metrics.get("total_trades", 0)),
            "exit_reasons": metrics.get("exit_reasons", {}),
        },
        "fixed_structure_overlay": {key: value for key, value in fixed.items() if key != "events"},
        "shadow": shadow,
    }


def main() -> None:
    args = parse_args()
    base_payload = load_config_payload(Path(args.config))
    payload, pressure_defaults = apply_pressure_params(base_payload, Path(args.pressure_params))
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=payload.get("regime_switcher_thresholds"),
    )

    baseline_result = evaluate_candidate(payload, prepared, args.start_date, args)
    candidates = build_candidate_params(args, payload, pressure_defaults)
    results: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates, start=1):
        candidate_payload = deepcopy(payload)
        candidate_payload.update(candidate)
        result = evaluate_candidate(candidate_payload, prepared, args.start_date, args)
        result["params"] = candidate
        shadow = result["shadow"]
        year = shadow.get("windows", {}).get("current_year", {})
        result["delta_vs_baseline"] = {
            "total_return_pct": round(float(shadow["total_return_pct"]) - float(baseline_result["shadow"]["total_return_pct"]), 4),
            "max_drawdown_pct": round(float(shadow["max_drawdown_pct"]) - float(baseline_result["shadow"]["max_drawdown_pct"]), 4),
            "current_year_return_pct": round(float(year.get("total_return_pct", 0.0)) - float(baseline_result["shadow"]["windows"]["current_year"]["total_return_pct"]), 4),
        }
        results.append(result)
        print(
            f"{index:03d}/{len(candidates):03d} score={result['score']:.2f} "
            f"full={shadow['total_return_pct']:.2f}%/{shadow['max_drawdown_pct']:.2f}% "
            f"year={year.get('total_return_pct', 0.0):.2f}% "
            f"delta={result['delta_vs_baseline']['total_return_pct']:+.2f}% "
            f"dd={result['delta_vs_baseline']['max_drawdown_pct']:+.2f}%"
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "config": str(Path(args.config).resolve()),
        "pressure_params": str(Path(args.pressure_params).resolve()),
        "data": {
            "start": str(prepared.start),
            "end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
        },
        "shadow_params": {
            "daily_loss_stop_pct": args.daily_loss_stop_pct,
            "equity_drawdown_stop_pct": args.equity_drawdown_stop_pct,
            "equity_drawdown_cooldown_days": args.equity_drawdown_cooldown_days,
            "consecutive_loss_stop": args.consecutive_loss_stop,
        },
        "baseline": baseline_result,
        "candidate_count": len(results),
        "top": results[: int(args.top)],
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)
    for rank, item in enumerate(results[: int(args.top)], start=1):
        shadow = item["shadow"]
        year = shadow.get("windows", {}).get("current_year", {})
        print(
            f"{rank:02d} score={item['score']:.2f} full={shadow['total_return_pct']:.2f}%/"
            f"{shadow['max_drawdown_pct']:.2f}% year={year.get('total_return_pct', 0.0):.2f}% "
            f"delta={item['delta_vs_baseline']['total_return_pct']:+.2f}% "
            f"dd={item['delta_vs_baseline']['max_drawdown_pct']:+.2f}% params={item['params']}"
        )


if __name__ == "__main__":
    main()
