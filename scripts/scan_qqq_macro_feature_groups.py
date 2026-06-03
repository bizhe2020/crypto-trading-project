#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qqq_drawdown_risk_model import (  # noqa: E402
    DEFAULT_BREADTH_PATH,
    DEFAULT_MACRO_PATH,
    DEFAULT_PUBLIC_DIR,
    build_feature_frame,
    run_walk_forward,
)


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_macro_feature_group_scan.json"


GROUP_MATCHERS: dict[str, list[str]] = {
    "rates": [
        "macro_fed_funds",
        "macro_treasury",
        "macro_real_yield",
        "macro_sofr",
    ],
    "inflation": [
        "macro_cpi",
        "macro_core_cpi",
        "macro_pce",
        "macro_core_pce",
    ],
    "labor_growth": [
        "macro_initial_claims",
        "macro_unemployment",
        "macro_nonfarm",
        "macro_avg_hourly",
        "macro_real_gdp",
    ],
    "dollar_credit_commodity_vol": [
        "macro_broad_dollar",
        "macro_high_yield",
        "macro_hy_oas",
        "macro_wti",
        "macro_vix_fred",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan optional FRED macro feature groups against the QQQ drawdown model.")
    parser.add_argument("--data-dir", default=str(DEFAULT_PUBLIC_DIR))
    parser.add_argument("--breadth-path", default=str(DEFAULT_BREADTH_PATH))
    parser.add_argument("--macro-path", default=str(DEFAULT_MACRO_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon-days", type=int, default=10)
    parser.add_argument("--drawdown-threshold-pct", type=float, default=5.0)
    parser.add_argument("--min-train-days", type=int, default=504)
    parser.add_argument("--calibration-days", type=int, default=126)
    parser.add_argument("--test-window-days", type=int, default=21)
    parser.add_argument("--embargo-days", type=int, default=10)
    parser.add_argument("--calibration-method", choices=["isotonic", "platt", "none"], default="platt")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def matches_any(feature: str, prefixes: list[str]) -> bool:
    return any(feature.startswith(prefix) for prefix in prefixes)


def macro_columns(feature_columns: list[str]) -> list[str]:
    return [feature for feature in feature_columns if feature.startswith("macro_")]


def evaluate_group(args: argparse.Namespace, frame: Any, feature_columns: list[str]) -> dict[str, Any]:
    _, metrics = run_walk_forward(
        frame,
        feature_columns,
        min_train_days=int(args.min_train_days),
        calibration_days=int(args.calibration_days),
        test_window_days=int(args.test_window_days),
        embargo_days=int(args.embargo_days),
        calibration_method=str(args.calibration_method),
        random_state=int(args.random_state),
    )
    return metrics


def main() -> None:
    args = parse_args()
    frame, all_features, missing_flags = build_feature_frame(
        data_dir=Path(args.data_dir),
        breadth_path=Path(args.breadth_path),
        macro_path=Path(args.macro_path),
        macro_groups=["all"],
        horizon_days=int(args.horizon_days),
        drawdown_threshold_pct=float(args.drawdown_threshold_pct),
    )
    macro_features = macro_columns(all_features)
    baseline_features = [feature for feature in all_features if not feature.startswith("macro_")]

    groups: dict[str, list[str]] = {"baseline": baseline_features}
    for group_name, prefixes in GROUP_MATCHERS.items():
        selected_macro = [feature for feature in macro_features if matches_any(feature, prefixes)]
        groups[group_name] = sorted(set(baseline_features + selected_macro))
    groups["all_macro"] = all_features

    results: list[dict[str, Any]] = []
    baseline_metrics: dict[str, Any] | None = None
    for group_name, features in groups.items():
        metrics = evaluate_group(args, frame, features)
        if group_name == "baseline":
            baseline_metrics = metrics
        delta: dict[str, float | None] = {}
        if baseline_metrics:
            for key in ["roc_auc_raw", "roc_auc_calibrated", "brier_raw", "brier_calibrated"]:
                current = metrics.get(key)
                base = baseline_metrics.get(key)
                delta[f"{key}_delta_vs_baseline"] = (
                    float(current) - float(base) if current is not None and base is not None else None
                )
        results.append(
            {
                "group": group_name,
                "feature_count": int(len(features)),
                "macro_feature_count": int(sum(feature.startswith("macro_") for feature in features)),
                "metrics": metrics,
                "delta_vs_baseline": delta,
            }
        )
        print(group_name, json.dumps(metrics, ensure_ascii=False), flush=True)

    payload = {
        "mode": "macro_feature_group_scan",
        "missing_data_flags": missing_flags,
        "groups": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output)


if __name__ == "__main__":
    main()
