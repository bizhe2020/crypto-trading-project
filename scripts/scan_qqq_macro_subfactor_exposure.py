#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qqq_drawdown_risk_model import (  # noqa: E402
    DEFAULT_BREADTH_PATH,
    DEFAULT_MACRO_PATH,
    DEFAULT_PUBLIC_DIR,
    build_feature_frame,
    risk_bucket,
    run_walk_forward,
    suggested_exposure,
    summarize_latest,
    top_feature_importance,
    train_latest_model,
)


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_macro_subfactor_exposure_scan.json"
DEFAULT_PREDICTIONS_CSV = ROOT / "var" / "reports" / "qqq_drawdown_lgb_shadow_predictions_macro_subfactor_core.csv"

FAMILY_PREFIXES: dict[str, list[str]] = {
    "broad_dollar_index": ["macro_broad_dollar_index"],
    "high_yield_oas": ["macro_high_yield_oas"],
    "hy_oas_plus_10y": ["macro_hy_oas_plus_10y"],
    "vix_fred": ["macro_vix_fred"],
    "wti_oil": ["macro_wti_oil"],
}

DEFAULT_FOCUS_FAMILIES = ["broad_dollar_index", "wti_oil"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QQQ macro subfactor ablation and out-of-sample exposure threshold scans."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_PUBLIC_DIR))
    parser.add_argument("--breadth-path", default=str(DEFAULT_BREADTH_PATH))
    parser.add_argument("--macro-path", default=str(DEFAULT_MACRO_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--predictions-csv", default=str(DEFAULT_PREDICTIONS_CSV))
    parser.add_argument(
        "--focus-families",
        default=",".join(DEFAULT_FOCUS_FAMILIES),
        help="Comma-separated macro families for variant-level exhaustive scan.",
    )
    parser.add_argument("--horizon-days", type=int, default=10)
    parser.add_argument("--drawdown-threshold-pct", type=float, default=5.0)
    parser.add_argument("--min-train-days", type=int, default=504)
    parser.add_argument("--calibration-days", type=int, default=126)
    parser.add_argument("--test-window-days", type=int, default=21)
    parser.add_argument("--embargo-days", type=int, default=10)
    parser.add_argument("--calibration-method", choices=["isotonic", "platt", "none"], default="platt")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.80)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def selected_family_features(macro_features: list[str], families: list[str]) -> list[str]:
    prefixes: list[str] = []
    for family in families:
        if family not in FAMILY_PREFIXES:
            raise ValueError(f"Unknown macro family: {family}")
        prefixes.extend(FAMILY_PREFIXES[family])
    return sorted(feature for feature in macro_features if any(feature.startswith(prefix) for prefix in prefixes))


def evaluate_feature_set(
    frame: pd.DataFrame,
    feature_columns: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
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


def metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    return float(value) if value is not None else float("-inf")


def scan_family_subsets(
    frame: pd.DataFrame,
    base_features: list[str],
    macro_features: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    family_names = list(FAMILY_PREFIXES)
    for size in range(len(family_names) + 1):
        for families_tuple in itertools.combinations(family_names, size):
            families = list(families_tuple)
            selected_macro = selected_family_features(macro_features, families)
            feature_columns = sorted(set(base_features + selected_macro))
            metrics = evaluate_feature_set(frame, feature_columns, args)
            row = {
                "families": families,
                "feature_count": int(len(feature_columns)),
                "macro_feature_count": int(len(selected_macro)),
                "macro_features": selected_macro,
                "metrics": metrics,
            }
            results.append(row)
            print(
                "family",
                ",".join(families) if families else "baseline",
                json.dumps(metrics, ensure_ascii=False),
                flush=True,
            )
    return sorted(results, key=lambda row: metric_value(row["metrics"], "roc_auc_raw"), reverse=True)


def scan_variant_subsets(
    frame: pd.DataFrame,
    base_features: list[str],
    candidate_macro_features: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for size in range(len(candidate_macro_features) + 1):
        for macro_tuple in itertools.combinations(candidate_macro_features, size):
            macros = list(macro_tuple)
            feature_columns = sorted(set(base_features + macros))
            metrics = evaluate_feature_set(frame, feature_columns, args)
            row = {
                "macro_features": macros,
                "feature_count": int(len(feature_columns)),
                "macro_feature_count": int(len(macros)),
                "metrics": metrics,
            }
            results.append(row)
            print(
                "variant",
                ",".join(macros) if macros else "baseline",
                json.dumps(metrics, ensure_ascii=False),
                flush=True,
            )
    return sorted(results, key=lambda row: metric_value(row["metrics"], "roc_auc_raw"), reverse=True)


def write_predictions_csv(predictions: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv = predictions.copy()
    if not csv.empty:
        csv["risk_score_0_100"] = (csv["model_prob_10d"] * 100.0).round(1)
        csv["risk_bucket"] = csv["model_prob_10d"].map(lambda value: risk_bucket(float(value)))
        csv["suggested_exposure"] = csv["model_prob_10d"].map(lambda value: suggested_exposure(float(value)))
    csv.to_csv(output_csv, index=False)


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min() * 100.0)


def exposure_metrics(frame: pd.DataFrame, exposure: pd.Series | None = None) -> dict[str, Any]:
    returns = pd.to_numeric(frame["next_ret"], errors="coerce").fillna(0.0).astype(float)
    if exposure is None:
        weights = pd.Series(1.0, index=returns.index)
    else:
        weights = exposure.reindex(returns.index).fillna(1.0).astype(float)
    strategy_returns = returns * weights
    equity = (1.0 + strategy_returns).cumprod()
    rows = int(len(strategy_returns))
    total_return = float(equity.iloc[-1] - 1.0) if rows else 0.0
    annual_return = float(equity.iloc[-1] ** (252.0 / rows) - 1.0) if rows and equity.iloc[-1] > 0 else None
    annual_vol = float(strategy_returns.std(ddof=0) * math.sqrt(252.0)) if rows else 0.0
    sharpe = (
        float(strategy_returns.mean() / strategy_returns.std(ddof=0) * math.sqrt(252.0))
        if rows and strategy_returns.std(ddof=0) > 0.0
        else None
    )
    target = pd.to_numeric(frame["target_drawdown"], errors="coerce").fillna(0).astype(int)
    event_weights = weights[target == 1]
    non_event_weights = weights[target == 0]
    return {
        "rows": rows,
        "total_return_pct": round(total_return * 100.0, 2),
        "annual_return_pct": round(annual_return * 100.0, 2) if annual_return is not None else None,
        "max_drawdown_pct": round(max_drawdown_pct(equity), 2),
        "annual_volatility_pct": round(annual_vol * 100.0, 2),
        "sharpe": round(sharpe, 3) if sharpe is not None else None,
        "avg_exposure": round(float(weights.mean()), 4),
        "event_avg_exposure": round(float(event_weights.mean()), 4) if not event_weights.empty else None,
        "non_event_avg_exposure": round(float(non_event_weights.mean()), 4) if not non_event_weights.empty else None,
        "reduced_days": int((weights < 0.999).sum()),
        "zero_days": int((weights <= 1e-12).sum()),
    }


def threshold_values(min_value: float, max_value: float, step: float) -> list[float]:
    if step <= 0.0:
        raise ValueError("--threshold-step must be positive")
    values: list[float] = []
    current = float(min_value)
    while current <= float(max_value) + 1e-12:
        values.append(round(current, 6))
        current += float(step)
    return values


def scan_exposure_thresholds(predictions: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    frame = predictions.copy()
    frame["next_ret"] = frame["qqq_close"].shift(-1) / frame["qqq_close"] - 1.0
    frame = frame.dropna(subset=["next_ret"]).reset_index(drop=True)

    baseline = exposure_metrics(frame)
    policies: list[dict[str, Any]] = []
    thresholds = threshold_values(float(args.threshold_min), float(args.threshold_max), float(args.threshold_step))
    for score_column in ["raw_prob_10d", "model_prob_10d"]:
        if score_column not in frame:
            continue
        for threshold in thresholds:
            for risk_exposure in [0.0, 0.25, 0.5, 0.75]:
                exposure = pd.Series(
                    np.where(frame[score_column].astype(float) >= float(threshold), float(risk_exposure), 1.0),
                    index=frame.index,
                )
                metrics = exposure_metrics(frame, exposure)
                return_retention = (
                    metrics["total_return_pct"] / baseline["total_return_pct"] * 100.0
                    if baseline["total_return_pct"]
                    else None
                )
                policy = {
                    "policy": "single_threshold",
                    "score_column": score_column,
                    "threshold": float(threshold),
                    "risk_exposure": float(risk_exposure),
                    **metrics,
                    "return_retention_pct": round(float(return_retention), 2) if return_retention is not None else None,
                    "max_dd_improvement_pct": round(
                        float(metrics["max_drawdown_pct"]) - float(baseline["max_drawdown_pct"]),
                        2,
                    ),
                }
                policies.append(policy)

    suggested = pd.Series(predictions["model_prob_10d"].map(lambda value: suggested_exposure(float(value)))).iloc[
        : len(frame)
    ]
    suggested_metrics = exposure_metrics(frame, suggested.reset_index(drop=True))
    suggested_retention = (
        suggested_metrics["total_return_pct"] / baseline["total_return_pct"] * 100.0
        if baseline["total_return_pct"]
        else None
    )
    suggested_policy = {
        "policy": "current_suggested_exposure",
        "score_column": "model_prob_10d",
        "threshold": None,
        "risk_exposure": None,
        **suggested_metrics,
        "return_retention_pct": round(float(suggested_retention), 2) if suggested_retention is not None else None,
        "max_dd_improvement_pct": round(
            float(suggested_metrics["max_drawdown_pct"]) - float(baseline["max_drawdown_pct"]),
            2,
        ),
    }
    policies.append(suggested_policy)

    eligible = [
        policy
        for policy in policies
        if policy.get("policy") == "single_threshold"
        and policy.get("return_retention_pct") is not None
        and float(policy["return_retention_pct"]) >= 90.0
        and float(policy["avg_exposure"]) >= 0.5
    ]
    eligible.sort(
        key=lambda item: (
            float(item["max_dd_improvement_pct"]),
            float(item["annual_return_pct"]) if item["annual_return_pct"] is not None else float("-inf"),
        ),
        reverse=True,
    )
    top_return = sorted(
        [policy for policy in policies if policy.get("policy") == "single_threshold"],
        key=lambda item: float(item["annual_return_pct"]) if item["annual_return_pct"] is not None else float("-inf"),
        reverse=True,
    )
    return {
        "return_model": "signal-date exposure applied to next trading day's close-to-close QQQ return; no costs.",
        "baseline_buy_and_hold": baseline,
        "current_suggested_exposure": suggested_policy,
        "best_tradeoff_90pct_return_retention": eligible[0] if eligible else None,
        "top_tradeoff_90pct_return_retention": eligible[: int(args.top)],
        "top_annual_return": top_return[: int(args.top)],
        "policies": policies,
    }


def json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def main() -> None:
    args = parse_args()
    focus_families = parse_csv_list(args.focus_families)

    frame, all_features, missing_flags = build_feature_frame(
        data_dir=Path(args.data_dir),
        breadth_path=Path(args.breadth_path),
        macro_path=Path(args.macro_path),
        macro_groups=["dollar_credit_commodity_vol"],
        horizon_days=int(args.horizon_days),
        drawdown_threshold_pct=float(args.drawdown_threshold_pct),
    )
    base_features = [feature for feature in all_features if not feature.startswith("macro_")]
    macro_features = [feature for feature in all_features if feature.startswith("macro_")]

    family_results = scan_family_subsets(frame, base_features, macro_features, args)
    variant_candidates = selected_family_features(macro_features, focus_families)
    variant_results = scan_variant_subsets(frame, base_features, variant_candidates, args)
    selected = variant_results[0] if variant_results else family_results[0]
    selected_features = sorted(set(base_features + selected["macro_features"]))

    selected_predictions, selected_metrics = run_walk_forward(
        frame,
        selected_features,
        min_train_days=int(args.min_train_days),
        calibration_days=int(args.calibration_days),
        test_window_days=int(args.test_window_days),
        embargo_days=int(args.embargo_days),
        calibration_method=str(args.calibration_method),
        random_state=int(args.random_state),
    )
    latest_fold, latest_train = train_latest_model(
        frame,
        selected_features,
        calibration_days=int(args.calibration_days),
        embargo_days=int(args.embargo_days),
        calibration_method=str(args.calibration_method),
        random_state=int(args.random_state),
    )
    latest_signal = summarize_latest(
        frame,
        selected_features,
        latest_fold,
        latest_train,
        missing_flags,
    )
    predictions_csv = Path(args.predictions_csv)
    write_predictions_csv(selected_predictions, predictions_csv)

    payload = {
        "generated_at_utc": str(pd.Timestamp.utcnow()),
        "mode": "qqq_macro_subfactor_exposure_scan",
        "scan_config": {
            "macro_group": "dollar_credit_commodity_vol",
            "focus_families": focus_families,
            "horizon_days": int(args.horizon_days),
            "drawdown_threshold_pct": -abs(float(args.drawdown_threshold_pct)),
            "min_train_days": int(args.min_train_days),
            "calibration_days": int(args.calibration_days),
            "test_window_days": int(args.test_window_days),
            "embargo_days": int(args.embargo_days),
            "calibration_method": str(args.calibration_method),
            "random_state": int(args.random_state),
        },
        "data": {
            "rows": int(len(frame)),
            "start": str(pd.Timestamp(frame["date"].min()).date()),
            "end": str(pd.Timestamp(frame["date"].max()).date()),
            "baseline_feature_count": int(len(base_features)),
            "macro_feature_count": int(len(macro_features)),
            "missing_data_flags": missing_flags,
        },
        "family_scan": {
            "families": FAMILY_PREFIXES,
            "results": family_results,
            "top_results": family_results[: int(args.top)],
        },
        "variant_scan": {
            "focus_families": focus_families,
            "candidate_macro_features": variant_candidates,
            "results": variant_results,
            "top_results": variant_results[: int(args.top)],
        },
        "selected_model": {
            "selection_rule": "best raw walk-forward AUC from variant scan over focus_families",
            "macro_features": selected["macro_features"],
            "feature_count": int(len(selected_features)),
            "macro_feature_count": int(len(selected["macro_features"])),
            "walk_forward_metrics": selected_metrics,
            "latest_shadow_signal": latest_signal,
            "top_feature_importance": top_feature_importance(latest_fold),
            "prediction_csv": str(predictions_csv),
        },
        "exposure_scan": scan_exposure_thresholds(selected_predictions, args),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default))
    print(output)
    print(json.dumps(payload["selected_model"]["walk_forward_metrics"], ensure_ascii=False))
    print(json.dumps(payload["exposure_scan"]["best_tradeoff_90pct_return_retention"], ensure_ascii=False))


if __name__ == "__main__":
    main()
