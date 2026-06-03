#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qqq_drawdown_risk_model import (  # noqa: E402
    DEFAULT_PUBLIC_DIR,
    build_feature_frame,
    run_walk_forward,
    summarize_latest,
    top_feature_importance,
    train_latest_model,
)
from scripts.qqq_risk_runtime_generation import (  # noqa: E402
    append_latest_signal_row,
    ensure_feature_columns,
    write_runtime_outputs,
)


DEFAULT_BREADTH_PATH = ROOT / "data" / "public" / "breadth" / "disabled_for_long_cycle.feather"
DEFAULT_OUTPUT_CSV = ROOT / "var" / "reports" / "qqq_long_cycle_correction20d10_qqqonly_lgb_predictions.csv"
DEFAULT_OUTPUT_JSON = ROOT / "var" / "reports" / "qqq_long_cycle_correction20d10_qqqonly_lgb_report.json"

LONG_CYCLE_FEATURES = [
    "qqq_20d_high_dist",
    "qqq_60d_drawdown",
    "qqq_below_ma20_days",
    "qqq_close_to_range_location",
    "qqq_downside_semivol_20d",
    "qqq_gap_down_freq_10d",
    "qqq_intraday_range_pct",
    "qqq_ma200_dist",
    "qqq_ma20_dist",
    "qqq_ma20_slope_10d",
    "qqq_ma50_dist",
    "qqq_ma50_slope_10d",
    "qqq_range_pctile_252",
    "qqq_realized_vol_10d",
    "qqq_realized_vol_20d",
    "qqq_ret_10d",
    "qqq_ret_1d",
    "qqq_ret_20d",
    "qqq_ret_5d",
    "qqq_ret_60d",
    "qqq_ret_kurt_20d",
    "qqq_ret_skew_20d",
    "qqq_spy_rel_20d",
    "qqq_volume_z_60d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the frozen long-cycle QQQ risk CSV used by live runtime.")
    parser.add_argument("--data-dir", default=str(DEFAULT_PUBLIC_DIR))
    parser.add_argument("--breadth-path", default=str(DEFAULT_BREADTH_PATH))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--horizon-days", type=int, default=20)
    parser.add_argument("--drawdown-threshold-pct", type=float, default=10.0)
    parser.add_argument("--min-train-days", type=int, default=756)
    parser.add_argument("--calibration-days", type=int, default=252)
    parser.add_argument("--test-window-days", type=int, default=63)
    parser.add_argument("--embargo-days", type=int, default=20)
    parser.add_argument("--calibration-method", choices=["isotonic", "platt", "none"], default="platt")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, _, missing_flags = build_feature_frame(
        data_dir=Path(args.data_dir),
        breadth_path=Path(args.breadth_path),
        macro_path=None,
        macro_groups=[],
        horizon_days=int(args.horizon_days),
        drawdown_threshold_pct=float(args.drawdown_threshold_pct),
    )
    selected_features = ensure_feature_columns(frame, LONG_CYCLE_FEATURES)
    predictions, metrics = run_walk_forward(
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
    predictions = append_latest_signal_row(
        predictions,
        frame,
        latest_signal,
        fold_train_end=latest_train["date"].iloc[-1] if not latest_train.empty else None,
    )
    payload = {
        "generated_at_utc": str(pd.Timestamp.utcnow()),
        "mode": "qqq_long_cycle_correction20d10_qqqonly_lgb",
        "label": {
            "name": "future_20d_max_drawdown",
            "horizon_days": int(args.horizon_days),
            "drawdown_threshold_pct": -abs(float(args.drawdown_threshold_pct)),
        },
        "model": {
            "type": "LightGBMClassifier",
            "calibration_method": str(args.calibration_method),
            "walk_forward": {
                "min_train_days": int(args.min_train_days),
                "calibration_days": int(args.calibration_days),
                "test_window_days": int(args.test_window_days),
                "embargo_days": int(args.embargo_days),
            },
        },
        "data": {
            "rows": int(len(frame)),
            "start": str(pd.Timestamp(frame["date"].min()).date()),
            "end": str(pd.Timestamp(frame["date"].max()).date()),
            "feature_count": int(len(selected_features)),
            "missing_data_flags": missing_flags,
        },
        "feature_set": "qqq_only_long_history_price_features",
        "feature_columns": selected_features,
        "walk_forward_metrics": metrics,
        "latest_shadow_signal": latest_signal,
        "top_feature_importance": top_feature_importance(latest_fold),
        "prediction_csv": str(Path(args.output_csv).resolve()),
    }
    write_runtime_outputs(
        predictions=predictions,
        report_payload=payload,
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
    )
    print(Path(args.output_csv))
    print(Path(args.output_json))
    print(json.dumps({"latest_shadow_signal": latest_signal, "walk_forward_metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
