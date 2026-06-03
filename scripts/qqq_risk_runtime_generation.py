from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.qqq_drawdown_risk_model import write_outputs


def ensure_feature_columns(frame: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Missing required feature columns: {', '.join(sorted(missing))}")
    return list(feature_columns)


def append_latest_signal_row(
    predictions: pd.DataFrame,
    frame: pd.DataFrame,
    latest_signal: dict[str, Any],
    *,
    fold_train_end: pd.Timestamp | str | None,
) -> pd.DataFrame:
    if predictions.empty:
        out = predictions.copy()
    else:
        out = predictions.copy().sort_values("date").reset_index(drop=True)
    latest_date = pd.Timestamp(str(latest_signal["date"]), tz="UTC")
    if "date" in out.columns and not out.empty:
        current_latest = pd.Timestamp(out["date"].max())
        if current_latest >= latest_date:
            return out

    row = {column: pd.NA for column in out.columns}
    row["date"] = latest_date
    row["target_drawdown"] = pd.NA
    row["future_drawdown_pct"] = pd.NA
    latest_frame = frame.loc[pd.to_datetime(frame["date"], utc=True, errors="coerce") == latest_date]
    if "qqq_close" in out.columns:
        row["qqq_close"] = float(latest_frame["qqq_close"].iloc[-1]) if not latest_frame.empty else pd.NA
    if "raw_prob_10d" in out.columns:
        row["raw_prob_10d"] = latest_signal.get("raw_prob_10d")
    if "model_prob_10d" in out.columns:
        row["model_prob_10d"] = latest_signal.get("model_prob_10d")
    if "fold_train_end" in out.columns and fold_train_end is not None:
        row["fold_train_end"] = pd.Timestamp(fold_train_end)
    next_index = len(out)
    for column in out.columns:
        out.loc[next_index, column] = row.get(column, pd.NA)
    return out


def write_runtime_outputs(
    *,
    predictions: pd.DataFrame,
    report_payload: dict[str, Any],
    output_csv: Path,
    output_json: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_outputs(predictions, output_csv, report_payload, output_json)


def json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return str(value)
    return str(value)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n")
