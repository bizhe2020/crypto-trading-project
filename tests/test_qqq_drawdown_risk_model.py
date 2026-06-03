from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_qqq_constituent_breadth import add_symbol_features, aggregate_breadth
from scripts.qqq_drawdown_risk_model import ProbabilityCalibrator, build_feature_frame


def write_ohlcv(path: Path, values: list[float]) -> None:
    dates = pd.date_range("2024-01-01", periods=len(values), freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": values,
            "high": [value * 1.01 for value in values],
            "low": [value * 0.99 for value in values],
            "close": values,
            "volume": [1000] * len(values),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_feather(path)


def test_future_drawdown_label_uses_next_rows_only(tmp_path: Path) -> None:
    data_dir = tmp_path / "etf"
    values = [100.0, 100.0, 94.0, 99.0, 101.0, 102.0]
    write_ohlcv(data_dir / "QQQ-1d.feather", values)

    frame, _, _ = build_feature_frame(
        data_dir,
        tmp_path / "missing_breadth.feather",
        tmp_path / "missing_macro.feather",
        [],
        horizon_days=2,
        drawdown_threshold_pct=5.0,
    )

    assert frame.loc[0, "target_drawdown"] == 1.0
    assert frame.loc[1, "target_drawdown"] == 1.0
    assert frame.loc[2, "target_drawdown"] == 0.0
    assert np.isnan(frame.loc[5, "target_drawdown"])


def test_probability_calibrator_orders_high_risk_above_low_risk() -> None:
    raw = np.array([0.05, 0.10, 0.20, 0.75, 0.85, 0.95])
    labels = np.array([0, 0, 0, 1, 1, 1])

    for method in ["isotonic", "platt"]:
        calibrator = ProbabilityCalibrator(method).fit(raw, labels)
        calibrated = calibrator.predict(np.array([0.1, 0.9]))
        assert 0.0 <= calibrated[0] <= 1.0
        assert 0.0 <= calibrated[1] <= 1.0
        assert calibrated[1] > calibrated[0]


def test_aggregate_breadth_counts_advancers_and_ma_membership() -> None:
    dates = pd.date_range("2024-01-01", periods=220, freq="B", tz="UTC")
    first = pd.DataFrame(
        {
            "date": dates,
            "open": np.linspace(90, 110, len(dates)),
            "high": np.linspace(91, 111, len(dates)),
            "low": np.linspace(89, 109, len(dates)),
            "close": np.linspace(90, 110, len(dates)),
            "volume": 1000,
        }
    )
    second = pd.DataFrame(
        {
            "date": dates,
            "open": np.linspace(110, 90, len(dates)),
            "high": np.linspace(111, 91, len(dates)),
            "low": np.linspace(109, 89, len(dates)),
            "close": np.linspace(110, 90, len(dates)),
            "volume": 1000,
        }
    )
    long_frame = pd.concat(
        [
            add_symbol_features(first, "AAA", 60.0),
            add_symbol_features(second, "BBB", 40.0),
        ],
        ignore_index=True,
    )

    breadth = aggregate_breadth(long_frame, total_symbols=2)
    latest = breadth.iloc[-1]

    assert latest["qqq_breadth_constituent_count"] == 2
    assert latest["qqq_breadth_data_coverage_pct"] == 100.0
    assert latest["qqq_breadth_advancers_pct"] == 50.0
    assert latest["qqq_breadth_above_ma200_pct"] == 50.0
