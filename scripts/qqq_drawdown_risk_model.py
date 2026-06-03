#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
except Exception as exc:  # pragma: no cover - exercised by runtime dependency checks.
    raise SystemExit(
        "Missing model dependencies. Install lightgbm and scikit-learn before running "
        "scripts/qqq_drawdown_risk_model.py."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_BREADTH_PATH = ROOT / "data" / "public" / "breadth" / "qqq_constituent_breadth-1d.feather"
DEFAULT_MACRO_PATH = ROOT / "data" / "public" / "macro" / "fred_macro-1d.feather"
DEFAULT_OUTPUT_JSON = ROOT / "var" / "reports" / "qqq_drawdown_lgb_shadow_report.json"
DEFAULT_OUTPUT_CSV = ROOT / "var" / "reports" / "qqq_drawdown_lgb_shadow_predictions.csv"


OPTIONAL_SYMBOLS: dict[str, str] = {
    "SPY": "spy",
    "^IXIC": "ixic",
    "^VIX": "vix",
    "^VIX3M": "vix3m",
    "^VVIX": "vvix",
    "^SKEW": "skew",
    "QQEW": "qqew",
    "RSP": "rsp",
    "SMH": "smh",
    "XLY": "xly",
    "XLP": "xlp",
    "HYG": "hyg",
    "IEF": "ief",
    "LQD": "lqd",
    "TLT": "tlt",
    "GLD": "gld",
    "USO": "uso",
    "CPER": "cper",
    "UUP": "uup",
    "BTC-USD": "btc",
}


MACRO_GROUP_PREFIXES: dict[str, list[str]] = {
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


FEATURE_LABELS: dict[str, str] = {
    "qqq_ret_10d": "QQQ 10d return",
    "qqq_ret_20d": "QQQ 20d return",
    "qqq_ma20_dist": "QQQ distance to MA20",
    "qqq_ma50_dist": "QQQ distance to MA50",
    "qqq_ma200_dist": "QQQ distance to MA200",
    "qqq_60d_drawdown": "QQQ drawdown from 60d high",
    "qqq_range_pctile_252": "QQQ intraday range percentile",
    "qqq_gap_down_freq_10d": "QQQ 10d gap-down frequency",
    "qqq_realized_vol_20d": "QQQ 20d realized vol",
    "qqq_downside_semivol_20d": "QQQ 20d downside semivol",
    "qqq_breadth_advancers_pct": "Nasdaq-100 advancers %",
    "qqq_breadth_above_ma20_pct": "Nasdaq-100 above MA20 %",
    "qqq_breadth_above_ma50_pct": "Nasdaq-100 above MA50 %",
    "qqq_breadth_above_ma200_pct": "Nasdaq-100 above MA200 %",
    "qqq_breadth_new_low_60_pct": "Nasdaq-100 60d new-low %",
    "qqq_breadth_median_ret_5d": "Nasdaq-100 median 5d return",
    "qqq_breadth_ret20_dispersion": "Nasdaq-100 20d return dispersion",
    "vix_close": "VIX level",
    "vix_change_5d": "VIX 5d change",
    "vix_ma20_ratio": "VIX / VIX MA20",
    "vix3m_vix_spread": "VIX3M - VIX",
    "vvix_close": "VVIX level",
    "skew_close": "SKEW level",
    "qqq_spy_rel_20d": "QQQ vs SPY 20d relative strength",
    "qqew_qqq_rel_20d": "QQEW vs QQQ 20d relative strength",
    "rsp_spy_rel_20d": "RSP vs SPY 20d relative strength",
    "hyg_ief_rel_20d": "HYG vs IEF 20d relative strength",
    "hyg_lqd_rel_20d": "HYG vs LQD 20d relative strength",
    "smh_qqq_rel_20d": "SMH vs QQQ 20d relative strength",
    "xly_xlp_rel_20d": "XLY vs XLP 20d relative strength",
    "tlt_spy_rel_20d": "TLT vs SPY 20d relative strength",
    "gld_spy_rel_20d": "Gold vs SPY 20d relative strength",
    "btc_ret_20d": "BTC 20d return",
    "btc_realized_vol_20d": "BTC 20d realized vol",
}


@dataclass
class FoldModel:
    model: Any
    calibrator: "ProbabilityCalibrator"
    feature_columns: list[str]
    importance: dict[str, float]
    direction: dict[str, float]


class ProbabilityCalibrator:
    def __init__(self, method: str = "isotonic") -> None:
        self.method = method
        self.model: Any | None = None
        self.fallback_probability: float | None = None

    def fit(self, raw_probabilities: np.ndarray, labels: np.ndarray) -> "ProbabilityCalibrator":
        probs = clip_probabilities(raw_probabilities)
        y = labels.astype(int)
        if len(np.unique(y)) < 2:
            self.fallback_probability = float(np.mean(y)) if len(y) else None
            return self
        if self.method == "none":
            return self
        if self.method == "platt":
            x = logit(probs).reshape(-1, 1)
            self.model = LogisticRegression(solver="lbfgs", max_iter=1000)
            self.model.fit(x, y)
            return self
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(probs, y)
            return self
        raise ValueError(f"Unsupported calibration method: {self.method}")

    def predict(self, raw_probabilities: np.ndarray) -> np.ndarray:
        probs = clip_probabilities(raw_probabilities)
        if self.fallback_probability is not None:
            return np.full_like(probs, self.fallback_probability, dtype=float)
        if self.model is None or self.method == "none":
            return probs
        if self.method == "platt":
            return self.model.predict_proba(logit(probs).reshape(-1, 1))[:, 1]
        return np.asarray(self.model.predict(probs), dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a shadow-only LightGBM QQQ drawdown risk model with probability calibration."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_PUBLIC_DIR))
    parser.add_argument("--breadth-path", default=str(DEFAULT_BREADTH_PATH))
    parser.add_argument(
        "--macro-path",
        default="",
        help=f"Optional FRED macro feather. Use {DEFAULT_MACRO_PATH} to enable the macro feature layer.",
    )
    parser.add_argument(
        "--macro-groups",
        default="all",
        help="Comma-separated macro groups when --macro-path is set: rates,inflation,labor_growth,dollar_credit_commodity_vol,all.",
    )
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--horizon-days", type=int, default=10)
    parser.add_argument("--drawdown-threshold-pct", type=float, default=5.0)
    parser.add_argument("--min-train-days", type=int, default=504)
    parser.add_argument("--calibration-days", type=int, default=126)
    parser.add_argument("--test-window-days", type=int, default=21)
    parser.add_argument("--embargo-days", type=int, default=10)
    parser.add_argument("--calibration-method", choices=["isotonic", "platt", "none"], default="platt")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def file_name_for_symbol(symbol: str) -> str:
    return f"{symbol.upper().replace('/', '_').replace(':', '_')}-1d.feather"


def load_ohlcv(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_feather(path)
    df.columns = [str(column).strip().lower() for column in df.columns]
    required = {"date", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
    if "volume" not in df.columns:
        df["volume"] = np.nan
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    df["session_day"] = df["date"].dt.normalize()
    df = df.sort_values("date").drop_duplicates("session_day", keep="last")
    columns = {
        "open": f"{prefix}_open",
        "high": f"{prefix}_high",
        "low": f"{prefix}_low",
        "close": f"{prefix}_close",
        "volume": f"{prefix}_volume",
    }
    return df[["session_day", "open", "high", "low", "close", "volume"]].rename(columns=columns).reset_index(drop=True)


def safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series.astype(float).pct_change(periods=periods, fill_method=None)


def realized_volatility(series: pd.Series, window: int) -> pd.Series:
    return safe_pct_change(series, 1).rolling(window, min_periods=max(5, window // 2)).std() * math.sqrt(252.0)


def rolling_percentile_last(series: pd.Series, window: int) -> pd.Series:
    min_periods = max(20, window // 3)

    def percentile(values: np.ndarray) -> float:
        if len(values) == 0 or np.isnan(values[-1]):
            return np.nan
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return np.nan
        return float((valid <= values[-1]).mean())

    return series.rolling(window, min_periods=min_periods).apply(percentile, raw=True)


def add_feature(features: dict[str, pd.Series], name: str, values: pd.Series) -> None:
    features[name] = pd.to_numeric(values, errors="coerce").astype(float)


def merge_optional_symbol(frame: pd.DataFrame, data_dir: Path, symbol: str, prefix: str, missing: list[str]) -> pd.DataFrame:
    path = data_dir / file_name_for_symbol(symbol)
    if not path.exists():
        missing.append(f"missing_symbol:{symbol}")
        return frame
    loaded = load_ohlcv(path, prefix)
    return frame.merge(loaded, on="session_day", how="left")


def merge_breadth(frame: pd.DataFrame, breadth_path: Path, missing: list[str]) -> pd.DataFrame:
    if not breadth_path.exists():
        missing.append(f"missing_breadth:{breadth_path}")
        return frame
    breadth = pd.read_feather(breadth_path)
    breadth.columns = [str(column).strip().lower() for column in breadth.columns]
    if "date" not in breadth.columns:
        missing.append(f"invalid_breadth_no_date:{breadth_path}")
        return frame
    breadth["session_day"] = pd.to_datetime(breadth["date"], utc=True, errors="coerce").dt.normalize()
    breadth = breadth.dropna(subset=["session_day"]).sort_values("session_day").drop_duplicates("session_day", keep="last")
    keep = ["session_day"] + [column for column in breadth.columns if column.startswith("qqq_breadth_")]
    if len(keep) == 1:
        missing.append(f"invalid_breadth_no_features:{breadth_path}")
        return frame
    return frame.merge(breadth[keep], on="session_day", how="left")


def selected_macro_columns(columns: list[str], macro_groups: list[str]) -> list[str]:
    macro_columns = [column for column in columns if column.startswith("macro_")]
    if not macro_groups or "all" in macro_groups:
        return macro_columns
    prefixes: list[str] = []
    for group in macro_groups:
        prefixes.extend(MACRO_GROUP_PREFIXES.get(group, []))
    return [column for column in macro_columns if any(column.startswith(prefix) for prefix in prefixes)]


def merge_macro(frame: pd.DataFrame, macro_path: Path | None, missing: list[str], macro_groups: list[str]) -> pd.DataFrame:
    if macro_path is None:
        return frame
    if not macro_path.exists():
        missing.append(f"missing_macro:{macro_path}")
        return frame
    macro = pd.read_feather(macro_path)
    macro.columns = [str(column).strip().lower() for column in macro.columns]
    if "date" not in macro.columns:
        missing.append(f"invalid_macro_no_date:{macro_path}")
        return frame
    macro["session_day"] = pd.to_datetime(macro["date"], utc=True, errors="coerce").dt.normalize()
    macro = macro.dropna(subset=["session_day"]).sort_values("session_day").drop_duplicates("session_day", keep="last")
    keep = ["session_day"] + selected_macro_columns(list(macro.columns), macro_groups)
    if len(keep) == 1:
        missing.append(f"invalid_macro_no_features:{macro_path}")
        return frame
    return frame.merge(macro[keep], on="session_day", how="left")


def build_feature_frame(
    data_dir: Path,
    breadth_path: Path,
    macro_path: Path | None,
    macro_groups: list[str],
    horizon_days: int,
    drawdown_threshold_pct: float,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    missing: list[str] = []
    qqq_path = data_dir / "QQQ-1d.feather"
    if not qqq_path.exists():
        raise FileNotFoundError(f"Required QQQ data not found: {qqq_path}")

    frame = load_ohlcv(qqq_path, "qqq")
    for symbol, prefix in OPTIONAL_SYMBOLS.items():
        frame = merge_optional_symbol(frame, data_dir, symbol, prefix, missing)
    frame = merge_breadth(frame, breadth_path, missing)
    frame = merge_macro(frame, macro_path, missing, macro_groups)
    frame = frame.sort_values("session_day").reset_index(drop=True)

    features: dict[str, pd.Series] = {}
    close = frame["qqq_close"]
    high = frame["qqq_high"]
    low = frame["qqq_low"]
    open_ = frame["qqq_open"]
    ret_1d = safe_pct_change(close, 1)

    for window in [1, 5, 10, 20, 60]:
        add_feature(features, f"qqq_ret_{window}d", safe_pct_change(close, window))
    for window in [20, 50, 200]:
        ma = close.rolling(window, min_periods=max(10, window // 2)).mean()
        add_feature(features, f"qqq_ma{window}_dist", close / ma - 1.0)
    add_feature(features, "qqq_ma20_slope_10d", close.rolling(20).mean().pct_change(10, fill_method=None))
    add_feature(features, "qqq_ma50_slope_10d", close.rolling(50).mean().pct_change(10, fill_method=None))
    add_feature(features, "qqq_20d_high_dist", close / close.rolling(20, min_periods=10).max() - 1.0)
    add_feature(features, "qqq_60d_drawdown", close / close.rolling(60, min_periods=20).max() - 1.0)
    add_feature(features, "qqq_below_ma20_days", (close < close.rolling(20, min_periods=10).mean()).rolling(20).sum())
    range_pct = (high - low) / close.replace(0, np.nan)
    add_feature(features, "qqq_intraday_range_pct", range_pct)
    add_feature(features, "qqq_range_pctile_252", rolling_percentile_last(range_pct, 252))
    gap_down = open_ / close.shift(1) - 1.0
    add_feature(features, "qqq_gap_down_freq_10d", (gap_down < -0.01).rolling(10, min_periods=5).mean())
    add_feature(features, "qqq_realized_vol_10d", realized_volatility(close, 10))
    add_feature(features, "qqq_realized_vol_20d", realized_volatility(close, 20))
    downside = ret_1d.where(ret_1d < 0.0, 0.0)
    add_feature(features, "qqq_downside_semivol_20d", downside.rolling(20, min_periods=10).std() * math.sqrt(252.0))
    add_feature(features, "qqq_close_to_range_location", ((close - low) - (high - close)) / (high - low).replace(0, np.nan))
    add_feature(features, "qqq_ret_skew_20d", ret_1d.rolling(20, min_periods=15).skew())
    add_feature(features, "qqq_ret_kurt_20d", ret_1d.rolling(20, min_periods=15).kurt())
    add_feature(features, "qqq_volume_z_60d", (frame["qqq_volume"] - frame["qqq_volume"].rolling(60).mean()) / frame["qqq_volume"].rolling(60).std())

    if "spy_close" in frame:
        add_feature(features, "qqq_spy_rel_20d", safe_pct_change(close, 20) - safe_pct_change(frame["spy_close"], 20))
        add_feature(features, "spy_ma50_dist", frame["spy_close"] / frame["spy_close"].rolling(50, min_periods=25).mean() - 1.0)
        add_feature(features, "spy_ma200_dist", frame["spy_close"] / frame["spy_close"].rolling(200, min_periods=100).mean() - 1.0)
    if "ixic_close" in frame:
        add_feature(features, "ixic_ma50_dist", frame["ixic_close"] / frame["ixic_close"].rolling(50, min_periods=25).mean() - 1.0)
        add_feature(features, "ixic_ma200_dist", frame["ixic_close"] / frame["ixic_close"].rolling(200, min_periods=100).mean() - 1.0)
    if "vix_close" in frame:
        add_feature(features, "vix_close", frame["vix_close"])
        add_feature(features, "vix_change_5d", frame["vix_close"].diff(5))
        add_feature(features, "vix_ma20_ratio", frame["vix_close"] / frame["vix_close"].rolling(20, min_periods=10).mean())
    if "vix3m_close" in frame and "vix_close" in frame:
        add_feature(features, "vix3m_vix_spread", frame["vix3m_close"] - frame["vix_close"])
    if "vvix_close" in frame:
        add_feature(features, "vvix_close", frame["vvix_close"])
        add_feature(features, "vvix_vix_ratio", frame["vvix_close"] / frame["vix_close"] if "vix_close" in frame else np.nan)
    if "skew_close" in frame:
        add_feature(features, "skew_close", frame["skew_close"])

    relative_pairs = [
        ("qqew", "qqq", "qqew_qqq_rel_20d"),
        ("rsp", "spy", "rsp_spy_rel_20d"),
        ("smh", "qqq", "smh_qqq_rel_20d"),
        ("xly", "xlp", "xly_xlp_rel_20d"),
        ("hyg", "ief", "hyg_ief_rel_20d"),
        ("hyg", "lqd", "hyg_lqd_rel_20d"),
        ("lqd", "ief", "lqd_ief_rel_20d"),
        ("tlt", "spy", "tlt_spy_rel_20d"),
        ("gld", "spy", "gld_spy_rel_20d"),
        ("cper", "gld", "cper_gld_rel_20d"),
    ]
    for left, right, name in relative_pairs:
        left_col = f"{left}_close"
        right_col = f"{right}_close"
        if left_col in frame and right_col in frame:
            add_feature(features, name, safe_pct_change(frame[left_col], 20) - safe_pct_change(frame[right_col], 20))
    if "uso_close" in frame:
        add_feature(features, "uso_ret_5d", safe_pct_change(frame["uso_close"], 5))
    if "uup_close" in frame:
        add_feature(features, "uup_ret_20d", safe_pct_change(frame["uup_close"], 20))
    if "btc_close" in frame:
        add_feature(features, "btc_ret_5d", safe_pct_change(frame["btc_close"], 5))
        add_feature(features, "btc_ret_20d", safe_pct_change(frame["btc_close"], 20))
        add_feature(features, "btc_realized_vol_20d", realized_volatility(frame["btc_close"], 20))

    for column in frame.columns:
        if column.startswith("qqq_breadth_"):
            add_feature(features, column, frame[column])
        if column.startswith("macro_"):
            series = pd.to_numeric(frame[column], errors="coerce")
            add_feature(features, column, series)
            add_feature(features, f"{column}_chg_20d", series.diff(20))
            rolling_mean = series.rolling(252, min_periods=80).mean()
            rolling_std = series.rolling(252, min_periods=80).std()
            add_feature(features, f"{column}_z_252d", (series - rolling_mean) / rolling_std.replace(0, np.nan))

    feature_columns = sorted(features)
    feature_frame = pd.DataFrame({"date": frame["session_day"], **features})
    future_lows = pd.concat([low.shift(-offset) for offset in range(1, horizon_days + 1)], axis=1).min(axis=1)
    feature_frame["future_drawdown_pct"] = (future_lows / close - 1.0) * 100.0
    feature_frame["target_drawdown"] = (feature_frame["future_drawdown_pct"] <= -abs(drawdown_threshold_pct)).astype(float)
    feature_frame.loc[future_lows.isna(), "target_drawdown"] = np.nan
    feature_frame["qqq_close"] = close
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    return feature_frame, feature_columns, missing


def clip_probabilities(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)


def logit(values: np.ndarray) -> np.ndarray:
    probs = clip_probabilities(values)
    return np.log(probs / (1.0 - probs))


def fit_lightgbm(train: pd.DataFrame, feature_columns: list[str], random_state: int) -> Any:
    model = LGBMClassifier(
        n_estimators=240,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=24,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=random_state,
        verbose=-1,
    )
    model.fit(train[feature_columns], train["target_drawdown"].astype(int))
    return model


def feature_directions(train: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    output: dict[str, float] = {}
    positives = train[train["target_drawdown"] == 1]
    negatives = train[train["target_drawdown"] == 0]
    for column in feature_columns:
        pos = positives[column].median(skipna=True)
        neg = negatives[column].median(skipna=True)
        if pd.isna(pos) or pd.isna(neg) or pos == neg:
            output[column] = 0.0
        else:
            output[column] = 1.0 if pos > neg else -1.0
    return output


def feature_importance(model: Any, feature_columns: list[str]) -> dict[str, float]:
    values = getattr(model, "feature_importances_", np.zeros(len(feature_columns)))
    return {feature: float(value) for feature, value in zip(feature_columns, values)}


def fit_fold_model(
    train_window: pd.DataFrame,
    feature_columns: list[str],
    calibration_days: int,
    calibration_method: str,
    random_state: int,
) -> FoldModel | None:
    if len(train_window) <= calibration_days + 50:
        return None
    model_train = train_window.iloc[:-calibration_days].copy()
    calibration = train_window.iloc[-calibration_days:].copy()
    if model_train["target_drawdown"].nunique(dropna=True) < 2:
        return None
    model = fit_lightgbm(model_train, feature_columns, random_state)
    raw_calibration = model.predict_proba(calibration[feature_columns])[:, 1]
    calibrator = ProbabilityCalibrator(calibration_method).fit(raw_calibration, calibration["target_drawdown"].to_numpy(dtype=int))
    return FoldModel(
        model=model,
        calibrator=calibrator,
        feature_columns=feature_columns,
        importance=feature_importance(model, feature_columns),
        direction=feature_directions(model_train, feature_columns),
    )


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=int)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, y_score))


def safe_brier(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=int)
    if len(y) == 0:
        return None
    return float(brier_score_loss(y, y_score))


def run_walk_forward(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    min_train_days: int,
    calibration_days: int,
    test_window_days: int,
    embargo_days: int,
    calibration_method: str,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labelled = frame.dropna(subset=["target_drawdown"]).reset_index(drop=True)
    predictions: list[pd.DataFrame] = []
    skipped_folds = 0
    first_test_start = min_train_days + calibration_days + embargo_days
    for test_start in range(first_test_start, len(labelled), test_window_days):
        train_end = test_start - embargo_days
        test_end = min(test_start + test_window_days, len(labelled))
        train_window = labelled.iloc[:train_end].copy()
        test_window = labelled.iloc[test_start:test_end].copy()
        if len(train_window) < min_train_days + calibration_days or test_window.empty:
            skipped_folds += 1
            continue
        fold = fit_fold_model(
            train_window,
            feature_columns,
            calibration_days=calibration_days,
            calibration_method=calibration_method,
            random_state=random_state,
        )
        if fold is None:
            skipped_folds += 1
            continue
        raw = fold.model.predict_proba(test_window[feature_columns])[:, 1]
        calibrated = fold.calibrator.predict(raw)
        out = test_window[["date", "target_drawdown", "future_drawdown_pct", "qqq_close"]].copy()
        out["raw_prob_10d"] = raw
        out["model_prob_10d"] = calibrated
        out["fold_train_end"] = labelled.iloc[train_end - 1]["date"]
        predictions.append(out)

    if predictions:
        prediction_frame = pd.concat(predictions, ignore_index=True)
    else:
        prediction_frame = pd.DataFrame(
            columns=[
                "date",
                "target_drawdown",
                "future_drawdown_pct",
                "qqq_close",
                "raw_prob_10d",
                "model_prob_10d",
                "fold_train_end",
            ]
        )

    y = prediction_frame["target_drawdown"].to_numpy(dtype=int) if not prediction_frame.empty else np.array([])
    raw_scores = prediction_frame["raw_prob_10d"].to_numpy(dtype=float) if not prediction_frame.empty else np.array([])
    calibrated_scores = (
        prediction_frame["model_prob_10d"].to_numpy(dtype=float) if not prediction_frame.empty else np.array([])
    )
    metrics = {
        "rows": int(len(prediction_frame)),
        "event_rate_pct": round(float(np.mean(y) * 100.0), 2) if len(y) else None,
        "roc_auc_raw": safe_auc(y, raw_scores),
        "roc_auc_calibrated": safe_auc(y, calibrated_scores),
        "brier_raw": safe_brier(y, raw_scores),
        "brier_calibrated": safe_brier(y, calibrated_scores),
        "skipped_folds": skipped_folds,
    }
    return prediction_frame, metrics


def risk_bucket(probability: float) -> str:
    if probability >= 0.65:
        return "severe"
    if probability >= 0.50:
        return "high"
    if probability >= 0.35:
        return "elevated"
    return "low"


def suggested_exposure(probability: float) -> float:
    if probability >= 0.80:
        return 0.0
    if probability >= 0.65:
        return 0.25
    if probability >= 0.50:
        return 0.50
    if probability >= 0.35:
        return 0.75
    return 1.0


def top_drivers(row: pd.Series, train: pd.DataFrame, fold: FoldModel, limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    importances = fold.importance
    max_importance = max(importances.values()) if importances else 0.0
    if max_importance <= 0.0:
        max_importance = 1.0
    for feature in fold.feature_columns:
        value = row.get(feature)
        if pd.isna(value):
            continue
        mean = train[feature].mean(skipna=True)
        std = train[feature].std(skipna=True)
        if pd.isna(mean) or pd.isna(std) or float(std) <= 1e-12:
            continue
        z_score = (float(value) - float(mean)) / float(std)
        direction = float(fold.direction.get(feature, 0.0))
        importance = float(importances.get(feature, 0.0)) / max_importance
        risk_alignment = z_score * direction * importance
        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "value": round(float(value), 6),
                "z_score": round(float(z_score), 3),
                "risk_alignment": round(float(risk_alignment), 3),
                "direction": "higher_risk" if direction > 0 else "lower_is_risk" if direction < 0 else "unknown",
                "importance": round(float(importance), 4),
            }
        )
    positive = [item for item in rows if item["risk_alignment"] > 0]
    source = positive if positive else rows
    source.sort(key=lambda item: abs(float(item["risk_alignment"])), reverse=True)
    return source[:limit]


def train_latest_model(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    calibration_days: int,
    embargo_days: int,
    calibration_method: str,
    random_state: int,
) -> tuple[FoldModel | None, pd.DataFrame]:
    labelled = frame.dropna(subset=["target_drawdown"]).reset_index(drop=True)
    if labelled.empty:
        return None, labelled
    train_end = max(0, len(labelled) - max(embargo_days, 0))
    train_window = labelled.iloc[:train_end].copy() if train_end > calibration_days + 50 else labelled.copy()
    fold = fit_fold_model(
        train_window,
        feature_columns,
        calibration_days=min(calibration_days, max(20, len(train_window) // 4)),
        calibration_method=calibration_method,
        random_state=random_state,
    )
    return fold, train_window


def summarize_latest(
    frame: pd.DataFrame,
    feature_columns: list[str],
    fold: FoldModel | None,
    train_window: pd.DataFrame,
    missing_flags: list[str],
) -> dict[str, Any]:
    latest = frame.dropna(subset=feature_columns, how="all").iloc[-1]
    if fold is None:
        return {
            "date": str(pd.Timestamp(latest["date"]).date()),
            "risk_score_0_100": None,
            "model_prob_10d": None,
            "raw_prob_10d": None,
            "suggested_exposure": None,
            "risk_bucket": "unavailable",
            "top_drivers": [],
            "missing_data_flags": missing_flags + ["latest_model_unavailable"],
        }
    raw = float(fold.model.predict_proba(pd.DataFrame([latest[feature_columns]], columns=feature_columns))[:, 1][0])
    calibrated = float(fold.calibrator.predict(np.array([raw]))[0])
    return {
        "date": str(pd.Timestamp(latest["date"]).date()),
        "risk_score_0_100": round(calibrated * 100.0, 1),
        "model_prob_10d": round(calibrated, 4),
        "raw_prob_10d": round(raw, 4),
        "suggested_exposure": suggested_exposure(calibrated),
        "risk_bucket": risk_bucket(calibrated),
        "top_drivers": top_drivers(latest, train_window, fold),
        "missing_data_flags": missing_flags,
    }


def top_feature_importance(fold: FoldModel | None, limit: int = 25) -> list[dict[str, Any]]:
    if fold is None:
        return []
    rows = [
        {
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "importance": importance,
        }
        for feature, importance in fold.importance.items()
    ]
    rows.sort(key=lambda item: float(item["importance"]), reverse=True)
    return rows[:limit]


def write_outputs(predictions: pd.DataFrame, output_csv: Path, payload: dict[str, Any], output_json: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    csv = predictions.copy()
    if not csv.empty:
        csv["risk_score_0_100"] = (csv["model_prob_10d"] * 100.0).round(1)
        csv["risk_bucket"] = csv["model_prob_10d"].map(lambda value: risk_bucket(float(value)))
        csv["suggested_exposure"] = csv["model_prob_10d"].map(lambda value: suggested_exposure(float(value)))
    csv.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    breadth_path = Path(args.breadth_path)
    frame, feature_columns, missing_flags = build_feature_frame(
        data_dir=data_dir,
        breadth_path=breadth_path,
        macro_path=Path(args.macro_path) if args.macro_path else None,
        macro_groups=[item.strip() for item in str(args.macro_groups).split(",") if item.strip()],
        horizon_days=int(args.horizon_days),
        drawdown_threshold_pct=float(args.drawdown_threshold_pct),
    )
    predictions, metrics = run_walk_forward(
        frame,
        feature_columns,
        min_train_days=int(args.min_train_days),
        calibration_days=int(args.calibration_days),
        test_window_days=int(args.test_window_days),
        embargo_days=int(args.embargo_days),
        calibration_method=str(args.calibration_method),
        random_state=int(args.random_state),
    )
    latest_fold, latest_train = train_latest_model(
        frame,
        feature_columns,
        calibration_days=int(args.calibration_days),
        embargo_days=int(args.embargo_days),
        calibration_method=str(args.calibration_method),
        random_state=int(args.random_state),
    )
    latest = summarize_latest(frame, feature_columns, latest_fold, latest_train, missing_flags)
    payload = {
        "generated_at_utc": str(pd.Timestamp.utcnow()),
        "mode": "shadow_only",
        "label": {
            "name": "future_10d_max_drawdown",
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
            "feature_count": int(len(feature_columns)),
            "missing_data_flags": missing_flags,
        },
        "walk_forward_metrics": metrics,
        "latest_shadow_signal": latest,
        "top_feature_importance": top_feature_importance(latest_fold),
        "prediction_csv": str(Path(args.output_csv)),
    }
    write_outputs(predictions, Path(args.output_csv), payload, Path(args.output_json))
    print(Path(args.output_json))
    print(json.dumps({"walk_forward_metrics": metrics, "latest_shadow_signal": latest}, ensure_ascii=False))


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
