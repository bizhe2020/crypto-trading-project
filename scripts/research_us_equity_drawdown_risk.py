#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_public_etf_history import fetch_timeframe, load_existing, output_path_for  # noqa: E402


DEFAULT_SYMBOLS = ["QQQ", "SPY", "^IXIC", "^VIX", "HYG", "IEF", "TLT", "RSP", "QQEW", "BTC-USD"]
REQUIRED_SYMBOLS = {"QQQ", "SPY", "^VIX"}
DEFAULT_OUTPUT_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_REPORT_JSON = ROOT / "var" / "reports" / "us_equity_drawdown_risk_v1.json"
DEFAULT_DAILY_CSV = ROOT / "var" / "reports" / "us_equity_drawdown_risk_v1_daily.csv"


@dataclass(frozen=True)
class RiskComponent:
    name: str
    category: str
    series: pd.Series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research a US equity drawdown risk index for QQQ/SPY risk gating.")
    parser.add_argument("--symbol", action="append", default=None, help="Repeatable Yahoo symbol override.")
    parser.add_argument("--start", default="2015-01-01T00:00:00Z")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--refresh", action="store_true", help="Fetch/extend Yahoo history before computing the index.")
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--daily-csv", default=str(DEFAULT_DAILY_CSV))
    parser.add_argument("--min-date", default="2016-01-01", help="Drop early warmup rows before evaluation.")
    parser.add_argument("--model-horizon", type=int, default=10, choices=[5, 10, 20])
    parser.add_argument("--model-min-train-days", type=int, default=756)
    parser.add_argument("--model-step-days", type=int, default=21)
    return parser.parse_args()


def symbol_prefix(symbol: str) -> str:
    return (
        symbol.lower()
        .replace("^", "")
        .replace("-", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(".", "_")
    )


def session_date(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, utc=True, errors="coerce")
    return timestamps.dt.date.astype("string")


def load_symbol_history(
    *,
    session: requests.Session,
    symbol: str,
    output_dir: Path,
    start: str,
    end: str | None,
    refresh: bool,
    sleep_seconds: float,
    proxy: str | None,
) -> pd.DataFrame:
    path = output_path_for(output_dir, symbol, "1d")
    if refresh:
        fetch_timeframe(
            session=session,
            symbol=symbol,
            timeframe="1d",
            start=start,
            end=end,
            output_path=path,
            sleep_seconds=sleep_seconds,
            proxy=proxy,
        )
    df = load_existing(path)
    if df.empty:
        raise ValueError(f"No daily data for {symbol}; run with --refresh or check {path}")
    prefix = symbol_prefix(symbol)
    result = df.copy()
    result["session"] = session_date(result["date"])
    result = result.rename(
        columns={
            "open": f"{prefix}_open",
            "high": f"{prefix}_high",
            "low": f"{prefix}_low",
            "close": f"{prefix}_close",
            "volume": f"{prefix}_volume",
        }
    )
    keep = ["session", f"{prefix}_open", f"{prefix}_high", f"{prefix}_low", f"{prefix}_close", f"{prefix}_volume"]
    return result[keep].drop_duplicates(subset=["session"], keep="last")


def load_market_data(args: argparse.Namespace) -> pd.DataFrame:
    symbols = args.symbol or DEFAULT_SYMBOLS
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    if args.proxy:
        session.proxies.update({"http": args.proxy, "https": args.proxy})
    frames: list[pd.DataFrame] = []
    skipped: dict[str, str] = {}
    for symbol in symbols:
        try:
            frames.append(
                load_symbol_history(
                    session=session,
                    symbol=symbol,
                    output_dir=Path(args.output_dir),
                    start=args.start,
                    end=args.end,
                    refresh=bool(args.refresh),
                    sleep_seconds=float(args.sleep_seconds),
                    proxy=args.proxy,
                )
            )
        except Exception as exc:
            if symbol in REQUIRED_SYMBOLS:
                raise
            skipped[symbol] = str(exc)
            print(f"skip optional {symbol}: {exc}", flush=True)
    if not frames:
        raise RuntimeError("No market data loaded")
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="session", how="outer")
    merged = merged.sort_values("session").reset_index(drop=True)
    merged["date"] = pd.to_datetime(merged["session"], utc=True)
    merged.attrs["skipped_symbols"] = skipped
    return merged


def zscore(series: pd.Series, window: int = 252, min_periods: int = 126) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return pd.to_numeric((series - mean) / std.where(std != 0), errors="coerce")


def rolling_percentile(series: pd.Series, window: int = 756, min_periods: int = 252) -> pd.Series:
    def rank_last(values: pd.Series) -> float:
        last = values.iloc[-1]
        if pd.isna(last):
            return math.nan
        valid = values.dropna()
        if valid.empty:
            return math.nan
        return float((valid <= last).sum() / len(valid))

    return series.rolling(window, min_periods=min_periods).apply(rank_last, raw=False)


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    returns = close.pct_change(fill_method=None)
    return returns.rolling(window, min_periods=max(5, window // 2)).std(ddof=0) * math.sqrt(252)


def pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods, fill_method=None)


def ratio(data: pd.DataFrame, numerator: str, denominator: str) -> pd.Series | None:
    if numerator not in data.columns or denominator not in data.columns:
        return None
    return data[numerator] / data[denominator].where(data[denominator] != 0)


def add_component(components: list[RiskComponent], name: str, category: str, series: pd.Series | None) -> None:
    if series is None:
        return
    clean = pd.to_numeric(series, errors="coerce").replace([math.inf, -math.inf], float("nan"))
    if clean.notna().sum() < 260:
        return
    components.append(RiskComponent(name=name, category=category, series=clean))


def build_components(data: pd.DataFrame) -> tuple[pd.DataFrame, list[RiskComponent]]:
    qqq = data["qqq_close"]
    components: list[RiskComponent] = []

    add_component(components, "qqq_below_ma50", "trend", -(qqq / qqq.rolling(50).mean() - 1.0))
    add_component(components, "qqq_below_ma200", "trend", -(qqq / qqq.rolling(200).mean() - 1.0))
    add_component(components, "qqq_20d_return_weakness", "trend", -pct_change(qqq, 20))
    add_component(components, "qqq_60d_drawdown", "trend", -(qqq / qqq.rolling(60).max() - 1.0))
    add_component(components, "qqq_ma50_extension", "fragility", qqq / qqq.rolling(50).mean() - 1.0)
    add_component(components, "qqq_60d_extension", "fragility", qqq / qqq.rolling(60).min() - 1.0)
    if "spy_close" in data:
        spy = data["spy_close"]
        add_component(components, "spy_below_ma200", "trend", -(spy / spy.rolling(200).mean() - 1.0))
    if "ixic_close" in data:
        ixic = data["ixic_close"]
        add_component(components, "ixic_below_ma50", "trend", -(ixic / ixic.rolling(50).mean() - 1.0))

    if "vix_close" in data:
        vix = data["vix_close"]
        add_component(components, "vix_level", "volatility", vix)
        add_component(components, "vix_vs_ma20", "volatility", vix / vix.rolling(20).mean() - 1.0)
        add_component(components, "vix_5d_change", "volatility", pct_change(vix, 5))
        add_component(components, "vix_compression", "fragility", -(vix / vix.rolling(20).mean() - 1.0))
    add_component(components, "qqq_realized_vol_20d", "volatility", realized_vol(qqq, 20))
    add_component(components, "qqq_realized_vol_compression", "fragility", -realized_vol(qqq, 20))
    if {"qqq_high", "qqq_low"}.issubset(data.columns):
        add_component(components, "qqq_intraday_range_10d", "volatility", (data["qqq_high"] / data["qqq_low"] - 1.0).rolling(10).mean())

    qqew_qqq = ratio(data, "qqew_close", "qqq_close")
    rsp_spy = ratio(data, "rsp_close", "spy_close")
    add_component(components, "qqew_qqq_20d_breadth_weakness", "breadth_proxy", -pct_change(qqew_qqq, 20) if qqew_qqq is not None else None)
    add_component(components, "qqew_qqq_below_ma50", "breadth_proxy", -(qqew_qqq / qqew_qqq.rolling(50).mean() - 1.0) if qqew_qqq is not None else None)
    add_component(components, "rsp_spy_20d_breadth_weakness", "breadth_proxy", -pct_change(rsp_spy, 20) if rsp_spy is not None else None)
    if qqew_qqq is not None:
        add_component(
            components,
            "qqq_up_breadth_down_divergence",
            "fragility",
            pct_change(qqq, 20).clip(lower=0.0) - pct_change(qqew_qqq, 20),
        )

    hyg_ief = ratio(data, "hyg_close", "ief_close")
    add_component(components, "hyg_ief_20d_credit_weakness", "credit", -pct_change(hyg_ief, 20) if hyg_ief is not None else None)
    add_component(components, "hyg_ief_below_ma50", "credit", -(hyg_ief / hyg_ief.rolling(50).mean() - 1.0) if hyg_ief is not None else None)
    if hyg_ief is not None:
        add_component(
            components,
            "qqq_up_credit_down_divergence",
            "fragility",
            pct_change(qqq, 20).clip(lower=0.0) - pct_change(hyg_ief, 20),
        )
    tlt_spy = ratio(data, "tlt_close", "spy_close")
    add_component(components, "tlt_spy_20d_riskoff_bid", "credit", pct_change(tlt_spy, 20) if tlt_spy is not None else None)

    if "btc_usd_close" in data:
        btc = data["btc_usd_close"]
        add_component(components, "btc_10d_weakness", "cross_asset", -pct_change(btc, 10))
        add_component(components, "btc_realized_vol_20d", "cross_asset", realized_vol(btc, 20))

    feature_frame = pd.DataFrame(index=data.index)
    for component in components:
        feature_frame[component.name] = zscore(component.series)
    return feature_frame, components


def category_scores(feature_frame: pd.DataFrame, components: list[RiskComponent]) -> pd.DataFrame:
    categories = sorted({component.category for component in components})
    scores = pd.DataFrame(index=feature_frame.index)
    for category in categories:
        names = [component.name for component in components if component.category == category and component.name in feature_frame]
        if names:
            scores[category] = feature_frame[names].mean(axis=1)
    return scores


def composite_risk_score(scores: pd.DataFrame) -> pd.DataFrame:
    weights = {
        "trend": 0.20,
        "breadth_proxy": 0.20,
        "volatility": 0.20,
        "credit": 0.15,
        "cross_asset": 0.10,
        "fragility": 0.15,
    }
    weighted = pd.Series(0.0, index=scores.index)
    weight_sum = pd.Series(0.0, index=scores.index)
    for category, weight in weights.items():
        if category not in scores:
            continue
        valid = scores[category].notna()
        weighted = weighted.add(scores[category].fillna(0.0) * weight, fill_value=0.0)
        weight_sum = weight_sum.add(valid.astype(float) * weight, fill_value=0.0)
    composite = pd.to_numeric(weighted / weight_sum.where(weight_sum != 0), errors="coerce")
    percentile = rolling_percentile(composite)
    fallback = (50.0 + composite * 15.0).clip(0.0, 100.0)
    risk_score = (percentile * 100.0).combine_first(fallback)
    output = scores.copy()
    output["composite_z"] = composite
    output["risk_score"] = risk_score.clip(0.0, 100.0)
    return output


def add_forward_labels(daily: pd.DataFrame) -> pd.DataFrame:
    result = daily.copy()
    close = result["qqq_close"]
    horizons = {5: -0.03, 10: -0.05, 20: -0.08}
    for horizon, threshold in horizons.items():
        future_min = pd.concat([close.shift(-step) for step in range(1, horizon + 1)], axis=1).min(axis=1)
        future_dd = (future_min / close - 1.0).where(close.shift(-horizon).notna())
        result[f"future_dd_{horizon}d"] = future_dd
        result[f"label_dd_{horizon}d"] = (future_dd <= threshold).where(future_dd.notna())
    return result


def exposure_from_risk(score: Any) -> float:
    if pd.isna(score):
        return 1.0
    score_float = float(score)
    if score_float >= 85:
        return 0.0
    if score_float >= 70:
        return 0.25
    if score_float >= 55:
        return 0.50
    if score_float >= 35:
        return 0.75
    return 1.0


def exposure_from_probability(probability: Any) -> float:
    if pd.isna(probability):
        return 1.0
    probability_float = float(probability)
    if probability_float >= 0.35:
        return 0.0
    if probability_float >= 0.28:
        return 0.25
    if probability_float >= 0.22:
        return 0.50
    if probability_float >= 0.16:
        return 0.75
    return 1.0


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def performance_summary(returns: pd.Series) -> dict[str, float]:
    clean = returns.dropna()
    if clean.empty:
        return {"total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "volatility": 0.0}
    equity = (1.0 + clean).cumprod()
    years = max(len(clean) / 252.0, 1 / 252.0)
    total_return = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    volatility = float(clean.std(ddof=0) * math.sqrt(252))
    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown(equity),
        "volatility": volatility,
    }


def evaluate_risk_index(daily: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for horizon in [5, 10, 20]:
        label = daily[f"label_dd_{horizon}d"]
        eligible = daily[label.notna() & daily["risk_score"].notna()]
        if eligible.empty:
            continue
        base_rate = float(eligible[f"label_dd_{horizon}d"].mean())
        rows: dict[str, Any] = {"base_event_rate": base_rate, "sample_size": int(len(eligible))}
        for threshold in [55, 70, 85]:
            bucket = eligible[eligible["risk_score"] >= threshold]
            rows[f"risk_ge_{threshold}"] = {
                "count": int(len(bucket)),
                "event_rate": float(bucket[f"label_dd_{horizon}d"].mean()) if len(bucket) else None,
                "avg_future_dd": float(bucket[f"future_dd_{horizon}d"].mean()) if len(bucket) else None,
            }
        output[f"{horizon}d"] = rows
    returns = daily["qqq_close"].pct_change(fill_method=None).fillna(0.0)
    exposure = daily["risk_score"].shift(1).apply(exposure_from_risk)
    overlay_returns = returns * exposure
    output["overlay"] = {
        "buy_hold": performance_summary(returns),
        "risk_scaled": performance_summary(overlay_returns),
        "avg_exposure": float(exposure.mean()),
        "zero_exposure_days": int((exposure == 0.0).sum()),
        "reduced_exposure_days": int((exposure < 1.0).sum()),
    }
    return output


def model_feature_matrix(daily: pd.DataFrame, feature_frame: pd.DataFrame) -> pd.DataFrame:
    score_columns = [
        column
        for column in ["trend", "breadth_proxy", "volatility", "credit", "cross_asset", "fragility", "composite_z", "risk_score"]
        if column in daily.columns
    ]
    features = pd.concat([feature_frame, daily[score_columns]], axis=1)
    return features.replace([math.inf, -math.inf], float("nan"))


def walk_forward_probabilities(
    daily: pd.DataFrame,
    features: pd.DataFrame,
    *,
    horizon: int,
    min_train_days: int,
    step_days: int,
) -> pd.Series:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    label_column = f"label_dd_{horizon}d"
    labels = daily[label_column]
    probabilities = pd.Series(float("nan"), index=daily.index, dtype=float)
    start_index = max(int(min_train_days) + int(horizon), 300)
    step = max(int(step_days), 1)
    for start in range(start_index, len(daily), step):
        end = min(start + step, len(daily))
        train_end = start - horizon
        train_mask = (features.index < train_end) & labels.notna()
        y_train = labels.loc[train_mask].astype(int)
        if len(y_train) < 300 or y_train.nunique() < 2:
            continue
        x_train = features.loc[train_mask]
        x_test = features.iloc[start:end]
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=0.5,
                        max_iter=1000,
                        random_state=17,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        probabilities.iloc[start:end] = model.predict_proba(x_test)[:, 1]
    return probabilities.clip(0.0, 1.0)


def evaluate_probability_model(daily: pd.DataFrame, probabilities: pd.Series, *, horizon: int) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    label_column = f"label_dd_{horizon}d"
    frame = pd.DataFrame({"probability": probabilities, "label": daily[label_column], "future_dd": daily[f"future_dd_{horizon}d"]})
    frame = frame.dropna(subset=["probability", "label"])
    if frame.empty:
        return {"horizon": horizon, "sample_size": 0}
    y = frame["label"].astype(int)
    p = frame["probability"].astype(float)
    metrics: dict[str, Any] = {
        "horizon": horizon,
        "sample_size": int(len(frame)),
        "base_event_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }
    metrics["roc_auc"] = float(roc_auc_score(y, p)) if y.nunique() > 1 else None
    for threshold in [0.16, 0.22, 0.28, 0.35]:
        bucket = frame[frame["probability"] >= threshold]
        metrics[f"prob_ge_{threshold:.2f}"] = {
            "count": int(len(bucket)),
            "event_rate": float(bucket["label"].mean()) if len(bucket) else None,
            "avg_future_dd": float(bucket["future_dd"].mean()) if len(bucket) else None,
        }
    cutoff = float(frame["probability"].quantile(0.90))
    top = frame[frame["probability"] >= cutoff]
    metrics["top_decile"] = {
        "threshold": cutoff,
        "count": int(len(top)),
        "event_rate": float(top["label"].mean()) if len(top) else None,
        "avg_future_dd": float(top["future_dd"].mean()) if len(top) else None,
    }

    returns = daily["qqq_close"].pct_change(fill_method=None).fillna(0.0)
    exposure = probabilities.shift(1).apply(exposure_from_probability)
    overlay_returns = returns * exposure
    metrics["overlay"] = {
        "risk_model_scaled": performance_summary(overlay_returns),
        "avg_exposure": float(exposure.mean()),
        "zero_exposure_days": int((exposure == 0.0).sum()),
        "reduced_exposure_days": int((exposure < 1.0).sum()),
    }
    return metrics


def latest_snapshot(daily: pd.DataFrame, feature_frame: pd.DataFrame, components: list[RiskComponent]) -> dict[str, Any]:
    latest = daily.dropna(subset=["risk_score"]).iloc[-1]
    latest_features = feature_frame.loc[latest.name].dropna().sort_values(ascending=False)
    component_categories = {component.name: component.category for component in components}
    top_drivers = [
        {"name": name, "category": component_categories.get(name), "z": float(value)}
        for name, value in latest_features.head(8).items()
    ]
    return {
        "date": str(latest["session"]),
        "risk_score": float(latest["risk_score"]),
        "suggested_exposure": exposure_from_risk(latest["risk_score"]),
        "model_probability": float(latest["model_probability"]) if "model_probability" in latest and pd.notna(latest["model_probability"]) else None,
        "model_suggested_exposure": exposure_from_probability(latest["model_probability"]) if "model_probability" in latest else None,
        "composite_z": float(latest["composite_z"]) if pd.notna(latest["composite_z"]) else None,
        "category_scores": {
            key: float(latest[key])
            for key in ["trend", "breadth_proxy", "volatility", "credit", "cross_asset", "fragility"]
            if key in daily.columns and pd.notna(latest[key])
        },
        "top_drivers": top_drivers,
    }


def build_daily_index(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[RiskComponent]]:
    data = data.sort_values("date").copy()
    data = data[data["qqq_close"].notna()].reset_index(drop=True)
    value_columns = [column for column in data.columns if column not in {"session", "date"}]
    data[value_columns] = data[value_columns].ffill()
    feature_frame, components = build_components(data)
    scores = composite_risk_score(category_scores(feature_frame, components))
    daily = pd.concat([data[["session", "date", "qqq_close"]], scores], axis=1)
    daily = add_forward_labels(daily)
    return daily, feature_frame, components


def main() -> None:
    args = parse_args()
    started_at = time.time()
    data = load_market_data(args)
    daily, feature_frame, components = build_daily_index(data)
    if args.min_date:
        mask = daily["date"] >= pd.Timestamp(args.min_date, tz="UTC")
        daily = daily[mask].copy()
        feature_frame = feature_frame.loc[daily.index].copy()
        daily = daily.reset_index(drop=True)
        feature_frame = feature_frame.reset_index(drop=True)
    features = model_feature_matrix(daily, feature_frame)
    model_probability = walk_forward_probabilities(
        daily,
        features,
        horizon=int(args.model_horizon),
        min_train_days=int(args.model_min_train_days),
        step_days=int(args.model_step_days),
    )
    daily["model_probability"] = model_probability
    report = {
        "schema_version": 1,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "config": {
            "symbols": args.symbol or DEFAULT_SYMBOLS,
            "start": args.start,
            "end": args.end,
            "min_date": args.min_date,
            "refresh": bool(args.refresh),
            "model_horizon": int(args.model_horizon),
            "model_min_train_days": int(args.model_min_train_days),
            "model_step_days": int(args.model_step_days),
        },
        "skipped_symbols": data.attrs.get("skipped_symbols", {}),
        "components": [{"name": item.name, "category": item.category} for item in components],
        "latest": latest_snapshot(daily, feature_frame, components),
        "evaluation": evaluate_risk_index(daily),
        "model_evaluation": evaluate_probability_model(daily, model_probability, horizon=int(args.model_horizon)),
        "runtime_seconds": round(time.time() - started_at, 3),
    }

    report_json = Path(args.report_json)
    daily_csv = Path(args.daily_csv)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    daily_csv.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    daily.to_csv(daily_csv, index=False)
    print(json.dumps({"report_json": str(report_json), "daily_csv": str(daily_csv), "latest": report["latest"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
