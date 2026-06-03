#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "var" / "reports" / "qqq_drawdown_lgb_shadow_predictions_macro_subfactor_core.csv"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_macro_subfactor_exposure_policy_robustness.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit QQQ macro subfactor exposure gates by subperiod and turnover cost."
    )
    parser.add_argument("--predictions-csv", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--score-columns", default="raw_prob_10d,model_prob_10d")
    parser.add_argument("--thresholds", default="0.15,0.20,0.35,0.50")
    parser.add_argument("--risk-exposures", default="0.0,0.25")
    parser.add_argument("--cost-bps-values", default="0,5,10,25")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "target_drawdown", "qqq_close", "raw_prob_10d", "model_prob_10d"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date", "qqq_close"]).sort_values("date").reset_index(drop=True)
    df["qqq_close"] = pd.to_numeric(df["qqq_close"], errors="coerce")
    df["target_drawdown"] = pd.to_numeric(df["target_drawdown"], errors="coerce").fillna(0).astype(int)
    df["next_ret"] = df["qqq_close"].shift(-1) / df["qqq_close"] - 1.0
    return df.dropna(subset=["next_ret"]).reset_index(drop=True)


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min() * 100.0)


def policy_exposure(frame: pd.DataFrame, policy: dict[str, Any]) -> pd.Series:
    kind = str(policy["kind"])
    if kind == "buy_hold":
        return pd.Series(1.0, index=frame.index)
    if kind == "current_suggested":
        if "suggested_exposure" not in frame.columns:
            raise ValueError("current_suggested policy requires suggested_exposure column")
        return pd.to_numeric(frame["suggested_exposure"], errors="coerce").fillna(1.0).astype(float)
    if kind == "threshold":
        score_column = str(policy["score_column"])
        threshold = float(policy["threshold"])
        risk_exposure = float(policy["risk_exposure"])
        return pd.Series(
            np.where(pd.to_numeric(frame[score_column], errors="coerce").astype(float) >= threshold, risk_exposure, 1.0),
            index=frame.index,
        )
    raise ValueError(f"Unsupported policy kind: {kind}")


def performance_metrics(frame: pd.DataFrame, exposure: pd.Series, cost_bps: float) -> dict[str, Any]:
    returns = pd.to_numeric(frame["next_ret"], errors="coerce").fillna(0.0).astype(float).reset_index(drop=True)
    weights = exposure.reset_index(drop=True).fillna(1.0).astype(float)
    turnover = weights.diff().abs().fillna(0.0)
    cost_rate = float(cost_bps) / 10000.0
    net_returns = returns * weights - turnover * cost_rate
    equity = (1.0 + net_returns).cumprod()
    rows = int(len(net_returns))
    total_return = float(equity.iloc[-1] - 1.0) if rows else 0.0
    annual_return = float(equity.iloc[-1] ** (252.0 / rows) - 1.0) if rows and equity.iloc[-1] > 0 else None
    annual_vol = float(net_returns.std(ddof=0) * math.sqrt(252.0)) if rows else 0.0
    sharpe = (
        float(net_returns.mean() / net_returns.std(ddof=0) * math.sqrt(252.0))
        if rows and net_returns.std(ddof=0) > 0.0
        else None
    )
    target = pd.to_numeric(frame["target_drawdown"], errors="coerce").fillna(0).astype(int).reset_index(drop=True)
    event_weights = weights[target == 1]
    non_event_weights = weights[target == 0]
    cost_drag = float((turnover * cost_rate).sum() * 100.0)
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
        "turnover_events": int((turnover > 1e-12).sum()),
        "total_turnover": round(float(turnover.sum()), 4),
        "cost_drag_pct": round(cost_drag, 4),
    }


def segment_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    dates = pd.to_datetime(frame["date"], utc=True)
    return {
        "full": pd.Series(True, index=frame.index),
        "2024_h2": (dates >= pd.Timestamp("2024-07-01", tz="UTC")) & (dates <= pd.Timestamp("2024-12-31", tz="UTC")),
        "2025": (dates >= pd.Timestamp("2025-01-01", tz="UTC")) & (dates <= pd.Timestamp("2025-12-31", tz="UTC")),
        "2025_stress_feb_apr": (dates >= pd.Timestamp("2025-02-01", tz="UTC"))
        & (dates <= pd.Timestamp("2025-04-30", tz="UTC")),
        "2025_recovery_may_dec": (dates >= pd.Timestamp("2025-05-01", tz="UTC"))
        & (dates <= pd.Timestamp("2025-12-31", tz="UTC")),
        "2026_ytd": dates >= pd.Timestamp("2026-01-01", tz="UTC"),
    }


def build_policies(score_columns: list[str], thresholds: list[float], risk_exposures: list[float]) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = [
        {"name": "buy_hold", "kind": "buy_hold"},
        {"name": "current_suggested_exposure", "kind": "current_suggested"},
    ]
    for score_column in score_columns:
        for threshold in thresholds:
            for risk_exposure in risk_exposures:
                score_label = score_column.replace("_prob_10d", "")
                policies.append(
                    {
                        "name": f"{score_label}_ge_{threshold:g}_to_{risk_exposure:g}",
                        "kind": "threshold",
                        "score_column": score_column,
                        "threshold": float(threshold),
                        "risk_exposure": float(risk_exposure),
                    }
                )
    return policies


def audit_policy(frame: pd.DataFrame, policy: dict[str, Any], cost_bps_values: list[float]) -> dict[str, Any]:
    exposure = policy_exposure(frame, policy)
    masks = segment_masks(frame)
    cost_results: dict[str, Any] = {}
    for cost_bps in cost_bps_values:
        segment_results: dict[str, Any] = {}
        for segment, mask in masks.items():
            sub_frame = frame.loc[mask].reset_index(drop=True)
            sub_exposure = exposure.loc[mask].reset_index(drop=True)
            if sub_frame.empty:
                continue
            segment_results[segment] = performance_metrics(sub_frame, sub_exposure, float(cost_bps))
        cost_results[f"{cost_bps:g}bps"] = segment_results
    return {
        "policy": policy,
        "cost_results": cost_results,
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
    frame = load_predictions(Path(args.predictions_csv))
    policies = build_policies(
        score_columns=parse_str_list(args.score_columns),
        thresholds=parse_float_list(args.thresholds),
        risk_exposures=parse_float_list(args.risk_exposures),
    )
    cost_bps_values = parse_float_list(args.cost_bps_values)
    results = [audit_policy(frame, policy, cost_bps_values) for policy in policies]
    payload = {
        "generated_at_utc": str(pd.Timestamp.utcnow()),
        "mode": "qqq_macro_subfactor_exposure_policy_robustness",
        "predictions_csv": str(Path(args.predictions_csv)),
        "return_model": "signal-date exposure applied to next trading day's close-to-close QQQ return; turnover cost on exposure changes.",
        "rows": int(len(frame)),
        "start": str(pd.Timestamp(frame["date"].min()).date()),
        "end": str(pd.Timestamp(frame["date"].max()).date()),
        "policies": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default))
    print(output)
    for result in results:
        full_10bps = result["cost_results"].get("10bps", {}).get("full")
        if full_10bps:
            print(result["policy"]["name"], json.dumps(full_10bps, ensure_ascii=False))


if __name__ == "__main__":
    main()
