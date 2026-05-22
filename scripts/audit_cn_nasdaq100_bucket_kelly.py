#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "var" / "reports" / "cn_nasdaq100_baseline_robustness_audit.json"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "cn_nasdaq100_bucket_kelly_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate conservative bucket-level Kelly fractions for the CN Nasdaq-100 baseline.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    if b <= 0:
        return 0.0
    return max(0.0, (b * win_rate - q) / b)


def bucket_for_trade(trade: dict[str, Any]) -> str:
    vix = str(trade.get("vix_label") or "")
    rel = str(trade.get("rel_strength_label") or "")
    ixic = str(trade.get("ixic_trend_label") or "")
    if vix == "vix_low" and rel == "qqq_strong":
        return "vix_low_strong"
    if vix == "vix_low":
        return "vix_low_other"
    if vix == "vix_normal" and rel == "qqq_strong" and ixic == "ixic_up":
        return "vix_normal_strong"
    if rel == "qqq_strong":
        return "qqq_strong_other"
    return "other"


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "kelly": 0.0, "quarter_kelly": 0.0}
    wins = [v for v in values if v > 0]
    losses = [abs(v) for v in values if v < 0]
    win_rate = len(wins) / len(values)
    avg_win = mean(wins) / 100.0 if wins else 0.0
    avg_loss = mean(losses) / 100.0 if losses else 0.0
    k = kelly_fraction(win_rate, avg_win, avg_loss)
    return {
        "count": len(values),
        "win_rate": round(win_rate * 100.0, 2),
        "avg_win": round(mean(wins), 2) if wins else 0.0,
        "avg_loss": round(-mean(losses), 2) if losses else 0.0,
        "kelly": round(k, 4),
        "quarter_kelly": round(k * 0.25, 4),
        "half_kelly": round(k * 0.5, 4),
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    trade_summary = payload.get("trade_summary", {})
    trades = payload.get("trades", [])
    if not trades:
        raise ValueError("Missing trade data.")

    buckets: dict[str, list[float]] = {}
    for item in trades:
        bucket = bucket_for_trade(item)
        buckets.setdefault(bucket, []).append(float(item["trade_return_pct"]))

    overall_values = [float(item["trade_return_pct"]) for item in trades]
    output = {
        "overall": summarize(overall_values),
        "buckets": {name: summarize(values) for name, values in buckets.items()},
        "interpretation": {
            "recommended_floor": 0.25,
            "recommended_cap": 0.5,
            "note": "Use fractional Kelly as an upper bound for bucket sizing, not a target exposure.",
        },
        "trade_summary": trade_summary,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(output_path)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
