#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_context" / "smc_htf_liquidity_targets_report_min1r_lookahead192.json"
DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_context" / "smc_runner_timeout_buckets.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze research-best timeout_original runner failures.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--runner-fraction", type=float, default=0.15)
    parser.add_argument("--min-target-rr", type=float, default=1.0)
    parser.add_argument("--lookahead-bars", type=int, default=192)
    parser.add_argument("--top-samples", type=int, default=20)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def categorize_rr(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 1.25:
        return "1_1p25r"
    if value < 1.5:
        return "1p25_1p5r"
    if value < 2.0:
        return "1p5_2r"
    return "gte_2r"


def categorize_score(score: int | None) -> str:
    if score is None:
        return "unknown"
    if score <= 1:
        return "0_1"
    if score == 2:
        return "2"
    if score == 3:
        return "3"
    return "4_plus"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "avg_return_pct": 0.0,
            "sum_return_pct": 0.0,
        }
    returns = [float(row.get("return", 0.0) or 0.0) for row in rows]
    wins = [value for value in returns if value > 0]
    return {
        "trades": len(rows),
        "wins": len(wins),
        "losses": len(rows) - len(wins),
        "win_rate_pct": round(len(wins) / len(rows) * 100.0, 2),
        "avg_return_pct": round(sum(returns) / len(rows) * 100.0, 4),
        "sum_return_pct": round(sum(returns) * 100.0, 4),
    }


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key, "unknown"))].append(row)
    return {name: summarize(bucket) for name, bucket in sorted(buckets.items())}


def main() -> None:
    args = parse_args()
    rows = json.loads(Path(args.input).read_text()).get("rows", [])
    eligible: list[dict[str, Any]] = []
    timeouts: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    for row in rows:
        target_rr = row.get("htf_selected_target_rr")
        if target_rr is None or float(target_rr) < float(args.min_target_rr):
            continue
        enriched = dict(row)
        enriched["target_rr_bucket"] = categorize_rr(None if target_rr is None else float(target_rr))
        enriched["smc_score_bucket"] = categorize_score(None if row.get("smc_score") is None else int(row.get("smc_score")))
        enriched["post_exit_hit_within_window"] = bool(
            row.get("htf_selected_target_post_exit_hit")
            and row.get("htf_selected_target_bars_after_exit") is not None
            and int(row.get("htf_selected_target_bars_after_exit")) <= int(args.lookahead_bars)
        )
        eligible.append(enriched)
        if enriched["post_exit_hit_within_window"]:
            hits.append(enriched)
        else:
            timeouts.append(enriched)

    report = {
        "metadata": {
            "input": str(Path(args.input).resolve()),
            "research_best_params": {
                "runner_fraction": args.runner_fraction,
                "min_target_rr": args.min_target_rr,
                "lookahead_bars": args.lookahead_bars,
                "timeout_mode": "original",
                "only_positive_original": False,
                "accounting_mode": "accounting",
            },
        },
        "summary": {
            "eligible": summarize(eligible),
            "post_exit_hits": summarize(hits),
            "timeouts": summarize(timeouts),
            "by_timeout_direction": grouped(timeouts, "direction"),
            "by_timeout_exit_reason": grouped(timeouts, "exit_reason"),
            "by_timeout_regime_label": grouped(timeouts, "regime_label"),
            "by_timeout_risk_mode": grouped(timeouts, "risk_mode"),
            "by_timeout_h4_pd_side": grouped(timeouts, "h4_pd_side"),
            "by_timeout_target_source": grouped(timeouts, "htf_selected_target_source"),
            "by_timeout_target_rr_bucket": grouped(timeouts, "target_rr_bucket"),
            "by_timeout_smc_score_bucket": grouped(timeouts, "smc_score_bucket"),
            "by_timeout_pressure_target_applied": grouped(timeouts, "pressure_target_applied"),
            "by_timeout_pressure_touch_lock_applied": grouped(timeouts, "pressure_touch_lock_applied"),
        },
        "samples": {
            "largest_timeout_losses": sorted(timeouts, key=lambda item: float(item.get("return", 0.0) or 0.0))[: args.top_samples],
            "largest_post_exit_hits": sorted(hits, key=lambda item: float(item.get("return", 0.0) or 0.0), reverse=True)[: args.top_samples],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    compact = {
        "eligible": report["summary"]["eligible"],
        "post_exit_hits": report["summary"]["post_exit_hits"],
        "timeouts": report["summary"]["timeouts"],
        "by_timeout_exit_reason": report["summary"]["by_timeout_exit_reason"],
        "by_timeout_regime_label": report["summary"]["by_timeout_regime_label"],
        "by_timeout_h4_pd_side": report["summary"]["by_timeout_h4_pd_side"],
        "by_timeout_target_source": report["summary"]["by_timeout_target_source"],
    }
    print(json.dumps(clean_for_json(compact), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
