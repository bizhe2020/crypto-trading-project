#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_shadow_utils import clean_for_json  # noqa: E402
from scripts.replay_sota_smc_live_shadow import replay_live_shadow  # noqa: E402


DEFAULT_INPUT = ROOT / "var" / "high_leverage_expansion" / "frozen_live_core_20260515.json"
DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "frozen_live_core_robustness_20260516.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether the frozen live core relies on a few top trades or one boosted bucket.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def current_year_return(summary: dict[str, Any]) -> float:
    return float((summary.get("windows") or {}).get("current_year", {}).get("total_return_pct", 0.0) or 0.0)


def current_year_dd(summary: dict[str, Any]) -> float:
    return float((summary.get("windows") or {}).get("current_year", {}).get("max_drawdown_pct", 0.0) or 0.0)


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_return_pct": round(float(summary.get("total_return_pct", 0.0) or 0.0), 2),
        "max_drawdown_pct": round(float(summary.get("max_drawdown_pct", 0.0) or 0.0), 2),
        "current_year_return_pct": round(current_year_return(summary), 2),
        "current_year_dd_pct": round(current_year_dd(summary), 2),
        "trades": int(summary.get("trades", len(summary.get("events", [])))),
        "profit_factor": round(float(summary.get("profit_factor", 0.0) or 0.0), 4),
    }


def event_key(event: dict[str, Any]) -> str:
    return f"{event.get('event_type')}|{event.get('entry_time')}|{event.get('exit_time')}|{float(event.get('return_pct', 0.0) or 0.0):.6f}"


def bucket_name(event: dict[str, Any]) -> str:
    decision = event.get("long_score_bucket_sizing")
    if not isinstance(decision, dict):
        return ""
    rule = decision.get("rule")
    if not isinstance(rule, dict):
        return ""
    return str(rule.get("name") or "")


def replay_from_events(events: list[dict[str, Any]], baseline: dict[str, Any], data_end: pd.Timestamp) -> dict[str, Any]:
    return replay_live_shadow(events, 1000.0, data_end, baseline)[0]


def remove_top_winners(events: list[dict[str, Any]], count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    winners = sorted(
        [event for event in events if float(event.get("return_pct", 0.0) or 0.0) > 0.0],
        key=lambda event: float(event.get("return_pct", 0.0) or 0.0),
        reverse=True,
    )
    removed = winners[:count]
    removed_keys = {event_key(event) for event in removed}
    kept = [event for event in events if event_key(event) not in removed_keys]
    return kept, removed


def demote_exact_bucket(events: list[dict[str, Any]], target_bucket: str, target_leverage: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    adjusted: list[dict[str, Any]] = []
    touched: list[dict[str, Any]] = []
    for event in events:
        updated = dict(event)
        if str(event.get("event_type") or "") != "sota_long":
            adjusted.append(updated)
            continue
        if bucket_name(event) != target_bucket:
            adjusted.append(updated)
            continue
        source_leverage = float(event.get("source_effective_leverage", 0.0) or 0.0)
        pre_bucket_leverage = float(event.get("pre_bucket_source_effective_leverage", source_leverage) or source_leverage)
        if source_leverage <= 0.0 or target_leverage <= 0.0:
            adjusted.append(updated)
            continue
        leverage_ratio = target_leverage / source_leverage
        original_return_pct = round(float(event.get("return_pct", 0.0) or 0.0), 4)
        updated["source_effective_leverage"] = target_leverage
        updated["return"] = float(event.get("return", 0.0) or 0.0) * leverage_ratio
        updated["return_pct"] = round(float(event.get("return_pct", 0.0) or 0.0) * leverage_ratio, 4)
        if isinstance(updated.get("long_score_bucket_sizing"), dict):
            decision = dict(updated["long_score_bucket_sizing"])
            decision["source_effective_leverage"] = target_leverage
            decision["demoted_from_frozen_target"] = source_leverage
            decision["demoted_to_pre_bucket"] = pre_bucket_leverage
            updated["long_score_bucket_sizing"] = decision
        updated["demoted_from_return_pct"] = original_return_pct
        touched.append(updated)
        adjusted.append(updated)
    return adjusted, touched


def build_report(payload: dict[str, Any]) -> dict[str, Any]:
    live = payload["live_shadow"]
    events = list(live.get("events", []))
    data_end = pd.Timestamp(payload.get("metadata", {}).get("data_end") or max(event["exit_time"] for event in events)).tz_convert("UTC")

    baseline = replay_from_events(events, live, data_end)
    top_share = {
        "top_1_share_of_positive_pct": round(
            max((float(event.get("return_pct", 0.0) or 0.0) for event in events if float(event.get("return_pct", 0.0) or 0.0) > 0.0), default=0.0)
            / max(sum(float(event.get("return_pct", 0.0) or 0.0) for event in events if float(event.get("return_pct", 0.0) or 0.0) > 0.0), 1e-9)
            * 100.0,
            2,
        ),
    }

    top_trade_removal: dict[str, Any] = {}
    for count in (1, 3, 5):
        kept, removed = remove_top_winners(events, count)
        replayed = replay_from_events(kept, live, data_end)
        top_trade_removal[f"remove_top_{count}"] = {
            "removed": [
                {
                    "event_type": event.get("event_type"),
                    "entry_time": event.get("entry_time"),
                    "return_pct": round(float(event.get("return_pct", 0.0) or 0.0), 4),
                    "bucket": bucket_name(event),
                    "source_effective_leverage": event.get("source_effective_leverage"),
                }
                for event in removed
            ],
            "summary": compact(replayed),
        }

    exact_bucket_name = "n3_b9_b6_conflict_target12"
    adjusted, touched = demote_exact_bucket(events, exact_bucket_name, 7.5)
    demoted_summary = replay_from_events(adjusted, live, data_end)

    bucket_counter = Counter(bucket_name(event) or "no_bucket" for event in events if str(event.get("event_type") or "") == "sota_long")
    bucket_return_counter: Counter[str] = Counter()
    for event in events:
        if str(event.get("event_type") or "") != "sota_long":
            continue
        bucket_return_counter[bucket_name(event) or "no_bucket"] += round(float(event.get("return_pct", 0.0) or 0.0), 4)

    return {
        "input": payload.get("metadata", {}).get("output_path"),
        "baseline": compact(baseline),
        "top_trade_positive_share": top_share,
        "top_trade_removal": top_trade_removal,
        "bucket_counts": dict(bucket_counter),
        "bucket_return_pct_sums": {key: round(value, 4) for key, value in bucket_return_counter.items()},
        "exact_bucket_demote_to_prebucket_like": {
            "bucket_name": exact_bucket_name,
            "demote_target_effective_leverage": 7.5,
            "affected_trades": len(touched),
            "summary": compact(demoted_summary),
            "samples": [
                {
                    "entry_time": event.get("entry_time"),
                    "frozen_return_pct": round(float(event.get("demoted_from_return_pct", 0.0) or 0.0), 4),
                    "demoted_return_pct": round(float(event.get("return_pct", 0.0) or 0.0), 4),
                    "source_effective_leverage": event.get("source_effective_leverage"),
                    "frozen_source_effective_leverage": event.get("long_score_bucket_sizing", {}).get("demoted_from_frozen_target"),
                }
                for event in touched[:10]
            ],
        },
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text())
    payload.setdefault("metadata", {})
    payload["metadata"]["output_path"] = str(input_path.resolve())
    report = build_report(payload)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    print("baseline", report["baseline"])
    for key, value in report["top_trade_removal"].items():
        print(key, value["summary"])
    print("exact_bucket_demote", report["exact_bucket_demote_to_prebucket_like"]["summary"])


if __name__ == "__main__":
    main()
