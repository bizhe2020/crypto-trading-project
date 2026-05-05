#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_stable_smc_live_shadow import replay_live_shadow  # noqa: E402
from scripts.reproduce_reverse_short_overlay_candidates import clean_for_json  # noqa: E402
from scripts.research_reverse_short_from_failed_longs import event_stream_summary  # noqa: E402


DEFAULT_INPUT = ROOT / "var" / "high_leverage_expansion" / "confirmed_multiframe_scores_full_execsync_on_20260505.json"
DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "confirmed_score_gates_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan score gates on confirmed multiframe scored events.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def passes_gate(
    event: dict[str, Any],
    *,
    net_min: int | None = None,
    net_max: int | None = None,
    bull_min: int | None = None,
    bull_max: int | None = None,
    bear_min: int | None = None,
    bear_max: int | None = None,
    conflict_mode: str = "any",
) -> bool:
    net_score = int(event.get("net_score", 0) or 0)
    bull_total = int(event.get("bull_total", 0) or 0)
    bear_total = int(event.get("bear_total", 0) or 0)
    conflict = bool(event.get("conflict"))
    if net_min is not None and net_score < net_min:
        return False
    if net_max is not None and net_score > net_max:
        return False
    if bull_min is not None and bull_total < bull_min:
        return False
    if bull_max is not None and bull_total > bull_max:
        return False
    if bear_min is not None and bear_total < bear_min:
        return False
    if bear_max is not None and bear_total > bear_max:
        return False
    if conflict_mode == "conflict" and not conflict:
        return False
    if conflict_mode == "clean" and conflict:
        return False
    return True


def compact_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_return_pct": round(float(result.get("total_return_pct", 0.0) or 0.0), 2),
        "max_drawdown_pct": round(float(result.get("max_drawdown_pct", 0.0) or 0.0), 2),
        "trades": int(result.get("trades", 0) or 0),
        "wins": int(result.get("wins", 0) or 0),
        "losses": int(result.get("losses", 0) or 0),
        "win_rate_pct": round(float(result.get("win_rate_pct", 0.0) or 0.0), 2),
        "avg_return_pct": round(float(result.get("avg_return_pct", 0.0) or 0.0), 4),
        "profit_factor": round(float(result.get("profit_factor", 0.0) or 0.0), 4),
        "event_type_counts": result.get("event_type_counts", {}),
        "exit_counts": result.get("exit_counts", {}),
        "current_year_return_pct": round(float(result.get("windows", {}).get("current_year", {}).get("total_return_pct", 0.0) or 0.0), 2),
        "current_year_dd_pct": round(float(result.get("windows", {}).get("current_year", {}).get("max_drawdown_pct", 0.0) or 0.0), 2),
    }


def score_scan(
    scored_events: list[dict[str, Any]],
    initial_capital: float,
    data_end: pd.Timestamp,
) -> dict[str, Any]:
    baseline_events = [event for event in scored_events if str(event.get("event_type")) == "sota_long"]
    baseline = event_stream_summary(baseline_events, initial_capital, data_end)
    stable_events = [event for event in scored_events if str(event.get("event_type")) == "stable_reverse_short"]
    smc_events = [event for event in scored_events if str(event.get("event_type")) == "smc_short"]

    sota_rules: list[dict[str, Any]] = []
    for net_min, bull_min, bear_max, conflict_mode in product(range(2, 11), range(8, 16), range(4, 8), ("any", "conflict", "clean")):
        filtered_base = [
            event for event in baseline_events
            if passes_gate(
                event,
                net_min=net_min,
                bull_min=bull_min,
                bear_max=bear_max,
                conflict_mode=conflict_mode,
            )
        ]
        if len(filtered_base) < 12:
            continue
        live, _decisions = replay_live_shadow(filtered_base + stable_events + smc_events, initial_capital, data_end, baseline)
        sota_rules.append(
            {
                "rule": {
                    "net_min": net_min,
                    "bull_min": bull_min,
                    "bear_max": bear_max,
                    "conflict_mode": conflict_mode,
                },
                "candidate_counts": {
                    "sota_long": len(filtered_base),
                    "stable_reverse_short": len(stable_events),
                    "smc_short": len(smc_events),
                },
                "live_shadow": compact_summary(live),
            }
        )

    smc_rules: list[dict[str, Any]] = []
    for net_max, bear_min, bull_max, conflict_mode in product(range(-2, -11, -1), range(4, 12), range(4, 8), ("any", "conflict", "clean")):
        filtered_smc = [
            event for event in smc_events
            if passes_gate(
                event,
                net_max=net_max,
                bear_min=bear_min,
                bull_max=bull_max,
                conflict_mode=conflict_mode,
            )
        ]
        if len(filtered_smc) < 5:
            continue
        live, _decisions = replay_live_shadow(baseline_events + stable_events + filtered_smc, initial_capital, data_end, baseline)
        smc_rules.append(
            {
                "rule": {
                    "net_max": net_max,
                    "bear_min": bear_min,
                    "bull_max": bull_max,
                    "conflict_mode": conflict_mode,
                },
                "candidate_counts": {
                    "sota_long": len(baseline_events),
                    "stable_reverse_short": len(stable_events),
                    "smc_short": len(filtered_smc),
                },
                "live_shadow": compact_summary(live),
            }
        )

    combo_rules: list[dict[str, Any]] = []
    sota_candidates = sorted(
        sota_rules,
        key=lambda item: (
            float(item["live_shadow"]["total_return_pct"]),
            -float(item["live_shadow"]["max_drawdown_pct"]),
            float(item["live_shadow"]["current_year_return_pct"]),
        ),
        reverse=True,
    )[:10]
    smc_candidates = sorted(
        smc_rules,
        key=lambda item: (
            float(item["live_shadow"]["total_return_pct"]),
            -float(item["live_shadow"]["max_drawdown_pct"]),
            float(item["live_shadow"]["current_year_return_pct"]),
        ),
        reverse=True,
    )[:10]
    for sota_rule in sota_candidates:
        for smc_rule in smc_candidates:
            filtered_base = [
                event for event in baseline_events
                if passes_gate(event, **sota_rule["rule"])
            ]
            filtered_smc = [
                event for event in smc_events
                if passes_gate(event, **smc_rule["rule"])
            ]
            live, _decisions = replay_live_shadow(filtered_base + stable_events + filtered_smc, initial_capital, data_end, baseline)
            combo_rules.append(
                {
                    "sota_rule": sota_rule["rule"],
                    "smc_rule": smc_rule["rule"],
                    "candidate_counts": {
                        "sota_long": len(filtered_base),
                        "stable_reverse_short": len(stable_events),
                        "smc_short": len(filtered_smc),
                    },
                    "live_shadow": compact_summary(live),
                }
            )

    def sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
        live = item["live_shadow"]
        return (
            float(live["total_return_pct"]),
            -float(live["max_drawdown_pct"]),
            float(live["current_year_return_pct"]),
        )

    return {
        "baseline_shadow_sota": compact_summary(baseline),
        "top_sota_rules": sorted(sota_rules, key=sort_key, reverse=True),
        "top_smc_rules": sorted(smc_rules, key=sort_key, reverse=True),
        "top_combo_rules": sorted(combo_rules, key=sort_key, reverse=True),
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text())
    scored_events = payload.get("scored_events", [])
    if not isinstance(scored_events, list) or not scored_events:
        raise ValueError(f"No scored_events found in {args.input}")
    initial_capital = 1000.0
    data_end = pd.Timestamp(payload["metadata"]["data_end"])
    results = score_scan(scored_events, initial_capital, data_end)
    report = {
        "metadata": {
            "input": str(Path(args.input).resolve()),
            "data_end": str(data_end),
        },
        "baseline_shadow_sota": results["baseline_shadow_sota"],
        "top_sota_rules": results["top_sota_rules"][: args.top_n],
        "top_smc_rules": results["top_smc_rules"][: args.top_n],
        "top_combo_rules": results["top_combo_rules"][: args.top_n],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
