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

from scripts.replay_sota_smc_live_shadow import replay_live_shadow  # noqa: E402
from scripts.live_shadow_utils import clean_for_json, event_stream_summary  # noqa: E402


DEFAULT_INPUT = ROOT / "var" / "high_leverage_expansion" / "confirmed_multiframe_scores_full_execsync_on_20260505.json"
DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "confirmed_score_gates_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan score gates on confirmed multiframe scored events.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--dd-budget-pct", type=float, default=2.0)
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


def delta_vs_baseline(summary: dict[str, Any], baseline_summary: dict[str, Any]) -> dict[str, float]:
    return {
        "total_return_pct": round(float(summary.get("total_return_pct", 0.0) or 0.0) - float(baseline_summary.get("total_return_pct", 0.0) or 0.0), 4),
        "max_drawdown_pct": round(float(summary.get("max_drawdown_pct", 0.0) or 0.0) - float(baseline_summary.get("max_drawdown_pct", 0.0) or 0.0), 4),
        "current_year_return_pct": round(float(summary.get("current_year_return_pct", 0.0) or 0.0) - float(baseline_summary.get("current_year_return_pct", 0.0) or 0.0), 4),
        "current_year_dd_pct": round(float(summary.get("current_year_dd_pct", 0.0) or 0.0) - float(baseline_summary.get("current_year_dd_pct", 0.0) or 0.0), 4),
    }


def score_scan(
    scored_events: list[dict[str, Any]],
    initial_capital: float,
    data_end: pd.Timestamp,
    dd_budget_pct: float,
) -> dict[str, Any]:
    baseline_events = [event for event in scored_events if str(event.get("event_type")) == "sota_long"]
    baseline = event_stream_summary(baseline_events, initial_capital, data_end)
    baseline_summary = compact_summary(baseline)
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
        live, _decisions = replay_live_shadow(filtered_base + smc_events, initial_capital, data_end, baseline)
        live_summary = compact_summary(live)
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
                    "smc_short": len(smc_events),
                },
                "live_shadow": live_summary,
                "delta_vs_baseline_shadow_sota": delta_vs_baseline(live_summary, baseline_summary),
                "within_dd_budget": float(live_summary["max_drawdown_pct"]) <= float(baseline_summary["max_drawdown_pct"]) + float(dd_budget_pct),
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
        live, _decisions = replay_live_shadow(baseline_events + filtered_smc, initial_capital, data_end, baseline)
        live_summary = compact_summary(live)
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
                    "smc_short": len(filtered_smc),
                },
                "live_shadow": live_summary,
                "delta_vs_baseline_shadow_sota": delta_vs_baseline(live_summary, baseline_summary),
                "within_dd_budget": float(live_summary["max_drawdown_pct"]) <= float(baseline_summary["max_drawdown_pct"]) + float(dd_budget_pct),
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
            live, _decisions = replay_live_shadow(filtered_base + filtered_smc, initial_capital, data_end, baseline)
            live_summary = compact_summary(live)
            combo_rules.append(
                {
                    "sota_rule": sota_rule["rule"],
                    "smc_rule": smc_rule["rule"],
                    "candidate_counts": {
                        "sota_long": len(filtered_base),
                        "smc_short": len(filtered_smc),
                    },
                    "live_shadow": live_summary,
                    "delta_vs_baseline_shadow_sota": delta_vs_baseline(live_summary, baseline_summary),
                    "within_dd_budget": float(live_summary["max_drawdown_pct"]) <= float(baseline_summary["max_drawdown_pct"]) + float(dd_budget_pct),
                }
            )

    def sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
        live = item["live_shadow"]
        return (
            float(live["total_return_pct"]),
            -float(live["max_drawdown_pct"]),
            float(live["current_year_return_pct"]),
        )

    def sort_key_2026(item: dict[str, Any]) -> tuple[float, float, float]:
        live = item["live_shadow"]
        return (
            float(live["current_year_return_pct"]),
            float(live["total_return_pct"]),
            -float(live["max_drawdown_pct"]),
        )

    sorted_sota = sorted(sota_rules, key=sort_key, reverse=True)
    sorted_smc = sorted(smc_rules, key=sort_key, reverse=True)
    sorted_combo = sorted(combo_rules, key=sort_key, reverse=True)
    sota_within_budget = [item for item in sorted_sota if bool(item.get("within_dd_budget"))]
    smc_within_budget = [item for item in sorted_smc if bool(item.get("within_dd_budget"))]
    combo_within_budget = [item for item in sorted_combo if bool(item.get("within_dd_budget"))]

    return {
        "baseline_shadow_sota": baseline_summary,
        "top_sota_rules": sorted_sota,
        "top_sota_rules_by_2026": sorted(sota_rules, key=sort_key_2026, reverse=True),
        "top_sota_rules_within_dd_budget": sota_within_budget,
        "top_smc_rules": sorted_smc,
        "top_smc_rules_by_2026": sorted(smc_rules, key=sort_key_2026, reverse=True),
        "top_smc_rules_within_dd_budget": smc_within_budget,
        "top_combo_rules": sorted_combo,
        "top_combo_rules_by_2026": sorted(combo_rules, key=sort_key_2026, reverse=True),
        "top_combo_rules_within_dd_budget": combo_within_budget,
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text())
    scored_events = payload.get("scored_events", [])
    if not isinstance(scored_events, list) or not scored_events:
        raise ValueError(f"No scored_events found in {args.input}")
    initial_capital = 1000.0
    data_end = pd.Timestamp(payload["metadata"]["data_end"])
    results = score_scan(scored_events, initial_capital, data_end, float(args.dd_budget_pct))
    report = {
        "metadata": {
            "input": str(Path(args.input).resolve()),
            "data_end": str(data_end),
            "dd_budget_pct": float(args.dd_budget_pct),
        },
        "baseline_shadow_sota": results["baseline_shadow_sota"],
        "top_sota_rules": results["top_sota_rules"][: args.top_n],
        "top_sota_rules_by_2026": results["top_sota_rules_by_2026"][: args.top_n],
        "top_sota_rules_within_dd_budget": results["top_sota_rules_within_dd_budget"][: args.top_n],
        "top_smc_rules": results["top_smc_rules"][: args.top_n],
        "top_smc_rules_by_2026": results["top_smc_rules_by_2026"][: args.top_n],
        "top_smc_rules_within_dd_budget": results["top_smc_rules_within_dd_budget"][: args.top_n],
        "top_combo_rules": results["top_combo_rules"][: args.top_n],
        "top_combo_rules_by_2026": results["top_combo_rules_by_2026"][: args.top_n],
        "top_combo_rules_within_dd_budget": results["top_combo_rules_within_dd_budget"][: args.top_n],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
