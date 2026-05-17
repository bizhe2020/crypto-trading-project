#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_shadow_utils import clean_for_json, standard_event_summary  # noqa: E402


DEFAULT_SCAN = ROOT / "var" / "reports" / "sota_rejected_smc_recall_scan_20260517.json"
DEFAULT_FROZEN = ROOT / "var" / "high_leverage_expansion" / "frozen_live_core_20260515.json"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "sota_recall_sweep_has_fvg_structure_normal_8x_audit_20260517.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the selected SOTA rejected-candidate recall bucket.")
    parser.add_argument("--scan", default=str(DEFAULT_SCAN))
    parser.add_argument("--frozen", default=str(DEFAULT_FROZEN))
    parser.add_argument("--condition", default="sweep_has_fvg")
    parser.add_argument("--dimensions-json", default='["regime_label", "sota_reject_stage"]')
    parser.add_argument("--values-json", default='{"regime_label": "normal", "sota_reject_stage": "structure_gate"}')
    parser.add_argument("--target-leverage", type=float, default=8.0)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def event_key(event: dict[str, Any]) -> str:
    return f"{event.get('event_type')}|{int(event.get('entry_idx', 0) or 0)}|{int(event.get('exit_idx', 0) or 0)}"


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    current_year = (summary.get("windows") or {}).get("current_year", {})
    return {
        "total_return_pct": round(float(summary.get("total_return_pct", 0.0) or 0.0), 2),
        "max_drawdown_pct": round(float(summary.get("max_drawdown_pct", 0.0) or 0.0), 2),
        "trades": int(summary.get("trades", 0) or 0),
        "wins": int(summary.get("wins", 0) or 0),
        "losses": int(summary.get("losses", 0) or 0),
        "win_rate_pct": round(float(summary.get("win_rate_pct", 0.0) or 0.0), 2),
        "avg_return_pct": round(float(summary.get("avg_return_pct", 0.0) or 0.0), 4),
        "profit_factor": round(float(summary.get("profit_factor", 0.0) or 0.0), 4),
        "event_type_counts": summary.get("event_type_counts", {}),
        "exit_counts": summary.get("exit_counts", {}),
        "current_year": {
            "total_return_pct": round(float(current_year.get("total_return_pct", 0.0) or 0.0), 2),
            "max_drawdown_pct": round(float(current_year.get("max_drawdown_pct", 0.0) or 0.0), 2),
            "trades": int(current_year.get("trades", 0) or 0),
            "win_rate_pct": round(float(current_year.get("win_rate_pct", 0.0) or 0.0), 2),
            "profit_factor": round(float(current_year.get("profit_factor", 0.0) or 0.0), 4),
        },
    }


def add_windows(summary: dict[str, Any], initial_capital: float, data_end: pd.Timestamp) -> dict[str, Any]:
    events = summary["events"]
    windows: dict[str, dict[str, Any]] = {}
    for name, start in {
        "current_year": pd.Timestamp(f"{data_end.year}-01-01", tz="UTC"),
        "last_60d": data_end - pd.Timedelta(days=60),
        "last_30d": data_end - pd.Timedelta(days=30),
    }.items():
        selected = [event for event in events if pd.Timestamp(event["entry_time"]) >= start]
        windows[name] = standard_event_summary(selected, initial_capital, "entry_idx")
        windows[name].pop("events", None)
    summary["windows"] = windows
    return summary


def summarize(events: list[dict[str, Any]], initial_capital: float, data_end: pd.Timestamp) -> dict[str, Any]:
    summary = standard_event_summary(sorted(events, key=lambda item: int(item.get("entry_idx", 0) or 0)), initial_capital, "entry_idx")
    return add_windows(summary, initial_capital, data_end)


def delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_return_pct": round(float(candidate["total_return_pct"]) - float(baseline["total_return_pct"]), 2),
        "max_drawdown_pct": round(float(candidate["max_drawdown_pct"]) - float(baseline["max_drawdown_pct"]), 2),
        "current_year_return_pct": round(
            float(candidate["windows"]["current_year"]["total_return_pct"])
            - float(baseline["windows"]["current_year"]["total_return_pct"]),
            2,
        ),
        "current_year_max_drawdown_pct": round(
            float(candidate["windows"]["current_year"]["max_drawdown_pct"])
            - float(baseline["windows"]["current_year"]["max_drawdown_pct"]),
            2,
        ),
    }


def yearly(events: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        groups.setdefault(str(pd.Timestamp(event["entry_time"]).year), []).append(event)
    out = {}
    for year, items in sorted(groups.items()):
        returns = [float(item.get("return_pct", 0.0) or 0.0) for item in items]
        wins = [value for value in returns if value > 0]
        losses = [value for value in returns if value < 0]
        out[year] = {
            "trades": len(items),
            "wins": len(wins),
            "losses": len(losses),
            "sum_return_pct": round(sum(returns), 4),
            "avg_return_pct": round(sum(returns) / len(returns), 4) if returns else 0.0,
            "best_return_pct": round(max(returns), 4) if returns else 0.0,
            "worst_return_pct": round(min(returns), 4) if returns else 0.0,
        }
    return out


def main() -> None:
    args = parse_args()
    scan = json.loads(Path(args.scan).read_text())
    frozen = json.loads(Path(args.frozen).read_text())
    dimensions = json.loads(args.dimensions_json)
    values = json.loads(args.values_json)
    selected = None
    for item in scan["subgroup_candidates"]:
        if (
            item["condition"] == args.condition
            and item["dimensions"] == dimensions
            and item["values"] == values
            and float(item["target_effective_leverage"]) == float(args.target_leverage)
        ):
            selected = item
            break
    if selected is None:
        raise SystemExit("selected recall candidate not found")

    frozen_events = list(frozen["live_shadow"]["events"])
    data_end = pd.Timestamp(frozen["metadata"]["data_end"])
    candidate_events = list(frozen_events)
    recall_events: list[dict[str, Any]] = []
    for sample in selected["sample_recalled_events"]:
        key = str(sample["event_key"])
        kind, entry_idx, exit_idx = key.split("|")
        event = dict(sample)
        event["event_type"] = kind
        event["entry_idx"] = int(entry_idx)
        event["exit_idx"] = int(exit_idx)
        event["direction"] = "BULL"
        event["return"] = float(event["return_pct"]) / 100.0
        event["source_effective_leverage"] = float(args.target_leverage)
        event["exit_reason"] = "recall_source"
        recall_events.append(event)
        candidate_events.append(event)

    baseline_summary = frozen["live_shadow"]
    candidate_summary = summarize(candidate_events, float(args.initial_capital), data_end)
    sorted_recall = sorted(recall_events, key=lambda event: float(event.get("return_pct", 0.0) or 0.0), reverse=True)

    remove_top = {}
    for count in [1, 2, 3]:
        remove_keys = {event_key(event) for event in sorted_recall[:count]}
        kept = [event for event in candidate_events if event_key(event) not in remove_keys]
        summary = summarize(kept, float(args.initial_capital), data_end)
        remove_top[f"remove_top_{count}"] = {
            "removed_events": [
                {
                    "event_key": event_key(event),
                    "entry_time": event.get("entry_time"),
                    "return_pct": event.get("return_pct"),
                }
                for event in sorted_recall[:count]
            ],
            "summary": compact_summary(summary),
            "delta_vs_frozen": delta(summary, baseline_summary),
        }

    per_event_revert = []
    for event in sorted_recall:
        kept = [item for item in candidate_events if event_key(item) != event_key(event)]
        summary = summarize(kept, float(args.initial_capital), data_end)
        per_event_revert.append(
            {
                "removed_event": {
                    "event_key": event_key(event),
                    "entry_time": event.get("entry_time"),
                    "return_pct": event.get("return_pct"),
                    "net_score": event.get("net_score"),
                    "bull_total": event.get("bull_total"),
                    "bear_total": event.get("bear_total"),
                    "recent_sweep_status": event.get("recent_sweep_status"),
                },
                "summary": compact_summary(summary),
                "delta_vs_frozen": delta(summary, baseline_summary),
            }
        )

    report = {
        "metadata": {
            "scan": str(Path(args.scan).resolve()),
            "frozen": str(Path(args.frozen).resolve()),
            "condition": args.condition,
            "dimensions": dimensions,
            "values": values,
            "target_effective_leverage": float(args.target_leverage),
        },
        "baseline": compact_summary(baseline_summary),
        "candidate": compact_summary(candidate_summary),
        "delta_vs_frozen": delta(candidate_summary, baseline_summary),
        "recall_events": sorted(recall_events, key=lambda item: int(item["entry_idx"])),
        "recall_yearly_stats": yearly(recall_events),
        "remove_top_recall_winners": remove_top,
        "per_event_revert": per_event_revert,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(clean_for_json(report), indent=2, ensure_ascii=False) + "\n")
    print(args.output)
    print("candidate", report["candidate"], report["delta_vs_frozen"])
    print("yearly", report["recall_yearly_stats"])
    print("remove_top_1", report["remove_top_recall_winners"]["remove_top_1"])


if __name__ == "__main__":
    main()
