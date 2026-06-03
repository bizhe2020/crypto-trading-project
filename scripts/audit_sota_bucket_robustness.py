#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_shadow_utils import (  # noqa: E402
    add_standard_windows,
    clean_for_json,
    event_return_stats,
    standard_event_summary,
)


DEFAULT_INPUT = ROOT / "var" / "reports" / "sota_fvg_chained_live_replay_20260517.json"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "sota_bucket_robustness_audit_20260517.json"
INITIAL_CAPITAL = 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Accepted-stream robustness audit for SOTA long score bucket sizing."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sample-events", type=int, default=8)
    return parser.parse_args()


def utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def pct(value: float) -> float:
    return round(value * 100.0, 4)


def event_key(event: dict[str, Any]) -> str:
    return f"{event.get('event_type')}|{int(event.get('entry_idx', 0) or 0)}|{int(event.get('exit_idx', 0) or 0)}"


def bucket_decision(event: dict[str, Any]) -> dict[str, Any]:
    decision = event.get("long_score_bucket_sizing")
    return decision if isinstance(decision, dict) else {}


def applied_rule_names(event: dict[str, Any]) -> list[str]:
    decision = bucket_decision(event)
    rules = decision.get("applied_rules")
    names: list[str] = []
    if isinstance(rules, list):
        for item in rules:
            if not isinstance(item, dict):
                continue
            rule = item.get("rule")
            if isinstance(rule, dict) and rule.get("name"):
                names.append(str(rule["name"]))
    if names:
        return names
    rule = decision.get("rule")
    if isinstance(rule, dict) and rule.get("name"):
        return [str(rule["name"])]
    return []


def final_bucket_name(event: dict[str, Any]) -> str:
    names = applied_rule_names(event)
    return names[-1] if names else ""


def applied_combo_name(event: dict[str, Any]) -> str:
    names = applied_rule_names(event)
    return " + ".join(names) if names else ""


def is_boosted_sota_long(event: dict[str, Any]) -> bool:
    return str(event.get("event_type") or "") == "sota_long" and bool(bucket_decision(event).get("applied"))


def event_summary(events: list[dict[str, Any]], data_end: pd.Timestamp) -> dict[str, Any]:
    summary = standard_event_summary(events, INITIAL_CAPITAL, "entry_idx")
    return add_standard_windows(summary, INITIAL_CAPITAL, data_end, "entry_idx")


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    current_year = (summary.get("windows") or {}).get("current_year", {})
    return {
        "total_return_pct": round(float(summary.get("total_return_pct", 0.0) or 0.0), 2),
        "max_drawdown_pct": round(float(summary.get("max_drawdown_pct", 0.0) or 0.0), 2),
        "current_year_return_pct": round(float(current_year.get("total_return_pct", 0.0) or 0.0), 2),
        "current_year_dd_pct": round(float(current_year.get("max_drawdown_pct", 0.0) or 0.0), 2),
        "trades": int(summary.get("trades", 0) or 0),
        "win_rate_pct": round(float(summary.get("win_rate_pct", 0.0) or 0.0), 2),
        "profit_factor": round(float(summary.get("profit_factor", 0.0) or 0.0), 4),
    }


def compact_delta(summary: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    compacted = compact_summary(summary)
    current_compacted = compact_summary(current)
    return {
        **compacted,
        "delta_total_return_pct": round(
            compacted["total_return_pct"] - current_compacted["total_return_pct"], 2
        ),
        "delta_max_drawdown_pct": round(
            compacted["max_drawdown_pct"] - current_compacted["max_drawdown_pct"], 2
        ),
        "delta_current_year_return_pct": round(
            compacted["current_year_return_pct"] - current_compacted["current_year_return_pct"], 2
        ),
        "delta_trades": compacted["trades"] - current_compacted["trades"],
    }


def positive_top_share(events: list[dict[str, Any]]) -> float:
    positives = [float(event.get("return", 0.0) or 0.0) for event in events if float(event.get("return", 0.0) or 0.0) > 0.0]
    if not positives:
        return 0.0
    return round(max(positives) / max(sum(positives), 1e-12) * 100.0, 2)


def year_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_year: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_year[str(utc_timestamp(event.get("entry_time")).year)].append(event)
    return {
        year: event_return_stats(sorted(items, key=lambda item: int(item.get("entry_idx", 0) or 0)), INITIAL_CAPITAL)
        for year, items in sorted(by_year.items())
    }


def sample_events(events: list[dict[str, Any]], sample_count: int) -> list[dict[str, Any]]:
    ordered = sorted(events, key=lambda item: float(item.get("return", 0.0) or 0.0), reverse=True)
    return [
        {
            "event_key": event_key(event),
            "entry_time": event.get("entry_time"),
            "exit_time": event.get("exit_time"),
            "return_pct": pct(float(event.get("return", 0.0) or 0.0)),
            "source_effective_leverage": event.get("source_effective_leverage"),
            "pre_bucket_source_effective_leverage": event.get("pre_bucket_source_effective_leverage"),
            "final_bucket": final_bucket_name(event),
            "applied_combo": applied_combo_name(event),
            "net_score": event.get("net_score"),
            "bull_total": event.get("bull_total"),
            "bear_total": event.get("bear_total"),
            "conflict": event.get("conflict"),
            "regime_label": event.get("regime_label"),
            "recent_fvg_near_entry": event.get("feature_recent_fvg_near_entry"),
        }
        for event in ordered[:sample_count]
    ]


def remove_positions(events: list[dict[str, Any]], positions: set[int]) -> list[dict[str, Any]]:
    return [event for idx, event in enumerate(events) if idx not in positions]


def deboost_positions(events: list[dict[str, Any]], positions: set[int]) -> tuple[list[dict[str, Any]], int]:
    adjusted: list[dict[str, Any]] = []
    touched = 0
    for idx, event in enumerate(events):
        updated = deepcopy(event)
        if idx not in positions:
            adjusted.append(updated)
            continue

        current_leverage = float(event.get("source_effective_leverage", 0.0) or 0.0)
        pre_bucket_leverage = float(event.get("pre_bucket_source_effective_leverage", 0.0) or 0.0)
        if current_leverage <= 0.0 or pre_bucket_leverage <= 0.0:
            adjusted.append(updated)
            continue

        ratio = pre_bucket_leverage / current_leverage
        updated["return"] = float(event.get("return", 0.0) or 0.0) * ratio
        updated["return_pct"] = pct(float(updated["return"]))
        updated["source_effective_leverage"] = pre_bucket_leverage
        updated["deboosted_from_effective_leverage"] = current_leverage
        updated["deboosted_to_effective_leverage"] = pre_bucket_leverage
        if isinstance(updated.get("long_score_bucket_sizing"), dict):
            decision = dict(updated["long_score_bucket_sizing"])
            decision["accepted_stream_deboosted"] = True
            decision["deboost_ratio"] = round(ratio, 6)
            updated["long_score_bucket_sizing"] = decision
        touched += 1
        adjusted.append(updated)
    return adjusted, touched


def scenario_summary(
    events: list[dict[str, Any]],
    current: dict[str, Any],
    data_end: pd.Timestamp,
) -> dict[str, Any]:
    return compact_delta(event_summary(events, data_end), current)


def top_event_position(group_positions: set[int], events: list[dict[str, Any]]) -> int | None:
    if not group_positions:
        return None
    return max(group_positions, key=lambda idx: float(events[idx].get("return", 0.0) or 0.0))


def score_tuple(event: dict[str, Any]) -> str:
    return (
        f"n{int(event.get('net_score', 0) or 0)}_"
        f"b{int(event.get('bull_total', 0) or 0)}_"
        f"bear{int(event.get('bear_total', 0) or 0)}_"
        f"{'conflict' if bool(event.get('conflict')) else 'clean'}"
    )


def group_report(
    *,
    name: str,
    group_events: list[dict[str, Any]],
    group_positions: set[int],
    all_events: list[dict[str, Any]],
    current: dict[str, Any],
    data_end: pd.Timestamp,
    sample_count: int,
) -> dict[str, Any]:
    remove_group = scenario_summary(remove_positions(all_events, group_positions), current, data_end)
    deboosted_events, deboosted_count = deboost_positions(all_events, group_positions)
    deboost_group = scenario_summary(deboosted_events, current, data_end)

    top_position = top_event_position(group_positions, all_events)
    remove_top_events = all_events
    removed_top: list[dict[str, Any]] = []
    if top_position is not None:
        remove_top_events = remove_positions(all_events, {top_position})
        removed_top = [all_events[top_position]]

    years = sorted({utc_timestamp(event.get("entry_time")).year for event in group_events})
    current_year = data_end.year
    score_counts: dict[str, int] = defaultdict(int)
    for event in group_events:
        score_counts[score_tuple(event)] += 1

    flags: list[str] = []
    if len(group_events) < 5:
        flags.append("small_sample_lt5")
    if not any(utc_timestamp(event.get("entry_time")).year == current_year for event in group_events):
        flags.append("no_current_year_sample")
    if positive_top_share(group_events) >= 50.0:
        flags.append("top1_positive_share_ge50")
    if remove_group["delta_total_return_pct"] > 0:
        flags.append("removal_improves_total")
    if deboost_group["delta_total_return_pct"] > 0:
        flags.append("deboost_improves_total")
    if remove_group["delta_max_drawdown_pct"] < 0:
        flags.append("removal_reduces_dd")

    robustness = "watch"
    if "removal_improves_total" in flags or "deboost_improves_total" in flags:
        robustness = "harmful_or_overlevered"
    elif len(group_events) >= 5 and positive_top_share(group_events) < 50.0:
        robustness = "robust_candidate"
    if "no_current_year_sample" in flags or "small_sample_lt5" in flags:
        robustness = "fragile_sample" if robustness == "watch" else robustness

    return {
        "name": name,
        "trades": len(group_events),
        "years": years,
        "current_year_trades": sum(1 for event in group_events if utc_timestamp(event.get("entry_time")).year == current_year),
        "standalone": event_return_stats(
            sorted(group_events, key=lambda item: int(item.get("entry_idx", 0) or 0)), INITIAL_CAPITAL
        ),
        "yearly": year_stats(group_events),
        "top1_positive_share_pct": positive_top_share(group_events),
        "score_tuple_counts": dict(sorted(score_counts.items())),
        "remove_group": remove_group,
        "deboost_group": {
            **deboost_group,
            "deboosted_trades": deboosted_count,
        },
        "remove_top1_in_group": {
            **scenario_summary(remove_top_events, current, data_end),
            "removed": sample_events(removed_top, 1),
        },
        "sample_best_events": sample_events(group_events, sample_count),
        "flags": flags,
        "robustness": robustness,
    }


def build_report(payload: dict[str, Any], sample_count: int) -> dict[str, Any]:
    metadata = payload.get("metadata") or {}
    data_end = utc_timestamp(metadata.get("data_end") or "2026-05-16 00:00:00+00:00")
    events = list((payload.get("live_shadow") or {}).get("events") or [])
    current = event_summary(events, data_end)

    boosted_positions = {idx for idx, event in enumerate(events) if is_boosted_sota_long(event)}
    final_rule_groups: dict[str, set[int]] = defaultdict(set)
    combo_groups: dict[str, set[int]] = defaultdict(set)
    for idx in boosted_positions:
        event = events[idx]
        final_rule_groups[final_bucket_name(event) or "unknown"].add(idx)
        combo_groups[applied_combo_name(event) or "unknown"].add(idx)

    all_boosted_events = [events[idx] for idx in sorted(boosted_positions)]
    report = {
        "input_report": str(DEFAULT_INPUT),
        "scope": "accepted_stream_sensitivity",
        "notes": [
            "This audit mutates/removes already accepted events from the formal live-shadow stream.",
            "It does not re-admit candidates previously rejected by single-position arbitration.",
        ],
        "current_live_shadow": compact_summary(current),
        "report_live_shadow": compact_summary(payload.get("live_shadow") or {}),
        "reference_base_priority_sota_first": compact_summary(payload.get("reference_base_priority_sota_first") or {}),
        "boosted_overview": {
            "boosted_sota_long_trades": len(boosted_positions),
            "all_sota_long_trades": sum(1 for event in events if str(event.get("event_type") or "") == "sota_long"),
            "all_events": len(events),
            "standalone": event_return_stats(
                sorted(all_boosted_events, key=lambda item: int(item.get("entry_idx", 0) or 0)), INITIAL_CAPITAL
            ),
            "top1_positive_share_pct": positive_top_share(all_boosted_events),
        },
        "all_boosted_sensitivity": group_report(
            name="all_boosted_sota_long",
            group_events=all_boosted_events,
            group_positions=boosted_positions,
            all_events=events,
            current=current,
            data_end=data_end,
            sample_count=sample_count,
        ),
        "by_final_rule": [],
        "by_applied_combo": [],
    }

    for name, positions in sorted(final_rule_groups.items()):
        report["by_final_rule"].append(
            group_report(
                name=name,
                group_events=[events[idx] for idx in sorted(positions)],
                group_positions=positions,
                all_events=events,
                current=current,
                data_end=data_end,
                sample_count=sample_count,
            )
        )

    for name, positions in sorted(combo_groups.items()):
        report["by_applied_combo"].append(
            group_report(
                name=name,
                group_events=[events[idx] for idx in sorted(positions)],
                group_positions=positions,
                all_events=events,
                current=current,
                data_end=data_end,
                sample_count=sample_count,
            )
        )

    return report


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    report = build_report(payload, args.sample_events)
    report["input_report"] = str(input_path.resolve())
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    print("current", report["current_live_shadow"])
    print("all_boosted_remove", report["all_boosted_sensitivity"]["remove_group"])
    print("all_boosted_deboost", report["all_boosted_sensitivity"]["deboost_group"])
    for item in report["by_final_rule"]:
        print(
            item["name"],
            "trades",
            item["trades"],
            "robustness",
            item["robustness"],
            "remove_delta",
            item["remove_group"]["delta_total_return_pct"],
            "deboost_delta",
            item["deboost_group"]["delta_total_return_pct"],
            "top1_share",
            item["top1_positive_share_pct"],
        )


if __name__ == "__main__":
    main()
