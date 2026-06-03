#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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


DEFAULT_INPUT = ROOT / "var" / "high_leverage_expansion" / "frozen_live_core_20260515.json"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "sota_fvg_bear6_target20_candidate_audit_20260517.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize and audit a sizing-only SOTA long bucket candidate from an accepted live-shadow report. "
            "This never adds new entries; it only up-sizes already accepted SOTA long trades that match the rule."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--condition", default="fvg_near")
    parser.add_argument("--dimensions", default="bear_total")
    parser.add_argument("--values-json", default='{"bear_total": 6}')
    parser.add_argument("--target-leverage", type=float, default=20.0)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--sample-events", type=int, default=8)
    return parser.parse_args()


def utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def is_sota_long(event: dict[str, Any]) -> bool:
    return str(event.get("event_type") or "") == "sota_long" and str(event.get("direction") or "") == "BULL"


def parse_dimensions(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def condition_match(event: dict[str, Any], name: str) -> bool:
    if name == "fvg_near":
        return bool(event.get("feature_recent_fvg_near_entry"))
    if name == "mss_with_fvg":
        return str(event.get("feature_recent_sweep_status") or "") == "mss_with_fvg"
    if name == "fvg_near_and_mss_with_fvg":
        return bool(event.get("feature_recent_fvg_near_entry")) and str(event.get("feature_recent_sweep_status") or "") == "mss_with_fvg"
    if name == "sweep_has_fvg":
        return bool(event.get("feature_recent_sweep_has_fvg"))
    if name == "sweep_only":
        return str(event.get("feature_recent_sweep_status") or "") == "sweep_only"
    if name == "no_recent_sweep":
        return not bool(event.get("feature_recent_sweep"))
    raise ValueError(f"Unsupported condition: {name}")


def normalized_bucket_value(event: dict[str, Any], field: str) -> Any:
    value = event.get(field)
    if field in {"net_score", "bull_total", "bear_total"}:
        return int(value or 0)
    if field == "conflict":
        return int(bool(value))
    if field == "source_effective_leverage":
        return round(float(value or 0.0), 4)
    return value


def candidate_rule_matches(
    event: dict[str, Any],
    *,
    condition: str,
    dimensions: list[str],
    values: dict[str, Any],
) -> bool:
    if not is_sota_long(event):
        return False
    if not condition_match(event, condition):
        return False
    for field in dimensions:
        if normalized_bucket_value(event, field) != values.get(field):
            return False
    return True


def applied_rule_names(event: dict[str, Any]) -> list[str]:
    decision = event.get("long_score_bucket_sizing")
    if not isinstance(decision, dict):
        return []
    names: list[str] = []
    applied_rules = decision.get("applied_rules")
    if isinstance(applied_rules, list):
        for item in applied_rules:
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


def final_rule_name(event: dict[str, Any]) -> str:
    names = applied_rule_names(event)
    return names[-1] if names else ""


def applied_combo_name(event: dict[str, Any]) -> str:
    names = applied_rule_names(event)
    return " + ".join(names) if names else ""


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


def summarize_events(
    events: list[dict[str, Any]],
    *,
    initial_capital: float,
    data_end: pd.Timestamp,
) -> dict[str, Any]:
    summary = standard_event_summary(events, initial_capital, "entry_idx")
    return add_standard_windows(summary, initial_capital, data_end, "entry_idx")


def yearly_stats(events: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        groups[str(utc_timestamp(event.get("entry_time")).year)].append(event)
    return {
        year: event_return_stats(sorted(items, key=lambda item: int(item.get("entry_idx", 0) or 0)), initial_capital)
        for year, items in sorted(groups.items())
    }


def matched_event_snapshot(event: dict[str, Any]) -> dict[str, Any]:
    candidate = event.get("candidate_bucket_sizing") if isinstance(event.get("candidate_bucket_sizing"), dict) else {}
    return {
        "entry_time": event.get("entry_time"),
        "exit_time": event.get("exit_time"),
        "event_key": f"{event.get('event_type')}|{int(event.get('entry_idx', 0) or 0)}|{int(event.get('exit_idx', 0) or 0)}",
        "return_pct": round(float(event.get("return_pct", 0.0) or 0.0), 4),
        "source_effective_leverage": event.get("source_effective_leverage"),
        "current_final_rule": final_rule_name(event),
        "current_applied_combo": applied_combo_name(event),
        "net_score": event.get("net_score"),
        "bull_total": event.get("bull_total"),
        "bear_total": event.get("bear_total"),
        "regime_label": event.get("regime_label"),
        "conflict": bool(event.get("conflict")),
        "recent_fvg_near_entry": bool(event.get("feature_recent_fvg_near_entry")),
        "recent_sweep_status": event.get("feature_recent_sweep_status"),
        "candidate_target_effective_leverage": candidate.get("target_effective_leverage"),
        "candidate_incremental_return_pct": candidate.get("incremental_return_pct"),
        "candidate_already_at_or_above_target": bool(candidate.get("already_at_or_above_target")),
        "candidate_was_boosted": bool(candidate.get("applied")),
    }


def matched_rule_breakdown(events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "current_final_rule_counts": dict(sorted(Counter(final_rule_name(event) or "unbucketed" for event in events).items())),
        "current_applied_combo_counts": dict(sorted(Counter(applied_combo_name(event) or "unbucketed" for event in events).items())),
        "current_leverage_counts": dict(sorted(Counter(str(event.get("source_effective_leverage")) for event in events).items())),
        "regime_counts": dict(sorted(Counter(str(event.get("regime_label") or "") for event in events).items())),
    }


def apply_candidate(
    events: list[dict[str, Any]],
    *,
    condition: str,
    dimensions: list[str],
    values: dict[str, Any],
    target_leverage: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    matched_positions: list[int] = []
    boosted_positions: list[int] = []
    already_at_target = 0
    incremental_return_sum = 0.0

    for idx, event in enumerate(events):
        updated = deepcopy(event)
        if not candidate_rule_matches(updated, condition=condition, dimensions=dimensions, values=values):
            adjusted.append(updated)
            continue

        matched_positions.append(idx)
        source_leverage = float(updated.get("source_effective_leverage", 0.0) or 0.0)
        source_return = float(updated.get("return", 0.0) or 0.0)
        candidate_payload = {
            "matched": True,
            "condition": condition,
            "dimensions": dimensions,
            "values": values,
            "source_effective_leverage": round(source_leverage, 6),
            "target_effective_leverage": round(float(target_leverage), 6),
            "source_return": source_return,
            "source_return_pct": round(source_return * 100.0, 4),
            "current_final_rule": final_rule_name(updated),
            "current_applied_combo": applied_combo_name(updated),
        }
        if source_leverage <= 0.0 or float(target_leverage) <= source_leverage + 1e-9:
            candidate_payload["applied"] = False
            candidate_payload["already_at_or_above_target"] = True
            candidate_payload["incremental_return_pct"] = 0.0
            updated["candidate_bucket_sizing"] = candidate_payload
            already_at_target += 1
            adjusted.append(updated)
            continue

        boosted_positions.append(idx)
        scale = float(target_leverage) / source_leverage
        target_return = source_return * scale
        incremental_return_pct = (target_return - source_return) * 100.0
        incremental_return_sum += incremental_return_pct
        updated["return"] = target_return
        updated["return_pct"] = round(target_return * 100.0, 4)
        updated["source_effective_leverage"] = round(float(target_leverage), 6)
        candidate_payload["applied"] = True
        candidate_payload["already_at_or_above_target"] = False
        candidate_payload["leverage_scale"] = round(scale, 6)
        candidate_payload["target_return_pct"] = round(target_return * 100.0, 4)
        candidate_payload["incremental_return_pct"] = round(incremental_return_pct, 4)
        updated["candidate_bucket_sizing"] = candidate_payload
        adjusted.append(updated)

    diagnostics = {
        "condition": condition,
        "dimensions": dimensions,
        "values": values,
        "target_effective_leverage": float(target_leverage),
        "matched_positions": matched_positions,
        "boosted_positions": boosted_positions,
        "matched_trades": len(matched_positions),
        "boosted_trades": len(boosted_positions),
        "already_at_or_above_target_trades": already_at_target,
        "incremental_return_sum_pct": round(incremental_return_sum, 4),
    }
    return adjusted, diagnostics


def revert_positions_to_source(events: list[dict[str, Any]], positions: list[int]) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    targets = set(int(pos) for pos in positions)
    for idx, event in enumerate(events):
        updated = deepcopy(event)
        candidate = updated.get("candidate_bucket_sizing")
        if idx not in targets or not isinstance(candidate, dict) or not bool(candidate.get("applied")):
            adjusted.append(updated)
            continue
        updated["source_effective_leverage"] = float(candidate.get("source_effective_leverage", updated.get("source_effective_leverage", 0.0)) or 0.0)
        updated["return"] = float(candidate.get("source_return", updated.get("return", 0.0)) or 0.0)
        updated["return_pct"] = round(float(updated["return"]) * 100.0, 4)
        candidate["reverted_to_source"] = True
        updated["candidate_bucket_sizing"] = candidate
        adjusted.append(updated)
    return adjusted


def delta_vs_baseline(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_return_pct": round(float(candidate.get("total_return_pct", 0.0) or 0.0) - float(baseline.get("total_return_pct", 0.0) or 0.0), 2),
        "max_drawdown_pct": round(float(candidate.get("max_drawdown_pct", 0.0) or 0.0) - float(baseline.get("max_drawdown_pct", 0.0) or 0.0), 2),
        "current_year_return_pct": round(
            float((candidate.get("windows") or {}).get("current_year", {}).get("total_return_pct", 0.0) or 0.0)
            - float((baseline.get("windows") or {}).get("current_year", {}).get("total_return_pct", 0.0) or 0.0),
            2,
        ),
        "current_year_max_drawdown_pct": round(
            float((candidate.get("windows") or {}).get("current_year", {}).get("max_drawdown_pct", 0.0) or 0.0)
            - float((baseline.get("windows") or {}).get("current_year", {}).get("max_drawdown_pct", 0.0) or 0.0),
            2,
        ),
    }


def contribution_audit(
    candidate_events: list[dict[str, Any]],
    boosted_positions: list[int],
    *,
    initial_capital: float,
    data_end: pd.Timestamp,
    baseline_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pos in boosted_positions:
        event = candidate_events[pos]
        candidate = event.get("candidate_bucket_sizing")
        if not isinstance(candidate, dict):
            continue
        reverted = revert_positions_to_source(candidate_events, [pos])
        reverted_summary = summarize_events(reverted, initial_capital=initial_capital, data_end=data_end)
        rows.append(
            {
                "event": matched_event_snapshot(event),
                "candidate_incremental_return_pct": round(float(candidate.get("incremental_return_pct", 0.0) or 0.0), 4),
                "reverted_summary": compact_summary(reverted_summary),
                "reverted_delta_vs_baseline": delta_vs_baseline(reverted_summary, baseline_summary),
            }
        )
    rows.sort(key=lambda item: float(item.get("candidate_incremental_return_pct", 0.0) or 0.0), reverse=True)
    total_incremental = sum(float(item.get("candidate_incremental_return_pct", 0.0) or 0.0) for item in rows)
    for item in rows:
        incremental = float(item.get("candidate_incremental_return_pct", 0.0) or 0.0)
        item["incremental_share_pct"] = round(incremental / total_incremental * 100.0, 2) if total_incremental > 0 else 0.0
    return rows


def remove_position(candidate_events: list[dict[str, Any]], position: int) -> list[dict[str, Any]]:
    return [deepcopy(event) for idx, event in enumerate(candidate_events) if idx != int(position)]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    baseline_events = list((payload.get("live_shadow") or {}).get("events") or [])
    if not baseline_events:
        raise ValueError(f"No live_shadow events found in {input_path}")

    metadata = payload.get("metadata") or {}
    data_end = utc_timestamp(metadata.get("data_end") or baseline_events[-1].get("exit_time"))
    condition = str(args.condition)
    dimensions = parse_dimensions(args.dimensions)
    values = json.loads(str(args.values_json or "{}"))
    initial_capital = float(args.initial_capital)

    baseline_summary = summarize_events(baseline_events, initial_capital=initial_capital, data_end=data_end)
    candidate_events, diagnostics = apply_candidate(
        baseline_events,
        condition=condition,
        dimensions=dimensions,
        values=values,
        target_leverage=float(args.target_leverage),
    )
    candidate_summary = summarize_events(candidate_events, initial_capital=initial_capital, data_end=data_end)

    matched_events = [candidate_events[idx] for idx in diagnostics["matched_positions"]]
    matched_source_events = revert_positions_to_source(matched_events, list(range(len(matched_events))))
    boosted_events = [candidate_events[idx] for idx in diagnostics["boosted_positions"]]
    boosted_source_events = revert_positions_to_source(boosted_events, list(range(len(boosted_events))))
    reverted_all = revert_positions_to_source(candidate_events, diagnostics["boosted_positions"])
    reverted_all_summary = summarize_events(reverted_all, initial_capital=initial_capital, data_end=data_end)
    contributions = contribution_audit(
        candidate_events,
        diagnostics["boosted_positions"],
        initial_capital=initial_capital,
        data_end=data_end,
        baseline_summary=baseline_summary,
    )

    matched_by_candidate_return = sorted(
        [(idx, candidate_events[idx]) for idx in diagnostics["matched_positions"]],
        key=lambda item: float(item[1].get("return", 0.0) or 0.0),
        reverse=True,
    )
    remove_top_trade = None
    if matched_by_candidate_return:
        top_idx, top_event = matched_by_candidate_return[0]
        minus_top_events = remove_position(candidate_events, top_idx)
        minus_top_summary = summarize_events(minus_top_events, initial_capital=initial_capital, data_end=data_end)
        remove_top_trade = {
            "removed_event": matched_event_snapshot(top_event),
            "summary": compact_summary(minus_top_summary),
            "delta_vs_baseline": delta_vs_baseline(minus_top_summary, baseline_summary),
        }

    report = {
        "metadata": {
            "input_report": str(input_path.resolve()),
            "source_metadata": metadata,
            "candidate_rule": {
                "condition": condition,
                "dimensions": dimensions,
                "values": values,
                "target_effective_leverage": float(args.target_leverage),
            },
            "initial_capital": initial_capital,
        },
        "baseline_live_shadow": compact_summary(baseline_summary),
        "candidate_live_shadow": compact_summary(candidate_summary) | {
            "delta_vs_baseline": delta_vs_baseline(candidate_summary, baseline_summary),
        },
        "candidate_subset": {
            "diagnostics": diagnostics,
            "matched_rule_breakdown": matched_rule_breakdown(matched_events),
            "matched_source_stats": event_return_stats(matched_source_events, initial_capital),
            "matched_candidate_stats": event_return_stats(matched_events, initial_capital),
            "matched_yearly_source_stats": yearly_stats(matched_source_events, initial_capital),
            "matched_yearly_candidate_stats": yearly_stats(matched_events, initial_capital),
            "boosted_source_stats": event_return_stats(boosted_source_events, initial_capital),
            "boosted_candidate_stats": event_return_stats(boosted_events, initial_capital),
            "matched_events": [matched_event_snapshot(event) for event in matched_events[: int(args.sample_events)]],
        },
        "audit": {
            "revert_all_boosts": compact_summary(reverted_all_summary) | {
                "delta_vs_baseline": delta_vs_baseline(reverted_all_summary, baseline_summary),
            },
            "remove_top_matched_trade": remove_top_trade,
            "per_boost_contribution": contributions,
            "top_boost": contributions[0] if contributions else None,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(output_path)
    print("baseline", compact_summary(baseline_summary))
    print("candidate", compact_summary(candidate_summary), "delta", delta_vs_baseline(candidate_summary, baseline_summary))
    print("diagnostics", diagnostics)
    if contributions:
        top = contributions[0]
        print(
            "top_boost",
            top["event"]["entry_time"],
            top["candidate_incremental_return_pct"],
            top["incremental_share_pct"],
            top["reverted_delta_vs_baseline"],
        )


if __name__ == "__main__":
    main()
