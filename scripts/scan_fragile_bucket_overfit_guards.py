#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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
from scripts.score_bucket_sizing_utils import apply_score_bucket_leverage  # noqa: E402


DEFAULT_INPUT = ROOT / "var" / "reports" / "sota_fvg_chained_live_replay_20260517.json"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "fragile_bucket_overfit_guard_scan_20260517.json"
INITIAL_CAPITAL = 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Accepted-stream scan for overfit guards on fragile long score buckets. "
            "This rescales existing accepted events from pre-bucket leverage and does not edit live config."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    current_year = (summary.get("windows") or {}).get("current_year", {})
    return {
        "total_return_pct": round(float(summary.get("total_return_pct", 0.0) or 0.0), 2),
        "max_drawdown_pct": round(float(summary.get("max_drawdown_pct", 0.0) or 0.0), 2),
        "current_year_return_pct": round(float(current_year.get("total_return_pct", 0.0) or 0.0), 2),
        "current_year_dd_pct": round(float(current_year.get("max_drawdown_pct", 0.0) or 0.0), 2),
        "trades": int(summary.get("trades", 0) or 0),
        "wins": int(summary.get("wins", 0) or 0),
        "losses": int(summary.get("losses", 0) or 0),
        "win_rate_pct": round(float(summary.get("win_rate_pct", 0.0) or 0.0), 2),
        "profit_factor": round(float(summary.get("profit_factor", 0.0) or 0.0), 4),
    }


def event_summary(events: list[dict[str, Any]], data_end: pd.Timestamp) -> dict[str, Any]:
    summary = standard_event_summary(events, INITIAL_CAPITAL, "entry_idx")
    return add_standard_windows(summary, INITIAL_CAPITAL, data_end, "entry_idx")


def event_key(event: dict[str, Any]) -> str:
    return f"{event.get('event_type')}|{int(event.get('entry_idx', 0) or 0)}|{int(event.get('exit_idx', 0) or 0)}"


def applied_rule_names(event: dict[str, Any]) -> list[str]:
    decision = event.get("long_score_bucket_sizing")
    if not isinstance(decision, dict):
        return []
    names: list[str] = []
    applied_rules = decision.get("applied_rules")
    if isinstance(applied_rules, list):
        for applied in applied_rules:
            if not isinstance(applied, dict):
                continue
            rule = applied.get("rule")
            if isinstance(rule, dict) and rule.get("name"):
                names.append(str(rule["name"]))
    if names:
        return names
    rule = decision.get("rule")
    if isinstance(rule, dict) and rule.get("name"):
        return [str(rule["name"])]
    return []


def score_payload(event: dict[str, Any]) -> dict[str, Any]:
    score: dict[str, Any] = {
        "net_score": int(event.get("net_score", 0) or 0),
        "bull_total": int(event.get("bull_total", 0) or 0),
        "bear_total": int(event.get("bear_total", 0) or 0),
        "conflict": bool(event.get("conflict")),
        "risk_mode": event.get("risk_mode"),
        "regime_label": event.get("regime_label"),
    }
    for key, value in event.items():
        if str(key).startswith("feature_"):
            score[str(key)] = value
    return score


def base_leverage(event: dict[str, Any]) -> float:
    pre_bucket = event.get("pre_bucket_source_effective_leverage")
    if pre_bucket is not None:
        return float(pre_bucket or 0.0)
    decision = event.get("long_score_bucket_sizing")
    if isinstance(decision, dict) and decision.get("source_effective_leverage") is not None:
        return float(decision.get("source_effective_leverage") or 0.0)
    return float(event.get("source_effective_leverage", 0.0) or 0.0)


def base_return(event: dict[str, Any]) -> float:
    current_leverage = float(event.get("source_effective_leverage", 0.0) or 0.0)
    pre_leverage = base_leverage(event)
    current_return = float(event.get("return", 0.0) or 0.0)
    if current_leverage <= 0.0 or pre_leverage <= 0.0:
        return current_return
    return current_return * pre_leverage / current_leverage


def rescale_events(events: list[dict[str, Any]], rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    for event in events:
        updated = deepcopy(event)
        if str(event.get("event_type") or "") != "sota_long":
            adjusted.append(updated)
            continue
        pre_leverage = base_leverage(event)
        if pre_leverage <= 0.0:
            adjusted.append(updated)
            continue
        selected_leverage, decision = apply_score_bucket_leverage(
            effective_leverage=pre_leverage,
            score=score_payload(event),
            enabled=True,
            rules=rules,
        )
        updated["pre_bucket_source_effective_leverage"] = pre_leverage
        updated["source_effective_leverage"] = selected_leverage
        updated["return"] = base_return(event) * (selected_leverage / pre_leverage)
        updated["return_pct"] = round(float(updated["return"]) * 100.0, 4)
        updated["long_score_bucket_sizing"] = decision
        adjusted.append(updated)
    return adjusted


def remove_rule(rules: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [deepcopy(rule) for rule in rules if str(rule.get("name") or "") != name]


def update_rule(rules: list[dict[str, Any]], name: str, updates: dict[str, Any]) -> list[dict[str, Any]]:
    updated_rules: list[dict[str, Any]] = []
    for rule in rules:
        copied = deepcopy(rule)
        if str(copied.get("name") or "") == name:
            for key, value in updates.items():
                if value is None:
                    copied.pop(key, None)
                else:
                    copied[key] = value
        updated_rules.append(copied)
    return updated_rules


def variant_rules(current_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []

    def add(name: str, rules: list[dict[str, Any]], note: str) -> None:
        variants.append({"name": name, "rules": rules, "note": note})

    add("current_rules_rescaled", deepcopy(current_rules), "Control: current frozen bucket rules.")

    for cap in (10.0, 12.0, 15.0):
        add(
            f"bear6_cap{int(cap)}",
            update_rule(current_rules, "bear_total_6_20x_boost", {"max_effective_leverage": cap}),
            f"Keep generic bear=6 boost but cap effective leverage at {cap}.",
        )

    add(
        "bear6_high_growth_only",
        update_rule(current_rules, "bear_total_6_20x_boost", {"regime_labels": ["high_growth"]}),
        "Only boost bear=6 when regime_label is high_growth.",
    )
    add(
        "bear6_high_growth_cap15",
        update_rule(
            current_rules,
            "bear_total_6_20x_boost",
            {"regime_labels": ["high_growth"], "max_effective_leverage": 15.0},
        ),
        "High-growth-only bear=6 boost with cap 15.",
    )
    add(
        "bear6_fvg_only",
        update_rule(
            current_rules,
            "bear_total_6_20x_boost",
            {"required_true_features": ["recent_fvg_near_entry"]},
        ),
        "Only boost bear=6 when recent_fvg_near_entry is true.",
    )
    add(
        "bear6_removed",
        remove_rule(current_rules, "bear_total_6_20x_boost"),
        "Remove generic bear=6 boost; later FVG bear=6 rule can still apply.",
    )

    for cap in (10.0, 12.0, 15.0):
        add(
            f"nbb_cap{int(cap)}",
            update_rule(current_rules, "nbb_6_11_5_conflict_2p5_cap20", {"max_effective_leverage": cap}),
            f"Keep NBB multiplier but cap effective leverage at {cap}.",
        )
    add(
        "nbb_fvg_only",
        update_rule(
            current_rules,
            "nbb_6_11_5_conflict_2p5_cap20",
            {"required_true_features": ["recent_fvg_near_entry"]},
        ),
        "Only boost NBB when recent_fvg_near_entry is true.",
    )
    add(
        "nbb_fvg_cap12",
        update_rule(
            current_rules,
            "nbb_6_11_5_conflict_2p5_cap20",
            {"required_true_features": ["recent_fvg_near_entry"], "max_effective_leverage": 12.0},
        ),
        "FVG-only NBB with cap 12.",
    )
    add(
        "nbb_removed",
        remove_rule(current_rules, "nbb_6_11_5_conflict_2p5_cap20"),
        "Remove NBB boost.",
    )

    add(
        "nbb_fvg_bear6_cap12",
        update_rule(
            update_rule(
                current_rules,
                "nbb_6_11_5_conflict_2p5_cap20",
                {"required_true_features": ["recent_fvg_near_entry"]},
            ),
            "bear_total_6_20x_boost",
            {"max_effective_leverage": 12.0},
        ),
        "NBB requires FVG; bear=6 capped at 12.",
    )
    add(
        "nbb_fvg_bear6_hg_only",
        update_rule(
            update_rule(
                current_rules,
                "nbb_6_11_5_conflict_2p5_cap20",
                {"required_true_features": ["recent_fvg_near_entry"]},
            ),
            "bear_total_6_20x_boost",
            {"regime_labels": ["high_growth"]},
        ),
        "NBB requires FVG; bear=6 requires high_growth; no leverage caps changed.",
    )
    add(
        "nbb_fvg_bear6_hg_cap15",
        update_rule(
            update_rule(
                current_rules,
                "nbb_6_11_5_conflict_2p5_cap20",
                {"required_true_features": ["recent_fvg_near_entry"]},
            ),
            "bear_total_6_20x_boost",
            {"regime_labels": ["high_growth"], "max_effective_leverage": 15.0},
        ),
        "NBB requires FVG; bear=6 requires high_growth and cap 15.",
    )
    add(
        "nbb_fvg_cap12_bear6_hg_cap15",
        update_rule(
            update_rule(
                current_rules,
                "nbb_6_11_5_conflict_2p5_cap20",
                {"required_true_features": ["recent_fvg_near_entry"], "max_effective_leverage": 12.0},
            ),
            "bear_total_6_20x_boost",
            {"regime_labels": ["high_growth"], "max_effective_leverage": 15.0},
        ),
        "Both fragile buckets constrained: NBB FVG cap 12; bear=6 high_growth cap 15.",
    )
    add(
        "fragile_removed",
        remove_rule(remove_rule(current_rules, "nbb_6_11_5_conflict_2p5_cap20"), "bear_total_6_20x_boost"),
        "Remove both fragile generic boosts; keep stable/FVG buckets.",
    )
    return variants


def positive_top_share(events: list[dict[str, Any]]) -> float:
    positives = [float(event.get("return", 0.0) or 0.0) for event in events if float(event.get("return", 0.0) or 0.0) > 0.0]
    if not positives:
        return 0.0
    return round(max(positives) / max(sum(positives), 1e-12) * 100.0, 2)


def remove_top_winner(events: list[dict[str, Any]], data_end: pd.Timestamp) -> dict[str, Any]:
    winners = [
        (idx, event)
        for idx, event in enumerate(events)
        if float(event.get("return", 0.0) or 0.0) > 0.0
    ]
    if not winners:
        return {"removed": None, "summary": compact_summary(event_summary(events, data_end))}
    idx, removed = max(winners, key=lambda item: float(item[1].get("return", 0.0) or 0.0))
    kept = [event for pos, event in enumerate(events) if pos != idx]
    return {
        "removed": {
            "event_key": event_key(removed),
            "entry_time": removed.get("entry_time"),
            "return_pct": round(float(removed.get("return", 0.0) or 0.0) * 100.0, 4),
            "source_effective_leverage": removed.get("source_effective_leverage"),
            "applied_rules": applied_rule_names(removed),
        },
        "summary": compact_summary(event_summary(kept, data_end)),
    }


def yearly_returns(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_year: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        year = str(utc_timestamp(event.get("entry_time")).year)
        by_year.setdefault(year, []).append(event)
    return {
        year: event_return_stats(sorted(items, key=lambda event: int(event.get("entry_idx", 0) or 0)), INITIAL_CAPITAL)
        for year, items in sorted(by_year.items())
    }


def bucket_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        if str(event.get("event_type") or "") != "sota_long":
            continue
        names = applied_rule_names(event)
        if not names:
            continue
        final_name = names[-1]
        counts[final_name] = counts.get(final_name, 0) + 1
    return dict(sorted(counts.items()))


def build_report(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata") or {}
    data_end = utc_timestamp(metadata.get("data_end") or "2026-05-16 00:00:00+00:00")
    events = list((payload.get("live_shadow") or {}).get("events") or [])
    current_rules = list((metadata.get("long_score_bucket_sizing") or {}).get("rules") or [])
    official = compact_summary(payload.get("live_shadow") or {})

    rows: list[dict[str, Any]] = []
    for variant in variant_rules(current_rules):
        adjusted = rescale_events(events, list(variant["rules"]))
        summary = event_summary(adjusted, data_end)
        compacted = compact_summary(summary)
        top_removed = remove_top_winner(adjusted, data_end)
        rows.append(
            {
                "name": variant["name"],
                "note": variant["note"],
                "summary": compacted,
                "delta_vs_official": {
                    "total_return_pct": round(compacted["total_return_pct"] - official["total_return_pct"], 2),
                    "max_drawdown_pct": round(compacted["max_drawdown_pct"] - official["max_drawdown_pct"], 2),
                    "current_year_return_pct": round(
                        compacted["current_year_return_pct"] - official["current_year_return_pct"],
                        2,
                    ),
                },
                "top1_positive_share_pct": positive_top_share(adjusted),
                "remove_top1": {
                    "removed": top_removed["removed"],
                    "delta_vs_variant": {
                        "total_return_pct": round(
                            top_removed["summary"]["total_return_pct"] - compacted["total_return_pct"],
                            2,
                        ),
                        "max_drawdown_pct": round(
                            top_removed["summary"]["max_drawdown_pct"] - compacted["max_drawdown_pct"],
                            2,
                        ),
                        "current_year_return_pct": round(
                            top_removed["summary"]["current_year_return_pct"] - compacted["current_year_return_pct"],
                            2,
                        ),
                    },
                    "summary": top_removed["summary"],
                },
                "bucket_counts": bucket_counts(adjusted),
                "yearly": yearly_returns(adjusted),
                "rules": variant["rules"],
            }
        )

    rows_by_total = sorted(
        rows,
        key=lambda row: (
            float(row["summary"]["total_return_pct"]),
            -float(row["summary"]["max_drawdown_pct"]),
            float(row["summary"]["current_year_return_pct"]),
        ),
        reverse=True,
    )
    rows_by_2026 = sorted(
        rows,
        key=lambda row: (
            float(row["summary"]["current_year_return_pct"]),
            float(row["summary"]["total_return_pct"]),
            -float(row["summary"]["max_drawdown_pct"]),
        ),
        reverse=True,
    )
    safer_than_current = [
        row
        for row in rows_by_total
        if row["summary"]["max_drawdown_pct"] <= official["max_drawdown_pct"] + 1.0
        and row["summary"]["current_year_return_pct"] >= official["current_year_return_pct"] - 20.0
    ]

    return {
        "scope": "accepted_stream_rescale_scan",
        "notes": [
            "This scan rescales the current accepted event stream from pre-bucket leverage.",
            "It does not rerun full candidate generation or re-admit position-lock rejected trades.",
        ],
        "official_live_shadow": official,
        "control_rescaled": rows_by_total[[row["name"] for row in rows_by_total].index("current_rules_rescaled")],
        "best_by_total_return": rows_by_total[:10],
        "best_by_current_year": rows_by_2026[:10],
        "safer_than_current_candidates": safer_than_current[:10],
        "all_variants": rows,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    report = build_report(payload)
    report["input_report"] = str(input_path.resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    print("official", report["official_live_shadow"])
    print("control", report["control_rescaled"]["summary"], report["control_rescaled"]["delta_vs_official"])
    print("best_by_total")
    for row in report["best_by_total_return"][:8]:
        print(row["name"], row["summary"], row["delta_vs_official"], "top1", row["top1_positive_share_pct"])
    print("safer_than_current")
    for row in report["safer_than_current_candidates"][:8]:
        print(row["name"], row["summary"], row["delta_vs_official"], "top1", row["top1_positive_share_pct"])


if __name__ == "__main__":
    main()
