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

from scripts.live_shadow_utils import (  # noqa: E402
    add_combo_deltas,
    add_standard_windows,
    clean_for_json,
    event_return_stats,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    standard_event_summary,
)
from scripts.scan_confirmed_score_gates import passes_gate  # noqa: E402


DEFAULT_INPUT = (
    ROOT
    / "var"
    / "high_leverage_expansion"
    / "sota_smc_scoregate_net3_atr_extreme_shadow_2026_top_conservative_20260506.json"
)
DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "long_score_bucket_sizing_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan conditional up-sizing for high-win-rate SOTA long score buckets after the promoted long gate."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--net-min-values", default="5,7,9,11,13")
    parser.add_argument("--bull-min-values", default="8,10,12,14,16")
    parser.add_argument("--bear-max-values", default="0,1,2,3,4,5,6")
    parser.add_argument("--conflict-modes", default="any,clean")
    parser.add_argument("--rule-modes", default="bucket,exact,threshold", help="Comma-separated: bucket,exact,threshold")
    parser.add_argument(
        "--bucket-dim-sets",
        default="net_score;bull_total;bear_total;net_score,bear_total;bull_total,bear_total;net_score,bull_total",
        help="Semicolon-separated dimension sets; each set is comma-separated, for example 'bear_total;net_score,bear_total'",
    )
    parser.add_argument("--min-bucket-trades", type=int, default=10)
    parser.add_argument("--min-bucket-win-rate-pct", type=float, default=48.0)
    parser.add_argument("--min-bucket-profit-factor", type=float, default=1.8)
    parser.add_argument("--min-2026-trades", type=int, default=1)
    parser.add_argument("--min-2026-avg-return-pct", type=float, default=-999.0)
    parser.add_argument("--leverage-multipliers", default="1.05,1.1,1.2,1.35")
    parser.add_argument("--max-leverage-values", default="8,9,10")
    parser.add_argument("--target-leverage-values", default="3,4,5,6,7.5")
    parser.add_argument("--max-dd-increase-pct", type=float, default=1.0)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--sample-events", type=int, default=0)
    return parser.parse_args()


def load_live_shadow_payload(path: Path) -> tuple[list[dict[str, Any]], pd.Timestamp, dict[str, Any]]:
    payload = json.loads(path.read_text())
    live = payload.get("live_shadow") if isinstance(payload.get("live_shadow"), dict) else payload
    events = live.get("events", [])
    if not isinstance(events, list) or not events:
        raise ValueError(f"No live_shadow events found in {path}")
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    data_end_raw = metadata.get("data_end")
    if data_end_raw is None:
        latest_exit = max(pd.Timestamp(event["exit_time"]) for event in events if event.get("exit_time"))
        data_end = latest_exit.tz_convert("UTC") if latest_exit.tzinfo else latest_exit.tz_localize("UTC")
    else:
        data_end = pd.Timestamp(data_end_raw)
        if data_end.tzinfo is None:
            data_end = data_end.tz_localize("UTC")
        else:
            data_end = data_end.tz_convert("UTC")
    return events, data_end, payload


def compact_summary(result: dict[str, Any], sample_events: int = 0) -> dict[str, Any]:
    payload = {
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
        "windows": result.get("windows", {}),
        "delta_vs_shadow_sota": result.get("delta_vs_shadow_sota", {}),
        "window_deltas_vs_shadow_sota": result.get("window_deltas_vs_shadow_sota", {}),
    }
    if sample_events > 0:
        payload["sample_events"] = result.get("events", [])[:sample_events]
    return payload


def event_year(event: dict[str, Any]) -> int:
    return int(pd.Timestamp(event["entry_time"]).year)


def parse_bucket_dim_sets(raw: str) -> list[tuple[str, ...]]:
    dim_sets: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for chunk in str(raw).split(";"):
        dims = tuple(item.strip() for item in str(chunk).split(",") if item.strip())
        if not dims or dims in seen:
            continue
        seen.add(dims)
        dim_sets.append(dims)
    return dim_sets


def is_sota_long_event(event: dict[str, Any]) -> bool:
    return str(event.get("event_type") or "") == "sota_long" and str(event.get("direction") or "") == "BULL"


def summarize_subset(events: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    return event_return_stats(events, initial_capital)


def windowed_bucket_stats(events: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    return {
        "all": summarize_subset(events, initial_capital),
        "train_2022_2024": summarize_subset([event for event in events if event_year(event) <= 2024], initial_capital),
        "validation_2025": summarize_subset([event for event in events if event_year(event) == 2025], initial_capital),
        "oos_2026": summarize_subset([event for event in events if event_year(event) == 2026], initial_capital),
    }


def exact_bucket_key(event: dict[str, Any]) -> str:
    return (
        f"net={int(event.get('net_score', 0) or 0)}|"
        f"bull={int(event.get('bull_total', 0) or 0)}|"
        f"bear={int(event.get('bear_total', 0) or 0)}|"
        f"conflict={int(bool(event.get('conflict')))}|"
        f"lev={float(event.get('source_effective_leverage', 0.0) or 0.0):.2f}"
    )


def normalized_bucket_value(event: dict[str, Any], field: str) -> Any:
    value = event.get(field)
    if field in {"net_score", "bull_total", "bear_total"}:
        return int(value or 0)
    if field == "conflict":
        return int(bool(value))
    if field == "source_effective_leverage":
        return round(float(value or 0.0), 2)
    return value


def bucket_rule_key(dimensions: tuple[str, ...], values: dict[str, Any]) -> str:
    parts = [f"{field}={values[field]}" for field in dimensions]
    return "|".join(parts)


def exact_bucket_report(sota_events: list[dict[str, Any]], initial_capital: float, min_trades: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for event in sota_events:
        buckets.setdefault(exact_bucket_key(event), []).append(event)
    rows: list[dict[str, Any]] = []
    for key, events in buckets.items():
        stats = windowed_bucket_stats(events, initial_capital)
        if int(stats["all"]["trades"]) < int(min_trades):
            continue
        rows.append({"bucket": key, "stats": stats})
    rows.sort(
        key=lambda item: (
            float(item["stats"]["all"].get("profit_factor", 0.0) or 0.0),
            float(item["stats"]["all"].get("win_rate_pct", 0.0) or 0.0),
            int(item["stats"]["all"].get("trades", 0) or 0),
        ),
        reverse=True,
    )
    return rows


def rule_events(events: list[dict[str, Any]], rule: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in events if is_sota_long_event(event) and event_matches_rule(event, rule)]


def eligible_rule(rule_stats: dict[str, Any], args: argparse.Namespace) -> bool:
    all_stats = rule_stats["all"]
    oos_stats = rule_stats["oos_2026"]
    if int(all_stats.get("trades", 0) or 0) < int(args.min_bucket_trades):
        return False
    if float(all_stats.get("win_rate_pct", 0.0) or 0.0) < float(args.min_bucket_win_rate_pct):
        return False
    if float(all_stats.get("profit_factor", 0.0) or 0.0) < float(args.min_bucket_profit_factor):
        return False
    if int(oos_stats.get("trades", 0) or 0) < int(args.min_2026_trades):
        return False
    if float(oos_stats.get("avg_return_pct", 0.0) or 0.0) < float(args.min_2026_avg_return_pct):
        return False
    return True


def event_matches_rule(event: dict[str, Any], rule: dict[str, Any]) -> bool:
    mode = str(rule.get("mode") or "")
    if mode == "exact" and "exact_bucket" in rule:
        return exact_bucket_key(event) == str(rule["exact_bucket"])
    if mode == "bucket":
        dimensions = tuple(str(item) for item in rule.get("dimensions", []))
        values = rule.get("values", {})
        for field in dimensions:
            if normalized_bucket_value(event, field) != values.get(field):
                return False
        return True
    return passes_gate(event, **rule)


def build_exact_boost_rules(events: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        buckets.setdefault(exact_bucket_key(event), []).append(event)
    rules: list[dict[str, Any]] = []
    for key, selected in buckets.items():
        stats = windowed_bucket_stats(selected, float(args.initial_capital))
        rules.append(
            {
                "rule_mode": "exact",
                "rule": {"mode": "exact", "exact_bucket": key},
                "eligible": eligible_rule(stats, args),
                "stats": stats,
            }
        )
    return rules


def build_bucket_boost_rules(events: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for dimensions in parse_bucket_dim_sets(args.bucket_dim_sets):
        buckets: dict[str, list[dict[str, Any]]] = {}
        bucket_values: dict[str, dict[str, Any]] = {}
        for event in events:
            values = {field: normalized_bucket_value(event, field) for field in dimensions}
            key = bucket_rule_key(dimensions, values)
            buckets.setdefault(key, []).append(event)
            bucket_values[key] = values
        for key, selected in buckets.items():
            stats = windowed_bucket_stats(selected, float(args.initial_capital))
            rules.append(
                {
                    "rule_mode": "bucket",
                    "rule": {
                        "mode": "bucket",
                        "dimensions": list(dimensions),
                        "values": bucket_values[key],
                        "bucket_key": key,
                    },
                    "eligible": eligible_rule(stats, args),
                    "stats": stats,
                }
            )
    return rules


def build_threshold_boost_rules(events: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    seen: set[str] = set()
    for net_min, bull_min, bear_max, conflict_mode in product(
        parse_int_list(args.net_min_values),
        parse_int_list(args.bull_min_values),
        parse_int_list(args.bear_max_values),
        parse_str_list(args.conflict_modes),
    ):
        rule = {
            "net_min": int(net_min),
            "bull_min": int(bull_min),
            "bear_max": int(bear_max),
            "conflict_mode": str(conflict_mode),
        }
        key = json.dumps(rule, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        selected = rule_events(events, rule)
        stats = windowed_bucket_stats(selected, float(args.initial_capital))
        rules.append(
            {
                "rule_mode": "threshold",
                "rule": rule,
                "eligible": eligible_rule(stats, args),
                "stats": stats,
            }
        )
    return rules


def build_boost_rules(events: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    modes = {item.strip().lower() for item in parse_str_list(args.rule_modes)}
    rules: list[dict[str, Any]] = []
    if "bucket" in modes:
        rules.extend(build_bucket_boost_rules(events, args))
    if "exact" in modes:
        rules.extend(build_exact_boost_rules(events, args))
    if "threshold" in modes:
        rules.extend(build_threshold_boost_rules(events, args))
    rules.sort(
        key=lambda item: (
            int(bool(item["eligible"])),
            float(item["stats"]["all"].get("profit_factor", 0.0) or 0.0),
            float(item["stats"]["oos_2026"].get("profit_factor", 0.0) or 0.0),
            float(item["stats"]["all"].get("win_rate_pct", 0.0) or 0.0),
            int(item["stats"]["all"].get("trades", 0) or 0),
        ),
        reverse=True,
    )
    return rules


def apply_bucket_sizing(
    events: list[dict[str, Any]],
    *,
    rule: dict[str, Any],
    leverage_multiplier: float,
    max_leverage: float | None,
    target_leverage: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    boosted = 0
    boosted_2026 = 0
    source_leverages: list[float] = []
    target_leverages: list[float] = []
    scales: list[float] = []
    boosted_returns: list[float] = []
    for event in events:
        updated = dict(event)
        if str(updated.get("event_type") or "") != "sota_long" or str(updated.get("direction") or "") != "BULL":
            adjusted.append(updated)
            continue
        if not event_matches_rule(updated, rule):
            adjusted.append(updated)
            continue
        source_leverage = float(updated.get("source_effective_leverage", 0.0) or 0.0)
        if source_leverage <= 0:
            adjusted.append(updated)
            continue
        if target_leverage is not None:
            selected_target_leverage = float(target_leverage)
        else:
            selected_target_leverage = source_leverage * float(leverage_multiplier)
            if max_leverage is not None:
                selected_target_leverage = min(selected_target_leverage, float(max_leverage))
        scale = selected_target_leverage / source_leverage
        if scale <= 1.0 + 1e-9:
            adjusted.append(updated)
            continue
        boosted += 1
        if event_year(updated) == 2026:
            boosted_2026 += 1
        source_leverages.append(source_leverage)
        target_leverages.append(selected_target_leverage)
        scales.append(scale)
        boosted_returns.append(float(updated.get("return", 0.0) or 0.0))
        updated["return"] = float(updated.get("return", 0.0) or 0.0) * scale
        updated["return_pct"] = round(float(updated["return"]) * 100.0, 4)
        updated["long_score_bucket_sizing"] = {
            "applied": True,
            "rule": rule,
            "source_effective_leverage": round(source_leverage, 6),
            "target_effective_leverage": round(selected_target_leverage, 6),
            "scale": round(scale, 6),
        }
        adjusted.append(updated)
    diagnostics = {
        "rule": rule,
        "leverage_multiplier": float(leverage_multiplier),
        "max_leverage": None if max_leverage is None else float(max_leverage),
        "target_leverage": None if target_leverage is None else float(target_leverage),
        "boosted_trades": boosted,
        "boosted_2026_trades": boosted_2026,
        "avg_scale": round(sum(scales) / len(scales), 6) if scales else 0.0,
        "avg_source_effective_leverage": round(sum(source_leverages) / len(source_leverages), 6) if source_leverages else 0.0,
        "avg_target_effective_leverage": round(sum(target_leverages) / len(target_leverages), 6) if target_leverages else 0.0,
        "boosted_source_stats": event_return_stats(
            [{"return": value, "exit_reason": "source"} for value in boosted_returns],
            1.0,
        ),
    }
    return adjusted, diagnostics


def scan_bucket_sizing(
    events: list[dict[str, Any]],
    data_end: pd.Timestamp,
    baseline: dict[str, Any],
    eligible_rules: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    initial_capital = float(args.initial_capital)
    baseline_dd = float(baseline.get("max_drawdown_pct", 0.0) or 0.0)
    sizing_candidates: list[dict[str, float | None]] = []
    for target_leverage in parse_float_list(args.target_leverage_values):
        sizing_candidates.append({"leverage_multiplier": 1.0, "max_leverage": None, "target_leverage": float(target_leverage)})
    for leverage_multiplier, max_leverage in product(parse_float_list(args.leverage_multipliers), parse_float_list(args.max_leverage_values)):
        sizing_candidates.append(
            {
                "leverage_multiplier": float(leverage_multiplier),
                "max_leverage": float(max_leverage),
                "target_leverage": None,
            }
        )
    for rule_item, sizing in product(eligible_rules, sizing_candidates):
        adjusted, diagnostics = apply_bucket_sizing(
            events,
            rule=rule_item["rule"],
            leverage_multiplier=float(sizing["leverage_multiplier"] or 1.0),
            max_leverage=sizing["max_leverage"],
            target_leverage=sizing["target_leverage"],
        )
        if int(diagnostics["boosted_trades"]) <= 0:
            continue
        summary = standard_event_summary(adjusted, initial_capital, "entry_idx")
        summary = add_standard_windows(summary, initial_capital, data_end, "entry_idx")
        summary = add_combo_deltas(summary, baseline)
        dd_increase = round(float(summary.get("max_drawdown_pct", 0.0) or 0.0) - baseline_dd, 4)
        results.append(
            {
                "boost": diagnostics,
                "rule_stats": rule_item["stats"],
                "live_shadow": compact_summary(summary, int(args.sample_events)),
                "within_dd_budget": dd_increase <= float(args.max_dd_increase_pct),
                "dd_increase_pct": dd_increase,
            }
        )
    return results


def sort_by_2026(item: dict[str, Any]) -> tuple[float, float, float]:
    live = item["live_shadow"]
    current_year = live.get("windows", {}).get("current_year", {})
    return (
        float(current_year.get("total_return_pct", 0.0) or 0.0),
        float(live.get("total_return_pct", 0.0) or 0.0),
        -float(live.get("max_drawdown_pct", 0.0) or 0.0),
    )


def sort_by_return(item: dict[str, Any]) -> tuple[float, float, float]:
    live = item["live_shadow"]
    current_year = live.get("windows", {}).get("current_year", {})
    return (
        float(live.get("total_return_pct", 0.0) or 0.0),
        -float(live.get("max_drawdown_pct", 0.0) or 0.0),
        float(current_year.get("total_return_pct", 0.0) or 0.0),
    )


def main() -> None:
    args = parse_args()
    events, data_end, source_payload = load_live_shadow_payload(Path(args.input))
    initial_capital = float(args.initial_capital)
    baseline = standard_event_summary(events, initial_capital, "entry_idx")
    baseline = add_standard_windows(baseline, initial_capital, data_end, "entry_idx")
    sota_events = [
        event
        for event in baseline.get("events", [])
        if str(event.get("event_type") or "") == "sota_long" and str(event.get("direction") or "") == "BULL"
    ]
    rules = build_boost_rules(sota_events, args)
    eligible_rules = [item for item in rules if bool(item["eligible"])]
    boosted = scan_bucket_sizing(baseline["events"], data_end, baseline, eligible_rules, args)
    boosted_by_2026 = sorted(boosted, key=sort_by_2026, reverse=True)
    boosted_by_return = sorted(boosted, key=sort_by_return, reverse=True)
    constrained_by_2026 = [item for item in boosted_by_2026 if bool(item["within_dd_budget"])]
    constrained_by_return = [item for item in boosted_by_return if bool(item["within_dd_budget"])]
    report = {
        "metadata": {
            "input": str(Path(args.input).resolve()),
            "source_metadata": source_payload.get("metadata", {}),
            "data_end": str(data_end),
            "initial_capital": initial_capital,
            "candidate_rule_count": len(rules),
            "eligible_rule_count": len(eligible_rules),
            "boosted_candidate_count": len(boosted),
            "filters": {
                "min_bucket_trades": int(args.min_bucket_trades),
                "min_bucket_win_rate_pct": float(args.min_bucket_win_rate_pct),
                "min_bucket_profit_factor": float(args.min_bucket_profit_factor),
                "min_2026_trades": int(args.min_2026_trades),
                "min_2026_avg_return_pct": float(args.min_2026_avg_return_pct),
                "max_dd_increase_pct": float(args.max_dd_increase_pct),
            },
        },
        "baseline": compact_summary(baseline, int(args.sample_events)),
        "top_exact_buckets": exact_bucket_report(sota_events, initial_capital, int(args.min_bucket_trades))[: int(args.top_n)],
        "top_rules": rules[: int(args.top_n)],
        "eligible_rules": eligible_rules[: int(args.top_n)],
        "top_by_2026": boosted_by_2026[: int(args.top_n)],
        "top_by_return": boosted_by_return[: int(args.top_n)],
        "top_constrained_by_2026": constrained_by_2026[: int(args.top_n)],
        "top_constrained_by_return": constrained_by_return[: int(args.top_n)],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output)
    base_year = baseline.get("windows", {}).get("current_year", {})
    print(
        f"Baseline full={float(baseline.get('total_return_pct', 0.0) or 0.0):.2f}%/"
        f"{float(baseline.get('max_drawdown_pct', 0.0) or 0.0):.2f}% "
        f"2026={float(base_year.get('total_return_pct', 0.0) or 0.0):.2f}%/"
        f"{float(base_year.get('max_drawdown_pct', 0.0) or 0.0):.2f}%"
    )
    print(f"Rules scanned={len(rules)} eligible={len(eligible_rules)} boosted_candidates={len(boosted)}")
    for label, rows in (("Best 2026", boosted_by_2026[:3]), ("Best constrained", constrained_by_2026[:3])):
        for idx, item in enumerate(rows, start=1):
            live = item["live_shadow"]
            year = live.get("windows", {}).get("current_year", {})
            print(
                f"{label} {idx}: full={float(live.get('total_return_pct', 0.0) or 0.0):.2f}%/"
                f"{float(live.get('max_drawdown_pct', 0.0) or 0.0):.2f}% "
                f"2026={float(year.get('total_return_pct', 0.0) or 0.0):.2f}%/"
                f"{float(year.get('max_drawdown_pct', 0.0) or 0.0):.2f}% "
                f"boosted={item['boost']['boosted_trades']} "
                f"mode={item['boost']['rule'].get('mode')} "
                f"rule={item['boost']['rule']} "
                f"mult={item['boost']['leverage_multiplier']} cap={item['boost']['max_leverage']} "
                f"target={item['boost']['target_leverage']} "
                f"dd_inc={item['dd_increase_pct']:+.2f}"
            )


if __name__ == "__main__":
    main()
