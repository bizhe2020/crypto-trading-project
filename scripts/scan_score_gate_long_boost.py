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
    event_stream_summary,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    standard_event_summary,
)
from scripts.replay_sota_smc_live_shadow import replay_live_shadow  # noqa: E402
from scripts.scan_confirmed_score_gates import passes_gate  # noqa: E402


DEFAULT_INPUT = ROOT / "var" / "high_leverage_expansion" / "confirmed_multiframe_scores_full_execsync_on_20260505.json"
DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "score_gate_long_boost_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan stricter long score gates plus gated leverage boosts on top of strict replay-scored SOTA+SMC events."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--base-sota-net-min-values", default="2,3,4,5")
    parser.add_argument("--base-sota-bull-min-values", default="8,9,10,11,12")
    parser.add_argument("--base-sota-bear-max-values", default="4,5,6")
    parser.add_argument("--base-sota-conflict-modes", default="any,clean")
    parser.add_argument("--base-top-k", type=int, default=12)
    parser.add_argument("--min-base-trades", type=int, default=24)
    parser.add_argument("--boost-net-min-values", default="10,12,14")
    parser.add_argument("--boost-bull-min-values", default="12,14,16")
    parser.add_argument("--boost-bear-max-values", default="1,2,3")
    parser.add_argument("--boost-conflict-modes", default="any,clean")
    parser.add_argument("--boost-leverage-multipliers", default="1.1,1.2,1.35,1.5")
    parser.add_argument("--boost-max-leverages", default="8,10,12")
    parser.add_argument("--min-boosted-trades", type=int, default=6)
    parser.add_argument("--max-dd-increase-pct", type=float, default=2.0)
    parser.add_argument("--top-n", type=int, default=40)
    return parser.parse_args()


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


def sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
    live = item["live_shadow"]
    return (
        float(live["total_return_pct"]),
        -float(live["max_drawdown_pct"]),
        float(live["current_year_return_pct"]),
    )


def boosted_event_summary(
    accepted_events: list[dict[str, Any]],
    initial_capital: float,
    data_end: pd.Timestamp,
    baseline: dict[str, Any],
    *,
    boost_rule: dict[str, Any],
    leverage_multiplier: float,
    max_leverage: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    boosted_count = 0
    boosted_2026_count = 0
    boosted_scales: list[float] = []
    source_leverages: list[float] = []
    target_leverages: list[float] = []

    for event in accepted_events:
        updated = dict(event)
        if str(updated.get("event_type") or "") != "sota_long" or str(updated.get("direction") or "") != "BULL":
            adjusted.append(updated)
            continue
        if not passes_gate(updated, **boost_rule):
            adjusted.append(updated)
            continue
        source_effective_leverage = float(updated.get("source_effective_leverage", 0.0) or 0.0)
        if source_effective_leverage <= 0.0:
            adjusted.append(updated)
            continue
        boosted_effective_leverage = min(source_effective_leverage * float(leverage_multiplier), float(max_leverage))
        scale = boosted_effective_leverage / source_effective_leverage
        boosted_count += 1
        if pd.Timestamp(updated["entry_time"]).year == 2026:
            boosted_2026_count += 1
        boosted_scales.append(scale)
        source_leverages.append(source_effective_leverage)
        target_leverages.append(boosted_effective_leverage)
        updated["return"] = float(updated.get("return", 0.0) or 0.0) * scale
        updated["return_pct"] = round(float(updated["return"]) * 100.0, 4)
        updated["long_gate_boost"] = {
            "applied": True,
            "boost_rule": boost_rule,
            "source_effective_leverage": round(source_effective_leverage, 6),
            "boosted_effective_leverage": round(boosted_effective_leverage, 6),
            "scale": round(scale, 6),
        }
        adjusted.append(updated)

    result = standard_event_summary(adjusted, initial_capital, "entry_idx")
    result = add_standard_windows(result, initial_capital, data_end, "entry_idx")
    result = add_combo_deltas(result, baseline)
    result["combo_mode"] = "live_shadow_long_gate_boost"
    diagnostics = {
        "boost_rule": boost_rule,
        "leverage_multiplier": float(leverage_multiplier),
        "max_leverage": float(max_leverage),
        "boosted_trades": boosted_count,
        "boosted_2026_trades": boosted_2026_count,
        "avg_scale": round(sum(boosted_scales) / len(boosted_scales), 6) if boosted_scales else 0.0,
        "avg_source_effective_leverage": round(sum(source_leverages) / len(source_leverages), 6) if source_leverages else 0.0,
        "avg_boosted_effective_leverage": round(sum(target_leverages) / len(target_leverages), 6) if target_leverages else 0.0,
    }
    return result, diagnostics


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text())
    scored_events = payload.get("scored_events", [])
    if not isinstance(scored_events, list) or not scored_events:
        raise ValueError(f"No scored_events found in {args.input}")

    initial_capital = 1000.0
    data_end = pd.Timestamp(payload["metadata"]["data_end"])
    all_sota = [event for event in scored_events if str(event.get("event_type")) == "sota_long"]
    all_smc = [event for event in scored_events if str(event.get("event_type")) == "smc_short"]
    baseline = event_stream_summary(all_sota, initial_capital, data_end)

    base_rules: list[dict[str, Any]] = []
    for net_min, bull_min, bear_max, conflict_mode in product(
        parse_int_list(args.base_sota_net_min_values),
        parse_int_list(args.base_sota_bull_min_values),
        parse_int_list(args.base_sota_bear_max_values),
        parse_str_list(args.base_sota_conflict_modes),
    ):
        filtered_sota = [
            event for event in all_sota
            if passes_gate(
                event,
                net_min=int(net_min),
                bull_min=int(bull_min),
                bear_max=int(bear_max),
                conflict_mode=str(conflict_mode),
            )
        ]
        if len(filtered_sota) < int(args.min_base_trades):
            continue
        live, decisions = replay_live_shadow(filtered_sota + all_smc, initial_capital, data_end, baseline)
        base_rules.append(
            {
                "base_rule": {
                    "net_min": int(net_min),
                    "bull_min": int(bull_min),
                    "bear_max": int(bear_max),
                    "conflict_mode": str(conflict_mode),
                },
                "candidate_counts": {
                    "sota_long": len(filtered_sota),
                    "smc_short": len(all_smc),
                },
                "live_shadow": compact_summary(live),
                "accepted_events": live.get("events", []),
                "decision_counts": live.get("decision_counts", {}) or {},
                "accepted_total": int(live.get("trades", 0) or 0),
                "rejected_total": int((live.get("decision_counts", {}) or {}).get("by_decision", {}).get("rejected", 0) or 0),
            }
        )

    ranked_base_rules = sorted(base_rules, key=sort_key, reverse=True)
    selected_base_rules = ranked_base_rules[: max(int(args.base_top_k), 1)]

    boosted_results: list[dict[str, Any]] = []
    baseline_dd = float(selected_base_rules[0]["live_shadow"]["max_drawdown_pct"]) if selected_base_rules else 0.0
    for base_result in selected_base_rules:
        accepted_events = base_result["accepted_events"]
        for boost_net_min, boost_bull_min, boost_bear_max, boost_conflict_mode, leverage_multiplier, max_leverage in product(
            parse_int_list(args.boost_net_min_values),
            parse_int_list(args.boost_bull_min_values),
            parse_int_list(args.boost_bear_max_values),
            parse_str_list(args.boost_conflict_modes),
            parse_float_list(args.boost_leverage_multipliers),
            parse_float_list(args.boost_max_leverages),
        ):
            boost_rule = {
                "net_min": int(boost_net_min),
                "bull_min": int(boost_bull_min),
                "bear_max": int(boost_bear_max),
                "conflict_mode": str(boost_conflict_mode),
            }
            boosted, diagnostics = boosted_event_summary(
                accepted_events,
                initial_capital,
                data_end,
                baseline,
                boost_rule=boost_rule,
                leverage_multiplier=float(leverage_multiplier),
                max_leverage=float(max_leverage),
            )
            if int(diagnostics["boosted_trades"]) < int(args.min_boosted_trades):
                continue
            boosted_summary = compact_summary(boosted)
            dd_increase = round(float(boosted_summary["max_drawdown_pct"]) - baseline_dd, 4)
            boosted_results.append(
                {
                    "base_rule": base_result["base_rule"],
                    "base_live_shadow": base_result["live_shadow"],
                    "candidate_counts": base_result["candidate_counts"],
                    "boost_diagnostics": diagnostics,
                    "live_shadow": boosted_summary,
                    "delta_vs_base_rule": {
                        "total_return_pct": round(float(boosted_summary["total_return_pct"]) - float(base_result["live_shadow"]["total_return_pct"]), 4),
                        "max_drawdown_pct": round(float(boosted_summary["max_drawdown_pct"]) - float(base_result["live_shadow"]["max_drawdown_pct"]), 4),
                        "current_year_return_pct": round(float(boosted_summary["current_year_return_pct"]) - float(base_result["live_shadow"]["current_year_return_pct"]), 4),
                    },
                    "within_dd_budget": dd_increase <= float(args.max_dd_increase_pct),
                }
            )

    def boosted_sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
        live = item["live_shadow"]
        return (
            float(live["total_return_pct"]),
            -float(live["max_drawdown_pct"]),
            float(live["current_year_return_pct"]),
        )

    boosted_results.sort(key=boosted_sort_key, reverse=True)
    constrained_results = [item for item in boosted_results if bool(item.get("within_dd_budget"))]
    report = {
        "metadata": {
            "input": str(Path(args.input).resolve()),
            "data_end": str(data_end),
            "base_top_k": int(args.base_top_k),
            "min_base_trades": int(args.min_base_trades),
            "min_boosted_trades": int(args.min_boosted_trades),
            "max_dd_increase_pct": float(args.max_dd_increase_pct),
        },
        "baseline_shadow_sota": compact_summary(baseline),
        "top_base_rules": selected_base_rules[: int(args.top_n)],
        "top_boosted_by_return": boosted_results[: int(args.top_n)],
        "top_boosted_within_dd_budget": constrained_results[: int(args.top_n)],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    if selected_base_rules:
        best_base = selected_base_rules[0]
        print(
            "Best base "
            f"full={float(best_base['live_shadow']['total_return_pct']):.2f}%/"
            f"{float(best_base['live_shadow']['max_drawdown_pct']):.2f}% "
            f"2026={float(best_base['live_shadow']['current_year_return_pct']):.2f}% "
            f"rule={best_base['base_rule']}"
        )
    if boosted_results:
        best_boost = boosted_results[0]
        print(
            "Best boosted "
            f"full={float(best_boost['live_shadow']['total_return_pct']):.2f}%/"
            f"{float(best_boost['live_shadow']['max_drawdown_pct']):.2f}% "
            f"2026={float(best_boost['live_shadow']['current_year_return_pct']):.2f}% "
            f"boosted={best_boost['boost_diagnostics']['boosted_trades']} "
            f"rule={best_boost['base_rule']} "
            f"boost={best_boost['boost_diagnostics']}"
        )
    if constrained_results:
        best_constrained = constrained_results[0]
        print(
            "Best constrained "
            f"full={float(best_constrained['live_shadow']['total_return_pct']):.2f}%/"
            f"{float(best_constrained['live_shadow']['max_drawdown_pct']):.2f}% "
            f"2026={float(best_constrained['live_shadow']['current_year_return_pct']):.2f}% "
            f"delta={best_constrained['delta_vs_base_rule']}"
        )


if __name__ == "__main__":
    main()
