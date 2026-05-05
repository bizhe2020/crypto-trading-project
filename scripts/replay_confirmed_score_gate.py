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

from scripts.replay_sota_smc_live_shadow import (  # noqa: E402
    compact_combo_with_events,
    compact_live_result,
    decision_counts,
    replay_live_shadow,
    write_paper_log,
)
from scripts.live_shadow_utils import (  # noqa: E402
    add_combo_deltas,
    add_standard_windows,
    clean_for_json,
    event_stream_summary,
    standard_event_summary,
)
from scripts.scan_confirmed_score_gates import passes_gate  # noqa: E402
from scripts.smc_live_utils import replay_base_priority_sota_first  # noqa: E402


DEFAULT_INPUT = ROOT / "var" / "high_leverage_expansion" / "confirmed_multiframe_scores_full_execsync_on_20260505.json"
DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "confirmed_score_gate_replay.json"
DEFAULT_PAPER_LOG = ROOT / "var" / "high_leverage_expansion" / "confirmed_score_gate_paper_decisions.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay confirmed multiframe score-gated SOTA + SMC candidates.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--paper-log-output", default=str(DEFAULT_PAPER_LOG))
    parser.add_argument("--sample-trades", type=int, default=40)
    parser.add_argument("--sota-net-min", type=int, default=2)
    parser.add_argument("--sota-bull-min", type=int, default=8)
    parser.add_argument("--sota-bear-max", type=int, default=6)
    parser.add_argument("--sota-conflict-mode", default="any", choices=("any", "conflict", "clean"))
    parser.add_argument("--smc-net-max", type=int)
    parser.add_argument("--smc-bear-min", type=int)
    parser.add_argument("--smc-bull-max", type=int)
    parser.add_argument("--smc-conflict-mode", default="any", choices=("any", "conflict", "clean"))
    return parser.parse_args()


def reference_gap(reference: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    reference_keys = {f"{event.get('event_type')}|{event.get('entry_idx')}|{event.get('exit_idx')}" for event in reference.get("events", [])}
    live_keys = {f"{event.get('event_type')}|{event.get('entry_idx')}|{event.get('exit_idx')}" for event in live.get("events", [])}
    return {
        "reference_total_return_pct": reference.get("total_return_pct"),
        "live_total_return_pct": live.get("total_return_pct"),
        "return_gap_pct": round(float(live.get("total_return_pct", 0.0) or 0.0) - float(reference.get("total_return_pct", 0.0) or 0.0), 4),
        "reference_max_drawdown_pct": reference.get("max_drawdown_pct"),
        "live_max_drawdown_pct": live.get("max_drawdown_pct"),
        "dd_gap_pct": round(float(live.get("max_drawdown_pct", 0.0) or 0.0) - float(reference.get("max_drawdown_pct", 0.0) or 0.0), 4),
        "accepted_only_in_reference": sorted(reference_keys - live_keys),
        "accepted_only_in_live": sorted(live_keys - reference_keys),
    }


def main() -> None:
    args = parse_args()
    source = json.loads(Path(args.input).read_text())
    scored_events = source.get("scored_events", [])
    if not isinstance(scored_events, list) or not scored_events:
        raise ValueError(f"No scored_events found in {args.input}")

    initial_capital = 1000.0
    data_end = pd.Timestamp(source["metadata"]["data_end"])
    original_sota = [event for event in scored_events if str(event.get("event_type")) == "sota_long"]
    original_smc = [event for event in scored_events if str(event.get("event_type")) == "smc_short"]
    baseline = event_stream_summary(original_sota, initial_capital, data_end)

    filtered_sota = [
        event for event in original_sota
        if passes_gate(
            event,
            net_min=int(args.sota_net_min),
            bull_min=int(args.sota_bull_min),
            bear_max=int(args.sota_bear_max),
            conflict_mode=str(args.sota_conflict_mode),
        )
    ]
    if args.smc_net_max is None and args.smc_bear_min is None and args.smc_bull_max is None:
        filtered_smc = original_smc
    else:
        filtered_smc = [
            event for event in original_smc
            if passes_gate(
                event,
                net_max=args.smc_net_max,
                bear_min=args.smc_bear_min,
                bull_max=args.smc_bull_max,
                conflict_mode=str(args.smc_conflict_mode),
            )
        ]

    filtered_baseline = event_stream_summary(filtered_sota, initial_capital, data_end)
    reference = replay_base_priority_sota_first(filtered_sota, filtered_smc, initial_capital, data_end, baseline)
    live, decisions = replay_live_shadow(filtered_sota + filtered_smc, initial_capital, data_end, baseline)
    parallel = standard_event_summary(filtered_sota + filtered_smc, initial_capital, "entry_idx")

    parallel = add_standard_windows(parallel, initial_capital, data_end, "entry_idx")
    parallel = add_combo_deltas(parallel, baseline)
    parallel["combo_mode"] = "parallel_no_position_lock"

    report = {
        "metadata": {
            **source.get("metadata", {}),
            "input": str(Path(args.input).resolve()),
            "paper_log_output": str(Path(args.paper_log_output).resolve()),
            "sota_gate": {
                "net_min": int(args.sota_net_min),
                "bull_min": int(args.sota_bull_min),
                "bear_max": int(args.sota_bear_max),
                "conflict_mode": str(args.sota_conflict_mode),
            },
            "smc_gate": {
                "net_max": args.smc_net_max,
                "bear_min": args.smc_bear_min,
                "bull_max": args.smc_bull_max,
                "conflict_mode": str(args.smc_conflict_mode),
                "enabled": not (args.smc_net_max is None and args.smc_bear_min is None and args.smc_bull_max is None),
            },
        },
        "candidate_generation": {
            "original_sota_candidates": len(original_sota),
            "filtered_sota_candidates": len(filtered_sota),
            "original_smc_candidates": len(original_smc),
            "filtered_smc_candidates": len(filtered_smc),
        },
        "baseline_shadow_sota": compact_combo_with_events(baseline, int(args.sample_trades)),
        "filtered_sota_only": compact_combo_with_events(filtered_baseline, int(args.sample_trades)),
        "parallel_no_position_lock": compact_combo_with_events(parallel, int(args.sample_trades)),
        "reference_base_priority_sota_first": compact_combo_with_events(reference, int(args.sample_trades)),
        "live_shadow": compact_live_result(live, int(args.sample_trades)),
        "standard_windows": live.get("windows", {}),
        "reference_gap": reference_gap(reference, live),
        "decision_counts": decision_counts(decisions),
        "decisions": decisions,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_paper_log(Path(args.paper_log_output), decisions)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    print(
        f"Filtered SOTA {len(filtered_sota)}/{len(original_sota)}; "
        f"SMC {len(filtered_smc)}/{len(original_smc)}"
    )
    print(
        f"Live-shadow full={live['total_return_pct']:.2f}%/{live['max_drawdown_pct']:.2f}% "
        f"2026={live['windows']['current_year']['total_return_pct']:.2f}% "
        f"trades={live['trades']}"
    )


if __name__ == "__main__":
    main()
