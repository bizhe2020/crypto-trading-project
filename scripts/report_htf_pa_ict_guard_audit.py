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

from scripts.reproduce_htf_pa_ict_guard import clean_for_json


DEFAULT_INPUT = Path("var/pa_ict_liquidity/htf_context/htf_pa_ict_guard_audit_source_events.json")
DEFAULT_OUTPUT = Path("var/pa_ict_liquidity/htf_context/htf_pa_ict_guard_audit.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit HTF PA/ICT guarded trades from reproduction events.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--baseline-multiplier", type=float, default=1.0)
    parser.add_argument("--guard-multiplier", type=float, default=0.70)
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args()


def result_by_multiplier(report: dict[str, Any], multiplier: float) -> dict[str, Any]:
    for result in report.get("results", []):
        if abs(float(result.get("multiplier", -999.0)) - multiplier) < 1e-9:
            return result
    raise ValueError(f"Missing multiplier result: {multiplier}")


def events_by_entry(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(event["entry_time"]): event for event in events}


def pct(value: float) -> float:
    return round(float(value) * 100.0, 4)


def context_text(event: dict[str, Any]) -> str:
    h4 = f"{event.get('htf_h4_state', 'none')}/{event.get('htf_h4_alignment', 'none')}"
    d1 = f"{event.get('htf_d1_state', 'none')}/{event.get('htf_d1_alignment', 'none')}"
    return f"4H {h4} age={event.get('htf_h4_age_bars')} | 1D {d1} age={event.get('htf_d1_age_bars')}"


def audit_rows(baseline: dict[str, Any], guarded: dict[str, Any]) -> list[dict[str, Any]]:
    base_events = events_by_entry(baseline.get("fixed_events", []))
    guarded_events = [
        event
        for event in guarded.get("fixed_events", [])
        if bool(event.get("htf_pa_ict_guard_applied"))
    ]
    rows: list[dict[str, Any]] = []
    for event in guarded_events:
        entry_time = str(event["entry_time"])
        base = base_events.get(entry_time, {})
        base_return = float(base.get("return", 0.0) or 0.0)
        guarded_return = float(event.get("return", 0.0) or 0.0)
        base_leverage = float(base.get("effective_leverage", 0.0) or 0.0)
        guarded_leverage = float(event.get("effective_leverage", 0.0) or 0.0)
        rows.append(
            {
                "entry_time": entry_time,
                "exit_time": str(event.get("exit_time")),
                "direction": str(event.get("direction") or ""),
                "exit_reason": str(event.get("exit_reason") or ""),
                "risk_mode": str(event.get("risk_mode") or ""),
                "regime_label": str(event.get("regime_label") or ""),
                "signal_return_pct": pct(float(event.get("signal_return", 0.0) or 0.0)),
                "baseline_return_pct": pct(base_return),
                "guarded_return_pct": pct(guarded_return),
                "return_delta_pct": pct(guarded_return - base_return),
                "baseline_leverage": round(base_leverage, 6),
                "guarded_leverage": round(guarded_leverage, 6),
                "leverage_delta": round(guarded_leverage - base_leverage, 6),
                "entry_price": event.get("entry_price"),
                "exit_price": event.get("exit_price"),
                "initial_stop_price": event.get("initial_stop_price"),
                "h4_state": event.get("htf_h4_state"),
                "h4_alignment": event.get("htf_h4_alignment"),
                "h4_age_bars": event.get("htf_h4_age_bars"),
                "d1_state": event.get("htf_d1_state"),
                "d1_alignment": event.get("htf_d1_alignment"),
                "d1_age_bars": event.get("htf_d1_age_bars"),
                "context": context_text(event),
                "reasons": event.get("reasons", []),
            }
        )
    return rows


def contribution_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    loss_rows = [row for row in rows if float(row["baseline_return_pct"]) < 0]
    win_rows = [row for row in rows if float(row["baseline_return_pct"]) > 0]
    return {
        "guarded_trades": len(rows),
        "baseline_return_sum_pct": round(sum(float(row["baseline_return_pct"]) for row in rows), 4),
        "guarded_return_sum_pct": round(sum(float(row["guarded_return_pct"]) for row in rows), 4),
        "return_delta_sum_pct": round(sum(float(row["return_delta_pct"]) for row in rows), 4),
        "baseline_losses": len(loss_rows),
        "baseline_wins": len(win_rows),
        "loss_delta_sum_pct": round(sum(float(row["return_delta_pct"]) for row in loss_rows), 4),
        "win_delta_sum_pct": round(sum(float(row["return_delta_pct"]) for row in win_rows), 4),
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "entry_time",
        "direction",
        "baseline_return_pct",
        "guarded_return_pct",
        "return_delta_pct",
        "risk_mode",
        "regime_label",
        "context",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = json.loads(Path(args.input).read_text())
    baseline = result_by_multiplier(report, args.baseline_multiplier)
    guarded = result_by_multiplier(report, args.guard_multiplier)
    rows = audit_rows(baseline, guarded)
    summary = contribution_summary(rows)
    output_report = {
        "input": str(Path(args.input)),
        "baseline_multiplier": args.baseline_multiplier,
        "guard_multiplier": args.guard_multiplier,
        "baseline_shadow": {key: value for key, value in baseline.get("shadow", {}).items() if key != "events"},
        "guarded_shadow": {key: value for key, value in guarded.get("shadow", {}).items() if key != "events"},
        "summary": summary,
        "rows": rows,
        "markdown_table": markdown_table(rows),
    }
    cleaned = clean_for_json(output_report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    print(json.dumps({"summary": cleaned["summary"], "rows": cleaned["rows"]}, ensure_ascii=False, indent=2, allow_nan=False))
    if args.stdout:
        print(cleaned["markdown_table"])


if __name__ == "__main__":
    main()
