#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_shadow_utils import (  # noqa: E402
    add_combo_deltas,
    add_standard_windows,
    clean_for_json,
    standard_event_summary,
)


DEFAULT_INPUT = ROOT / "var" / "reports" / "live_config_recall_on_replay_20260517.json"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "kelly_bucket_sizing_scan_20260520.json"


@dataclass(frozen=True)
class Candidate:
    name: str
    target_leverages: tuple[float, ...]
    predicate: Callable[[dict[str, Any]], bool]
    min_trades: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan Kelly-inspired fixed target leverage candidates on frozen live-shadow events."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--max-dd-increase-pct", type=float, default=1.0)
    parser.add_argument("--sample-events", type=int, default=20)
    return parser.parse_args()


def load_events(path: Path) -> tuple[list[dict[str, Any]], pd.Timestamp, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    live = payload.get("live_shadow") if isinstance(payload.get("live_shadow"), dict) else payload
    events = live.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError(f"No live_shadow.events found in {path}")
    raw_end = payload.get("metadata", {}).get("data_end") if isinstance(payload.get("metadata"), dict) else None
    if raw_end is None:
        raw_end = max(event["exit_time"] for event in events if event.get("exit_time"))
    data_end = pd.Timestamp(raw_end)
    if data_end.tzinfo is None:
        data_end = data_end.tz_localize("UTC")
    else:
        data_end = data_end.tz_convert("UTC")
    return [dict(event) for event in events], data_end, payload


def event_year(event: dict[str, Any]) -> int:
    return int(pd.Timestamp(event["entry_time"]).year)


def is_sota_long(event: dict[str, Any]) -> bool:
    return str(event.get("event_type") or "") == "sota_long" and str(event.get("direction") or "") == "BULL"


def source_leverage(event: dict[str, Any]) -> float:
    raw = event.get("source_effective_leverage")
    if raw is not None:
        try:
            value = float(raw)
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    event_type = str(event.get("event_type") or "")
    if event_type == "smc_short":
        return 10.0
    if event_type == "gap_smc_short_expansion":
        return 3.0
    return 0.0


def unit_return(event: dict[str, Any]) -> float | None:
    if event.get("smc_unit_return_pct") is not None:
        return float(event["smc_unit_return_pct"]) / 100.0
    leverage = source_leverage(event)
    if leverage <= 0:
        return None
    return float(event.get("return", 0.0) or 0.0) / leverage


def compact_summary(result: dict[str, Any], sample_events: int = 0) -> dict[str, Any]:
    output = {key: value for key, value in result.items() if key != "events"}
    if sample_events > 0:
        output["sample_events"] = result.get("events", [])[:sample_events]
    return output


def summarize_subset(events: list[dict[str, Any]]) -> dict[str, Any]:
    units = [unit_return(event) for event in events]
    units = [value for value in units if value is not None]
    wins = [value for value in units if value > 0]
    losses = [value for value in units if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    leverages = [source_leverage(event) for event in events if source_leverage(event) > 0]
    return {
        "trades": len(events),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(units) * 100.0, 2) if units else 0.0,
        "unit_avg_return_pct": round(sum(units) / len(units) * 100.0, 4) if units else 0.0,
        "unit_avg_win_pct": round(sum(wins) / len(wins) * 100.0, 4) if wins else 0.0,
        "unit_avg_loss_pct": round(sum(losses) / len(losses) * 100.0, 4) if losses else 0.0,
        "unit_profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "avg_source_leverage": round(sum(leverages) / len(leverages), 4) if leverages else 0.0,
        "years": {str(year): sum(1 for event in events if event_year(event) == year) for year in sorted({event_year(event) for event in events})},
    }


def candidates() -> list[Candidate]:
    return [
        Candidate(
            name="sota_fvg_near_bear3",
            target_leverages=(3.5, 4.0, 5.0),
            predicate=lambda event: is_sota_long(event)
            and int(event.get("bear_total", 0) or 0) == 3
            and bool(event.get("feature_recent_fvg_near_entry")),
            min_trades=10,
        ),
        Candidate(
            name="sota_fvg_hg_net8",
            target_leverages=(6.0, 8.0),
            predicate=lambda event: is_sota_long(event)
            and int(event.get("net_score", 0) or 0) == 8
            and str(event.get("regime_label") or "") == "high_growth"
            and bool(event.get("feature_recent_fvg_near_entry")),
            min_trades=5,
        ),
        Candidate(
            name="gap_smc_short",
            target_leverages=(5.0, 8.0, 10.0),
            predicate=lambda event: str(event.get("event_type") or "") == "gap_smc_short_expansion",
            min_trades=5,
        ),
        Candidate(
            name="sota_bear2_normal",
            target_leverages=(4.0, 5.0, 6.0),
            predicate=lambda event: is_sota_long(event)
            and int(event.get("bear_total", 0) or 0) == 2
            and str(event.get("regime_label") or "") == "normal",
            min_trades=10,
        ),
    ]


def apply_target_leverage(
    events: list[dict[str, Any]],
    candidate: Candidate,
    target_leverage: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    boosted: list[dict[str, Any]] = []
    for event in events:
        updated = dict(event)
        if not candidate.predicate(updated):
            adjusted.append(updated)
            continue
        source = source_leverage(updated)
        unit = unit_return(updated)
        if source <= 0 or unit is None or target_leverage <= source + 1e-9:
            adjusted.append(updated)
            continue
        updated["return"] = unit * float(target_leverage)
        updated["return_pct"] = round(float(updated["return"]) * 100.0, 4)
        updated["kelly_bucket_sizing"] = {
            "candidate": candidate.name,
            "source_effective_leverage": round(source, 6),
            "target_effective_leverage": round(float(target_leverage), 6),
            "scale": round(float(target_leverage) / source, 6),
        }
        boosted.append(updated)
        adjusted.append(updated)
    diagnostics = {
        "candidate": candidate.name,
        "target_effective_leverage": float(target_leverage),
        "boosted_trades": len(boosted),
        "boosted_2026_trades": sum(1 for event in boosted if event_year(event) == 2026),
        "source_stats": summarize_subset([event for event in events if candidate.predicate(event)]),
        "boosted_source_stats": summarize_subset(boosted),
        "boosted_events": [
            {
                "entry_time": event.get("entry_time"),
                "exit_time": event.get("exit_time"),
                "event_type": event.get("event_type"),
                "direction": event.get("direction"),
                "return_pct": event.get("return_pct"),
                "exit_reason": event.get("exit_reason"),
                "source_effective_leverage": event.get("kelly_bucket_sizing", {}).get("source_effective_leverage"),
                "target_effective_leverage": event.get("kelly_bucket_sizing", {}).get("target_effective_leverage"),
                "net_score": event.get("net_score"),
                "bull_total": event.get("bull_total"),
                "bear_total": event.get("bear_total"),
                "regime_label": event.get("regime_label"),
                "feature_recent_fvg_near_entry": event.get("feature_recent_fvg_near_entry"),
            }
            for event in boosted
        ],
    }
    return adjusted, diagnostics


def drop_top_trade(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not events:
        return []
    max_idx = max(range(len(events)), key=lambda idx: float(events[idx].get("return", 0.0) or 0.0))
    return [event for idx, event in enumerate(events) if idx != max_idx]


def run_summary(events: list[dict[str, Any]], initial_capital: float, data_end: pd.Timestamp) -> dict[str, Any]:
    summary = standard_event_summary(events, initial_capital, "entry_idx")
    return add_standard_windows(summary, initial_capital, data_end, "entry_idx")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    events, data_end, payload = load_events(input_path)
    initial_capital = float(args.initial_capital)
    baseline = run_summary(events, initial_capital, data_end)
    baseline_drop_top1 = run_summary(drop_top_trade(baseline["events"]), initial_capital, data_end)
    baseline_2026 = baseline.get("windows", {}).get("current_year", {})
    baseline_dd = float(baseline.get("max_drawdown_pct", 0.0) or 0.0)

    results: list[dict[str, Any]] = []
    for candidate in candidates():
        selected = [event for event in events if candidate.predicate(event)]
        if len(selected) < candidate.min_trades:
            results.append(
                {
                    "candidate": candidate.name,
                    "skipped": True,
                    "reason": "not_enough_trades",
                    "source_stats": summarize_subset(selected),
                }
            )
            continue
        for target in candidate.target_leverages:
            adjusted, diagnostics = apply_target_leverage(events, candidate, float(target))
            if int(diagnostics["boosted_trades"]) <= 0:
                continue
            summary = run_summary(adjusted, initial_capital, data_end)
            summary = add_combo_deltas(summary, baseline)
            dropped = run_summary(drop_top_trade(summary["events"]), initial_capital, data_end)
            year = summary.get("windows", {}).get("current_year", {})
            result = {
                "candidate": candidate.name,
                "target_effective_leverage": float(target),
                "boost": diagnostics,
                "live_shadow": compact_summary(summary, int(args.sample_events)),
                "drop_top1": compact_summary(dropped, 0),
                "dd_increase_pct": round(float(summary.get("max_drawdown_pct", 0.0) or 0.0) - baseline_dd, 4),
                "return_delta_pct": round(float(summary.get("total_return_pct", 0.0) or 0.0) - float(baseline.get("total_return_pct", 0.0) or 0.0), 4),
                "current_year_delta_pct": round(float(year.get("total_return_pct", 0.0) or 0.0) - float(baseline_2026.get("total_return_pct", 0.0) or 0.0), 4),
                "drop_top1_delta_vs_baseline_drop_top1_pct": round(
                    float(dropped.get("total_return_pct", 0.0) or 0.0)
                    - float(baseline_drop_top1.get("total_return_pct", 0.0) or 0.0),
                    4,
                ),
            }
            result["passes_constraints"] = (
                result["dd_increase_pct"] <= float(args.max_dd_increase_pct)
                and result["drop_top1_delta_vs_baseline_drop_top1_pct"] > 0.0
                and result["current_year_delta_pct"] >= 0.0
            )
            results.append(result)

    sortable = [row for row in results if not row.get("skipped")]
    top_constrained = sorted(
        [row for row in sortable if bool(row["passes_constraints"])],
        key=lambda row: (
            float(row["current_year_delta_pct"]),
            float(row["return_delta_pct"]),
            -float(row["dd_increase_pct"]),
        ),
        reverse=True,
    )
    top_by_return = sorted(
        sortable,
        key=lambda row: (
            float(row["return_delta_pct"]),
            float(row["current_year_delta_pct"]),
            -float(row["dd_increase_pct"]),
        ),
        reverse=True,
    )
    report = {
        "metadata": {
            "input": str(input_path.resolve()),
            "source_metadata": payload.get("metadata", {}),
            "data_end": str(data_end),
            "initial_capital": initial_capital,
            "max_dd_increase_pct": float(args.max_dd_increase_pct),
            "constraints": {
                "dd_increase_lte_pct": float(args.max_dd_increase_pct),
                "drop_top1_delta_vs_baseline_drop_top1_gt_pct": 0.0,
                "current_year_delta_gte_pct": 0.0,
            },
        },
        "baseline": compact_summary(baseline, 0),
        "baseline_drop_top1": compact_summary(baseline_drop_top1, 0),
        "all_results": results,
        "top_constrained": top_constrained,
        "top_by_return": top_by_return,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(output)
    print(
        f"baseline={float(baseline.get('total_return_pct', 0.0) or 0.0):.2f}%/"
        f"{float(baseline.get('max_drawdown_pct', 0.0) or 0.0):.2f}% "
        f"2026={float(baseline_2026.get('total_return_pct', 0.0) or 0.0):.2f}%"
    )
    print(f"results={len(sortable)} constrained={len(top_constrained)}")
    for idx, row in enumerate(top_constrained[:8], start=1):
        live = row["live_shadow"]
        year = live.get("windows", {}).get("current_year", {})
        print(
            f"{idx}. {row['candidate']} target={row['target_effective_leverage']:.2f}x "
            f"full={float(live.get('total_return_pct', 0.0) or 0.0):.2f}% "
            f"dd={float(live.get('max_drawdown_pct', 0.0) or 0.0):.2f}% "
            f"2026={float(year.get('total_return_pct', 0.0) or 0.0):.2f}% "
            f"dd_inc={row['dd_increase_pct']:+.2f} "
            f"drop_top_delta={row['drop_top1_delta_vs_baseline_drop_top1_pct']:+.2f}"
        )


if __name__ == "__main__":
    main()
