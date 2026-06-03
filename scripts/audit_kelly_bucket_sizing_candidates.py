#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
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
    standard_event_summary,
)
from scripts.scan_kelly_bucket_sizing import (  # noqa: E402
    Candidate,
    apply_target_leverage,
    candidates,
    load_events,
)


DEFAULT_INPUT = ROOT / "var" / "reports" / "live_config_recall_on_replay_20260517.json"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "kelly_bucket_sizing_candidate_audit_20260520.json"
DEFAULT_TRADES_CSV = ROOT / "var" / "reports" / "kelly_bucket_sizing_candidate_trades_20260520.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit selected Kelly-inspired bucket sizing candidates on frozen live-shadow events."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--trades-csv", default=str(DEFAULT_TRADES_CSV))
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate spec in name:target_leverage format. Defaults to sota_fvg_near_bear3:5 and gap_smc_short:8.",
    )
    parser.add_argument("--extra-roundtrip-bps", default="2.5,5,10,20")
    return parser.parse_args()


def parse_candidate_specs(values: list[str]) -> list[tuple[str, float]]:
    specs = values or ["sota_fvg_near_bear3:5", "gap_smc_short:8"]
    parsed: list[tuple[str, float]] = []
    for raw in specs:
        if ":" not in raw:
            raise ValueError(f"Invalid candidate spec {raw!r}; expected name:target")
        name, target = raw.split(":", 1)
        parsed.append((name.strip(), float(target.strip())))
    return parsed


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    current_year = (summary.get("windows") or {}).get("current_year", {})
    last_60d = (summary.get("windows") or {}).get("last_60d", {})
    last_30d = (summary.get("windows") or {}).get("last_30d", {})
    return {
        "total_return_pct": round(float(summary.get("total_return_pct", 0.0) or 0.0), 2),
        "max_drawdown_pct": round(float(summary.get("max_drawdown_pct", 0.0) or 0.0), 2),
        "trades": int(summary.get("trades", 0) or 0),
        "wins": int(summary.get("wins", 0) or 0),
        "losses": int(summary.get("losses", 0) or 0),
        "win_rate_pct": round(float(summary.get("win_rate_pct", 0.0) or 0.0), 2),
        "avg_return_pct": round(float(summary.get("avg_return_pct", 0.0) or 0.0), 4),
        "profit_factor": round(float(summary.get("profit_factor", 0.0) or 0.0), 4),
        "current_year": compact_window(current_year),
        "last_60d": compact_window(last_60d),
        "last_30d": compact_window(last_30d),
        "event_type_counts": summary.get("event_type_counts", {}),
        "exit_counts": summary.get("exit_counts", {}),
    }


def compact_window(window: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_return_pct": round(float(window.get("total_return_pct", 0.0) or 0.0), 2),
        "max_drawdown_pct": round(float(window.get("max_drawdown_pct", 0.0) or 0.0), 2),
        "trades": int(window.get("trades", 0) or 0),
        "win_rate_pct": round(float(window.get("win_rate_pct", 0.0) or 0.0), 2),
        "profit_factor": round(float(window.get("profit_factor", 0.0) or 0.0), 4),
    }


def run_summary(events: list[dict[str, Any]], initial_capital: float, data_end: pd.Timestamp) -> dict[str, Any]:
    summary = standard_event_summary(events, initial_capital, "entry_idx")
    return add_standard_windows(summary, initial_capital, data_end, "entry_idx")


def event_year(event: dict[str, Any]) -> int:
    return int(utc_timestamp(event["entry_time"]).year)


def yearly_summary(events: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    years = sorted({event_year(event) for event in events})
    output: dict[str, Any] = {}
    for year in years:
        year_events = [event for event in events if event_year(event) == year]
        stats = event_return_stats(year_events, initial_capital)
        stream = standard_event_summary(year_events, initial_capital, "entry_idx")
        stats["max_drawdown_pct"] = round(float(stream.get("max_drawdown_pct", 0.0) or 0.0), 2)
        stats["total_return_pct"] = round(float(stream.get("total_return_pct", 0.0) or 0.0), 2)
        output[str(year)] = stats
    return output


def remove_top_n(events: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0:
        return list(events)
    top_positions = {
        idx
        for idx, _event in sorted(
            enumerate(events),
            key=lambda item: float(item[1].get("return", 0.0) or 0.0),
            reverse=True,
        )[:n]
    }
    return [event for idx, event in enumerate(events) if idx not in top_positions]


def summarize_remove_top_n(
    events: list[dict[str, Any]],
    *,
    initial_capital: float,
    data_end: pd.Timestamp,
    max_n: int = 3,
) -> dict[str, Any]:
    return {
        f"drop_top{n}": compact_summary(run_summary(remove_top_n(events, n), initial_capital, data_end))
        for n in range(1, max_n + 1)
    }


def event_key(event: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(event.get("event_type") or ""),
        int(event.get("entry_idx", 0) or 0),
        int(event.get("exit_idx", 0) or 0),
        str(event.get("entry_time") or ""),
    )


def boosted_key_set(events: list[dict[str, Any]]) -> set[tuple[str, int, int, str]]:
    return {event_key(event) for event in events if isinstance(event.get("kelly_bucket_sizing"), dict)}


def apply_extra_roundtrip_cost(
    events: list[dict[str, Any]],
    boosted_keys: set[tuple[str, int, int, str]],
    extra_bps: float,
) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    for event in events:
        updated = dict(event)
        if event_key(event) in boosted_keys:
            sizing = event.get("kelly_bucket_sizing") if isinstance(event.get("kelly_bucket_sizing"), dict) else {}
            leverage = float(sizing.get("target_effective_leverage") or event.get("source_effective_leverage") or 0.0)
            penalty = leverage * float(extra_bps) / 10000.0
            updated["return"] = float(updated.get("return", 0.0) or 0.0) - penalty
            updated["return_pct"] = round(float(updated["return"]) * 100.0, 4)
            updated["extra_roundtrip_cost_bps"] = float(extra_bps)
        adjusted.append(updated)
    return adjusted


def cost_sensitivity(
    events: list[dict[str, Any]],
    boosted_keys: set[tuple[str, int, int, str]],
    extra_bps_values: list[float],
    *,
    initial_capital: float,
    data_end: pd.Timestamp,
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for extra_bps in extra_bps_values:
        adjusted = apply_extra_roundtrip_cost(events, boosted_keys, extra_bps)
        summary = run_summary(adjusted, initial_capital, data_end)
        rows.append(
            {
                "extra_roundtrip_bps_on_boosted_trades": float(extra_bps),
                "summary": compact_summary(summary),
                "delta_vs_baseline": delta_vs_baseline(summary, baseline),
            }
        )
    return rows


def delta_vs_baseline(summary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    current = (summary.get("windows") or {}).get("current_year", {})
    base_current = (baseline.get("windows") or {}).get("current_year", {})
    return {
        "total_return_pct": round(float(summary.get("total_return_pct", 0.0) or 0.0) - float(baseline.get("total_return_pct", 0.0) or 0.0), 2),
        "max_drawdown_pct": round(float(summary.get("max_drawdown_pct", 0.0) or 0.0) - float(baseline.get("max_drawdown_pct", 0.0) or 0.0), 2),
        "current_year_return_pct": round(float(current.get("total_return_pct", 0.0) or 0.0) - float(base_current.get("total_return_pct", 0.0) or 0.0), 2),
        "current_year_max_drawdown_pct": round(float(current.get("max_drawdown_pct", 0.0) or 0.0) - float(base_current.get("max_drawdown_pct", 0.0) or 0.0), 2),
    }


def boosted_trade_row(candidate_name: str, target: float, event: dict[str, Any], baseline_by_key: dict[tuple[str, int, int, str], dict[str, Any]]) -> dict[str, Any]:
    sizing = event.get("kelly_bucket_sizing") if isinstance(event.get("kelly_bucket_sizing"), dict) else {}
    baseline_event = baseline_by_key.get(event_key(event), {})
    source_return_pct = round(float(baseline_event.get("return", 0.0) or 0.0) * 100.0, 4)
    target_return_pct = round(float(event.get("return", 0.0) or 0.0) * 100.0, 4)
    return {
        "candidate": candidate_name,
        "target_effective_leverage": float(target),
        "entry_time": event.get("entry_time"),
        "exit_time": event.get("exit_time"),
        "event_type": event.get("event_type"),
        "direction": event.get("direction"),
        "exit_reason": event.get("exit_reason"),
        "source_effective_leverage": sizing.get("source_effective_leverage"),
        "target_effective_leverage_applied": sizing.get("target_effective_leverage"),
        "source_return_pct": source_return_pct,
        "target_return_pct": target_return_pct,
        "incremental_return_pct": round(target_return_pct - source_return_pct, 4),
        "net_score": event.get("net_score"),
        "bull_total": event.get("bull_total"),
        "bear_total": event.get("bear_total"),
        "regime_label": event.get("regime_label"),
        "feature_recent_fvg_near_entry": event.get("feature_recent_fvg_near_entry"),
    }


def find_candidate(name: str, all_candidates: list[Candidate]) -> Candidate:
    for candidate in all_candidates:
        if candidate.name == name:
            return candidate
    available = ", ".join(candidate.name for candidate in all_candidates)
    raise ValueError(f"Unknown candidate {name!r}; available: {available}")


def write_trades_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    events, data_end, payload = load_events(input_path)
    initial_capital = float(args.initial_capital)
    extra_bps_values = parse_float_list(args.extra_roundtrip_bps)
    baseline = run_summary(events, initial_capital, data_end)
    baseline_by_key = {event_key(event): event for event in events}
    available_candidates = candidates()
    trade_rows: list[dict[str, Any]] = []
    audited: list[dict[str, Any]] = []

    for name, target in parse_candidate_specs(args.candidate):
        candidate = find_candidate(name, available_candidates)
        adjusted, diagnostics = apply_target_leverage(events, candidate, target)
        summary = add_combo_deltas(run_summary(adjusted, initial_capital, data_end), baseline)
        boosted = [event for event in adjusted if isinstance(event.get("kelly_bucket_sizing"), dict)]
        boosted_keys = boosted_key_set(adjusted)
        candidate_trade_rows = [boosted_trade_row(name, target, event, baseline_by_key) for event in boosted]
        trade_rows.extend(candidate_trade_rows)
        boosted_events_by_year = yearly_summary(boosted, initial_capital) if boosted else {}
        audited.append(
            {
                "candidate": name,
                "target_effective_leverage": float(target),
                "diagnostics": diagnostics,
                "summary": compact_summary(summary),
                "delta_vs_baseline": delta_vs_baseline(summary, baseline),
                "yearly_all_events": yearly_summary(adjusted, initial_capital),
                "yearly_boosted_events": boosted_events_by_year,
                "remove_top_n_all_events": summarize_remove_top_n(
                    adjusted,
                    initial_capital=initial_capital,
                    data_end=data_end,
                    max_n=3,
                ),
                "remove_top_n_delta_vs_baseline_drop_top_n": {
                    f"drop_top{n}": delta_vs_baseline(
                        run_summary(remove_top_n(adjusted, n), initial_capital, data_end),
                        run_summary(remove_top_n(events, n), initial_capital, data_end),
                    )
                    for n in range(1, 4)
                },
                "cost_sensitivity": cost_sensitivity(
                    adjusted,
                    boosted_keys,
                    extra_bps_values,
                    initial_capital=initial_capital,
                    data_end=data_end,
                    baseline=baseline,
                ),
                "boosted_trades": candidate_trade_rows,
            }
        )

    report = {
        "metadata": {
            "input": str(input_path.resolve()),
            "source_metadata": payload.get("metadata", {}),
            "data_end": str(data_end),
            "initial_capital": initial_capital,
            "extra_roundtrip_bps_values": extra_bps_values,
            "note": "Extra cost sensitivity subtracts additional roundtrip bps only from boosted trades, scaled by target effective leverage.",
        },
        "baseline": compact_summary(baseline),
        "baseline_yearly": yearly_summary(events, initial_capital),
        "baseline_remove_top_n": summarize_remove_top_n(events, initial_capital=initial_capital, data_end=data_end, max_n=3),
        "audits": audited,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_trades_csv(Path(args.trades_csv), trade_rows)

    print(output_path)
    print(Path(args.trades_csv))
    base_year = report["baseline"]["current_year"]["total_return_pct"]
    print(
        f"baseline={report['baseline']['total_return_pct']:.2f}% "
        f"dd={report['baseline']['max_drawdown_pct']:.2f}% 2026={base_year:.2f}%"
    )
    for row in audited:
        summary = row["summary"]
        delta = row["delta_vs_baseline"]
        print(
            f"{row['candidate']} target={row['target_effective_leverage']:.2f}x "
            f"full={summary['total_return_pct']:.2f}% dd={summary['max_drawdown_pct']:.2f}% "
            f"2026={summary['current_year']['total_return_pct']:.2f}% "
            f"delta={delta['total_return_pct']:+.2f}% dd_delta={delta['max_drawdown_pct']:+.2f}% "
            f"boosted={row['diagnostics']['boosted_trades']}"
        )


if __name__ == "__main__":
    main()
