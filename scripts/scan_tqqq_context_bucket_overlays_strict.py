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

from scripts.tqqq_sqqq_strategy_utils import load_strategy_frame, run_strategy_path  # noqa: E402


DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "tqqq_context_bucket_overlay_scan_strict.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict-execution scan for TQQQ/SQQQ context bucket overlays.")
    parser.add_argument("--data-root", default=str(DEFAULT_PUBLIC_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--entry-fast-window", type=int, default=25)
    parser.add_argument("--entry-slow-window", type=int, default=200)
    parser.add_argument("--switch-cost-bps", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def annual_returns(equity: pd.DataFrame) -> dict[str, float]:
    if equity.empty:
        return {}
    frame = equity.copy()
    frame["year"] = frame["date"].dt.year.astype(str)
    out: dict[str, float] = {}
    for year, group in frame.groupby("year"):
        start = float(group.iloc[0]["equity"])
        end = float(group.iloc[-1]["equity"])
        out[year] = round((end / start - 1.0) * 100.0, 2) if start > 0 else 0.0
    return out


def summarize_path(path: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    if path.empty:
        return {
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "yearly_returns_pct": {},
            "trades": 0,
            "tqqq_trades": 0,
            "sqqq_trades": 0,
            "invested_days": 0,
            "score": 0.0,
        }

    equity = path[["date", "capital"]].rename(columns={"capital": "equity"}).copy()
    yearly = annual_returns(equity)
    peak = path["capital"].cummax()
    max_dd = float(((peak - path["capital"]) / peak.replace(0, pd.NA) * 100.0).max(skipna=True) or 0.0)
    entry_rows = path[(path["position"] != path["position"].shift(1)) & (path["position"] != "CASH")].copy()
    trades = int(len(entry_rows))
    tqqq_trades = int((entry_rows["position"] == "TQQQ").sum())
    sqqq_trades = int((entry_rows["position"] == "SQQQ").sum())
    total_return_pct = round((float(path.iloc[-1]["capital"]) / float(initial_capital) - 1.0) * 100.0, 2)
    score = round(total_return_pct - 2.0 * max_dd + trades * 30.0 + float(yearly.get("2026", 0.0)) * 0.8, 4)
    return {
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": round(max_dd, 2),
        "yearly_returns_pct": yearly,
        "trades": trades,
        "tqqq_trades": tqqq_trades,
        "sqqq_trades": sqqq_trades,
        "invested_days": int((path["position"] != "CASH").sum()),
        "score": score,
        "latest_position": str(path.iloc[-1]["position"]),
    }


def candidate_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    summary = item["summary"]
    yearly = summary.get("yearly_returns_pct", {})
    return (
        float(summary.get("score", 0.0)),
        float(summary.get("total_return_pct", 0.0)),
        float(yearly.get("2026", 0.0)),
        -float(summary.get("max_drawdown_pct", 0.0)),
    )


def run_candidate(
    frame: pd.DataFrame,
    *,
    long_profile_name: str,
    short_rule_name: str,
    short_exit_profile: tuple[int, int, float],
    initial_capital: float,
    switch_cost_bps: float,
) -> dict[str, Any]:
    path = run_strategy_path(
        frame,
        long_profile_name=long_profile_name,
        short_rule_name=short_rule_name,
        short_exit_profile=short_exit_profile,
        initial_capital=initial_capital,
        switch_cost_bps=switch_cost_bps,
    )
    return {
        "candidate": {
            "long_profile_name": long_profile_name,
            "short_rule_name": short_rule_name,
            "short_exit_profile": {
                "max_hold_days": int(short_exit_profile[0]),
                "trailing_lookback_days": int(short_exit_profile[1]),
                "trailing_drawdown_pct": float(short_exit_profile[2]),
            },
        },
        "summary": summarize_path(path, initial_capital),
    }


def main() -> None:
    args = parse_args()
    frame = load_strategy_frame(
        data_root=Path(args.data_root),
        entry_fast_window=int(args.entry_fast_window),
        entry_slow_window=int(args.entry_slow_window),
    )

    long_profiles = [
        "stable_base",
        "weak_tight_score2",
        "weak_tight_score3",
        "three_bucket_score",
        "breakout_loose_else_mid",
    ]
    short_rules = [
        "off",
        "narrow_base",
        "narrow_score4",
        "narrow_score5",
        "bearish_score5",
        "breakdown_confirm",
    ]
    short_exit_profiles = [
        (20, 5, 6.0),
        (20, 5, 8.0),
        (20, 10, 8.0),
        (40, 10, 8.0),
    ]

    results: list[dict[str, Any]] = []
    for long_profile_name in long_profiles:
        for short_rule_name in short_rules:
            for short_exit_profile in short_exit_profiles:
                results.append(
                    run_candidate(
                        frame,
                        long_profile_name=long_profile_name,
                        short_rule_name=short_rule_name,
                        short_exit_profile=short_exit_profile,
                        initial_capital=float(args.initial_capital),
                        switch_cost_bps=float(args.switch_cost_bps),
                    )
                )

    ranked = sorted(results, key=candidate_sort_key, reverse=True)
    payload = {
        "reference": {
            "execution_mode": "strict_t_plus_1_open",
            "entry_fast_window": int(args.entry_fast_window),
            "entry_slow_window": int(args.entry_slow_window),
            "switch_cost_bps": float(args.switch_cost_bps),
            "description": "Strict execution: t close signal, t+1 open execution, no same-day flip capture.",
        },
        "scan_size": len(results),
        "top_candidates": ranked[: int(args.top)],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    for item in ranked[: min(int(args.top), 12)]:
        summary = item["summary"]
        candidate = item["candidate"]
        yearly = summary["yearly_returns_pct"]
        print(
            f"score={summary['score']:.2f} full={summary['total_return_pct']:.2f}% dd={summary['max_drawdown_pct']:.2f}% "
            f"trades={summary['trades']} tqqq={summary['tqqq_trades']} sqqq={summary['sqqq_trades']} "
            f"2026={float(yearly.get('2026', 0.0)):.2f}% long={candidate['long_profile_name']} "
            f"short={candidate['short_rule_name']} s_exit={candidate['short_exit_profile']}"
        )


if __name__ == "__main__":
    main()
