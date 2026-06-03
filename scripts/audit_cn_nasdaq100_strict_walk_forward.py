#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cn_nasdaq100_strict_utils import load_config, load_strict_frame, run_strict_path, summarize_path  # noqa: E402


OUTPUT = ROOT / "var" / "reports" / "cn_nasdaq100_strict_walk_forward.json"


def make_base_config() -> dict[str, Any]:
    config = load_config(ROOT / "config" / "config.paper.cn-nasdaq100-etf.json")
    config["conditional_leverage_enabled"] = False
    config["conditional_leverage_value"] = 1.0
    config["tiered_leverage_enabled"] = True
    config["tiered_leverage_rules"] = [
        {
            "vix_label": "vix_normal",
            "rel_strength_label": "qqq_strong",
            "leverage": 2.0,
        },
        {
            "vix_label": "vix_normal",
            "rel_strength_label": "qqq_neutral",
            "leverage": 1.5,
        },
    ]
    return config


def split_summary(path: pd.DataFrame, start: str, end: str, initial_capital: float) -> dict[str, Any]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    segment = path[(path["date"] >= start_ts) & (path["date"] <= end_ts)].reset_index(drop=True)
    if segment.empty:
        return {
            "days": 0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "trades": 0,
            "invested_days": 0,
            "yearly_returns_pct": {},
            "latest_position": "CASH",
        }
    segment = segment.copy()
    rebased = [float(initial_capital)]
    prev_capital = float(segment.iloc[0]["capital"]) / max(1e-12, 1.0 + float(segment.iloc[0]["daily_return"]))
    for _, row in segment.iterrows():
        prev_capital = prev_capital * (1.0 + float(row["daily_return"]))
        rebased.append(prev_capital)
    segment["capital"] = rebased[1:]
    summary = summarize_path(segment, initial_capital=float(initial_capital))
    summary["days"] = int(len(segment))
    return summary


def main() -> None:
    config = make_base_config()
    candidate_grid = []
    for fast in [20, 21, 22]:
        for slow in [190, 200, 210]:
            for hold in [90, 120, 135]:
                for tlb in [3, 4, 5]:
                    for tdd in [4.0, 5.0]:
                        candidate_grid.append(
                            {
                                "entry_fast_window": fast,
                                "entry_slow_window": slow,
                                "regime_filter": "ixic_filter",
                                "max_hold_days": hold,
                                "trailing_lookback_days": tlb,
                                "trailing_drawdown_pct": tdd,
                            }
                        )

    folds = [
        {"name": "train_2022_test_2023", "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31", "test_year": "2023"},
        {"name": "train_2022_2023_test_2024", "train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31", "test_year": "2024"},
        {"name": "train_2022_2024_test_2025", "train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-12-31", "test_year": "2025"},
        {"name": "train_2022_2025_test_2026", "train_end": "2025-12-31", "test_start": "2026-01-01", "test_end": "2026-12-31", "test_year": "2026"},
    ]

    fold_results: list[dict[str, Any]] = []
    for fold in folds:
        train_end = pd.Timestamp(fold["train_end"])
        scored: list[dict[str, Any]] = []
        for candidate in candidate_grid:
            local = dict(config)
            local.update(candidate)
            frame = load_strict_frame(local)
            path = run_strict_path(frame, local)
            train_path = path[path["date"] <= train_end].reset_index(drop=True)
            train_summary = summarize_path(train_path, initial_capital=float(local.get("initial_capital", 1000.0)))
            yearly = train_summary.get("yearly_returns_pct", {})
            score = round(
                float(train_summary.get("total_return_pct", 0.0))
                - 1.2 * float(train_summary.get("max_drawdown_pct", 0.0))
                + 0.8 * float(yearly.get("2025", 0.0))
                + 0.8 * float(yearly.get("2026", 0.0)),
                4,
            )
            scored.append(
                {
                    "candidate": candidate,
                    "train_summary": train_summary,
                    "score": score,
                }
            )
        best = sorted(
            scored,
            key=lambda item: (
                float(item["score"]),
                float(item["train_summary"].get("total_return_pct", 0.0)),
                -float(item["train_summary"].get("max_drawdown_pct", 0.0)),
            ),
            reverse=True,
        )[0]
        chosen = dict(config)
        chosen.update(best["candidate"])
        full_frame = load_strict_frame(chosen)
        full_path = run_strict_path(full_frame, chosen)
        test_summary = split_summary(full_path, fold["test_start"], fold["test_end"], float(chosen.get("initial_capital", 1000.0)))
        fold_results.append(
            {
                "fold": fold["name"],
                "train_end": fold["train_end"],
                "test_year": fold["test_year"],
                "best_train_candidate": best,
                "test_summary": test_summary,
                "test_year_return_pct": float(test_summary.get("yearly_returns_pct", {}).get(fold["test_year"], 0.0)),
            }
        )

    payload = {
        "candidate_family": "strict_cn_nasdaq100_tiered",
        "fixed_leverage_rules": config["tiered_leverage_rules"],
        "folds": fold_results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(OUTPUT)
    for fold in fold_results:
        best = fold["best_train_candidate"]
        candidate = best["candidate"]
        print(
            fold["fold"],
            candidate["entry_fast_window"],
            candidate["entry_slow_window"],
            candidate["max_hold_days"],
            candidate["trailing_lookback_days"],
            candidate["trailing_drawdown_pct"],
            best["train_summary"]["total_return_pct"],
            fold["test_year_return_pct"],
        )


if __name__ == "__main__":
    main()
