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

from scripts.audit_tqqq_cash_regime_context import build_regime_frame, load_df  # noqa: E402
from scripts.scan_tqqq_cash_exit_profiles import parse_float_list, parse_int_list, run_exit_candidate  # noqa: E402

DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "tqqq_cash_walk_forward_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward audit for the TQQQ/CASH regime-aware candidate family.")
    parser.add_argument("--tqqq", default=str(DEFAULT_PUBLIC_DIR / "TQQQ-1d.feather"))
    parser.add_argument("--qqq", default=str(DEFAULT_PUBLIC_DIR / "QQQ-1d.feather"))
    parser.add_argument("--spy", default=str(DEFAULT_PUBLIC_DIR / "SPY-1d.feather"))
    parser.add_argument("--ixic", default=str(DEFAULT_PUBLIC_DIR / "^IXIC-1d.feather"))
    parser.add_argument("--vix", default=str(DEFAULT_PUBLIC_DIR / "^VIX-1d.feather"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--entry-fast-window", type=int, default=25)
    parser.add_argument("--entry-slow-window", type=int, default=150)
    parser.add_argument("--switch-cost-bps", type=float, default=5.0)
    parser.add_argument("--max-hold-days-values", default="0,40,60,90,120")
    parser.add_argument("--trailing-lookback-days-values", default="0,10,20,30")
    parser.add_argument("--trailing-drawdown-pct-values", default="0,5,8,10,12,15")
    parser.add_argument("--hold-mode", choices=["hard_exit", "timer_refresh"], default="hard_exit")
    return parser.parse_args()


def annual_return_for_year(summary: dict[str, Any], year: str) -> float:
    return float((summary.get("yearly_returns_pct") or {}).get(year, 0.0) or 0.0)


def build_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    vix_allow = frame["vix_label"].isin(["vix_low", "vix_normal"])
    ixic_allow = frame["ixic_trend_label"].eq("ixic_up")
    rel_allow = frame["rel_strength_label"].ne("qqq_weak")
    return {
        "base": pd.Series([True] * len(frame)),
        "vix_filter": vix_allow,
        "ixic_filter": ixic_allow,
        "rel_filter": rel_allow,
        "vix_ixic": vix_allow & ixic_allow,
        "vix_rel": vix_allow & rel_allow,
        "ixic_rel": ixic_allow & rel_allow,
        "all_three": vix_allow & ixic_allow & rel_allow,
    }


def score_summary(summary: dict[str, Any]) -> float:
    return float(summary.get("total_return_pct", 0.0)) - 2.0 * float(summary.get("max_drawdown_pct", 0.0))


def scan_candidates(frame: pd.DataFrame, masks: dict[str, pd.Series], args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for mask_name, allow_mask in masks.items():
        for max_hold_days in parse_int_list(args.max_hold_days_values):
            for trailing_lookback_days in parse_int_list(args.trailing_lookback_days_values):
                for trailing_drawdown_pct in parse_float_list(args.trailing_drawdown_pct_values):
                    if trailing_lookback_days == 0 and trailing_drawdown_pct > 0:
                        continue
                    if trailing_lookback_days > 0 and trailing_drawdown_pct == 0:
                        continue
                    result = run_exit_candidate(
                        frame,
                        initial_capital=float(args.initial_capital),
                        switch_cost_bps=float(args.switch_cost_bps),
                        max_hold_days=int(max_hold_days),
                        trailing_lookback_days=int(trailing_lookback_days),
                        trailing_drawdown_pct=float(trailing_drawdown_pct),
                        allow_mask=allow_mask,
                        hold_mode=str(args.hold_mode),
                    )
                    candidates.append(
                        {
                            "mask": mask_name,
                            "params": {
                                "max_hold_days": int(max_hold_days),
                                "trailing_lookback_days": int(trailing_lookback_days),
                                "trailing_drawdown_pct": float(trailing_drawdown_pct),
                            },
                            "summary": result,
                            "score": round(score_summary(result), 4),
                        }
                    )
    return candidates


def pick_best(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        candidates,
        key=lambda item: (
            float(item["score"]),
            float(item["summary"].get("max_drawdown_pct", 0.0)),
            float(item["summary"].get("total_return_pct", 0.0)),
        ),
        reverse=True,
    )[0]


def main() -> None:
    args = parse_args()
    qqq = load_df(Path(args.qqq))
    tqqq = load_df(Path(args.tqqq))
    spy = load_df(Path(args.spy))
    ixic = load_df(Path(args.ixic))
    vix = load_df(Path(args.vix))

    frame = build_regime_frame(qqq, tqqq, spy, ixic, vix, int(args.entry_fast_window), int(args.entry_slow_window))
    masks = build_masks(frame)

    folds = [
        {"name": "train_2022_test_2023", "train_end": "2022-12-31", "test_end": "2023-12-31", "test_year": "2023"},
        {"name": "train_2022_2023_test_2024", "train_end": "2023-12-31", "test_end": "2024-12-31", "test_year": "2024"},
        {"name": "train_2022_2024_test_2025", "train_end": "2024-12-31", "test_end": "2025-12-31", "test_year": "2025"},
        {"name": "train_2022_2025_test_2026", "train_end": "2025-12-31", "test_end": "2026-12-31", "test_year": "2026"},
    ]

    fold_results: list[dict[str, Any]] = []
    for fold in folds:
        train_end = pd.Timestamp(fold["train_end"], tz="UTC")
        test_end = pd.Timestamp(fold["test_end"], tz="UTC")
        train_mask = frame["date"] <= train_end
        test_mask = frame["date"] <= test_end
        train_frame = frame.loc[train_mask].reset_index(drop=True)
        test_frame = frame.loc[test_mask].reset_index(drop=True)
        train_candidates = scan_candidates(
            train_frame,
            {k: v.loc[train_mask].reset_index(drop=True) for k, v in masks.items()},
            args,
        )
        best = pick_best(train_candidates)
        test_result = run_exit_candidate(
            test_frame,
            initial_capital=float(args.initial_capital),
            switch_cost_bps=float(args.switch_cost_bps),
            max_hold_days=int(best["params"]["max_hold_days"]),
            trailing_lookback_days=int(best["params"]["trailing_lookback_days"]),
            trailing_drawdown_pct=float(best["params"]["trailing_drawdown_pct"]),
            allow_mask=masks[best["mask"]].loc[test_mask].reset_index(drop=True),
            hold_mode=str(args.hold_mode),
        )
        fold_results.append(
            {
                "fold": fold["name"],
                "train_end": fold["train_end"],
                "test_year": fold["test_year"],
                "best_train_candidate": best,
                "test_summary": test_result,
                "test_year_return_pct": annual_return_for_year(test_result, fold["test_year"]),
            }
        )

    payload = {
        "entry_reference": {
            "fast_window": int(args.entry_fast_window),
            "slow_window": int(args.entry_slow_window),
            "switch_cost_bps": float(args.switch_cost_bps),
        },
        "folds": fold_results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    for fold in fold_results:
        best = fold["best_train_candidate"]
        print(
            fold["fold"],
            best["mask"],
            best["params"],
            best["summary"]["total_return_pct"],
            best["summary"]["max_drawdown_pct"],
            fold["test_year_return_pct"],
        )


if __name__ == "__main__":
    main()
