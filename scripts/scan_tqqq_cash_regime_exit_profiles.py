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
from scripts.scan_tqqq_cash_exit_profiles import build_entry_frame, parse_float_list, parse_int_list, run_exit_candidate  # noqa: E402

DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "tqqq_cash_regime_exit_profile_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan exit profiles under regime masks on top of TQQQ/CASH.")
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
    parser.add_argument("--top", type=int, default=10)
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


def summarize_candidates(items: list[dict[str, Any]], top_n: int) -> dict[str, Any]:
    ranked = sorted(
        items,
        key=lambda item: (
            float(item.get("score", 0.0)),
            float(item.get("yearly_returns_pct", {}).get("2022", 0.0)),
            float(item.get("yearly_returns_pct", {}).get("2026", 0.0)),
            -float(item.get("max_drawdown_pct", 0.0)),
        ),
        reverse=True,
    )
    return {
        "scan_size": len(items),
        "top_candidates": ranked[:top_n],
    }


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


def main() -> None:
    args = parse_args()
    qqq = load_df(Path(args.qqq))
    tqqq = load_df(Path(args.tqqq))
    spy = load_df(Path(args.spy))
    ixic = load_df(Path(args.ixic))
    vix = load_df(Path(args.vix))

    frame = build_regime_frame(qqq, tqqq, spy, ixic, vix, int(args.entry_fast_window), int(args.entry_slow_window))
    masks = build_masks(frame)

    results: dict[str, Any] = {}
    for mask_name, allow_mask in masks.items():
        candidates: list[dict[str, Any]] = []
        for max_hold_days in parse_int_list(args.max_hold_days_values):
            for trailing_lookback_days in parse_int_list(args.trailing_lookback_days_values):
                for trailing_drawdown_pct in parse_float_list(args.trailing_drawdown_pct_values):
                    if trailing_lookback_days == 0 and trailing_drawdown_pct > 0:
                        continue
                    if trailing_lookback_days > 0 and trailing_drawdown_pct == 0:
                        continue
                    item = run_exit_candidate(
                        frame,
                        initial_capital=float(args.initial_capital),
                        switch_cost_bps=float(args.switch_cost_bps),
                        max_hold_days=int(max_hold_days),
                        trailing_lookback_days=int(trailing_lookback_days),
                        trailing_drawdown_pct=float(trailing_drawdown_pct),
                        allow_mask=allow_mask,
                        hold_mode=str(args.hold_mode),
                    )
                    item["mask_name"] = mask_name
                    candidates.append(item)
        results[mask_name] = summarize_candidates(candidates, int(args.top))

    payload = {
        "entry_reference": {
            "fast_window": int(args.entry_fast_window),
            "slow_window": int(args.entry_slow_window),
            "description": "QQQ 25/150 MA -> TQQQ/CASH under regime masks",
            "switch_cost_bps": float(args.switch_cost_bps),
        },
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    for name, block in results.items():
        if not block["top_candidates"]:
            continue
        top = block["top_candidates"][0]
        print(name, top["total_return_pct"], top["max_drawdown_pct"], top["yearly_returns_pct"])


if __name__ == "__main__":
    main()
