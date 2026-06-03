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

from scripts.scan_tqqq_context_bucket_overlays import prepare_frame, run_candidate  # noqa: E402
from scripts.audit_tqqq_cash_regime_context import load_df  # noqa: E402


DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "tqqq_context_bucket_overlay_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a BTC-style context/bucket TQQQ+SQQQ overlay candidate.")
    parser.add_argument("--tqqq", default=str(DEFAULT_PUBLIC_DIR / "TQQQ-1d.feather"))
    parser.add_argument("--sqqq", default=str(DEFAULT_PUBLIC_DIR / "SQQQ-1d.feather"))
    parser.add_argument("--qqq", default=str(DEFAULT_PUBLIC_DIR / "QQQ-1d.feather"))
    parser.add_argument("--spy", default=str(DEFAULT_PUBLIC_DIR / "SPY-1d.feather"))
    parser.add_argument("--ixic", default=str(DEFAULT_PUBLIC_DIR / "^IXIC-1d.feather"))
    parser.add_argument("--vix", default=str(DEFAULT_PUBLIC_DIR / "^VIX-1d.feather"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--entry-fast-window", type=int, default=25)
    parser.add_argument("--entry-slow-window", type=int, default=200)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--switch-cost-bps", type=float, default=10.0)
    parser.add_argument("--long-profile-name", default="stable_base")
    parser.add_argument("--short-rule-name", default="bearish_score5")
    parser.add_argument("--short-max-hold-days", type=int, default=20)
    parser.add_argument("--short-trailing-lookback-days", type=int, default=5)
    parser.add_argument("--short-trailing-drawdown-pct", type=float, default=6.0)
    parser.add_argument("--extra-cost-bps-values", default="0,2.5,5,10")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def trade_side_breakdown(trades: list[dict[str, Any]]) -> dict[str, Any]:
    tqqq = [item for item in trades if item.get("asset") == "TQQQ"]
    sqqq = [item for item in trades if item.get("asset") == "SQQQ"]
    return {
        "tqqq_trades": int(len(tqqq)),
        "sqqq_trades": int(len(sqqq)),
        "sqqq_trade_returns_pct": [float(item.get("trade_return_pct", 0.0) or 0.0) for item in sqqq],
        "sqqq_trade_windows": [
            {
                "entry_date": item.get("entry_date"),
                "exit_date": item.get("exit_date"),
                "trade_return_pct": float(item.get("trade_return_pct", 0.0) or 0.0),
                "exit_reason": item.get("exit_reason"),
            }
            for item in sqqq
        ],
    }


def leave_one_out(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not trades:
        return []
    returns = pd.Series([float(item.get("trade_return_pct", 0.0) or 0.0) for item in trades], dtype=float)
    full_growth = float((1.0 + returns / 100.0).prod())
    output: list[dict[str, Any]] = []
    for idx, trade in enumerate(trades):
        remaining = returns.drop(returns.index[idx])
        remaining_growth = float((1.0 + remaining / 100.0).prod()) if not remaining.empty else 1.0
        output.append(
            {
                "removed_index": int(idx),
                "asset": trade.get("asset"),
                "entry_date": trade.get("entry_date"),
                "exit_date": trade.get("exit_date"),
                "trade_return_pct": float(trade.get("trade_return_pct", 0.0) or 0.0),
                "remaining_total_return_pct": round((remaining_growth - 1.0) * 100.0, 2),
                "relative_drop_pct": round((full_growth - remaining_growth) / full_growth * 100.0, 2) if full_growth > 0 else 0.0,
            }
        )
    return output


def per_year_trade_breakdown(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {}
    frame = pd.DataFrame(trades)
    frame["entry_year"] = pd.to_datetime(frame["entry_date"]).dt.year.astype(str)
    output: dict[str, Any] = {}
    for year, group in frame.groupby("entry_year"):
        output[str(year)] = {
            "trades": int(len(group)),
            "tqqq_trades": int((group["asset"] == "TQQQ").sum()),
            "sqqq_trades": int((group["asset"] == "SQQQ").sum()),
            "mean_trade_return_pct": round(float(group["trade_return_pct"].mean()), 2),
            "sum_trade_return_pct": round(float(group["trade_return_pct"].sum()), 2),
        }
    return output


def main() -> None:
    args = parse_args()
    qqq = load_df(Path(args.qqq))
    tqqq = load_df(Path(args.tqqq))
    sqqq = load_df(Path(args.sqqq))
    spy = load_df(Path(args.spy))
    ixic = load_df(Path(args.ixic))
    vix = load_df(Path(args.vix))
    frame = prepare_frame(
        qqq,
        tqqq,
        sqqq,
        spy,
        ixic,
        vix,
        int(args.entry_fast_window),
        int(args.entry_slow_window),
    )

    base_result = run_candidate(
        frame,
        long_profile_name=str(args.long_profile_name),
        short_rule_name=str(args.short_rule_name),
        short_exit_profile=(
            int(args.short_max_hold_days),
            int(args.short_trailing_lookback_days),
            float(args.short_trailing_drawdown_pct),
        ),
        initial_capital=float(args.initial_capital),
        switch_cost_bps=float(args.switch_cost_bps),
    )

    cost_sensitivity: list[dict[str, Any]] = []
    for extra_cost_bps in parse_float_list(args.extra_cost_bps_values):
        result = run_candidate(
            frame,
            long_profile_name=str(args.long_profile_name),
            short_rule_name=str(args.short_rule_name),
            short_exit_profile=(
                int(args.short_max_hold_days),
                int(args.short_trailing_lookback_days),
                float(args.short_trailing_drawdown_pct),
            ),
            initial_capital=float(args.initial_capital),
            switch_cost_bps=float(args.switch_cost_bps) + float(extra_cost_bps),
        )
        cost_sensitivity.append(
            {
                "switch_cost_bps": round(float(args.switch_cost_bps) + float(extra_cost_bps), 2),
                "extra_cost_bps": float(extra_cost_bps),
                "summary": result["summary"],
            }
        )

    payload = {
        "candidate": base_result["candidate"],
        "summary": base_result["summary"],
        "trade_side_breakdown": trade_side_breakdown(base_result["trades"]),
        "per_year_trade_breakdown": per_year_trade_breakdown(base_result["trades"]),
        "leave_one_out": leave_one_out(base_result["trades"]),
        "cost_sensitivity": cost_sensitivity,
        "sample_trades": base_result["sample_trades"],
        "trades": base_result["trades"],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
