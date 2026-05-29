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
from scripts.scan_tqqq_cash_exit_profiles import build_entry_frame  # noqa: E402
from scripts.replay_tqqq_sqqq_trend_baseline import max_drawdown_pct  # noqa: E402


DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "tqqq_cash_regime_filter_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan regime filters on top of the TQQQ/CASH main candidate.")
    parser.add_argument("--tqqq", default=str(DEFAULT_PUBLIC_DIR / "TQQQ-1d.feather"))
    parser.add_argument("--qqq", default=str(DEFAULT_PUBLIC_DIR / "QQQ-1d.feather"))
    parser.add_argument("--spy", default=str(DEFAULT_PUBLIC_DIR / "SPY-1d.feather"))
    parser.add_argument("--ixic", default=str(DEFAULT_PUBLIC_DIR / "^IXIC-1d.feather"))
    parser.add_argument("--vix", default=str(DEFAULT_PUBLIC_DIR / "^VIX-1d.feather"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--entry-fast-window", type=int, default=25)
    parser.add_argument("--entry-slow-window", type=int, default=150)
    parser.add_argument("--max-hold-days", type=int, default=90)
    parser.add_argument("--trailing-lookback-days", type=int, default=10)
    parser.add_argument("--trailing-drawdown-pct", type=float, default=12.0)
    parser.add_argument("--switch-cost-bps", type=float, default=5.0)
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


def run_main_candidate(frame: pd.DataFrame, initial_capital: float, switch_cost_bps: float, max_hold_days: int, trailing_lookback_days: int, trailing_drawdown_pct: float, allow_mask: pd.Series) -> pd.DataFrame:
    capital = initial_capital
    previous_position = "CASH"
    active = False
    hold_days = 0
    rolling_peak = 0.0
    exit_override = False
    rows: list[dict[str, Any]] = []

    for idx, row in frame.iterrows():
        desired_active = int(row["planned_position"]) > 0 and bool(allow_mask.iloc[idx])
        if not desired_active:
            active = False
            hold_days = 0
            rolling_peak = 0.0
            exit_override = False
        else:
            if not active and not exit_override:
                active = True
                hold_days = 0
                rolling_peak = float(row["tqqq_close"])
            elif exit_override:
                active = False

        position = "TQQQ" if active else "CASH"
        daily_ret = 0.0
        trailing_exit = False
        time_exit = False
        if idx > 0 and position == "TQQQ":
            prev_close = float(frame.iloc[idx - 1]["tqqq_close"])
            cur_close = float(row["tqqq_close"])
            daily_ret = cur_close / prev_close - 1.0 if prev_close > 0 else 0.0
            hold_days += 1
            rolling_peak = max(rolling_peak, cur_close)
            if trailing_lookback_days > 0 and trailing_drawdown_pct > 0 and hold_days >= trailing_lookback_days and rolling_peak > 0:
                drawdown_from_peak = (rolling_peak - cur_close) / rolling_peak * 100.0
                trailing_exit = drawdown_from_peak >= trailing_drawdown_pct
            if max_hold_days > 0 and hold_days >= max_hold_days:
                time_exit = True
            if trailing_exit or time_exit:
                active = False
                exit_override = True
                hold_days = 0
                rolling_peak = 0.0
        elif position == "CASH" and desired_active:
            exit_override = False

        if idx > 0 and position != previous_position:
            daily_ret -= float(switch_cost_bps) / 10000.0
        previous_position = position
        capital *= 1.0 + daily_ret
        rows.append(
            {
                "date": row["date"],
                "position": position,
                "daily_return": daily_ret,
                "capital": capital,
                "vix_label": row["vix_label"],
                "rel_strength_label": row["rel_strength_label"],
                "ixic_trend_label": row["ixic_trend_label"],
                "allow": bool(allow_mask.iloc[idx]),
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"days": 0, "invested_days": 0, "total_return_pct": 0.0, "max_drawdown_pct": 0.0, "annual_returns_pct": {}}
    equity = 1000.0 * (1.0 + df["daily_return"]).cumprod()
    return {
        "days": int(len(df)),
        "invested_days": int((df["position"] == "TQQQ").sum()),
        "total_return_pct": round((float(equity.iloc[-1]) / 1000.0 - 1.0) * 100.0, 2),
        "max_drawdown_pct": round(max_drawdown_pct(equity), 2),
        "annual_returns_pct": annual_returns(pd.DataFrame({"date": df["date"], "equity": equity})),
    }


def main() -> None:
    args = parse_args()
    qqq = load_df(Path(args.qqq))
    tqqq = load_df(Path(args.tqqq))
    spy = load_df(Path(args.spy))
    ixic = load_df(Path(args.ixic))
    vix = load_df(Path(args.vix))
    frame = build_regime_frame(qqq, tqqq, spy, ixic, vix, int(args.entry_fast_window), int(args.entry_slow_window))
    base = build_entry_frame(qqq, tqqq, int(args.entry_fast_window), int(args.entry_slow_window))

    vix_allow = frame["vix_label"].isin(["vix_low", "vix_normal"])
    ixic_allow = frame["ixic_trend_label"].eq("ixic_up")
    rel_allow = frame["rel_strength_label"].ne("qqq_weak")

    candidates = {
        "base": pd.Series([True] * len(frame)),
        "vix_filter": vix_allow,
        "ixic_filter": ixic_allow,
        "rel_filter": rel_allow,
        "vix_ixic": vix_allow & ixic_allow,
        "vix_rel": vix_allow & rel_allow,
        "ixic_rel": ixic_allow & rel_allow,
        "all_three": vix_allow & ixic_allow & rel_allow,
    }

    results: dict[str, Any] = {}
    for name, mask in candidates.items():
        out = run_main_candidate(
            frame,
            initial_capital=float(args.initial_capital),
            switch_cost_bps=float(args.switch_cost_bps),
            max_hold_days=int(args.max_hold_days),
            trailing_lookback_days=int(args.trailing_lookback_days),
            trailing_drawdown_pct=float(args.trailing_drawdown_pct),
            allow_mask=mask.reset_index(drop=True),
        )
        summary = summarize(out)
        summary["allow_days"] = int(mask.sum())
        summary["allow_ratio_pct"] = round(float(mask.mean() * 100.0), 2)
        results[name] = summary

    payload = {
        "entry_reference": {
            "fast_window": int(args.entry_fast_window),
            "slow_window": int(args.entry_slow_window),
            "max_hold_days": int(args.max_hold_days),
            "trailing_lookback_days": int(args.trailing_lookback_days),
            "trailing_drawdown_pct": float(args.trailing_drawdown_pct),
            "switch_cost_bps": float(args.switch_cost_bps),
        },
        "results": results,
        "baseline": {
            "total_return_pct": 399.97,
            "max_drawdown_pct": 35.67,
            "annual_returns_pct": {"2022": 0.0, "2023": 106.8, "2024": 96.94, "2025": 13.52, "2026": 15.45},
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    for name, stats in results.items():
        print(name, stats["total_return_pct"], stats["max_drawdown_pct"], stats["annual_returns_pct"])


if __name__ == "__main__":
    main()
