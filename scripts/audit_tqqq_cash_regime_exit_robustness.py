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
from scripts.scan_tqqq_cash_exit_profiles import build_entry_frame, parse_float_list, parse_int_list  # noqa: E402
from scripts.replay_tqqq_sqqq_trend_baseline import max_drawdown_pct  # noqa: E402

DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "tqqq_cash_regime_exit_robustness_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit robustness of the strongest regime-aware TQQQ/CASH candidate.")
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
    parser.add_argument("--max-hold-days", type=int, default=90)
    parser.add_argument("--trailing-lookback-days", type=int, default=10)
    parser.add_argument("--trailing-drawdown-pct", type=float, default=12.0)
    parser.add_argument("--mask", default="ixic_filter", choices=["base", "vix_filter", "ixic_filter", "rel_filter", "vix_ixic", "vix_rel", "ixic_rel", "all_three"])
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


def run_candidate(frame: pd.DataFrame, allow_mask: pd.Series, *, initial_capital: float, switch_cost_bps: float, max_hold_days: int, trailing_lookback_days: int, trailing_drawdown_pct: float) -> dict[str, Any]:
    allow_mask = allow_mask.reset_index(drop=True)
    capital = initial_capital
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    active = False
    hold_days = 0
    rolling_peak = 0.0
    exit_override = False
    entry_equity = capital
    entry_date = None
    entry_price = None
    previous_position = "CASH"

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
                entry_equity = capital
                entry_date = row["date"]
                entry_price = float(row["tqqq_close"])
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
                if entry_date is not None:
                    trades.append(
                        {
                            "entry_date": str(entry_date.date()),
                            "exit_date": str(row["date"].date()),
                            "entry_price": round(float(entry_price or 0.0), 4),
                            "exit_price": round(float(row["tqqq_close"]), 4),
                            "trade_return_pct": round((capital * (1.0 + daily_ret) / entry_equity - 1.0) * 100.0, 2),
                            "exit_reason": "trailing" if trailing_exit else "time",
                            "hold_days": int(hold_days),
                        }
                    )
                    entry_date = None
                    entry_price = None
        elif position == "CASH" and desired_active:
            exit_override = False

        if idx > 0 and position != previous_position:
            daily_ret -= float(switch_cost_bps) / 10000.0
        previous_position = position
        capital *= 1.0 + daily_ret
        rows.append({"date": row["date"], "equity": capital, "position": position, "daily_return": daily_ret})

    equity = pd.DataFrame(rows)
    total_return_pct = round((capital / initial_capital - 1.0) * 100.0, 2)
    max_dd = round(max_drawdown_pct(equity["equity"]), 2)
    yearly = annual_returns(equity)
    trade_returns = pd.Series([t["trade_return_pct"] for t in trades], dtype=float) if trades else pd.Series(dtype=float)
    leave_one_out_total_return_pct: list[dict[str, Any]] = []
    if not trade_returns.empty:
        full_growth = float((1.0 + trade_returns / 100.0).prod())
        for i, trade in enumerate(trades):
            remaining = trade_returns.drop(trade_returns.index[i])
            remaining_growth = float((1.0 + remaining / 100.0).prod()) if not remaining.empty else 1.0
            leave_one_out_total_return_pct.append(
                {
                    "removed_trade_index": int(i),
                    "removed_entry_date": trade.get("entry_date"),
                    "removed_exit_date": trade.get("exit_date"),
                    "remaining_total_return_pct": round((remaining_growth - 1.0) * 100.0, 2),
                    "relative_drop_pp": round((full_growth - remaining_growth) / full_growth * 100.0, 2) if full_growth > 0 else 0.0,
                }
            )
    return {
        "summary": {
            "days": int(len(equity)),
            "trades": int(len(trades)),
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd,
            "annual_returns_pct": yearly,
            "median_trade_return_pct": round(float(trade_returns.median()), 2) if not trade_returns.empty else 0.0,
            "mean_trade_return_pct": round(float(trade_returns.mean()), 2) if not trade_returns.empty else 0.0,
            "positive_trade_ratio_pct": round(float((trade_returns > 0).mean() * 100.0), 2) if not trade_returns.empty else 0.0,
            "best_trade_return_pct": round(float(trade_returns.max()), 2) if not trade_returns.empty else 0.0,
            "worst_trade_return_pct": round(float(trade_returns.min()), 2) if not trade_returns.empty else 0.0,
        },
        "leave_one_out": leave_one_out_total_return_pct,
        "trades": trades,
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
    result = run_candidate(
        frame,
        masks[args.mask],
        initial_capital=float(args.initial_capital),
        switch_cost_bps=float(args.switch_cost_bps),
        max_hold_days=int(args.max_hold_days),
        trailing_lookback_days=int(args.trailing_lookback_days),
        trailing_drawdown_pct=float(args.trailing_drawdown_pct),
    )
    payload = {
        "mask": args.mask,
        "entry_reference": {
            "fast_window": int(args.entry_fast_window),
            "slow_window": int(args.entry_slow_window),
            "max_hold_days": int(args.max_hold_days),
            "trailing_lookback_days": int(args.trailing_lookback_days),
            "trailing_drawdown_pct": float(args.trailing_drawdown_pct),
            "switch_cost_bps": float(args.switch_cost_bps),
        },
        "summary": result["summary"],
        "leave_one_out": result["leave_one_out"],
        "trades": result["trades"],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    print(result["summary"])


if __name__ == "__main__":
    main()
