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

from scripts.scan_tqqq_context_bucket_overlays import prepare_frame  # noqa: E402
from scripts.tqqq_cash_strict_utils import (  # noqa: E402
    build_allow_mask,
    de_risk_fraction,
    load_strict_frame,
    max_drawdown_pct,
)
from scripts.audit_tqqq_cash_regime_context import load_df  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "tqqq_recovery_reentry_overlay_scan.json"
DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan narrow recovery re-entry overlays for TQQQ strict baseline.")
    parser.add_argument("--data-root", default=str(DEFAULT_PUBLIC_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--entry-fast-window", type=int, default=25)
    parser.add_argument("--entry-slow-window", type=int, default=150)
    parser.add_argument("--regime-filter", default="ixic_filter")
    parser.add_argument("--max-hold-days", type=int, default=90)
    parser.add_argument("--trailing-lookback-days", type=int, default=10)
    parser.add_argument("--trailing-drawdown-pct", type=float, default=12.0)
    parser.add_argument("--de-risk-signal-name", default="breakout_fail_score_le3_flat")
    parser.add_argument("--switch-cost-bps", type=float, default=10.0)
    parser.add_argument("--cooldown-days-values", default="0,1,2,3")
    parser.add_argument(
        "--overlay-rules",
        default=(
            "off,"
            "score_ge3,"
            "score_ge4,"
            "score_ge5,"
            "score_ge3_breakout20,"
            "score_ge4_breakout20,"
            "score_ge5_breakout20,"
            "score_ge3_breakout_or_reclaim20,"
            "score_ge4_breakout_or_reclaim20,"
            "score_ge5_breakout_or_reclaim20,"
            "rel_strong_breakout20,"
            "rel_strong_breakout_or_reclaim20,"
            "vix_not_high_rel_strong_breakout_or_reclaim20,"
            "score_ge4_breakout_or_reclaim20_mom_pos"
        ),
    )
    parser.add_argument("--start-date", default="2023-01-03")
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [str(item.strip()) for item in str(value).split(",") if item.strip()]


def load_overlay_frame(*, data_root: Path, entry_fast_window: int, entry_slow_window: int) -> pd.DataFrame:
    frame = load_strict_frame(
        data_root=data_root,
        entry_fast_window=entry_fast_window,
        entry_slow_window=entry_slow_window,
    )
    qqq = load_df(data_root / "QQQ-1d.feather")
    tqqq = load_df(data_root / "TQQQ-1d.feather")
    spy = load_df(data_root / "SPY-1d.feather")
    ixic = load_df(data_root / "^IXIC-1d.feather")
    vix = load_df(data_root / "^VIX-1d.feather")
    rich = prepare_frame(qqq, tqqq, tqqq, spy, ixic, vix, entry_fast_window, entry_slow_window)
    rich = rich[
        [
            "date",
            "qqq_breakout_20",
            "qqq_sweep_reclaim_20",
            "qqq_volume_ratio_20",
            "qqq_compression_60",
        ]
    ].copy()
    merged = frame.merge(rich, on="date", how="left")
    for column in ["qqq_breakout_20", "qqq_sweep_reclaim_20", "qqq_compression_60"]:
        merged[column] = merged[column].fillna(False).astype(bool)
    merged["qqq_volume_ratio_20"] = merged["qqq_volume_ratio_20"].fillna(0.0)
    return merged


def build_recovery_mask(frame: pd.DataFrame, rule_name: str) -> pd.Series:
    if rule_name == "off":
        return pd.Series(False, index=frame.index)
    score = frame["long_context_score"].fillna(0).astype(int)
    breakout = frame["qqq_breakout_20"].fillna(False)
    reclaim = frame["qqq_sweep_reclaim_20"].fillna(False)
    rel_strong = frame["rel_strength_label"].eq("qqq_strong")
    vix_not_high = frame["vix_label"].isin(["vix_low", "vix_normal"])
    mom_pos = frame["qqq_mom_20"].fillna(0.0) > 0

    mapping: dict[str, pd.Series] = {
        "score_ge3": score >= 3,
        "score_ge4": score >= 4,
        "score_ge5": score >= 5,
        "score_ge3_breakout20": (score >= 3) & breakout,
        "score_ge4_breakout20": (score >= 4) & breakout,
        "score_ge5_breakout20": (score >= 5) & breakout,
        "score_ge3_breakout_or_reclaim20": (score >= 3) & (breakout | reclaim),
        "score_ge4_breakout_or_reclaim20": (score >= 4) & (breakout | reclaim),
        "score_ge5_breakout_or_reclaim20": (score >= 5) & (breakout | reclaim),
        "rel_strong_breakout20": rel_strong & breakout,
        "rel_strong_breakout_or_reclaim20": rel_strong & (breakout | reclaim),
        "vix_not_high_rel_strong_breakout_or_reclaim20": vix_not_high & rel_strong & (breakout | reclaim),
        "score_ge4_breakout_or_reclaim20_mom_pos": (score >= 4) & (breakout | reclaim) & mom_pos,
    }
    if rule_name not in mapping:
        raise ValueError(f"Unsupported overlay rule: {rule_name}")
    return mapping[rule_name].fillna(False)


def annual_returns(equity: pd.DataFrame) -> dict[str, float]:
    if equity.empty:
        return {}
    yearly: dict[str, float] = {}
    frame = equity.copy()
    frame["year"] = frame["date"].dt.year.astype(str)
    for year, group in frame.groupby("year"):
        start = float(group.iloc[0]["equity"])
        end = float(group.iloc[-1]["equity"])
        yearly[year] = round((end / start - 1.0) * 100.0, 2) if start > 0 else 0.0
    return yearly


def summarize_trades(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "count": 0,
            "win_rate_pct": 0.0,
            "avg_hold_days": 0.0,
            "median_hold_days": 0.0,
            "avg_trade_return_pct": 0.0,
            "median_trade_return_pct": 0.0,
            "overlay_entries": 0,
        }
    trade_df = pd.DataFrame(trades)
    wins = (trade_df["trade_return_pct"] > 0).sum()
    overlay_entries = int((trade_df["entry_type"] == "overlay").sum())
    return {
        "count": int(len(trade_df)),
        "win_rate_pct": round(float(wins) / len(trade_df) * 100.0, 2),
        "avg_hold_days": round(float(trade_df["hold_days"].mean()), 2),
        "median_hold_days": round(float(trade_df["hold_days"].median()), 2),
        "avg_trade_return_pct": round(float(trade_df["trade_return_pct"].mean()), 2),
        "median_trade_return_pct": round(float(trade_df["trade_return_pct"].median()), 2),
        "overlay_entries": overlay_entries,
    }


def run_candidate(
    frame: pd.DataFrame,
    *,
    regime_filter: str,
    max_hold_days: int,
    trailing_lookback_days: int,
    trailing_drawdown_pct: float,
    switch_cost_bps: float,
    initial_capital: float,
    de_risk_signal_name: str,
    overlay_rule_name: str,
    cooldown_days: int,
) -> dict[str, Any]:
    allow_mask = build_allow_mask(frame, regime_filter).reset_index(drop=True)
    overlay_mask = build_recovery_mask(frame, overlay_rule_name).reset_index(drop=True)
    capital = float(initial_capital)
    holding = False
    overlay_mode = False
    pending_exit = False
    pending_exit_reason = ""
    exit_override = False
    hold_days = 0
    rolling_peak = 0.0
    previous_close = 0.0
    cooldown_left = 0
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    entry_equity = 0.0
    entry_date: pd.Timestamp | None = None
    entry_type = "base"

    for idx, row in frame.iterrows():
        start_capital = capital
        prev_signal_on = bool(int(frame.iloc[idx - 1]["entry_signal"]) > 0) if idx > 0 else False
        prev_allow = bool(allow_mask.iloc[idx - 1]) if idx > 0 else False
        base_desired_today = prev_signal_on and prev_allow
        can_overlay_today = (
            idx > 0
            and prev_signal_on
            and (not prev_allow)
            and cooldown_left == 0
            and bool(overlay_mask.iloc[idx - 1])
        )
        desired_today = base_desired_today or can_overlay_today or (overlay_mode and prev_signal_on)

        if exit_override and not desired_today:
            exit_override = False

        if holding and previous_close > 0:
            capital *= float(row["tqqq_open"]) / previous_close

        entered_today = False
        exited_today = False
        action_cost = 0.0

        if holding and (pending_exit or not desired_today):
            exit_reason = pending_exit_reason if pending_exit else "signal_off"
            action_cost += float(switch_cost_bps) / 10000.0
            capital *= 1.0 - float(switch_cost_bps) / 10000.0
            exited_today = True
            if entry_date is not None:
                trades.append(
                    {
                        "entry_date": str(entry_date.date()),
                        "exit_date": str(pd.Timestamp(row["date"]).date()),
                        "entry_type": entry_type,
                        "trade_return_pct": round((capital / entry_equity - 1.0) * 100.0, 2),
                        "hold_days": int(hold_days),
                        "exit_reason": exit_reason,
                    }
                )
            holding = False
            overlay_mode = False
            pending_exit = False
            pending_exit_reason = ""
            hold_days = 0
            rolling_peak = 0.0
            cooldown_left = max(int(cooldown_days), 0)

        if (not holding) and desired_today and not exit_override:
            action_cost += float(switch_cost_bps) / 10000.0
            capital *= 1.0 - float(switch_cost_bps) / 10000.0
            holding = True
            entered_today = True
            hold_days = 0
            rolling_peak = float(row["tqqq_open"])
            entry_equity = start_capital
            entry_date = pd.Timestamp(row["date"])
            entry_type = "overlay" if can_overlay_today and not base_desired_today else "base"
            overlay_mode = entry_type == "overlay"
        elif (not holding) and cooldown_left > 0:
            cooldown_left -= 1

        trailing_exit = False
        time_exit = False
        leverage = 0.0
        if holding:
            leverage = de_risk_fraction(frame.iloc[idx - 1], de_risk_signal_name) if idx > 0 else 1.0
            open_price = float(row["tqqq_open"])
            close_price = float(row["tqqq_close"])
            if open_price > 0:
                capital *= 1.0 + leverage * (close_price / open_price - 1.0)
            hold_days += 1
            rolling_peak = max(rolling_peak, close_price)
            if base_desired_today:
                overlay_mode = False
            if trailing_lookback_days > 0 and trailing_drawdown_pct > 0 and hold_days >= trailing_lookback_days and rolling_peak > 0:
                drawdown_from_peak = (rolling_peak - close_price) / rolling_peak * 100.0
                trailing_exit = drawdown_from_peak >= trailing_drawdown_pct
            if max_hold_days > 0 and hold_days >= max_hold_days:
                time_exit = True
            if trailing_exit or time_exit:
                pending_exit = True
                pending_exit_reason = "trailing" if trailing_exit else "time"
                exit_override = True

        previous_close = float(row["tqqq_close"])
        rows.append(
            {
                "date": pd.Timestamp(row["date"]),
                "position": "TQQQ" if holding else "CASH",
                "capital": float(capital),
                "daily_return": float(capital / start_capital - 1.0) if start_capital > 0 else 0.0,
                "entered_today": bool(entered_today),
                "exited_today": bool(exited_today),
                "entry_type_active": entry_type if holding else "none",
                "overlay_mode": bool(overlay_mode),
                "leverage": float(leverage),
                "cooldown_left": int(cooldown_left),
                "trailing_exit": bool(trailing_exit),
                "time_exit": bool(time_exit),
                "action_cost": float(action_cost),
            }
        )

    path = pd.DataFrame(rows)
    equity = path[["date", "capital"]].rename(columns={"capital": "equity"})
    yearly = annual_returns(equity)
    trade_summary = summarize_trades(trades)
    summary = {
        "total_return_pct": round((float(path.iloc[-1]["capital"]) / float(initial_capital) - 1.0) * 100.0, 2) if not path.empty else 0.0,
        "max_drawdown_pct": round(max_drawdown_pct(path["capital"]), 2) if not path.empty else 0.0,
        "yearly_returns_pct": yearly,
        "trades": trade_summary["count"],
        "win_rate_pct": trade_summary["win_rate_pct"],
        "avg_hold_days": trade_summary["avg_hold_days"],
        "median_hold_days": trade_summary["median_hold_days"],
        "avg_trade_return_pct": trade_summary["avg_trade_return_pct"],
        "median_trade_return_pct": trade_summary["median_trade_return_pct"],
        "overlay_entries": trade_summary["overlay_entries"],
        "invested_days": int((path["position"] == "TQQQ").sum()) if not path.empty else 0,
        "invested_ratio_pct": round(float((path["position"] == "TQQQ").mean()) * 100.0, 2) if not path.empty else 0.0,
    }
    score = round(
        float(summary["total_return_pct"])
        - 1.75 * float(summary["max_drawdown_pct"])
        + float(summary["yearly_returns_pct"].get("2026", 0.0))
        + 0.15 * float(summary["overlay_entries"]),
        4,
    )
    return {
        "summary": summary,
        "score": score,
        "trades": trades,
        "path": path,
    }


def candidate_sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
    summary = item["summary"]
    return (
        float(item["score"]),
        float(summary["total_return_pct"]),
        -float(summary["max_drawdown_pct"]),
    )


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    frame = load_overlay_frame(
        data_root=data_root,
        entry_fast_window=int(args.entry_fast_window),
        entry_slow_window=int(args.entry_slow_window),
    )
    if args.start_date:
        frame = frame[frame["date"] >= pd.Timestamp(str(args.start_date), tz="UTC")].reset_index(drop=True)

    baseline = run_candidate(
        frame,
        regime_filter=str(args.regime_filter),
        max_hold_days=int(args.max_hold_days),
        trailing_lookback_days=int(args.trailing_lookback_days),
        trailing_drawdown_pct=float(args.trailing_drawdown_pct),
        switch_cost_bps=float(args.switch_cost_bps),
        initial_capital=float(args.initial_capital),
        de_risk_signal_name=str(args.de_risk_signal_name),
        overlay_rule_name="off",
        cooldown_days=0,
    )

    candidates: list[dict[str, Any]] = []
    for overlay_rule_name in parse_str_list(args.overlay_rules):
        for cooldown_days in parse_int_list(args.cooldown_days_values):
            if overlay_rule_name == "off" and cooldown_days > 0:
                continue
            result = run_candidate(
                frame,
                regime_filter=str(args.regime_filter),
                max_hold_days=int(args.max_hold_days),
                trailing_lookback_days=int(args.trailing_lookback_days),
                trailing_drawdown_pct=float(args.trailing_drawdown_pct),
                switch_cost_bps=float(args.switch_cost_bps),
                initial_capital=float(args.initial_capital),
                de_risk_signal_name=str(args.de_risk_signal_name),
                overlay_rule_name=overlay_rule_name,
                cooldown_days=int(cooldown_days),
            )
            item = {
                "overlay_rule_name": overlay_rule_name,
                "cooldown_days": int(cooldown_days),
                "summary": result["summary"],
                "score": result["score"],
                "delta_total_return_pct": round(
                    float(result["summary"]["total_return_pct"]) - float(baseline["summary"]["total_return_pct"]),
                    2,
                ),
                "delta_max_drawdown_pct": round(
                    float(result["summary"]["max_drawdown_pct"]) - float(baseline["summary"]["max_drawdown_pct"]),
                    2,
                ),
            }
            candidates.append(item)

    ranked = sorted(candidates, key=candidate_sort_key, reverse=True)
    payload = {
        "baseline": {
            "config": {
                "entry_fast_window": int(args.entry_fast_window),
                "entry_slow_window": int(args.entry_slow_window),
                "regime_filter": str(args.regime_filter),
                "max_hold_days": int(args.max_hold_days),
                "trailing_lookback_days": int(args.trailing_lookback_days),
                "trailing_drawdown_pct": float(args.trailing_drawdown_pct),
                "de_risk_signal_name": str(args.de_risk_signal_name),
                "switch_cost_bps": float(args.switch_cost_bps),
                "start_date": str(args.start_date),
            },
            "summary": baseline["summary"],
        },
        "top_candidates": ranked[: int(args.top)],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    print("BASELINE", json.dumps(baseline["summary"], ensure_ascii=False))
    for item in ranked[: min(int(args.top), 12)]:
        summary = item["summary"]
        print(
            f"{item['overlay_rule_name']} cooldown={item['cooldown_days']} "
            f"full={summary['total_return_pct']:.2f}% dd={summary['max_drawdown_pct']:.2f}% "
            f"trades={summary['trades']} win={summary['win_rate_pct']:.2f}% hold={summary['avg_hold_days']:.2f}d "
            f"overlay_entries={summary['overlay_entries']} delta={item['delta_total_return_pct']:+.2f}%"
        )


if __name__ == "__main__":
    main()
