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

from scripts.scan_tqqq_recovery_reentry_overlay import (  # noqa: E402
    annual_returns,
    load_overlay_frame,
    max_drawdown_pct,
    summarize_trades,
)
from scripts.tqqq_cash_strict_utils import build_allow_mask, de_risk_fraction  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "tqqq_drawdown_ladder_overlay_scan.json"
DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan drawdown ladder overlay on top of TQQQ strict baseline.")
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
    parser.add_argument("--start-date", default="2023-01-03")
    parser.add_argument("--drawdown-source", default="tqqq", choices=["tqqq", "qqq"])
    parser.add_argument("--drawdown-threshold-values", default="25,30,35,40")
    parser.add_argument("--peak-lookback-values", default="60,90,120,180")
    parser.add_argument("--ladder-schemes", default="two_equal,three_40_30_30,three_50_30_20")
    parser.add_argument("--rebound-exit-threshold-values", default="12,15,20,25")
    parser.add_argument("--max-overlay-hold-values", default="10,15,20,30")
    parser.add_argument("--vix-allow-values", default="all,vix_low_normal,not_extreme")
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [str(item.strip()) for item in str(value).split(",") if item.strip()]


def ladder_steps(name: str) -> list[tuple[float, float]]:
    if name == "two_equal":
        return [(0.0, 0.5), (5.0, 0.5)]
    if name == "three_40_30_30":
        return [(0.0, 0.4), (5.0, 0.3), (10.0, 0.3)]
    if name == "three_50_30_20":
        return [(0.0, 0.5), (5.0, 0.3), (10.0, 0.2)]
    raise ValueError(f"Unsupported ladder scheme: {name}")


def vix_ok(label: str, rule: str) -> bool:
    if rule == "all":
        return True
    if rule == "vix_low_normal":
        return label in {"vix_low", "vix_normal"}
    if rule == "not_extreme":
        return label != "vix_extreme"
    raise ValueError(f"Unsupported vix rule: {rule}")


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
    drawdown_source: str,
    drawdown_threshold_pct: float,
    peak_lookback_days: int,
    ladder_scheme: str,
    rebound_exit_threshold_pct: float,
    max_overlay_hold_days: int,
    vix_rule: str,
) -> dict[str, Any]:
    allow_mask = build_allow_mask(frame, regime_filter).reset_index(drop=True)
    source_column = "tqqq_close" if drawdown_source == "tqqq" else "qqq_close"
    rolling_peak_ref = frame[source_column].rolling(int(peak_lookback_days)).max().shift(1)
    dd_from_peak = (rolling_peak_ref - frame[source_column]) / rolling_peak_ref * 100.0

    capital = float(initial_capital)
    previous_close = 0.0
    holding = False
    overlay_mode = False
    pending_exit = False
    pending_exit_reason = ""
    exit_override = False
    hold_days = 0
    rolling_peak = 0.0
    entry_equity = 0.0
    entry_date: pd.Timestamp | None = None
    entry_type = "base"
    overlay_days = 0
    overlay_entry_close = 0.0
    overlay_step_idx = -1
    overlay_allocation = 0.0
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    steps = ladder_steps(ladder_scheme)

    for idx, row in frame.iterrows():
        start_capital = capital
        prev_signal_on = bool(int(frame.iloc[idx - 1]["entry_signal"]) > 0) if idx > 0 else False
        prev_allow = bool(allow_mask.iloc[idx - 1]) if idx > 0 else False
        base_desired_today = prev_signal_on and prev_allow

        overlay_trigger = False
        new_overlay_allocation = overlay_allocation
        new_overlay_step_idx = overlay_step_idx
        if idx > 0 and prev_signal_on and (not prev_allow) and pd.notna(dd_from_peak.iloc[idx - 1]):
            peak_dd = float(dd_from_peak.iloc[idx - 1])
            label = str(frame.iloc[idx - 1]["vix_label"])
            if peak_dd >= float(drawdown_threshold_pct) and vix_ok(label, vix_rule):
                extra_dd = peak_dd - float(drawdown_threshold_pct)
                candidate_step_idx = -1
                candidate_allocation = 0.0
                for i, (step_extra_dd, allocation) in enumerate(steps):
                    if extra_dd >= float(step_extra_dd):
                        candidate_step_idx = i
                        candidate_allocation += float(allocation)
                if candidate_step_idx >= 0:
                    overlay_trigger = True
                    new_overlay_step_idx = candidate_step_idx
                    new_overlay_allocation = candidate_allocation

        desired_today = base_desired_today or overlay_trigger or (overlay_mode and prev_signal_on)

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
                        "overlay_allocation": round(float(overlay_allocation), 4),
                    }
                )
            holding = False
            overlay_mode = False
            pending_exit = False
            pending_exit_reason = ""
            hold_days = 0
            rolling_peak = 0.0
            overlay_days = 0
            overlay_entry_close = 0.0
            overlay_step_idx = -1
            overlay_allocation = 0.0

        if (not holding) and desired_today and not exit_override:
            action_cost += float(switch_cost_bps) / 10000.0
            capital *= 1.0 - float(switch_cost_bps) / 10000.0
            holding = True
            entered_today = True
            hold_days = 0
            rolling_peak = float(row["tqqq_open"])
            entry_equity = start_capital
            entry_date = pd.Timestamp(row["date"])
            if overlay_trigger and not base_desired_today:
                entry_type = "overlay"
                overlay_mode = True
                overlay_days = 0
                overlay_entry_close = float(frame.iloc[idx - 1]["qqq_close"]) if idx > 0 else float(row["qqq_close"])
                overlay_step_idx = new_overlay_step_idx
                overlay_allocation = new_overlay_allocation
            else:
                entry_type = "base"
                overlay_mode = False
                overlay_days = 0
                overlay_entry_close = 0.0
                overlay_step_idx = -1
                overlay_allocation = 1.0

        trailing_exit = False
        time_exit = False
        leverage = 0.0
        if holding:
            base_fraction = de_risk_fraction(frame.iloc[idx - 1], de_risk_signal_name) if idx > 0 else 1.0
            leverage = float(overlay_allocation if overlay_mode else 1.0) * float(base_fraction)
            open_price = float(row["tqqq_open"])
            close_price = float(row["tqqq_close"])
            if open_price > 0:
                capital *= 1.0 + leverage * (close_price / open_price - 1.0)
            hold_days += 1
            rolling_peak = max(rolling_peak, close_price)
            if trailing_lookback_days > 0 and trailing_drawdown_pct > 0 and hold_days >= trailing_lookback_days and rolling_peak > 0:
                drawdown_from_peak = (rolling_peak - close_price) / rolling_peak * 100.0
                trailing_exit = drawdown_from_peak >= trailing_drawdown_pct
            if max_hold_days > 0 and hold_days >= max_hold_days:
                time_exit = True
            if overlay_mode:
                overlay_days += 1
                if overlay_entry_close > 0:
                    rebound_pct = (float(row["qqq_close"]) / overlay_entry_close - 1.0) * 100.0
                    if rebound_pct >= float(rebound_exit_threshold_pct):
                        pending_exit = True
                        pending_exit_reason = "overlay_rebound"
                        exit_override = True
                if not pending_exit and overlay_days >= int(max_overlay_hold_days):
                    pending_exit = True
                    pending_exit_reason = "overlay_time"
                    exit_override = True
                if base_desired_today:
                    overlay_mode = False
                    overlay_allocation = 1.0
                    overlay_step_idx = -1
                    overlay_days = 0
                    overlay_entry_close = 0.0
            if (not pending_exit) and (trailing_exit or time_exit):
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
                "overlay_allocation": float(overlay_allocation),
                "overlay_step_idx": int(overlay_step_idx),
                "leverage": float(leverage),
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
        "overlay_trade_count": int(sum(1 for t in trades if t["entry_type"] == "overlay")),
    }
    score = round(
        float(summary["total_return_pct"])
        - 1.75 * float(summary["max_drawdown_pct"])
        + float(summary["yearly_returns_pct"].get("2026", 0.0))
        + 0.1 * float(summary["overlay_trade_count"]),
        4,
    )
    return {
        "summary": summary,
        "score": score,
        "trades": trades,
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
        drawdown_source=str(args.drawdown_source),
        drawdown_threshold_pct=1000.0,
        peak_lookback_days=60,
        ladder_scheme="two_equal",
        rebound_exit_threshold_pct=1000.0,
        max_overlay_hold_days=1,
        vix_rule="all",
    )

    candidates: list[dict[str, Any]] = []
    for drawdown_threshold_pct in parse_float_list(args.drawdown_threshold_values):
        for peak_lookback_days in parse_int_list(args.peak_lookback_values):
            for ladder_scheme in parse_str_list(args.ladder_schemes):
                for rebound_exit_threshold_pct in parse_float_list(args.rebound_exit_threshold_values):
                    for max_overlay_hold_days in parse_int_list(args.max_overlay_hold_values):
                        for vix_rule in parse_str_list(args.vix_allow_values):
                            result = run_candidate(
                                frame,
                                regime_filter=str(args.regime_filter),
                                max_hold_days=int(args.max_hold_days),
                                trailing_lookback_days=int(args.trailing_lookback_days),
                                trailing_drawdown_pct=float(args.trailing_drawdown_pct),
                                switch_cost_bps=float(args.switch_cost_bps),
                                initial_capital=float(args.initial_capital),
                                de_risk_signal_name=str(args.de_risk_signal_name),
                                drawdown_source=str(args.drawdown_source),
                                drawdown_threshold_pct=float(drawdown_threshold_pct),
                                peak_lookback_days=int(peak_lookback_days),
                                ladder_scheme=ladder_scheme,
                                rebound_exit_threshold_pct=float(rebound_exit_threshold_pct),
                                max_overlay_hold_days=int(max_overlay_hold_days),
                                vix_rule=vix_rule,
                            )
                            item = {
                                "drawdown_threshold_pct": float(drawdown_threshold_pct),
                                "peak_lookback_days": int(peak_lookback_days),
                                "ladder_scheme": ladder_scheme,
                                "rebound_exit_threshold_pct": float(rebound_exit_threshold_pct),
                                "max_overlay_hold_days": int(max_overlay_hold_days),
                                "vix_rule": vix_rule,
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
                "drawdown_source": str(args.drawdown_source),
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
            f"dd={item['drawdown_threshold_pct']} lookback={item['peak_lookback_days']} ladder={item['ladder_scheme']} "
            f"rebound={item['rebound_exit_threshold_pct']} hold={item['max_overlay_hold_days']} vix={item['vix_rule']} "
            f"full={summary['total_return_pct']:.2f}% ddmax={summary['max_drawdown_pct']:.2f}% "
            f"trades={summary['trades']} win={summary['win_rate_pct']:.2f}% overlay={summary['overlay_trade_count']} "
            f"delta={item['delta_total_return_pct']:+.2f}%"
        )


if __name__ == "__main__":
    main()
