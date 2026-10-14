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

from scripts.tqqq_cash_strict_utils import (  # noqa: E402
    annual_returns,
    build_allow_mask,
    build_recovery_mask,
    de_risk_fraction,
    load_strict_frame_with_overlay_context,
    max_drawdown_pct,
    recovery_overlay_trigger,
    strict_overlay_trade_stats,
)


DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "tqqq_hard_exit_reset_signal_scan_20260616.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan independent signal-reset rules for the TQQQ hard-exit lock.")
    parser.add_argument("--data-root", default=str(DEFAULT_PUBLIC_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--csv-output")
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--entry-fast-window", type=int, default=25)
    parser.add_argument("--entry-slow-window", type=int, default=150)
    parser.add_argument("--regime-filter", default="ixic_filter")
    parser.add_argument("--max-hold-days", type=int, default=90)
    parser.add_argument("--trailing-lookback-days", type=int, default=10)
    parser.add_argument("--trailing-drawdown-pct", type=float, default=12.0)
    parser.add_argument("--de-risk-signal-name", default="breakout_fail_score_le3_flat")
    parser.add_argument("--recovery-reentry-rule", default="score_ge3")
    parser.add_argument("--recovery-reentry-cooldown-days", type=int, default=0)
    parser.add_argument("--switch-cost-bps", type=float, default=10.0)
    parser.add_argument("--ma-windows", default="5,10,15,20,25,30,40,50,60,75,100")
    parser.add_argument("--fast-windows", default="5,10,15,20,25")
    parser.add_argument("--slow-windows", default="20,30,40,50,60,75,100,150")
    parser.add_argument("--momentum-windows", default="5,10,20,30")
    parser.add_argument("--momentum-thresholds-pct", default="-2,0,2")
    parser.add_argument("--reentry-delay-bars-values", default="1")
    parser.add_argument("--top", type=int, default=40)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def recent_return_pct(path: pd.DataFrame, days: int) -> float:
    if path.empty:
        return 0.0
    frame = path.sort_values("date").reset_index(drop=True)
    end = pd.Timestamp(frame.iloc[-1]["date"])
    start = end - pd.Timedelta(days=int(days))
    before = frame[frame["date"] <= start]
    anchor = before.iloc[-1] if not before.empty else frame.iloc[0]
    start_capital = float(anchor["capital"])
    end_capital = float(frame.iloc[-1]["capital"])
    return round((end_capital / start_capital - 1.0) * 100.0, 2) if start_capital > 0 else 0.0


def summarize_path(
    path: pd.DataFrame,
    *,
    initial_capital: float,
    next_open_target: str,
    final_lock_active: bool,
    lock_reset_count: int,
    lock_blocked_days: int,
) -> dict[str, Any]:
    equity = path[["date", "capital"]].rename(columns={"capital": "equity"})
    yearly = annual_returns(equity)
    total_return_pct = (
        round((float(path.iloc[-1]["capital"]) / float(initial_capital) - 1.0) * 100.0, 2)
        if not path.empty
        else 0.0
    )
    max_dd = round(max_drawdown_pct(path["capital"]), 2) if not path.empty else 0.0
    invested_days = int((path["position"] == "TQQQ").sum()) if not path.empty else 0
    trade_stats = strict_overlay_trade_stats(path)
    score = round(
        total_return_pct
        - max_dd * 1.75
        + float(yearly.get("2022", 0.0)) * 1.2
        + float(yearly.get("2023", 0.0)) * 0.6
        + float(yearly.get("2024", 0.0)) * 0.7
        + float(yearly.get("2025", 0.0)) * 0.8
        + float(yearly.get("2026", 0.0)) * 1.2,
        4,
    )
    return {
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd,
        "yearly_returns_pct": yearly,
        "return_2026_pct": float(yearly.get("2026", 0.0)),
        "recent_60d_return_pct": recent_return_pct(path, 60),
        "recent_30d_return_pct": recent_return_pct(path, 30),
        "trades": int(path["entered_today"].sum()) if not path.empty else 0,
        "closed_trades": int(trade_stats["trades_closed"]),
        "win_rate_pct": float(trade_stats["win_rate_pct"]),
        "avg_hold_days": float(trade_stats["avg_hold_days"]),
        "invested_days": invested_days,
        "invested_ratio_pct": round(invested_days / len(path) * 100.0, 2) if len(path) else 0.0,
        "latest_position": str(path.iloc[-1]["position"]) if not path.empty else "CASH",
        "next_open_target": str(next_open_target),
        "final_lock_active": bool(final_lock_active),
        "lock_reset_count": int(lock_reset_count),
        "lock_blocked_days": int(lock_blocked_days),
        "score": score,
    }


def add_reset_signal_columns(frame: pd.DataFrame, args: argparse.Namespace) -> dict[str, pd.Series]:
    signals: dict[str, pd.Series] = {}
    main_active = (frame["entry_signal"].astype(int) > 0) & build_allow_mask(frame, str(args.regime_filter)).reset_index(drop=True)
    signals["main_desired"] = main_active.fillna(False)

    for source in ["qqq", "ixic", "tqqq"]:
        close_col = f"{source}_close"
        if close_col not in frame.columns:
            continue
        close = frame[close_col].astype(float)
        for window in parse_int_list(args.ma_windows):
            ma = close.rolling(int(window)).mean()
            signals[f"{source}_close_gt_ma{window}"] = (close > ma).fillna(False)

    for source in ["qqq", "ixic"]:
        close_col = f"{source}_close"
        if close_col not in frame.columns:
            continue
        close = frame[close_col].astype(float)
        for window in parse_int_list(args.momentum_windows):
            mom_pct = (close / close.shift(int(window)) - 1.0) * 100.0
            for threshold in parse_float_list(args.momentum_thresholds_pct):
                threshold_label = str(threshold).replace("-", "m").replace(".", "p")
                signals[f"{source}_mom{window}_ge_{threshold_label}pct"] = (mom_pct >= float(threshold)).fillna(False)

    qqq_close = frame["qqq_close"].astype(float)
    for fast in parse_int_list(args.fast_windows):
        fast_ma = qqq_close.rolling(int(fast)).mean()
        for slow in parse_int_list(args.slow_windows):
            if int(fast) >= int(slow):
                continue
            slow_ma = qqq_close.rolling(int(slow)).mean()
            signals[f"qqq_ma{fast}_gt_ma{slow}"] = (fast_ma > slow_ma).fillna(False)

    for window in parse_int_list(args.ma_windows):
        qqq_name = f"qqq_close_gt_ma{window}"
        ixic_name = f"ixic_close_gt_ma{window}"
        if qqq_name in signals and ixic_name in signals:
            signals[f"qqq_ixic_close_gt_ma{window}"] = (signals[qqq_name] & signals[ixic_name]).fillna(False)

    return {name: value.reset_index(drop=True) for name, value in signals.items()}


def run_candidate(
    frame: pd.DataFrame,
    *,
    reset_signal: pd.Series,
    reset_signal_name: str,
    reentry_delay_bars: int,
    regime_filter: str,
    max_hold_days: int,
    trailing_lookback_days: int,
    trailing_drawdown_pct: float,
    switch_cost_bps: float,
    initial_capital: float,
    de_risk_signal_name: str,
    recovery_reentry_rule: str,
    recovery_reentry_cooldown_days: int,
) -> dict[str, Any]:
    allow_mask = build_allow_mask(frame, regime_filter).reset_index(drop=True)
    recovery_mask = build_recovery_mask(frame, recovery_reentry_rule).reset_index(drop=True)

    capital = float(initial_capital)
    holding = False
    pending_exit = False
    lock_active = False
    lock_reset_delay_left = 0
    lock_reset_count = 0
    lock_blocked_days = 0
    hold_days = 0
    rolling_peak = 0.0
    previous_close = 0.0
    recovery_cooldown_left = 0
    overlay_mode = False
    overlay_entry_type = "base"
    overlay_allocation = 1.0
    rows: list[dict[str, Any]] = []

    for idx, row in frame.iterrows():
        start_capital = capital
        recovery_trigger = recovery_overlay_trigger(
            frame,
            idx,
            recovery_rule_name=recovery_reentry_rule,
            recovery_cooldown_days=recovery_reentry_cooldown_days,
            recovery_cooldown_left=recovery_cooldown_left,
            allow_mask=allow_mask,
            recovery_mask=recovery_mask,
        )
        if idx > 0:
            prev_signal_on = bool(int(frame.iloc[idx - 1]["entry_signal"]) > 0)
            base_desired_today = bool(prev_signal_on and bool(allow_mask.iloc[idx - 1]))
            desired_today = base_desired_today or recovery_trigger or (overlay_mode and prev_signal_on)
            reset_active_today = bool(reset_signal.iloc[idx - 1])
        else:
            base_desired_today = False
            desired_today = False
            reset_active_today = False

        if lock_active and not reset_active_today:
            lock_active = False
            lock_reset_delay_left = max(int(reentry_delay_bars), 0)
            lock_reset_count += 1
        elif lock_active and desired_today:
            lock_blocked_days += 1

        if holding and previous_close > 0:
            capital *= float(row["tqqq_open"]) / previous_close

        action_cost = 0.0
        entered_today = False
        exited_today = False

        if holding and (pending_exit or not desired_today):
            action_cost += float(switch_cost_bps) / 10000.0
            capital *= 1.0 - float(switch_cost_bps) / 10000.0
            holding = False
            pending_exit = False
            hold_days = 0
            rolling_peak = 0.0
            exited_today = True
            if overlay_mode and recovery_reentry_cooldown_days > 0:
                recovery_cooldown_left = int(recovery_reentry_cooldown_days)
            overlay_mode = False
            overlay_entry_type = "base"
            overlay_allocation = 1.0

        can_enter_today = desired_today and not lock_active and lock_reset_delay_left <= 0
        if (not holding) and can_enter_today:
            action_cost += float(switch_cost_bps) / 10000.0
            capital *= 1.0 - float(switch_cost_bps) / 10000.0
            holding = True
            hold_days = 0
            rolling_peak = float(row["tqqq_open"])
            entered_today = True
            if recovery_trigger and not base_desired_today:
                overlay_mode = True
                overlay_entry_type = "recovery_reentry"
                overlay_allocation = 1.0
            else:
                overlay_mode = False
                overlay_entry_type = "base"
                overlay_allocation = 1.0
        elif not holding:
            if recovery_cooldown_left > 0:
                recovery_cooldown_left -= 1
            if lock_reset_delay_left > 0:
                lock_reset_delay_left -= 1

        trailing_exit = False
        time_exit = False
        leverage = 0.0
        if holding:
            base_fraction = de_risk_fraction(frame.iloc[idx - 1], de_risk_signal_name) if idx > 0 else 1.0
            leverage = float(base_fraction) * float(overlay_allocation if overlay_mode else 1.0)
            open_price = float(row["tqqq_open"])
            close_price = float(row["tqqq_close"])
            if open_price > 0:
                capital *= 1.0 + leverage * (close_price / open_price - 1.0)
            hold_days += 1
            rolling_peak = max(rolling_peak, close_price)
            if base_desired_today and overlay_mode:
                overlay_mode = False
                overlay_entry_type = "base"
                overlay_allocation = 1.0
            if trailing_lookback_days > 0 and trailing_drawdown_pct > 0 and hold_days >= trailing_lookback_days and rolling_peak > 0:
                drawdown_from_peak = (rolling_peak - close_price) / rolling_peak * 100.0
                trailing_exit = drawdown_from_peak >= float(trailing_drawdown_pct)
            if max_hold_days > 0 and hold_days >= max_hold_days:
                time_exit = True
            if trailing_exit or time_exit:
                pending_exit = True
                lock_active = True
                lock_reset_delay_left = 0

        previous_close = float(row["tqqq_close"])
        position = "TQQQ" if holding else "CASH"
        daily_return = capital / start_capital - 1.0 if start_capital > 0 else 0.0
        rows.append(
            {
                "date": pd.Timestamp(row["date"]),
                "position": position,
                "daily_return": float(daily_return),
                "capital": float(capital),
                "entered_today": bool(entered_today),
                "exited_today": bool(exited_today),
                "pending_exit": bool(pending_exit),
                "trailing_exit": bool(trailing_exit),
                "time_exit": bool(time_exit),
                "entry_type": overlay_entry_type if holding else "none",
                "lock_active": bool(lock_active),
                "reset_active_today": bool(reset_active_today),
                "lock_reset_delay_left": int(lock_reset_delay_left),
                "action_cost": float(action_cost),
                "leverage": float(leverage),
            }
        )

    path = pd.DataFrame(rows)
    if frame.empty:
        next_open_target = "CASH"
    else:
        last_idx = len(frame) - 1
        last = frame.iloc[last_idx]
        base_next = bool(int(last["entry_signal"]) > 0 and bool(allow_mask.iloc[last_idx]))
        recovery_next = bool(
            int(last["entry_signal"]) > 0
            and (not bool(allow_mask.iloc[last_idx]))
            and recovery_cooldown_left <= 0
            and bool(recovery_mask.iloc[last_idx])
        )
        desired_next = base_next or recovery_next or (overlay_mode and bool(int(last["entry_signal"]) > 0))
        reset_active_next = bool(reset_signal.iloc[last_idx])
        lock_for_next = bool(lock_active and reset_active_next)
        if holding and pending_exit:
            next_open_target = "CASH"
        elif holding:
            next_open_target = "TQQQ"
        else:
            next_open_target = "TQQQ" if desired_next and not lock_for_next and lock_reset_delay_left <= 0 else "CASH"

    summary = summarize_path(
        path,
        initial_capital=float(initial_capital),
        next_open_target=next_open_target,
        final_lock_active=bool(lock_active),
        lock_reset_count=int(lock_reset_count),
        lock_blocked_days=int(lock_blocked_days),
    )
    return {
        "params": {
            "reset_signal_name": reset_signal_name,
            "reentry_delay_bars": int(reentry_delay_bars),
        },
        "summary": summary,
        "path": path,
    }


def flatten_item(item: dict[str, Any]) -> dict[str, Any]:
    params = item["params"]
    summary = item["summary"]
    yearly = summary.get("yearly_returns_pct", {})
    return {
        **params,
        "total_return_pct": summary["total_return_pct"],
        "max_drawdown_pct": summary["max_drawdown_pct"],
        "return_2023_pct": float(yearly.get("2023", 0.0)),
        "return_2024_pct": float(yearly.get("2024", 0.0)),
        "return_2025_pct": float(yearly.get("2025", 0.0)),
        "return_2026_pct": summary["return_2026_pct"],
        "recent_60d_return_pct": summary["recent_60d_return_pct"],
        "recent_30d_return_pct": summary["recent_30d_return_pct"],
        "trades": summary["trades"],
        "closed_trades": summary["closed_trades"],
        "win_rate_pct": summary["win_rate_pct"],
        "invested_ratio_pct": summary["invested_ratio_pct"],
        "latest_position": summary["latest_position"],
        "next_open_target": summary["next_open_target"],
        "final_lock_active": summary["final_lock_active"],
        "lock_reset_count": summary["lock_reset_count"],
        "lock_blocked_days": summary["lock_blocked_days"],
        "score": summary["score"],
    }


def sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    summary = item["summary"]
    return (
        float(summary["score"]),
        float(summary["total_return_pct"]),
        float(summary["return_2026_pct"]),
        -float(summary["max_drawdown_pct"]),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    csv_output = Path(args.csv_output) if args.csv_output else output_path.with_suffix(".csv")
    frame = load_strict_frame_with_overlay_context(
        data_root=Path(args.data_root),
        entry_fast_window=int(args.entry_fast_window),
        entry_slow_window=int(args.entry_slow_window),
    ).copy()
    signals = add_reset_signal_columns(frame, args)

    common = {
        "regime_filter": str(args.regime_filter),
        "max_hold_days": int(args.max_hold_days),
        "trailing_lookback_days": int(args.trailing_lookback_days),
        "trailing_drawdown_pct": float(args.trailing_drawdown_pct),
        "switch_cost_bps": float(args.switch_cost_bps),
        "initial_capital": float(args.initial_capital),
        "de_risk_signal_name": str(args.de_risk_signal_name),
        "recovery_reentry_rule": str(args.recovery_reentry_rule),
        "recovery_reentry_cooldown_days": int(args.recovery_reentry_cooldown_days),
    }

    candidates: list[dict[str, Any]] = []
    for reentry_delay_bars in parse_int_list(args.reentry_delay_bars_values):
        for name, signal in signals.items():
            result = run_candidate(
                frame,
                reset_signal=signal,
                reset_signal_name=name,
                reentry_delay_bars=int(reentry_delay_bars),
                **common,
            )
            candidates.append({"params": result["params"], "summary": result["summary"]})

    ranked = sorted(candidates, key=sort_key, reverse=True)
    csv_rows = [flatten_item(item) for item in ranked]
    write_csv(csv_output, csv_rows)

    baseline = next(
        item
        for item in candidates
        if item["params"]["reset_signal_name"] == "main_desired"
        and int(item["params"]["reentry_delay_bars"]) == parse_int_list(args.reentry_delay_bars_values)[0]
    )
    payload = {
        "reference": {
            "strategy": "TQQQ mail-chain strict strategy with independent hard-exit reset signal scan",
            "data_start": str(pd.Timestamp(frame.iloc[0]["date"]).date()) if not frame.empty else None,
            "data_end": str(pd.Timestamp(frame.iloc[-1]["date"]).date()) if not frame.empty else None,
            "execution_mode": "T+1 open execution with intraday open-to-close PnL",
            "reset_semantics": "hard-exit lock clears after the reset signal has one inactive bar; reentry_delay_bars controls same/next-bar reentry.",
            "common_config": {
                "entry_fast_window": int(args.entry_fast_window),
                "entry_slow_window": int(args.entry_slow_window),
                **common,
            },
        },
        "baseline_main_desired_reset": baseline["summary"],
        "scan_size": len(candidates),
        "csv_output": str(csv_output),
        "top_candidates": ranked[: int(args.top)],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(output_path)
    print(csv_output)
    print("BASELINE", json.dumps(baseline["summary"], ensure_ascii=False))
    for item in ranked[: min(int(args.top), 20)]:
        params = item["params"]
        summary = item["summary"]
        print(
            f"score={summary['score']:.2f} full={summary['total_return_pct']:.2f}% "
            f"dd={summary['max_drawdown_pct']:.2f}% 2026={summary['return_2026_pct']:.2f}% "
            f"60d={summary['recent_60d_return_pct']:.2f}% 30d={summary['recent_30d_return_pct']:.2f}% "
            f"latest={summary['latest_position']} next={summary['next_open_target']} "
            f"reset={params['reset_signal_name']} delay={params['reentry_delay_bars']} "
            f"resets={summary['lock_reset_count']} blocked={summary['lock_blocked_days']}"
        )


if __name__ == "__main__":
    main()
