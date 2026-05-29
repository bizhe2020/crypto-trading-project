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

from scripts.replay_tqqq_sqqq_trend_baseline import load_df, max_drawdown_pct


DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "tqqq_cash_exit_profile_scan.json"


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan TQQQ/CASH exit profiles on top of fixed QQQ 25/150 entry.")
    parser.add_argument("--tqqq", default=str(DEFAULT_PUBLIC_DIR / "TQQQ-1d.feather"))
    parser.add_argument("--qqq", default=str(DEFAULT_PUBLIC_DIR / "QQQ-1d.feather"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--entry-fast-window", type=int, default=25)
    parser.add_argument("--entry-slow-window", type=int, default=150)
    parser.add_argument("--switch-cost-bps", type=float, default=5.0)
    parser.add_argument("--max-hold-days-values", default="0,40,60,90,120")
    parser.add_argument("--trailing-lookback-days-values", default="0,10,20,30")
    parser.add_argument("--trailing-drawdown-pct-values", default="0,5,8,10,12,15")
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


def build_entry_frame(qqq: pd.DataFrame, tqqq: pd.DataFrame, fast_window: int, slow_window: int) -> pd.DataFrame:
    merged = qqq[["date", "close"]].rename(columns={"close": "qqq_close"})
    merged = merged.merge(tqqq[["date", "open", "high", "low", "close"]].rename(columns={"open": "tqqq_open", "high": "tqqq_high", "low": "tqqq_low", "close": "tqqq_close"}), on="date", how="inner")
    merged["fast_ma"] = merged["qqq_close"].rolling(fast_window).mean()
    merged["slow_ma"] = merged["qqq_close"].rolling(slow_window).mean()
    merged["entry_signal"] = (merged["fast_ma"] > merged["slow_ma"]).astype(int)
    merged = merged.dropna(subset=["fast_ma", "slow_ma"]).reset_index(drop=True)
    merged["planned_position"] = merged["entry_signal"].shift(1).fillna(0).astype(int)
    return merged


def run_exit_candidate(
    frame: pd.DataFrame,
    *,
    initial_capital: float,
    switch_cost_bps: float,
    max_hold_days: int,
    trailing_lookback_days: int,
    trailing_drawdown_pct: float,
    allow_mask: pd.Series | None = None,
    hold_mode: str = "hard_exit",
    price_column: str = "tqqq_close",
) -> dict[str, Any]:
    capital = initial_capital
    capitals: list[float] = []
    daily_returns: list[float] = []
    positions: list[str] = []
    hold_days = 0
    active = False
    rolling_peak = 0.0
    exit_override = False
    previous_position = "CASH"
    if allow_mask is None:
        allow_mask = pd.Series([True] * len(frame), index=frame.index)
    else:
        allow_mask = allow_mask.reset_index(drop=True)

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
                rolling_peak = float(row[price_column])
            elif exit_override:
                active = False

        position = "TQQQ" if active else "CASH"
        positions.append(position)
        daily_ret = 0.0
        if idx > 0 and position == "TQQQ":
            prev_close = float(frame.iloc[idx - 1][price_column])
            cur_close = float(row[price_column])
            daily_ret = cur_close / prev_close - 1.0 if prev_close > 0 else 0.0
            hold_days += 1
            rolling_peak = max(rolling_peak, cur_close)
            trailing_exit = False
            time_exit = False
            if trailing_lookback_days > 0 and trailing_drawdown_pct > 0 and hold_days >= trailing_lookback_days and rolling_peak > 0:
                drawdown_from_peak = (rolling_peak - cur_close) / rolling_peak * 100.0
                trailing_exit = drawdown_from_peak >= trailing_drawdown_pct
            if max_hold_days > 0 and hold_days >= max_hold_days:
                time_exit = True
            if trailing_exit or time_exit:
                active = False
                exit_override = True
                if hold_mode == "timer_refresh" and time_exit and not trailing_exit:
                    active = True
                    hold_days = 0
                    rolling_peak = cur_close
                    exit_override = False
                else:
                    hold_days = 0
                    rolling_peak = 0.0
        elif position == "CASH" and desired_active:
            exit_override = False

        if idx > 0 and position != previous_position:
            daily_ret -= float(switch_cost_bps) / 10000.0
        previous_position = position
        capital *= 1.0 + daily_ret
        capitals.append(capital)
        daily_returns.append(daily_ret)

    equity = pd.DataFrame({"date": frame["date"], "equity": capitals, "position": positions})
    yearly = annual_returns(equity)
    total_return_pct = round((capital / initial_capital - 1.0) * 100.0, 2)
    max_dd = round(max_drawdown_pct(equity["equity"]), 2)
    trades = int((equity["position"] != equity["position"].shift(1)).sum())
    invested_days = int((equity["position"] == "TQQQ").sum())
    sharpe_like = round((pd.Series(daily_returns).mean() / pd.Series(daily_returns).std()) if len(daily_returns) > 1 and pd.Series(daily_returns).std() > 0 else 0.0, 3)

    score = (
        total_return_pct
        - max_dd * 2.0
        + float(yearly.get("2022", 0.0)) * 1.8
        + float(yearly.get("2023", 0.0)) * 0.7
        + float(yearly.get("2024", 0.0)) * 0.7
        + float(yearly.get("2025", 0.0)) * 0.5
        + float(yearly.get("2026", 0.0)) * 0.8
    )
    return {
        "params": {
            "max_hold_days": max_hold_days,
            "trailing_lookback_days": trailing_lookback_days,
            "trailing_drawdown_pct": trailing_drawdown_pct,
            "switch_cost_bps": switch_cost_bps,
        },
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd,
        "sharpe_like": sharpe_like,
        "trades": trades,
        "invested_days": invested_days,
        "invested_ratio_pct": round(invested_days / len(equity) * 100.0, 2) if len(equity) else 0.0,
        "yearly_returns_pct": yearly,
        "score": round(score, 4),
    }


def candidate_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    yearly = item.get("yearly_returns_pct", {})
    return (
        float(item.get("score", 0.0)),
        float(yearly.get("2022", 0.0)),
        float(yearly.get("2026", 0.0)),
        -float(item.get("max_drawdown_pct", 0.0)),
    )


def main() -> None:
    args = parse_args()
    qqq = load_df(Path(args.qqq))
    tqqq = load_df(Path(args.tqqq))
    frame = build_entry_frame(qqq, tqqq, int(args.entry_fast_window), int(args.entry_slow_window))

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
                )
                candidates.append(item)

    ranked = sorted(candidates, key=candidate_sort_key, reverse=True)
    payload = {
        "entry_reference": {
            "fast_window": int(args.entry_fast_window),
            "slow_window": int(args.entry_slow_window),
            "description": "QQQ 25/150 MA -> TQQQ/CASH, previous-day signal, 5bps switch cost",
            "reference_total_return_pct": 331.92,
            "reference_max_drawdown_pct": 38.05,
            "reference_yearly_returns_pct": {"2022": 0.0, "2023": 112.74, "2024": 64.41, "2025": 19.00, "2026": 10.79},
        },
        "scan_size": len(candidates),
        "top_candidates": ranked[: int(args.top)],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    for item in ranked[: min(int(args.top), 10)]:
        yearly = item["yearly_returns_pct"]
        params = item["params"]
        print(
            f"score={item['score']:.2f} full={item['total_return_pct']:.2f}% dd={item['max_drawdown_pct']:.2f}% "
            f"2022={float(yearly.get('2022', 0.0)):.2f}% 2023={float(yearly.get('2023', 0.0)):.2f}% "
            f"2024={float(yearly.get('2024', 0.0)):.2f}% 2026={float(yearly.get('2026', 0.0)):.2f}% "
            f"hold={params['max_hold_days']} tlb={params['trailing_lookback_days']} tdd={params['trailing_drawdown_pct']}"
        )


if __name__ == "__main__":
    main()
