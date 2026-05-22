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

from scripts.nasdaq100_cn_strategy_utils import build_allow_mask, load_config, load_strategy_frame


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "cn_nasdaq100_secondary_reentry_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan secondary re-entry overlays for CN Nasdaq-100 ETF.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.paper.cn-nasdaq100-etf.json"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--reentry-delay-values", default="0,1,3,5,10")
    parser.add_argument("--breakout-lookback-values", default="20,40,60")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


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


def max_drawdown_pct(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = ((peak - equity) / peak.replace(0, pd.NA) * 100.0).max(skipna=True)
    return float(drawdown or 0.0)


def build_reentry_masks(frame: pd.DataFrame, breakout_lookback: int) -> dict[str, pd.Series]:
    breakout_high = frame["qqq_close"].rolling(breakout_lookback).max().shift(1)
    breakout = frame["qqq_close"] > breakout_high
    return {
        "breakout": breakout.fillna(False),
        "qqq_strong_and_breakout": (frame["rel_strength_label"].eq("qqq_strong") & breakout).fillna(False),
        "vix_low_and_breakout": (frame["vix_label"].eq("vix_low") & breakout).fillna(False),
        "vix_low_and_qqq_strong_and_breakout": (
            frame["vix_label"].eq("vix_low") & frame["rel_strength_label"].eq("qqq_strong") & breakout
        ).fillna(False),
    }


def simulate_candidate(
    frame: pd.DataFrame,
    allow_mask: pd.Series,
    reentry_mask: pd.Series,
    *,
    initial_capital: float,
    switch_cost_bps: float,
    max_hold_days: int,
    trailing_lookback_days: int,
    trailing_drawdown_pct: float,
    reentry_delay_days: int,
    leverage_enabled: bool,
    leverage_trigger: str,
    leverage_value: float,
) -> dict[str, Any]:
    allow_mask = allow_mask.reset_index(drop=True)
    reentry_mask = reentry_mask.reset_index(drop=True)
    capital = float(initial_capital)
    active = False
    hold_days = 0
    rolling_peak = 0.0
    reentry_wait = 0
    previous_position = "CASH"
    rows: list[dict[str, Any]] = []
    exits: list[dict[str, Any]] = []
    reentry_count = 0
    leverage_days = 0

    for idx, row in frame.iterrows():
        desired_active = int(row["planned_position"]) > 0 and bool(allow_mask.iloc[idx])
        if not desired_active:
            active = False
            hold_days = 0
            rolling_peak = 0.0
            reentry_wait = 0
        else:
            if not active:
                if reentry_wait > 0:
                    reentry_wait -= 1
                elif bool(reentry_mask.iloc[idx]):
                    active = True
                    hold_days = 0
                    rolling_peak = float(row["asset_close"])
                    if previous_position == "CASH":
                        reentry_count += 1

        position = "LONG" if active else "CASH"
        daily_ret = 0.0
        trailing_exit = False
        time_exit = False
        if idx > 0 and position == "LONG":
            prev_close = float(frame.iloc[idx - 1]["asset_close"])
            cur_close = float(row["asset_close"])
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
                reentry_wait = max(int(reentry_delay_days), 0)
                hold_days = 0
                rolling_peak = 0.0
                exits.append(
                    {
                        "date": str(pd.Timestamp(row["date"]).date()),
                        "reason": "trailing" if trailing_exit else "time",
                    }
                )
        elif position == "CASH" and desired_active:
            # Keep the re-entry countdown alive while the base signal remains valid.
            pass

        if idx > 0 and position != previous_position:
            daily_ret -= float(switch_cost_bps) / 10000.0

        leverage = 1.0 if position == "LONG" else 0.0
        if position == "LONG" and leverage_enabled and leverage_value > 1.0:
            trigger_hit = False
            if leverage_trigger == "vix_low":
                trigger_hit = str(row["vix_label"]) == "vix_low"
            elif leverage_trigger == "qqq_strong":
                trigger_hit = str(row["rel_strength_label"]) == "qqq_strong"
            elif leverage_trigger == "vix_low_and_qqq_strong":
                trigger_hit = str(row["vix_label"]) == "vix_low" and str(row["rel_strength_label"]) == "qqq_strong"
            if trigger_hit:
                cost_component = 0.0
                if idx > 0 and previous_position != "LONG":
                    cost_component = daily_ret - ((float(row["asset_close"]) / float(frame.iloc[idx - 1]["asset_close"])) - 1.0)
                asset_ret = daily_ret - cost_component
                daily_ret = asset_ret * leverage_value + cost_component
                leverage = leverage_value
                leverage_days += 1

        previous_position = position
        capital *= 1.0 + daily_ret
        rows.append(
            {
                "date": row["date"],
                "position": position,
                "daily_return": daily_ret,
                "equity": capital,
                "reentry_wait": reentry_wait,
                "vix_label": row["vix_label"],
                "ixic_trend_label": row["ixic_trend_label"],
                "rel_strength_label": row["rel_strength_label"],
                "leverage": leverage,
                "asset_close": float(row["asset_close"]),
            }
        )

    equity = pd.DataFrame(rows)
    yearly = annual_returns(equity)
    total_return_pct = round((capital / float(initial_capital) - 1.0) * 100.0, 2)
    max_dd = round(max_drawdown_pct(equity["equity"]), 2) if not equity.empty else 0.0
    score = total_return_pct - 2.0 * max_dd + float(yearly.get("2026", 0.0) or 0.0)
    return {
        "summary": {
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd,
            "yearly_returns_pct": yearly,
            "trades": int((equity["position"] != equity["position"].shift(1)).sum()) if not equity.empty else 0,
            "reentry_count": int(reentry_count),
            "reentry_exits": len(exits),
            "leverage_days": int(leverage_days),
        },
        "score": round(score, 4),
        "exits": exits,
    }


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    frame = load_strategy_frame(config).copy().reset_index(drop=True)
    allow_mask = build_allow_mask(frame, config).reset_index(drop=True)
    leverage_enabled = bool(config.get("conditional_leverage_enabled", False))
    leverage_trigger = str(config.get("conditional_leverage_trigger", "none"))
    leverage_value = float(config.get("conditional_leverage_value", 1.0))

    masks = {
        "breakout": lambda n: build_reentry_masks(frame, n)["breakout"],
        "qqq_strong_and_breakout": lambda n: build_reentry_masks(frame, n)["qqq_strong_and_breakout"],
        "vix_low_and_breakout": lambda n: build_reentry_masks(frame, n)["vix_low_and_breakout"],
        "vix_low_and_qqq_strong_and_breakout": lambda n: build_reentry_masks(frame, n)["vix_low_and_qqq_strong_and_breakout"],
    }

    results: list[dict[str, Any]] = []
    for breakout_lookback in parse_int_list(args.breakout_lookback_values):
        reentry_masks = build_reentry_masks(frame, breakout_lookback)
        for reentry_delay in parse_int_list(args.reentry_delay_values):
            for name, mask in reentry_masks.items():
                result = simulate_candidate(
                    frame,
                    allow_mask,
                    mask,
                    initial_capital=float(config.get("initial_capital", 1000.0)),
                    switch_cost_bps=float(config.get("switch_cost_bps", 10.0)),
                    max_hold_days=int(config.get("max_hold_days", 90)),
                    trailing_lookback_days=int(config.get("trailing_lookback_days", 10)),
                    trailing_drawdown_pct=float(config.get("trailing_drawdown_pct", 5.0)),
                    reentry_delay_days=int(reentry_delay),
                    leverage_enabled=leverage_enabled,
                    leverage_trigger=leverage_trigger,
                    leverage_value=leverage_value,
                )
                results.append(
                    {
                        "breakout_lookback_days": breakout_lookback,
                        "reentry_delay_days": reentry_delay,
                        "reentry_rule": name,
                        "summary": result["summary"],
                        "score": result["score"],
                    }
                )

    ranked = sorted(results, key=lambda item: (float(item["score"]), float(item["summary"]["total_return_pct"])), reverse=True)
    payload = {
        "baseline": {
            "entry_fast_window": int(config.get("entry_fast_window", 25)),
            "entry_slow_window": int(config.get("entry_slow_window", 200)),
            "max_hold_days": int(config.get("max_hold_days", 90)),
            "trailing_lookback_days": int(config.get("trailing_lookback_days", 10)),
            "trailing_drawdown_pct": float(config.get("trailing_drawdown_pct", 5.0)),
            "reentry_note": "secondary re-entry after exit, not hold extension",
        },
        "top_candidates": ranked[:20],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    for item in ranked[:10]:
        s = item["summary"]
        print(
            item["reentry_rule"],
            item["breakout_lookback_days"],
            item["reentry_delay_days"],
            s["total_return_pct"],
            s["max_drawdown_pct"],
            s["yearly_returns_pct"],
            "reentries",
            s["reentry_count"],
        )


if __name__ == "__main__":
    main()
