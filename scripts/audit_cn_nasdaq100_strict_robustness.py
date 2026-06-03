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

from scripts.cn_nasdaq100_strict_utils import load_config, load_strict_frame, run_strict_path, summarize_path  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "cn_nasdaq100_strict_robustness_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness audit for the strict CN Nasdaq100 ETF candidate.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.paper.cn-nasdaq100-etf.json"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--extra-switch-cost-bps-values", default="2.5,5.0,10.0")
    return parser.parse_args()


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def strict_candidate_config(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    config["conditional_leverage_enabled"] = False
    config["conditional_leverage_value"] = 1.0
    config["tiered_leverage_enabled"] = True
    config["tiered_leverage_rules"] = [
        {
            "vix_label": "vix_normal",
            "rel_strength_label": "qqq_strong",
            "leverage": 2.0,
        },
        {
            "vix_label": "vix_normal",
            "rel_strength_label": "qqq_neutral",
            "leverage": 1.5,
        },
    ]
    config["entry_fast_window"] = 21
    config["entry_slow_window"] = 200
    config["regime_filter"] = "ixic_filter"
    config["max_hold_days"] = 120
    config["trailing_lookback_days"] = 4
    config["trailing_drawdown_pct"] = 4.0
    return config


def extract_trades(path: pd.DataFrame) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    prev_row: pd.Series | None = None

    for _, row in path.iterrows():
        entered = bool(row["entered_today"])
        exited = bool(row["exited_today"])

        if entered:
            pre_entry_capital = float(prev_row["capital"]) if prev_row is not None else float(row["capital"]) / max(1e-12, 1.0 + float(row["daily_return"]))
            active = {
                "entry_date": str(pd.Timestamp(row["date"]).date()),
                "entry_capital": float(pre_entry_capital),
                "entry_price": round(float(row["asset_open"]), 4),
                "vix_label": str(row["vix_label"]),
                "ixic_trend_label": str(row["ixic_trend_label"]),
                "rel_strength_label": str(row["rel_strength_label"]),
                "leverage": float(row["leverage"]),
            }

        if exited and active is not None and prev_row is not None:
            exit_capital = float(prev_row["capital"])
            trade_return_pct = ((exit_capital / float(active["entry_capital"])) - 1.0) * 100.0 if active["entry_capital"] else 0.0
            trades.append(
                {
                    "entry_date": active["entry_date"],
                    "exit_date": str(pd.Timestamp(row["date"]).date()),
                    "entry_price": active["entry_price"],
                    "exit_price": round(float(row["asset_open"]), 4),
                    "trade_return_pct": round(trade_return_pct, 2),
                    "vix_label": active["vix_label"],
                    "ixic_trend_label": active["ixic_trend_label"],
                    "rel_strength_label": active["rel_strength_label"],
                    "leverage": active["leverage"],
                }
            )
            active = None

        prev_row = row

    if active is not None and prev_row is not None:
        exit_capital = float(prev_row["capital"])
        trade_return_pct = ((exit_capital / float(active["entry_capital"])) - 1.0) * 100.0 if active["entry_capital"] else 0.0
        trades.append(
            {
                "entry_date": active["entry_date"],
                "exit_date": str(pd.Timestamp(prev_row["date"]).date()),
                "entry_price": active["entry_price"],
                "exit_price": round(float(prev_row["asset_close"]), 4),
                "trade_return_pct": round(trade_return_pct, 2),
                "vix_label": active["vix_label"],
                "ixic_trend_label": active["ixic_trend_label"],
                "rel_strength_label": active["rel_strength_label"],
                "leverage": active["leverage"],
            }
        )

    return trades


def main() -> None:
    args = parse_args()
    config = strict_candidate_config(Path(args.config))
    frame = load_strict_frame(config)
    path = run_strict_path(frame, config).reset_index(drop=True)
    summary = summarize_path(path, initial_capital=float(config.get("initial_capital", 1000.0)))
    trades = extract_trades(path)

    trade_returns = pd.Series([float(trade["trade_return_pct"]) / 100.0 for trade in trades], dtype=float)
    full_growth = float((1.0 + trade_returns).prod()) if not trade_returns.empty else 1.0

    leave_one_out: list[dict[str, Any]] = []
    for idx, trade in enumerate(trades):
        remaining = trade_returns.drop(trade_returns.index[idx])
        remaining_growth = float((1.0 + remaining).prod()) if not remaining.empty else 1.0
        leave_one_out.append(
            {
                "removed_entry_date": trade["entry_date"],
                "removed_exit_date": trade["exit_date"],
                "removed_trade_return_pct": trade["trade_return_pct"],
                "remaining_total_return_pct": round((remaining_growth - 1.0) * 100.0, 2),
                "relative_drop_pct": round(((full_growth - remaining_growth) / full_growth) * 100.0, 2) if full_growth > 0 else 0.0,
            }
        )

    cost_sensitivity: list[dict[str, Any]] = []
    for extra_cost in parse_floats(args.extra_switch_cost_bps_values):
        stressed = dict(config)
        stressed["switch_cost_bps"] = float(config.get("switch_cost_bps", 10.0)) + float(extra_cost)
        stressed_path = run_strict_path(frame, stressed)
        stressed_summary = summarize_path(stressed_path, initial_capital=float(stressed.get("initial_capital", 1000.0)))
        cost_sensitivity.append(
            {
                "extra_switch_cost_bps": float(extra_cost),
                "total_return_pct": stressed_summary["total_return_pct"],
                "max_drawdown_pct": stressed_summary["max_drawdown_pct"],
                "yearly_returns_pct": stressed_summary["yearly_returns_pct"],
                "return_delta_pct": round(float(stressed_summary["total_return_pct"]) - float(summary["total_return_pct"]), 2),
                "dd_delta_pct": round(float(stressed_summary["max_drawdown_pct"]) - float(summary["max_drawdown_pct"]), 2),
            }
        )

    leverage_breakdown: dict[str, Any] = {}
    if trades:
        by_lev = pd.DataFrame(trades).groupby("leverage")["trade_return_pct"]
        for leverage, series in by_lev:
            leverage_breakdown[str(leverage)] = {
                "trades": int(series.count()),
                "mean_trade_return_pct": round(float(series.mean()), 2),
                "median_trade_return_pct": round(float(series.median()), 2),
                "positive_trade_ratio_pct": round(float((series > 0).mean() * 100.0), 2),
            }

    payload = {
        "candidate": {
            "entry_fast_window": 21,
            "entry_slow_window": 200,
            "regime_filter": "ixic_filter",
            "max_hold_days": 120,
            "trailing_lookback_days": 4,
            "trailing_drawdown_pct": 4.0,
            "tiered_leverage_rules": config["tiered_leverage_rules"],
        },
        "baseline_summary": summary,
        "trade_summary": {
            "trade_count": len(trades),
            "best_trade_return_pct": round(max((float(t["trade_return_pct"]) for t in trades), default=0.0), 2),
            "worst_trade_return_pct": round(min((float(t["trade_return_pct"]) for t in trades), default=0.0), 2),
            "median_trade_return_pct": round(float(pd.Series([float(t["trade_return_pct"]) for t in trades]).median()) if trades else 0.0, 2),
            "positive_trade_ratio_pct": round(float((pd.Series([float(t["trade_return_pct"]) for t in trades]) > 0).mean() * 100.0) if trades else 0.0, 2),
        },
        "leverage_breakdown": leverage_breakdown,
        "leave_one_out": sorted(leave_one_out, key=lambda item: float(item["remaining_total_return_pct"])),
        "yearly_breakdown": summary["yearly_returns_pct"],
        "cost_sensitivity": cost_sensitivity,
        "trades": trades,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    print(json.dumps(payload["baseline_summary"], ensure_ascii=False))
    print(json.dumps(payload["trade_summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
