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

from scripts.nasdaq100_cn_strategy_utils import load_config, load_strategy_frame, run_full_strategy, run_full_strategy_path


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "cn_nasdaq100_baseline_robustness_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness audit for the formal CN Nasdaq-100 ETF baseline.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.paper.cn-nasdaq100-etf.json"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--extra-switch-cost-bps-values", default="2.5,5.0,10.0")
    return parser.parse_args()


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def annual_returns_from_equity(equity: pd.DataFrame) -> dict[str, float]:
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


def extract_trades(path: pd.DataFrame) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    in_trade = False
    entry_idx = None
    entry_date = None
    entry_equity = None
    entry_price = None
    for idx, row in path.iterrows():
        position = str(row["position"])
        if not in_trade and position == "LONG":
            in_trade = True
            entry_idx = idx
            entry_date = row["date"]
            entry_equity = float(path.iloc[idx - 1]["capital"]) if idx > 0 else float(row["capital"]) / (1.0 + float(row["daily_return"])) if (1.0 + float(row["daily_return"])) != 0 else float(row["capital"])
            entry_price = float(row.get("execution_price", row.get("asset_close", 0.0)) or 0.0)
        elif in_trade and position == "CASH":
            exit_idx = idx
            exit_date = row["date"]
            end_equity = float(path.iloc[idx - 1]["capital"]) if idx > 0 else float(row["capital"])
            trade_return_pct = ((end_equity / float(entry_equity or 1.0)) - 1.0) * 100.0 if entry_equity else 0.0
            exit_price = float(path.iloc[idx - 1].get("execution_price", path.iloc[idx - 1].get("asset_close", 0.0)) or 0.0) if idx > 0 else float(row.get("execution_price", row.get("asset_close", 0.0)) or 0.0)
            trades.append(
                {
                    "entry_idx": int(entry_idx or 0),
                    "exit_idx": int(exit_idx),
                    "entry_date": str(pd.Timestamp(entry_date).date()) if entry_date is not None else "",
                    "exit_date": str(pd.Timestamp(exit_date).date()),
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "trade_return_pct": round(trade_return_pct, 2),
                    "vix_label": str(path.iloc[entry_idx]["vix_label"]) if entry_idx is not None else "",
                    "ixic_trend_label": str(path.iloc[entry_idx]["ixic_trend_label"]) if entry_idx is not None else "",
                    "rel_strength_label": str(path.iloc[entry_idx]["rel_strength_label"]) if entry_idx is not None else "",
                }
            )
            in_trade = False
            entry_idx = None
            entry_date = None
            entry_equity = None
            entry_price = None
    if in_trade and entry_idx is not None:
        end_equity = float(path.iloc[-1]["capital"])
        trade_return_pct = ((end_equity / float(entry_equity or 1.0)) - 1.0) * 100.0 if entry_equity else 0.0
        trades.append(
            {
                "entry_idx": int(entry_idx),
                "exit_idx": int(len(path) - 1),
                "entry_date": str(pd.Timestamp(entry_date).date()) if entry_date is not None else "",
                "exit_date": str(pd.Timestamp(path.iloc[-1]["date"]).date()),
                "entry_price": round(float(entry_price or 0.0), 4),
                "exit_price": round(float(path.iloc[-1].get("execution_price", path.iloc[-1].get("asset_close", 0.0)) or 0.0), 4),
                "trade_return_pct": round(trade_return_pct, 2),
                "vix_label": str(path.iloc[entry_idx]["vix_label"]) if entry_idx is not None else "",
                "ixic_trend_label": str(path.iloc[entry_idx]["ixic_trend_label"]) if entry_idx is not None else "",
                "rel_strength_label": str(path.iloc[entry_idx]["rel_strength_label"]) if entry_idx is not None else "",
            }
        )
    return trades


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    frame = load_strategy_frame(config)
    summary = run_full_strategy(frame, config)
    path = run_full_strategy_path(frame, config).reset_index(drop=True)
    trades = extract_trades(path)

    leave_one_out: list[dict[str, Any]] = []
    trade_returns = pd.Series([float(trade["trade_return_pct"]) / 100.0 for trade in trades], dtype=float)
    full_growth = float((1.0 + trade_returns).prod()) if not trade_returns.empty else 1.0
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

    yearly_breakdown = summary["yearly_returns_pct"]
    cost_sensitivity: list[dict[str, Any]] = []
    for extra_cost in parse_floats(args.extra_switch_cost_bps_values):
        stressed_config = dict(config)
        stressed_config["switch_cost_bps"] = float(config.get("switch_cost_bps", 10.0)) + float(extra_cost)
        stressed_frame = load_strategy_frame(stressed_config)
        stressed_summary = run_full_strategy(stressed_frame, stressed_config)
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

    payload = {
        "baseline_summary": {
            "total_return_pct": summary["total_return_pct"],
            "max_drawdown_pct": summary["max_drawdown_pct"],
            "yearly_returns_pct": summary["yearly_returns_pct"],
            "trades": summary["trades"],
        },
        "trades": trades,
        "trade_summary": {
            "trade_count": len(trades),
            "best_trade_return_pct": round(max((float(t["trade_return_pct"]) for t in trades), default=0.0), 2),
            "worst_trade_return_pct": round(min((float(t["trade_return_pct"]) for t in trades), default=0.0), 2),
            "median_trade_return_pct": round(float(pd.Series([float(t["trade_return_pct"]) for t in trades]).median()) if trades else 0.0, 2),
            "positive_trade_ratio_pct": round(float((pd.Series([float(t["trade_return_pct"]) for t in trades]) > 0).mean() * 100.0) if trades else 0.0, 2),
        },
        "leave_one_out": sorted(leave_one_out, key=lambda item: float(item["remaining_total_return_pct"])),
        "yearly_breakdown": yearly_breakdown,
        "cost_sensitivity": cost_sensitivity,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    print(json.dumps(payload["baseline_summary"], ensure_ascii=False))
    print(json.dumps(payload["trade_summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
