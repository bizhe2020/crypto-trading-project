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
from scripts.scan_tqqq_cash_exit_profiles import run_exit_candidate


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "cn_nasdaq100_param_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan parameters for the CN Nasdaq-100 ETF strategy.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.paper.cn-nasdaq100-etf.json"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--entry-fast-values", default="10,15,20,25,30,40")
    parser.add_argument("--entry-slow-values", default="100,150,200")
    parser.add_argument("--max-hold-values", default="0,30,60,90,120,180")
    parser.add_argument("--trailing-lookback-values", default="0,10,20,30")
    parser.add_argument("--trailing-drawdown-values", default="0,5,8,10,12,15")
    parser.add_argument("--switch-cost-bps", type=float, default=10.0)
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def annual_returns(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    frame = df.copy()
    frame["year"] = frame["date"].dt.year.astype(str)
    out: dict[str, float] = {}
    for year, group in frame.groupby("year"):
        start = float(group.iloc[0]["capital"])
        end = float(group.iloc[-1]["capital"])
        out[year] = round((end / start - 1.0) * 100.0, 2) if start > 0 else 0.0
    return out


def score(summary: dict[str, Any]) -> float:
    return float(summary.get("total_return_pct", 0.0)) - 2.0 * float(summary.get("max_drawdown_pct", 0.0)) + float(summary.get("yearly_returns_pct", {}).get("2026", 0.0) or 0.0)


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    frame = load_strategy_frame(config)
    allow_mask = build_allow_mask(frame, config)
    results: list[dict[str, Any]] = []

    for fast in parse_ints(args.entry_fast_values):
        for slow in parse_ints(args.entry_slow_values):
            if fast >= slow:
                continue
            for max_hold in parse_ints(args.max_hold_values):
                for lookback in parse_ints(args.trailing_lookback_values):
                    for drawdown in parse_floats(args.trailing_drawdown_values):
                        if lookback == 0 and drawdown > 0:
                            continue
                        if lookback > 0 and drawdown == 0:
                            continue
                        local = frame.copy()
                        local["fast_ma"] = local["qqq_close"].rolling(fast).mean()
                        local["slow_ma"] = local["qqq_close"].rolling(slow).mean()
                        local = local.dropna(subset=["fast_ma", "slow_ma"]).reset_index(drop=True)
                        local["planned_position"] = (local["fast_ma"] > local["slow_ma"]).shift(1).fillna(0).astype(int)
                        result = run_exit_candidate(
                            local,
                            initial_capital=float(config.get("initial_capital", 1000.0)),
                            switch_cost_bps=float(args.switch_cost_bps),
                            max_hold_days=int(max_hold),
                            trailing_lookback_days=int(lookback),
                            trailing_drawdown_pct=float(drawdown),
                            allow_mask=allow_mask.loc[local.index].reset_index(drop=True),
                            hold_mode=str(config.get("hold_mode", "hard_exit")),
                            price_column="asset_close",
                        )
                        results.append(
                            {
                                "entry_fast_window": fast,
                                "entry_slow_window": slow,
                                "max_hold_days": max_hold,
                                "trailing_lookback_days": lookback,
                                "trailing_drawdown_pct": drawdown,
                                "summary": result,
                                "score": round(score(result), 4),
                            }
                        )

    ranked = sorted(results, key=lambda item: (float(item["score"]), float(item["summary"]["total_return_pct"])), reverse=True)
    payload = {
        "scan_size": len(results),
        "top_candidates": ranked[:20],
        "frozen_reference": {
            "total_return_pct": 211.87,
            "max_drawdown_pct": 6.7,
            "yearly_returns_pct": {"2022": 0.0, "2023": 50.02, "2024": 57.57, "2025": 24.68, "2026": 5.18},
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    for item in ranked[:10]:
        s = item["summary"]
        print(item["entry_fast_window"], item["entry_slow_window"], item["max_hold_days"], item["trailing_lookback_days"], item["trailing_drawdown_pct"], s["total_return_pct"], s["max_drawdown_pct"], s["yearly_returns_pct"])


if __name__ == "__main__":
    main()
