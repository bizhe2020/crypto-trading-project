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

from scripts.nasdaq100_cn_strategy_utils import load_config, load_strategy_frame, run_full_strategy_path


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "cn_nasdaq100_tiered_sizing_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan tiered sizing overlays for CN Nasdaq-100 ETF.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.paper.cn-nasdaq100-etf.json"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--weak-low-leverage-values", default="1.25,1.5,1.75,2.0")
    parser.add_argument("--mid-leverage-values", default="1.0,1.25,1.5")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


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


def run_tier_overlay(
    base_path: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    weak_low_leverage: float,
    mid_leverage: float,
    initial_capital: float,
) -> dict[str, Any]:
    capital = float(initial_capital)
    rows: list[dict[str, Any]] = []
    previous_position = "CASH"
    tier_hits = {"vix_low_strong": 0, "vix_low_weak": 0, "vix_normal_strong": 0}

    for idx, row in base_path.iterrows():
        active = str(row["position"]) == "LONG"
        daily_ret = float(row["daily_return"])
        leverage = 1.0 if active else 0.0
        if active:
            vix_low = str(frame.iloc[idx]["vix_label"]) == "vix_low"
            qqq_strong = str(frame.iloc[idx]["rel_strength_label"]) == "qqq_strong"
            vix_normal = str(frame.iloc[idx]["vix_label"]) == "vix_normal"
            if vix_low and qqq_strong:
                leverage = 2.0
                tier_hits["vix_low_strong"] += 1
            elif vix_low and not qqq_strong:
                leverage = float(weak_low_leverage)
                tier_hits["vix_low_weak"] += 1
            elif vix_normal and qqq_strong:
                leverage = float(mid_leverage)
                tier_hits["vix_normal_strong"] += 1

            if leverage > 1.0:
                cost_component = 0.0
                if idx > 0 and str(base_path.iloc[idx - 1]["position"]) != "LONG":
                    prev_close = float(base_path.iloc[idx - 1]["asset_close"])
                    cur_close = float(row["asset_close"])
                    cost_component = daily_ret - (cur_close / prev_close - 1.0 if prev_close > 0 else 0.0)
                asset_ret = daily_ret - cost_component
                daily_ret = asset_ret * leverage + cost_component

        previous_position = str(row["position"])
        capital *= 1.0 + daily_ret
        rows.append({"date": row["date"], "equity": capital, "position": row["position"], "daily_return": daily_ret, "leverage": leverage})

    equity = pd.DataFrame(rows)
    yearly = annual_returns(equity[["date", "equity"]])
    total_return_pct = round((capital / float(initial_capital) - 1.0) * 100.0, 2)
    max_dd = round(max_drawdown_pct(equity["equity"]), 2)
    score = total_return_pct - 2.0 * max_dd + float(yearly.get("2026", 0.0) or 0.0)
    return {
        "summary": {
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd,
            "yearly_returns_pct": yearly,
            "trades": int((equity["position"] != equity["position"].shift(1)).sum()),
            "leveraged_days": int((equity["leverage"] > 1.0).sum()),
            "tier_hits": tier_hits,
        },
        "score": round(score, 4),
    }


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    base_config = dict(config)
    base_config["conditional_leverage_enabled"] = False
    frame = load_strategy_frame(base_config).reset_index(drop=True)
    base_path = run_full_strategy_path(frame, base_config).reset_index(drop=True)

    # Current baseline for comparison.
    baseline = run_tier_overlay(
        base_path,
        frame,
        weak_low_leverage=2.0,
        mid_leverage=1.0,
        initial_capital=float(config.get("initial_capital", 1000.0)),
    )

    results: list[dict[str, Any]] = []
    for weak_low_leverage in parse_float_list(args.weak_low_leverage_values):
        for mid_leverage in parse_float_list(args.mid_leverage_values):
            result = run_tier_overlay(
                base_path,
                frame,
                weak_low_leverage=weak_low_leverage,
                mid_leverage=mid_leverage,
                initial_capital=float(config.get("initial_capital", 1000.0)),
            )
            results.append(
                {
                    "weak_low_leverage": weak_low_leverage,
                    "mid_leverage": mid_leverage,
                    "summary": result["summary"],
                    "score": result["score"],
                }
            )

    ranked = sorted(results, key=lambda item: (float(item["score"]), float(item["summary"]["total_return_pct"])), reverse=True)
    payload = {
        "baseline_2x": baseline,
        "top_candidates": ranked[:20],
        "scan_meta": {
            "weak_low_leverage_values": parse_float_list(args.weak_low_leverage_values),
            "mid_leverage_values": parse_float_list(args.mid_leverage_values),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    print("baseline_2x", baseline["summary"])
    for item in ranked[:10]:
        s = item["summary"]
        print(item["weak_low_leverage"], item["mid_leverage"], s["total_return_pct"], s["max_drawdown_pct"], s["yearly_returns_pct"], "tier_hits", s["tier_hits"])


if __name__ == "__main__":
    main()
