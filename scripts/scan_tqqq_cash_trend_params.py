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
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "tqqq_cash_trend_param_scan.json"


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan TQQQ/CASH daily trend parameters.")
    parser.add_argument("--tqqq", default=str(DEFAULT_PUBLIC_DIR / "TQQQ-1d.feather"))
    parser.add_argument("--qqq", default=str(DEFAULT_PUBLIC_DIR / "QQQ-1d.feather"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--fast-windows", default="10,15,20,25,30,40")
    parser.add_argument("--slow-windows", default="60,80,100,120,150,200")
    parser.add_argument("--momentum-windows", default="0,5,10,20")
    parser.add_argument("--momentum-thresholds-pct", default="0,1,2,3,5")
    parser.add_argument("--switch-cost-bps-values", default="5,10")
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


def run_candidate(
    qqq: pd.DataFrame,
    tqqq: pd.DataFrame,
    *,
    initial_capital: float,
    fast_window: int,
    slow_window: int,
    momentum_window: int,
    momentum_threshold_pct: float,
    switch_cost_bps: float,
) -> dict[str, Any] | None:
    if fast_window >= slow_window:
        return None

    merged = qqq[["date", "close"]].rename(columns={"close": "qqq_close"})
    merged = merged.merge(tqqq[["date", "close"]].rename(columns={"close": "tqqq_close"}), on="date", how="inner")
    merged["fast_ma"] = merged["qqq_close"].rolling(fast_window).mean()
    merged["slow_ma"] = merged["qqq_close"].rolling(slow_window).mean()
    merged["trend"] = (merged["fast_ma"] > merged["slow_ma"]).astype(int)

    if momentum_window > 0:
        merged["momentum_pct"] = (merged["qqq_close"] / merged["qqq_close"].shift(momentum_window) - 1.0) * 100.0
        merged["trend"] = merged["trend"] * (merged["momentum_pct"] >= momentum_threshold_pct).astype(int)
    else:
        merged["momentum_pct"] = 0.0
        if momentum_threshold_pct > 0:
            merged["trend"] = 0

    merged = merged.dropna(subset=["fast_ma", "slow_ma"]).reset_index(drop=True)
    merged["position"] = merged["trend"].shift(1).fillna(0).map(lambda value: "TQQQ" if int(value) > 0 else "CASH")

    capital = initial_capital
    capitals: list[float] = []
    daily_returns: list[float] = []
    previous_position = "CASH"

    for idx, row in merged.iterrows():
        position = str(row["position"])
        daily_ret = 0.0
        if idx > 0 and position == "TQQQ":
            prev = float(merged.iloc[idx - 1]["tqqq_close"])
            cur = float(row["tqqq_close"])
            daily_ret = cur / prev - 1.0 if prev > 0 else 0.0
        if idx > 0 and position != previous_position:
            daily_ret -= float(switch_cost_bps) / 10000.0
        previous_position = position
        capital *= 1.0 + daily_ret
        capitals.append(capital)
        daily_returns.append(daily_ret)

    equity = pd.DataFrame({"date": merged["date"], "equity": capitals, "position": merged["position"]})
    yearly = annual_returns(equity)
    total_return_pct = round((capital / initial_capital - 1.0) * 100.0, 2)
    max_dd = round(max_drawdown_pct(equity["equity"]), 2)
    sharpe_like = round((pd.Series(daily_returns).mean() / pd.Series(daily_returns).std()) if len(daily_returns) > 1 and pd.Series(daily_returns).std() > 0 else 0.0, 3)
    trades = int((equity["position"] != equity["position"].shift(1)).sum())
    invested_days = int((equity["position"] == "TQQQ").sum())

    score = (
        total_return_pct
        - max_dd * 2.0
        + float(yearly.get("2023", 0.0)) * 0.8
        + float(yearly.get("2024", 0.0)) * 0.8
        + float(yearly.get("2025", 0.0)) * 0.6
        + float(yearly.get("2026", 0.0)) * 1.0
        + float(yearly.get("2022", 0.0)) * 1.5
    )

    return {
        "params": {
            "fast_window": fast_window,
            "slow_window": slow_window,
            "momentum_window": momentum_window,
            "momentum_threshold_pct": momentum_threshold_pct,
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
        float(yearly.get("2026", 0.0)),
        float(yearly.get("2022", 0.0)),
        -float(item.get("max_drawdown_pct", 0.0)),
    )


def main() -> None:
    args = parse_args()
    qqq = load_df(Path(args.qqq))
    tqqq = load_df(Path(args.tqqq))

    candidates: list[dict[str, Any]] = []
    for fast_window in parse_int_list(args.fast_windows):
        for slow_window in parse_int_list(args.slow_windows):
            if fast_window >= slow_window:
                continue
            for momentum_window in parse_int_list(args.momentum_windows):
                for momentum_threshold_pct in parse_float_list(args.momentum_thresholds_pct):
                    for switch_cost_bps in parse_float_list(args.switch_cost_bps_values):
                        candidate = run_candidate(
                            qqq,
                            tqqq,
                            initial_capital=float(args.initial_capital),
                            fast_window=int(fast_window),
                            slow_window=int(slow_window),
                            momentum_window=int(momentum_window),
                            momentum_threshold_pct=float(momentum_threshold_pct),
                            switch_cost_bps=float(switch_cost_bps),
                        )
                        if candidate is not None:
                            candidates.append(candidate)

    ranked = sorted(candidates, key=candidate_sort_key, reverse=True)
    payload = {
        "baseline_reference": {
            "description": "QQQ 20/100 MA, TQQQ/CASH, previous-day signal, 5bps switch cost",
            "total_return_pct": 150.54,
            "max_drawdown_pct": 41.05,
            "yearly_returns_pct": {"2022": -35.29, "2023": 88.05, "2024": 64.41, "2025": 15.95, "2026": 15.30},
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
            f"fast={params['fast_window']} slow={params['slow_window']} mw={params['momentum_window']} "
            f"mt={params['momentum_threshold_pct']} cost={params['switch_cost_bps']}"
        )


if __name__ == "__main__":
    main()
