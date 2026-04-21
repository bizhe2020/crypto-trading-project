#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.okx_executor import ExecutorConfig
from strategy.scalp_robust_v2_core import (
    ScalpRobustEngine,
    align_timeframes,
    build_precomputed_state,
    dataframe_to_candles,
)


OPTIMIZED_OVERRIDES = {
    "allow_bear_strong_short": True,
    "bear_strong_short_pullback_window": 10,
    "bear_strong_short_retrace_min_ob_fill_pct": 0.8,
    "bear_strong_short_entry_min_ob_fill_pct": 0.55,
}

BASELINE_RESET_OVERRIDES = {
    "allow_bear_strong_short": True,
    "bear_strong_short_pullback_window": None,
    "bear_strong_short_sl_buffer_pct": None,
    "bear_strong_short_retrace_min_ob_fill_pct": None,
    "bear_strong_short_entry_min_ob_fill_pct": None,
    "bear_strong_short_rr_ratio_override": None,
    "bear_strong_short_trail_style_override": None,
    "bear_strong_short_max_hold_bars": None,
    "bear_strong_short_atr_activation_rr": None,
    "bear_strong_short_atr_loose_multiplier": None,
    "bear_strong_short_atr_normal_multiplier": None,
    "bear_strong_short_atr_tight_multiplier": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare bear_strong short optimization cases")
    parser.add_argument("--config", default="config/config.live.5x-3pct.json")
    parser.add_argument(
        "--data-15m",
        default="/Users/laoji/projects/crypto-trading-project/deployment/data/okx/futures/BTC_USDT_USDT-15m-futures.feather",
    )
    parser.add_argument(
        "--data-4h",
        default="/Users/laoji/projects/crypto-trading-project/deployment/data/okx/futures/BTC_USDT_USDT-4h-futures.feather",
    )
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-04-18")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def load_filtered_feather(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    df = pd.read_feather(path)
    if "date" in df.columns:
        ts = pd.to_datetime(df["date"], utc=True)
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        raise ValueError(f"Unsupported dataframe columns for {path}")
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
    return df[(ts >= start_ts) & (ts <= end_ts)].reset_index(drop=True)


def run_case(
    name: str,
    strategy_config: Any,
    c4h: list[Any],
    c15m: list[Any],
    mapping: list[int],
    precomputed: Any,
    start_date: str,
) -> dict[str, Any]:
    start_dt = pd.Timestamp(start_date, tz="UTC")
    start_idx = next((i for i, candle in enumerate(c15m) if candle.ts >= start_dt.timestamp()), 0)
    engine = ScalpRobustEngine(c4h, c15m, mapping, precomputed, strategy_config)
    engine.evaluate_range(start_idx + 100, len(c15m) - 1)
    if engine.position:
        engine.close_position(
            len(c15m) - 1,
            "end_of_data",
            entry_risk_regime=engine.position.entry_risk_regime,
            trail_style=engine.position.trail_style,
        )

    metrics = engine.compute_metrics()
    trades = pd.DataFrame([t.__dict__ for t in engine.trades])
    if not trades.empty:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
        trades["hold_hours"] = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 3600.0

    return {
        "name": name,
        "overall": {
            "total_return_pct": round(float(metrics["total_return_pct"]), 2),
            "max_drawdown_pct": round(float(metrics["max_drawdown_pct"]), 2),
            "sharpe_ratio": round(float(metrics["sharpe_ratio"]), 3),
            "profit_factor": round(float(metrics["profit_factor"]), 3),
            "win_rate": round(float(metrics["win_rate"]), 1),
            "total_trades": int(metrics["total_trades"]),
            "final_capital": round(float(metrics["final_capital"]), 2),
        },
        "bear_strong_short": summarize_bucket(
            trades,
            entry_risk_regime="bear_strong",
            direction="BEAR",
        ),
        "regime_direction": summarize_by_regime_direction(trades),
    }


def summarize_bucket(
    trades: pd.DataFrame,
    *,
    entry_risk_regime: str,
    direction: str,
) -> dict[str, Any]:
    if trades.empty:
        return {
            "trades": 0,
            "pnl": 0.0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "avg_rr": 0.0,
            "stop_loss_pct": 0.0,
            "target_pct": 0.0,
            "avg_hold_hours": 0.0,
            "median_hold_hours": 0.0,
        }
    bucket = trades[
        (trades["entry_risk_regime"] == entry_risk_regime)
        & (trades["direction"] == direction)
    ].copy()
    if bucket.empty:
        return {
            "trades": 0,
            "pnl": 0.0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "avg_rr": 0.0,
            "stop_loss_pct": 0.0,
            "target_pct": 0.0,
            "avg_hold_hours": 0.0,
            "median_hold_hours": 0.0,
        }
    return {
        "trades": int(len(bucket)),
        "pnl": round(float(bucket["pnl"].sum()), 2),
        "win_rate": round(float((bucket["pnl"] > 0).mean() * 100), 1),
        "avg_pnl": round(float(bucket["pnl"].mean()), 2),
        "avg_rr": round(float(bucket["rr_ratio"].mean()), 3),
        "stop_loss_pct": round(float((bucket["exit_reason"] == "stop_loss").mean() * 100), 1),
        "target_pct": round(float(bucket["exit_reason"].astype(str).str.contains("target").mean() * 100), 1),
        "avg_hold_hours": round(float(bucket["hold_hours"].mean()), 2),
        "median_hold_hours": round(float(bucket["hold_hours"].median()), 2),
    }


def summarize_by_regime_direction(trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = trades.groupby(["entry_risk_regime", "direction"], dropna=False)
    for (regime, direction), bucket in grouped:
        rows.append(
            {
                "entry_risk_regime": regime or "unknown",
                "direction": direction,
                "trades": int(len(bucket)),
                "pnl": round(float(bucket["pnl"].sum()), 2),
                "win_rate": round(float((bucket["pnl"] > 0).mean() * 100), 1),
                "avg_pnl": round(float(bucket["pnl"].mean()), 2),
            }
        )
    rows.sort(key=lambda row: (str(row["entry_risk_regime"]), str(row["direction"])))
    return rows


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(Path(args.config).read_text())
    base_config = ExecutorConfig.from_dict(payload).to_scalp_strategy_config()
    baseline_config = replace(base_config, **BASELINE_RESET_OVERRIDES)
    optimized_config = replace(base_config, **OPTIMIZED_OVERRIDES)
    disabled_overrides = {**BASELINE_RESET_OVERRIDES, "allow_bear_strong_short": False}
    disabled_config = replace(base_config, **disabled_overrides)

    c15m_df = load_filtered_feather(Path(args.data_15m), args.start_date, args.end_date)
    c4h_df = load_filtered_feather(Path(args.data_4h), args.start_date, args.end_date)
    c15m = dataframe_to_candles(c15m_df)
    c4h = dataframe_to_candles(c4h_df)
    mapping = align_timeframes(c4h, c15m)
    precomputed = build_precomputed_state(c4h, c15m)

    cases = [
        run_case("baseline", baseline_config, c4h, c15m, mapping, precomputed, args.start_date),
        run_case("optimized", optimized_config, c4h, c15m, mapping, precomputed, args.start_date),
        run_case("disabled_reference", disabled_config, c4h, c15m, mapping, precomputed, args.start_date),
    ]
    case_map = {case["name"]: case for case in cases}

    baseline = case_map["baseline"]
    optimized = case_map["optimized"]
    disabled = case_map["disabled_reference"]

    return {
        "config": str(Path(args.config)),
        "date_range": {"start": args.start_date, "end": args.end_date},
        "optimized_overrides": OPTIMIZED_OVERRIDES,
        "cases": cases,
        "delta_vs_baseline": {
            "optimized": build_delta(baseline, optimized),
            "disabled_reference": build_delta(baseline, disabled),
        },
    }


def build_delta(base_case: dict[str, Any], next_case: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall": {
            "total_return_pct": round(
                next_case["overall"]["total_return_pct"] - base_case["overall"]["total_return_pct"],
                2,
            ),
            "sharpe_ratio": round(
                next_case["overall"]["sharpe_ratio"] - base_case["overall"]["sharpe_ratio"],
                3,
            ),
            "profit_factor": round(
                next_case["overall"]["profit_factor"] - base_case["overall"]["profit_factor"],
                3,
            ),
            "win_rate": round(
                next_case["overall"]["win_rate"] - base_case["overall"]["win_rate"],
                1,
            ),
            "total_trades": next_case["overall"]["total_trades"] - base_case["overall"]["total_trades"],
        },
        "bear_strong_short": {
            "trades": next_case["bear_strong_short"]["trades"] - base_case["bear_strong_short"]["trades"],
            "pnl": round(next_case["bear_strong_short"]["pnl"] - base_case["bear_strong_short"]["pnl"], 2),
            "win_rate": round(
                next_case["bear_strong_short"]["win_rate"] - base_case["bear_strong_short"]["win_rate"],
                1,
            ),
            "avg_pnl": round(
                next_case["bear_strong_short"]["avg_pnl"] - base_case["bear_strong_short"]["avg_pnl"],
                2,
            ),
            "stop_loss_pct": round(
                next_case["bear_strong_short"]["stop_loss_pct"] - base_case["bear_strong_short"]["stop_loss_pct"],
                1,
            ),
        },
    }


def print_human_report(payload: dict[str, Any]) -> None:
    print("=== OVERALL ===")
    for case in payload["cases"]:
        overall = case["overall"]
        print(
            f"{case['name']:<18} "
            f"Return={overall['total_return_pct']:>8.2f}%  "
            f"Sharpe={overall['sharpe_ratio']:>6.3f}  "
            f"PF={overall['profit_factor']:>5.3f}  "
            f"WR={overall['win_rate']:>5.1f}%  "
            f"Trades={overall['total_trades']:>4d}"
        )

    print("\n=== BEAR_STRONG SHORT ===")
    for case in payload["cases"]:
        bucket = case["bear_strong_short"]
        print(
            f"{case['name']:<18} "
            f"Trades={bucket['trades']:>4d}  "
            f"PnL={bucket['pnl']:>9.2f}  "
            f"WinRate={bucket['win_rate']:>5.1f}%  "
            f"AvgPnL={bucket['avg_pnl']:>7.2f}  "
            f"StopLoss={bucket['stop_loss_pct']:>5.1f}%  "
            f"AvgHold={bucket['avg_hold_hours']:>6.2f}h"
        )

    print("\n=== DELTA VS BASELINE ===")
    for name, delta in payload["delta_vs_baseline"].items():
        overall = delta["overall"]
        bucket = delta["bear_strong_short"]
        print(
            f"{name:<18} "
            f"Return={overall['total_return_pct']:+8.2f}%  "
            f"Sharpe={overall['sharpe_ratio']:+6.3f}  "
            f"Trades={overall['total_trades']:+4d}  "
            f"BS_PnL={bucket['pnl']:+9.2f}  "
            f"BS_Trades={bucket['trades']:+4d}  "
            f"BS_StopLoss={bucket['stop_loss_pct']:+6.1f}pp"
        )


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print_human_report(payload)


if __name__ == "__main__":
    main()
