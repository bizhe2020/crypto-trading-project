#!/usr/bin/env python3
"""
Parameter scanning script for bear_strong_short and bull_weak_long parameters.
Uses ExecutorConfig.from_dict().to_scalp_strategy_config() path.
"""

import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.okx_executor import ExecutorConfig
from strategy.scalp_robust_v2_core import (
    ScalpRobustEngine,
    align_timeframes,
    build_precomputed_state,
    dataframe_to_candles,
)


def load_filtered_feather(path: Path, start_date: str, end_date: str = "2026-04-18") -> pd.DataFrame:
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
    return {
        "name": name,
        "total_return_pct": round(float(metrics["total_return_pct"]), 2),
        "sharpe_ratio": round(float(metrics["sharpe_ratio"]), 3),
        "max_drawdown_pct": round(float(metrics["max_drawdown_pct"]), 2),
        "win_rate": round(float(metrics["win_rate"]), 1),
        "total_trades": int(metrics["total_trades"]),
    }


def main():
    # Paths
    config_path = PROJECT_ROOT / "config" / "config.live.5x-3pct.json"
    data_root = Path("/Users/laoji/projects/crypto-trading-project/deployment/data/okx/futures")
    c15m_path = data_root / "BTC_USDT_USDT-15m-futures.feather"
    c4h_path = data_root / "BTC_USDT_USDT-4h-futures.feather"
    start_date = "2023-01-01"

    print("Loading data...")
    c15m_df = load_filtered_feather(c15m_path, start_date)
    c4h_df = load_filtered_feather(c4h_path, start_date)
    c15m = dataframe_to_candles(c15m_df)
    c4h = dataframe_to_candles(c4h_df)
    mapping = align_timeframes(c4h, c15m)
    precomputed = build_precomputed_state(c4h, c15m)
    print(f"Loaded {len(c15m)} 15m candles, {len(c4h)} 4h candles")

    # Load base config
    payload = json.loads(config_path.read_text())
    base_config = ExecutorConfig.from_dict(payload).to_scalp_strategy_config()

    # Fixed parameters for all cases (StrategyConfig fields only)
    # NOTE: atr_regime_filter is silently IGNORED if not a valid StrategyConfig field
    # The base config already has: atr_regime_filter=tight_style_off, atr_activation_rr=2.12
    # Only include overrides for what we actually want to CHANGE from base.
    fixed_base = {
        "pullback_window": 40,
        "sl_buffer_pct": 1.25,
        "bull_strong_long_risk_per_trade": 0.06,
        "enable_target_rr_cap": True,
        "enable_atr_trailing": True,
        # ATR params already in base config via to_scalp_strategy_config():
        # atr_activation_rr=2.12, atr_loose_multiplier=2.7, atr_normal_multiplier=2.25,
        # atr_tight_multiplier=1.8, atr_regime_filter=tight_style_off
    }

    # ============================================================
    # FIRST ROUND: single parameter changes
    # ============================================================
    cases_round1 = [
        # name, bear_strong_short_risk, short_strong_rr, bull_weak_long_trail, bear_weak_short_trail
        ("1_baseline",          0.02, 4.5, None,       "tight"),
        ("2_bss_risk_0.015",    0.015, 4.5, None,       "tight"),
        ("3_bss_risk_0.025",    0.025, 4.5, None,       "tight"),
        ("4_bss_risk_0.03",     0.03,  4.5, None,       "tight"),
        ("5_ssrr_5.0",          0.02,  5.0, None,       "tight"),
        ("6_ssrr_5.5",          0.02,  5.5, None,       "tight"),
        ("7_bwl_trail_normal",  0.02,  4.5, "normal",   "tight"),
        ("8_bwl_trail_tight",   0.02,  4.5, "tight",   "tight"),
        ("9_bws_trail_normal",  0.02,  4.5, None,       "normal"),
    ]

    print("\n" + "=" * 90)
    print("FIRST ROUND: single parameter changes")
    print("=" * 90)
    print(f"{'Case':<22} {'Return%':>10} {'Sharpe':>8} {'MDD%':>8} {'Trades':>7} {'WR%':>6}")
    print("-" * 90)

    round1_results = []
    for case_name, bss_risk, ssrr, bwl_trail, bws_trail in cases_round1:
        overrides = {
            **fixed_base,
            "bear_strong_short_risk_per_trade": bss_risk,
            "short_strong_rr_ratio": ssrr,
        }
        if bwl_trail is not None:
            overrides["bull_weak_long_trail_style_override"] = bwl_trail
        if bws_trail is not None:
            overrides["bear_weak_short_trail_style_override"] = bws_trail

        config = replace(base_config, **overrides)
        result = run_case(case_name, config, c4h, c15m, mapping, precomputed, start_date)
        round1_results.append(result)

        # Only print if > 2500%
        if result["total_return_pct"] >= 2500:
            print(
                f"{result['name']:<22} "
                f"{result['total_return_pct']:>10.2f} "
                f"{result['sharpe_ratio']:>8.3f} "
                f"{result['max_drawdown_pct']:>8.2f} "
                f"{result['total_trades']:>7.0f} "
                f"{result['win_rate']:>6.1f}"
            )
        else:
            print(
                f"{result['name']:<22} "
                f"{result['total_return_pct']:>10.2f} "
                f"(below 2500%)"
            )

    # ============================================================
    # SECOND ROUND: top combinations
    # ============================================================
    # Find top 3 from round 1 to build combinations
    top3 = sorted(round1_results, key=lambda x: x["total_return_pct"], reverse=True)[:3]
    print("\n" + "=" * 90)
    print("TOP 3 FROM ROUND 1:")
    for r in top3:
        print(f"  {r['name']}: {r['total_return_pct']:.2f}%")
    print("=" * 90)

    # Build second round cases based on round 1 insights:
    # Get best risk per trade
    risk_results = [(r, float(r["name"].split("_")[-1].replace("0.", "0."))) 
                    for r in round1_results if "bss_risk" in r["name"]]
    best_risk_result = max(risk_results, key=lambda x: x[0]["total_return_pct"])
    best_risk_val = best_risk_result[1]
    best_risk_name = best_risk_result[0]["name"]

    # Get best ssrr
    ssrr_results = [r for r in round1_results if "ssrr" in r["name"]]
    best_ssrr_result = max(ssrr_results, key=lambda x: x["total_return_pct"])
    best_ssrr_val = float(best_ssrr_result["name"].split("_")[-1].replace("p", "."))
    best_ssrr_name = best_ssrr_result["name"]

    # The baseline bss_risk=0.02 ssrr=4.5 is the reference
    baseline_return = next(r["total_return_pct"] for r in round1_results if r["name"] == "1_baseline")

    # Build more targeted combinations based on round 1 insights:
    # risk=0.015 and ssrr=5.0 were best individual improvements
    # Combining them should give the best result
    round2_cases = [
        # name, bss_risk, ssrr, bwl_trail, bws_trail
        ("10_combo_best_all",       best_risk_val, best_ssrr_val, "normal", "normal"),
        ("11_combo_best_tight_bws", best_risk_val, best_ssrr_val, "normal", "tight"),
        ("12_combo_best_bws_null",  best_risk_val, best_ssrr_val, "normal", None),
        ("13_combo_ssrr6",          best_risk_val, 6.0, "normal", "normal"),
        ("14_combo_ssrr7",          best_risk_val, 7.0, "normal", "normal"),
        ("15_combo_risk_ssrr_only", best_risk_val, best_ssrr_val, None, "tight"),
        ("16_combo_risk_only",      best_risk_val, 4.5, None, "tight"),
        ("17_combo_ssrr_only",      0.02, best_ssrr_val, None, "tight"),
        ("18_combo_risk_0p01",      0.01, best_ssrr_val, None, "tight"),
        ("19_combo_risk_0p012",     0.012, best_ssrr_val, None, "tight"),
        ("20_combo_risk_0p018",    0.018, best_ssrr_val, None, "tight"),
        ("21_combo_risk_0p015_ssrr5p5", 0.015, 5.5, None, "tight"),
        ("22_combo_risk_0p015_ssrr6p0", 0.015, 6.0, None, "tight"),
    ]

    # Deduplicate against round1
    seen_names = {r["name"] for r in round1_results}
    round2_cases = [(n, *rest) for n, *rest in round2_cases if n not in seen_names]

    print("\n" + "=" * 90)
    print("SECOND ROUND: combinations")
    print("=" * 90)
    print(f"{'Case':<22} {'Return%':>10} {'Sharpe':>8} {'MDD%':>8} {'Trades':>7} {'WR%':>6}")
    print("-" * 90)

    round2_results = []
    for case_name, bss_risk, ssrr, bwl_trail, bws_trail in round2_cases:
        overrides = {
            **fixed_base,
            "bear_strong_short_risk_per_trade": bss_risk,
            "short_strong_rr_ratio": ssrr,
        }
        if bwl_trail is not None:
            overrides["bull_weak_long_trail_style_override"] = bwl_trail
        if bws_trail is not None:
            overrides["bear_weak_short_trail_style_override"] = bws_trail

        config = replace(base_config, **overrides)
        result = run_case(case_name, config, c4h, c15m, mapping, precomputed, start_date)
        round2_results.append(result)

        if result["total_return_pct"] >= 2500:
            print(
                f"{result['name']:<22} "
                f"{result['total_return_pct']:>10.2f} "
                f"{result['sharpe_ratio']:>8.3f} "
                f"{result['max_drawdown_pct']:>8.2f} "
                f"{result['total_trades']:>7.0f} "
                f"{result['win_rate']:>6.1f}"
            )
        else:
            print(
                f"{result['name']:<22} "
                f"{result['total_return_pct']:>10.2f} "
                f"(below 2500%)"
            )

    # ============================================================
    # FINAL SUMMARY: all results >= 2500%
    # ============================================================
    all_results = round1_results + round2_results
    qualifying = [r for r in all_results if r["total_return_pct"] >= 2500]
    qualifying_sorted = sorted(qualifying, key=lambda x: x["total_return_pct"], reverse=True)

    print("\n" + "=" * 90)
    print("FINAL SUMMARY: All results >= 2500%")
    print("=" * 90)
    print(f"{'Case':<22} {'Return%':>10} {'Sharpe':>8} {'MDD%':>8} {'Trades':>7} {'WR%':>6}")
    print("-" * 90)
    if qualifying_sorted:
        for r in qualifying_sorted:
            print(
                f"{r['name']:<22} "
                f"{r['total_return_pct']:>10.2f} "
                f"{r['sharpe_ratio']:>8.3f} "
                f"{r['max_drawdown_pct']:>8.2f} "
                f"{r['total_trades']:>7.0f} "
                f"{r['win_rate']:>6.1f}"
            )
    else:
        print("No combinations reached 2500%")

    if qualifying_sorted:
        best = qualifying_sorted[0]
        print(f"\n*** BEST: {best['name']} with {best['total_return_pct']:.2f}% return, "
              f"Sharpe={best['sharpe_ratio']:.3f}, MDD={best['max_drawdown_pct']:.2f}% ***")


if __name__ == "__main__":
    main()
