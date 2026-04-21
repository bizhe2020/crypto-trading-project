#!/usr/bin/env python3
"""
Parameter scanning script for ATR trailing activation RR.
Keeps the best long/short ATR multipliers fixed and scans atr_activation_rr values.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy.scalp_robust_v2_core import (
    ScalpRobustEngine,
    StrategyConfig,
    dataframe_to_candles,
)


def load_feather_data(path: str, start_date: str = "2023-01-01", end_date: str = "2026-04-18") -> list:
    """Load feather file and filter by date range."""
    df = pd.read_feather(path)

    # Ensure timestamp column exists
    if "timestamp" not in df.columns and "date" not in df.columns:
        raise ValueError(f"Feather file must contain 'timestamp' or 'date' column")

    # Parse dates
    if "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # Filter by date range
    start_dt = pd.to_datetime(start_date, utc=True)
    end_dt = pd.to_datetime(end_date, utc=True)
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)].reset_index(drop=True)

    return df


def load_config(config_path: str) -> dict:
    """Load JSON config file."""
    with open(config_path, "r") as f:
        return json.load(f)


# Fields that exist in StrategyConfig dataclass
_STRATEGY_CONFIG_FIELDS = {
    f.name for f in StrategyConfig.__dataclass_fields__.values()
}


def create_strategy_config(base_config: dict, **overrides) -> StrategyConfig:
    """Create StrategyConfig from dict, applying overrides.
    Filters out fields not in StrategyConfig to avoid TypeError.
    """
    config_dict = base_config.copy()
    # Filter to only StrategyConfig fields
    config_dict = {k: v for k, v in config_dict.items() if k in _STRATEGY_CONFIG_FIELDS}
    config_dict.update(overrides)
    return StrategyConfig(**config_dict)


def run_backtest(
    c4h_df: pd.DataFrame,
    c15m_df: pd.DataFrame,
    config: StrategyConfig,
) -> dict[str, Any]:
    """Run backtest with given config."""
    c4h_candles = dataframe_to_candles(c4h_df)
    c15m_candles = dataframe_to_candles(c15m_df)

    engine = ScalpRobustEngine.from_candles(c4h_candles, c15m_candles, config)
    metrics = engine.run_backtest(start_date="2023-01-01")

    return metrics


def format_metrics(metrics: dict) -> dict:
    """Extract key metrics for display."""
    return {
        "total_return_pct": metrics.get("total_return_pct", 0),
        "sharpe_ratio": metrics.get("sharpe_ratio", 0),
        "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
        "win_rate": metrics.get("win_rate", 0),
        "profit_factor": metrics.get("profit_factor", 0),
        "total_trades": metrics.get("total_trades", 0),
        "wl_ratio": metrics.get("wl_ratio", 0),
    }


def scan_parameters():
    """Main scanning loop."""
    # Data paths
    data_root = Path("/Users/laoji/projects/crypto-trading-project/deployment/data/okx/futures")
    c15m_path = data_root / "BTC_USDT_USDT-15m-futures.feather"
    c4h_path = data_root / "BTC_USDT_USDT-4h-futures.feather"

    # Config path
    config_path = Path(__file__).parent.parent / "config" / "config.live.5x-3pct.template.json"

    print(f"Loading data from {c15m_path} and {c4h_path}...")
    c15m_df = load_feather_data(str(c15m_path))
    c4h_df = load_feather_data(str(c4h_path))
    print(f"Loaded {len(c15m_df)} 15m candles and {len(c4h_df)} 4h candles")

    print(f"Loading config from {config_path}...")
    base_config_dict = load_config(str(config_path))

    # Fixed parameters
    fixed_params = {
        "disable_fixed_target_exit": True,
        "atr_loose_multiplier": 2.7,
        "atr_normal_multiplier": 2.25,
        "atr_tight_multiplier": 1.8,
        "atr_regime_filter": "tight_style_off",
        "long_atr_loose_multiplier": 2.7,
        "long_atr_normal_multiplier": 2.7,
        "long_atr_tight_multiplier": 2.7,
        "short_atr_loose_multiplier": 1.5,
        "short_atr_normal_multiplier": 1.5,
        "short_atr_tight_multiplier": 1.5,
        "long_atr_activation_rr": None,
        "short_atr_activation_rr": None,
    }

    activation_rr_values = [1.5, 1.8, 2.0, 2.2, 2.5]

    results = []
    total_combinations = len(activation_rr_values)
    current = 0

    print(f"\nScanning {total_combinations} parameter combinations...")
    print("=" * 100)
    print("Fixed multipliers: long=2.7 short=1.5 disable_fixed_target_exit=true")

    for activation_rr in activation_rr_values:
        current += 1

        overrides = {
            **fixed_params,
            "atr_activation_rr": activation_rr,
        }

        config = create_strategy_config(base_config_dict, **overrides)

        try:
            metrics = run_backtest(c4h_df, c15m_df, config)
            formatted = format_metrics(metrics)

            result = {
                "atr_activation_rr": activation_rr,
                **formatted,
            }
            results.append(result)

            print(
                f"[{current:2d}/{total_combinations}] "
                f"ATR_RR={activation_rr:<3.1f} | "
                f"Return={formatted['total_return_pct']:7.2f}% | "
                f"Sharpe={formatted['sharpe_ratio']:6.2f} | "
                f"DD={formatted['max_drawdown_pct']:6.2f}% | "
                f"WR={formatted['win_rate']:5.1f}% | "
                f"PF={formatted['profit_factor']:5.2f} | "
                f"Trades={formatted['total_trades']:3.0f}"
            )
        except Exception as e:
            print(f"[{current:2d}/{total_combinations}] ATR_RR={activation_rr:<3.1f} | ERROR: {e}")

    print("=" * 100)

    # Sort by total return
    results_by_return = sorted(results, key=lambda x: x["total_return_pct"], reverse=True)

    # Sort by sharpe
    results_by_sharpe = sorted(results, key=lambda x: x["sharpe_ratio"], reverse=True)

    print("\n" + "=" * 100)
    print("TOP 5 BY TOTAL RETURN")
    print("=" * 100)
    print(f"{'Rank':<5} {'ATR RR':<8} {'Return%':<10} {'Sharpe':<8} {'DD%':<8} {'WR%':<7} {'PF':<6} {'Trades':<7}")
    print("-" * 100)
    for i, result in enumerate(results_by_return[:5], 1):
        print(
            f"{i:<5} {result['atr_activation_rr']:<8.1f} {result['total_return_pct']:<10.2f} "
            f"{result['sharpe_ratio']:<8.2f} "
            f"{result['max_drawdown_pct']:<8.2f} {result['win_rate']:<7.1f} "
            f"{result['profit_factor']:<6.2f} {result['total_trades']:<7.0f}"
        )

    print("\n" + "=" * 100)
    print("TOP 5 BY SHARPE RATIO")
    print("=" * 100)
    print(f"{'Rank':<5} {'ATR RR':<8} {'Sharpe':<8} {'Return%':<10} {'DD%':<8} {'WR%':<7} {'PF':<6} {'Trades':<7}")
    print("-" * 100)
    for i, result in enumerate(results_by_sharpe[:5], 1):
        print(
            f"{i:<5} {result['atr_activation_rr']:<8.1f} {result['sharpe_ratio']:<8.2f} "
            f"{result['total_return_pct']:<10.2f} "
            f"{result['max_drawdown_pct']:<8.2f} {result['win_rate']:<7.1f} "
            f"{result['profit_factor']:<6.2f} {result['total_trades']:<7.0f}"
        )

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total combinations scanned: {len(results)}")
    print("Fixed multipliers: long=2.7, short=1.5")
    print(f"Best return: {results_by_return[0]['total_return_pct']:.2f}% "
          f"(ATR RR={results_by_return[0]['atr_activation_rr']:.1f})")
    print(f"Best Sharpe: {results_by_sharpe[0]['sharpe_ratio']:.2f} "
          f"(ATR RR={results_by_sharpe[0]['atr_activation_rr']:.1f})")


if __name__ == "__main__":
    scan_parameters()
