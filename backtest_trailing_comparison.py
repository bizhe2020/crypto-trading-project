#!/usr/bin/env python3
"""
Backtest trailing 策略对比：
1. Baseline：当前 trailing（固定 RR + 分阶段锁盈）
2. Variant A：动态布林带 + 保留固定目标
3. Variant B：纯动态布林带（禁用固定目标）
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from deployment.bot.market_data import OhlcvRepository
from deployment.strategy.price_band_trailing import PriceBandTrailingConfig
from deployment.strategy.scalp_robust_v2_core import (
    ScalpRobustEngine,
    StrategyConfig,
    dataframe_to_candles,
)


def load_market_data(symbol: str = "BTC/USDT:USDT"):
    repo = OhlcvRepository("deployment/data/okx/futures")
    bundle = repo.load_pair(symbol, timeframe="15m", informative_timeframe="4h")
    return dataframe_to_candles(bundle.informative_candles), dataframe_to_candles(bundle.primary_candles)


def backtest_with_config(c4h, c15m, config: StrategyConfig):
    engine = ScalpRobustEngine.from_candles(c4h, c15m, config=config)
    metrics = engine.run_backtest()
    return engine, metrics


def main():
    print("Loading BTC candles...")
    try:
        c4h, c15m = load_market_data()
        print(f"Loaded 4h candles: {len(c4h)}, 15m candles: {len(c15m)}")
        if c15m:
            start_time = datetime.fromtimestamp(c15m[0].ts, tz=timezone.utc)
            end_time = datetime.fromtimestamp(c15m[-1].ts, tz=timezone.utc)
            print(f"Period: {start_time} to {end_time}")
    except Exception as e:
        print(f"Error loading candles: {e}")
        return

    print("\n" + "=" * 120)
    print("TRAILING STRATEGY COMPARISON")
    print("=" * 120)

    baseline_config = StrategyConfig(
        enable_price_band_trailing=False,
        disable_fixed_target=False,
    )
    variant_a_config = StrategyConfig(
        enable_price_band_trailing=True,
        price_band_trailing_config=PriceBandTrailingConfig(
            window=20,
            std_mult_base=2.0,
            std_mult_sensitivity=0.5,
            trigger_min_rr=0.5,
            keep_fixed_target=True,
            enabled=True,
        ),
        disable_fixed_target=False,
    )
    variant_b_config = StrategyConfig(
        enable_price_band_trailing=True,
        price_band_trailing_config=PriceBandTrailingConfig(
            window=20,
            std_mult_base=2.0,
            std_mult_sensitivity=0.5,
            trigger_min_rr=0.5,
            keep_fixed_target=False,
            enabled=True,
        ),
        disable_fixed_target=True,
    )

    runs = [
        ("Baseline", baseline_config),
        ("Variant A", variant_a_config),
        ("Variant B", variant_b_config),
    ]

    results = {}
    engines = {}
    for name, config in runs:
        print(f"\n[{name}]")
        print("-" * 120)
        try:
            engine, metrics = backtest_with_config(c4h, c15m, config)
            engines[name] = engine
            results[name] = metrics
            print(f"Trades: {metrics['total_trades']:<10} Win Rate: {metrics['win_rate']:<8}%")
            print(f"Final Capital: ${metrics['final_capital']:<10.2f} Total Return: {metrics['total_return_pct']:<10.2f}%")
            print(f"Sharpe: {metrics['sharpe_ratio']:<10.2f} MaxDD%: {metrics['max_drawdown_pct']:<10.2f}")
            print(f"Exit Reasons: {engine.exit_reasons}")
        except Exception as e:
            print(f"Error: {e}")
            results[name] = {"error": str(e)}

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"{'Strategy':<15} {'Trades':<10} {'Return%':<12} {'Sharpe':<10} {'MaxDD%':<10}")
    print("-" * 120)
    for name, metrics in results.items():
        if "error" in metrics:
            print(f"{name:<15} ERROR: {metrics['error']}")
        else:
            print(
                f"{name:<15} {metrics['total_trades']:<10} {metrics['total_return_pct']:<12.2f} "
                f"{metrics['sharpe_ratio']:<10.2f} {metrics['max_drawdown_pct']:<10.2f}"
            )

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "exit_reasons": {name: engine.exit_reasons for name, engine in engines.items()},
    }
    output_file = Path("backtest_trailing_comparison.json")
    output_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
