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


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_bools(raw: str) -> list[bool]:
    mapping = {"true": True, "false": False}
    values: list[bool] = []
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"Unsupported bool value: {item}")
        values.append(mapping[key])
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize fee-sensitive ATR exit parameters")
    parser.add_argument("--config", default="config/config.live.5x-3pct.json")
    parser.add_argument("--data-root", default="../deployment/data/okx/futures")
    parser.add_argument("--symbol-stem", default="BTC_USDT_USDT")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--activation-values", default="2.0,2.1,2.2")
    parser.add_argument("--atr-period-values", default="14")
    parser.add_argument("--scale-values", default="0.9,1.0,1.1")
    parser.add_argument("--disable-fixed-target-exit-values", default="false,true")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _load_engine_inputs(data_root: Path, symbol_stem: str) -> tuple[list[Any], list[Any], list[int], Any]:
    path_15m = data_root / f"{symbol_stem}-15m-futures.feather"
    path_4h = data_root / f"{symbol_stem}-4h-futures.feather"
    c15m = dataframe_to_candles(pd.read_feather(path_15m))
    c4h = dataframe_to_candles(pd.read_feather(path_4h))
    mapping = align_timeframes(c4h, c15m)
    precomputed = build_precomputed_state(c4h, c15m)
    return c4h, c15m, mapping, precomputed


def _run_case(
    name: str,
    strategy_config: Any,
    c4h: list[Any],
    c15m: list[Any],
    mapping: list[int],
    precomputed: Any,
    start_date: str,
) -> dict[str, Any]:
    engine = ScalpRobustEngine(c4h, c15m, mapping, precomputed, strategy_config)
    metrics = engine.run_backtest(start_date=start_date)
    initial_capital = float(metrics["initial_capital"])
    final_capital = float(metrics["final_capital"])
    net_pnl = final_capital - initial_capital
    total_fees = float(metrics["total_fees_paid"])
    net_pnl_to_fees_ratio = net_pnl / total_fees if total_fees > 0 else 0.0
    return {
        "name": name,
        "disable_fixed_target_exit": strategy_config.disable_fixed_target_exit,
        "atr_period": strategy_config.atr_period,
        "atr_activation_rr": strategy_config.atr_activation_rr,
        "atr_loose_multiplier": strategy_config.atr_loose_multiplier,
        "atr_normal_multiplier": strategy_config.atr_normal_multiplier,
        "atr_tight_multiplier": strategy_config.atr_tight_multiplier,
        "total_return_pct": round(float(metrics["total_return_pct"]), 4),
        "final_capital": round(final_capital, 4),
        "max_drawdown_pct": round(float(metrics["max_drawdown_pct"]), 4),
        "sharpe_ratio": round(float(metrics["sharpe_ratio"]), 4),
        "risk_adjusted_return": round(float(metrics["risk_adjusted_return"]), 4),
        "profit_factor": round(float(metrics["profit_factor"]), 4),
        "win_rate": round(float(metrics["win_rate"]), 4),
        "wl_ratio": round(float(metrics["wl_ratio"]), 4),
        "target_hit_rate": round(float(metrics["target_hit_rate"]), 4),
        "total_trades": int(metrics["total_trades"]),
        "net_pnl": round(net_pnl, 4),
        "total_fees_paid": round(total_fees, 4),
        "net_pnl_to_fees_ratio": round(net_pnl_to_fees_ratio, 6),
        "exit_reasons": metrics["exit_reasons"],
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    data_root = Path(args.data_root)
    payload = json.loads(config_path.read_text())
    base_config = ExecutorConfig.from_dict(payload).to_scalp_strategy_config()
    activations = _parse_csv_floats(args.activation_values)
    atr_period_values = [int(value) for value in _parse_csv_floats(args.atr_period_values)]
    scales = _parse_csv_floats(args.scale_values)
    disable_values = _parse_csv_bools(args.disable_fixed_target_exit_values)
    c4h, c15m, mapping, precomputed = _load_engine_inputs(data_root, args.symbol_stem)

    results: list[dict[str, Any]] = []

    current = _run_case(
        "current_config",
        base_config,
        c4h,
        c15m,
        mapping,
        precomputed,
        args.start_date,
    )
    results.append(current)

    for atr_period in atr_period_values:
        for activation in activations:
            for scale in scales:
                for disable_target in disable_values:
                    candidate = replace(
                        base_config,
                        disable_fixed_target_exit=disable_target,
                        enable_atr_trailing=True,
                        atr_period=atr_period,
                        atr_activation_rr=activation,
                        atr_loose_multiplier=round(3.0 * scale, 4),
                        atr_normal_multiplier=round(2.5 * scale, 4),
                        atr_tight_multiplier=round(2.0 * scale, 4),
                    )
                    name = (
                        f"p{atr_period}_a{activation}_s{scale}_"
                        f"target_{'off' if disable_target else 'on'}"
                    )
                    results.append(
                        _run_case(
                            name,
                            candidate,
                            c4h,
                            c15m,
                            mapping,
                            precomputed,
                            args.start_date,
                        )
                    )

    top_by_sharpe = sorted(results, key=lambda row: row["sharpe_ratio"], reverse=True)[: args.top_k]
    top_by_fee_efficiency = sorted(results, key=lambda row: row["net_pnl_to_fees_ratio"], reverse=True)[: args.top_k]
    best_sharpe = top_by_sharpe[0]
    best_fee = top_by_fee_efficiency[0]

    output = {
        "config_path": str(config_path),
        "data_root": str(data_root),
        "start_date": args.start_date,
        "top_by_sharpe": top_by_sharpe,
        "top_by_net_pnl_to_fees_ratio": top_by_fee_efficiency,
        "best_sharpe_delta_vs_current": {
            "name": best_sharpe["name"],
            "total_return_pct": round(best_sharpe["total_return_pct"] - current["total_return_pct"], 4),
            "sharpe_ratio": round(best_sharpe["sharpe_ratio"] - current["sharpe_ratio"], 4),
            "net_pnl_to_fees_ratio": round(best_sharpe["net_pnl_to_fees_ratio"] - current["net_pnl_to_fees_ratio"], 6),
        },
        "best_fee_delta_vs_current": {
            "name": best_fee["name"],
            "total_return_pct": round(best_fee["total_return_pct"] - current["total_return_pct"], 4),
            "sharpe_ratio": round(best_fee["sharpe_ratio"] - current["sharpe_ratio"], 4),
            "net_pnl_to_fees_ratio": round(best_fee["net_pnl_to_fees_ratio"] - current["net_pnl_to_fees_ratio"], 6),
        },
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
