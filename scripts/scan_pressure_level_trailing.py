#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay, parse_float_list, parse_str_list  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, add_windows, replay_shadow_events  # noqa: E402


def parse_bool_list(value: str) -> list[bool]:
    items = []
    for item in parse_str_list(value):
        normalized = item.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            items.append(True)
        elif normalized in {"0", "false", "no", "off"}:
            items.append(False)
        else:
            raise ValueError(f"Invalid boolean value: {item}")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan pressure-level trailing with the fixed high-leverage shadow overlay.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--include-disabled-baseline", action="store_true")
    parser.add_argument("--pressure-min-rr-values", default="2.0,2.5,3.0")
    parser.add_argument("--pressure-lock-rr-values", default="0.4,0.6")
    parser.add_argument("--pressure-atr-multiplier-values", default="2.5,3.0")
    parser.add_argument("--pressure-proximity-pct-values", default="0.15,0.25")
    parser.add_argument("--pressure-rejection-min-rr-values", default="3.0")
    parser.add_argument("--pressure-take-profit-on-rejection-values", default="false,true")
    parser.add_argument("--pressure-enable-target-cap-values", default="false,true")
    parser.add_argument("--pressure-target-min-rr-values", default="1.5,2.0")
    parser.add_argument("--pressure-target-buffer-pct-values", default="0.05")
    parser.add_argument(
        "--pressure-regime-label-sets",
        default="flat+normal",
        help="Comma-separated label sets. Use + inside a set, e.g. flat+normal,normal,all.",
    )
    parser.add_argument("--pressure-rejection-wick-ratio-values", default="0.55")
    parser.add_argument("--pressure-rejection-close-pct-values", default="0.2")
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", default=str(ROOT / "var" / "high_leverage_expansion" / "pressure_level_trailing_scan.json"))
    return parser.parse_args()


def pressure_param_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if args.include_disabled_baseline:
        candidates.append({"enable_pressure_level_trailing": False})

    for (
        min_rr,
        lock_rr,
        atr_multiplier,
        proximity_pct,
        rejection_min_rr,
        take_profit,
        target_cap,
        target_min_rr,
        target_buffer_pct,
        regime_label_set,
        wick_ratio,
        close_pct,
    ) in itertools.product(
        parse_float_list(args.pressure_min_rr_values),
        parse_float_list(args.pressure_lock_rr_values),
        parse_float_list(args.pressure_atr_multiplier_values),
        parse_float_list(args.pressure_proximity_pct_values),
        parse_float_list(args.pressure_rejection_min_rr_values),
        parse_bool_list(args.pressure_take_profit_on_rejection_values),
        parse_bool_list(args.pressure_enable_target_cap_values),
        parse_float_list(args.pressure_target_min_rr_values),
        parse_float_list(args.pressure_target_buffer_pct_values),
        parse_str_list(args.pressure_regime_label_sets),
        parse_float_list(args.pressure_rejection_wick_ratio_values),
        parse_float_list(args.pressure_rejection_close_pct_values),
    ):
        candidates.append(
            {
                "enable_pressure_level_trailing": True,
                "pressure_min_rr": min_rr,
                "pressure_rejection_min_rr": rejection_min_rr,
                "pressure_lock_rr": lock_rr,
                "pressure_atr_multiplier": atr_multiplier,
                "pressure_proximity_pct": proximity_pct,
                "pressure_rejection_wick_ratio": wick_ratio,
                "pressure_rejection_close_pct": close_pct,
                "pressure_take_profit_on_rejection": take_profit,
                "pressure_enable_target_cap": target_cap,
                "pressure_target_min_rr": target_min_rr,
                "pressure_target_buffer_pct": target_buffer_pct,
                "pressure_regime_labels": None if regime_label_set == "all" else regime_label_set.split("+"),
                "pressure_round_steps_usdt": [1000.0, 500.0],
                "pressure_cluster_lookback_bars": 192,
                "pressure_cluster_bin_usdt": 250.0,
                "pressure_cluster_min_touches": 4,
                "pressure_cluster_min_volume_ratio": 1.25,
                "pressure_swing_lookback_bars": 96,
                "pressure_min_bars_held": 1,
            }
        )
    return candidates


def evaluate_candidate(
    base_payload: dict[str, Any],
    pressure_params: dict[str, Any],
    prepared: Any,
    start_date: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = deepcopy(base_payload)
    payload.update(pressure_params)
    metrics, engine = run_engine(payload, prepared, start_date)
    trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
    initial_capital = float(metrics.get("initial_capital", 1000.0))
    fixed = expansion_overlay(trades, initial_capital, FIXED_STRUCTURE_PARAMS, include_events=True)
    shadow = replay_shadow_events(
        fixed["events"],
        initial_capital,
        daily_loss_stop_pct=float(args.daily_loss_stop_pct),
        equity_drawdown_stop_pct=float(args.equity_drawdown_stop_pct),
        consecutive_loss_stop=int(args.consecutive_loss_stop),
        equity_drawdown_cooldown_days=int(args.equity_drawdown_cooldown_days),
    )
    shadow = add_windows(shadow, initial_capital)
    year = shadow.get("windows", {}).get("current_year", {})
    recent_60d = shadow.get("windows", {}).get("last_60d", {})
    recent_30d = shadow.get("windows", {}).get("last_30d", {})
    score = round(
        float(shadow["total_return_pct"])
        + float(year.get("total_return_pct", 0.0)) * 150.0
        + float(recent_60d.get("total_return_pct", 0.0)) * 80.0
        + float(recent_30d.get("total_return_pct", 0.0)) * 30.0
        - float(shadow["max_drawdown_pct"]) * 30.0
        - float(year.get("max_drawdown_pct", 0.0)) * 40.0,
        4,
    )
    return {
        "score": score,
        "pressure_params": pressure_params,
        "engine": {
            "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 2),
            "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0)), 2),
            "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
            "total_trades": int(metrics.get("total_trades", 0)),
            "exit_reasons": metrics.get("exit_reasons", {}),
        },
        "fixed_structure_overlay": {key: value for key, value in fixed.items() if key != "events"},
        "shadow": shadow,
    }


def main() -> None:
    args = parse_args()
    base_payload = load_config_payload(Path(args.config))
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=base_payload.get("regime_switcher_thresholds"),
    )

    results: list[dict[str, Any]] = []
    candidates = pressure_param_grid(args)
    for index, params in enumerate(candidates, start=1):
        result = evaluate_candidate(base_payload, params, prepared, args.start_date, args)
        results.append(result)
        shadow = result["shadow"]
        year = shadow.get("windows", {}).get("current_year", {})
        print(
            f"{index:03d}/{len(candidates):03d} score={result['score']:.2f} "
            f"full={shadow['total_return_pct']:.2f}%/{shadow['max_drawdown_pct']:.2f}% "
            f"year={year.get('total_return_pct', 0.0):.2f}% "
            f"engine_exits={result['engine']['exit_reasons']} params={params}",
            flush=True,
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "config": str(Path(args.config).resolve()),
        "data": {
            "start": str(prepared.start),
            "end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
        },
        "shadow_params": {
            "daily_loss_stop_pct": args.daily_loss_stop_pct,
            "equity_drawdown_stop_pct": args.equity_drawdown_stop_pct,
            "equity_drawdown_cooldown_days": args.equity_drawdown_cooldown_days,
            "consecutive_loss_stop": args.consecutive_loss_stop,
        },
        "candidate_count": len(results),
        "top": results[: args.top],
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)
    for idx, item in enumerate(results[: args.top], start=1):
        shadow = item["shadow"]
        year = shadow.get("windows", {}).get("current_year", {})
        recent_60d = shadow.get("windows", {}).get("last_60d", {})
        print(
            f"{idx:02d} score={item['score']:.2f} full={shadow['total_return_pct']:.2f}%/"
            f"{shadow['max_drawdown_pct']:.2f}% year={year.get('total_return_pct', 0.0):.2f}% "
            f"60d={recent_60d.get('total_return_pct', 0.0):.2f}% params={item['pressure_params']}"
        )


if __name__ == "__main__":
    main()
