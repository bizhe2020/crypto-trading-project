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
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay, parse_int_list  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, add_windows, replay_shadow_events  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan OB pullback-window settings on the promoted high-leverage strategy.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--pullback-window-values", default="30,40,50")
    parser.add_argument("--short-pullback-window-values", default="20,25,30,35")
    parser.add_argument("--hg-pullback-window-values", default="30,40,50")
    parser.add_argument("--normal-pullback-window-values", default="20,30,40")
    parser.add_argument("--flat-pullback-window-values", default="30,40,50")
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", default=str(ROOT / "var" / "high_leverage_expansion" / "pullback_window_scan.json"))
    return parser.parse_args()


def set_override_window(payload: dict[str, Any], key: str, value: int) -> None:
    overrides = payload.get(key)
    if not isinstance(overrides, dict):
        overrides = {}
    overrides = deepcopy(overrides)
    overrides["pullback_window"] = int(value)
    payload[key] = overrides


def apply_window_params(base_payload: dict[str, Any], params: dict[str, int]) -> dict[str, Any]:
    payload = deepcopy(base_payload)
    payload["pullback_window"] = int(params["pullback_window"])
    payload["short_pullback_window"] = int(params["short_pullback_window"])
    set_override_window(payload, "regime_switcher_hg_overrides", int(params["hg_pullback_window"]))
    set_override_window(payload, "regime_switcher_normal_overrides", int(params["normal_pullback_window"]))
    set_override_window(payload, "regime_switcher_flat_overrides", int(params["flat_pullback_window"]))
    return payload


def candidate_grid(args: argparse.Namespace) -> list[dict[str, int]]:
    candidates: list[dict[str, int]] = []
    seen: set[str] = set()
    for values in itertools.product(
        parse_int_list(args.pullback_window_values),
        parse_int_list(args.short_pullback_window_values),
        parse_int_list(args.hg_pullback_window_values),
        parse_int_list(args.normal_pullback_window_values),
        parse_int_list(args.flat_pullback_window_values),
    ):
        params = {
            "pullback_window": values[0],
            "short_pullback_window": values[1],
            "hg_pullback_window": values[2],
            "normal_pullback_window": values[3],
            "flat_pullback_window": values[4],
        }
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(params)
    return candidates


def score_shadow(shadow: dict[str, Any]) -> float:
    year = shadow.get("windows", {}).get("current_year", {})
    recent_60d = shadow.get("windows", {}).get("last_60d", {})
    recent_30d = shadow.get("windows", {}).get("last_30d", {})
    return round(
        float(shadow["total_return_pct"])
        + float(year.get("total_return_pct", 0.0)) * 150.0
        + float(recent_60d.get("total_return_pct", 0.0)) * 80.0
        + float(recent_30d.get("total_return_pct", 0.0)) * 30.0
        - float(shadow["max_drawdown_pct"]) * 30.0
        - float(year.get("max_drawdown_pct", 0.0)) * 40.0,
        4,
    )


def evaluate_candidate(
    base_payload: dict[str, Any],
    params: dict[str, int],
    prepared: Any,
    start_date: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = apply_window_params(base_payload, params)
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
    return {
        "score": score_shadow(shadow),
        "params": params,
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


def compact_result(result: dict[str, Any]) -> str:
    shadow = result["shadow"]
    year = shadow.get("windows", {}).get("current_year", {})
    recent_60d = shadow.get("windows", {}).get("last_60d", {})
    recent_30d = shadow.get("windows", {}).get("last_30d", {})
    engine = result["engine"]
    return (
        f"score={result['score']:.2f} "
        f"full={shadow['total_return_pct']:.2f}%/{shadow['max_drawdown_pct']:.2f}% "
        f"2026={year.get('total_return_pct', 0.0):.2f}%/{year.get('max_drawdown_pct', 0.0):.2f}% "
        f"60d={recent_60d.get('total_return_pct', 0.0):.2f}% "
        f"30d={recent_30d.get('total_return_pct', 0.0):.2f}% "
        f"trades={engine['total_trades']} accepted={shadow.get('accepted_trades', 0)} "
        f"params={result['params']}"
    )


def main() -> None:
    args = parse_args()
    payload = load_config_payload(Path(args.config))
    pressure_params_path = None
    pressure_params: dict[str, Any] = {}
    if str(args.pressure_params).strip().lower() != "none":
        pressure_params_path = str(Path(args.pressure_params).resolve())
        payload, pressure_params = apply_pressure_params(payload, Path(args.pressure_params))
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=payload.get("regime_switcher_thresholds"),
    )

    results: list[dict[str, Any]] = []
    candidates = candidate_grid(args)
    for index, params in enumerate(candidates, start=1):
        result = evaluate_candidate(payload, params, prepared, args.start_date, args)
        results.append(result)
        print(f"{index:04d}/{len(candidates):04d} {compact_result(result)}", flush=True)

    results.sort(key=lambda item: item["score"], reverse=True)
    baseline_params = {
        "pullback_window": int(payload.get("pullback_window", 0) or 0),
        "short_pullback_window": int(payload.get("short_pullback_window", 0) or 0),
        "hg_pullback_window": int((payload.get("regime_switcher_hg_overrides") or {}).get("pullback_window", 0) or 0),
        "normal_pullback_window": int((payload.get("regime_switcher_normal_overrides") or {}).get("pullback_window", 0) or 0),
        "flat_pullback_window": int((payload.get("regime_switcher_flat_overrides") or {}).get("pullback_window", 0) or 0),
    }
    report = {
        "config": str(Path(args.config).resolve()),
        "pressure_params_path": pressure_params_path,
        "pressure_params": pressure_params,
        "baseline_params": baseline_params,
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
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)
    for index, result in enumerate(results[: args.top], start=1):
        print(f"{index:02d} {compact_result(result)}")


if __name__ == "__main__":
    main()
