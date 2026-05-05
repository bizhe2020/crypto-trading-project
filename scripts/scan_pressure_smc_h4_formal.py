#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
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
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, load_pressure_params  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.report_smc_trade_context import load_best_shadow_params  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay, parse_float_list  # noqa: E402
from scripts.scan_pressure_level_trailing import parse_bool_list, parse_str_list  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, add_windows, replay_shadow_events  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_context" / "pressure_smc_h4_formal_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formal scan for pressure target-cap variants plus H4 favorable SMC sizing."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--pressure-target-min-rr-values", default="1.0,1.1,1.25,1.4")
    parser.add_argument("--pressure-target-buffer-pct-values", default="0.02,0.03,0.05")
    parser.add_argument("--pressure-dynamic-target-min-rr-enabled-values", default="false,true")
    parser.add_argument("--pressure-dynamic-target-compression-rr-values", default="0.9,1.0")
    parser.add_argument("--pressure-dynamic-target-flat-rr-values", default="1.0,1.1,1.25")
    parser.add_argument("--pressure-dynamic-target-breakout-rr-values", default="1.4,1.5")
    parser.add_argument("--pressure-regime-label-sets", default="flat")
    parser.add_argument("--smc-h4-favorable-multiplier-values", default="1.0,1.05,1.08,1.1")
    parser.add_argument("--h4-range-lookback-bars", type=int, default=42)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def pressure_candidates(args: argparse.Namespace, base_pressure: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for (
        target_min_rr,
        target_buffer_pct,
        dynamic_enabled,
        compression_rr,
        flat_rr,
        breakout_rr,
        regime_label_set,
    ) in itertools.product(
        parse_float_list(args.pressure_target_min_rr_values),
        parse_float_list(args.pressure_target_buffer_pct_values),
        parse_bool_list(args.pressure_dynamic_target_min_rr_enabled_values),
        parse_float_list(args.pressure_dynamic_target_compression_rr_values),
        parse_float_list(args.pressure_dynamic_target_flat_rr_values),
        parse_float_list(args.pressure_dynamic_target_breakout_rr_values),
        parse_str_list(args.pressure_regime_label_sets),
    ):
        if not dynamic_enabled:
            compression_rr = float(base_pressure.get("pressure_dynamic_target_compression_rr", 1.0))
            flat_rr = float(base_pressure.get("pressure_dynamic_target_flat_rr", 1.25))
            breakout_rr = float(base_pressure.get("pressure_dynamic_target_breakout_rr", 1.5))
        candidate = deepcopy(base_pressure)
        candidate.update(
            {
                "pressure_enable_target_cap": True,
                "pressure_target_min_rr": target_min_rr,
                "pressure_target_buffer_pct": target_buffer_pct,
                "pressure_dynamic_target_min_rr_enabled": dynamic_enabled,
                "pressure_dynamic_target_compression_rr": compression_rr,
                "pressure_dynamic_target_flat_rr": flat_rr,
                "pressure_dynamic_target_breakout_rr": breakout_rr,
                "pressure_regime_labels": None if regime_label_set == "all" else regime_label_set.split("+"),
            }
        )
        key = json.dumps(candidate, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def completed_4h_idx(mapping: list[int], entry_idx: int) -> int:
    if entry_idx < 0 or entry_idx >= len(mapping):
        return -1
    return max(0, int(mapping[entry_idx]) - 1)


def h4_pd_side(prepared: Any, entry_idx: int, entry_price: float, direction: str, lookback: int) -> str:
    h4_idx = completed_4h_idx(prepared.mapping, entry_idx)
    if h4_idx < 0:
        return "unknown"
    start = max(0, h4_idx - max(int(lookback), 1) + 1)
    window = prepared.c4h[start : h4_idx + 1]
    if not window:
        return "unknown"
    high = max(float(candle.h) for candle in window)
    low = min(float(candle.l) for candle in window)
    if high <= low:
        return "unknown"
    position_pct = (float(entry_price) - low) / (high - low) * 100.0
    if direction == "BULL":
        return "favorable" if position_pct < 50.0 else "adverse"
    if direction == "BEAR":
        return "favorable" if position_pct > 50.0 else "adverse"
    return "unknown"


def attach_h4_smc_tags(trades: pd.DataFrame, prepared: Any, lookback: int) -> pd.DataFrame:
    tagged = trades.copy()
    tagged["smc_h4_pd_side"] = "unknown"
    tagged["smc_session_bucket"] = "unknown"
    tagged["smc_score"] = 2
    tagged["smc_recent_sweep_mss"] = False
    for idx, trade in tagged.iterrows():
        entry_idx = trade.get("entry_idx")
        if pd.isna(entry_idx):
            continue
        side = h4_pd_side(
            prepared=prepared,
            entry_idx=int(entry_idx),
            entry_price=float(trade.get("entry_price", 0.0) or 0.0),
            direction=str(trade.get("direction") or ""),
            lookback=lookback,
        )
        tagged.at[idx, "smc_h4_pd_side"] = side
    return tagged


def score_result(shadow: dict[str, Any]) -> float:
    year = shadow.get("windows", {}).get("current_year", {})
    recent_60d = shadow.get("windows", {}).get("last_60d", {})
    recent_30d = shadow.get("windows", {}).get("last_30d", {})
    return round(
        float(shadow["total_return_pct"])
        + float(year.get("total_return_pct", 0.0)) * 220.0
        + float(recent_60d.get("total_return_pct", 0.0)) * 100.0
        + float(recent_30d.get("total_return_pct", 0.0)) * 80.0
        - float(shadow["max_drawdown_pct"]) * 60.0
        - float(year.get("max_drawdown_pct", 0.0)) * 50.0,
        4,
    )


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    if pd.isna(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    base_payload = load_config_payload(Path(args.config))
    base_pressure = load_pressure_params(Path(args.pressure_params))
    shadow_params = load_best_shadow_params(Path(args.pressure_params))
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=base_payload.get("regime_switcher_thresholds"),
    )

    results: list[dict[str, Any]] = []
    pressure_grid = pressure_candidates(args, base_pressure)
    smc_values = parse_float_list(args.smc_h4_favorable_multiplier_values)
    total = len(pressure_grid) * len(smc_values)
    counter = 0
    for pressure_params in pressure_grid:
        payload = deepcopy(base_payload)
        payload.update(pressure_params)
        metrics, engine = run_engine(payload, prepared, args.start_date)
        base_trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
        trades = attach_h4_smc_tags(base_trades, prepared, args.h4_range_lookback_bars)
        initial_capital = float(metrics.get("initial_capital", 1000.0))
        for smc_multiplier in smc_values:
            counter += 1
            overlay_params = dict(FIXED_STRUCTURE_PARAMS)
            overlay_params.update(
                {
                    "smc_context_overlay_enabled": smc_multiplier != 1.0,
                    "smc_h4_favorable_multiplier": smc_multiplier,
                    "smc_h4_adverse_multiplier": 1.0,
                    "smc_low_score_multiplier": 1.0,
                    "smc_london_multiplier": 1.0,
                    "smc_recent_sweep_mss_multiplier": 1.0,
                }
            )
            fixed = expansion_overlay(trades, initial_capital, overlay_params, include_events=True)
            shadow = replay_shadow_events(
                fixed["events"],
                initial_capital,
                daily_loss_stop_pct=shadow_params["daily_loss_stop_pct"],
                equity_drawdown_stop_pct=shadow_params["equity_drawdown_stop_pct"],
                consecutive_loss_stop=shadow_params["consecutive_loss_stop"],
                equity_drawdown_cooldown_days=shadow_params["equity_drawdown_cooldown_days"],
            )
            shadow = add_windows(shadow, initial_capital)
            result = {
                "score": score_result(shadow),
                "pressure_params": pressure_params,
                "smc_params": {
                    "smc_h4_favorable_multiplier": smc_multiplier,
                    "smc_h4_adverse_multiplier": 1.0,
                    "smc_low_score_multiplier": 1.0,
                    "smc_london_multiplier": 1.0,
                    "smc_recent_sweep_mss_multiplier": 1.0,
                },
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
            results.append(result)
            year = shadow.get("windows", {}).get("current_year", {})
            recent_60d = shadow.get("windows", {}).get("last_60d", {})
            print(
                f"{counter:03d}/{total:03d} score={result['score']:.2f} "
                f"full={shadow['total_return_pct']:.2f}%/{shadow['max_drawdown_pct']:.2f}% "
                f"2026={year.get('total_return_pct', 0.0):.2f}% "
                f"60d={recent_60d.get('total_return_pct', 0.0):.2f}% "
                f"smc={smc_multiplier:g} target_rr={pressure_params.get('pressure_target_min_rr')} "
                f"buffer={pressure_params.get('pressure_target_buffer_pct')} "
                f"dyn={pressure_params.get('pressure_dynamic_target_min_rr_enabled')} "
                f"flat_rr={pressure_params.get('pressure_dynamic_target_flat_rr')}",
                flush=True,
            )

    results.sort(key=lambda item: item["score"], reverse=True)
    report = {
        "metadata": {
            "config": str(Path(args.config).resolve()),
            "pressure_params": str(Path(args.pressure_params).resolve()),
            "start_date": args.start_date,
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
            "shadow_params": shadow_params,
            "candidate_count": len(results),
            "h4_range_lookback_bars": args.h4_range_lookback_bars,
        },
        "top": results[: args.top],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    for idx, item in enumerate(results[: args.top], start=1):
        shadow = item["shadow"]
        year = shadow.get("windows", {}).get("current_year", {})
        recent_60d = shadow.get("windows", {}).get("last_60d", {})
        recent_30d = shadow.get("windows", {}).get("last_30d", {})
        print(
            f"{idx:02d} score={item['score']:.2f} full={shadow['total_return_pct']:.2f}%/"
            f"{shadow['max_drawdown_pct']:.2f}% 2026={year.get('total_return_pct', 0.0):.2f}%/"
            f"{year.get('max_drawdown_pct', 0.0):.2f}% 60d={recent_60d.get('total_return_pct', 0.0):.2f}% "
            f"30d={recent_30d.get('total_return_pct', 0.0):.2f}% "
            f"accepted={shadow['accepted_trades']} skipped={shadow['skipped_trades']} "
            f"smc={item['smc_params']} pressure={item['pressure_params']}"
        )


if __name__ == "__main__":
    main()
