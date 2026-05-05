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

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.report_htf_pa_ict_context import (  # noqa: E402
    DEFAULT_DATA_15M,
    DEFAULT_DATA_4H,
    add_htf_contexts,
    daily_from_4h,
    load_df,
    namespace_for,
)
from scripts.report_pa_ict_liquidity_features import scan_events  # noqa: E402
from scripts.scan_high_leverage_expansion import (  # noqa: E402
    enrich_trades_with_regime_features,
    expansion_overlay,
    parse_float_list,
    window_metrics_from_events,
)
from scripts.scan_pa_ict_shadow_quality_overlay import (  # noqa: E402
    add_strategy_trade_fields,
    apply_best_pressure_params,
    load_best_params,
    promoted_fixed_params,
    promoted_shadow_params,
)
from scripts.scan_shadow_on_fixed_high_leverage import replay_shadow_events  # noqa: E402
from strategy.scalp_robust_v2_core import dataframe_to_candles  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "htf_context" / "htf_pa_ict_guard_reproduction.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the promoted high-leverage strategy with optional HTF PA/ICT guard."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--best-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--multipliers", default="1.0,0.70,0.75,0.25,0.0")
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument("--stdout", action="store_true")

    parser.add_argument("--swing-n", type=int, default=2)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--min-body-atr", type=float, default=0.7)
    parser.add_argument("--min-range-atr", type=float, default=1.1)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.05)
    parser.add_argument("--target-rr", type=float, default=2.0)
    parser.add_argument("--require-mss", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--h4-swing-lookback", type=int, default=30)
    parser.add_argument("--h4-liquidity-lookback-bars", type=int, default=180)
    parser.add_argument("--h4-mss-lookahead-bars", type=int, default=12)
    parser.add_argument("--h4-fvg-lookback-bars", type=int, default=6)
    parser.add_argument("--h4-entry-lookahead-bars", type=int, default=18)
    parser.add_argument("--h4-outcome-lookahead-bars", type=int, default=36)
    parser.add_argument("--h4-context-ttl-bars", type=int, default=42)

    parser.add_argument("--d1-swing-lookback", type=int, default=20)
    parser.add_argument("--d1-liquidity-lookback-bars", type=int, default=90)
    parser.add_argument("--d1-mss-lookahead-bars", type=int, default=5)
    parser.add_argument("--d1-fvg-lookback-bars", type=int, default=4)
    parser.add_argument("--d1-entry-lookahead-bars", type=int, default=10)
    parser.add_argument("--d1-outcome-lookahead-bars", type=int, default=20)
    parser.add_argument("--d1-context-ttl-bars", type=int, default=14)
    return parser.parse_args()


def annotate_trades_with_htf_context(
    trades: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if trades.empty:
        return trades, {"h4": 0, "d1": 0}

    df4 = load_df(Path(args.data_4h))
    h4_candles = dataframe_to_candles(df4)
    d1_candles = dataframe_to_candles(daily_from_4h(df4))
    h4_events = scan_events(h4_candles, namespace_for(args, "4h"))
    d1_events = scan_events(d1_candles, namespace_for(args, "1d"))

    out = trades.sort_values("entry_time").reset_index(drop=True).copy()
    records = out.to_dict("records")
    enriched = add_htf_contexts(records, h4_events, h4_candles, d1_events, d1_candles, args)
    htf_columns = sorted(
        {
            key
            for event in enriched
            for key in event
            if key.startswith("htf_")
        }
    )
    for column in htf_columns:
        out[column] = [event.get(column) for event in enriched]

    return out, {
        "h4": len(h4_events),
        "d1": len(d1_events),
        "h4_by_status": {
            status: sum(1 for event in h4_events if event.status == status)
            for status in sorted({event.status for event in h4_events})
        },
        "d1_by_status": {
            status: sum(1 for event in d1_events if event.status == status)
            for status in sorted({event.status for event in d1_events})
        },
    }


def guard_params(base_params: dict[str, Any], multiplier: float) -> dict[str, Any]:
    params = dict(base_params)
    params.update(
        {
            "htf_pa_ict_guard_enabled": multiplier < 1.0,
            "htf_pa_ict_guard_multiplier": multiplier,
            "htf_pa_ict_guard_directions": ["BULL"],
            "htf_pa_ict_guard_h4_alignment": "opposed",
            "htf_pa_ict_guard_d1_alignment": "opposed",
            "htf_pa_ict_guard_h4_states": ["bearish"],
            "htf_pa_ict_guard_d1_states": ["bearish"],
        }
    )
    return params


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_return_pct": result.get("total_return_pct"),
        "max_drawdown_pct": result.get("max_drawdown_pct"),
        "sharpe_ratio": result.get("sharpe_ratio"),
        "accepted_trades": result.get("accepted_trades"),
        "skipped_trades": result.get("skipped_trades"),
        "htf_pa_ict_guard_applied": result.get("htf_pa_ict_guard_applied"),
        "trigger_counts": result.get("trigger_counts"),
        "windows": result.get("windows", {}),
    }


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_for_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return clean_for_json(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return None if pd.isna(value) else value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def run_candidate(
    trades: pd.DataFrame,
    initial_capital: float,
    base_params: dict[str, Any],
    shadow_params: dict[str, Any],
    multiplier: float,
    include_events: bool,
) -> dict[str, Any]:
    fixed = expansion_overlay(
        trades,
        initial_capital,
        guard_params(base_params, multiplier),
        include_events=True,
    )
    fixed_events = add_strategy_trade_fields(fixed["events"], trades)
    shadow = replay_shadow_events(
        fixed_events,
        initial_capital,
        daily_loss_stop_pct=float(shadow_params["daily_loss_stop_pct"]),
        equity_drawdown_stop_pct=float(shadow_params["equity_drawdown_stop_pct"]),
        consecutive_loss_stop=int(shadow_params["consecutive_loss_stop"]),
        equity_drawdown_cooldown_days=int(shadow_params["equity_drawdown_cooldown_days"]),
    )
    shadow["windows"] = window_metrics_from_events(shadow["events"], initial_capital)
    shadow["htf_pa_ict_guard_applied"] = fixed.get("htf_pa_ict_guard_applied", 0)
    result = {
        "multiplier": multiplier,
        "fixed_structure_overlay": {key: value for key, value in fixed.items() if key != "events"},
        "shadow": {key: value for key, value in shadow.items() if key != "events"},
    }
    if include_events:
        result["fixed_events"] = fixed["events"]
        result["shadow_events"] = shadow["events"]
    return result


def main() -> None:
    args = parse_args()
    best_payload = load_best_params(Path(args.best_params))
    payload = load_config_payload(Path(args.config))
    payload, pressure_params = apply_best_pressure_params(payload, args.pressure_params)
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=payload.get("regime_switcher_thresholds"),
    )
    metrics, engine = run_engine(payload, prepared, args.start_date)
    trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
    trades, htf_event_counts = annotate_trades_with_htf_context(trades, args)
    initial_capital = float(metrics.get("initial_capital", 1000.0) or 1000.0)
    base_params = promoted_fixed_params(best_payload)
    shadow_params = promoted_shadow_params(best_payload)

    results = [
        run_candidate(
            trades=trades,
            initial_capital=initial_capital,
            base_params=base_params,
            shadow_params=shadow_params,
            multiplier=multiplier,
            include_events=args.include_events,
        )
        for multiplier in parse_float_list(args.multipliers)
    ]
    report = {
        "metadata": {
            "config": str(Path(args.config).resolve()),
            "best_params": str(Path(args.best_params).resolve()),
            "pressure_params_path": None
            if str(args.pressure_params).lower() == "none"
            else str(Path(args.pressure_params).resolve()),
            "pressure_params": pressure_params,
            "start_date": args.start_date,
            "data": {
                "data_15m": str(Path(args.data_15m).resolve()),
                "data_4h": str(Path(args.data_4h).resolve()),
                "start": str(prepared.start),
                "end": str(prepared.end),
                "candles_15m": len(prepared.c15m),
                "candles_4h": len(prepared.c4h),
            },
            "engine": metrics,
            "htf_event_counts": htf_event_counts,
            "shadow_params": shadow_params,
        },
        "guard_rule": {
            "description": "BULL trade + active 4H opposed bearish context + active 1D opposed bearish context.",
            "h4_context_ttl_bars": args.h4_context_ttl_bars,
            "d1_context_ttl_bars": args.d1_context_ttl_bars,
            "require_mss": args.require_mss,
        },
        "results": results,
        "compact": [
            {
                "multiplier": item["multiplier"],
                "fixed": summarize_result(item["fixed_structure_overlay"]),
                "shadow": summarize_result(item["shadow"]),
            }
            for item in results
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned = clean_for_json(report)
    output.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    print(json.dumps(cleaned["compact"], ensure_ascii=False, indent=2, allow_nan=False))
    if args.stdout:
        print(json.dumps(cleaned, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
