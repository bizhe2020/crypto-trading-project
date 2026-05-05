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
from scripts.reproduce_htf_pa_ict_guard import clean_for_json, guard_params  # noqa: E402
from scripts.scan_high_leverage_expansion import (  # noqa: E402
    enrich_trades_with_regime_features,
    expansion_overlay,
    parse_float_list,
)
from scripts.scan_htf_pa_ict_guard_live_feasible import run_shadow  # noqa: E402
from scripts.scan_pa_ict_shadow_quality_overlay import (  # noqa: E402
    add_strategy_trade_fields,
    apply_best_pressure_params,
    load_best_params,
    promoted_fixed_params,
    promoted_shadow_params,
)
from strategy.scalp_robust_v2_core import dataframe_to_candles  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "htf_context" / "htf_pa_ict_generator_param_scan.json"

H4_PRESETS: dict[str, dict[str, int]] = {
    "current": {
        "swing_lookback": 30,
        "liquidity_lookback_bars": 180,
        "mss_lookahead_bars": 12,
        "fvg_lookback_bars": 6,
    },
    "tight": {
        "swing_lookback": 24,
        "liquidity_lookback_bars": 120,
        "mss_lookahead_bars": 8,
        "fvg_lookback_bars": 4,
    },
    "quick_mss": {
        "swing_lookback": 30,
        "liquidity_lookback_bars": 180,
        "mss_lookahead_bars": 8,
        "fvg_lookback_bars": 4,
    },
    "deep_liq": {
        "swing_lookback": 30,
        "liquidity_lookback_bars": 240,
        "mss_lookahead_bars": 12,
        "fvg_lookback_bars": 6,
    },
    "wide": {
        "swing_lookback": 36,
        "liquidity_lookback_bars": 240,
        "mss_lookahead_bars": 16,
        "fvg_lookback_bars": 8,
    },
}

D1_PRESETS: dict[str, dict[str, int]] = {
    "current": {
        "swing_lookback": 20,
        "liquidity_lookback_bars": 90,
        "mss_lookahead_bars": 5,
        "fvg_lookback_bars": 4,
    },
    "tight": {
        "swing_lookback": 16,
        "liquidity_lookback_bars": 60,
        "mss_lookahead_bars": 4,
        "fvg_lookback_bars": 3,
    },
    "deep_liq": {
        "swing_lookback": 20,
        "liquidity_lookback_bars": 120,
        "mss_lookahead_bars": 5,
        "fvg_lookback_bars": 4,
    },
    "wide": {
        "swing_lookback": 24,
        "liquidity_lookback_bars": 120,
        "mss_lookahead_bars": 7,
        "fvg_lookback_bars": 6,
    },
}

DISPLACEMENT_PRESETS: dict[str, dict[str, float]] = {
    "loose": {"min_body_atr": 0.5, "min_range_atr": 0.9},
    "current": {"min_body_atr": 0.7, "min_range_atr": 1.1},
    "strict": {"min_body_atr": 0.9, "min_range_atr": 1.3},
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan low-level HTF PA/ICT event generator parameters for the double-opposed guard."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--best-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--multipliers", default="0.70,0.72")
    parser.add_argument("--h4-presets", default="current,tight,quick_mss,deep_liq,wide")
    parser.add_argument("--d1-presets", default="current,tight,deep_liq,wide")
    parser.add_argument("--displacement-presets", default="loose,current,strict")
    parser.add_argument("--swing-n-values", default="2,3")
    parser.add_argument("--h4-context-ttl-bars", type=int, default=42)
    parser.add_argument("--d1-context-ttl-bars", type=int, default=14)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument("--stdout", action="store_true")

    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.05)
    parser.add_argument("--target-rr", type=float, default=2.0)
    parser.add_argument("--require-mss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--h4-entry-lookahead-bars", type=int, default=18)
    parser.add_argument("--h4-outcome-lookahead-bars", type=int, default=36)
    parser.add_argument("--d1-entry-lookahead-bars", type=int, default=10)
    parser.add_argument("--d1-outcome-lookahead-bars", type=int, default=20)
    return parser.parse_args()


def set_prefixed_params(args: argparse.Namespace, prefix: str, preset: dict[str, int]) -> None:
    for key, value in preset.items():
        setattr(args, f"{prefix}_{key}", value)


def apply_scan_params(
    base_args: argparse.Namespace,
    *,
    swing_n: int,
    h4_preset_name: str,
    d1_preset_name: str,
    displacement_name: str,
) -> argparse.Namespace:
    args = argparse.Namespace(**vars(base_args))
    args.swing_n = swing_n
    set_prefixed_params(args, "h4", H4_PRESETS[h4_preset_name])
    set_prefixed_params(args, "d1", D1_PRESETS[d1_preset_name])
    displacement = DISPLACEMENT_PRESETS[displacement_name]
    args.min_body_atr = displacement["min_body_atr"]
    args.min_range_atr = displacement["min_range_atr"]
    return args


def context_columns(events: list[dict[str, Any]]) -> list[str]:
    return sorted({key for event in events for key in event if key.startswith("htf_")})


def annotate_trades(
    trades: pd.DataFrame,
    h4_events: list[Any],
    h4_candles: list[Any],
    d1_events: list[Any],
    d1_candles: list[Any],
    args: argparse.Namespace,
) -> pd.DataFrame:
    out = trades.sort_values("entry_time").reset_index(drop=True).copy()
    enriched = add_htf_contexts(
        out.to_dict("records"),
        h4_events,
        h4_candles,
        d1_events,
        d1_candles,
        args,
    )
    for column in context_columns(enriched):
        out[column] = [event.get(column) for event in enriched]
    return out


def main() -> None:
    args = parse_args()
    missing = [
        name
        for name in parse_csv(args.h4_presets)
        if name not in H4_PRESETS
    ] + [
        name
        for name in parse_csv(args.d1_presets)
        if name not in D1_PRESETS
    ] + [
        name
        for name in parse_csv(args.displacement_presets)
        if name not in DISPLACEMENT_PRESETS
    ]
    if missing:
        raise ValueError(f"Unknown preset(s): {', '.join(missing)}")

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
    base_trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
    base_trades = base_trades.sort_values("entry_time").reset_index(drop=True)
    initial_capital = float(metrics.get("initial_capital", 1000.0) or 1000.0)
    fixed_params = promoted_fixed_params(best_payload)
    shadow_params = promoted_shadow_params(best_payload)

    df4 = load_df(Path(args.data_4h))
    h4_candles = dataframe_to_candles(df4)
    d1_candles = dataframe_to_candles(daily_from_4h(df4))

    rows: list[dict[str, Any]] = []
    top_events: list[dict[str, Any]] = []
    for swing_n_text in parse_csv(args.swing_n_values):
        swing_n = int(swing_n_text)
        for h4_name in parse_csv(args.h4_presets):
            for d1_name in parse_csv(args.d1_presets):
                for displacement_name in parse_csv(args.displacement_presets):
                    scan_args = apply_scan_params(
                        args,
                        swing_n=swing_n,
                        h4_preset_name=h4_name,
                        d1_preset_name=d1_name,
                        displacement_name=displacement_name,
                    )
                    h4_events = scan_events(h4_candles, namespace_for(scan_args, "4h"))
                    d1_events = scan_events(d1_candles, namespace_for(scan_args, "1d"))
                    trades = annotate_trades(
                        base_trades,
                        h4_events,
                        h4_candles,
                        d1_events,
                        d1_candles,
                        scan_args,
                    )
                    for multiplier in parse_float_list(args.multipliers):
                        params = guard_params(fixed_params, multiplier)
                        fixed = expansion_overlay(trades, initial_capital, params, include_events=True)
                        fixed_events = add_strategy_trade_fields(fixed["events"], trades)
                        shadow = run_shadow(fixed_events, initial_capital, shadow_params)
                        current_year = shadow["windows"].get("current_year", {})
                        last_60d = shadow["windows"].get("last_60d", {})
                        last_30d = shadow["windows"].get("last_30d", {})
                        row = {
                            "swing_n": swing_n,
                            "h4_preset": h4_name,
                            "d1_preset": d1_name,
                            "displacement_preset": displacement_name,
                            "h4_params": H4_PRESETS[h4_name],
                            "d1_params": D1_PRESETS[d1_name],
                            "min_body_atr": DISPLACEMENT_PRESETS[displacement_name]["min_body_atr"],
                            "min_range_atr": DISPLACEMENT_PRESETS[displacement_name]["min_range_atr"],
                            "h4_context_ttl_bars": scan_args.h4_context_ttl_bars,
                            "d1_context_ttl_bars": scan_args.d1_context_ttl_bars,
                            "multiplier": multiplier,
                            "h4_event_count": len(h4_events),
                            "d1_event_count": len(d1_events),
                            "total_return_pct": shadow["total_return_pct"],
                            "max_drawdown_pct": shadow["max_drawdown_pct"],
                            "sharpe_ratio": shadow["sharpe_ratio"],
                            "current_year_return_pct": current_year.get("total_return_pct"),
                            "current_year_max_drawdown_pct": current_year.get("max_drawdown_pct"),
                            "last_60d_return_pct": last_60d.get("total_return_pct"),
                            "last_30d_return_pct": last_30d.get("total_return_pct"),
                            "accepted_trades": shadow["accepted_trades"],
                            "skipped_trades": shadow["skipped_trades"],
                            "guard_applied": fixed.get("htf_pa_ict_guard_applied"),
                            "trigger_counts": shadow.get("trigger_counts", {}),
                        }
                        rows.append(row)
                        if args.include_events:
                            maybe_best = not top_events or float(row["total_return_pct"]) > float(top_events[0]["total_return_pct"])
                            if maybe_best:
                                top_events = [
                                    row | {"events": shadow["events"]},
                                ]

    rows.sort(
        key=lambda item: (
            float(item["total_return_pct"]),
            float(item.get("current_year_return_pct") or -999.0),
            -float(item["max_drawdown_pct"]),
        ),
        reverse=True,
    )
    report: dict[str, Any] = {
        "metadata": {
            "config": str(Path(args.config).resolve()),
            "best_params": str(Path(args.best_params).resolve()),
            "pressure_params_path": None
            if str(args.pressure_params).lower() == "none"
            else str(Path(args.pressure_params).resolve()),
            "pressure_params": pressure_params,
            "data": {
                "data_15m": str(Path(args.data_15m).resolve()),
                "data_4h": str(Path(args.data_4h).resolve()),
                "start": str(prepared.start),
                "end": str(prepared.end),
                "candles_15m": len(prepared.c15m),
                "candles_4h": len(prepared.c4h),
            },
            "engine": metrics,
            "shadow_params": shadow_params,
            "multipliers": args.multipliers,
            "h4_presets": args.h4_presets,
            "d1_presets": args.d1_presets,
            "displacement_presets": args.displacement_presets,
            "swing_n_values": args.swing_n_values,
            "require_mss": args.require_mss,
        },
        "candidate_count": len(rows),
        "top": rows[: args.top],
        "all": rows,
    }
    if args.include_events and top_events:
        report["top_events"] = top_events

    cleaned = clean_for_json(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    print(json.dumps(cleaned["top"], ensure_ascii=False, indent=2, allow_nan=False))
    if args.stdout:
        print(json.dumps(cleaned, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
