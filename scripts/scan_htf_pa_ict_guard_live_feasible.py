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
    parse_int_list,
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


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "htf_context" / "htf_pa_ict_guard_live_feasible_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan live-feasible HTF PA/ICT guard settings before shadow replay."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--best-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--multipliers", default="0.70,0.72,0.75,0.80,1.0")
    parser.add_argument("--h4-context-ttl-bars-values", default="24,36,42,48,60")
    parser.add_argument("--d1-context-ttl-bars-values", default="7,10,14,21")
    parser.add_argument("--shadow-daily-loss-values", default=None)
    parser.add_argument("--shadow-equity-dd-values", default=None)
    parser.add_argument("--shadow-cooldown-days-values", default=None)
    parser.add_argument("--shadow-loss-streak-values", default=None)
    parser.add_argument("--top", type=int, default=30)
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


def float_values_or_default(value: str | None, default: float) -> list[float]:
    return parse_float_list(value) if value else [float(default)]


def int_values_or_default(value: str | None, default: int) -> list[int]:
    return parse_int_list(value) if value else [int(default)]


def annotate_for_ttl(
    trades: pd.DataFrame,
    h4_events: list[Any],
    h4_candles: list[Any],
    d1_events: list[Any],
    d1_candles: list[Any],
    args: argparse.Namespace,
    h4_ttl: int,
    d1_ttl: int,
) -> pd.DataFrame:
    args.h4_context_ttl_bars = h4_ttl
    args.d1_context_ttl_bars = d1_ttl
    out = trades.sort_values("entry_time").reset_index(drop=True).copy()
    enriched = add_htf_contexts(
        out.to_dict("records"),
        h4_events,
        h4_candles,
        d1_events,
        d1_candles,
        args,
    )
    htf_columns = sorted({key for event in enriched for key in event if key.startswith("htf_")})
    for column in htf_columns:
        out[column] = [event.get(column) for event in enriched]
    return out


def run_shadow(
    events: list[dict[str, Any]],
    initial_capital: float,
    shadow_params: dict[str, Any],
) -> dict[str, Any]:
    shadow = replay_shadow_events(
        events,
        initial_capital,
        daily_loss_stop_pct=float(shadow_params["daily_loss_stop_pct"]),
        equity_drawdown_stop_pct=float(shadow_params["equity_drawdown_stop_pct"]),
        consecutive_loss_stop=int(shadow_params["consecutive_loss_stop"]),
        equity_drawdown_cooldown_days=int(shadow_params["equity_drawdown_cooldown_days"]),
    )
    shadow["windows"] = window_metrics_from_events(shadow["events"], initial_capital)
    return shadow


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
    base_trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
    base_trades = base_trades.sort_values("entry_time").reset_index(drop=True)
    initial_capital = float(metrics.get("initial_capital", 1000.0) or 1000.0)
    fixed_params = promoted_fixed_params(best_payload)
    shadow_params = promoted_shadow_params(best_payload)

    df4 = load_df(Path(args.data_4h))
    h4_candles = dataframe_to_candles(df4)
    d1_candles = dataframe_to_candles(daily_from_4h(df4))
    h4_events = scan_events(h4_candles, namespace_for(args, "4h"))
    d1_events = scan_events(d1_candles, namespace_for(args, "1d"))
    shadow_grid = [
        {
            "daily_loss_stop_pct": daily_loss,
            "equity_drawdown_stop_pct": equity_dd,
            "equity_drawdown_cooldown_days": cooldown_days,
            "consecutive_loss_stop": loss_streak,
        }
        for daily_loss in float_values_or_default(
            args.shadow_daily_loss_values,
            float(shadow_params["daily_loss_stop_pct"]),
        )
        for equity_dd in float_values_or_default(
            args.shadow_equity_dd_values,
            float(shadow_params["equity_drawdown_stop_pct"]),
        )
        for cooldown_days in int_values_or_default(
            args.shadow_cooldown_days_values,
            int(shadow_params["equity_drawdown_cooldown_days"]),
        )
        for loss_streak in int_values_or_default(
            args.shadow_loss_streak_values,
            int(shadow_params["consecutive_loss_stop"]),
        )
    ]

    rows: list[dict[str, Any]] = []
    for h4_ttl in parse_int_list(args.h4_context_ttl_bars_values):
        for d1_ttl in parse_int_list(args.d1_context_ttl_bars_values):
            trades = annotate_for_ttl(
                base_trades,
                h4_events,
                h4_candles,
                d1_events,
                d1_candles,
                args,
                h4_ttl,
                d1_ttl,
            )
            for multiplier in parse_float_list(args.multipliers):
                params = guard_params(fixed_params, multiplier)
                fixed = expansion_overlay(trades, initial_capital, params, include_events=True)
                fixed_events = add_strategy_trade_fields(fixed["events"], trades)
                for shadow_candidate in shadow_grid:
                    shadow = run_shadow(fixed_events, initial_capital, shadow_candidate)
                    current_year = shadow["windows"].get("current_year", {})
                    last_60d = shadow["windows"].get("last_60d", {})
                    last_30d = shadow["windows"].get("last_30d", {})
                    rows.append(
                        {
                            "h4_ttl": h4_ttl,
                            "d1_ttl": d1_ttl,
                            "multiplier": multiplier,
                            "shadow_daily_loss_stop_pct": shadow_candidate["daily_loss_stop_pct"],
                            "shadow_equity_drawdown_stop_pct": shadow_candidate["equity_drawdown_stop_pct"],
                            "shadow_equity_drawdown_cooldown_days": shadow_candidate["equity_drawdown_cooldown_days"],
                            "shadow_consecutive_loss_stop": shadow_candidate["consecutive_loss_stop"],
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
                    )

    rows.sort(
        key=lambda item: (
            float(item["total_return_pct"]),
            float(item.get("current_year_return_pct") or -999.0),
        ),
        reverse=True,
    )
    report = {
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
            "htf_event_counts": {"h4": len(h4_events), "d1": len(d1_events)},
            "shadow_params": shadow_params,
            "multipliers": args.multipliers,
            "h4_context_ttl_bars_values": args.h4_context_ttl_bars_values,
            "d1_context_ttl_bars_values": args.d1_context_ttl_bars_values,
            "shadow_daily_loss_values": args.shadow_daily_loss_values,
            "shadow_equity_dd_values": args.shadow_equity_dd_values,
            "shadow_cooldown_days_values": args.shadow_cooldown_days_values,
            "shadow_loss_streak_values": args.shadow_loss_streak_values,
        },
        "top": rows[: args.top],
        "all": rows,
    }
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
