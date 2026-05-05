#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_readiness_report import _high_leverage_trade_diagnostics, load_prepared_data  # noqa: E402
from scripts.report_smc_trade_context import daily_candles_from_4h  # noqa: E402
from scripts.reproduce_smc_standalone_v2_1_10x import summarize_leveraged_rows, window_summary, yearly_summary  # noqa: E402
from scripts.reproduce_smc_standalone_v2_1_10x import strategy_args as v2_1_strategy_args  # noqa: E402
from scripts.research_smc_standalone_v1 import apply_max_open_positions, build_event_scan_args, clean_for_json  # noqa: E402
from scripts.research_smc_standalone_v1 import scan_events, trade_rows_for_events  # noqa: E402
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_short_only_v1_10x_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce standalone SMC short-only v1 under 10x leverage audit.")
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--target-rr", type=float, default=2.0)
    parser.add_argument("--allowed-time-buckets", default="other")
    parser.add_argument("--swing-n", type=int, default=3)
    parser.add_argument("--min-body-atr", type=float, default=0.7)
    parser.add_argument("--min-range-atr", type=float, default=1.1)
    parser.add_argument("--entry-lookahead-bars", type=int, default=40)
    parser.add_argument("--max-open-positions", type=int, default=1)
    parser.add_argument("--min-displacement-body-atr", type=float, default=0.0)
    parser.add_argument("--min-displacement-range-atr", type=float, default=0.0)
    parser.add_argument("--max-mss-lag-bars", type=int, default=15)
    parser.add_argument("--global-min-mss-lag-bars", type=int, default=0)
    parser.add_argument("--global-max-mss-lag-bars", type=int, default=0)
    parser.add_argument("--ny-max-mss-lag-bars", type=int, default=0)
    parser.add_argument("--other-min-mss-lag-bars", type=int, default=0)
    parser.add_argument("--drop-asia-session", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--position-size-pct", type=float, default=1.0)
    parser.add_argument("--maintenance-margin-pct", type=float, default=0.5)
    parser.add_argument("--min-liq-buffer-pct", type=float, default=1.2)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def strategy_args(args: argparse.Namespace) -> argparse.Namespace:
    out = v2_1_strategy_args(args)
    out.swing_n = int(args.swing_n)
    out.min_body_atr = float(args.min_body_atr)
    out.min_range_atr = float(args.min_range_atr)
    out.entry_lookahead_bars = int(args.entry_lookahead_bars)
    out.max_open_positions = int(args.max_open_positions)
    out.min_displacement_body_atr = float(args.min_displacement_body_atr)
    out.min_displacement_range_atr = float(args.min_displacement_range_atr)
    out.allowed_directions = "BEAR"
    out.target_rr = float(args.target_rr)
    out.allowed_time_buckets = str(args.allowed_time_buckets)
    out.max_mss_lag_bars = int(args.max_mss_lag_bars)
    return out


def main() -> None:
    args = parse_args()
    smc_args = strategy_args(args)
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=None,
    )
    daily = daily_candles_from_4h(prepared.c4h)
    h4_highs, h4_lows = precompute_swings(prepared.c4h, n=2, lookback=80)
    d1_highs, d1_lows = precompute_swings(daily, n=2, lookback=20)
    rows = trade_rows_for_events(
        scan_events(prepared.c15m, build_event_scan_args(smc_args)),
        prepared,
        daily,
        h4_highs,
        h4_lows,
        d1_highs,
        d1_lows,
        smc_args,
    )
    if int(args.global_min_mss_lag_bars) > 0:
        floor = int(args.global_min_mss_lag_bars)
        rows = [
            row
            for row in rows
            if row["mss_lag_bars"] is None or int(row["mss_lag_bars"]) >= floor
        ]
    if int(args.global_max_mss_lag_bars) > 0:
        ceiling = int(args.global_max_mss_lag_bars)
        rows = [
            row
            for row in rows
            if row["mss_lag_bars"] is None or int(row["mss_lag_bars"]) <= ceiling
        ]
    if int(args.ny_max_mss_lag_bars) > 0:
        ny_limit = int(args.ny_max_mss_lag_bars)
        rows = [
            row
            for row in rows
            if row["time_bucket"] != "ny_am_killzone"
            or row["mss_lag_bars"] is None
            or int(row["mss_lag_bars"]) <= ny_limit
        ]
    if int(args.other_min_mss_lag_bars) > 0:
        other_floor = int(args.other_min_mss_lag_bars)
        rows = [
            row
            for row in rows
            if row["time_bucket"] != "other"
            or row["mss_lag_bars"] is None
            or int(row["mss_lag_bars"]) >= other_floor
        ]
    if bool(args.drop_asia_session):
        rows = [row for row in rows if row["time_bucket"] != "asia_evening_ny"]
    raw_trade_count = len(rows)
    rows, slot_skipped = apply_max_open_positions(rows, int(smc_args.max_open_positions))

    capital = float(args.initial_capital)
    accepted_rows: list[dict[str, object]] = []
    failures: dict[str, int] = {}
    guard_skipped = 0
    for row in rows:
        trade = pd.Series(
            {
                "entry_time": row["entry_time"],
                "direction": row["direction"],
                "entry_price": row["entry_price"],
                "initial_stop_price": row["stop_price"],
                "notional": capital * float(args.leverage) * float(args.position_size_pct),
            }
        )
        diagnostics = _high_leverage_trade_diagnostics(
            trade,
            capital=capital,
            leverage=float(args.leverage),
            maintenance_margin_pct=float(args.maintenance_margin_pct),
        )
        if float(diagnostics["liquidation_buffer_pct"]) < float(args.min_liq_buffer_pct):
            failures["liquidation_buffer_too_small"] = failures.get("liquidation_buffer_too_small", 0) + 1
            guard_skipped += 1
            continue
        leveraged_return = float(row["signal_return_pct"] or 0.0) / 100.0 * float(args.leverage) * float(args.position_size_pct)
        capital *= 1.0 + leveraged_return
        accepted_rows.append(
            {
                **row,
                "leveraged_return": leveraged_return,
                "shadow_capital": capital,
                "liquidation_buffer_pct": diagnostics["liquidation_buffer_pct"],
                "account_effective_leverage": diagnostics["account_effective_leverage"],
            }
        )

    report = {
        "metadata": {
            "start_date": args.start_date,
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
        },
        "strategy_params": clean_for_json(vars(smc_args)),
        "session_filters": {
            "global_min_mss_lag_bars": int(args.global_min_mss_lag_bars),
            "global_max_mss_lag_bars": int(args.global_max_mss_lag_bars),
            "ny_max_mss_lag_bars": int(args.ny_max_mss_lag_bars),
            "other_min_mss_lag_bars": int(args.other_min_mss_lag_bars),
            "drop_asia_session": bool(args.drop_asia_session),
        },
        "leverage_params": {
            "leverage": args.leverage,
            "position_size_pct": args.position_size_pct,
            "maintenance_margin_pct": args.maintenance_margin_pct,
            "min_liq_buffer_pct": args.min_liq_buffer_pct,
            "initial_capital": args.initial_capital,
        },
        "execution_summary": {
            "raw_trades": raw_trade_count,
            "slot_trades": len(rows),
            "slot_skipped_trades": slot_skipped,
            "guard_skipped_trades": guard_skipped,
            "failure_counts": failures,
        },
        "overall": summarize_leveraged_rows(accepted_rows, float(args.initial_capital)),
        "yearly": yearly_summary(accepted_rows, float(args.initial_capital)),
        "windows": window_summary(accepted_rows, float(args.initial_capital)),
        "risk_diagnostics": {
            "min_liquidation_buffer_pct": round(min((float(row["liquidation_buffer_pct"]) for row in accepted_rows), default=0.0), 4),
            "max_account_effective_leverage": round(max((float(row["account_effective_leverage"]) for row in accepted_rows), default=0.0), 4),
            "max_stop_distance_pct": round(max((float(row["stop_distance_pct"] or 0.0) for row in accepted_rows), default=0.0), 4),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    print(
        json.dumps(
            {
                "overall": report["overall"],
                "yearly": report["yearly"],
                "windows": report["windows"],
                "risk_diagnostics": report["risk_diagnostics"],
                "execution_summary": report["execution_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
