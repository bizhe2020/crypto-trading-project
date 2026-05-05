#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_readiness_report import _high_leverage_trade_diagnostics, load_prepared_data  # noqa: E402
from scripts.report_smc_trade_context import daily_candles_from_4h  # noqa: E402
from scripts.reproduce_smc_short_only_v1_10x import strategy_args  # noqa: E402
from scripts.reproduce_smc_standalone_v2_1_10x import summarize_leveraged_rows, window_summary, yearly_summary  # noqa: E402
from scripts.research_smc_standalone_v1 import apply_max_open_positions, build_event_scan_args, clean_for_json  # noqa: E402
from scripts.research_smc_standalone_v1 import scan_events, trade_rows_for_events  # noqa: E402
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_short_only_v1_aggressive_loss_attribution.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Loss attribution for aggressive standalone SMC short-only v1.")
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--target-rr", type=float, default=2.0)
    parser.add_argument("--allowed-time-buckets", default="other+asia_evening_ny+ny_am_killzone")
    parser.add_argument("--swing-n", type=int, default=3)
    parser.add_argument("--min-body-atr", type=float, default=0.7)
    parser.add_argument("--min-range-atr", type=float, default=1.1)
    parser.add_argument("--entry-lookahead-bars", type=int, default=40)
    parser.add_argument("--max-open-positions", type=int, default=1)
    parser.add_argument("--min-displacement-body-atr", type=float, default=0.0)
    parser.add_argument("--min-displacement-range-atr", type=float, default=0.0)
    parser.add_argument("--max-mss-lag-bars", type=int, default=15)
    parser.add_argument("--ny-max-mss-lag-bars", type=int, default=0)
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--position-size-pct", type=float, default=1.0)
    parser.add_argument("--maintenance-margin-pct", type=float, default=0.5)
    parser.add_argument("--min-liq-buffer-pct", type=float, default=1.2)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def stop_distance_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.4:
        return "<0.4%"
    if value < 0.6:
        return "0.4-0.6%"
    if value < 0.8:
        return "0.6-0.8%"
    return ">=0.8%"


def mss_lag_bucket(value: int | None) -> str:
    if value is None:
        return "unknown"
    if value <= 3:
        return "<=3"
    if value <= 6:
        return "4-6"
    if value <= 9:
        return "7-9"
    if value <= 12:
        return "10-12"
    return ">=13"


def mfe_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.25:
        return "<0.25R"
    if value < 0.5:
        return "0.25-0.5R"
    if value < 1.0:
        return "0.5-1.0R"
    if value < 1.5:
        return "1.0-1.5R"
    return ">=1.5R"


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.median(values)), 4)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.mean(values)), 4)


def bucket_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    out: list[dict[str, Any]] = []
    for bucket, bucket_rows in sorted(grouped.items()):
        wins = sum(1 for row in bucket_rows if float(row["rr_result"]) > 0)
        losses = sum(1 for row in bucket_rows if float(row["rr_result"]) < 0)
        out.append(
            {
                "bucket": bucket,
                "trades": len(bucket_rows),
                "wins": wins,
                "losses": losses,
                "win_rate_pct": round(wins / len(bucket_rows) * 100.0, 2) if bucket_rows else 0.0,
                "total_return_pct": round(sum(float(row["leveraged_return"]) for row in bucket_rows) * 100.0, 4),
                "avg_mfe_r": mean_or_none([float(row["mfe_r"]) for row in bucket_rows if row["mfe_r"] is not None]),
                "median_mfe_r": median_or_none([float(row["mfe_r"]) for row in bucket_rows if row["mfe_r"] is not None]),
            }
        )
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
    if int(args.ny_max_mss_lag_bars) > 0:
        ny_limit = int(args.ny_max_mss_lag_bars)
        rows = [
            row
            for row in rows
            if row["time_bucket"] != "ny_am_killzone"
            or row["mss_lag_bars"] is None
            or int(row["mss_lag_bars"]) <= ny_limit
        ]
    raw_trade_count = len(rows)
    rows, slot_skipped = apply_max_open_positions(rows, int(smc_args.max_open_positions))

    capital = float(args.initial_capital)
    accepted_rows: list[dict[str, Any]] = []
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
            guard_skipped += 1
            continue
        leveraged_return = float(row["signal_return_pct"] or 0.0) / 100.0 * float(args.leverage) * float(args.position_size_pct)
        capital *= 1.0 + leveraged_return
        accepted_rows.append(
            {
                **row,
                "leveraged_return": leveraged_return,
                "stop_distance_bucket": stop_distance_bucket(row["stop_distance_pct"]),
                "mss_lag_bucket": mss_lag_bucket(row["mss_lag_bars"]),
                "mfe_bucket": mfe_bucket(row["mfe_r"]),
                "liquidation_buffer_pct": diagnostics["liquidation_buffer_pct"],
            }
        )

    losses = [row for row in accepted_rows if float(row["rr_result"]) < 0]
    wins = [row for row in accepted_rows if float(row["rr_result"]) > 0]

    report = {
        "metadata": {
            "start_date": args.start_date,
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
        },
        "strategy_params": clean_for_json(vars(smc_args)),
        "execution_summary": {
            "raw_trades": raw_trade_count,
            "slot_trades": len(rows),
            "slot_skipped_trades": slot_skipped,
            "guard_skipped_trades": guard_skipped,
        },
        "overall": summarize_leveraged_rows(accepted_rows, float(args.initial_capital)),
        "yearly": yearly_summary(accepted_rows, float(args.initial_capital)),
        "windows": window_summary(accepted_rows, float(args.initial_capital)),
        "loss_overview": {
            "loss_count": len(losses),
            "loss_return_sum_pct": round(sum(float(row["leveraged_return"]) for row in losses) * 100.0, 4),
            "win_count": len(wins),
            "win_return_sum_pct": round(sum(float(row["leveraged_return"]) for row in wins) * 100.0, 4),
            "losses_with_mfe_ge_1r": sum(1 for row in losses if float(row["mfe_r"] or 0.0) >= 1.0),
            "losses_with_mfe_lt_0_5r": sum(1 for row in losses if float(row["mfe_r"] or 0.0) < 0.5),
            "losses_with_mss_lag_ge_10": sum(1 for row in losses if int(row["mss_lag_bars"] or 0) >= 10),
        },
        "bucket_summaries": {
            "time_bucket": bucket_summary(accepted_rows, "time_bucket"),
            "mss_lag_bucket": bucket_summary(accepted_rows, "mss_lag_bucket"),
            "stop_distance_bucket": bucket_summary(accepted_rows, "stop_distance_bucket"),
            "mfe_bucket": bucket_summary(accepted_rows, "mfe_bucket"),
        },
        "loss_mfe_stats": {
            "mean_mfe_r": mean_or_none([float(row["mfe_r"]) for row in losses if row["mfe_r"] is not None]),
            "median_mfe_r": median_or_none([float(row["mfe_r"]) for row in losses if row["mfe_r"] is not None]),
        },
        "win_mfe_stats": {
            "mean_mfe_r": mean_or_none([float(row["mfe_r"]) for row in wins if row["mfe_r"] is not None]),
            "median_mfe_r": median_or_none([float(row["mfe_r"]) for row in wins if row["mfe_r"] is not None]),
        },
        "losses": [
            {
                "entry_time": row["entry_time"],
                "exit_time": row["exit_time"],
                "time_bucket": row["time_bucket"],
                "entry_price": row["entry_price"],
                "stop_distance_pct": round(float(row["stop_distance_pct"] or 0.0), 6),
                "sweep_distance_pct": round(float(row["sweep_distance_pct"] or 0.0), 6),
                "mss_lag_bars": row["mss_lag_bars"],
                "displacement_body_atr": row["displacement_body_atr"],
                "displacement_range_atr": row["displacement_range_atr"],
                "fvg_touched": row["fvg_touched"],
                "fvg_size_pct": row["fvg_size_pct"],
                "mfe_r": row["mfe_r"],
                "mae_r": row["mae_r"],
                "outcome": row["outcome"],
                "leveraged_return_pct": round(float(row["leveraged_return"]) * 100.0, 4),
            }
            for row in losses
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    print(
        json.dumps(
            {
                "overall": report["overall"],
                "loss_overview": report["loss_overview"],
                "time_bucket": report["bucket_summaries"]["time_bucket"],
                "mss_lag_bucket": report["bucket_summaries"]["mss_lag_bucket"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
