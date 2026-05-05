#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_readiness_report import _high_leverage_trade_diagnostics, load_prepared_data, max_drawdown_from_capitals  # noqa: E402
from scripts.report_smc_trade_context import daily_candles_from_4h  # noqa: E402
from scripts.research_smc_standalone_v1 import apply_max_open_positions, build_event_scan_args, clean_for_json  # noqa: E402
from scripts.research_smc_standalone_v1 import scan_events, trade_rows_for_events  # noqa: E402
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_standalone_v2_1_10x_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce standalone SMC v2.1 core under 10x leverage audit.")
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--position-size-pct", type=float, default=1.0)
    parser.add_argument("--maintenance-margin-pct", type=float, default=0.5)
    parser.add_argument("--min-liq-buffer-pct", type=float, default=1.2)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def strategy_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        data_15m=args.data_15m,
        data_4h=args.data_4h,
        start_date=args.start_date,
        swing_n=3,
        swing_lookback=80,
        liquidity_lookback_bars=192,
        mss_lookahead_bars=24,
        fvg_lookback_bars=8,
        entry_lookahead_bars=40,
        outcome_lookahead_bars=96,
        atr_period=14,
        min_body_atr=0.7,
        min_range_atr=1.1,
        stop_buffer_atr=0.05,
        target_rr=2.0,
        require_confirmed_retest=True,
        require_fvg_touch=False,
        allow_ote_only=True,
        require_htf_bias_align=True,
        require_h4_bias_align=True,
        require_d1_bias_align=False,
        allowed_time_buckets="other",
        allowed_directions="all",
        require_ote_touch=True,
        min_displacement_body_atr=0.0,
        min_displacement_range_atr=0.0,
        bull_min_displacement_body_atr=0.9,
        bull_max_displacement_body_atr=1.3,
        bull_min_displacement_range_atr=0.0,
        bull_max_displacement_range_atr=0.0,
        max_mss_lag_bars=15,
        min_fvg_size_pct=0.0,
        max_fvg_fill_pct=0.0,
        bear_min_sweep_distance_pct=0.03,
        bear_require_fvg_touch=False,
        bear_min_fvg_size_pct=0.0,
        max_open_positions=1,
        position_risk_fraction=1.0,
        initial_capital=args.initial_capital,
    )


def summarize_leveraged_rows(rows: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    if not rows:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }
    capital = initial_capital
    capitals: list[float] = []
    wins = 0
    for row in rows:
        trade_return = float(row["leveraged_return"])
        capital *= 1.0 + trade_return
        capitals.append(capital)
        if trade_return > 0:
            wins += 1
    return {
        "trades": len(rows),
        "wins": wins,
        "losses": len(rows) - wins,
        "win_rate_pct": round(wins / len(rows) * 100.0, 2),
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100.0, 2),
        "max_drawdown_pct": round(max_drawdown_from_capitals(capitals, initial_capital), 2),
    }


def yearly_summary(rows: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(pd.Timestamp(row["entry_time"]).year)].append(row)
    return {year: summarize_leveraged_rows(bucket, initial_capital) for year, bucket in sorted(buckets.items())}


def window_summary(rows: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    if not rows:
        return {}
    end = max(pd.Timestamp(row["exit_time"], tz="UTC") for row in rows)
    windows = {
        "current_year": pd.Timestamp(f"{end.year}-01-01", tz="UTC"),
        "last_60d": end - pd.Timedelta(days=60),
        "last_30d": end - pd.Timedelta(days=30),
    }
    out: dict[str, Any] = {}
    for name, start in windows.items():
        selected = [row for row in rows if pd.Timestamp(row["entry_time"], tz="UTC") >= start]
        out[name] = summarize_leveraged_rows(selected, initial_capital)
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
    raw_trade_count = len(rows)
    rows, slot_skipped = apply_max_open_positions(rows, int(smc_args.max_open_positions))

    capital = float(args.initial_capital)
    accepted_rows: list[dict[str, Any]] = []
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
            "min_liquidation_buffer_pct": round(min((row["liquidation_buffer_pct"] for row in accepted_rows), default=0.0), 4),
            "max_account_effective_leverage": round(max((row["account_effective_leverage"] for row in accepted_rows), default=0.0), 4),
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
