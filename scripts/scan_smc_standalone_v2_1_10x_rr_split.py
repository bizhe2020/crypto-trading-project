#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_readiness_report import _high_leverage_trade_diagnostics, load_prepared_data  # noqa: E402
from scripts.report_smc_trade_context import daily_candles_from_4h  # noqa: E402
from scripts.reproduce_smc_standalone_v2_1_10x import strategy_args, summarize_leveraged_rows, window_summary, yearly_summary  # noqa: E402
from scripts.research_smc_standalone_v1 import apply_max_open_positions, build_event_scan_args, clean_for_json  # noqa: E402
from scripts.research_smc_standalone_v1 import scan_events, trade_rows_for_events  # noqa: E402
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_standalone_v2_1_10x_rr_split_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan direction-specific target RR for standalone SMC v2.1 at 10x.")
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--position-size-pct", type=float, default=1.0)
    parser.add_argument("--maintenance-margin-pct", type=float, default=0.5)
    parser.add_argument("--min-liq-buffer-pct", type=float, default=1.2)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--bull-rr-values", default="1.25,1.5,1.75,2.0")
    parser.add_argument("--bear-rr-values", default="1.5,2.0,2.5,3.0")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def normalized_ts(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def score_candidate(result: dict[str, Any]) -> float:
    overall = result["overall"]
    yearly = result["yearly"].get("2026", {})
    windows = result["windows"]
    return round(
        float(overall["total_return_pct"])
        + float(yearly.get("total_return_pct", 0.0)) * 6.0
        + float(windows.get("last_60d", {}).get("total_return_pct", 0.0)) * 2.0
        + float(windows.get("last_30d", {}).get("total_return_pct", 0.0))
        - float(overall["max_drawdown_pct"]) * 8.0,
        4,
    )


def main() -> None:
    args = parse_args()
    base_strategy = strategy_args(args)
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=None,
    )
    daily = daily_candles_from_4h(prepared.c4h)
    h4_highs, h4_lows = precompute_swings(prepared.c4h, n=2, lookback=80)
    d1_highs, d1_lows = precompute_swings(daily, n=2, lookback=20)

    rr_values = sorted(set(parse_float_list(args.bull_rr_values) + parse_float_list(args.bear_rr_values)))
    event_cache: dict[float, list[Any]] = {}
    row_cache: dict[float, list[dict[str, Any]]] = {}
    for rr in rr_values:
        rr_args = argparse.Namespace(**vars(base_strategy))
        rr_args.target_rr = rr
        events = scan_events(prepared.c15m, build_event_scan_args(rr_args))
        event_cache[rr] = events
        row_cache[rr] = trade_rows_for_events(events, prepared, daily, h4_highs, h4_lows, d1_highs, d1_lows, rr_args)

    candidates: list[dict[str, Any]] = []
    for bull_rr, bear_rr in itertools.product(parse_float_list(args.bull_rr_values), parse_float_list(args.bear_rr_values)):
        merged_rows = [row for row in row_cache[bull_rr] if row["direction"] == "BULL"]
        merged_rows.extend(row for row in row_cache[bear_rr] if row["direction"] == "BEAR")
        merged_rows.sort(key=lambda row: normalized_ts(row["entry_time"]))
        raw_trade_count = len(merged_rows)
        slot_rows, slot_skipped = apply_max_open_positions(merged_rows, int(base_strategy.max_open_positions))

        capital = float(args.initial_capital)
        accepted_rows: list[dict[str, Any]] = []
        failure_counts: dict[str, int] = {}
        guard_skipped = 0
        for row in slot_rows:
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
                failure_counts["liquidation_buffer_too_small"] = failure_counts.get("liquidation_buffer_too_small", 0) + 1
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

        overall = summarize_leveraged_rows(accepted_rows, float(args.initial_capital))
        yearly = yearly_summary(accepted_rows, float(args.initial_capital))
        windows = window_summary(accepted_rows, float(args.initial_capital))
        candidate = {
            "params": {
                "bull_target_rr": bull_rr,
                "bear_target_rr": bear_rr,
                "leverage": args.leverage,
                "position_size_pct": args.position_size_pct,
                "maintenance_margin_pct": args.maintenance_margin_pct,
                "min_liq_buffer_pct": args.min_liq_buffer_pct,
            },
            "execution_summary": {
                "raw_trades": raw_trade_count,
                "slot_trades": len(slot_rows),
                "slot_skipped_trades": slot_skipped,
                "guard_skipped_trades": guard_skipped,
                "failure_counts": failure_counts,
            },
            "overall": overall,
            "yearly": yearly,
            "windows": windows,
            "risk_diagnostics": {
                "min_liquidation_buffer_pct": round(min((row["liquidation_buffer_pct"] for row in accepted_rows), default=0.0), 4),
                "max_account_effective_leverage": round(max((row["account_effective_leverage"] for row in accepted_rows), default=0.0), 4),
                "max_stop_distance_pct": round(max((float(row["stop_distance_pct"] or 0.0) for row in accepted_rows), default=0.0), 4),
            },
        }
        candidate["score"] = score_candidate(candidate)
        candidates.append(candidate)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    report = {
        "metadata": {
            "start_date": args.start_date,
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
            "candidate_count": len(candidates),
            "base_strategy_params": clean_for_json(vars(base_strategy)),
        },
        "top": candidates[: args.top],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    for idx, item in enumerate(candidates[: args.top], start=1):
        overall = item["overall"]
        y2026 = item["yearly"].get("2026", {})
        w60 = item["windows"].get("last_60d", {})
        w30 = item["windows"].get("last_30d", {})
        print(
            f"{idx:02d} score={item['score']:.2f} full={overall['total_return_pct']:.2f}%/"
            f"{overall['max_drawdown_pct']:.2f}% 2026={y2026.get('total_return_pct', 0.0):.2f}% "
            f"60d={w60.get('total_return_pct', 0.0):.2f}% 30d={w30.get('total_return_pct', 0.0):.2f}% "
            f"trades={overall['trades']} params={item['params']}"
        )


if __name__ == "__main__":
    main()
