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

from scripts.live_readiness_report import _high_leverage_trade_diagnostics, load_prepared_data  # noqa: E402
from scripts.report_smc_trade_context import daily_candles_from_4h  # noqa: E402
from scripts.reproduce_smc_short_only_v1_10x import strategy_args  # noqa: E402
from scripts.reproduce_smc_standalone_v2_1_10x import summarize_leveraged_rows, window_summary, yearly_summary  # noqa: E402
from scripts.research_smc_standalone_v1 import apply_max_open_positions, build_event_scan_args, clean_for_json  # noqa: E402
from scripts.research_smc_standalone_v1 import scan_events, trade_rows_for_events  # noqa: E402
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_short_only_v1_10x_rr_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan fixed target RR for standalone SMC short-only v1 at 10x.")
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--position-size-pct", type=float, default=1.0)
    parser.add_argument("--maintenance-margin-pct", type=float, default=0.5)
    parser.add_argument("--min-liq-buffer-pct", type=float, default=1.2)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--rr-values", default="1.5,1.75,2.0,2.25,2.5,3.0")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def score_candidate(result: dict[str, Any]) -> float:
    overall = result["overall"]
    yearly_2026 = result["yearly"].get("2026", {})
    windows = result["windows"]
    return round(
        float(overall["total_return_pct"])
        + float(yearly_2026.get("total_return_pct", 0.0)) * 8.0
        + float(windows.get("last_60d", {}).get("total_return_pct", 0.0)) * 2.0
        + float(windows.get("last_30d", {}).get("total_return_pct", 0.0))
        - float(overall["max_drawdown_pct"]) * 8.0,
        4,
    )


def main() -> None:
    args = parse_args()
    rr_values = parse_float_list(args.rr_values)
    base_strategy = strategy_args(argparse.Namespace(**vars(args), target_rr=2.0))
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=None,
    )
    daily = daily_candles_from_4h(prepared.c4h)
    h4_highs, h4_lows = precompute_swings(prepared.c4h, n=2, lookback=80)
    d1_highs, d1_lows = precompute_swings(daily, n=2, lookback=20)

    candidates: list[dict[str, Any]] = []
    for rr in rr_values:
        smc_args = strategy_args(argparse.Namespace(**vars(args), target_rr=rr))
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

        candidate = {
            "params": {
                "target_rr": rr,
                "leverage": args.leverage,
                "position_size_pct": args.position_size_pct,
                "maintenance_margin_pct": args.maintenance_margin_pct,
                "min_liq_buffer_pct": args.min_liq_buffer_pct,
            },
            "strategy_params": clean_for_json(vars(smc_args)),
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
