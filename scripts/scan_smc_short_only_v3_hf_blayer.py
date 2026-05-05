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
from scripts.reproduce_smc_short_only_v1_10x import strategy_args  # noqa: E402
from scripts.reproduce_smc_standalone_v2_1_10x import summarize_leveraged_rows, window_summary, yearly_summary  # noqa: E402
from scripts.research_smc_standalone_v1 import apply_max_open_positions, build_event_scan_args, clean_for_json  # noqa: E402
from scripts.research_smc_standalone_v1 import scan_events, trade_rows_for_events  # noqa: E402
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_short_only_v3_hf_blayer_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan high-frequency B-layer variants for standalone short-only SMC.")
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--target-rr", type=float, default=2.0)
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--position-size-pct", type=float, default=1.0)
    parser.add_argument("--maintenance-margin-pct", type=float, default=0.5)
    parser.add_argument("--min-liq-buffer-pct", type=float, default=1.2)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--top", type=int, default=24)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def annualized_2026_trades(y2026_trades: int) -> float:
    return round(float(y2026_trades) * (366.0 / 117.0), 2)


def score_candidate(result: dict[str, Any]) -> float:
    overall = result["overall"]
    yearly_2026 = result["yearly"].get("2026", {})
    windows = result["windows"]
    ann_2026 = annualized_2026_trades(int(yearly_2026.get("trades", 0)))
    return round(
        ann_2026 * 15.0
        + float(yearly_2026.get("total_return_pct", 0.0)) * 2.0
        + float(windows.get("last_60d", {}).get("total_return_pct", 0.0))
        + float(overall["total_return_pct"]) * 0.2
        - float(overall["max_drawdown_pct"]) * 4.0,
        4,
    )


def main() -> None:
    args = parse_args()
    base_cfg = {
        "allowed_time_buckets": "all",
        "swing_n": 2,
        "min_body_atr": 0.5,
        "min_range_atr": 0.9,
        "entry_lookahead_bars": 56,
        "max_open_positions": 1,
        "max_mss_lag_bars": 24,
        "min_displacement_body_atr": 0.0,
        "min_displacement_range_atr": 0.0,
        "ny_max_mss_lag_bars": 0,
        "other_min_mss_lag_bars": 0,
    }

    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=None,
    )
    daily = daily_candles_from_4h(prepared.c4h)
    h4_highs, h4_lows = precompute_swings(prepared.c4h, n=2, lookback=80)
    d1_highs, d1_lows = precompute_swings(daily, n=2, lookback=20)

    grid = itertools.product(
        ["all", "other+asia_evening_ny+ny_am_killzone"],
        [0.5, 0.4],
        [0.9, 0.8],
        [56, 72],
        [1, 2],
        [24, 32],
        [0.0, 0.3],
    )

    candidates: list[dict[str, Any]] = []
    for allowed_time_buckets, min_body_atr, min_range_atr, entry_lookahead_bars, max_open_positions, max_mss_lag_bars, min_displacement_body_atr in grid:
        cfg = {
            **base_cfg,
            "allowed_time_buckets": allowed_time_buckets,
            "min_body_atr": min_body_atr,
            "min_range_atr": min_range_atr,
            "entry_lookahead_bars": entry_lookahead_bars,
            "max_open_positions": max_open_positions,
            "max_mss_lag_bars": max_mss_lag_bars,
            "min_displacement_body_atr": min_displacement_body_atr,
        }
        smc_args = strategy_args(argparse.Namespace(**{
            "data_15m": args.data_15m,
            "data_4h": args.data_4h,
            "start_date": args.start_date,
            "target_rr": args.target_rr,
            "allowed_time_buckets": cfg["allowed_time_buckets"],
            "swing_n": cfg["swing_n"],
            "min_body_atr": cfg["min_body_atr"],
            "min_range_atr": cfg["min_range_atr"],
            "entry_lookahead_bars": cfg["entry_lookahead_bars"],
            "max_open_positions": cfg["max_open_positions"],
            "min_displacement_body_atr": cfg["min_displacement_body_atr"],
            "min_displacement_range_atr": cfg["min_displacement_range_atr"],
            "max_mss_lag_bars": cfg["max_mss_lag_bars"],
            "ny_max_mss_lag_bars": cfg["ny_max_mss_lag_bars"],
            "other_min_mss_lag_bars": cfg["other_min_mss_lag_bars"],
            "leverage": args.leverage,
            "position_size_pct": args.position_size_pct,
            "maintenance_margin_pct": args.maintenance_margin_pct,
            "min_liq_buffer_pct": args.min_liq_buffer_pct,
            "initial_capital": args.initial_capital,
        }))
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
                    "shadow_capital": capital,
                    "liquidation_buffer_pct": diagnostics["liquidation_buffer_pct"],
                    "account_effective_leverage": diagnostics["account_effective_leverage"],
                }
            )

        candidate = {
            "params": clean_for_json(cfg),
            "execution_summary": {
                "slot_trades": len(rows),
                "slot_skipped_trades": slot_skipped,
                "guard_skipped_trades": guard_skipped,
            },
            "overall": summarize_leveraged_rows(accepted_rows, float(args.initial_capital)),
            "yearly": yearly_summary(accepted_rows, float(args.initial_capital)),
            "windows": window_summary(accepted_rows, float(args.initial_capital)),
        }
        candidate["annualized_2026_trades"] = annualized_2026_trades(int(candidate["yearly"].get("2026", {}).get("trades", 0)))
        candidate["score"] = score_candidate(candidate)
        candidates.append(candidate)

    candidates.sort(key=lambda item: (item["annualized_2026_trades"], item["score"]), reverse=True)
    report = {
        "metadata": {
            "start_date": args.start_date,
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
            "candidate_count": len(candidates),
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
            f"{idx:02d} ann2026={item['annualized_2026_trades']:.2f} "
            f"full={overall['total_return_pct']:.2f}%/{overall['max_drawdown_pct']:.2f}% "
            f"2026={y2026.get('total_return_pct', 0.0):.2f}% "
            f"60d={w60.get('total_return_pct', 0.0):.2f}% "
            f"30d={w30.get('total_return_pct', 0.0):.2f}% "
            f"trades={overall['trades']} score={item['score']:.2f} "
            f"params={item['params']}"
        )


if __name__ == "__main__":
    main()
