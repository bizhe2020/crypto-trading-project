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

from scripts.live_readiness_report import load_prepared_data  # noqa: E402
from scripts.report_smc_trade_context import daily_candles_from_4h  # noqa: E402
from scripts.research_smc_standalone_v1 import apply_max_open_positions, build_event_scan_args, clean_for_json  # noqa: E402
from scripts.research_smc_standalone_v1 import group_summary, scan_events, summarize_rows, trade_rows_for_events, yearly_summary  # noqa: E402
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_standalone_v2_formal_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal scan for standalone SMC v2 quality filters.")
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))

    parser.add_argument("--swing-n", type=int, default=3)
    parser.add_argument("--swing-lookback", type=int, default=80)
    parser.add_argument("--liquidity-lookback-bars", type=int, default=192)
    parser.add_argument("--mss-lookahead-bars", type=int, default=24)
    parser.add_argument("--fvg-lookback-bars", type=int, default=8)
    parser.add_argument("--entry-lookahead-bars", type=int, default=40)
    parser.add_argument("--outcome-lookahead-bars", type=int, default=96)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--min-body-atr", type=float, default=0.7)
    parser.add_argument("--min-range-atr", type=float, default=1.1)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.05)

    parser.add_argument("--target-rr-values", default="1.5,2.0")
    parser.add_argument("--time-bucket-sets", default="other,other+ny_am_killzone,other+ny_am_killzone+asia_evening_ny")
    parser.add_argument("--entry-modes", default="ote_only,fvg_only,fvg_or_ote")
    parser.add_argument("--direction-sets", default="all,BULL,BEAR")
    parser.add_argument("--require-ote-touch-values", default="false,true")
    parser.add_argument("--min-displacement-body-atr-values", default="0.0,0.95")
    parser.add_argument("--min-displacement-range-atr-values", default="0.0,1.25,1.5")
    parser.add_argument("--max-mss-lag-bars-values", default="0,15")
    parser.add_argument("--min-fvg-size-pct-values", default="0.0,0.05,0.1")
    parser.add_argument("--max-fvg-fill-pct-values", default="0.0,0.35,0.7")
    parser.add_argument("--max-open-positions", type=int, default=1)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_bool_list(raw: str) -> list[bool]:
    values: list[bool] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"1", "true", "yes", "on"}:
            values.append(True)
        elif token in {"0", "false", "no", "off"}:
            values.append(False)
        else:
            raise ValueError(f"Unsupported boolean token: {item}")
    return values


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def entry_mode_flags(mode: str) -> tuple[bool, bool]:
    if mode == "fvg_only":
        return True, False
    if mode == "ote_only":
        return False, True
    if mode == "fvg_or_ote":
        return False, False
    raise ValueError(f"Unsupported entry mode: {mode}")


def score_candidate(overall: dict[str, Any], yearly: dict[str, Any]) -> float:
    year_2026 = yearly.get("2026", {})
    return round(
        float(overall["total_return_pct"])
        + float(year_2026.get("total_return_pct", 0.0)) * 4.0
        - float(overall["max_drawdown_pct"]) * 2.5
        + float(overall["win_rate_pct"]) * 0.5,
        4,
    )


def main() -> None:
    args = parse_args()
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=None,
    )
    daily = daily_candles_from_4h(prepared.c4h)
    h4_highs, h4_lows = precompute_swings(prepared.c4h, n=2, lookback=80)
    d1_highs, d1_lows = precompute_swings(daily, n=2, lookback=20)

    base_scan = argparse.Namespace(
        swing_n=args.swing_n,
        swing_lookback=args.swing_lookback,
        liquidity_lookback_bars=args.liquidity_lookback_bars,
        mss_lookahead_bars=args.mss_lookahead_bars,
        fvg_lookback_bars=args.fvg_lookback_bars,
        entry_lookahead_bars=args.entry_lookahead_bars,
        outcome_lookahead_bars=args.outcome_lookahead_bars,
        atr_period=args.atr_period,
        min_body_atr=args.min_body_atr,
        min_range_atr=args.min_range_atr,
        stop_buffer_atr=args.stop_buffer_atr,
        target_rr=2.0,
    )
    event_cache: dict[float, list[Any]] = {}
    for target_rr in parse_float_list(args.target_rr_values):
        scan_args = argparse.Namespace(**base_scan.__dict__)
        scan_args.target_rr = target_rr
        event_cache[target_rr] = scan_events(prepared.c15m, build_event_scan_args(scan_args))

    candidates: list[dict[str, Any]] = []
    for (
        target_rr,
        time_buckets,
        entry_mode,
        directions,
        require_ote_touch,
        min_body_atr,
        min_range_atr,
        max_mss_lag_bars,
        min_fvg_size_pct,
        max_fvg_fill_pct,
    ) in itertools.product(
        parse_float_list(args.target_rr_values),
        parse_str_list(args.time_bucket_sets),
        parse_str_list(args.entry_modes),
        parse_str_list(args.direction_sets),
        parse_bool_list(args.require_ote_touch_values),
        parse_float_list(args.min_displacement_body_atr_values),
        parse_float_list(args.min_displacement_range_atr_values),
        parse_int_list(args.max_mss_lag_bars_values),
        parse_float_list(args.min_fvg_size_pct_values),
        parse_float_list(args.max_fvg_fill_pct_values),
    ):
        require_fvg_touch, allow_ote_only = entry_mode_flags(entry_mode)
        runtime_args = argparse.Namespace(
            target_rr=target_rr,
            require_confirmed_retest=True,
            require_fvg_touch=require_fvg_touch,
            allow_ote_only=allow_ote_only,
            require_htf_bias_align=True,
            require_h4_bias_align=True,
            require_d1_bias_align=False,
            allowed_time_buckets=time_buckets,
            allowed_directions=directions,
            require_ote_touch=require_ote_touch,
            min_displacement_body_atr=min_body_atr,
            min_displacement_range_atr=min_range_atr,
            max_mss_lag_bars=max_mss_lag_bars,
            min_fvg_size_pct=min_fvg_size_pct,
            max_fvg_fill_pct=max_fvg_fill_pct,
            max_open_positions=args.max_open_positions,
            position_risk_fraction=1.0,
            initial_capital=args.initial_capital,
        )
        raw_rows = trade_rows_for_events(
            event_cache[target_rr],
            prepared,
            daily,
            h4_highs,
            h4_lows,
            d1_highs,
            d1_lows,
            runtime_args,
        )
        rows, slot_skipped = apply_max_open_positions(raw_rows, args.max_open_positions)
        overall = summarize_rows(rows, args.initial_capital)
        yearly = yearly_summary(rows, args.initial_capital)
        candidate = {
            "params": {
                "target_rr": target_rr,
                "time_buckets": time_buckets,
                "entry_mode": entry_mode,
                "allowed_directions": directions,
                "require_ote_touch": require_ote_touch,
                "min_displacement_body_atr": min_body_atr,
                "min_displacement_range_atr": min_range_atr,
                "max_mss_lag_bars": max_mss_lag_bars,
                "min_fvg_size_pct": min_fvg_size_pct,
                "max_fvg_fill_pct": max_fvg_fill_pct,
                "max_open_positions": args.max_open_positions,
            },
            "execution": {
                "raw_trades": len(raw_rows),
                "accepted_trades": len(rows),
                "slot_skipped_trades": slot_skipped,
            },
            "overall": overall,
            "yearly": yearly,
            "by_direction": group_summary(rows, "direction", args.initial_capital),
            "score": score_candidate(overall, yearly),
        }
        candidates.append(candidate)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    report = {
        "metadata": {
            "start_date": args.start_date,
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
            "daily_candles": len(daily),
            "candidate_count": len(candidates),
            "max_open_positions": args.max_open_positions,
        },
        "top": candidates[: args.top],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    for idx, item in enumerate(candidates[: args.top], start=1):
        overall = item["overall"]
        year_2026 = item["yearly"].get("2026", {})
        print(
            f"{idx:02d} score={item['score']:.2f} full={overall['total_return_pct']:.2f}%/"
            f"{overall['max_drawdown_pct']:.2f}% 2026={year_2026.get('total_return_pct', 0.0):.2f}% "
            f"trades={overall['trades']} win={overall['win_rate_pct']:.2f}% "
            f"slot={item['execution']['accepted_trades']}/{item['execution']['raw_trades']} params={item['params']}"
        )


if __name__ == "__main__":
    main()
