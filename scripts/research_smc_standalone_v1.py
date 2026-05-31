#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_pa_ict_liquidity_features import LiquidityEvent, atr_series  # noqa: E402
from scripts.report_pa_ict_liquidity_features import scan_events, summarize as summarize_paict, time_bucket  # noqa: E402
from scripts.report_smc_trade_context import completed_4h_idx_for_entry, completed_d1_idx_for_entry, daily_candles_from_4h  # noqa: E402
from scripts.live_readiness_report import load_prepared_data  # noqa: E402
from strategy.scalp_robust_v2_core import Candle, precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_standalone_v1_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone SMC v1 research backtest.")
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
    parser.add_argument("--target-rr", type=float, default=2.0)

    parser.add_argument("--require-confirmed-retest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-fvg-touch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-ote-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-htf-bias-align", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-h4-bias-align", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-d1-bias-align", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allowed-time-buckets", default="all")
    parser.add_argument("--allowed-directions", default="all")
    parser.add_argument("--require-ote-touch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-displacement-body-atr", type=float, default=0.0)
    parser.add_argument("--min-displacement-range-atr", type=float, default=0.0)
    parser.add_argument("--bull-min-displacement-body-atr", type=float, default=0.0)
    parser.add_argument("--bull-max-displacement-body-atr", type=float, default=0.0)
    parser.add_argument("--bull-min-displacement-range-atr", type=float, default=0.0)
    parser.add_argument("--bull-max-displacement-range-atr", type=float, default=0.0)
    parser.add_argument("--max-mss-lag-bars", type=int, default=0)
    parser.add_argument("--min-fvg-size-pct", type=float, default=0.0)
    parser.add_argument("--max-fvg-fill-pct", type=float, default=0.0)
    parser.add_argument("--bear-min-sweep-distance-pct", type=float, default=0.0)
    parser.add_argument("--bear-require-fvg-touch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bear-min-fvg-size-pct", type=float, default=0.0)
    parser.add_argument("--max-open-positions", type=int, default=0)
    parser.add_argument("--position-risk-fraction", type=float, default=1.0)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    return parser.parse_args()


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_for_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def build_event_scan_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
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
        target_rr=args.target_rr,
        allow_incomplete_tail=False,
    )


def htf_structure_bias(candles: list[Candle], highs: list[int], lows: list[int], idx: int) -> str:
    prev_highs = [swing_idx for swing_idx in highs if swing_idx <= idx]
    prev_lows = [swing_idx for swing_idx in lows if swing_idx <= idx]
    if len(prev_highs) < 2 or len(prev_lows) < 2:
        return "NONE"
    last_high_idx, prev_high_idx = prev_highs[-1], prev_highs[-2]
    last_low_idx, prev_low_idx = prev_lows[-1], prev_lows[-2]
    last_high = float(candles[last_high_idx].h)
    prev_high = float(candles[prev_high_idx].h)
    last_low = float(candles[last_low_idx].l)
    prev_low = float(candles[prev_low_idx].l)
    if last_high > prev_high and last_low > prev_low:
        return "BULL"
    if last_high < prev_high and last_low < prev_low:
        return "BEAR"
    return "NONE"


def allowed_bucket(bucket: str, allowed: str) -> bool:
    if allowed == "all":
        return True
    return bucket in {item.strip() for item in allowed.split("+") if item.strip()}


def allowed_direction(direction: str, allowed: str) -> bool:
    if allowed == "all":
        return True
    return direction in {item.strip() for item in allowed.split("+") if item.strip()}


def event_rr_result(event: LiquidityEvent, target_rr: float) -> float | None:
    if event.retest is None:
        return None
    if event.retest.target_rr_hit:
        return float(target_rr)
    if event.retest.stopped:
        return -1.0
    return None
def trade_rows_for_events(
    events: list[LiquidityEvent],
    prepared: Any,
    daily: list[Candle],
    h4_highs: list[int],
    h4_lows: list[int],
    d1_highs: list[int],
    d1_lows: list[int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    daily_ts = [candle.ts for candle in daily]
    atr_values = atr_series(prepared.c15m, int(args.atr_period))
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.retest is None:
            continue
        if args.require_confirmed_retest and not bool(event.retest.confirmed):
            continue
        if args.require_fvg_touch and not bool(event.retest.fvg_touched):
            continue
        if not args.allow_ote_only and not bool(event.retest.fvg_touched):
            continue

        entry_idx = int(event.retest.idx)
        entry_candle = prepared.c15m[entry_idx]
        h4_idx = completed_4h_idx_for_entry(prepared.mapping, entry_idx)
        d1_idx = completed_d1_idx_for_entry(daily_ts, entry_candle.ts)
        h4_bias = htf_structure_bias(prepared.c4h, h4_highs, h4_lows, h4_idx) if h4_idx >= 0 else "NONE"
        d1_bias = htf_structure_bias(daily, d1_highs, d1_lows, d1_idx) if d1_idx >= 0 else "NONE"
        direction = str(event.direction)
        bucket, ny_time = time_bucket(entry_candle.ts)

        if not allowed_bucket(bucket, args.allowed_time_buckets):
            continue
        if not allowed_direction(direction, str(args.allowed_directions)):
            continue
        if args.require_h4_bias_align and args.require_htf_bias_align and h4_bias != direction:
            continue
        if args.require_h4_bias_align and not args.require_htf_bias_align and h4_bias not in {direction, "NONE"}:
            continue
        if args.require_d1_bias_align and args.require_htf_bias_align and d1_bias != direction:
            continue
        if args.require_d1_bias_align and not args.require_htf_bias_align and d1_bias not in {direction, "NONE"}:
            continue
        if bool(args.require_ote_touch) and not bool(event.retest.ote_touched):
            continue
        if float(event.displacement_body_atr or 0.0) < float(args.min_displacement_body_atr):
            continue
        if float(event.displacement_range_atr or 0.0) < float(args.min_displacement_range_atr):
            continue
        if int(args.max_mss_lag_bars) > 0 and event.mss_idx is not None:
            if int(event.mss_idx) - int(event.sweep_idx) > int(args.max_mss_lag_bars):
                continue
        fvg_size_pct = float(event.fvg.size_pct) if event.fvg is not None else None
        if float(args.min_fvg_size_pct) > 0.0:
            if fvg_size_pct is None or fvg_size_pct < float(args.min_fvg_size_pct):
                continue
        fvg_fill_pct = event.retest.fvg_fill_pct
        if float(args.max_fvg_fill_pct) > 0.0 and fvg_fill_pct is not None:
            if float(fvg_fill_pct) > float(args.max_fvg_fill_pct):
                continue
        if direction == "BULL":
            body_atr = float(event.displacement_body_atr or 0.0)
            range_atr = float(event.displacement_range_atr or 0.0)
            if float(args.bull_min_displacement_body_atr) > 0.0 and body_atr < float(args.bull_min_displacement_body_atr):
                continue
            if float(args.bull_max_displacement_body_atr) > 0.0 and body_atr > float(args.bull_max_displacement_body_atr):
                continue
            if float(args.bull_min_displacement_range_atr) > 0.0 and range_atr < float(args.bull_min_displacement_range_atr):
                continue
            if float(args.bull_max_displacement_range_atr) > 0.0 and range_atr > float(args.bull_max_displacement_range_atr):
                continue
        if direction == "BEAR":
            if float(args.bear_min_sweep_distance_pct) > 0.0 and float(event.sweep_distance_pct or 0.0) < float(args.bear_min_sweep_distance_pct):
                continue
            if bool(args.bear_require_fvg_touch) and not bool(event.retest.fvg_touched):
                continue
            if float(args.bear_min_fvg_size_pct) > 0.0:
                if fvg_size_pct is None or fvg_size_pct < float(args.bear_min_fvg_size_pct):
                    continue

        rr_result = event_rr_result(event, args.target_rr)
        if rr_result is None:
            continue
        stop_buffer = atr_values[entry_idx] * float(args.stop_buffer_atr) if entry_idx < len(atr_values) else 0.0
        if direction == "BULL":
            stop_price = float(event.sweep_extreme) - stop_buffer
            risk_points = float(event.retest.close) - stop_price
            target_price = float(event.retest.close) + risk_points * float(args.target_rr)
        else:
            stop_price = float(event.sweep_extreme) + stop_buffer
            risk_points = stop_price - float(event.retest.close)
            target_price = float(event.retest.close) - risk_points * float(args.target_rr)
        stop_distance_pct = (risk_points / float(event.retest.close) * 100.0) if float(event.retest.close) > 0 and risk_points > 0 else None
        signal_return_pct = rr_result * stop_distance_pct if stop_distance_pct is not None else None
        capital_return = rr_result * float(args.position_risk_fraction) / 100.0
        rows.append(
            {
                "entry_time": event.retest.timestamp,
                "exit_time": event.retest.outcome_time or event.retest.timestamp,
                "sweep_time": event.sweep_time,
                "direction": direction,
                "entry_idx": entry_idx,
                "exit_idx": event.retest.outcome_idx if event.retest.outcome_idx is not None else entry_idx,
                "entry_price": float(event.retest.close),
                "stop_price": stop_price,
                "target_price": target_price,
                "stop_distance_pct": stop_distance_pct,
                "signal_return_pct": signal_return_pct,
                "sweep_idx": int(event.sweep_idx),
                "swept_level": float(event.swept_level),
                "sweep_extreme": float(event.sweep_extreme),
                "sweep_distance_pct": float(event.sweep_distance_pct),
                "mss_idx": event.mss_idx,
                "mss_time": event.mss_time,
                "mss_lag_bars": (int(event.mss_idx) - int(event.sweep_idx)) if event.mss_idx is not None else None,
                "displacement_body_atr": event.displacement_body_atr,
                "displacement_range_atr": event.displacement_range_atr,
                "time_bucket": bucket,
                "ny_time": ny_time,
                "h4_bias": h4_bias,
                "d1_bias": d1_bias,
                "fvg_touched": bool(event.retest.fvg_touched),
                "fvg_fill_pct": event.retest.fvg_fill_pct,
                "fvg_size_pct": event.fvg.size_pct if event.fvg is not None else None,
                "ote_touched": bool(event.retest.ote_touched),
                "mfe_r": event.retest.mfe_r,
                "mae_r": event.retest.mae_r,
                "target_rr_hit": bool(event.retest.target_rr_hit),
                "stopped": bool(event.retest.stopped),
                "outcome": event.retest.outcome,
                "rr_result": rr_result,
                "return": capital_return,
                "status": event.status,
            }
        )
    rows.sort(key=lambda item: pd.Timestamp(item["entry_time"], tz="UTC"))
    return rows


def apply_max_open_positions(rows: list[dict[str, Any]], max_open_positions: int) -> tuple[list[dict[str, Any]], int]:
    if max_open_positions <= 0:
        return list(rows), 0
    accepted: list[dict[str, Any]] = []
    active_exits: list[pd.Timestamp] = []
    skipped = 0
    for row in rows:
        entry_time = pd.Timestamp(row["entry_time"], tz="UTC")
        while active_exits and active_exits[0] <= entry_time:
            heapq.heappop(active_exits)
        if len(active_exits) >= max_open_positions:
            skipped += 1
            continue
        accepted.append(row)
        heapq.heappush(active_exits, pd.Timestamp(row["exit_time"], tz="UTC"))
    return accepted, skipped


def summarize_rows(rows: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    if not rows:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "avg_rr": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }
    capital = initial_capital
    peak = initial_capital
    max_dd = 0.0
    wins = 0
    rr_values: list[float] = []
    for row in rows:
        trade_return = float(row["return"])
        rr_values.append(float(row["rr_result"]))
        capital *= 1.0 + trade_return
        peak = max(peak, capital)
        if peak > 0:
            max_dd = max(max_dd, (peak - capital) / peak * 100.0)
        if trade_return > 0:
            wins += 1
    return {
        "trades": len(rows),
        "wins": wins,
        "losses": len(rows) - wins,
        "win_rate_pct": round(wins / len(rows) * 100.0, 2),
        "avg_rr": round(sum(rr_values) / len(rr_values), 3),
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100.0, 2),
        "max_drawdown_pct": round(max_dd, 2),
    }


def group_summary(rows: list[dict[str, Any]], key: str, initial_capital: float) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key, "unknown"))].append(row)
    return {name: summarize_rows(bucket, initial_capital) for name, bucket in sorted(buckets.items())}


def yearly_summary(rows: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        year = str(pd.Timestamp(row["entry_time"]).year)
        buckets[year].append(row)
    return {year: summarize_rows(bucket, initial_capital) for year, bucket in sorted(buckets.items())}


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

    events = scan_events(prepared.c15m, build_event_scan_args(args))
    raw_rows = trade_rows_for_events(events, prepared, daily, h4_highs, h4_lows, d1_highs, d1_lows, args)
    rows, slot_skipped = apply_max_open_positions(raw_rows, int(args.max_open_positions))
    report = {
        "metadata": {
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
            "daily_candles": len(daily),
        },
        "parameters": {
            "start_date": args.start_date,
            "target_rr": args.target_rr,
            "position_risk_fraction": args.position_risk_fraction,
            "require_confirmed_retest": args.require_confirmed_retest,
            "require_fvg_touch": args.require_fvg_touch,
            "allow_ote_only": args.allow_ote_only,
            "require_4h_bias_align": args.require_h4_bias_align,
            "require_1d_bias_align": args.require_d1_bias_align,
            "allowed_time_buckets": args.allowed_time_buckets,
            "allowed_directions": args.allowed_directions,
            "require_ote_touch": args.require_ote_touch,
            "min_displacement_body_atr": args.min_displacement_body_atr,
            "min_displacement_range_atr": args.min_displacement_range_atr,
            "bull_min_displacement_body_atr": args.bull_min_displacement_body_atr,
            "bull_max_displacement_body_atr": args.bull_max_displacement_body_atr,
            "bull_min_displacement_range_atr": args.bull_min_displacement_range_atr,
            "bull_max_displacement_range_atr": args.bull_max_displacement_range_atr,
            "max_mss_lag_bars": args.max_mss_lag_bars,
            "min_fvg_size_pct": args.min_fvg_size_pct,
            "max_fvg_fill_pct": args.max_fvg_fill_pct,
            "bear_min_sweep_distance_pct": args.bear_min_sweep_distance_pct,
            "bear_require_fvg_touch": args.bear_require_fvg_touch,
            "bear_min_fvg_size_pct": args.bear_min_fvg_size_pct,
            "max_open_positions": args.max_open_positions,
        },
        "execution_summary": {
            "raw_trades": len(raw_rows),
            "accepted_trades": len(rows),
            "slot_skipped_trades": slot_skipped,
        },
        "event_summary": summarize_paict(events),
        "trade_summary": {
            "overall": summarize_rows(rows, args.initial_capital),
            "by_direction": group_summary(rows, "direction", args.initial_capital),
            "by_time_bucket": group_summary(rows, "time_bucket", args.initial_capital),
            "by_h4_bias": group_summary(rows, "h4_bias", args.initial_capital),
            "by_d1_bias": group_summary(rows, "d1_bias", args.initial_capital),
            "by_status": group_summary(rows, "status", args.initial_capital),
            "yearly": yearly_summary(rows, args.initial_capital),
        },
        "sample_trades": rows[:50],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    compact = {
        "overall": report["trade_summary"]["overall"],
        "by_direction": report["trade_summary"]["by_direction"],
        "yearly": report["trade_summary"]["yearly"],
    }
    print(json.dumps(clean_for_json(compact), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
