#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import heapq
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from scripts.live_readiness_report import _high_leverage_trade_diagnostics
from strategy.scalp_robust_v2_core import Candle, precompute_swings
from strategy.sota_overlay_state import leveraged_net_return


SMC_CASES: dict[str, dict[str, Any]] = {
    "v1_base_other_10x": {
        "target_rr": 2.0,
        "allowed_time_buckets": "other",
        "swing_n": 3,
        "min_body_atr": 0.7,
        "min_range_atr": 1.1,
        "entry_lookahead_bars": 40,
        "max_open_positions": 1,
        "max_mss_lag_bars": 15,
        "leverage": 10.0,
        "position_size_pct": 1.0,
    },
    "v1_aggressive_maxlag9_10x": {
        "target_rr": 2.0,
        "allowed_time_buckets": "other+asia_evening_ny+ny_am_killzone",
        "swing_n": 3,
        "min_body_atr": 0.7,
        "min_range_atr": 1.1,
        "entry_lookahead_bars": 40,
        "max_open_positions": 1,
        "max_mss_lag_bars": 9,
        "leverage": 10.0,
        "position_size_pct": 1.0,
    },
    "v2_medium_dispbody05_otherlag4_10x": {
        "target_rr": 2.0,
        "allowed_time_buckets": "other+asia_evening_ny+ny_am_killzone",
        "swing_n": 2,
        "min_body_atr": 0.7,
        "min_range_atr": 1.1,
        "entry_lookahead_bars": 40,
        "max_open_positions": 1,
        "max_mss_lag_bars": 15,
        "min_displacement_body_atr": 0.5,
        "other_min_mss_lag_bars": 4,
        "leverage": 10.0,
        "position_size_pct": 1.0,
    },
    "v3_lag4_9_10x": {
        "target_rr": 2.0,
        "allowed_time_buckets": "other+asia_evening_ny+ny_am_killzone",
        "swing_n": 2,
        "min_body_atr": 0.5,
        "min_range_atr": 0.9,
        "entry_lookahead_bars": 72,
        "max_open_positions": 1,
        "max_mss_lag_bars": 24,
        "min_displacement_body_atr": 0.3,
        "global_min_mss_lag_bars": 4,
        "global_max_mss_lag_bars": 9,
        "leverage": 10.0,
        "position_size_pct": 1.0,
    },
}


FORMAL_SMC_CASE_NAMES: tuple[str, ...] = (
    "v1_base_other_10x",
    "v1_aggressive_maxlag9_10x",
    "v2_medium_dispbody05_otherlag4_10x",
    "v3_lag4_9_10x",
)


@dataclass
class FvgZone:
    direction: str
    idx: int
    timestamp: str
    bottom: float
    top: float
    size_pct: float


@dataclass
class OteZone:
    bottom: float
    top: float
    leg_low: float
    leg_high: float


@dataclass
class Retest:
    idx: int
    timestamp: str
    close: float
    fvg_touched: bool
    fvg_fill_pct: float | None
    ote_touched: bool
    confirmed: bool
    mfe_r: float | None
    mae_r: float | None
    target_rr_hit: bool | None
    stopped: bool | None
    outcome: str | None
    outcome_idx: int | None
    outcome_time: str | None


@dataclass
class LiquidityEvent:
    direction: str
    sweep_idx: int
    sweep_time: str
    swept_level: float
    swept_level_idx: int
    swept_level_time: str
    sweep_extreme: float
    sweep_distance_pct: float
    time_bucket: str
    ny_time: str
    mss_idx: int | None
    mss_time: str | None
    mss_level: float | None
    mss_level_idx: int | None
    displacement_body_atr: float | None
    displacement_range_atr: float | None
    fvg: FvgZone | None
    ote: OteZone | None
    retest: Retest | None
    status: str


def timestamp_for(candles: list[Candle], idx: int) -> str:
    return datetime.fromtimestamp(candles[idx].ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def daily_candles_from_4h(c4h: list[Candle]) -> list[Candle]:
    buckets: dict[pd.Timestamp, list[Candle]] = defaultdict(list)
    for candle in c4h:
        day = pd.Timestamp(candle.ts, unit="s", tz="UTC").normalize()
        buckets[day].append(candle)
    daily: list[Candle] = []
    for day in sorted(buckets):
        candles = buckets[day]
        daily.append(
            Candle(
                ts=day.timestamp(),
                o=float(candles[0].o),
                h=max(float(candle.h) for candle in candles),
                l=min(float(candle.l) for candle in candles),
                c=float(candles[-1].c),
                v=sum(float(candle.v) for candle in candles),
            )
        )
    return daily


def completed_4h_idx_for_entry(mapping: list[int], entry_idx: int) -> int:
    if entry_idx < 0 or entry_idx >= len(mapping):
        return -1
    return max(0, int(mapping[entry_idx]) - 1)


def completed_d1_idx_for_entry(daily_ts: list[float], entry_ts: float) -> int:
    entry_day = pd.Timestamp(entry_ts, unit="s", tz="UTC").normalize().timestamp()
    return bisect.bisect_left(daily_ts, entry_day) - 1


def atr_series(candles: list[Candle], period: int) -> list[float]:
    effective_period = max(int(period), 1)
    out: list[float] = []
    true_ranges: list[float] = []
    atr = 0.0
    for idx, candle in enumerate(candles):
        prev_close = candles[idx - 1].c if idx > 0 else candle.c
        tr = max(candle.h - candle.l, abs(candle.h - prev_close), abs(candle.l - prev_close))
        true_ranges.append(tr)
        if idx == 0:
            atr = tr
        elif idx < effective_period:
            atr = sum(true_ranges) / len(true_ranges)
        elif idx == effective_period:
            atr = sum(true_ranges[-effective_period:]) / effective_period
        else:
            atr = ((atr * (effective_period - 1)) + tr) / effective_period
        out.append(atr)
    return out


def previous_idx(indices: list[int], idx: int, min_idx: int) -> int | None:
    pos = bisect.bisect_left(indices, idx) - 1
    while pos >= 0:
        candidate = indices[pos]
        if candidate >= min_idx:
            return candidate
        break
    return None


def time_bucket(ts: float) -> tuple[str, str]:
    dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    dt_ny = dt_utc.astimezone(ZoneInfo("America/New_York"))
    minutes = dt_ny.hour * 60 + dt_ny.minute
    if 2 * 60 <= minutes < 5 * 60:
        bucket = "london_open"
    elif 8 * 60 + 30 <= minutes < 11 * 60:
        bucket = "ny_am_killzone"
    elif 12 * 60 <= minutes < 13 * 60:
        bucket = "ny_lunch"
    elif 13 * 60 + 30 <= minutes < 16 * 60:
        bucket = "ny_pm_killzone"
    elif minutes >= 20 * 60 or minutes < 30:
        bucket = "asia_evening_ny"
    else:
        bucket = "other"
    return bucket, dt_ny.strftime("%Y-%m-%d %H:%M")


def detect_fvg_at(candles: list[Candle], idx: int, direction: str) -> FvgZone | None:
    if idx < 2:
        return None
    first = candles[idx - 2]
    third = candles[idx]
    if direction == "BULL" and first.h < third.l:
        bottom = first.h
        top = third.l
    elif direction == "BEAR" and first.l > third.h:
        bottom = third.h
        top = first.l
    else:
        return None
    ref = candles[idx].c if candles[idx].c > 0 else 1.0
    return FvgZone(
        direction=direction,
        idx=idx,
        timestamp=timestamp_for(candles, idx),
        bottom=float(bottom),
        top=float(top),
        size_pct=(float(top) - float(bottom)) / ref * 100.0,
    )


def recent_fvg(candles: list[Candle], direction: str, start_idx: int, end_idx: int) -> FvgZone | None:
    for idx in range(end_idx, max(start_idx, 2) - 1, -1):
        fvg = detect_fvg_at(candles, idx, direction)
        if fvg is not None:
            return fvg
    return None


def build_ote(direction: str, sweep_extreme: float, candles: list[Candle], start_idx: int, end_idx: int) -> OteZone:
    if direction == "BULL":
        leg_low = float(sweep_extreme)
        leg_high = max(float(candle.h) for candle in candles[start_idx : end_idx + 1])
        rng = max(leg_high - leg_low, 0.0)
        bottom = leg_high - rng * 0.79
        top = leg_high - rng * 0.62
    else:
        leg_high = float(sweep_extreme)
        leg_low = min(float(candle.l) for candle in candles[start_idx : end_idx + 1])
        rng = max(leg_high - leg_low, 0.0)
        bottom = leg_low + rng * 0.62
        top = leg_low + rng * 0.79
    return OteZone(bottom=float(min(bottom, top)), top=float(max(bottom, top)), leg_low=leg_low, leg_high=leg_high)


def zone_touched(candle: Candle, bottom: float, top: float) -> bool:
    return candle.l <= top and candle.h >= bottom


def fvg_fill_pct(candle: Candle, fvg: FvgZone) -> float:
    height = max(fvg.top - fvg.bottom, 1e-9)
    if fvg.direction == "BULL":
        return max(0.0, min(1.0, (fvg.top - candle.l) / height))
    return max(0.0, min(1.0, (candle.h - fvg.bottom) / height))


def simulate_outcome(
    candles: list[Candle],
    atr: list[float],
    direction: str,
    entry_idx: int,
    entry_price: float,
    sweep_extreme: float,
    *,
    stop_buffer_atr: float,
    target_rr: float,
    outcome_lookahead_bars: int,
) -> tuple[float | None, float | None, bool | None, bool | None, str | None, int | None]:
    stop_buffer = atr[entry_idx] * stop_buffer_atr if entry_idx < len(atr) else 0.0
    if direction == "BULL":
        stop = sweep_extreme - stop_buffer
        risk = entry_price - stop
        target = entry_price + risk * target_rr
    else:
        stop = sweep_extreme + stop_buffer
        risk = stop - entry_price
        target = entry_price - risk * target_rr
    if risk <= 0:
        return None, None, None, None, None, None

    end_idx = min(len(candles) - 1, entry_idx + outcome_lookahead_bars)
    max_favorable = 0.0
    max_adverse = 0.0
    target_hit = False
    stopped = False
    outcome = "open"
    outcome_idx: int | None = end_idx
    for idx in range(entry_idx + 1, end_idx + 1):
        candle = candles[idx]
        if direction == "BULL":
            max_favorable = max(max_favorable, candle.h - entry_price)
            max_adverse = max(max_adverse, entry_price - candle.l)
            stop_hit = candle.l <= stop
            target_reached = candle.h >= target
        else:
            max_favorable = max(max_favorable, entry_price - candle.l)
            max_adverse = max(max_adverse, candle.h - entry_price)
            stop_hit = candle.h >= stop
            target_reached = candle.l <= target
        if stop_hit and target_reached:
            stopped = True
            outcome = "stop_and_target_same_bar_assume_stop"
            outcome_idx = idx
            break
        if stop_hit:
            stopped = True
            outcome = "stop"
            outcome_idx = idx
            break
        if target_reached:
            target_hit = True
            outcome = f"target_{target_rr:.1f}r"
            outcome_idx = idx
            break

    return max_favorable / risk, max_adverse / risk, target_hit, stopped, outcome, outcome_idx


def find_retest(
    candles: list[Candle],
    atr: list[float],
    direction: str,
    sweep_extreme: float,
    fvg: FvgZone | None,
    ote: OteZone | None,
    start_idx: int,
    *,
    entry_lookahead_bars: int,
    outcome_lookahead_bars: int,
    stop_buffer_atr: float,
    target_rr: float,
) -> Retest | None:
    if fvg is None and ote is None:
        return None
    end_idx = min(len(candles) - 1, start_idx + entry_lookahead_bars)
    for idx in range(start_idx + 1, end_idx + 1):
        candle = candles[idx]
        invalidated = candle.l <= sweep_extreme if direction == "BULL" else candle.h >= sweep_extreme
        if invalidated:
            return None
        fvg_touched = fvg is not None and zone_touched(candle, fvg.bottom, fvg.top)
        ote_touched = ote is not None and zone_touched(candle, ote.bottom, ote.top)
        if not fvg_touched and not ote_touched:
            continue
        confirmed = candle.c > candle.o if direction == "BULL" else candle.c < candle.o
        mfe_r, mae_r, target_hit, stopped, outcome, outcome_idx = simulate_outcome(
            candles,
            atr,
            direction,
            idx,
            float(candle.c),
            sweep_extreme,
            stop_buffer_atr=stop_buffer_atr,
            target_rr=target_rr,
            outcome_lookahead_bars=outcome_lookahead_bars,
        )
        return Retest(
            idx=idx,
            timestamp=timestamp_for(candles, idx),
            close=float(candle.c),
            fvg_touched=bool(fvg_touched),
            fvg_fill_pct=fvg_fill_pct(candle, fvg) if fvg_touched and fvg is not None else None,
            ote_touched=bool(ote_touched),
            confirmed=bool(confirmed),
            mfe_r=mfe_r,
            mae_r=mae_r,
            target_rr_hit=target_hit,
            stopped=stopped,
            outcome=outcome,
            outcome_idx=outcome_idx,
            outcome_time=timestamp_for(candles, outcome_idx) if outcome_idx is not None else None,
        )
    return None


def scan_events(candles: list[Candle], args: argparse.Namespace) -> list[LiquidityEvent]:
    highs, lows = precompute_swings(candles, n=args.swing_n, lookback=args.swing_lookback)
    atr = atr_series(candles, args.atr_period)
    events: list[LiquidityEvent] = []
    min_idx = max(args.swing_lookback, args.liquidity_lookback_bars)
    tail_guard = 0 if bool(getattr(args, "allow_incomplete_tail", False)) else args.mss_lookahead_bars + 2
    max_idx = len(candles) - tail_guard

    for idx in range(min_idx, max_idx):
        candle = candles[idx]
        floor_idx = max(0, idx - args.liquidity_lookback_bars)
        prior_low_idx = previous_idx(lows, idx, floor_idx)
        prior_high_idx = previous_idx(highs, idx, floor_idx)
        candidates: list[tuple[str, int, float, float]] = []
        if prior_low_idx is not None:
            level = float(candles[prior_low_idx].l)
            if candle.l < level and candle.c > level:
                candidates.append(("BULL", prior_low_idx, level, float(candle.l)))
        if prior_high_idx is not None:
            level = float(candles[prior_high_idx].h)
            if candle.h > level and candle.c < level:
                candidates.append(("BEAR", prior_high_idx, level, float(candle.h)))

        for direction, swept_idx, swept_level, sweep_extreme in candidates:
            if direction == "BULL":
                mss_level_idx = previous_idx(highs, idx, floor_idx)
                mss_level = float(candles[mss_level_idx].h) if mss_level_idx is not None else None
            else:
                mss_level_idx = previous_idx(lows, idx, floor_idx)
                mss_level = float(candles[mss_level_idx].l) if mss_level_idx is not None else None
            if mss_level_idx is None or mss_level is None:
                continue

            bucket, ny_time = time_bucket(candle.ts)
            mss_idx: int | None = None
            body_atr: float | None = None
            range_atr: float | None = None
            for forward_idx in range(idx + 1, min(len(candles), idx + args.mss_lookahead_bars + 1)):
                forward = candles[forward_idx]
                broken = forward.c > mss_level if direction == "BULL" else forward.c < mss_level
                if not broken:
                    continue
                current_atr = max(atr[forward_idx], 1e-9)
                body_atr = abs(forward.c - forward.o) / current_atr
                range_atr = (forward.h - forward.l) / current_atr
                if body_atr >= args.min_body_atr or range_atr >= args.min_range_atr:
                    mss_idx = forward_idx
                    break

            fvg: FvgZone | None = None
            ote: OteZone | None = None
            retest: Retest | None = None
            if mss_idx is not None:
                fvg = recent_fvg(candles, direction, max(idx + 2, mss_idx - args.fvg_lookback_bars), mss_idx)
                ote = build_ote(direction, sweep_extreme, candles, idx, mss_idx)
                retest = find_retest(
                    candles,
                    atr,
                    direction,
                    sweep_extreme,
                    fvg,
                    ote,
                    mss_idx,
                    entry_lookahead_bars=args.entry_lookahead_bars,
                    outcome_lookahead_bars=args.outcome_lookahead_bars,
                    stop_buffer_atr=args.stop_buffer_atr,
                    target_rr=args.target_rr,
                )

            status = "sweep_only"
            if mss_idx is not None:
                status = "mss_no_fvg" if fvg is None else "mss_with_fvg"
            if retest is not None:
                status = "confirmed_retest" if retest.confirmed else "unconfirmed_retest"
            sweep_distance_pct = (
                (swept_level - sweep_extreme) / swept_level * 100.0
                if direction == "BULL" and swept_level > 0
                else (sweep_extreme - swept_level) / swept_level * 100.0
                if swept_level > 0
                else 0.0
            )
            events.append(
                LiquidityEvent(
                    direction=direction,
                    sweep_idx=idx,
                    sweep_time=timestamp_for(candles, idx),
                    swept_level=swept_level,
                    swept_level_idx=swept_idx,
                    swept_level_time=timestamp_for(candles, swept_idx),
                    sweep_extreme=sweep_extreme,
                    sweep_distance_pct=sweep_distance_pct,
                    time_bucket=bucket,
                    ny_time=ny_time,
                    mss_idx=mss_idx,
                    mss_time=timestamp_for(candles, mss_idx) if mss_idx is not None else None,
                    mss_level=mss_level,
                    mss_level_idx=mss_level_idx,
                    displacement_body_atr=body_atr,
                    displacement_range_atr=range_atr,
                    fvg=fvg,
                    ote=ote,
                    retest=retest,
                    status=status,
                )
            )
    return events


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
            fvg_size_pct = float(event.fvg.size_pct) if event.fvg is not None else None
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
        rows.append(
            {
                "entry_time": event.retest.timestamp,
                "exit_time": event.retest.outcome_time or event.retest.timestamp,
                "direction": direction,
                "entry_idx": entry_idx,
                "exit_idx": event.retest.outcome_idx if event.retest.outcome_idx is not None else entry_idx,
                "entry_price": float(event.retest.close),
                "stop_price": stop_price,
                "target_price": target_price,
                "stop_distance_pct": stop_distance_pct,
                "signal_return_pct": signal_return_pct,
                "mss_lag_bars": (int(event.mss_idx) - int(event.sweep_idx)) if event.mss_idx is not None else None,
                "displacement_body_atr": event.displacement_body_atr,
                "displacement_range_atr": event.displacement_range_atr,
                "time_bucket": bucket,
                "ny_time": ny_time,
                "h4_bias": h4_bias,
                "d1_bias": d1_bias,
                "fvg_touched": bool(event.retest.fvg_touched),
                "fvg_fill_pct": event.retest.fvg_fill_pct,
                "ote_touched": bool(event.retest.ote_touched),
                "mfe_r": event.retest.mfe_r,
                "mae_r": event.retest.mae_r,
                "target_rr_hit": bool(event.retest.target_rr_hit),
                "stopped": bool(event.retest.stopped),
                "outcome": event.retest.outcome,
                "rr_result": rr_result,
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


def smc_case_namespace(args: argparse.Namespace, case_params: dict[str, Any]) -> argparse.Namespace:
    defaults = {
        "target_rr": 2.0,
        "allowed_time_buckets": "other",
        "swing_n": 3,
        "min_body_atr": 0.7,
        "min_range_atr": 1.1,
        "entry_lookahead_bars": 40,
        "max_open_positions": 1,
        "min_displacement_body_atr": 0.0,
        "min_displacement_range_atr": 0.0,
        "max_mss_lag_bars": 15,
        "global_min_mss_lag_bars": 0,
        "global_max_mss_lag_bars": 0,
        "ny_max_mss_lag_bars": 0,
        "other_min_mss_lag_bars": 0,
        "drop_asia_session": False,
        "leverage": 10.0,
        "position_size_pct": 1.0,
        "maintenance_margin_pct": 0.5,
        "min_liq_buffer_pct": 1.2,
        "initial_capital": 1000.0,
    }
    merged = defaults | case_params
    merged["data_15m"] = args.data_15m
    merged["data_4h"] = args.data_4h
    merged["start_date"] = args.start_date
    return argparse.Namespace(**merged)


def smc_strategy_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        data_15m=args.data_15m,
        data_4h=args.data_4h,
        start_date=args.start_date,
        swing_n=int(args.swing_n),
        swing_lookback=80,
        liquidity_lookback_bars=192,
        mss_lookahead_bars=24,
        fvg_lookback_bars=8,
        entry_lookahead_bars=int(args.entry_lookahead_bars),
        outcome_lookahead_bars=96,
        atr_period=14,
        min_body_atr=float(args.min_body_atr),
        min_range_atr=float(args.min_range_atr),
        stop_buffer_atr=0.05,
        target_rr=float(args.target_rr),
        require_confirmed_retest=True,
        require_fvg_touch=False,
        allow_ote_only=True,
        require_htf_bias_align=True,
        require_h4_bias_align=True,
        require_d1_bias_align=False,
        allowed_time_buckets=str(args.allowed_time_buckets),
        allowed_directions="BEAR",
        require_ote_touch=True,
        min_displacement_body_atr=float(args.min_displacement_body_atr),
        min_displacement_range_atr=float(args.min_displacement_range_atr),
        bull_min_displacement_body_atr=0.9,
        bull_max_displacement_body_atr=1.3,
        bull_min_displacement_range_atr=0.0,
        bull_max_displacement_range_atr=0.0,
        max_mss_lag_bars=int(args.max_mss_lag_bars),
        bear_min_sweep_distance_pct=0.03,
        bear_require_fvg_touch=False,
        bear_min_fvg_size_pct=0.0,
        max_open_positions=int(args.max_open_positions),
        position_risk_fraction=1.0,
        initial_capital=float(args.initial_capital),
    )


def normalize_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def event_return_stats(events: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    returns = [float(event.get("return", 0.0) or 0.0) for event in events]
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value <= 0]
    capital = initial_capital
    for value in returns:
        capital *= 1.0 + value
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "trades": len(events),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(returns) * 100.0, 2) if returns else 0.0,
        "sum_return_pct": round(sum(returns) * 100.0, 4),
        "compounded_return_pct": round((capital - initial_capital) / initial_capital * 100.0, 4),
        "avg_return_pct": round(sum(returns) / len(returns) * 100.0, 4) if returns else 0.0,
        "best_return_pct": round(max(returns) * 100.0, 4) if returns else 0.0,
        "worst_return_pct": round(min(returns) * 100.0, 4) if returns else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
    }


def build_smc_events(
    case_name: str,
    case_params: dict[str, Any],
    args: argparse.Namespace,
    prepared: Any,
    daily: list[Candle],
    h4_highs: list[int],
    h4_lows: list[int],
    d1_highs: list[int],
    d1_lows: list[int],
    allocation: float,
    *,
    taker_fee_rate: float,
    slippage_bps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_args = smc_case_namespace(args, case_params)
    smc_args = smc_strategy_args(case_args)
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
    if int(getattr(case_args, "global_min_mss_lag_bars", 0)) > 0:
        floor = int(case_args.global_min_mss_lag_bars)
        rows = [row for row in rows if row["mss_lag_bars"] is None or int(row["mss_lag_bars"]) >= floor]
    if int(getattr(case_args, "global_max_mss_lag_bars", 0)) > 0:
        ceiling = int(case_args.global_max_mss_lag_bars)
        rows = [row for row in rows if row["mss_lag_bars"] is None or int(row["mss_lag_bars"]) <= ceiling]
    if int(getattr(case_args, "ny_max_mss_lag_bars", 0)) > 0:
        ny_limit = int(case_args.ny_max_mss_lag_bars)
        rows = [
            row for row in rows
            if row["time_bucket"] != "ny_am_killzone"
            or row["mss_lag_bars"] is None
            or int(row["mss_lag_bars"]) <= ny_limit
        ]
    if int(getattr(case_args, "other_min_mss_lag_bars", 0)) > 0:
        other_floor = int(case_args.other_min_mss_lag_bars)
        rows = [
            row for row in rows
            if row["time_bucket"] != "other"
            or row["mss_lag_bars"] is None
            or int(row["mss_lag_bars"]) >= other_floor
        ]
    if bool(getattr(case_args, "drop_asia_session", False)):
        rows = [row for row in rows if row["time_bucket"] != "asia_evening_ny"]

    raw_trades = len(rows)
    rows, slot_skipped = apply_max_open_positions(rows, int(smc_args.max_open_positions))
    capital = float(case_args.initial_capital)
    accepted_rows: list[dict[str, Any]] = []
    guard_skipped = 0
    failures: dict[str, int] = {}
    for row in rows:
        trade = pd.Series(
            {
                "entry_time": row["entry_time"],
                "direction": row["direction"],
                "entry_price": row["entry_price"],
                "initial_stop_price": row["stop_price"],
                "notional": capital * float(case_args.leverage) * float(case_args.position_size_pct),
            }
        )
        diagnostics = _high_leverage_trade_diagnostics(
            trade,
            capital=capital,
            leverage=float(case_args.leverage),
            maintenance_margin_pct=float(case_args.maintenance_margin_pct),
        )
        if float(diagnostics["liquidation_buffer_pct"]) < float(case_args.min_liq_buffer_pct):
            failures["liquidation_buffer_too_small"] = failures.get("liquidation_buffer_too_small", 0) + 1
            guard_skipped += 1
            continue
        return_model = leveraged_net_return(
            signal_return_pct=float(row["signal_return_pct"] or 0.0),
            leverage=float(case_args.leverage),
            position_size_pct=float(case_args.position_size_pct),
            allocation=float(allocation),
            taker_fee_rate=float(taker_fee_rate),
            slippage_bps=float(slippage_bps),
        )
        leveraged_return = float(return_model["account_return"])
        capital *= 1.0 + leveraged_return
        accepted_rows.append(row | {"leveraged_return": leveraged_return, "return_model": return_model})

    events: list[dict[str, Any]] = []
    for row in accepted_rows:
        return_model = row["return_model"]
        return_value = float(row["leveraged_return"])
        events.append(
            {
                "event_type": "smc_short",
                "entry_idx": int(row["entry_idx"]),
                "exit_idx": int(row["exit_idx"]),
                "entry_time": str(normalize_ts(row["entry_time"])),
                "exit_time": str(normalize_ts(row["exit_time"])),
                "direction": "BEAR",
                "return": return_value,
                "return_pct": round(return_value * 100.0, 4),
                "exit_reason": str(row.get("outcome") or row.get("status") or "unknown"),
                "smc_case": case_name,
                "smc_allocation": float(allocation),
                "smc_rr_result": float(row.get("rr_result", 0.0) or 0.0),
                "smc_signal_return_pct": round(float(row.get("signal_return_pct", 0.0) or 0.0), 4),
                "smc_roundtrip_cost_pct": round(float(return_model["roundtrip_cost_pct"]), 4),
                "smc_unit_return_pct": round(float(return_model["net_unit_return_pct"]), 4),
                "smc_taker_fee_rate": float(taker_fee_rate),
                "smc_slippage_bps": float(slippage_bps),
                "smc_time_bucket": str(row.get("time_bucket") or ""),
                "smc_mss_lag_bars": row.get("mss_lag_bars"),
            }
        )
    summary = {
        "raw_trades": raw_trades,
        "slot_trades": len(rows),
        "slot_skipped": slot_skipped,
        "guard_skipped": guard_skipped,
        "failures": failures,
        "accepted_trades": len(events),
        "allocation": allocation,
        "case_params": case_params,
        "fee_model": {
            "taker_fee_rate": float(taker_fee_rate),
            "slippage_bps": float(slippage_bps),
            "roundtrip_cost_pct": round((2.0 * float(taker_fee_rate) + 2.0 * float(slippage_bps) / 10000.0) * 100.0, 4),
        },
        "standalone_event_stats": event_return_stats(events, float(case_args.initial_capital)),
    }
    return events, summary
