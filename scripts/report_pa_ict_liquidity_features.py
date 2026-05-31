#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.okx_executor import ExecutorConfig  # noqa: E402
from strategy.scalp_robust_v2_core import Candle, dataframe_to_candles, precompute_swings  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "var" / "pa_ict_liquidity"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only PA/ICT liquidity feature report: sweep, MSS, FVG, OTE, and time buckets."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "config.paper.high-leverage-structure.json"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
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
    parser.add_argument(
        "--allow-incomplete-tail",
        action="store_true",
        help="Scan recent tail bars with only currently available lookahead; useful for live/cutoff audits.",
    )
    parser.add_argument("--include-events", action="store_true", help="Include every detected event in the JSON report.")
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> ExecutorConfig:
    payload = json.loads(path.read_text())
    return ExecutorConfig.from_dict(payload)


def symbol_file_prefix(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def load_dataframe(path: Path, start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
    df = pd.read_feather(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    elif "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        raise ValueError(f"Unsupported dataframe format: {path}")
    df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    return df.sort_values("date").reset_index(drop=True)


def timestamp_for(candles: list[Candle], idx: int) -> str:
    return datetime.fromtimestamp(candles[idx].ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


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
    elif minutes >= 20 * 60 or minutes < 0 * 60 + 30:
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
        return None, None, None, None, None

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
                distance_pct = (level - candle.l) / level * 100.0 if level > 0 else 0.0
                candidates.append(("BULL", prior_low_idx, level, float(candle.l)))
        if prior_high_idx is not None:
            level = float(candles[prior_high_idx].h)
            if candle.h > level and candle.c < level:
                distance_pct = (candle.h - level) / level * 100.0 if level > 0 else 0.0
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
                displaced = body_atr >= args.min_body_atr or range_atr >= args.min_range_atr
                if displaced:
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


def pct(numerator: int | float, denominator: int | float) -> float:
    return round((float(numerator) / float(denominator) * 100.0), 2) if denominator else 0.0


def summarize_group(events: list[LiquidityEvent]) -> dict[str, Any]:
    total = len(events)
    with_mss = [event for event in events if event.mss_idx is not None]
    with_fvg = [event for event in events if event.fvg is not None]
    with_retest = [event for event in events if event.retest is not None]
    confirmed = [event for event in with_retest if event.retest and event.retest.confirmed]
    target_hits = [event for event in confirmed if event.retest and event.retest.target_rr_hit]
    stops = [event for event in confirmed if event.retest and event.retest.stopped]
    mfe_values = [event.retest.mfe_r for event in confirmed if event.retest and event.retest.mfe_r is not None]
    mae_values = [event.retest.mae_r for event in confirmed if event.retest and event.retest.mae_r is not None]
    by_retest_zone = {
        "fvg_only": [
            event
            for event in confirmed
            if event.retest and event.retest.fvg_touched and not event.retest.ote_touched
        ],
        "ote_only": [
            event
            for event in confirmed
            if event.retest and event.retest.ote_touched and not event.retest.fvg_touched
        ],
        "fvg_and_ote": [
            event
            for event in confirmed
            if event.retest and event.retest.fvg_touched and event.retest.ote_touched
        ],
    }
    zone_summary = {}
    for zone_name, zone_events in by_retest_zone.items():
        zone_hits = [event for event in zone_events if event.retest and event.retest.target_rr_hit]
        zone_stops = [event for event in zone_events if event.retest and event.retest.stopped]
        zone_mfe = [event.retest.mfe_r for event in zone_events if event.retest and event.retest.mfe_r is not None]
        zone_summary[zone_name] = {
            "confirmed_retests": len(zone_events),
            "target_hit_pct": pct(len(zone_hits), len(zone_events)),
            "stop_pct": pct(len(zone_stops), len(zone_events)),
            "avg_mfe_r": round(sum(zone_mfe) / len(zone_mfe), 3) if zone_mfe else 0.0,
        }
    return {
        "events": total,
        "with_mss": len(with_mss),
        "with_mss_pct": pct(len(with_mss), total),
        "with_fvg": len(with_fvg),
        "with_fvg_pct": pct(len(with_fvg), total),
        "with_retest": len(with_retest),
        "with_retest_pct": pct(len(with_retest), total),
        "confirmed_retests": len(confirmed),
        "confirmed_retest_pct": pct(len(confirmed), total),
        "target_hits": len(target_hits),
        "target_hit_pct_of_confirmed": pct(len(target_hits), len(confirmed)),
        "stops": len(stops),
        "stop_pct_of_confirmed": pct(len(stops), len(confirmed)),
        "avg_mfe_r": round(sum(mfe_values) / len(mfe_values), 3) if mfe_values else 0.0,
        "avg_mae_r": round(sum(mae_values) / len(mae_values), 3) if mae_values else 0.0,
        "by_retest_zone": zone_summary,
    }


def summarize(events: list[LiquidityEvent]) -> dict[str, Any]:
    by_direction = {direction: summarize_group([event for event in events if event.direction == direction]) for direction in ("BULL", "BEAR")}
    by_status = dict(Counter(event.status for event in events))
    by_bucket: dict[str, Any] = {}
    for bucket in sorted(set(event.time_bucket for event in events)):
        by_bucket[bucket] = summarize_group([event for event in events if event.time_bucket == bucket])
    by_year: dict[str, Any] = {}
    for event in events:
        year = event.sweep_time[:4]
        by_year.setdefault(year, []).append(event)
    by_year_summary = {year: summarize_group(items) for year, items in sorted(by_year.items())}
    return {
        "overall": summarize_group(events),
        "by_direction": by_direction,
        "by_status": by_status,
        "by_time_bucket": by_bucket,
        "by_year": by_year_summary,
    }


def output_path_for(output_dir: Path, start: pd.Timestamp, end: pd.Timestamp | None) -> Path:
    end_text = end.strftime("%Y-%m-%d") if end is not None else "latest"
    return output_dir / f"pa_ict_liquidity_features_{start.strftime('%Y-%m-%d')}_to_{end_text}.json"


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    start = pd.Timestamp(args.start_date, tz="UTC")
    end = pd.Timestamp(args.end_date, tz="UTC") if args.end_date else None
    data_root = Path(args.data_root or config.data_root)
    prefix = symbol_file_prefix(config.symbol)
    data_15m = data_root / f"{prefix}-{config.timeframe}-futures.feather"
    df15 = load_dataframe(data_15m, start, end)
    candles = dataframe_to_candles(df15)
    events = scan_events(candles, args)
    report = {
        "config": str(config_path),
        "data": {
            "path_15m": str(data_15m),
            "rows_15m": len(df15),
            "start": str(df15["date"].iloc[0]) if not df15.empty else None,
            "end": str(df15["date"].iloc[-1]) if not df15.empty else None,
        },
        "parameters": {
            "swing_n": args.swing_n,
            "swing_lookback": args.swing_lookback,
            "liquidity_lookback_bars": args.liquidity_lookback_bars,
            "mss_lookahead_bars": args.mss_lookahead_bars,
            "fvg_lookback_bars": args.fvg_lookback_bars,
            "entry_lookahead_bars": args.entry_lookahead_bars,
            "outcome_lookahead_bars": args.outcome_lookahead_bars,
            "atr_period": args.atr_period,
            "min_body_atr": args.min_body_atr,
            "min_range_atr": args.min_range_atr,
            "stop_buffer_atr": args.stop_buffer_atr,
            "target_rr": args.target_rr,
        },
        "summary": summarize(events),
        "recent_examples": [asdict(event) for event in events[-20:]],
    }
    if args.include_events:
        report["events"] = [asdict(event) for event in events]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_path_for(output_dir, start, end)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output_path)
    compact = {
        "events": report["summary"]["overall"],
        "by_direction": report["summary"]["by_direction"],
        "by_time_bucket": {
            key: {
                "events": value["events"],
                "confirmed_retests": value["confirmed_retests"],
                "target_hit_pct_of_confirmed": value["target_hit_pct_of_confirmed"],
                "stop_pct_of_confirmed": value["stop_pct_of_confirmed"],
                "avg_mfe_r": value["avg_mfe_r"],
                "by_retest_zone": value["by_retest_zone"],
            }
            for key, value in report["summary"]["by_time_bucket"].items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    if args.stdout:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
