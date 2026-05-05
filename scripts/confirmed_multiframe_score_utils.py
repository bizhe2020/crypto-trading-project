from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from scripts.regime_detector import _adx_series
from strategy.scalp_robust_v2_core import Candle, compute_ema_series


@dataclass
class ScoreSnapshot:
    bull_4h: int
    bear_4h: int
    bull_1h: int
    bear_1h: int
    bull_15m: int
    bear_15m: int
    bull_total: int
    bear_total: int
    net_score: int
    conflict: bool


def resample_confirmed_1h(c15m: list[Candle]) -> list[Candle]:
    if not c15m:
        return []
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp(c.ts, unit="s", tz="UTC") for c in c15m],
            "open": [c.o for c in c15m],
            "high": [c.h for c in c15m],
            "low": [c.l for c in c15m],
            "close": [c.c for c in c15m],
            "volume": [c.v for c in c15m],
        }
    ).set_index("date")
    aggregated = (
        frame.resample("1h", label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    return [
        Candle(
            ts=float(pd.Timestamp(row["date"]).timestamp()),
            o=float(row["open"]),
            h=float(row["high"]),
            l=float(row["low"]),
            c=float(row["close"]),
            v=float(row["volume"]),
        )
        for _, row in aggregated.iterrows()
    ]


def align_confirmed_mapping(source: list[Candle], target: list[Candle], source_seconds: int = 3600) -> list[int]:
    mapping: list[int] = []
    src_idx = -1
    for candle in target:
        while src_idx + 1 < len(source) and source[src_idx + 1].ts + source_seconds <= candle.ts:
            src_idx += 1
        mapping.append(src_idx)
    return mapping


def structure_flags(candles: list[Candle], end_idx: int, window: int) -> dict[str, bool]:
    window = max(int(window), 2)
    if end_idx + 1 < window * 2:
        return {"higher_high": False, "higher_low": False, "lower_high": False, "lower_low": False}
    highs = [c.h for c in candles]
    lows = [c.l for c in candles]
    recent_start = end_idx - window + 1
    prev_start = end_idx - window * 2 + 1
    recent_high = max(highs[recent_start : end_idx + 1])
    prev_high = max(highs[prev_start:recent_start])
    recent_low = min(lows[recent_start : end_idx + 1])
    prev_low = min(lows[prev_start:recent_start])
    return {
        "higher_high": recent_high > prev_high,
        "higher_low": recent_low > prev_low,
        "lower_high": recent_high < prev_high,
        "lower_low": recent_low < prev_low,
    }


def score_confirmed_4h(prepared: Any, idx: int) -> tuple[int, int]:
    p = prepared.precomputed
    bull = 0
    bear = 0
    if p.bias_for_15m is not None and idx < len(p.bias_for_15m):
        bias = p.bias_for_15m[idx]
        bull += int(bias == "BULL")
        bear += int(bias == "BEAR")
    if p.regime_1d_bull_100_for_15m is not None and idx < len(p.regime_1d_bull_100_for_15m):
        bull += int(bool(p.regime_1d_bull_100_for_15m[idx]))
    if p.regime_1d_bull_200_for_15m is not None and idx < len(p.regime_1d_bull_200_for_15m):
        bull += int(bool(p.regime_1d_bull_200_for_15m[idx]))
    if p.regime_1d_bear_100_for_15m is not None and idx < len(p.regime_1d_bear_100_for_15m):
        bear += int(bool(p.regime_1d_bear_100_for_15m[idx]))
    if p.regime_1d_bear_200_for_15m is not None and idx < len(p.regime_1d_bear_200_for_15m):
        bear += int(bool(p.regime_1d_bear_200_for_15m[idx]))
    if p.bull_trend_score_for_15m is not None and idx < len(p.bull_trend_score_for_15m):
        bull += int(p.bull_trend_score_for_15m[idx] >= 3)
    if p.bear_trend_score_for_15m is not None and idx < len(p.bear_trend_score_for_15m):
        bear += int(p.bear_trend_score_for_15m[idx] >= 3)
    return bull, bear


def score_confirmed_1h(c1h: list[Candle], mapping_1h: list[int], idx: int) -> tuple[int, int]:
    confirmed_idx = mapping_1h[idx]
    if confirmed_idx < 0:
        return 0, 0
    history = c1h[: confirmed_idx + 1]
    closes = [c.c for c in history]
    highs = [c.h for c in history]
    lows = [c.l for c in history]
    ema20 = compute_ema_series(history, 20)
    ema50 = compute_ema_series(history, 50)
    ema100 = compute_ema_series(history, 100)
    adx = _adx_series(highs, lows, closes, 14)
    end = len(history) - 1
    bull = 0
    bear = 0
    if end >= 0:
        close = closes[end]
        bull += int(end < len(ema20) and close > ema20[end])
        bear += int(end < len(ema20) and close < ema20[end])
        bull += int(end < len(ema50) and close > ema50[end])
        bear += int(end < len(ema50) and close < ema50[end])
        bull += int(end < len(ema100) and close > ema100[end])
        bear += int(end < len(ema100) and close < ema100[end])
        bull += int(end < len(ema50) and end < len(ema100) and ema50[end] > ema100[end])
        bear += int(end < len(ema50) and end < len(ema100) and ema50[end] < ema100[end])
        if end >= 6:
            momentum = close / closes[end - 6] - 1.0
            bull += int(momentum > 0.0)
            bear += int(momentum < 0.0)
        bull += int(end < len(adx) and adx[end] >= 20.0 and end >= 6 and close >= closes[max(end - 6, 0)])
        bear += int(end < len(adx) and adx[end] >= 20.0 and end >= 6 and close <= closes[max(end - 6, 0)])
        structure = structure_flags(history, end, 4)
        bull += int(structure["higher_high"] and structure["higher_low"])
        bear += int(structure["lower_high"] and structure["lower_low"])
    return bull, bear


def score_15m(c15m: list[Candle], idx: int) -> tuple[int, int]:
    if idx <= 0 or idx >= len(c15m):
        return 0, 0
    history = c15m[: idx + 1]
    closes = [c.c for c in history]
    highs = [c.h for c in history]
    lows = [c.l for c in history]
    volumes = [c.v for c in history]
    ema20 = compute_ema_series(history, 20)
    ema50 = compute_ema_series(history, 50)
    adx = _adx_series(highs, lows, closes, 14)
    current = history[-1]
    bull = 0
    bear = 0
    bull += int(current.c > current.o)
    bear += int(current.c < current.o)
    bull += int(idx < len(ema20) and current.c > ema20[idx])
    bear += int(idx < len(ema20) and current.c < ema20[idx])
    bull += int(idx < len(ema50) and current.c > ema50[idx])
    bear += int(idx < len(ema50) and current.c < ema50[idx])
    if idx >= 8:
        momentum = current.c / closes[idx - 8] - 1.0
        bull += int(momentum > 0.0)
        bear += int(momentum < 0.0)
    if idx >= 20:
        avg_volume = sum(volumes[idx - 20 : idx]) / 20.0
        bull += int(avg_volume > 0 and current.v / avg_volume >= 1.25 and current.c > current.o)
        bear += int(avg_volume > 0 and current.v / avg_volume >= 1.25 and current.c < current.o)
    bull += int(idx < len(adx) and adx[idx] >= 18.0 and current.c >= closes[max(idx - 4, 0)])
    bear += int(idx < len(adx) and adx[idx] >= 18.0 and current.c <= closes[max(idx - 4, 0)])
    structure = structure_flags(history, idx, 6)
    bull += int(structure["higher_high"] and structure["higher_low"])
    bear += int(structure["lower_high"] and structure["lower_low"])
    return bull, bear


def score_snapshot(prepared: Any, c1h: list[Candle], mapping_1h: list[int], idx: int) -> ScoreSnapshot:
    bull_4h, bear_4h = score_confirmed_4h(prepared, idx)
    bull_1h, bear_1h = score_confirmed_1h(c1h, mapping_1h, idx)
    bull_15m, bear_15m = score_15m(prepared.c15m, idx)
    bull_total = bull_4h + bull_1h + bull_15m
    bear_total = bear_4h + bear_1h + bear_15m
    return ScoreSnapshot(
        bull_4h=bull_4h,
        bear_4h=bear_4h,
        bull_1h=bull_1h,
        bear_1h=bear_1h,
        bull_15m=bull_15m,
        bear_15m=bear_15m,
        bull_total=bull_total,
        bear_total=bear_total,
        net_score=bull_total - bear_total,
        conflict=bool(bull_total >= 5 and bear_total >= 5),
    )


def passes_score_gate(
    event: dict[str, Any],
    *,
    net_min: int | None = None,
    net_max: int | None = None,
    bull_min: int | None = None,
    bull_max: int | None = None,
    bear_min: int | None = None,
    bear_max: int | None = None,
    conflict_mode: str = "any",
) -> bool:
    net_score = int(event.get("net_score", 0) or 0)
    bull_total = int(event.get("bull_total", 0) or 0)
    bear_total = int(event.get("bear_total", 0) or 0)
    conflict = bool(event.get("conflict"))
    if net_min is not None and net_score < net_min:
        return False
    if net_max is not None and net_score > net_max:
        return False
    if bull_min is not None and bull_total < bull_min:
        return False
    if bull_max is not None and bull_total > bull_max:
        return False
    if bear_min is not None and bear_total < bear_min:
        return False
    if bear_max is not None and bear_total > bear_max:
        return False
    if conflict_mode == "conflict" and not conflict:
        return False
    if conflict_mode == "clean" and conflict:
        return False
    return True
