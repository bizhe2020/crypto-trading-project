from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterable


@dataclass
class RegimeThresholds:
    atr_period: int = 14
    atr_baseline_window: int = 90
    adx_period: int = 14
    momentum_window: int = 30
    ema_fast_period: int = 20
    ema_slow_period: int = 50
    structure_window: int = 12
    high_growth_score_min: int = 3
    compression_growth_score_min: int = 4
    flat_score_min: int = 3
    normal_score_min: int = 1
    strong_high_growth_adx_min: float = 35.0
    strong_high_growth_momentum_min: float = 0.04
    high_growth_adx_min: float = 20.0
    high_growth_momentum_min: float = -0.01
    high_growth_ema_gap_min: float = 0.01
    compression_growth_adx_max: float = 18.5
    compression_growth_atr_ratio_min: float = 0.75
    compression_growth_atr_ratio_max: float = 0.95
    compression_growth_momentum_min: float = -0.02
    compression_growth_ema_gap_min: float = -0.002
    flat_adx_max: float = 20.0
    flat_momentum_abs_max: float = 0.03
    flat_momentum_min: float = -0.005
    flat_ema_gap_min: float = 0.0
    flat_atr_ratio_max: float = 0.9
    normal_adx_min: float = 20.0
    normal_momentum_max: float = 0.0


def _candle_value(candle: Any, key: str) -> float:
    if isinstance(candle, dict):
        return float(candle[key])
    return float(getattr(candle, key))


def _to_thresholds(threshold_dict: dict[str, Any] | RegimeThresholds | None) -> RegimeThresholds:
    if threshold_dict is None:
        return RegimeThresholds()
    if isinstance(threshold_dict, RegimeThresholds):
        return threshold_dict
    allowed = {field.name for field in fields(RegimeThresholds)}
    filtered = {key: value for key, value in threshold_dict.items() if key in allowed}
    return RegimeThresholds(**filtered)


def _ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    period = max(int(period), 1)
    multiplier = 2.0 / (period + 1.0)
    out = [values[0]]
    for value in values[1:]:
        out.append((value - out[-1]) * multiplier + out[-1])
    return out


def _atr_series(highs: list[float], lows: list[float], closes: list[float], period: int) -> list[float]:
    if not highs:
        return []
    period = max(int(period), 1)
    trs: list[float] = []
    atrs: list[float] = []
    atr = 0.0
    for idx, (high, low, close) in enumerate(zip(highs, lows, closes)):
        prev_close = closes[idx - 1] if idx > 0 else close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        if idx == 0:
            atr = tr
        elif idx < period:
            atr = sum(trs) / len(trs)
        elif idx == period:
            atr = sum(trs[-period:]) / period
        else:
            atr = ((atr * (period - 1)) + tr) / period
        atrs.append(atr)
    return atrs


def _adx_series(highs: list[float], lows: list[float], closes: list[float], period: int) -> list[float]:
    if len(highs) < 2:
        return [0.0] * len(highs)
    period = max(int(period), 1)
    tr_list: list[float] = [0.0]
    plus_dm: list[float] = [0.0]
    minus_dm: list[float] = [0.0]
    for idx in range(1, len(highs)):
        up_move = highs[idx] - highs[idx - 1]
        down_move = lows[idx - 1] - lows[idx]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        tr = max(
            highs[idx] - lows[idx],
            abs(highs[idx] - closes[idx - 1]),
            abs(lows[idx] - closes[idx - 1]),
        )
        tr_list.append(tr)

    adx: list[float] = [0.0] * len(highs)
    if len(highs) <= period:
        return adx

    tr_smooth = sum(tr_list[1 : period + 1])
    plus_smooth = sum(plus_dm[1 : period + 1])
    minus_smooth = sum(minus_dm[1 : period + 1])
    dx_values: list[float] = [0.0] * len(highs)

    for idx in range(period + 1, len(highs)):
        tr_smooth = tr_smooth - (tr_smooth / period) + tr_list[idx]
        plus_smooth = plus_smooth - (plus_smooth / period) + plus_dm[idx]
        minus_smooth = minus_smooth - (minus_smooth / period) + minus_dm[idx]
        plus_di = (plus_smooth / tr_smooth * 100.0) if tr_smooth > 0 else 0.0
        minus_di = (minus_smooth / tr_smooth * 100.0) if tr_smooth > 0 else 0.0
        di_sum = plus_di + minus_di
        dx_values[idx] = (abs(plus_di - minus_di) / di_sum * 100.0) if di_sum > 0 else 0.0

    seed_start = period + 1
    seed_end = min(seed_start + period, len(highs))
    if seed_end <= seed_start:
        return adx
    first_adx = sum(dx_values[seed_start:seed_end]) / max(seed_end - seed_start, 1)
    adx[seed_end - 1] = first_adx
    for idx in range(seed_end, len(highs)):
        adx[idx] = ((adx[idx - 1] * (period - 1)) + dx_values[idx]) / period
    return adx


def _structure_flags(highs: list[float], lows: list[float], window: int) -> dict[str, bool]:
    window = max(int(window), 2)
    if len(highs) < window * 2:
        return {"higher_high": False, "higher_low": False, "lower_high": False, "lower_low": False}
    recent_high = max(highs[-window:])
    prev_high = max(highs[-window * 2 : -window])
    recent_low = min(lows[-window:])
    prev_low = min(lows[-window * 2 : -window])
    return {
        "higher_high": recent_high > prev_high,
        "higher_low": recent_low > prev_low,
        "lower_high": recent_high < prev_high,
        "lower_low": recent_low < prev_low,
    }


def compute_regime_features(
    candles_4h: Iterable[Any],
    threshold_dict: dict[str, Any] | RegimeThresholds | None = None,
) -> dict[str, Any]:
    thresholds = _to_thresholds(threshold_dict)
    candles = list(candles_4h)
    if len(candles) < max(
        thresholds.atr_baseline_window,
        thresholds.momentum_window,
        thresholds.ema_slow_period,
        thresholds.structure_window * 2,
    ):
        raise ValueError("Not enough 4h candles to compute regime features")

    highs = [_candle_value(candle, "h") for candle in candles]
    lows = [_candle_value(candle, "l") for candle in candles]
    closes = [_candle_value(candle, "c") for candle in candles]

    atr_values = _atr_series(highs, lows, closes, thresholds.atr_period)
    adx_values = _adx_series(highs, lows, closes, thresholds.adx_period)
    ema_fast = _ema(closes, thresholds.ema_fast_period)
    ema_slow = _ema(closes, thresholds.ema_slow_period)
    structure = _structure_flags(highs, lows, thresholds.structure_window)

    atr_now = atr_values[-1]
    atr_baseline_slice = atr_values[-thresholds.atr_baseline_window :]
    atr_baseline = sum(atr_baseline_slice) / len(atr_baseline_slice)
    atr_ratio = atr_now / atr_baseline if atr_baseline > 0 else 1.0
    momentum = closes[-1] / closes[-1 - thresholds.momentum_window] - 1.0
    ema_gap = ema_fast[-1] / ema_slow[-1] - 1.0 if ema_slow[-1] != 0 else 0.0
    adx = adx_values[-1]

    trend_conflict = (momentum > 0 and ema_gap < 0) or (momentum < 0 and ema_gap > 0)
    bearish_structure = structure["lower_high"] and structure["lower_low"]
    bullish_structure = structure["higher_high"] and structure["higher_low"]

    strong_growth_score = sum(
        [
            adx >= thresholds.high_growth_adx_min,
            momentum >= thresholds.high_growth_momentum_min,
            ema_gap >= thresholds.high_growth_ema_gap_min,
            bullish_structure,
        ]
    )
    compression_growth_score = sum(
        [
            adx <= thresholds.compression_growth_adx_max,
            thresholds.compression_growth_atr_ratio_min <= atr_ratio <= thresholds.compression_growth_atr_ratio_max,
            momentum >= thresholds.compression_growth_momentum_min,
            ema_gap >= thresholds.compression_growth_ema_gap_min,
        ]
    )
    flat_score = sum(
        [
            adx <= thresholds.flat_adx_max,
            atr_ratio <= thresholds.flat_atr_ratio_max,
            abs(momentum) <= thresholds.flat_momentum_abs_max,
            ema_gap >= thresholds.flat_ema_gap_min,
        ]
    )
    normal_score = sum(
        [
            momentum < thresholds.normal_momentum_max,
            adx >= thresholds.normal_adx_min,
            bearish_structure or trend_conflict,
        ]
    )

    return {
        "atr": atr_now,
        "atr_ratio": atr_ratio,
        "adx": adx,
        "momentum": momentum,
        "ema_gap": ema_gap,
        "structure": structure,
        "bullish_structure": bullish_structure,
        "bearish_structure": bearish_structure,
        "trend_conflict": trend_conflict,
        "strong_growth_score": strong_growth_score,
        "compression_growth_score": compression_growth_score,
        "high_growth_score": max(strong_growth_score, compression_growth_score),
        "flat_score": flat_score,
        "normal_score": normal_score,
    }


def detect_regime(
    candles_4h: Iterable[Any],
    threshold_dict: dict[str, Any] | RegimeThresholds | None = None,
) -> str:
    thresholds = _to_thresholds(threshold_dict)
    features = compute_regime_features(candles_4h, thresholds)
    if (
        features["adx"] >= thresholds.strong_high_growth_adx_min
        and features["momentum"] >= thresholds.strong_high_growth_momentum_min
    ):
        return "high_growth"
    if features["compression_growth_score"] >= thresholds.compression_growth_score_min:
        return "high_growth"
    # Early breakout / trend resumption: positive EMA spread, non-negative momentum,
    # and no confirmed bearish structure. This keeps 2025-05 out of "normal".
    if (
        features["adx"] >= thresholds.high_growth_adx_min
        and features["momentum"] >= thresholds.high_growth_momentum_min
        and features["ema_gap"] >= thresholds.high_growth_ema_gap_min
        and not features["bearish_structure"]
    ):
        return "high_growth"
    # Low-volatility drift is only "flat" when the EMA spread is non-negative.
    # Otherwise low-vol down-drift months (for example 2025-01) stay in "normal".
    if (
        features["flat_score"] >= thresholds.flat_score_min
        and features["momentum"] >= thresholds.flat_momentum_min
        and features["ema_gap"] >= thresholds.flat_ema_gap_min
    ):
        return "flat"
    if features["normal_score"] >= thresholds.normal_score_min:
        return "normal"
    return "normal"
