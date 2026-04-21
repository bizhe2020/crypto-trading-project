from __future__ import annotations

import csv
import math
from bisect import bisect_right
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from strategy.funding_oi_trailing import FundingOIOverlay, FundingOIOverlayConfig, FundingOISnapshot
from strategy.price_band_trailing import PriceBandTrailingConfig, PriceBandTrailingOverlay


class Direction:
    BULL = "BULL"
    BEAR = "BEAR"
    NONE = "NONE"


class ActionType(str, Enum):
    HOLD = "HOLD"
    OPEN_LONG = "OPEN_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE_POSITION = "CLOSE_POSITION"
    UPDATE_STOP = "UPDATE_STOP"


@dataclass
class Candle:
    ts: float
    o: float
    h: float
    l: float
    c: float
    v: float


@dataclass
class Trade:
    entry_time: str
    exit_time: str
    direction: str
    signal_entry_price: float
    entry_price: float
    signal_exit_price: float
    exit_price: float
    gross_pnl: float
    fees: float
    slippage_cost: float
    pnl: float
    pnl_pct: float
    rr_ratio: float
    exit_reason: str
    capital_at_entry: float


@dataclass
class StrategyConfig:
    rr_ratio: float = 4.0
    trailing_config: str = "B"
    sl_buffer_pct: float = 1.0
    use_hfvf_filter: bool = True
    leverage: float = 3.0
    initial_capital: float = 1000.0
    pullback_window: int = 30
    risk_per_trade: float = 0.01
    position_size_pct: float = 0.35
    fixed_notional_usdt: float | None = None
    allow_long: bool = True
    allow_short: bool = True
    regime_filter_1d_ema_period: int | None = None
    regime_filter_1d_direction: str = "bull"
    enable_directional_regime_switch: bool = False
    long_regime_filter_1d_ema_period: int | None = None
    short_regime_filter_1d_ema_period: int | None = None
    enable_dual_pending_state: bool = False
    regime_position_size_pct_below_ema: float | None = None
    enable_regime_layered_exit: bool = False
    enable_short_regime_layered_exit: bool = False
    short_pullback_window: int | None = None
    short_sl_buffer_pct: float | None = None
    short_strong_rr_ratio: float = 5.0
    short_mid_rr_ratio: float = 4.0
    short_weak_rr_ratio: float = 3.0
    enable_target_rr_cap: bool = False
    loose_target_rr_cap: float | None = None
    normal_target_rr_cap: float | None = None
    tight_target_rr_cap: float | None = None
    enable_regime_directional_risk: bool = False
    bull_strong_long_risk_per_trade: float | None = None
    bull_strong_short_risk_per_trade: float | None = None
    bull_weak_long_risk_per_trade: float | None = None
    bull_weak_short_risk_per_trade: float | None = None
    bear_weak_long_risk_per_trade: float | None = None
    bear_weak_short_risk_per_trade: float | None = None
    bear_strong_long_risk_per_trade: float | None = None
    bear_strong_short_risk_per_trade: float | None = None
    bull_weak_long_rr_ratio_override: float | None = None
    bull_weak_long_trail_style_override: str | None = None
    bear_weak_short_rr_ratio_override: float | None = None
    bear_weak_short_trail_style_override: str | None = None
    enable_price_band_trailing: bool = False
    price_band_trailing_config: PriceBandTrailingConfig | None = None
    disable_fixed_target: bool = False
    enable_funding_oi_trailing: bool = False
    funding_oi_trailing_config: FundingOIOverlayConfig | None = None
    funding_oi_snapshots: list[FundingOISnapshot] | None = None
    enable_atr_trailing: bool = False
    atr_period: int = 14
    atr_activation_rr: float = 2.0
    atr_loose_multiplier: float = 2.7
    atr_normal_multiplier: float = 2.25
    atr_tight_multiplier: float = 1.8
    atr_regime_filter: str = "all"
    disable_fixed_target_exit: bool = False
    enable_regime_switching: bool = False
    regime_switcher_thresholds: dict[str, Any] | None = None
    regime_switcher_hg_overrides: dict[str, Any] | None = None
    regime_switcher_normal_overrides: dict[str, Any] | None = None
    regime_switcher_flat_overrides: dict[str, Any] | None = None
    taker_fee_rate: float = 0.0005
    slippage_bps: float = 2.0


@dataclass
class StrategyAction:
    type: ActionType
    timestamp: str
    direction: str | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    reason: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class PositionState:
    direction: str
    signal_entry_price: float
    entry_price: float
    sl_price: float
    initial_sl_price: float
    target_price: float
    entry_time: str
    capital_at_entry: float
    risk_amount: float
    notional: float
    quantity: float
    entry_fee: float
    entry_slippage_cost: float
    entry_idx: int
    entry_regime_score: int
    target_rr: float
    max_hold_bars: int | None
    trail_style: str
    stage: int = -1
    exchange_order_id: str | None = None
    exchange_attach_algo_id: str | None = None
    exchange_attach_algo_client_id: str | None = None


@dataclass
class PendingPullback:
    direction: str
    bos_idx: int
    ob_zone: dict[str, float]
    pullback_window: int


@dataclass
class StrategySnapshot:
    capital: float
    position: dict[str, Any] | None
    exit_reasons: dict[str, int]
    trade_count: int


@dataclass
class PrecomputedState:
    bias_4h: list[str]
    regime_1d_bull_100: list[bool]
    regime_1d_bull_200: list[bool]
    regime_1d_bear_100: list[bool]
    regime_1d_bear_200: list[bool]
    ema50_4h: list[float]
    ema200_4h: list[float]
    bull_trend_score_4h: list[int]
    bear_trend_score_4h: list[int]
    highs_15m: list[int]
    lows_15m: list[int]
    highs_set: set[int]
    lows_set: set[int]
    broken_bear: list[bool]
    reclaimed_bear: list[bool]
    broken_bull: list[bool]
    reclaimed_bull: list[bool]


def load_candles(path: str | Path) -> list[Candle]:
    candles: list[Candle] = []
    with open(path, "r") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candles.append(
                Candle(
                    ts=float(row["timestamp"]) / 1000.0,
                    o=float(row["open"]),
                    h=float(row["high"]),
                    l=float(row["low"]),
                    c=float(row["close"]),
                    v=float(row["volume"]),
                )
            )
    return candles


def dataframe_to_candles(dataframe: pd.DataFrame) -> list[Candle]:
    df = dataframe.copy()
    df.columns = [column.lower() for column in df.columns]
    if "date" in df.columns:
        ts_series = pd.to_datetime(df["date"], utc=True)
        timestamps = ts_series.astype("int64") / 1_000_000_000
    elif "timestamp" in df.columns:
        timestamps = pd.to_numeric(df["timestamp"], errors="coerce") / 1000.0
    else:
        raise ValueError("Dataframe must contain date or timestamp column")

    candles: list[Candle] = []
    for idx, row in df.iterrows():
        candles.append(
            Candle(
                ts=float(timestamps.iloc[idx]),
                o=float(row["open"]),
                h=float(row["high"]),
                l=float(row["low"]),
                c=float(row["close"]),
                v=float(row.get("volume", 0.0)),
            )
        )
    return candles


def align_timeframes(c4h: list[Candle], c15m: list[Candle]) -> list[int]:
    mapping: list[int] = []
    c4h_idx = 0
    for candle in c15m:
        while c4h_idx + 1 < len(c4h) and c4h[c4h_idx + 1].ts <= candle.ts:
            c4h_idx += 1
        mapping.append(c4h_idx)
    return mapping


def precompute_swings(candles: list[Candle], n: int = 3, lookback: int = 80) -> tuple[list[int], list[int]]:
    highs: list[int] = []
    lows: list[int] = []
    for i in range(lookback, len(candles) - 1):
        is_high = all(candles[j].h <= candles[i].h for j in range(i - n, i + 1))
        if is_high:
            highs.append(i)
        is_low = all(candles[j].l >= candles[i].l for j in range(i - n, i + 1))
        if is_low:
            lows.append(i)
    return highs, lows


def precompute_fvgs_4h(candles: list[Candle]) -> list[tuple[int, str, float, float]]:
    fvgs: list[tuple[int, str, float, float]] = []
    for i in range(2, len(candles)):
        if candles[i - 2].l > candles[i].h:
            fvgs.append((i, "bull", candles[i - 2].l, candles[i].h))
        if candles[i - 2].h < candles[i].l:
            fvgs.append((i, "bear", candles[i].l, candles[i - 2].h))
    return fvgs


def precompute_4h_bias(c4h: list[Candle], fvgs: list[tuple[int, str, float, float]]) -> list[str]:
    biases: list[str] = []
    for c4h_idx in range(len(c4h)):
        cp = c4h[c4h_idx].c
        bull_above = False
        bear_below = False
        for fvg_idx, ftype, top, bottom in reversed(fvgs):
            if fvg_idx > c4h_idx:
                continue
            if ftype == "bull" and cp > top:
                bull_above = True
                break
            if ftype == "bear" and cp < bottom:
                bear_below = True
                break
        if bull_above and not bear_below:
            biases.append(Direction.BULL)
        elif bear_below and not bull_above:
            biases.append(Direction.BEAR)
        elif bull_above:
            biases.append(Direction.BULL)
        elif bear_below:
            biases.append(Direction.BEAR)
        else:
            biases.append(Direction.NONE)
    return biases


def precompute_1d_regime(candles: list[Candle], ema_period: int = 200) -> list[bool]:
    regime = [False] * len(candles)
    if not candles:
        return regime
    alpha = 2.0 / (ema_period + 1.0)
    ema = candles[0].c
    for idx, candle in enumerate(candles):
        if idx == 0:
            ema = candle.c
        else:
            ema = alpha * candle.c + (1 - alpha) * ema
        regime[idx] = idx >= ema_period - 1 and candle.c > ema
    return regime


def precompute_1d_bear_regime(candles: list[Candle], ema_period: int = 200) -> list[bool]:
    regime = [False] * len(candles)
    if not candles:
        return regime
    alpha = 2.0 / (ema_period + 1.0)
    ema = candles[0].c
    for idx, candle in enumerate(candles):
        if idx == 0:
            ema = candle.c
        else:
            ema = alpha * candle.c + (1 - alpha) * ema
        regime[idx] = idx >= ema_period - 1 and candle.c < ema
    return regime


def compute_ema_series(candles: list[Candle], period: int) -> list[float]:
    if not candles:
        return []
    alpha = 2.0 / (period + 1.0)
    ema = candles[0].c
    values: list[float] = []
    for idx, candle in enumerate(candles):
        if idx == 0:
            ema = candle.c
        else:
            ema = alpha * candle.c + (1 - alpha) * ema
        values.append(ema)
    return values


def precompute_trend_score_4h(candles: list[Candle], ema50: list[float], ema200: list[float]) -> list[int]:
    scores = [0] * len(candles)
    for idx, candle in enumerate(candles):
        score = 0
        if idx < len(ema50) and candle.c > ema50[idx]:
            score += 1
        if idx < len(ema200) and ema50[idx] > ema200[idx]:
            score += 1
        if idx >= 3 and ema50[idx] > ema50[idx - 3]:
            score += 1
        if idx < len(ema200) and candle.c > 0 and (ema50[idx] - ema200[idx]) / candle.c > 0.02:
            score += 1
        scores[idx] = score
    return scores


def precompute_bear_trend_score_4h(candles: list[Candle], ema50: list[float], ema200: list[float]) -> list[int]:
    scores = [0] * len(candles)
    for idx, candle in enumerate(candles):
        score = 0
        if idx < len(ema50) and candle.c < ema50[idx]:
            score += 1
        if idx < len(ema200) and ema50[idx] < ema200[idx]:
            score += 1
        if idx >= 3 and ema50[idx] < ema50[idx - 3]:
            score += 1
        if idx < len(ema200) and candle.c > 0 and (ema200[idx] - ema50[idx]) / candle.c > 0.02:
            score += 1
        scores[idx] = score
    return scores


def precompute_mss(
    c15m: list[Candle], highs: list[int], lows: list[int], lookback: int = 80
) -> tuple[list[bool], list[bool], list[bool], list[bool]]:
    n = len(c15m)
    highs_arr = [c15m[i].h for i in range(n)]
    lows_arr = [c15m[i].l for i in range(n)]
    closes_arr = [c15m[i].c for i in range(n)]

    broken_bear = [False] * n
    reclaimed_bear = [False] * n
    broken_bull = [False] * n
    reclaimed_bull = [False] * n

    for idx_pos in range(1, len(lows)):
        i = lows[idx_pos]
        prev_idx = lows[idx_pos - 1]
        prev_h = highs_arr[prev_idx]
        broken = False
        for j in range(prev_idx + 1, max(prev_idx + 1, i - 2)):
            if highs_arr[j] > prev_h:
                broken = True
                break
        broken_bear[i] = broken
        if broken:
            for j in range(max(prev_idx + 1, i - 20), i):
                if closes_arr[j] > prev_h:
                    reclaimed_bear[i] = True
                    break

    for idx_pos in range(1, len(highs)):
        i = highs[idx_pos]
        prev_idx = highs[idx_pos - 1]
        prev_l = lows_arr[prev_idx]
        broken = False
        for j in range(prev_idx + 1, max(prev_idx + 1, i - 2)):
            if lows_arr[j] < prev_l:
                broken = True
                break
        broken_bull[i] = broken
        if broken:
            for j in range(max(prev_idx + 1, i - 20), i):
                if closes_arr[j] < prev_l:
                    reclaimed_bull[i] = True
                    break

    return broken_bear, reclaimed_bear, broken_bull, reclaimed_bull


def find_ob(candles: list[Candle], idx: int, direction: str) -> dict[str, float] | None:
    start = max(0, idx - 10)
    for i in range(start, idx):
        range_value = candles[i].h - candles[i].l
        if range_value == 0:
            continue
        body = abs(candles[i].c - candles[i].o)
        if body / range_value < 0.6:
            continue
        if direction == Direction.BEAR and candles[i].c > candles[i].o:
            return {"top": max(candles[i].o, candles[i].c), "bottom": min(candles[i].o, candles[i].c)}
        if direction == Direction.BULL and candles[i].c < candles[i].o:
            return {"top": max(candles[i].o, candles[i].c), "bottom": min(candles[i].o, candles[i].c)}
    return None


def build_precomputed_state(c4h: list[Candle], c15m: list[Candle], swing_n: int = 3, lookback: int = 80) -> PrecomputedState:
    highs_15m, lows_15m = precompute_swings(c15m, n=swing_n, lookback=lookback)
    fvgs_4h = precompute_fvgs_4h(c4h)
    bias_4h = precompute_4h_bias(c4h, fvgs_4h)
    regime_1d_bull_100 = precompute_1d_regime(c4h, ema_period=100)
    regime_1d_bull_200 = precompute_1d_regime(c4h, ema_period=200)
    regime_1d_bear_100 = precompute_1d_bear_regime(c4h, ema_period=100)
    regime_1d_bear_200 = precompute_1d_bear_regime(c4h, ema_period=200)
    ema50_4h = compute_ema_series(c4h, 50)
    ema200_4h = compute_ema_series(c4h, 200)
    bull_trend_score_4h = precompute_trend_score_4h(c4h, ema50_4h, ema200_4h)
    bear_trend_score_4h = precompute_bear_trend_score_4h(c4h, ema50_4h, ema200_4h)
    broken_bear, reclaimed_bear, broken_bull, reclaimed_bull = precompute_mss(c15m, highs_15m, lows_15m, lookback=lookback)
    return PrecomputedState(
        bias_4h=bias_4h,
        regime_1d_bull_100=regime_1d_bull_100,
        regime_1d_bull_200=regime_1d_bull_200,
        regime_1d_bear_100=regime_1d_bear_100,
        regime_1d_bear_200=regime_1d_bear_200,
        ema50_4h=ema50_4h,
        ema200_4h=ema200_4h,
        bull_trend_score_4h=bull_trend_score_4h,
        bear_trend_score_4h=bear_trend_score_4h,
        highs_15m=highs_15m,
        lows_15m=lows_15m,
        highs_set=set(highs_15m),
        lows_set=set(lows_15m),
        broken_bear=broken_bear,
        reclaimed_bear=reclaimed_bear,
        broken_bull=broken_bull,
        reclaimed_bull=reclaimed_bull,
    )


class ScalpRobustEngine:
    def __init__(
        self,
        c4h: list[Candle],
        c15m: list[Candle],
        mapping: list[int],
        precomputed: PrecomputedState,
        config: StrategyConfig | None = None,
    ):
        self.c4h = c4h
        self.c15m = c15m
        self.mapping = mapping
        self.precomputed = precomputed
        self.config = config or StrategyConfig()
        self._base_config = deepcopy(self.config)
        self.capital = self.config.initial_capital
        self.trades: list[Trade] = []
        self.restored_trade_count = 0
        self.position: PositionState | None = None
        self.exit_reasons: dict[str, int] = {}
        self._regime_switch_cache: dict[int, tuple[str, StrategyConfig]] = {}
        band_config = self.config.price_band_trailing_config or PriceBandTrailingConfig()
        self.price_band_overlay = PriceBandTrailingOverlay(band_config)
        funding_config = self.config.funding_oi_trailing_config or FundingOIOverlayConfig()
        self.funding_oi_overlay = FundingOIOverlay(funding_config)
        funding_snapshots = self.config.funding_oi_snapshots or []
        funding_snapshots = sorted(funding_snapshots, key=lambda item: item.timestamp_ms)
        self._funding_oi_snapshots = funding_snapshots
        self._funding_oi_timestamps = [item.timestamp_ms for item in funding_snapshots]
        self._atr_15m = self._compute_atr_series(self.config.atr_period)

    @classmethod
    def from_candles(
        cls,
        c4h: list[Candle],
        c15m: list[Candle],
        config: StrategyConfig | None = None,
    ) -> "ScalpRobustEngine":
        mapping = align_timeframes(c4h, c15m)
        precomputed = build_precomputed_state(c4h, c15m)
        return cls(c4h, c15m, mapping, precomputed, config)

    def snapshot(self) -> StrategySnapshot:
        return StrategySnapshot(
            capital=self.capital,
            position=asdict(self.position) if self.position else None,
            exit_reasons=dict(self.exit_reasons),
            trade_count=self.restored_trade_count + len(self.trades),
        )

    def restore_snapshot(self, snapshot: dict[str, Any] | None) -> None:
        if not snapshot:
            return
        self.capital = float(snapshot.get("capital", self.capital))
        position_payload = snapshot.get("position")
        self.position = PositionState(**position_payload) if isinstance(position_payload, dict) else None
        exit_reasons = snapshot.get("exit_reasons")
        self.exit_reasons = dict(exit_reasons) if isinstance(exit_reasons, dict) else {}
        try:
            self.restored_trade_count = int(snapshot.get("trade_count", 0) or 0)
        except (TypeError, ValueError):
            self.restored_trade_count = 0

    def open_position(self, idx: int, direction: str, entry_price: float, sl_price: float, target_price: float) -> StrategyAction:
        applied_risk_per_trade, risk_regime = self._risk_per_trade_for_idx(idx, direction)
        risk_amount = self.capital * applied_risk_per_trade
        filled_entry_price = self._apply_entry_slippage(entry_price, direction)
        position_size_pct = self._position_size_pct_for_idx(idx)
        entry_regime_score, target_rr, max_hold_bars, trail_style = self._exit_template_for_idx(idx, direction)
        filled_target_price = self._target_price_from_rr(filled_entry_price, sl_price, direction, target_rr)
        max_notional = (
            float(self.config.fixed_notional_usdt)
            if self.config.fixed_notional_usdt is not None
            else self.capital * position_size_pct * self.config.leverage
        )
        stop_distance = abs(filled_entry_price - sl_price)
        risk_based_notional = (
            (risk_amount / stop_distance) * filled_entry_price
            if stop_distance > 0
            else max_notional
        )
        notional = min(max_notional, risk_based_notional)
        quantity = notional / filled_entry_price if filled_entry_price > 0 else 0.0
        entry_fee = notional * self.config.taker_fee_rate
        entry_slippage_cost = quantity * abs(filled_entry_price - entry_price)
        self.position = PositionState(
            direction=direction,
            signal_entry_price=entry_price,
            entry_price=filled_entry_price,
            sl_price=sl_price,
            initial_sl_price=sl_price,
            target_price=filled_target_price,
            entry_time=datetime.fromtimestamp(self.c15m[idx].ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
            capital_at_entry=self.capital,
            risk_amount=risk_amount,
            notional=notional,
            quantity=quantity,
            entry_fee=entry_fee,
            entry_slippage_cost=entry_slippage_cost,
            entry_idx=idx,
            entry_regime_score=entry_regime_score,
            target_rr=target_rr,
            max_hold_bars=max_hold_bars,
            trail_style=trail_style,
        )
        return StrategyAction(
            type=ActionType.OPEN_LONG if direction == Direction.BULL else ActionType.OPEN_SHORT,
            timestamp=self.position.entry_time,
            direction=direction,
            entry_price=filled_entry_price,
            stop_price=sl_price,
            target_price=filled_target_price,
            metadata={
                "index": idx,
                "position_size_pct": position_size_pct,
                "entry_regime_score": entry_regime_score,
                "target_rr": target_rr,
                "max_hold_bars": max_hold_bars,
                "trail_style": trail_style,
                "signal_entry_price": entry_price,
                "signal_target_price": target_price,
                "capital_at_entry": self.capital,
                "notional": notional,
                "quantity": quantity,
                "max_notional": max_notional,
                "risk_based_notional": risk_based_notional,
                "risk_per_trade": applied_risk_per_trade,
                "risk_amount": risk_amount,
                "risk_regime": risk_regime,
                "entry_fee": entry_fee,
                "entry_slippage_cost": entry_slippage_cost,
            },
        )

    def close_position(self, idx: int, reason: str, exit_price: float | None = None) -> StrategyAction:
        if not self.position:
            return StrategyAction(type=ActionType.HOLD, timestamp=self._timestamp_for_idx(idx), reason="no_position")
        pos = self.position
        if exit_price is None:
            exit_price = self.c15m[idx].c
        filled_exit_price = self._apply_exit_slippage(exit_price, pos.direction)
        if pos.direction == Direction.BULL:
            gross_pnl = pos.quantity * (filled_exit_price - pos.entry_price)
        else:
            gross_pnl = pos.quantity * (pos.entry_price - filled_exit_price)
        exit_fee = pos.quantity * filled_exit_price * self.config.taker_fee_rate
        fees = pos.entry_fee + exit_fee
        slippage_cost = pos.entry_slippage_cost + pos.quantity * abs(filled_exit_price - exit_price)
        pnl = gross_pnl - fees
        rr_ratio = pnl / pos.risk_amount if pos.risk_amount > 0 else 0.0
        self.trades.append(
            Trade(
                entry_time=pos.entry_time,
                exit_time=self._timestamp_for_idx(idx),
                direction=pos.direction,
                signal_entry_price=pos.signal_entry_price,
                entry_price=pos.entry_price,
                signal_exit_price=exit_price,
                exit_price=filled_exit_price,
                gross_pnl=gross_pnl,
                fees=fees,
                slippage_cost=slippage_cost,
                pnl=pnl,
                pnl_pct=pnl / pos.capital_at_entry,
                rr_ratio=rr_ratio,
                exit_reason=reason,
                capital_at_entry=pos.capital_at_entry,
            )
        )
        self.capital += pnl
        self.exit_reasons[reason] = self.exit_reasons.get(reason, 0) + 1
        self.position = None
        return StrategyAction(
            type=ActionType.CLOSE_POSITION,
            timestamp=self._timestamp_for_idx(idx),
            direction=pos.direction,
            exit_price=filled_exit_price,
            reason=reason,
            metadata={
                "index": idx,
                "gross_pnl": gross_pnl,
                "fees": fees,
                "slippage_cost": slippage_cost,
                "net_pnl": pnl,
                "signal_exit_price": exit_price,
            },
        )

    def manage_position(self, idx: int) -> list[StrategyAction]:
        if not self.position:
            return []
        pos = self.position
        curr = self.c15m[idx]
        actions: list[StrategyAction] = []

        if pos.max_hold_bars is not None and idx - pos.entry_idx >= pos.max_hold_bars:
            actions.append(self.close_position(idx, "time_exit"))
            return actions

        if pos.direction == Direction.BULL:
            if curr.l <= pos.sl_price:
                actions.append(self.close_position(idx, "stop_loss", pos.sl_price))
                return actions

            if self.config.enable_funding_oi_trailing:
                funding_snapshot = self._funding_oi_snapshot_for_idx(idx)
                if funding_snapshot is not None:
                    funding_decision = self.funding_oi_overlay.evaluate(funding_snapshot, pos)
                    if funding_decision.action == "exit":
                        actions.append(
                            self.close_position(
                                idx,
                                funding_decision.reason or "funding_oi_exit",
                                funding_decision.exit_price,
                            )
                        )
                        return actions
                    if funding_decision.action == "tighten" and funding_decision.stop_price is not None:
                        if funding_decision.stop_price > pos.sl_price:
                            pos.sl_price = funding_decision.stop_price
                            actions.append(
                                StrategyAction(
                                    type=ActionType.UPDATE_STOP,
                                    timestamp=self._timestamp_for_idx(idx),
                                    direction=pos.direction,
                                    stop_price=funding_decision.stop_price,
                                    reason=funding_decision.reason or "funding_oi_tighten",
                                    metadata=funding_decision.metrics or {},
                                )
                            )

            if self.config.enable_price_band_trailing:
                band_decision = self.price_band_overlay.evaluate(curr, pos)
                if band_decision.action == "exit":
                    actions.append(self.close_position(idx, band_decision.reason or "band_exit", band_decision.exit_price))
                    return actions
                if band_decision.action == "tighten" and band_decision.stop_price is not None:
                    if band_decision.stop_price > pos.sl_price:
                        pos.sl_price = band_decision.stop_price
                        actions.append(
                            StrategyAction(
                                type=ActionType.UPDATE_STOP,
                                timestamp=self._timestamp_for_idx(idx),
                                direction=pos.direction,
                                stop_price=band_decision.stop_price,
                                reason=band_decision.reason or "band_tighten",
                                metadata=band_decision.metrics or {},
                            )
                        )

            update = self._apply_trailing_bull(pos, curr, idx)
            atr_update = self._apply_atr_trailing_bull(pos, curr, idx)
            final_update = atr_update or update
            if final_update:
                actions.append(final_update)
            if (
                self.position
                and not self._fixed_target_exit_disabled()
                and curr.h >= self.position.target_price
            ):
                actions.append(self.close_position(idx, "target_rr", self.position.target_price))
                return actions
        else:
            if curr.h >= pos.sl_price:
                actions.append(self.close_position(idx, "stop_loss", pos.sl_price))
                return actions

            if self.config.enable_funding_oi_trailing:
                funding_snapshot = self._funding_oi_snapshot_for_idx(idx)
                if funding_snapshot is not None:
                    funding_decision = self.funding_oi_overlay.evaluate(funding_snapshot, pos)
                    if funding_decision.action == "exit":
                        actions.append(
                            self.close_position(
                                idx,
                                funding_decision.reason or "funding_oi_exit",
                                funding_decision.exit_price,
                            )
                        )
                        return actions
                    if funding_decision.action == "tighten" and funding_decision.stop_price is not None:
                        if funding_decision.stop_price < pos.sl_price:
                            pos.sl_price = funding_decision.stop_price
                            actions.append(
                                StrategyAction(
                                    type=ActionType.UPDATE_STOP,
                                    timestamp=self._timestamp_for_idx(idx),
                                    direction=pos.direction,
                                    stop_price=funding_decision.stop_price,
                                    reason=funding_decision.reason or "funding_oi_tighten",
                                    metadata=funding_decision.metrics or {},
                                )
                            )

            if self.config.enable_price_band_trailing:
                band_decision = self.price_band_overlay.evaluate(curr, pos)
                if band_decision.action == "exit":
                    actions.append(self.close_position(idx, band_decision.reason or "band_exit", band_decision.exit_price))
                    return actions
                if band_decision.action == "tighten" and band_decision.stop_price is not None:
                    if band_decision.stop_price < pos.sl_price:
                        pos.sl_price = band_decision.stop_price
                        actions.append(
                            StrategyAction(
                                type=ActionType.UPDATE_STOP,
                                timestamp=self._timestamp_for_idx(idx),
                                direction=pos.direction,
                                stop_price=band_decision.stop_price,
                                reason=band_decision.reason or "band_tighten",
                                metadata=band_decision.metrics or {},
                            )
                        )

            update = self._apply_trailing_bear(pos, curr, idx)
            atr_update = self._apply_atr_trailing_bear(pos, curr, idx)
            final_update = atr_update or update
            if final_update:
                actions.append(final_update)
            if (
                self.position
                and not self._fixed_target_exit_disabled()
                and curr.l <= self.position.target_price
            ):
                actions.append(self.close_position(idx, "target_rr", self.position.target_price))
                return actions

        return actions

    def _apply_trailing_bull(self, pos: PositionState, curr: Candle, idx: int) -> StrategyAction | None:
        pnl = pos.quantity * (curr.c - pos.entry_price)
        stage = pos.stage
        new_stop = None
        new_stage = stage

        risk_price = abs(pos.entry_price - pos.initial_sl_price)
        if risk_price <= 0:
            return None
        templates = {
            "loose": [(2.0, 0.0), (4.0, 1.5)],
            "normal": [(1.0, 0.0), (2.0, 0.5), (3.0, 1.0)],
            "tight": [(0.75, 0.25), (1.5, 0.75)],
        }
        template = templates.get(pos.trail_style, templates["normal"])
        for new_stage_candidate, (trigger_r, lock_r) in enumerate(template):
            if pnl >= trigger_r * pos.risk_amount and stage < new_stage_candidate:
                new_stop = pos.entry_price + risk_price * lock_r
                new_stage = new_stage_candidate

        if new_stop is None:
            return None
        pos.sl_price = new_stop
        pos.stage = new_stage
        target_update = self._maybe_cap_target_price(pos)
        return StrategyAction(
            type=ActionType.UPDATE_STOP,
            timestamp=self._timestamp_for_idx(idx),
            direction=pos.direction,
            stop_price=new_stop,
            reason=f"trail_stage_{new_stage}",
            target_price=target_update,
            metadata={"index": idx, "target_price": target_update, "target_rr": pos.target_rr},
        )

    def _apply_trailing_bear(self, pos: PositionState, curr: Candle, idx: int) -> StrategyAction | None:
        pnl = pos.quantity * (pos.entry_price - curr.c)
        stage = pos.stage
        new_stop = None
        new_stage = stage

        risk_price = abs(pos.entry_price - pos.initial_sl_price)
        if risk_price <= 0:
            return None
        templates = {
            "loose": [(2.0, 0.0), (4.0, 1.5)],
            "normal": [(1.0, 0.0), (2.0, 0.5), (3.0, 1.0)],
            "tight": [(0.75, 0.25), (1.5, 0.75)],
        }
        template = templates.get(pos.trail_style, templates["normal"])
        for new_stage_candidate, (trigger_r, lock_r) in enumerate(template):
            if pnl >= trigger_r * pos.risk_amount and stage < new_stage_candidate:
                new_stop = pos.entry_price - risk_price * lock_r
                new_stage = new_stage_candidate

        if new_stop is None:
            return None
        pos.sl_price = new_stop
        pos.stage = new_stage
        target_update = self._maybe_cap_target_price(pos)
        return StrategyAction(
            type=ActionType.UPDATE_STOP,
            timestamp=self._timestamp_for_idx(idx),
            direction=pos.direction,
            stop_price=new_stop,
            reason=f"trail_stage_{new_stage}",
            target_price=target_update,
            metadata={"index": idx, "target_price": target_update, "target_rr": pos.target_rr},
        )

    def _target_rr_cap_for_style(self, trail_style: str) -> float | None:
        if trail_style == "loose":
            return self.config.loose_target_rr_cap
        if trail_style == "tight":
            return self.config.tight_target_rr_cap
        return self.config.normal_target_rr_cap

    def _target_price_from_rr(self, entry_price: float, sl_price: float, direction: str, target_rr: float) -> float:
        risk_price = abs(entry_price - sl_price)
        if direction == Direction.BULL:
            return entry_price + risk_price * target_rr
        return entry_price - risk_price * target_rr

    def _maybe_cap_target_price(self, pos: PositionState) -> float | None:
        if not self.config.enable_target_rr_cap:
            return None
        cap_rr = self._target_rr_cap_for_style(pos.trail_style)
        if cap_rr is None or pos.target_rr <= cap_rr:
            return None
        pos.target_rr = cap_rr
        pos.target_price = self._target_price_from_rr(pos.entry_price, pos.initial_sl_price, pos.direction, cap_rr)
        return pos.target_price

    def _fixed_target_exit_disabled(self) -> bool:
        return bool(self.config.disable_fixed_target or self.config.disable_fixed_target_exit)

    def _funding_oi_snapshot_for_idx(self, idx: int) -> FundingOISnapshot | None:
        if not self._funding_oi_timestamps:
            return None
        timestamp_ms = int(self.c15m[idx].ts * 1000)
        pos = bisect_right(self._funding_oi_timestamps, timestamp_ms) - 1
        if pos < 0:
            return None
        return self._funding_oi_snapshots[pos]

    def _compute_atr_series(self, period: int) -> list[float]:
        if not self.c15m:
            return []
        effective_period = max(int(period or 1), 1)
        atr_values: list[float] = []
        tr_values: list[float] = []
        atr = 0.0
        for idx, candle in enumerate(self.c15m):
            prev_close = self.c15m[idx - 1].c if idx > 0 else candle.c
            tr = max(candle.h - candle.l, abs(candle.h - prev_close), abs(candle.l - prev_close))
            tr_values.append(tr)
            if idx == 0:
                atr = tr
            elif idx < effective_period:
                atr = sum(tr_values) / len(tr_values)
            elif idx == effective_period:
                atr = sum(tr_values[-effective_period:]) / effective_period
            else:
                atr = ((atr * (effective_period - 1)) + tr) / effective_period
            atr_values.append(atr)
        return atr_values

    def _atr_for_idx(self, idx: int) -> float:
        if idx < 0 or idx >= len(self._atr_15m):
            return 0.0
        return self._atr_15m[idx]

    def _unrealized_rr(self, pos: PositionState, price: float) -> float:
        if pos.risk_amount <= 0:
            return 0.0
        if pos.direction == Direction.BULL:
            pnl = pos.quantity * (price - pos.entry_price)
        else:
            pnl = pos.quantity * (pos.entry_price - price)
        return pnl / pos.risk_amount

    def _price_extrema_since_entry(self, pos: PositionState, idx: int) -> tuple[float, float]:
        start_idx = max(0, min(pos.entry_idx, idx))
        window = self.c15m[start_idx : idx + 1]
        highest_price = max(candle.h for candle in window)
        lowest_price = min(candle.l for candle in window)
        return highest_price, lowest_price

    def _atr_multiplier_for_style(self, trail_style: str) -> float:
        if trail_style == "loose":
            return self.config.atr_loose_multiplier
        if trail_style == "tight":
            return self.config.atr_tight_multiplier
        return self.config.atr_normal_multiplier

    def _atr_trailing_enabled_for_position(self, pos: PositionState) -> bool:
        if not self.config.enable_atr_trailing:
            return False
        regime_filter = self.config.atr_regime_filter
        if regime_filter == "all":
            return True
        entry_regime = self._risk_regime_for_idx(pos.entry_idx)
        if regime_filter == "tight_style_off":
            return pos.trail_style != "tight"
        if regime_filter == "bull_weak_off":
            return entry_regime != "bull_weak"
        if regime_filter == "bear_weak_off":
            return entry_regime != "bear_weak"
        raise ValueError(f"Unsupported ATR regime filter: {regime_filter}")

    def _apply_atr_trailing_bull(self, pos: PositionState, curr: Candle, idx: int) -> StrategyAction | None:
        if not self._atr_trailing_enabled_for_position(pos):
            return None
        if self._unrealized_rr(pos, curr.c) < self.config.atr_activation_rr:
            return None
        atr = self._atr_for_idx(idx)
        if atr <= 0:
            return None
        highest_price, _ = self._price_extrema_since_entry(pos, idx)
        new_stop = highest_price - self._atr_multiplier_for_style(pos.trail_style) * atr
        if new_stop <= pos.sl_price:
            return None
        pos.sl_price = new_stop
        return StrategyAction(
            type=ActionType.UPDATE_STOP,
            timestamp=self._timestamp_for_idx(idx),
            direction=pos.direction,
            stop_price=new_stop,
            reason="atr_trail",
            metadata={"index": idx, "atr": atr, "highest_price": highest_price},
        )

    def _apply_atr_trailing_bear(self, pos: PositionState, curr: Candle, idx: int) -> StrategyAction | None:
        if not self._atr_trailing_enabled_for_position(pos):
            return None
        if self._unrealized_rr(pos, curr.c) < self.config.atr_activation_rr:
            return None
        atr = self._atr_for_idx(idx)
        if atr <= 0:
            return None
        _, lowest_price = self._price_extrema_since_entry(pos, idx)
        new_stop = lowest_price + self._atr_multiplier_for_style(pos.trail_style) * atr
        if new_stop >= pos.sl_price:
            return None
        pos.sl_price = new_stop
        return StrategyAction(
            type=ActionType.UPDATE_STOP,
            timestamp=self._timestamp_for_idx(idx),
            direction=pos.direction,
            stop_price=new_stop,
            reason="atr_trail",
            metadata={"index": idx, "atr": atr, "lowest_price": lowest_price},
        )

    def run_backtest(self, start_date: str = "2023-01-01") -> dict[str, Any]:
        start_dt = datetime.fromisoformat(start_date)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        else:
            start_dt = start_dt.astimezone(timezone.utc)
        start_ts = start_dt.timestamp()
        start_idx = next((i for i, candle in enumerate(self.c15m) if candle.ts >= start_ts), 0)

        self.evaluate_range(start_idx + 100, len(self.c15m) - 1)

        if self.position:
            self.close_position(len(self.c15m) - 1, "end_of_data")
        return self.compute_metrics()

    def evaluate_range(self, start_idx: int, end_idx: int) -> list[StrategyAction]:
        waiting_for_pullback = False
        bos_idx = -1
        ob_zone: dict[str, float] | None = None
        waiting_direction: str | None = None
        waiting_pullback_window = self.config.pullback_window
        pending_by_direction: dict[str, PendingPullback | None] = {
            Direction.BULL: None,
            Direction.BEAR: None,
        }
        actions: list[StrategyAction] = []

        for i in range(start_idx, end_idx):
            self._apply_regime_switch_for_idx(i)
            if self.position:
                position_actions = self.manage_position(i)
                actions.extend(position_actions)
                if self.position is None:
                    continue

            bias = self.precomputed.bias_4h[self.mapping[i]]

            active_pending = waiting_for_pullback or any(pending_by_direction.values())
            if self.config.use_hfvf_filter and bias == Direction.NONE and not active_pending:
                continue

            if self.config.enable_dual_pending_state:
                if not self.position:
                    for direction in (Direction.BULL, Direction.BEAR):
                        pending = pending_by_direction[direction]
                        if pending is None:
                            continue
                        action = self._open_action_from_pending(i, pending)
                        if action:
                            actions.append(action)
                            pending_by_direction[Direction.BULL] = None
                            pending_by_direction[Direction.BEAR] = None
                            break
                        if self._pending_expired(i, pending):
                            pending_by_direction[direction] = None
                if self.position:
                    continue
                if i in (self.precomputed.highs_set | self.precomputed.lows_set):
                    pending = self._build_pending_pullback(i, bias)
                    if pending and pending_by_direction[pending.direction] is None:
                        pending_by_direction[pending.direction] = pending
                continue

            if waiting_for_pullback and ob_zone and waiting_direction:
                pending = PendingPullback(
                    direction=waiting_direction,
                    bos_idx=bos_idx,
                    ob_zone=ob_zone,
                    pullback_window=waiting_pullback_window,
                )
                action = self._open_action_from_pending(i, pending)
                if action:
                    actions.append(action)
                    waiting_for_pullback = False
                    ob_zone = None
                    waiting_direction = None
                    continue
                if self._pending_expired(i, pending):
                    waiting_for_pullback = False
                    ob_zone = None
                    waiting_direction = None
                    continue

            if not self.position and not waiting_for_pullback and i in (self.precomputed.highs_set | self.precomputed.lows_set):
                pending = self._build_pending_pullback(i, bias)
                if pending:
                    waiting_for_pullback = True
                    bos_idx = pending.bos_idx
                    ob_zone = pending.ob_zone
                    waiting_direction = pending.direction
                    waiting_pullback_window = pending.pullback_window

        return actions

    def _effective_regime_history(self, idx: int) -> list[Candle]:
        if idx < 0 or idx >= len(self.mapping):
            return []
        c4h_idx = self.mapping[idx]
        if c4h_idx <= 0:
            return []
        return self.c4h[:c4h_idx]

    def _regime_switch_overrides(self, regime: str) -> dict[str, Any]:
        if regime == "high_growth":
            return dict(self._base_config.regime_switcher_hg_overrides or {})
        if regime == "flat":
            return dict(self._base_config.regime_switcher_flat_overrides or {})
        return dict(self._base_config.regime_switcher_normal_overrides or {})

    def _config_for_regime(self, regime: str) -> StrategyConfig:
        config_copy = deepcopy(self._base_config)
        for field_name, value in self._regime_switch_overrides(regime).items():
            if hasattr(config_copy, field_name):
                setattr(config_copy, field_name, value)
        return config_copy

    def _regime_label_for_idx(self, idx: int) -> str:
        history = self._effective_regime_history(idx)
        if not history:
            return "flat"
        try:
            from scripts.regime_detector import detect_regime

            thresholds = self._base_config.regime_switcher_thresholds
            return detect_regime(history, thresholds)
        except Exception:
            return "flat"

    def _apply_regime_switch_for_idx(self, idx: int) -> str:
        if not self._base_config.enable_regime_switching:
            self.config = self._base_config
            return "static"
        c4h_idx = self.mapping[idx]
        cached = self._regime_switch_cache.get(c4h_idx)
        if cached is None:
            regime = self._regime_label_for_idx(idx)
            cached = (regime, self._config_for_regime(regime))
            self._regime_switch_cache[c4h_idx] = cached
        self.config = cached[1]
        return cached[0]

    def compute_metrics(self) -> dict[str, Any]:
        if not self.trades:
            return {"total_trades": 0, "total_return_pct": 0}
        wins = [trade for trade in self.trades if trade.pnl > 0]
        losses = [trade for trade in self.trades if trade.pnl <= 0]
        total = len(self.trades)
        win_rate = len(wins) / total * 100 if total > 0 else 0.0
        gross_profit = sum(trade.pnl for trade in wins)
        gross_loss = abs(sum(trade.pnl for trade in losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else 0.0
        total_return = (self.capital - self.config.initial_capital) / self.config.initial_capital * 100
        if len(self.trades) > 1:
            returns = [trade.pnl_pct for trade in self.trades]
            mean_r = sum(returns) / len(returns)
            std_r = math.sqrt(sum((value - mean_r) ** 2 for value in returns) / len(returns))
            sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
        else:
            sharpe = 0.0
        peak = self.config.initial_capital
        max_dd = 0.0
        run_cap = self.config.initial_capital
        for trade in self.trades:
            run_cap += trade.pnl
            if run_cap > peak:
                peak = run_cap
            dd = (peak - run_cap) / peak * 100
            if dd > max_dd:
                max_dd = dd
        target_hits = sum(1 for trade in self.trades if "target" in trade.exit_reason)
        target_hit_rate = target_hits / total * 100 if total > 0 else 0.0
        avg_win = gross_profit / len(wins) if wins else 0.0
        avg_loss = gross_loss / len(losses) if losses else 0.0
        wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        total_fees = sum(trade.fees for trade in self.trades)
        total_slippage_cost = sum(trade.slippage_cost for trade in self.trades)
        gross_pnl_before_fees = sum(trade.gross_pnl for trade in self.trades)
        return {
            "initial_capital": self.config.initial_capital,
            "final_capital": self.capital,
            "total_return_pct": total_return,
            "max_drawdown_pct": max_dd,
            "sharpe_ratio": sharpe,
            "risk_adjusted_return": total_return / max_dd if max_dd > 0 else 0.0,
            "total_trades": total,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "profit_factor": pf,
            "wl_ratio": wl_ratio,
            "target_hit_rate": target_hit_rate,
            "gross_pnl_before_fees": gross_pnl_before_fees,
            "total_fees_paid": total_fees,
            "total_slippage_cost": total_slippage_cost,
            "exit_reasons": dict(self.exit_reasons),
            "parameters": {
                "rr_ratio": self.config.rr_ratio,
                "trailing_config": self.config.trailing_config,
                "sl_buffer_pct": self.config.sl_buffer_pct,
                "use_hfvf_filter": self.config.use_hfvf_filter,
                "leverage": self.config.leverage,
                "position_size_pct": self.config.position_size_pct,
                "fixed_notional_usdt": self.config.fixed_notional_usdt,
                "allow_long": self.config.allow_long,
                "allow_short": self.config.allow_short,
                "regime_filter_1d_ema_period": self.config.regime_filter_1d_ema_period,
                "regime_filter_1d_direction": self.config.regime_filter_1d_direction,
                "enable_directional_regime_switch": self.config.enable_directional_regime_switch,
                "long_regime_filter_1d_ema_period": self.config.long_regime_filter_1d_ema_period,
                "short_regime_filter_1d_ema_period": self.config.short_regime_filter_1d_ema_period,
                "enable_dual_pending_state": self.config.enable_dual_pending_state,
                "regime_position_size_pct_below_ema": self.config.regime_position_size_pct_below_ema,
                "enable_regime_layered_exit": self.config.enable_regime_layered_exit,
                "enable_short_regime_layered_exit": self.config.enable_short_regime_layered_exit,
                "short_pullback_window": self.config.short_pullback_window,
                "short_sl_buffer_pct": self.config.short_sl_buffer_pct,
                "short_strong_rr_ratio": self.config.short_strong_rr_ratio,
                "short_mid_rr_ratio": self.config.short_mid_rr_ratio,
                "short_weak_rr_ratio": self.config.short_weak_rr_ratio,
                "enable_target_rr_cap": self.config.enable_target_rr_cap,
                "loose_target_rr_cap": self.config.loose_target_rr_cap,
                "normal_target_rr_cap": self.config.normal_target_rr_cap,
                "tight_target_rr_cap": self.config.tight_target_rr_cap,
                "enable_price_band_trailing": self.config.enable_price_band_trailing,
                "disable_fixed_target": self.config.disable_fixed_target,
                "enable_funding_oi_trailing": self.config.enable_funding_oi_trailing,
                "enable_atr_trailing": self.config.enable_atr_trailing,
                "atr_period": self.config.atr_period,
                "atr_activation_rr": self.config.atr_activation_rr,
                "atr_loose_multiplier": self.config.atr_loose_multiplier,
                "atr_normal_multiplier": self.config.atr_normal_multiplier,
                "atr_tight_multiplier": self.config.atr_tight_multiplier,
                "atr_regime_filter": self.config.atr_regime_filter,
                "disable_fixed_target_exit": self.config.disable_fixed_target_exit,
                "taker_fee_rate": self.config.taker_fee_rate,
                "slippage_bps": self.config.slippage_bps,
            },
        }

    def _timestamp_for_idx(self, idx: int) -> str:
        return datetime.fromtimestamp(self.c15m[idx].ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    def _apply_entry_slippage(self, price: float, direction: str) -> float:
        slip = self.config.slippage_bps / 10_000.0
        if direction == Direction.BULL:
            return price * (1 + slip)
        return price * (1 - slip)

    def _apply_exit_slippage(self, price: float, direction: str) -> float:
        slip = self.config.slippage_bps / 10_000.0
        if direction == Direction.BULL:
            return price * (1 - slip)
        return price * (1 + slip)

    def _regime_ok_for_idx(self, idx: int) -> bool:
        period = self.config.regime_filter_1d_ema_period
        if period is None:
            return True
        mapped_idx = self.mapping[idx]
        direction = self.config.regime_filter_1d_direction
        if period == 100 and direction == "bull":
            return self.precomputed.regime_1d_bull_100[mapped_idx]
        if period == 200 and direction == "bull":
            return self.precomputed.regime_1d_bull_200[mapped_idx]
        if period == 100 and direction == "bear":
            return self.precomputed.regime_1d_bear_100[mapped_idx]
        if period == 200 and direction == "bear":
            return self.precomputed.regime_1d_bear_200[mapped_idx]
        raise ValueError(f"Unsupported regime EMA period: {period}")

    def _regime_ok_for_direction_idx(self, idx: int, direction: str) -> bool:
        if not self.config.enable_directional_regime_switch:
            period = self.config.regime_filter_1d_ema_period
            if period is None:
                return True
            return self._regime_ok_for_idx(idx)
        period = (
            self.config.long_regime_filter_1d_ema_period
            if direction == Direction.BULL
            else self.config.short_regime_filter_1d_ema_period
        )
        if period is None:
            period = self.config.regime_filter_1d_ema_period
        if period is None:
            return True
        mapped_idx = self.mapping[idx]
        if period == 100 and direction == Direction.BULL:
            return self.precomputed.regime_1d_bull_100[mapped_idx]
        if period == 200 and direction == Direction.BULL:
            return self.precomputed.regime_1d_bull_200[mapped_idx]
        if period == 100 and direction == Direction.BEAR:
            return self.precomputed.regime_1d_bear_100[mapped_idx]
        if period == 200 and direction == Direction.BEAR:
            return self.precomputed.regime_1d_bear_200[mapped_idx]
        raise ValueError(f"Unsupported regime EMA period: {period}")

    def _position_size_pct_for_idx(self, idx: int) -> float:
        if self.config.fixed_notional_usdt is not None:
            return self.config.position_size_pct
        if self.config.enable_directional_regime_switch:
            return self.config.position_size_pct
        if self.config.regime_filter_1d_ema_period is None:
            return self.config.position_size_pct
        if self._regime_ok_for_idx(idx):
            return self.config.position_size_pct
        if self.config.regime_position_size_pct_below_ema is not None:
            return self.config.regime_position_size_pct_below_ema
        return self.config.position_size_pct

    def _pullback_window_for_direction(self, direction: str) -> int:
        if direction == Direction.BEAR and self.config.short_pullback_window is not None:
            return self.config.short_pullback_window
        return self.config.pullback_window

    def _sl_buffer_pct_for_direction(self, direction: str) -> float:
        if direction == Direction.BEAR and self.config.short_sl_buffer_pct is not None:
            return self.config.short_sl_buffer_pct
        return self.config.sl_buffer_pct

    def _open_action_from_pending(self, idx: int, pending: PendingPullback) -> StrategyAction | None:
        curr = self.c15m[idx]
        direction = pending.direction
        in_ob = pending.ob_zone["bottom"] <= curr.l <= pending.ob_zone["top"] or pending.ob_zone["bottom"] <= curr.h <= pending.ob_zone["top"]
        direction_allowed = (
            direction == Direction.BULL
            and self.config.allow_long
            and self._regime_ok_for_direction_idx(idx, Direction.BULL)
        ) or (
            direction == Direction.BEAR
            and self.config.allow_short
            and self._regime_ok_for_direction_idx(idx, Direction.BEAR)
        )
        if not direction_allowed or not in_ob:
            return None
        if direction == Direction.BULL and curr.c <= curr.o:
            return None
        if direction == Direction.BEAR and curr.c >= curr.o:
            return None
        entry = curr.c
        sl_raw = pending.ob_zone["bottom"] if direction == Direction.BULL else pending.ob_zone["top"]
        sl_buffer_pct = self._sl_buffer_pct_for_direction(direction)
        sl = (
            sl_raw * (1 - sl_buffer_pct / 100)
            if direction == Direction.BULL
            else sl_raw * (1 + sl_buffer_pct / 100)
        )
        risk = abs(entry - sl)
        _, target_rr, _, _ = self._exit_template_for_idx(idx, direction)
        tp = entry + risk * target_rr if direction == Direction.BULL else entry - risk * target_rr
        return self.open_position(idx, direction, entry, sl, tp)

    def _pending_expired(self, idx: int, pending: PendingPullback) -> bool:
        return idx - pending.bos_idx > pending.pullback_window

    def _build_pending_pullback(self, idx: int, bias: str) -> PendingPullback | None:
        if bias == Direction.BEAR:
            if not (self.config.allow_short and self._regime_ok_for_direction_idx(idx, Direction.BEAR)):
                return None
            if not (self.precomputed.broken_bear[idx] and self.precomputed.reclaimed_bear[idx]):
                return None
            prev_lows = [low for low in self.precomputed.lows_15m if low < idx]
            if len(prev_lows) <= 1:
                return None
            next_low = self.c15m[prev_lows[-2]].l
            for j in range(max(0, idx - 40), idx):
                if self.c15m[j].l < next_low:
                    ob = find_ob(self.c15m, idx, bias)
                    if ob:
                        return PendingPullback(
                            direction=Direction.BEAR,
                            bos_idx=j,
                            ob_zone=ob,
                            pullback_window=self._pullback_window_for_direction(Direction.BEAR),
                        )
                    return None
            return None

        if bias == Direction.BULL:
            if not (self.config.allow_long and self._regime_ok_for_direction_idx(idx, Direction.BULL)):
                return None
            if not (self.precomputed.broken_bull[idx] and self.precomputed.reclaimed_bull[idx]):
                return None
            prev_highs = [high for high in self.precomputed.highs_15m if high < idx]
            if not prev_highs:
                return None
            next_highs_list = [high for high in self.precomputed.highs_15m if high < prev_highs[-1]]
            if not next_highs_list:
                return None
            next_high = self.c15m[next_highs_list[-1]].h
            for j in range(max(0, idx - 40), idx):
                if self.c15m[j].h > next_high:
                    ob = find_ob(self.c15m, idx, bias)
                    if ob:
                        return PendingPullback(
                            direction=Direction.BULL,
                            bos_idx=j,
                            ob_zone=ob,
                            pullback_window=self._pullback_window_for_direction(Direction.BULL),
                        )
                    return None
            return None

        return None

    def _btc_bull_trend_score_for_idx(self, idx: int) -> int:
        mapped_idx = self.mapping[idx]
        return self.precomputed.bull_trend_score_4h[mapped_idx]

    def _btc_bear_trend_score_for_idx(self, idx: int) -> int:
        mapped_idx = self.mapping[idx]
        return self.precomputed.bear_trend_score_4h[mapped_idx]

    def _risk_regime_for_idx(self, idx: int) -> str:
        mapped_idx = self.mapping[idx]
        bull_200 = self.precomputed.regime_1d_bull_200[mapped_idx]
        bear_200 = self.precomputed.regime_1d_bear_200[mapped_idx]
        bull_score = self.precomputed.bull_trend_score_4h[mapped_idx]
        bear_score = self.precomputed.bear_trend_score_4h[mapped_idx]

        if bull_200 and bull_score >= 3:
            return "bull_strong"
        if bull_200:
            return "bull_weak"
        if bear_200 and bear_score >= 3:
            return "bear_strong"
        return "bear_weak"

    def _risk_per_trade_for_idx(self, idx: int, direction: str) -> tuple[float, str]:
        if not self.config.enable_regime_directional_risk:
            return self.config.risk_per_trade, "fixed"

        regime = self._risk_regime_for_idx(idx)
        field_map = {
            ("bull_strong", Direction.BULL): self.config.bull_strong_long_risk_per_trade,
            ("bull_strong", Direction.BEAR): self.config.bull_strong_short_risk_per_trade,
            ("bull_weak", Direction.BULL): self.config.bull_weak_long_risk_per_trade,
            ("bull_weak", Direction.BEAR): self.config.bull_weak_short_risk_per_trade,
            ("bear_weak", Direction.BULL): self.config.bear_weak_long_risk_per_trade,
            ("bear_weak", Direction.BEAR): self.config.bear_weak_short_risk_per_trade,
            ("bear_strong", Direction.BULL): self.config.bear_strong_long_risk_per_trade,
            ("bear_strong", Direction.BEAR): self.config.bear_strong_short_risk_per_trade,
        }
        value = field_map.get((regime, direction))
        if value is None:
            return self.config.risk_per_trade, regime
        return value, regime

    def _exit_template_for_idx(self, idx: int, direction: str) -> tuple[int, float, int | None, str]:
        regime = self._risk_regime_for_idx(idx)
        if direction == Direction.BULL:
            score = self._btc_bull_trend_score_for_idx(idx)
            if not self.config.enable_regime_layered_exit:
                rr = self.config.rr_ratio
                style = "normal"
            elif score >= 3:
                rr = 5.5
                style = "loose"
            elif score >= 2:
                rr = 5.0
                style = "normal"
            else:
                rr = 3.0
                style = "tight"

            if regime == "bull_weak":
                rr = self.config.bull_weak_long_rr_ratio_override or rr
                style = self.config.bull_weak_long_trail_style_override or style
            return score, rr, None, style

        score = self._btc_bear_trend_score_for_idx(idx)
        if not self.config.enable_short_regime_layered_exit:
            rr = self.config.rr_ratio
            style = "normal"
        elif score >= 3:
            rr = self.config.short_strong_rr_ratio
            style = "loose"
        elif score >= 2:
            rr = self.config.short_mid_rr_ratio
            style = "normal"
        else:
            rr = self.config.short_weak_rr_ratio
            style = "tight"

        if regime == "bear_weak":
            rr = self.config.bear_weak_short_rr_ratio_override or rr
            style = self.config.bear_weak_short_trail_style_override or style
        return score, rr, None, style
