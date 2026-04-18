from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from strategy.scalp_robust_v2_core import (
    ActionType,
    Candle,
    Direction,
    StrategyAction,
    StrategySnapshot,
)


@dataclass
class EmaCrossVolumeConfig:
    leverage: float = 3.0
    risk_per_trade: float = 0.01
    position_size_pct: float = 0.35
    fixed_notional_usdt: float | None = None
    allow_long: bool = True
    allow_short: bool = True
    ema_fast_period: int = 9
    ema_slow_period: int = 21
    volume_ma_period: int = 20
    volume_multiplier: float = 1.2
    taker_fee_rate: float = 0.0005
    slippage_bps: float = 2.0
    initial_capital: float = 1000.0


@dataclass
class EmaCrossPositionState:
    direction: str
    signal_entry_price: float
    entry_price: float
    entry_time: str
    capital_at_entry: float
    risk_amount: float
    notional: float
    quantity: float
    entry_fee: float
    entry_slippage_cost: float
    entry_idx: int


@dataclass
class EmaCrossTrade:
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
    exit_reason: str
    capital_at_entry: float


class EmaCrossVolumeEngine:
    def __init__(self, candles: list[Candle], config: EmaCrossVolumeConfig | None = None):
        self.candles = candles
        self.config = config or EmaCrossVolumeConfig()
        self.capital = self.config.initial_capital
        self.position: EmaCrossPositionState | None = None
        self.trades: list[EmaCrossTrade] = []
        self.restored_trade_count = 0
        self.exit_reasons: dict[str, int] = {}
        self._ema_fast = self._compute_ema_series(self.config.ema_fast_period)
        self._ema_slow = self._compute_ema_series(self.config.ema_slow_period)
        self._volume_ma = self._compute_volume_ma(self.config.volume_ma_period)

    @classmethod
    def from_candles(
        cls,
        primary_candles: list[Candle],
        informative_candles: list[Candle] | None = None,
        config: EmaCrossVolumeConfig | None = None,
    ) -> "EmaCrossVolumeEngine":
        return cls(primary_candles, config)

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
        self.position = EmaCrossPositionState(**position_payload) if isinstance(position_payload, dict) else None
        exit_reasons = snapshot.get("exit_reasons")
        self.exit_reasons = dict(exit_reasons) if isinstance(exit_reasons, dict) else {}
        try:
            self.restored_trade_count = int(snapshot.get("trade_count", 0) or 0)
        except (TypeError, ValueError):
            self.restored_trade_count = 0

    def evaluate_range(self, start_idx: int, end_idx: int) -> list[StrategyAction]:
        actions: list[StrategyAction] = []
        if len(self.candles) < 3:
            return actions

        safe_start = max(start_idx, max(self.config.ema_slow_period, self.config.volume_ma_period) + 1)
        safe_end = min(end_idx, len(self.candles) - 1)

        for idx in range(safe_start, safe_end + 1):
            cross_up = self._is_cross_up(idx)
            cross_down = self._is_cross_down(idx)

            if self.position:
                if self.position.direction == Direction.BULL and cross_down:
                    actions.append(self.close_position(idx, "ema_cross_down"))
                elif self.position.direction == Direction.BEAR and cross_up:
                    actions.append(self.close_position(idx, "ema_cross_up"))

            if self.position:
                continue

            if not self._volume_ok(idx):
                continue

            if cross_up and self.config.allow_long:
                actions.append(self.open_position(idx, Direction.BULL))
            elif cross_down and self.config.allow_short:
                actions.append(self.open_position(idx, Direction.BEAR))

        return actions

    def open_position(self, idx: int, direction: str) -> StrategyAction:
        candle = self.candles[idx]
        signal_entry_price = candle.c
        filled_entry_price = self._apply_entry_slippage(signal_entry_price, direction)
        risk_amount = self.capital * self.config.risk_per_trade
        max_notional = (
            float(self.config.fixed_notional_usdt)
            if self.config.fixed_notional_usdt is not None
            else self.capital * self.config.position_size_pct * self.config.leverage
        )
        quantity = max_notional / filled_entry_price if filled_entry_price > 0 else 0.0
        entry_fee = max_notional * self.config.taker_fee_rate
        entry_slippage_cost = quantity * abs(filled_entry_price - signal_entry_price)
        timestamp = self._timestamp_for_idx(idx)
        self.position = EmaCrossPositionState(
            direction=direction,
            signal_entry_price=signal_entry_price,
            entry_price=filled_entry_price,
            entry_time=timestamp,
            capital_at_entry=self.capital,
            risk_amount=risk_amount,
            notional=max_notional,
            quantity=quantity,
            entry_fee=entry_fee,
            entry_slippage_cost=entry_slippage_cost,
            entry_idx=idx,
        )
        return StrategyAction(
            type=ActionType.OPEN_LONG if direction == Direction.BULL else ActionType.OPEN_SHORT,
            timestamp=timestamp,
            direction=direction,
            entry_price=filled_entry_price,
            reason="ema_cross",
            metadata={
                "index": idx,
                "signal_entry_price": signal_entry_price,
                "ema_fast": self._ema_fast[idx],
                "ema_slow": self._ema_slow[idx],
                "volume": candle.v,
                "volume_ma": self._volume_ma[idx],
                "volume_multiplier": self.config.volume_multiplier,
                "notional": max_notional,
                "entry_fee": entry_fee,
                "entry_slippage_cost": entry_slippage_cost,
            },
        )

    def close_position(self, idx: int, reason: str) -> StrategyAction:
        if not self.position:
            return StrategyAction(type=ActionType.HOLD, timestamp=self._timestamp_for_idx(idx), reason="no_position")
        pos = self.position
        signal_exit_price = self.candles[idx].c
        filled_exit_price = self._apply_exit_slippage(signal_exit_price, pos.direction)
        if pos.direction == Direction.BULL:
            gross_pnl = pos.quantity * (filled_exit_price - pos.entry_price)
        else:
            gross_pnl = pos.quantity * (pos.entry_price - filled_exit_price)
        exit_fee = pos.quantity * filled_exit_price * self.config.taker_fee_rate
        fees = pos.entry_fee + exit_fee
        slippage_cost = pos.entry_slippage_cost + pos.quantity * abs(filled_exit_price - signal_exit_price)
        pnl = gross_pnl - fees
        self.trades.append(
            EmaCrossTrade(
                entry_time=pos.entry_time,
                exit_time=self._timestamp_for_idx(idx),
                direction=pos.direction,
                signal_entry_price=pos.signal_entry_price,
                entry_price=pos.entry_price,
                signal_exit_price=signal_exit_price,
                exit_price=filled_exit_price,
                gross_pnl=gross_pnl,
                fees=fees,
                slippage_cost=slippage_cost,
                pnl=pnl,
                pnl_pct=pnl / pos.capital_at_entry if pos.capital_at_entry > 0 else 0.0,
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
                "signal_exit_price": signal_exit_price,
                "ema_fast": self._ema_fast[idx],
                "ema_slow": self._ema_slow[idx],
                "gross_pnl": gross_pnl,
                "fees": fees,
                "slippage_cost": slippage_cost,
                "net_pnl": pnl,
            },
        )

    def _compute_ema_series(self, period: int) -> list[float]:
        if not self.candles:
            return []
        alpha = 2.0 / (period + 1.0)
        ema_values: list[float] = []
        ema = self.candles[0].c
        for idx, candle in enumerate(self.candles):
            ema = candle.c if idx == 0 else alpha * candle.c + (1 - alpha) * ema
            ema_values.append(ema)
        return ema_values

    def _compute_volume_ma(self, period: int) -> list[float | None]:
        values: list[float | None] = []
        for idx in range(len(self.candles)):
            if idx + 1 < period:
                values.append(None)
                continue
            window = self.candles[idx - period + 1 : idx + 1]
            values.append(sum(candle.v for candle in window) / period)
        return values

    def _is_cross_up(self, idx: int) -> bool:
        if idx <= 0:
            return False
        return self._ema_fast[idx - 1] <= self._ema_slow[idx - 1] and self._ema_fast[idx] > self._ema_slow[idx]

    def _is_cross_down(self, idx: int) -> bool:
        if idx <= 0:
            return False
        return self._ema_fast[idx - 1] >= self._ema_slow[idx - 1] and self._ema_fast[idx] < self._ema_slow[idx]

    def _volume_ok(self, idx: int) -> bool:
        volume_ma = self._volume_ma[idx]
        if volume_ma is None or volume_ma <= 0:
            return False
        return self.candles[idx].v > volume_ma * self.config.volume_multiplier

    def _timestamp_for_idx(self, idx: int) -> str:
        return datetime.fromtimestamp(self.candles[idx].ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

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
