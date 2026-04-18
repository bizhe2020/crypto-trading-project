from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OrderBookLevel:
    price: float
    size: float


@dataclass(frozen=True)
class OrderBookSnapshot:
    inst_id: str
    timestamp_ms: int
    bids: tuple[OrderBookLevel, ...]
    asks: tuple[OrderBookLevel, ...]

    @property
    def mid_price(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / 2.0


@dataclass
class OBIOverlayConfig:
    distance_weights: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)
    min_profit_rr: float = 0.75
    fast_ema_seconds: float = 0.5
    slow_ema_seconds: float = 2.0
    tighten_dwell_seconds: float = 1.5
    tighten_obi_threshold: float = 0.12
    tighten_edge_bps: float = 0.15
    low_lock_rr: float = 0.25
    mid_lock_rr: float = 0.5
    high_lock_rr: float = 0.75
    force_exit_min_profit_rr: float = 1.5
    force_exit_dwell_seconds: float = 2.0
    force_exit_obi_threshold: float = 0.2
    force_exit_edge_bps: float = 0.25


@dataclass
class OBIOverlayDecision:
    action: str
    reason: str
    stop_price: float | None = None
    exit_price: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class OBIOverlayState:
    position_key: str | None = None
    last_timestamp_ms: int | None = None
    obi_fast: float | None = None
    obi_slow: float | None = None
    tighten_active_since_ms: int | None = None
    exit_active_since_ms: int | None = None
    last_emitted_stop_price: float | None = None


class OBIOverlay:
    def __init__(self, config: OBIOverlayConfig | None = None):
        self.config = config or OBIOverlayConfig()
        self.state = OBIOverlayState()

    def reset(self) -> None:
        self.state = OBIOverlayState()

    def evaluate(self, snapshot: OrderBookSnapshot, position: Any) -> OBIOverlayDecision:
        position_key = f"{getattr(position, 'direction', '')}:{getattr(position, 'entry_time', '')}"
        if self.state.position_key != position_key:
            self.reset()
            self.state.position_key = position_key

        mid_price = snapshot.mid_price
        if mid_price is None:
            return OBIOverlayDecision(action="hold", reason="missing_mid_price")

        risk_price = abs(float(getattr(position, "entry_price", 0.0)) - float(getattr(position, "initial_sl_price", 0.0)))
        if risk_price <= 0:
            return OBIOverlayDecision(action="hold", reason="invalid_risk_price")

        profit_rr = self._profit_rr(position, mid_price, risk_price)
        metrics = self._compute_metrics(snapshot)
        obi = metrics["obi"]
        self._update_emas(snapshot.timestamp_ms, obi)
        obi_fast = self.state.obi_fast if self.state.obi_fast is not None else obi
        obi_slow = self.state.obi_slow if self.state.obi_slow is not None else obi
        obi_mom = obi_fast - obi_slow
        edge_bps = metrics["edge_bps"]

        decision_metrics = {
            "obi": obi,
            "obi_fast": obi_fast,
            "obi_slow": obi_slow,
            "obi_mom": obi_mom,
            "edge_bps": edge_bps,
            "spread_bps": metrics["spread_bps"],
            "top_depth": metrics["top_depth"],
            "profit_rr": profit_rr,
            "snapshot_timestamp_ms": snapshot.timestamp_ms,
        }

        if profit_rr < self.config.min_profit_rr:
            self.state.tighten_active_since_ms = None
            self.state.exit_active_since_ms = None
            return OBIOverlayDecision(action="hold", reason="below_min_profit_rr", metrics=decision_metrics)

        adverse = self._is_adverse(position, obi, obi_mom, edge_bps)
        if not adverse:
            self.state.tighten_active_since_ms = None
            self.state.exit_active_since_ms = None
            return OBIOverlayDecision(action="hold", reason="no_adverse_microstructure", metrics=decision_metrics)

        force_exit_adverse = self._is_force_exit_adverse(position, profit_rr, obi, obi_mom, edge_bps)
        if force_exit_adverse:
            if self.state.exit_active_since_ms is None:
                self.state.exit_active_since_ms = snapshot.timestamp_ms
            exit_active_ms = snapshot.timestamp_ms - self.state.exit_active_since_ms
            decision_metrics["force_exit_active_ms"] = exit_active_ms
            if exit_active_ms >= int(self.config.force_exit_dwell_seconds * 1000):
                self.state.tighten_active_since_ms = None
                return OBIOverlayDecision(
                    action="exit",
                    reason="obi_force_exit",
                    exit_price=mid_price,
                    metrics=decision_metrics,
                )
        else:
            self.state.exit_active_since_ms = None

        if self.state.tighten_active_since_ms is None:
            self.state.tighten_active_since_ms = snapshot.timestamp_ms
            return OBIOverlayDecision(action="hold", reason="tighten_dwell_started", metrics=decision_metrics)

        active_ms = snapshot.timestamp_ms - self.state.tighten_active_since_ms
        decision_metrics["adverse_active_ms"] = active_ms
        if active_ms < int(self.config.tighten_dwell_seconds * 1000):
            return OBIOverlayDecision(action="hold", reason="tighten_dwell_pending", metrics=decision_metrics)

        lock_rr = self._lock_rr_for_profit(profit_rr)
        suggested_stop = self._stop_price_from_lock_rr(position, risk_price, lock_rr)
        current_stop = float(getattr(position, "sl_price", 0.0) or 0.0)
        if not self._improves_stop(position, current_stop, suggested_stop):
            return OBIOverlayDecision(action="hold", reason="no_stop_improvement", metrics=decision_metrics)

        if self.state.last_emitted_stop_price is not None and not self._improves_stop(
            position,
            self.state.last_emitted_stop_price,
            suggested_stop,
        ):
            return OBIOverlayDecision(action="hold", reason="already_emitted_stop", metrics=decision_metrics)

        self.state.last_emitted_stop_price = suggested_stop
        return OBIOverlayDecision(
            action="tighten",
            reason="obi_tighten_stop",
            stop_price=suggested_stop,
            metrics={**decision_metrics, "lock_rr": lock_rr},
        )

    def _compute_metrics(self, snapshot: OrderBookSnapshot) -> dict[str, float]:
        weights = self.config.distance_weights
        weighted_bid = self._weighted_depth(snapshot.bids, weights)
        weighted_ask = self._weighted_depth(snapshot.asks, weights)
        total_depth = weighted_bid + weighted_ask
        obi = 0.0 if total_depth <= 0 else (weighted_bid - weighted_ask) / total_depth
        spread_bps = 0.0
        edge_bps = 0.0
        mid_price = snapshot.mid_price
        if mid_price and snapshot.bids and snapshot.asks:
            spread_bps = (snapshot.asks[0].price - snapshot.bids[0].price) / mid_price * 10_000
            top_bid_size = snapshot.bids[0].size
            top_ask_size = snapshot.asks[0].size
            if top_bid_size + top_ask_size > 0:
                microprice = (
                    snapshot.asks[0].price * top_bid_size + snapshot.bids[0].price * top_ask_size
                ) / (top_bid_size + top_ask_size)
                edge_bps = (microprice - mid_price) / mid_price * 10_000
        return {
            "obi": obi,
            "spread_bps": spread_bps,
            "edge_bps": edge_bps,
            "top_depth": total_depth,
        }

    def _update_emas(self, timestamp_ms: int, obi: float) -> None:
        if self.state.last_timestamp_ms is None:
            self.state.last_timestamp_ms = timestamp_ms
            self.state.obi_fast = obi
            self.state.obi_slow = obi
            return
        delta_seconds = max((timestamp_ms - self.state.last_timestamp_ms) / 1000.0, 0.0)
        self.state.last_timestamp_ms = timestamp_ms
        self.state.obi_fast = self._ema(self.state.obi_fast, obi, delta_seconds, self.config.fast_ema_seconds)
        self.state.obi_slow = self._ema(self.state.obi_slow, obi, delta_seconds, self.config.slow_ema_seconds)

    def _ema(self, current: float | None, value: float, delta_seconds: float, horizon_seconds: float) -> float:
        if current is None or horizon_seconds <= 0:
            return value
        alpha = 1.0 if delta_seconds <= 0 else 1.0 - pow(2.718281828459045, -delta_seconds / horizon_seconds)
        return current + alpha * (value - current)

    def _weighted_depth(self, levels: tuple[OrderBookLevel, ...], weights: tuple[float, ...]) -> float:
        total = 0.0
        for idx, level in enumerate(levels):
            weight = weights[idx] if idx < len(weights) else weights[-1]
            total += level.size * weight
        return total

    def _profit_rr(self, position: Any, mid_price: float, risk_price: float) -> float:
        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        direction = str(getattr(position, "direction", ""))
        if direction == "BULL":
            return (mid_price - entry_price) / risk_price
        return (entry_price - mid_price) / risk_price

    def _is_adverse(self, position: Any, obi: float, obi_mom: float, edge_bps: float) -> bool:
        direction = str(getattr(position, "direction", ""))
        if direction == "BULL":
            return (
                obi <= -self.config.tighten_obi_threshold
                and obi_mom <= 0
                and edge_bps <= -self.config.tighten_edge_bps
            )
        return (
            obi >= self.config.tighten_obi_threshold
            and obi_mom >= 0
            and edge_bps >= self.config.tighten_edge_bps
        )

    def _is_force_exit_adverse(self, position: Any, profit_rr: float, obi: float, obi_mom: float, edge_bps: float) -> bool:
        if profit_rr < self.config.force_exit_min_profit_rr:
            return False
        direction = str(getattr(position, "direction", ""))
        if direction == "BULL":
            return (
                obi <= -self.config.force_exit_obi_threshold
                and obi_mom < 0
                and edge_bps <= -self.config.force_exit_edge_bps
            )
        return (
            obi >= self.config.force_exit_obi_threshold
            and obi_mom > 0
            and edge_bps >= self.config.force_exit_edge_bps
        )

    def _lock_rr_for_profit(self, profit_rr: float) -> float:
        if profit_rr >= 3.0:
            return self.config.high_lock_rr
        if profit_rr >= 1.5:
            return self.config.mid_lock_rr
        return self.config.low_lock_rr

    def _stop_price_from_lock_rr(self, position: Any, risk_price: float, lock_rr: float) -> float:
        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        direction = str(getattr(position, "direction", ""))
        if direction == "BULL":
            return entry_price + risk_price * lock_rr
        return entry_price - risk_price * lock_rr

    def _improves_stop(self, position: Any, current_stop: float, suggested_stop: float) -> bool:
        direction = str(getattr(position, "direction", ""))
        if direction == "BULL":
            return suggested_stop > current_stop
        return suggested_stop < current_stop or current_stop <= 0
