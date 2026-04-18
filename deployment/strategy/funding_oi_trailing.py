from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Literal


Action = Literal["hold", "tighten", "exit"]


@dataclass(frozen=True)
class FundingOISnapshot:
    inst_id: str
    timestamp_ms: int
    mark_price: float
    funding_rate: float
    open_interest: float


@dataclass(frozen=True)
class FundingOIOverlayConfig:
    history_window: int = 96
    oi_lookback_bars: int = 12
    price_lookback_bars: int = 12
    min_profit_rr_tighten: float = 1.0
    min_profit_rr_exit: float = 1.75
    funding_zscore_tighten: float = 1.5
    funding_zscore_exit: float = 2.5
    oi_growth_tighten_pct: float = 0.03
    oi_growth_exit_pct: float = 0.08
    oi_flush_exit_pct: float = 0.08
    squeeze_price_change_pct: float = 0.018
    low_lock_rr: float = 0.5
    mid_lock_rr: float = 1.0
    high_lock_rr: float = 1.5


@dataclass(frozen=True)
class FundingOIOverlayDecision:
    action: Action
    reason: str | None = None
    stop_price: float | None = None
    exit_price: float | None = None
    metrics: dict[str, Any] | None = None


class FundingOIOverlay:
    def __init__(self, config: FundingOIOverlayConfig | None = None):
        self.config = config or FundingOIOverlayConfig()
        self._history: deque[FundingOISnapshot] = deque(maxlen=self.config.history_window)

    def evaluate(self, snapshot: FundingOISnapshot, position: Any) -> FundingOIOverlayDecision:
        self._history.append(snapshot)
        metrics = self._compute_metrics(snapshot, position)
        if metrics is None:
            return FundingOIOverlayDecision(action="hold", metrics={})

        direction = str(position.direction)
        if direction == "BULL":
            return self._evaluate_bull(snapshot, position, metrics)
        if direction == "BEAR":
            return self._evaluate_bear(snapshot, position, metrics)
        return FundingOIOverlayDecision(action="hold", metrics=metrics)

    def _evaluate_bull(self, snapshot: FundingOISnapshot, position: Any, metrics: dict[str, Any]) -> FundingOIOverlayDecision:
        profit_rr = float(metrics["profit_rr"])
        crowded_exit = (
            profit_rr >= self.config.min_profit_rr_exit
            and metrics["funding_rate"] > 0
            and metrics["funding_zscore"] >= self.config.funding_zscore_exit
            and metrics["oi_change_pct"] >= self.config.oi_growth_exit_pct
        )
        squeeze_exit = (
            profit_rr >= self.config.min_profit_rr_exit
            and metrics["price_change_pct"] >= self.config.squeeze_price_change_pct
            and metrics["oi_change_pct"] <= -self.config.oi_flush_exit_pct
            and metrics["recent_min_funding_rate"] < 0
            and metrics["funding_reversion_ratio"] >= 0.66
        )
        momentum_exit = (
            profit_rr >= self.config.min_profit_rr_exit
            and metrics["price_change_pct"] >= self.config.squeeze_price_change_pct
            and metrics["oi_change_pct"] >= self.config.oi_growth_exit_pct
            and metrics["recent_max_funding_rate"] > 0
        )
        if crowded_exit:
            return FundingOIOverlayDecision("exit", "crowded_long_exit", exit_price=snapshot.mark_price, metrics=metrics)
        if squeeze_exit:
            return FundingOIOverlayDecision("exit", "short_squeeze_exit", exit_price=snapshot.mark_price, metrics=metrics)
        if momentum_exit:
            return FundingOIOverlayDecision(
                "exit",
                "bull_momentum_exhaustion_exit",
                exit_price=snapshot.mark_price,
                metrics=metrics,
            )

        crowded_tighten = (
            profit_rr >= self.config.min_profit_rr_tighten
            and metrics["funding_rate"] > 0
            and metrics["funding_zscore"] >= self.config.funding_zscore_tighten
            and metrics["oi_change_pct"] >= self.config.oi_growth_tighten_pct
        )
        squeeze_tighten = (
            profit_rr >= self.config.min_profit_rr_tighten
            and metrics["price_change_pct"] >= self.config.squeeze_price_change_pct * 0.75
            and metrics["oi_change_pct"] <= -self.config.oi_flush_exit_pct * 0.75
        )
        momentum_tighten = (
            profit_rr >= self.config.min_profit_rr_tighten
            and metrics["price_change_pct"] >= self.config.squeeze_price_change_pct * 0.75
            and metrics["oi_change_pct"] >= self.config.oi_growth_tighten_pct
            and metrics["recent_max_funding_rate"] > 0
        )
        if crowded_tighten or squeeze_tighten or momentum_tighten:
            stop_price = self._tightened_stop(position, metrics)
            current_stop = float(getattr(position, "sl_price", getattr(position, "stop_price", 0.0)) or 0.0)
            if stop_price is not None and stop_price > current_stop:
                if crowded_tighten:
                    reason = "crowded_long_tighten"
                elif squeeze_tighten:
                    reason = "short_squeeze_tighten"
                else:
                    reason = "bull_momentum_exhaustion_tighten"
                return FundingOIOverlayDecision("tighten", reason, stop_price=stop_price, metrics=metrics)
        return FundingOIOverlayDecision(action="hold", metrics=metrics)

    def _evaluate_bear(self, snapshot: FundingOISnapshot, position: Any, metrics: dict[str, Any]) -> FundingOIOverlayDecision:
        profit_rr = float(metrics["profit_rr"])
        crowded_exit = (
            profit_rr >= self.config.min_profit_rr_exit
            and metrics["funding_rate"] < 0
            and metrics["funding_zscore"] <= -self.config.funding_zscore_exit
            and metrics["oi_change_pct"] >= self.config.oi_growth_exit_pct
        )
        capitulation_exit = (
            profit_rr >= self.config.min_profit_rr_exit
            and metrics["price_change_pct"] <= -self.config.squeeze_price_change_pct
            and metrics["oi_change_pct"] <= -self.config.oi_flush_exit_pct
            and metrics["recent_max_funding_rate"] > 0
            and metrics["funding_reversion_ratio"] >= 0.66
        )
        momentum_exit = (
            profit_rr >= self.config.min_profit_rr_exit
            and metrics["price_change_pct"] <= -self.config.squeeze_price_change_pct
            and metrics["oi_change_pct"] >= self.config.oi_growth_exit_pct
            and metrics["recent_min_funding_rate"] < 0
        )
        if crowded_exit:
            return FundingOIOverlayDecision("exit", "crowded_short_exit", exit_price=snapshot.mark_price, metrics=metrics)
        if capitulation_exit:
            return FundingOIOverlayDecision(
                "exit",
                "long_capitulation_exit",
                exit_price=snapshot.mark_price,
                metrics=metrics,
            )
        if momentum_exit:
            return FundingOIOverlayDecision(
                "exit",
                "bear_momentum_exhaustion_exit",
                exit_price=snapshot.mark_price,
                metrics=metrics,
            )

        crowded_tighten = (
            profit_rr >= self.config.min_profit_rr_tighten
            and metrics["funding_rate"] < 0
            and metrics["funding_zscore"] <= -self.config.funding_zscore_tighten
            and metrics["oi_change_pct"] >= self.config.oi_growth_tighten_pct
        )
        capitulation_tighten = (
            profit_rr >= self.config.min_profit_rr_tighten
            and metrics["price_change_pct"] <= -self.config.squeeze_price_change_pct * 0.75
            and metrics["oi_change_pct"] <= -self.config.oi_flush_exit_pct * 0.75
        )
        momentum_tighten = (
            profit_rr >= self.config.min_profit_rr_tighten
            and metrics["price_change_pct"] <= -self.config.squeeze_price_change_pct * 0.75
            and metrics["oi_change_pct"] >= self.config.oi_growth_tighten_pct
            and metrics["recent_min_funding_rate"] < 0
        )
        if crowded_tighten or capitulation_tighten or momentum_tighten:
            stop_price = self._tightened_stop(position, metrics)
            current_stop = float(getattr(position, "sl_price", getattr(position, "stop_price", 0.0)) or 0.0)
            if stop_price is not None and (current_stop == 0.0 or stop_price < current_stop):
                if crowded_tighten:
                    reason = "crowded_short_tighten"
                elif capitulation_tighten:
                    reason = "long_capitulation_tighten"
                else:
                    reason = "bear_momentum_exhaustion_tighten"
                return FundingOIOverlayDecision("tighten", reason, stop_price=stop_price, metrics=metrics)
        return FundingOIOverlayDecision(action="hold", metrics=metrics)

    def _tightened_stop(self, position: Any, metrics: dict[str, Any]) -> float | None:
        entry_price = float(position.entry_price)
        initial_stop = float(position.initial_sl_price)
        risk_price = abs(entry_price - initial_stop)
        if risk_price <= 0:
            return None
        profit_rr = float(metrics["profit_rr"])
        if profit_rr >= 3.0:
            lock_rr = self.config.high_lock_rr
        elif profit_rr >= 2.0:
            lock_rr = self.config.mid_lock_rr
        else:
            lock_rr = self.config.low_lock_rr
        direction = str(position.direction)
        if direction == "BULL":
            return entry_price + risk_price * lock_rr
        if direction == "BEAR":
            return entry_price - risk_price * lock_rr
        return None

    def _compute_metrics(self, snapshot: FundingOISnapshot, position: Any) -> dict[str, Any] | None:
        if len(self._history) < max(self.config.oi_lookback_bars, self.config.price_lookback_bars) + 1:
            return None

        funding_values = [item.funding_rate for item in self._history]
        funding_mean = mean(funding_values)
        funding_std = pstdev(funding_values) or 0.0
        funding_zscore = 0.0 if funding_std == 0 else (snapshot.funding_rate - funding_mean) / funding_std

        oi_ref = self._history[-self.config.oi_lookback_bars - 1]
        price_ref = self._history[-self.config.price_lookback_bars - 1]

        oi_change_pct = 0.0
        if oi_ref.open_interest > 0:
            oi_change_pct = (snapshot.open_interest - oi_ref.open_interest) / oi_ref.open_interest

        price_change_pct = 0.0
        if price_ref.mark_price > 0:
            price_change_pct = (snapshot.mark_price - price_ref.mark_price) / price_ref.mark_price

        recent_window = list(self._history)[-max(self.config.oi_lookback_bars, self.config.price_lookback_bars) :]
        recent_min_funding_rate = min(item.funding_rate for item in recent_window)
        recent_max_funding_rate = max(item.funding_rate for item in recent_window)
        if snapshot.funding_rate < 0 and recent_min_funding_rate < 0:
            funding_reversion_ratio = min(1.0, snapshot.funding_rate / recent_min_funding_rate)
        elif snapshot.funding_rate > 0 and recent_max_funding_rate > 0:
            funding_reversion_ratio = min(1.0, snapshot.funding_rate / recent_max_funding_rate)
        else:
            funding_reversion_ratio = 0.0

        entry_price = float(position.entry_price)
        initial_stop = float(position.initial_sl_price)
        risk_price = abs(entry_price - initial_stop)
        if risk_price <= 0:
            profit_rr = 0.0
        elif str(position.direction) == "BULL":
            profit_rr = (snapshot.mark_price - entry_price) / risk_price
        else:
            profit_rr = (entry_price - snapshot.mark_price) / risk_price

        return {
            "timestamp_ms": snapshot.timestamp_ms,
            "mark_price": snapshot.mark_price,
            "funding_rate": snapshot.funding_rate,
            "funding_mean": funding_mean,
            "funding_std": funding_std,
            "funding_zscore": funding_zscore,
            "open_interest": snapshot.open_interest,
            "oi_change_pct": oi_change_pct,
            "price_change_pct": price_change_pct,
            "recent_min_funding_rate": recent_min_funding_rate,
            "recent_max_funding_rate": recent_max_funding_rate,
            "funding_reversion_ratio": funding_reversion_ratio,
            "profit_rr": profit_rr,
        }
