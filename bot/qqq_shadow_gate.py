from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QqqShadowGateProfile:
    enabled: bool
    clock: str = "execution_bar"
    reentry_rule: str = "clear"
    reentry_clear_bars: int = 0
    loss_streak_stop: int = 0
    loss_streak_cooldown_bars: int = 0
    equity_dd_stop_pct: float = 0.0
    equity_dd_cooldown_bars: int = 0
    initial_capital: float = 1000.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QqqShadowGateProfile":
        raw = config.get("shadow_gate_replay_profile")
        if not isinstance(raw, dict):
            return cls(enabled=False)
        explicit = raw.get("runtime_enabled")
        status = str(raw.get("status") or "").lower()
        enabled = bool(explicit) if explicit is not None else status in {"runtime_enabled", "enabled"}
        return cls(
            enabled=enabled,
            clock=str(raw.get("clock", raw.get("cooldown_clock", "execution_bar")) or "execution_bar").lower(),
            reentry_rule=str(raw.get("reentry_rule", "clear") or "clear"),
            reentry_clear_bars=int(raw.get("reentry_clear_bars", 0) or 0),
            loss_streak_stop=int(raw.get("loss_streak_stop", 0) or 0),
            loss_streak_cooldown_bars=int(raw.get("loss_streak_cooldown_bars", 0) or 0),
            equity_dd_stop_pct=float(raw.get("equity_dd_stop_pct", 0.0) or 0.0),
            equity_dd_cooldown_bars=int(raw.get("equity_dd_cooldown_bars", 0) or 0),
            initial_capital=float(config.get("initial_capital", 1000.0) or 1000.0),
        )


class QqqShadowGateStateMachine:
    def __init__(self, profile: QqqShadowGateProfile):
        self.profile = profile

    def default_state(self) -> dict[str, Any]:
        capital = float(self.profile.initial_capital)
        return {
            "version": 1,
            "enabled": bool(self.profile.enabled),
            "capital": capital,
            "equity_peak": capital,
            "loss_streak": 0,
            "gate_remaining_bars": 0,
            "gate_reason": None,
            "stopped_after_stop": False,
            "bars_since_stop": None,
            "clear_streak": 0,
            "last_bar_timestamp": None,
            "observed_timestamps": [],
            "position": None,
            "events": [],
        }

    def normalize_state(self, state: dict[str, Any] | None) -> dict[str, Any]:
        default = self.default_state()
        if not isinstance(state, dict):
            return default
        normalized = {**default, **state}
        normalized["enabled"] = bool(self.profile.enabled)
        normalized["capital"] = float(normalized.get("capital", default["capital"]) or default["capital"])
        normalized["equity_peak"] = float(normalized.get("equity_peak", normalized["capital"]) or normalized["capital"])
        normalized["loss_streak"] = int(normalized.get("loss_streak", 0) or 0)
        normalized["gate_remaining_bars"] = max(0, int(normalized.get("gate_remaining_bars", 0) or 0))
        normalized["clear_streak"] = max(0, int(normalized.get("clear_streak", 0) or 0))
        if normalized.get("bars_since_stop") is not None:
            normalized["bars_since_stop"] = max(0, int(normalized.get("bars_since_stop", 0) or 0))
        observed = normalized.get("observed_timestamps")
        normalized["observed_timestamps"] = [str(item) for item in observed] if isinstance(observed, list) else []
        events = normalized.get("events")
        normalized["events"] = events if isinstance(events, list) else []
        return normalized

    def observe_bar(
        self,
        state: dict[str, Any],
        *,
        timestamp: str | None,
        allow_long: bool,
        defense_state: bool,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        state = self.normalize_state(state)
        if not self.profile.enabled:
            return state, None
        ts = str(timestamp or "runtime")
        if self._already_observed(state, ts):
            return state, None

        previous_gate = int(state.get("gate_remaining_bars", 0) or 0)
        if previous_gate > 0:
            state["gate_remaining_bars"] = previous_gate - 1
            if state["gate_remaining_bars"] <= 0:
                state["gate_reason"] = None

        if state.get("bars_since_stop") is not None:
            state["bars_since_stop"] = int(state["bars_since_stop"]) + 1

        if allow_long:
            state["clear_streak"] = int(state.get("clear_streak", 0) or 0) + (0 if defense_state else 1)
            if defense_state:
                state["clear_streak"] = 0
        else:
            state["clear_streak"] = 0
            state["stopped_after_stop"] = False
            state["bars_since_stop"] = None

        state["last_bar_timestamp"] = ts
        self._record_observed(state, ts)
        event = {
            "event": "observe_bar",
            "timestamp": ts,
            "allow_long": bool(allow_long),
            "defense_state": bool(defense_state),
            "gate_remaining_bars": int(state.get("gate_remaining_bars", 0) or 0),
            "clear_streak": int(state.get("clear_streak", 0) or 0),
            "stopped_after_stop": bool(state.get("stopped_after_stop", False)),
        }
        self._append_event(state, event)
        return state, event

    def entry_decision(self, state: dict[str, Any]) -> dict[str, Any]:
        state = self.normalize_state(state)
        if not self.profile.enabled:
            return {"allow": True, "reason": "disabled"}
        gate_remaining = int(state.get("gate_remaining_bars", 0) or 0)
        if gate_remaining > 0:
            return {
                "allow": False,
                "reason": "gate_cooldown",
                "gate_remaining_bars": gate_remaining,
                "gate_reason": state.get("gate_reason"),
            }
        if (
            self.profile.reentry_rule == "clear"
            and bool(state.get("stopped_after_stop", False))
            and int(state.get("clear_streak", 0) or 0) < int(self.profile.reentry_clear_bars)
        ):
            return {
                "allow": False,
                "reason": "reentry_clear_bars",
                "clear_streak": int(state.get("clear_streak", 0) or 0),
                "required_clear_bars": int(self.profile.reentry_clear_bars),
            }
        return {"allow": True, "reason": "ok"}

    def record_open(
        self,
        state: dict[str, Any],
        *,
        timestamp: str | None,
        entry_price: float,
        leverage: float,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        state = self.normalize_state(state)
        if not self.profile.enabled:
            return state, None
        state["clear_streak"] = 0
        state["position"] = {
            "entry_timestamp": str(timestamp or "runtime"),
            "entry_price": float(entry_price),
            "leverage": float(leverage),
        }
        event = {
            "event": "open",
            "timestamp": str(timestamp or "runtime"),
            "entry_price": float(entry_price),
            "leverage": float(leverage),
            "capital": float(state["capital"]),
        }
        self._append_event(state, event)
        return state, event

    def record_close(
        self,
        state: dict[str, Any],
        *,
        timestamp: str | None,
        exit_price: float,
        reason: str,
        taker_fee_rate: float,
        slippage_bps: float,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        state = self.normalize_state(state)
        if not self.profile.enabled:
            return state, None
        position = state.get("position") if isinstance(state.get("position"), dict) else None
        if not position:
            return state, None

        entry_price = float(position.get("entry_price", 0.0) or 0.0)
        leverage = float(position.get("leverage", 0.0) or 0.0)
        gross_return = (float(exit_price) / entry_price - 1.0) * leverage if entry_price > 0 else 0.0
        roundtrip_cost = 2.0 * (float(taker_fee_rate) + float(slippage_bps) / 10000.0) * leverage
        trade_return = gross_return - roundtrip_cost
        old_capital = float(state["capital"])
        new_capital = max(0.0, old_capital * (1.0 + trade_return))
        state["capital"] = new_capital
        state["equity_peak"] = max(float(state.get("equity_peak", old_capital) or old_capital), new_capital)

        loss_trade = trade_return <= 0.0
        state["loss_streak"] = int(state.get("loss_streak", 0) or 0) + 1 if loss_trade else 0
        triggers: list[str] = []
        if self.profile.loss_streak_stop > 0 and int(state["loss_streak"]) >= self.profile.loss_streak_stop:
            self._trigger_gate(state, self.profile.loss_streak_cooldown_bars, "loss_streak")
            triggers.append("loss_streak")
            state["loss_streak"] = 0

        peak = float(state.get("equity_peak", new_capital) or new_capital)
        dd_pct = (peak - new_capital) / peak * 100.0 if peak > 0 else 0.0
        if self.profile.equity_dd_stop_pct > 0 and dd_pct >= self.profile.equity_dd_stop_pct:
            self._trigger_gate(state, self.profile.equity_dd_cooldown_bars, "equity_dd")
            triggers.append("equity_dd")
            state["equity_peak"] = new_capital
            dd_pct = 0.0

        stop_reason = "stop" in str(reason)
        state["stopped_after_stop"] = bool(stop_reason)
        state["bars_since_stop"] = 0 if stop_reason else None
        state["clear_streak"] = 0
        state["position"] = None
        if triggers:
            state["observed_timestamps"] = []

        event = {
            "event": "close",
            "timestamp": str(timestamp or "runtime"),
            "reason": str(reason),
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "leverage": leverage,
            "trade_return_pct": round(trade_return * 100.0, 4),
            "capital": round(new_capital, 6),
            "drawdown_pct": round(dd_pct, 4),
            "loss_streak": int(state.get("loss_streak", 0) or 0),
            "gate_remaining_bars": int(state.get("gate_remaining_bars", 0) or 0),
            "gate_reason": state.get("gate_reason"),
            "triggers": triggers,
        }
        self._append_event(state, event)
        return state, event

    def _trigger_gate(self, state: dict[str, Any], bars: int, reason: str) -> None:
        if bars <= 0:
            return
        state["gate_remaining_bars"] = max(int(state.get("gate_remaining_bars", 0) or 0), int(bars))
        state["gate_reason"] = reason

    @staticmethod
    def _session_sort_key(timestamp: str) -> str | None:
        prefix = "signal_session:"
        if not str(timestamp).startswith(prefix):
            return None
        return str(timestamp)[len(prefix) :]

    def _already_observed(self, state: dict[str, Any], timestamp: str) -> bool:
        observed = state.get("observed_timestamps")
        if not isinstance(observed, list):
            observed = []
        ts = str(timestamp)
        if ts in {str(item) for item in observed}:
            return True
        if state.get("last_bar_timestamp") == ts:
            return True
        session = self._session_sort_key(ts)
        if session is None:
            return False
        observed_sessions = [
            existing
            for item in observed
            for existing in [self._session_sort_key(str(item))]
            if existing is not None
        ]
        return bool(observed_sessions and session <= max(observed_sessions))

    @staticmethod
    def _record_observed(state: dict[str, Any], timestamp: str) -> None:
        observed = state.get("observed_timestamps")
        if not isinstance(observed, list):
            observed = []
        observed.append(str(timestamp))
        state["observed_timestamps"] = [str(item) for item in observed[-250:]]

    @staticmethod
    def _append_event(state: dict[str, Any], event: dict[str, Any]) -> None:
        events = state.get("events")
        if not isinstance(events, list):
            events = []
        events.append(event)
        state["events"] = events[-200:]
