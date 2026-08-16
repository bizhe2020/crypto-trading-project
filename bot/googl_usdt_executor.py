from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any

import pandas as pd

from bot.okx_client import OkxClient, OkxCredentials
from bot.qqq_runtime_policy import filter_closed_bars, market_time_window_status
from bot.qqq_shadow_gate import QqqShadowGateProfile, QqqShadowGateStateMachine
from bot.state_store import StateStore
from bot.strategy_router import RoutedSignalCandidate, StrategyRouterConfig
from scripts.replay_googl_usdt_4h import attach_googl_daily_state, load_okx_4h


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GooglOrderContext:
    symbol: str
    margin_mode: str
    leverage: float
    stop_loss_pct: float
    reference_price: float
    latest_low: float
    stop_price: float
    stop_hit: bool
    route_score: float
    candidate: dict[str, Any]
    ramp_full_leverage: float = 0.0
    ramped: bool = False
    be_locked: bool = False
    bar_timestamp: str | None = None


class GooglUsdtExecutionEngine:
    def __init__(self, router_config: StrategyRouterConfig, googl_config_path: str | Path):
        self.router_config = router_config
        self.googl_config_path = Path(googl_config_path).resolve()
        self.googl_config = json.loads(self.googl_config_path.read_text())
        self.shadow_gate_profile = QqqShadowGateProfile.from_config(self.googl_config)
        self.shadow_gate = QqqShadowGateStateMachine(self.shadow_gate_profile)
        credentials = self._load_credentials()
        self.client = OkxClient(credentials, trading_mode=router_config.mode, proxy=self._proxy())
        self.store = StateStore(router_config.googl_state_db_path)
        self._markets_cache: dict[str, Any] | None = None

    def _load_credentials(self) -> OkxCredentials | None:
        if not self.router_config.execution_credentials_config:
            return None
        path = Path(self.router_config.execution_credentials_config)
        payload = json.loads(path.read_text())
        api_key = payload.get("api_key")
        api_secret = payload.get("api_secret")
        api_passphrase = payload.get("api_passphrase")
        if api_key and api_secret and api_passphrase:
            return OkxCredentials(
                api_key=str(api_key),
                api_secret=str(api_secret),
                api_passphrase=str(api_passphrase),
            )
        return None

    def _proxy(self) -> str | None:
        if not self.router_config.execution_credentials_config:
            return None
        path = Path(self.router_config.execution_credentials_config)
        payload = json.loads(path.read_text())
        proxy = payload.get("proxy")
        return str(proxy) if proxy else None

    def _load_markets(self) -> dict[str, Any]:
        if self._markets_cache is None:
            cache = self._load_markets_cache()
            if cache:
                self._markets_cache = cache
                self._hydrate_exchange_markets(cache)
            else:
                self._markets_cache = self.client.load_markets()
        return self._markets_cache

    def _hydrate_exchange_markets(self, markets: dict[str, Any]) -> None:
        try:
            self.client.exchange.set_markets(list(markets.values()))
        except Exception:
            return

    def _load_markets_cache(self) -> dict[str, Any]:
        path = Path(self.router_config.okx_markets_cache_path)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        symbol = str(self.googl_config["execution_symbol"])
        market = payload.get(symbol)
        if isinstance(market, dict):
            return {symbol: market}
        return {key: value for key, value in payload.items() if isinstance(value, dict)}

    def _market(self) -> dict[str, Any]:
        symbol = str(self.googl_config["execution_symbol"])
        market = self._load_markets().get(symbol)
        if market is None:
            raise ValueError(f"Market metadata missing for {symbol}")
        return market

    def _exchange_leverage(self) -> float:
        configured = self.googl_config.get("exchange_leverage", self.googl_config.get("base_leverage", 10.0))
        return float(configured or 10.0)

    def bootstrap(self) -> dict[str, Any]:
        symbol = str(self.googl_config["execution_symbol"])
        exchange_leverage = self._exchange_leverage()
        leverage = int(round(exchange_leverage))
        error = None
        bootstrap_step = "start"
        market_loaded = False
        try:
            bootstrap_step = "load_markets"
            markets = self._load_markets()
            market_loaded = symbol in markets
            if self.router_config.mode != "paper":
                bootstrap_step = "set_leverage"
                self.client.set_leverage(leverage, symbol, margin_mode=self.router_config.googl_margin_mode, pos_side="long")
            bootstrap_step = "completed"
        except Exception as exc:
            error = str(exc)
        payload = {
            "status": "ok" if error is None else "error",
            "symbol": symbol,
            "market_loaded": market_loaded,
            "market_cache": str(self.router_config.okx_markets_cache_path),
            "leverage": leverage,
            "exchange_leverage": exchange_leverage,
            "bootstrap_step": bootstrap_step,
            "error": error,
        }
        self.store.append_action("bootstrap", "BOOTSTRAP", payload)
        return payload

    def evaluate_latest(self, candidate: RoutedSignalCandidate | None) -> dict[str, Any]:
        context = self._build_context(candidate) if candidate is not None and candidate.active else None
        exchange_position = self.fetch_position_state()
        state = self.load_state()
        actions: list[dict[str, Any]] = []
        local_position = state.get("position") if isinstance(state.get("position"), dict) else None
        if local_position is not None and float(exchange_position.get("contracts", 0.0) or 0.0) <= 0:
            sync_result = self.sync_external_flat(reason="exchange_position_flat", context=context, state=state)
            actions.append(sync_result)
            if sync_result.get("status") == "synced":
                return {"status": "ok", "symbol": self.symbol, "actions": actions, "position_open": False}
            state = self.load_state()

        if context is None:
            if exchange_position["contracts"] > 0:
                actions.append(self.close_position(reason="router_no_googl_signal"))
            self.save_state({"position": None, "last_candidate": candidate.to_dict() if candidate else None})
            return {"status": "ok", "symbol": self.symbol, "actions": actions, "position_open": False}

        if context.stop_hit:
            if exchange_position["contracts"] > 0:
                actions.append(
                    self.close_position(
                        reason="googl_trailing_stop_hit",
                        exit_price=context.stop_price,
                        timestamp=str(context.candidate.get("timestamp") or "runtime"),
                    )
                )
            else:
                actions.append({"status": "skipped", "reason": "googl_stop_hit_no_exchange_position"})
                self._shadow_gate_record_close(
                    state.get("position") if isinstance(state.get("position"), dict) else None,
                    reason="googl_trailing_stop_hit",
                    exit_price=context.stop_price,
                    timestamp=str(context.candidate.get("timestamp") or "runtime"),
                )
            self.save_state(
                {
                    "position": None,
                    "last_candidate": context.candidate,
                    "last_stop_hit": {
                        "candidate_timestamp": context.candidate.get("timestamp"),
                        "stop_price": context.stop_price,
                        "latest_low": context.latest_low,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    "last_flat_event": {
                        "candidate_timestamp": context.candidate.get("timestamp"),
                        "reason": "googl_trailing_stop_hit",
                        "stop_like": True,
                        "stop_price": context.stop_price,
                        "latest_low": context.latest_low,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
            )
            return {"status": "ok", "symbol": self.symbol, "actions": actions, "position_open": False}

        if exchange_position["contracts"] <= 0:
            if not self._candidate_meets_entry_score(context):
                actions.append(
                    {
                        "status": "skipped",
                        "reason": "googl_entry_score_below_min",
                        "action": "open_googl_usdt_long",
                        "route_score": context.route_score,
                        "min_route_score": float(self.router_config.googl_min_route_score),
                    }
                )
                self.save_state({"position": None, "last_candidate": context.candidate})
                return {"status": "ok", "symbol": self.symbol, "actions": actions, "position_open": False}
            risk_on_status = self._risk_on_window_status()
            if not bool(risk_on_status["open"]):
                actions.append(
                    {
                        "status": "skipped",
                        "reason": "googl_risk_on_window_closed",
                        "action": "open_googl_usdt_long",
                        "target_leverage": context.leverage,
                        "market_window": risk_on_status,
                    }
                )
                self.save_state({"position": None, "last_candidate": context.candidate})
                return {"status": "ok", "symbol": self.symbol, "actions": actions, "position_open": False}
            if self._same_signal_stop_locked(context):
                actions.append(
                    {
                        "status": "skipped",
                        "reason": "googl_stop_hit_same_signal_lock",
                        "candidate_timestamp": context.candidate.get("timestamp"),
                    }
                )
                return {"status": "ok", "symbol": self.symbol, "actions": actions, "position_open": False}
            if not self._stop_price_valid_for_open(context):
                actions.append(
                    {
                        "status": "skipped",
                        "reason": "googl_invalid_initial_stop_price",
                        "action": "open_googl_usdt_long",
                        "reference_price": context.reference_price,
                        "stop_price": context.stop_price,
                    }
                )
                self.save_state({"position": None, "last_candidate": context.candidate})
                return {"status": "ok", "symbol": self.symbol, "actions": actions, "position_open": False}
            shadow_gate = self._shadow_gate_pre_open(context)
            if not bool(shadow_gate.get("allow", True)):
                actions.append(
                    {
                        "status": "skipped",
                        "reason": "googl_shadow_gate_blocked",
                        "action": "open_googl_usdt_long",
                        "shadow_gate": shadow_gate,
                    }
                )
                self.save_state({"position": None, "last_candidate": context.candidate})
                return {"status": "ok", "symbol": self.symbol, "actions": actions, "position_open": False}
            open_result = self.open_position(context)
            actions.append(open_result)
            position_open = open_result.get("status") in {"paper_opened", "submitted"}
            if position_open:
                self._shadow_gate_record_open(context)
            position_state = self._state_from_context(context) if position_open else None
            position_state = self._with_exchange_stop_fields(position_state, open_result.get("exchange_stop"))
            self.save_state(
                {
                    "position": position_state,
                    "last_candidate": context.candidate,
                }
            )
            return {"status": "ok", "symbol": self.symbol, "actions": actions, "position_open": position_open}

        current = state.get("position") if isinstance(state.get("position"), dict) else {}
        current_leverage = float(current.get("leverage", 0.0) or 0.0)
        target_notional = self._rebalance_target_notional(context, exchange_position, current)
        current_notional = float(exchange_position["notional_usdt"])
        notional_gap = abs(current_notional - target_notional)
        leverage_changed = abs(current_leverage - float(context.leverage)) > 1e-9
        notional_gap_large = self._rebalance_gap_large(notional_gap, current_notional, target_notional)
        leverage_rebalance = bool(self.router_config.googl_rebalance_on_leverage_change) and leverage_changed
        notional_rebalance = bool(self.router_config.googl_rebalance_on_notional_gap)
        should_rebalance = (leverage_rebalance or notional_rebalance) and notional_gap_large
        cooldown_status = self._rebalance_cooldown_status(current)
        if should_rebalance and bool(cooldown_status.get("active")):
            should_rebalance = False
        risk_on_blocked = False
        risk_on_status: dict[str, Any] | None = None
        if should_rebalance and target_notional > current_notional:
            risk_on_status = self._risk_on_window_status()
            risk_on_blocked = not bool(risk_on_status["open"])
            should_rebalance = should_rebalance and not risk_on_blocked
        next_position_state = self._state_from_context(context)
        if should_rebalance:
            rebalance_result = self.rebalance_position(
                context,
                exchange_position,
                current,
                target_notional=target_notional,
            )
            actions.append(rebalance_result)
            if rebalance_result.get("status") == "error":
                next_position_state = current
            else:
                next_position_state = self._with_exchange_stop_fields(
                    next_position_state,
                    rebalance_result.get("exchange_stop"),
                )
                if self._rebalance_submitted(rebalance_result):
                    next_position_state["last_rebalance_at"] = datetime.now(timezone.utc).isoformat()
                    next_position_state["last_rebalance_side"] = rebalance_result.get("side")
                    next_position_state["last_rebalance_delta_notional_usdt"] = rebalance_result.get("delta_notional_usdt")
            position_open = True
        elif risk_on_blocked:
            actions.append(
                {
                    "status": "skipped",
                    "action": "rebalance_googl_position",
                    "reason": "googl_risk_on_window_closed",
                    "current_leverage": current_leverage,
                    "target_leverage": context.leverage,
                    "current_notional_usdt": float(exchange_position["notional_usdt"]),
                    "target_notional_usdt": target_notional,
                    "notional_gap_usdt": notional_gap,
                    "market_window": risk_on_status,
                }
            )
            next_position_state = current
            position_open = True
        elif bool(cooldown_status.get("active")) and (leverage_rebalance or notional_rebalance):
            actions.append(
                {
                    "status": "skipped",
                    "action": "rebalance_googl_position",
                    "reason": "rebalance_cooldown_active",
                    "current_leverage": current_leverage,
                    "target_leverage": context.leverage,
                    "current_notional_usdt": current_notional,
                    "target_notional_usdt": target_notional,
                    "notional_gap_usdt": notional_gap,
                    "cooldown": cooldown_status,
                }
            )
            next_position_state = current
            position_open = True
        elif leverage_changed:
            actions.append(
                {
                    "status": "skipped",
                    "action": "rebalance_googl_position",
                    "reason": "notional_gap_too_small",
                    "current_leverage": current_leverage,
                    "target_leverage": context.leverage,
                    "current_notional_usdt": current_notional,
                    "target_notional_usdt": target_notional,
                    "notional_gap_usdt": notional_gap,
                }
            )
            next_position_state = current
            position_open = True
        else:
            stop_update = self.update_trailing_stop(context, exchange_position)
            if stop_update is not None:
                actions.append(stop_update)
                if stop_update.get("status") == "error":
                    next_position_state = current
                else:
                    next_position_state = self._with_exchange_stop_fields(
                        next_position_state,
                        stop_update.get("exchange_stop"),
                    )
            position_open = True
        self.save_state(
            {
                "position": next_position_state if position_open else None,
                "last_candidate": context.candidate,
            }
        )
        return {"status": "ok", "symbol": self.symbol, "actions": actions, "position_open": position_open}

    @property
    def symbol(self) -> str:
        return str(self.googl_config["execution_symbol"])

    def load_state(self) -> dict[str, Any]:
        raw = self.store.get_value("googl_usdt_state")
        if not raw:
            return {}
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}

    def save_state(self, payload: dict[str, Any]) -> None:
        self.store.set_value("googl_usdt_state", json.dumps(payload, ensure_ascii=False, indent=2))

    def _load_shadow_gate_state(self) -> dict[str, Any]:
        raw = self.store.get_value("googl_shadow_gate_state")
        if not raw:
            return self.shadow_gate.default_state()
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return self.shadow_gate.default_state()
        return self.shadow_gate.normalize_state(decoded if isinstance(decoded, dict) else {})

    def _save_shadow_gate_state(self, state: dict[str, Any]) -> None:
        self.store.set_value("googl_shadow_gate_state", json.dumps(self.shadow_gate.normalize_state(state), ensure_ascii=False, indent=2))

    def _shadow_gate_state_summary(self, state: dict[str, Any]) -> dict[str, Any]:
        normalized = self.shadow_gate.normalize_state(state)
        return {
            "enabled": bool(self.shadow_gate_profile.enabled),
            "capital": round(float(normalized.get("capital", 0.0) or 0.0), 6),
            "equity_peak": round(float(normalized.get("equity_peak", 0.0) or 0.0), 6),
            "loss_streak": int(normalized.get("loss_streak", 0) or 0),
            "gate_remaining_bars": int(normalized.get("gate_remaining_bars", 0) or 0),
            "gate_reason": normalized.get("gate_reason"),
            "clear_streak": int(normalized.get("clear_streak", 0) or 0),
            "stopped_after_stop": bool(normalized.get("stopped_after_stop", False)),
            "last_bar_timestamp": normalized.get("last_bar_timestamp"),
            "position_open": isinstance(normalized.get("position"), dict),
        }

    def _append_shadow_gate_event(self, timestamp: str | None, event: dict[str, Any], state: dict[str, Any], decision: dict[str, Any] | None = None) -> None:
        payload = {
            "event": event,
            "decision": decision,
            "state": self._shadow_gate_state_summary(state),
            "profile": {
                "clock": self.shadow_gate_profile.clock,
                "reentry_rule": self.shadow_gate_profile.reentry_rule,
                "reentry_clear_bars": self.shadow_gate_profile.reentry_clear_bars,
                "loss_streak_stop": self.shadow_gate_profile.loss_streak_stop,
                "loss_streak_cooldown_bars": self.shadow_gate_profile.loss_streak_cooldown_bars,
                "equity_dd_stop_pct": self.shadow_gate_profile.equity_dd_stop_pct,
                "equity_dd_cooldown_bars": self.shadow_gate_profile.equity_dd_cooldown_bars,
            },
        }
        self.store.append_action(str(timestamp or "runtime"), "GOOGL_SHADOW_GATE", payload)

    def _shadow_gate_observation_timestamp(self, context: GooglOrderContext) -> str:
        clock = str(getattr(self.shadow_gate_profile, "clock", "execution_bar") or "execution_bar").lower()
        metadata = context.candidate.get("metadata") if isinstance(context.candidate.get("metadata"), dict) else {}
        if clock in {"signal_session", "daily_signal", "session_day", "daily"}:
            raw = metadata.get("daily_signal_timestamp") or metadata.get("session_day") or context.candidate.get("timestamp")
            if raw is None:
                return "signal_session:runtime"
            try:
                timestamp = pd.Timestamp(raw)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize("UTC")
                else:
                    timestamp = timestamp.tz_convert("UTC")
                return f"signal_session:{timestamp.date().isoformat()}"
            except Exception:
                return f"signal_session:{raw}"
        return str(context.candidate.get("timestamp") or "runtime")

    def _shadow_gate_observe_context(self, context: GooglOrderContext) -> tuple[dict[str, Any], dict[str, Any] | None]:
        state = self._load_shadow_gate_state()
        metadata = context.candidate.get("metadata") if isinstance(context.candidate.get("metadata"), dict) else {}
        state, event = self.shadow_gate.observe_bar(
            state,
            timestamp=self._shadow_gate_observation_timestamp(context),
            allow_long=bool(context.candidate.get("active", True)),
            defense_state=bool(metadata.get("defense_state", False)),
        )
        self._save_shadow_gate_state(state)
        if event is not None:
            self._append_shadow_gate_event(context.candidate.get("timestamp"), event, state)
        return state, event

    def _shadow_gate_pre_open(self, context: GooglOrderContext) -> dict[str, Any]:
        if not self.shadow_gate_profile.enabled:
            return {"enabled": False, "allow": True, "reason": "disabled"}
        state, _ = self._shadow_gate_observe_context(context)
        decision = self.shadow_gate.entry_decision(state)
        if not bool(decision.get("allow", False)):
            self._append_shadow_gate_event(
                context.candidate.get("timestamp"),
                {"event": "entry_blocked", "timestamp": str(context.candidate.get("timestamp") or "runtime")},
                state,
                decision,
            )
        return {"enabled": True, **decision, "state": self._shadow_gate_state_summary(state)}

    def shadow_gate_observe_candidate(self, candidate: RoutedSignalCandidate | None) -> dict[str, Any]:
        if not self.shadow_gate_profile.enabled:
            return {"enabled": False, "allow": True, "reason": "disabled"}
        if candidate is None or not candidate.timestamp:
            return {"enabled": True, "allow": True, "reason": "no_candidate_timestamp"}
        context = self._build_context(candidate)
        state, _ = self._shadow_gate_observe_context(context)
        decision = self.shadow_gate.entry_decision(state)
        return {"enabled": True, **decision, "state": self._shadow_gate_state_summary(state)}

    def shadow_gate_pre_switch_status(self, candidate: RoutedSignalCandidate | None) -> dict[str, Any]:
        if not self.shadow_gate_profile.enabled or candidate is None or not candidate.active:
            return {"enabled": bool(self.shadow_gate_profile.enabled), "allow": True, "reason": "disabled_or_inactive"}
        shadow_gate = self.shadow_gate_observe_candidate(candidate)
        if not bool(shadow_gate.get("allow", True)):
            self._append_shadow_gate_event(
                candidate.timestamp,
                {"event": "entry_blocked", "timestamp": str(candidate.timestamp or "runtime")},
                self._load_shadow_gate_state(),
                shadow_gate,
            )
        return shadow_gate

    def _shadow_gate_record_open(self, context: GooglOrderContext) -> None:
        if not self.shadow_gate_profile.enabled:
            return
        state = self._load_shadow_gate_state()
        state, event = self.shadow_gate.record_open(
            state,
            timestamp=context.candidate.get("timestamp"),
            entry_price=float(context.reference_price),
            leverage=float(context.leverage),
        )
        self._save_shadow_gate_state(state)
        if event is not None:
            self._append_shadow_gate_event(context.candidate.get("timestamp"), event, state)

    def _shadow_gate_record_close(
        self,
        position_state: dict[str, Any] | None,
        *,
        reason: str,
        exit_price: float | None = None,
        timestamp: str | None = None,
    ) -> None:
        if not self.shadow_gate_profile.enabled:
            return
        state = self._load_shadow_gate_state()
        position = position_state if isinstance(position_state, dict) else {}
        if not isinstance(state.get("position"), dict) and position:
            state["position"] = {
                "entry_timestamp": str(position.get("entry_candidate_timestamp") or "runtime"),
                "entry_price": float(position.get("entry_price", position.get("peak_price", 0.0)) or 0.0),
                "leverage": float(position.get("leverage", self.googl_config.get("base_leverage", 10.0)) or self.googl_config.get("base_leverage", 10.0)),
            }
        resolved_exit = exit_price
        if resolved_exit is None:
            try:
                resolved_exit = float(self._latest_bar()["close"])
            except Exception:
                resolved_exit = float(position.get("latest_low", position.get("peak_price", position.get("entry_price", 0.0))) or 0.0)
        state, event = self.shadow_gate.record_close(
            state,
            timestamp=str(timestamp or "runtime"),
            exit_price=float(resolved_exit or 0.0),
            reason=str(reason),
            taker_fee_rate=float(self.googl_config.get("taker_fee_rate", 0.0) or 0.0),
            slippage_bps=float(self.googl_config.get("slippage_bps", 0.0) or 0.0),
        )
        self._save_shadow_gate_state(state)
        if event is not None:
            self._append_shadow_gate_event(event.get("timestamp"), event, state)

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (ROOT / value).resolve()

    def _signal_overrides(self) -> dict[str, Any]:
        overrides = self.googl_config.get("signal_overrides", {})
        if overrides is None:
            return {}
        if not isinstance(overrides, dict):
            raise TypeError("signal_overrides must be an object")
        return dict(overrides)

    def _load_bars(self) -> pd.DataFrame:
        """GOOGL 执行层 bar 源。优先 4h 合约数据；缺省时回退日线分辨率 bar。

        服务器自闭环：4h feather 由部署脚本推送，日线 prices.csv 由
        scripts/fetch_googl_daily_prices.py 每日刷新（ticker,date,open,close）。
        日线 fallback 无 high/low，用 open/close 极值近似 —— 仅保证 shadow 观察
        不因缺 4h 文件而崩溃，真实执行层观察仍以 4h 为准。
        """
        data_4h = self._resolve_path(str(self.googl_config.get("data_4h", "") or ""))
        if data_4h and data_4h.exists():
            return load_okx_4h(data_4h)
        daily = self._resolve_path(
            str(self.googl_config.get("data_daily_fallback", "var/runtime/googl/prices.csv") or "")
        )
        if daily and daily.exists():
            prices = pd.read_csv(daily)
            if "ticker" in prices.columns:
                prices = prices[prices["ticker"].str.upper().isin(["GOOGL", "GOOG"])]
            prices["date"] = pd.to_datetime(prices["date"], utc=True, errors="coerce")
            prices["open"] = pd.to_numeric(prices.get("open"), errors="coerce")
            prices["close"] = pd.to_numeric(prices.get("close"), errors="coerce")
            prices = prices.dropna(subset=["date", "open", "close"]).sort_values("date").reset_index(drop=True)
            if prices.empty:
                raise FileNotFoundError(f"GOOGL 日线 fallback 无有效行: {daily}")
            return pd.DataFrame(
                {
                    "date": prices["date"],
                    "open": prices["open"],
                    "high": prices[["open", "close"]].max(axis=1),
                    "low": prices[["open", "close"]].min(axis=1),
                    "close": prices["close"],
                }
            )
        raise FileNotFoundError(
            f"GOOGL 执行层数据缺失: data_4h {data_4h} 与日线 fallback {daily} 均不存在"
        )

    def _load_googl_signal_path(self, signal_source: Path) -> pd.DataFrame:
        """载入日线信号 CSV（date,position,berkshire_conviction,leverage_tier,target_leverage）。

        attach_googl_daily_state 需要 leverage_tier/target_leverage 列；旧信号缺列时
        按 position 兜底（GOOGL→base / FLAT→flat），保证不 KeyError。
        """
        if not signal_source.exists():
            raise FileNotFoundError(f"GOOGL daily signal CSV not found: {signal_source}")
        frame = pd.read_csv(signal_source)
        missing = [c for c in ("date", "position") if c not in frame.columns]
        if missing:
            raise ValueError(f"GOOGL daily signal CSV missing columns: {missing}")
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if "leverage_tier" not in frame.columns:
            frame["leverage_tier"] = frame["position"].map({"GOOGL": "base", "FLAT": "flat", "CASH": "flat"}).fillna("flat")
        if "target_leverage" not in frame.columns:
            frame["target_leverage"] = 0.0
        return frame

    def _closed_bars(self) -> pd.DataFrame:
        """日线信号附着到 4h bars 后，过滤出已收盘 bar。"""
        signal_source = self._resolve_path(str(self.googl_config["signal_source"]))
        signal_path = self._load_googl_signal_path(signal_source)
        bars = attach_googl_daily_state(self._load_bars(), signal_path, trim_to_signal_end=False)
        if bool(self.googl_config.get("use_closed_execution_bars", True)):
            bars = filter_closed_bars(
                bars,
                timeframe=str(self.googl_config.get("execution_timeframe", "4h")),
                grace_seconds=int(self.googl_config.get("closed_bar_grace_seconds", 30) or 0),
            )
        if bars.empty:
            raise RuntimeError("No closed GOOGL/USDT bars available")
        return bars.reset_index(drop=True)

    def _latest_bar(self) -> pd.Series:
        return self._closed_bars().iloc[-1]

    def _latest_bars(self) -> tuple[pd.Series, float | None]:
        """最近两根已收盘 bar：(latest, prev_close)。

        prev_close 供杠杆爬坡确认（无前视语义，与 run_googl_4h_replay 一致）：
        在最新已收盘 bar 上决策时，其上一根 bar 的 close 是已知的。
        """
        bars = self._closed_bars()
        latest = bars.iloc[-1]
        prev_close = None
        if len(bars) >= 2:
            prev_close = float(bars.iloc[-2]["close"])
        return latest, prev_close

    def _build_context(self, candidate: RoutedSignalCandidate) -> GooglOrderContext:
        latest, prev_close = self._latest_bars()
        reference_price = float(latest["close"])
        latest_low = float(latest["low"])
        candidate_payload = candidate.to_dict()
        candidate_metadata = candidate_payload.get("metadata")
        metadata = dict(candidate_metadata) if isinstance(candidate_metadata, dict) else {}
        base_leverage = float(self.googl_config["base_leverage"])
        full_leverage = float(candidate.leverage or base_leverage)
        stop_loss_pct = float(self.googl_config["stop_loss_pct"])
        ramp_confirm_pct = float(self.googl_config.get("ramp_confirm_pct", 0.0) or 0.0)
        ramp_pre_stop_pct = float(self.googl_config.get("ramp_pre_stop_pct", 0.0) or 0.0)
        state = self.load_state()
        previous = state.get("position") if isinstance(state.get("position"), dict) else {}
        entry_price = float(previous.get("entry_price", 0.0) or 0.0)

        # --- 杠杆爬坡状态机（镜像 run_googl_4h_replay） ---
        # 无前视语义：在最新已收盘 bar 上决策，用其上一根已收盘 close（prev_close）
        # 做确认；只在持仓中（previous 非空，=was_holding）才生效，入场 bar 自身不爬坡。
        if previous:
            entry_leverage = float(previous.get("entry_leverage", 0.0) or 0.0) or full_leverage
            ramp_full_leverage = float(previous.get("ramp_full_leverage", 0.0) or 0.0)
            ramped = bool(previous.get("ramped", False))
            be_locked = bool(previous.get("be_locked", False))
            if (
                ramp_full_leverage > 0
                and not ramped
                and entry_price > 0
                and prev_close is not None
                and prev_close >= entry_price * (1.0 + ramp_confirm_pct / 100.0)
            ):
                entry_leverage = ramp_full_leverage
                ramped = True
        else:
            entry_leverage = full_leverage
            ramp_full_leverage = 0.0
            ramped = False
            be_locked = False
            if ramp_confirm_pct > 0 and full_leverage > base_leverage > 0:
                ramp_full_leverage = full_leverage
                entry_leverage = base_leverage

        # 止损宽度：爬坡档确认前用 ramp_pre_stop_pct（若设置）；已确认或非爬坡单用 stop_loss_pct。
        if ramp_full_leverage > 0 and not ramped and ramp_pre_stop_pct > 0:
            effective_stop_loss_pct = ramp_pre_stop_pct
        else:
            effective_stop_loss_pct = stop_loss_pct

        previous_stop = float(previous.get("stop_price", 0.0) or 0.0)
        peak_price = max(float(previous.get("peak_price", 0.0) or 0.0), reference_price)
        stop_price = max(previous_stop, peak_price * (1.0 - effective_stop_loss_pct / 100.0))
        bar_timestamp = str(pd.Timestamp(latest["date"]))
        # 入场 bar 保护：仅跳过"入场那一根已收盘 bar"的止损判定。
        # GOOGL 的 candidate.timestamp 是日线信号日期（非 4h bar），不能直接用于 same_entry_bar，
        # 否则会把入场当天后续所有 4h bar 的止损都误压掉（爬坡 pre_stop 失效）。
        entry_bar_timestamp = previous.get("entry_bar_timestamp")
        same_entry_bar = bool(entry_bar_timestamp and bar_timestamp and str(entry_bar_timestamp) == str(bar_timestamp))
        stop_hit = bool(previous_stop > 0 and latest_low <= previous_stop and not same_entry_bar)
        return GooglOrderContext(
            symbol=self.symbol,
            margin_mode=str(self.router_config.googl_margin_mode),
            leverage=entry_leverage,
            stop_loss_pct=effective_stop_loss_pct,
            reference_price=reference_price,
            latest_low=latest_low,
            stop_price=stop_price,
            stop_hit=stop_hit,
            route_score=float(candidate.route_score),
            candidate=candidate_payload,
            ramp_full_leverage=ramp_full_leverage,
            ramped=ramped,
            be_locked=be_locked,
            bar_timestamp=bar_timestamp,
        )

    def _risk_on_window_status(self) -> dict[str, Any]:
        return market_time_window_status(
            enabled=bool(self.router_config.googl_rebalance_risk_on_market_hours_only),
            timezone_name=str(self.router_config.googl_market_hours_timezone),
            start_time=str(self.router_config.googl_market_hours_start),
            end_time=str(self.router_config.googl_market_hours_end),
            trading_calendar=str(self.router_config.googl_market_calendar),
        )

    def risk_on_window_status(self) -> dict[str, Any]:
        return self._risk_on_window_status()

    def _state_from_context(self, context: GooglOrderContext) -> dict[str, Any]:
        state = self.load_state()
        previous = state.get("position") if isinstance(state.get("position"), dict) else {}
        peak_price = max(float(previous.get("peak_price", 0.0) or 0.0), float(context.reference_price))
        payload = {
            "symbol": context.symbol,
            "leverage": context.leverage,
            "stop_loss_pct": context.stop_loss_pct,
            "stop_price": context.stop_price,
            "latest_low": context.latest_low,
            "peak_price": peak_price,
            "entry_price": float(previous.get("entry_price", context.reference_price) or context.reference_price),
            "entry_leverage": float(context.leverage),
            "ramp_full_leverage": float(context.ramp_full_leverage or 0.0),
            "ramped": bool(context.ramped),
            "be_locked": bool(context.be_locked),
            "entry_candidate_timestamp": previous.get("entry_candidate_timestamp") or context.candidate.get("timestamp"),
            "entry_bar_timestamp": previous.get("entry_bar_timestamp") or context.bar_timestamp,
            "route_score": context.route_score,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata = context.candidate.get("metadata") if isinstance(context.candidate.get("metadata"), dict) else {}
        if previous.get("strength_label"):
            payload["strength_label"] = previous["strength_label"]
        elif context.candidate.get("strength_label"):
            payload["strength_label"] = context.candidate["strength_label"]
        for key in (
            "exchange_order_id",
            "exchange_attach_algo_id",
            "exchange_attach_algo_client_id",
            "last_rebalance_at",
            "last_rebalance_side",
            "last_rebalance_delta_notional_usdt",
        ):
            if previous.get(key):
                payload[key] = previous[key]
        return payload

    def _same_signal_stop_locked(self, context: GooglOrderContext) -> bool:
        state = self.load_state()
        last_flat_event = state.get("last_flat_event") if isinstance(state.get("last_flat_event"), dict) else {}
        flat_timestamp = last_flat_event.get("candidate_timestamp")
        candidate_timestamp = context.candidate.get("timestamp")
        if flat_timestamp and candidate_timestamp and str(flat_timestamp) == str(candidate_timestamp):
            return True
        last_stop_hit = state.get("last_stop_hit") if isinstance(state.get("last_stop_hit"), dict) else {}
        stop_timestamp = last_stop_hit.get("candidate_timestamp")
        return bool(stop_timestamp and candidate_timestamp and str(stop_timestamp) == str(candidate_timestamp))

    def _candidate_meets_entry_score(self, context: GooglOrderContext) -> bool:
        return float(context.route_score) >= float(self.router_config.googl_min_route_score)

    @staticmethod
    def _stop_price_valid_for_open(context: GooglOrderContext) -> bool:
        return float(context.stop_price) > 0 and float(context.stop_price) < float(context.reference_price)

    def _extract_available_usdt(self, balance: dict[str, Any]) -> float:
        usdt = balance.get("USDT") if isinstance(balance, dict) else None
        if isinstance(usdt, dict):
            for key in ("free", "available", "availableBalance", "cash", "total"):
                value = usdt.get(key)
                if value not in (None, ""):
                    numeric = float(value)
                    if numeric > 0:
                        return numeric
        info = balance.get("info") if isinstance(balance, dict) else None
        if isinstance(info, dict):
            for row in info.get("data", []) or []:
                for detail in row.get("details", []) or []:
                    if detail.get("ccy") != "USDT":
                        continue
                    for key in ("availBal", "cashBal", "eq", "availEq"):
                        value = detail.get(key)
                        if value not in (None, ""):
                            numeric = float(value)
                            if numeric > 0:
                                return numeric
        raise ValueError("Unable to extract positive USDT balance")

    def _extract_total_usdt_equity(self, balance: dict[str, Any]) -> float:
        usdt = balance.get("USDT") if isinstance(balance, dict) else None
        if isinstance(usdt, dict):
            for key in ("total", "equity", "eq", "eqUsd"):
                value = usdt.get(key)
                if value not in (None, ""):
                    numeric = float(value)
                    if numeric > 0:
                        return numeric
        info = balance.get("info") if isinstance(balance, dict) else None
        if isinstance(info, dict):
            for row in info.get("data", []) or []:
                for detail in row.get("details", []) or []:
                    if detail.get("ccy") != "USDT":
                        continue
                    for key in ("eq", "eqUsd", "cashBal", "availEq"):
                        value = detail.get(key)
                        if value not in (None, ""):
                            numeric = float(value)
                            if numeric > 0:
                                return numeric
            for row in info.get("data", []) or []:
                value = row.get("totalEq")
                if value not in (None, ""):
                    numeric = float(value)
                    if numeric > 0:
                        return numeric
        raise ValueError("Unable to extract positive USDT equity")

    def _sizing_capital_usdt(self) -> float:
        if self.router_config.mode == "paper":
            capital = 1000.0
        else:
            balance = self.client.fetch_balance()
            basis = str(self.router_config.googl_sizing_basis or "available").strip().lower()
            if basis in {"total_equity", "equity", "total"}:
                capital = self._extract_total_usdt_equity(balance)
            elif basis in {"available", "free", "available_balance"}:
                capital = self._extract_available_usdt(balance)
            else:
                raise ValueError(f"Unsupported googl_sizing_basis: {self.router_config.googl_sizing_basis}")
        buffer_usdt = max(0.0, float(self.router_config.googl_sizing_cash_buffer_usdt or 0.0))
        return max(0.0, float(capital) - buffer_usdt)

    def _target_notional(self, context: GooglOrderContext) -> float:
        notional = self._sizing_capital_usdt() * float(self.router_config.googl_position_size_pct) * float(context.leverage)
        if self.router_config.googl_max_notional_usdt is not None:
            notional = min(notional, float(self.router_config.googl_max_notional_usdt))
        return max(notional, 0.0)

    def _rebalance_target_notional(
        self,
        context: GooglOrderContext,
        exchange_position: dict[str, Any],
        current: dict[str, Any],
    ) -> float:
        current_notional = float(exchange_position.get("notional_usdt", 0.0) or 0.0)
        current_leverage = float(current.get("leverage", 0.0) or 0.0)
        if bool(self.router_config.googl_rebalance_on_notional_gap):
            target = self._target_notional(context)
        elif current_notional > 0 and current_leverage > 0:
            target = current_notional * float(context.leverage) / current_leverage
        else:
            target = self._target_notional(context)
        if self.router_config.googl_max_notional_usdt is not None:
            target = min(target, float(self.router_config.googl_max_notional_usdt))
        return max(float(target), 0.0)

    def _rebalance_gap_large(self, notional_gap: float, current_notional: float, target_notional: float) -> bool:
        min_abs = max(0.0, float(self.router_config.googl_min_rebalance_notional_usdt or 0.0))
        if float(notional_gap) < min_abs:
            return False
        min_ratio = max(0.0, float(self.router_config.googl_min_rebalance_gap_ratio or 0.0))
        if min_ratio <= 0:
            return True
        basis = max(abs(float(current_notional)), abs(float(target_notional)), 1.0)
        return float(notional_gap) / basis >= min_ratio

    def _rebalance_cooldown_status(self, current: dict[str, Any]) -> dict[str, Any]:
        cooldown_seconds = max(0.0, float(self.router_config.googl_rebalance_cooldown_seconds or 0.0))
        if cooldown_seconds <= 0:
            return {"enabled": False, "active": False}
        raw = current.get("last_rebalance_at") if isinstance(current, dict) else None
        if not raw:
            return {"enabled": True, "active": False, "cooldown_seconds": cooldown_seconds}
        try:
            last = pd.Timestamp(raw)
        except Exception:
            return {"enabled": True, "active": False, "cooldown_seconds": cooldown_seconds, "last_rebalance_at": raw}
        if last.tzinfo is None:
            last = last.tz_localize("UTC")
        now = pd.Timestamp.now(tz="UTC")
        elapsed = max(0.0, float((now - last).total_seconds()))
        remaining = max(0.0, cooldown_seconds - elapsed)
        return {
            "enabled": True,
            "active": remaining > 0,
            "cooldown_seconds": cooldown_seconds,
            "last_rebalance_at": last.isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "remaining_seconds": round(remaining, 3),
        }

    @staticmethod
    def _rebalance_submitted(result: dict[str, Any]) -> bool:
        return str(result.get("status") or "") in {"submitted", "submitted_with_leverage_error", "paper_rebalanced"}

    def _order_amount(self, notional: float, reference_price: float) -> float:
        market = self._market()
        contract_size = float(market.get("contractSize") or 1.0) if market.get("contract") else 0.0
        base_amount = notional / reference_price if reference_price > 0 else 0.0
        if market.get("contract"):
            amount = base_amount / contract_size if contract_size > 0 else 0.0
        else:
            amount = base_amount
        return float(self._amount_to_precision(market, amount))

    def _amount_to_precision(self, market: dict[str, Any], amount: float) -> str:
        step = ((market.get("precision") or {}).get("amount")) or ((market.get("limits") or {}).get("amount") or {}).get("min")
        return self._floor_to_step(amount, step)

    def _price_to_precision(self, market: dict[str, Any], price: float) -> str:
        step = (market.get("precision") or {}).get("price")
        return self._floor_to_step(price, step)

    @staticmethod
    def _floor_to_step(value: float, step: Any) -> str:
        if step in (None, "", 0):
            return str(round(float(value), 12))
        decimal_value = Decimal(str(value))
        decimal_step = Decimal(str(step))
        if decimal_step <= 0:
            return str(round(float(value), 12))
        units = (decimal_value / decimal_step).to_integral_value(rounding=ROUND_DOWN)
        return format(units * decimal_step, "f")

    def fetch_position_state(self) -> dict[str, Any]:
        if self.router_config.mode == "paper":
            state = self.load_state()
            position = state.get("position") if isinstance(state.get("position"), dict) else None
            if not position:
                return {"contracts": 0.0, "notional_usdt": 0.0, "raw": None}
            notional = self._target_notional(self._build_context_from_state(position))
            return {"contracts": 1.0, "notional_usdt": notional, "raw": position}
        positions = self.client.fetch_positions([self.symbol])
        for position in positions:
            if not isinstance(position, dict):
                continue
            symbol = position.get("symbol") or position.get("instId")
            if symbol and symbol != self.symbol:
                continue
            info = position.get("info") if isinstance(position.get("info"), dict) else {}
            pos_side = position.get("posSide") or position.get("side") or info.get("posSide") or info.get("side")
            if pos_side and str(pos_side).lower() != "long":
                continue
            contracts = 0.0
            for key in ("contracts", "positionAmt", "pos", "size"):
                value = position.get(key, info.get(key))
                if value not in (None, ""):
                    contracts = abs(float(value))
                    if contracts > 0:
                        break
            if contracts <= 0:
                continue
            notional = 0.0
            for key in ("notional", "notionalUsd", "positionValue", "posValue"):
                value = position.get(key, info.get(key))
                if value not in (None, ""):
                    notional = abs(float(value))
                    if notional > 0:
                        break
            close_order_algos = info.get("closeOrderAlgo") if isinstance(info.get("closeOrderAlgo"), list) else []
            return {
                "contracts": contracts,
                "notional_usdt": notional,
                "close_order_algos": close_order_algos,
                "raw": position,
            }
        return {"contracts": 0.0, "notional_usdt": 0.0, "close_order_algos": [], "raw": None}

    def _build_context_from_state(self, position: dict[str, Any]) -> GooglOrderContext:
        return GooglOrderContext(
            symbol=self.symbol,
            margin_mode=str(self.router_config.googl_margin_mode),
            leverage=float(position.get("leverage", self.googl_config["base_leverage"]) or self.googl_config["base_leverage"]),
            stop_loss_pct=float(position.get("stop_loss_pct", self.googl_config["stop_loss_pct"]) or self.googl_config["stop_loss_pct"]),
            reference_price=float(position.get("peak_price", 1.0) or 1.0),
            latest_low=float(position.get("latest_low", position.get("peak_price", 1.0)) or 1.0),
            stop_price=float(position.get("stop_price", 0.0) or 0.0),
            stop_hit=False,
            route_score=float(position.get("route_score", 0.0) or 0.0),
            candidate={},
            ramp_full_leverage=float(position.get("ramp_full_leverage", 0.0) or 0.0),
            ramped=bool(position.get("ramped", False)),
            be_locked=bool(position.get("be_locked", False)),
            bar_timestamp=position.get("entry_bar_timestamp"),
        )

    def open_position(self, context: GooglOrderContext) -> dict[str, Any]:
        if not self._stop_price_valid_for_open(context):
            return {
                "status": "skipped",
                "reason": "googl_invalid_initial_stop_price",
                "notional_usdt": 0.0,
                "reference_price": context.reference_price,
                "stop_price": context.stop_price,
            }
        notional = self._target_notional(context)
        if notional < float(self.router_config.googl_min_order_notional_usdt):
            return {"status": "skipped", "reason": "notional_too_small", "notional_usdt": notional}
        amount = round(notional / context.reference_price, 6) if self.router_config.mode == "paper" else self._order_amount(notional, context.reference_price)
        if amount <= 0:
            return {"status": "error", "reason": "non_positive_amount", "notional_usdt": notional}
        target_exposure_leverage = context.ramp_full_leverage if context.ramp_full_leverage > 0 else context.leverage
        if self.router_config.mode == "paper":
            return {
                "status": "paper_opened",
                "symbol": context.symbol,
                "amount": amount,
                "notional_usdt": round(notional, 6),
                "leverage": context.leverage,
                "target_exposure_leverage": target_exposure_leverage,
                "stop_price": context.stop_price,
                "ramp_full_leverage": context.ramp_full_leverage,
            }
        exchange_leverage = self._exchange_leverage()
        self.client.set_leverage(int(round(exchange_leverage)), context.symbol, margin_mode=context.margin_mode, pos_side="long")
        chunks = self._market_order_chunks(amount)
        orders = []
        exchange_stops = []
        for idx, chunk_amount in enumerate(chunks):
            params: dict[str, Any] = {"tdMode": context.margin_mode, "posSide": "long"}
            exchange_stop = self._attach_stop_to_order_params(params, context)
            order = self.client.create_order(context.symbol, "market", "buy", chunk_amount, params=params)
            orders.append({"order": order, "amount": chunk_amount})
            if exchange_stop is not None:
                exchange_stops.append(self._with_order_id(exchange_stop, order))
            chunk_delay = self._market_order_chunk_delay_seconds()
            if idx < len(chunks) - 1 and chunk_delay > 0:
                time.sleep(chunk_delay)
        order = orders[0]["order"] if orders else None
        exchange_stop = exchange_stops[-1] if exchange_stops else None
        payload = {
            "status": "submitted",
            "action": "open_googl_usdt_long",
            "order": order,
            "orders": orders,
            "amount": amount,
            "notional_usdt": round(notional, 6),
            "leverage": context.leverage,
            "target_exposure_leverage": target_exposure_leverage,
            "exchange_leverage": exchange_leverage,
            "stop_price": context.stop_price,
            "exchange_stop": exchange_stop,
            "ramp_full_leverage": context.ramp_full_leverage,
            "ramped": bool(context.ramped),
        }
        if exchange_stops:
            payload["exchange_stops"] = exchange_stops
        self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "OPEN_GOOGL_USDT", payload)
        return payload

    def restore_position(self, rollback: dict[str, Any] | None = None) -> dict[str, Any]:
        """Re-open the GOOGL long after a failed strategy switch, bypassing entry gates.

        The router captures a rollback context (position state + exchange position)
        BEFORE flattening the incumbent. If the target strategy fails to open, this
        restores the GOOGL position at market with the captured stop, so the account is
        not left flat by a botched switch.
        """
        if self.router_config.mode == "paper":
            return {"status": "paper_restored", "symbol": self.symbol, "action": "restore_googl_usdt_long"}

        state = self.load_state()
        position_state = state.get("position") if isinstance(state.get("position"), dict) else None
        exchange_state: dict[str, Any] = {}
        if isinstance(rollback, dict):
            captured_position = rollback.get("position") if isinstance(rollback.get("position"), dict) else None
            captured_exchange = rollback.get("exchange_position") if isinstance(rollback.get("exchange_position"), dict) else {}
            if position_state is None:
                position_state = captured_position
            if isinstance(captured_exchange, dict):
                exchange_state = captured_exchange

        contracts = float(exchange_state.get("contracts", 0.0) or 0.0)
        notional = float(exchange_state.get("notional_usdt", 0.0) or 0.0)
        if contracts <= 0 and isinstance(position_state, dict):
            contracts = float(position_state.get("contracts", 0.0) or 0.0)
        if contracts <= 0 and notional <= 0:
            return {"status": "skipped", "reason": "no_restore_size"}

        latest = self._latest_bar()
        reference_price = float(latest["close"])
        latest_low = float(latest["low"])
        if reference_price <= 0:
            return {"status": "error", "reason": "invalid_reference_price"}

        if contracts <= 0:
            contracts = self._order_amount(notional, reference_price)
        if contracts <= 0:
            return {"status": "error", "reason": "non_positive_amount"}

        stop_loss_pct = float(self.googl_config.get("stop_loss_pct", 4.0) or 4.0)
        stop_price = float(position_state.get("stop_price", 0.0) or 0.0) if isinstance(position_state, dict) else 0.0
        if stop_price <= 0 or stop_price >= reference_price:
            stop_price = reference_price * (1.0 - stop_loss_pct / 100.0)

        leverage = float(
            position_state.get("leverage", self.googl_config["base_leverage"])
            if isinstance(position_state, dict)
            else self.googl_config["base_leverage"]
        ) or float(self.googl_config["base_leverage"])
        exchange_leverage = self._exchange_leverage()
        self.client.set_leverage(
            int(round(exchange_leverage)),
            self.symbol,
            margin_mode=str(self.router_config.googl_margin_mode),
            pos_side="long",
        )

        context = GooglOrderContext(
            symbol=self.symbol,
            margin_mode=str(self.router_config.googl_margin_mode),
            leverage=leverage,
            stop_loss_pct=stop_loss_pct,
            reference_price=reference_price,
            latest_low=latest_low,
            stop_price=stop_price,
            stop_hit=False,
            route_score=float(position_state.get("route_score", 100.0) or 100.0) if isinstance(position_state, dict) else 100.0,
            candidate=dict(position_state) if isinstance(position_state, dict) else {},
        )

        chunks = self._market_order_chunks(contracts)
        orders = []
        exchange_stops = []
        for idx, chunk_amount in enumerate(chunks):
            params: dict[str, Any] = {"tdMode": context.margin_mode, "posSide": "long"}
            exchange_stop = self._attach_stop_to_order_params(params, context)
            order = self.client.create_order(self.symbol, "market", "buy", chunk_amount, params=params)
            orders.append({"order": order, "amount": chunk_amount})
            if exchange_stop is not None:
                exchange_stops.append(self._with_order_id(exchange_stop, order))
            chunk_delay = self._market_order_chunk_delay_seconds()
            if idx < len(chunks) - 1 and chunk_delay > 0:
                time.sleep(chunk_delay)

        previous_peak = float(position_state.get("peak_price", reference_price) or reference_price) if isinstance(position_state, dict) else reference_price
        entry_price = float(position_state.get("entry_price", reference_price) or reference_price) if isinstance(position_state, dict) else reference_price
        entry_candidate_ts = position_state.get("entry_candidate_timestamp") if isinstance(position_state, dict) else None
        saved_position = {
            "symbol": self.symbol,
            "leverage": leverage,
            "stop_loss_pct": stop_loss_pct,
            "stop_price": stop_price,
            "latest_low": latest_low,
            "peak_price": max(previous_peak, reference_price),
            "entry_price": entry_price,
            "entry_leverage": float(position_state.get("entry_leverage", leverage) or leverage) if isinstance(position_state, dict) else leverage,
            "ramp_full_leverage": float(position_state.get("ramp_full_leverage", 0.0) or 0.0) if isinstance(position_state, dict) else 0.0,
            "ramped": bool(position_state.get("ramped", False)) if isinstance(position_state, dict) else False,
            "be_locked": bool(position_state.get("be_locked", False)) if isinstance(position_state, dict) else False,
            "entry_bar_timestamp": position_state.get("entry_bar_timestamp") if isinstance(position_state, dict) else None,
            "entry_candidate_timestamp": entry_candidate_ts,
            "route_score": context.route_score,
            "restored": True,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save_state({"position": saved_position, "last_candidate": position_state})

        payload = {
            "status": "submitted",
            "action": "restore_googl_usdt_long",
            "orders": orders,
            "amount": contracts,
            "notional_usdt": round(contracts * reference_price, 6),
            "leverage": leverage,
            "exchange_leverage": exchange_leverage,
            "stop_price": stop_price,
            "exchange_stop": exchange_stops[-1] if exchange_stops else None,
            "restore_source": "router_switch_rollback",
        }
        if exchange_stops:
            payload["exchange_stops"] = exchange_stops
        self.store.append_action("runtime", "RESTORE_GOOGL_USDT", payload)
        return payload

    def sync_leverage_setting(self, context: GooglOrderContext) -> dict[str, Any]:
        exchange_leverage = self._exchange_leverage()
        if self.router_config.mode == "paper":
            payload = {
                "status": "paper_synced",
                "action": "sync_googl_leverage_setting",
                "leverage": exchange_leverage,
                "exchange_leverage": exchange_leverage,
                "target_exposure_leverage": context.leverage,
            }
            self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "SYNC_GOOGL_LEVERAGE", payload)
            return payload
        try:
            response = self.client.set_leverage(
                int(round(exchange_leverage)),
                context.symbol,
                margin_mode=context.margin_mode,
                pos_side="long",
            )
        except Exception as exc:
            payload = {
                "status": "error",
                "action": "sync_googl_leverage_setting",
                "reason": "set_leverage_failed",
                "leverage": exchange_leverage,
                "exchange_leverage": exchange_leverage,
                "target_exposure_leverage": context.leverage,
                "error": str(exc),
            }
            self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "SYNC_GOOGL_LEVERAGE_FAILED", payload)
            return payload
        payload = {
            "status": "submitted",
            "action": "sync_googl_leverage_setting",
            "leverage": exchange_leverage,
            "exchange_leverage": exchange_leverage,
            "target_exposure_leverage": context.leverage,
            "response": response,
        }
        self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "SYNC_GOOGL_LEVERAGE", payload)
        return payload

    def rebalance_position(
        self,
        context: GooglOrderContext,
        exchange_position: dict[str, Any],
        current: dict[str, Any],
        *,
        target_notional: float,
    ) -> dict[str, Any]:
        current_notional = float(exchange_position.get("notional_usdt", 0.0) or 0.0)
        delta_notional = float(target_notional) - current_notional
        min_gap = float(self.router_config.googl_min_rebalance_notional_usdt)
        if abs(delta_notional) < min_gap:
            return {
                "status": "skipped",
                "action": "rebalance_googl_position",
                "reason": "notional_gap_too_small",
                "current_notional_usdt": current_notional,
                "target_notional_usdt": float(target_notional),
                "delta_notional_usdt": delta_notional,
                "leverage": context.leverage,
            }
        side = "buy" if delta_notional > 0 else "sell"
        exchange_leverage = self._exchange_leverage()
        if self.router_config.mode == "paper":
            amount = round(abs(delta_notional) / context.reference_price, 6) if context.reference_price > 0 else 0.0
            payload = {
                "status": "paper_rebalanced",
                "action": "rebalance_googl_position",
                "side": side,
                "amount": amount,
                "current_notional_usdt": round(current_notional, 6),
                "target_notional_usdt": round(float(target_notional), 6),
                "delta_notional_usdt": round(delta_notional, 6),
                "old_leverage": float(current.get("leverage", 0.0) or 0.0),
                "new_leverage": context.leverage,
                "target_exposure_leverage": context.leverage,
                "exchange_leverage": exchange_leverage,
            }
            self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "REBALANCE_GOOGL_POSITION", payload)
            return payload

        if side == "buy":
            leverage_result = self.sync_leverage_setting(context)
            if leverage_result.get("status") == "error":
                payload = {
                    "status": "error",
                    "action": "rebalance_googl_position",
                    "reason": "set_leverage_before_add_failed",
                    "leverage_result": leverage_result,
                    "current_notional_usdt": current_notional,
                    "target_notional_usdt": float(target_notional),
                    "delta_notional_usdt": delta_notional,
                }
                self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "REBALANCE_GOOGL_POSITION_FAILED", payload)
                return payload
            amount = self._order_amount(abs(delta_notional), context.reference_price)
            if amount <= 0:
                return {"status": "skipped", "action": "rebalance_googl_position", "reason": "non_positive_add_amount"}
            chunks = self._market_order_chunks(amount)
            orders = []
            exchange_stops = []
            for idx, chunk_amount in enumerate(chunks):
                params: dict[str, Any] = {"tdMode": context.margin_mode, "posSide": "long"}
                exchange_stop = self._attach_stop_to_order_params(params, context)
                order = self.client.create_order(context.symbol, "market", "buy", chunk_amount, params=params)
                orders.append({"order": order, "amount": chunk_amount})
                if exchange_stop is not None:
                    exchange_stops.append(self._with_order_id(exchange_stop, order))
                chunk_delay = self._market_order_chunk_delay_seconds()
                if idx < len(chunks) - 1 and chunk_delay > 0:
                    time.sleep(chunk_delay)
            payload = {
                "status": "submitted",
                "action": "rebalance_googl_position",
                "side": "buy",
                "amount": amount,
                "order": orders[0]["order"] if orders else None,
                "orders": orders,
                "current_notional_usdt": round(current_notional, 6),
                "target_notional_usdt": round(float(target_notional), 6),
                "delta_notional_usdt": round(delta_notional, 6),
                "old_leverage": float(current.get("leverage", 0.0) or 0.0),
                "new_leverage": context.leverage,
                "target_exposure_leverage": context.leverage,
                "exchange_leverage": exchange_leverage,
                "leverage_result": leverage_result,
                "exchange_stop": exchange_stops[-1] if exchange_stops else None,
            }
            if exchange_stops:
                payload["exchange_stops"] = exchange_stops
            self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "REBALANCE_GOOGL_POSITION", payload)
            return payload

        amount = self._order_amount(abs(delta_notional), context.reference_price)
        contracts = float(exchange_position.get("contracts", 0.0) or 0.0)
        amount = min(float(amount), contracts)
        if amount <= 0:
            return {"status": "skipped", "action": "rebalance_googl_position", "reason": "non_positive_reduce_amount"}
        chunks = self._close_order_chunks(amount, abs(delta_notional))
        orders = []
        for idx, chunk_amount in enumerate(chunks):
            order = self.client.create_order(
                context.symbol,
                "market",
                "sell",
                chunk_amount,
                params={"reduceOnly": True, "tdMode": context.margin_mode, "posSide": "long"},
            )
            orders.append({"order": order, "amount": chunk_amount})
            chunk_delay = self._sell_chunk_delay_seconds()
            if idx < len(chunks) - 1 and chunk_delay > 0:
                time.sleep(chunk_delay)
        leverage_result = self.sync_leverage_setting(context)
        status = "submitted_with_leverage_error" if leverage_result.get("status") == "error" else "submitted"
        payload = {
            "status": status,
            "action": "rebalance_googl_position",
            "side": "sell",
            "amount": amount,
            "order": orders[0]["order"] if orders else None,
            "orders": orders,
            "current_notional_usdt": round(current_notional, 6),
            "target_notional_usdt": round(float(target_notional), 6),
            "delta_notional_usdt": round(delta_notional, 6),
            "old_leverage": float(current.get("leverage", 0.0) or 0.0),
            "new_leverage": context.leverage,
            "target_exposure_leverage": context.leverage,
            "exchange_leverage": exchange_leverage,
            "leverage_result": leverage_result,
        }
        self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "REBALANCE_GOOGL_POSITION", payload)
        return payload

    def close_position(self, *, reason: str, exit_price: float | None = None, timestamp: str | None = None) -> dict[str, Any]:
        state_before = self.load_state()
        position_state = state_before.get("position") if isinstance(state_before.get("position"), dict) else None
        position = self.fetch_position_state()
        amount = float(position.get("contracts", 0.0) or 0.0)
        if amount <= 0:
            return {"status": "skipped", "reason": "no_open_googl_position"}
        if self.router_config.mode == "paper":
            self._shadow_gate_record_close(position_state, reason=reason, exit_price=exit_price, timestamp=timestamp)
            self.save_state({"position": None})
            return {"status": "paper_closed", "symbol": self.symbol, "amount": amount, "reason": reason}
        chunks = self._close_order_chunks(amount, float(position.get("notional_usdt", 0.0) or 0.0))
        orders = []
        for idx, chunk_amount in enumerate(chunks):
            order = self.client.create_order(
                self.symbol,
                "market",
                "sell",
                chunk_amount,
                params={"reduceOnly": True, "tdMode": self.router_config.googl_margin_mode, "posSide": "long"},
            )
            orders.append({"order": order, "amount": chunk_amount})
            chunk_delay = self._sell_chunk_delay_seconds()
            if idx < len(chunks) - 1 and chunk_delay > 0:
                time.sleep(chunk_delay)
        confirmed = self._wait_until_flat()
        status = "closed_confirmed" if confirmed["contracts"] <= 0 else "submitted_but_unconfirmed"
        payload = {
            "status": status,
            "action": "close_googl_usdt_long",
            "orders": orders,
            "amount": amount,
            "remaining_contracts": confirmed["contracts"],
            "remaining_notional_usdt": confirmed["notional_usdt"],
            "reason": reason,
        }
        self.store.append_action("runtime", "CLOSE_GOOGL_USDT", payload)
        if confirmed["contracts"] <= 0:
            self._shadow_gate_record_close(position_state, reason=reason, exit_price=exit_price, timestamp=timestamp)
            self.save_state({"position": None})
        return payload

    def sync_external_flat(
        self,
        *,
        reason: str,
        context: GooglOrderContext | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = state if isinstance(state, dict) else self.load_state()
        position_state = state.get("position") if isinstance(state.get("position"), dict) else None
        if position_state is None:
            return {"status": "skipped", "reason": "no_local_googl_position", "action": "sync_external_flat"}
        exit_price = None
        timestamp = None
        stop_price = float(position_state.get("stop_price", 0.0) or 0.0)
        latest_low = float(position_state.get("latest_low", 0.0) or 0.0)
        stop_like = False
        if context is not None:
            exit_price = context.stop_price if context.stop_price > 0 else context.reference_price
            timestamp = str(context.candidate.get("timestamp") or "runtime")
            stop_like = bool(stop_price > 0 and context.latest_low <= stop_price)
        else:
            exit_price = stop_price if stop_price > 0 else position_state.get("entry_price")
            timestamp = "runtime"
            stop_like = bool(stop_price > 0 and latest_low <= stop_price)
        close_reason = "googl_external_stop_sync" if stop_like else str(reason)
        self._shadow_gate_record_close(
            position_state,
            reason=close_reason,
            exit_price=float(exit_price or 0.0),
            timestamp=timestamp,
        )
        next_state = {
            "position": None,
            "last_candidate": context.candidate if context is not None else state.get("last_candidate"),
            "last_flat_event": {
                "candidate_timestamp": context.candidate.get("timestamp") if context is not None else position_state.get("entry_candidate_timestamp"),
                "reason": close_reason,
                "source_reason": reason,
                "stop_like": stop_like,
                "stop_price": stop_price,
                "latest_low": context.latest_low if context is not None else latest_low,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "source": "external_flat_sync",
            },
        }
        if stop_like:
            next_state["last_stop_hit"] = {
                "candidate_timestamp": next_state["last_flat_event"]["candidate_timestamp"],
                "stop_price": stop_price,
                "latest_low": context.latest_low if context is not None else latest_low,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "source": "external_flat_sync",
            }
        self.save_state(next_state)
        payload = {
            "status": "synced",
            "action": "sync_external_flat",
            "reason": close_reason,
            "source_reason": reason,
            "stop_like": stop_like,
            "stop_price": stop_price,
            "exit_price": float(exit_price or 0.0),
            "candidate_timestamp": next_state.get("last_stop_hit", {}).get("candidate_timestamp"),
        }
        self.store.append_action(str(timestamp or "runtime"), "SYNC_GOOGL_EXTERNAL_FLAT", payload)
        return payload

    def _close_order_chunks(self, amount: float, notional_usdt: float) -> list[float]:
        chunk_limits: list[float] = []
        if self.router_config.googl_max_close_order_contracts is not None:
            chunk_limits.append(float(self.router_config.googl_max_close_order_contracts))
        if self.router_config.googl_max_close_order_notional_usdt is not None and notional_usdt > 0:
            chunk_limits.append(float(amount) * float(self.router_config.googl_max_close_order_notional_usdt) / float(notional_usdt))
        chunk_limits.extend(self._market_order_chunk_limits())
        return self._amount_chunks(amount, chunk_limits)

    def _market_order_chunks(self, amount: float) -> list[float]:
        return self._amount_chunks(amount, self._market_order_chunk_limits())

    def _market_order_chunk_limits(self) -> list[float]:
        chunk_limits: list[float] = []
        if self.router_config.googl_max_market_order_contracts is not None:
            chunk_limits.append(float(self.router_config.googl_max_market_order_contracts))
        try:
            market = self._market()
            market_max = (((market.get("limits") or {}).get("amount") or {}).get("max"))
            if market_max not in (None, ""):
                chunk_limits.append(float(market_max))
            info = market.get("info") if isinstance(market.get("info"), dict) else {}
            max_market_size = info.get("maxMktSz")
            if max_market_size not in (None, ""):
                chunk_limits.append(float(max_market_size))
        except Exception:
            pass
        return chunk_limits

    def _amount_chunks(self, amount: float, chunk_limits: list[float]) -> list[float]:
        max_contracts = min(chunk_limits) if chunk_limits else None
        if max_contracts is None or float(max_contracts) <= 0 or float(max_contracts) >= float(amount):
            return [float(amount)]
        chunks = []
        remaining = float(amount)
        market = self._market()
        while remaining > 0:
            chunk = min(remaining, float(max_contracts))
            chunk = float(self._amount_to_precision(market, chunk))
            if chunk <= 0:
                break
            chunks.append(chunk)
            remaining = max(0.0, remaining - chunk)
        return chunks or [float(amount)]

    def _market_order_chunk_delay_seconds(self) -> float:
        configured = self.router_config.googl_market_order_chunk_delay_seconds
        if configured is not None:
            return max(0.0, float(configured))
        return max(0.0, float(self.router_config.googl_close_chunk_delay_seconds))

    def _sell_chunk_delay_seconds(self) -> float:
        return self._market_order_chunk_delay_seconds()

    def _wait_until_flat(self) -> dict[str, float]:
        deadline = time.time() + max(0.0, float(self.router_config.googl_close_confirm_timeout_seconds))
        poll = max(0.1, float(self.router_config.googl_close_confirm_poll_seconds))
        latest = self.fetch_position_state()
        while float(latest.get("contracts", 0.0) or 0.0) > 0 and time.time() < deadline:
            time.sleep(poll)
            latest = self.fetch_position_state()
        return {
            "contracts": float(latest.get("contracts", 0.0) or 0.0),
            "notional_usdt": float(latest.get("notional_usdt", 0.0) or 0.0),
        }

    def _attach_stop_to_order_params(self, params: dict[str, Any], context: GooglOrderContext) -> dict[str, Any] | None:
        if not bool(self.router_config.googl_enable_exchange_stop) or context.stop_price <= 0:
            return None
        market = self._market()
        attach_algo_client_id = self._generate_attach_algo_client_id()
        params["attachAlgoOrds"] = [
            {
                "slTriggerPx": self._price_to_precision(market, context.stop_price),
                "slOrdPx": "-1",
                "slTriggerPxType": "mark",
                "attachAlgoClOrdId": attach_algo_client_id,
            }
        ]
        return {
            "status": "client_id_pending",
            "algo_id": None,
            "algo_client_id": attach_algo_client_id,
            "stop_price": context.stop_price,
        }

    def _with_order_id(self, exchange_stop: dict[str, Any] | None, order: dict[str, Any]) -> dict[str, Any] | None:
        if exchange_stop is None:
            return None
        updated = dict(exchange_stop)
        order_id = self._extract_order_id(order)
        if order_id:
            updated["order_id"] = order_id
        return updated

    def update_trailing_stop(self, context: GooglOrderContext, exchange_position: dict[str, Any]) -> dict[str, Any] | None:
        state = self.load_state()
        current = state.get("position") if isinstance(state.get("position"), dict) else {}
        old_stop = float(current.get("stop_price", 0.0) or 0.0)
        local_stop_advanced = context.stop_price > old_stop
        exchange_stop: dict[str, Any] | None = None
        exchange_sync_needed = False
        if self.router_config.mode != "paper" and bool(self.router_config.googl_enable_exchange_stop):
            current_stop_fields = self._extract_exchange_stop_fields(exchange_position, current)
            stop_status = str(current_stop_fields.get("status") or "")
            stop_price_min = self._safe_float(current_stop_fields.get("stop_price_min"))
            stop_price_max = self._safe_float(current_stop_fields.get("stop_price_max"))
            tolerance = max(0.1, abs(float(context.stop_price)) * 0.001)
            exchange_sync_needed = stop_status not in {"found"} or (
                stop_price_min is not None and abs(float(context.stop_price) - stop_price_min) > tolerance
            ) or (
                stop_price_max is not None and abs(float(context.stop_price) - stop_price_max) > tolerance
            )
            if exchange_sync_needed:
                exchange_stop = self._amend_exchange_stop(context.stop_price, current, exchange_position)
            if exchange_stop is not None and exchange_stop.get("status") == "error":
                payload = {
                    "status": "error",
                    "action": "update_googl_trailing_stop",
                    "reason": "exchange_stop_amend_failed",
                    "old_stop_price": old_stop,
                    "new_stop_price": context.stop_price,
                    "exchange_contracts": exchange_position.get("contracts"),
                    "exchange_stop": exchange_stop,
                }
                self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "UPDATE_GOOGL_STOP_FAILED", payload)
                return payload
        if not local_stop_advanced and not exchange_sync_needed:
            return None
        payload = {
            "status": "tracked" if local_stop_advanced else "synced",
            "action": "update_googl_trailing_stop",
            "old_stop_price": old_stop,
            "new_stop_price": context.stop_price,
            "exchange_contracts": exchange_position.get("contracts"),
        }
        if exchange_stop is not None:
            payload["exchange_stop"] = exchange_stop
        self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "UPDATE_GOOGL_STOP", payload)
        return payload

    def _with_exchange_stop_fields(
        self,
        position: dict[str, Any] | None,
        fields: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if position is None or not fields:
            return position
        updated = dict(position)
        for source_key, state_key in (
            ("order_id", "exchange_order_id"),
            ("algo_id", "exchange_attach_algo_id"),
            ("algo_client_id", "exchange_attach_algo_client_id"),
        ):
            value = fields.get(source_key)
            if value:
                updated[state_key] = value
        return updated

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _generate_attach_algo_client_id(self) -> str:
        return f"googls{uuid.uuid4().hex[:26]}"

    def _extract_attached_algo_identity(self, position_state: dict[str, Any] | None) -> dict[str, str | None]:
        if not isinstance(position_state, dict):
            return {"algo_id": None, "algo_client_id": None}
        close_order_algos = position_state.get("close_order_algos") or []
        if not close_order_algos:
            raw = position_state.get("raw")
            info = raw.get("info") if isinstance(raw, dict) and isinstance(raw.get("info"), dict) else {}
            close_order_algos = info.get("closeOrderAlgo") if isinstance(info.get("closeOrderAlgo"), list) else []
        for algo in close_order_algos:
            if not isinstance(algo, dict):
                continue
            algo_id = algo.get("attachAlgoId") or algo.get("algoId")
            algo_client_id = (
                algo.get("attachAlgoClOrdId")
                or algo.get("algoClOrdId")
                or algo.get("slAttachAlgoClOrdId")
            )
            if algo_id or algo_client_id:
                return {
                    "algo_id": str(algo_id) if algo_id else None,
                    "algo_client_id": str(algo_client_id) if algo_client_id else None,
                }
        return {"algo_id": None, "algo_client_id": None}

    def _fetch_pending_algo_orders(self) -> list[dict[str, Any]]:
        orders: list[dict[str, Any]] = []
        errors: list[str] = []
        seen: set[tuple[str, str]] = set()
        for order_type in ("oco", "conditional"):
            try:
                response = self.client.fetch_pending_algo_orders({"ordType": order_type})
            except Exception as exc:
                errors.append(f"{order_type}: {exc}")
                continue
            data = response.get("data") if isinstance(response, dict) else None
            if not isinstance(data, list):
                continue
            for order in data:
                if not isinstance(order, dict):
                    continue
                identity = (str(order.get("ordType") or order_type), str(order.get("algoId") or order.get("algoClOrdId") or id(order)))
                if identity in seen:
                    continue
                seen.add(identity)
                orders.append(order)
        if not orders and errors:
            raise RuntimeError("; ".join(errors))
        return orders

    def _select_pending_algo_orders(self, local_position: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        pending_orders = self._fetch_pending_algo_orders()
        market_id = self._market().get("id")
        local_algo_id = str((local_position or {}).get("exchange_attach_algo_id") or "")
        local_algo_client_id = str((local_position or {}).get("exchange_attach_algo_client_id") or "")
        candidates = []
        for order in pending_orders:
            if not isinstance(order, dict):
                continue
            if market_id and order.get("instId") != market_id:
                continue
            if order.get("ordType") not in {"oco", "conditional"}:
                continue
            if order.get("state") not in {"live", "effective"}:
                continue
            if order.get("posSide") not in (None, "", "long"):
                continue
            if order.get("side") not in (None, "", "sell"):
                continue
            reduce_only = order.get("reduceOnly")
            if reduce_only not in (None, "", "true", True):
                continue
            candidates.append(order)
        if not candidates or (not local_algo_id and not local_algo_client_id):
            return candidates
        matched = []
        unmatched = []
        for order in candidates:
            if local_algo_id and str(order.get("algoId") or "") == local_algo_id:
                matched.append(order)
                continue
            if local_algo_client_id and str(order.get("algoClOrdId") or "") == local_algo_client_id:
                matched.append(order)
                continue
            unmatched.append(order)
        return matched + unmatched

    def _select_pending_algo_order(self, local_position: dict[str, Any] | None = None) -> dict[str, Any] | None:
        candidates = self._select_pending_algo_orders(local_position)
        return candidates[0] if candidates else None

    def _extract_exchange_stop_orders(
        self,
        exchange_position: dict[str, Any],
        local_position: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        orders: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        def _append_order(
            *,
            algo_id: Any,
            algo_client_id: Any,
            stop_price: Any,
            order_type: Any,
            state: Any,
            size: Any,
            raw: dict[str, Any],
        ) -> None:
            normalized_algo_id = str(algo_id) if algo_id else None
            normalized_algo_client_id = str(algo_client_id) if algo_client_id else None
            normalized_type = str(order_type or "")
            key = (normalized_algo_id or "", normalized_algo_client_id or "", normalized_type)
            if key in seen:
                return
            seen.add(key)
            orders.append(
                {
                    "algo_id": normalized_algo_id,
                    "algo_client_id": normalized_algo_client_id,
                    "stop_price": self._safe_float(stop_price),
                    "ord_type": normalized_type or None,
                    "state": str(state) if state not in (None, "") else None,
                    "size": self._safe_float(size),
                    "raw": raw,
                }
            )

        close_order_algos = exchange_position.get("close_order_algos") if isinstance(exchange_position, dict) else None
        if not isinstance(close_order_algos, list):
            close_order_algos = []
        raw = exchange_position.get("raw") if isinstance(exchange_position, dict) else None
        info = raw.get("info") if isinstance(raw, dict) and isinstance(raw.get("info"), dict) else {}
        if not close_order_algos:
            close_order_algos = info.get("closeOrderAlgo") if isinstance(info.get("closeOrderAlgo"), list) else []
        for algo in close_order_algos:
            if not isinstance(algo, dict):
                continue
            _append_order(
                algo_id=algo.get("attachAlgoId") or algo.get("algoId"),
                algo_client_id=(
                    algo.get("attachAlgoClOrdId")
                    or algo.get("algoClOrdId")
                    or algo.get("slAttachAlgoClOrdId")
                ),
                stop_price=algo.get("slTriggerPx") or algo.get("triggerPx"),
                order_type=algo.get("ordType") or "oco",
                state=algo.get("state"),
                size=algo.get("sz"),
                raw=algo,
            )

        try:
            pending_orders = self._select_pending_algo_orders(local_position)
        except Exception:
            pending_orders = []
        for order in pending_orders:
            if not isinstance(order, dict):
                continue
            _append_order(
                algo_id=order.get("algoId"),
                algo_client_id=order.get("algoClOrdId"),
                stop_price=order.get("slTriggerPx") or order.get("triggerPx"),
                order_type=order.get("ordType"),
                state=order.get("state"),
                size=order.get("sz"),
                raw=order,
            )

        return orders

    def _extract_exchange_stop_fields(
        self,
        exchange_position: dict[str, Any],
        local_position: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        orders = self._extract_exchange_stop_orders(exchange_position, local_position)
        if not orders:
            return {
                "status": "not_found",
                "algo_id": None,
                "algo_client_id": None,
                "stop_price": None,
                "order_count": 0,
                "orders": [],
            }

        local_algo_id = str((local_position or {}).get("exchange_attach_algo_id") or "")
        local_algo_client_id = str((local_position or {}).get("exchange_attach_algo_client_id") or "")
        primary = next(
            (
                order
                for order in orders
                if (local_algo_id and order.get("algo_id") == local_algo_id)
                or (local_algo_client_id and order.get("algo_client_id") == local_algo_client_id)
            ),
            None,
        )
        if primary is None:
            primary = max(
                orders,
                key=lambda item: (
                    item.get("stop_price") is not None,
                    float(item.get("stop_price") or 0.0),
                ),
            )
        stop_prices = [float(item["stop_price"]) for item in orders if item.get("stop_price") is not None]
        status = "found"
        if not stop_prices:
            status = "missing_price"
        elif len({round(price, 8) for price in stop_prices}) > 1:
            status = "diverged"
        return {
            "status": status,
            "algo_id": primary.get("algo_id"),
            "algo_client_id": primary.get("algo_client_id"),
            "stop_price": primary.get("stop_price"),
            "order_count": len(orders),
            "stop_price_min": min(stop_prices) if stop_prices else None,
            "stop_price_max": max(stop_prices) if stop_prices else None,
            "orders": orders,
        }

    def _refresh_exchange_stop_identity(
        self,
        *,
        attach_algo_client_id: str | None = None,
        order_id: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "disabled" if not bool(self.router_config.googl_enable_exchange_stop) else "not_found",
            "order_id": order_id,
            "algo_id": None,
            "algo_client_id": attach_algo_client_id,
        }
        if self.router_config.mode == "paper" or not bool(self.router_config.googl_enable_exchange_stop):
            return result
        try:
            position = self.fetch_position_state()
            identity = self._extract_attached_algo_identity(position)
            if not identity["algo_id"] and not identity["algo_client_id"]:
                pending = self._select_pending_algo_order()
                if pending:
                    identity = {
                        "algo_id": str(pending.get("algoId")) if pending.get("algoId") else None,
                        "algo_client_id": str(pending.get("algoClOrdId")) if pending.get("algoClOrdId") else attach_algo_client_id,
                    }
            result.update(
                {
                    "status": "found" if identity["algo_id"] or identity["algo_client_id"] else "not_found",
                    "order_id": order_id or self._extract_order_id(position.get("raw")),
                    "algo_id": identity["algo_id"],
                    "algo_client_id": identity["algo_client_id"] or attach_algo_client_id,
                }
            )
            if result["status"] == "not_found" and attach_algo_client_id:
                result["status"] = "client_id_pending"
        except Exception as exc:
            result.update({"status": "error", "error": str(exc)})
        return result

    def _extract_order_id(self, order_or_position: dict[str, Any] | None) -> str | None:
        if not isinstance(order_or_position, dict):
            return None
        for key in ("id", "order", "ordId"):
            value = order_or_position.get(key)
            if value:
                return str(value)
        info = order_or_position.get("info")
        if isinstance(info, dict):
            for key in ("ordId", "orderId"):
                value = info.get(key)
                if value:
                    return str(value)
        return None

    def _build_algo_amend_request(
        self,
        *,
        attach_algo_id: str | None,
        attach_algo_client_id: str | None,
        stop_price: float,
    ) -> dict[str, Any]:
        market = self._market()
        request = {
            "instId": self._market()["id"],
            "newSlTriggerPx": self._price_to_precision(market, stop_price),
            "newSlOrdPx": "-1",
            "newSlTriggerPxType": "mark",
        }
        if attach_algo_id:
            request["algoId"] = attach_algo_id
        elif attach_algo_client_id:
            request["algoClOrdId"] = attach_algo_client_id
        else:
            raise ValueError("Missing GOOGL attached stop identifier")
        return request

    def _build_conditional_algo_amend_request(
        self,
        *,
        attach_algo_id: str | None,
        attach_algo_client_id: str | None,
        stop_price: float,
    ) -> dict[str, Any]:
        market = self._market()
        request = {
            "instId": market["id"],
            "newTriggerPx": self._price_to_precision(market, stop_price),
            "newOrdPx": "-1",
            "newTriggerPxType": "mark",
        }
        if attach_algo_id:
            request["algoId"] = attach_algo_id
        elif attach_algo_client_id:
            request["algoClOrdId"] = attach_algo_client_id
        else:
            raise ValueError("Missing GOOGL conditional stop identifier")
        return request

    def _amend_exchange_stop(
        self,
        stop_price: float,
        local_position: dict[str, Any],
        exchange_position: dict[str, Any],
    ) -> dict[str, Any]:
        stop_orders = self._extract_exchange_stop_orders(exchange_position, local_position)
        if not stop_orders:
            identity = self._extract_attached_algo_identity(exchange_position)
            attach_algo_id = identity["algo_id"] or local_position.get("exchange_attach_algo_id")
            attach_algo_client_id = identity["algo_client_id"] or local_position.get("exchange_attach_algo_client_id")
            if not attach_algo_id and not attach_algo_client_id:
                return {"status": "error", "reason": "missing_exchange_stop_identifier"}
            stop_orders = [
                {
                    "algo_id": attach_algo_id,
                    "algo_client_id": attach_algo_client_id,
                    "ord_type": None,
                }
            ]

        amended_orders = []
        for stop_order in stop_orders:
            attach_algo_id = stop_order.get("algo_id")
            attach_algo_client_id = stop_order.get("algo_client_id")
            if not attach_algo_id and not attach_algo_client_id:
                continue
            request = self._build_algo_amend_request(
                attach_algo_id=attach_algo_id,
                attach_algo_client_id=attach_algo_client_id,
                stop_price=stop_price,
            )
            try:
                response = self.client.amend_algo_order(request)
            except Exception as exc:
                primary_error = str(exc)
                fallback_request = self._build_conditional_algo_amend_request(
                    attach_algo_id=attach_algo_id,
                    attach_algo_client_id=attach_algo_client_id,
                    stop_price=stop_price,
                )
                try:
                    response = self.client.amend_algo_order(fallback_request)
                    request = fallback_request
                except Exception as fallback_exc:
                    return {
                        "status": "error",
                        "reason": "amend_algo_order_failed",
                        "error": str(fallback_exc),
                        "primary_error": primary_error,
                        "pending_order_type": stop_order.get("ord_type"),
                        "request": fallback_request,
                        "failed_algo_id": attach_algo_id,
                        "failed_algo_client_id": attach_algo_client_id,
                        "amended_orders": amended_orders,
                    }
            amended_orders.append(
                {
                    "algo_id": attach_algo_id,
                    "algo_client_id": attach_algo_client_id,
                    "stop_price": stop_price,
                    "request": request,
                    "response": response,
                    "ord_type": stop_order.get("ord_type"),
                }
            )

        if not amended_orders:
            return {"status": "error", "reason": "missing_exchange_stop_identifier"}

        local_algo_id = str(local_position.get("exchange_attach_algo_id") or "")
        local_algo_client_id = str(local_position.get("exchange_attach_algo_client_id") or "")
        primary = next(
            (
                order
                for order in amended_orders
                if (local_algo_id and order.get("algo_id") == local_algo_id)
                or (local_algo_client_id and order.get("algo_client_id") == local_algo_client_id)
            ),
            amended_orders[0],
        )
        return {
            "status": "amended",
            "algo_id": primary.get("algo_id"),
            "algo_client_id": primary.get("algo_client_id"),
            "stop_price": stop_price,
            "request": primary.get("request"),
            "response": primary.get("response"),
            "orders": amended_orders,
            "amended_count": len(amended_orders),
        }
