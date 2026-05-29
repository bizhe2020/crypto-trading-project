from __future__ import annotations

import json
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from bot.okx_client import OkxClient, OkxCredentials
from bot.state_store import StateStore
from bot.strategy_router import RoutedSignalCandidate, StrategyRouterConfig
from scripts.scan_qqq_usdt_4h_triggers import attach_daily_state, load_okx_4h, load_signal_path
from scripts.scan_qqq_usdt_aggressive_grid import AGGRESSIVE_LEVERAGE_PROFILES
from scripts.scan_qqq_usdt_btc_logic_overlays import enrich_bars


ROOT = Path(__file__).resolve().parents[1]


class PartialOrderSubmissionError(RuntimeError):
    def __init__(self, message: str, *, orders: list[dict[str, Any]], failed_chunk: float):
        super().__init__(message)
        self.orders = orders
        self.failed_chunk = failed_chunk


@dataclass
class QqqOrderContext:
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


class QqqUsdtExecutionEngine:
    def __init__(self, router_config: StrategyRouterConfig, qqq_config_path: str | Path):
        self.router_config = router_config
        self.qqq_config_path = Path(qqq_config_path).resolve()
        self.qqq_config = json.loads(self.qqq_config_path.read_text())
        credentials = self._load_credentials()
        self.client = OkxClient(credentials, trading_mode=router_config.mode, proxy=self._proxy())
        self.store = StateStore(router_config.qqq_state_db_path)
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
            else:
                self._markets_cache = self.client.load_markets()
        return self._markets_cache

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
        symbol = str(self.qqq_config["execution_symbol"])
        market = payload.get(symbol)
        if isinstance(market, dict):
            return {symbol: market}
        return {key: value for key, value in payload.items() if isinstance(value, dict)}

    def _market(self) -> dict[str, Any]:
        symbol = str(self.qqq_config["execution_symbol"])
        market = self._load_markets().get(symbol)
        if market is None:
            raise ValueError(f"Market metadata missing for {symbol}")
        return market

    def bootstrap(self) -> dict[str, Any]:
        symbol = str(self.qqq_config["execution_symbol"])
        leverage = int(round(float(self.qqq_config["offense_leverage"])))
        error = None
        market_loaded = False
        try:
            market_loaded = bool(self._load_markets_cache())
            if self.router_config.mode != "paper":
                self._load_markets()
                market_loaded = True
                self.client.set_leverage(leverage, symbol, margin_mode=self.router_config.qqq_margin_mode, pos_side="long")
        except Exception as exc:
            error = str(exc)
        payload = {
            "status": "ok" if error is None else "error",
            "symbol": symbol,
            "market_loaded": market_loaded,
            "market_cache": str(self.router_config.okx_markets_cache_path),
            "leverage": leverage,
            "error": error,
        }
        self.store.append_action("bootstrap", "BOOTSTRAP", payload)
        return payload

    def evaluate_latest(self, candidate: RoutedSignalCandidate | None) -> dict[str, Any]:
        context = self._build_context(candidate) if candidate is not None and candidate.active else None
        exchange_position = self.fetch_position_state()
        state = self.load_state()
        actions: list[dict[str, Any]] = []

        if context is None:
            if exchange_position["contracts"] > 0:
                actions.append(self.close_position(reason="router_no_qqq_signal"))
            self.save_state({"position": None, "last_candidate": candidate.to_dict() if candidate else None})
            return {"status": "ok", "symbol": self.symbol, "actions": actions, "position_open": False}

        if context.stop_hit:
            if exchange_position["contracts"] > 0:
                actions.append(self.close_position(reason="qqq_trailing_stop_hit"))
            else:
                actions.append({"status": "skipped", "reason": "qqq_stop_hit_no_exchange_position"})
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
                }
            )
            return {"status": "ok", "symbol": self.symbol, "actions": actions, "position_open": False}

        if exchange_position["contracts"] > 0 and not isinstance(state.get("position"), dict):
            sync_result = self.sync_existing_exchange_position(context, exchange_position)
            actions.append(sync_result)
            self.save_state(
                {
                    "position": sync_result.get("position"),
                    "last_candidate": context.candidate,
                }
            )
            return {"status": "ok", "symbol": self.symbol, "actions": actions, "position_open": True}

        if exchange_position["contracts"] <= 0:
            if self._same_signal_stop_locked(context):
                actions.append(
                    {
                        "status": "skipped",
                        "reason": "qqq_stop_hit_same_signal_lock",
                        "candidate_timestamp": context.candidate.get("timestamp"),
                    }
                )
                return {"status": "ok", "symbol": self.symbol, "actions": actions, "position_open": False}
            open_result = self.open_position(context)
            actions.append(open_result)
            position_open = open_result.get("status") in {"paper_opened", "submitted"}
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
        notional_gap = abs(float(exchange_position["notional_usdt"]) - target_notional)
        leverage_changed = abs(current_leverage - float(context.leverage)) > 1e-9
        should_rebalance = (
            (
                bool(self.router_config.qqq_rebalance_on_leverage_change)
                and leverage_changed
            )
            or bool(getattr(self.router_config, "qqq_rebalance_on_notional_gap", False))
        ) and (
            notional_gap >= float(self.router_config.qqq_min_rebalance_notional_usdt)
        )
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
            position_open = True
        elif leverage_changed:
            actions.append(
                {
                    "status": "skipped",
                    "action": "rebalance_qqq_position",
                    "reason": "notional_gap_too_small",
                    "current_leverage": current_leverage,
                    "target_leverage": context.leverage,
                    "current_notional_usdt": float(exchange_position["notional_usdt"]),
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
        return str(self.qqq_config["execution_symbol"])

    def load_state(self) -> dict[str, Any]:
        raw = self.store.get_value("qqq_usdt_state")
        if not raw:
            return {}
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}

    def save_state(self, payload: dict[str, Any]) -> None:
        self.store.set_value("qqq_usdt_state", json.dumps(payload, ensure_ascii=False, indent=2))

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (ROOT / value).resolve()

    def _latest_bar(self) -> pd.Series:
        signal_source = self._resolve_path(str(self.qqq_config["signal_source"]))
        data_4h = self._resolve_path(str(self.qqq_config["data_4h"]))
        _, signal_path = load_signal_path(signal_source)
        bars = enrich_bars(attach_daily_state(load_okx_4h(data_4h), signal_path))
        if bars.empty:
            raise RuntimeError("No QQQ/USDT bars available")
        return bars.iloc[-1]

    def _build_context(self, candidate: RoutedSignalCandidate) -> QqqOrderContext:
        latest = self._latest_bar()
        reference_price = float(latest["close"])
        latest_low = float(latest["low"])
        leverage = float(candidate.leverage or self.qqq_config["base_leverage"])
        stop_loss_pct = float(self.qqq_config["stop_loss_pct"])
        state = self.load_state()
        previous = state.get("position") if isinstance(state.get("position"), dict) else {}
        previous_stop = float(previous.get("stop_price", 0.0) or 0.0)
        peak_price = max(float(previous.get("peak_price", 0.0) or 0.0), reference_price)
        stop_price = max(previous_stop, peak_price * (1.0 - stop_loss_pct / 100.0))
        candidate_payload = candidate.to_dict()
        entry_timestamp = previous.get("entry_candidate_timestamp")
        candidate_timestamp = candidate_payload.get("timestamp")
        same_entry_bar = bool(entry_timestamp and candidate_timestamp and str(entry_timestamp) == str(candidate_timestamp))
        stop_hit = bool(previous_stop > 0 and latest_low <= previous_stop and not same_entry_bar)
        return QqqOrderContext(
            symbol=self.symbol,
            margin_mode=str(self.router_config.qqq_margin_mode),
            leverage=leverage,
            stop_loss_pct=stop_loss_pct,
            reference_price=reference_price,
            latest_low=latest_low,
            stop_price=stop_price,
            stop_hit=stop_hit,
            route_score=float(candidate.route_score),
            candidate=candidate_payload,
        )

    def _state_from_context(self, context: QqqOrderContext) -> dict[str, Any]:
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
            "entry_candidate_timestamp": previous.get("entry_candidate_timestamp") or context.candidate.get("timestamp"),
            "route_score": context.route_score,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        for key in ("exchange_order_id", "exchange_attach_algo_id", "exchange_attach_algo_client_id"):
            if previous.get(key):
                payload[key] = previous[key]
        return payload

    def _same_signal_stop_locked(self, context: QqqOrderContext) -> bool:
        state = self.load_state()
        last_stop_hit = state.get("last_stop_hit") if isinstance(state.get("last_stop_hit"), dict) else {}
        stop_timestamp = last_stop_hit.get("candidate_timestamp")
        candidate_timestamp = context.candidate.get("timestamp")
        return bool(stop_timestamp and candidate_timestamp and str(stop_timestamp) == str(candidate_timestamp))

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

    def _extract_total_usdt(self, balance: dict[str, Any]) -> float:
        usdt = balance.get("USDT") if isinstance(balance, dict) else None
        if isinstance(usdt, dict):
            for key in ("total", "equity", "cash", "free"):
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
                    for key in ("eq", "cashBal", "availEq", "availBal"):
                        value = detail.get(key)
                        if value not in (None, ""):
                            numeric = float(value)
                            if numeric > 0:
                                return numeric
        raise ValueError("Unable to extract positive total USDT balance")

    def _sizing_usdt(self, balance: dict[str, Any]) -> float:
        basis = str(getattr(self.router_config, "qqq_sizing_basis", "available") or "available").lower()
        if basis in {"total", "equity", "total_equity"}:
            return self._extract_total_usdt(balance)
        return self._extract_available_usdt(balance)

    def _target_notional(self, context: QqqOrderContext) -> float:
        if self.router_config.mode == "paper":
            sizing_usdt = 1000.0
        else:
            sizing_usdt = self._sizing_usdt(self.client.fetch_balance())
        notional = sizing_usdt * float(self.router_config.qqq_position_size_pct) * float(context.leverage)
        if self.router_config.qqq_max_notional_usdt is not None:
            notional = min(notional, float(self.router_config.qqq_max_notional_usdt))
        return max(notional, 0.0)

    def _rebalance_target_notional(
        self,
        context: QqqOrderContext,
        exchange_position: dict[str, Any],
        current: dict[str, Any],
    ) -> float:
        current_notional = float(exchange_position.get("notional_usdt", 0.0) or 0.0)
        current_leverage = float(current.get("leverage", 0.0) or 0.0)
        if (
            current_notional > 0
            and current_leverage > 0
            and not bool(getattr(self.router_config, "qqq_rebalance_on_notional_gap", False))
        ):
            target = current_notional * float(context.leverage) / current_leverage
        else:
            target = self._target_notional(context)
        if self.router_config.qqq_max_notional_usdt is not None:
            target = min(target, float(self.router_config.qqq_max_notional_usdt))
        return max(float(target), 0.0)

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

    def _build_context_from_state(self, position: dict[str, Any]) -> QqqOrderContext:
        return QqqOrderContext(
            symbol=self.symbol,
            margin_mode=str(self.router_config.qqq_margin_mode),
            leverage=float(position.get("leverage", self.qqq_config["base_leverage"]) or self.qqq_config["base_leverage"]),
            stop_loss_pct=float(position.get("stop_loss_pct", self.qqq_config["stop_loss_pct"]) or self.qqq_config["stop_loss_pct"]),
            reference_price=float(position.get("peak_price", 1.0) or 1.0),
            latest_low=float(position.get("latest_low", position.get("peak_price", 1.0)) or 1.0),
            stop_price=float(position.get("stop_price", 0.0) or 0.0),
            stop_hit=False,
            route_score=float(position.get("route_score", 0.0) or 0.0),
            candidate={},
        )

    def open_position(self, context: QqqOrderContext) -> dict[str, Any]:
        notional = self._target_notional(context)
        if notional < float(self.router_config.qqq_min_order_notional_usdt):
            return {"status": "skipped", "reason": "notional_too_small", "notional_usdt": notional}
        amount = round(notional / context.reference_price, 6) if self.router_config.mode == "paper" else self._order_amount(notional, context.reference_price)
        if amount <= 0:
            return {"status": "error", "reason": "non_positive_amount", "notional_usdt": notional}
        if self.router_config.mode == "paper":
            return {
                "status": "paper_opened",
                "symbol": context.symbol,
                "amount": amount,
                "notional_usdt": round(notional, 6),
                "leverage": context.leverage,
                "stop_price": context.stop_price,
            }
        self.client.set_leverage(int(round(context.leverage)), context.symbol, margin_mode=context.margin_mode, pos_side="long")
        params: dict[str, Any] = {"tdMode": context.margin_mode, "posSide": "long"}
        attach_algo_client_ids: list[str] = []
        try:
            params_factory = None
            if bool(self.router_config.qqq_enable_exchange_stop) and context.stop_price > 0:
                params_factory = self._chunk_params_factory(context, generated_client_ids=attach_algo_client_ids)
            orders = self._submit_market_orders(
                context.symbol,
                "buy",
                amount,
                params=params,
                params_factory=params_factory,
            )
        except PartialOrderSubmissionError as exc:
            exchange_position = self.fetch_position_state()
            payload = {
                "status": "partial_submitted_error",
                "action": "open_qqq_usdt_long",
                "reason": "partial_order_submission_failed",
                "error": str(exc),
                "failed_chunk": exc.failed_chunk,
                "orders": exc.orders,
                "amount": amount,
                "notional_usdt": round(notional, 6),
                "leverage": context.leverage,
                "stop_price": context.stop_price,
                "exchange_position": exchange_position,
            }
            self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "OPEN_QQQ_USDT_PARTIAL_FAILED", payload)
            return payload
        order = orders[-1] if orders else {}
        attach_algo_client_id = attach_algo_client_ids[0] if attach_algo_client_ids else None
        exchange_stop = self._refresh_exchange_stop_identity(
            attach_algo_client_id=attach_algo_client_id,
            attach_algo_client_ids=attach_algo_client_ids,
            order_id=self._extract_order_id(order),
        )
        payload = {
            "status": "submitted",
            "action": "open_qqq_usdt_long",
            "order": order,
            "orders": orders,
            "amount": amount,
            "notional_usdt": round(notional, 6),
            "leverage": context.leverage,
            "stop_price": context.stop_price,
            "exchange_stop": exchange_stop,
        }
        self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "OPEN_QQQ_USDT", payload)
        return payload

    def sync_leverage_setting(self, context: QqqOrderContext) -> dict[str, Any]:
        if self.router_config.mode == "paper":
            payload = {
                "status": "paper_synced",
                "action": "sync_qqq_leverage_setting",
                "leverage": context.leverage,
            }
            self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "SYNC_QQQ_LEVERAGE", payload)
            return payload
        try:
            response = self.client.set_leverage(
                int(round(context.leverage)),
                context.symbol,
                margin_mode=context.margin_mode,
                pos_side="long",
            )
        except Exception as exc:
            payload = {
                "status": "error",
                "action": "sync_qqq_leverage_setting",
                "reason": "set_leverage_failed",
                "leverage": context.leverage,
                "error": str(exc),
            }
            self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "SYNC_QQQ_LEVERAGE_FAILED", payload)
            return payload
        payload = {
            "status": "submitted",
            "action": "sync_qqq_leverage_setting",
            "leverage": context.leverage,
            "response": response,
        }
        self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "SYNC_QQQ_LEVERAGE", payload)
        return payload

    def rebalance_position(
        self,
        context: QqqOrderContext,
        exchange_position: dict[str, Any],
        current: dict[str, Any],
        *,
        target_notional: float,
    ) -> dict[str, Any]:
        current_notional = float(exchange_position.get("notional_usdt", 0.0) or 0.0)
        delta_notional = float(target_notional) - current_notional
        min_gap = float(self.router_config.qqq_min_rebalance_notional_usdt)
        if abs(delta_notional) < min_gap:
            return {
                "status": "skipped",
                "action": "rebalance_qqq_position",
                "reason": "notional_gap_too_small",
                "current_notional_usdt": current_notional,
                "target_notional_usdt": float(target_notional),
                "delta_notional_usdt": delta_notional,
                "leverage": context.leverage,
            }
        side = "buy" if delta_notional > 0 else "sell"
        if self.router_config.mode == "paper":
            amount = round(abs(delta_notional) / context.reference_price, 6) if context.reference_price > 0 else 0.0
            payload = {
                "status": "paper_rebalanced",
                "action": "rebalance_qqq_position",
                "side": side,
                "amount": amount,
                "current_notional_usdt": round(current_notional, 6),
                "target_notional_usdt": round(float(target_notional), 6),
                "delta_notional_usdt": round(delta_notional, 6),
                "old_leverage": float(current.get("leverage", 0.0) or 0.0),
                "new_leverage": context.leverage,
            }
            self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "REBALANCE_QQQ_POSITION", payload)
            return payload

        if side == "buy":
            leverage_result = self.sync_leverage_setting(context)
            if leverage_result.get("status") == "error":
                payload = {
                    "status": "error",
                    "action": "rebalance_qqq_position",
                    "reason": "set_leverage_before_add_failed",
                    "leverage_result": leverage_result,
                    "current_notional_usdt": current_notional,
                    "target_notional_usdt": float(target_notional),
                    "delta_notional_usdt": delta_notional,
                }
                self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "REBALANCE_QQQ_POSITION_FAILED", payload)
                return payload
            amount = self._order_amount(abs(delta_notional), context.reference_price)
            if amount <= 0:
                return {"status": "skipped", "action": "rebalance_qqq_position", "reason": "non_positive_add_amount"}
            params: dict[str, Any] = {"tdMode": context.margin_mode, "posSide": "long"}
            try:
                attach_algo_client_ids = []
                params_factory = None
                if bool(self.router_config.qqq_enable_exchange_stop) and context.stop_price > 0:
                    params_factory = self._chunk_params_factory(context, generated_client_ids=attach_algo_client_ids)
                orders = self._submit_market_orders(
                    context.symbol,
                    "buy",
                    amount,
                    params=params,
                    params_factory=params_factory,
                )
            except PartialOrderSubmissionError as exc:
                refreshed_position = self.fetch_position_state()
                payload = {
                    "status": "partial_submitted_error",
                    "action": "rebalance_qqq_position",
                    "reason": "partial_order_submission_failed",
                    "error": str(exc),
                    "failed_chunk": exc.failed_chunk,
                    "side": "buy",
                    "orders": exc.orders,
                    "current_notional_usdt": round(current_notional, 6),
                    "target_notional_usdt": round(float(target_notional), 6),
                    "delta_notional_usdt": round(delta_notional, 6),
                    "old_leverage": float(current.get("leverage", 0.0) or 0.0),
                    "new_leverage": context.leverage,
                    "leverage_result": leverage_result,
                    "exchange_position": refreshed_position,
                }
                self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "REBALANCE_QQQ_POSITION_PARTIAL_FAILED", payload)
                return payload
            order = orders[-1] if orders else {}
            exchange_stop = self._refresh_exchange_stop_identity(
                attach_algo_client_id=attach_algo_client_ids[0] if attach_algo_client_ids else None,
                attach_algo_client_ids=attach_algo_client_ids,
                order_id=self._extract_order_id(order),
            )
            payload = {
                "status": "submitted",
                "action": "rebalance_qqq_position",
                "side": "buy",
                "amount": amount,
                "order": order,
                "orders": orders,
                "current_notional_usdt": round(current_notional, 6),
                "target_notional_usdt": round(float(target_notional), 6),
                "delta_notional_usdt": round(delta_notional, 6),
                "old_leverage": float(current.get("leverage", 0.0) or 0.0),
                "new_leverage": context.leverage,
                "leverage_result": leverage_result,
                "exchange_stop": self._with_order_id(exchange_stop, order),
            }
            self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "REBALANCE_QQQ_POSITION", payload)
            return payload

        amount = self._order_amount(abs(delta_notional), context.reference_price)
        contracts = float(exchange_position.get("contracts", 0.0) or 0.0)
        amount = min(float(amount), contracts)
        if amount <= 0:
            return {"status": "skipped", "action": "rebalance_qqq_position", "reason": "non_positive_reduce_amount"}
        orders = self._submit_market_orders(
            context.symbol,
            "sell",
            amount,
            params={"reduceOnly": True, "tdMode": context.margin_mode, "posSide": "long"},
        )
        order = orders[-1] if orders else {}
        leverage_result = self.sync_leverage_setting(context)
        status = "submitted_with_leverage_error" if leverage_result.get("status") == "error" else "submitted"
        payload = {
            "status": status,
            "action": "rebalance_qqq_position",
            "side": "sell",
            "amount": amount,
            "order": order,
            "orders": orders,
            "current_notional_usdt": round(current_notional, 6),
            "target_notional_usdt": round(float(target_notional), 6),
            "delta_notional_usdt": round(delta_notional, 6),
            "old_leverage": float(current.get("leverage", 0.0) or 0.0),
            "new_leverage": context.leverage,
            "leverage_result": leverage_result,
        }
        self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "REBALANCE_QQQ_POSITION", payload)
        return payload

    def sync_existing_exchange_position(
        self,
        context: QqqOrderContext,
        exchange_position: dict[str, Any],
    ) -> dict[str, Any]:
        position_state = self._state_from_context(context)
        entry_price = self._extract_exchange_entry_price(exchange_position)
        if entry_price > 0:
            position_state["exchange_entry_price"] = entry_price
            position_state["peak_price"] = max(float(position_state.get("peak_price", 0.0) or 0.0), entry_price)
        exchange_stop = self._extract_exchange_stop_fields(exchange_position)
        if exchange_stop.get("stop_price"):
            position_state["stop_price"] = float(exchange_stop["stop_price"])
        exchange_leverage = self._extract_exchange_leverage(exchange_position)
        if exchange_leverage > 0:
            position_state["leverage"] = exchange_leverage
        position_state["exchange_contracts"] = float(exchange_position.get("contracts", 0.0) or 0.0)
        position_state["exchange_notional_usdt"] = float(exchange_position.get("notional_usdt", 0.0) or 0.0)
        position_state = self._with_exchange_stop_fields(position_state, exchange_stop)
        payload = {
            "status": "synced",
            "action": "sync_existing_qqq_usdt_position",
            "reason": "exchange_position_without_local_state",
            "contracts": position_state["exchange_contracts"],
            "notional_usdt": position_state["exchange_notional_usdt"],
            "leverage": position_state["leverage"],
            "stop_price": position_state["stop_price"],
            "exchange_stop": exchange_stop,
            "position": position_state,
        }
        self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "SYNC_QQQ_POSITION", payload)
        return payload

    def close_position(self, *, reason: str) -> dict[str, Any]:
        position = self.fetch_position_state()
        amount = float(position.get("contracts", 0.0) or 0.0)
        if amount <= 0:
            return {"status": "skipped", "reason": "no_open_qqq_position"}
        if self.router_config.mode == "paper":
            self.save_state({"position": None})
            return {"status": "paper_closed", "symbol": self.symbol, "amount": amount, "reason": reason}
        orders = self._submit_market_orders(
            self.symbol,
            "sell",
            amount,
            params={"reduceOnly": True, "tdMode": self.router_config.qqq_margin_mode, "posSide": "long"},
        )
        order = orders[-1] if orders else {}
        payload = {
            "status": "submitted",
            "action": "close_qqq_usdt_long",
            "order": order,
            "orders": orders,
            "amount": amount,
            "reason": reason,
        }
        self.store.append_action("runtime", "CLOSE_QQQ_USDT", payload)
        self.save_state({"position": None})
        return payload

    def _attach_stop_to_order_params(self, params: dict[str, Any], context: QqqOrderContext) -> dict[str, Any] | None:
        if not bool(self.router_config.qqq_enable_exchange_stop) or context.stop_price <= 0:
            return None
        attach_algo_client_id = self._generate_attach_algo_client_id()
        params["attachAlgoOrds"] = [self._build_stop_attach_algo(context, attach_algo_client_id)]
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

    def update_trailing_stop(self, context: QqqOrderContext, exchange_position: dict[str, Any]) -> dict[str, Any] | None:
        state = self.load_state()
        current = state.get("position") if isinstance(state.get("position"), dict) else {}
        old_stop = float(current.get("stop_price", 0.0) or 0.0)
        if context.stop_price <= old_stop:
            return None
        exchange_stop: dict[str, Any] | None = None
        if self.router_config.mode != "paper" and bool(self.router_config.qqq_enable_exchange_stop):
            exchange_stop = self._amend_exchange_stop(context.stop_price, current, exchange_position)
            if exchange_stop.get("status") == "error":
                payload = {
                    "status": "error",
                    "action": "update_qqq_trailing_stop",
                    "reason": "exchange_stop_amend_failed",
                    "old_stop_price": old_stop,
                    "new_stop_price": context.stop_price,
                    "exchange_contracts": exchange_position.get("contracts"),
                    "exchange_stop": exchange_stop,
                }
                self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "UPDATE_QQQ_STOP_FAILED", payload)
                return payload
        payload = {
            "status": "tracked",
            "action": "update_qqq_trailing_stop",
            "old_stop_price": old_stop,
            "new_stop_price": context.stop_price,
            "exchange_contracts": exchange_position.get("contracts"),
        }
        if exchange_stop is not None:
            payload["exchange_stop"] = exchange_stop
        self.store.append_action(str(context.candidate.get("timestamp") or "runtime"), "UPDATE_QQQ_STOP", payload)
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
        if fields.get("algo_ids"):
            updated["exchange_attach_algo_ids"] = fields["algo_ids"]
        if fields.get("algo_client_ids"):
            updated["exchange_attach_algo_client_ids"] = fields["algo_client_ids"]
        return updated

    def _generate_attach_algo_client_id(self) -> str:
        return f"qqqsl{uuid.uuid4().hex[:27]}"

    def _build_stop_attach_algo(self, context: QqqOrderContext, attach_algo_client_id: str) -> dict[str, Any]:
        market = self._market()
        return {
            "slTriggerPx": self._price_to_precision(market, context.stop_price),
            "slOrdPx": "-1",
            "slTriggerPxType": "mark",
            "attachAlgoClOrdId": attach_algo_client_id,
        }

    def _chunk_params_factory(
        self,
        context: QqqOrderContext,
        *,
        generated_client_ids: list[str] | None = None,
    ) -> Callable[[int, float], dict[str, Any]]:
        def factory(_index: int, _chunk: float) -> dict[str, Any]:
            params: dict[str, Any] = {"tdMode": context.margin_mode, "posSide": "long"}
            if bool(self.router_config.qqq_enable_exchange_stop) and context.stop_price > 0:
                attach_algo_client_id = self._generate_attach_algo_client_id()
                if generated_client_ids is not None:
                    generated_client_ids.append(attach_algo_client_id)
                params["attachAlgoOrds"] = [self._build_stop_attach_algo(context, attach_algo_client_id)]
            return params

        return factory

    def _extract_attached_algo_identity(self, position_state: dict[str, Any] | None) -> dict[str, str | None]:
        if not isinstance(position_state, dict):
            return {"algo_id": None, "algo_client_id": None}
        identities = self._extract_attached_algo_identities(position_state)
        if identities:
            return {
                "algo_id": identities[0].get("algo_id"),
                "algo_client_id": identities[0].get("algo_client_id"),
            }
        return {"algo_id": None, "algo_client_id": None}

    def _extract_attached_algo_identities(self, position_state: dict[str, Any] | None) -> list[dict[str, str | None]]:
        if not isinstance(position_state, dict):
            return []
        close_order_algos = position_state.get("close_order_algos") or []
        if not close_order_algos:
            raw = position_state.get("raw")
            info = raw.get("info") if isinstance(raw, dict) and isinstance(raw.get("info"), dict) else {}
            close_order_algos = info.get("closeOrderAlgo") if isinstance(info.get("closeOrderAlgo"), list) else []
        identities: list[dict[str, str | None]] = []
        seen: set[tuple[str | None, str | None]] = set()
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
                identity = {
                    "algo_id": str(algo_id) if algo_id else None,
                    "algo_client_id": str(algo_client_id) if algo_client_id else None,
                }
                key = (identity["algo_id"], identity["algo_client_id"])
                if key not in seen:
                    seen.add(key)
                    identities.append(identity)
        return identities

    @staticmethod
    def _positive_float(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            numeric = abs(float(value))
        except (TypeError, ValueError):
            return None
        return numeric if numeric > 0 else None

    def _extract_exchange_entry_price(self, exchange_position: dict[str, Any]) -> float:
        raw = exchange_position.get("raw") if isinstance(exchange_position, dict) else None
        candidates: list[Any] = []
        if isinstance(raw, dict):
            candidates.extend(raw.get(key) for key in ("entryPrice", "avgPx", "markPrice"))
            info = raw.get("info")
            if isinstance(info, dict):
                candidates.extend(info.get(key) for key in ("entryPrice", "avgPx", "markPx", "last"))
        for value in candidates:
            numeric = self._positive_float(value)
            if numeric is not None:
                return numeric
        return 0.0

    def _extract_exchange_leverage(self, exchange_position: dict[str, Any]) -> float:
        raw = exchange_position.get("raw") if isinstance(exchange_position, dict) else None
        candidates: list[Any] = []
        if isinstance(raw, dict):
            candidates.append(raw.get("leverage"))
            info = raw.get("info")
            if isinstance(info, dict):
                candidates.append(info.get("lever"))
        for value in candidates:
            numeric = self._positive_float(value)
            if numeric is not None:
                return numeric
        return 0.0

    def _extract_stop_price_from_algo(self, algo: dict[str, Any] | None) -> float | None:
        if not isinstance(algo, dict):
            return None
        for key in ("slTriggerPx", "triggerPx", "newSlTriggerPx", "newTriggerPx"):
            numeric = self._positive_float(algo.get(key))
            if numeric is not None:
                return numeric
        return None

    def _extract_exchange_stop_fields(self, exchange_position: dict[str, Any]) -> dict[str, Any]:
        identity = self._extract_attached_algo_identity(exchange_position)
        stop_price = None
        close_order_algos = exchange_position.get("close_order_algos") if isinstance(exchange_position, dict) else None
        if not close_order_algos:
            raw = exchange_position.get("raw") if isinstance(exchange_position, dict) else None
            info = raw.get("info") if isinstance(raw, dict) and isinstance(raw.get("info"), dict) else {}
            close_order_algos = info.get("closeOrderAlgo") if isinstance(info.get("closeOrderAlgo"), list) else []
        for algo in close_order_algos or []:
            stop_price = self._extract_stop_price_from_algo(algo)
            if stop_price is not None:
                break
        if (not identity["algo_id"] and not identity["algo_client_id"]) or stop_price is None:
            try:
                pending = self._select_pending_algo_order()
            except Exception:
                pending = None
            if pending:
                if not identity["algo_id"] and pending.get("algoId"):
                    identity["algo_id"] = str(pending.get("algoId"))
                if not identity["algo_client_id"] and pending.get("algoClOrdId"):
                    identity["algo_client_id"] = str(pending.get("algoClOrdId"))
                stop_price = stop_price or self._extract_stop_price_from_algo(pending)
        pending_orders: list[dict[str, Any]] = []
        try:
            pending_orders = self._select_pending_algo_orders()
        except Exception:
            pending_orders = []
        algo_ids = []
        algo_client_ids = []
        for item in pending_orders:
            if item.get("algoId"):
                algo_ids.append(str(item["algoId"]))
            if item.get("algoClOrdId"):
                algo_client_ids.append(str(item["algoClOrdId"]))
        for item in self._extract_attached_algo_identities(exchange_position):
            if item.get("algo_id") and item["algo_id"] not in algo_ids:
                algo_ids.append(str(item["algo_id"]))
            if item.get("algo_client_id") and item["algo_client_id"] not in algo_client_ids:
                algo_client_ids.append(str(item["algo_client_id"]))
        return {
            "status": "found" if identity["algo_id"] or identity["algo_client_id"] or stop_price else "not_found",
            "algo_id": identity["algo_id"],
            "algo_client_id": identity["algo_client_id"],
            "algo_ids": algo_ids,
            "algo_client_ids": algo_client_ids,
            "stop_price": stop_price,
        }

    def _max_market_order_amount(self) -> float | None:
        market = self._market()
        candidates = [
            ((market.get("limits") or {}).get("amount") or {}).get("max"),
            (market.get("info") or {}).get("maxMktSz"),
        ]
        for value in candidates:
            numeric = self._positive_float(value)
            if numeric is not None:
                return numeric
        return None

    def _split_order_amount(self, amount: float) -> list[float]:
        total = Decimal(str(amount))
        if total <= 0:
            return []
        max_amount = self._max_market_order_amount()
        if max_amount is None or max_amount <= 0 or total <= Decimal(str(max_amount)):
            return [float(total)]
        chunks: list[float] = []
        remaining = total
        max_decimal = Decimal(str(max_amount))
        while remaining > max_decimal:
            chunks.append(float(self._amount_to_precision(self._market(), float(max_decimal))))
            remaining -= max_decimal
        if remaining > 0:
            chunks.append(float(self._amount_to_precision(self._market(), float(remaining))))
        return [chunk for chunk in chunks if chunk > 0]

    def _submit_market_orders(
        self,
        symbol: str,
        side: str,
        amount: float,
        *,
        params: dict[str, Any] | None = None,
        params_factory: Callable[[int, float], dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        orders = []
        for index, chunk in enumerate(self._split_order_amount(float(amount))):
            chunk_params = params_factory(index, chunk) if params_factory is not None else deepcopy(params or {})
            try:
                orders.append(self.client.create_order(symbol, "market", side, chunk, params=chunk_params))
            except Exception as exc:
                if orders:
                    raise PartialOrderSubmissionError(str(exc), orders=orders, failed_chunk=float(chunk)) from exc
                raise
        return orders

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

    def _select_pending_algo_order(self, local_position: dict[str, Any] | None = None) -> dict[str, Any] | None:
        candidates = self._select_pending_algo_orders(local_position)
        return candidates[0] if candidates else None

    def _select_pending_algo_orders(self, local_position: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        pending_orders = self._fetch_pending_algo_orders()
        market_id = self._market().get("id")
        local_algo_id = str((local_position or {}).get("exchange_attach_algo_id") or "")
        local_algo_client_id = str((local_position or {}).get("exchange_attach_algo_client_id") or "")
        local_algo_ids = {
            str(item)
            for item in (local_position or {}).get("exchange_attach_algo_ids", [])
            if item not in (None, "")
        }
        local_algo_client_ids = {
            str(item)
            for item in (local_position or {}).get("exchange_attach_algo_client_ids", [])
            if item not in (None, "")
        }
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
            candidates.append(order)
        matched = []
        for order in candidates:
            algo_id = str(order.get("algoId") or "")
            algo_client_id = str(order.get("algoClOrdId") or "")
            if local_algo_id and algo_id == local_algo_id:
                matched.append(order)
                continue
            if local_algo_client_id and algo_client_id == local_algo_client_id:
                matched.append(order)
                continue
            if local_algo_ids and algo_id in local_algo_ids:
                matched.append(order)
                continue
            if local_algo_client_ids and algo_client_id in local_algo_client_ids:
                matched.append(order)
        return matched or candidates

    def _refresh_exchange_stop_identity(
        self,
        *,
        attach_algo_client_id: str | None = None,
        attach_algo_client_ids: list[str] | None = None,
        order_id: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "disabled" if not bool(self.router_config.qqq_enable_exchange_stop) else "not_found",
            "order_id": order_id,
            "algo_id": None,
            "algo_client_id": attach_algo_client_id,
        }
        if self.router_config.mode == "paper" or not bool(self.router_config.qqq_enable_exchange_stop):
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
            pending_orders = self._select_pending_algo_orders()
            algo_ids = [
                str(order.get("algoId"))
                for order in pending_orders
                if order.get("algoId")
            ]
            algo_client_ids = [
                str(order.get("algoClOrdId"))
                for order in pending_orders
                if order.get("algoClOrdId")
            ]
            for generated_id in attach_algo_client_ids or []:
                if generated_id and generated_id not in algo_client_ids:
                    algo_client_ids.append(generated_id)
            result.update(
                {
                    "status": "found" if identity["algo_id"] or identity["algo_client_id"] else "not_found",
                    "order_id": order_id or self._extract_order_id(position.get("raw")),
                    "algo_id": identity["algo_id"],
                    "algo_client_id": identity["algo_client_id"] or attach_algo_client_id,
                    "algo_ids": algo_ids,
                    "algo_client_ids": algo_client_ids,
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
            raise ValueError("Missing QQQ attached stop identifier")
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
            raise ValueError("Missing QQQ conditional stop identifier")
        return request

    def _amend_exchange_stop(
        self,
        stop_price: float,
        local_position: dict[str, Any],
        exchange_position: dict[str, Any],
    ) -> dict[str, Any]:
        identity = self._extract_attached_algo_identity(exchange_position)
        attach_algo_id = identity["algo_id"] or local_position.get("exchange_attach_algo_id")
        attach_algo_client_id = identity["algo_client_id"] or local_position.get("exchange_attach_algo_client_id")
        pending_order_type = None
        if not attach_algo_id and not attach_algo_client_id:
            try:
                pending = self._select_pending_algo_order(local_position)
            except Exception as exc:
                return {"status": "error", "reason": "pending_algo_lookup_failed", "error": str(exc)}
            if pending:
                pending_order_type = str(pending.get("ordType") or "")
                attach_algo_id = str(pending.get("algoId")) if pending.get("algoId") else None
                attach_algo_client_id = str(pending.get("algoClOrdId")) if pending.get("algoClOrdId") else None
        if not attach_algo_id and not attach_algo_client_id:
            return {"status": "error", "reason": "missing_exchange_stop_identifier"}
        try:
            pending_orders = self._select_pending_algo_orders(local_position)
        except Exception:
            pending_orders = []
        amend_targets: list[dict[str, str | None]] = []
        seen: set[tuple[str | None, str | None]] = set()
        for algo_id, algo_client_id in [(attach_algo_id, attach_algo_client_id)]:
            key = (algo_id, algo_client_id)
            if key not in seen:
                seen.add(key)
                amend_targets.append({"algo_id": algo_id, "algo_client_id": algo_client_id})
        for pending in pending_orders:
            algo_id = str(pending.get("algoId")) if pending.get("algoId") else None
            algo_client_id = str(pending.get("algoClOrdId")) if pending.get("algoClOrdId") else None
            if not algo_id and not algo_client_id:
                continue
            key = (algo_id, algo_client_id)
            if key in seen:
                continue
            seen.add(key)
            amend_targets.append({"algo_id": algo_id, "algo_client_id": algo_client_id})

        responses = []
        errors = []
        last_request = None
        for target in amend_targets:
            request = self._build_algo_amend_request(
                attach_algo_id=target["algo_id"],
                attach_algo_client_id=target["algo_client_id"],
                stop_price=stop_price,
            )
            try:
                response = self.client.amend_algo_order(request)
            except Exception as exc:
                primary_error = str(exc)
                fallback_request = self._build_conditional_algo_amend_request(
                    attach_algo_id=target["algo_id"],
                    attach_algo_client_id=target["algo_client_id"],
                    stop_price=stop_price,
                )
                try:
                    response = self.client.amend_algo_order(fallback_request)
                    request = fallback_request
                except Exception as fallback_exc:
                    errors.append(
                        {
                            "algo_id": target["algo_id"],
                            "algo_client_id": target["algo_client_id"],
                            "error": str(fallback_exc),
                            "primary_error": primary_error,
                            "request": fallback_request,
                        }
                    )
                    continue
            last_request = request
            responses.append(
                {
                    "algo_id": target["algo_id"],
                    "algo_client_id": target["algo_client_id"],
                    "response": response,
                }
            )
        if errors:
            return {
                "status": "error",
                "reason": "amend_algo_order_failed",
                "errors": errors,
                "updated_count": len(responses),
                "pending_order_type": pending_order_type,
            }
        refreshed_stop = self._extract_exchange_stop_fields(self.fetch_position_state())
        return {
            "status": "amended",
            "algo_id": attach_algo_id,
            "algo_client_id": attach_algo_client_id,
            "algo_ids": refreshed_stop.get("algo_ids", []),
            "algo_client_ids": refreshed_stop.get("algo_client_ids", []),
            "stop_price": stop_price,
            "request": last_request,
            "response": responses[-1]["response"] if responses else None,
            "responses": responses,
            "updated_count": len(responses),
        }
