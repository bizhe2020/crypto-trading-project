from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from bot.okx_executor import OkxExecutionEngine
from bot.qqq_usdt_executor import QqqUsdtExecutionEngine
from bot.strategy_router import RoutedSignalCandidate, StrategyRouter


class StrategyRouterExecutionEngine:
    def __init__(self, router: StrategyRouter):
        self.router = router
        self.config = router.config
        self.btc_executor = OkxExecutionEngine.from_file(self.config.btc_strategy_config)
        self.qqq_executor = QqqUsdtExecutionEngine(self.config, self.config.qqq_strategy_config)
        self.execution_state_path = self.router.state_path.with_suffix(self.router.state_path.suffix + ".execution")
        self.audit_log_path = self._resolve_audit_log_path()

    @classmethod
    def from_file(cls, path: str | Path) -> "StrategyRouterExecutionEngine":
        return cls(StrategyRouter.from_file(path))

    def bootstrap(self) -> dict[str, Any]:
        payload = {
            "status": "ok",
            "mode": self.config.mode,
            "router_config": str(self.router.config_path),
            "btc": self.btc_executor.bootstrap(),
            "qqq": self.qqq_executor.bootstrap(),
        }
        if bool(self.config.telegram_notify_startup):
            self._send_telegram(self._format_startup_message(payload))
        return payload

    def evaluate_latest(self) -> dict[str, Any]:
        previous_executed = self._current_executed_strategy()
        route = self.router.evaluate_latest(current_strategy_override=previous_executed)
        selected_strategy = route.get("selected_strategy")
        selected_candidate = route.get("selected_candidate") if isinstance(route.get("selected_candidate"), dict) else None
        execution_results: list[dict[str, Any]] = []
        risk_on_window = None
        if selected_strategy == "qqq_usdt_aggressive" and previous_executed != "qqq_usdt_aggressive":
            risk_on_window = self.qqq_executor.risk_on_window_status()
            if not bool(risk_on_window.get("open")):
                execution_results.append(
                    {
                        "strategy": "qqq_usdt_aggressive",
                        "result": {
                            "status": "skipped",
                            "reason": "qqq_risk_on_window_closed_before_switch",
                            "market_window": risk_on_window,
                        },
                    }
                )
                selected_strategy = previous_executed

        if selected_strategy != previous_executed and bool(self.config.flatten_before_switch):
            execution_results.extend(self._flatten_strategy(previous_executed, reason=f"router_switch_to_{selected_strategy or 'cash'}"))
            if not self._flatten_confirmed(execution_results):
                self._set_current_executed_strategy(previous_executed)
                payload = {
                    "status": "blocked",
                    "mode": self.config.mode,
                    "route": route,
                    "previous_executed_strategy": previous_executed,
                    "current_executed_strategy": self._current_executed_strategy(),
                    "execution_results": execution_results,
                    "blocked_reason": "flatten_not_confirmed",
                    "updated_at": int(time.time()),
                }
                self._attach_execution_diagnostics(payload)
                self._maybe_send_telegram_notifications(payload)
                execution_state = self._load_execution_state()
                execution_state.update(
                    {
                        "current_executed_strategy": self._current_executed_strategy(),
                        "last_status": payload,
                        "updated_at": int(time.time()),
                    }
                )
                self._save_execution_state(execution_state)
                self._append_audit_log(payload)
                return payload

        if selected_strategy == "btc_sota":
            execution_results.append({"strategy": "btc_sota", "result": self.btc_executor.evaluate_latest()})
            self._set_current_executed_strategy("btc_sota")
        elif selected_strategy == "qqq_usdt_aggressive":
            candidate = self._candidate_from_payload(selected_candidate)
            execution_results.append({"strategy": "qqq_usdt_aggressive", "result": self.qqq_executor.evaluate_latest(candidate)})
            self._set_current_executed_strategy("qqq_usdt_aggressive")
        else:
            if bool(self.config.flatten_on_no_signal):
                execution_results.extend(self._flatten_strategy(previous_executed, reason="router_no_signal"))
            self._set_current_executed_strategy(None)

        payload = {
            "status": "ok",
            "mode": self.config.mode,
            "route": route,
            "previous_executed_strategy": previous_executed,
            "current_executed_strategy": self._current_executed_strategy(),
            "execution_results": execution_results,
            "updated_at": int(time.time()),
        }
        self._attach_execution_diagnostics(payload)
        self._maybe_send_telegram_notifications(payload)
        execution_state = self._load_execution_state()
        execution_state.update(
            {
                "current_executed_strategy": self._current_executed_strategy(),
                "last_status": payload,
                "updated_at": int(time.time()),
            }
        )
        self._save_execution_state(execution_state)
        self._append_audit_log(payload)
        return payload

    @staticmethod
    def _flatten_confirmed(results: list[dict[str, Any]]) -> bool:
        for item in results:
            result = item.get("result") if isinstance(item, dict) else None
            if not isinstance(result, dict):
                continue
            status = str(result.get("status") or "")
            if status in {"submitted_but_unconfirmed", "partial_submitted_error", "error"}:
                return False
        return True

    def run_loop(self, poll_interval_seconds: int = 30) -> None:
        bootstrap = self.bootstrap()
        print(json.dumps({"event": "bootstrap", **bootstrap}, ensure_ascii=False))
        while True:
            try:
                status = self.evaluate_latest()
                print(json.dumps({"event": "evaluate", **status}, ensure_ascii=False))
            except KeyboardInterrupt:
                print(json.dumps({"event": "stopped"}, ensure_ascii=False))
                raise
            except Exception as exc:
                print(json.dumps({"event": "error", "error": str(exc)}, ensure_ascii=False))
                if bool(self.config.telegram_notify_errors):
                    self._send_telegram(f"Router 异常\n错误: {exc}")
            time.sleep(max(1, int(poll_interval_seconds)))

    def _candidate_from_payload(self, payload: dict[str, Any] | None) -> RoutedSignalCandidate | None:
        if not payload:
            return None
        return RoutedSignalCandidate(
            strategy_id=str(payload.get("strategy_id") or "qqq_usdt_aggressive"),
            symbol=str(payload.get("symbol") or self.qqq_executor.symbol),
            active=bool(payload.get("active", False)),
            route_score=float(payload.get("route_score", 0.0) or 0.0),
            timestamp=payload.get("timestamp"),
            direction=payload.get("direction"),
            event_type=payload.get("event_type"),
            leverage=float(payload["leverage"]) if payload.get("leverage") not in (None, "") else None,
            strength_label=payload.get("strength_label"),
            source_config=payload.get("source_config"),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        )

    def _current_executed_strategy(self) -> str | None:
        raw = self._load_execution_state().get("current_executed_strategy")
        if not raw or raw == "null":
            return None
        return str(raw)

    def _last_notified_strategy(self) -> str | None:
        raw = self._load_execution_state().get("last_notified_strategy")
        if not raw or raw == "null":
            return None
        return str(raw)

    def _last_warning_key(self) -> str | None:
        raw = self._load_execution_state().get("last_warning_key")
        if not raw or raw == "null":
            return None
        return str(raw)

    def _last_gap_key(self) -> str | None:
        raw = self._load_execution_state().get("last_gap_key")
        if not raw or raw == "null":
            return None
        return str(raw)

    def _set_current_executed_strategy(self, strategy_id: str | None) -> None:
        state = self._load_execution_state()
        state["current_executed_strategy"] = strategy_id
        state["updated_at"] = int(time.time())
        self._save_execution_state(state)

    def _set_last_notified_strategy(self, strategy_id: str | None) -> None:
        state = self._load_execution_state()
        state["last_notified_strategy"] = strategy_id
        state["updated_at"] = int(time.time())
        self._save_execution_state(state)

    def _set_last_warning_key(self, warning_key: str | None) -> None:
        state = self._load_execution_state()
        state["last_warning_key"] = warning_key
        state["updated_at"] = int(time.time())
        self._save_execution_state(state)

    def _set_last_gap_key(self, gap_key: str | None) -> None:
        state = self._load_execution_state()
        state["last_gap_key"] = gap_key
        state["updated_at"] = int(time.time())
        self._save_execution_state(state)

    def _load_execution_state(self) -> dict[str, Any]:
        if not self.execution_state_path.exists():
            return {}
        try:
            decoded = json.loads(self.execution_state_path.read_text())
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}

    def _save_execution_state(self, payload: dict[str, Any]) -> None:
        self.execution_state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.execution_state_path.with_suffix(self.execution_state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        tmp.replace(self.execution_state_path)

    def _resolve_audit_log_path(self) -> Path:
        configured = self.config.router_audit_log_path
        if configured:
            return Path(configured)
        return self.router.state_path.with_suffix(self.router.state_path.suffix + ".audit.jsonl")

    def _append_audit_log(self, payload: dict[str, Any]) -> None:
        if not bool(self.config.router_audit_log_enabled):
            return
        record = self._build_audit_record(payload)
        try:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            return

    def _build_audit_record(self, payload: dict[str, Any]) -> dict[str, Any]:
        errors: list[dict[str, str]] = []
        record = {
            "event": "strategy_router_evaluate",
            "schema_version": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "mode": self.config.mode,
            "router_config": str(self.router.config_path),
            "state_path": str(self.router.state_path),
            "execution_state_path": str(self.execution_state_path),
            "route": payload.get("route"),
            "previous_executed_strategy": payload.get("previous_executed_strategy"),
            "current_executed_strategy": payload.get("current_executed_strategy"),
            "execution_results": payload.get("execution_results", []),
            "execution_diagnostics": payload.get("execution_diagnostics"),
            "runtime": {
                "updated_at": payload.get("updated_at"),
                "pid": self._safe_pid(),
            },
            "local_state": {
                "router_execution": self._safe_call("router_execution_state", self._load_execution_state, errors),
                "btc_snapshot": self._safe_call("btc_snapshot", self.btc_executor._load_snapshot_payload, errors),
                "qqq_state": self._safe_call("qqq_state", self.qqq_executor.load_state, errors),
            },
            "exchange_state": {
                "btc": self._safe_call("btc_exchange_state", self._btc_exchange_state, errors),
                "qqq": self._safe_call("qqq_exchange_state", self._qqq_exchange_state, errors),
            },
        }
        if errors:
            record["audit_errors"] = errors
        return record

    def _attach_execution_diagnostics(self, payload: dict[str, Any]) -> None:
        payload["execution_diagnostics"] = self._build_execution_diagnostics(payload)

    def _build_execution_diagnostics(self, payload: dict[str, Any]) -> dict[str, Any]:
        errors: list[dict[str, str]] = []
        route = payload.get("route") if isinstance(payload.get("route"), dict) else {}
        selected_strategy = route.get("selected_strategy")
        current_strategy = payload.get("current_executed_strategy")
        previous_strategy = payload.get("previous_executed_strategy")
        selected_candidate = route.get("selected_candidate") if isinstance(route.get("selected_candidate"), dict) else {}
        metadata = selected_candidate.get("metadata") if isinstance(selected_candidate.get("metadata"), dict) else {}
        daily_refresh = metadata.get("daily_signal_refresh") if isinstance(metadata.get("daily_signal_refresh"), dict) else {}
        daily_stale = metadata.get("daily_signal_stale") if isinstance(metadata.get("daily_signal_stale"), dict) else {}
        data_refresh = metadata.get("data_refresh") if isinstance(metadata.get("data_refresh"), dict) else {}

        btc_snapshot = self._safe_call(
            "diag_btc_snapshot",
            getattr(self.btc_executor, "_load_snapshot_payload", lambda: None),
            errors,
        )
        qqq_state = self._safe_call(
            "diag_qqq_state",
            getattr(self.qqq_executor, "load_state", lambda: None),
            errors,
        )
        btc_exchange = self._safe_call("diag_btc_exchange_state", self._btc_exchange_state, errors)
        qqq_exchange = self._safe_call("diag_qqq_exchange_state", self._qqq_exchange_state, errors)

        btc_long = self._nested_float(btc_exchange, "long", "contracts")
        btc_short = self._nested_float(btc_exchange, "short", "contracts")
        qqq_local_position = self._nested_dict(qqq_state, "position")
        qqq_exchange_position = self._nested_dict(qqq_exchange, "position")
        qqq_local_contracts = self._first_float(
            qqq_local_position.get("exchange_contracts"),
            qqq_local_position.get("contracts"),
        )
        qqq_exchange_contracts = self._float_or_none(qqq_exchange_position.get("contracts"))
        qqq_local_stop = self._float_or_none(qqq_local_position.get("stop_price"))
        qqq_stop_fields = self._safe_call(
            "diag_qqq_exchange_stop_fields",
            lambda: self._qqq_exchange_stop_fields(qqq_exchange_position),
            errors,
        )
        qqq_exchange_stop = self._float_or_none(
            qqq_stop_fields.get("stop_price") if isinstance(qqq_stop_fields, dict) else None
        )

        items: list[dict[str, str]] = []

        def add_item(severity: str, key: str, message: str) -> None:
            items.append({"severity": severity, "key": key, "message": message})

        if selected_strategy != current_strategy:
            add_item(
                "info",
                "router_selected_execution_diverged",
                f"路由目标 {self._strategy_label(selected_strategy)}，当前执行 {self._strategy_label(current_strategy)}",
            )

        execution_statuses = self._payload_execution_statuses(payload)
        for status in execution_statuses:
            if status in {"error", "partial_submitted_error"}:
                add_item("critical", f"execution_status:{status}", f"执行状态异常: {status}")
            elif status == "submitted_but_unconfirmed":
                add_item("warning", f"execution_status:{status}", "执行已提交但交易所未确认完全同步")

        btc_open = self._positive(btc_long) or self._positive(btc_short)
        qqq_open = self._positive(qqq_exchange_contracts)
        if current_strategy != "btc_sota" and btc_open:
            add_item(
                "critical",
                "unmanaged_btc_exchange_position",
                f"交易所仍有 BTC 仓位 L/S={self._fmt_num(btc_long)}/{self._fmt_num(btc_short)}，但当前执行不是 BTC",
            )
        if current_strategy != "qqq_usdt_aggressive" and qqq_open:
            add_item(
                "critical",
                "unmanaged_qqq_exchange_position",
                f"交易所仍有 QQQ 仓位 {self._fmt_num(qqq_exchange_contracts)}，但当前执行不是 QQQ",
            )
        if current_strategy == "qqq_usdt_aggressive" and qqq_exchange_contracts is not None and not qqq_open:
            add_item("critical", "router_qqq_exchange_flat", "router 当前执行 QQQ，但交易所 QQQ 仓位为 0")
        if qqq_local_contracts is not None and qqq_exchange_contracts is not None:
            contract_gap = abs(float(qqq_local_contracts) - float(qqq_exchange_contracts))
            tolerance = max(0.01, abs(float(qqq_exchange_contracts)) * 0.001)
            if contract_gap > tolerance:
                add_item(
                    "warning",
                    "qqq_local_exchange_contract_gap",
                    (
                        "QQQ 本地/交易所仓位不一致: "
                        f"state={self._fmt_num(qqq_local_contracts)} exchange={self._fmt_num(qqq_exchange_contracts)}"
                    ),
                )
        if qqq_open and bool(self.config.qqq_enable_exchange_stop):
            stop_found = isinstance(qqq_stop_fields, dict) and str(qqq_stop_fields.get("status") or "") == "found"
            if not stop_found and qqq_exchange_stop is None:
                add_item("critical", "qqq_exchange_stop_missing", "QQQ 交易所保护止损未找到")
            elif qqq_local_stop is not None and qqq_exchange_stop is not None:
                stop_gap = abs(float(qqq_local_stop) - float(qqq_exchange_stop))
                tolerance = max(0.1, abs(float(qqq_local_stop)) * 0.001)
                if stop_gap > tolerance:
                    add_item(
                        "warning",
                        "qqq_local_exchange_stop_gap",
                        (
                            "QQQ 本地/交易所止损不一致: "
                            f"state={self._fmt_num(qqq_local_stop)} exchange={self._fmt_num(qqq_exchange_stop)}"
                        ),
                    )

        for error in errors:
            section = error.get("section") or "unknown"
            add_item("warning", f"diagnostic_read_failed:{section}", f"诊断读取失败: {section}")

        status = self._diagnostic_status(items)
        return {
            "status": status,
            "runtime": {"pid": self._safe_pid(), "updated_at": payload.get("updated_at")},
            "router": {
                "selected_strategy": selected_strategy,
                "current_executed_strategy": current_strategy,
                "previous_executed_strategy": previous_strategy,
                "decision_reason": route.get("decision_reason"),
                "selected_route_score": route.get("selected_route_score"),
            },
            "data": {
                "data_refresh_status": data_refresh.get("status"),
                "data_latest": data_refresh.get("latest"),
                "daily_refresh_status": daily_refresh.get("status"),
                "daily_errors": sorted((daily_refresh.get("errors") or {}).keys())
                if isinstance(daily_refresh.get("errors"), dict)
                else [],
                "daily_stale": daily_stale.get("stale"),
                "daily_latest": daily_stale.get("latest"),
                "daily_lag_days": daily_stale.get("lag_days"),
                "daily_signal_timestamp": metadata.get("daily_signal_timestamp"),
            },
            "execution": {
                "payload_status": payload.get("status"),
                "statuses": execution_statuses,
                "blocked_reason": payload.get("blocked_reason"),
            },
            "exchange": {
                "btc": {
                    "long_contracts": btc_long,
                    "short_contracts": btc_short,
                    "local_direction": self._nested_value(btc_snapshot, "position", "direction"),
                },
                "qqq": {
                    "local_contracts": qqq_local_contracts,
                    "exchange_contracts": qqq_exchange_contracts,
                    "notional_usdt": self._float_or_none(qqq_exchange_position.get("notional_usdt")),
                    "local_stop_price": qqq_local_stop,
                    "exchange_stop_price": qqq_exchange_stop,
                    "stop_status": qqq_stop_fields.get("status") if isinstance(qqq_stop_fields, dict) else None,
                    "stop_algo_id": qqq_stop_fields.get("algo_id") if isinstance(qqq_stop_fields, dict) else None,
                    "stop_algo_client_id": qqq_stop_fields.get("algo_client_id") if isinstance(qqq_stop_fields, dict) else None,
                    "mark_price": self._float_or_none(qqq_exchange_position.get("markPrice")),
                    "liquidation_price": self._float_or_none(qqq_exchange_position.get("liquidationPrice")),
                },
            },
            "items": items,
        }

    def _qqq_exchange_stop_fields(self, exchange_position: dict[str, Any]) -> dict[str, Any]:
        extractor = getattr(self.qqq_executor, "_extract_exchange_stop_fields", None)
        if callable(extractor):
            result = extractor(exchange_position)
            return result if isinstance(result, dict) else {}
        return {}

    @staticmethod
    def _payload_execution_statuses(payload: dict[str, Any]) -> list[str]:
        statuses: list[str] = []
        for item in payload.get("execution_results", []) or []:
            if not isinstance(item, dict):
                continue
            result = item.get("result")
            if not isinstance(result, dict):
                continue
            actions = result.get("actions")
            if isinstance(actions, list) and actions:
                for action in actions:
                    if isinstance(action, dict):
                        raw = action.get("status") or action.get("reason")
                        if raw:
                            statuses.append(str(raw))
                continue
            raw = result.get("status") or result.get("reason")
            if raw:
                statuses.append(str(raw))
        return statuses

    @staticmethod
    def _diagnostic_status(items: list[dict[str, str]]) -> str:
        severities = {str(item.get("severity") or "") for item in items}
        if "critical" in severities:
            return "critical"
        if "warning" in severities:
            return "warning"
        return "ok"

    @staticmethod
    def _nested_dict(payload: Any, *keys: str) -> dict[str, Any]:
        value: Any = payload
        for key in keys:
            if not isinstance(value, dict):
                return {}
            value = value.get(key)
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _nested_value(payload: Any, *keys: str) -> Any:
        value: Any = payload
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    @classmethod
    def _nested_float(cls, payload: Any, *keys: str) -> float | None:
        return cls._float_or_none(cls._nested_value(payload, *keys))

    @classmethod
    def _first_float(cls, *values: Any) -> float | None:
        for value in values:
            numeric = cls._float_or_none(value)
            if numeric is not None:
                return numeric
        return None

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _positive(cls, value: Any) -> bool:
        numeric = cls._float_or_none(value)
        return numeric is not None and abs(float(numeric)) > 1e-9

    @staticmethod
    def _fmt_num(value: Any, digits: int = 4) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "-"
        text = f"{numeric:.{digits}f}".rstrip("0").rstrip(".")
        return text or "0"

    @staticmethod
    def _short_id(value: Any) -> str:
        text = str(value or "")
        if not text:
            return "-"
        if len(text) <= 12:
            return text
        return f"{text[:4]}...{text[-6:]}"

    @staticmethod
    def _fmt_time(value: Any) -> str:
        if value in (None, ""):
            return "-"
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        except (TypeError, ValueError, OSError):
            return str(value)

    @staticmethod
    def _safe_pid() -> int | None:
        try:
            import os

            return int(os.getpid())
        except Exception:
            return None

    @staticmethod
    def _safe_call(name: str, callback: Any, errors: list[dict[str, str]]) -> Any:
        try:
            return callback()
        except Exception as exc:
            errors.append({"section": name, "error": str(exc)})
            return None

    def _btc_exchange_state(self) -> dict[str, Any]:
        if self.config.mode == "paper":
            return {"mode": "paper", "enabled": False}
        return {
            "symbol": self.btc_executor.config.symbol,
            "long": self.btc_executor._fetch_position_state("long"),
            "short": self.btc_executor._fetch_position_state("short"),
        }

    def _qqq_exchange_state(self) -> dict[str, Any]:
        return {
            "symbol": self.qqq_executor.symbol,
            "position": self.qqq_executor.fetch_position_state(),
        }

    def _flatten_strategy(self, strategy_id: str | None, *, reason: str) -> list[dict[str, Any]]:
        if strategy_id == "qqq_usdt_aggressive":
            return [{"strategy": strategy_id, "result": self.qqq_executor.close_position(reason=reason)}]
        if strategy_id == "btc_sota":
            return [{"strategy": strategy_id, "result": self._flatten_btc(reason=reason)}]
        return []

    def _flatten_btc(self, *, reason: str) -> dict[str, Any]:
        if self.config.mode == "paper":
            return {"status": "paper_flatten_skipped", "reason": reason}
        results = []
        for pos_side, side in [("long", "sell"), ("short", "buy")]:
            state = self.btc_executor._fetch_position_state(pos_side)
            amount = float(state.get("contracts", 0.0) or 0.0)
            if amount <= 0:
                continue
            order = self.btc_executor.client.create_order(
                self.btc_executor.config.symbol,
                "market",
                side,
                amount,
                params={"reduceOnly": True, "tdMode": self.btc_executor.config.margin_mode, "posSide": pos_side},
            )
            results.append({"pos_side": pos_side, "amount": amount, "order": order})
        return {"status": "submitted" if results else "skipped", "reason": reason, "orders": results}

    def _telegram_credentials(self) -> tuple[str | None, str | None, str | None]:
        token = self.config.telegram_token
        chat_id = self.config.telegram_chat_id
        proxy = self.config.telegram_proxy
        if (not token or not chat_id) and self.config.execution_credentials_config:
            try:
                payload = json.loads(Path(self.config.execution_credentials_config).read_text())
            except Exception:
                payload = {}
            token = token or payload.get("telegram_token")
            chat_id = chat_id or payload.get("telegram_chat_id")
            proxy = proxy or payload.get("proxy")
        return (str(token) if token else None, str(chat_id) if chat_id else None, str(proxy) if proxy else None)

    def _send_telegram(self, message: str) -> bool:
        if not bool(self.config.telegram_enabled):
            return False
        token, chat_id, proxy = self._telegram_credentials()
        if not token or not chat_id:
            return False
        proxies = {"http": proxy, "https": proxy} if proxy else None
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message},
                timeout=10,
                proxies=proxies,
            )
            response.raise_for_status()
            return True
        except Exception:
            return False

    def _maybe_send_telegram_notifications(self, payload: dict[str, Any]) -> None:
        route = payload.get("route") if isinstance(payload.get("route"), dict) else {}
        current = payload.get("current_executed_strategy")
        previous = payload.get("previous_executed_strategy")
        if bool(self.config.telegram_notify_route_change) and current != self._last_notified_strategy():
            if self._send_telegram(self._format_route_message(payload, route)):
                self._set_last_notified_strategy(str(current) if current else None)
        if bool(self.config.telegram_notify_execution):
            messages = self._execution_messages(payload)
            for message in messages:
                self._send_telegram(message)
        if bool(self.config.telegram_notify_data_warnings):
            warning = self._data_warning(route)
            warning_key = warning.get("key") if warning else None
            if warning and warning_key != self._last_warning_key():
                if self._send_telegram(warning["message"]):
                    self._set_last_warning_key(str(warning_key))
            elif warning is None and self._last_warning_key():
                self._set_last_warning_key(None)
        if bool(getattr(self.config, "telegram_notify_gap_warnings", True)):
            diagnostics = payload.get("execution_diagnostics") if isinstance(payload.get("execution_diagnostics"), dict) else {}
            warning = self._gap_warning(diagnostics)
            gap_key = warning.get("key") if warning else None
            if warning and gap_key != self._last_gap_key():
                if self._send_telegram(warning["message"]):
                    self._set_last_gap_key(str(gap_key))
            elif warning is None and self._last_gap_key():
                self._set_last_gap_key(None)

    @staticmethod
    def _strategy_label(strategy_id: Any) -> str:
        labels = {
            "btc_sota": "BTC SOTA",
            "qqq_usdt_aggressive": "QQQ/USDT",
            None: "空仓",
        }
        return labels.get(strategy_id, str(strategy_id or "空仓"))

    @staticmethod
    def _action_statuses(result: Any) -> list[str]:
        if not isinstance(result, dict):
            return []
        actions = result.get("actions")
        if isinstance(actions, list):
            return [str(item.get("status") or item.get("reason") or "-") for item in actions if isinstance(item, dict)]
        status = result.get("status")
        return [str(status)] if status else []

    def _format_startup_message(self, payload: dict[str, Any]) -> str:
        btc_status = "ok" if isinstance(payload.get("btc"), dict) and payload["btc"].get("status") != "error" else "异常"
        qqq_status = "ok" if isinstance(payload.get("qqq"), dict) and payload["qqq"].get("status") != "error" else "异常"
        return "\n".join(
            [
                "Router 启动",
                f"模式: {payload.get('mode')}",
                "链路: BTC SOTA / QQQ-USDT",
                f"BTC: {btc_status}",
                f"QQQ: {qqq_status}",
            ]
        )

    @staticmethod
    def _format_route_message(payload: dict[str, Any], route: dict[str, Any]) -> str:
        candidate = route.get("selected_candidate") if isinstance(route.get("selected_candidate"), dict) else {}
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        stale = metadata.get("daily_signal_stale") if isinstance(metadata.get("daily_signal_stale"), dict) else {}
        diagnostics = (
            payload.get("execution_diagnostics") if isinstance(payload.get("execution_diagnostics"), dict) else {}
        )
        lines = [
            "Router 选路更新",
            f"路由目标: {StrategyRouterExecutionEngine._strategy_label(route.get('selected_strategy'))}",
            f"当前执行: {StrategyRouterExecutionEngine._strategy_label(payload.get('current_executed_strategy'))}",
            f"上次执行: {StrategyRouterExecutionEngine._strategy_label(payload.get('previous_executed_strategy'))}",
            f"原因: {route.get('decision_reason') or '-'}",
            f"分数: {route.get('selected_route_score') or 0}",
        ]
        if candidate:
            lines.append(f"信号时间: {candidate.get('timestamp') or '-'}")
            if candidate.get("leverage") is not None:
                lines.append(f"杠杆: {candidate.get('leverage')}x")
        if stale:
            lines.append(f"QQQ日线: {stale.get('latest') or '-'} / lag {stale.get('lag_days')}")
        lines.extend(StrategyRouterExecutionEngine._format_diagnostics_lines(diagnostics))
        return "\n".join(lines)

    def _execution_messages(self, payload: dict[str, Any]) -> list[str]:
        out: list[str] = []
        for item in payload.get("execution_results", []) or []:
            if not isinstance(item, dict):
                continue
            strategy = item.get("strategy")
            result = item.get("result")
            statuses = self._action_statuses(result)
            if not statuses:
                continue
            actionable = [status for status in statuses if status not in {"skipped", "paper_flatten_skipped"}]
            if not actionable:
                continue
            out.append(
                "\n".join(
                    [
                        "Router 执行",
                        f"策略: {self._strategy_label(strategy)}",
                        f"动作: {', '.join(statuses)}",
                        *self._format_diagnostics_lines(
                            payload.get("execution_diagnostics")
                            if isinstance(payload.get("execution_diagnostics"), dict)
                            else {}
                        ),
                    ]
                )
            )
        return out

    @staticmethod
    def _format_diagnostics_lines(diagnostics: dict[str, Any], *, max_items: int = 3) -> list[str]:
        if not diagnostics:
            return []
        router = diagnostics.get("router") if isinstance(diagnostics.get("router"), dict) else {}
        data = diagnostics.get("data") if isinstance(diagnostics.get("data"), dict) else {}
        exchange = diagnostics.get("exchange") if isinstance(diagnostics.get("exchange"), dict) else {}
        btc = exchange.get("btc") if isinstance(exchange.get("btc"), dict) else {}
        qqq = exchange.get("qqq") if isinstance(exchange.get("qqq"), dict) else {}
        runtime = diagnostics.get("runtime") if isinstance(diagnostics.get("runtime"), dict) else {}
        items = [item for item in diagnostics.get("items", []) or [] if isinstance(item, dict)]
        lines = [
            f"实盘Gap: {str(diagnostics.get('status') or 'unknown').upper()}",
            (
                "路由/执行: "
                f"{StrategyRouterExecutionEngine._strategy_label(router.get('selected_strategy'))}"
                " -> "
                f"{StrategyRouterExecutionEngine._strategy_label(router.get('current_executed_strategy'))}"
            ),
            (
                "交易所: "
                f"BTC L/S {StrategyRouterExecutionEngine._fmt_num(btc.get('long_contracts'))}/"
                f"{StrategyRouterExecutionEngine._fmt_num(btc.get('short_contracts'))}; "
                f"QQQ ex {StrategyRouterExecutionEngine._fmt_num(qqq.get('exchange_contracts'))}"
                f" / state {StrategyRouterExecutionEngine._fmt_num(qqq.get('local_contracts'))}"
            ),
            (
                "QQQ止损: "
                f"{qqq.get('stop_status') or '-'} @ "
                f"{StrategyRouterExecutionEngine._fmt_num(qqq.get('exchange_stop_price') or qqq.get('local_stop_price'))}"
                f" / algo {StrategyRouterExecutionEngine._short_id(qqq.get('stop_algo_id') or qqq.get('stop_algo_client_id'))}"
            ),
            (
                "数据: "
                f"4h {data.get('data_refresh_status') or '-'}; "
                f"1d {data.get('daily_refresh_status') or '-'}; "
                f"过期={data.get('daily_stale')}; "
                f"日线信号={data.get('daily_signal_timestamp') or '-'}"
            ),
            f"运行: pid {runtime.get('pid') or '-'} / updated {StrategyRouterExecutionEngine._fmt_time(runtime.get('updated_at'))}",
        ]
        alert_items = [item for item in items if item.get("severity") in {"critical", "warning"}]
        if alert_items:
            rendered = [str(item.get("message") or item.get("key") or "-") for item in alert_items[:max_items]]
            suffix = "" if len(alert_items) <= max_items else f" (+{len(alert_items) - max_items})"
            lines.append("Gap项: " + "；".join(rendered) + suffix)
        return lines

    @staticmethod
    def _gap_warning(diagnostics: dict[str, Any]) -> dict[str, str] | None:
        if not diagnostics or diagnostics.get("status") == "ok":
            return None
        items = [
            item
            for item in diagnostics.get("items", []) or []
            if isinstance(item, dict) and item.get("severity") in {"critical", "warning"}
        ]
        if not items:
            return None
        key = "|".join(
            f"{item.get('severity')}:{item.get('key')}"
            for item in sorted(items, key=lambda item: str(item.get("key") or ""))
        )
        return {
            "key": key[:240],
            "message": "\n".join(
                [
                    "Router 实盘Gap告警",
                    *StrategyRouterExecutionEngine._format_diagnostics_lines(diagnostics, max_items=6),
                ]
            ),
        }

    @staticmethod
    def _data_warning(route: dict[str, Any]) -> dict[str, str] | None:
        for candidate in route.get("candidates", []) or []:
            if not isinstance(candidate, dict) or candidate.get("strategy_id") != "qqq_usdt_aggressive":
                continue
            metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
            stale = metadata.get("daily_signal_stale") if isinstance(metadata.get("daily_signal_stale"), dict) else {}
            refresh = metadata.get("daily_signal_refresh") if isinstance(metadata.get("daily_signal_refresh"), dict) else {}
            if bool(stale.get("stale")):
                key = f"qqq_stale:{stale.get('latest')}:{stale.get('lag_days')}"
                return {
                    "key": key,
                    "message": "\n".join(
                        [
                            "Router 数据告警",
                            "QQQ 日线信号过期，QQQ候选已禁用",
                            f"最新: {stale.get('latest') or '-'}",
                            f"滞后: {stale.get('lag_days')} 天",
                        ]
                    ),
                }
            if refresh.get("status") == "error":
                errors = refresh.get("errors") if isinstance(refresh.get("errors"), dict) else {}
                key = "qqq_refresh_error:" + ",".join(sorted(errors))[:120]
                error_symbols = ", ".join(sorted(errors)) if errors else "-"
                return {
                    "key": key,
                    "message": "\n".join(
                        [
                            "Router 数据提示",
                            "本轮 QQQ 日线在线刷新失败，已使用本地缓存",
                            f"失败标的: {error_symbols}",
                            f"缓存最新: {stale.get('latest') or '-'}",
                            f"滞后: {stale.get('lag_days', '-')}",
                        ]
                    ),
                }
        return None
