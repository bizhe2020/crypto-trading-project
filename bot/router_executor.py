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

        if selected_strategy != previous_executed and bool(self.config.flatten_before_switch):
            execution_results.extend(self._flatten_strategy(previous_executed, reason=f"router_switch_to_{selected_strategy or 'cash'}"))

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
        lines = [
            "Router 选路更新",
            f"当前: {StrategyRouterExecutionEngine._strategy_label(payload.get('current_executed_strategy'))}",
            f"上次: {StrategyRouterExecutionEngine._strategy_label(payload.get('previous_executed_strategy'))}",
            f"原因: {route.get('decision_reason') or '-'}",
            f"分数: {route.get('selected_route_score') or 0}",
        ]
        if candidate:
            lines.append(f"信号时间: {candidate.get('timestamp') or '-'}")
            if candidate.get("leverage") is not None:
                lines.append(f"杠杆: {candidate.get('leverage')}x")
        if stale:
            lines.append(f"QQQ日线: {stale.get('latest') or '-'} / lag {stale.get('lag_days')}")
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
                    ]
                )
            )
        return out

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
                return {
                    "key": key,
                    "message": "\n".join(
                        [
                            "Router 数据提示",
                            "本轮 QQQ 日线在线刷新失败，已使用本地缓存",
                            f"缓存最新: {stale.get('latest') or '-'}",
                            f"滞后: {stale.get('lag_days', '-')}",
                        ]
                    ),
                }
        return None
