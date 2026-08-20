from __future__ import annotations

import os
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from bot.googl_usdt_executor import GooglUsdtExecutionEngine
from bot.gold_usdt_executor import GoldUsdtExecutionEngine
from bot.okx_executor import OkxExecutionEngine
from bot.qqq_usdt_executor import QqqUsdtExecutionEngine
from bot.strategy_router import RoutedSignalCandidate, StrategyRouter


class EvaluationTimeoutError(TimeoutError):
    pass


class StrategyRouterExecutionEngine:
    def __init__(self, router: StrategyRouter):
        self.router = router
        self.config = router.config
        self.btc_executor = OkxExecutionEngine.from_file(self.config.btc_strategy_config)
        self.qqq_executor = QqqUsdtExecutionEngine(self.config, self.config.qqq_strategy_config)
        self.googl_executor = GooglUsdtExecutionEngine(self.config, self.config.googl_strategy_config)
        self.gold_executor = GoldUsdtExecutionEngine(self.config, self.config.gold_strategy_config)
        self.router.candidate_preprocessor = self._preprocess_route_candidates
        self.execution_state_path = self.router.state_path.with_suffix(self.router.state_path.suffix + ".execution")
        self.heartbeat_path = self._resolve_heartbeat_path()

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
            "googl": self.googl_executor.bootstrap(),
            "gold": self.gold_executor.bootstrap(),
        }
        if bool(self.config.telegram_notify_startup):
            self._send_telegram(self._format_startup_message(payload))
        return payload

    def evaluate_latest(self) -> dict[str, Any]:
        stored_previous_executed = self._current_executed_strategy()
        position_sync = self._sync_executed_strategy_with_exchange(stored_previous_executed)
        previous_executed = self._current_executed_strategy()
        route = self.router.evaluate_latest(current_strategy=previous_executed)
        selected_strategy = route.get("selected_strategy")
        selected_candidate = route.get("selected_candidate") if isinstance(route.get("selected_candidate"), dict) else None
        qqq_candidate = self._candidate_from_payload(selected_candidate) if selected_strategy == "qqq_usdt_aggressive" else None
        btc_candidate = self._candidate_from_payload(selected_candidate) if selected_strategy == "btc_sota" else None
        googl_candidate = self._candidate_from_payload(selected_candidate) if selected_strategy == "googl_usdt_aggressive" else None
        gold_candidate = self._candidate_from_payload(selected_candidate) if selected_strategy == "gold_usdt_trend" else None
        external_qqq_flat_sync = self._sync_external_qqq_flat_after_route(position_sync, route)
        if external_qqq_flat_sync is not None:
            position_sync["qqq_external_flat_sync"] = external_qqq_flat_sync
        external_googl_flat_sync = self._sync_external_googl_flat_after_route(position_sync, route)
        if external_googl_flat_sync is not None:
            position_sync["googl_external_flat_sync"] = external_googl_flat_sync
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
            else:
                shadow_gate = self.qqq_executor.shadow_gate_pre_switch_status(qqq_candidate)
                if not bool(shadow_gate.get("allow", True)):
                    execution_results.append(
                        {
                            "strategy": "qqq_usdt_aggressive",
                            "result": {
                                "status": "skipped",
                                "reason": "qqq_shadow_gate_blocked_before_switch",
                                "shadow_gate": shadow_gate,
                            },
                        }
                    )
                    selected_strategy = previous_executed
            if selected_strategy != "qqq_usdt_aggressive":
                selected_candidate = self._candidate_payload_for_strategy(route, selected_strategy)
                qqq_candidate = (
                    self._candidate_from_payload(selected_candidate)
                    if selected_strategy == "qqq_usdt_aggressive"
                    else None
                )
                googl_candidate = (
                    self._candidate_from_payload(selected_candidate)
                    if selected_strategy == "googl_usdt_aggressive"
                    else None
                )

        if selected_strategy == "btc_sota" and previous_executed != "btc_sota":
            btc_shadow_gate = self._btc_pre_switch_status(btc_candidate)
            btc_open_check = self._btc_pre_switch_open_confirmation()
            if not bool(btc_shadow_gate.get("allow", True)):
                execution_results.append(
                    {
                        "strategy": "btc_sota",
                        "result": {
                            "status": "skipped",
                            "reason": "btc_shadow_gate_blocked_before_switch",
                            "shadow_gate": btc_shadow_gate,
                        },
                    }
                )
                selected_strategy = previous_executed
                selected_candidate = self._candidate_payload_for_strategy(route, selected_strategy)
                qqq_candidate = (
                    self._candidate_from_payload(selected_candidate)
                    if selected_strategy == "qqq_usdt_aggressive"
                    else None
                )
                googl_candidate = (
                    self._candidate_from_payload(selected_candidate)
                    if selected_strategy == "googl_usdt_aggressive"
                    else None
                )
            elif not bool(btc_open_check.get("allow", True)):
                execution_results.append(
                    {
                        "strategy": "btc_sota",
                        "result": {
                            "status": "skipped",
                            "reason": "btc_open_not_confirmed_before_switch",
                            "open_check": btc_open_check,
                        },
                    }
                )
                selected_strategy = previous_executed
                selected_candidate = self._candidate_payload_for_strategy(route, selected_strategy)
                qqq_candidate = (
                    self._candidate_from_payload(selected_candidate)
                    if selected_strategy == "qqq_usdt_aggressive"
                    else None
                )
                googl_candidate = (
                    self._candidate_from_payload(selected_candidate)
                    if selected_strategy == "googl_usdt_aggressive"
                    else None
                )

        if selected_strategy == "googl_usdt_aggressive" and previous_executed != "googl_usdt_aggressive":
            risk_on_window = self.googl_executor.risk_on_window_status()
            if not bool(risk_on_window.get("open")):
                execution_results.append(
                    {
                        "strategy": "googl_usdt_aggressive",
                        "result": {
                            "status": "skipped",
                            "reason": "googl_risk_on_window_closed_before_switch",
                            "market_window": risk_on_window,
                        },
                    }
                )
                selected_strategy = previous_executed
            else:
                shadow_gate = self.googl_executor.shadow_gate_pre_switch_status(googl_candidate)
                if not bool(shadow_gate.get("allow", True)):
                    execution_results.append(
                        {
                            "strategy": "googl_usdt_aggressive",
                            "result": {
                                "status": "skipped",
                                "reason": "googl_shadow_gate_blocked_before_switch",
                                "shadow_gate": shadow_gate,
                            },
                        }
                    )
                    selected_strategy = previous_executed
            if selected_strategy != "googl_usdt_aggressive":
                selected_candidate = self._candidate_payload_for_strategy(route, selected_strategy)
                qqq_candidate = (
                    self._candidate_from_payload(selected_candidate)
                    if selected_strategy == "qqq_usdt_aggressive"
                    else None
                )
                googl_candidate = (
                    self._candidate_from_payload(selected_candidate)
                    if selected_strategy == "googl_usdt_aggressive"
                    else None
                )

        rollback_context = None
        hold_incumbent = False
        hold_incumbent_reason = "switch_rollback_cooldown_hold"
        if selected_strategy != previous_executed and previous_executed and bool(self.config.flatten_before_switch):
            googl_cooldown_remaining = self._googl_switch_cooldown_remaining(previous_executed, selected_strategy)
            if googl_cooldown_remaining is not None:
                execution_results.append(
                    {
                        "strategy": selected_strategy,
                        "result": {
                            "status": "skipped",
                            "reason": "googl_switch_cooldown",
                            "cooldown_remaining_seconds": round(float(googl_cooldown_remaining), 1),
                        },
                    }
                )
                selected_strategy = previous_executed
                selected_candidate = self._candidate_payload_for_strategy(route, selected_strategy)
                qqq_candidate = (
                    self._candidate_from_payload(selected_candidate)
                    if selected_strategy == "qqq_usdt_aggressive"
                    else None
                )
                googl_candidate = (
                    self._candidate_from_payload(selected_candidate)
                    if selected_strategy == "googl_usdt_aggressive"
                    else None
                )
                hold_incumbent = True
                hold_incumbent_reason = "googl_switch_cooldown_hold"
            elif self._switch_rollback_guard_active(previous_executed):
                execution_results.append(
                    {
                        "strategy": selected_strategy,
                        "result": {
                            "status": "skipped",
                            "reason": "switch_rollback_cooldown",
                            "rollback_guard": self._switch_rollback_guard(),
                        },
                    }
                )
                selected_strategy = previous_executed
                selected_candidate = self._candidate_payload_for_strategy(route, selected_strategy)
                qqq_candidate = (
                    self._candidate_from_payload(selected_candidate)
                    if selected_strategy == "qqq_usdt_aggressive"
                    else None
                )
                googl_candidate = (
                    self._candidate_from_payload(selected_candidate)
                    if selected_strategy == "googl_usdt_aggressive"
                    else None
                )
                hold_incumbent = True
            else:
                rollback_context = self._capture_rollback_context(previous_executed)

        if selected_strategy != previous_executed and bool(self.config.flatten_before_switch):
            execution_results.extend(self._flatten_strategy(previous_executed, reason=f"router_switch_to_{selected_strategy or 'cash'}"))
            if not self._flatten_confirmed(execution_results):
                self._set_current_executed_strategy(previous_executed)
                payload = {
                    "status": "blocked",
                    "mode": self.config.mode,
                    "route": route,
                    "stored_previous_executed_strategy": stored_previous_executed,
                    "previous_executed_strategy": previous_executed,
                    "current_executed_strategy": self._current_executed_strategy(),
                    "exchange_position_sync": position_sync,
                    "execution_results": execution_results,
                    "blocked_reason": "flatten_not_confirmed",
                    "updated_at": int(time.time()),
                }
                self._maybe_send_telegram_notifications(payload)
                execution_state = self._load_execution_state()
                execution_state.update(
                    {
                        "current_executed_strategy": self._current_executed_strategy(),
                        "exchange_position_sync": position_sync,
                        "last_status": payload,
                        "updated_at": int(time.time()),
                    }
                )
                self._save_execution_state(execution_state)
                return payload

        if hold_incumbent:
            execution_results.append(
                {
                    "strategy": previous_executed,
                    "result": {
                        "status": "held",
                        "reason": hold_incumbent_reason,
                    },
                }
            )
            self._set_current_executed_strategy(previous_executed)
        elif selected_strategy == "btc_sota":
            btc_result = None
            btc_error = None
            try:
                btc_result = self.btc_executor.evaluate_latest()
            except Exception as exc:
                btc_error = str(exc)
            if btc_error is not None:
                execution_results.append(
                    {
                        "strategy": "btc_sota",
                        "result": {"status": "error", "reason": "btc_evaluate_latest_failed", "error": btc_error},
                    }
                )
                self._rollback_after_switch_failure(
                    previous_executed,
                    rollback_context,
                    reason="btc_evaluate_latest_failed",
                    execution_results=execution_results,
                )
            elif self._btc_position_open_confirmed(btc_result):
                execution_results.append({"strategy": "btc_sota", "result": btc_result})
                self._set_current_executed_strategy("btc_sota")
            else:
                execution_results.append({"strategy": "btc_sota", "result": btc_result})
                self._rollback_after_switch_failure(
                    previous_executed,
                    rollback_context,
                    reason="btc_open_not_confirmed",
                    execution_results=execution_results,
                )
        elif selected_strategy == "qqq_usdt_aggressive":
            candidate = qqq_candidate or self._candidate_from_payload(selected_candidate)
            qqq_result = None
            qqq_error = None
            try:
                qqq_result = self.qqq_executor.evaluate_latest(candidate)
            except Exception as exc:
                qqq_error = str(exc)
            if qqq_error is not None:
                execution_results.append(
                    {
                        "strategy": "qqq_usdt_aggressive",
                        "result": {"status": "error", "reason": "qqq_evaluate_latest_failed", "error": qqq_error},
                    }
                )
                self._rollback_after_switch_failure(
                    previous_executed,
                    rollback_context,
                    reason="qqq_evaluate_latest_failed",
                    execution_results=execution_results,
                )
            elif bool(qqq_result.get("position_open")):
                execution_results.append({"strategy": "qqq_usdt_aggressive", "result": qqq_result})
                self._set_current_executed_strategy("qqq_usdt_aggressive")
                if self._is_googl_switch(previous_executed, "qqq_usdt_aggressive"):
                    self._record_googl_switch()
            else:
                execution_results.append({"strategy": "qqq_usdt_aggressive", "result": qqq_result})
                self._rollback_after_switch_failure(
                    previous_executed,
                    rollback_context,
                    reason="qqq_open_not_confirmed",
                    execution_results=execution_results,
                )
        elif selected_strategy == "googl_usdt_aggressive":
            candidate = googl_candidate or self._candidate_from_payload(selected_candidate)
            googl_result = None
            googl_error = None
            try:
                googl_result = self.googl_executor.evaluate_latest(candidate)
            except Exception as exc:
                googl_error = str(exc)
            if googl_error is not None:
                execution_results.append(
                    {
                        "strategy": "googl_usdt_aggressive",
                        "result": {"status": "error", "reason": "googl_evaluate_latest_failed", "error": googl_error},
                    }
                )
                self._rollback_after_switch_failure(
                    previous_executed,
                    rollback_context,
                    reason="googl_evaluate_latest_failed",
                    execution_results=execution_results,
                )
            elif bool(googl_result.get("position_open")):
                execution_results.append({"strategy": "googl_usdt_aggressive", "result": googl_result})
                self._set_current_executed_strategy("googl_usdt_aggressive")
                if self._is_googl_switch(previous_executed, "googl_usdt_aggressive"):
                    self._record_googl_switch()
            else:
                execution_results.append({"strategy": "googl_usdt_aggressive", "result": googl_result})
                self._rollback_after_switch_failure(
                    previous_executed,
                    rollback_context,
                    reason="googl_open_not_confirmed",
                    execution_results=execution_results,
                )
        else:
            if bool(self.config.flatten_on_no_signal):
                execution_results.extend(self._flatten_strategy(previous_executed, reason="router_no_signal"))
            self._set_current_executed_strategy(None)

        payload = {
            "status": "ok",
            "mode": self.config.mode,
            "route": route,
            "stored_previous_executed_strategy": stored_previous_executed,
            "previous_executed_strategy": previous_executed,
            "current_executed_strategy": self._current_executed_strategy(),
            "exchange_position_sync": position_sync,
            "execution_results": execution_results,
            "updated_at": int(time.time()),
        }
        self._maybe_send_telegram_notifications(payload)
        execution_state = self._load_execution_state()
        execution_state.update(
            {
                "current_executed_strategy": self._current_executed_strategy(),
                "exchange_position_sync": position_sync,
                "last_status": payload,
                "updated_at": int(time.time()),
            }
        )
        self._save_execution_state(execution_state)
        return payload

    def _preprocess_route_candidates(self, candidates: list[RoutedSignalCandidate]) -> list[RoutedSignalCandidate]:
        processed: list[RoutedSignalCandidate] = []
        for candidate in candidates:
            if candidate.strategy_id not in ("qqq_usdt_aggressive", "googl_usdt_aggressive"):
                processed.append(candidate)
                continue
            executor = (
                self.qqq_executor if candidate.strategy_id == "qqq_usdt_aggressive" else self.googl_executor
            )
            shadow_gate = executor.shadow_gate_observe_candidate(candidate)
            if not bool(shadow_gate.get("allow", True)) and self._current_executed_strategy() != candidate.strategy_id:
                metadata = dict(candidate.metadata)
                metadata["shadow_gate"] = shadow_gate
                metadata["shadow_gate_blocked"] = True
                metadata["pre_shadow_gate_active"] = bool(candidate.active)
                metadata["pre_shadow_gate_route_score"] = float(candidate.route_score)
                processed.append(
                    RoutedSignalCandidate(
                        strategy_id=candidate.strategy_id,
                        symbol=candidate.symbol,
                        active=False,
                        route_score=0.0,
                        timestamp=candidate.timestamp,
                        direction=None,
                        event_type=None,
                        leverage=None,
                        strength_label="shadow_gate_blocked",
                        source_config=candidate.source_config,
                        metadata=metadata,
                    )
                )
                continue
            metadata = dict(candidate.metadata)
            metadata["shadow_gate"] = shadow_gate
            processed.append(
                RoutedSignalCandidate(
                    strategy_id=candidate.strategy_id,
                    symbol=candidate.symbol,
                    active=candidate.active,
                    route_score=candidate.route_score,
                    timestamp=candidate.timestamp,
                    direction=candidate.direction,
                    event_type=candidate.event_type,
                    leverage=candidate.leverage,
                    strength_label=candidate.strength_label,
                    source_config=candidate.source_config,
                    metadata=metadata,
                )
            )
        return processed

    @staticmethod
    def _btc_position_open_confirmed(result: dict[str, Any]) -> bool:
        if bool(result.get("position_open")):
            return True
        snapshot = result.get("snapshot")
        if isinstance(snapshot, dict) and snapshot.get("position") is not None:
            return True
        return False

    def _btc_pre_switch_status(self, candidate: RoutedSignalCandidate | None) -> dict[str, Any]:
        status_fn = getattr(self.btc_executor, "shadow_gate_pre_switch_status", None)
        if status_fn is None:
            return {"enabled": False, "allow": True, "reason": "unavailable"}
        status = status_fn(candidate)
        return status if isinstance(status, dict) else {"enabled": False, "allow": True, "reason": "invalid_status"}

    def _btc_pre_switch_open_confirmation(self) -> dict[str, Any]:
        """Dry-run whether the BTC executor would actually open right now.

        Guards against a BTC takeover that flattens QQQ but then cannot open BTC
        (arbitration/score-gate rejection at a new candle boundary, sizing failure,
        high-leverage guard, etc.). Any rejection here holds the incumbent instead.
        """
        confirm_fn = getattr(self.btc_executor, "pre_switch_open_confirm", None)
        if confirm_fn is None:
            return {"enabled": False, "allow": True, "reason": "unavailable"}
        try:
            confirmation = confirm_fn()
        except Exception as exc:
            return {"enabled": True, "allow": False, "reason": "btc_open_confirm_error", "error": str(exc)}
        if not isinstance(confirmation, dict):
            return {"enabled": False, "allow": True, "reason": "invalid_confirmation"}
        return confirmation

    @staticmethod
    def _candidate_payload_for_strategy(route: dict[str, Any], strategy_id: str | None) -> dict[str, Any] | None:
        if not strategy_id:
            return None
        for item in route.get("candidates", []) if isinstance(route.get("candidates"), list) else []:
            if isinstance(item, dict) and item.get("strategy_id") == strategy_id:
                return item
        return None

    @staticmethod
    def _flatten_confirmed(results: list[dict[str, Any]]) -> bool:
        for item in results:
            result = item.get("result") if isinstance(item, dict) else None
            if not isinstance(result, dict):
                continue
            status = str(result.get("status") or "")
            if status in {"submitted_but_unconfirmed", "error"}:
                return False
        return True

    def run_loop(self, poll_interval_seconds: int = 30) -> None:
        self._write_heartbeat("bootstrap_started")
        bootstrap = self.bootstrap()
        self._write_heartbeat("bootstrap_completed", status=bootstrap.get("status"))
        print(json.dumps({"event": "bootstrap", **bootstrap}, ensure_ascii=False))
        while True:
            timeout_seconds = self._evaluation_timeout_seconds()
            started_at = time.time()
            self._write_heartbeat(
                "evaluate_started",
                timeout_seconds=timeout_seconds,
                deadline_at=started_at + timeout_seconds if timeout_seconds > 0 else None,
            )
            try:
                status = self._evaluate_latest_with_timeout(timeout_seconds)
                self._write_heartbeat(
                    "evaluate_completed",
                    status=status.get("status"),
                    duration_seconds=round(time.time() - started_at, 3),
                )
                print(json.dumps({"event": "evaluate", **status}, ensure_ascii=False))
            except KeyboardInterrupt:
                self._write_heartbeat("stopped")
                print(json.dumps({"event": "stopped"}, ensure_ascii=False))
                raise
            except EvaluationTimeoutError as exc:
                status = self._timeout_status(str(exc), timeout_seconds, started_at)
                self._save_loop_status(status)
                self._write_heartbeat(
                    "evaluate_timeout",
                    error=str(exc),
                    timeout_seconds=timeout_seconds,
                    duration_seconds=round(time.time() - started_at, 3),
                )
                print(json.dumps({"event": "error", **status}, ensure_ascii=False))
                if bool(self.config.telegram_notify_errors):
                    self._send_telegram(f"Router 评估超时\n错误: {exc}")
            except Exception as exc:
                status = self._exception_status(exc, started_at)
                self._save_loop_status(status)
                self._write_heartbeat(
                    "evaluate_error",
                    error=str(exc),
                    duration_seconds=round(time.time() - started_at, 3),
                )
                print(json.dumps({"event": "error", "error": str(exc)}, ensure_ascii=False))
                if bool(self.config.telegram_notify_errors):
                    self._send_telegram(f"Router 异常\n错误: {exc}")
            time.sleep(max(1, int(poll_interval_seconds)))

    def _evaluation_timeout_seconds(self) -> float:
        configured = self.config.router_evaluation_timeout_seconds
        if configured is None:
            return 240.0
        try:
            return max(0.0, float(configured))
        except (TypeError, ValueError):
            return 240.0

    def _evaluate_latest_with_timeout(self, timeout_seconds: float) -> dict[str, Any]:
        if timeout_seconds <= 0 or not hasattr(signal, "SIGALRM"):
            return self.evaluate_latest()

        def _raise_timeout(_signum: int, _frame: Any) -> None:
            raise EvaluationTimeoutError(f"evaluate_latest exceeded {timeout_seconds:.1f}s")

        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        try:
            return self.evaluate_latest()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])

    def _resolve_heartbeat_path(self) -> Path:
        configured = self.config.router_heartbeat_path
        if configured:
            return Path(configured)
        return self.router.state_path.with_suffix(self.router.state_path.suffix + ".heartbeat")

    def _write_heartbeat(self, phase: str, **extra: Any) -> None:
        payload = {
            "phase": phase,
            "pid": os.getpid(),
            "updated_at": int(time.time()),
            "updated_at_iso": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        try:
            self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.heartbeat_path.with_suffix(self.heartbeat_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            tmp.replace(self.heartbeat_path)
        except Exception:
            return

    def _timeout_status(self, error: str, timeout_seconds: float, started_at: float) -> dict[str, Any]:
        return {
            "status": "error",
            "mode": self.config.mode,
            "error_type": "evaluation_timeout",
            "error": error,
            "timeout_seconds": timeout_seconds,
            "duration_seconds": round(time.time() - started_at, 3),
            "current_executed_strategy": self._current_executed_strategy(),
            "updated_at": int(time.time()),
        }

    def _exception_status(self, exc: Exception, started_at: float) -> dict[str, Any]:
        return {
            "status": "error",
            "mode": self.config.mode,
            "error_type": "evaluation_exception",
            "error": str(exc),
            "duration_seconds": round(time.time() - started_at, 3),
            "current_executed_strategy": self._current_executed_strategy(),
            "updated_at": int(time.time()),
        }

    def _save_loop_status(self, status: dict[str, Any]) -> None:
        execution_state = self._load_execution_state()
        execution_state.update(
            {
                "current_executed_strategy": self._current_executed_strategy(),
                "last_status": status,
                "updated_at": int(time.time()),
            }
        )
        self._save_execution_state(execution_state)

    def _sync_executed_strategy_with_exchange(self, stored_strategy: str | None) -> dict[str, Any]:
        if self.config.mode != "live":
            return {"status": "skipped", "reason": "not_live", "current_executed_strategy": stored_strategy}
        sync = self._fetch_exchange_position_sync()
        if sync.get("status") != "ok":
            return sync
        exchange_strategy = self._executed_strategy_from_position_sync(sync, stored_strategy)
        if exchange_strategy == stored_strategy:
            state = self._load_execution_state()
            state["exchange_position_sync"] = sync
            state["updated_at"] = int(time.time())
            self._save_execution_state(state)
            return {**sync, "current_executed_strategy": stored_strategy, "synced": False}
        if stored_strategy == "qqq_usdt_aggressive" and exchange_strategy is None:
            sync["requires_qqq_external_flat_sync"] = True
        if stored_strategy == "googl_usdt_aggressive" and exchange_strategy is None:
            sync["requires_googl_external_flat_sync"] = True
        self._set_current_executed_strategy(exchange_strategy)
        state = self._load_execution_state()
        state["exchange_position_sync"] = sync
        state["last_external_execution_sync"] = {
            "from": stored_strategy,
            "to": exchange_strategy,
            "reason": "exchange_position_mismatch",
            "updated_at": int(time.time()),
        }
        state["updated_at"] = int(time.time())
        self._save_execution_state(state)
        return {
            **sync,
            "synced": True,
            "previous_current_executed_strategy": stored_strategy,
            "current_executed_strategy": exchange_strategy,
        }

    def _sync_external_qqq_flat_after_route(self, position_sync: dict[str, Any], route: dict[str, Any]) -> dict[str, Any] | None:
        if not bool(position_sync.get("requires_qqq_external_flat_sync")):
            return None
        candidate = self._route_candidate(route, "qqq_usdt_aggressive")
        context = None
        if candidate is not None and hasattr(self.qqq_executor, "_build_context"):
            try:
                context = self.qqq_executor._build_context(candidate)
            except Exception as exc:
                return {
                    "status": "error",
                    "reason": "qqq_external_flat_context_failed",
                    "error": str(exc),
                }
        try:
            return self.qqq_executor.sync_external_flat(reason="router_exchange_flat_sync", context=context)
        except TypeError:
            return self.qqq_executor.sync_external_flat(reason="router_exchange_flat_sync")
        except Exception as exc:
            return {"status": "error", "reason": "qqq_external_flat_sync_failed", "error": str(exc)}

    def _sync_external_googl_flat_after_route(self, position_sync: dict[str, Any], route: dict[str, Any]) -> dict[str, Any] | None:
        if not bool(position_sync.get("requires_googl_external_flat_sync")):
            return None
        candidate = self._route_candidate(route, "googl_usdt_aggressive")
        context = None
        if candidate is not None and hasattr(self.googl_executor, "_build_context"):
            try:
                context = self.googl_executor._build_context(candidate)
            except Exception as exc:
                return {
                    "status": "error",
                    "reason": "googl_external_flat_context_failed",
                    "error": str(exc),
                }
        try:
            return self.googl_executor.sync_external_flat(reason="router_exchange_flat_sync", context=context)
        except TypeError:
            return self.googl_executor.sync_external_flat(reason="router_exchange_flat_sync")
        except Exception as exc:
            return {"status": "error", "reason": "googl_external_flat_sync_failed", "error": str(exc)}

    def _route_candidate(self, route: dict[str, Any], strategy_id: str) -> RoutedSignalCandidate | None:
        selected = route.get("selected_candidate") if isinstance(route.get("selected_candidate"), dict) else None
        if selected and selected.get("strategy_id") == strategy_id:
            return self._candidate_from_payload(selected)
        for payload in route.get("candidates", []) or []:
            if isinstance(payload, dict) and payload.get("strategy_id") == strategy_id:
                return self._candidate_from_payload(payload)
        return None

    def _fetch_exchange_position_sync(self) -> dict[str, Any]:
        try:
            qqq_position = self.qqq_executor.fetch_position_state()
            googl_position = self.googl_executor.fetch_position_state()
            btc_long = self.btc_executor._fetch_position_state("long")
            btc_short = self.btc_executor._fetch_position_state("short")
        except Exception as exc:
            return {"status": "error", "reason": "exchange_position_sync_failed", "error": str(exc)}
        return {
            "status": "ok",
            "btc_long_contracts": float(btc_long.get("contracts", 0.0) or 0.0),
            "btc_short_contracts": float(btc_short.get("contracts", 0.0) or 0.0),
            "qqq_contracts": float(qqq_position.get("contracts", 0.0) or 0.0),
            "qqq_notional_usdt": float(qqq_position.get("notional_usdt", 0.0) or 0.0),
            "googl_contracts": float(googl_position.get("contracts", 0.0) or 0.0),
            "googl_notional_usdt": float(googl_position.get("notional_usdt", 0.0) or 0.0),
            "updated_at": int(time.time()),
        }

    @staticmethod
    def _executed_strategy_from_position_sync(sync: dict[str, Any], fallback: str | None = None) -> str | None:
        btc_open = (
            float(sync.get("btc_long_contracts", 0.0) or 0.0) > 0
            or float(sync.get("btc_short_contracts", 0.0) or 0.0) > 0
        )
        qqq_open = float(sync.get("qqq_contracts", 0.0) or 0.0) > 0
        googl_open = float(sync.get("googl_contracts", 0.0) or 0.0) > 0
        open_strategies = int(btc_open) + int(qqq_open) + int(googl_open)
        if open_strategies > 1:
            return fallback
        if btc_open:
            return "btc_sota"
        if qqq_open:
            return "qqq_usdt_aggressive"
        if googl_open:
            return "googl_usdt_aggressive"
        return None

    def _candidate_from_payload(self, payload: dict[str, Any] | None) -> RoutedSignalCandidate | None:
        if not payload:
            return None
        strategy_id = str(payload.get("strategy_id") or "qqq_usdt_aggressive")
        symbol = payload.get("symbol")
        if symbol in (None, ""):
            symbol = getattr(self.qqq_executor, "symbol", "QQQ/USDT:USDT")
            if strategy_id == "googl_usdt_aggressive":
                symbol = getattr(self.googl_executor, "symbol", "GOOGL/USDT:USDT")
            elif strategy_id == "btc_sota":
                try:
                    symbol = self.btc_executor.config.symbol
                except AttributeError:
                    symbol = "BTC/USDT:USDT"
        return RoutedSignalCandidate(
            strategy_id=strategy_id,
            symbol=str(symbol or "QQQ/USDT:USDT"),
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

    def _flatten_strategy(self, strategy_id: str | None, *, reason: str) -> list[dict[str, Any]]:
        if strategy_id == "qqq_usdt_aggressive":
            return [{"strategy": strategy_id, "result": self.qqq_executor.close_position(reason=reason)}]
        if strategy_id == "googl_usdt_aggressive":
            return [{"strategy": strategy_id, "result": self.googl_executor.close_position(reason=reason)}]
        if strategy_id == "btc_sota":
            return [{"strategy": strategy_id, "result": self._flatten_btc(reason=reason)}]
        return []

    def _flatten_btc(self, *, reason: str) -> dict[str, Any]:
        return self.btc_executor.close_for_router_switch(reason=reason)

    def _capture_rollback_context(self, strategy_id: str | None) -> dict[str, Any] | None:
        """Snapshot the incumbent before flattening so a failed switch can restore it."""
        if not strategy_id:
            return None
        context: dict[str, Any] = {"strategy": strategy_id, "captured_at": datetime.now(timezone.utc).isoformat()}
        try:
            if strategy_id == "qqq_usdt_aggressive":
                state = self.qqq_executor.load_state()
                position = state.get("position") if isinstance(state.get("position"), dict) else None
                exchange_position = self.qqq_executor.fetch_position_state()
                context["position"] = position
                context["exchange_position"] = exchange_position if isinstance(exchange_position, dict) else {}
            elif strategy_id == "googl_usdt_aggressive":
                state = self.googl_executor.load_state()
                position = state.get("position") if isinstance(state.get("position"), dict) else None
                exchange_position = self.googl_executor.fetch_position_state()
                context["position"] = position
                context["exchange_position"] = exchange_position if isinstance(exchange_position, dict) else {}
            elif strategy_id == "btc_sota":
                btc_state = self.btc_executor.store.load_snapshot() or {}
                context["position"] = btc_state.get("position") if isinstance(btc_state.get("position"), dict) else None
        except Exception as exc:
            context["capture_error"] = str(exc)
        return context

    def _exchange_position_open_state(self) -> tuple[bool, bool, bool, dict[str, Any]] | None:
        """Return (btc_open, qqq_open, googl_open, sync) from the live exchange, or None on failure.

        Used before a rollback restore so the router does not re-open an incumbent
        on top of a target that actually filled, and does not restore an incumbent
        that was never flattened on the exchange in the first place.
        """
        try:
            sync = self._fetch_exchange_position_sync()
        except Exception:
            return None
        if not isinstance(sync, dict) or sync.get("status") != "ok":
            return None
        btc_open = (
            float(sync.get("btc_long_contracts", 0.0) or 0.0) > 0
            or float(sync.get("btc_short_contracts", 0.0) or 0.0) > 0
        )
        qqq_open = float(sync.get("qqq_contracts", 0.0) or 0.0) > 0
        googl_open = float(sync.get("googl_contracts", 0.0) or 0.0) > 0
        return (btc_open, qqq_open, googl_open, sync)

    @staticmethod
    def _rollback_strategy_tag(strategy_id: str | None) -> str:
        return {
            "btc_sota": "btc",
            "qqq_usdt_aggressive": "qqq",
            "googl_usdt_aggressive": "googl",
        }.get(str(strategy_id or ""), str(strategy_id or "unknown"))

    def _rollback_after_switch_failure(
        self,
        previous_executed: str | None,
        rollback_context: dict[str, Any] | None,
        *,
        reason: str,
        execution_results: list[dict[str, Any]],
    ) -> None:
        """Restore the incumbent after a failed switch so the account is not left flat."""
        if rollback_context is None:
            # No incumbent was flattened this cycle (the selected strategy was
            # already being managed, or the account was flat). Keep the flat
            # outcome; the next cycle reconciles current_executed_strategy with
            # the exchange.
            self._set_current_executed_strategy(None)
            return
        incumbent = rollback_context.get("strategy") or previous_executed
        if incumbent in ("qqq_usdt_aggressive", "googl_usdt_aggressive"):
            self._rollback_stock_incumbent(incumbent, rollback_context, reason=reason, execution_results=execution_results)
        elif incumbent == "btc_sota":
            try:
                redo = self.btc_executor.evaluate_latest()
            except Exception as exc:
                redo = {"status": "error", "reason": "btc_restore_evaluate_failed", "error": str(exc)}
            execution_results.append(
                {
                    "strategy": "btc_sota",
                    "result": redo,
                    "rollback": True,
                    "rollback_reason": reason,
                }
            )
            if self._btc_position_open_confirmed(redo):
                self._set_current_executed_strategy("btc_sota")
                self._record_switch_rollback("btc_sota")
            else:
                self._set_current_executed_strategy(None)
        else:
            self._set_current_executed_strategy(None)

    def _rollback_stock_incumbent(
        self,
        incumbent: str,
        rollback_context: dict[str, Any] | None,
        *,
        reason: str,
        execution_results: list[dict[str, Any]],
    ) -> None:
        """Restore a QQQ/GOOGL incumbent after a failed switch, exchange-aware.

        Adopts the single non-incumbent strategy that actually opened on the
        exchange (so we never stack a restore on top of a filled target), retains
        the incumbent when it was never flattened, keeps the stored incumbent on
        ambiguous double-open, and only re-buys at market when the account is flat.
        """
        exchange_state = self._exchange_position_open_state()
        if exchange_state is not None:
            btc_open, qqq_open, googl_open, exchange_sync = exchange_state
            open_strategies: list[str] = []
            if btc_open:
                open_strategies.append("btc_sota")
            if qqq_open:
                open_strategies.append("qqq_usdt_aggressive")
            if googl_open:
                open_strategies.append("googl_usdt_aggressive")
            incumbent_open = incumbent in open_strategies
            others_open = [strategy for strategy in open_strategies if strategy != incumbent]

            if not incumbent_open and len(others_open) == 1:
                # The target (or another strategy) actually opened despite the
                # failure signal; restoring the incumbent on top of it would
                # create a double position. Adopt the opened strategy; the next
                # cycle's exchange sync reconciles residual mismatch.
                adopted = others_open[0]
                execution_results.append(
                    {
                        "strategy": adopted,
                        "result": {
                            "status": f"{self._rollback_strategy_tag(adopted)}_open_detected_on_rollback",
                            "rollback_reason": reason,
                            "exchange_position_sync": exchange_sync,
                        },
                        "rollback": True,
                        "rollback_reason": reason,
                    }
                )
                self._set_current_executed_strategy(adopted)
                return
            if incumbent_open and others_open:
                # Double position -- never add more. Keep the stored incumbent
                # and let the next cycle's exchange sync reconcile.
                self._set_current_executed_strategy(incumbent)
                return
            if not incumbent_open and len(others_open) > 1:
                # Multiple non-incumbent strategies open -- ambiguous. Keep the
                # stored incumbent and let the next cycle reconcile.
                self._set_current_executed_strategy(incumbent)
                return
            if incumbent_open:
                # The incumbent was never actually flattened on the exchange, so
                # there is nothing to restore -- the position we want is already open.
                execution_results.append(
                    {
                        "strategy": incumbent,
                        "result": {
                            "status": f"{self._rollback_strategy_tag(incumbent)}_retained_on_rollback",
                            "rollback_reason": reason,
                            "exchange_position_sync": exchange_sync,
                        },
                        "rollback": True,
                        "rollback_reason": reason,
                    }
                )
                self._set_current_executed_strategy(incumbent)
                return

        executor = self.qqq_executor if incumbent == "qqq_usdt_aggressive" else self.googl_executor
        try:
            restore = executor.restore_position(rollback_context)
        except Exception as exc:
            restore = {"status": "error", "reason": f"restore_{self._rollback_strategy_tag(incumbent)}_failed", "error": str(exc)}
        execution_results.append(
            {
                "strategy": incumbent,
                "result": restore,
                "rollback": True,
                "rollback_reason": reason,
            }
        )
        if str(restore.get("status")) in {"submitted", "paper_restored"}:
            self._set_current_executed_strategy(incumbent)
            self._record_switch_rollback(incumbent)
        else:
            self._set_current_executed_strategy(None)

    def _record_switch_rollback(self, strategy_id: str) -> None:
        state = self._load_execution_state()
        state["switch_rollback_guard"] = {
            "strategy": strategy_id,
            "until_ts": int(time.time()) + max(0.0, float(self.config.switch_rollback_cooldown_seconds)),
            "updated_at": int(time.time()),
        }
        self._save_execution_state(state)

    def _switch_rollback_guard(self) -> dict[str, Any] | None:
        state = self._load_execution_state()
        guard = state.get("switch_rollback_guard")
        return guard if isinstance(guard, dict) else None

    def _switch_rollback_guard_active(self, incumbent_strategy: str | None) -> bool:
        guard = self._switch_rollback_guard()
        if not isinstance(guard, dict):
            return False
        if str(guard.get("strategy") or "") != str(incumbent_strategy or ""):
            return False
        until_ts = float(guard.get("until_ts", 0.0) or 0.0)
        return time.time() < until_ts

    @staticmethod
    def _is_googl_switch(from_strategy: str | None, to_strategy: str | None) -> bool:
        """True when a switch is between QQQ and GOOGL (either direction)."""
        googl_pair = {"qqq_usdt_aggressive", "googl_usdt_aggressive"}
        return from_strategy in googl_pair and to_strategy in googl_pair and from_strategy != to_strategy

    def _record_googl_switch(self) -> None:
        """Record the wall-clock time of a successful QQQ<->GOOGL switch."""
        state = self._load_execution_state()
        state["last_googl_switch_at"] = int(time.time())
        state["updated_at"] = int(time.time())
        self._save_execution_state(state)

    def _googl_switch_cooldown_remaining(self, from_strategy: str | None, to_strategy: str | None) -> float | None:
        """Seconds remaining on the QQQ<->GOOGL switch cooldown, or None when not constrained."""
        if not self._is_googl_switch(from_strategy, to_strategy):
            return None
        state = self._load_execution_state()
        last_ts = state.get("last_googl_switch_at")
        if last_ts is None:
            return None
        try:
            cooldown_seconds = max(0.0, float(self.config.googl_switch_cooldown_seconds))
        except (TypeError, ValueError):
            cooldown_seconds = 0.0
        remaining = float(last_ts) + cooldown_seconds - time.time()
        if remaining <= 0:
            return None
        return remaining

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
            "googl_usdt_aggressive": "GOOGL/USDT",
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
        btc = payload.get("btc") if isinstance(payload.get("btc"), dict) else {}
        qqq = payload.get("qqq") if isinstance(payload.get("qqq"), dict) else {}
        googl = payload.get("googl") if isinstance(payload.get("googl"), dict) else {}
        btc_status, btc_error = self._bootstrap_component_status(btc)
        qqq_status, qqq_error = self._bootstrap_component_status(qqq)
        googl_status, googl_error = self._bootstrap_component_status(googl)
        lines = [
            "Router 启动",
            f"模式: {payload.get('mode')}",
            "链路: BTC SOTA / QQQ-USDT / GOOGL-USDT",
            f"BTC: {btc_status}",
            f"QQQ: {qqq_status}",
            f"GOOGL: {googl_status}",
        ]
        if btc_error:
            lines.append(f"BTC错误: {btc_error}")
        if qqq_error:
            lines.append(f"QQQ错误: {qqq_error}")
        if googl_error:
            lines.append(f"GOOGL错误: {googl_error}")
        return "\n".join(lines)

    @staticmethod
    def _bootstrap_component_status(component: dict[str, Any]) -> tuple[str, str | None]:
        if not isinstance(component, dict):
            return "异常", "missing bootstrap payload"
        error = component.get("error") or component.get("bootstrap_error")
        if component.get("status") == "error" or error:
            return "异常", str(error) if error else None
        return "ok", None

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
