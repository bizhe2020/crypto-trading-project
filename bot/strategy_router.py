from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RoutedSignalCandidate:
    strategy_id: str
    symbol: str
    active: bool
    route_score: float
    timestamp: str | None = None
    direction: str | None = None
    event_type: str | None = None
    leverage: float | None = None
    strength_label: str | None = None
    source_config: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["route_score"] = round(float(self.route_score), 2)
        if self.leverage is not None:
            payload["leverage"] = round(float(self.leverage), 4)
        return payload


@dataclass
class StrategyRouterConfig:
    mode: str
    state_path: str
    btc_strategy_config: str
    qqq_strategy_config: str
    enable_btc: bool = True
    enable_qqq: bool = True
    btc_min_route_score: float = 30.0
    qqq_min_route_score: float = 45.0
    switch_advantage: float = 8.0
    strategy_priority: list[str] | None = None
    persist_state: bool = True
    execution_enabled: bool = False
    execution_credentials_config: str | None = None
    flatten_before_switch: bool = True
    flatten_on_no_signal: bool = True
    qqq_state_db_path: str = "state/runtime_qqq_usdt_router.db"
    qqq_margin_mode: str = "isolated"
    qqq_position_size_pct: float = 1.0
    qqq_max_notional_usdt: float | None = None
    qqq_min_order_notional_usdt: float = 10.0
    qqq_min_rebalance_notional_usdt: float = 10.0
    qqq_rebalance_on_leverage_change: bool = True
    qqq_enable_exchange_stop: bool = False
    okx_markets_cache_path: str = "var/okx/markets_cache.json"
    telegram_enabled: bool = False
    telegram_token: str | None = None
    telegram_chat_id: str | None = None
    telegram_proxy: str | None = None
    telegram_notify_startup: bool = True
    telegram_notify_route_change: bool = True
    telegram_notify_execution: bool = True
    telegram_notify_data_warnings: bool = True
    telegram_notify_errors: bool = True
    router_audit_log_enabled: bool = True
    router_audit_log_path: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrategyRouterConfig":
        filtered = {key: value for key, value in payload.items() if key in cls.__dataclass_fields__}
        return cls(**filtered)


class StrategyRouter:
    def __init__(self, config: StrategyRouterConfig, config_path: str | Path | None = None):
        self.config = config
        self.config_path = Path(config_path).resolve() if config_path else None
        self.state_path = Path(config.state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.btc_adapter = None
        self.qqq_adapter = None

    @classmethod
    def from_file(cls, path: str | Path) -> "StrategyRouter":
        config_path = Path(path).resolve()
        payload = json.loads(config_path.read_text())
        payload = cls._resolve_paths(config_path.parent, payload)
        return cls(StrategyRouterConfig.from_dict(payload), config_path=config_path)

    @staticmethod
    def _resolve_paths(base: Path, payload: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(payload)
        project_root = Path(__file__).resolve().parents[1]
        for key in [
            "state_path",
            "btc_strategy_config",
            "qqq_strategy_config",
            "execution_credentials_config",
            "qqq_state_db_path",
            "okx_markets_cache_path",
            "router_audit_log_path",
        ]:
            value = resolved.get(key)
            if not isinstance(value, str) or not value:
                continue
            path = Path(value)
            if path.is_absolute():
                resolved[key] = str(path)
                continue
            if path.parts and path.parts[0] in {"config", "state", "data", "bot", "scripts", "var"}:
                resolved[key] = str((project_root / path).resolve())
            else:
                resolved[key] = str((base / path).resolve())
        return resolved

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            payload = json.loads(self.state_path.read_text())
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save_state(self, payload: dict[str, Any]) -> None:
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        tmp.replace(self.state_path)

    def _strategy_priority_value(self, strategy_id: str) -> int:
        configured = self.config.strategy_priority or ["btc_sota", "qqq_usdt_aggressive"]
        try:
            return list(configured).index(strategy_id)
        except ValueError:
            return len(configured)

    def _min_score_for(self, strategy_id: str) -> float:
        if strategy_id == "btc_sota":
            return float(self.config.btc_min_route_score)
        if strategy_id == "qqq_usdt_aggressive":
            return float(self.config.qqq_min_route_score)
        return 0.0

    def _collect_candidates(self) -> list[RoutedSignalCandidate]:
        from bot.btc_signal_adapter import BtcSignalAdapter
        from bot.qqq_usdt_signal_adapter import QqqUsdtSignalAdapter

        if self.btc_adapter is None:
            self.btc_adapter = BtcSignalAdapter(Path(self.config.btc_strategy_config))
        if self.qqq_adapter is None:
            self.qqq_adapter = QqqUsdtSignalAdapter(Path(self.config.qqq_strategy_config))
        candidates: list[RoutedSignalCandidate] = []
        if self.config.enable_btc:
            candidates.append(self.btc_adapter.preview())
        if self.config.enable_qqq:
            candidates.append(self.qqq_adapter.preview())
        return candidates

    def _choose_candidate(
        self,
        candidates: list[RoutedSignalCandidate],
        current_strategy: str | None,
    ) -> tuple[RoutedSignalCandidate | None, str]:
        current_candidate = next(
            (
                candidate
                for candidate in candidates
                if candidate.strategy_id == current_strategy and candidate.active
            ),
            None,
        )
        eligible = [
            candidate
            for candidate in candidates
            if candidate.active and float(candidate.route_score) >= self._min_score_for(candidate.strategy_id)
        ]
        if not eligible:
            if current_candidate is not None:
                return current_candidate, "hold_current_active_below_threshold"
            return None, "no_eligible_candidates"

        eligible.sort(
            key=lambda item: (
                float(item.route_score),
                -self._strategy_priority_value(item.strategy_id),
            ),
            reverse=True,
        )
        best = eligible[0]
        current = current_candidate or next((item for item in eligible if item.strategy_id == current_strategy), None)
        if current is not None and current.strategy_id != best.strategy_id:
            score_delta = float(best.route_score) - float(current.route_score)
            if score_delta < float(self.config.switch_advantage):
                return current, "hold_current_hysteresis"
        return best, "best_route_score"

    def evaluate_latest(self) -> dict[str, Any]:
        state = self._load_state()
        previous_strategy = state.get("selected_strategy")
        candidates = self._collect_candidates()
        selected, reason = self._choose_candidate(candidates, str(previous_strategy) if previous_strategy else None)

        payload = {
            "status": "ok",
            "mode": self.config.mode,
            "selected_strategy": selected.strategy_id if selected is not None else None,
            "selected_symbol": selected.symbol if selected is not None else None,
            "selected_direction": selected.direction if selected is not None else None,
            "selected_route_score": round(float(selected.route_score), 2) if selected is not None else 0.0,
            "decision_reason": reason,
            "previous_selected_strategy": previous_strategy,
            "candidates": [item.to_dict() for item in candidates],
            "updated_at": int(time.time()),
        }
        if selected is not None:
            payload["selected_candidate"] = selected.to_dict()
        if bool(self.config.persist_state):
            self._save_state(payload)
        return payload

    def run_loop(self, poll_interval_seconds: int = 30) -> None:
        while True:
            try:
                status = self.evaluate_latest()
                print(json.dumps(status, ensure_ascii=False))
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "status": "error",
                            "error": str(exc),
                            "updated_at": int(time.time()),
                        },
                        ensure_ascii=False,
                    )
                )
            time.sleep(max(1, int(poll_interval_seconds)))
