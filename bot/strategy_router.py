from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_CURRENT_STRATEGY_UNSET = object()


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
    googl_strategy_config: str = "config/config.paper.googl-high-leverage-runtime.json"
    enable_btc: bool = True
    enable_qqq: bool = True
    enable_googl: bool = False
    btc_min_route_score: float = 30.0
    qqq_min_route_score: float = 45.0
    googl_min_route_score: float = 0.0
    switch_advantage: float = 8.0
    btc_takeover_advantage: float | None = None
    qqq_takeover_advantage: float | None = None
    googl_takeover_advantage: float | None = None
    googl_execution_enabled: bool = False
    strategy_priority: list[str] | None = None
    persist_state: bool = True
    execution_enabled: bool = False
    execution_credentials_config: str | None = None
    flatten_before_switch: bool = True
    flatten_on_no_signal: bool = True
    qqq_state_db_path: str = "state/runtime_qqq_usdt_router.db"
    qqq_margin_mode: str = "isolated"
    qqq_position_size_pct: float = 1.0
    qqq_sizing_basis: str = "available"
    qqq_sizing_cash_buffer_usdt: float = 0.0
    qqq_max_notional_usdt: float | None = None
    qqq_min_order_notional_usdt: float = 10.0
    qqq_min_rebalance_notional_usdt: float = 10.0
    qqq_min_rebalance_gap_ratio: float = 0.0
    qqq_rebalance_cooldown_seconds: float = 0.0
    qqq_rebalance_on_leverage_change: bool = True
    qqq_rebalance_on_notional_gap: bool = False
    qqq_max_close_order_contracts: float | None = None
    qqq_max_close_order_notional_usdt: float | None = None
    qqq_max_market_order_contracts: float | None = None
    qqq_market_order_chunk_delay_seconds: float = 0.0
    qqq_close_confirm_timeout_seconds: float = 15.0
    qqq_close_confirm_poll_seconds: float = 1.0
    qqq_close_chunk_delay_seconds: float = 0.2
    qqq_enable_exchange_stop: bool = False
    qqq_rebalance_risk_on_market_hours_only: bool = False
    qqq_market_hours_timezone: str = "America/New_York"
    qqq_market_hours_start: str = "09:30"
    qqq_market_hours_end: str = "16:00"
    qqq_market_calendar: str = "NYSE"
    qqq_profit_roll_enabled: bool = False
    qqq_profit_roll_min_actual_leverage: float = 9.5
    qqq_profit_roll_trigger: str = "any"
    qqq_profit_roll_max_rolls_per_trade: int = 4
    qqq_profit_roll_cooldown_bars: int = 1
    qqq_profit_roll_skip_defense: bool = True
    qqq_profit_roll_min_notional_usdt: float | None = None
    okx_markets_cache_path: str = "var/okx/markets_cache.json"
    telegram_enabled: bool = False
    telegram_token: str | None = None
    telegram_chat_id: str | None = None
    telegram_proxy: str | None = None
    telegram_notify_startup: bool = True
    telegram_notify_route_change: bool = True
    telegram_notify_execution: bool = True
    telegram_notify_data_warnings: bool = True
    telegram_notify_gap_warnings: bool = True
    telegram_notify_errors: bool = True
    router_audit_log_enabled: bool = True
    router_audit_log_path: str | None = None
    router_evaluation_timeout_seconds: float = 240.0
    router_heartbeat_path: str | None = None
    switch_rollback_cooldown_seconds: float = 3600.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrategyRouterConfig":
        filtered = {key: value for key, value in payload.items() if key in cls.__dataclass_fields__}
        if "qqq_rebalance_on_notional_gap" not in payload:
            basis = str(filtered.get("qqq_sizing_basis", cls.__dataclass_fields__["qqq_sizing_basis"].default) or "available").strip().lower()
            filtered["qqq_rebalance_on_notional_gap"] = basis in {"total_equity", "equity", "total"}
        return cls(**filtered)


class StrategyRouter:
    def __init__(self, config: StrategyRouterConfig, config_path: str | Path | None = None):
        self.config = config
        self.config_path = Path(config_path).resolve() if config_path else None
        self.state_path = Path(config.state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.btc_adapter = None
        self.qqq_adapter = None
        self.googl_adapter = None
        self.candidate_preprocessor: Callable[[list[RoutedSignalCandidate]], list[RoutedSignalCandidate]] | None = None

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
            "googl_strategy_config",
            "execution_credentials_config",
            "qqq_state_db_path",
            "okx_markets_cache_path",
            "router_audit_log_path",
            "router_heartbeat_path",
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
        if strategy_id == "googl_usdt_aggressive":
            return float(self.config.googl_min_route_score)
        return 0.0

    def _takeover_advantage_for(self, challenger_strategy_id: str) -> float:
        if challenger_strategy_id == "btc_sota" and self.config.btc_takeover_advantage is not None:
            return float(self.config.btc_takeover_advantage)
        if challenger_strategy_id == "qqq_usdt_aggressive" and self.config.qqq_takeover_advantage is not None:
            return float(self.config.qqq_takeover_advantage)
        if challenger_strategy_id == "googl_usdt_aggressive" and self.config.googl_takeover_advantage is not None:
            return float(self.config.googl_takeover_advantage)
        return float(self.config.switch_advantage)

    def _is_live_eligible(self, strategy_id: str) -> bool:
        """策略是否可被选为实盘路由目标。影子模式（googl_execution_enabled=False）
        下 GOOGL 仍参与评估并被记录，但永不进入实盘选择。"""
        if strategy_id == "googl_usdt_aggressive":
            return bool(self.config.googl_execution_enabled)
        return True

    def _collect_candidates(self) -> list[RoutedSignalCandidate]:
        from bot.btc_signal_adapter import BtcSignalAdapter
        from bot.googl_usdt_signal_adapter import GooglUsdtSignalAdapter
        from bot.qqq_usdt_signal_adapter import QqqUsdtSignalAdapter

        if self.btc_adapter is None:
            self.btc_adapter = BtcSignalAdapter(Path(self.config.btc_strategy_config))
        if self.qqq_adapter is None:
            self.qqq_adapter = QqqUsdtSignalAdapter(Path(self.config.qqq_strategy_config))
        if self.googl_adapter is None:
            self.googl_adapter = GooglUsdtSignalAdapter(Path(self.config.googl_strategy_config))
        candidates: list[RoutedSignalCandidate] = []
        if self.config.enable_btc:
            candidates.append(self.btc_adapter.preview())
        if self.config.enable_qqq:
            candidates.append(self.qqq_adapter.preview())
        if self.config.enable_googl:
            candidates.append(self.googl_adapter.preview())
        return candidates

    def _choose_candidate(
        self,
        candidates: list[RoutedSignalCandidate],
        current_strategy: str | None,
    ) -> tuple[RoutedSignalCandidate | None, str]:
        active = [
            candidate
            for candidate in candidates
            if candidate.active and self._is_live_eligible(candidate.strategy_id)
        ]
        current = next((item for item in active if item.strategy_id == current_strategy), None) if current_strategy else None
        eligible_challengers = [
            candidate
            for candidate in active
            if candidate.strategy_id != current_strategy
            and float(candidate.route_score) >= self._min_score_for(candidate.strategy_id)
        ]

        if current is not None:
            if not eligible_challengers:
                return current, "hold_current_no_challenger"
            eligible_challengers.sort(
                key=lambda item: (
                    float(item.route_score),
                    -self._strategy_priority_value(item.strategy_id),
                ),
                reverse=True,
            )
            best_challenger = eligible_challengers[0]
            score_delta = float(best_challenger.route_score) - float(current.route_score)
            if score_delta < self._takeover_advantage_for(best_challenger.strategy_id):
                return current, "hold_current_hysteresis"
            return best_challenger, "best_route_score"

        eligible = [
            candidate
            for candidate in active
            if float(candidate.route_score) >= self._min_score_for(candidate.strategy_id)
        ]
        if not eligible:
            return None, "no_eligible_candidates"
        eligible.sort(
            key=lambda item: (
                float(item.route_score),
                -self._strategy_priority_value(item.strategy_id),
            ),
            reverse=True,
        )
        return eligible[0], "best_route_score"

    def evaluate_latest(
        self,
        current_strategy: str | None | object = _CURRENT_STRATEGY_UNSET,
        *,
        current_strategy_override: str | None | object = _CURRENT_STRATEGY_UNSET,
    ) -> dict[str, Any]:
        state = self._load_state()
        previous_strategy = state.get("selected_strategy")
        if (
            current_strategy is not _CURRENT_STRATEGY_UNSET
            and current_strategy_override is not _CURRENT_STRATEGY_UNSET
            and current_strategy != current_strategy_override
        ):
            raise TypeError("current_strategy and current_strategy_override disagree")
        effective_override = (
            current_strategy_override
            if current_strategy_override is not _CURRENT_STRATEGY_UNSET
            else current_strategy
        )
        if effective_override is _CURRENT_STRATEGY_UNSET:
            effective_current_strategy = str(previous_strategy) if previous_strategy else None
        else:
            effective_current_strategy = str(effective_override) if effective_override else None
        candidates = self._collect_candidates()
        if self.candidate_preprocessor is not None:
            candidates = list(self.candidate_preprocessor(candidates))
        selected, reason = self._choose_candidate(candidates, effective_current_strategy)

        payload = {
            "status": "ok",
            "mode": self.config.mode,
            "selected_strategy": selected.strategy_id if selected is not None else None,
            "selected_symbol": selected.symbol if selected is not None else None,
            "selected_direction": selected.direction if selected is not None else None,
            "selected_route_score": round(float(selected.route_score), 2) if selected is not None else 0.0,
            "decision_reason": reason,
            "previous_selected_strategy": previous_strategy,
            "route_current_strategy": effective_current_strategy,
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
