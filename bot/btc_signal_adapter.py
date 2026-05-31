from __future__ import annotations

from pathlib import Path
from typing import Any

from bot.btc_route_scoring import btc_effective_leverage, btc_route_score, btc_strength_label
from bot.okx_executor import OkxExecutionEngine
from bot.strategy_router import RoutedSignalCandidate


class BtcSignalAdapter:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()

    def preview(self) -> RoutedSignalCandidate:
        executor = OkxExecutionEngine.from_file(self.config_path)
        snapshot = executor.store.load_snapshot() or {}
        last_processed = executor.store.get_value("last_processed_candle_time")
        position = snapshot.get("position") if isinstance(snapshot.get("position"), dict) else None
        if position:
            return self._candidate_from_snapshot_position(executor, position, last_processed)

        try:
            engine, start_idx = executor.load_engine()
            latest_closed_idx = executor._latest_closed_index(engine)
            if latest_closed_idx is None:
                return RoutedSignalCandidate(
                    strategy_id="btc_sota",
                    symbol=executor.config.symbol,
                    active=False,
                    route_score=0.0,
                    source_config=str(self.config_path),
                    metadata={"reason": "waiting_for_closed_candle"},
                )
            latest_timestamp = engine._timestamp_for_idx(latest_closed_idx)
            span_start = max(start_idx, latest_closed_idx - 96)
            actions = engine.evaluate_range(span_start, latest_closed_idx + 1)
            selected = self._preview_selected_candidate(executor, engine, actions, latest_closed_idx, latest_timestamp)
            if selected is not None:
                return selected
            position_obj = getattr(engine, "position", None)
            if position_obj is not None:
                return self._candidate_from_position(executor, position_obj)
            return RoutedSignalCandidate(
                strategy_id="btc_sota",
                symbol=executor.config.symbol,
                active=False,
                route_score=0.0,
                timestamp=latest_timestamp,
                source_config=str(self.config_path),
                metadata={"reason": "no_live_candidate"},
            )
        except Exception as exc:
            return RoutedSignalCandidate(
                strategy_id="btc_sota",
                symbol=executor.config.symbol,
                active=False,
                route_score=0.0,
                timestamp=last_processed,
                source_config=str(self.config_path),
                metadata={"reason": "preview_error", "error": str(exc)},
            )

    def _preview_selected_candidate(
        self,
        executor: OkxExecutionEngine,
        engine: Any,
        actions: list[Any],
        latest_closed_idx: int,
        latest_timestamp: str,
    ) -> RoutedSignalCandidate | None:
        raw_open_actions = [item for item in actions if item.type.value in {"OPEN_LONG", "OPEN_SHORT"}]
        if not raw_open_actions:
            return None

        if executor._live_candidate_arbitration_enabled():
            open_actions, _score_gate_rejected = executor._apply_sota_score_gate_to_open_actions(
                engine,
                raw_open_actions,
                latest_closed_idx,
            )
            open_actions, _structure_gate_rejected, _structure_gate_recalled = (
                executor._apply_sota_structure_gate_to_open_actions(open_actions)
            )
            candidates: list[dict[str, Any]] = []
            for action in open_actions:
                event_type = executor._open_action_event_type(action)
                metadata = dict(action.metadata or {})
                metadata.setdefault("candidate_event_type", event_type)
                action.metadata = metadata
                candidates.append(
                    {
                        "event_type": event_type,
                        "source_key": f"{event_type}|{action.timestamp}|{action.direction}|{action.entry_price}",
                        "entry_idx": int(metadata.get("index", latest_closed_idx) or latest_closed_idx),
                        "action": action,
                    }
                )
            for builder_name in ["_smc_short_candidate", "_gap_smc_short_candidate", "_smc_long_candidate"]:
                builder = getattr(executor, builder_name, None)
                if builder is None:
                    continue
                candidate = builder(engine, latest_closed_idx)
                if candidate is not None:
                    candidates.append(candidate)
            if not candidates:
                return None
            candidates.sort(
                key=lambda item: (
                    int(item.get("entry_idx", latest_closed_idx) or latest_closed_idx),
                    -executor._live_candidate_priority_value(str(item.get("event_type") or "")),
                ),
            )
            selected = candidates[-1]
            return self._candidate_from_selected(executor, selected, latest_timestamp)

        fresh_actions = [item for item in raw_open_actions if str(item.timestamp) == str(latest_timestamp)]
        if not fresh_actions:
            return None
        return self._candidate_from_action(executor, fresh_actions[-1], latest_timestamp)

    def _candidate_from_selected(
        self,
        executor: OkxExecutionEngine,
        selected: dict[str, Any],
        latest_timestamp: str,
    ) -> RoutedSignalCandidate | None:
        action = selected.get("action")
        if action is not None:
            if str(action.timestamp) != str(latest_timestamp):
                return None
            return self._candidate_from_action(executor, action, latest_timestamp)

        entry_idx = int(selected.get("entry_idx", -1) or -1)
        direction = str(selected.get("direction") or "BULL")
        event_type = str(selected.get("event_type") or "btc_overlay")
        metadata = dict(selected)
        metadata.setdefault("event_type", event_type)
        metadata.setdefault("direction", direction)
        leverage = btc_effective_leverage(metadata)
        route_score = btc_route_score(metadata)
        return RoutedSignalCandidate(
            strategy_id="btc_sota",
            symbol=executor.config.symbol,
            active=entry_idx >= 0,
            route_score=route_score,
            timestamp=latest_timestamp,
            direction=direction,
            event_type=event_type,
            leverage=leverage if leverage > 0 else None,
            strength_label=btc_strength_label(metadata),
            source_config=str(self.config_path),
            metadata={
                "source": selected.get("source"),
                "entry_idx": entry_idx,
                "source_key": selected.get("source_key"),
                "route_score_source": "btc_route_scoring",
            },
        )

    def _candidate_from_action(
        self,
        executor: OkxExecutionEngine,
        action: Any,
        latest_timestamp: str,
    ) -> RoutedSignalCandidate:
        metadata = dict(action.metadata or {})
        event_type = str(metadata.get("candidate_event_type") or executor._open_action_event_type(action))
        score_payload = {**metadata, "event_type": event_type, "direction": action.direction}
        route_score = btc_route_score(score_payload)
        return RoutedSignalCandidate(
            strategy_id="btc_sota",
            symbol=executor.config.symbol,
            active=str(action.timestamp) == str(latest_timestamp),
            route_score=route_score,
            timestamp=action.timestamp,
            direction=action.direction,
            event_type=event_type,
            leverage=btc_effective_leverage(score_payload) or None,
            strength_label=btc_strength_label(score_payload),
            source_config=str(self.config_path),
            metadata={
                "entry_idx": metadata.get("index"),
                "event_type": event_type,
                "entry_regime_score": metadata.get("entry_regime_score"),
                "risk_regime": metadata.get("risk_regime"),
                "regime_label": metadata.get("regime_label"),
                "trail_style": metadata.get("trail_style"),
                "feature_recent_fvg_near_entry": metadata.get("feature_recent_fvg_near_entry"),
                "feature_recent_sweep_status": metadata.get("feature_recent_sweep_status"),
                "sota_score_gate": metadata.get("sota_score_gate"),
                "route_score_source": "btc_route_scoring",
            },
        )

    def _candidate_from_position(self, executor: OkxExecutionEngine, position: Any) -> RoutedSignalCandidate:
        metadata = self._payload_from_position(position)
        event_type = str(metadata.get("event_type") or "sota_long")
        route_score = btc_route_score(metadata)
        return RoutedSignalCandidate(
            strategy_id="btc_sota",
            symbol=executor.config.symbol,
            active=True,
            route_score=route_score,
            timestamp=str(getattr(position, "entry_time", None) or ""),
            direction=str(getattr(position, "direction", None) or ""),
            event_type=event_type,
            leverage=btc_effective_leverage(metadata) or None,
            strength_label=btc_strength_label(metadata),
            source_config=str(self.config_path),
            metadata={
                "entry_idx": getattr(position, "entry_idx", None),
                "entry_regime_score": getattr(position, "entry_regime_score", None),
                "risk_regime": getattr(position, "risk_regime", None),
                "regime_label": getattr(position, "regime_label", None),
                "trail_style": getattr(position, "trail_style", None),
                "route_score_source": "btc_route_scoring",
            },
        )

    def _candidate_from_snapshot_position(
        self,
        executor: OkxExecutionEngine,
        position: dict[str, Any],
        last_processed: str | None,
    ) -> RoutedSignalCandidate:
        metadata = self._payload_from_snapshot_position(position)
        event_type = str(metadata.get("event_type") or "sota_long")
        route_score = btc_route_score(metadata)
        return RoutedSignalCandidate(
            strategy_id="btc_sota",
            symbol=executor.config.symbol,
            active=True,
            route_score=route_score,
            timestamp=str(position.get("entry_time") or last_processed or ""),
            direction=str(position.get("direction") or ""),
            event_type=event_type,
            leverage=btc_effective_leverage(metadata) or None,
            strength_label=btc_strength_label(metadata),
            source_config=str(self.config_path),
            metadata={
                "entry_idx": position.get("entry_idx"),
                "entry_regime_score": position.get("entry_regime_score"),
                "risk_regime": position.get("risk_regime"),
                "regime_label": position.get("regime_label"),
                "trail_style": position.get("trail_style"),
                "from_snapshot": True,
                "route_score_source": "btc_route_scoring",
            },
        )

    @staticmethod
    def _payload_from_position(position: Any) -> dict[str, Any]:
        keys = [
            "entry_idx",
            "entry_regime_score",
            "risk_regime",
            "regime_label",
            "trail_style",
            "execution_effective_leverage",
            "requested_effective_leverage",
            "source_effective_leverage",
            "net_score",
            "bull_total",
            "bear_total",
            "feature_recent_fvg_near_entry",
            "feature_recent_sweep_status",
            "feature_bearish_structure",
            "feature_bullish_structure",
        ]
        payload = {key: getattr(position, key, None) for key in keys}
        payload["event_type"] = getattr(position, "candidate_event_type", None) or "sota_long"
        payload["direction"] = getattr(position, "direction", None)
        return payload

    @staticmethod
    def _payload_from_snapshot_position(position: dict[str, Any]) -> dict[str, Any]:
        payload = dict(position)
        payload["event_type"] = position.get("candidate_event_type") or position.get("event_type") or "sota_long"
        return payload
