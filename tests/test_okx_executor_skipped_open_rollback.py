from __future__ import annotations

import unittest
from types import SimpleNamespace

from bot.okx_executor import ActionType, Direction, OkxExecutionEngine, StrategyAction


class StubStore:
    def __init__(self) -> None:
        self.actions: list[tuple[str, str, dict]] = []

    def append_action(self, timestamp: str, action_type: str, payload: dict) -> None:
        self.actions.append((timestamp, action_type, payload))


class SkippedOpenRollbackTest(unittest.TestCase):
    def build_executor(self) -> OkxExecutionEngine:
        executor = object.__new__(OkxExecutionEngine)
        executor.store = StubStore()
        executor.config = SimpleNamespace(overlay_skip_dynamic_high_leverage=False)
        executor._telegram_open_paused = lambda: False
        executor._shadow_gate_pre_execute = lambda action, engine: None
        executor._resolve_order_sizing = lambda action, engine: {"status": "ok", "amount": 1.0}
        executor._is_overlay_open_action = lambda action: False
        return executor

    def test_dynamic_skipped_open_rolls_back_local_position(self) -> None:
        executor = self.build_executor()
        action = StrategyAction(
            type=ActionType.OPEN_LONG,
            timestamp="2026-05-21 07:45",
            direction=Direction.BULL,
            entry_price=77848.5,
            stop_price=76668.41375,
            target_price=81544.45575,
        )
        engine = SimpleNamespace(
            position=SimpleNamespace(entry_time=action.timestamp, direction=Direction.BULL)
        )
        decision = {
            "status": "dynamic_high_leverage_skipped_open",
            "action": action.type.value,
            "direction": action.direction,
            "reason": "stop_distance_too_wide",
        }
        executor._dynamic_high_leverage_pre_open = lambda action, sizing, engine: (sizing, decision)

        result = executor.execute_action(action, engine)

        self.assertEqual(result["status"], "dynamic_high_leverage_skipped_open")
        self.assertIsNone(engine.position)
        action_types = [item[1] for item in executor.store.actions]
        self.assertIn("UNEXECUTED_OPEN_ROLLBACK", action_types)
        self.assertIn("EXECUTION_SKIPPED", action_types)
        rollback_payload = next(
            payload
            for _, action_type, payload in executor.store.actions
            if action_type == "UNEXECUTED_OPEN_ROLLBACK"
        )
        self.assertTrue(rollback_payload["rolled_back"])


if __name__ == "__main__":
    unittest.main()
