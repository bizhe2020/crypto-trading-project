from __future__ import annotations

import unittest
from types import SimpleNamespace

from bot.okx_executor import ActionType, Direction, OkxExecutionEngine, StrategyAction
from strategy.scalp_robust_v2_core import Candle, ScalpRobustEngine, StrategyConfig


class StubStore:
    def __init__(self) -> None:
        self.actions: list[tuple[str, str, dict]] = []
        self.values: dict[str, str] = {}

    def append_action(self, timestamp: str, action_type: str, payload: dict) -> None:
        self.actions.append((timestamp, action_type, payload))

    def get_value(self, key: str) -> str | None:
        return self.values.get(key)

    def set_value(self, key: str, value: str) -> None:
        self.values[key] = value


class SkippedOpenRollbackTest(unittest.TestCase):
    def build_executor(self) -> OkxExecutionEngine:
        executor = object.__new__(OkxExecutionEngine)
        executor.store = StubStore()
        executor.config = SimpleNamespace(
            overlay_skip_dynamic_high_leverage=False,
            enable_dynamic_high_leverage_structure=True,
        )
        executor._telegram_open_paused = lambda: False
        executor._shadow_gate_pre_execute = lambda action, engine: None
        executor._resolve_order_sizing = lambda action, engine: {"status": "ok", "amount": 1.0}
        executor._is_overlay_open_action = lambda action: False
        executor._safe_float = OkxExecutionEngine._safe_float.__get__(executor, OkxExecutionEngine)
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

    def test_dynamic_skipped_open_records_nonblocking_paper_position(self) -> None:
        executor = self.build_executor()
        action = StrategyAction(
            type=ActionType.OPEN_LONG,
            timestamp="2026-05-21 07:45",
            direction=Direction.BULL,
            entry_price=77848.5,
            stop_price=76668.41375,
            target_price=81544.45575,
            metadata={
                "index": 10,
                "capital_at_entry": 1000.0,
                "notional": 4000.0,
                "quantity": 0.051,
                "exit_profile": "fvg_bear6_loose_runner",
                "exit_profile_reason": "score_bucket:fvg_near_bear6_target20",
                "exit_profile_overrides": {"atr_activation_rr": 2.6},
            },
        )
        engine = SimpleNamespace(
            capital=1000.0,
            position=SimpleNamespace(
                entry_time=action.timestamp,
                direction=Direction.BULL,
                capital_at_entry=1000.0,
                notional=4000.0,
                quantity=0.051,
            ),
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
        state = __import__("json").loads(executor.store.values["dynamic_high_leverage_structure_state"])
        self.assertEqual(state["paper_position"]["entry_time"], "2026-05-21 07:45")
        self.assertEqual(state["paper_position"]["notional"], 4000.0)
        self.assertEqual(state["paper_position"]["exit_profile"], "fvg_bear6_loose_runner")
        self.assertEqual(state["paper_position"]["exit_profile_overrides"], {"atr_activation_rr": 2.6})
        action_types = [item[1] for item in executor.store.actions]
        self.assertIn("DYNAMIC_PAPER_OPEN", action_types)

    def test_shadow_skipped_close_updates_dynamic_health_from_paper_position(self) -> None:
        executor = self.build_executor()
        executor.store.set_value(
            "dynamic_high_leverage_structure_state",
            __import__("json").dumps(
                {
                    "mode": "offense",
                    "capital": 1000.0,
                    "drawdown_peak": 1000.0,
                    "unit_returns": [],
                    "loss_streak": 0,
                    "win_streak": 0,
                    "paper_position": {
                        "entry_time": "2026-05-21 07:45",
                        "direction": Direction.BULL,
                        "notional": 4000.0,
                        "capital_at_entry": 1000.0,
                    },
                    "paper_entry_time": "2026-05-21 07:45",
                }
            ),
        )
        action = StrategyAction(
            type=ActionType.CLOSE_POSITION,
            timestamp="2026-05-21 12:00",
            direction=Direction.BULL,
            exit_price=76000.0,
            reason="stop_loss",
            metadata={"net_pnl": -80.0, "entry_time": "2026-05-21 07:45"},
        )
        engine = SimpleNamespace(capital=1000.0, trades=[])
        executor._shadow_gate_pre_execute = lambda action, engine: {
            "status": "shadow_gate_skipped_close",
            "action": action.type.value,
            "direction": action.direction,
            "reason": "paper_position_not_mirrored",
        }

        result = executor.execute_action(action, engine)

        self.assertEqual(result["status"], "shadow_gate_skipped_close")
        state = __import__("json").loads(executor.store.values["dynamic_high_leverage_structure_state"])
        self.assertEqual(state["unit_returns"], [-0.02])
        self.assertEqual(state["loss_streak"], 0)
        self.assertEqual(state["capital"], 1000.0)
        self.assertIsNone(state["paper_position"])
        action_types = [item[1] for item in executor.store.actions]
        self.assertIn("DYNAMIC_PAPER_CLOSE", action_types)

    def test_paper_dynamic_position_is_evaluated_without_restoring_real_position(self) -> None:
        executor = self.build_executor()
        candles = [
            Candle(ts=1_700_000_000, o=100.0, h=101.0, l=99.0, c=100.0, v=1.0),
            Candle(ts=1_700_000_900, o=100.0, h=101.0, l=94.0, c=95.0, v=1.0),
        ]
        engine = ScalpRobustEngine.from_candles(
            candles,
            candles,
            StrategyConfig(initial_capital=1000.0, taker_fee_rate=0.0, slippage_bps=0.0),
        )
        engine.position = None
        executor.store.set_value(
            "dynamic_high_leverage_structure_state",
            __import__("json").dumps(
                {
                    "mode": "offense",
                    "capital": 1000.0,
                    "drawdown_peak": 1000.0,
                    "unit_returns": [],
                    "loss_streak": 0,
                    "win_streak": 0,
                    "paper_position": {
                        "entry_time": "2023-11-14 22:13",
                        "direction": Direction.BULL,
                        "entry_price": 100.0,
                        "signal_entry_price": 100.0,
                        "stop_price": 95.0,
                        "target_price": 120.0,
                        "notional": 1000.0,
                        "quantity": 10.0,
                        "capital_at_entry": 1000.0,
                        "entry_idx": 0,
                        "entry_fee": 0.0,
                        "entry_slippage_cost": 0.0,
                        "target_rr": 4.0,
                        "trail_style": "normal",
                    },
                    "paper_entry_time": "2023-11-14 22:13",
                }
            ),
        )

        payload = executor._dynamic_evaluate_paper_position(engine, 1)

        self.assertIsNotNone(payload)
        self.assertEqual(payload["source"], "paper_skipped_close")
        self.assertIsNone(engine.position)
        self.assertEqual(engine.trades, [])
        state = __import__("json").loads(executor.store.values["dynamic_high_leverage_structure_state"])
        self.assertEqual(state["unit_returns"], [-0.05])
        self.assertEqual(state["loss_streak"], 0)
        self.assertEqual(state["capital"], 1000.0)
        self.assertIsNone(state["paper_position"])

    def test_paper_dynamic_position_can_close_while_real_position_is_open(self) -> None:
        executor = self.build_executor()
        real_position = SimpleNamespace(entry_time="real-live-position")
        candles = [
            Candle(ts=1_700_000_000, o=100.0, h=101.0, l=99.0, c=100.0, v=1.0),
            Candle(ts=1_700_000_900, o=100.0, h=101.0, l=94.0, c=95.0, v=1.0),
        ]
        engine = ScalpRobustEngine.from_candles(
            candles,
            candles,
            StrategyConfig(initial_capital=1000.0, taker_fee_rate=0.0, slippage_bps=0.0),
        )
        engine.position = real_position
        executor.store.set_value(
            "dynamic_high_leverage_structure_state",
            __import__("json").dumps(
                {
                    "mode": "offense",
                    "capital": 1000.0,
                    "drawdown_peak": 1000.0,
                    "unit_returns": [],
                    "loss_streak": 0,
                    "win_streak": 0,
                    "paper_position": {
                        "entry_time": "2023-11-14 22:13",
                        "direction": Direction.BULL,
                        "entry_price": 100.0,
                        "signal_entry_price": 100.0,
                        "stop_price": 95.0,
                        "target_price": 120.0,
                        "notional": 1000.0,
                        "quantity": 10.0,
                        "capital_at_entry": 1000.0,
                        "entry_idx": 0,
                        "entry_fee": 0.0,
                        "entry_slippage_cost": 0.0,
                        "target_rr": 4.0,
                        "trail_style": "normal",
                    },
                    "paper_entry_time": "2023-11-14 22:13",
                }
            ),
        )

        payload = executor._dynamic_evaluate_paper_position(engine, 1)

        self.assertIsNotNone(payload)
        self.assertEqual(payload["source"], "paper_skipped_close")
        self.assertIs(engine.position, real_position)
        self.assertEqual(engine.trades, [])
        state = __import__("json").loads(executor.store.values["dynamic_high_leverage_structure_state"])
        self.assertEqual(state["unit_returns"], [-0.05])
        self.assertEqual(state["loss_streak"], 0)
        self.assertEqual(state["capital"], 1000.0)
        self.assertIsNone(state["paper_position"])

    def test_paper_dynamic_evaluation_continues_after_stop_update(self) -> None:
        executor = self.build_executor()
        executor.store.set_value(
            "dynamic_high_leverage_structure_state",
            __import__("json").dumps(
                {
                    "mode": "offense",
                    "capital": 1000.0,
                    "drawdown_peak": 1000.0,
                    "unit_returns": [],
                    "loss_streak": 0,
                    "win_streak": 0,
                    "paper_position": {
                        "entry_time": "2026-05-21 07:45",
                        "direction": Direction.BULL,
                        "entry_price": 100.0,
                        "signal_entry_price": 100.0,
                        "stop_price": 95.0,
                        "target_price": 120.0,
                        "notional": 1000.0,
                        "quantity": 10.0,
                        "capital_at_entry": 1000.0,
                        "entry_idx": 0,
                        "target_rr": 4.0,
                        "trail_style": "normal",
                    },
                    "paper_entry_time": "2026-05-21 07:45",
                }
            ),
        )

        class UpdatingEngine:
            def __init__(self) -> None:
                self.position = None
                self.capital = 1000.0
                self.trades = []
                self.exit_reasons = {}

            def manage_position(self, idx: int) -> list[StrategyAction]:
                if idx == 1:
                    self.position.sl_price = 99.0
                    return [
                        StrategyAction(
                            type=ActionType.UPDATE_STOP,
                            timestamp="2026-05-21 08:00",
                            direction=Direction.BULL,
                            stop_price=99.0,
                        )
                    ]
                self.trades.append(SimpleNamespace(notional=1000.0))
                return [
                    StrategyAction(
                        type=ActionType.CLOSE_POSITION,
                        timestamp="2026-05-21 08:15",
                        direction=Direction.BULL,
                        exit_price=99.0,
                        reason="stop_loss",
                        metadata={"net_pnl": -10.0, "entry_time": "2026-05-21 07:45"},
                    )
                ]

        engine = UpdatingEngine()

        payload = executor._dynamic_evaluate_paper_position(engine, 2)

        self.assertIsNotNone(payload)
        self.assertEqual(payload["source"], "paper_skipped_close")
        state = __import__("json").loads(executor.store.values["dynamic_high_leverage_structure_state"])
        self.assertEqual(state["unit_returns"], [-0.01])
        self.assertEqual(state["loss_streak"], 0)
        self.assertEqual(state["capital"], 1000.0)
        self.assertIsNone(state["paper_position"])
        self.assertIsNone(engine.position)
        self.assertEqual(engine.trades, [])


if __name__ == "__main__":
    unittest.main()
