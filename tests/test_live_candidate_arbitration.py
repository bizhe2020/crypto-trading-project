from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bot.okx_executor import ExecutorConfig, OkxExecutionEngine
from bot.state_store import StateStore
from strategy.scalp_robust_v2_core import ActionType, Candle, Direction, ScalpRobustEngine, StrategyAction, StrategyConfig


class DummyExchange:
    def amount_to_precision(self, _symbol: str, amount: float) -> str:
        return str(round(amount, 6))


class DummyClient:
    def __init__(self) -> None:
        self.exchange = DummyExchange()

    def load_markets(self) -> dict[str, dict[str, float | bool]]:
        return {"BTC/USDT:USDT": {"contract": False, "contractSize": 1.0}}


def make_executor(config: ExecutorConfig | None = None) -> OkxExecutionEngine:
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / "state.db"
    executor = object.__new__(OkxExecutionEngine)
    executor.config = config or ExecutorConfig(
        mode="paper",
        symbol="BTC/USDT:USDT",
        timeframe="15m",
        informative_timeframe="4h",
        leverage=10,
        margin_mode="cross",
        max_open_positions=1,
        risk_per_trade=0.01,
        state_db_path=str(path),
    )
    executor.config_path = None
    executor.client = DummyClient()
    executor.store = StateStore(path)
    executor.market_data = None
    executor._markets_cache = None
    executor._test_tmp = tmp
    return executor


def make_engine() -> ScalpRobustEngine:
    candles = [
        Candle(ts=1_700_000_000, o=1000.0, h=1005.0, l=995.0, c=1000.0, v=100.0),
        Candle(ts=1_700_000_900, o=1000.0, h=1010.0, l=990.0, c=1000.0, v=100.0),
    ]
    engine = ScalpRobustEngine.from_candles(candles, candles, StrategyConfig(initial_capital=1000.0))
    engine.capital = 1000.0
    return engine


class LiveCandidateArbitrationTest(unittest.TestCase):
    def test_overlay_open_skips_dynamic_high_leverage_in_paper(self) -> None:
        config = ExecutorConfig(
            mode="paper",
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            informative_timeframe="4h",
            leverage=10,
            margin_mode="cross",
            max_open_positions=1,
            risk_per_trade=0.01,
            state_db_path=":memory:",
            enable_dynamic_high_leverage_structure=True,
            overlay_skip_dynamic_high_leverage=True,
        )
        executor = make_executor(config)
        engine = make_engine()
        action = engine.open_position(
            0,
            Direction.BEAR,
            1000.0,
            1020.0,
            950.0,
            target_rr_override=2.5,
            max_hold_bars_override=40,
            trail_style_override="tight",
            candidate_event_type="smc_short",
            requested_notional_override=5000.0,
        )
        metadata = dict(action.metadata or {})
        metadata["candidate_leverage"] = 5.0
        action.metadata = metadata

        result = executor.execute_action(action, engine)

        self.assertEqual(result["status"], "paper_recorded")
        self.assertTrue(result["overlay_skipped_dynamic_high_leverage"])
        self.assertEqual(result.get("dynamic_high_leverage"), None)
        self.assertAlmostEqual(float(result["notional_usdt"]), 5000.0)
        self.assertEqual(engine.position.execution_risk_mode, "overlay_fixed")
        self.assertEqual(engine.position.execution_leverage_reasons, ["overlay_fixed:smc_short"])

    def test_smc_long_overlay_open_skips_dynamic_high_leverage_in_paper(self) -> None:
        config = ExecutorConfig(
            mode="paper",
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            informative_timeframe="4h",
            leverage=10,
            margin_mode="cross",
            max_open_positions=1,
            risk_per_trade=0.01,
            state_db_path=":memory:",
            enable_dynamic_high_leverage_structure=True,
            overlay_skip_dynamic_high_leverage=True,
        )
        executor = make_executor(config)
        engine = make_engine()
        action = engine.open_position(
            0,
            Direction.BULL,
            1000.0,
            980.0,
            1030.0,
            target_rr_override=1.5,
            max_hold_bars_override=40,
            trail_style_override="tight",
            candidate_event_type="smc_long",
            requested_notional_override=5000.0,
        )
        metadata = dict(action.metadata or {})
        metadata["candidate_leverage"] = 5.0
        action.metadata = metadata

        result = executor.execute_action(action, engine)

        self.assertEqual(result["status"], "paper_recorded")
        self.assertTrue(result["overlay_skipped_dynamic_high_leverage"])
        self.assertEqual(result.get("dynamic_high_leverage"), None)
        self.assertAlmostEqual(float(result["notional_usdt"]), 5000.0)
        self.assertEqual(engine.position.execution_risk_mode, "overlay_fixed")
        self.assertEqual(engine.position.execution_leverage_reasons, ["overlay_fixed:smc_long"])

    def test_high_leverage_guard_uses_candidate_leverage_for_overlay(self) -> None:
        config = ExecutorConfig(
            mode="paper",
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            informative_timeframe="4h",
            leverage=10,
            margin_mode="cross",
            max_open_positions=1,
            risk_per_trade=0.01,
            state_db_path=":memory:",
            enable_high_leverage_guard=True,
            high_leverage_guard_min_leverage=8.0,
            high_leverage_min_liquidation_buffer_pct=1.2,
        )
        executor = make_executor(config)
        action = StrategyAction(
            type=ActionType.OPEN_SHORT,
            timestamp="2023-11-14 22:13",
            direction=Direction.BEAR,
            entry_price=1000.0,
            stop_price=1015.0,
            target_price=958.75,
            metadata={
                "candidate_event_type": "smc_short",
                "candidate_leverage": 5.0,
                "candidate_maintenance_margin_pct": 0.5,
                "capital_at_entry": 1000.0,
            },
        )
        sizing = {"status": "ok", "expected_notional_usdt": 5000.0, "available_usdt": 1000.0}

        diagnostics = executor._high_leverage_open_diagnostics(action, sizing)
        failures = executor._high_leverage_guard_failures(diagnostics)

        self.assertAlmostEqual(diagnostics["configured_leverage"], 5.0)
        self.assertGreater(diagnostics["liquidation_buffer_pct"], 1.2)
        self.assertNotIn("liquidation_buffer_too_small", failures)

    def test_sota_open_still_uses_dynamic_high_leverage(self) -> None:
        config = ExecutorConfig(
            mode="paper",
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            informative_timeframe="4h",
            leverage=10,
            margin_mode="cross",
            max_open_positions=1,
            risk_per_trade=0.01,
            state_db_path=":memory:",
            enable_dynamic_high_leverage_structure=True,
        )
        executor = make_executor(config)
        engine = make_engine()
        action = StrategyAction(
            type=ActionType.OPEN_LONG,
            timestamp="2023-11-14 22:13",
            direction=Direction.BULL,
            entry_price=1000.0,
            stop_price=985.0,
            target_price=1040.0,
            metadata={"capital_at_entry": 1000.0, "notional": 1000.0, "risk_based_notional": 1000.0},
        )

        result = executor.execute_action(action, engine)

        self.assertEqual(result["status"], "paper_recorded")
        self.assertFalse(result["overlay_skipped_dynamic_high_leverage"])
        self.assertIsInstance(result.get("dynamic_high_leverage"), dict)

    def test_long_score_bucket_sizing_boosts_matching_sota_open(self) -> None:
        config = ExecutorConfig(
            mode="paper",
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            informative_timeframe="4h",
            leverage=10,
            margin_mode="cross",
            max_open_positions=1,
            risk_per_trade=0.01,
            state_db_path=":memory:",
            enable_dynamic_high_leverage_structure=True,
            dynamic_base_leverage=4.0,
            dynamic_max_effective_leverage=8.0,
            enable_long_score_bucket_sizing_live=True,
            long_score_bucket_sizing_rules=[
                {
                    "name": "bear_total_6_light_boost",
                    "bear_eq": 6,
                    "leverage_multiplier": 1.35,
                    "max_effective_leverage": 8.0,
                }
            ],
        )
        executor = make_executor(config)
        engine = make_engine()
        action = StrategyAction(
            type=ActionType.OPEN_LONG,
            timestamp="2023-11-14 22:13",
            direction=Direction.BULL,
            entry_price=1000.0,
            stop_price=985.0,
            target_price=1040.0,
            metadata={
                "capital_at_entry": 1000.0,
                "notional": 1000.0,
                "risk_based_notional": 1000.0,
                "candidate_event_type": "sota_long",
                "sota_score_gate": {
                    "score": {
                        "net_score": 4,
                        "bull_total": 10,
                        "bear_total": 6,
                        "conflict": True,
                    }
                },
            },
        )

        result = executor.execute_action(action, engine)

        self.assertEqual(result["status"], "paper_recorded")
        dynamic = result["dynamic_high_leverage"]
        self.assertAlmostEqual(dynamic["effective_leverage"], 5.4)
        self.assertTrue(dynamic["score_bucket_sizing"]["applied"])
        self.assertIn("score_bucket:bear_total_6_light_boost", dynamic["leverage_reasons"])

    def test_long_score_bucket_sizing_leaves_non_matching_sota_open_unchanged(self) -> None:
        config = ExecutorConfig(
            mode="paper",
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            informative_timeframe="4h",
            leverage=10,
            margin_mode="cross",
            max_open_positions=1,
            risk_per_trade=0.01,
            state_db_path=":memory:",
            enable_dynamic_high_leverage_structure=True,
            dynamic_base_leverage=4.0,
            dynamic_max_effective_leverage=8.0,
            enable_long_score_bucket_sizing_live=True,
        )
        executor = make_executor(config)
        engine = make_engine()
        action = StrategyAction(
            type=ActionType.OPEN_LONG,
            timestamp="2023-11-14 22:13",
            direction=Direction.BULL,
            entry_price=1000.0,
            stop_price=985.0,
            target_price=1040.0,
            metadata={
                "capital_at_entry": 1000.0,
                "notional": 1000.0,
                "risk_based_notional": 1000.0,
                "candidate_event_type": "sota_long",
                "sota_score_gate": {
                    "score": {
                        "net_score": 8,
                        "bull_total": 10,
                        "bear_total": 2,
                        "conflict": False,
                    }
                },
            },
        )

        result = executor.execute_action(action, engine)

        self.assertEqual(result["status"], "paper_recorded")
        dynamic = result["dynamic_high_leverage"]
        self.assertAlmostEqual(dynamic["effective_leverage"], 4.0)
        self.assertFalse(dynamic["score_bucket_sizing"]["applied"])

    def test_sota_score_gate_rejects_open_candidate(self) -> None:
        config = ExecutorConfig(
            mode="paper",
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            informative_timeframe="4h",
            leverage=10,
            margin_mode="cross",
            max_open_positions=1,
            risk_per_trade=0.01,
            state_db_path=":memory:",
            enable_live_candidate_arbitration=True,
            enable_sota_score_gate_live=True,
            sota_score_net_min=99,
            sota_score_bull_min=99,
            sota_score_bear_max=0,
            sota_score_conflict_mode="any",
        )
        executor = make_executor(config)
        engine = make_engine()
        action = StrategyAction(
            type=ActionType.OPEN_LONG,
            timestamp="2023-11-14 22:13",
            direction=Direction.BULL,
            entry_price=1000.0,
            stop_price=990.0,
            target_price=1040.0,
            metadata={"index": 1},
        )

        filtered_actions, decision = executor._apply_live_candidate_arbitration(engine, [action], 1)

        self.assertEqual(filtered_actions, [])
        self.assertEqual(decision["decision"], "no_candidates")
        self.assertEqual(len(decision["score_gate_rejected"]), 1)
        self.assertEqual(decision["score_gate_rejected"][0]["reason"], "sota_score_gate")
        self.assertFalse(decision["score_gate_rejected"][0]["accepted"])

    def test_sota_score_gate_keeps_open_candidate(self) -> None:
        config = ExecutorConfig(
            mode="paper",
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            informative_timeframe="4h",
            leverage=10,
            margin_mode="cross",
            max_open_positions=1,
            risk_per_trade=0.01,
            state_db_path=":memory:",
            enable_live_candidate_arbitration=True,
            enable_sota_score_gate_live=True,
            sota_score_net_min=-99,
            sota_score_bull_min=0,
            sota_score_bear_max=99,
            sota_score_conflict_mode="any",
        )
        executor = make_executor(config)
        engine = make_engine()
        action = StrategyAction(
            type=ActionType.OPEN_LONG,
            timestamp="2023-11-14 22:13",
            direction=Direction.BULL,
            entry_price=1000.0,
            stop_price=990.0,
            target_price=1040.0,
            metadata={"index": 1},
        )

        filtered_actions, decision = executor._apply_live_candidate_arbitration(engine, [action], 1)

        self.assertEqual(len(filtered_actions), 1)
        self.assertEqual(filtered_actions[0].type, ActionType.OPEN_LONG)
        self.assertEqual(decision["decision"], "accepted")
        self.assertEqual(decision["selected"]["event_type"], "sota_long")
        self.assertEqual(decision["score_gate_rejected"], [])

    def test_smc_long_candidate_can_be_selected_when_enabled(self) -> None:
        config = ExecutorConfig(
            mode="paper",
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            informative_timeframe="4h",
            leverage=10,
            margin_mode="cross",
            max_open_positions=1,
            risk_per_trade=0.01,
            state_db_path=":memory:",
            enable_live_candidate_arbitration=True,
            enable_smc_long_live=True,
            live_candidate_priority=["sota_long", "smc_short", "smc_long"],
        )
        executor = make_executor(config)
        engine = make_engine()
        executor._smc_long_candidate = lambda _engine, _idx: {
            "event_type": "smc_long",
            "source_key": "smc_long|unit",
            "entry_idx": 1,
            "direction": Direction.BULL,
            "entry_price": 1000.0,
            "stop_price": 980.0,
            "target_price": 1030.0,
            "target_rr": 1.5,
            "max_hold_bars": 40,
            "trail_style": "tight",
            "leverage": 5.0,
            "position_size_pct": 0.5,
            "maintenance_margin_pct": 0.5,
            "requested_notional": 2500.0,
            "source": {"smc_case": "unit"},
        }

        filtered_actions, decision = executor._apply_live_candidate_arbitration(engine, [], 1)

        self.assertEqual(len(filtered_actions), 1)
        self.assertEqual(filtered_actions[0].type, ActionType.OPEN_LONG)
        self.assertEqual(filtered_actions[0].metadata["candidate_event_type"], "smc_long")
        self.assertEqual(decision["decision"], "accepted")
        self.assertEqual(decision["selected"]["event_type"], "smc_long")


if __name__ == "__main__":
    unittest.main()
