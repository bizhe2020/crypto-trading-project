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
            stop_price=990.0,
            target_price=1040.0,
            metadata={"capital_at_entry": 1000.0, "notional": 1000.0, "risk_based_notional": 1000.0},
        )

        result = executor.execute_action(action, engine)

        self.assertEqual(result["status"], "paper_recorded")
        self.assertFalse(result["overlay_skipped_dynamic_high_leverage"])
        self.assertIsInstance(result.get("dynamic_high_leverage"), dict)


if __name__ == "__main__":
    unittest.main()
