from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bot.okx_executor import ExecutorConfig, OkxExecutionEngine
from bot.state_store import StateStore
from strategy.scalp_robust_v2_core import ActionType, Candle, Direction, ScalpRobustEngine, StrategyAction, StrategyConfig, Trade


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
    def test_guarded_weak_loss_selector_uses_research_thresholds(self) -> None:
        executor = make_executor(
            ExecutorConfig(
                mode="paper",
                symbol="BTC/USDT:USDT",
                timeframe="15m",
                informative_timeframe="4h",
                leverage=10,
                margin_mode="cross",
                max_open_positions=1,
                risk_per_trade=0.01,
                state_db_path=":memory:",
                stable_selector="guarded_weak_loss",
            )
        )

        source = {
            "direction": Direction.BULL,
            "exit_reason": "stop_loss",
            "return": -0.02,
            "regime_label": "high_growth",
            "risk_mode": "offense",
            "effective_leverage": 7.5,
            "failed_breakout_guard_applied": False,
            "feature_momentum": 0.05,
            "feature_ema_gap": 0.015,
            "feature_adx": 37.0,
            "feature_bullish_structure": False,
        }

        self.assertTrue(executor._stable_selector_allows(source))

    def test_trailing_stop_profit_reverse_selector_accepts_profitable_stop_loss(self) -> None:
        executor = make_executor(
            ExecutorConfig(
                mode="paper",
                symbol="BTC/USDT:USDT",
                timeframe="15m",
                informative_timeframe="4h",
                leverage=10,
                margin_mode="cross",
                max_open_positions=1,
                risk_per_trade=0.01,
                state_db_path=":memory:",
                stable_selector="trailing_stop_profit_reverse",
            )
        )

        source = {
            "direction": Direction.BULL,
            "exit_reason": "stop_loss",
            "return": 0.0125,
            "regime_label": "high_growth",
            "risk_mode": "offense",
            "effective_leverage": 7.5,
            "failed_breakout_guard_applied": False,
            "feature_momentum": 0.05,
            "feature_ema_gap": 0.015,
            "feature_adx": 37.0,
            "feature_bullish_structure": False,
        }

        self.assertTrue(executor._stable_selector_allows(source))

    def test_trailing_stage_profit_reverse_requires_stage_stop_update(self) -> None:
        executor = make_executor(
            ExecutorConfig(
                mode="paper",
                symbol="BTC/USDT:USDT",
                timeframe="15m",
                informative_timeframe="4h",
                leverage=10,
                margin_mode="cross",
                max_open_positions=1,
                risk_per_trade=0.01,
                state_db_path=":memory:",
                stable_selector="trailing_stage_profit_reverse",
            )
        )

        source = {
            "direction": Direction.BULL,
            "exit_reason": "stop_loss",
            "return": 0.0125,
            "regime_label": "high_growth",
            "risk_mode": "offense",
            "last_stop_update_reason": "trail_stage_1",
        }
        self.assertTrue(executor._stable_selector_allows(source))
        source["last_stop_update_reason"] = "atr_trail"
        self.assertFalse(executor._stable_selector_allows(source))

    def test_trailing_pressure_profit_reverse_accepts_pressure_touch_lock(self) -> None:
        executor = make_executor(
            ExecutorConfig(
                mode="paper",
                symbol="BTC/USDT:USDT",
                timeframe="15m",
                informative_timeframe="4h",
                leverage=10,
                margin_mode="cross",
                max_open_positions=1,
                risk_per_trade=0.01,
                state_db_path=":memory:",
                stable_selector="trailing_pressure_profit_reverse",
            )
        )

        source = {
            "direction": Direction.BULL,
            "exit_reason": "stop_loss",
            "return": 0.0125,
            "regime_label": "high_growth",
            "risk_mode": "offense",
            "pressure_touch_lock_applied": True,
        }
        self.assertTrue(executor._stable_selector_allows(source))

    def test_plain_stop_profit_reverse_rejects_trailing_updated_stops(self) -> None:
        executor = make_executor(
            ExecutorConfig(
                mode="paper",
                symbol="BTC/USDT:USDT",
                timeframe="15m",
                informative_timeframe="4h",
                leverage=10,
                margin_mode="cross",
                max_open_positions=1,
                risk_per_trade=0.01,
                state_db_path=":memory:",
                stable_selector="plain_stop_profit_reverse",
            )
        )

        source = {
            "direction": Direction.BULL,
            "exit_reason": "stop_loss",
            "return": 0.0125,
            "regime_label": "high_growth",
            "risk_mode": "offense",
            "last_stop_update_reason": "",
        }
        self.assertTrue(executor._stable_selector_allows(source))
        source["last_stop_update_reason"] = "trail_stage_0"
        self.assertFalse(executor._stable_selector_allows(source))

    def test_stable_candidate_uses_replay_sizing_defaults(self) -> None:
        executor = make_executor(
            ExecutorConfig(
                mode="paper",
                symbol="BTC/USDT:USDT",
                timeframe="15m",
                informative_timeframe="4h",
                leverage=10,
                margin_mode="cross",
                max_open_positions=1,
                risk_per_trade=0.01,
                state_db_path=":memory:",
                enable_stable_reverse_short_live=True,
                stable_selector="guarded_weak_loss",
            )
        )
        engine = make_engine()
        engine.trades.append(
            Trade(
                entry_time="2023-11-14 22:13",
                exit_time="2023-11-14 22:28",
                direction=Direction.BULL,
                signal_entry_price=1000.0,
                entry_price=1000.0,
                signal_exit_price=985.0,
                exit_price=985.0,
                gross_pnl=-15.0,
                fees=0.0,
                slippage_cost=0.0,
                pnl=-15.0,
                pnl_pct=-0.015,
                rr_ratio=-1.0,
                exit_reason="stop_loss",
                capital_at_entry=1000.0,
                notional=7500.0,
                quantity=7.5,
                entry_idx=0,
                exit_idx=1,
                initial_stop_price=985.0,
                risk_regime="offense",
                regime_label="high_growth",
                execution_effective_leverage=7.5,
                execution_risk_mode="offense",
                execution_leverage_reasons=["base"],
                execution_guard_diagnostics={
                    "feature_momentum": 0.05,
                    "feature_ema_gap": 0.015,
                    "feature_adx": 37.0,
                    "feature_bullish_structure": False,
                },
            )
        )

        candidate = executor._stable_reverse_short_candidate(engine, None, 1)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate["event_type"], "stable_reverse_short")
        self.assertAlmostEqual(candidate["requested_notional"], 5000.0)
        self.assertAlmostEqual(candidate["leverage"], 5.0)
        self.assertAlmostEqual(candidate["position_size_pct"], 1.0)

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
            candidate_event_type="stable_reverse_short",
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
        self.assertEqual(engine.position.execution_leverage_reasons, ["overlay_fixed:stable_reverse_short"])

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
                "candidate_event_type": "stable_reverse_short",
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
