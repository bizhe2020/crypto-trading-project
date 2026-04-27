from __future__ import annotations

import unittest

from strategy.scalp_robust_v2_core import (
    ActionType,
    Candle,
    Direction,
    PositionState,
    ScalpRobustEngine,
    StrategyConfig,
)


def build_engine(direction: str) -> ScalpRobustEngine:
    if direction == Direction.BULL:
        candles = [
            Candle(ts=1_700_000_000, o=1000.0, h=1010.0, l=990.0, c=1000.0, v=100.0),
            Candle(ts=1_700_000_900, o=1080.0, h=1100.0, l=1070.0, c=1090.0, v=100.0),
        ]
    else:
        candles = [
            Candle(ts=1_700_000_000, o=1000.0, h=1010.0, l=990.0, c=1000.0, v=100.0),
            Candle(ts=1_700_000_900, o=920.0, h=930.0, l=900.0, c=910.0, v=100.0),
        ]
    config = StrategyConfig(
        enable_pressure_level_trailing=True,
        pressure_min_rr=1.0,
        pressure_lock_rr=0.2,
        pressure_atr_multiplier=0.0,
        pressure_proximity_pct=0.2,
        pressure_round_steps_usdt=[100.0],
        pressure_swing_lookback_bars=0,
        pressure_cluster_lookback_bars=0,
        pressure_take_profit_on_rejection=False,
        pressure_enable_target_cap=False,
        pressure_touch_lock_enabled=True,
        pressure_touch_lock_min_rr=1.0,
        pressure_touch_lock_buffer_pct=0.2,
        pressure_touch_lock_atr_multiplier=0.0,
        pressure_touch_lock_requires_touch=True,
    )
    return ScalpRobustEngine.from_candles(candles, candles, config)


class PressureTouchLockTest(unittest.TestCase):
    def test_long_touch_lock_moves_stop_near_current_close(self) -> None:
        engine = build_engine(Direction.BULL)
        engine.position = PositionState(
            direction=Direction.BULL,
            signal_entry_price=1000.0,
            entry_price=1000.0,
            sl_price=950.0,
            initial_sl_price=950.0,
            target_price=1200.0,
            entry_time="2023-11-14 22:13",
            capital_at_entry=1000.0,
            risk_amount=50.0,
            notional=1000.0,
            quantity=1.0,
            entry_fee=0.0,
            entry_slippage_cost=0.0,
            entry_idx=0,
            entry_regime_score=0,
            target_rr=4.0,
            max_hold_bars=None,
            trail_style="normal",
            regime_label="flat",
        )

        action = engine._apply_pressure_level_exit_or_trail(engine.position, engine.c15m[1], 1)

        self.assertIsNotNone(action)
        self.assertEqual(action.type, ActionType.UPDATE_STOP)
        self.assertEqual(action.reason, "pressure_level_trail")
        self.assertAlmostEqual(action.stop_price, 1090.0 * 0.9995)
        self.assertTrue(action.metadata["touch_lock_enabled"])
        self.assertTrue(action.metadata["touch_lock_requires_touch"])
        self.assertEqual(engine.position.sl_price, action.stop_price)

    def test_short_touch_lock_moves_stop_near_current_close(self) -> None:
        engine = build_engine(Direction.BEAR)
        engine.position = PositionState(
            direction=Direction.BEAR,
            signal_entry_price=1000.0,
            entry_price=1000.0,
            sl_price=1050.0,
            initial_sl_price=1050.0,
            target_price=800.0,
            entry_time="2023-11-14 22:13",
            capital_at_entry=1000.0,
            risk_amount=50.0,
            notional=1000.0,
            quantity=1.0,
            entry_fee=0.0,
            entry_slippage_cost=0.0,
            entry_idx=0,
            entry_regime_score=0,
            target_rr=4.0,
            max_hold_bars=None,
            trail_style="normal",
            regime_label="flat",
        )

        action = engine._apply_pressure_level_exit_or_trail(engine.position, engine.c15m[1], 1)

        self.assertIsNotNone(action)
        self.assertEqual(action.type, ActionType.UPDATE_STOP)
        self.assertEqual(action.reason, "pressure_level_trail")
        self.assertAlmostEqual(action.stop_price, 910.0 * 1.0005)
        self.assertTrue(action.metadata["touch_lock_enabled"])
        self.assertTrue(action.metadata["touch_lock_requires_touch"])
        self.assertEqual(engine.position.sl_price, action.stop_price)


if __name__ == "__main__":
    unittest.main()
