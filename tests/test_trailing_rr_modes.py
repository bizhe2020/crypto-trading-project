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


def build_mode_engine(config: StrategyConfig) -> ScalpRobustEngine:
    candles = [
        Candle(ts=1_700_000_000, o=1000.0, h=1005.0, l=995.0, c=1000.0, v=100.0),
        Candle(ts=1_700_000_900, o=1000.0, h=1060.0, l=998.0, c=1020.0, v=100.0),
    ]
    return ScalpRobustEngine.from_candles(candles, candles, config)

def build_long_position() -> PositionState:
    return PositionState(
        direction=Direction.BULL,
        signal_entry_price=1000.0,
        entry_price=1000.0,
        sl_price=980.0,
        initial_sl_price=980.0,
        target_price=1080.0,
        entry_time="2023-11-14 22:13",
        capital_at_entry=1000.0,
        risk_amount=20.0,
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


class TrailingRRModeTest(unittest.TestCase):
    def test_stage_trigger_extreme_can_advance_without_close_confirmation(self) -> None:
        close_engine = build_mode_engine(StrategyConfig(stage_trigger_rr_mode="close"))
        close_engine.position = build_long_position()
        close_action = close_engine._apply_trailing_bull(close_engine.position, close_engine.c15m[1], 1)
        self.assertIsNotNone(close_action)
        self.assertEqual(close_action.reason, "trail_stage_0")
        self.assertAlmostEqual(close_action.stop_price, 1000.0)

        extreme_engine = build_mode_engine(StrategyConfig(stage_trigger_rr_mode="extreme"))
        extreme_engine.position = build_long_position()
        extreme_action = extreme_engine._apply_trailing_bull(extreme_engine.position, extreme_engine.c15m[1], 1)

        self.assertIsNotNone(extreme_action)
        self.assertEqual(extreme_action.type, ActionType.UPDATE_STOP)
        self.assertEqual(extreme_action.reason, "trail_stage_2")
        self.assertAlmostEqual(extreme_action.stop_price, 1020.0)

    def test_time_trailing_extreme_can_enter_breathe_stage_earlier(self) -> None:
        close_engine = build_mode_engine(
            StrategyConfig(
                enable_time_based_trailing=True,
                time_trailing_rr_mode="close",
                S1_trigger_rr=1.5,
                S3_trigger_rr=4.0,
            )
        )
        close_engine.position = build_long_position()
        close_state = close_engine._time_based_trailing_state(close_engine.position, close_engine.c15m[1], 1)
        self.assertEqual(close_state.stage, 0)

        extreme_engine = build_mode_engine(
            StrategyConfig(
                enable_time_based_trailing=True,
                time_trailing_rr_mode="extreme",
                S1_trigger_rr=1.5,
                S3_trigger_rr=4.0,
            )
        )
        extreme_engine.position = build_long_position()
        extreme_state = extreme_engine._time_based_trailing_state(extreme_engine.position, extreme_engine.c15m[1], 1)
        self.assertEqual(extreme_state.stage, 1)
        self.assertEqual(extreme_state.label, "S1_breathe")

    def test_atr_activation_extreme_can_enable_trail_before_close_hits_threshold(self) -> None:
        close_engine = build_mode_engine(
            StrategyConfig(
                enable_atr_trailing=True,
                atr_activation_rr=2.5,
                atr_activation_rr_mode="close",
                atr_tight_multiplier=1.0,
            )
        )
        close_engine.position = build_long_position()
        close_engine.position.trail_style = "tight"
        close_action = close_engine._apply_atr_trailing_bull(close_engine.position, close_engine.c15m[1], 1)
        self.assertIsNone(close_action)

        extreme_engine = build_mode_engine(
            StrategyConfig(
                enable_atr_trailing=True,
                atr_activation_rr=2.5,
                atr_activation_rr_mode="extreme",
                atr_tight_multiplier=1.0,
            )
        )
        extreme_engine.position = build_long_position()
        extreme_engine.position.trail_style = "tight"
        extreme_action = extreme_engine._apply_atr_trailing_bull(extreme_engine.position, extreme_engine.c15m[1], 1)

        self.assertIsNotNone(extreme_action)
        self.assertEqual(extreme_action.type, ActionType.UPDATE_STOP)
        self.assertEqual(extreme_action.reason, "atr_trail")

    def test_exit_profile_overrides_stage_trigger_mode_per_position(self) -> None:
        engine = build_mode_engine(StrategyConfig(stage_trigger_rr_mode="close"))
        engine.position = build_long_position()
        engine.position.exit_profile = "unit_extreme"
        engine.position.exit_profile_overrides = {"stage_trigger_rr_mode": "extreme"}

        action = engine._apply_trailing_bull(engine.position, engine.c15m[1], 1)

        self.assertIsNotNone(action)
        self.assertEqual(action.reason, "trail_stage_2")
        self.assertAlmostEqual(action.stop_price, 1020.0)

    def test_exit_profile_overrides_time_trailing_mode_per_position(self) -> None:
        engine = build_mode_engine(
            StrategyConfig(
                enable_time_based_trailing=True,
                time_trailing_rr_mode="close",
                S1_trigger_rr=1.5,
                S3_trigger_rr=4.0,
            )
        )
        engine.position = build_long_position()
        engine.position.exit_profile = "unit_extreme"
        engine.position.exit_profile_overrides = {"time_trailing_rr_mode": "extreme"}

        state = engine._time_based_trailing_state(engine.position, engine.c15m[1], 1)

        self.assertEqual(state.stage, 1)
        self.assertEqual(state.label, "S1_breathe")

    def test_exit_profile_overrides_atr_activation_and_multiplier_per_position(self) -> None:
        engine = build_mode_engine(
            StrategyConfig(
                enable_atr_trailing=True,
                atr_activation_rr=99.0,
                atr_activation_rr_mode="close",
                atr_tight_multiplier=5.0,
            )
        )
        engine.position = build_long_position()
        engine.position.trail_style = "tight"
        engine.position.exit_profile = "unit_extreme"
        engine.position.exit_profile_overrides = {
            "atr_activation_rr": 2.5,
            "atr_activation_rr_mode": "extreme",
            "atr_tight_multiplier": 1.0,
        }

        action = engine._apply_atr_trailing_bull(engine.position, engine.c15m[1], 1)

        self.assertIsNotNone(action)
        self.assertEqual(action.reason, "atr_trail")
        self.assertEqual(action.metadata["atr_multiplier"], 1.0)


if __name__ == "__main__":
    unittest.main()
