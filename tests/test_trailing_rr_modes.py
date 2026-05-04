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


if __name__ == "__main__":
    unittest.main()
