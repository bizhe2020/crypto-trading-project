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


def make_engine(*, require_stop_at_breakeven: bool = True) -> ScalpRobustEngine:
    candles = [
        Candle(ts=float(i * 900), o=100.0 + i, h=101.0 + i, l=99.0 + i, c=100.0 + i, v=1.0)
        for i in range(12)
    ]
    config = StrategyConfig(
        enable_controlled_scale_in=True,
        scale_in_trigger_rr=0.25,
        scale_in_min_bars_held=1,
        scale_in_risk_fraction=0.5,
        scale_in_total_risk_multiplier=1.0,
        scale_in_max_total_notional_multiplier=1.0,
        scale_in_min_target_rr=1.5,
        scale_in_require_stop_at_breakeven=require_stop_at_breakeven,
        scale_in_regime_labels=["static"],
        position_size_pct=1.0,
        leverage=3.0,
        initial_capital=1000.0,
        slippage_bps=0.0,
        taker_fee_rate=0.0,
    )
    engine = ScalpRobustEngine.from_candles(candles, candles, config)
    engine.position = PositionState(
        direction=Direction.BULL,
        signal_entry_price=100.0,
        entry_price=100.0,
        sl_price=100.0,
        initial_sl_price=98.0,
        target_price=110.0,
        entry_time="1970-01-01 00:00",
        capital_at_entry=1000.0,
        risk_amount=100.0,
        notional=1000.0,
        quantity=10.0,
        entry_fee=0.0,
        entry_slippage_cost=0.0,
        entry_idx=0,
        entry_regime_score=3,
        target_rr=5.0,
        max_hold_bars=None,
        trail_style="loose",
        risk_regime="bull_strong",
        regime_label="high_growth",
    )
    return engine


class ControlledScaleInTest(unittest.TestCase):
    def test_scale_in_adds_second_slot_with_shared_risk_cap(self) -> None:
        engine = make_engine()
        assert engine.position is not None

        action = engine._maybe_controlled_scale_in(engine.position, engine.c15m[3], 3)

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.type, ActionType.SCALE_IN)
        self.assertEqual(engine.position.scale_in_slots, 2)
        self.assertLessEqual(engine._open_risk_at_stop(engine.position, engine.position.sl_price), engine.position.risk_amount)
        self.assertGreater(engine.position.quantity, 10.0)

    def test_scale_in_requires_breakeven_stop_when_enabled(self) -> None:
        engine = make_engine(require_stop_at_breakeven=True)
        assert engine.position is not None
        engine.position.sl_price = 99.0

        action = engine._maybe_controlled_scale_in(engine.position, engine.c15m[3], 3)

        self.assertIsNone(action)
        self.assertEqual(engine.position.scale_in_slots, 1)


if __name__ == "__main__":
    unittest.main()
