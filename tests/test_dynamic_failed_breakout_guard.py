from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bot.okx_executor import ExecutorConfig, OkxExecutionEngine


class DynamicFailedBreakoutGuardTest(unittest.TestCase):
    def build_engine(self) -> OkxExecutionEngine:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        return OkxExecutionEngine(
            ExecutorConfig(
                mode="paper",
                symbol="BTC/USDT:USDT",
                timeframe="15m",
                informative_timeframe="4h",
                leverage=10,
                margin_mode="isolated",
                max_open_positions=1,
                risk_per_trade=0.035,
                telegram_enabled=False,
                state_db_path=str(Path(tmpdir.name) / "state.db"),
                dynamic_failed_breakout_guard_enabled=True,
                dynamic_failed_breakout_guard_leverage=2.0,
                dynamic_failed_breakout_guard_min_leverage=7.5,
                dynamic_failed_breakout_guard_min_quality_score=2,
                dynamic_failed_breakout_guard_min_momentum_pct=6.0,
                dynamic_failed_breakout_guard_min_ema_gap_pct=2.0,
                dynamic_failed_breakout_guard_min_adx=38.0,
                dynamic_failed_breakout_guard_regime_labels=["high_growth"],
                dynamic_failed_breakout_guard_risk_modes=["offense"],
                dynamic_failed_breakout_guard_directions=["BULL"],
            )
        )

    def test_dynamic_failed_breakout_guard_reduces_weak_offense_long(self) -> None:
        engine = self.build_engine()
        leverage, reasons = engine._dynamic_failed_breakout_guard(
            leverage=8.0,
            risk_mode="offense",
            diagnostics={
                "direction": "BULL",
                "regime_label": "high_growth",
                "feature_momentum": 0.03,
                "feature_ema_gap": 0.01,
                "feature_adx": 25.0,
                "feature_bullish_structure": False,
            },
        )
        self.assertEqual(leverage, 2.0)
        self.assertEqual(reasons, ["failed_breakout_guard:0/2"])

    def test_dynamic_failed_breakout_guard_keeps_strong_offense_long(self) -> None:
        engine = self.build_engine()
        leverage, reasons = engine._dynamic_failed_breakout_guard(
            leverage=8.0,
            risk_mode="offense",
            diagnostics={
                "direction": "BULL",
                "regime_label": "high_growth",
                "feature_momentum": 0.065,
                "feature_ema_gap": 0.021,
                "feature_adx": 39.0,
                "feature_bullish_structure": False,
            },
        )
        self.assertEqual(leverage, 8.0)
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
