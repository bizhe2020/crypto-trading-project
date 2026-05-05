from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.backtest_config_report import load_config_payload
from scripts.live_readiness_report import load_prepared_data
from bot.okx_executor import ExecutorConfig
from strategy.scalp_robust_v2_core import Direction, ScalpRobustEngine


ROOT = Path(__file__).resolve().parents[1]


class InformativeAsofReplayTest(unittest.TestCase):
    def build_engine(
        self,
        *,
        informative_asof_from_15m: bool = False,
        confirmed_4h_only: bool = False,
        timestamp: str = "2026-05-04 09:15",
    ) -> tuple[ScalpRobustEngine, int]:
        payload = load_config_payload(ROOT / "var" / "tokyo_audit" / "config.live.high-leverage-structure.json")
        prepared = load_prepared_data(
            data_15m_path=ROOT / "var" / "tokyo_audit" / "BTC_USDT_USDT-15m-futures.remote-tail.feather",
            data_4h_path=ROOT / "var" / "tokyo_audit" / "BTC_USDT_USDT-4h-futures.remote-tail.feather",
            start=pd.Timestamp("2026-04-15 16:00", tz="UTC"),
            threshold_payload=payload.get("regime_switcher_thresholds"),
            informative_asof_from_15m=informative_asof_from_15m,
            confirmed_4h_only=confirmed_4h_only,
        )
        engine = ScalpRobustEngine(
            prepared.c4h,
            prepared.c15m,
            prepared.mapping,
            prepared.precomputed,
            ExecutorConfig.from_dict(payload).to_scalp_strategy_config(),
        )
        idx = next(idx for idx, candle in enumerate(engine.c15m) if engine._timestamp_for_idx(idx) == timestamp)
        return engine, idx

    def test_asof_4h_reproduces_live_pending_at_0915(self) -> None:
        engine, idx = self.build_engine(informative_asof_from_15m=True)

        engine._apply_regime_switch_for_idx(idx)
        pending = engine._build_pending_pullback(idx, engine._bias_for_idx(idx))

        self.assertEqual(engine._bias_for_idx(idx), Direction.BULL)
        self.assertEqual(engine._regime_switch_label_for_idx(idx), "normal")
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.direction, Direction.BULL)
        self.assertEqual(pending.bos_idx, 1772)
        self.assertEqual(pending.ob_zone, {"top": 79828.7, "bottom": 79665.3})
        self.assertEqual(pending.pullback_window, 30)

    def test_finalized_4h_does_not_reproduce_live_pending_at_0915(self) -> None:
        engine, idx = self.build_engine(informative_asof_from_15m=False)

        self.assertEqual(engine._bias_for_idx(idx), Direction.BEAR)
        self.assertIsNone(engine._build_pending_pullback(idx, engine._bias_for_idx(idx)))

    def test_confirmed_4h_matches_last_closed_4h_before_1200(self) -> None:
        engine, idx = self.build_engine(confirmed_4h_only=True)

        engine._apply_regime_switch_for_idx(idx)
        pending = engine._build_pending_pullback(idx, engine._bias_for_idx(idx))

        self.assertEqual(engine._bias_for_idx(idx), Direction.BULL)
        self.assertEqual(engine._regime_switch_label_for_idx(idx), "normal")
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.direction, Direction.BULL)
        self.assertEqual(pending.bos_idx, 1772)
        self.assertEqual(pending.ob_zone, {"top": 79828.7, "bottom": 79665.3})
        self.assertEqual(pending.pullback_window, 30)

    def test_confirmed_4h_switches_after_4h_close(self) -> None:
        engine, idx = self.build_engine(confirmed_4h_only=True, timestamp="2026-05-04 12:00")

        self.assertEqual(engine._bias_for_idx(idx), Direction.BEAR)
        self.assertIsNone(engine._build_pending_pullback(idx, engine._bias_for_idx(idx)))


if __name__ == "__main__":
    unittest.main()
