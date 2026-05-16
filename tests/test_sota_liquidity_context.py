from __future__ import annotations

import unittest

from scripts.report_smc_trade_context import event_support_context
from scripts.sota_liquidity_context import flatten_context_features, liquidity_context_for_entry
from strategy.scalp_robust_v2_core import Candle


def candle(idx: int, o: float, h: float, l: float, c: float) -> Candle:
    return Candle(ts=float(idx * 900), o=o, h=h, l=l, c=c, v=1.0)


class SotaLiquidityContextTest(unittest.TestCase):
    def test_detects_recent_fvg_near_entry_without_sweep_requirement(self) -> None:
        candles = [
            candle(0, 100, 101, 99, 100),
            candle(1, 100, 101, 99, 100),
            candle(2, 102, 105, 104, 104),
            candle(3, 104, 105, 102, 103),
        ]

        context = liquidity_context_for_entry(candles, 3, "BULL", recent_fvg_lookback_bars=4)
        features = flatten_context_features(context)

        self.assertTrue(context["recent_fvg_near_entry"])
        self.assertEqual(context["recent_sweep_status"], None)
        self.assertTrue(features["feature_recent_fvg_near_entry"])
        self.assertFalse(features["feature_recent_sweep_mss"])

    def test_uses_only_events_available_by_entry_idx(self) -> None:
        candles = [
            candle(0, 100, 101, 99, 100),
            candle(1, 100, 101, 99, 100),
            candle(2, 100, 101, 98, 100.5),
            candle(3, 100.5, 102, 100, 101.5),
        ]
        future_like_event = type(
            "Event",
            (),
            {
                "direction": "BULL",
                "sweep_idx": 2,
                "mss_idx": 4,
                "fvg": None,
                "retest": None,
                "sweep_distance_pct": 1.0,
            },
        )()

        context = liquidity_context_for_entry(
            candles,
            3,
            "BULL",
            liquidity_events=[future_like_event],
            recent_sweep_lookback_bars=8,
        )

        self.assertTrue(context["recent_sweep"])
        self.assertFalse(context["recent_sweep_mss"])
        self.assertEqual(context["recent_sweep_status"], "sweep_only")

    def test_event_support_context_does_not_use_future_final_status(self) -> None:
        future_final_event = type(
            "Event",
            (),
            {
                "direction": "BULL",
                "sweep_idx": 2,
                "mss_idx": 5,
                "fvg": type("Fvg", (), {"idx": 5})(),
                "retest": None,
                "status": "mss_with_fvg",
                "swept_level": 100.0,
                "sweep_extreme": 98.0,
                "sweep_distance_pct": 2.0,
            },
        )()

        context = event_support_context({"BULL": [future_final_event]}, "BULL", entry_idx=3, lookback=8)

        self.assertTrue(context["recent_sweep"])
        self.assertFalse(context["recent_sweep_mss"])
        self.assertFalse(context["event_has_fvg"])
        self.assertEqual(context["event_status"], "sweep_only")


if __name__ == "__main__":
    unittest.main()
