from __future__ import annotations

import unittest

import pandas as pd

from strategy.scalp_robust_v2_core import dataframe_to_candles


class DataframeToCandlesTest(unittest.TestCase):
    def test_converts_date_dataframe(self) -> None:
        dataframe = pd.DataFrame(
            {
                "date": ["2026-05-25 17:00:00+00:00", "2026-05-25 17:15:00+00:00"],
                "open": [77745, "77800.5"],
                "high": [77900.2, 78010],
                "low": [77600, 77750],
                "close": [77850, 77990.1],
                "volume": [12.3, "45.6"],
            }
        )

        candles = dataframe_to_candles(dataframe)

        self.assertEqual(len(candles), 2)
        self.assertEqual(candles[0].ts, 1779728400.0)
        self.assertEqual(candles[0].o, 77745.0)
        self.assertEqual(candles[1].c, 77990.1)
        self.assertEqual(candles[1].v, 45.6)

    def test_converts_timestamp_dataframe_without_volume(self) -> None:
        dataframe = pd.DataFrame(
            {
                "timestamp": [1779728400000, 1779729300000],
                "open": [1, 2],
                "high": [3, 4],
                "low": [0.5, 1.5],
                "close": [2.5, 3.5],
            }
        )

        candles = dataframe_to_candles(dataframe)

        self.assertEqual([candle.ts for candle in candles], [1779728400.0, 1779729300.0])
        self.assertEqual([candle.v for candle in candles], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
