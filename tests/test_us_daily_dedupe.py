from __future__ import annotations

import pandas as pd

from scripts.fetch_public_etf_history import dedupe_daily_by_us_trade_day
from scripts.tqqq_cash_strict_utils import dedupe_last_by_us_trade_day


def test_us_daily_dedupe_keeps_last_timestamp_per_trade_day() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-29T13:30:00Z",
                    "2026-05-29T14:40:00Z",
                    "2026-06-01T13:30:00Z",
                ],
                utc=True,
            ),
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "volume": [100, 200, 300],
        }
    )

    fetched = dedupe_daily_by_us_trade_day(frame)
    strict = dedupe_last_by_us_trade_day(frame)

    assert list(fetched["close"]) == [2.0, 3.0]
    assert list(strict["close"]) == [2.0, 3.0]
