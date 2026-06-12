from __future__ import annotations

import pandas as pd

from scripts.fetch_public_etf_history import dedupe_daily_by_us_trade_day, merge_history
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


def test_daily_merge_prefers_refetched_same_trade_day_bar() -> None:
    existing = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-04T13:30:00Z", "2026-06-05T13:30:00Z"], utc=True),
            "open": [83.47, 81.55],
            "high": [86.25, 82.08],
            "low": [82.48, 76.76],
            "close": [85.22, 76.77],
            "volume": [58_790_800, 58_392_341],
        }
    )
    fetched = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-05T13:30:00Z"], utc=True),
            "open": [81.55],
            "high": [82.08],
            "low": [72.68],
            "close": [73.05],
            "volume": [112_658_916],
        }
    )

    merged = merge_history(existing, fetched, "1d")

    assert list(merged["date"].dt.strftime("%Y-%m-%d")) == ["2026-06-04", "2026-06-05"]
    assert float(merged.iloc[-1]["close"]) == 73.05
    assert int(merged.iloc[-1]["volume"]) == 112_658_916
