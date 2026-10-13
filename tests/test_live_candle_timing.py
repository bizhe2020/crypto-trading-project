from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

import bot.okx_executor as okx_executor
from bot.okx_executor import OkxExecutionEngine
from strategy.scalp_robust_v2_core import dataframe_to_candles


def _engine_at(now: datetime) -> OkxExecutionEngine:
    engine = OkxExecutionEngine.__new__(OkxExecutionEngine)
    engine.config = SimpleNamespace(timeframe="15m")

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D401 - test clock
            return now

    okx_executor.datetime = _FrozenDatetime
    return engine


@pytest.fixture(autouse=True)
def _restore_datetime():
    original = okx_executor.datetime
    yield
    okx_executor.datetime = original


@pytest.mark.parametrize(
    "now",
    [
        # 醒在 [边界, 边界+buffer) 内：旧实现会睡满 900s，整根 K 线晚 15 分钟。
        datetime(2026, 10, 13, 7, 30, 4, 500000, tzinfo=timezone.utc),
        datetime(2026, 10, 13, 7, 30, 0, 100000, tzinfo=timezone.utc),
        # 普通等待：唤醒点必须落在边界+buffer 之后（不能被 int 截断提前）。
        datetime(2026, 10, 13, 7, 16, 33, 400000, tzinfo=timezone.utc),
        datetime(2026, 10, 13, 7, 29, 59, 900000, tzinfo=timezone.utc),
    ],
)
def test_wake_up_lands_right_after_next_close_buffer(now: datetime) -> None:
    engine = _engine_at(now)
    wait = engine.seconds_until_next_close(5)
    wake = now + timedelta(seconds=wait)
    expected_close = datetime(2026, 10, 13, 7, 30, 5, tzinfo=timezone.utc)
    assert expected_close <= wake < expected_close + timedelta(seconds=1)
    # 醒来时 latest_closed 必须已推进到 07:15，否则 run_loop 会再 WAIT 一整根。
    engine_after = _engine_at(wake)
    assert engine_after.latest_closed_candle_time(5) == "2026-10-13 07:15"


def test_dataframe_to_candles_matches_row_semantics() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-10-13 07:00", "2026-10-13 07:15"], utc=True),
            "Open": [1.5, 2],
            "high": [3, 4.25],
            "low": [1, 1.5],
            "close": [2, 3],
            "volume": [10, 0.5],
        }
    )
    candles = dataframe_to_candles(df)
    assert [(c.ts, c.o, c.h, c.l, c.c, c.v) for c in candles] == [
        (datetime(2026, 10, 13, 7, 0, tzinfo=timezone.utc).timestamp(), 1.5, 3.0, 1.0, 2.0, 10.0),
        (datetime(2026, 10, 13, 7, 15, tzinfo=timezone.utc).timestamp(), 2.0, 4.25, 1.5, 3.0, 0.5),
    ]
    assert all(isinstance(c.o, float) for c in candles)
    no_volume = dataframe_to_candles(df.drop(columns=["volume"]))
    assert [c.v for c in no_volume] == [0.0, 0.0]
