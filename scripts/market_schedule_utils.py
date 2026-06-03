#!/usr/bin/env python3
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


LOCAL_TZ = ZoneInfo("Asia/Shanghai")
US_TZ = ZoneInfo("America/New_York")

CN_2026_HOLIDAYS = {
    date(2026, 1, 1),
    date(2026, 1, 2),
    date(2026, 1, 5),
    date(2026, 2, 16),
    date(2026, 2, 17),
    date(2026, 2, 18),
    date(2026, 2, 19),
    date(2026, 2, 20),
    date(2026, 2, 23),
    date(2026, 4, 6),
    date(2026, 5, 1),
    date(2026, 5, 4),
    date(2026, 5, 5),
    date(2026, 6, 19),
    date(2026, 9, 25),
    date(2026, 9, 28),
    date(2026, 10, 1),
    date(2026, 10, 2),
    date(2026, 10, 5),
    date(2026, 10, 6),
    date(2026, 10, 7),
}

US_2026_HOLIDAYS = {
    date(2026, 1, 1),
    date(2026, 1, 19),
    date(2026, 2, 16),
    date(2026, 4, 3),
    date(2026, 5, 25),
    date(2026, 6, 19),
    date(2026, 7, 3),
    date(2026, 9, 7),
    date(2026, 11, 26),
    date(2026, 12, 25),
}

US_2026_EARLY_CLOSES = {
    date(2026, 11, 27): (13, 5),
    date(2026, 12, 24): (13, 5),
}


def is_cn_trade_day(day: date) -> bool:
    return day.weekday() < 5 and day not in CN_2026_HOLIDAYS


def is_us_trade_day(day: date) -> bool:
    return day.weekday() < 5 and day not in US_2026_HOLIDAYS


def next_trade_day(day: date, *, market: str) -> date:
    current = day + timedelta(days=1)
    checker = is_cn_trade_day if market == "cn" else is_us_trade_day
    while not checker(current):
        current += timedelta(days=1)
    return current


def now_market_day(market: str) -> date:
    if market == "cn":
        return datetime.now(LOCAL_TZ).date()
    if market == "us":
        return datetime.now(US_TZ).date()
    raise ValueError(f"Unsupported market: {market}")


def is_scheduled_window(mode: str, market: str, now_local: datetime | None = None) -> bool:
    local_now = now_local or datetime.now(LOCAL_TZ)
    if market == "cn":
        if mode == "preopen":
            return local_now.hour == 9 and 25 <= local_now.minute <= 29
        if mode == "close":
            return local_now.hour == 15 and 5 <= local_now.minute <= 15
        return False
    if market == "us":
        ny_now = local_now.astimezone(US_TZ)
        if mode == "preopen":
            return ny_now.hour == 9 and 25 <= ny_now.minute <= 29
        if mode == "close":
            return local_now.hour == 10 and 0 <= local_now.minute <= 10
        return False
    raise ValueError(f"Unsupported market: {market}")
