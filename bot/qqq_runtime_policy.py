from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    value = str(timeframe or "").strip().lower()
    if value.endswith("m"):
        return pd.Timedelta(minutes=int(value[:-1]))
    if value.endswith("h"):
        return pd.Timedelta(hours=int(value[:-1]))
    if value.endswith("d"):
        return pd.Timedelta(days=int(value[:-1]))
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def filter_closed_bars(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    now: pd.Timestamp | datetime | None = None,
    grace_seconds: int = 0,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    current = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")
    close_cutoff = current - pd.Timedelta(seconds=max(0, int(grace_seconds)))
    bar_delta = timeframe_to_timedelta(timeframe)
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], utc=True)
    return result[result["date"] + bar_delta <= close_cutoff].reset_index(drop=True)


def latest_closed_bar(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    now: pd.Timestamp | datetime | None = None,
    grace_seconds: int = 0,
) -> pd.Series:
    closed = filter_closed_bars(frame, timeframe=timeframe, now=now, grace_seconds=grace_seconds)
    if closed.empty:
        raise RuntimeError(f"No closed {timeframe} bars available")
    return closed.iloc[-1]


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    current = date(year, month, 1)
    days = (weekday - current.weekday()) % 7
    return current + timedelta(days=days + 7 * (nth - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        current = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        current = date(year, month + 1, 1) - timedelta(days=1)
    return current - timedelta(days=(current.weekday() - weekday) % 7)


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    actual = date(year, month, day)
    if actual.weekday() == 5:
        return actual - timedelta(days=1)
    if actual.weekday() == 6:
        return actual + timedelta(days=1)
    return actual


def _easter_date(year: int) -> date:
    # Gregorian computus. Good Friday is an NYSE holiday.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nyse_holidays_for_year(year: int) -> set[date]:
    holidays = {
        _observed_fixed_holiday(year, 1, 1),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter_date(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed_fixed_holiday(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed_fixed_holiday(year, 12, 25),
    }
    if year >= 2022:
        holidays.add(_observed_fixed_holiday(year, 6, 19))
    return holidays


def nyse_calendar_status(local_date: date) -> dict[str, object]:
    holidays: set[date] = set()
    for year in (local_date.year - 1, local_date.year, local_date.year + 1):
        holidays.update(_nyse_holidays_for_year(year))
    weekday_ok = local_date.weekday() < 5
    holiday = local_date in holidays
    trading_day = weekday_ok and not holiday

    thanksgiving = _nth_weekday(local_date.year, 11, 3, 4)
    half_day = False
    early_close = None
    if trading_day:
        if local_date == thanksgiving + timedelta(days=1):
            half_day = True
        elif local_date.month == 7 and local_date.day == 3:
            half_day = True
        elif local_date.month == 12 and local_date.day == 24:
            half_day = True
        if half_day:
            early_close = "13:00"

    return {
        "calendar": "NYSE",
        "trading_day": bool(trading_day),
        "weekday_ok": bool(weekday_ok),
        "holiday": bool(holiday),
        "half_day": bool(half_day),
        "early_close": early_close,
    }


def trading_calendar_status(calendar_name: str | None, local_date: date) -> dict[str, object]:
    calendar = str(calendar_name or "").strip().upper()
    if not calendar or calendar in {"NONE", "OFF", "FALSE"}:
        return {
            "calendar": calendar or "none",
            "trading_day": True,
            "weekday_ok": local_date.weekday() < 5,
            "holiday": False,
            "half_day": False,
            "early_close": None,
        }
    if calendar != "NYSE":
        raise ValueError(f"Unsupported trading calendar: {calendar_name}")
    return nyse_calendar_status(local_date)


def market_time_window_status(
    *,
    enabled: bool,
    timezone_name: str,
    start_time: str,
    end_time: str,
    trading_calendar: str | None = "NYSE",
    now: pd.Timestamp | datetime | None = None,
) -> dict[str, object]:
    current_utc = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    if current_utc.tzinfo is None:
        current_utc = current_utc.tz_localize("UTC")
    else:
        current_utc = current_utc.tz_convert("UTC")
    local = current_utc.tz_convert(ZoneInfo(str(timezone_name)))
    calendar_status = trading_calendar_status(trading_calendar, local.date())
    weekday_ok = bool(calendar_status.get("weekday_ok", int(local.weekday()) < 5))
    clock = local.strftime("%H:%M")
    effective_end = str(end_time)
    early_close = calendar_status.get("early_close")
    if early_close:
        effective_end = min(effective_end, str(early_close))
    trading_day = bool(calendar_status.get("trading_day", weekday_ok))
    if not bool(enabled):
        open_now = True
        reason = "disabled"
    elif not trading_day:
        open_now = False
        reason = "non_trading_day"
    elif clock < str(start_time):
        open_now = False
        reason = "before_window"
    elif clock > effective_end:
        open_now = False
        reason = "after_window"
    else:
        open_now = True
        reason = "inside_window"
    return {
        "enabled": bool(enabled),
        "open": bool(open_now),
        "reason": reason,
        "timezone": str(timezone_name),
        "start": str(start_time),
        "end": effective_end,
        "configured_end": str(end_time),
        "now_utc": str(current_utc),
        "now_local": str(local),
        "weekday_ok": bool(weekday_ok),
        "trading_calendar": calendar_status,
    }
