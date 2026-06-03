from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return ROOT / path


def _rolling_zscore(series: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0.0, pd.NA)


def _normalize_signal_days(calendar_source: pd.DataFrame | pd.Series | list[Any]) -> pd.Series:
    if isinstance(calendar_source, pd.DataFrame):
        source = pd.Series(pd.NaT, index=calendar_source.index, dtype="datetime64[ns, UTC]")
        for column in ["daily_signal_timestamp", "session_day", "date"]:
            if column not in calendar_source.columns:
                continue
            values = pd.to_datetime(calendar_source[column], utc=True, errors="coerce").dt.normalize()
            source = source.where(source.notna(), values)
        return source
    return pd.to_datetime(pd.Series(calendar_source), utc=True, errors="coerce").dt.normalize()


def build_macro_proxy_context(config: dict[str, Any], calendar_source: pd.DataFrame | pd.Series | list[Any]) -> dict[str, Any]:
    enabled = bool(config.get("macro_proxy_overlay_enabled", False))
    payload: dict[str, Any] = {
        "enabled": enabled,
        "frame": None,
        "mode": str(config.get("macro_proxy_overlay_mode", "")),
    }
    if not enabled:
        return payload

    fail_open = bool(config.get("macro_proxy_overlay_fail_open", True))
    path = _resolve_path(config.get("macro_proxy_overlay_path"))
    mode = str(config.get("macro_proxy_overlay_mode", "dollar_zscore_cap") or "dollar_zscore_cap")
    value_column = str(config.get("macro_proxy_overlay_value_column", "macro_broad_dollar_index") or "macro_broad_dollar_index")
    if mode != "dollar_zscore_cap":
        if fail_open:
            payload.update({"ignored": True, "reason": "unsupported_mode", "mode": mode})
            return payload
        raise ValueError(f"Unsupported macro proxy overlay mode: {mode}")
    if path is None:
        if fail_open:
            payload.update({"ignored": True, "reason": "missing_path"})
            return payload
        raise RuntimeError("macro proxy overlay path is not configured")

    try:
        macro = pd.read_feather(path)
        if "date" not in macro.columns:
            raise ValueError(f"{path} missing date column")
        if value_column not in macro.columns:
            raise ValueError(f"{path} missing value column: {value_column}")
        macro = macro[["date", value_column]].copy()
        macro["macro_signal_day"] = pd.to_datetime(macro["date"], utc=True, errors="coerce").dt.normalize()
        macro[value_column] = pd.to_numeric(macro[value_column], errors="coerce")
        macro = (
            macro.dropna(subset=["macro_signal_day", value_column])
            .sort_values("macro_signal_day")
            .drop_duplicates("macro_signal_day", keep="last")
            .reset_index(drop=True)
        )
        calendar = pd.DataFrame({"signal_day": _normalize_signal_days(calendar_source)})
        calendar = (
            calendar.dropna(subset=["signal_day"])
            .sort_values("signal_day")
            .drop_duplicates("signal_day", keep="last")
            .reset_index(drop=True)
        )
        if calendar.empty:
            raise ValueError("macro proxy overlay calendar is empty")
        frame = pd.merge_asof(
            calendar,
            macro,
            left_on="signal_day",
            right_on="macro_signal_day",
            direction="backward",
            allow_exact_matches=True,
        )
        window = int(config.get("macro_proxy_overlay_dollar_z_window", 252) or 252)
        min_periods = int(config.get("macro_proxy_overlay_dollar_z_min_periods", 80) or 80)
        score_column = "macro_proxy_overlay_score"
        frame[score_column] = _rolling_zscore(frame[value_column], window=window, min_periods=min_periods)
        payload.update(
            {
                "frame": frame,
                "mode": mode,
                "value_column": value_column,
                "score_column": score_column,
                "path": str(path),
                "window": window,
                "min_periods": min_periods,
            }
        )
        return payload
    except Exception as exc:
        if fail_open:
            payload.update({"ignored": True, "reason": "error_fail_open", "error": str(exc)})
            return payload
        raise


def _signal_for_bar(
    frame: pd.DataFrame | None,
    *,
    score_column: str,
    value_column: str,
    bar_date: pd.Timestamp,
    signal_timestamp: pd.Timestamp | None,
    use_previous: bool,
    max_stale_calendar_days: int,
    stale_guard_enabled: bool,
) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {"available": False, "reason": "missing_or_empty"}
    bar_ts = pd.Timestamp(bar_date)
    bar_ts = bar_ts.tz_localize("UTC") if bar_ts.tzinfo is None else bar_ts.tz_convert("UTC")
    if signal_timestamp is None or pd.isna(signal_timestamp):
        signal_ts = bar_ts
    else:
        signal_ts = pd.Timestamp(signal_timestamp)
    signal_ts = signal_ts.tz_localize("UTC") if signal_ts.tzinfo is None else signal_ts.tz_convert("UTC")
    signal_day = signal_ts.normalize()
    eligible = frame[frame["signal_day"] < signal_day] if use_previous else frame[frame["signal_day"] <= signal_day]
    if eligible.empty:
        return {"available": False, "reason": "no_signal_before_bar"}
    row = eligible.iloc[-1]
    score = row.get(score_column)
    if pd.isna(score):
        return {"available": False, "reason": "score_nan", "signal_date": pd.Timestamp(row["signal_day"]).tz_convert("UTC")}
    row_signal_day = pd.Timestamp(row["signal_day"]).tz_convert("UTC")
    macro_signal_day = pd.Timestamp(row["macro_signal_day"]).tz_convert("UTC") if pd.notna(row.get("macro_signal_day")) else None
    lag_days = int((signal_day - row_signal_day.normalize()).days)
    source_lag_days = int((signal_day - macro_signal_day.normalize()).days) if macro_signal_day is not None else None
    if not use_previous and source_lag_days not in (None, 0):
        return {
            "available": False,
            "reason": "macro_signal_not_current",
            "score": round(float(score), 6),
            "signal_date": row_signal_day,
            "macro_signal_date": macro_signal_day,
            "lag_days": lag_days,
            "source_lag_days": source_lag_days,
        }
    stale = bool(stale_guard_enabled and source_lag_days is not None and source_lag_days > int(max_stale_calendar_days))
    if stale:
        return {
            "available": False,
            "reason": "stale",
            "score": round(float(score), 6),
            "signal_date": row_signal_day,
            "macro_signal_date": macro_signal_day,
            "lag_days": lag_days,
            "source_lag_days": source_lag_days,
        }
    value = row.get(value_column)
    return {
        "available": True,
        "reason": "ok",
        "score": round(float(score), 6),
        "raw_value": round(float(value), 6) if pd.notna(value) else None,
        "signal_date": row_signal_day,
        "macro_signal_date": macro_signal_day,
        "lag_days": lag_days,
        "source_lag_days": source_lag_days,
    }


def macro_proxy_overlay_for_bar(
    config: dict[str, Any],
    macro_context: dict[str, Any],
    bar_date: pd.Timestamp,
    *,
    signal_timestamp: pd.Timestamp | None = None,
) -> dict[str, Any]:
    enabled = bool(macro_context.get("enabled"))
    payload: dict[str, Any] = {
        "enabled": enabled,
        "cash_gate": False,
        "leverage_multiplier": 1.0,
        "triggered": False,
        "available": False,
        "mode": str(macro_context.get("mode", "")),
    }
    if not enabled:
        return payload

    fail_open = bool(config.get("macro_proxy_overlay_fail_open", True))
    frame = macro_context.get("frame")
    if frame is None or str(macro_context.get("reason", "")) == "error_fail_open":
        if fail_open:
            payload.update(
                {
                    "ignored": True,
                    "reason": str(macro_context.get("reason", "missing_context") or "missing_context"),
                    "error": macro_context.get("error"),
                }
            )
            return payload
        raise RuntimeError(f"macro proxy overlay context unavailable: {macro_context}")

    signal = _signal_for_bar(
        frame,
        score_column=str(macro_context["score_column"]),
        value_column=str(macro_context["value_column"]),
        bar_date=bar_date,
        signal_timestamp=signal_timestamp,
        use_previous=bool(config.get("macro_proxy_overlay_use_previous_signal", False)),
        max_stale_calendar_days=int(config.get("macro_proxy_overlay_max_stale_calendar_days", 5) or 5),
        stale_guard_enabled=bool(config.get("macro_proxy_overlay_stale_guard_enabled", True)),
    )
    payload.update(
        {
            "available": bool(signal.get("available", False)),
            "reason": signal.get("reason"),
            "score": signal.get("score"),
            "raw_value": signal.get("raw_value"),
            "signal_date": str(signal.get("signal_date")) if signal.get("signal_date") is not None else None,
            "macro_signal_date": str(signal.get("macro_signal_date")) if signal.get("macro_signal_date") is not None else None,
            "lag_days": signal.get("lag_days"),
            "source_lag_days": signal.get("source_lag_days"),
        }
    )
    if not bool(signal.get("available", False)):
        if fail_open or str(signal.get("reason")) in {"score_nan", "no_signal_before_bar", "macro_signal_not_current"}:
            payload["ignored"] = True
            return payload
        raise RuntimeError(f"macro proxy overlay unavailable for bar {bar_date}: {signal}")

    threshold = float(config.get("macro_proxy_overlay_dollar_z_threshold", 1.5) or 1.5)
    multiplier = max(0.0, min(1.0, float(config.get("macro_proxy_overlay_leverage_multiplier", 1.0) or 1.0)))
    payload["threshold"] = threshold
    if float(signal["score"]) >= threshold:
        payload["triggered"] = True
        if multiplier <= 1e-12:
            payload["cash_gate"] = True
            payload["reason"] = "dollar_zscore_cash_gate"
            return payload
        payload["leverage_multiplier"] = multiplier
        payload["reason"] = "dollar_zscore_cap"
    return payload


def apply_macro_proxy_overlay(*, allow_long: bool, leverage_target: float, overlay: dict[str, Any]) -> dict[str, Any]:
    adjusted_allow = bool(allow_long)
    adjusted_leverage = float(leverage_target) if adjusted_allow else 0.0
    if not adjusted_allow:
        return {
            "allow_long": False,
            "leverage_target": 0.0,
            "triggered": False,
            "capped": False,
            "cash_gate": False,
            "reason": "inactive",
        }
    if bool(overlay.get("cash_gate", False)):
        return {
            "allow_long": False,
            "leverage_target": 0.0,
            "triggered": bool(overlay.get("triggered", False)),
            "capped": False,
            "cash_gate": True,
            "reason": overlay.get("reason"),
        }
    multiplier = max(0.0, min(1.0, float(overlay.get("leverage_multiplier", 1.0) or 1.0)))
    adjusted_leverage *= multiplier
    if adjusted_leverage <= 1e-12:
        adjusted_allow = False
        adjusted_leverage = 0.0
    return {
        "allow_long": bool(adjusted_allow),
        "leverage_target": float(adjusted_leverage),
        "triggered": bool(overlay.get("triggered", False)),
        "capped": bool(multiplier < 0.999 and adjusted_allow),
        "cash_gate": False,
        "reason": overlay.get("reason"),
    }
