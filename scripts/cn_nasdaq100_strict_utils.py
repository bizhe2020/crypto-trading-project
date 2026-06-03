#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any
from bisect import bisect_right

import pandas as pd
from zoneinfo import ZoneInfo

from scripts.audit_tqqq_cash_regime_context import label_rel_strength, label_vix, load_df


ROOT = Path(__file__).resolve().parents[1]
LOCAL_CN = ZoneInfo("Asia/Shanghai")
LOCAL_US = ZoneInfo("America/New_York")


def load_config(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text())


def _dedupe_last_by_local_day(df: pd.DataFrame, tz: ZoneInfo) -> pd.DataFrame:
    local = df.copy()
    local["local_day"] = pd.to_datetime(local["date"], utc=True).dt.tz_convert(tz).dt.date
    return local.sort_values("date").drop_duplicates("local_day", keep="last").reset_index(drop=True)


def _build_us_signal_frame(data_root: Path, fast_window: int, slow_window: int) -> pd.DataFrame:
    qqq = _dedupe_last_by_local_day(load_df(data_root / "QQQ-1d.feather"), LOCAL_US)
    spy = _dedupe_last_by_local_day(load_df(data_root / "SPY-1d.feather"), LOCAL_US)
    ixic = _dedupe_last_by_local_day(load_df(data_root / "^IXIC-1d.feather"), LOCAL_US)
    vix = _dedupe_last_by_local_day(load_df(data_root / "^VIX-1d.feather"), LOCAL_US)

    frame = qqq[["date", "local_day", "close"]].rename(columns={"date": "signal_date", "close": "qqq_close", "local_day": "us_day"})
    frame = frame.merge(spy[["local_day", "close"]].rename(columns={"local_day": "us_day", "close": "spy_close"}), on="us_day", how="left")
    frame = frame.merge(ixic[["local_day", "close"]].rename(columns={"local_day": "us_day", "close": "ixic_close"}), on="us_day", how="left")
    frame = frame.merge(vix[["local_day", "close"]].rename(columns={"local_day": "us_day", "close": "vix_close"}), on="us_day", how="left")
    frame["fast_ma"] = frame["qqq_close"].rolling(fast_window).mean()
    frame["slow_ma"] = frame["qqq_close"].rolling(slow_window).mean()
    frame["entry_signal"] = (frame["fast_ma"] > frame["slow_ma"]).astype(int)
    frame["qqq_spy_rel_20"] = (frame["qqq_close"] / frame["qqq_close"].shift(20)) - (frame["spy_close"] / frame["spy_close"].shift(20))
    frame["ixic_ma_50"] = frame["ixic_close"].rolling(50).mean()
    frame["ixic_trend_label"] = "unknown"
    ixic_ready = frame["ixic_close"].notna() & frame["ixic_ma_50"].notna()
    frame.loc[ixic_ready & (frame["ixic_close"] > frame["ixic_ma_50"]), "ixic_trend_label"] = "ixic_up"
    frame.loc[ixic_ready & (frame["ixic_close"] <= frame["ixic_ma_50"]), "ixic_trend_label"] = "ixic_down"
    frame["vix_label"] = frame["vix_close"].map(lambda value: label_vix(float(value)) if pd.notna(value) else "unknown")
    frame["rel_strength_label"] = frame["qqq_spy_rel_20"].map(lambda value: label_rel_strength(float(value)) if pd.notna(value) else "unknown")
    frame = frame.dropna(subset=["fast_ma", "slow_ma", "qqq_close", "spy_close", "ixic_close", "vix_close"]).reset_index(drop=True)
    return frame


def load_strict_frame(config: dict[str, Any]) -> pd.DataFrame:
    data_root = ROOT / str(config["data_root"])
    fast_window = int(config["entry_fast_window"])
    slow_window = int(config["entry_slow_window"])
    signal_frame = _build_us_signal_frame(data_root, fast_window, slow_window)

    execution_symbol = str(config.get("execution_symbol", "513100.SS"))
    cn = _dedupe_last_by_local_day(load_df(data_root / f"{execution_symbol}-1d.feather"), LOCAL_CN)
    cn = cn[["date", "local_day", "open", "high", "low", "close"]].rename(
        columns={
            "local_day": "cn_day",
            "open": "asset_open",
            "high": "asset_high",
            "low": "asset_low",
            "close": "asset_close",
        }
    )

    cn_trade_days = sorted(cn["cn_day"].tolist())
    mapped_cn_days: list[Any] = []
    for us_day in signal_frame["us_day"].tolist():
        insert_at = bisect_right(cn_trade_days, us_day)
        if insert_at >= len(cn_trade_days):
            mapped_cn_days.append(None)
        else:
            mapped_cn_days.append(cn_trade_days[insert_at])
    signal_frame["cn_day"] = mapped_cn_days
    signal_frame = signal_frame.dropna(subset=["cn_day"]).reset_index(drop=True)
    merged = signal_frame.merge(cn, on="cn_day", how="inner")
    merged["planned_position"] = merged["entry_signal"].astype(int)
    merged["date"] = merged["cn_day"].map(pd.Timestamp)
    merged = merged.sort_values("cn_day").reset_index(drop=True)
    return merged


def build_allow_mask(frame: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    regime_filter = str(config.get("regime_filter", "vix_ixic"))
    vix_allow = frame["vix_label"].isin(list(config.get("vix_allowed_labels", ["vix_low", "vix_normal"])))
    ixic_allow = frame["ixic_trend_label"].eq(str(config.get("ixic_trend_required", "ixic_up")))
    rel_allow = frame["rel_strength_label"].ne("qqq_weak")
    if regime_filter == "vix_ixic":
        return (vix_allow & ixic_allow).reset_index(drop=True)
    if regime_filter == "ixic_filter":
        return ixic_allow.reset_index(drop=True)
    if regime_filter == "all_three":
        return (vix_allow & ixic_allow & rel_allow).reset_index(drop=True)
    return pd.Series([True] * len(frame))


def annual_returns(equity: pd.DataFrame) -> dict[str, float]:
    if equity.empty:
        return {}
    frame = equity.copy()
    frame["year"] = frame["date"].dt.year.astype(str)
    out: dict[str, float] = {}
    for year, group in frame.groupby("year"):
        start = float(group.iloc[0]["equity"])
        end = float(group.iloc[-1]["equity"])
        out[year] = round((end / start - 1.0) * 100.0, 2) if start > 0 else 0.0
    return out


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0, pd.NA) * 100.0
    return float(dd.max(skipna=True) or 0.0)


def _tiered_leverage_for_row(current_row: pd.Series, config: dict[str, Any]) -> float:
    if not bool(config.get("tiered_leverage_enabled", False)):
        return 1.0
    for rule in list(config.get("tiered_leverage_rules", [])):
        if not isinstance(rule, dict):
            continue
        vix_label = str(rule.get("vix_label", "") or "").strip()
        rel_strength_label = str(rule.get("rel_strength_label", "") or "").strip()
        ixic_trend_label = str(rule.get("ixic_trend_label", "") or "").strip()
        if vix_label and str(current_row["vix_label"]) != vix_label:
            continue
        if rel_strength_label and str(current_row["rel_strength_label"]) != rel_strength_label:
            continue
        if ixic_trend_label and str(current_row["ixic_trend_label"]) != ixic_trend_label:
            continue
        return float(rule.get("leverage", 1.0) or 1.0)
    return 1.0


def run_strict_path(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    allow_mask = build_allow_mask(frame, config)
    capital = float(config.get("initial_capital", 1000.0))
    holding = False
    pending_exit = False
    exit_override = False
    hold_days = 0
    rolling_peak = 0.0
    previous_close = 0.0
    rows: list[dict[str, Any]] = []

    max_hold_days = int(config.get("max_hold_days", 90))
    trailing_lookback_days = int(config.get("trailing_lookback_days", 5))
    trailing_drawdown_pct = float(config.get("trailing_drawdown_pct", 8.0))
    switch_cost_bps = float(config.get("switch_cost_bps", 10.0))

    for idx, row in frame.iterrows():
        start_capital = capital
        desired_today = bool(int(row["planned_position"]) > 0 and bool(allow_mask.iloc[idx]))

        if exit_override and not desired_today:
            exit_override = False

        action_cost = 0.0
        entered_today = False
        exited_today = False

        if holding and previous_close > 0:
            capital *= float(row["asset_open"]) / previous_close

        if holding and (pending_exit or not desired_today):
            action_cost += switch_cost_bps / 10000.0
            capital *= 1.0 - switch_cost_bps / 10000.0
            holding = False
            pending_exit = False
            hold_days = 0
            rolling_peak = 0.0
            exited_today = True

        leverage = 0.0
        if (not holding) and desired_today and not exit_override:
            action_cost += switch_cost_bps / 10000.0
            capital *= 1.0 - switch_cost_bps / 10000.0
            holding = True
            hold_days = 0
            rolling_peak = float(row["asset_open"])
            entered_today = True

        trailing_exit = False
        time_exit = False
        if holding:
            leverage = _tiered_leverage_for_row(row, config)
            open_price = float(row["asset_open"])
            close_price = float(row["asset_close"])
            if open_price > 0:
                capital *= 1.0 + leverage * (close_price / open_price - 1.0)
            hold_days += 1
            rolling_peak = max(rolling_peak, close_price)
            if trailing_lookback_days > 0 and trailing_drawdown_pct > 0 and hold_days >= trailing_lookback_days and rolling_peak > 0:
                drawdown_from_peak = (rolling_peak - close_price) / rolling_peak * 100.0
                trailing_exit = drawdown_from_peak >= trailing_drawdown_pct
            if max_hold_days > 0 and hold_days >= max_hold_days:
                time_exit = True
            if trailing_exit or time_exit:
                pending_exit = True
                exit_override = True

        previous_close = float(row["asset_close"])
        daily_return = capital / start_capital - 1.0 if start_capital > 0 else 0.0
        rows.append(
            {
                "date": pd.Timestamp(row["cn_day"]),
                "position": "LONG" if holding else "CASH",
                "daily_return": float(daily_return),
                "capital": float(capital),
                "entered_today": bool(entered_today),
                "exited_today": bool(exited_today),
                "pending_exit": bool(pending_exit),
                "trailing_exit": bool(trailing_exit),
                "time_exit": bool(time_exit),
                "vix_label": str(row["vix_label"]),
                "ixic_trend_label": str(row["ixic_trend_label"]),
                "rel_strength_label": str(row["rel_strength_label"]),
                "asset_open": float(row["asset_open"]),
                "asset_close": float(row["asset_close"]),
                "leverage": float(leverage),
                "signal_us_day": str(row["us_day"]),
                "cn_day": str(row["cn_day"]),
            }
        )

    return pd.DataFrame(rows)


def summarize_path(path: pd.DataFrame, initial_capital: float = 1000.0) -> dict[str, Any]:
    if path.empty:
        return {
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "trades": 0,
            "invested_days": 0,
            "yearly_returns_pct": {},
            "latest_position": "CASH",
        }
    equity = path[["date", "capital"]].rename(columns={"capital": "equity"})
    return {
        "total_return_pct": round((float(path.iloc[-1]["capital"]) / float(initial_capital) - 1.0) * 100.0, 2) if len(path) > 0 else 0.0,
        "max_drawdown_pct": round(max_drawdown_pct(path["capital"]), 2),
        "trades": int(path["entered_today"].sum()),
        "invested_days": int((path["position"] == "LONG").sum()),
        "yearly_returns_pct": annual_returns(equity),
        "latest_position": str(path.iloc[-1]["position"]),
    }
