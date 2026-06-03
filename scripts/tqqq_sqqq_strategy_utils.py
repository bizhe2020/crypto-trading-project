#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from scripts.audit_tqqq_cash_regime_context import load_df
from scripts.scan_tqqq_context_bucket_overlays import allow_short, prepare_frame, select_long_profile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"


def asset_price_column(asset: str) -> str:
    if asset == "TQQQ":
        return "tqqq_close"
    if asset == "SQQQ":
        return "sqqq_close"
    raise ValueError(f"Unsupported asset: {asset}")


def load_strategy_frame(
    *,
    data_root: Path = DEFAULT_PUBLIC_DIR,
    entry_fast_window: int = 25,
    entry_slow_window: int = 200,
) -> pd.DataFrame:
    qqq = load_df(data_root / "QQQ-1d.feather")
    tqqq = load_df(data_root / "TQQQ-1d.feather")
    sqqq = load_df(data_root / "SQQQ-1d.feather")
    spy = load_df(data_root / "SPY-1d.feather")
    ixic = load_df(data_root / "^IXIC-1d.feather")
    vix = load_df(data_root / "^VIX-1d.feather")
    return prepare_frame(qqq, tqqq, sqqq, spy, ixic, vix, entry_fast_window, entry_slow_window)


def run_strategy_path(
    frame: pd.DataFrame,
    *,
    long_profile_name: str = "stable_base",
    short_rule_name: str = "bearish_score5",
    short_exit_profile: tuple[int, int, float] = (20, 5, 6.0),
    initial_capital: float = 1000.0,
    switch_cost_bps: float = 10.0,
) -> pd.DataFrame:
    long_mask = frame["vix_label"].isin(["vix_low", "vix_normal"]) & frame["ixic_trend_label"].eq("ixic_up")
    strict_long_mask = pd.Series(long_mask.to_numpy(dtype=bool), index=frame.index).shift(1, fill_value=False)
    strict_short_allow_base = frame.apply(lambda row: allow_short(row, short_rule_name), axis=1)
    strict_short_allow = pd.Series(strict_short_allow_base.to_numpy(dtype=bool), index=frame.index).shift(1, fill_value=False)
    capital = float(initial_capital)
    previous_position = "CASH"
    active_asset = "CASH"
    active_max_hold_days = 0
    active_trailing_lookback_days = 0
    active_trailing_drawdown_pct = 0.0
    hold_days = 0
    rolling_peak = 0.0
    exit_override_asset: str | None = None
    rows: list[dict[str, Any]] = []

    for idx, row in frame.iterrows():
        raw_desired_asset = "CASH"
        desired_long = int(row["planned_trend"]) > 0 and bool(strict_long_mask.iloc[idx])
        desired_short = int(row["planned_trend"]) < 0 and bool(strict_short_allow.iloc[idx])
        context_row = frame.iloc[max(idx - 1, 0)]
        candidate_long_profile = select_long_profile(context_row, long_profile_name)
        if desired_long:
            raw_desired_asset = "TQQQ"
        elif desired_short:
            raw_desired_asset = "SQQQ"

        daily_ret = 0.0
        if idx > 0 and active_asset != "CASH":
            price_col = asset_price_column(active_asset)
            prev_close = float(frame.iloc[idx - 1][price_col])
            cur_open = float(row[f"{active_asset.lower()}_open"])
            daily_ret = cur_open / prev_close - 1.0 if prev_close > 0 else 0.0
            capital *= 1.0 + daily_ret

        if active_asset != "CASH" and raw_desired_asset != active_asset:
            capital *= 1.0 - float(switch_cost_bps) / 10000.0
            active_asset = "CASH"
            hold_days = 0
            rolling_peak = 0.0
            if exit_override_asset == raw_desired_asset:
                raw_desired_asset = "CASH"

        desired_asset = raw_desired_asset
        if desired_asset != "CASH" and exit_override_asset == desired_asset:
            desired_asset = "CASH"

        if active_asset == "CASH" and desired_asset != "CASH":
            active_asset = desired_asset
            capital *= 1.0 - float(switch_cost_bps) / 10000.0
            hold_days = 0
            rolling_peak = float(row[asset_price_column(active_asset)])
            if active_asset == "TQQQ":
                active_max_hold_days, active_trailing_lookback_days, active_trailing_drawdown_pct = candidate_long_profile
            else:
                active_max_hold_days, active_trailing_lookback_days, active_trailing_drawdown_pct = short_exit_profile

        position = active_asset
        trailing_exit = False
        time_exit = False
        if position != "CASH":
            cur_open = float(row[f"{position.lower()}_open"])
            cur_close = float(row[asset_price_column(position)])
            intraday_ret = cur_close / cur_open - 1.0 if cur_open > 0 else 0.0
            capital *= 1.0 + intraday_ret
            daily_ret += intraday_ret
            hold_days += 1
            rolling_peak = max(rolling_peak, cur_close)
            if (
                active_trailing_lookback_days > 0
                and active_trailing_drawdown_pct > 0
                and hold_days >= active_trailing_lookback_days
                and rolling_peak > 0
            ):
                drawdown_from_peak = (rolling_peak - cur_close) / rolling_peak * 100.0
                trailing_exit = drawdown_from_peak >= active_trailing_drawdown_pct
            if active_max_hold_days > 0 and hold_days >= active_max_hold_days:
                time_exit = True
            if trailing_exit or time_exit:
                exit_override_asset = position
                active_asset = "CASH"
                hold_days = 0
                rolling_peak = 0.0

        previous_position = position

        if position == "CASH" and raw_desired_asset != "CASH" and exit_override_asset == raw_desired_asset:
            exit_override_asset = None
        elif raw_desired_asset == "CASH":
            exit_override_asset = None

        rows.append(
            {
                "date": pd.Timestamp(row["date"]),
                "capital": float(capital),
                "position": position,
                "daily_return": float(daily_ret),
                "signal_source_day": str(pd.Timestamp(row["date"]).date()),
            }
        )

    return pd.DataFrame(rows)
