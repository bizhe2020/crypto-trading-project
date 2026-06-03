#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.audit_tqqq_cash_regime_context import build_regime_frame, load_df
from scripts.scan_tqqq_cash_exit_profiles import run_exit_candidate


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def load_strategy_frame(config: dict[str, Any]) -> pd.DataFrame:
    data_root = ROOT / str(config["data_root"])
    qqq = load_df(data_root / "QQQ-1d.feather")
    execution_symbol = str(config.get("execution_symbol", "513100.SS"))
    execution_df = load_df(data_root / f"{execution_symbol}-1d.feather")
    spy = load_df(data_root / "SPY-1d.feather")
    ixic = load_df(data_root / "^IXIC-1d.feather")
    vix = load_df(data_root / "^VIX-1d.feather")
    frame = build_regime_frame(
        qqq,
        qqq.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"}),
        spy,
        ixic,
        vix,
        int(config["entry_fast_window"]),
        int(config["entry_slow_window"]),
    )
    execution_frame = execution_df[["date", "open", "high", "low", "close"]].copy()
    execution_frame["session_day"] = pd.to_datetime(execution_frame["date"], utc=True).dt.normalize()
    execution_frame = execution_frame.sort_values("session_day").drop_duplicates("session_day", keep="last").reset_index(drop=True)
    frame["session_day"] = pd.to_datetime(frame["date"], utc=True).dt.normalize()
    merged = frame.merge(
        execution_frame.rename(columns={"open": "asset_open", "high": "asset_high", "low": "asset_low", "close": "asset_close"})[
            ["session_day", "asset_open", "asset_high", "asset_low", "asset_close"]
        ],
        on="session_day",
        how="left",
    )
    required_ready = (
        merged["qqq_close"].notna()
        & merged["spy_close"].notna()
        & merged["ixic_close"].notna()
        & merged["vix_close"].notna()
        & merged["asset_open"].notna()
        & merged["asset_close"].notna()
    )
    merged["data_complete"] = required_ready
    missing_reasons: list[str] = []
    for _, row in merged.iterrows():
        reasons: list[str] = []
        for column in ["qqq_close", "spy_close", "ixic_close", "vix_close", "asset_open", "asset_close"]:
            if pd.isna(row[column]):
                reasons.append(column)
        missing_reasons.append(",".join(reasons))
    merged["missing_fields"] = missing_reasons

    incomplete_rows = merged.loc[~required_ready, ["date", "missing_fields"]].copy()
    merged.attrs["data_quality"] = {
        "total_rows": int(len(merged)),
        "complete_rows": int(required_ready.sum()),
        "incomplete_rows": int((~required_ready).sum()),
        "latest_complete_date": str(pd.Timestamp(merged.loc[required_ready, "date"].iloc[-1]).date()) if required_ready.any() else None,
        "latest_row_complete": bool(required_ready.iloc[-1]) if len(required_ready) else False,
        "incomplete_examples": [
            {
                "date": str(pd.Timestamp(row["date"]).date()),
                "missing_fields": str(row["missing_fields"]),
            }
            for _, row in incomplete_rows.head(10).iterrows()
        ],
    }
    if required_ready.any():
        last_complete_index = int(required_ready[required_ready].index[-1])
        merged = merged.iloc[: last_complete_index + 1].reset_index(drop=True)
    else:
        merged = merged.iloc[0:0].copy()
    merged["tqqq_open"] = merged["asset_open"]
    merged["tqqq_high"] = merged["asset_high"]
    merged["tqqq_low"] = merged["asset_low"]
    merged["tqqq_close"] = merged["asset_close"]
    return merged


def build_allow_mask(frame: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    if str(config.get("regime_filter")) == "vix_ixic":
        vix_allow = frame["vix_label"].isin(list(config.get("vix_allowed_labels", ["vix_low", "vix_normal"])))
        ixic_allow = frame["ixic_trend_label"].eq(str(config.get("ixic_trend_required", "ixic_up")))
        return (vix_allow & ixic_allow).reset_index(drop=True)
    return pd.Series([True] * len(frame))


def run_full_strategy(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    path = run_full_strategy_path(frame, config)
    if path.empty:
        return {
            "params": {
                "max_hold_days": int(config.get("max_hold_days", 90)),
                "trailing_lookback_days": int(config.get("trailing_lookback_days", 10)),
                "trailing_drawdown_pct": float(config.get("trailing_drawdown_pct", 5.0)),
                "switch_cost_bps": float(config.get("switch_cost_bps", 10.0)),
            },
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_like": 0.0,
            "trades": 0,
            "invested_days": 0,
            "invested_ratio_pct": 0.0,
            "yearly_returns_pct": {},
            "score": 0.0,
        }
    equity = path[["date", "capital"]].rename(columns={"capital": "equity"})
    total_return_pct = round((float(path.iloc[-1]["capital"]) / float(config.get("initial_capital", 1000.0)) - 1.0) * 100.0, 2)
    peak = path["capital"].cummax()
    dd = ((peak - path["capital"]) / peak.replace(0, pd.NA) * 100.0).max(skipna=True)
    daily_returns = pd.Series(path["daily_return"], dtype=float)
    return {
        "params": {
            "max_hold_days": int(config.get("max_hold_days", 90)),
            "trailing_lookback_days": int(config.get("trailing_lookback_days", 10)),
            "trailing_drawdown_pct": float(config.get("trailing_drawdown_pct", 5.0)),
            "switch_cost_bps": float(config.get("switch_cost_bps", 10.0)),
        },
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": round(float(dd or 0.0), 2),
        "sharpe_like": round((daily_returns.mean() / daily_returns.std()) if len(daily_returns) > 1 and daily_returns.std() > 0 else 0.0, 3),
        "trades": int((path["position"] != path["position"].shift(1)).sum()),
        "invested_days": int((path["position"] == "LONG").sum()),
        "invested_ratio_pct": round(float((path["position"] == "LONG").mean() * 100.0), 2),
        "yearly_returns_pct": annual_returns(equity),
        "score": round(total_return_pct - 2.0 * float(dd or 0.0) + float(annual_returns(equity).get("2026", 0.0)), 4),
    }


def run_full_strategy_path(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    allow_mask = build_allow_mask(frame, config).reset_index(drop=True)
    local = frame.copy().reset_index(drop=True)
    execution_price_mode = str(config.get("execution_price_mode", "same_close"))
    local["exec_same_close"] = local["asset_close"]
    local["exec_next_open"] = local["asset_open"].shift(-1)
    local["exec_next_close"] = local["asset_close"].shift(-1)
    local["exec_next_mid"] = (local["asset_high"].shift(-1) + local["asset_low"].shift(-1)) / 2.0

    def execution_price(row_index: int) -> float:
        column_map = {
            "same_close": "exec_same_close",
            "next_open": "exec_next_open",
            "next_close": "exec_next_close",
            "next_mid": "exec_next_mid",
        }
        selected_column = column_map.get(execution_price_mode, "exec_same_close")
        selected = local.iloc[row_index].get(selected_column)
        if pd.notna(selected):
            return float(selected)
        return float(local.iloc[row_index]["asset_close"])

    capital = float(config.get("initial_capital", 1000.0))
    previous_position = "CASH"
    hold_days = 0
    active = False
    rolling_peak = 0.0
    exit_override = False
    rows: list[dict[str, Any]] = []
    max_hold_days = int(config.get("max_hold_days", 90))
    trailing_lookback_days = int(config.get("trailing_lookback_days", 10))
    trailing_drawdown_pct = float(config.get("trailing_drawdown_pct", 5.0))
    hold_mode = str(config.get("hold_mode", "hard_exit"))
    switch_cost_bps = float(config.get("switch_cost_bps", 10.0))
    leverage_enabled = bool(config.get("conditional_leverage_enabled", False))
    leverage_trigger = str(config.get("conditional_leverage_trigger", "none"))
    leverage_value = float(config.get("conditional_leverage_value", 1.0))
    tiered_leverage_enabled = bool(config.get("tiered_leverage_enabled", False))
    tiered_rules = list(config.get("tiered_leverage_rules", []))

    def tiered_leverage_for_row(current_row: pd.Series) -> float:
        if not tiered_leverage_enabled or not tiered_rules:
            return 1.0
        for rule in tiered_rules:
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

    for idx, row in local.iterrows():
        desired_active = int(row["planned_position"]) > 0 and bool(allow_mask.iloc[idx])
        if not desired_active:
            active = False
            hold_days = 0
            rolling_peak = 0.0
            exit_override = False
        else:
            if not active and not exit_override:
                active = True
                hold_days = 0
                rolling_peak = float(row["asset_close"])
            elif exit_override:
                active = False

        position = "LONG" if active else "CASH"
        daily_ret = 0.0
        trailing_exit = False
        time_exit = False
        if idx > 0 and position == "LONG":
            prev_exec_price = execution_price(idx - 1)
            cur_exec_price = execution_price(idx)
            daily_ret = cur_exec_price / prev_exec_price - 1.0 if prev_exec_price > 0 else 0.0
            hold_days += 1
            rolling_peak = max(rolling_peak, float(row["asset_close"]))
            if trailing_lookback_days > 0 and trailing_drawdown_pct > 0 and hold_days >= trailing_lookback_days and rolling_peak > 0:
                drawdown_from_peak = (rolling_peak - float(row["asset_close"])) / rolling_peak * 100.0
                trailing_exit = drawdown_from_peak >= trailing_drawdown_pct
            if max_hold_days > 0 and hold_days >= max_hold_days:
                time_exit = True
            if trailing_exit or time_exit:
                active = False
                exit_override = True
                if hold_mode == "timer_refresh" and time_exit and not trailing_exit:
                    active = True
                    hold_days = 0
                    rolling_peak = float(row["asset_close"])
                    exit_override = False
                else:
                    hold_days = 0
                    rolling_peak = 0.0
        elif position == "CASH" and desired_active:
            exit_override = False

        if idx > 0 and position != previous_position:
            daily_ret -= switch_cost_bps / 10000.0
        leverage = 1.0 if position == "LONG" else 0.0
        target_leverage = leverage
        if position == "LONG":
            if tiered_leverage_enabled:
                target_leverage = tiered_leverage_for_row(row)
            elif leverage_enabled and leverage_value > 1.0:
                trigger_hit = False
                if leverage_trigger == "vix_low":
                    trigger_hit = str(row["vix_label"]) == "vix_low"
                elif leverage_trigger == "qqq_strong":
                    trigger_hit = str(row["rel_strength_label"]) == "qqq_strong"
                elif leverage_trigger == "vix_low_and_qqq_strong":
                    trigger_hit = str(row["vix_label"]) == "vix_low" and str(row["rel_strength_label"]) == "qqq_strong"
                if trigger_hit:
                    target_leverage = leverage_value
            if target_leverage > 1.0:
                cost_component = 0.0
                if idx > 0 and previous_position != "LONG":
                    prev_exec_price = execution_price(idx - 1)
                    cur_exec_price = execution_price(idx)
                    cost_component = daily_ret - ((cur_exec_price / prev_exec_price) - 1.0 if prev_exec_price > 0 else 0.0)
                asset_ret = daily_ret - cost_component
                daily_ret = asset_ret * target_leverage + cost_component
                leverage = target_leverage
        previous_position = position
        capital *= 1.0 + daily_ret
        rows.append(
            {
                "date": row["date"],
                "position": position,
                "daily_return": daily_ret,
                "capital": capital,
                "base_signal_active": bool(desired_active),
                "vix_label": row["vix_label"],
                "ixic_trend_label": row["ixic_trend_label"],
                "rel_strength_label": row["rel_strength_label"],
                "asset_close": float(row["asset_close"]),
                "execution_price": execution_price(idx),
                "execution_price_mode": execution_price_mode,
                "leverage": leverage,
            }
        )
    return pd.DataFrame(rows)


def latest_decision(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    allow_mask = build_allow_mask(frame, config)
    latest = frame.reset_index(drop=True).iloc[-1]
    planned = bool(int(latest["planned_position"]) > 0)
    allowed = bool(allow_mask.iloc[-1])
    should_hold = planned and allowed
    return {
        "date": str(pd.Timestamp(latest["date"]).date()),
        "signal_symbol": "QQQ",
        "execution_symbol": str(config.get("execution_symbol", "513100.SS")),
        "planned_signal": str(config.get("execution_symbol", "513100.SS")) if planned else "CASH",
        "allowed": allowed,
        "decision": str(config.get("execution_symbol", "513100.SS")) if should_hold else "CASH",
        "qqq_close": round(float(latest["qqq_close"]), 4),
        "execution_close": round(float(latest["asset_close"]), 4),
        "vix_label": str(latest["vix_label"]),
        "ixic_trend_label": str(latest["ixic_trend_label"]),
        "rel_strength_label": str(latest["rel_strength_label"]),
    }
