#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_qqq_shadow_gate_v2_combined import (  # noqa: E402
    CANDIDATES,
    DEFAULT_BTC_FROZEN,
    DEFAULT_NQ_4H,
    DEFAULT_NQ_FUNDING,
    DEFAULT_QQQ_USDT_CONFIG,
    DEFAULT_REAL_4H,
    DEFAULT_REAL_FUNDING,
    DEFAULT_ROUTER_CONFIG,
    annual_metrics,
    bar_closure_audit,
    closed_only_bars,
    load_enriched_bars,
    overlap_consistency,
    parse_end_timestamp,
    reentry_ready,
    rolling_compare,
    route_candidate,
    trigger_gate,
)
from scripts.qqq_drawdown_risk_model import (  # noqa: E402
    DEFAULT_BREADTH_PATH,
    DEFAULT_MACRO_PATH,
    DEFAULT_PUBLIC_DIR,
    build_feature_frame,
)
from scripts.replay_proxy_strategy_router import (  # noqa: E402
    _load_risk_predictions,
    _risk_overlay_for_bar,
    build_btc_path_from_frozen_artifact,
    max_drawdown_pct,
)
from scripts.replay_qqq_usdt_10x import is_funding_settlement_bar, load_funding  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_v2_macro_proxy_overlay_audit_20220101_20260529.json"
BASELINE_PARAMS = dict(CANDIDATES["shadow_v2_low_dd_plus_risk"])

MACRO_POLICIES: dict[str, dict[str, Any]] = {
    "baseline_v2": {
        "kind": "none",
        "label": "Current V2 baseline",
    },
    "dollar_flat": {
        "kind": "cash",
        "rule": "dollar",
        "label": "Dollar stress flat",
    },
    "dollar_flat_z1_5": {
        "kind": "cash",
        "rule": "dollar",
        "dollar_z_threshold": 1.5,
        "label": "Dollar stress flat (z >= 1.5)",
    },
    "dollar_cap50_z1_5": {
        "kind": "tiered",
        "rule": "dollar",
        "dollar_z_threshold": 1.5,
        "multiplier": 0.5,
        "label": "Dollar stress cap 50% (z >= 1.5)",
    },
    "dollar_credit_flat": {
        "kind": "cash",
        "rule": "dollar_and_credit",
        "label": "Dollar + credit stress flat",
    },
    "dollar_credit_or_rates_flat": {
        "kind": "cash",
        "rule": "dollar_and_credit_or_rates",
        "label": "Dollar + credit/rates stress flat",
    },
    "macro_5way_tiered": {
        "kind": "tiered",
        "rule": "stress_score",
        "cap_count": 2,
        "cash_count": 3,
        "multiplier": 0.5,
        "label": "2 of 5 stress groups cap 50%, 3+ flat",
    },
    "macro_5way_flat": {
        "kind": "cash",
        "rule": "stress_score",
        "min_count": 2,
        "label": "2 of 5 stress groups flat",
    },
}


def rolling_zscore(series: pd.Series, window: int = 252, min_periods: int = 80) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0.0, np.nan)


def load_daily_macro_proxy_context() -> tuple[pd.DataFrame, list[str]]:
    frame, _, missing_flags = build_feature_frame(
        data_dir=DEFAULT_PUBLIC_DIR,
        breadth_path=DEFAULT_BREADTH_PATH,
        macro_path=DEFAULT_MACRO_PATH,
        macro_groups=["all"],
        horizon_days=10,
        drawdown_threshold_pct=5.0,
    )
    daily = frame.copy()
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    for column in [
        "hyg_ief_rel_20d",
        "tlt_spy_rel_20d",
        "vvix_close",
        "skew_close",
        "qqq_breadth_ret20_dispersion",
        "qqew_qqq_rel_20d",
    ]:
        if column in daily.columns:
            values = pd.to_numeric(daily[column], errors="coerce")
            daily[f"{column}_z_252d"] = rolling_zscore(values)
    keep = [
        "date",
        "macro_broad_dollar_index_z_252d",
        "macro_high_yield_oas_z_252d",
        "macro_real_yield_10y_z_252d",
        "hyg_ief_rel_20d_z_252d",
        "tlt_spy_rel_20d_z_252d",
        "vix3m_vix_spread",
        "vvix_close_z_252d",
        "skew_close_z_252d",
        "qqq_breadth_ret20_dispersion_z_252d",
        "qqew_qqq_rel_20d_z_252d",
    ]
    return daily[[column for column in keep if column in daily.columns]].sort_values("date").reset_index(drop=True), missing_flags


def attach_macro_context(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge_asof(
        bars.sort_values("date"),
        daily.sort_values("date"),
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.reset_index(drop=True)


def _is_ge(value: Any, threshold: float) -> bool:
    return bool(pd.notna(value) and float(value) >= float(threshold))


def _is_le(value: Any, threshold: float) -> bool:
    return bool(pd.notna(value) and float(value) <= float(threshold))


def macro_stress_flags(row: Any, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = policy or {}
    dollar_z_threshold = float(policy.get("dollar_z_threshold", 1.0) or 1.0)
    credit_oas_z_threshold = float(policy.get("credit_oas_z_threshold", 1.0) or 1.0)
    credit_rel_z_threshold = float(policy.get("credit_rel_z_threshold", -1.0) or -1.0)
    rates_real_yield_z_threshold = float(policy.get("rates_real_yield_z_threshold", 1.0) or 1.0)
    rates_rel_z_threshold = float(policy.get("rates_rel_z_threshold", -1.0) or -1.0)
    vol_vvix_z_threshold = float(policy.get("vol_vvix_z_threshold", 1.0) or 1.0)
    vol_skew_z_threshold = float(policy.get("vol_skew_z_threshold", -1.0) or -1.0)
    breadth_dispersion_z_threshold = float(policy.get("breadth_dispersion_z_threshold", 1.0) or 1.0)
    breadth_rel_z_threshold = float(policy.get("breadth_rel_z_threshold", -1.0) or -1.0)

    dollar_stress = _is_ge(getattr(row, "macro_broad_dollar_index_z_252d", np.nan), dollar_z_threshold)
    credit_stress = _is_ge(getattr(row, "macro_high_yield_oas_z_252d", np.nan), credit_oas_z_threshold) or _is_le(
        getattr(row, "hyg_ief_rel_20d_z_252d", np.nan), credit_rel_z_threshold
    )
    rates_stress = _is_ge(getattr(row, "macro_real_yield_10y_z_252d", np.nan), rates_real_yield_z_threshold) or _is_le(
        getattr(row, "tlt_spy_rel_20d_z_252d", np.nan), rates_rel_z_threshold
    )
    vol_stress = (
        _is_le(getattr(row, "vix3m_vix_spread", np.nan), 0.0)
        or _is_ge(getattr(row, "vvix_close_z_252d", np.nan), vol_vvix_z_threshold)
        or _is_le(getattr(row, "skew_close_z_252d", np.nan), vol_skew_z_threshold)
    )
    breadth_stress = _is_ge(
        getattr(row, "qqq_breadth_ret20_dispersion_z_252d", np.nan), breadth_dispersion_z_threshold
    ) or _is_le(
        getattr(row, "qqew_qqq_rel_20d_z_252d", np.nan), breadth_rel_z_threshold
    )
    stress_count = sum(
        1 for value in [dollar_stress, credit_stress, rates_stress, vol_stress, breadth_stress] if bool(value)
    )
    return {
        "dollar_stress": dollar_stress,
        "credit_stress": credit_stress,
        "rates_stress": rates_stress,
        "vol_stress": vol_stress,
        "breadth_stress": breadth_stress,
        "stress_count": stress_count,
    }


def apply_macro_overlay(
    *,
    row: Any,
    allow_long: bool,
    leverage_target: float,
    policy: dict[str, Any],
) -> dict[str, Any]:
    flags = macro_stress_flags(row, policy)
    base = {
        "allow_long": bool(allow_long),
        "leverage_target": float(leverage_target),
        "triggered": False,
        "capped": False,
        "reason": "disabled" if policy.get("kind") == "none" else "inactive",
        **flags,
    }
    if policy.get("kind") == "none":
        return base
    if not allow_long or leverage_target <= 1e-12:
        return base

    rule = str(policy.get("rule") or "")
    flat = False
    capped = False
    reason = "inactive"

    if rule == "dollar":
        if policy.get("kind") == "cash":
            flat = bool(flags["dollar_stress"])
            reason = "dollar_stress"
        elif policy.get("kind") == "tiered":
            capped = bool(flags["dollar_stress"])
            reason = "dollar_cap"
    elif rule == "dollar_and_credit":
        if policy.get("kind") == "cash":
            flat = bool(flags["dollar_stress"] and flags["credit_stress"])
            reason = "dollar_credit_stress"
        elif policy.get("kind") == "tiered":
            capped = bool(flags["dollar_stress"] and flags["credit_stress"])
            reason = "dollar_credit_cap"
    elif rule == "dollar_and_credit_or_rates":
        if policy.get("kind") == "cash":
            flat = bool(flags["dollar_stress"] and (flags["credit_stress"] or flags["rates_stress"]))
            reason = "dollar_credit_or_rates_stress"
        elif policy.get("kind") == "tiered":
            capped = bool(flags["dollar_stress"] and (flags["credit_stress"] or flags["rates_stress"]))
            reason = "dollar_credit_or_rates_cap"
    elif rule == "stress_score":
        stress_count = int(flags["stress_count"])
        if policy.get("kind") == "cash":
            flat = stress_count >= int(policy.get("min_count", 99) or 99)
            reason = "macro_score_cash_gate"
        elif policy.get("kind") == "tiered":
            if stress_count >= int(policy.get("cash_count", 99) or 99):
                flat = True
                reason = "macro_score_cash_gate"
            elif stress_count >= int(policy.get("cap_count", 99) or 99):
                capped = True
                reason = "macro_score_cap"

    if flat:
        return {
            "allow_long": False,
            "leverage_target": 0.0,
            "triggered": True,
            "capped": False,
            "reason": reason,
            **flags,
        }
    if capped:
        multiplier = max(0.0, min(1.0, float(policy.get("multiplier", 1.0) or 1.0)))
        adjusted = float(leverage_target) * multiplier
        return {
            "allow_long": adjusted > 1e-12,
            "leverage_target": adjusted,
            "triggered": True,
            "capped": adjusted + 1e-12 < float(leverage_target),
            "reason": reason,
            **flags,
        }
    return {
        "allow_long": True,
        "leverage_target": float(leverage_target),
        "triggered": False,
        "capped": False,
        "reason": "not_triggered",
        **flags,
    }


def simulate_macro_path(
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    config: dict[str, Any],
    params: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    recent_frame, recent_score_column = _load_risk_predictions(
        config,
        path_key="recent_risk_predictions_csv",
        score_column_key="recent_risk_score_column",
        default_score_column="raw_prob_10d",
    )
    long_cycle_frame, long_cycle_score_column = _load_risk_predictions(
        config,
        path_key="long_cycle_risk_predictions_csv",
        score_column_key="long_cycle_risk_score_column",
        default_score_column="raw_prob_10d",
    )
    risk_context = {
        "enabled": bool(params.get("risk_overlay_enabled", True)),
        "recent_frame": recent_frame,
        "recent_score_column": recent_score_column,
        "long_cycle_frame": long_cycle_frame,
        "long_cycle_score_column": long_cycle_score_column,
    }
    merged = pd.merge_asof(
        bars.sort_values("date"),
        funding.sort_values("date"),
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["funding_rate_value"] = merged["funding_rate_value"].fillna(0.0)
    merged["funding_event_time"] = merged["funding_event_time"].where(merged["funding_event_time"].notna(), pd.NaT)

    capital = float(config["initial_capital"])
    equity_peak = capital
    holding = False
    stop_price = 0.0
    peak_close = 0.0
    prev_leverage = 0.0
    stopped_after_stop = False
    bars_since_stop: int | None = None
    clear_streak = 0
    loss_streak = 0
    gate_until_idx = -1
    current_trade: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    trigger_counts: dict[str, int] = {}

    stop_loss_pct = float(params["stop_loss_pct"])
    per_side_cost = float(config["taker_fee_rate"]) + float(config["slippage_bps"]) / 10000.0
    leverage_profile = {
        "base": float(config["base_leverage"]),
        "offense": float(config.get("offense_leverage", config["base_leverage"])),
        "defense": float(config.get("defense_leverage", config["base_leverage"])),
    }

    for idx, row in enumerate(merged.itertuples(index=False)):
        start_capital = capital
        base_allow = bool(row.allow_long)
        risk_overlay = _risk_overlay_for_bar(config, risk_context, pd.Timestamp(row.date))
        base_leverage = 0.0
        if base_allow or holding:
            if bool(row.high_growth):
                base_leverage = leverage_profile["offense"]
            elif bool(row.defense_state):
                base_leverage = leverage_profile["defense"]
            else:
                base_leverage = leverage_profile["base"]
        leverage_target = base_leverage * float(risk_overlay["leverage_multiplier"]) if base_allow else 0.0
        risk_cash_gate = bool(risk_overlay["cash_gate"])
        effective_allow = bool(base_allow and not risk_cash_gate and leverage_target > 1e-12)
        macro_overlay = apply_macro_overlay(
            row=row,
            allow_long=effective_allow,
            leverage_target=leverage_target,
            policy=policy,
        )
        effective_allow = bool(macro_overlay["allow_long"])
        leverage_target = float(macro_overlay["leverage_target"])
        gate_active = idx < gate_until_idx

        entered = False
        exited = False
        stop_hit = False
        risk_exit = False
        funding_cost = 0.0
        fee_cost = 0.0
        leverage_now = prev_leverage if holding and prev_leverage > 0 else leverage_target

        if effective_allow:
            clear_streak = clear_streak + 1 if not bool(row.defense_state) else 0
        else:
            clear_streak = 0
            if not base_allow:
                stopped_after_stop = False
                bars_since_stop = None
        if bars_since_stop is not None:
            bars_since_stop += 1

        if holding and not effective_allow:
            fee_cost += per_side_cost * leverage_now
            capital *= 1.0 - per_side_cost * leverage_now
            holding = False
            exited = True
            risk_exit = bool(base_allow and not effective_allow)
            if current_trade is not None:
                trade_return = capital / float(current_trade["entry_capital"]) - 1.0
                trades.append(
                    {
                        "entry_date": current_trade["entry_date"],
                        "exit_date": str(pd.Timestamp(row.date)),
                        "trade_return_pct": round(trade_return * 100.0, 2),
                        "exit_reason": "risk_macro_or_signal",
                    }
                )
                loss_streak = loss_streak + 1 if trade_return <= 0.0 else 0
                if int(params.get("loss_streak_stop", 0) or 0) > 0 and loss_streak >= int(params["loss_streak_stop"]):
                    gate_until_idx = trigger_gate(
                        idx=idx,
                        bars=int(params.get("loss_streak_cooldown_bars", 0) or 0),
                        current_gate_until=gate_until_idx,
                    )
                    trigger_counts["loss_streak"] = trigger_counts.get("loss_streak", 0) + 1
                    loss_streak = 0
            current_trade = None
            prev_leverage = 0.0

        can_reenter = reentry_ready(
            rule=str(params["reentry_rule"]),
            stopped_after_stop=stopped_after_stop,
            bars_since_stop=bars_since_stop,
            clear_streak=clear_streak,
            clear_bars=int(params.get("reentry_clear_bars", 0) or 0),
            cooldown_bars=int(params.get("reentry_cooldown_bars", 0) or 0),
            high_growth=bool(row.high_growth),
        )
        can_open = bool(effective_allow and not holding and not gate_active and can_reenter)
        if can_open:
            leverage_now = leverage_target
            fee_cost += per_side_cost * leverage_now
            capital *= 1.0 - per_side_cost * leverage_now
            holding = True
            entered = True
            stopped_after_stop = False
            bars_since_stop = None
            stop_price = float(row.open) * (1.0 - stop_loss_pct / 100.0)
            peak_close = float(row.open)
            current_trade = {"entry_date": str(pd.Timestamp(row.date)), "entry_capital": capital}
            prev_leverage = leverage_now

        if holding:
            leverage_now = leverage_target
            open_price = float(row.open)
            low_price = float(row.low)
            close_price = float(row.close)
            peak_close = max(peak_close, close_price)
            stop_price = max(stop_price, peak_close * (1.0 - stop_loss_pct / 100.0))
            if low_price <= stop_price:
                stop_hit = True
                exit_price = stop_price
                bar_return = exit_price / open_price - 1.0 if open_price > 0 else 0.0
                capital *= 1.0 + leverage_now * bar_return
                fee_cost += per_side_cost * leverage_now
                capital *= 1.0 - per_side_cost * leverage_now
                holding = False
                exited = True
                stopped_after_stop = True
                bars_since_stop = 0
                prev_leverage = 0.0
                if current_trade is not None:
                    trade_return = capital / float(current_trade["entry_capital"]) - 1.0
                    trades.append(
                        {
                            "entry_date": current_trade["entry_date"],
                            "exit_date": str(pd.Timestamp(row.date)),
                            "trade_return_pct": round(trade_return * 100.0, 2),
                            "exit_reason": "trailing_stop",
                        }
                    )
                    loss_streak = loss_streak + 1 if trade_return <= 0.0 else 0
                    if int(params.get("loss_streak_stop", 0) or 0) > 0 and loss_streak >= int(params["loss_streak_stop"]):
                        gate_until_idx = trigger_gate(
                            idx=idx,
                            bars=int(params.get("loss_streak_cooldown_bars", 0) or 0),
                            current_gate_until=gate_until_idx,
                        )
                        trigger_counts["loss_streak"] = trigger_counts.get("loss_streak", 0) + 1
                        loss_streak = 0
                current_trade = None
            else:
                bar_return = close_price / open_price - 1.0 if open_price > 0 else 0.0
                capital *= 1.0 + leverage_now * bar_return
                if is_funding_settlement_bar(row.date, row.funding_event_time):
                    funding_cost = float(row.funding_rate_value) * leverage_now
                    capital *= 1.0 - funding_cost
                prev_leverage = leverage_now

        equity_peak = max(equity_peak, capital)
        equity_dd_pct = (equity_peak - capital) / equity_peak * 100.0 if equity_peak > 0 else 0.0
        equity_dd_stop = float(params.get("equity_dd_stop_pct", 0.0) or 0.0)
        if equity_dd_stop > 0.0 and equity_dd_pct >= equity_dd_stop:
            gate_until_idx = trigger_gate(
                idx=idx,
                bars=int(params.get("equity_dd_cooldown_bars", 0) or 0),
                current_gate_until=gate_until_idx,
            )
            trigger_counts["equity_dd"] = trigger_counts.get("equity_dd", 0) + 1
            equity_peak = capital

        rows.append(
            {
                "date": pd.Timestamp(row.date),
                "session_day": pd.Timestamp(row.date).floor("D"),
                "bar_return": float(capital / start_capital - 1.0 if start_capital > 0 else 0.0),
                "capital": float(capital),
                "holding": bool(holding),
                "allow_long": bool(effective_allow and not gate_active),
                "base_allow_long": bool(base_allow),
                "entered": bool(entered),
                "exited": bool(exited),
                "stop_hit": bool(stop_hit),
                "risk_exit": bool(risk_exit),
                "risk_cash_gate": bool(risk_cash_gate),
                "risk_capped": bool(float(risk_overlay["leverage_multiplier"]) < 0.999 and base_allow),
                "macro_triggered": bool(macro_overlay["triggered"]),
                "macro_capped": bool(macro_overlay["capped"]),
                "macro_cash_gate": bool(str(macro_overlay["reason"]).endswith("cash_gate") or leverage_target <= 1e-12 and bool(macro_overlay["triggered"])),
                "macro_reason": str(macro_overlay["reason"]),
                "dollar_stress": bool(macro_overlay["dollar_stress"]),
                "credit_stress": bool(macro_overlay["credit_stress"]),
                "rates_stress": bool(macro_overlay["rates_stress"]),
                "vol_stress": bool(macro_overlay["vol_stress"]),
                "breadth_stress": bool(macro_overlay["breadth_stress"]),
                "macro_stress_count": int(macro_overlay["stress_count"]),
                "gate_active": bool(gate_active),
                "recent_risk_score": risk_overlay["recent_score"],
                "long_cycle_risk_score": risk_overlay["long_cycle_score"],
                "high_growth": bool(row.high_growth),
                "defense_state": bool(row.defense_state),
                "leverage_now": float(leverage_now if holding or entered else 0.0),
                "funding_cost": float(funding_cost),
                "fee_cost": float(fee_cost),
            }
        )

    path_4h = pd.DataFrame(rows)
    trades_df = pd.DataFrame(trades)
    daily_rows: list[dict[str, Any]] = []
    for day, group in path_4h.groupby("session_day", sort=True):
        active = bool((group["holding"] | group["allow_long"] | group["entered"] | group["exited"]).any())
        avg_leverage = (
            float(group.loc[group["leverage_now"] > 0, "leverage_now"].mean())
            if bool((group["leverage_now"] > 0).any())
            else 0.0
        )
        qqq_score = 0.0
        if active:
            qqq_score = 72.0 + avg_leverage * 4.0
            if bool(group["high_growth"].any()):
                qqq_score += 10.0
            if bool(group["defense_state"].any()):
                qqq_score -= 6.0
        daily_rows.append(
            {
                "date": pd.Timestamp(day),
                "qqq_equity_raw": float(group["capital"].iloc[-1]),
                "qqq_return": float((1.0 + group["bar_return"]).prod() - 1.0),
                "qqq_active": active,
                "qqq_score": round(float(qqq_score), 2),
                "position": "QQQ_USDT_LONG" if active else "CASH",
                "entry_type": "combined_shadow_risk_macro_proxy",
                "risk_cash_day": bool(group["risk_cash_gate"].any()),
                "risk_capped_day": bool(group["risk_capped"].any()),
                "macro_trigger_day": bool(group["macro_triggered"].any()),
                "macro_cash_day": bool(group["macro_cash_gate"].any()),
                "macro_cap_day": bool(group["macro_capped"].any()),
                "shadow_gate_day": bool(group["gate_active"].any()),
                "avg_leverage_when_active": avg_leverage,
                "dollar_stress_day": bool(group["dollar_stress"].any()),
                "credit_stress_day": bool(group["credit_stress"].any()),
                "rates_stress_day": bool(group["rates_stress"].any()),
                "vol_stress_day": bool(group["vol_stress"].any()),
                "breadth_stress_day": bool(group["breadth_stress"].any()),
                "macro_stress_count_max": int(group["macro_stress_count"].max()),
            }
        )
    daily_path = pd.DataFrame(daily_rows)
    summary = {
        "total_return_pct": round((float(path_4h["capital"].iloc[-1]) / float(config["initial_capital"]) - 1.0) * 100.0, 2),
        "max_drawdown_pct": round(max_drawdown_pct(path_4h["capital"]), 2),
        "bars": int(len(path_4h)),
        "days": int(len(daily_path)),
        "invested_bars": int(path_4h["holding"].sum()),
        "invested_days": int(daily_path["qqq_active"].sum()),
        "trades": int(len(trades_df)),
        "stop_hits": int(path_4h["stop_hit"].sum()),
        "risk_cash_bars": int(path_4h["risk_cash_gate"].sum()),
        "risk_capped_bars": int(path_4h["risk_capped"].sum()),
        "macro_trigger_bars": int(path_4h["macro_triggered"].sum()),
        "macro_cap_bars": int(path_4h["macro_capped"].sum()),
        "macro_cash_bars": int(path_4h["macro_cash_gate"].sum()),
        "risk_exit_events": int(path_4h["risk_exit"].sum()),
        "gate_days": int(daily_path["shadow_gate_day"].sum()),
        "trigger_counts": trigger_counts,
        "funding_cost_pct_est": round(float(path_4h["funding_cost"].sum() * 100.0), 2),
        "fee_cost_pct_est": round(float(path_4h["fee_cost"].sum() * 100.0), 2),
        "avg_leverage_when_in": (
            round(float(path_4h.loc[path_4h["holding"], "leverage_now"].mean()), 2)
            if bool(path_4h["holding"].any())
            else 0.0
        ),
        "stress_days": {
            "dollar": int(daily_path["dollar_stress_day"].sum()),
            "credit": int(daily_path["credit_stress_day"].sum()),
            "rates": int(daily_path["rates_stress_day"].sum()),
            "vol": int(daily_path["vol_stress_day"].sum()),
            "breadth": int(daily_path["breadth_stress_day"].sum()),
        },
    }
    return daily_path, path_4h, summary


def run_policy(
    *,
    policy_name: str,
    policy: dict[str, Any],
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    btc_path: pd.DataFrame,
    config: dict[str, Any],
    router_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    qqq_path, path_4h, qqq_summary = simulate_macro_path(
        bars,
        funding,
        config=config,
        params=BASELINE_PARAMS,
        policy=policy,
    )
    full, routed = route_candidate(
        btc_path=btc_path,
        qqq_path=qqq_path,
        initial_capital=float(config["initial_capital"]),
        router_config=router_config,
    )
    return full, {
        "policy_name": policy_name,
        "policy": policy,
        "router": routed["router"],
        "qqq": routed["qqq"],
        "selection": routed["selection"],
        "qqq_path_summary": qqq_summary,
        "path_4h_rows": int(len(path_4h)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit macro proxy overlays on top of current QQQ shadow gate V2 runtime replay.")
    parser.add_argument("--config", default=str(DEFAULT_QQQ_USDT_CONFIG))
    parser.add_argument("--router-config", default=str(DEFAULT_ROUTER_CONFIG))
    parser.add_argument("--nq-data-4h", default=str(DEFAULT_NQ_4H))
    parser.add_argument("--nq-funding", default=str(DEFAULT_NQ_FUNDING))
    parser.add_argument("--real-data-4h", default=str(DEFAULT_REAL_4H))
    parser.add_argument("--real-funding", default=str(DEFAULT_REAL_FUNDING))
    parser.add_argument("--btc-frozen", default=str(DEFAULT_BTC_FROZEN))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2026-05-29")
    parser.add_argument("--real-start-date", default="2026-03-04")
    parser.add_argument("--reference-now", default="2026-05-30T00:00:00+08:00")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    router_config = json.loads(Path(args.router_config).read_text())
    start = pd.Timestamp(args.start_date, tz="UTC")
    end = parse_end_timestamp(args.end_date)
    real_start = pd.Timestamp(args.real_start_date, tz="UTC")
    reference_now = pd.Timestamp(args.reference_now).tz_convert("UTC")
    initial_capital = float(config["initial_capital"])

    daily_macro, missing_flags = load_daily_macro_proxy_context()
    nq_bars = attach_macro_context(load_enriched_bars(config, Path(args.nq_data_4h), start=start, end=end), daily_macro)
    real_bars = attach_macro_context(load_enriched_bars(config, Path(args.real_data_4h), start=real_start, end=end), daily_macro)
    nq_closed = closed_only_bars(nq_bars, reference_now)
    real_closed = closed_only_bars(real_bars, reference_now)
    nq_funding = load_funding(Path(args.nq_funding))
    real_funding = load_funding(Path(args.real_funding))
    btc_full, _ = build_btc_path_from_frozen_artifact(
        frozen_path=Path(args.btc_frozen),
        start=start,
        end=end,
        initial_capital=initial_capital,
    )
    btc_real, _ = build_btc_path_from_frozen_artifact(
        frozen_path=Path(args.btc_frozen),
        start=real_start,
        end=end,
        initial_capital=initial_capital,
    )

    full_paths: dict[str, pd.DataFrame] = {}
    full_results: dict[str, Any] = {}
    for name, policy in MACRO_POLICIES.items():
        path, summary = run_policy(
            policy_name=name,
            policy=policy,
            bars=nq_bars,
            funding=nq_funding,
            btc_path=btc_full,
            config=config,
            router_config=router_config,
        )
        full_paths[name] = path
        full_results[name] = summary | {"annual": annual_metrics(path)}

    closed_only: dict[str, Any] = {}
    for name, policy in MACRO_POLICIES.items():
        _, summary = run_policy(
            policy_name=name,
            policy=policy,
            bars=nq_closed,
            funding=nq_funding,
            btc_path=btc_full,
            config=config,
            router_config=router_config,
        )
        closed_only[name] = summary

    real_overlap: dict[str, Any] = {}
    for name, policy in MACRO_POLICIES.items():
        nq_overlap_path, nq_summary = run_policy(
            policy_name=name,
            policy=policy,
            bars=nq_bars.loc[nq_bars["date"] >= real_start].copy(),
            funding=nq_funding,
            btc_path=btc_real,
            config=config,
            router_config=router_config,
        )
        real_path, real_summary = run_policy(
            policy_name=name,
            policy=policy,
            bars=real_bars,
            funding=real_funding,
            btc_path=btc_real,
            config=config,
            router_config=router_config,
        )
        _, real_closed_summary = run_policy(
            policy_name=name,
            policy=policy,
            bars=real_closed,
            funding=real_funding,
            btc_path=btc_real,
            config=config,
            router_config=router_config,
        )
        real_overlap[name] = {
            "nq_overlap": nq_summary,
            "real": real_summary,
            "real_closed_only": real_closed_summary,
            "consistency": overlap_consistency(nq_overlap_path, real_path),
        }

    baseline = "baseline_v2"
    rolling: dict[str, Any] = {}
    for name in MACRO_POLICIES:
        if name == baseline:
            continue
        rolling[name] = {
            "126d": rolling_compare(full_paths[name], full_paths[baseline], window=126, step=21),
            "252d": rolling_compare(full_paths[name], full_paths[baseline], window=252, step=21),
        }

    report = {
        "mode": "qqq_v2_macro_proxy_overlay_audit",
        "period": {"start": args.start_date, "end": args.end_date, "real_start": args.real_start_date},
        "baseline_profile": BASELINE_PARAMS,
        "router_config": {
            "path": str(Path(args.router_config)),
            "btc_min_route_score": router_config.get("btc_min_route_score"),
            "qqq_min_route_score": router_config.get("qqq_min_route_score"),
            "switch_advantage": router_config.get("switch_advantage"),
            "btc_takeover_advantage": router_config.get("btc_takeover_advantage"),
            "qqq_takeover_advantage": router_config.get("qqq_takeover_advantage"),
        },
        "macro_policies": MACRO_POLICIES,
        "macro_context": {
            "rows": int(len(daily_macro)),
            "start": str(pd.Timestamp(daily_macro["date"].min()).date()) if not daily_macro.empty else None,
            "end": str(pd.Timestamp(daily_macro["date"].max()).date()) if not daily_macro.empty else None,
            "missing_data_flags": missing_flags,
        },
        "bar_closure_audit": {
            "reference_now": str(reference_now),
            "nq": bar_closure_audit(nq_bars, reference_now),
            "nq_closed_only_rows": int(len(nq_closed)),
            "real": bar_closure_audit(real_bars, reference_now),
            "real_closed_only_rows": int(len(real_closed)),
        },
        "full_nq": full_results,
        "closed_only": closed_only,
        "real_overlap": real_overlap,
        "rolling_window_vs_baseline": rolling,
        "method_notes": [
            "Macro proxy overlays are applied after the current runtime risk overlay and before shadow gate entry decisions.",
            "Dollar stress uses broad dollar index 252d z-score >= 1.0.",
            "Credit stress uses HY OAS z-score >= 1.0 or HYG/IEF relative-strength z-score <= -1.0.",
            "Rates stress uses 10y real-yield z-score >= 1.0 or TLT/SPY relative-strength z-score <= -1.0.",
            "Vol stress uses VIX3M-VIX backwardation or VVIX/SKEW stress z-scores.",
            "Breadth stress uses breadth-dispersion z-score >= 1.0 or QQEW/QQQ relative-strength z-score <= -1.0.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(output)
    print(
        json.dumps(
            {
                "baseline": full_results[baseline]["router"],
                "variants": {
                    name: {
                        "router": full_results[name]["router"],
                        "qqq_path_summary": full_results[name]["qqq_path_summary"],
                    }
                    for name in MACRO_POLICIES
                    if name != baseline
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
