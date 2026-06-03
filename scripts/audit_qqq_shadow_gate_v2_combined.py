#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.qqq_macro_proxy_overlay import apply_macro_proxy_overlay, build_macro_proxy_context, macro_proxy_overlay_for_bar  # noqa: E402
from scripts.replay_proxy_strategy_router import (  # noqa: E402
    DEFAULT_BTC_FROZEN,
    DEFAULT_QQQ_USDT_CONFIG,
    _load_risk_predictions,
    _risk_overlay_for_bar,
    build_btc_path_from_frozen_artifact,
    equity_from_returns,
    max_drawdown_pct,
    run_router,
)
from scripts.replay_qqq_usdt_10x import is_funding_settlement_bar, load_funding  # noqa: E402
from scripts.scan_qqq_usdt_4h_triggers import attach_daily_state, load_okx_4h, load_signal_path  # noqa: E402
from scripts.scan_qqq_usdt_btc_logic_overlays import enrich_bars  # noqa: E402


DEFAULT_NQ_4H = ROOT / "data" / "proxy" / "qqq_usdt_nq_continuous" / "QQQ_USDT_USDT-4h-futures-nq-continuous-dailyproxy-long.feather"
DEFAULT_NQ_FUNDING = ROOT / "data" / "proxy" / "qqq_usdt_nq_continuous" / "QQQ_USDT_USDT-8h-funding_rate-zero-nq-continuous-scaled-long.feather"
DEFAULT_REAL_4H = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-4h-futures.feather"
DEFAULT_REAL_FUNDING = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-8h-funding_rate.feather"
DEFAULT_ROUTER_CONFIG = ROOT / "config" / "config.paper.strategy-router.json"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_shadow_gate_v2_combined_audit_20220101_20260529.json"


BASE = {
    "stop_loss_pct": 4.0,
    "reentry_rule": "clear",
    "reentry_clear_bars": 2,
    "reentry_cooldown_bars": 0,
    "loss_streak_stop": 0,
    "loss_streak_cooldown_bars": 0,
    "equity_dd_stop_pct": 0.0,
    "equity_dd_cooldown_bars": 0,
    "risk_overlay_enabled": True,
}

CANDIDATES: dict[str, dict[str, Any]] = {
    "risk_only_stop4_clear2": BASE,
    "current_balanced_plus_risk": {
        **BASE,
        "loss_streak_stop": 2,
        "loss_streak_cooldown_bars": 20,
        "equity_dd_stop_pct": 25.0,
        "equity_dd_cooldown_bars": 10,
    },
    "shadow_v2_low_dd_plus_risk": {
        **BASE,
        "equity_dd_stop_pct": 15.0,
        "equity_dd_cooldown_bars": 20,
    },
}


def parse_end_timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)


def summarize_equity(equity: pd.Series) -> dict[str, Any]:
    if equity.empty:
        return {"total_return_pct": 0.0, "max_drawdown_pct": 0.0, "daily_cvar5_pct": 0.0, "calmar_like": None}
    equity = pd.to_numeric(equity, errors="coerce").ffill()
    returns = equity.pct_change().fillna(0.0)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if float(equity.iloc[0]) > 0 else 0.0
    dd = max_drawdown_pct(equity)
    cvar_count = max(1, int(len(returns) * 0.05))
    cvar5 = float(returns.nsmallest(cvar_count).mean() * 100.0) if len(returns) else 0.0
    return {
        "total_return_pct": round(total * 100.0, 2),
        "max_drawdown_pct": round(dd, 2),
        "daily_cvar5_pct": round(cvar5, 4),
        "calmar_like": round((total * 100.0) / dd, 4) if dd > 0 else None,
    }


def load_enriched_bars(config: dict[str, Any], data_path: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    _, signal_path = load_signal_path(ROOT / str(config["signal_source"]))
    bars = enrich_bars(attach_daily_state(load_okx_4h(data_path), signal_path))
    return bars.loc[(bars["date"] >= start) & (bars["date"] <= end)].copy()


def closed_only_bars(bars: pd.DataFrame, reference_now: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(bars["date"], utc=True).sort_values()
    deltas = dates.diff().dropna()
    if deltas.empty:
        return bars.copy()
    return bars.loc[bars["date"] <= reference_now - deltas.median()].copy()


def bar_closure_audit(bars: pd.DataFrame, reference_now: pd.Timestamp) -> dict[str, Any]:
    dates = pd.to_datetime(bars["date"], utc=True).sort_values()
    deltas = dates.diff().dropna()
    median_delta = deltas.median() if not deltas.empty else pd.Timedelta(0)
    last_open = dates.iloc[-1] if not dates.empty else pd.NaT
    last_close = last_open + median_delta if pd.notna(last_open) else pd.NaT
    return {
        "rows": int(len(dates)),
        "start": str(dates.iloc[0]) if len(dates) else None,
        "end_open": str(last_open) if pd.notna(last_open) else None,
        "median_bar_delta": str(median_delta),
        "estimated_last_close": str(last_close) if pd.notna(last_close) else None,
        "reference_now": str(reference_now),
        "last_bar_closed_by_reference": bool(pd.notna(last_close) and last_close <= reference_now),
    }


def reentry_ready(
    *,
    rule: str,
    stopped_after_stop: bool,
    bars_since_stop: int | None,
    clear_streak: int,
    clear_bars: int,
    cooldown_bars: int,
    high_growth: bool,
) -> bool:
    if not stopped_after_stop:
        return True
    if rule == "signal_reset":
        return False
    if rule == "cooldown":
        return (bars_since_stop or 0) >= int(cooldown_bars)
    if rule == "clear":
        return int(clear_streak) >= int(clear_bars)
    if rule == "high_growth_or_clear":
        return bool(high_growth) or int(clear_streak) >= int(clear_bars)
    raise ValueError(f"Unsupported reentry rule: {rule}")


def trigger_gate(*, idx: int, bars: int, current_gate_until: int) -> int:
    if bars <= 0:
        return current_gate_until
    return max(current_gate_until, idx + int(bars))


def simulate_combined_path(
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    config: dict[str, Any],
    params: dict[str, Any],
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
    macro_context = build_macro_proxy_context(config, merged)

    capital = float(config["initial_capital"])
    equity_peak = capital
    holding = False
    entry_price = 0.0
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
        risk_adjusted_allow = bool(base_allow and not risk_cash_gate and leverage_target > 1e-12)
        macro_overlay = macro_proxy_overlay_for_bar(
            config,
            macro_context,
            pd.Timestamp(row.date),
            signal_timestamp=getattr(row, "daily_signal_timestamp", None),
        )
        macro_adjusted = apply_macro_proxy_overlay(
            allow_long=risk_adjusted_allow,
            leverage_target=leverage_target,
            overlay=macro_overlay,
        )
        effective_allow = bool(macro_adjusted["allow_long"])
        leverage_target = float(macro_adjusted["leverage_target"])
        gate_active = idx < gate_until_idx

        entered = False
        exited = False
        stop_hit = False
        risk_exit = False
        macro_exit = False
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
            risk_exit = bool(base_allow and not risk_adjusted_allow)
            macro_exit = bool(risk_adjusted_allow and not effective_allow)
            if current_trade is not None:
                trade_return = capital / float(current_trade["entry_capital"]) - 1.0
                trades.append(
                    {
                        "entry_date": current_trade["entry_date"],
                        "exit_date": str(pd.Timestamp(row.date)),
                        "trade_return_pct": round(trade_return * 100.0, 2),
                        "exit_reason": "risk_or_signal",
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
            entry_price = float(row.open)
            stop_price = entry_price * (1.0 - stop_loss_pct / 100.0)
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
                "risk_capped": bool(float(risk_overlay["leverage_multiplier"]) < 0.999 and risk_adjusted_allow),
                "macro_triggered": bool(macro_adjusted["triggered"]),
                "macro_capped": bool(macro_adjusted["capped"]),
                "macro_cash_gate": bool(macro_adjusted["cash_gate"]),
                "macro_exit": bool(macro_exit),
                "macro_multiplier": float(macro_overlay["leverage_multiplier"]),
                "macro_score": macro_overlay.get("score"),
                "macro_raw_value": macro_overlay.get("raw_value"),
                "macro_signal_date": macro_overlay.get("signal_date"),
                "macro_macro_signal_date": macro_overlay.get("macro_signal_date"),
                "macro_reason": macro_overlay.get("reason"),
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
                "entry_type": "combined_shadow_risk",
                "vix_label": "",
                "ixic_trend_label": "",
                "rel_strength_label": "",
                "risk_cash_day": bool(group["risk_cash_gate"].any()),
                "risk_capped_day": bool(group["risk_capped"].any()),
                "macro_trigger_day": bool(group["macro_triggered"].any()),
                "macro_cash_day": bool(group["macro_cash_gate"].any()),
                "macro_cap_day": bool(group["macro_capped"].any()),
                "shadow_gate_day": bool(group["gate_active"].any()),
                "avg_leverage_when_active": avg_leverage,
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
        "macro_exit_events": int(path_4h["macro_exit"].sum()),
        "gate_days": int(daily_path["shadow_gate_day"].sum()),
        "trigger_counts": trigger_counts,
        "funding_cost_pct_est": round(float(path_4h["funding_cost"].sum() * 100.0), 2),
        "fee_cost_pct_est": round(float(path_4h["fee_cost"].sum() * 100.0), 2),
        "avg_leverage_when_in": (
            round(float(path_4h.loc[path_4h["holding"], "leverage_now"].mean()), 2)
            if bool(path_4h["holding"].any())
            else 0.0
        ),
    }
    return daily_path, path_4h, summary


def route_candidate(
    *,
    btc_path: pd.DataFrame,
    qqq_path: pd.DataFrame,
    initial_capital: float,
    router_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = pd.merge(btc_path, qqq_path, on="date", how="inner").sort_values("date").reset_index(drop=True)
    merged["btc_equity"] = equity_from_returns(merged["btc_return"], initial_capital)
    merged["qqq_equity"] = equity_from_returns(merged["qqq_return"], initial_capital)
    router_path = run_router(
        merged,
        initial_capital=initial_capital,
        btc_min_score=float(router_config.get("btc_min_route_score", 35.0)),
        qqq_min_score=float(router_config.get("qqq_min_route_score", 60.0)),
        switch_advantage=float(router_config.get("switch_advantage", 8.0)),
        btc_takeover_advantage=(
            float(router_config["btc_takeover_advantage"])
            if router_config.get("btc_takeover_advantage") is not None
            else None
        ),
        qqq_takeover_advantage=(
            float(router_config["qqq_takeover_advantage"])
            if router_config.get("qqq_takeover_advantage") is not None
            else None
        ),
        switch_cost_bps=10.0,
    )
    full = pd.concat([merged, router_path.drop(columns=["date"])], axis=1)
    return full, {
        "router": summarize_equity(full["router_equity"]),
        "qqq": summarize_equity(full["qqq_equity"]),
        "selection": {
            "btc_days": int((full["selected_strategy"] == "BTC").sum()),
            "qqq_proxy_days": int((full["selected_strategy"] == "QQQ_PROXY").sum()),
            "cash_days": int((full["selected_strategy"] == "CASH").sum()),
            "switches": int(full["switched"].sum()),
        },
    }


def run_candidate(
    *,
    params: dict[str, Any],
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    btc_path: pd.DataFrame,
    config: dict[str, Any],
    router_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    qqq_path, path_4h, qqq_summary = simulate_combined_path(bars, funding, config=config, params=params)
    full, routed = route_candidate(
        btc_path=btc_path,
        qqq_path=qqq_path,
        initial_capital=float(config["initial_capital"]),
        router_config=router_config,
    )
    return full, {
        "params": params,
        "router": routed["router"],
        "qqq": routed["qqq"],
        "selection": routed["selection"],
        "qqq_path_summary": qqq_summary,
        "path_4h_rows": int(len(path_4h)),
    }


def rolling_compare(candidate: pd.DataFrame, baseline: pd.DataFrame, *, window: int, step: int) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    n = min(len(candidate), len(baseline))
    for start in range(0, max(0, n - window + 1), step):
        cand_slice = candidate.iloc[start : start + window]
        base_slice = baseline.iloc[start : start + window]
        cand = summarize_equity(cand_slice["router_equity"])
        base = summarize_equity(base_slice["router_equity"])
        windows.append(
            {
                "start": str(pd.Timestamp(cand_slice["date"].iloc[0]).date()),
                "end": str(pd.Timestamp(cand_slice["date"].iloc[-1]).date()),
                "candidate": cand,
                "baseline": base,
                "dd_improved": cand["max_drawdown_pct"] < base["max_drawdown_pct"],
                "cvar_improved": cand["daily_cvar5_pct"] > base["daily_cvar5_pct"],
                "calmar_improved": (cand["calmar_like"] or -999999.0) > (base["calmar_like"] or -999999.0),
                "return_improved": cand["total_return_pct"] > base["total_return_pct"],
            }
        )
    return {
        "window_days": window,
        "step_days": step,
        "count": len(windows),
        "dd_improved_pct": round(sum(item["dd_improved"] for item in windows) / len(windows) * 100.0, 2) if windows else 0.0,
        "cvar_improved_pct": round(sum(item["cvar_improved"] for item in windows) / len(windows) * 100.0, 2) if windows else 0.0,
        "calmar_improved_pct": round(sum(item["calmar_improved"] for item in windows) / len(windows) * 100.0, 2) if windows else 0.0,
        "return_improved_pct": round(sum(item["return_improved"] for item in windows) / len(windows) * 100.0, 2) if windows else 0.0,
        "worst_candidate_dd": max((item["candidate"]["max_drawdown_pct"] for item in windows), default=0.0),
        "windows": windows,
    }


def annual_metrics(path: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    frame = path.copy()
    frame["year"] = pd.to_datetime(frame["date"], utc=True).dt.year.astype(str)
    for year, group in frame.groupby("year"):
        out[year] = {
            "router": summarize_equity(group["router_equity"]),
            "qqq": summarize_equity(group["qqq_equity"]),
        }
    return out


def overlap_consistency(nq_path: pd.DataFrame, real_path: pd.DataFrame) -> dict[str, Any]:
    merged = nq_path.merge(real_path, on="date", suffixes=("_nq", "_real"), how="inner")
    if merged.empty:
        return {"days": 0}
    def col(name: str) -> pd.Series:
        value = merged[name]
        if isinstance(value, pd.DataFrame):
            return value.iloc[:, 0]
        return value

    return {
        "days": int(len(merged)),
        "start": str(pd.Timestamp(merged["date"].iloc[0]).date()),
        "end": str(pd.Timestamp(merged["date"].iloc[-1]).date()),
        "selected_match_pct": round(float((col("selected_strategy_nq") == col("selected_strategy_real")).mean() * 100.0), 2),
        "qqq_active_match_pct": round(float((col("qqq_active_nq") == col("qqq_active_real")).mean() * 100.0), 2),
        "router_return_corr": round(float(col("router_return_nq").corr(col("router_return_real"))), 4),
        "qqq_return_corr": round(float(col("qqq_return_nq").corr(col("qqq_return_real"))), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined risk overlay + shadow gate V2 audit.")
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

    nq_bars = load_enriched_bars(config, Path(args.nq_data_4h), start=start, end=end)
    real_bars = load_enriched_bars(config, Path(args.real_data_4h), start=real_start, end=end)
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
    for name, params in CANDIDATES.items():
        path, summary = run_candidate(
            params=params,
            bars=nq_bars,
            funding=nq_funding,
            btc_path=btc_full,
            config=config,
            router_config=router_config,
        )
        full_paths[name] = path
        full_results[name] = summary | {"annual": annual_metrics(path)}

    closed_only: dict[str, Any] = {}
    for name, params in CANDIDATES.items():
        _, summary = run_candidate(
            params=params,
            bars=nq_closed,
            funding=nq_funding,
            btc_path=btc_full,
            config=config,
            router_config=router_config,
        )
        closed_only[name] = summary

    real_overlap: dict[str, Any] = {}
    for name, params in CANDIDATES.items():
        nq_overlap_path, nq_summary = run_candidate(
            params=params,
            bars=nq_bars.loc[nq_bars["date"] >= real_start].copy(),
            funding=nq_funding,
            btc_path=btc_real,
            config=config,
            router_config=router_config,
        )
        real_path, real_summary = run_candidate(
            params=params,
            bars=real_bars,
            funding=real_funding,
            btc_path=btc_real,
            config=config,
            router_config=router_config,
        )
        _, real_closed_summary = run_candidate(
            params=params,
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

    target = "shadow_v2_low_dd_plus_risk"
    rolling = {
        "vs_risk_only_stop4_clear2": {
            "126d": rolling_compare(full_paths[target], full_paths["risk_only_stop4_clear2"], window=126, step=21),
            "252d": rolling_compare(full_paths[target], full_paths["risk_only_stop4_clear2"], window=252, step=21),
        },
        "vs_current_balanced_plus_risk": {
            "126d": rolling_compare(full_paths[target], full_paths["current_balanced_plus_risk"], window=126, step=21),
            "252d": rolling_compare(full_paths[target], full_paths["current_balanced_plus_risk"], window=252, step=21),
        },
    }

    report = {
        "mode": "combined_risk_overlay_shadow_gate_v2_audit",
        "period": {"start": args.start_date, "end": args.end_date, "real_start": args.real_start_date},
        "candidate_under_review": target,
        "router_config": {
            "path": str(Path(args.router_config)),
            "btc_min_route_score": router_config.get("btc_min_route_score"),
            "qqq_min_route_score": router_config.get("qqq_min_route_score"),
            "switch_advantage": router_config.get("switch_advantage"),
            "btc_takeover_advantage": router_config.get("btc_takeover_advantage"),
            "qqq_takeover_advantage": router_config.get("qqq_takeover_advantage"),
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
        "rolling_window": rolling,
        "method_notes": [
            "Risk overlay is evaluated before shadow gate entry decisions.",
            "Macro proxy overlay is evaluated after the risk overlay and before shadow-gate clear-bar counting.",
            "Risk score dated D is only applied to later bars when risk_overlay_use_previous_signal=true.",
            "Macro proxy signal-day alignment uses daily_signal_timestamp/session-day context, not natural 4h calendar days.",
            "Shadow gate clear-bar counting uses the macro-adjusted allow signal after the risk overlay.",
            "NQ path uses zero/scaled proxy funding; real overlap uses OKX QQQ/USDT funding data.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(output)
    print(
        json.dumps(
            {
                "candidate": target,
                "full_nq": full_results[target]["router"],
                "closed_only": closed_only[target]["router"],
                "real_overlap": real_overlap[target]["real"]["router"],
                "rolling": {
                    "vs_current_126d": {
                        key: value
                        for key, value in rolling["vs_current_balanced_plus_risk"]["126d"].items()
                        if key != "windows"
                    },
                    "vs_current_252d": {
                        key: value
                        for key, value in rolling["vs_current_balanced_plus_risk"]["252d"].items()
                        if key != "windows"
                    },
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
