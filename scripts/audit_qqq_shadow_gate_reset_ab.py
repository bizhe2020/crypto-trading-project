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

from scripts.audit_qqq_shadow_gate_v2_combined import (  # noqa: E402
    DEFAULT_BTC_FROZEN,
    DEFAULT_NQ_4H,
    DEFAULT_NQ_FUNDING,
    DEFAULT_REAL_4H,
    DEFAULT_REAL_FUNDING,
    DEFAULT_ROUTER_CONFIG,
    annual_metrics,
    closed_only_bars,
    load_enriched_bars,
    overlap_consistency,
    parse_end_timestamp,
    route_candidate,
    summarize_equity,
)
from scripts.replay_proxy_strategy_router import DEFAULT_QQQ_USDT_CONFIG, build_btc_path_from_frozen_artifact  # noqa: E402
from scripts.replay_qqq_usdt_10x import is_funding_settlement_bar, load_funding  # noqa: E402
from scripts.scan_qqq_usdt_4h_triggers import load_signal_path  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_shadow_gate_reset_ab_20220101_20260616.json"


BASE_PARAMS = {
    "name": "shadow_v2_low_dd_no_risk_macro",
    "stop_loss_pct": 4.0,
    "reentry_rule": "clear",
    "reentry_clear_bars": 2,
    "reentry_cooldown_bars": 0,
    "loss_streak_stop": 0,
    "loss_streak_cooldown_bars": 0,
    "equity_dd_stop_pct": 15.0,
    "equity_dd_cooldown_bars": 20,
    "risk_overlay_enabled": False,
}


def load_gate_reset_sessions(
    *,
    config: dict[str, Any],
    reset_signal: str,
    delay_bars: int,
) -> pd.DataFrame:
    overrides = {
        "hard_exit_reset_signal": reset_signal,
        "hard_exit_reset_reentry_delay_bars": int(delay_bars),
    }
    _, signal_path = load_signal_path(ROOT / str(config["signal_source"]), overrides=overrides)
    out = signal_path[["date", "hard_exit_reset_active"]].copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out["session_day"] = out["date"].dt.floor("D")
    out["gate_reset_active"] = out["hard_exit_reset_active"].astype(bool)
    out["gate_reset_event"] = ~out["gate_reset_active"]
    return out[["session_day", "gate_reset_active", "gate_reset_event"]]


def attach_gate_reset(bars: pd.DataFrame, reset_sessions: pd.DataFrame) -> pd.DataFrame:
    reset = reset_sessions.copy().sort_values("session_day")
    left = bars.copy()
    if "daily_signal_timestamp" in left.columns:
        left["session_day"] = pd.to_datetime(left["daily_signal_timestamp"], utc=True).dt.floor("D")
    else:
        left["session_day"] = pd.to_datetime(left["date"], utc=True).dt.floor("D")
    merged = left.merge(reset, on="session_day", how="left")
    merged["gate_reset_active"] = merged["gate_reset_active"].map(lambda value: True if pd.isna(value) else bool(value))
    merged["gate_reset_event"] = merged["gate_reset_event"].map(lambda value: False if pd.isna(value) else bool(value))
    return merged


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


def simulate_gate_reset_path(
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    config: dict[str, Any],
    params: dict[str, Any],
    gate_release_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from bot.qqq_macro_proxy_overlay import apply_macro_proxy_overlay, build_macro_proxy_context, macro_proxy_overlay_for_bar
    from scripts.replay_proxy_strategy_router import _load_risk_predictions, _risk_overlay_for_bar, max_drawdown_pct

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
        "enabled": bool(params.get("risk_overlay_enabled", False)),
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
    stop_price = 0.0
    peak_close = 0.0
    prev_leverage = 0.0
    stopped_after_stop = False
    bars_since_stop: int | None = None
    clear_streak = 0
    loss_streak = 0
    gate_remaining_sessions = 0
    gate_signal_delay_left: int | None = None
    gate_reason: str | None = None
    last_gate_session: pd.Timestamp | None = None
    current_trade: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    trigger_counts: dict[str, int] = {}
    release_counts: dict[str, int] = {}

    stop_loss_pct = float(params["stop_loss_pct"])
    per_side_cost = float(config["taker_fee_rate"]) + float(config["slippage_bps"]) / 10000.0
    leverage_profile = {
        "base": float(config["base_leverage"]),
        "offense": float(config.get("offense_leverage", config["base_leverage"])),
        "defense": float(config.get("defense_leverage", config["base_leverage"])),
    }

    def trigger_gate(bars_count: int, reason: str) -> None:
        nonlocal gate_remaining_sessions, gate_reason, last_gate_session, gate_signal_delay_left
        if bars_count <= 0:
            return
        gate_remaining_sessions = max(gate_remaining_sessions, int(bars_count))
        gate_reason = reason
        gate_signal_delay_left = None
        last_gate_session = None

    for row in merged.itertuples(index=False):
        start_capital = capital
        session = pd.Timestamp(getattr(row, "session_day", row.date)).floor("D")
        if last_gate_session is None or session > last_gate_session:
            if gate_release_mode == "fixed20" and gate_remaining_sessions > 0:
                gate_remaining_sessions -= 1
                if gate_remaining_sessions <= 0:
                    gate_reason = None
                    release_counts["fixed_elapsed"] = release_counts.get("fixed_elapsed", 0) + 1
            elif gate_release_mode == "ixic_mom15_delay2" and gate_remaining_sessions > 0:
                if gate_signal_delay_left is not None:
                    gate_signal_delay_left -= 1
                    if gate_signal_delay_left <= 0:
                        gate_remaining_sessions = 0
                        gate_signal_delay_left = None
                        gate_reason = None
                        release_counts["ixic_mom15_delay2"] = release_counts.get("ixic_mom15_delay2", 0) + 1
                if gate_remaining_sessions > 0 and gate_signal_delay_left is None and bool(getattr(row, "gate_reset_event", False)):
                    gate_signal_delay_left = max(int(params.get("gate_reset_delay_bars", 2) or 0), 0)
                    if gate_signal_delay_left <= 0:
                        gate_remaining_sessions = 0
                        gate_signal_delay_left = None
                        gate_reason = None
                        release_counts["ixic_mom15_delay2"] = release_counts.get("ixic_mom15_delay2", 0) + 1
            last_gate_session = session

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
        gate_active = gate_remaining_sessions > 0

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
                    trigger_gate(int(params.get("loss_streak_cooldown_bars", 0) or 0), "loss_streak")
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
                        trigger_gate(int(params.get("loss_streak_cooldown_bars", 0) or 0), "loss_streak")
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
            trigger_gate(int(params.get("equity_dd_cooldown_bars", 0) or 0), "equity_dd")
            trigger_counts["equity_dd"] = trigger_counts.get("equity_dd", 0) + 1
            equity_peak = capital

        rows.append(
            {
                "date": pd.Timestamp(row.date),
                "session_day": session,
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
                "gate_active": bool(gate_active),
                "gate_remaining_bars": int(gate_remaining_sessions),
                "gate_reason": gate_reason,
                "gate_signal_delay_left": gate_signal_delay_left,
                "gate_reset_active": bool(getattr(row, "gate_reset_active", True)),
                "gate_reset_event": bool(getattr(row, "gate_reset_event", False)),
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
                "entry_type": gate_release_mode,
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
        "cash_days": int((~daily_path["qqq_active"]).sum()),
        "trades": int(len(trades_df)),
        "stop_hits": int(path_4h["stop_hit"].sum()),
        "gate_days": int(daily_path["shadow_gate_day"].sum()),
        "gate_bars": int(path_4h["gate_active"].sum()),
        "trigger_counts": trigger_counts,
        "release_counts": release_counts,
        "funding_cost_pct_est": round(float(path_4h["funding_cost"].sum() * 100.0), 2),
        "fee_cost_pct_est": round(float(path_4h["fee_cost"].sum() * 100.0), 2),
        "avg_leverage_when_in": (
            round(float(path_4h.loc[path_4h["holding"], "leverage_now"].mean()), 2)
            if bool(path_4h["holding"].any())
            else 0.0
        ),
    }
    return daily_path, path_4h, summary


def run_ab_candidate(
    *,
    name: str,
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    btc_path: pd.DataFrame,
    config: dict[str, Any],
    router_config: dict[str, Any],
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    qqq_path, path_4h, qqq_summary = simulate_gate_reset_path(
        bars,
        funding,
        config=config,
        params=params,
        gate_release_mode=name,
    )
    full, routed = route_candidate(
        btc_path=btc_path,
        qqq_path=qqq_path,
        initial_capital=float(config["initial_capital"]),
        router_config=router_config,
    )
    return full, {
        "params": {**params, "gate_release_mode": name},
        "router": routed["router"],
        "qqq": routed["qqq"],
        "selection": routed["selection"],
        "qqq_path_summary": qqq_summary,
        "annual": annual_metrics(full),
        "path_4h_rows": int(len(path_4h)),
    }


def delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "router_return_delta_pct": round(candidate["router"]["total_return_pct"] - baseline["router"]["total_return_pct"], 2),
        "router_dd_delta_pct": round(candidate["router"]["max_drawdown_pct"] - baseline["router"]["max_drawdown_pct"], 2),
        "router_calmar_delta": round((candidate["router"]["calmar_like"] or 0.0) - (baseline["router"]["calmar_like"] or 0.0), 4),
        "qqq_return_delta_pct": round(candidate["qqq"]["total_return_pct"] - baseline["qqq"]["total_return_pct"], 2),
        "qqq_dd_delta_pct": round(candidate["qqq"]["max_drawdown_pct"] - baseline["qqq"]["max_drawdown_pct"], 2),
        "qqq_gate_days_delta": int(candidate["qqq_path_summary"]["gate_days"] - baseline["qqq_path_summary"]["gate_days"]),
        "qqq_cash_days_delta": int(candidate["qqq_path_summary"]["cash_days"] - baseline["qqq_path_summary"]["cash_days"]),
        "qqq_invested_days_delta": int(candidate["qqq_path_summary"]["invested_days"] - baseline["qqq_path_summary"]["invested_days"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AB audit for QQQ shadow gate fixed 20-session cooldown versus IXIC reset release.")
    parser.add_argument("--config", default=str(DEFAULT_QQQ_USDT_CONFIG))
    parser.add_argument("--router-config", default=str(DEFAULT_ROUTER_CONFIG))
    parser.add_argument("--nq-data-4h", default=str(DEFAULT_NQ_4H))
    parser.add_argument("--nq-funding", default=str(DEFAULT_NQ_FUNDING))
    parser.add_argument("--real-data-4h", default=str(DEFAULT_REAL_4H))
    parser.add_argument("--real-funding", default=str(DEFAULT_REAL_FUNDING))
    parser.add_argument("--btc-frozen", default=str(DEFAULT_BTC_FROZEN))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2026-06-12")
    parser.add_argument("--real-start-date", default="2026-03-04")
    parser.add_argument("--reference-now", default="2026-06-16T00:00:00+08:00")
    parser.add_argument("--reset-signal", default="ixic_mom15_ge_-2.5%")
    parser.add_argument("--reset-delay-bars", type=int, default=2)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    router_config = json.loads(Path(args.router_config).read_text())
    start = pd.Timestamp(args.start_date, tz="UTC")
    end = parse_end_timestamp(args.end_date)
    real_start = pd.Timestamp(args.real_start_date, tz="UTC")
    reference_now = pd.Timestamp(args.reference_now).tz_convert("UTC")
    initial_capital = float(config["initial_capital"])
    params = dict(BASE_PARAMS)

    reset_sessions = load_gate_reset_sessions(
        config=config,
        reset_signal=str(args.reset_signal),
        delay_bars=int(args.reset_delay_bars),
    )
    nq_bars = attach_gate_reset(load_enriched_bars(config, Path(args.nq_data_4h), start=start, end=end), reset_sessions)
    real_bars = attach_gate_reset(load_enriched_bars(config, Path(args.real_data_4h), start=real_start, end=end), reset_sessions)
    nq_closed = closed_only_bars(nq_bars, reference_now)
    real_closed = closed_only_bars(real_bars, reference_now)
    nq_funding = load_funding(Path(args.nq_funding))
    real_funding = load_funding(Path(args.real_funding))
    btc_full, btc_meta = build_btc_path_from_frozen_artifact(
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

    modes = ["fixed20", "ixic_mom15_delay2"]
    full_paths: dict[str, pd.DataFrame] = {}
    full_nq: dict[str, Any] = {}
    closed_only: dict[str, Any] = {}
    real_overlap: dict[str, Any] = {}
    for mode in modes:
        path, summary = run_ab_candidate(
            name=mode,
            bars=nq_bars,
            funding=nq_funding,
            btc_path=btc_full,
            config=config,
            router_config=router_config,
            params=params,
        )
        full_paths[mode] = path
        full_nq[mode] = summary
        _, closed_summary = run_ab_candidate(
            name=mode,
            bars=nq_closed,
            funding=nq_funding,
            btc_path=btc_full,
            config=config,
            router_config=router_config,
            params=params,
        )
        closed_only[mode] = closed_summary
        nq_overlap_path, nq_overlap_summary = run_ab_candidate(
            name=mode,
            bars=nq_bars.loc[nq_bars["date"] >= real_start].copy(),
            funding=nq_funding,
            btc_path=btc_real,
            config=config,
            router_config=router_config,
            params=params,
        )
        real_path, real_summary = run_ab_candidate(
            name=mode,
            bars=real_bars,
            funding=real_funding,
            btc_path=btc_real,
            config=config,
            router_config=router_config,
            params=params,
        )
        _, real_closed_summary = run_ab_candidate(
            name=mode,
            bars=real_closed,
            funding=real_funding,
            btc_path=btc_real,
            config=config,
            router_config=router_config,
            params=params,
        )
        real_overlap[mode] = {
            "nq_overlap": nq_overlap_summary,
            "real": real_summary,
            "real_closed_only": real_closed_summary,
            "consistency": overlap_consistency(nq_overlap_path, real_path),
        }

    report = {
        "mode": "qqq_shadow_gate_reset_ab",
        "period": {"start": args.start_date, "end": args.end_date, "real_start": args.real_start_date},
        "reset_rule": {"signal": args.reset_signal, "delay_bars": int(args.reset_delay_bars)},
        "params": params,
        "config": {
            "source_config": str(Path(args.config)),
            "router_config": str(Path(args.router_config)),
            "btc_meta": btc_meta,
        },
        "full_nq": full_nq,
        "closed_only": closed_only,
        "real_overlap": real_overlap,
        "delta_ixic_vs_fixed20": {
            "full_nq": delta(full_nq["ixic_mom15_delay2"], full_nq["fixed20"]),
            "closed_only": delta(closed_only["ixic_mom15_delay2"], closed_only["fixed20"]),
            "real_overlap": delta(real_overlap["ixic_mom15_delay2"]["real"], real_overlap["fixed20"]["real"]),
            "real_closed_only": delta(real_overlap["ixic_mom15_delay2"]["real_closed_only"], real_overlap["fixed20"]["real_closed_only"]),
        },
        "method_notes": [
            "fixed20 keeps the current equity_dd gate as a 20 signal-session cooldown.",
            "ixic_mom15_delay2 keeps the same gate trigger, but releases an active gate once ixic_mom15_ge_-2.5% has been ready for two daily signal sessions.",
            "The daily TQQQ/QQQ base signal is not replaced in this AB; only the shadow_gate release condition changes.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n")
    print(output)
    print(
        json.dumps(
            {
                "delta_ixic_vs_fixed20": report["delta_ixic_vs_fixed20"],
                "full_nq": {
                    key: {
                        "router": value["router"],
                        "qqq": value["qqq"],
                        "qqq_path_summary": value["qqq_path_summary"],
                        "selection": value["selection"],
                    }
                    for key, value in full_nq.items()
                },
                "real_overlap": {
                    key: {
                        "router": value["real"]["router"],
                        "qqq": value["real"]["qqq"],
                        "qqq_path_summary": value["real"]["qqq_path_summary"],
                        "selection": value["real"]["selection"],
                    }
                    for key, value in real_overlap.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
