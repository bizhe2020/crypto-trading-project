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
    summarize_equity,
    trigger_gate,
)
from scripts.replay_proxy_strategy_router import (  # noqa: E402
    _load_risk_predictions,
    _risk_overlay_for_bar,
    build_btc_path_from_frozen_artifact,
    max_drawdown_pct,
)
from scripts.replay_qqq_usdt_10x import is_funding_settlement_bar, load_funding  # noqa: E402
from scripts.tqqq_cash_strict_utils import load_strict_config, load_strict_frame_with_overlay_context  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_v2_heat_overlay_audit_20220101_20260529.json"
BASELINE_PARAMS = dict(CANDIDATES["shadow_v2_low_dd_plus_risk"])

HEAT_POLICIES: dict[str, dict[str, Any]] = {
    "baseline_v2": {
        "kind": "none",
        "label": "Current V2 baseline",
    },
    "vix_lt14_cap50": {
        "kind": "cap",
        "vix_below": 14.0,
        "multiplier": 0.5,
        "label": "VIX<14 cap 50%",
    },
    "vix_lt14_flat": {
        "kind": "cash",
        "vix_below": 14.0,
        "label": "VIX<14 flat",
    },
    "vix_lt14_qstrong_cap50": {
        "kind": "cap",
        "vix_below": 14.0,
        "require_rel_strength": "qqq_strong",
        "multiplier": 0.5,
        "label": "VIX<14 + qqq_strong cap 50% (fear/greed proxy)",
    },
    "vix_lt14_qstrong_flat": {
        "kind": "cash",
        "vix_below": 14.0,
        "require_rel_strength": "qqq_strong",
        "label": "VIX<14 + qqq_strong flat (fear/greed proxy)",
    },
}


def load_daily_heat_context(config: dict[str, Any]) -> pd.DataFrame:
    signal_config = load_strict_config(ROOT / str(config["signal_source"]))
    frame = load_strict_frame_with_overlay_context(
        data_root=ROOT / str(signal_config["data_root"]),
        entry_fast_window=int(signal_config["entry_fast_window"]),
        entry_slow_window=int(signal_config["entry_slow_window"]),
    )
    columns = ["date", "vix_close", "vix_label", "rel_strength_label", "ixic_trend_label"]
    daily = frame[columns].copy()
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    return daily.sort_values("date").reset_index(drop=True)


def attach_heat_context(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge_asof(
        bars.sort_values("date"),
        daily.sort_values("date"),
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.reset_index(drop=True)


def apply_heat_overlay(
    *,
    row: Any,
    allow_long: bool,
    leverage_target: float,
    policy: dict[str, Any],
) -> dict[str, Any]:
    if policy.get("kind") == "none":
        return {
            "allow_long": bool(allow_long),
            "leverage_target": float(leverage_target),
            "triggered": False,
            "capped": False,
            "reason": "disabled",
        }
    if not allow_long or leverage_target <= 1e-12:
        return {
            "allow_long": bool(allow_long),
            "leverage_target": float(leverage_target),
            "triggered": False,
            "capped": False,
            "reason": "inactive",
        }

    vix_below = policy.get("vix_below")
    vix_close = float(getattr(row, "vix_close", 0.0) or 0.0)
    if vix_below is not None and not (vix_close > 0 and vix_close < float(vix_below)):
        return {
            "allow_long": True,
            "leverage_target": float(leverage_target),
            "triggered": False,
            "capped": False,
            "reason": "vix_not_triggered",
        }

    required_rel = str(policy.get("require_rel_strength") or "")
    rel_strength = str(getattr(row, "rel_strength_label", "") or "")
    if required_rel and rel_strength != required_rel:
        return {
            "allow_long": True,
            "leverage_target": float(leverage_target),
            "triggered": False,
            "capped": False,
            "reason": "rel_strength_not_triggered",
        }

    if policy.get("kind") == "cash":
        return {
            "allow_long": False,
            "leverage_target": 0.0,
            "triggered": True,
            "capped": False,
            "reason": "heat_cash_gate",
        }

    multiplier = max(0.0, min(1.0, float(policy.get("multiplier", 1.0) or 1.0)))
    adjusted = float(leverage_target) * multiplier
    return {
        "allow_long": adjusted > 1e-12,
        "leverage_target": adjusted,
        "triggered": True,
        "capped": adjusted + 1e-12 < float(leverage_target),
        "reason": "heat_cap",
    }


def simulate_heat_path(
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
        effective_allow = bool(base_allow and not risk_cash_gate and leverage_target > 1e-12)
        heat_overlay = apply_heat_overlay(
            row=row,
            allow_long=effective_allow,
            leverage_target=leverage_target,
            policy=policy,
        )
        effective_allow = bool(heat_overlay["allow_long"])
        leverage_target = float(heat_overlay["leverage_target"])
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
                        "exit_reason": "risk_or_heat_or_signal",
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
                "risk_capped": bool(float(risk_overlay["leverage_multiplier"]) < 0.999 and base_allow),
                "heat_triggered": bool(heat_overlay["triggered"]),
                "heat_capped": bool(heat_overlay["capped"]),
                "heat_cash_gate": bool(heat_overlay["reason"] == "heat_cash_gate"),
                "heat_reason": str(heat_overlay["reason"]),
                "gate_active": bool(gate_active),
                "recent_risk_score": risk_overlay["recent_score"],
                "long_cycle_risk_score": risk_overlay["long_cycle_score"],
                "high_growth": bool(row.high_growth),
                "defense_state": bool(row.defense_state),
                "leverage_now": float(leverage_now if holding or entered else 0.0),
                "funding_cost": float(funding_cost),
                "fee_cost": float(fee_cost),
                "vix_close": float(getattr(row, "vix_close", 0.0) or 0.0),
                "vix_label": str(getattr(row, "vix_label", "") or ""),
                "rel_strength_label": str(getattr(row, "rel_strength_label", "") or ""),
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
                "entry_type": "combined_shadow_risk_heat",
                "vix_label": str(group["vix_label"].iloc[-1]),
                "ixic_trend_label": "",
                "rel_strength_label": str(group["rel_strength_label"].iloc[-1]),
                "risk_cash_day": bool(group["risk_cash_gate"].any()),
                "risk_capped_day": bool(group["risk_capped"].any()),
                "heat_trigger_day": bool(group["heat_triggered"].any()),
                "heat_cash_day": bool(group["heat_cash_gate"].any()),
                "heat_cap_day": bool(group["heat_capped"].any()),
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
        "heat_trigger_bars": int(path_4h["heat_triggered"].sum()),
        "heat_cap_bars": int(path_4h["heat_capped"].sum()),
        "heat_cash_bars": int(path_4h["heat_cash_gate"].sum()),
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
    qqq_path, path_4h, qqq_summary = simulate_heat_path(
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
    parser = argparse.ArgumentParser(description="Audit VIX heat overlays on top of current QQQ shadow gate V2 runtime replay.")
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

    daily_heat = load_daily_heat_context(config)
    nq_bars = attach_heat_context(load_enriched_bars(config, Path(args.nq_data_4h), start=start, end=end), daily_heat)
    real_bars = attach_heat_context(load_enriched_bars(config, Path(args.real_data_4h), start=real_start, end=end), daily_heat)
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
    for name, policy in HEAT_POLICIES.items():
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
    for name, policy in HEAT_POLICIES.items():
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
    for name, policy in HEAT_POLICIES.items():
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
    for name in HEAT_POLICIES:
        if name == baseline:
            continue
        rolling[name] = {
            "126d": rolling_compare(full_paths[name], full_paths[baseline], window=126, step=21),
            "252d": rolling_compare(full_paths[name], full_paths[baseline], window=252, step=21),
        }

    report = {
        "mode": "qqq_v2_heat_overlay_audit",
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
        "heat_policies": HEAT_POLICIES,
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
            "Heat overlays are applied after the current risk overlay and before shadow gate entry decisions.",
            "VIX<14 rules use the daily vix_close series already present in the strict daily signal frame.",
            "Fear & Greed is approximated by VIX<14 plus rel_strength_label=qqq_strong; this is not CNN's original history.",
            "QQQ PE is not tested here because the repo does not contain an audited historical valuation series.",
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
                    for name in HEAT_POLICIES
                    if name != baseline
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
