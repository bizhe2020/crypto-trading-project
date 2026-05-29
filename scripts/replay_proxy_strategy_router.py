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

from bot.btc_route_scoring import btc_effective_leverage, btc_route_score  # noqa: E402
from bot.okx_executor import ExecutorConfig  # noqa: E402
from bot.qqq_runtime_policy import market_time_window_status  # noqa: E402
from scripts.backtest_config_report import load_dataframe, parse_end_timestamp  # noqa: E402
from scripts.replay_qqq_usdt_10x import load_funding  # noqa: E402
from scripts.scan_qqq_usdt_4h_triggers import attach_daily_state, load_okx_4h, load_signal_path  # noqa: E402
from scripts.scan_qqq_usdt_btc_logic_overlays import enrich_bars  # noqa: E402
from scripts.tqqq_cash_strict_utils import (  # noqa: E402
    load_strict_config,
    load_strict_frame_with_overlay_context,
    run_strict_candidate,
)
from strategy.scalp_robust_v2_core import ScalpRobustEngine, dataframe_to_candles  # noqa: E402


DEFAULT_BTC_CONFIG = ROOT / "config" / "config.paper.high-leverage-structure.json"
DEFAULT_QQQ_PROXY_CONFIG = ROOT / "config" / "config.paper.tqqq-only-strict-recovery-frozen.json"
DEFAULT_QQQ_USDT_CONFIG = ROOT / "config" / "config.paper.qqq-usdt-aggressive-frozen.json"
DEFAULT_BTC_15M = ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"
DEFAULT_BTC_4H = ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"
DEFAULT_BTC_REPORT = ROOT / "var" / "reports" / "backtest_config.paper.high-leverage-structure_2022-01-01_to_2026-05-29.json"
DEFAULT_BTC_FROZEN = ROOT / "var" / "high_leverage_expansion" / "frozen_live_core_20260515.json"
DEFAULT_OUTPUT_JSON = ROOT / "var" / "reports" / "proxy_strategy_router_replay_20260529.json"
DEFAULT_OUTPUT_MD = ROOT / "var" / "reports" / "proxy_strategy_router_replay_20260529.md"
DEFAULT_OUTPUT_CSV = ROOT / "var" / "reports" / "proxy_strategy_router_replay_20260529.csv"


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0, pd.NA) * 100.0
    return float(dd.max(skipna=True) or 0.0)


def equity_from_returns(returns: pd.Series, initial_capital: float) -> pd.Series:
    equity = (1.0 + returns.fillna(0.0)).cumprod() * float(initial_capital)
    if not equity.empty:
        equity.iloc[0] = float(initial_capital) * (1.0 + float(returns.fillna(0.0).iloc[0]))
    return equity


def summarize_path(path: pd.DataFrame, equity_column: str, return_column: str) -> dict[str, Any]:
    if path.empty:
        return {
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "start": None,
            "end": None,
            "days": 0,
            "annual_returns_pct": {},
        }
    start_equity = float(path[equity_column].iloc[0])
    end_equity = float(path[equity_column].iloc[-1])
    frame = path.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year.astype(str)
    yearly: dict[str, float] = {}
    for year, group in frame.groupby("year"):
        y_start = float(group[equity_column].iloc[0])
        y_end = float(group[equity_column].iloc[-1])
        yearly[year] = round((y_end / y_start - 1.0) * 100.0, 2) if y_start > 0 else 0.0
    return {
        "total_return_pct": round((end_equity / start_equity - 1.0) * 100.0, 2) if start_equity > 0 else 0.0,
        "max_drawdown_pct": round(max_drawdown_pct(frame[equity_column]), 2),
        "start": str(pd.Timestamp(frame["date"].iloc[0]).date()),
        "end": str(pd.Timestamp(frame["date"].iloc[-1]).date()),
        "days": int(len(frame)),
        "positive_days_pct": round(float((frame[return_column] > 0).mean() * 100.0), 2),
        "annual_returns_pct": yearly,
    }


def _safe_bool(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def qqq_replay_signal_leverage(signal: dict[str, Any], lev_profile: dict[str, float]) -> float:
    if not _safe_bool(signal.get("allow_long")):
        return 0.0
    if _safe_bool(signal.get("high_growth")):
        return float(lev_profile["offense"])
    if _safe_bool(signal.get("defense_state")):
        return float(lev_profile["defense"])
    return float(lev_profile["base"])


def qqq_replay_risk_on_allowed(config: dict[str, Any], timestamp: Any) -> tuple[bool, dict[str, Any]]:
    status = market_time_window_status(
        enabled=bool(config.get("qqq_rebalance_risk_on_market_hours_only", False)),
        timezone_name=str(config.get("qqq_market_hours_timezone", "America/New_York")),
        start_time=str(config.get("qqq_market_hours_start", "09:30")),
        end_time=str(config.get("qqq_market_hours_end", "16:00")),
        trading_calendar=str(config.get("qqq_market_calendar", "NYSE")),
        now=pd.Timestamp(timestamp),
    )
    return bool(status["open"]), status


def slim_btc_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "total_return_pct",
        "final_capital",
        "sharpe_ratio",
        "max_drawdown_pct",
        "trades",
        "wins",
        "losses",
        "win_rate_pct",
        "avg_return_pct",
        "profit_factor",
        "event_type_counts",
        "exit_counts",
        "windows",
        "decision_counts",
        "combo_mode",
    ]
    return {key: metrics[key] for key in keep if key in metrics}


def score_btc_trade(trade: Any, fallback_leverage: float) -> float:
    payload = {
        "event_type": getattr(trade, "candidate_event_type", None) or "sota_long",
        "direction": getattr(trade, "direction", None),
        "execution_effective_leverage": getattr(trade, "execution_effective_leverage", None),
        "requested_effective_leverage": getattr(trade, "requested_effective_leverage", None),
        "source_effective_leverage": getattr(trade, "source_effective_leverage", None) or fallback_leverage,
        "risk_regime": getattr(trade, "risk_regime", None),
        "regime_label": getattr(trade, "regime_label", None),
        "net_score": getattr(trade, "net_score", None),
        "bull_total": getattr(trade, "bull_total", None),
        "bear_total": getattr(trade, "bear_total", None),
        "conflict": getattr(trade, "conflict", None),
        "feature_recent_fvg_near_entry": getattr(trade, "feature_recent_fvg_near_entry", None),
        "feature_recent_sweep_status": getattr(trade, "feature_recent_sweep_status", None),
        "feature_bearish_structure": getattr(trade, "feature_bearish_structure", None),
        "feature_bullish_structure": getattr(trade, "feature_bullish_structure", None),
    }
    if btc_effective_leverage(payload) <= 0 and fallback_leverage:
        payload["source_effective_leverage"] = fallback_leverage
    return btc_route_score(payload)


def build_btc_proxy_path(
    *,
    config_path: Path,
    data_15m_path: Path,
    data_4h_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_capital: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config_payload = json.loads(config_path.read_text())
    strategy_config = ExecutorConfig.from_dict(config_payload).to_scalp_strategy_config()

    df15 = load_dataframe(data_15m_path, start=start, end=end)
    df4 = load_dataframe(data_4h_path, end=end)
    engine = ScalpRobustEngine.from_candles(
        dataframe_to_candles(df4),
        dataframe_to_candles(df15),
        strategy_config,
    )
    metrics = engine.run_backtest(start_date=start.strftime("%Y-%m-%d"))

    price = df15[["date", "close"]].copy()
    price["day"] = price["date"].dt.floor("D")
    daily_close = price.groupby("day")["close"].last().reset_index()
    daily_close = daily_close.rename(columns={"day": "date", "close": "btc_close"})

    trades = list(engine.trades)
    rows: list[dict[str, Any]] = []
    previous_equity: float | None = None

    for row in daily_close.itertuples(index=False):
        day = pd.Timestamp(row.date)
        day_end = day + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        closed_pnl = 0.0
        active_unrealized = 0.0
        active_scores: list[float] = []
        active_directions: list[str] = []
        active_trades = 0

        for trade in trades:
            entry_time = pd.Timestamp(trade.entry_time, tz="UTC") if pd.Timestamp(trade.entry_time).tzinfo is None else pd.Timestamp(trade.entry_time).tz_convert("UTC")
            exit_time = pd.Timestamp(trade.exit_time, tz="UTC") if pd.Timestamp(trade.exit_time).tzinfo is None else pd.Timestamp(trade.exit_time).tz_convert("UTC")
            if exit_time <= day_end:
                closed_pnl += float(trade.pnl)
                continue
            if entry_time <= day_end < exit_time:
                quantity = float(trade.quantity or 0.0)
                if str(trade.direction) == "BULL":
                    active_unrealized += quantity * (float(row.btc_close) - float(trade.entry_price))
                else:
                    active_unrealized += quantity * (float(trade.entry_price) - float(row.btc_close))
                active_scores.append(score_btc_trade(trade, float(strategy_config.leverage)))
                active_directions.append(str(trade.direction))
                active_trades += 1

        equity = float(initial_capital) + closed_pnl + active_unrealized
        daily_return = 0.0 if previous_equity is None or previous_equity <= 0 else equity / previous_equity - 1.0
        previous_equity = equity
        rows.append(
            {
                "date": day,
                "btc_equity_raw": equity,
                "btc_return": daily_return,
                "btc_active": active_trades > 0,
                "btc_score": max(active_scores) if active_scores else 0.0,
                "btc_direction": active_directions[0] if active_directions else "",
                "btc_active_trades": active_trades,
            }
        )

    path = pd.DataFrame(rows)
    return path, {
        "metrics": metrics,
        "trades": int(len(trades)),
        "date_range": {
            "start": str(start.date()),
            "end": str(end.date()),
        },
    }


def build_btc_path_from_report(
    *,
    report_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_capital: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = json.loads(report_path.read_text())
    overall = payload["overall"]
    total_return = float(overall["total_return_pct"]) / 100.0
    days = pd.date_range(start=start.floor("D"), end=end.floor("D"), freq="D", tz="UTC")
    if len(days) == 0:
        raise RuntimeError("Empty date range for BTC report proxy path.")
    daily_rate = (1.0 + total_return) ** (1.0 / max(len(days), 1)) - 1.0
    rows = []
    equity = float(initial_capital)
    for idx, day in enumerate(days):
        ret = 0.0 if idx == 0 else daily_rate
        equity *= 1.0 + ret
        rows.append(
            {
                "date": pd.Timestamp(day),
                "btc_equity_raw": float(equity),
                "btc_return": float(ret),
                "btc_active": True,
                "btc_score": 70.0,
                "btc_direction": "BULL",
                "btc_active_trades": 1,
            }
        )
    return pd.DataFrame(rows), {
        "metrics": overall,
        "trades": int(overall.get("total_trades", 0)),
        "date_range": payload.get("date_range", {}),
        "proxy_method": "constant_daily_return_from_existing_backtest_report",
    }


def _daily_event_return(event: dict[str, Any]) -> float:
    return_pct = float(event.get("return_pct", 0.0) or 0.0) / 100.0
    return return_pct


def score_frozen_btc_event(event: dict[str, Any]) -> float:
    return btc_route_score(event)


def build_btc_path_from_frozen_artifact(
    *,
    frozen_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_capital: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = json.loads(frozen_path.read_text())
    live_shadow = payload["live_shadow"]
    events = [
        event
        for event in live_shadow.get("events", [])
        if str(event.get("decision", "accepted")) == "accepted" or event.get("decision") is None
    ]
    days = pd.date_range(start=start.floor("D"), end=end.floor("D"), freq="D", tz="UTC")
    event_returns_by_day: dict[pd.Timestamp, list[dict[str, Any]]] = {pd.Timestamp(day): [] for day in days}
    for event in events:
        exit_time = event.get("exit_time") or event.get("entry_time")
        if not exit_time:
            continue
        exit_day = pd.Timestamp(exit_time)
        if exit_day.tzinfo is None:
            exit_day = exit_day.tz_localize("UTC")
        else:
            exit_day = exit_day.tz_convert("UTC")
        exit_day = exit_day.floor("D")
        if exit_day in event_returns_by_day:
            event_returns_by_day[exit_day].append(event)

    rows: list[dict[str, Any]] = []
    equity = float(initial_capital)
    previous_equity = equity
    open_event: dict[str, Any] | None = None
    sorted_events = sorted(events, key=lambda event: str(event.get("entry_time") or ""))
    event_idx = 0

    for day in days:
        day_end = pd.Timestamp(day) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        while event_idx < len(sorted_events):
            entry_time = sorted_events[event_idx].get("entry_time")
            if not entry_time:
                event_idx += 1
                continue
            entry_ts = pd.Timestamp(entry_time)
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize("UTC")
            else:
                entry_ts = entry_ts.tz_convert("UTC")
            if entry_ts <= day_end:
                open_event = sorted_events[event_idx]
                event_idx += 1
                continue
            break

        closed_today = event_returns_by_day.get(pd.Timestamp(day), [])
        route_event = closed_today[-1] if closed_today else open_event
        active_for_route = route_event is not None
        active_score = score_frozen_btc_event(route_event) if route_event is not None else 0.0
        direction = str(route_event.get("direction") or "") if route_event is not None else ""

        day_return = 0.0
        for event in closed_today:
            event_return = _daily_event_return(event)
            equity *= 1.0 + event_return
            day_return = equity / previous_equity - 1.0 if previous_equity > 0 else 0.0
            previous_equity = equity
            if open_event is event:
                open_event = None

        rows.append(
            {
                "date": pd.Timestamp(day),
                "btc_equity_raw": float(equity),
                "btc_return": float(day_return),
                "btc_active": bool(active_for_route),
                "btc_score": round(float(active_score), 2),
                "btc_direction": direction,
                "btc_active_trades": 1 if active_for_route else 0,
            }
        )

    return pd.DataFrame(rows), {
        "metrics": live_shadow,
        "trades": int(live_shadow.get("trades", 0)),
        "date_range": {
            "start": payload.get("metadata", {}).get("data_start"),
            "end": payload.get("metadata", {}).get("data_end"),
        },
        "artifact": str(frozen_path),
        "proxy_method": "frozen_live_shadow_events_closed_on_exit_day",
    }


def score_qqq_row(row: pd.Series) -> float:
    if str(row.get("position", "")) != "TQQQ":
        return 0.0
    score = 80.0
    rel_label = str(row.get("rel_strength_label", "") or "")
    if rel_label == "qqq_strong":
        score += 8.0
    elif rel_label == "qqq_neutral":
        score += 4.0
    entry_type = str(row.get("entry_type", "") or "")
    if entry_type == "recovery_reentry":
        score += 10.0
    elif entry_type == "base":
        score += 4.0
    if bool(row.get("overlay_mode", False)):
        score += 6.0
    vix_label = str(row.get("vix_label", "") or "")
    if vix_label == "vix_low":
        score += 4.0
    elif vix_label == "vix_normal":
        score += 2.0
    return round(score, 2)


def build_qqq_proxy_path(
    *,
    config_path: Path,
    initial_capital: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = load_strict_config(config_path)
    frame = load_strict_frame_with_overlay_context(
        data_root=ROOT / str(config["data_root"]),
        entry_fast_window=int(config["entry_fast_window"]),
        entry_slow_window=int(config["entry_slow_window"]),
    )
    result = run_strict_candidate(
        frame,
        regime_filter=str(config["regime_filter"]),
        max_hold_days=int(config["max_hold_days"]),
        trailing_lookback_days=int(config["trailing_lookback_days"]),
        trailing_drawdown_pct=float(config["trailing_drawdown_pct"]),
        switch_cost_bps=float(config["switch_cost_bps"]),
        initial_capital=float(config["initial_capital"]),
        de_risk_signal_name=str(config.get("de_risk_signal_name", "off")),
        recovery_reentry_rule=str(config.get("recovery_reentry_rule", "off")),
        recovery_reentry_cooldown_days=int(config.get("recovery_reentry_cooldown_days", 0)),
        drawdown_ladder_enabled=bool(config.get("drawdown_ladder_enabled", False)),
        drawdown_ladder_source=str(config.get("drawdown_ladder_source", "tqqq")),
        drawdown_ladder_threshold_pct=float(config.get("drawdown_ladder_threshold_pct", 0.0)),
        drawdown_ladder_peak_lookback_days=int(config.get("drawdown_ladder_peak_lookback_days", 90)),
        drawdown_ladder_scheme=str(config.get("drawdown_ladder_scheme", "two_equal")),
        drawdown_ladder_vix_rule=str(config.get("drawdown_ladder_vix_rule", "all")),
        drawdown_ladder_rebound_exit_pct=float(config.get("drawdown_ladder_rebound_exit_pct", 10.0)),
        drawdown_ladder_max_hold_days=int(config.get("drawdown_ladder_max_hold_days", 15)),
    )
    path = result["path"].copy()
    path["date"] = pd.to_datetime(path["date"], utc=True).dt.floor("D")
    path["qqq_return"] = pd.to_numeric(path["daily_return"], errors="coerce").fillna(0.0)
    path["qqq_active"] = path["position"].eq("TQQQ")
    path["qqq_score"] = path.apply(score_qqq_row, axis=1)
    path["qqq_equity_raw"] = equity_from_returns(path["qqq_return"], initial_capital)
    return (
        path[
            [
                "date",
                "qqq_equity_raw",
                "qqq_return",
                "qqq_active",
                "qqq_score",
                "position",
                "entry_type",
                "vix_label",
                "ixic_trend_label",
                "rel_strength_label",
            ]
        ].copy(),
        {
            "config": config,
            "summary": result["summary"],
        },
    )


def build_qqq_usdt_leveraged_path(
    *,
    config_path: Path,
    initial_capital: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = json.loads(config_path.read_text())
    signal_config, signal_path = load_signal_path(ROOT / str(config["signal_source"]))
    bars = enrich_bars(attach_daily_state(load_okx_4h(ROOT / str(config["data_4h"])), signal_path))
    funding = load_funding(ROOT / str(config["funding_history_path"]))
    merged = pd.merge_asof(
        bars.sort_values("date"),
        funding.sort_values("date"),
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["funding_rate_value"] = merged["funding_rate_value"].fillna(0.0)
    merged = merged.sort_values("date").reset_index(drop=True)
    for column in ("allow_long", "high_growth", "defense_state", "breakout_12"):
        merged[f"signal_{column}"] = merged[column].shift(1)
    merged["signal_date"] = merged["date"].shift(1)

    lev_profile = {
        "base": float(config["base_leverage"]),
        "offense": float(config["offense_leverage"]),
        "defense": float(config["defense_leverage"]),
    }
    stop_loss_pct = float(config["stop_loss_pct"])
    per_side_cost = float(config["taker_fee_rate"]) + float(config["slippage_bps"]) / 10000.0

    capital = float(initial_capital)
    holding = False
    entry_price = 0.0
    stop_price = 0.0
    peak_close = 0.0
    current_leverage = 0.0
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    current_trade: dict[str, Any] | None = None
    risk_on_blocked_entries = 0
    risk_on_blocked_leverage_increases = 0
    risk_off_rebalance_bars = 0
    risk_on_rebalance_bars = 0

    for row in merged.itertuples(index=False):
        start_capital = capital
        signal = {
            "allow_long": row.signal_allow_long,
            "high_growth": row.signal_high_growth,
            "defense_state": row.signal_defense_state,
            "breakout_12": row.signal_breakout_12,
        }
        allow_now = _safe_bool(signal["allow_long"])
        signal_high_growth = _safe_bool(signal["high_growth"])
        signal_defense = _safe_bool(signal["defense_state"])
        target_leverage = qqq_replay_signal_leverage(signal, lev_profile)
        entered = False
        exited = False
        stop_hit = False
        risk_on_blocked = False
        risk_on_open: bool | None = None
        risk_on_reason: str | None = None
        funding_cost = 0.0
        fee_cost = 0.0
        rebalance_fee_cost = 0.0
        leverage_now = current_leverage

        if holding and not allow_now:
            fee_cost += per_side_cost * current_leverage
            capital *= 1.0 - fee_cost
            holding = False
            exited = True
            if current_trade is not None:
                trades.append(
                    {
                        "entry_date": current_trade["entry_date"],
                        "exit_date": str(pd.Timestamp(row.date)),
                        "trade_return_pct": round((capital / float(current_trade["entry_capital"]) - 1.0) * 100.0, 2),
                    }
                )
            current_trade = None
            current_leverage = 0.0
            leverage_now = 0.0

        if allow_now and not holding and target_leverage > 0.0:
            risk_on_open, risk_on_status = qqq_replay_risk_on_allowed(config, row.date)
            risk_on_reason = str(risk_on_status.get("reason"))
            if risk_on_open:
                current_leverage = target_leverage
                leverage_now = current_leverage
                fee_cost += per_side_cost * current_leverage
                capital *= 1.0 - fee_cost
                holding = True
                entered = True
                entry_price = float(row.open)
                stop_price = entry_price * (1.0 - stop_loss_pct / 100.0)
                peak_close = float(row.open)
                current_trade = {"entry_date": str(pd.Timestamp(row.date)), "entry_capital": capital}
            else:
                risk_on_blocked = True
                risk_on_blocked_entries += 1

        if holding and allow_now and target_leverage != current_leverage:
            leverage_delta = abs(target_leverage - current_leverage)
            if target_leverage < current_leverage:
                rebalance_fee_cost += per_side_cost * leverage_delta
                capital *= 1.0 - rebalance_fee_cost
                current_leverage = target_leverage
                risk_off_rebalance_bars += 1
            elif target_leverage > current_leverage:
                risk_on_open, risk_on_status = qqq_replay_risk_on_allowed(config, row.date)
                risk_on_reason = str(risk_on_status.get("reason"))
                if risk_on_open:
                    rebalance_fee_cost += per_side_cost * leverage_delta
                    capital *= 1.0 - rebalance_fee_cost
                    current_leverage = target_leverage
                    risk_on_rebalance_bars += 1
                else:
                    risk_on_blocked = True
                    risk_on_blocked_leverage_increases += 1
            leverage_now = current_leverage

        if holding:
            leverage_now = current_leverage
            open_price = float(row.open)
            low_price = float(row.low)
            close_price = float(row.close)
            peak_close = max(peak_close, close_price)
            stop_price = max(stop_price, peak_close * (1.0 - stop_loss_pct / 100.0))

            if low_price <= stop_price:
                stop_hit = True
                exit_price = stop_price
                bar_ret = exit_price / open_price - 1.0 if open_price > 0 else 0.0
                capital *= 1.0 + leverage_now * bar_ret
                exit_fee_cost = per_side_cost * leverage_now
                fee_cost += exit_fee_cost
                capital *= 1.0 - exit_fee_cost
                holding = False
                exited = True
                if current_trade is not None:
                    trades.append(
                        {
                            "entry_date": current_trade["entry_date"],
                            "exit_date": str(pd.Timestamp(row.date)),
                            "trade_return_pct": round((capital / float(current_trade["entry_capital"]) - 1.0) * 100.0, 2),
                        }
                    )
                current_trade = None
                current_leverage = 0.0
            else:
                bar_ret = close_price / open_price - 1.0 if open_price > 0 else 0.0
                capital *= 1.0 + leverage_now * bar_ret
                funding_cost = max(float(row.funding_rate_value), 0.0) * leverage_now
                capital *= 1.0 - funding_cost

        rows.append(
            {
                "date": pd.Timestamp(row.date),
                "signal_date": pd.Timestamp(row.signal_date) if not pd.isna(row.signal_date) else pd.NaT,
                "session_day": pd.Timestamp(row.date).floor("D"),
                "bar_return": float(capital / start_capital - 1.0 if start_capital > 0 else 0.0),
                "capital": float(capital),
                "holding": bool(holding),
                "allow_long": bool(allow_now),
                "entered": bool(entered),
                "exited": bool(exited),
                "stop_hit": bool(stop_hit),
                "high_growth": bool(signal_high_growth),
                "defense_state": bool(signal_defense),
                "target_leverage": float(target_leverage),
                "leverage_now": float(leverage_now),
                "risk_on_window_open": risk_on_open,
                "risk_on_window_reason": risk_on_reason,
                "risk_on_blocked": bool(risk_on_blocked),
                "funding_cost": float(funding_cost),
                "fee_cost": float(fee_cost),
                "rebalance_fee_cost": float(rebalance_fee_cost),
                "raw_allow_long": bool(row.allow_long),
                "raw_high_growth": bool(row.high_growth),
                "raw_defense_state": bool(row.defense_state),
            }
        )

    path_4h = pd.DataFrame(rows)
    if path_4h.empty:
        raise RuntimeError("Empty QQQ/USDT leveraged replay path.")

    daily_rows: list[dict[str, Any]] = []
    for day, group in path_4h.groupby("session_day", sort=True):
        active = bool((group["holding"] | group["entered"] | group["exited"]).any())
        candidate_active = bool(group["allow_long"].any())
        avg_leverage = float(group.loc[group["leverage_now"] > 0, "leverage_now"].mean()) if bool((group["leverage_now"] > 0).any()) else 0.0
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
                "qqq_candidate_active": candidate_active,
                "qqq_score": round(float(qqq_score), 2),
                "position": "QQQ_USDT_LONG" if active else "CASH",
                "entry_type": "leveraged_contract",
                "vix_label": "",
                "ixic_trend_label": "",
                "rel_strength_label": "",
            }
        )

    daily_path = pd.DataFrame(daily_rows)
    trades_df = pd.DataFrame(trades)
    return daily_path, {
        "config": config,
        "signal_config": signal_config,
        "summary": {
            "total_return_pct": round((float(path_4h.iloc[-1]["capital"]) / float(initial_capital) - 1.0) * 100.0, 2),
            "max_drawdown_pct": round(max_drawdown_pct(path_4h["capital"]), 2),
            "bars": int(len(path_4h)),
            "days": int(len(daily_path)),
            "invested_bars": int(path_4h["holding"].sum()),
            "invested_days": int(daily_path["qqq_active"].sum()),
            "trades": int(len(trades_df)),
            "win_rate_pct": round(float((trades_df["trade_return_pct"] > 0).mean() * 100.0), 2) if not trades_df.empty else 0.0,
            "avg_trade_return_pct": round(float(trades_df["trade_return_pct"].mean()), 2) if not trades_df.empty else 0.0,
            "funding_cost_pct_est": round(float(path_4h["funding_cost"].sum() * 100.0), 2),
            "fee_cost_pct_est": round(float(path_4h["fee_cost"].sum() * 100.0), 2),
            "rebalance_fee_cost_pct_est": round(float(path_4h["rebalance_fee_cost"].sum() * 100.0), 2),
            "avg_leverage_when_in": round(float(path_4h.loc[path_4h["holding"], "leverage_now"].mean()), 2) if bool(path_4h["holding"].any()) else 0.0,
            "execution_policy": {
                "signal_lag_4h_bars": 1,
                "uses_prior_closed_4h_signal": True,
                "risk_off_changes_immediate": True,
                "risk_on_changes_market_window_only": bool(config.get("qqq_rebalance_risk_on_market_hours_only", False)),
                "market_hours_timezone": str(config.get("qqq_market_hours_timezone", "America/New_York")),
                "market_hours_start": str(config.get("qqq_market_hours_start", "09:30")),
                "market_hours_end": str(config.get("qqq_market_hours_end", "16:00")),
                "market_calendar": str(config.get("qqq_market_calendar", "NYSE")),
            },
            "risk_on_blocked_entries": int(risk_on_blocked_entries),
            "risk_on_blocked_leverage_increases": int(risk_on_blocked_leverage_increases),
            "risk_off_rebalance_bars": int(risk_off_rebalance_bars),
            "risk_on_rebalance_bars": int(risk_on_rebalance_bars),
            "start": str(path_4h.iloc[0]["date"]),
            "end": str(path_4h.iloc[-1]["date"]),
        },
    }


def choose_strategy(
    *,
    btc_active: bool,
    btc_score: float,
    qqq_active: bool,
    qqq_score: float,
    current: str,
    btc_min_score: float,
    qqq_min_score: float,
    switch_advantage: float,
) -> tuple[str, str]:
    candidates: list[tuple[str, float]] = []
    if btc_active and btc_score >= btc_min_score:
        candidates.append(("BTC", float(btc_score)))
    if qqq_active and qqq_score >= qqq_min_score:
        candidates.append(("QQQ_PROXY", float(qqq_score)))
    if not candidates:
        return "CASH", "no_eligible_candidates"

    candidates.sort(key=lambda item: item[1], reverse=True)
    best_strategy, best_score = candidates[0]
    current_score = next((score for strategy, score in candidates if strategy == current), None)
    if current_score is not None and current != best_strategy and (best_score - current_score) < switch_advantage:
        return current, "hold_current_hysteresis"
    return best_strategy, "best_route_score"


def run_router(
    merged: pd.DataFrame,
    *,
    initial_capital: float,
    btc_min_score: float,
    qqq_min_score: float,
    switch_advantage: float,
    switch_cost_bps: float,
) -> pd.DataFrame:
    capital = float(initial_capital)
    current = "CASH"
    rows: list[dict[str, Any]] = []

    for row in merged.itertuples(index=False):
        previous = current
        selected, reason = choose_strategy(
            btc_active=bool(row.btc_active),
            btc_score=float(row.btc_score),
            qqq_active=bool(row.qqq_active),
            qqq_score=float(row.qqq_score),
            current=current,
            btc_min_score=btc_min_score,
            qqq_min_score=qqq_min_score,
            switch_advantage=switch_advantage,
        )
        selected_return = 0.0
        if selected == "BTC":
            selected_return = float(row.btc_return)
        elif selected == "QQQ_PROXY":
            selected_return = float(row.qqq_return)
        switched = selected != previous
        if switched and (selected != "CASH" or previous != "CASH"):
            capital *= 1.0 - float(switch_cost_bps) / 10000.0
        capital *= 1.0 + selected_return
        current = selected
        rows.append(
            {
                "date": pd.Timestamp(row.date),
                "selected_strategy": selected,
                "decision_reason": reason,
                "router_return": selected_return,
                "router_equity": capital,
                "switched": switched,
                "btc_score": float(row.btc_score),
                "qqq_score": float(row.qqq_score),
                "btc_active": bool(row.btc_active),
                "qqq_active": bool(row.qqq_active),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Proxy historical route replay: BTC real strategy vs QQQ ETF proxy or QQQ/USDT leveraged leg.")
    parser.add_argument("--btc-config", default=str(DEFAULT_BTC_CONFIG))
    parser.add_argument("--qqq-proxy-config", default=str(DEFAULT_QQQ_PROXY_CONFIG))
    parser.add_argument("--qqq-usdt-config", default=str(DEFAULT_QQQ_USDT_CONFIG))
    parser.add_argument("--qqq-source", choices=["etf_proxy", "usdt_leveraged"], default="etf_proxy")
    parser.add_argument("--btc-15m", default=str(DEFAULT_BTC_15M))
    parser.add_argument("--btc-4h", default=str(DEFAULT_BTC_4H))
    parser.add_argument("--btc-report", default=str(DEFAULT_BTC_REPORT))
    parser.add_argument("--btc-frozen", default=str(DEFAULT_BTC_FROZEN))
    parser.add_argument("--btc-source", choices=["frozen", "report", "engine"], default="frozen")
    parser.add_argument("--recompute-btc-engine", action="store_true")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2026-05-29")
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--btc-min-score", type=float, default=35.0)
    parser.add_argument("--qqq-min-score", type=float, default=60.0)
    parser.add_argument("--switch-advantage", type=float, default=8.0)
    parser.add_argument("--switch-cost-bps", type=float, default=10.0)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    args = parser.parse_args()

    start = pd.Timestamp(args.start_date, tz="UTC")
    end = parse_end_timestamp(args.end_date)
    initial_capital = float(args.initial_capital)

    btc_source = "engine" if bool(args.recompute_btc_engine) else str(args.btc_source)
    if btc_source == "engine":
        btc_path, btc_meta = build_btc_proxy_path(
            config_path=Path(args.btc_config),
            data_15m_path=Path(args.btc_15m),
            data_4h_path=Path(args.btc_4h),
            start=start,
            end=end,
            initial_capital=initial_capital,
        )
    elif btc_source == "frozen":
        btc_path, btc_meta = build_btc_path_from_frozen_artifact(
            frozen_path=Path(args.btc_frozen),
            start=start,
            end=end,
            initial_capital=initial_capital,
        )
    else:
        btc_path, btc_meta = build_btc_path_from_report(
            report_path=Path(args.btc_report),
            start=start,
            end=end,
            initial_capital=initial_capital,
        )
    if str(args.qqq_source) == "usdt_leveraged":
        qqq_path, qqq_meta = build_qqq_usdt_leveraged_path(
            config_path=Path(args.qqq_usdt_config),
            initial_capital=initial_capital,
        )
    else:
        qqq_path, qqq_meta = build_qqq_proxy_path(
            config_path=Path(args.qqq_proxy_config),
            initial_capital=initial_capital,
        )

    merged = pd.merge(btc_path, qqq_path, on="date", how="inner").sort_values("date").reset_index(drop=True)
    if merged.empty:
        raise RuntimeError("No overlapping dates between BTC real path and QQQ proxy path.")

    merged["btc_equity"] = equity_from_returns(merged["btc_return"], initial_capital)
    merged["qqq_equity"] = equity_from_returns(merged["qqq_return"], initial_capital)
    router_path = run_router(
        merged,
        initial_capital=initial_capital,
        btc_min_score=float(args.btc_min_score),
        qqq_min_score=float(args.qqq_min_score),
        switch_advantage=float(args.switch_advantage),
        switch_cost_bps=float(args.switch_cost_bps),
    )
    full_path = pd.concat([merged, router_path.drop(columns=["date"])], axis=1)

    summary = {
        "mode": "proxy_history_router_replay",
        "important_limitations": [
            "QQQ leg uses TQQQ/QQQ ETF proxy unless --qqq-source usdt_leveraged is selected.",
            "QQQ/USDT leveraged mode is limited by short OKX contract history.",
            "QQQ/USDT leveraged mode uses the prior closed 4h signal and gates risk-on leverage increases to configured market hours.",
            "Default BTC path uses the research frozen live-shadow artifact events; use --btc-source report for the smoothed report fallback or --recompute-btc-engine for a slower engine path.",
            "Router is evaluated daily, not intraday.",
        ],
        "date_range": {
            "start": str(pd.Timestamp(full_path["date"].iloc[0]).date()),
            "end": str(pd.Timestamp(full_path["date"].iloc[-1]).date()),
            "days": int(len(full_path)),
        },
        "router_params": {
            "btc_min_score": float(args.btc_min_score),
            "qqq_min_score": float(args.qqq_min_score),
            "switch_advantage": float(args.switch_advantage),
            "switch_cost_bps": float(args.switch_cost_bps),
        },
        "btc_source": btc_source,
        "qqq_source": str(args.qqq_source),
        "btc_only": summarize_path(full_path, "btc_equity", "btc_return"),
        "qqq_proxy_only": summarize_path(full_path, "qqq_equity", "qqq_return"),
        "router": summarize_path(full_path, "router_equity", "router_return"),
        "selection": {
            "btc_days": int((full_path["selected_strategy"] == "BTC").sum()),
            "qqq_proxy_days": int((full_path["selected_strategy"] == "QQQ_PROXY").sum()),
            "cash_days": int((full_path["selected_strategy"] == "CASH").sum()),
            "switches": int(full_path["switched"].sum()),
        },
        "source_summaries": {
            "btc_artifact": btc_meta.get("artifact"),
            "btc_proxy_method": btc_meta.get("proxy_method", btc_source),
            "btc_full_engine_metrics": slim_btc_metrics(dict(btc_meta["metrics"])),
            "btc_trades": btc_meta["trades"],
            "qqq_proxy_summary": qqq_meta["summary"],
        },
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    full_path.to_csv(out_csv, index=False)

    if str(args.qqq_source) == "usdt_leveraged":
        qqq_label = "QQQ/USDT leveraged aggressive frozen"
        qqq_result_label = "QQQ/USDT-only"
        qqq_beats_btc = summary["qqq_proxy_only"]["total_return_pct"] > summary["btc_only"]["total_return_pct"]
        router_beats_btc = summary["router"]["total_return_pct"] > summary["btc_only"]["total_return_pct"]
        router_dd_higher = summary["router"]["max_drawdown_pct"] > summary["btc_only"]["max_drawdown_pct"]
        interpretation = [
            "This corrected replay uses the research frozen BTC strategy and the leveraged QQQ/USDT frozen candidate with prior-closed-4h signal execution.",
            f"On the short OKX QQQ/USDT overlap window, the leveraged QQQ leg {'beats' if qqq_beats_btc else 'does not beat'} BTC frozen.",
            f"Daily routing {'beats' if router_beats_btc else 'does not beat'} BTC frozen on return and has {'higher' if router_dd_higher else 'lower'} drawdown.",
            "The QQQ/USDT sample is still too short to prove long-cycle robustness.",
        ]
    else:
        qqq_label = "TQQQ/QQQ ETF proxy path from the current recovery frozen strategy"
        qqq_result_label = "QQQ-proxy-only"
        interpretation = [
            "This corrected replay uses the research frozen BTC strategy, not the current-branch simplified BTC report.",
            "On the overlapping proxy window, simple daily routing does not beat the BTC frozen leg; it mainly adds switching and drawdown.",
            "It still cannot prove QQQ/USDT contract execution alpha before enough real QQQ/USDT history exists.",
        ]

    md = [
        "# Proxy Strategy Router Replay",
        "",
        "This is a proxy history replay, not a real long-history QQQ/USDT contract backtest.",
        "",
        "## Scope",
        "",
        f"- Period: `{summary['date_range']['start']} -> {summary['date_range']['end']}`",
        f"- Days: `{summary['date_range']['days']}`",
        f"- BTC leg: research frozen live-shadow artifact `{btc_meta.get('artifact', args.btc_frozen)}`",
        f"- QQQ leg: {qqq_label}",
        "- Router cadence: daily",
        "",
        "## Results",
        "",
        f"- BTC-only: `{summary['btc_only']['total_return_pct']}% / DD {summary['btc_only']['max_drawdown_pct']}%`",
        f"- {qqq_result_label}: `{summary['qqq_proxy_only']['total_return_pct']}% / DD {summary['qqq_proxy_only']['max_drawdown_pct']}%`",
        f"- Router: `{summary['router']['total_return_pct']}% / DD {summary['router']['max_drawdown_pct']}%`",
        "",
        "## Selection",
        "",
        f"- BTC days: `{summary['selection']['btc_days']}`",
        f"- QQQ proxy days: `{summary['selection']['qqq_proxy_days']}`",
        f"- Cash days: `{summary['selection']['cash_days']}`",
        f"- Switches: `{summary['selection']['switches']}`",
        "",
        "## Interpretation",
        "",
        *interpretation,
    ]
    out_md.write_text("\n".join(md) + "\n")

    print(out_json)
    print(out_md)
    print(out_csv)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
