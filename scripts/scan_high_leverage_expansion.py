#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_config_report import DEFAULT_DATA_15M, DEFAULT_DATA_4H, load_config_payload  # noqa: E402
from scripts.live_readiness_report import (  # noqa: E402
    _high_leverage_failures,
    _high_leverage_trade_diagnostics,
    load_prepared_data,
    max_drawdown_from_capitals,
    run_engine,
    shadow_risk_gate_overlay,
    trade_dataframe,
    trade_return_sharpe,
)


DEFAULT_OUTPUT_DIR = ROOT / "var" / "high_leverage_expansion"
HISTORICAL_MAIN_BEST_RETURN_PCT = 9240.42
HISTORICAL_MAIN_BEST_MAX_DRAWDOWN_PCT = 36.02


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan dynamic 10x high leverage expansion overlays.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.research.10x.json"))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--base-leverage", default="2.0,3.0")
    parser.add_argument("--high-growth-leverage", default="5.0,6.0,7.5")
    parser.add_argument("--tight-stop-leverage", default="6.0,8.0,10.0")
    parser.add_argument("--recovery-leverage", default="1.0,1.5,2.0")
    parser.add_argument("--drawdown-leverage", default="1.0,1.5,2.0")
    parser.add_argument("--unhealthy-leverage", default=None)
    parser.add_argument("--tight-stop-pct", default="1.0,1.25,1.5")
    parser.add_argument("--max-stop-distance-pct", default="2.0")
    parser.add_argument("--high-growth-max-stop-distance-pct", default=None)
    parser.add_argument("--wide-stop-mode", default="high_growth", help="Comma list: high_growth, healthy, all_healthy")
    parser.add_argument("--max-effective-leverage", default="10.0")
    parser.add_argument("--loss-streak-threshold", default="1,2")
    parser.add_argument("--win-streak-threshold", default="2,3")
    parser.add_argument("--drawdown-threshold-pct", default="10.0,15.0,20.0")
    parser.add_argument("--health-lookback-trades", default="0")
    parser.add_argument("--health-min-unit-return-pct", default="0.0")
    parser.add_argument("--health-min-win-rate-pct", default="0.0")
    parser.add_argument("--state-lookback-trades", default="6")
    parser.add_argument("--defense-enter-unit-return-pct", default="0.0")
    parser.add_argument("--defense-enter-win-rate-pct", default="40.0")
    parser.add_argument("--offense-enter-unit-return-pct", default="0.5")
    parser.add_argument("--offense-enter-win-rate-pct", default="50.0")
    parser.add_argument("--reattack-lookback-trades", default="3")
    parser.add_argument("--reattack-unit-return-pct", default="0.0")
    parser.add_argument("--reattack-win-rate-pct", default="50.0")
    parser.add_argument(
        "--reattack-signal-mode",
        default="high_growth_or_tight",
        help="Comma list: any, high_growth, tight_stop, structure, high_growth_or_tight, high_growth_or_structure, high_growth_or_tight_or_structure",
    )
    parser.add_argument(
        "--price-structure-reattack-mode",
        default="none",
        help="Comma list: none, structure, high_growth_or_structure, high_growth_or_tight_or_structure",
    )
    parser.add_argument("--structure-reattack-min-momentum-pct", default="0.0")
    parser.add_argument("--structure-reattack-min-ema-gap-pct", default="0.0")
    parser.add_argument("--structure-reattack-min-adx", default="0.0")
    parser.add_argument("--defense-leverage", default=None)
    parser.add_argument("--defense-max-stop-distance-pct", default=None)
    parser.add_argument("--defense-structure-max-stop-distance-pct", default=None)
    parser.add_argument("--min-liq-buffer-pct", type=float, default=1.2)
    parser.add_argument("--maintenance-margin-pct", type=float, default=0.5)
    parser.add_argument("--max-drawdown-pct", type=float, default=45.0)
    parser.add_argument("--min-2026-return-pct", type=float, default=0.0)
    parser.add_argument("--max-2026-drawdown-pct", type=float, default=30.0)
    parser.add_argument("--min-60d-return-pct", type=float, default=0.0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args()


def unit_trade_return(trade: pd.Series) -> float:
    notional = abs(float(trade.get("notional", 0.0) or 0.0))
    if notional <= 0:
        entry_price = float(trade.get("entry_price", 0.0) or 0.0)
        notional = abs(float(trade.get("quantity", 0.0) or 0.0) * entry_price)
    if notional <= 0:
        return 0.0
    return float(trade.get("pnl", 0.0) or 0.0) / notional


def select_effective_leverage(
    trade: pd.Series,
    diagnostics: dict[str, Any],
    params: dict[str, Any],
    loss_streak: int,
    win_streak: int,
    drawdown_pct: float,
    market_healthy: bool,
    risk_mode: str,
) -> tuple[float, list[str]]:
    leverage = float(params["base_leverage"])
    reasons = ["base"]
    regime_label = str(trade.get("regime_label") or "")
    trail_style = str(trade.get("trail_style") or "")
    stop_distance_pct = float(diagnostics["stop_distance_pct"])

    if regime_label == "high_growth":
        leverage = max(leverage, float(params["high_growth_leverage"]))
        reasons.append("high_growth")
    if stop_distance_pct <= float(params["tight_stop_pct"]):
        leverage = max(leverage, float(params["tight_stop_leverage"]))
        reasons.append("tight_stop")
    if trail_style == "tight":
        leverage = max(leverage, float(params["tight_stop_leverage"]))
        reasons.append("tight_trail")
    if win_streak >= int(params["win_streak_threshold"]):
        leverage = min(float(params["max_effective_leverage"]), leverage * 1.15)
        reasons.append("win_streak_expand")
    if loss_streak >= int(params["loss_streak_threshold"]):
        leverage = min(leverage, float(params["recovery_leverage"]))
        reasons.append("loss_streak_reduce")
    if drawdown_pct >= float(params["drawdown_threshold_pct"]):
        leverage = min(leverage, float(params["drawdown_leverage"]))
        reasons.append("drawdown_reduce")
    if not market_healthy:
        leverage = min(leverage, float(params["unhealthy_leverage"]))
        reasons.append("market_unhealthy_reduce")
    if risk_mode == "defense":
        leverage = min(leverage, float(params["defense_leverage"]))
        reasons.append("state_defense_reduce")

    leverage = max(0.0, min(leverage, float(params["max_effective_leverage"])))
    return leverage, reasons


def recent_signal_stats(signal_returns: list[float], lookback: int) -> dict[str, Any]:
    if lookback <= 0 or len(signal_returns) < lookback:
        return {
            "ready": False,
            "lookback": lookback,
            "recent_unit_return_pct": 0.0,
            "recent_win_rate_pct": 0.0,
        }
    recent = signal_returns[-lookback:]
    return {
        "ready": True,
        "lookback": lookback,
        "recent_unit_return_pct": round(sum(recent) * 100.0, 6),
        "recent_win_rate_pct": round(sum(1 for value in recent if value > 0) / len(recent) * 100.0, 6),
    }


def signal_allows_reattack(trade: pd.Series, diagnostics: dict[str, Any], params: dict[str, Any]) -> bool:
    mode = str(params.get("reattack_signal_mode", "high_growth_or_tight"))
    if mode == "any":
        return True
    regime_label = str(trade.get("regime_label") or "")
    trail_style = str(trade.get("trail_style") or "")
    tight_signal = (
        float(diagnostics["stop_distance_pct"]) <= float(params["tight_stop_pct"])
        or trail_style == "tight"
    )
    high_growth = regime_label == "high_growth"
    if mode == "high_growth":
        return high_growth
    if mode == "tight_stop":
        return tight_signal
    structure = price_structure_qualified(trade, params)
    if mode == "structure":
        return structure
    if mode == "high_growth_or_structure":
        return high_growth or structure
    if mode == "high_growth_or_tight_or_structure":
        return high_growth or tight_signal or structure
    return high_growth or tight_signal


def price_structure_qualified(trade: pd.Series, params: dict[str, Any]) -> bool:
    direction = str(trade.get("direction") or "")
    momentum = float(trade.get("feature_momentum", 0.0) or 0.0)
    ema_gap = float(trade.get("feature_ema_gap", 0.0) or 0.0)
    adx = float(trade.get("feature_adx", 0.0) or 0.0)
    min_momentum = float(params.get("structure_reattack_min_momentum_pct", 0.0) or 0.0) / 100.0
    min_ema_gap = float(params.get("structure_reattack_min_ema_gap_pct", 0.0) or 0.0) / 100.0
    min_adx = float(params.get("structure_reattack_min_adx", 0.0) or 0.0)
    bullish = bool(trade.get("feature_bullish_structure", False))
    bearish = bool(trade.get("feature_bearish_structure", False))
    if adx < min_adx:
        return False
    if direction == "BULL":
        return bullish and momentum >= min_momentum and ema_gap >= min_ema_gap
    if direction == "BEAR":
        return bearish and momentum <= -min_momentum and ema_gap <= -min_ema_gap
    return False


def signal_allows_price_structure_reattack(
    trade: pd.Series,
    diagnostics: dict[str, Any],
    params: dict[str, Any],
) -> bool:
    mode = str(params.get("price_structure_reattack_mode", "none"))
    if mode == "none":
        return False
    structure = price_structure_qualified(trade, params)
    if mode == "structure":
        return structure
    regime_label = str(trade.get("regime_label") or "")
    high_growth = regime_label == "high_growth"
    if mode == "high_growth_or_structure":
        return high_growth or structure
    trail_style = str(trade.get("trail_style") or "")
    tight_signal = (
        float(diagnostics["stop_distance_pct"]) <= float(params["tight_stop_pct"])
        or trail_style == "tight"
    )
    return high_growth or tight_signal or structure


def next_risk_mode(
    trade: pd.Series,
    diagnostics: dict[str, Any],
    current_mode: str,
    signal_returns: list[float],
    loss_streak: int,
    drawdown_pct: float,
    params: dict[str, Any],
) -> tuple[str, list[str], dict[str, Any]]:
    lookback = int(params.get("state_lookback_trades", 0) or 0)
    stats = recent_signal_stats(signal_returns, lookback)
    short_stats = recent_signal_stats(signal_returns, int(params.get("reattack_lookback_trades", 0) or 0))
    stats["reattack"] = short_stats
    reasons: list[str] = []
    if not stats["ready"]:
        return current_mode, reasons, stats

    recent_return = float(stats["recent_unit_return_pct"])
    recent_win_rate = float(stats["recent_win_rate_pct"])
    defense_return = float(params["defense_enter_unit_return_pct"])
    defense_win_rate = float(params["defense_enter_win_rate_pct"])
    offense_return = float(params["offense_enter_unit_return_pct"])
    offense_win_rate = float(params["offense_enter_win_rate_pct"])

    if current_mode == "offense":
        if recent_return <= defense_return:
            reasons.append("low_recent_unit_return")
        if recent_win_rate <= defense_win_rate:
            reasons.append("low_recent_win_rate")
        if loss_streak >= int(params["loss_streak_threshold"]):
            reasons.append("loss_streak")
        if drawdown_pct >= float(params["drawdown_threshold_pct"]):
            reasons.append("drawdown")
        if reasons:
            return "defense", reasons, stats
        return "offense", reasons, stats

    if (
        recent_return >= offense_return
        and recent_win_rate >= offense_win_rate
        and loss_streak < int(params["loss_streak_threshold"])
        and drawdown_pct < float(params["drawdown_threshold_pct"])
    ):
        reasons.append("recovered_recent_signal")
        return "offense", reasons, stats
    if short_stats["ready"]:
        short_return = float(short_stats["recent_unit_return_pct"])
        short_win_rate = float(short_stats["recent_win_rate_pct"])
        if (
            short_return >= float(params["reattack_unit_return_pct"])
            and short_win_rate >= float(params["reattack_win_rate_pct"])
            and loss_streak < int(params["loss_streak_threshold"])
            and drawdown_pct < float(params["drawdown_threshold_pct"])
            and signal_allows_reattack(trade, diagnostics, params)
        ):
            reasons.append("short_window_reattack")
            return "offense", reasons, stats
    if (
        loss_streak < int(params["loss_streak_threshold"])
        and drawdown_pct < float(params["drawdown_threshold_pct"])
        and signal_allows_price_structure_reattack(trade, diagnostics, params)
    ):
        reasons.append("price_structure_reattack")
        return "offense", reasons, stats
    return "defense", reasons, stats


def expansion_overlay(
    trades: pd.DataFrame,
    initial_capital: float,
    params: dict[str, Any],
    include_events: bool = False,
) -> dict[str, Any]:
    base = {
        "mode": "dynamic_high_leverage_expansion_overlay",
        "params": params,
        "total_return_pct": 0.0,
        "final_capital": round(initial_capital, 2),
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "accepted_trades": 0,
        "skipped_trades": 0,
        "failure_counts": {},
        "risk_mode_counts": {"offense": 0, "defense": 0},
        "accepted_risk_mode_counts": {"offense": 0, "defense": 0},
        "mode_switches": 0,
        "mode_switch_events": [],
        "avg_effective_leverage": 0.0,
        "max_effective_leverage_seen": 0.0,
        "worst_liquidation_buffer_pct": None,
        "widest_stop_distance_pct": None,
        "beats_historical_main_return": False,
        "beats_historical_main_return_and_drawdown": False,
        "windows": {},
        "first_skipped_trade": None,
    }
    if trades.empty:
        return base

    capital = initial_capital
    peak = initial_capital
    loss_streak = 0
    win_streak = 0
    accepted = 0
    skipped = 0
    returns: list[float] = []
    signal_health_returns: list[float] = []
    capitals: list[float] = []
    leverages: list[float] = []
    events: list[dict[str, Any]] = []
    buffers: list[float] = []
    stop_distances: list[float] = []
    failure_counts: dict[str, int] = {}
    first_skipped_trade: dict[str, Any] | None = None
    risk_mode = "offense"
    risk_mode_counts = {"offense": 0, "defense": 0}
    accepted_risk_mode_counts = {"offense": 0, "defense": 0}
    mode_switches = 0
    mode_switch_events: list[dict[str, Any]] = []

    for _, trade in trades.sort_values("entry_time").reset_index(drop=True).iterrows():
        diagnostics = _high_leverage_trade_diagnostics(
            trade,
            capital=capital,
            leverage=10.0,
            maintenance_margin_pct=float(params["maintenance_margin_pct"]),
        )
        buffers.append(float(diagnostics["liquidation_buffer_pct"]))
        stop_distances.append(float(diagnostics["stop_distance_pct"]))
        drawdown_pct = (peak - capital) / peak * 100.0 if peak > 0 else 0.0
        market_healthy = is_market_healthy(signal_health_returns, params)
        previous_mode = risk_mode
        risk_mode, mode_reasons, mode_stats = next_risk_mode(
            trade,
            diagnostics,
            risk_mode,
            signal_health_returns,
            loss_streak=loss_streak,
            drawdown_pct=drawdown_pct,
            params=params,
        )
        if risk_mode != previous_mode:
            mode_switches += 1
            if len(mode_switch_events) < 20:
                mode_switch_events.append(
                    {
                        "entry_time": str(trade.get("entry_time")),
                        "from": previous_mode,
                        "to": risk_mode,
                        "reasons": mode_reasons,
                        "stats": mode_stats,
                        "drawdown_pct": round(drawdown_pct, 6),
                        "loss_streak": loss_streak,
                    }
                )
        risk_mode_counts[risk_mode] = risk_mode_counts.get(risk_mode, 0) + 1
        max_stop_distance_pct = dynamic_stop_distance_cap(
            trade=trade,
            drawdown_pct=drawdown_pct,
            loss_streak=loss_streak,
            win_streak=win_streak,
            market_healthy=market_healthy,
            params=params,
        )
        if risk_mode == "defense":
            max_stop_distance_pct = min(max_stop_distance_pct, float(params["defense_max_stop_distance_pct"]))
            if price_structure_qualified(trade, params):
                max_stop_distance_pct = max(
                    max_stop_distance_pct,
                    float(params["defense_structure_max_stop_distance_pct"]),
                )
        signal_unit_return = unit_trade_return(trade)
        failures = _high_leverage_failures(
            diagnostics,
            min_liquidation_buffer_pct=float(params["min_liq_buffer_pct"]),
            max_stop_distance_pct=max_stop_distance_pct,
            max_account_effective_leverage=0.0,
        )
        if failures:
            signal_health_returns.append(signal_unit_return)
            skipped += 1
            for failure in failures:
                failure_counts[failure] = failure_counts.get(failure, 0) + 1
            if first_skipped_trade is None:
                first_skipped_trade = {"failures": failures, "diagnostics": diagnostics}
            continue

        effective_leverage, reasons = select_effective_leverage(
            trade,
            diagnostics,
            params,
            loss_streak=loss_streak,
            win_streak=win_streak,
            drawdown_pct=drawdown_pct,
            market_healthy=market_healthy,
            risk_mode=risk_mode,
        )
        trade_return = signal_unit_return * effective_leverage
        signal_health_returns.append(signal_unit_return)
        capital_before = capital
        capital = max(0.0, capital * (1.0 + trade_return))
        peak = max(peak, capital)
        accepted += 1
        accepted_risk_mode_counts[risk_mode] = accepted_risk_mode_counts.get(risk_mode, 0) + 1
        returns.append(trade_return)
        capitals.append(capital)
        leverages.append(effective_leverage)
        events.append(
            {
                "entry_time": str(trade.get("entry_time")),
                "exit_time": str(trade.get("exit_time")),
                "return": trade_return,
                "capital": capital,
                "effective_leverage": effective_leverage,
                "regime_label": str(trade.get("regime_label") or ""),
                "trail_style": str(trade.get("trail_style") or ""),
                "direction": str(trade.get("direction") or ""),
                "feature_adx": float(trade.get("feature_adx", 0.0) or 0.0),
                "feature_momentum": float(trade.get("feature_momentum", 0.0) or 0.0),
                "feature_ema_gap": float(trade.get("feature_ema_gap", 0.0) or 0.0),
                "feature_bullish_structure": bool(trade.get("feature_bullish_structure", False)),
                "feature_bearish_structure": bool(trade.get("feature_bearish_structure", False)),
                "stop_distance_pct": float(diagnostics["stop_distance_pct"]),
                "stop_distance_cap_pct": max_stop_distance_pct,
                "reasons": reasons,
                "market_healthy": market_healthy,
                "risk_mode": risk_mode,
                "risk_mode_stats": mode_stats,
            }
        )

        if capital > capital_before:
            win_streak += 1
            loss_streak = 0
        else:
            loss_streak += 1
            win_streak = 0

    total_return_pct = (capital - initial_capital) / initial_capital * 100.0 if initial_capital > 0 else 0.0
    max_drawdown_pct = max_drawdown_from_capitals(capitals, initial_capital)
    base.update(
        {
            "total_return_pct": round(total_return_pct, 2),
            "final_capital": round(capital, 2),
            "sharpe_ratio": round(trade_return_sharpe(returns), 3),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "accepted_trades": accepted,
            "skipped_trades": skipped,
            "failure_counts": failure_counts,
            "risk_mode_counts": risk_mode_counts,
            "accepted_risk_mode_counts": accepted_risk_mode_counts,
            "mode_switches": mode_switches,
            "mode_switch_events": mode_switch_events,
            "avg_effective_leverage": round(sum(leverages) / len(leverages), 6) if leverages else 0.0,
            "max_effective_leverage_seen": round(max(leverages), 6) if leverages else 0.0,
            "worst_liquidation_buffer_pct": round(min(buffers), 6) if buffers else None,
            "widest_stop_distance_pct": round(max(stop_distances), 6) if stop_distances else None,
            "beats_historical_main_return": total_return_pct > HISTORICAL_MAIN_BEST_RETURN_PCT,
            "beats_historical_main_return_and_drawdown": (
                total_return_pct > HISTORICAL_MAIN_BEST_RETURN_PCT
                and max_drawdown_pct <= HISTORICAL_MAIN_BEST_MAX_DRAWDOWN_PCT
            ),
            "windows": window_metrics_from_events(events, initial_capital),
            "first_skipped_trade": first_skipped_trade,
        }
    )
    if include_events:
        base["events"] = events
    return base


def dynamic_stop_distance_cap(
    trade: pd.Series,
    drawdown_pct: float,
    loss_streak: int,
    win_streak: int,
    market_healthy: bool,
    params: dict[str, Any],
) -> float:
    base_cap = float(params["max_stop_distance_pct"])
    high_growth_cap = float(params.get("high_growth_max_stop_distance_pct", base_cap))
    regime_label = str(trade.get("regime_label") or "")
    mode = str(params.get("wide_stop_mode", "high_growth"))
    healthy = (
        market_healthy
        and drawdown_pct < float(params["drawdown_threshold_pct"])
        and loss_streak < int(params["loss_streak_threshold"])
    )
    if mode == "all_healthy" and healthy:
        return max(base_cap, high_growth_cap)
    if mode == "healthy" and healthy and (
        regime_label in {"high_growth", "normal"} or win_streak >= int(params["win_streak_threshold"])
    ):
        return max(base_cap, high_growth_cap)
    if mode == "high_growth" and healthy and regime_label == "high_growth":
        return max(base_cap, high_growth_cap)
    return base_cap


def is_market_healthy(signal_returns: list[float], params: dict[str, Any]) -> bool:
    lookback = int(params.get("health_lookback_trades", 0) or 0)
    if lookback <= 0 or len(signal_returns) < lookback:
        return True
    recent = signal_returns[-lookback:]
    min_return = float(params.get("health_min_unit_return_pct", 0.0) or 0.0) / 100.0
    min_win_rate = float(params.get("health_min_win_rate_pct", 0.0) or 0.0) / 100.0
    recent_return = sum(recent)
    recent_win_rate = sum(1 for value in recent if value > 0) / len(recent)
    return recent_return >= min_return and recent_win_rate >= min_win_rate


def enrich_trades_with_regime_features(trades: pd.DataFrame, prepared: Any) -> pd.DataFrame:
    if trades.empty or "entry_idx" not in trades.columns:
        return trades
    out = trades.copy()
    feature_columns = {
        "feature_adx": 0.0,
        "feature_momentum": 0.0,
        "feature_ema_gap": 0.0,
        "feature_bullish_structure": False,
        "feature_bearish_structure": False,
    }
    for column, default in feature_columns.items():
        if column not in out.columns:
            out[column] = default
    mapping = getattr(prepared, "mapping", [])
    features_by_idx = getattr(prepared, "regime_features", {})
    for row_idx, entry_idx in out["entry_idx"].items():
        if pd.isna(entry_idx):
            continue
        c15_idx = int(entry_idx)
        if c15_idx < 0 or c15_idx >= len(mapping):
            continue
        features = features_by_idx.get(mapping[c15_idx], {})
        if not features:
            continue
        out.at[row_idx, "feature_adx"] = float(features.get("adx", 0.0) or 0.0)
        out.at[row_idx, "feature_momentum"] = float(features.get("momentum", 0.0) or 0.0)
        out.at[row_idx, "feature_ema_gap"] = float(features.get("ema_gap", 0.0) or 0.0)
        out.at[row_idx, "feature_bullish_structure"] = bool(features.get("bullish_structure", False))
        out.at[row_idx, "feature_bearish_structure"] = bool(features.get("bearish_structure", False))
    return out


def replay_window(events: list[dict[str, Any]], initial_capital: float, start: pd.Timestamp) -> dict[str, Any]:
    selected = [
        event
        for event in events
        if pd.Timestamp(event["entry_time"]).tz_convert("UTC") >= start
    ]
    capital = initial_capital
    capitals: list[float] = []
    returns: list[float] = []
    leverages: list[float] = []
    mode_counts = {"offense": 0, "defense": 0}
    for event in selected:
        trade_return = float(event["return"])
        capital = max(0.0, capital * (1.0 + trade_return))
        capitals.append(capital)
        returns.append(trade_return)
        leverages.append(float(event["effective_leverage"]))
        mode = str(event.get("risk_mode", ""))
        if mode:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
    return {
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100.0, 2),
        "final_capital": round(capital, 2),
        "sharpe_ratio": round(trade_return_sharpe(returns), 3),
        "max_drawdown_pct": round(max_drawdown_from_capitals(capitals, initial_capital), 2),
        "trades": len(selected),
        "avg_effective_leverage": round(sum(leverages) / len(leverages), 6) if leverages else 0.0,
        "max_effective_leverage": round(max(leverages), 6) if leverages else 0.0,
        "risk_mode_counts": mode_counts,
    }


def window_metrics_from_events(events: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    if not events:
        return {}
    exits = [pd.Timestamp(event["exit_time"]).tz_convert("UTC") for event in events]
    end = max(exits)
    starts = {
        "current_year": pd.Timestamp(f"{end.year}-01-01", tz="UTC"),
        "last_60d": end - pd.Timedelta(days=60),
        "last_30d": end - pd.Timedelta(days=30),
    }
    return {
        name: replay_window(events, initial_capital, start)
        for name, start in starts.items()
    }


def candidate_params(args: argparse.Namespace) -> list[dict[str, Any]]:
    combos = itertools.product(
        parse_float_list(args.base_leverage),
        parse_float_list(args.high_growth_leverage),
        parse_float_list(args.tight_stop_leverage),
        parse_float_list(args.recovery_leverage),
        parse_float_list(args.drawdown_leverage),
        parse_float_list(args.unhealthy_leverage or args.drawdown_leverage),
        parse_float_list(args.tight_stop_pct),
        parse_float_list(args.max_stop_distance_pct),
        parse_float_list(args.high_growth_max_stop_distance_pct or args.max_stop_distance_pct),
        parse_str_list(args.wide_stop_mode),
        parse_float_list(args.max_effective_leverage),
        parse_int_list(args.loss_streak_threshold),
        parse_int_list(args.win_streak_threshold),
        parse_float_list(args.drawdown_threshold_pct),
        parse_int_list(args.health_lookback_trades),
        parse_float_list(args.health_min_unit_return_pct),
        parse_float_list(args.health_min_win_rate_pct),
        parse_int_list(args.state_lookback_trades),
        parse_float_list(args.defense_enter_unit_return_pct),
        parse_float_list(args.defense_enter_win_rate_pct),
        parse_float_list(args.offense_enter_unit_return_pct),
        parse_float_list(args.offense_enter_win_rate_pct),
        parse_int_list(args.reattack_lookback_trades),
        parse_float_list(args.reattack_unit_return_pct),
        parse_float_list(args.reattack_win_rate_pct),
        parse_str_list(args.reattack_signal_mode),
        parse_str_list(args.price_structure_reattack_mode),
        parse_float_list(args.structure_reattack_min_momentum_pct),
        parse_float_list(args.structure_reattack_min_ema_gap_pct),
        parse_float_list(args.structure_reattack_min_adx),
        parse_float_list(args.defense_leverage or args.unhealthy_leverage or args.drawdown_leverage),
        parse_float_list(args.defense_max_stop_distance_pct or args.max_stop_distance_pct),
        parse_float_list(args.defense_structure_max_stop_distance_pct or args.defense_max_stop_distance_pct or args.max_stop_distance_pct),
    )
    params: list[dict[str, Any]] = []
    for (
        base_leverage,
        high_growth_leverage,
        tight_stop_leverage,
        recovery_leverage,
        drawdown_leverage,
        unhealthy_leverage,
        tight_stop_pct,
        max_stop_distance_pct,
        high_growth_max_stop_distance_pct,
        wide_stop_mode,
        max_effective_leverage,
        loss_streak_threshold,
        win_streak_threshold,
        drawdown_threshold_pct,
        health_lookback_trades,
        health_min_unit_return_pct,
        health_min_win_rate_pct,
        state_lookback_trades,
        defense_enter_unit_return_pct,
        defense_enter_win_rate_pct,
        offense_enter_unit_return_pct,
        offense_enter_win_rate_pct,
        reattack_lookback_trades,
        reattack_unit_return_pct,
        reattack_win_rate_pct,
        reattack_signal_mode,
        price_structure_reattack_mode,
        structure_reattack_min_momentum_pct,
        structure_reattack_min_ema_gap_pct,
        structure_reattack_min_adx,
        defense_leverage,
        defense_max_stop_distance_pct,
        defense_structure_max_stop_distance_pct,
    ) in combos:
        if base_leverage > max_effective_leverage:
            continue
        if defense_leverage > max_effective_leverage:
            continue
        if defense_enter_win_rate_pct > offense_enter_win_rate_pct:
            continue
        if defense_enter_unit_return_pct > offense_enter_unit_return_pct:
            continue
        params.append(
            {
                "base_leverage": base_leverage,
                "high_growth_leverage": min(high_growth_leverage, max_effective_leverage),
                "tight_stop_leverage": min(tight_stop_leverage, max_effective_leverage),
                "recovery_leverage": min(recovery_leverage, max_effective_leverage),
                "drawdown_leverage": min(drawdown_leverage, max_effective_leverage),
                "unhealthy_leverage": min(unhealthy_leverage, max_effective_leverage),
                "tight_stop_pct": tight_stop_pct,
                "max_stop_distance_pct": max_stop_distance_pct,
                "high_growth_max_stop_distance_pct": max(high_growth_max_stop_distance_pct, max_stop_distance_pct),
                "wide_stop_mode": wide_stop_mode,
                "max_effective_leverage": max_effective_leverage,
                "loss_streak_threshold": loss_streak_threshold,
                "win_streak_threshold": win_streak_threshold,
                "drawdown_threshold_pct": drawdown_threshold_pct,
                "health_lookback_trades": health_lookback_trades,
                "health_min_unit_return_pct": health_min_unit_return_pct,
                "health_min_win_rate_pct": health_min_win_rate_pct,
                "state_lookback_trades": state_lookback_trades,
                "defense_enter_unit_return_pct": defense_enter_unit_return_pct,
                "defense_enter_win_rate_pct": defense_enter_win_rate_pct,
                "offense_enter_unit_return_pct": offense_enter_unit_return_pct,
                "offense_enter_win_rate_pct": offense_enter_win_rate_pct,
                "reattack_lookback_trades": reattack_lookback_trades,
                "reattack_unit_return_pct": reattack_unit_return_pct,
                "reattack_win_rate_pct": reattack_win_rate_pct,
                "reattack_signal_mode": reattack_signal_mode,
                "price_structure_reattack_mode": price_structure_reattack_mode,
                "structure_reattack_min_momentum_pct": structure_reattack_min_momentum_pct,
                "structure_reattack_min_ema_gap_pct": structure_reattack_min_ema_gap_pct,
                "structure_reattack_min_adx": structure_reattack_min_adx,
                "defense_leverage": min(defense_leverage, max_effective_leverage),
                "defense_max_stop_distance_pct": min(defense_max_stop_distance_pct, max_stop_distance_pct),
                "defense_structure_max_stop_distance_pct": max(
                    min(defense_structure_max_stop_distance_pct, high_growth_max_stop_distance_pct),
                    min(defense_max_stop_distance_pct, max_stop_distance_pct),
                ),
                "min_liq_buffer_pct": args.min_liq_buffer_pct,
                "maintenance_margin_pct": args.maintenance_margin_pct,
            }
        )
    return params


def score_result(
    result: dict[str, Any],
    max_drawdown_pct: float,
    min_2026_return_pct: float,
    max_2026_drawdown_pct: float,
    min_60d_return_pct: float,
) -> float:
    total_return = float(result["total_return_pct"])
    drawdown = float(result["max_drawdown_pct"])
    sharpe = float(result["sharpe_ratio"])
    current_year = result.get("windows", {}).get("current_year", {})
    recent_60d = result.get("windows", {}).get("last_60d", {})
    year_return = float(current_year.get("total_return_pct", 0.0) or 0.0)
    year_drawdown = float(current_year.get("max_drawdown_pct", 0.0) or 0.0)
    recent_60d_return = float(recent_60d.get("total_return_pct", 0.0) or 0.0)
    penalty = max(0.0, drawdown - max_drawdown_pct) * 500.0
    penalty += max(0.0, min_2026_return_pct - year_return) * 400.0
    penalty += max(0.0, year_drawdown - max_2026_drawdown_pct) * 250.0
    penalty += max(0.0, min_60d_return_pct - recent_60d_return) * 250.0
    return (
        total_return
        + sharpe * 100.0
        + year_return * 120.0
        + recent_60d_return * 80.0
        - drawdown * 20.0
        - year_drawdown * 30.0
        - penalty
    )


def output_path_for(output_dir: Path, start_date: str, end: pd.Timestamp) -> Path:
    return output_dir / f"dynamic_expansion_scan_{start_date}_to_{end.strftime('%Y-%m-%d')}.json"


def main() -> None:
    args = parse_args()
    payload = load_config_payload(Path(args.config))
    prepared = load_prepared_data(
        Path(args.data_15m),
        Path(args.data_4h),
        pd.Timestamp(args.start_date, tz="UTC"),
        payload.get("regime_switcher_thresholds"),
    )
    metrics, engine = run_engine(payload, prepared, args.start_date)
    trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
    initial_capital = float(metrics.get("initial_capital", 1000.0))
    current_shadow = shadow_risk_gate_overlay(
        trades=trades,
        initial_capital=initial_capital,
        daily_loss_stop_pct=float(payload.get("shadow_daily_loss_stop_pct", 6.0) or 0.0),
        equity_drawdown_stop_pct=float(payload.get("shadow_equity_drawdown_stop_pct", 20.0) or 0.0),
        consecutive_loss_stop=int(payload.get("shadow_consecutive_loss_stop", 4) or 0),
        equity_drawdown_cooldown_days=int(payload.get("shadow_equity_drawdown_cooldown_days", 7) or 0),
    )
    results = []
    params_list = candidate_params(args)
    for idx, params in enumerate(params_list, start=1):
        if idx == 1 or idx % 100 == 0:
            print(f"[{idx}/{len(params_list)}] scanning expansion params", flush=True)
        overlay = expansion_overlay(trades, initial_capital, params)
        overlay["score"] = round(
            score_result(
                overlay,
                max_drawdown_pct=args.max_drawdown_pct,
                min_2026_return_pct=args.min_2026_return_pct,
                max_2026_drawdown_pct=args.max_2026_drawdown_pct,
                min_60d_return_pct=args.min_60d_return_pct,
            ),
            4,
        )
        overlay["passes_drawdown_constraint"] = overlay["max_drawdown_pct"] <= args.max_drawdown_pct
        current_year = overlay.get("windows", {}).get("current_year", {})
        recent_60d = overlay.get("windows", {}).get("last_60d", {})
        overlay["passes_recent_constraints"] = (
            float(current_year.get("total_return_pct", 0.0) or 0.0) >= args.min_2026_return_pct
            and float(current_year.get("max_drawdown_pct", 0.0) or 0.0) <= args.max_2026_drawdown_pct
            and float(recent_60d.get("total_return_pct", 0.0) or 0.0) >= args.min_60d_return_pct
        )
        results.append(overlay)

    ranked = sorted(
        results,
        key=lambda item: (
            bool(item["beats_historical_main_return_and_drawdown"]),
            bool(item["beats_historical_main_return"]),
            bool(item["passes_drawdown_constraint"]),
            bool(item["passes_recent_constraints"]),
            float(item["score"]),
        ),
        reverse=True,
    )
    report = {
        "config": str(Path(args.config).resolve()),
        "data": {
            "data_15m": str(Path(args.data_15m).resolve()),
            "data_4h": str(Path(args.data_4h).resolve()),
            "start": str(prepared.start),
            "end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
        },
        "historical_main_best_record": {
            "total_return_pct": HISTORICAL_MAIN_BEST_RETURN_PCT,
            "max_drawdown_pct": HISTORICAL_MAIN_BEST_MAX_DRAWDOWN_PCT,
            "note": "README/reference record, not recomputed in this scan.",
        },
        "base_strategy_metrics": {
            "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 2),
            "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0)), 2),
            "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
            "total_trades": int(metrics.get("total_trades", 0)),
        },
        "current_main_shadow_metrics": {
            "total_return_pct": current_shadow["total_return_pct"],
            "max_drawdown_pct": current_shadow["max_drawdown_pct"],
            "sharpe_ratio": current_shadow["sharpe_ratio"],
            "total_trades": current_shadow["total_trades"],
            "skipped_trades": current_shadow["skipped_trades"],
            "params": {
                "daily_loss_stop_pct": current_shadow["daily_loss_stop_pct"],
                "equity_drawdown_stop_pct": current_shadow["equity_drawdown_stop_pct"],
                "consecutive_loss_stop": current_shadow["consecutive_loss_stop"],
                "equity_drawdown_cooldown_days": current_shadow["equity_drawdown_cooldown_days"],
            },
        },
        "candidate_count": len(results),
        "top": ranked[: args.top],
        "all_results": ranked,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_path_for(output_dir, args.start_date, prepared.end)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output_path)
    if args.stdout:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
