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

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import (  # noqa: E402
    load_prepared_data,
    max_drawdown_from_capitals,
    run_engine,
    trade_dataframe,
    trade_return_sharpe,
)
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research reverse short candidates derived from failed or weak high-leverage long events."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument(
        "--pressure-params",
        default=str(DEFAULT_PRESSURE_PARAMS_PATH),
        help="JSON reproduction file with pressure_level_target_cap_params. Use 'none' to skip.",
    )
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument(
        "--source-streams",
        default="shadow,fixed",
        help="Comma list: shadow,fixed. shadow means SOTA after shadow gate; fixed means before shadow gate.",
    )
    parser.add_argument(
        "--selectors",
        default="guarded_weak,guarded_weak_loss,weak_quality,weak_quality_loss,all_high_growth_offense,actual_loss_oracle",
        help=(
            "Comma list: guarded_weak,guarded_weak_loss,weak_quality,weak_quality_loss,"
            "weak_or_guarded,weak_or_guarded_loss,all_high_growth_offense,actual_loss_oracle,"
            "all_long_stop_loss_loss,bull_stop_loss_loss,bull_high_growth_offense_loss,"
            "trailing_stop_profit_reverse,trailing_stage_profit_reverse,trailing_atr_profit_reverse,"
            "trailing_pressure_profit_reverse,trailing_pressure_touch_lock_profit_reverse,"
            "trailing_time_enabled_profit_reverse,plain_stop_profit_reverse."
        ),
    )
    parser.add_argument(
        "--trigger-modes",
        default="entry_reversal,stop_loss_reversal",
        help=(
            "entry_reversal replaces the long at entry; stop_loss_reversal enters after the long stop-loss event; "
            "virtual_invalidation_reversal enters after a source long breaches a virtual invalidation line."
        ),
    )
    parser.add_argument("--max-quality-score", type=int, default=1)
    parser.add_argument("--virtual-invalidation-rr-values", default="0.5,0.75,1.0")
    parser.add_argument("--virtual-invalidation-lookahead-bars-values", default="8,16,32")
    parser.add_argument("--target-rr-values", default="0.75,1.0,1.5,2.0")
    parser.add_argument("--max-hold-bars-values", default="8,16,32,64")
    parser.add_argument("--leverage-values", default="1.0,2.0,4.0")
    parser.add_argument("--stop-multiplier-values", default="0.75,1.0,1.25")
    parser.add_argument("--overlay-allocation-values", default="0.25,0.5,1.0")
    parser.add_argument("--max-short-stop-pct", type=float, default=3.0)
    parser.add_argument(
        "--max-short-stop-pct-values",
        default=None,
        help="Optional comma list to scan max short stop distance pct. Defaults to --max-short-stop-pct.",
    )
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=0)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--sample-trades", type=int, default=30)
    parser.add_argument("--output", default=str(ROOT / "var" / "high_leverage_expansion" / "reverse_short_from_failed_longs.json"))
    return parser.parse_args()


def candle_time(candle: Any) -> str:
    return str(pd.Timestamp(float(candle.ts), unit="s", tz="UTC"))


def event_timestamp(event: dict[str, Any], key: str) -> pd.Timestamp:
    return pd.Timestamp(event[key]).tz_convert("UTC")


def pct(value: float) -> float:
    return round(value * 100.0, 4)


def quality_snapshot(event: dict[str, Any]) -> dict[str, Any]:
    direction = str(event.get("direction") or "")
    sign = 1.0 if direction == "BULL" else -1.0
    momentum_pct = float(event.get("feature_momentum", 0.0) or 0.0) * 100.0 * sign
    ema_gap_pct = float(event.get("feature_ema_gap", 0.0) or 0.0) * 100.0 * sign
    adx = float(event.get("feature_adx", 0.0) or 0.0)
    structure_ok = (
        bool(event.get("feature_bullish_structure"))
        if direction == "BULL"
        else bool(event.get("feature_bearish_structure"))
    )
    checks = {
        "momentum": momentum_pct >= float(FIXED_STRUCTURE_PARAMS["failed_breakout_guard_min_momentum_pct"]),
        "ema_gap": ema_gap_pct >= float(FIXED_STRUCTURE_PARAMS["failed_breakout_guard_min_ema_gap_pct"]),
        "adx": adx >= float(FIXED_STRUCTURE_PARAMS["failed_breakout_guard_min_adx"]),
        "structure": structure_ok,
    }
    return {
        "quality_score": sum(1 for passed in checks.values() if passed),
        "directional_momentum_pct": round(momentum_pct, 6),
        "directional_ema_gap_pct": round(ema_gap_pct, 6),
        "adx": round(adx, 6),
        "checks": checks,
    }


def common_failed_long_source(event: dict[str, Any]) -> bool:
    return (
        str(event.get("direction") or "") == "BULL"
        and str(event.get("regime_label") or "") == "high_growth"
        and str(event.get("risk_mode") or "") == "offense"
    )


def selected_by(event: dict[str, Any], selector: str, max_quality_score: int) -> bool:
    direction = str(event.get("direction") or "")
    exit_reason = str(event.get("exit_reason") or "")
    return_value = float(event.get("return", 0.0) or 0.0)
    stop_update_reason = str(event.get("last_stop_update_reason") or "")
    time_based_trailing_enabled = bool(event.get("time_based_trailing_enabled"))
    pressure_target_applied = bool(event.get("pressure_target_applied"))
    pressure_touch_lock_applied = bool(event.get("pressure_touch_lock_applied"))
    if selector == "all_long_stop_loss_loss":
        return direction == "BULL" and exit_reason == "stop_loss" and return_value < 0.0
    if selector == "bull_stop_loss_loss":
        return direction == "BULL" and exit_reason == "stop_loss" and return_value < 0.0
    if selector in {
        "trailing_stop_profit_reverse",
        "trailing_stage_profit_reverse",
        "trailing_atr_profit_reverse",
        "trailing_pressure_profit_reverse",
        "trailing_pressure_touch_lock_profit_reverse",
        "trailing_time_enabled_profit_reverse",
        "plain_stop_profit_reverse",
    }:
        base = (
            direction == "BULL"
            and exit_reason == "stop_loss"
            and return_value > 0.0
            and str(event.get("regime_label") or "") == "high_growth"
            and str(event.get("risk_mode") or "") == "offense"
        )
        if not base:
            return False
        if selector == "trailing_stop_profit_reverse":
            return True
        if selector == "trailing_stage_profit_reverse":
            return stop_update_reason.startswith("trail_stage_")
        if selector == "trailing_atr_profit_reverse":
            return stop_update_reason == "atr_trail"
        if selector == "trailing_pressure_profit_reverse":
            return stop_update_reason == "pressure_level_trail" or pressure_target_applied or pressure_touch_lock_applied
        if selector == "trailing_pressure_touch_lock_profit_reverse":
            return pressure_touch_lock_applied
        if selector == "trailing_time_enabled_profit_reverse":
            return time_based_trailing_enabled
        if selector == "plain_stop_profit_reverse":
            return not stop_update_reason
        return (
            direction == "BULL"
            and exit_reason == "stop_loss"
            and return_value > 0.0
        )
    if selector == "bull_high_growth_offense_loss":
        return (
            direction == "BULL"
            and str(event.get("regime_label") or "") == "high_growth"
            and str(event.get("risk_mode") or "") == "offense"
            and exit_reason == "stop_loss"
            and return_value < 0.0
        )
    if not common_failed_long_source(event):
        return False
    quality = quality_snapshot(event)
    guarded = bool(event.get("failed_breakout_guard_applied"))
    weak_quality = int(quality["quality_score"]) <= max_quality_score
    if selector == "guarded_weak":
        return guarded
    if selector == "guarded_weak_loss":
        return guarded and float(event.get("return", 0.0) or 0.0) < 0.0
    if selector == "weak_quality":
        return weak_quality
    if selector == "weak_quality_loss":
        return weak_quality and float(event.get("return", 0.0) or 0.0) < 0.0
    if selector == "weak_or_guarded":
        return weak_quality or guarded
    if selector == "weak_or_guarded_loss":
        return (weak_quality or guarded) and float(event.get("return", 0.0) or 0.0) < 0.0
    if selector == "all_high_growth_offense":
        return True
    if selector == "actual_loss_oracle":
        return float(event.get("return", 0.0) or 0.0) < 0.0
    raise ValueError(f"Unsupported selector: {selector}")


def short_entry_for_event(
    event: dict[str, Any],
    candles: list[Any],
    trigger_mode: str,
    virtual_invalidation_rr: float | None = None,
    virtual_invalidation_lookahead_bars: int | None = None,
) -> tuple[int, float, str] | None:
    if trigger_mode == "entry_reversal":
        entry_idx = event.get("entry_idx")
        entry_price = float(event.get("entry_price", 0.0) or 0.0)
        entry_time = str(event.get("entry_time") or "")
    elif trigger_mode == "stop_loss_reversal":
        if str(event.get("exit_reason") or "") != "stop_loss":
            return None
        entry_idx = event.get("exit_idx")
        entry_price = float(event.get("exit_price", 0.0) or 0.0)
        entry_time = str(event.get("exit_time") or "")
    elif trigger_mode == "virtual_invalidation_reversal":
        if str(event.get("direction") or "") != "BULL":
            return None
        source_entry_idx = event.get("entry_idx")
        source_exit_idx = event.get("exit_idx")
        source_entry_price = float(event.get("entry_price", 0.0) or 0.0)
        source_stop_pct = float(event.get("stop_distance_pct", 0.0) or 0.0) / 100.0
        if source_entry_idx is None or source_exit_idx is None or source_entry_price <= 0 or source_stop_pct <= 0:
            return None
        invalidation_rr = float(virtual_invalidation_rr if virtual_invalidation_rr is not None else 1.0)
        lookahead = int(virtual_invalidation_lookahead_bars if virtual_invalidation_lookahead_bars is not None else 16)
        risk_price = source_entry_price * source_stop_pct
        invalidation_price = source_entry_price - risk_price * invalidation_rr
        start = int(source_entry_idx) + 1
        end = min(len(candles) - 1, int(source_exit_idx), int(source_entry_idx) + max(1, lookahead))
        if start > end or invalidation_price <= 0:
            return None
        for idx in range(start, end + 1):
            if float(candles[idx].l) <= invalidation_price:
                return idx, invalidation_price, candle_time(candles[idx])
        return None
    else:
        raise ValueError(f"Unsupported trigger mode: {trigger_mode}")
    if entry_idx is None or entry_price <= 0:
        return None
    return int(entry_idx), entry_price, entry_time


def simulate_short_trade(
    event: dict[str, Any],
    candles: list[Any],
    trigger_mode: str,
    target_rr: float,
    max_hold_bars: int,
    leverage: float,
    stop_multiplier: float,
    max_short_stop_pct: float,
    virtual_invalidation_rr: float | None,
    virtual_invalidation_lookahead_bars: int | None,
    taker_fee_rate: float,
    slippage_bps: float,
) -> dict[str, Any] | None:
    entry = short_entry_for_event(
        event,
        candles,
        trigger_mode,
        virtual_invalidation_rr=virtual_invalidation_rr,
        virtual_invalidation_lookahead_bars=virtual_invalidation_lookahead_bars,
    )
    if entry is None:
        return None
    entry_idx, entry_price, entry_time = entry
    if entry_idx + 1 >= len(candles):
        return None

    source_stop_pct = float(event.get("stop_distance_pct", 0.0) or 0.0) / 100.0
    stop_pct = source_stop_pct * stop_multiplier
    if stop_pct <= 0 or stop_pct * 100.0 > max_short_stop_pct:
        return None
    target_price = entry_price * (1.0 - stop_pct * target_rr)
    stop_price = entry_price * (1.0 + stop_pct)
    if target_price <= 0 or stop_price <= entry_price:
        return None

    start = entry_idx + 1
    end = min(len(candles) - 1, entry_idx + max(1, max_hold_bars))
    if start > end:
        return None

    exit_idx = end
    exit_price = float(candles[end].c)
    exit_reason = "time_exit"
    mfe = 0.0
    mae = 0.0
    risk_price = entry_price * stop_pct
    for idx in range(start, end + 1):
        candle = candles[idx]
        high = float(candle.h)
        low = float(candle.l)
        mfe = max(mfe, entry_price - low)
        mae = max(mae, high - entry_price)
        # Conservative same-candle ordering: if both stop and target trade, count the stop.
        if high >= stop_price:
            exit_idx = idx
            exit_price = stop_price
            exit_reason = "stop_loss"
            break
        if low <= target_price:
            exit_idx = idx
            exit_price = target_price
            exit_reason = "target_rr"
            break

    gross_unit_return = (entry_price - exit_price) / entry_price
    roundtrip_cost = 2.0 * float(taker_fee_rate) + 2.0 * float(slippage_bps) / 10000.0
    unit_return = gross_unit_return - roundtrip_cost
    trade_return = unit_return * leverage
    source_quality = quality_snapshot(event)
    return {
        "source_entry_time": event.get("entry_time"),
        "source_exit_time": event.get("exit_time"),
        "source_return_pct": pct(float(event.get("return", 0.0) or 0.0)),
        "source_exit_reason": event.get("exit_reason"),
        "source_effective_leverage": event.get("effective_leverage"),
        "source_failed_breakout_guard_applied": bool(event.get("failed_breakout_guard_applied")),
        "source_quality_score": source_quality["quality_score"],
        "source_quality": source_quality,
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "entry_time": entry_time,
        "exit_time": candle_time(candles[exit_idx]),
        "entry_price": round(entry_price, 6),
        "exit_price": round(exit_price, 6),
        "stop_price": round(stop_price, 6),
        "target_price": round(target_price, 6),
        "stop_distance_pct": round(stop_pct * 100.0, 6),
        "target_rr": target_rr,
        "max_hold_bars": max_hold_bars,
        "leverage": leverage,
        "stop_multiplier": stop_multiplier,
        "virtual_invalidation_rr": virtual_invalidation_rr,
        "virtual_invalidation_lookahead_bars": virtual_invalidation_lookahead_bars,
        "exit_reason": exit_reason,
        "gross_unit_return_pct": pct(gross_unit_return),
        "unit_return_pct": pct(unit_return),
        "return": trade_return,
        "return_pct": pct(trade_return),
        "mfe_rr": round(mfe / risk_price, 4) if risk_price > 0 else 0.0,
        "mae_rr": round(mae / risk_price, 4) if risk_price > 0 else 0.0,
    }


def replay_non_overlapping(trades: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    ordered = sorted(trades, key=lambda item: (int(item["entry_idx"]), int(item["exit_idx"])))
    capital = initial_capital
    capitals: list[float] = []
    returns: list[float] = []
    accepted: list[dict[str, Any]] = []
    skipped_overlap = 0
    last_exit_idx = -1
    exit_counts: dict[str, int] = {}
    gross_profit = 0.0
    gross_loss = 0.0

    for trade in ordered:
        if int(trade["entry_idx"]) <= last_exit_idx:
            skipped_overlap += 1
            continue
        trade_return = float(trade["return"])
        capital = max(0.0, capital * (1.0 + trade_return))
        returns.append(trade_return)
        capitals.append(capital)
        accepted_trade = dict(trade)
        accepted_trade["capital"] = round(capital, 2)
        accepted.append(accepted_trade)
        last_exit_idx = int(trade["exit_idx"])
        exit_reason = str(trade["exit_reason"])
        exit_counts[exit_reason] = exit_counts.get(exit_reason, 0) + 1
        if trade_return > 0:
            gross_profit += trade_return
        else:
            gross_loss += abs(trade_return)

    wins = sum(1 for value in returns if value > 0)
    losses = len(returns) - wins
    return {
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100.0, 2),
        "final_capital": round(capital, 2),
        "sharpe_ratio": round(trade_return_sharpe(returns), 3),
        "max_drawdown_pct": round(max_drawdown_from_capitals(capitals, initial_capital), 2),
        "trades": len(accepted),
        "raw_candidates": len(ordered),
        "skipped_overlap": skipped_overlap,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wins / len(returns) * 100.0, 2) if returns else 0.0,
        "avg_return_pct": round(sum(returns) / len(returns) * 100.0, 4) if returns else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "exit_counts": exit_counts,
        "events": accepted,
    }


def replay_window(events: list[dict[str, Any]], initial_capital: float, start: pd.Timestamp) -> dict[str, Any]:
    selected = [event for event in events if event_timestamp(event, "entry_time") >= start]
    return {key: value for key, value in replay_non_overlapping(selected, initial_capital).items() if key != "events"}


def add_windows(result: dict[str, Any], initial_capital: float, data_end: pd.Timestamp) -> dict[str, Any]:
    events = result["events"]
    starts = {
        "current_year": pd.Timestamp(f"{data_end.year}-01-01", tz="UTC"),
        "last_60d": data_end - pd.Timedelta(days=60),
        "last_30d": data_end - pd.Timedelta(days=30),
    }
    result["windows"] = {name: replay_window(events, initial_capital, start) for name, start in starts.items()}
    return result


def event_stream_summary(events: list[dict[str, Any]], initial_capital: float, data_end: pd.Timestamp) -> dict[str, Any]:
    trades = []
    for event in events:
        copied = dict(event)
        copied["entry_idx"] = int(copied.get("entry_idx") or 0)
        copied["exit_idx"] = int(copied.get("exit_idx") or copied["entry_idx"])
        copied["return_pct"] = pct(float(copied.get("return", 0.0) or 0.0))
        trades.append(copied)
    result = replay_non_overlapping(trades, initial_capital)
    return add_windows(result, initial_capital, data_end)


def compact_result(result: dict[str, Any], sample_trades: int) -> dict[str, Any]:
    payload = {key: value for key, value in result.items() if key != "events"}
    payload["sample_events"] = result.get("events", [])[:sample_trades]
    return payload


def score_result(result: dict[str, Any]) -> float:
    year = result.get("windows", {}).get("current_year", {})
    last_60d = result.get("windows", {}).get("last_60d", {})
    last_30d = result.get("windows", {}).get("last_30d", {})
    return round(
        float(result.get("total_return_pct", 0.0))
        + float(year.get("total_return_pct", 0.0)) * 80.0
        + float(last_60d.get("total_return_pct", 0.0)) * 40.0
        + float(last_30d.get("total_return_pct", 0.0)) * 20.0
        - float(result.get("max_drawdown_pct", 0.0)) * 15.0
        - float(year.get("max_drawdown_pct", 0.0)) * 20.0,
        4,
    )


def standard_sota_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": "sota_long",
        "entry_idx": int(event.get("entry_idx") or 0),
        "exit_idx": int(event.get("exit_idx") or event.get("entry_idx") or 0),
        "entry_time": str(event.get("entry_time")),
        "exit_time": str(event.get("exit_time")),
        "direction": event.get("direction"),
        "return": float(event.get("return", 0.0) or 0.0),
        "return_pct": pct(float(event.get("return", 0.0) or 0.0)),
        "exit_reason": event.get("exit_reason"),
        "source_effective_leverage": event.get("effective_leverage"),
        "source_failed_breakout_guard_applied": bool(event.get("failed_breakout_guard_applied")),
        "source_quality_score": quality_snapshot(event)["quality_score"],
    }


def standard_reverse_short_event(event: dict[str, Any], overlay_allocation: float) -> dict[str, Any]:
    copied = dict(event)
    raw_return = float(copied.get("return", 0.0) or 0.0)
    copied["event_type"] = "reverse_short"
    copied["raw_return"] = raw_return
    copied["raw_return_pct"] = pct(raw_return)
    copied["overlay_allocation"] = overlay_allocation
    copied["return"] = raw_return * overlay_allocation
    copied["return_pct"] = pct(float(copied["return"]))
    copied["direction"] = "BEAR"
    return copied


def standard_event_summary(
    events: list[dict[str, Any]],
    initial_capital: float,
    order_key: str,
) -> dict[str, Any]:
    ordered = sorted(events, key=lambda item: (int(item.get(order_key, 0) or 0), int(item.get("exit_idx", 0) or 0)))
    capital = initial_capital
    capitals: list[float] = []
    returns: list[float] = []
    accepted: list[dict[str, Any]] = []
    event_type_counts: dict[str, int] = {}
    exit_counts: dict[str, int] = {}
    gross_profit = 0.0
    gross_loss = 0.0
    for event in ordered:
        trade_return = float(event.get("return", 0.0) or 0.0)
        capital = max(0.0, capital * (1.0 + trade_return))
        capitals.append(capital)
        returns.append(trade_return)
        accepted_event = dict(event)
        accepted_event["capital"] = round(capital, 2)
        accepted.append(accepted_event)
        event_type = str(event.get("event_type") or "unknown")
        exit_reason = str(event.get("exit_reason") or "unknown")
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        exit_counts[exit_reason] = exit_counts.get(exit_reason, 0) + 1
        if trade_return > 0:
            gross_profit += trade_return
        else:
            gross_loss += abs(trade_return)

    wins = sum(1 for value in returns if value > 0)
    losses = len(returns) - wins
    return {
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100.0, 2),
        "final_capital": round(capital, 2),
        "sharpe_ratio": round(trade_return_sharpe(returns), 3),
        "max_drawdown_pct": round(max_drawdown_from_capitals(capitals, initial_capital), 2),
        "trades": len(accepted),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wins / len(returns) * 100.0, 2) if returns else 0.0,
        "avg_return_pct": round(sum(returns) / len(returns) * 100.0, 4) if returns else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "event_type_counts": event_type_counts,
        "exit_counts": exit_counts,
        "events": accepted,
    }


def replay_standard_single_slot(events: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    ordered = sorted(
        events,
        key=lambda item: (
            int(item.get("entry_idx", 0) or 0),
            0 if str(item.get("event_type")) == "sota_long" else 1,
            int(item.get("exit_idx", 0) or 0),
        ),
    )
    accepted: list[dict[str, Any]] = []
    skipped_by_type: dict[str, int] = {}
    last_exit_idx = -1
    for event in ordered:
        entry_idx = int(event.get("entry_idx", 0) or 0)
        if entry_idx < last_exit_idx:
            event_type = str(event.get("event_type") or "unknown")
            skipped_by_type[event_type] = skipped_by_type.get(event_type, 0) + 1
            continue
        accepted.append(event)
        last_exit_idx = max(last_exit_idx, int(event.get("exit_idx", entry_idx) or entry_idx))
    result = standard_event_summary(accepted, initial_capital, "entry_idx")
    result["skipped_by_type"] = skipped_by_type
    return result


def overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_entry = int(a.get("entry_idx", 0) or 0)
    a_exit = int(a.get("exit_idx", a_entry) or a_entry)
    b_entry = int(b.get("entry_idx", 0) or 0)
    b_exit = int(b.get("exit_idx", b_entry) or b_entry)
    return a_entry < b_exit and a_exit > b_entry


def filter_overlay_for_base_priority(
    base_events: list[dict[str, Any]],
    overlay_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    accepted: list[dict[str, Any]] = []
    skipped = 0
    for event in overlay_events:
        if any(overlaps(event, base_event) for base_event in base_events):
            skipped += 1
            continue
        accepted.append(event)
    return accepted, skipped


def add_standard_windows(
    result: dict[str, Any],
    initial_capital: float,
    data_end: pd.Timestamp,
    order_key: str,
) -> dict[str, Any]:
    events = result["events"]
    starts = {
        "current_year": pd.Timestamp(f"{data_end.year}-01-01", tz="UTC"),
        "last_60d": data_end - pd.Timedelta(days=60),
        "last_30d": data_end - pd.Timedelta(days=30),
    }
    result["windows"] = {
        name: {key: value for key, value in standard_event_summary(
            [event for event in events if event_timestamp(event, "entry_time") >= start],
            initial_capital,
            order_key,
        ).items() if key != "events"}
        for name, start in starts.items()
    }
    return result


def add_combo_deltas(result: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    result["delta_vs_shadow_sota"] = {
        "total_return_pct": round(float(result.get("total_return_pct", 0.0)) - float(baseline.get("total_return_pct", 0.0)), 4),
        "max_drawdown_pct": round(float(result.get("max_drawdown_pct", 0.0)) - float(baseline.get("max_drawdown_pct", 0.0)), 4),
    }
    result["window_deltas_vs_shadow_sota"] = {}
    for name, window in result.get("windows", {}).items():
        base_window = baseline.get("windows", {}).get(name, {})
        result["window_deltas_vs_shadow_sota"][name] = {
            "total_return_pct": round(float(window.get("total_return_pct", 0.0)) - float(base_window.get("total_return_pct", 0.0)), 4),
            "max_drawdown_pct": round(float(window.get("max_drawdown_pct", 0.0)) - float(base_window.get("max_drawdown_pct", 0.0)), 4),
        }
    return result


def score_combo_result(result: dict[str, Any]) -> float:
    delta = result.get("delta_vs_shadow_sota", {})
    window_deltas = result.get("window_deltas_vs_shadow_sota", {})
    year = window_deltas.get("current_year", {})
    last_60d = window_deltas.get("last_60d", {})
    last_30d = window_deltas.get("last_30d", {})
    dd_penalty = max(0.0, float(delta.get("max_drawdown_pct", 0.0))) * 120.0
    year_dd_penalty = max(0.0, float(year.get("max_drawdown_pct", 0.0))) * 160.0
    return round(
        float(delta.get("total_return_pct", 0.0))
        + float(year.get("total_return_pct", 0.0)) * 150.0
        + float(last_60d.get("total_return_pct", 0.0)) * 80.0
        + float(last_30d.get("total_return_pct", 0.0)) * 40.0
        - dd_penalty
        - year_dd_penalty,
        4,
    )


def build_combo_results(
    base_events: list[dict[str, Any]],
    overlay_events: list[dict[str, Any]],
    initial_capital: float,
    data_end: pd.Timestamp,
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    parallel = standard_event_summary(base_events + overlay_events, initial_capital, "exit_idx")
    parallel = add_standard_windows(parallel, initial_capital, data_end, "exit_idx")
    parallel = add_combo_deltas(parallel, baseline)
    parallel["combo_mode"] = "parallel_overlay"
    parallel["score"] = score_combo_result(parallel)

    base_priority_overlay, skipped_base_priority = filter_overlay_for_base_priority(base_events, overlay_events)
    base_priority = replay_standard_single_slot(base_events + base_priority_overlay, initial_capital)
    base_priority = add_standard_windows(base_priority, initial_capital, data_end, "entry_idx")
    base_priority = add_combo_deltas(base_priority, baseline)
    base_priority["combo_mode"] = "base_priority_single_slot"
    base_priority["base_priority_overlay_skipped"] = skipped_base_priority
    base_priority["score"] = score_combo_result(base_priority)

    overlay_priority = replay_standard_single_slot(base_events + overlay_events, initial_capital)
    overlay_priority = add_standard_windows(overlay_priority, initial_capital, data_end, "entry_idx")
    overlay_priority = add_combo_deltas(overlay_priority, baseline)
    overlay_priority["combo_mode"] = "overlay_priority_single_slot"
    overlay_priority["score"] = score_combo_result(overlay_priority)
    return [parallel, base_priority, overlay_priority]


def compact_combo_result(result: dict[str, Any], sample_trades: int) -> dict[str, Any]:
    payload = {key: value for key, value in result.items() if key != "events"}
    payload["sample_events"] = result.get("events", [])[:sample_trades]
    return payload


def best_combos_by_mode(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for result in results:
        mode = str(result.get("combo_mode") or "")
        if mode not in best or float(result["score"]) > float(best[mode]["score"]):
            best[mode] = result
    return sorted(best.values(), key=lambda item: float(item["score"]), reverse=True)


def best_by_family(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for result in results:
        key = (
            str(result["params"]["source_stream"]),
            str(result["params"]["selector"]),
            str(result["params"]["trigger_mode"]),
        )
        if key not in best or float(result["score"]) > float(best[key]["score"]):
            best[key] = result
    return sorted(best.values(), key=lambda item: float(item["score"]), reverse=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    payload = load_config_payload(config_path)
    pressure_params: dict[str, Any] = {}
    pressure_params_path: str | None = None
    if str(args.pressure_params).strip().lower() != "none":
        payload, pressure_params = apply_pressure_params(payload, Path(args.pressure_params))
        pressure_params_path = str(Path(args.pressure_params).resolve())

    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=payload.get("regime_switcher_thresholds"),
    )
    metrics, engine = run_engine(payload, prepared, args.start_date)
    trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
    initial_capital = float(metrics.get("initial_capital", 1000.0))
    fixed = expansion_overlay(trades, initial_capital, FIXED_STRUCTURE_PARAMS, include_events=True)
    fixed_events = fixed["events"]
    shadow = replay_shadow_events(
        fixed_events,
        initial_capital,
        daily_loss_stop_pct=float(args.daily_loss_stop_pct),
        equity_drawdown_stop_pct=float(args.equity_drawdown_stop_pct),
        consecutive_loss_stop=int(args.consecutive_loss_stop),
        equity_drawdown_cooldown_days=int(args.equity_drawdown_cooldown_days),
    )
    shadow_events = shadow["events"]

    stream_events = {
        "fixed": fixed_events,
        "shadow": shadow_events,
    }
    source_streams = parse_str_list(args.source_streams)
    selectors = parse_str_list(args.selectors)
    trigger_modes = parse_str_list(args.trigger_modes)
    target_rr_values = parse_float_list(args.target_rr_values)
    max_hold_bars_values = parse_int_list(args.max_hold_bars_values)
    leverage_values = parse_float_list(args.leverage_values)
    stop_multiplier_values = parse_float_list(args.stop_multiplier_values)
    overlay_allocation_values = parse_float_list(args.overlay_allocation_values)
    virtual_invalidation_rr_values = parse_float_list(args.virtual_invalidation_rr_values)
    virtual_invalidation_lookahead_bars_values = parse_int_list(args.virtual_invalidation_lookahead_bars_values)
    max_short_stop_pct_values = (
        parse_float_list(args.max_short_stop_pct_values)
        if args.max_short_stop_pct_values
        else [float(args.max_short_stop_pct)]
    )
    taker_fee_rate = float(payload.get("taker_fee_rate", 0.0005) or 0.0)
    slippage_bps = float(payload.get("slippage_bps", 0.0) or 0.0)

    base_shadow_summary = event_stream_summary(shadow_events, initial_capital, prepared.end)
    standard_base_events = [standard_sota_event(event) for event in shadow_events]
    results: list[dict[str, Any]] = []
    combo_results: list[dict[str, Any]] = []
    trigger_param_sets: dict[str, list[tuple[float | None, int | None]]] = {
        "entry_reversal": [(None, None)],
        "stop_loss_reversal": [(None, None)],
        "virtual_invalidation_reversal": list(itertools.product(virtual_invalidation_rr_values, virtual_invalidation_lookahead_bars_values)),
    }

    for source_stream, selector, trigger_mode, target_rr, max_hold_bars, leverage, stop_multiplier, max_short_stop_pct in itertools.product(
        source_streams,
        selectors,
        trigger_modes,
        target_rr_values,
        max_hold_bars_values,
        leverage_values,
        stop_multiplier_values,
        max_short_stop_pct_values,
    ):
        if source_stream not in stream_events:
            raise ValueError(f"Unsupported source stream: {source_stream}")
        if trigger_mode not in trigger_param_sets:
            raise ValueError(f"Unsupported trigger mode: {trigger_mode}")
        for virtual_invalidation_rr, virtual_invalidation_lookahead_bars in trigger_param_sets[trigger_mode]:
            candidates = [
                simulate_short_trade(
                    event=event,
                    candles=prepared.c15m,
                    trigger_mode=trigger_mode,
                    target_rr=target_rr,
                    max_hold_bars=max_hold_bars,
                    leverage=leverage,
                    stop_multiplier=stop_multiplier,
                    max_short_stop_pct=max_short_stop_pct,
                    virtual_invalidation_rr=virtual_invalidation_rr,
                    virtual_invalidation_lookahead_bars=virtual_invalidation_lookahead_bars,
                    taker_fee_rate=taker_fee_rate,
                    slippage_bps=slippage_bps,
                )
                for event in stream_events[source_stream]
                if selected_by(event, selector, int(args.max_quality_score))
            ]
            simulated = [trade for trade in candidates if trade is not None]
            result = replay_non_overlapping(simulated, initial_capital)
            result = add_windows(result, initial_capital, prepared.end)
            result["params"] = {
                "source_stream": source_stream,
                "selector": selector,
                "trigger_mode": trigger_mode,
                "target_rr": target_rr,
                "max_hold_bars": max_hold_bars,
                "leverage": leverage,
                "stop_multiplier": stop_multiplier,
                "max_short_stop_pct": max_short_stop_pct,
                "virtual_invalidation_rr": virtual_invalidation_rr,
                "virtual_invalidation_lookahead_bars": virtual_invalidation_lookahead_bars,
            }
            result["score"] = score_result(result)
            results.append(result)
            for overlay_allocation in overlay_allocation_values:
                standard_overlay_events = [
                    standard_reverse_short_event(event, overlay_allocation)
                    for event in result["events"]
                ]
                for combo in build_combo_results(
                    standard_base_events,
                    standard_overlay_events,
                    initial_capital,
                    prepared.end,
                    base_shadow_summary,
                ):
                    combo["params"] = {
                        **result["params"],
                        "overlay_allocation": overlay_allocation,
                        "combo_mode": combo["combo_mode"],
                    }
                    combo_results.append(combo)

    results.sort(key=lambda item: float(item["score"]), reverse=True)
    combo_results.sort(key=lambda item: float(item["score"]), reverse=True)
    family_best = best_by_family(results)
    combo_mode_best = best_combos_by_mode(combo_results)
    report = {
        "config": str(config_path.resolve()),
        "pressure_params_path": pressure_params_path,
        "pressure_params": pressure_params,
        "data": {
            "start": str(prepared.start),
            "end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
        },
        "baseline": {
            "engine": {
                "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 2),
                "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0)), 2),
                "total_trades": int(metrics.get("total_trades", 0)),
                "win_rate": round(float(metrics.get("win_rate", 0.0)), 2),
            },
            "fixed_structure": compact_result(event_stream_summary(fixed_events, initial_capital, prepared.end), 0),
            "shadow_sota": compact_result(base_shadow_summary, 0),
        },
        "experiment": {
            "source_streams": source_streams,
            "selectors": selectors,
            "trigger_modes": trigger_modes,
            "target_rr_values": target_rr_values,
            "max_hold_bars_values": max_hold_bars_values,
            "leverage_values": leverage_values,
            "stop_multiplier_values": stop_multiplier_values,
            "overlay_allocation_values": overlay_allocation_values,
            "virtual_invalidation_rr_values": virtual_invalidation_rr_values,
            "virtual_invalidation_lookahead_bars_values": virtual_invalidation_lookahead_bars_values,
            "max_short_stop_pct_values": max_short_stop_pct_values,
            "max_quality_score": args.max_quality_score,
            "taker_fee_rate": taker_fee_rate,
            "slippage_bps": slippage_bps,
            "candidate_count": len(results),
            "combo_candidate_count": len(combo_results),
        },
        "top": [compact_result(result, int(args.sample_trades)) for result in results[: int(args.top)]],
        "best_by_source_selector_trigger": [
            compact_result(result, min(5, int(args.sample_trades))) for result in family_best
        ],
        "combo_top": [compact_combo_result(result, int(args.sample_trades)) for result in combo_results[: int(args.top)]],
        "best_combo_by_mode": [
            compact_combo_result(result, min(5, int(args.sample_trades))) for result in combo_mode_best
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print(output)
    print("Baseline shadow SOTA:")
    shadow_summary = report["baseline"]["shadow_sota"]
    print(
        f"  full={shadow_summary['total_return_pct']:.2f}%/"
        f"{shadow_summary['max_drawdown_pct']:.2f}% trades={shadow_summary['trades']} "
        f"win_rate={shadow_summary['win_rate_pct']:.2f}%"
    )
    print("Top reverse-short candidates:")
    for idx, item in enumerate(report["top"][:10], start=1):
        params = item["params"]
        year = item.get("windows", {}).get("current_year", {})
        recent_60d = item.get("windows", {}).get("last_60d", {})
        print(
            f"{idx:02d} score={item['score']:.2f} full={item['total_return_pct']:.2f}%/"
            f"{item['max_drawdown_pct']:.2f}% year={year.get('total_return_pct', 0.0):.2f}% "
            f"60d={recent_60d.get('total_return_pct', 0.0):.2f}% "
            f"trades={item['trades']} win={item['win_rate_pct']:.2f}% params={params}"
        )
    print("Top combined candidates vs shadow SOTA:")
    for idx, item in enumerate(report["combo_top"][:10], start=1):
        params = item["params"]
        delta = item.get("delta_vs_shadow_sota", {})
        year_delta = item.get("window_deltas_vs_shadow_sota", {}).get("current_year", {})
        recent_60d_delta = item.get("window_deltas_vs_shadow_sota", {}).get("last_60d", {})
        print(
            f"{idx:02d} score={item['score']:.2f} full={item['total_return_pct']:.2f}%/"
            f"{item['max_drawdown_pct']:.2f}% delta={delta.get('total_return_pct', 0.0):.2f}%/"
            f"{delta.get('max_drawdown_pct', 0.0):+.2f}dd "
            f"year_delta={year_delta.get('total_return_pct', 0.0):.2f}% "
            f"60d_delta={recent_60d_delta.get('total_return_pct', 0.0):.2f}% "
            f"trades={item['trades']} params={params}"
        )


if __name__ == "__main__":
    main()
