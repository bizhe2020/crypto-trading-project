#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.report_pa_ict_liquidity_features import recent_fvg, scan_events, time_bucket, zone_touched  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, add_windows, replay_shadow_events  # noqa: E402
from strategy.scalp_robust_v2_core import Candle, precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_context" / "smc_trade_context_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only SMC context report for promoted high-leverage shadow accepted trades."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--stdout", action="store_true")

    parser.add_argument("--h4-range-lookback-bars", type=int, default=42)
    parser.add_argument("--d1-range-lookback-bars", type=int, default=60)
    parser.add_argument("--h4-swing-lookback-bars", type=int, default=120)
    parser.add_argument("--d1-swing-lookback-bars", type=int, default=90)
    parser.add_argument("--liquidity-target-lookback-15m", type=int, default=384)
    parser.add_argument("--liquidity-target-lookback-4h", type=int, default=180)
    parser.add_argument("--recent-sweep-lookback-bars", type=int, default=96)
    parser.add_argument("--recent-fvg-lookback-bars", type=int, default=32)
    parser.add_argument("--min-liquidity-target-rr", type=float, default=1.5)

    parser.add_argument("--swing-n", type=int, default=3)
    parser.add_argument("--swing-lookback", type=int, default=80)
    parser.add_argument("--liquidity-lookback-bars", type=int, default=192)
    parser.add_argument("--mss-lookahead-bars", type=int, default=24)
    parser.add_argument("--fvg-lookback-bars", type=int, default=8)
    parser.add_argument("--entry-lookahead-bars", type=int, default=40)
    parser.add_argument("--outcome-lookahead-bars", type=int, default=96)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--min-body-atr", type=float, default=0.7)
    parser.add_argument("--min-range-atr", type=float, default=1.1)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.05)
    parser.add_argument("--target-rr", type=float, default=2.0)
    return parser.parse_args()


def load_best_shadow_params(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    params = payload.get("shadow_gate_scan_params") or payload.get("runtime_shadow_gate_params") or {}
    return {
        "daily_loss_stop_pct": float(params.get("daily_loss_stop_pct", 6.0) or 0.0),
        "equity_drawdown_stop_pct": float(params.get("equity_drawdown_stop_pct", 15.0) or 0.0),
        "equity_drawdown_cooldown_days": int(params.get("equity_drawdown_cooldown_days", 2) or 0),
        "consecutive_loss_stop": int(params.get("consecutive_loss_stop", 0) or 0),
    }


def daily_candles_from_4h(c4h: list[Candle]) -> list[Candle]:
    buckets: dict[pd.Timestamp, list[Candle]] = defaultdict(list)
    for candle in c4h:
        day = pd.Timestamp(candle.ts, unit="s", tz="UTC").normalize()
        buckets[day].append(candle)
    daily: list[Candle] = []
    for day in sorted(buckets):
        candles = buckets[day]
        daily.append(
            Candle(
                ts=day.timestamp(),
                o=float(candles[0].o),
                h=max(float(candle.h) for candle in candles),
                l=min(float(candle.l) for candle in candles),
                c=float(candles[-1].c),
                v=sum(float(candle.v) for candle in candles),
            )
        )
    return daily


def completed_4h_idx_for_entry(mapping: list[int], entry_idx: int) -> int:
    if entry_idx < 0 or entry_idx >= len(mapping):
        return -1
    return max(0, int(mapping[entry_idx]) - 1)


def completed_d1_idx_for_entry(daily_ts: list[float], entry_ts: float) -> int:
    entry_day = pd.Timestamp(entry_ts, unit="s", tz="UTC").normalize().timestamp()
    return bisect.bisect_left(daily_ts, entry_day) - 1


def range_context(candles: list[Candle], idx: int, lookback: int, price: float) -> dict[str, Any]:
    if idx < 0 or not candles:
        return {"zone": "unknown", "position_pct": None, "range_low": None, "range_high": None}
    start = max(0, idx - max(int(lookback), 1) + 1)
    window = candles[start : idx + 1]
    if not window:
        return {"zone": "unknown", "position_pct": None, "range_low": None, "range_high": None}
    high = max(float(candle.h) for candle in window)
    low = min(float(candle.l) for candle in window)
    if high <= low:
        return {"zone": "unknown", "position_pct": None, "range_low": low, "range_high": high}
    position_pct = (float(price) - low) / (high - low) * 100.0
    if position_pct < 45.0:
        zone = "discount"
    elif position_pct > 55.0:
        zone = "premium"
    else:
        zone = "equilibrium"
    return {
        "zone": zone,
        "position_pct": round(position_pct, 2),
        "range_low": round(low, 2),
        "range_high": round(high, 2),
    }


def pd_favorable(direction: str, context: dict[str, Any]) -> bool:
    position_pct = context.get("position_pct")
    if position_pct is None:
        return False
    if direction == "BULL":
        return float(position_pct) < 50.0
    if direction == "BEAR":
        return float(position_pct) > 50.0
    return False


def pd_side(direction: str, context: dict[str, Any]) -> str:
    if context.get("position_pct") is None:
        return "unknown"
    return "favorable" if pd_favorable(direction, context) else "adverse"


def h4_bias_for_idx(precomputed: Any, idx: int) -> dict[str, Any]:
    if idx < 0:
        return {"bias": "NONE", "bull_score": None, "bear_score": None}
    bull_scores = getattr(precomputed, "bull_trend_score_4h", [])
    bear_scores = getattr(precomputed, "bear_trend_score_4h", [])
    if idx >= len(bull_scores) or idx >= len(bear_scores):
        return {"bias": "NONE", "bull_score": None, "bear_score": None}
    bull_score = int(bull_scores[idx])
    bear_score = int(bear_scores[idx])
    if bull_score > bear_score:
        bias = "BULL"
    elif bear_score > bull_score:
        bias = "BEAR"
    else:
        bias = "NONE"
    return {"bias": bias, "bull_score": bull_score, "bear_score": bear_score}


def structure_bias_for_idx(candles: list[Candle], highs: list[int], lows: list[int], idx: int) -> dict[str, Any]:
    prev_highs = [swing_idx for swing_idx in highs if swing_idx <= idx]
    prev_lows = [swing_idx for swing_idx in lows if swing_idx <= idx]
    if len(prev_highs) < 2 or len(prev_lows) < 2:
        return {"bias": "NONE", "last_high": None, "prev_high": None, "last_low": None, "prev_low": None}
    last_high_idx, prev_high_idx = prev_highs[-1], prev_highs[-2]
    last_low_idx, prev_low_idx = prev_lows[-1], prev_lows[-2]
    last_high = float(candles[last_high_idx].h)
    prev_high = float(candles[prev_high_idx].h)
    last_low = float(candles[last_low_idx].l)
    prev_low = float(candles[prev_low_idx].l)
    if last_high > prev_high and last_low > prev_low:
        bias = "BULL"
    elif last_high < prev_high and last_low < prev_low:
        bias = "BEAR"
    else:
        bias = "NONE"
    return {
        "bias": bias,
        "last_high": round(last_high, 2),
        "prev_high": round(prev_high, 2),
        "last_low": round(last_low, 2),
        "prev_low": round(prev_low, 2),
    }


def nearest_level(
    direction: str,
    entry_price: float,
    entry_idx: int,
    entry_h4_idx: int,
    c15m: list[Candle],
    c4h: list[Candle],
    highs_15m: list[int],
    lows_15m: list[int],
    highs_4h: list[int],
    lows_4h: list[int],
    lookback_15m: int,
    lookback_4h: int,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    def add_candidate(level: float, source: str, idx: int) -> None:
        if direction == "BULL" and level <= entry_price:
            return
        if direction == "BEAR" and level >= entry_price:
            return
        candidates.append({"level": float(level), "source": source, "idx": int(idx)})

    start_15m = max(0, entry_idx - max(int(lookback_15m), 1))
    if direction == "BULL":
        for idx in highs_15m:
            if start_15m <= idx < entry_idx:
                add_candidate(float(c15m[idx].h), "15m_bsl", idx)
    else:
        for idx in lows_15m:
            if start_15m <= idx < entry_idx:
                add_candidate(float(c15m[idx].l), "15m_ssl", idx)

    start_4h = max(0, entry_h4_idx - max(int(lookback_4h), 1))
    if direction == "BULL":
        for idx in highs_4h:
            if start_4h <= idx <= entry_h4_idx:
                add_candidate(float(c4h[idx].h), "4h_bsl", idx)
    else:
        for idx in lows_4h:
            if start_4h <= idx <= entry_h4_idx:
                add_candidate(float(c4h[idx].l), "4h_ssl", idx)

    if not candidates:
        return {"level": None, "source": None, "distance_pct": None}
    if direction == "BULL":
        best = min(candidates, key=lambda item: item["level"] - entry_price)
        distance_pct = (best["level"] - entry_price) / entry_price * 100.0
    else:
        best = min(candidates, key=lambda item: entry_price - item["level"])
        distance_pct = (entry_price - best["level"]) / entry_price * 100.0
    return {
        "level": round(float(best["level"]), 2),
        "source": best["source"],
        "idx": best["idx"],
        "distance_pct": round(float(distance_pct), 3),
    }


def liquidity_target_rr(target: dict[str, Any], event: dict[str, Any]) -> float | None:
    level = target.get("level")
    if level is None:
        return None
    entry = float(event.get("entry_price", 0.0) or 0.0)
    stop = event.get("initial_stop_price")
    if stop is None:
        return None
    risk = abs(entry - float(stop))
    if risk <= 0:
        return None
    direction = str(event.get("direction") or "")
    if direction == "BULL":
        rr = (float(level) - entry) / risk
    else:
        rr = (entry - float(level)) / risk
    return round(float(rr), 3)


def target_rr_bucket(rr: float | None) -> str:
    if rr is None:
        return "unknown"
    if rr < 1.0:
        return "lt_1r"
    if rr < 1.5:
        return "1_1p5r"
    if rr < 2.0:
        return "1p5_2r"
    return "gte_2r"


def scan_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        swing_n=args.swing_n,
        swing_lookback=args.swing_lookback,
        liquidity_lookback_bars=args.liquidity_lookback_bars,
        mss_lookahead_bars=args.mss_lookahead_bars,
        fvg_lookback_bars=args.fvg_lookback_bars,
        entry_lookahead_bars=args.entry_lookahead_bars,
        outcome_lookahead_bars=args.outcome_lookahead_bars,
        atr_period=args.atr_period,
        min_body_atr=args.min_body_atr,
        min_range_atr=args.min_range_atr,
        stop_buffer_atr=args.stop_buffer_atr,
        target_rr=args.target_rr,
        allow_incomplete_tail=True,
    )


def event_support_context(events_by_direction: dict[str, list[Any]], direction: str, entry_idx: int, lookback: int) -> dict[str, Any]:
    candidates: list[tuple[int, Any]] = []
    for event in events_by_direction.get(direction, []):
        if event.sweep_idx > entry_idx:
            continue
        lag = entry_idx - int(event.sweep_idx)
        if 0 <= lag <= lookback:
            candidates.append((lag, event))
    if not candidates:
        return {
            "recent_sweep": False,
            "recent_sweep_mss": False,
            "sweep_lag_bars": None,
            "event_status": None,
            "event_has_fvg": False,
            "event_retest_confirmed": False,
        }
    lag, event = sorted(candidates, key=lambda item: (item[0], -item[1].sweep_idx))[0]
    mss_available = event.mss_idx is not None and int(event.mss_idx) <= entry_idx
    fvg_available = event.fvg is not None and int(event.fvg.idx) <= entry_idx
    retest_available = event.retest is not None and int(event.retest.idx) <= entry_idx
    status = "sweep_only"
    if mss_available:
        status = "mss_with_fvg" if fvg_available else "mss_no_fvg"
    if retest_available:
        status = "confirmed_retest" if bool(event.retest.confirmed) else "unconfirmed_retest"
    return {
        "recent_sweep": True,
        "recent_sweep_mss": bool(mss_available),
        "sweep_lag_bars": int(lag),
        "event_status": status,
        "event_has_fvg": bool(fvg_available),
        "event_retest_confirmed": bool(retest_available and event.retest and event.retest.confirmed),
        "swept_level": round(float(event.swept_level), 2),
        "sweep_extreme": round(float(event.sweep_extreme), 2),
        "sweep_distance_pct": round(float(event.sweep_distance_pct), 4),
    }


def smc_grade(score: int) -> str:
    if score >= 5:
        return "A_plus_or_A"
    if score == 4:
        return "B_plus"
    if score == 3:
        return "B"
    return "C"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "sum_return_pct": 0.0,
            "compounded_return_pct": 0.0,
            "avg_return_pct": 0.0,
            "profit_factor": 0.0,
        }
    returns = [float(row.get("return", 0.0) or 0.0) for row in rows]
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value <= 0]
    capital = 1.0
    for value in returns:
        capital *= 1.0 + value
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "trades": len(rows),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(rows) * 100.0, 2),
        "sum_return_pct": round(sum(returns) * 100.0, 4),
        "compounded_return_pct": round((capital - 1.0) * 100.0, 4),
        "avg_return_pct": round(sum(returns) / len(rows) * 100.0, 4),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else 0.0,
        "avg_smc_score": round(sum(int(row.get("smc_score", 0)) for row in rows) / len(rows), 3),
    }


def grouped_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(row)
    return {name: summarize_rows(bucket) for name, bucket in sorted(groups.items())}


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_for_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def build_promoted_shadow_events(args: argparse.Namespace) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    base_payload = load_config_payload(Path(args.config))
    payload, pressure_params = apply_pressure_params(base_payload, Path(args.pressure_params))
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=base_payload.get("regime_switcher_thresholds"),
    )
    metrics, engine = run_engine(payload, prepared, args.start_date)
    trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
    initial_capital = float(metrics.get("initial_capital", 1000.0))
    fixed = expansion_overlay(trades, initial_capital, FIXED_STRUCTURE_PARAMS, include_events=True)
    shadow_params = load_best_shadow_params(Path(args.pressure_params))
    shadow = replay_shadow_events(
        fixed["events"],
        initial_capital,
        daily_loss_stop_pct=shadow_params["daily_loss_stop_pct"],
        equity_drawdown_stop_pct=shadow_params["equity_drawdown_stop_pct"],
        consecutive_loss_stop=shadow_params["consecutive_loss_stop"],
        equity_drawdown_cooldown_days=shadow_params["equity_drawdown_cooldown_days"],
    )
    shadow_summary = add_windows(dict(shadow), initial_capital)
    metadata = {
        "engine": metrics,
        "fixed_structure_overlay": {key: value for key, value in fixed.items() if key != "events"},
        "fixed_structure_events": fixed["events"],
        "shadow": shadow_summary,
        "pressure_params": pressure_params,
        "shadow_params": shadow_params,
    }
    return metadata, prepared, shadow, {"config": base_payload}


def annotate_trade(
    event: dict[str, Any],
    prepared: Any,
    daily: list[Candle],
    daily_ts: list[float],
    daily_highs: list[int],
    daily_lows: list[int],
    h4_highs: list[int],
    h4_lows: list[int],
    events_by_direction: dict[str, list[Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    entry_idx = int(event.get("entry_idx"))
    entry_price = float(event.get("entry_price", 0.0) or 0.0)
    direction = str(event.get("direction") or "")
    c15 = prepared.c15m[entry_idx]
    h4_idx = completed_4h_idx_for_entry(prepared.mapping, entry_idx)
    d1_idx = completed_d1_idx_for_entry(daily_ts, c15.ts)

    h4_range = range_context(prepared.c4h, h4_idx, args.h4_range_lookback_bars, entry_price)
    d1_range = range_context(daily, d1_idx, args.d1_range_lookback_bars, entry_price)
    h4_bias = h4_bias_for_idx(prepared.precomputed, h4_idx)
    d1_bias = structure_bias_for_idx(daily, daily_highs, daily_lows, d1_idx)
    sweep_context = event_support_context(events_by_direction, direction, entry_idx, args.recent_sweep_lookback_bars)

    fvg = recent_fvg(prepared.c15m, direction, max(2, entry_idx - args.recent_fvg_lookback_bars), entry_idx)
    fvg_near = bool(fvg and zone_touched(c15, fvg.bottom, fvg.top))
    target = nearest_level(
        direction=direction,
        entry_price=entry_price,
        entry_idx=entry_idx,
        entry_h4_idx=h4_idx,
        c15m=prepared.c15m,
        c4h=prepared.c4h,
        highs_15m=prepared.precomputed.highs_15m,
        lows_15m=prepared.precomputed.lows_15m,
        highs_4h=h4_highs,
        lows_4h=h4_lows,
        lookback_15m=args.liquidity_target_lookback_15m,
        lookback_4h=args.liquidity_target_lookback_4h,
    )
    target_rr = liquidity_target_rr(target, event)
    bucket, ny_time = time_bucket(c15.ts)

    reasons: list[str] = []
    score = 0
    if h4_bias["bias"] == direction or d1_bias["bias"] == direction:
        score += 1
        reasons.append("htf_bias_aligned")
    if pd_favorable(direction, h4_range) or pd_favorable(direction, d1_range):
        score += 1
        reasons.append("premium_discount_favorable")
    if sweep_context["recent_sweep_mss"]:
        score += 1
        reasons.append("recent_sweep_mss")
    elif sweep_context["recent_sweep"]:
        reasons.append("recent_sweep_only")
    if fvg_near or bool(sweep_context.get("event_has_fvg")):
        score += 1
        reasons.append("fvg_context")
    if target_rr is not None and target_rr >= args.min_liquidity_target_rr:
        score += 1
        reasons.append("liquidity_target_room")
    if bucket in {"london_open", "ny_am_killzone", "ny_pm_killzone"}:
        score += 1
        reasons.append("session_window")

    return {
        "entry_time": str(event.get("entry_time")),
        "exit_time": str(event.get("exit_time")),
        "entry_idx": entry_idx,
        "direction": direction,
        "entry_price": round(entry_price, 2),
        "initial_stop_price": event.get("initial_stop_price"),
        "return": float(event.get("return", 0.0) or 0.0),
        "return_pct": round(float(event.get("return", 0.0) or 0.0) * 100.0, 4),
        "exit_reason": event.get("exit_reason"),
        "risk_mode": event.get("risk_mode"),
        "regime_label": event.get("regime_label"),
        "effective_leverage": event.get("effective_leverage"),
        "pressure_target_applied": bool(event.get("pressure_target_applied")),
        "pressure_touch_lock_applied": bool(event.get("pressure_touch_lock_applied")),
        "failed_breakout_guard_applied": bool(event.get("failed_breakout_guard_applied")),
        "h4_idx_closed": h4_idx,
        "d1_idx_closed": d1_idx,
        "h4_bias": h4_bias["bias"],
        "d1_bias": d1_bias["bias"],
        "h4_pd_zone": h4_range["zone"],
        "h4_pd_side": pd_side(direction, h4_range),
        "h4_pd_position_pct": h4_range["position_pct"],
        "h4_range_low": h4_range["range_low"],
        "h4_range_high": h4_range["range_high"],
        "d1_pd_zone": d1_range["zone"],
        "d1_pd_side": pd_side(direction, d1_range),
        "d1_pd_position_pct": d1_range["position_pct"],
        "d1_range_low": d1_range["range_low"],
        "d1_range_high": d1_range["range_high"],
        "recent_sweep": sweep_context["recent_sweep"],
        "recent_sweep_mss": sweep_context["recent_sweep_mss"],
        "recent_sweep_lag_bars": sweep_context["sweep_lag_bars"],
        "recent_sweep_status": sweep_context["event_status"],
        "recent_sweep_has_fvg": sweep_context["event_has_fvg"],
        "recent_fvg_near_entry": fvg_near,
        "liquidity_target_level": target.get("level"),
        "liquidity_target_source": target.get("source"),
        "liquidity_target_distance_pct": target.get("distance_pct"),
        "liquidity_target_rr": target_rr,
        "liquidity_target_rr_bucket": target_rr_bucket(target_rr),
        "session_bucket": bucket,
        "ny_time": ny_time,
        "smc_score": score,
        "smc_grade": smc_grade(score),
        "smc_reasons": reasons,
    }


def main() -> None:
    args = parse_args()
    metadata, prepared, shadow, _payload = build_promoted_shadow_events(args)
    daily = daily_candles_from_4h(prepared.c4h)
    daily_ts = [candle.ts for candle in daily]
    daily_highs, daily_lows = precompute_swings(daily, n=2, lookback=20)
    h4_highs, h4_lows = precompute_swings(prepared.c4h, n=2, lookback=80)
    liquidity_events = scan_events(prepared.c15m, scan_args(args))
    events_by_direction: dict[str, list[Any]] = defaultdict(list)
    for event in liquidity_events:
        events_by_direction[event.direction].append(event)

    fixed_rows = [
        annotate_trade(
            event=event,
            prepared=prepared,
            daily=daily,
            daily_ts=daily_ts,
            daily_highs=daily_highs,
            daily_lows=daily_lows,
            h4_highs=h4_highs,
            h4_lows=h4_lows,
            events_by_direction=events_by_direction,
            args=args,
        )
        for event in metadata["fixed_structure_events"]
        if event.get("entry_idx") is not None
    ]
    shadow_rows = [
        annotate_trade(
            event=event,
            prepared=prepared,
            daily=daily,
            daily_ts=daily_ts,
            daily_highs=daily_highs,
            daily_lows=daily_lows,
            h4_highs=h4_highs,
            h4_lows=h4_lows,
            events_by_direction=events_by_direction,
            args=args,
        )
        for event in shadow["events"]
        if event.get("entry_idx") is not None
    ]

    report = {
        "metadata": {
            "config": str(Path(args.config).resolve()),
            "pressure_params": str(Path(args.pressure_params).resolve()),
            "data_15m": str(Path(args.data_15m).resolve()),
            "data_4h": str(Path(args.data_4h).resolve()),
            "start_date": args.start_date,
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
            "daily_candles": len(daily),
            "liquidity_events": len(liquidity_events),
            "smc_parameters": {
                "h4_range_lookback_bars": args.h4_range_lookback_bars,
                "d1_range_lookback_bars": args.d1_range_lookback_bars,
                "recent_sweep_lookback_bars": args.recent_sweep_lookback_bars,
                "recent_fvg_lookback_bars": args.recent_fvg_lookback_bars,
                "min_liquidity_target_rr": args.min_liquidity_target_rr,
                "htf_context_uses_completed_4h_and_d1": True,
            },
        },
        "promoted_reproduction": {
            "engine": metadata["engine"],
            "fixed_structure_overlay": metadata["fixed_structure_overlay"],
            "fixed_structure_events": metadata["fixed_structure_events"],
            "shadow": metadata["shadow"],
            "shadow_params": metadata["shadow_params"],
        },
        "summary": {
            "all": summarize_rows(shadow_rows),
            "by_smc_score": grouped_summary(shadow_rows, "smc_score"),
            "by_smc_grade": grouped_summary(shadow_rows, "smc_grade"),
            "by_direction": grouped_summary(shadow_rows, "direction"),
            "by_regime_label": grouped_summary(shadow_rows, "regime_label"),
            "by_risk_mode": grouped_summary(shadow_rows, "risk_mode"),
            "by_session_bucket": grouped_summary(shadow_rows, "session_bucket"),
            "by_h4_pd_side": grouped_summary(shadow_rows, "h4_pd_side"),
            "by_d1_pd_side": grouped_summary(shadow_rows, "d1_pd_side"),
            "by_recent_sweep_mss": grouped_summary(shadow_rows, "recent_sweep_mss"),
            "by_liquidity_target_rr_bucket": grouped_summary(shadow_rows, "liquidity_target_rr_bucket"),
        },
        "fixed_summary": {
            "all": summarize_rows(fixed_rows),
            "by_smc_score": grouped_summary(fixed_rows, "smc_score"),
            "by_smc_grade": grouped_summary(fixed_rows, "smc_grade"),
            "by_direction": grouped_summary(fixed_rows, "direction"),
            "by_regime_label": grouped_summary(fixed_rows, "regime_label"),
            "by_risk_mode": grouped_summary(fixed_rows, "risk_mode"),
            "by_session_bucket": grouped_summary(fixed_rows, "session_bucket"),
            "by_h4_pd_side": grouped_summary(fixed_rows, "h4_pd_side"),
            "by_d1_pd_side": grouped_summary(fixed_rows, "d1_pd_side"),
            "by_recent_sweep_mss": grouped_summary(fixed_rows, "recent_sweep_mss"),
            "by_liquidity_target_rr_bucket": grouped_summary(fixed_rows, "liquidity_target_rr_bucket"),
        },
        "rows": shadow_rows,
        "fixed_rows": fixed_rows,
    }

    cleaned = clean_for_json(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    compact = {
        "all": cleaned["summary"]["all"],
        "by_smc_grade": cleaned["summary"]["by_smc_grade"],
        "by_h4_pd_side": cleaned["summary"]["by_h4_pd_side"],
        "by_d1_pd_side": cleaned["summary"]["by_d1_pd_side"],
        "by_liquidity_target_rr_bucket": cleaned["summary"]["by_liquidity_target_rr_bucket"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False))
    if args.stdout:
        print(json.dumps(cleaned, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
