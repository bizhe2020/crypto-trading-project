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

from scripts.live_readiness_report import max_drawdown_from_capitals, trade_return_sharpe  # noqa: E402
from scripts.report_pa_ict_liquidity_features import scan_events  # noqa: E402
from scripts.report_pa_ict_trade_alignment import (  # noqa: E402
    compact_trade,
    load_config,
    load_market_data,
    match_event,
    parse_timestamp,
    run_strategy,
)
from strategy.scalp_robust_v2_core import dataframe_to_candles  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "var" / "pa_ict_liquidity"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan PA/ICT quality-score overlays as risk multipliers on current strategy trades."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "config.paper.high-leverage-structure.json"))
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--support-lookback-bars", type=int, default=96)
    parser.add_argument("--require-confirmed-retest", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--high-score-thresholds", default="2.0,3.0")
    parser.add_argument("--low-score-thresholds", default="-1.0,0.0,1.0")
    parser.add_argument("--high-risk-multipliers", default="1.0")
    parser.add_argument("--mid-risk-multipliers", default="1.0")
    parser.add_argument("--low-risk-multipliers", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--include-trades", action="store_true")
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args()


def profit_factor(pnls: list[float]) -> float:
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = abs(sum(value for value in pnls if value <= 0))
    return round(gross_profit / gross_loss, 3) if gross_loss > 0 else 0.0


def retest_zone_score(zone_type: str | None) -> float:
    if zone_type == "fvg_and_ote":
        return 2.0
    if zone_type == "fvg_only":
        return 0.75
    if zone_type == "ote_only":
        return -1.0
    return -0.25


def time_bucket_score(bucket: str | None) -> float:
    return {
        "asia_evening_ny": 1.0,
        "london_open": 0.5,
        "ny_am_killzone": 0.25,
        "ny_pm_killzone": -0.25,
        "ny_lunch": -1.0,
        "other": -1.0,
    }.get(str(bucket or "none"), 0.0)


def regime_score(regime_label: str | None, risk_regime: str | None, direction: str | None) -> float:
    score = {
        "flat": 1.0,
        "normal": 0.0,
        "high_growth": -0.5,
    }.get(str(regime_label or "none"), 0.0)
    score += {
        "bull_strong": 1.0,
        "bull_weak": -0.75,
        "bear_strong": -0.75,
        "bear_weak": -0.25,
    }.get(str(risk_regime or "none"), 0.0)
    if direction == "BEAR":
        score -= 0.5
    return score


def quality_score(trade: dict[str, Any]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    zone_type = trade.get("pa_ict_zone_type")
    zone_delta = retest_zone_score(str(zone_type) if zone_type is not None else None)
    score += zone_delta
    reasons.append(f"zone={zone_type or 'none'}:{zone_delta:+.2f}")

    bucket = trade.get("pa_ict_time_bucket")
    time_delta = time_bucket_score(str(bucket) if bucket is not None else None)
    score += time_delta
    reasons.append(f"time={bucket or 'none'}:{time_delta:+.2f}")

    regime_delta = regime_score(
        trade.get("regime_label"),
        trade.get("risk_regime"),
        trade.get("direction"),
    )
    score += regime_delta
    reasons.append(f"regime={trade.get('regime_label') or 'none'}/{trade.get('risk_regime') or 'none'}:{regime_delta:+.2f}")

    if bool(trade.get("pressure_target_applied")):
        score += 1.0
        reasons.append("pressure_target:+1.00")
    if bool(trade.get("pressure_touch_lock_applied")):
        score += 0.5
        reasons.append("pressure_touch_lock:+0.50")
    if bool(trade.get("time_based_trailing_enabled")):
        score += 0.25
        reasons.append("auto_tit:+0.25")
    return round(score, 4), reasons


def select_multiplier(score: float, params: dict[str, float]) -> tuple[str, float]:
    if score >= params["high_score_threshold"]:
        return "high", params["high_risk_multiplier"]
    if score <= params["low_score_threshold"]:
        return "low", params["low_risk_multiplier"]
    return "mid", params["mid_risk_multiplier"]


def summarize_replay(events: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    capital = initial_capital
    capitals: list[float] = []
    returns: list[float] = []
    pnls: list[float] = []
    wins = 0
    active = 0
    bucket_counts: dict[str, int] = {"high": 0, "mid": 0, "low": 0}
    bucket_pnl: dict[str, float] = {"high": 0.0, "mid": 0.0, "low": 0.0}
    for event in events:
        trade_return = float(event.get("weighted_return", 0.0) or 0.0)
        capital_before = capital
        pnl = capital_before * trade_return
        capital += pnl
        capitals.append(capital)
        returns.append(trade_return)
        pnls.append(pnl)
        bucket = str(event.get("quality_bucket") or "mid")
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        bucket_pnl[bucket] = bucket_pnl.get(bucket, 0.0) + pnl
        if float(event.get("risk_multiplier", 0.0) or 0.0) > 0:
            active += 1
        if pnl > 0:
            wins += 1
    return {
        "initial_capital": round(initial_capital, 2),
        "final_capital": round(capital, 2),
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100.0, 2),
        "max_drawdown_pct": round(max_drawdown_from_capitals(capitals, initial_capital), 2),
        "sharpe": round(trade_return_sharpe(returns), 3),
        "trades": len(events),
        "active_trades": active,
        "win_rate": round(wins / len(events) * 100.0, 2) if events else 0.0,
        "profit_factor": profit_factor(pnls),
        "bucket_counts": bucket_counts,
        "bucket_pnl": {key: round(value, 2) for key, value in sorted(bucket_pnl.items())},
    }


def replay_window(events: list[dict[str, Any]], initial_capital: float, start: pd.Timestamp) -> dict[str, Any]:
    selected = [event for event in events if parse_timestamp(str(event["exit_time"])) >= start]
    return summarize_replay(selected, initial_capital)


def add_windows(metrics: dict[str, Any], events: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    if not events:
        metrics["windows"] = {}
        return metrics
    last_exit = max(parse_timestamp(str(event["exit_time"])) for event in events)
    starts = {
        "current_year": pd.Timestamp(f"{last_exit.year}-01-01", tz="UTC"),
        "last_60d": last_exit - pd.Timedelta(days=60),
        "last_30d": last_exit - pd.Timedelta(days=30),
    }
    metrics["windows"] = {name: replay_window(events, initial_capital, start) for name, start in starts.items()}
    return metrics


def replay_overlay(trades: list[dict[str, Any]], initial_capital: float, params: dict[str, float]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for trade in trades:
        score, reasons = quality_score(trade)
        bucket, multiplier = select_multiplier(score, params)
        raw_return = float(trade.get("pnl_pct", 0.0) or 0.0)
        events.append(
            {
                "entry_time": trade.get("entry_time"),
                "exit_time": trade.get("exit_time"),
                "direction": trade.get("direction"),
                "exit_reason": trade.get("exit_reason"),
                "raw_return": raw_return,
                "weighted_return": raw_return * multiplier,
                "risk_multiplier": multiplier,
                "quality_score": score,
                "quality_bucket": bucket,
                "quality_reasons": reasons,
                "regime_label": trade.get("regime_label"),
                "risk_regime": trade.get("risk_regime"),
                "pa_ict_zone_type": trade.get("pa_ict_zone_type"),
                "pa_ict_time_bucket": trade.get("pa_ict_time_bucket"),
                "pressure_target_applied": trade.get("pressure_target_applied"),
                "pressure_touch_lock_applied": trade.get("pressure_touch_lock_applied"),
            }
        )
    metrics = summarize_replay(events, initial_capital)
    metrics = add_windows(metrics, events, initial_capital)
    metrics["params"] = params
    metrics["recent_events"] = events[-10:]
    return metrics


def objective(metrics: dict[str, Any]) -> float:
    current_year = metrics.get("windows", {}).get("current_year", {})
    last_60d = metrics.get("windows", {}).get("last_60d", {})
    last_30d = metrics.get("windows", {}).get("last_30d", {})
    return (
        float(metrics.get("total_return_pct", 0.0) or 0.0)
        + float(current_year.get("total_return_pct", 0.0) or 0.0) * 2.0
        + float(last_60d.get("total_return_pct", 0.0) or 0.0)
        + float(last_30d.get("total_return_pct", 0.0) or 0.0)
        - float(metrics.get("max_drawdown_pct", 0.0) or 0.0) * 5.0
    )


def build_aligned_trades(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = load_config(Path(args.config))
    df15, df4 = load_market_data(config, args)
    candles = dataframe_to_candles(df15)
    events = scan_events(candles, args)
    events_by_direction = {"BULL": [], "BEAR": []}
    for event in events:
        events_by_direction.setdefault(event.direction, []).append(event)
    engine, backtest_metrics = run_strategy(config, df15, df4, args.start_date)
    aligned: list[dict[str, Any]] = []
    for trade in engine.trades:
        payload = trade.__dict__.copy()
        event, lag = match_event(
            payload,
            events_by_direction,
            lookback_bars=args.support_lookback_bars,
            require_confirmed_retest=bool(args.require_confirmed_retest),
        )
        aligned.append(compact_trade(payload, event, lag))
    metadata = {
        "rows_15m": len(df15),
        "rows_4h": len(df4),
        "events": len(events),
        "backtest_metrics": backtest_metrics,
    }
    return aligned, metadata


def output_path_for(output_dir: Path, start_date: str, require_confirmed_retest: bool) -> Path:
    mode = "confirmed" if require_confirmed_retest else "mss_or_retest"
    return output_dir / f"pa_ict_quality_overlay_scan_{start_date}_{mode}.json"


def main() -> None:
    args = parse_args()
    trades, metadata = build_aligned_trades(args)
    initial_capital = float(metadata["backtest_metrics"].get("initial_capital", 1000.0) or 1000.0)
    baseline_params = {
        "high_score_threshold": 999.0,
        "low_score_threshold": -999.0,
        "high_risk_multiplier": 1.0,
        "mid_risk_multiplier": 1.0,
        "low_risk_multiplier": 1.0,
    }
    baseline = replay_overlay(trades, initial_capital, baseline_params)

    candidates: list[dict[str, Any]] = []
    for high_threshold, low_threshold, high_mult, mid_mult, low_mult in itertools.product(
        parse_float_list(args.high_score_thresholds),
        parse_float_list(args.low_score_thresholds),
        parse_float_list(args.high_risk_multipliers),
        parse_float_list(args.mid_risk_multipliers),
        parse_float_list(args.low_risk_multipliers),
    ):
        if low_threshold >= high_threshold:
            continue
        params = {
            "high_score_threshold": high_threshold,
            "low_score_threshold": low_threshold,
            "high_risk_multiplier": high_mult,
            "mid_risk_multiplier": mid_mult,
            "low_risk_multiplier": low_mult,
        }
        result = replay_overlay(trades, initial_capital, params)
        result["objective"] = round(objective(result), 4)
        candidates.append(result)

    top = sorted(candidates, key=lambda item: item["objective"], reverse=True)[: max(args.top, 1)]
    report: dict[str, Any] = {
        "config": str(Path(args.config)),
        "parameters": {
            "start_date": args.start_date,
            "support_lookback_bars": args.support_lookback_bars,
            "require_confirmed_retest": args.require_confirmed_retest,
            "high_score_thresholds": args.high_score_thresholds,
            "low_score_thresholds": args.low_score_thresholds,
            "high_risk_multipliers": args.high_risk_multipliers,
            "mid_risk_multipliers": args.mid_risk_multipliers,
            "low_risk_multipliers": args.low_risk_multipliers,
        },
        "metadata": metadata,
        "baseline": baseline,
        "top": top,
    }
    if args.include_trades:
        report["trades"] = trades

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_path_for(output_dir, args.start_date, bool(args.require_confirmed_retest))
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output_path)
    compact = {
        "baseline": {
            "total_return_pct": baseline["total_return_pct"],
            "max_drawdown_pct": baseline["max_drawdown_pct"],
            "current_year": baseline["windows"]["current_year"]["total_return_pct"],
            "last_60d": baseline["windows"]["last_60d"]["total_return_pct"],
            "last_30d": baseline["windows"]["last_30d"]["total_return_pct"],
        },
        "top": [
            {
                "objective": item["objective"],
                "params": item["params"],
                "total_return_pct": item["total_return_pct"],
                "max_drawdown_pct": item["max_drawdown_pct"],
                "current_year": item["windows"]["current_year"]["total_return_pct"],
                "last_60d": item["windows"]["last_60d"]["total_return_pct"],
                "last_30d": item["windows"]["last_30d"]["total_return_pct"],
                "bucket_counts": item["bucket_counts"],
                "bucket_pnl": item["bucket_pnl"],
            }
            for item in top[:5]
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    if args.stdout:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
