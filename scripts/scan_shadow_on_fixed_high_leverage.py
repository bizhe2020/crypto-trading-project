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
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_return_sharpe, max_drawdown_from_capitals, trade_dataframe  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay, parse_float_list, parse_int_list  # noqa: E402


FIXED_STRUCTURE_PARAMS: dict[str, Any] = {
    "base_leverage": 4.0,
    "high_growth_leverage": 7.5,
    "tight_stop_leverage": 8.0,
    "recovery_leverage": 2.0,
    "drawdown_leverage": 2.0,
    "unhealthy_leverage": 2.0,
    "tight_stop_pct": 1.25,
    "max_stop_distance_pct": 1.5,
    "high_growth_max_stop_distance_pct": 2.0,
    "wide_stop_mode": "all_healthy",
    "max_effective_leverage": 8.0,
    "loss_streak_threshold": 3,
    "win_streak_threshold": 2,
    "drawdown_threshold_pct": 20.0,
    "health_lookback_trades": 6,
    "health_min_unit_return_pct": 0.0,
    "health_min_win_rate_pct": 25.0,
    "state_lookback_trades": 8,
    "defense_enter_unit_return_pct": -2.0,
    "defense_enter_win_rate_pct": 20.0,
    "offense_enter_unit_return_pct": -0.5,
    "offense_enter_win_rate_pct": 40.0,
    "reattack_lookback_trades": 2,
    "reattack_unit_return_pct": 0.5,
    "reattack_win_rate_pct": 33.0,
    "reattack_signal_mode": "high_growth_or_tight_or_structure",
    "price_structure_reattack_mode": "none",
    "structure_reattack_min_momentum_pct": 0.0,
    "structure_reattack_min_ema_gap_pct": 0.25,
    "structure_reattack_min_adx": 0.0,
    "defense_leverage": 2.0,
    "defense_max_stop_distance_pct": 1.5,
    "defense_structure_max_stop_distance_pct": 1.9,
    "failed_breakout_guard_enabled": True,
    "failed_breakout_guard_leverage": 2.0,
    "failed_breakout_guard_min_leverage": 7.5,
    "failed_breakout_guard_min_quality_score": 2,
    "failed_breakout_guard_min_momentum_pct": 6.0,
    "failed_breakout_guard_min_ema_gap_pct": 2.0,
    "failed_breakout_guard_min_adx": 38.0,
    "failed_breakout_guard_regime_labels": ["high_growth"],
    "failed_breakout_guard_risk_modes": ["offense"],
    "failed_breakout_guard_directions": ["BULL"],
    "min_liq_buffer_pct": 1.2,
    "maintenance_margin_pct": 0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan shadow gate on the fixed structure high-leverage overlay.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--daily-loss-values", default="0,4,6,8,10,12")
    parser.add_argument("--equity-dd-values", default="0,15,18,21,25,30")
    parser.add_argument("--equity-cooldown-values", default="0,3,6,10")
    parser.add_argument("--loss-streak-values", default="0,3,4,5,6")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", default=str(ROOT / "var" / "high_leverage_expansion" / "shadow_on_fixed_structure_scan.json"))
    return parser.parse_args()


def replay_shadow_events(
    events: list[dict[str, Any]],
    initial_capital: float,
    daily_loss_stop_pct: float,
    equity_drawdown_stop_pct: float,
    consecutive_loss_stop: int,
    equity_drawdown_cooldown_days: int,
) -> dict[str, Any]:
    capital = initial_capital
    drawdown_peak = initial_capital
    loss_streak = 0
    pause_until = pd.Timestamp.min.tz_localize("UTC")
    day_start_capital: dict[pd.Timestamp, float] = {}
    day_pnl: dict[pd.Timestamp, float] = {}
    accepted_events: list[dict[str, Any]] = []
    capitals: list[float] = []
    returns: list[float] = []
    skipped = 0
    trigger_counts: dict[str, int] = {}

    for event in events:
        entry_time = pd.Timestamp(event["entry_time"]).tz_convert("UTC")
        exit_time = pd.Timestamp(event["exit_time"]).tz_convert("UTC")
        if entry_time < pause_until:
            skipped += 1
            continue

        capital_before = capital
        trade_return = float(event["return"])
        pnl = capital_before * trade_return
        capital += pnl
        returns.append(trade_return)
        capitals.append(capital)
        accepted = dict(event)
        accepted["shadow_capital"] = capital
        accepted_events.append(accepted)
        drawdown_peak = max(drawdown_peak, capital)

        exit_day = exit_time.normalize()
        if exit_day not in day_start_capital:
            day_start_capital[exit_day] = capital_before
            day_pnl[exit_day] = 0.0
        day_pnl[exit_day] += pnl

        if pnl > 0:
            loss_streak = 0
        else:
            loss_streak += 1

        triggered: list[str] = []
        if daily_loss_stop_pct > 0 and day_start_capital[exit_day] > 0:
            daily_loss_pct = -day_pnl[exit_day] / day_start_capital[exit_day] * 100.0
            if daily_loss_pct >= daily_loss_stop_pct:
                triggered.append("daily_loss")
                pause_until = max(pause_until, exit_day + pd.Timedelta(days=1))
        if consecutive_loss_stop > 0 and loss_streak >= consecutive_loss_stop:
            triggered.append("consecutive_loss")
            pause_until = max(pause_until, exit_day + pd.Timedelta(days=1))
            loss_streak = 0
        if equity_drawdown_stop_pct > 0 and drawdown_peak > 0:
            drawdown_pct = (drawdown_peak - capital) / drawdown_peak * 100.0
            if drawdown_pct >= equity_drawdown_stop_pct:
                triggered.append("equity_drawdown")
                pause_until = max(pause_until, exit_day + pd.Timedelta(days=equity_drawdown_cooldown_days))
                drawdown_peak = capital
                loss_streak = 0
        for reason in triggered:
            trigger_counts[reason] = trigger_counts.get(reason, 0) + 1

    return {
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100.0, 2),
        "final_capital": round(capital, 2),
        "sharpe_ratio": round(trade_return_sharpe(returns), 3),
        "max_drawdown_pct": round(max_drawdown_from_capitals(capitals, initial_capital), 2),
        "accepted_trades": len(accepted_events),
        "skipped_trades": skipped,
        "trigger_counts": trigger_counts,
        "events": accepted_events,
    }


def replay_window(events: list[dict[str, Any]], initial_capital: float, start: pd.Timestamp) -> dict[str, Any]:
    selected = [event for event in events if pd.Timestamp(event["entry_time"]).tz_convert("UTC") >= start]
    capital = initial_capital
    capitals: list[float] = []
    returns: list[float] = []
    for event in selected:
        trade_return = float(event["return"])
        capital *= 1.0 + trade_return
        capitals.append(capital)
        returns.append(trade_return)
    return {
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100.0, 2),
        "sharpe_ratio": round(trade_return_sharpe(returns), 3),
        "max_drawdown_pct": round(max_drawdown_from_capitals(capitals, initial_capital), 2),
        "trades": len(selected),
    }


def add_windows(result: dict[str, Any], initial_capital: float) -> dict[str, Any]:
    events = result.pop("events")
    if not events:
        result["windows"] = {}
        return result
    exits = [pd.Timestamp(event["exit_time"]).tz_convert("UTC") for event in events]
    end = max(exits)
    starts = {
        "current_year": pd.Timestamp(f"{end.year}-01-01", tz="UTC"),
        "last_60d": end - pd.Timedelta(days=60),
        "last_30d": end - pd.Timedelta(days=30),
    }
    result["windows"] = {name: replay_window(events, initial_capital, start) for name, start in starts.items()}
    return result


def main() -> None:
    args = parse_args()
    payload = load_config_payload(Path(args.config))
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
    events = fixed["events"]

    candidates: list[dict[str, Any]] = []
    for daily_loss, equity_dd, cooldown, loss_streak in itertools.product(
        parse_float_list(args.daily_loss_values),
        parse_float_list(args.equity_dd_values),
        parse_int_list(args.equity_cooldown_values),
        parse_int_list(args.loss_streak_values),
    ):
        if equity_dd <= 0 and cooldown > 0:
            continue
        if equity_dd > 0 and cooldown <= 0:
            continue
        result = replay_shadow_events(
            events,
            initial_capital,
            daily_loss_stop_pct=daily_loss,
            equity_drawdown_stop_pct=equity_dd,
            consecutive_loss_stop=loss_streak,
            equity_drawdown_cooldown_days=cooldown,
        )
        result = add_windows(result, initial_capital)
        year = result.get("windows", {}).get("current_year", {})
        recent_60d = result.get("windows", {}).get("last_60d", {})
        result["params"] = {
            "daily_loss_stop_pct": daily_loss,
            "equity_drawdown_stop_pct": equity_dd,
            "equity_drawdown_cooldown_days": cooldown,
            "consecutive_loss_stop": loss_streak,
        }
        result["score"] = round(
            float(result["total_return_pct"])
            + float(year.get("total_return_pct", 0.0)) * 150.0
            + float(recent_60d.get("total_return_pct", 0.0)) * 80.0
            - float(result["max_drawdown_pct"]) * 30.0
            - float(year.get("max_drawdown_pct", 0.0)) * 40.0,
            4,
        )
        candidates.append(result)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    report = {
        "config": str(Path(args.config).resolve()),
        "data": {
            "start": str(prepared.start),
            "end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
        },
        "fixed_structure_overlay": {k: v for k, v in fixed.items() if k != "events"},
        "candidate_count": len(candidates),
        "top": candidates[: args.top],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)
    for idx, item in enumerate(candidates[: args.top], start=1):
        year = item.get("windows", {}).get("current_year", {})
        recent_60d = item.get("windows", {}).get("last_60d", {})
        print(
            f"{idx:02d} score={item['score']:.2f} full={item['total_return_pct']:.2f}%/"
            f"{item['max_drawdown_pct']:.2f}% ytd={year.get('total_return_pct', 0.0):.2f}%/"
            f"{year.get('max_drawdown_pct', 0.0):.2f}% 60d={recent_60d.get('total_return_pct', 0.0):.2f}% "
            f"skip={item['skipped_trades']} params={item['params']}"
        )


if __name__ == "__main__":
    main()
