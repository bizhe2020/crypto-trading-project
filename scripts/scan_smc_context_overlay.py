#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_context" / "smc_trade_context_report.json"
DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_context" / "smc_context_overlay_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay SMC context risk multipliers on shadow-accepted trades.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--h4-favorable-multipliers", default="1.0,1.05,1.10,1.15")
    parser.add_argument("--h4-adverse-multipliers", default="1.0,0.95,0.90")
    parser.add_argument("--low-score-multipliers", default="1.0,0.95,0.90")
    parser.add_argument("--london-multipliers", default="1.0,1.05,1.10")
    parser.add_argument("--recent-sweep-mss-multipliers", default="1.0,0.95,0.90")
    parser.add_argument("--max-effective-leverage", type=float, default=8.0)
    parser.add_argument("--live-feasible", action="store_true", help="Apply multipliers to fixed events before shadow replay.")
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def max_drawdown(capitals: list[float], initial: float) -> float:
    peak = initial
    worst = 0.0
    for capital in capitals:
        peak = max(peak, capital)
        if peak > 0:
            worst = max(worst, (peak - capital) / peak * 100.0)
    return round(worst, 2)


def sharpe(returns: list[float]) -> float:
    if len(returns) <= 1:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    std = math.sqrt(variance)
    return round(mean / std * math.sqrt(len(returns)), 3) if std > 0 else 0.0


def replay(rows: list[dict[str, Any]], params: dict[str, float], initial: float = 1000.0) -> dict[str, Any]:
    capital = initial
    capitals: list[float] = []
    returns: list[float] = []
    events: list[dict[str, Any]] = []
    multiplier_counts: dict[str, int] = {}
    adjusted_count = 0
    leverage_delta_sum = 0.0
    for row in rows:
        signal_return = float(row.get("signal_return", 0.0) or 0.0)
        base_leverage = float(row.get("effective_leverage", 0.0) or 0.0)
        multiplier = 1.0
        reasons: list[str] = []
        if row.get("h4_pd_side") == "favorable":
            multiplier *= params["h4_favorable_multiplier"]
            reasons.append("h4_favorable")
        if row.get("h4_pd_side") == "adverse":
            multiplier *= params["h4_adverse_multiplier"]
            reasons.append("h4_adverse")
        if int(row.get("smc_score", 0) or 0) <= 1:
            multiplier *= params["low_score_multiplier"]
            reasons.append("low_score")
        if row.get("session_bucket") == "london_open":
            multiplier *= params["london_multiplier"]
            reasons.append("london_open")
        if bool(row.get("recent_sweep_mss")):
            multiplier *= params["recent_sweep_mss_multiplier"]
            reasons.append("recent_sweep_mss")

        effective_leverage = min(params["max_effective_leverage"], max(0.0, base_leverage * multiplier))
        if abs(effective_leverage - base_leverage) > 1e-9:
            adjusted_count += 1
            leverage_delta_sum += effective_leverage - base_leverage
        trade_return = signal_return * effective_leverage
        capital *= 1.0 + trade_return
        capitals.append(capital)
        returns.append(trade_return)
        for reason in reasons:
            multiplier_counts[reason] = multiplier_counts.get(reason, 0) + 1
        event = dict(row)
        event["base_effective_leverage"] = base_leverage
        event["smc_multiplier"] = round(multiplier, 6)
        event["effective_leverage"] = round(effective_leverage, 6)
        event["return"] = trade_return
        event["capital"] = capital
        events.append(event)

    return {
        "total_return_pct": round((capital - initial) / initial * 100.0, 2),
        "max_drawdown_pct": max_drawdown(capitals, initial),
        "sharpe_ratio": sharpe(returns),
        "trades": len(rows),
        "multiplier_counts": multiplier_counts,
        "adjusted_trades": adjusted_count,
        "leverage_delta_sum": round(leverage_delta_sum, 6),
        "events": events,
    }


def row_multiplier(row: dict[str, Any], params: dict[str, float]) -> tuple[float, list[str]]:
    multiplier = 1.0
    reasons: list[str] = []
    if row.get("h4_pd_side") == "favorable":
        multiplier *= params["h4_favorable_multiplier"]
        reasons.append("h4_favorable")
    if row.get("h4_pd_side") == "adverse":
        multiplier *= params["h4_adverse_multiplier"]
        reasons.append("h4_adverse")
    if int(row.get("smc_score", 0) or 0) <= 1:
        multiplier *= params["low_score_multiplier"]
        reasons.append("low_score")
    if row.get("session_bucket") == "london_open":
        multiplier *= params["london_multiplier"]
        reasons.append("london_open")
    if bool(row.get("recent_sweep_mss")):
        multiplier *= params["recent_sweep_mss_multiplier"]
        reasons.append("recent_sweep_mss")
    return multiplier, reasons


def replay_fixed_before_shadow(
    fixed_events: list[dict[str, Any]],
    rows_by_entry: dict[str, dict[str, Any]],
    params: dict[str, float],
    shadow_params: dict[str, Any],
    initial: float = 1000.0,
) -> dict[str, Any]:
    adjusted_events: list[dict[str, Any]] = []
    multiplier_counts: dict[str, int] = {}
    adjusted_count = 0
    leverage_delta_sum = 0.0
    for event in fixed_events:
        row = rows_by_entry.get(str(event.get("entry_time")))
        adjusted = dict(event)
        if row is None:
            adjusted_events.append(adjusted)
            continue
        base_leverage = float(event.get("effective_leverage", 0.0) or 0.0)
        signal_return = float(event.get("signal_return", 0.0) or 0.0)
        multiplier, reasons = row_multiplier(row, params)
        effective_leverage = min(params["max_effective_leverage"], max(0.0, base_leverage * multiplier))
        if abs(effective_leverage - base_leverage) > 1e-9:
            adjusted_count += 1
            leverage_delta_sum += effective_leverage - base_leverage
        adjusted["effective_leverage"] = round(effective_leverage, 6)
        adjusted["return"] = signal_return * effective_leverage
        adjusted["smc_multiplier"] = round(multiplier, 6)
        adjusted["smc_multiplier_reasons"] = reasons
        adjusted["smc_score"] = row.get("smc_score")
        adjusted["h4_pd_side"] = row.get("h4_pd_side")
        adjusted["session_bucket"] = row.get("session_bucket")
        adjusted_events.append(adjusted)
        for reason in reasons:
            multiplier_counts[reason] = multiplier_counts.get(reason, 0) + 1

    result = replay_shadow_like(
        adjusted_events,
        initial,
        daily_loss_stop_pct=float(shadow_params["daily_loss_stop_pct"]),
        equity_drawdown_stop_pct=float(shadow_params["equity_drawdown_stop_pct"]),
        consecutive_loss_stop=int(shadow_params["consecutive_loss_stop"]),
        equity_drawdown_cooldown_days=int(shadow_params["equity_drawdown_cooldown_days"]),
    )
    result["multiplier_counts"] = multiplier_counts
    result["adjusted_trades"] = adjusted_count
    result["leverage_delta_sum"] = round(leverage_delta_sum, 6)
    return add_windows(result, initial)


def replay_shadow_like(
    events: list[dict[str, Any]],
    initial: float,
    daily_loss_stop_pct: float,
    equity_drawdown_stop_pct: float,
    consecutive_loss_stop: int,
    equity_drawdown_cooldown_days: int,
) -> dict[str, Any]:
    capital = initial
    drawdown_peak = initial
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
        "total_return_pct": round((capital - initial) / initial * 100.0, 2),
        "max_drawdown_pct": max_drawdown(capitals, initial),
        "sharpe_ratio": sharpe(returns),
        "accepted_trades": len(accepted_events),
        "skipped_trades": skipped,
        "trigger_counts": trigger_counts,
        "events": accepted_events,
    }


def replay_window(events: list[dict[str, Any]], initial: float, start: pd.Timestamp) -> dict[str, Any]:
    selected = [event for event in events if pd.Timestamp(event["entry_time"]).tz_convert("UTC") >= start]
    capital = initial
    capitals: list[float] = []
    returns: list[float] = []
    for event in selected:
        trade_return = float(event["return"])
        capital *= 1.0 + trade_return
        capitals.append(capital)
        returns.append(trade_return)
    return {
        "total_return_pct": round((capital - initial) / initial * 100.0, 2),
        "max_drawdown_pct": max_drawdown(capitals, initial),
        "sharpe_ratio": sharpe(returns),
        "trades": len(selected),
    }


def add_windows(result: dict[str, Any], initial: float = 1000.0) -> dict[str, Any]:
    events = result["events"]
    if not events:
        result["windows"] = {}
        return result
    end = max(pd.Timestamp(event["exit_time"]).tz_convert("UTC") for event in events)
    starts = {
        "current_year": pd.Timestamp(f"{end.year}-01-01", tz="UTC"),
        "last_60d": end - pd.Timedelta(days=60),
        "last_30d": end - pd.Timedelta(days=30),
    }
    result["windows"] = {name: replay_window(events, initial, start) for name, start in starts.items()}
    years = sorted({pd.Timestamp(event["entry_time"]).tz_convert("UTC").year for event in events})
    result["yearly"] = {
        str(year): replay_window(events, initial, pd.Timestamp(f"{year}-01-01", tz="UTC"))
        for year in years
    }
    for year in years:
        next_year_start = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        year_events = [
            event
            for event in events
            if pd.Timestamp(event["entry_time"]).tz_convert("UTC") < next_year_start
            and pd.Timestamp(event["entry_time"]).tz_convert("UTC") >= pd.Timestamp(f"{year}-01-01", tz="UTC")
        ]
        result["yearly"][str(year)] = replay_window(year_events, initial, pd.Timestamp(f"{year}-01-01", tz="UTC"))
    return result


def score_result(result: dict[str, Any]) -> float:
    year = result.get("windows", {}).get("current_year", {})
    recent_60d = result.get("windows", {}).get("last_60d", {})
    recent_30d = result.get("windows", {}).get("last_30d", {})
    return round(
        float(result["total_return_pct"])
        + float(year.get("total_return_pct", 0.0)) * 150.0
        + float(recent_60d.get("total_return_pct", 0.0)) * 80.0
        + float(recent_30d.get("total_return_pct", 0.0)) * 30.0
        - float(result["max_drawdown_pct"]) * 30.0
        - float(year.get("max_drawdown_pct", 0.0)) * 40.0,
        4,
    )


def main() -> None:
    args = parse_args()
    source = json.loads(Path(args.input).read_text())
    rows = source["rows"]
    fixed_events = source.get("promoted_reproduction", {}).get("fixed_structure_events", [])
    shadow_params = source.get("promoted_reproduction", {}).get("shadow_params", {})
    fixed_rows = source.get("fixed_rows", rows)
    rows_by_entry = {str(row.get("entry_time")): row for row in fixed_rows}
    for row in rows:
        effective_leverage = float(row.get("effective_leverage", 0.0) or 0.0)
        row["signal_return"] = float(row.get("return", 0.0) or 0.0) / effective_leverage if effective_leverage > 0 else 0.0

    candidates: list[dict[str, Any]] = []
    for values in itertools.product(
        parse_float_list(args.h4_favorable_multipliers),
        parse_float_list(args.h4_adverse_multipliers),
        parse_float_list(args.low_score_multipliers),
        parse_float_list(args.london_multipliers),
        parse_float_list(args.recent_sweep_mss_multipliers),
    ):
        params = {
            "h4_favorable_multiplier": values[0],
            "h4_adverse_multiplier": values[1],
            "low_score_multiplier": values[2],
            "london_multiplier": values[3],
            "recent_sweep_mss_multiplier": values[4],
            "max_effective_leverage": float(args.max_effective_leverage),
        }
        if args.live_feasible:
            result = replay_fixed_before_shadow(fixed_events, rows_by_entry, params, shadow_params)
        else:
            result = add_windows(replay(rows, params))
        result["params"] = params
        result["score"] = score_result(result)
        slim = dict(result)
        slim.pop("events", None)
        candidates.append(slim)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    baseline = source.get("summary", {}).get("all", {})
    report = {
        "input": str(Path(args.input).resolve()),
        "baseline_summary": baseline,
        "candidate_count": len(candidates),
        "top": candidates[: args.top],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    for idx, item in enumerate(candidates[: args.top], start=1):
        year = item.get("windows", {}).get("current_year", {})
        recent_60d = item.get("windows", {}).get("last_60d", {})
        recent_30d = item.get("windows", {}).get("last_30d", {})
        print(
            f"{idx:02d} score={item['score']:.2f} full={item['total_return_pct']:.2f}%/"
            f"{item['max_drawdown_pct']:.2f}% ytd={year.get('total_return_pct', 0.0):.2f}% "
            f"60d={recent_60d.get('total_return_pct', 0.0):.2f}% "
            f"30d={recent_30d.get('total_return_pct', 0.0):.2f}% params={item['params']}"
        )


if __name__ == "__main__":
    main()
