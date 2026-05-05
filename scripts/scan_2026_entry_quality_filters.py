#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine  # noqa: E402
from scripts.live_readiness_report import trade_dataframe, trade_return_sharpe, max_drawdown_from_capitals  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, add_windows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan post-overlay entry-quality skip rules on promoted high-leverage events.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", default=str(ROOT / "var" / "high_leverage_expansion" / "entry_quality_filter_scan.json"))
    return parser.parse_args()


def directional_feature(event: dict[str, Any], key: str) -> float:
    direction = str(event.get("direction") or "")
    sign = 1.0 if direction == "BULL" else -1.0
    return float(event.get(key, 0.0) or 0.0) * sign


def event_matches_filter(event: dict[str, Any], rule: dict[str, Any]) -> bool:
    if not bool(rule.get("enabled", True)):
        return False
    direction_set = rule.get("directions")
    if direction_set and str(event.get("direction") or "") not in direction_set:
        return False
    regime_set = rule.get("regimes")
    if regime_set and str(event.get("regime_label") or "") not in regime_set:
        return False
    risk_mode_set = rule.get("risk_modes")
    if risk_mode_set and str(event.get("risk_mode") or "") not in risk_mode_set:
        return False
    max_effective_leverage = rule.get("max_effective_leverage")
    if max_effective_leverage is not None and float(event.get("effective_leverage", 0.0) or 0.0) > float(max_effective_leverage):
        return False
    min_effective_leverage = rule.get("min_effective_leverage")
    if min_effective_leverage is not None and float(event.get("effective_leverage", 0.0) or 0.0) < float(min_effective_leverage):
        return False
    adx_lt = rule.get("adx_lt")
    if adx_lt is not None and float(event.get("feature_adx", 0.0) or 0.0) >= float(adx_lt):
        return False
    adx_gt = rule.get("adx_gt")
    if adx_gt is not None and float(event.get("feature_adx", 0.0) or 0.0) <= float(adx_gt):
        return False
    momentum_lt = rule.get("directional_momentum_lt")
    if momentum_lt is not None and directional_feature(event, "feature_momentum") >= float(momentum_lt) / 100.0:
        return False
    ema_gap_lt = rule.get("directional_ema_gap_lt")
    if ema_gap_lt is not None and directional_feature(event, "feature_ema_gap") >= float(ema_gap_lt) / 100.0:
        return False
    no_pressure_only = bool(rule.get("no_pressure_only", False))
    if no_pressure_only and (event.get("pressure_target_applied") or event.get("pressure_touch_lock_applied")):
        return False
    return True


def replay_events_with_filter(
    events: list[dict[str, Any]],
    initial_capital: float,
    rule: dict[str, Any],
    *,
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
    skipped_by_shadow = 0
    skipped_by_filter = 0
    skipped_filter_return_sum = 0.0
    capitals: list[float] = []
    returns: list[float] = []
    trigger_counts: dict[str, int] = {}

    for event in events:
        entry_time = pd.Timestamp(event["entry_time"]).tz_convert("UTC")
        exit_time = pd.Timestamp(event["exit_time"]).tz_convert("UTC")
        if entry_time < pause_until:
            skipped_by_shadow += 1
            continue
        if event_matches_filter(event, rule):
            skipped_by_filter += 1
            skipped_filter_return_sum += float(event.get("return", 0.0) or 0.0)
            continue

        exit_day = exit_time.normalize()
        if exit_day not in day_start_capital:
            day_start_capital[exit_day] = capital
            day_pnl[exit_day] = 0.0

        capital_before = capital
        trade_return = float(event["return"])
        pnl = capital_before * trade_return
        capital += pnl
        day_pnl[exit_day] += pnl
        returns.append(trade_return)
        capitals.append(capital)
        accepted = dict(event)
        accepted["shadow_capital"] = capital
        accepted_events.append(accepted)
        if pnl > 0:
            loss_streak = 0
        else:
            loss_streak += 1
        drawdown_peak = max(drawdown_peak, capital)

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

    result = {
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100.0, 2),
        "final_capital": round(capital, 2),
        "sharpe_ratio": round(trade_return_sharpe(returns), 3),
        "max_drawdown_pct": round(max_drawdown_from_capitals(capitals, initial_capital), 2),
        "accepted_trades": len(accepted_events),
        "skipped_shadow_trades": skipped_by_shadow,
        "skipped_filter_trades": skipped_by_filter,
        "skipped_filter_return_sum_pct": round(skipped_filter_return_sum * 100.0, 4),
        "trigger_counts": trigger_counts,
        "events": accepted_events,
    }
    return add_windows(result, initial_capital)


def candidate_rules() -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = [{"name": "baseline_no_extra_filter", "enabled": False}]
    for adx_lt in (18.0, 20.0, 22.0):
        rules.append(
            {
                "name": f"skip_normal_offense_bull_adx_lt_{adx_lt:g}",
                "directions": ["BULL"],
                "regimes": ["normal"],
                "risk_modes": ["offense"],
                "adx_lt": adx_lt,
                "no_pressure_only": True,
            }
        )
    for momentum_lt in (-1.0, 0.0, 1.0, 2.0):
        rules.append(
            {
                "name": f"skip_defense_bull_momentum_lt_{momentum_lt:g}",
                "directions": ["BULL"],
                "risk_modes": ["defense"],
                "directional_momentum_lt": momentum_lt,
                "no_pressure_only": True,
            }
        )
    for ema_lt in (-0.5, 0.0, 0.5, 1.0):
        rules.append(
            {
                "name": f"skip_defense_bull_ema_lt_{ema_lt:g}",
                "directions": ["BULL"],
                "risk_modes": ["defense"],
                "directional_ema_gap_lt": ema_lt,
                "no_pressure_only": True,
            }
        )
    for adx_lt, momentum_lt in ((20.0, 3.0), (22.0, 3.0), (30.0, 4.0), (35.0, 5.0)):
        rules.append(
            {
                "name": f"skip_high_growth_bull_weak_adx_{adx_lt:g}_mom_{momentum_lt:g}",
                "directions": ["BULL"],
                "regimes": ["high_growth"],
                "adx_lt": adx_lt,
                "directional_momentum_lt": momentum_lt,
                "no_pressure_only": True,
            }
        )
    rules.append(
        {
            "name": "skip_all_no_pressure_defense_bull",
            "directions": ["BULL"],
            "risk_modes": ["defense"],
            "no_pressure_only": True,
        }
    )
    return rules


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


def compact(result: dict[str, Any]) -> str:
    y = result.get("windows", {}).get("current_year", {})
    w60 = result.get("windows", {}).get("last_60d", {})
    w30 = result.get("windows", {}).get("last_30d", {})
    return (
        f"score={result['score']:.2f} full={result['total_return_pct']:.2f}%/{result['max_drawdown_pct']:.2f}% "
        f"2026={y.get('total_return_pct', 0.0):.2f}%/{y.get('max_drawdown_pct', 0.0):.2f}% "
        f"60d={w60.get('total_return_pct', 0.0):.2f}% 30d={w30.get('total_return_pct', 0.0):.2f}% "
        f"skip_filter={result['skipped_filter_trades']} skip_ret={result['skipped_filter_return_sum_pct']:.2f}%"
    )


def main() -> None:
    args = parse_args()
    payload = load_config_payload(Path(args.config))
    pressure_params_path = None
    pressure_params: dict[str, Any] = {}
    if str(args.pressure_params).strip().lower() != "none":
        pressure_params_path = str(Path(args.pressure_params).resolve())
        payload, pressure_params = apply_pressure_params(payload, Path(args.pressure_params))
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

    results: list[dict[str, Any]] = []
    for rule in candidate_rules():
        result = replay_events_with_filter(
            events,
            initial_capital,
            rule,
            daily_loss_stop_pct=float(args.daily_loss_stop_pct),
            equity_drawdown_stop_pct=float(args.equity_drawdown_stop_pct),
            consecutive_loss_stop=int(args.consecutive_loss_stop),
            equity_drawdown_cooldown_days=int(args.equity_drawdown_cooldown_days),
        )
        result["rule"] = rule
        result["score"] = score_result(result)
        results.append(result)
        print(f"{compact(result)} rule={rule['name']}", flush=True)

    results.sort(key=lambda item: item["score"], reverse=True)
    report = {
        "config": str(Path(args.config).resolve()),
        "pressure_params_path": pressure_params_path,
        "pressure_params": pressure_params,
        "engine": {
            "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 2),
            "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0)), 2),
            "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
            "total_trades": int(metrics.get("total_trades", 0)),
        },
        "fixed_structure_overlay": {key: value for key, value in fixed.items() if key != "events"},
        "candidate_count": len(results),
        "top": results[: args.top],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)
    for index, result in enumerate(results[: args.top], start=1):
        print(f"{index:02d} {compact(result)} rule={result['rule']['name']}")


if __name__ == "__main__":
    main()
