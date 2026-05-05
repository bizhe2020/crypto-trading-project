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
from scripts.high_leverage_repro_params import apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import (  # noqa: E402
    load_prepared_data,
    max_drawdown_from_capitals,
    run_engine,
    trade_dataframe,
    trade_return_sharpe,
)
from scripts.report_pa_ict_liquidity_features import LiquidityEvent, scan_events  # noqa: E402
from scripts.report_pa_ict_trade_alignment import match_event, retest_zone_type  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_pa_ict_quality_overlay import parse_float_list, quality_score, select_multiplier  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "shadow_overlay" / "pa_ict_shadow_quality_overlay_scan.json"
DEFAULT_BEST_PARAMS = ROOT / "config" / "high_leverage_pressure_target_cap_best.params.json"
DEFAULT_DATA_15M = ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"
DEFAULT_DATA_4H = ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply PA/ICT quality-score risk multipliers to promoted high-leverage shadow overlay events."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--best-params", default=str(DEFAULT_BEST_PARAMS))
    parser.add_argument("--pressure-params", default=str(DEFAULT_BEST_PARAMS))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
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
    parser.add_argument(
        "--activation-modes",
        default="all,shadow_defense,weak_no_pressure,defense_or_low_no_pressure",
        help="Comma list: all, shadow_defense, weak_or_defense, weak_no_pressure, low_no_pressure, defense_or_low_no_pressure",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args()


def load_best_params(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def promoted_fixed_params(best_payload: dict[str, Any]) -> dict[str, Any]:
    params = best_payload.get("fixed_structure_overlay_params")
    return dict(params) if isinstance(params, dict) else dict(FIXED_STRUCTURE_PARAMS)


def promoted_shadow_params(best_payload: dict[str, Any]) -> dict[str, Any]:
    params = best_payload.get("shadow_gate_scan_params")
    if not isinstance(params, dict):
        params = {}
    return {
        "daily_loss_stop_pct": float(params.get("daily_loss_stop_pct", 6.0) or 0.0),
        "equity_drawdown_stop_pct": float(params.get("equity_drawdown_stop_pct", 15.0) or 0.0),
        "equity_drawdown_cooldown_days": int(params.get("equity_drawdown_cooldown_days", 2) or 0),
        "consecutive_loss_stop": int(params.get("consecutive_loss_stop", 0) or 0),
    }


def apply_best_pressure_params(payload: dict[str, Any], path_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if str(path_text).strip().lower() == "none":
        return payload, {}
    return apply_pressure_params(payload, Path(path_text))


def events_by_direction(events: list[LiquidityEvent]) -> dict[str, list[LiquidityEvent]]:
    out: dict[str, list[LiquidityEvent]] = {"BULL": [], "BEAR": []}
    for event in events:
        out.setdefault(event.direction, []).append(event)
    return out


def add_strategy_trade_fields(events: list[dict[str, Any]], trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty or "entry_idx" not in trades.columns:
        return events
    by_entry_idx: dict[int, pd.Series] = {}
    for _, trade in trades.iterrows():
        entry_idx = trade.get("entry_idx")
        if pd.isna(entry_idx):
            continue
        by_entry_idx[int(entry_idx)] = trade
    fields = [
        "risk_regime",
        "time_based_trailing_enabled",
        "auto_tit_reason",
        "pressure_target_applied",
        "pressure_target_source",
        "pressure_target_level",
        "pressure_target_rr",
        "pressure_target_min_rr",
        "pressure_target_dynamic_reason",
        "pressure_touch_lock_applied",
        "pressure_touch_lock_source",
        "pressure_touch_lock_level",
        "pressure_touch_lock_rr",
    ]
    out: list[dict[str, Any]] = []
    for event in events:
        row = by_entry_idx.get(int(event["entry_idx"])) if event.get("entry_idx") is not None else None
        enriched = dict(event)
        if row is not None:
            for field in fields:
                value = row.get(field)
                if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
                    value = None
                enriched[field] = value
        out.append(enriched)
    return out


def add_pa_ict_fields(
    events: list[dict[str, Any]],
    pa_events: dict[str, list[LiquidityEvent]],
    *,
    lookback_bars: int,
    require_confirmed_retest: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events:
        matched, lag = match_event(
            event,
            pa_events,
            lookback_bars=lookback_bars,
            require_confirmed_retest=require_confirmed_retest,
        )
        enriched = dict(event)
        enriched["pa_ict_supported"] = matched is not None
        enriched["pa_ict_lag_bars"] = lag
        if matched is not None:
            enriched.update(
                {
                    "pa_ict_sweep_time": matched.sweep_time,
                    "pa_ict_mss_time": matched.mss_time,
                    "pa_ict_retest_time": matched.retest.timestamp if matched.retest else None,
                    "pa_ict_time_bucket": matched.time_bucket,
                    "pa_ict_zone_type": retest_zone_type(matched),
                    "pa_ict_status": matched.status,
                }
            )
        return_event = enriched
        out.append(return_event)
    return out


def has_pressure_protection(event: dict[str, Any]) -> bool:
    return bool(event.get("pressure_target_applied")) or bool(event.get("pressure_touch_lock_applied"))


def weak_context(event: dict[str, Any]) -> bool:
    risk_mode = str(event.get("risk_mode") or "")
    risk_regime = str(event.get("risk_regime") or "")
    regime_label = str(event.get("regime_label") or "")
    return (
        risk_mode == "defense"
        or risk_regime in {"bull_weak", "bear_weak", "bear_strong"}
        or regime_label == "high_growth"
    )


def activation_allows(event: dict[str, Any], mode: str, score: float, low_threshold: float) -> bool:
    if mode == "all":
        return True
    if mode == "shadow_defense":
        return str(event.get("risk_mode") or "") == "defense"
    if mode == "weak_or_defense":
        return weak_context(event)
    if mode == "weak_no_pressure":
        return weak_context(event) and not has_pressure_protection(event)
    if mode == "low_no_pressure":
        return score <= low_threshold and not has_pressure_protection(event)
    if mode == "defense_or_low_no_pressure":
        return str(event.get("risk_mode") or "") == "defense" or (
            score <= low_threshold and not has_pressure_protection(event)
        )
    raise ValueError(f"Unsupported activation mode: {mode}")


def profit_factor(pnls: list[float]) -> float:
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = abs(sum(value for value in pnls if value <= 0))
    return round(gross_profit / gross_loss, 3) if gross_loss > 0 else 0.0


def replay_window(events: list[dict[str, Any]], initial_capital: float, start: pd.Timestamp) -> dict[str, Any]:
    selected = [event for event in events if pd.Timestamp(event["entry_time"]).tz_convert("UTC") >= start]
    capital = initial_capital
    capitals: list[float] = []
    returns: list[float] = []
    for event in selected:
        trade_return = float(event["return"])
        capital = max(0.0, capital * (1.0 + trade_return))
        capitals.append(capital)
        returns.append(trade_return)
    return {
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100.0, 2),
        "final_capital": round(capital, 2),
        "sharpe_ratio": round(trade_return_sharpe(returns), 3),
        "max_drawdown_pct": round(max_drawdown_from_capitals(capitals, initial_capital), 2),
        "trades": len(selected),
    }


def add_windows(result: dict[str, Any], events: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
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


def replay_shadow_quality(
    events: list[dict[str, Any]],
    initial_capital: float,
    shadow_params: dict[str, Any],
    quality_params: dict[str, float | str],
) -> dict[str, Any]:
    capital = initial_capital
    drawdown_peak = initial_capital
    loss_streak = 0
    pause_until = pd.Timestamp.min.tz_localize("UTC")
    day_start_capital: dict[pd.Timestamp, float] = {}
    day_pnl: dict[pd.Timestamp, float] = {}
    capitals: list[float] = []
    returns: list[float] = []
    pnls: list[float] = []
    accepted_events: list[dict[str, Any]] = []
    skipped = 0
    trigger_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {"high": 0, "mid": 0, "low": 0, "inactive": 0}
    activation_count = 0

    for event in events:
        entry_time = pd.Timestamp(event["entry_time"]).tz_convert("UTC")
        exit_time = pd.Timestamp(event["exit_time"]).tz_convert("UTC")
        if entry_time < pause_until:
            skipped += 1
            continue

        score, reasons = quality_score(event)
        bucket, multiplier = select_multiplier(score, quality_params)  # type: ignore[arg-type]
        activation_mode = str(quality_params["activation_mode"])
        if not activation_allows(event, activation_mode, score, float(quality_params["low_score_threshold"])):
            bucket = "inactive"
            multiplier = 1.0
        else:
            activation_count += 1
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        capital_before = capital
        original_return = float(event["return"])
        trade_return = original_return * float(multiplier)
        pnl = capital_before * trade_return
        capital = max(0.0, capital + pnl)
        returns.append(trade_return)
        pnls.append(pnl)
        capitals.append(capital)

        accepted = dict(event)
        accepted["original_return"] = original_return
        accepted["return"] = trade_return
        accepted["quality_score"] = score
        accepted["quality_bucket"] = bucket
        accepted["quality_reasons"] = reasons
        accepted["quality_activation_mode"] = activation_mode
        accepted["risk_multiplier"] = float(multiplier)
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
        daily_loss_stop_pct = float(shadow_params["daily_loss_stop_pct"])
        if daily_loss_stop_pct > 0 and day_start_capital[exit_day] > 0:
            daily_loss_pct = -day_pnl[exit_day] / day_start_capital[exit_day] * 100.0
            if daily_loss_pct >= daily_loss_stop_pct:
                triggered.append("daily_loss")
                pause_until = max(pause_until, exit_day + pd.Timedelta(days=1))
        consecutive_loss_stop = int(shadow_params["consecutive_loss_stop"])
        if consecutive_loss_stop > 0 and loss_streak >= consecutive_loss_stop:
            triggered.append("consecutive_loss")
            pause_until = max(pause_until, exit_day + pd.Timedelta(days=1))
            loss_streak = 0
        equity_drawdown_stop_pct = float(shadow_params["equity_drawdown_stop_pct"])
        if equity_drawdown_stop_pct > 0 and drawdown_peak > 0:
            drawdown_pct = (drawdown_peak - capital) / drawdown_peak * 100.0
            if drawdown_pct >= equity_drawdown_stop_pct:
                triggered.append("equity_drawdown")
                pause_until = max(
                    pause_until,
                    exit_day + pd.Timedelta(days=int(shadow_params["equity_drawdown_cooldown_days"])),
                )
                drawdown_peak = capital
                loss_streak = 0
        for reason in triggered:
            trigger_counts[reason] = trigger_counts.get(reason, 0) + 1

    wins = sum(1 for pnl in pnls if pnl > 0)
    result = {
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100.0, 2),
        "final_capital": round(capital, 2),
        "sharpe_ratio": round(trade_return_sharpe(returns), 3),
        "max_drawdown_pct": round(max_drawdown_from_capitals(capitals, initial_capital), 2),
        "accepted_trades": len(accepted_events),
        "skipped_trades": skipped,
        "win_rate": round(wins / len(pnls) * 100.0, 2) if pnls else 0.0,
        "profit_factor": profit_factor(pnls),
        "trigger_counts": trigger_counts,
        "bucket_counts": bucket_counts,
        "quality_activation_count": activation_count,
        "params": quality_params,
    }
    return add_windows(result, accepted_events, initial_capital) | {"events": accepted_events}


def score_result(result: dict[str, Any]) -> float:
    current_year = result.get("windows", {}).get("current_year", {})
    recent_60d = result.get("windows", {}).get("last_60d", {})
    recent_30d = result.get("windows", {}).get("last_30d", {})
    return round(
        float(result["total_return_pct"])
        + float(current_year.get("total_return_pct", 0.0)) * 160.0
        + float(recent_60d.get("total_return_pct", 0.0)) * 90.0
        + float(recent_30d.get("total_return_pct", 0.0)) * 50.0
        - float(result["max_drawdown_pct"]) * 30.0
        - float(current_year.get("max_drawdown_pct", 0.0)) * 40.0,
        4,
    )


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    windows = result.get("windows", {})
    return {
        "score": result.get("score"),
        "total_return_pct": result["total_return_pct"],
        "max_drawdown_pct": result["max_drawdown_pct"],
        "accepted_trades": result["accepted_trades"],
        "skipped_trades": result["skipped_trades"],
        "quality_activation_count": result["quality_activation_count"],
        "bucket_counts": result["bucket_counts"],
        "current_year": windows.get("current_year", {}),
        "last_60d": windows.get("last_60d", {}),
        "last_30d": windows.get("last_30d", {}),
        "params": result["params"],
    }


def build_promoted_events(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], float, dict[str, Any]]:
    best_payload = load_best_params(Path(args.best_params))
    payload = load_config_payload(Path(args.config))
    payload, pressure_params = apply_best_pressure_params(payload, args.pressure_params)
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=payload.get("regime_switcher_thresholds"),
    )
    metrics, engine = run_engine(payload, prepared, args.start_date)
    trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
    initial_capital = float(metrics.get("initial_capital", 1000.0) or 1000.0)
    fixed_params = promoted_fixed_params(best_payload)
    fixed = expansion_overlay(trades, initial_capital, fixed_params, include_events=True)
    events = add_strategy_trade_fields(fixed["events"], trades)

    pa_events = events_by_direction(scan_events(prepared.c15m, args))
    events = add_pa_ict_fields(
        events,
        pa_events,
        lookback_bars=args.support_lookback_bars,
        require_confirmed_retest=bool(args.require_confirmed_retest),
    )
    metadata = {
        "config": str(Path(args.config).resolve()),
        "best_params": str(Path(args.best_params).resolve()),
        "pressure_params_path": None if str(args.pressure_params).lower() == "none" else str(Path(args.pressure_params).resolve()),
        "pressure_params": pressure_params,
        "data": {
            "start": str(prepared.start),
            "end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
        },
        "engine": metrics,
        "fixed_structure_overlay": {key: value for key, value in fixed.items() if key != "events"},
        "fixed_structure_params": fixed_params,
        "pa_ict_events": sum(len(items) for items in pa_events.values()),
    }
    return events, metadata, initial_capital, promoted_shadow_params(best_payload)


def main() -> None:
    args = parse_args()
    events, metadata, initial_capital, shadow_params = build_promoted_events(args)
    baseline_params = {
        "high_score_threshold": 999.0,
        "low_score_threshold": -999.0,
        "high_risk_multiplier": 1.0,
        "mid_risk_multiplier": 1.0,
        "low_risk_multiplier": 1.0,
        "activation_mode": "all",
    }
    baseline = replay_shadow_quality(events, initial_capital, shadow_params, baseline_params)
    baseline["score"] = score_result(baseline)

    candidates: list[dict[str, Any]] = []
    for high_threshold, low_threshold, high_mult, mid_mult, low_mult, activation_mode in itertools.product(
        parse_float_list(args.high_score_thresholds),
        parse_float_list(args.low_score_thresholds),
        parse_float_list(args.high_risk_multipliers),
        parse_float_list(args.mid_risk_multipliers),
        parse_float_list(args.low_risk_multipliers),
        parse_str_list(args.activation_modes),
    ):
        if low_threshold >= high_threshold:
            continue
        params: dict[str, float | str] = {
            "high_score_threshold": high_threshold,
            "low_score_threshold": low_threshold,
            "high_risk_multiplier": high_mult,
            "mid_risk_multiplier": mid_mult,
            "low_risk_multiplier": low_mult,
            "activation_mode": activation_mode,
        }
        result = replay_shadow_quality(events, initial_capital, shadow_params, params)
        result["score"] = score_result(result)
        candidates.append(result)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    top = candidates[: max(int(args.top), 1)]
    report: dict[str, Any] = {
        "metadata": metadata,
        "shadow_params": shadow_params,
        "scan_parameters": {
            "start_date": args.start_date,
            "support_lookback_bars": args.support_lookback_bars,
            "require_confirmed_retest": args.require_confirmed_retest,
            "high_score_thresholds": args.high_score_thresholds,
            "low_score_thresholds": args.low_score_thresholds,
            "high_risk_multipliers": args.high_risk_multipliers,
            "mid_risk_multipliers": args.mid_risk_multipliers,
            "low_risk_multipliers": args.low_risk_multipliers,
            "activation_modes": args.activation_modes,
        },
        "baseline": {key: value for key, value in baseline.items() if key != "events"},
        "top": [{key: value for key, value in item.items() if key != "events"} for item in top],
    }
    if args.include_events:
        report["events"] = events
        report["baseline_events"] = baseline["events"]
        report["top_events"] = top[0]["events"] if top else []

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    compact = {
        "baseline": compact_result(baseline),
        "top": [compact_result(item) for item in top[:5]],
    }
    print(output)
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    if args.stdout:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
