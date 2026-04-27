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

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze 2026 high-growth offense losses in the fixed high-leverage overlay.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument(
        "--pressure-params",
        default=str(DEFAULT_PRESSURE_PARAMS_PATH),
        help="JSON reproduction file with pressure_level_target_cap_params. Use 'none' to skip.",
    )
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--min-effective-leverage", type=float, default=7.5)
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=0)
    parser.add_argument("--output", default=str(ROOT / "var" / "high_leverage_expansion" / "offense_losses_2026.json"))
    return parser.parse_args()


def pct(value: float) -> float:
    return round(value * 100.0, 4)


def risk_price(event: dict[str, Any]) -> float:
    return abs(float(event.get("entry_price") or 0.0) - float(event.get("initial_stop_price") or 0.0))


def excursion_rr(event: dict[str, Any], candles: list[Any], bars: int) -> dict[str, float]:
    entry_idx = event.get("entry_idx")
    risk = risk_price(event)
    if entry_idx is None or risk <= 0:
        return {"mfe_rr": 0.0, "mae_rr": 0.0}
    start = int(entry_idx)
    end = min(len(candles), start + max(1, bars) + 1)
    if start >= end:
        return {"mfe_rr": 0.0, "mae_rr": 0.0}
    entry = float(event.get("entry_price") or 0.0)
    direction = str(event.get("direction") or "")
    window = candles[start:end]
    if direction == "BULL":
        mfe = max(float(candle.h) for candle in window) - entry
        mae = entry - min(float(candle.l) for candle in window)
    else:
        mfe = entry - min(float(candle.l) for candle in window)
        mae = max(float(candle.h) for candle in window) - entry
    return {
        "mfe_rr": round(max(0.0, mfe / risk), 4),
        "mae_rr": round(max(0.0, mae / risk), 4),
    }


def quality_snapshot(event: dict[str, Any]) -> dict[str, Any]:
    direction = str(event.get("direction") or "")
    sign = 1.0 if direction == "BULL" else -1.0
    momentum_pct = float(event.get("feature_momentum", 0.0) or 0.0) * 100.0 * sign
    ema_gap_pct = float(event.get("feature_ema_gap", 0.0) or 0.0) * 100.0 * sign
    structure = bool(event.get("feature_bullish_structure")) if direction == "BULL" else bool(event.get("feature_bearish_structure"))
    return {
        "adx": round(float(event.get("feature_adx", 0.0) or 0.0), 4),
        "directional_momentum_pct": round(momentum_pct, 4),
        "directional_ema_gap_pct": round(ema_gap_pct, 4),
        "directional_structure": structure,
    }


def compact_event(event: dict[str, Any], candles: list[Any]) -> dict[str, Any]:
    payload = {
        "entry_time": event.get("entry_time"),
        "exit_time": event.get("exit_time"),
        "direction": event.get("direction"),
        "return_pct": pct(float(event.get("return", 0.0) or 0.0)),
        "signal_return_pct": pct(float(event.get("signal_return", 0.0) or 0.0)),
        "exit_reason": event.get("exit_reason"),
        "regime_label": event.get("regime_label"),
        "risk_mode": event.get("risk_mode"),
        "effective_leverage": event.get("effective_leverage"),
        "stop_distance_pct": event.get("stop_distance_pct"),
        "reasons": event.get("reasons"),
        "quality": quality_snapshot(event),
        "first_4_bars": excursion_rr(event, candles, 4),
        "first_8_bars": excursion_rr(event, candles, 8),
        "first_16_bars": excursion_rr(event, candles, 16),
    }
    return payload


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {"trades": 0, "return_sum_pct": 0.0, "avg_return_pct": 0.0}
    total = sum(pct(float(event.get("return", 0.0) or 0.0)) for event in events)
    return {
        "trades": len(events),
        "return_sum_pct": round(total, 4),
        "avg_return_pct": round(total / len(events), 4),
        "wins": sum(1 for event in events if float(event.get("return", 0.0) or 0.0) > 0),
        "losses": sum(1 for event in events if float(event.get("return", 0.0) or 0.0) < 0),
    }


def main() -> None:
    args = parse_args()
    payload = load_config_payload(Path(args.config))
    pressure_params: dict[str, Any] = {}
    pressure_params_path: str | None = None
    if str(args.pressure_params).strip().lower() != "none":
        path = Path(args.pressure_params)
        payload, pressure_params = apply_pressure_params(payload, path)
        pressure_params_path = str(path.resolve())
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
    shadow = replay_shadow_events(
        fixed["events"],
        initial_capital,
        daily_loss_stop_pct=float(args.daily_loss_stop_pct),
        equity_drawdown_stop_pct=float(args.equity_drawdown_stop_pct),
        consecutive_loss_stop=int(args.consecutive_loss_stop),
        equity_drawdown_cooldown_days=int(args.equity_drawdown_cooldown_days),
    )

    start = pd.Timestamp(f"{args.year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{args.year + 1}-01-01", tz="UTC")
    events = [
        event
        for event in shadow["events"]
        if start <= pd.Timestamp(event["entry_time"]).tz_convert("UTC") < end
    ]
    offense = [
        event
        for event in events
        if str(event.get("regime_label")) == "high_growth"
        and str(event.get("risk_mode")) == "offense"
        and float(event.get("effective_leverage", 0.0) or 0.0) >= args.min_effective_leverage
    ]
    losses = [event for event in offense if float(event.get("return", 0.0) or 0.0) < 0.0]
    wins = [event for event in offense if float(event.get("return", 0.0) or 0.0) > 0.0]

    report = {
        "config": str(Path(args.config)),
        "pressure_params_path": pressure_params_path,
        "pressure_params": pressure_params,
        "year": args.year,
        "params": {
            "min_effective_leverage": args.min_effective_leverage,
            "daily_loss_stop_pct": args.daily_loss_stop_pct,
            "equity_drawdown_stop_pct": args.equity_drawdown_stop_pct,
            "equity_drawdown_cooldown_days": args.equity_drawdown_cooldown_days,
            "consecutive_loss_stop": args.consecutive_loss_stop,
        },
        "summary": {
            "accepted_events": len(events),
            "high_growth_offense_events": summarize(offense),
            "high_growth_offense_wins": summarize(wins),
            "high_growth_offense_losses": summarize(losses),
        },
        "worst_offense_losses": [compact_event(event, prepared.c15m) for event in sorted(losses, key=lambda item: float(item.get("return", 0.0)))[:10]],
        "best_offense_wins": [compact_event(event, prepared.c15m) for event in sorted(wins, key=lambda item: float(item.get("return", 0.0)), reverse=True)[:10]],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
