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
    parser = argparse.ArgumentParser(description="Build a 2026 pressure-level loss-bucket report for the high-leverage overlay.")
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
    parser.add_argument("--future-lookahead-bars", type=int, default=32)
    parser.add_argument("--early-cap-missed-rr", type=float, default=1.0)
    parser.add_argument("--late-cap-mfe-rr", type=float, default=1.0)
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=0)
    parser.add_argument("--output", default=str(ROOT / "var" / "high_leverage_expansion" / "pressure_loss_buckets_2026.json"))
    return parser.parse_args()


def pct(value: float) -> float:
    return round(value * 100.0, 4)


def source_class(event: dict[str, Any]) -> str:
    source = str(event.get("pressure_target_source") or event.get("pressure_touch_lock_source") or "")
    if source.startswith("round_"):
        return "integer_level"
    if source in {"swing_high", "swing_low", "volume_cluster"}:
        return "pressure_level"
    return "no_pressure_event"


def risk_price(event: dict[str, Any]) -> float:
    try:
        return abs(float(event.get("entry_price") or 0.0) - float(event.get("initial_stop_price") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def favorable_rr_until_exit(event: dict[str, Any], candles: list[Any]) -> float:
    entry_idx = event.get("entry_idx")
    exit_idx = event.get("exit_idx")
    risk = risk_price(event)
    if entry_idx is None or exit_idx is None or risk <= 0:
        return 0.0
    start = max(0, int(entry_idx))
    end = min(len(candles) - 1, int(exit_idx))
    if end < start:
        return 0.0
    entry = float(event.get("entry_price") or 0.0)
    direction = str(event.get("direction") or "")
    window = candles[start : end + 1]
    if direction == "BULL":
        high = max(float(candle.h) for candle in window)
        return max(0.0, (high - entry) / risk)
    low = min(float(candle.l) for candle in window)
    return max(0.0, (entry - low) / risk)


def future_missed_rr(event: dict[str, Any], candles: list[Any], lookahead_bars: int) -> float:
    exit_idx = event.get("exit_idx")
    risk = risk_price(event)
    if exit_idx is None or risk <= 0:
        return 0.0
    start = int(exit_idx) + 1
    end = min(len(candles), start + max(0, lookahead_bars))
    if start >= end:
        return 0.0
    exit_price = float(event.get("exit_price") or 0.0)
    direction = str(event.get("direction") or "")
    window = candles[start:end]
    if direction == "BULL":
        high = max(float(candle.h) for candle in window)
        return max(0.0, (high - exit_price) / risk)
    low = min(float(candle.l) for candle in window)
    return max(0.0, (exit_price - low) / risk)


def add_bucket(buckets: dict[str, dict[str, Any]], name: str, event: dict[str, Any]) -> None:
    bucket = buckets.setdefault(name, {"trades": 0, "return_sum_pct": 0.0, "avg_return_pct": 0.0})
    bucket["trades"] += 1
    bucket["return_sum_pct"] += pct(float(event.get("return", 0.0) or 0.0))


def classify_primary_loss(event: dict[str, Any], mfe_rr: float, late_cap_mfe_rr: float) -> str:
    exit_reason = str(event.get("exit_reason") or "")
    if exit_reason == "time_stop_exit":
        return "time_stop_loss"
    if bool(event.get("pressure_target_applied")) and exit_reason == "stop_loss":
        return "late_cap_failed_after_target_loss"
    if not bool(event.get("pressure_target_applied")) and mfe_rr >= late_cap_mfe_rr:
        return "late_cap_no_target_mfe_loss"
    klass = source_class(event)
    if klass == "integer_level":
        return "integer_level_loss"
    if klass == "pressure_level":
        return "pressure_level_loss"
    if exit_reason == "stop_loss":
        return "plain_stop_loss"
    return f"other_{exit_reason or 'unknown'}_loss"


def summarize_buckets(buckets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, bucket in sorted(buckets.items()):
        trades = int(bucket["trades"])
        total = round(float(bucket["return_sum_pct"]), 4)
        out[name] = {
            "trades": trades,
            "return_sum_pct": total,
            "avg_return_pct": round(total / trades, 4) if trades else 0.0,
        }
    return out


def compact_event(event: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "entry_time": event.get("entry_time"),
        "exit_time": event.get("exit_time"),
        "direction": event.get("direction"),
        "return_pct": pct(float(event.get("return", 0.0) or 0.0)),
        "exit_reason": event.get("exit_reason"),
        "regime_label": event.get("regime_label"),
        "source_class": source_class(event),
        "pressure_target_applied": bool(event.get("pressure_target_applied")),
        "pressure_target_source": event.get("pressure_target_source"),
        "pressure_target_rr": event.get("pressure_target_rr"),
        "pressure_target_min_rr": event.get("pressure_target_min_rr"),
        "pressure_target_dynamic_reason": event.get("pressure_target_dynamic_reason"),
        "pressure_touch_lock_applied": bool(event.get("pressure_touch_lock_applied")),
        "pressure_touch_lock_source": event.get("pressure_touch_lock_source"),
        "effective_leverage": event.get("effective_leverage"),
        "risk_mode": event.get("risk_mode"),
        "feature_adx": event.get("feature_adx"),
        "feature_momentum": event.get("feature_momentum"),
        "feature_ema_gap": event.get("feature_ema_gap"),
    }
    if extra:
        payload.update(extra)
    return payload


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    payload = load_config_payload(config_path)
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
    losses = [event for event in events if float(event.get("return", 0.0) or 0.0) < 0.0]

    primary_buckets: dict[str, dict[str, Any]] = {}
    source_buckets: dict[str, dict[str, Any]] = {}
    cap_timing_buckets: dict[str, dict[str, Any]] = {}
    loss_samples: list[dict[str, Any]] = []
    early_cap_events: list[dict[str, Any]] = []

    for event in losses:
        mfe_rr = favorable_rr_until_exit(event, prepared.c15m)
        primary = classify_primary_loss(event, mfe_rr, args.late_cap_mfe_rr)
        add_bucket(primary_buckets, primary, event)
        add_bucket(source_buckets, source_class(event), event)
        if primary.startswith("late_cap"):
            add_bucket(cap_timing_buckets, primary, event)
        elif str(event.get("exit_reason") or "") == "time_stop_exit":
            add_bucket(cap_timing_buckets, "time_stop_loss", event)
        elif bool(event.get("pressure_target_applied")):
            add_bucket(cap_timing_buckets, "target_applied_but_loss", event)
        else:
            add_bucket(cap_timing_buckets, "no_target_cap_loss", event)
        loss_samples.append(compact_event(event, {"primary_bucket": primary, "mfe_rr": round(mfe_rr, 4)}))

    for event in events:
        if not bool(event.get("pressure_target_applied")):
            continue
        if str(event.get("exit_reason") or "") != "target_rr":
            continue
        missed_rr = future_missed_rr(event, prepared.c15m, args.future_lookahead_bars)
        if missed_rr >= args.early_cap_missed_rr:
            early_cap_events.append(compact_event(event, {"future_missed_rr": round(missed_rr, 4)}))

    loss_samples.sort(key=lambda item: float(item["return_pct"]))
    early_cap_events.sort(key=lambda item: float(item["future_missed_rr"]), reverse=True)
    report = {
        "config": str(config_path),
        "pressure_params_path": pressure_params_path,
        "pressure_params": pressure_params,
        "year": args.year,
        "window": {"start": str(start), "end": str(end)},
        "data": {
            "start": str(prepared.start),
            "end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
        },
        "params": {
            "future_lookahead_bars": args.future_lookahead_bars,
            "early_cap_missed_rr": args.early_cap_missed_rr,
            "late_cap_mfe_rr": args.late_cap_mfe_rr,
            "daily_loss_stop_pct": args.daily_loss_stop_pct,
            "equity_drawdown_stop_pct": args.equity_drawdown_stop_pct,
            "equity_drawdown_cooldown_days": args.equity_drawdown_cooldown_days,
            "consecutive_loss_stop": args.consecutive_loss_stop,
        },
        "summary": {
            "accepted_events": len(events),
            "loss_events": len(losses),
            "loss_return_sum_pct": round(sum(pct(float(event.get("return", 0.0) or 0.0)) for event in losses), 4),
            "early_cap_opportunity_events": len(early_cap_events),
        },
        "primary_loss_buckets": summarize_buckets(primary_buckets),
        "source_loss_buckets": summarize_buckets(source_buckets),
        "cap_timing_buckets": summarize_buckets(cap_timing_buckets),
        "worst_loss_samples": loss_samples[:10],
        "early_cap_opportunity_samples": early_cap_events[:10],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(report["primary_loss_buckets"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
