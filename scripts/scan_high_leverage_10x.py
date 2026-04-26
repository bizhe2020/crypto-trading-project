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
    compact_metrics,
    high_leverage_guard_overlay,
    load_prepared_data,
    shadow_risk_gate_overlay,
    trade_dataframe,
    worst_trade_streak,
    run_engine,
)


DEFAULT_OUTPUT_DIR = ROOT / "var" / "high_leverage_scan"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan 10x high leverage research parameters.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.research.10x.json"))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--risk-per-trade", default="0.0125,0.02,0.03,0.04")
    parser.add_argument("--position-size-pct", default="0.5,0.75,1.0")
    parser.add_argument("--max-stop-distance-pct", default="2.0,2.5,3.0")
    parser.add_argument("--max-account-effective-leverage", default="5.0,7.5,10.0")
    parser.add_argument("--slippage-bps", default="7.5,10.0")
    parser.add_argument("--max-drawdown-pct", type=float, default=30.0)
    parser.add_argument("--min-trades", type=int, default=50)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args()


def update_regime_risk(payload: dict[str, Any], risk_per_trade: float) -> None:
    base = risk_per_trade
    payload["risk_per_trade"] = base
    payload["bull_strong_long_risk_per_trade"] = min(base * 1.6, 0.08)
    payload["bull_strong_short_risk_per_trade"] = min(base * 0.4, 0.03)
    payload["bull_weak_long_risk_per_trade"] = min(base * 0.9, 0.05)
    payload["bull_weak_short_risk_per_trade"] = min(base * 0.7, 0.04)
    payload["bear_weak_long_risk_per_trade"] = min(base * 0.7, 0.04)
    payload["bear_weak_short_risk_per_trade"] = min(base * 0.8, 0.04)
    payload["bear_strong_long_risk_per_trade"] = min(base * 0.4, 0.03)
    payload["bear_strong_short_risk_per_trade"] = min(base * 0.6, 0.04)


def candidate_payloads(base_payload: dict[str, Any], args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    risks = parse_float_list(args.risk_per_trade)
    sizes = parse_float_list(args.position_size_pct)
    stop_caps = parse_float_list(args.max_stop_distance_pct)
    leverage_caps = parse_float_list(args.max_account_effective_leverage)
    slippages = parse_float_list(args.slippage_bps)
    candidates: list[tuple[str, dict[str, Any]]] = []
    for risk, size, stop_cap, leverage_cap, slippage in itertools.product(
        risks,
        sizes,
        stop_caps,
        leverage_caps,
        slippages,
    ):
        payload = dict(base_payload)
        payload["mode"] = "paper"
        payload["leverage"] = 10
        payload["position_size_pct"] = size
        payload["slippage_bps"] = slippage
        payload["enable_high_leverage_guard"] = True
        payload["high_leverage_guard_min_leverage"] = 10.0
        payload["high_leverage_max_stop_distance_pct"] = stop_cap
        payload["high_leverage_max_account_effective_leverage"] = leverage_cap
        payload["high_leverage_min_liquidation_buffer_pct"] = float(
            base_payload.get("high_leverage_min_liquidation_buffer_pct", 1.2) or 1.2
        )
        payload["high_leverage_maintenance_margin_pct"] = float(
            base_payload.get("high_leverage_maintenance_margin_pct", 0.5) or 0.5
        )
        update_regime_risk(payload, risk)
        label = (
            f"risk={risk:.4f}_size={size:.2f}_stop={stop_cap:.1f}_"
            f"maxlev={leverage_cap:.1f}_slip={slippage:.1f}"
        )
        candidates.append((label, payload))
    return candidates


def score_case(raw: dict[str, Any], high_guard: dict[str, Any]) -> float:
    total_return = float(high_guard.get("total_return_pct", 0.0) or 0.0)
    drawdown = float(high_guard.get("max_drawdown_pct", 0.0) or 0.0)
    skipped = float(high_guard.get("skipped_trades", 0.0) or 0.0)
    raw_drawdown = float(raw.get("max_drawdown_pct", 0.0) or 0.0)
    return total_return - drawdown * 2.0 - raw_drawdown * 0.5 - skipped * 0.1


def run_candidate(
    label: str,
    payload: dict[str, Any],
    prepared: Any,
    start_date: str,
    max_drawdown_pct: float,
    min_trades: int,
) -> dict[str, Any]:
    metrics, engine = run_engine(payload, prepared, start_date)
    trades = trade_dataframe(engine)
    raw = compact_metrics(metrics)
    high_guard = high_leverage_guard_overlay(trades, float(metrics.get("initial_capital", 1000.0)), payload)
    shadow = shadow_risk_gate_overlay(
        trades=trades,
        initial_capital=float(metrics.get("initial_capital", 1000.0)),
        daily_loss_stop_pct=float(payload.get("shadow_daily_loss_stop_pct", 3.0) or 0.0),
        equity_drawdown_stop_pct=float(payload.get("shadow_equity_drawdown_stop_pct", 12.0) or 0.0),
        consecutive_loss_stop=int(payload.get("shadow_consecutive_loss_stop", 2) or 0),
        equity_drawdown_cooldown_days=int(payload.get("shadow_equity_drawdown_cooldown_days", 6) or 0),
    )
    accepted = int(high_guard.get("accepted_trades", 0) or 0)
    drawdown = float(high_guard.get("max_drawdown_pct", 0.0) or 0.0)
    passes = accepted >= min_trades and drawdown <= max_drawdown_pct
    return {
        "label": label,
        "passes_constraints": passes,
        "score": round(score_case(raw, high_guard), 4),
        "params": {
            "risk_per_trade": payload["risk_per_trade"],
            "position_size_pct": payload["position_size_pct"],
            "slippage_bps": payload["slippage_bps"],
            "high_leverage_max_stop_distance_pct": payload["high_leverage_max_stop_distance_pct"],
            "high_leverage_max_account_effective_leverage": payload["high_leverage_max_account_effective_leverage"],
        },
        "raw": raw,
        "high_leverage_guard_overlay": high_guard,
        "shadow_risk_gate_overlay": shadow,
        "worst_loss_streak": worst_trade_streak(trades),
    }


def output_path_for(output_dir: Path, start_date: str, end: pd.Timestamp) -> Path:
    return output_dir / f"high_leverage_10x_scan_{start_date}_to_{end.strftime('%Y-%m-%d')}.json"


def main() -> None:
    args = parse_args()
    base_payload = load_config_payload(Path(args.config))
    prepared = load_prepared_data(
        Path(args.data_15m),
        Path(args.data_4h),
        pd.Timestamp(args.start_date, tz="UTC"),
        base_payload.get("regime_switcher_thresholds"),
    )
    results = []
    candidates = candidate_payloads(base_payload, args)
    for idx, (label, payload) in enumerate(candidates, start=1):
        print(f"[{idx}/{len(candidates)}] {label}", flush=True)
        results.append(
            run_candidate(
                label=label,
                payload=payload,
                prepared=prepared,
                start_date=args.start_date,
                max_drawdown_pct=args.max_drawdown_pct,
                min_trades=args.min_trades,
            )
        )
    ranked = sorted(
        results,
        key=lambda row: (bool(row["passes_constraints"]), float(row["score"])),
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
        "constraints": {
            "max_drawdown_pct": args.max_drawdown_pct,
            "min_trades": args.min_trades,
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
