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

from scripts.backtest_config_report import DEFAULT_DATA_15M, DEFAULT_DATA_4H, load_config_payload, load_dataframe  # noqa: E402
from scripts.live_readiness_report import (  # noqa: E402
    PreparedData,
    compact_metrics,
    date_string,
    precompute_regime_state,
    run_engine,
    shadow_risk_gate_overlay,
    trade_dataframe,
)
from strategy.scalp_robust_v2_core import align_timeframes, build_precomputed_state, dataframe_to_candles  # noqa: E402


DEFAULT_END_TS = "2026-04-25 03:45:00+00:00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the fixed best autoTIT + shadow gate result.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-ts", default=DEFAULT_END_TS)
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=21.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=6)
    parser.add_argument("--consecutive-loss-stop", type=int, default=4)
    parser.add_argument("--output", default=str(ROOT / "var" / "live_readiness" / "shadow_gate_best_reproduction.json"))
    parser.add_argument("--stdout-json", action="store_true")
    return parser.parse_args()


def load_prepared_data_at_end(
    data_15m_path: Path,
    data_4h_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    threshold_payload: dict[str, Any] | None,
) -> PreparedData:
    df15 = load_dataframe(data_15m_path, start=start, end=end)
    df4 = load_dataframe(data_4h_path, end=end)
    if df15.empty:
        raise ValueError(f"No 15m data loaded from {data_15m_path}")
    if df4.empty:
        raise ValueError(f"No 4h data loaded from {data_4h_path}")

    actual_end = pd.Timestamp(df15["date"].max()).tz_convert("UTC")
    df4 = df4[df4["date"] <= actual_end].sort_values("date").reset_index(drop=True)
    c4h = dataframe_to_candles(df4)
    c15m = dataframe_to_candles(df15)
    mapping = align_timeframes(c4h, c15m)
    precomputed = build_precomputed_state(c4h, c15m)
    regime_labels, regime_features = precompute_regime_state(c4h, sorted(set(mapping)), threshold_payload)
    return PreparedData(
        c4h=c4h,
        c15m=c15m,
        mapping=mapping,
        precomputed=precomputed,
        start=pd.Timestamp(df15["date"].min()).tz_convert("UTC"),
        end=actual_end,
        regime_labels=regime_labels,
        regime_features=regime_features,
    )


def run_reproduction_case(
    payload: dict[str, Any],
    prepared: PreparedData,
    start_date: str,
    shadow_params: dict[str, Any],
) -> dict[str, Any]:
    metrics, engine = run_engine(payload, prepared, start_date)
    trades = trade_dataframe(engine)
    raw = compact_metrics(metrics)
    overlay = shadow_risk_gate_overlay(
        trades=trades,
        initial_capital=float(metrics.get("initial_capital", 1000.0)),
        daily_loss_stop_pct=float(shadow_params["daily_loss_stop_pct"]),
        equity_drawdown_stop_pct=float(shadow_params["equity_drawdown_stop_pct"]),
        consecutive_loss_stop=int(shadow_params["consecutive_loss_stop"]),
        equity_drawdown_cooldown_days=int(shadow_params["equity_drawdown_cooldown_days"]),
    )
    total_trades = max(1, int(raw.get("total_trades", 0)))
    return {
        "start_date": start_date,
        "raw_autotit": raw,
        "shadow_gate": overlay,
        "skip_ratio": round(float(overlay["skipped_trades"]) / total_trades, 4),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    payload = load_config_payload(config_path)
    start = pd.Timestamp(args.start_date, tz="UTC")
    end = pd.Timestamp(args.end_ts)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")

    shadow_params = {
        "daily_loss_stop_pct": float(args.daily_loss_stop_pct),
        "equity_drawdown_stop_pct": float(args.equity_drawdown_stop_pct),
        "equity_drawdown_cooldown_days": int(args.equity_drawdown_cooldown_days),
        "consecutive_loss_stop": int(args.consecutive_loss_stop),
    }
    prepared = load_prepared_data_at_end(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=start,
        end=end,
        threshold_payload=payload.get("regime_switcher_thresholds"),
    )
    window_starts = {
        "full": args.start_date,
        "current_year": f"{prepared.end.year}-01-01",
        "recent_60d": date_string(max(prepared.start, prepared.end - pd.Timedelta(days=60))),
        "recent_30d": date_string(max(prepared.start, prepared.end - pd.Timedelta(days=30))),
    }
    windows = {
        name: run_reproduction_case(payload, prepared, start_date, shadow_params)
        for name, start_date in window_starts.items()
    }
    report = {
        "config": str(config_path.resolve()),
        "data": {
            "data_15m": str(Path(args.data_15m).resolve()),
            "data_4h": str(Path(args.data_4h).resolve()),
            "start": str(prepared.start),
            "end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
        },
        "windows": window_starts,
        "shadow_params": shadow_params,
        "results": windows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print(output_path)
    print(f"data_end={prepared.end}")
    for name, result in windows.items():
        raw = result["raw_autotit"]
        shadow = result["shadow_gate"]
        print(
            f"{name:12s} raw={raw['total_return_pct']:8.2f}%/{raw['sharpe_ratio']:.3f}/{raw['max_drawdown_pct']:.2f}% "
            f"shadow={shadow['total_return_pct']:8.2f}%/{shadow['sharpe_ratio']:.3f}/{shadow['max_drawdown_pct']:.2f}% "
            f"skipped={shadow['skipped_trades']} skip_ratio={result['skip_ratio']:.2%}"
        )
    if args.stdout_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
