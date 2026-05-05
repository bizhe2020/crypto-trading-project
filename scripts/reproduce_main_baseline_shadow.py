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
from scripts.report_smc_trade_context import load_best_shadow_params  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, add_windows, replay_shadow_events  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "main_baseline_shadow_events.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce promoted main strategy accepted shadow events.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    if pd.isna(value):
        return None
    return value


def main() -> None:
    args = parse_args()
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
    fixed = expansion_overlay(trades, float(metrics.get("initial_capital", 1000.0)), FIXED_STRUCTURE_PARAMS, include_events=True)
    shadow_params = load_best_shadow_params(Path(args.pressure_params))
    shadow = replay_shadow_events(
        fixed["events"],
        float(metrics.get("initial_capital", 1000.0)),
        daily_loss_stop_pct=shadow_params["daily_loss_stop_pct"],
        equity_drawdown_stop_pct=shadow_params["equity_drawdown_stop_pct"],
        consecutive_loss_stop=shadow_params["consecutive_loss_stop"],
        equity_drawdown_cooldown_days=shadow_params["equity_drawdown_cooldown_days"],
    )
    shadow_with_windows = add_windows(dict(shadow), float(metrics.get("initial_capital", 1000.0)))
    report = {
        "metadata": {
            "config": str(Path(args.config).resolve()),
            "pressure_params": str(Path(args.pressure_params).resolve()),
            "start_date": args.start_date,
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
            "pressure_params_applied": pressure_params,
            "shadow_params": shadow_params,
        },
        "summary": shadow_with_windows,
        "events": shadow["events"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    print(
        json.dumps(
            {
                "overall": {
                    "total_return_pct": shadow_with_windows["total_return_pct"],
                    "max_drawdown_pct": shadow_with_windows["max_drawdown_pct"],
                    "accepted_trades": shadow_with_windows["accepted_trades"],
                    "skipped_trades": shadow_with_windows["skipped_trades"],
                },
                "windows": shadow_with_windows["windows"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
