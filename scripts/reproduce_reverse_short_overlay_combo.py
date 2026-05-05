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
from scripts.research_reverse_short_from_failed_longs import (  # noqa: E402
    add_combo_deltas,
    add_standard_windows,
    add_windows,
    build_combo_results,
    compact_combo_result,
    compact_result,
    event_stream_summary,
    replay_non_overlapping,
    score_result,
    selected_by,
    simulate_short_trade,
    standard_reverse_short_event,
    standard_sota_event,
)
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "reverse_short_overlay_combo_reproduction.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce one fixed reverse-short overlay combo candidate on top of the shadow SOTA stream.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--source-stream", default="shadow")
    parser.add_argument("--selector", default="guarded_weak_loss")
    parser.add_argument("--trigger-mode", default="stop_loss_reversal")
    parser.add_argument("--target-rr", type=float, default=2.5)
    parser.add_argument("--max-hold-bars", type=int, default=64)
    parser.add_argument("--leverage", type=float, default=5.0)
    parser.add_argument("--stop-multiplier", type=float, default=1.25)
    parser.add_argument("--max-short-stop-pct", type=float, default=2.0)
    parser.add_argument("--overlay-allocation", type=float, default=1.0)
    parser.add_argument("--combo-mode", default="base_priority_single_slot")
    parser.add_argument("--max-quality-score", type=int, default=1)
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=0)
    parser.add_argument("--sample-trades", type=int, default=20)
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

    source_events = shadow["events"] if args.source_stream == "shadow" else fixed["events"]
    reverse_candidates = []
    for event in source_events:
        if not selected_by(event, str(args.selector), int(args.max_quality_score)):
            continue
        simulated = simulate_short_trade(
            event=event,
            candles=prepared.c15m,
            trigger_mode=str(args.trigger_mode),
            target_rr=float(args.target_rr),
            max_hold_bars=int(args.max_hold_bars),
            leverage=float(args.leverage),
            stop_multiplier=float(args.stop_multiplier),
            max_short_stop_pct=float(args.max_short_stop_pct),
            virtual_invalidation_rr=None,
            virtual_invalidation_lookahead_bars=None,
            taker_fee_rate=float(payload.get("taker_fee_rate", 0.0005) or 0.0),
            slippage_bps=float(payload.get("slippage_bps", 0.0) or 0.0),
        )
        if simulated is not None:
            reverse_candidates.append(simulated)

    reverse_only = replay_non_overlapping(reverse_candidates, initial_capital)
    reverse_only = add_windows(reverse_only, initial_capital, prepared.end)
    reverse_only["params"] = {
        "source_stream": args.source_stream,
        "selector": args.selector,
        "trigger_mode": args.trigger_mode,
        "target_rr": args.target_rr,
        "max_hold_bars": args.max_hold_bars,
        "leverage": args.leverage,
        "stop_multiplier": args.stop_multiplier,
        "max_short_stop_pct": args.max_short_stop_pct,
    }
    reverse_only["score"] = score_result(reverse_only)

    base_shadow_summary = event_stream_summary(shadow["events"], initial_capital, prepared.end)
    standard_base_events = [standard_sota_event(event) for event in shadow["events"]]
    standard_overlay_events = [
        standard_reverse_short_event(event, float(args.overlay_allocation))
        for event in reverse_only["events"]
    ]
    combos = build_combo_results(
        standard_base_events,
        standard_overlay_events,
        initial_capital,
        prepared.end,
        base_shadow_summary,
    )
    combo = next((item for item in combos if str(item.get("combo_mode")) == str(args.combo_mode)), None)
    if combo is None:
        raise ValueError(f"Unsupported combo mode: {args.combo_mode}")
    combo["params"] = {
        **reverse_only["params"],
        "overlay_allocation": args.overlay_allocation,
        "combo_mode": args.combo_mode,
    }

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
        },
        "baseline_shadow_sota": compact_result(base_shadow_summary, 0),
        "reverse_only": compact_result(reverse_only, int(args.sample_trades)),
        "combo": compact_combo_result(combo, int(args.sample_trades)),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    print("Baseline shadow SOTA:")
    print(
        f"  full={report['baseline_shadow_sota']['total_return_pct']:.2f}%/"
        f"{report['baseline_shadow_sota']['max_drawdown_pct']:.2f}% "
        f"trades={report['baseline_shadow_sota']['trades']}"
    )
    print("Reverse short only:")
    print(
        f"  full={report['reverse_only']['total_return_pct']:.2f}%/"
        f"{report['reverse_only']['max_drawdown_pct']:.2f}% "
        f"trades={report['reverse_only']['trades']} score={report['reverse_only']['score']:.2f}"
    )
    print("Combo:")
    combo_delta = report["combo"]["delta_vs_shadow_sota"]
    print(
        f"  mode={report['combo']['params']['combo_mode']} "
        f"full={report['combo']['total_return_pct']:.2f}%/{report['combo']['max_drawdown_pct']:.2f}% "
        f"delta={combo_delta['total_return_pct']:.2f}%/{combo_delta['max_drawdown_pct']:+.2f}dd "
        f"trades={report['combo']['trades']}"
    )


if __name__ == "__main__":
    main()
