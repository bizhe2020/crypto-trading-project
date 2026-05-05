#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.confirmed_multiframe_score_utils import (  # noqa: E402
    align_confirmed_mapping,
    resample_confirmed_1h,
    score_snapshot,
)
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.live_shadow_utils import clean_for_json, standard_sota_event  # noqa: E402
from scripts.report_smc_trade_context import daily_candles_from_4h  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402
from scripts.smc_live_utils import SMC_CASES, build_smc_events  # noqa: E402
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "confirmed_multiframe_scores.json"
RR_MODE_CHOICES = ("close", "extreme")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score confirmed 4h/1h/15m bullish/bearish context at replay event entries.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--replay-sync-entry-to-signal-price", action="store_true")
    parser.add_argument("--smc-case", default="v2_medium_dispbody05_otherlag4_10x", choices=sorted(SMC_CASES))
    parser.add_argument("--smc-allocation", type=float, default=1.0)
    parser.add_argument("--stage-trigger-rr-mode", default="close", choices=RR_MODE_CHOICES)
    parser.add_argument("--time-trailing-rr-mode", default="close", choices=RR_MODE_CHOICES)
    parser.add_argument("--atr-activation-rr-mode", default="close", choices=RR_MODE_CHOICES)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def summarize_bucket(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {"count": 0, "avg_return_pct": 0.0, "win_rate_pct": 0.0, "sum_return_pct": 0.0}
    returns = [float(item.get("return", 0.0) or 0.0) * 100.0 for item in events]
    wins = sum(1 for value in returns if value > 0.0)
    return {
        "count": len(events),
        "avg_return_pct": round(sum(returns) / len(returns), 4),
        "win_rate_pct": round(wins / len(returns) * 100.0, 2),
        "sum_return_pct": round(sum(returns), 4),
    }


def build_buckets(events: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        bucket_key = str(event.get(key))
        buckets.setdefault(bucket_key, []).append(event)
    return {bucket: summarize_bucket(items) for bucket, items in sorted(buckets.items(), key=lambda item: item[0])}


def event_type_reports(events: list[dict[str, Any]]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for event_type in sorted({str(item.get("event_type")) for item in events}):
        subset = [item for item in events if str(item.get("event_type")) == event_type]
        reports[event_type] = {
            "overall": summarize_bucket(subset),
            "by_net_score": build_buckets(subset, "net_score"),
            "by_bull_total": build_buckets(subset, "bull_total"),
            "by_bear_total": build_buckets(subset, "bear_total"),
            "conflict_vs_clean": {
                "conflict": summarize_bucket([item for item in subset if bool(item.get("conflict"))]),
                "clean": summarize_bucket([item for item in subset if not bool(item.get("conflict"))]),
            },
        }
    return reports


def main() -> None:
    args = parse_args()
    base_payload = load_config_payload(Path(args.config))
    payload, _pressure_params = apply_pressure_params(base_payload, Path(args.pressure_params))
    payload["replay_sync_entry_to_signal_price"] = bool(args.replay_sync_entry_to_signal_price)
    payload["stage_trigger_rr_mode"] = str(args.stage_trigger_rr_mode)
    payload["time_trailing_rr_mode"] = str(args.time_trailing_rr_mode)
    payload["atr_activation_rr_mode"] = str(args.atr_activation_rr_mode)
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=payload.get("regime_switcher_thresholds"),
        confirmed_4h_only=True,
    )
    metrics, engine = run_engine(payload, prepared, args.start_date)
    trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
    initial_capital = float(metrics.get("initial_capital", 1000.0) or 1000.0)
    fixed = expansion_overlay(trades, initial_capital, FIXED_STRUCTURE_PARAMS, include_events=True)
    shadow = replay_shadow_events(
        fixed["events"],
        initial_capital,
        daily_loss_stop_pct=6.0,
        equity_drawdown_stop_pct=15.0,
        consecutive_loss_stop=0,
        equity_drawdown_cooldown_days=2,
    )
    shadow_events = shadow["events"]
    base_events = [standard_sota_event(event) for event in shadow_events]
    daily = daily_candles_from_4h(prepared.c4h)
    h4_highs, h4_lows = precompute_swings(prepared.c4h, n=2, lookback=80)
    d1_highs, d1_lows = precompute_swings(daily, n=2, lookback=20)
    smc_events, smc_summary = build_smc_events(
        args.smc_case,
        SMC_CASES[str(args.smc_case)],
        args,
        prepared,
        daily,
        h4_highs,
        h4_lows,
        d1_highs,
        d1_lows,
        float(args.smc_allocation),
        taker_fee_rate=float(payload.get("taker_fee_rate", 0.0005) or 0.0),
        slippage_bps=float(payload.get("slippage_bps", 0.0) or 0.0),
    )

    c1h = resample_confirmed_1h(prepared.c15m)
    mapping_1h = align_confirmed_mapping(c1h, prepared.c15m)
    all_events = base_events + smc_events
    scored_events: list[dict[str, Any]] = []
    for event in all_events:
        entry_idx = int(event.get("entry_idx", 0) or 0)
        snapshot = score_snapshot(prepared, c1h, mapping_1h, entry_idx)
        enriched = dict(event)
        enriched.update(asdict(snapshot))
        scored_events.append(enriched)

    top_bull = sorted(scored_events, key=lambda item: (int(item["bull_total"]), -int(item["bear_total"]), float(item.get("return", 0.0))), reverse=True)[
        : max(int(args.top_n), 0)
    ]
    top_bear = sorted(scored_events, key=lambda item: (int(item["bear_total"]), -int(item["bull_total"]), -float(item.get("return", 0.0))), reverse=True)[
        : max(int(args.top_n), 0)
    ]

    report = {
        "metadata": {
            "config": str(Path(args.config).resolve()),
            "start_date": args.start_date,
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
            "candles_1h": len(c1h),
            "confirmed_4h_only": True,
            "replay_sync_entry_to_signal_price": bool(args.replay_sync_entry_to_signal_price),
            "trailing_rr_modes": {
                "stage_trigger_rr_mode": str(args.stage_trigger_rr_mode),
                "time_trailing_rr_mode": str(args.time_trailing_rr_mode),
                "atr_activation_rr_mode": str(args.atr_activation_rr_mode),
            },
        },
        "candidate_generation": {
            "sota_candidates": len(base_events),
            "smc_candidates": len(smc_events),
            "smc_summary": smc_summary,
        },
        "overall": summarize_bucket(scored_events),
        "by_event_type": {
            event_type: summarize_bucket([item for item in scored_events if item.get("event_type") == event_type])
            for event_type in sorted({str(item.get("event_type")) for item in scored_events})
        },
        "by_net_score": build_buckets(scored_events, "net_score"),
        "by_bull_total": build_buckets(scored_events, "bull_total"),
        "by_bear_total": build_buckets(scored_events, "bear_total"),
        "conflict_vs_clean": {
            "conflict": summarize_bucket([item for item in scored_events if bool(item.get("conflict"))]),
            "clean": summarize_bucket([item for item in scored_events if not bool(item.get("conflict"))]),
        },
        "by_event_type_detailed": event_type_reports(scored_events),
        "top_bull": top_bull,
        "top_bear": top_bear,
        "scored_events": scored_events,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
