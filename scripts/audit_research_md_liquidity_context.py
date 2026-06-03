#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_readiness_report import load_prepared_data
from scripts.report_pa_ict_liquidity_features import recent_fvg, scan_events, time_bucket, zone_touched
from scripts.report_smc_trade_context import (
    completed_4h_idx_for_entry,
    completed_d1_idx_for_entry,
    daily_candles_from_4h,
    event_support_context,
    h4_bias_for_idx,
    pd_side,
    range_context,
    structure_bias_for_idx,
)
from strategy.scalp_robust_v2_core import precompute_swings


DEFAULT_FROZEN_REPORT = ROOT / "var" / "high_leverage_expansion" / "frozen_live_core_20260515.json"
DEFAULT_CONFIG = ROOT / "config" / "config.live.high-leverage-structure.template.json"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "research_md_liquidity_context_audit_20260516.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit ICT/SMC liquidity-context ideas against the frozen live SOTA long sample."
    )
    parser.add_argument("--frozen-report", default=str(DEFAULT_FROZEN_REPORT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--recent-sweep-lookback-bars", type=int, default=96)
    parser.add_argument("--recent-fvg-lookback-bars", type=int, default=32)
    parser.add_argument("--h4-range-lookback-bars", type=int, default=42)
    parser.add_argument("--d1-range-lookback-bars", type=int, default=60)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    wins = sum(1 for row in rows if bool(row["win"]))
    return {
        "trades": len(rows),
        "win_rate_pct": round(wins / len(rows) * 100.0, 2),
        "avg_return_pct": round(sum(float(row["return_pct"]) for row in rows) / len(rows), 4),
        "sum_return_pct": round(sum(float(row["return_pct"]) for row in rows), 4),
    }


def grouped_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    values = sorted({str(row.get(key)) for row in rows})
    for value in values:
        out[value] = summarize([row for row in rows if str(row.get(key)) == value])
    return out


def composite_summary(rows: list[dict[str, Any]], name: str, predicate: Any) -> dict[str, Any]:
    subset = [row for row in rows if predicate(row)]
    rest = [row for row in rows if not predicate(row)]
    return {
        "name": name,
        "subset": summarize(subset),
        "rest": summarize(rest),
    }


def build_scan_args() -> SimpleNamespace:
    return SimpleNamespace(
        swing_n=3,
        swing_lookback=80,
        liquidity_lookback_bars=192,
        mss_lookahead_bars=24,
        fvg_lookback_bars=8,
        entry_lookahead_bars=40,
        outcome_lookahead_bars=96,
        atr_period=14,
        min_body_atr=0.7,
        min_range_atr=1.1,
        stop_buffer_atr=0.05,
        target_rr=2.0,
        allow_incomplete_tail=True,
    )


def annotate_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    frozen = load_json(Path(args.frozen_report))
    config = load_json(Path(args.config))
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=config.get("regime_switcher_thresholds"),
    )
    daily = daily_candles_from_4h(prepared.c4h)
    daily_ts = [candle.ts for candle in daily]
    daily_highs, daily_lows = precompute_swings(daily, n=2, lookback=20)
    liquidity_events = scan_events(prepared.c15m, build_scan_args())
    events_by_direction: dict[str, list[Any]] = defaultdict(list)
    for event in liquidity_events:
        events_by_direction[str(event.direction)].append(event)

    rows: list[dict[str, Any]] = []
    for event in frozen["live_shadow"]["events"]:
        if str(event.get("event_type") or "") != "sota_long":
            continue
        entry_idx = int(event["entry_idx"])
        direction = str(event["direction"])
        c15 = prepared.c15m[entry_idx]
        h4_idx = completed_4h_idx_for_entry(prepared.mapping, entry_idx)
        d1_idx = completed_d1_idx_for_entry(daily_ts, c15.ts)
        h4_range = range_context(prepared.c4h, h4_idx, int(args.h4_range_lookback_bars), float(c15.c))
        d1_range = range_context(daily, d1_idx, int(args.d1_range_lookback_bars), float(c15.c))
        h4_bias = h4_bias_for_idx(prepared.precomputed, h4_idx)
        d1_bias = structure_bias_for_idx(daily, daily_highs, daily_lows, d1_idx)
        sweep_context = event_support_context(
            events_by_direction,
            direction,
            entry_idx,
            int(args.recent_sweep_lookback_bars),
        )
        fvg = recent_fvg(prepared.c15m, direction, max(2, entry_idx - int(args.recent_fvg_lookback_bars)), entry_idx)
        recent_fvg_near_entry = bool(fvg and zone_touched(c15, fvg.bottom, fvg.top))
        bucket, _ny_time = time_bucket(c15.ts)
        bucket_payload = event.get("long_score_bucket_sizing")
        boosted = bool(isinstance(bucket_payload, dict) and bucket_payload.get("applied"))
        rows.append(
            {
                "entry_time": event.get("entry_time"),
                "return_pct": float(event.get("return_pct", 0.0) or 0.0),
                "win": float(event.get("return_pct", 0.0) or 0.0) > 0.0,
                "regime_label": event.get("regime_label"),
                "session_bucket": bucket,
                "h4_bias": h4_bias["bias"],
                "d1_bias": d1_bias["bias"],
                "h4_pd_side": pd_side(direction, h4_range),
                "d1_pd_side": pd_side(direction, d1_range),
                "recent_sweep": bool(sweep_context["recent_sweep"]),
                "recent_sweep_mss": bool(sweep_context["recent_sweep_mss"]),
                "recent_sweep_has_fvg": bool(sweep_context.get("event_has_fvg")),
                "recent_sweep_confirmed": bool(sweep_context.get("event_retest_confirmed")),
                "recent_sweep_lag_bars": sweep_context.get("sweep_lag_bars"),
                "recent_sweep_status": sweep_context.get("event_status"),
                "recent_fvg_near_entry": recent_fvg_near_entry,
                "high_growth": str(event.get("regime_label") or "") == "high_growth",
                "bucket_boosted": boosted,
                "feature_adx": event.get("feature_adx"),
                "feature_momentum": event.get("feature_momentum"),
                "feature_ema_gap": event.get("feature_ema_gap"),
                "net_score": event.get("net_score"),
                "bull_total": event.get("bull_total"),
                "bear_total": event.get("bear_total"),
                "conflict": event.get("conflict"),
            }
        )
    return rows


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    rows = annotate_rows(args)
    composites = [
        composite_summary(
            rows,
            "mss_plus_fvg_plus_h4_discount",
            lambda row: bool(row["recent_sweep_mss"]) and bool(row["recent_fvg_near_entry"]) and str(row["h4_pd_side"]) == "favorable",
        ),
        composite_summary(
            rows,
            "mss_plus_h4_discount",
            lambda row: bool(row["recent_sweep_mss"]) and str(row["h4_pd_side"]) == "favorable",
        ),
        composite_summary(rows, "mss_only", lambda row: bool(row["recent_sweep_mss"])),
        composite_summary(rows, "fvg_near_only", lambda row: bool(row["recent_fvg_near_entry"])),
        composite_summary(
            rows,
            "high_growth_plus_mss",
            lambda row: bool(row["high_growth"]) and bool(row["recent_sweep_mss"]),
        ),
        composite_summary(
            rows,
            "boosted_plus_mss",
            lambda row: bool(row["bucket_boosted"]) and bool(row["recent_sweep_mss"]),
        ),
    ]
    return {
        "metadata": {
            "frozen_report": str(Path(args.frozen_report).resolve()),
            "config": str(Path(args.config).resolve()),
            "data_15m": str(Path(args.data_15m).resolve()),
            "data_4h": str(Path(args.data_4h).resolve()),
            "start_date": args.start_date,
            "recent_sweep_lookback_bars": int(args.recent_sweep_lookback_bars),
            "recent_fvg_lookback_bars": int(args.recent_fvg_lookback_bars),
            "h4_range_lookback_bars": int(args.h4_range_lookback_bars),
            "d1_range_lookback_bars": int(args.d1_range_lookback_bars),
        },
        "overall": summarize(rows),
        "grouped": {
            "recent_sweep": grouped_summary(rows, "recent_sweep"),
            "recent_sweep_mss": grouped_summary(rows, "recent_sweep_mss"),
            "recent_fvg_near_entry": grouped_summary(rows, "recent_fvg_near_entry"),
            "h4_pd_side": grouped_summary(rows, "h4_pd_side"),
            "d1_pd_side": grouped_summary(rows, "d1_pd_side"),
            "session_bucket": grouped_summary(rows, "session_bucket"),
            "regime_label": grouped_summary(rows, "regime_label"),
            "recent_sweep_status": grouped_summary(rows, "recent_sweep_status"),
        },
        "composites": composites,
    }


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    print(json.dumps(report["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
