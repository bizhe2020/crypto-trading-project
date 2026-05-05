#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_htf_pa_ict_context import (  # noqa: E402
    DEFAULT_DATA_4H,
    active_context_for_entry,
    daily_from_4h,
    load_df,
    namespace_for,
)
from scripts.report_pa_ict_liquidity_features import scan_events  # noqa: E402
from scripts.reproduce_htf_pa_ict_guard import clean_for_json  # noqa: E402
from strategy.scalp_robust_v2_core import dataframe_to_candles  # noqa: E402


DEFAULT_INPUT = ROOT / "var" / "pa_ict_liquidity" / "htf_context" / "htf_pa_ict_guard_audit.json"
DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "htf_context" / "htf_pa_ict_context_replay_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit HTF PA/ICT context by recomputing it with entry-time cutoffs only."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--all-guarded", action="store_true", default=True)
    parser.add_argument("--stdout", action="store_true")

    parser.add_argument("--swing-n", type=int, default=2)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--min-body-atr", type=float, default=0.7)
    parser.add_argument("--min-range-atr", type=float, default=1.1)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.05)
    parser.add_argument("--target-rr", type=float, default=2.0)
    parser.add_argument("--require-mss", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--h4-swing-lookback", type=int, default=30)
    parser.add_argument("--h4-liquidity-lookback-bars", type=int, default=180)
    parser.add_argument("--h4-mss-lookahead-bars", type=int, default=12)
    parser.add_argument("--h4-fvg-lookback-bars", type=int, default=6)
    parser.add_argument("--h4-entry-lookahead-bars", type=int, default=18)
    parser.add_argument("--h4-outcome-lookahead-bars", type=int, default=36)
    parser.add_argument("--h4-context-ttl-bars", type=int, default=42)

    parser.add_argument("--d1-swing-lookback", type=int, default=20)
    parser.add_argument("--d1-liquidity-lookback-bars", type=int, default=90)
    parser.add_argument("--d1-mss-lookahead-bars", type=int, default=5)
    parser.add_argument("--d1-fvg-lookback-bars", type=int, default=4)
    parser.add_argument("--d1-entry-lookahead-bars", type=int, default=10)
    parser.add_argument("--d1-outcome-lookahead-bars", type=int, default=20)
    parser.add_argument("--d1-context-ttl-bars", type=int, default=14)
    parser.set_defaults(allow_incomplete_tail=True)
    return parser.parse_args()


def cutoff_candles(candles: list[Any], entry_time: pd.Timestamp) -> list[Any]:
    entry_ts = entry_time.timestamp()
    ts_values = [candle.ts for candle in candles]
    end_idx = bisect.bisect_right(ts_values, entry_ts) - 1
    if end_idx < 0:
        return []
    return candles[: end_idx + 1]


def normalize_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": context.get("state", "none"),
        "alignment": context.get("alignment", "none"),
        "age_bars": context.get("age_bars"),
        "anchor_idx": context.get("anchor_idx"),
        "event_status": context.get("event_status"),
        "event_sweep_time": context.get("event_sweep_time"),
        "event_mss_time": context.get("event_mss_time"),
        "event_retest_time": context.get("event_retest_time"),
    }


def expected_context(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        "state": row.get(f"{prefix}_state"),
        "alignment": row.get(f"{prefix}_alignment"),
        "age_bars": row.get(f"{prefix}_age_bars"),
    }


def context_matches(expected: dict[str, Any], replay: dict[str, Any]) -> bool:
    def age(value: Any) -> int | None:
        if value is None:
            return None
        return int(float(value))

    return (
        str(expected.get("state") or "none") == str(replay.get("state") or "none")
        and str(expected.get("alignment") or "none") == str(replay.get("alignment") or "none")
        and age(expected.get("age_bars")) == age(replay.get("age_bars"))
    )


def main() -> None:
    args = parse_args()
    source = json.loads(Path(args.input).read_text())
    rows = source.get("rows", [])
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")

    df4 = load_df(Path(args.data_4h))
    h4_candles_full = dataframe_to_candles(df4)
    d1_candles_full = dataframe_to_candles(daily_from_4h(df4))

    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        entry_time = pd.Timestamp(row["entry_time"]).tz_convert("UTC")
        direction = str(row.get("direction") or "")
        h4_candles = cutoff_candles(h4_candles_full, entry_time)
        d1_candles = cutoff_candles(d1_candles_full, entry_time)
        h4_events = scan_events(h4_candles, namespace_for(args, "4h"))
        d1_events = scan_events(d1_candles, namespace_for(args, "1d"))
        h4_context = active_context_for_entry(
            h4_events,
            h4_candles,
            entry_time,
            args.h4_context_ttl_bars,
            trade_direction=direction,
            require_mss=bool(args.require_mss),
        )
        d1_context = active_context_for_entry(
            d1_events,
            d1_candles,
            entry_time,
            args.d1_context_ttl_bars,
            trade_direction=direction,
            require_mss=bool(args.require_mss),
        )
        expected_h4 = expected_context(row, "h4")
        expected_d1 = expected_context(row, "d1")
        replay_h4 = normalize_context(h4_context)
        replay_d1 = normalize_context(d1_context)
        audit_rows.append(
            {
                "entry_time": row["entry_time"],
                "direction": direction,
                "h4_match": context_matches(expected_h4, replay_h4),
                "d1_match": context_matches(expected_d1, replay_d1),
                "match": context_matches(expected_h4, replay_h4) and context_matches(expected_d1, replay_d1),
                "expected_h4": expected_h4,
                "replay_h4": replay_h4,
                "expected_d1": expected_d1,
                "replay_d1": replay_d1,
                "h4_events_at_cutoff": len(h4_events),
                "d1_events_at_cutoff": len(d1_events),
                "h4_candles_at_cutoff": len(h4_candles),
                "d1_candles_at_cutoff": len(d1_candles),
            }
        )

    mismatches = [row for row in audit_rows if not row["match"]]
    report = {
        "input": str(Path(args.input)),
        "data_4h": str(Path(args.data_4h)),
        "parameters": {
            "swing_n": args.swing_n,
            "require_mss": args.require_mss,
            "h4_context_ttl_bars": args.h4_context_ttl_bars,
            "d1_context_ttl_bars": args.d1_context_ttl_bars,
            "h4_swing_lookback": args.h4_swing_lookback,
            "h4_liquidity_lookback_bars": args.h4_liquidity_lookback_bars,
            "h4_mss_lookahead_bars": args.h4_mss_lookahead_bars,
            "h4_fvg_lookback_bars": args.h4_fvg_lookback_bars,
            "d1_swing_lookback": args.d1_swing_lookback,
            "d1_liquidity_lookback_bars": args.d1_liquidity_lookback_bars,
            "d1_mss_lookahead_bars": args.d1_mss_lookahead_bars,
            "d1_fvg_lookback_bars": args.d1_fvg_lookback_bars,
            "min_body_atr": args.min_body_atr,
            "min_range_atr": args.min_range_atr,
        },
        "summary": {
            "rows": len(audit_rows),
            "matched": len(audit_rows) - len(mismatches),
            "mismatched": len(mismatches),
            "h4_mismatched": sum(1 for row in audit_rows if not row["h4_match"]),
            "d1_mismatched": sum(1 for row in audit_rows if not row["d1_match"]),
        },
        "rows": audit_rows,
        "mismatches": mismatches,
    }
    cleaned = clean_for_json(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    print(json.dumps({"summary": cleaned["summary"], "mismatches": cleaned["mismatches"]}, ensure_ascii=False, indent=2, allow_nan=False))
    if args.stdout:
        print(json.dumps(cleaned, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
