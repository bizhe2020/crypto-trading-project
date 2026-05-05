#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_htf_pa_ict_context import (  # noqa: E402
    add_htf_contexts,
    build_promoted_shadow_events,
    daily_from_4h,
    load_df,
    namespace_for,
)
from scripts.report_pa_ict_liquidity_features import scan_events  # noqa: E402
from scripts.scan_htf_pa_ict_context_overlay import compact, parse_float_list, replay, score_result  # noqa: E402
from strategy.scalp_robust_v2_core import dataframe_to_candles  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "htf_context" / "htf_pa_ict_guard_ttl_sensitivity.json"


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan TTL sensitivity for the tiny HTF PA/ICT double-opposed guard."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--best-params", default=str(ROOT / "config" / "high_leverage_pressure_target_cap_best.params.json"))
    parser.add_argument("--pressure-params", default=str(ROOT / "config" / "high_leverage_pressure_target_cap_best.params.json"))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument("--stdout", action="store_true")

    parser.add_argument("--h4-ttl-bars-values", default="24,36,42,60")
    parser.add_argument("--d1-ttl-bars-values", default="7,10,14,21")
    parser.add_argument("--multipliers", default="0,0.25,0.5")
    parser.add_argument("--top", type=int, default=20)

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
    return parser.parse_args()


def passes_promotion_floor(result: dict[str, Any], baseline: dict[str, Any]) -> bool:
    windows = result.get("windows", {})
    base_windows = baseline.get("windows", {})
    return (
        float(result["total_return_pct"]) > float(baseline["total_return_pct"])
        and float(result["max_drawdown_pct"]) <= float(baseline["max_drawdown_pct"])
        and float(windows.get("current_year", {}).get("compounded_return_pct", 0.0))
        >= float(base_windows.get("current_year", {}).get("compounded_return_pct", 0.0))
        and float(windows.get("last_60d", {}).get("compounded_return_pct", 0.0))
        >= float(base_windows.get("last_60d", {}).get("compounded_return_pct", 0.0))
        and float(windows.get("last_30d", {}).get("compounded_return_pct", 0.0))
        >= float(base_windows.get("last_30d", {}).get("compounded_return_pct", 0.0))
    )


def build_base_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[Any], list[Any], list[Any], list[Any], dict[str, Any], float]:
    shadow_events, metadata, initial_capital = build_promoted_shadow_events(args)
    df4 = load_df(Path(args.data_4h))
    h4_candles = dataframe_to_candles(df4)
    d1_candles = dataframe_to_candles(daily_from_4h(df4))
    h4_events = scan_events(h4_candles, namespace_for(args, "4h"))
    d1_events = scan_events(d1_candles, namespace_for(args, "1d"))
    metadata["htf_event_counts"] = {"h4": len(h4_events), "d1": len(d1_events)}
    return shadow_events, h4_events, h4_candles, d1_events, d1_candles, metadata, initial_capital


def main() -> None:
    args = parse_args()
    (
        shadow_events,
        h4_events,
        h4_candles,
        d1_events,
        d1_candles,
        metadata,
        initial_capital,
    ) = build_base_inputs(args)
    baseline_events = add_htf_contexts(
        shadow_events,
        h4_events,
        h4_candles,
        d1_events,
        d1_candles,
        args,
    )
    baseline = replay(baseline_events, initial_capital, "double_opposed_only", 1.0)
    baseline["score"] = score_result(baseline)

    results: list[dict[str, Any]] = []
    h4_values = parse_int_list(args.h4_ttl_bars_values)
    d1_values = parse_int_list(args.d1_ttl_bars_values)
    multipliers = parse_float_list(args.multipliers)
    for h4_ttl in h4_values:
        for d1_ttl in d1_values:
            args.h4_context_ttl_bars = h4_ttl
            args.d1_context_ttl_bars = d1_ttl
            events = add_htf_contexts(
                shadow_events,
                h4_events,
                h4_candles,
                d1_events,
                d1_candles,
                args,
            )
            for multiplier in multipliers:
                result = replay(events, initial_capital, "double_opposed_only", multiplier)
                result["score"] = score_result(result)
                result["h4_context_ttl_bars"] = h4_ttl
                result["d1_context_ttl_bars"] = d1_ttl
                result["passes_promotion_floor"] = passes_promotion_floor(result, baseline)
                result["metadata_snapshot"] = {
                    "htf_event_counts": metadata.get("htf_event_counts", {}),
                }
                results.append(result)

    results.sort(key=lambda item: item["score"], reverse=True)
    passing = [item for item in results if item.get("passes_promotion_floor")]
    by_ttl: dict[str, dict[str, Any]] = {}
    for item in results:
        key = f"h4={item['h4_context_ttl_bars']},d1={item['d1_context_ttl_bars']}"
        best = by_ttl.get(key)
        if best is None or float(item["score"]) > float(best["score"]):
            by_ttl[key] = item

    report: dict[str, Any] = {
        "metadata": metadata,
        "parameters": {
            "start_date": args.start_date,
            "h4_ttl_bars_values": h4_values,
            "d1_ttl_bars_values": d1_values,
            "multipliers": multipliers,
            "require_mss": args.require_mss,
        },
        "baseline": {key: value for key, value in baseline.items() if key != "events"},
        "candidate_count": len(results),
        "passing_count": len(passing),
        "top": [{key: value for key, value in item.items() if key != "events"} for item in results[: args.top]],
        "passing": [{key: value for key, value in item.items() if key != "events"} for item in passing],
        "best_by_ttl": {
            key: {field: value for field, value in item.items() if field != "events"}
            for key, item in sorted(by_ttl.items())
        },
    }
    if args.include_events and results:
        report["top_events"] = results[0]["events"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)
    print(
        json.dumps(
            {
                "baseline": compact(baseline),
                "candidate_count": len(results),
                "passing_count": len(passing),
                "top": [
                    compact(item)
                    | {
                        "h4_context_ttl_bars": item["h4_context_ttl_bars"],
                        "d1_context_ttl_bars": item["d1_context_ttl_bars"],
                        "passes_promotion_floor": item["passes_promotion_floor"],
                    }
                    for item in results[: args.top]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.stdout:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
