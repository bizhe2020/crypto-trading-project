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

from scripts.live_readiness_report import max_drawdown_from_capitals, trade_return_sharpe  # noqa: E402
from scripts.report_htf_pa_ict_context import (  # noqa: E402
    DEFAULT_DATA_15M,
    DEFAULT_DATA_4H,
    DEFAULT_OUTPUT,
    add_htf_contexts,
    build_promoted_shadow_events,
    daily_from_4h,
    load_df,
    namespace_for,
    parse_args as _report_parse_args,
    summarize_events,
    window_start_times,
)
from scripts.report_pa_ict_liquidity_features import scan_events  # noqa: E402
from strategy.scalp_robust_v2_core import dataframe_to_candles  # noqa: E402


DEFAULT_SCAN_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "htf_context" / "htf_pa_ict_context_overlay_scan.json"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    report_parser = argparse.ArgumentParser(add_help=False)
    for action in _report_parse_args.__annotations__.values():
        _ = action
    parser = argparse.ArgumentParser(
        description="Scan small HTF PA/ICT context overlays on promoted high-leverage shadow events."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--best-params", default=str(ROOT / "config" / "high_leverage_pressure_target_cap_best.params.json"))
    parser.add_argument("--pressure-params", default=str(ROOT / "config" / "high_leverage_pressure_target_cap_best.params.json"))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--output", default=str(DEFAULT_SCAN_OUTPUT))
    parser.add_argument("--include-events", action="store_true")
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

    parser.add_argument("--double-opposed-multipliers", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--rule-modes",
        default="double_opposed_only,d1_opposed_only,not_htf_supported",
        help="Comma list: double_opposed_only, d1_opposed_only, not_htf_supported",
    )
    return parser.parse_args()


def parse_modes(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def event_matches_rule(event: dict[str, Any], mode: str) -> bool:
    h4 = str(event.get("htf_h4_alignment") or "none")
    d1 = str(event.get("htf_d1_alignment") or "none")
    if mode == "double_opposed_only":
        return h4 == "opposed" and d1 == "opposed"
    if mode == "d1_opposed_only":
        return d1 == "opposed"
    if mode == "not_htf_supported":
        return not (d1 == "aligned" or (h4 == "aligned" and d1 != "opposed"))
    raise ValueError(f"Unsupported rule mode: {mode}")


def replay(events: list[dict[str, Any]], initial_capital: float, mode: str, multiplier: float) -> dict[str, Any]:
    capital = initial_capital
    capitals: list[float] = []
    returns: list[float] = []
    out_events: list[dict[str, Any]] = []
    adjusted_count = 0
    adjusted_original_return_sum = 0.0
    adjusted_weighted_return_sum = 0.0

    for event in events:
        original_return = float(event.get("return", 0.0) or 0.0)
        matched = event_matches_rule(event, mode)
        applied_multiplier = multiplier if matched else 1.0
        weighted_return = original_return * applied_multiplier
        capital = max(0.0, capital * (1.0 + weighted_return))
        capitals.append(capital)
        returns.append(weighted_return)
        if matched:
            adjusted_count += 1
            adjusted_original_return_sum += original_return
            adjusted_weighted_return_sum += weighted_return
        enriched = dict(event)
        enriched["htf_overlay_rule_mode"] = mode
        enriched["htf_overlay_rule_matched"] = matched
        enriched["htf_overlay_multiplier"] = applied_multiplier
        enriched["original_return"] = original_return
        enriched["return"] = weighted_return
        out_events.append(enriched)

    result = {
        "mode": mode,
        "multiplier": multiplier,
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100.0, 2),
        "final_capital": round(capital, 2),
        "max_drawdown_pct": round(max_drawdown_from_capitals(capitals, initial_capital), 2),
        "sharpe_ratio": round(trade_return_sharpe(returns), 3),
        "trades": len(out_events),
        "adjusted_trades": adjusted_count,
        "adjusted_original_return_sum_pct": round(adjusted_original_return_sum * 100.0, 2),
        "adjusted_weighted_return_sum_pct": round(adjusted_weighted_return_sum * 100.0, 2),
        "summary": summarize_events(out_events, initial_capital),
        "windows": {
            name: summarize_events(
                [
                    event
                    for event in out_events
                    if pd.Timestamp(event["entry_time"]).tz_convert("UTC") >= start
                ],
                initial_capital,
            )
            for name, start in window_start_times(out_events).items()
        },
    }
    return result | {"events": out_events}


def score_result(result: dict[str, Any]) -> float:
    current_year = result.get("windows", {}).get("current_year", {})
    last_60d = result.get("windows", {}).get("last_60d", {})
    last_30d = result.get("windows", {}).get("last_30d", {})
    return round(
        float(result["total_return_pct"])
        + float(current_year.get("compounded_return_pct", 0.0) or 0.0) * 160.0
        + float(last_60d.get("compounded_return_pct", 0.0) or 0.0) * 90.0
        + float(last_30d.get("compounded_return_pct", 0.0) or 0.0) * 50.0
        - float(result["max_drawdown_pct"]) * 30.0,
        4,
    )


def build_events(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    shadow_events, metadata, initial_capital = build_promoted_shadow_events(args)
    df4 = load_df(Path(args.data_4h))
    h4_candles = dataframe_to_candles(df4)
    d1_candles = dataframe_to_candles(daily_from_4h(df4))
    h4_events = scan_events(h4_candles, namespace_for(args, "4h"))
    d1_events = scan_events(d1_candles, namespace_for(args, "1d"))
    enriched_events = add_htf_contexts(shadow_events, h4_events, h4_candles, d1_events, d1_candles, args)
    metadata["htf_event_counts"] = {"h4": len(h4_events), "d1": len(d1_events)}
    return enriched_events, metadata, initial_capital


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": result.get("score"),
        "mode": result["mode"],
        "multiplier": result["multiplier"],
        "total_return_pct": result["total_return_pct"],
        "max_drawdown_pct": result["max_drawdown_pct"],
        "adjusted_trades": result["adjusted_trades"],
        "adjusted_original_return_sum_pct": result["adjusted_original_return_sum_pct"],
        "current_year": result["windows"].get("current_year", {}),
        "last_60d": result["windows"].get("last_60d", {}),
        "last_30d": result["windows"].get("last_30d", {}),
    }


def main() -> None:
    args = parse_args()
    events, metadata, initial_capital = build_events(args)
    baseline = replay(events, initial_capital, "double_opposed_only", 1.0)
    baseline["score"] = score_result(baseline)

    candidates: list[dict[str, Any]] = []
    for mode in parse_modes(args.rule_modes):
        for multiplier in parse_float_list(args.double_opposed_multipliers):
            result = replay(events, initial_capital, mode, multiplier)
            result["score"] = score_result(result)
            candidates.append(result)
    candidates.sort(key=lambda item: item["score"], reverse=True)

    report: dict[str, Any] = {
        "metadata": metadata,
        "parameters": {
            "start_date": args.start_date,
            "require_mss": args.require_mss,
            "rule_modes": args.rule_modes,
            "double_opposed_multipliers": args.double_opposed_multipliers,
        },
        "baseline": {key: value for key, value in baseline.items() if key != "events"},
        "top": [{key: value for key, value in item.items() if key != "events"} for item in candidates],
    }
    if args.include_events:
        report["events"] = events
        report["top_events"] = candidates[0]["events"] if candidates else []

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)
    print(
        json.dumps(
            {
                "baseline": compact(baseline),
                "top": [compact(item) for item in candidates[:10]],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.stdout:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
