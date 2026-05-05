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
    add_windows,
    build_combo_results,
    compact_combo_result,
    compact_result,
    event_stream_summary,
    event_timestamp,
    replay_non_overlapping,
    score_result,
    selected_by,
    simulate_short_trade,
    standard_reverse_short_event,
    standard_sota_event,
)
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "reverse_short_overlay_candidate_comparison.json"


STABLE_CANDIDATE: dict[str, Any] = {
    "label": "stable",
    "source_stream": "shadow",
    "selector": "guarded_weak_loss",
    "trigger_mode": "stop_loss_reversal",
    "target_rr": 2.75,
    "max_hold_bars": 80,
    "leverage": 5.0,
    "stop_multiplier": 1.1,
    "max_short_stop_pct": 1.75,
    "overlay_allocation": 1.0,
    "combo_mode": "base_priority_single_slot",
    "max_quality_score": 1,
}


BROADER_CANDIDATE: dict[str, Any] = {
    "label": "broader_raw",
    "source_stream": "shadow",
    "selector": "bull_high_growth_offense_loss",
    "trigger_mode": "stop_loss_reversal",
    "target_rr": 1.25,
    "max_hold_bars": 32,
    "leverage": 6.0,
    "stop_multiplier": 1.0,
    "max_short_stop_pct": 1.5,
    "overlay_allocation": 1.0,
    "combo_mode": "base_priority_single_slot",
    "max_quality_score": 1,
}


BROADER_GATE_CANDIDATE: dict[str, Any] = {
    "label": "broader_gate",
    "source_stream": "shadow",
    "selector": "bull_high_growth_offense_loss",
    "trigger_mode": "stop_loss_reversal",
    "target_rr": 1.25,
    "max_hold_bars": 12,
    "leverage": 6.0,
    "stop_multiplier": 1.0,
    "max_short_stop_pct": 1.5,
    "overlay_allocation": 1.0,
    "combo_mode": "base_priority_single_slot",
    "max_quality_score": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the fixed stable and broader reverse-short overlay candidates side by side.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
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


def compounded_return_pct(events: list[dict[str, Any]], initial_capital: float) -> float:
    capital = initial_capital
    for event in sorted(events, key=lambda item: (int(item.get("entry_idx", 0) or 0), int(item.get("exit_idx", 0) or 0))):
        capital = max(0.0, capital * (1.0 + float(event.get("return", 0.0) or 0.0)))
    return round((capital - initial_capital) / initial_capital * 100.0, 4)


def event_return_stats(events: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    returns = [float(event.get("return", 0.0) or 0.0) for event in events]
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    ordered_returns = sorted(returns, reverse=True)
    return {
        "trades": len(events),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(returns) * 100.0, 2) if returns else 0.0,
        "sum_return_pct": round(sum(returns) * 100.0, 4),
        "compounded_return_pct": compounded_return_pct(events, initial_capital),
        "avg_return_pct": round(sum(returns) / len(returns) * 100.0, 4) if returns else 0.0,
        "best_return_pct": round(max(returns) * 100.0, 4) if returns else 0.0,
        "worst_return_pct": round(min(returns) * 100.0, 4) if returns else 0.0,
        "top_3_sum_return_pct": round(sum(ordered_returns[:3]) * 100.0, 4) if returns else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "exit_counts": {
            reason: sum(1 for event in events if str(event.get("exit_reason") or "unknown") == reason)
            for reason in sorted({str(event.get("exit_reason") or "unknown") for event in events})
        },
    }


def overlay_attribution(combo: dict[str, Any], initial_capital: float, data_end: pd.Timestamp) -> dict[str, Any]:
    overlay_events = [
        event for event in combo.get("events", [])
        if str(event.get("event_type") or "") == "reverse_short"
    ]
    starts = {
        "current_year": pd.Timestamp(f"{data_end.year}-01-01", tz="UTC"),
        "last_60d": data_end - pd.Timedelta(days=60),
        "last_30d": data_end - pd.Timedelta(days=30),
    }
    return {
        "accepted_overlay": event_return_stats(overlay_events, initial_capital),
        "windows": {
            name: event_return_stats(
                [event for event in overlay_events if event_timestamp(event, "entry_time") >= start],
                initial_capital,
            )
            for name, start in starts.items()
        },
    }


def year_breakdown(events: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    years = sorted({event_timestamp(event, "entry_time").year for event in events})
    breakdown: dict[str, Any] = {}
    for year in years:
        year_events = [event for event in events if event_timestamp(event, "entry_time").year == year]
        breakdown[str(year)] = event_return_stats(year_events, initial_capital)
    return breakdown


def source_funnel(
    source_events: list[dict[str, Any]],
    selector_matches: int,
    reverse_candidates: list[dict[str, Any]],
    reverse_only: dict[str, Any],
    combo: dict[str, Any],
) -> dict[str, Any]:
    event_type_counts = combo.get("event_type_counts", {})
    return {
        "source_events": len(source_events),
        "selector_matches": selector_matches,
        "simulated_short_candidates": len(reverse_candidates),
        "reverse_only_accepted": int(reverse_only.get("trades", 0) or 0),
        "reverse_only_skipped_overlap": int(reverse_only.get("skipped_overlap", 0) or 0),
        "combo_reverse_short_accepted": int(event_type_counts.get("reverse_short", 0) or 0),
        "combo_overlay_skipped_by_base": int(combo.get("base_priority_overlay_skipped", 0) or 0),
    }


def acceptance_gate(label: str, combo: dict[str, Any], base_shadow_summary: dict[str, Any]) -> dict[str, Any]:
    delta = combo.get("delta_vs_shadow_sota", {})
    window_deltas = combo.get("window_deltas_vs_shadow_sota", {})
    year_delta = window_deltas.get("current_year", {})
    last_60d_delta = window_deltas.get("last_60d", {})
    accepted_short = combo.get("overlay_attribution", {}).get("accepted_overlay", {})
    year_short = combo.get("overlay_attribution", {}).get("windows", {}).get("current_year", {})
    max_dd_delta = float(delta.get("max_drawdown_pct", 0.0) or 0.0)
    dd_limit = 0.0 if label == "stable" else 1.0
    checks = {
        "beats_baseline_full_return": float(delta.get("total_return_pct", 0.0) or 0.0) > 0.0,
        "dd_within_candidate_limit": max_dd_delta <= dd_limit,
        "positive_2026_delta": float(year_delta.get("total_return_pct", 0.0) or 0.0) > 0.0,
        "non_negative_last_60d_delta": float(last_60d_delta.get("total_return_pct", 0.0) or 0.0) >= 0.0,
        "accepted_short_count_min_8": int(accepted_short.get("trades", 0) or 0) >= 8,
    }
    if label.startswith("broader"):
        checks["not_single_2026_short_dependent"] = int(year_short.get("trades", 0) or 0) > 1
    return {
        "baseline_full_return_pct": base_shadow_summary.get("total_return_pct"),
        "baseline_max_drawdown_pct": base_shadow_summary.get("max_drawdown_pct"),
        "dd_delta_limit_pct": dd_limit,
        "checks": checks,
        "passes": all(checks.values()),
    }


def reproduce_candidate(
    candidate: dict[str, Any],
    payload: dict[str, Any],
    prepared: Any,
    shadow_events: list[dict[str, Any]],
    fixed_events: list[dict[str, Any]],
    initial_capital: float,
    base_shadow_summary: dict[str, Any],
) -> dict[str, Any]:
    source_events = shadow_events if str(candidate["source_stream"]) == "shadow" else fixed_events
    reverse_candidates = []
    selector_matches = 0
    for event in source_events:
        if not selected_by(event, str(candidate["selector"]), int(candidate["max_quality_score"])):
            continue
        selector_matches += 1
        simulated = simulate_short_trade(
            event=event,
            candles=prepared.c15m,
            trigger_mode=str(candidate["trigger_mode"]),
            target_rr=float(candidate["target_rr"]),
            max_hold_bars=int(candidate["max_hold_bars"]),
            leverage=float(candidate["leverage"]),
            stop_multiplier=float(candidate["stop_multiplier"]),
            max_short_stop_pct=float(candidate["max_short_stop_pct"]),
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
        key: value for key, value in candidate.items() if key != "label"
    }
    reverse_only["score"] = score_result(reverse_only)

    standard_base_events = [standard_sota_event(event) for event in shadow_events]
    standard_overlay_events = [
        standard_reverse_short_event(event, float(candidate["overlay_allocation"]))
        for event in reverse_only["events"]
    ]
    combos = build_combo_results(
        standard_base_events,
        standard_overlay_events,
        initial_capital,
        prepared.end,
        base_shadow_summary,
    )
    combo = next(item for item in combos if str(item["combo_mode"]) == str(candidate["combo_mode"]))
    combo["params"] = reverse_only["params"] | {
        "overlay_allocation": candidate["overlay_allocation"],
        "combo_mode": candidate["combo_mode"],
    }
    combo["overlay_attribution"] = overlay_attribution(combo, initial_capital, prepared.end)
    combo["reverse_short_year_breakdown"] = year_breakdown(
        [
            event for event in combo.get("events", [])
            if str(event.get("event_type") or "") == "reverse_short"
        ],
        initial_capital,
    )
    combo["source_funnel"] = source_funnel(
        source_events,
        selector_matches,
        reverse_candidates,
        reverse_only,
        combo,
    )
    combo["acceptance_gate"] = acceptance_gate(str(candidate["label"]), combo, base_shadow_summary)

    return {
        "label": candidate["label"],
        "reverse_only": reverse_only,
        "combo": combo,
    }


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
    base_shadow_summary = event_stream_summary(shadow["events"], initial_capital, prepared.end)

    candidate_results = [
        reproduce_candidate(
            candidate,
            payload,
            prepared,
            shadow["events"],
            fixed["events"],
            initial_capital,
            base_shadow_summary,
        )
        for candidate in (STABLE_CANDIDATE, BROADER_CANDIDATE, BROADER_GATE_CANDIDATE)
    ]

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
        "candidates": [
            {
                "label": item["label"],
                "reverse_only": compact_result(item["reverse_only"], int(args.sample_trades)),
                "combo": compact_combo_result(item["combo"], int(args.sample_trades)),
            }
            for item in candidate_results
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    print("Baseline shadow SOTA:")
    base = report["baseline_shadow_sota"]
    print(f"  full={base['total_return_pct']:.2f}%/{base['max_drawdown_pct']:.2f}% 2026={base['windows']['current_year']['total_return_pct']:.2f}% trades={base['trades']}")
    for item in report["candidates"]:
        combo = item["combo"]
        delta = combo["delta_vs_shadow_sota"]
        year = combo["windows"]["current_year"]
        year_delta = combo["window_deltas_vs_shadow_sota"]["current_year"]
        funnel = combo["source_funnel"]
        accepted_overlay = combo["overlay_attribution"]["accepted_overlay"]
        year_overlay = combo["overlay_attribution"]["windows"]["current_year"]
        print(
            f"{item['label']:8s} full={combo['total_return_pct']:.2f}%/{combo['max_drawdown_pct']:.2f}% "
            f"delta={delta['total_return_pct']:.2f}%/{delta['max_drawdown_pct']:+.2f}dd "
            f"2026={year['total_return_pct']:.2f}% y_delta={year_delta['total_return_pct']:.2f}% "
            f"trades={combo['trades']}"
        )
        print(
            f"  funnel selector={funnel['selector_matches']} simulated={funnel['simulated_short_candidates']} "
            f"combo_short={funnel['combo_reverse_short_accepted']} skipped_by_base={funnel['combo_overlay_skipped_by_base']}"
        )
        print(
            f"  accepted_short trades={accepted_overlay['trades']} win={accepted_overlay['win_rate_pct']:.2f}% "
            f"compounded={accepted_overlay['compounded_return_pct']:.2f}% "
            f"2026_short_trades={year_overlay['trades']} 2026_short={year_overlay['compounded_return_pct']:.2f}%"
        )
        print(
            f"  gate passes={combo['acceptance_gate']['passes']} "
            f"checks={combo['acceptance_gate']['checks']}"
        )


if __name__ == "__main__":
    main()
