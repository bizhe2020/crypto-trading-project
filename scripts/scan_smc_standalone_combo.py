#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, max_drawdown_from_capitals, run_engine, trade_dataframe  # noqa: E402
from scripts.report_smc_trade_context import daily_candles_from_4h, load_best_shadow_params  # noqa: E402
from scripts.reproduce_main_baseline_shadow import DEFAULT_OUTPUT as DEFAULT_MAIN_SHADOW_REPORT  # noqa: E402
from scripts.research_smc_standalone_v1 import build_event_scan_args, clean_for_json  # noqa: E402
from scripts.research_smc_standalone_v1 import group_summary, scan_events, summarize_rows, trade_rows_for_events, yearly_summary  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, add_windows, replay_shadow_events  # noqa: E402
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_standalone_combo_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study portfolio combinations between promoted main strategy and standalone SMC sleeves.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--main-shadow-report", default=str(DEFAULT_MAIN_SHADOW_REPORT))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--standalone-cases", default="top1_ote_other_2r,fvg15_balanced")
    parser.add_argument("--standalone-risk-fraction-values", default="1.0,2.0,3.0")
    parser.add_argument("--main-weight-values", default="1.0,0.95,0.9,0.8,0.7,0.5")
    parser.add_argument("--standalone-max-open-positions", type=int, default=1)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def standalone_case_params(name: str) -> dict[str, Any]:
    cases: dict[str, dict[str, Any]] = {
        "baseline": {
            "target_rr": 2.0,
            "require_fvg_touch": True,
            "allow_ote_only": False,
            "allowed_time_buckets": "all",
            "require_d1_bias_align": False,
        },
        "top1_ote_other_2r": {
            "target_rr": 2.0,
            "require_fvg_touch": False,
            "allow_ote_only": True,
            "allowed_time_buckets": "other",
            "require_d1_bias_align": False,
        },
        "fvg15_balanced": {
            "target_rr": 1.5,
            "require_fvg_touch": True,
            "allow_ote_only": False,
            "allowed_time_buckets": "other+ny_am_killzone+asia_evening_ny",
            "require_d1_bias_align": False,
        },
    }
    if name not in cases:
        raise ValueError(f"Unsupported standalone case: {name}")
    return dict(cases[name])


def canonicalize_event(event: dict[str, Any]) -> dict[str, Any]:
    row = dict(event)
    row["entry_time"] = normalize_ts(row["entry_time"])
    row["exit_time"] = normalize_ts(row["exit_time"])
    row["return"] = float(row.get("return", 0.0) or 0.0)
    row["direction"] = str(row.get("direction") or "")
    return row


def normalize_event_list(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [canonicalize_event(event) for event in events]
    normalized.sort(key=lambda item: (item["entry_time"], item["exit_time"]))
    return normalized


def apply_max_open_positions(rows: list[dict[str, Any]], max_open_positions: int) -> tuple[list[dict[str, Any]], int]:
    if max_open_positions <= 0:
        ordered = sorted(rows, key=lambda item: (normalize_ts(item["entry_time"]), normalize_ts(item["exit_time"])))
        return ordered, 0
    ordered = sorted(rows, key=lambda item: (normalize_ts(item["entry_time"]), normalize_ts(item["exit_time"])))
    active_exits: list[pd.Timestamp] = []
    accepted: list[dict[str, Any]] = []
    skipped = 0
    for row in ordered:
        entry_time = normalize_ts(row["entry_time"])
        while active_exits and active_exits[0] <= entry_time:
            heapq.heappop(active_exits)
        if len(active_exits) >= max_open_positions:
            skipped += 1
            continue
        accepted.append(row)
        heapq.heappush(active_exits, normalize_ts(row["exit_time"]))
    return accepted, skipped


def scale_rows(rows: list[dict[str, Any]], risk_fraction: float) -> list[dict[str, Any]]:
    scaled: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["return"] = float(copied["rr_result"]) * float(risk_fraction) / 100.0
        copied["position_risk_fraction"] = float(risk_fraction)
        scaled.append(copied)
    return scaled


def build_equity_curve(events: list[dict[str, Any]], start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    start_day = start.normalize()
    end_day = max(start_day, end.normalize())
    index = pd.date_range(start_day, end_day, freq="D", tz="UTC")
    grouped: dict[pd.Timestamp, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[event["exit_time"].normalize()].append(event)
    capital = 1.0
    values: list[float] = []
    for day in index:
        for event in sorted(grouped.get(day, []), key=lambda item: (item["exit_time"], item["entry_time"])):
            capital *= 1.0 + float(event["return"])
        values.append(capital)
    return pd.Series(values, index=index, dtype="float64")


def curve_summary(curve: pd.Series) -> dict[str, Any]:
    if curve.empty:
        return {"total_return_pct": 0.0, "max_drawdown_pct": 0.0}
    return {
        "total_return_pct": round((float(curve.iloc[-1]) - 1.0) * 100.0, 2),
        "max_drawdown_pct": round(max_drawdown_from_capitals(curve.tolist(), 1.0), 2),
    }


def summarize_window(events: list[dict[str, Any]], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    selected = [event for event in events if start <= event["exit_time"] <= end]
    curve = build_equity_curve(selected, start, end)
    summary = curve_summary(curve)
    summary["trades"] = len(selected)
    return summary


def combined_window_summary(
    main_events: list[dict[str, Any]],
    smc_events: list[dict[str, Any]],
    main_weight: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    main_selected = [event for event in main_events if start <= event["exit_time"] <= end]
    smc_selected = [event for event in smc_events if start <= event["exit_time"] <= end]
    main_curve = build_equity_curve(main_selected, start, end)
    smc_curve = build_equity_curve(smc_selected, start, end)
    combo = main_curve * float(main_weight) + smc_curve * (1.0 - float(main_weight))
    summary = curve_summary(combo)
    summary["main_trades"] = len(main_selected)
    summary["smc_trades"] = len(smc_selected)
    return summary


def overlap_stats(a_events: list[dict[str, Any]], b_events: list[dict[str, Any]]) -> dict[str, Any]:
    overlap_a = 0
    same_dir_a = 0
    opposite_dir_a = 0
    touched_b: set[int] = set()
    for event_a in a_events:
        hit = False
        same = False
        opposite = False
        for idx_b, event_b in enumerate(b_events):
            if event_b["entry_time"] >= event_a["exit_time"] or event_b["exit_time"] <= event_a["entry_time"]:
                continue
            hit = True
            touched_b.add(idx_b)
            if event_a["direction"] == event_b["direction"]:
                same = True
            else:
                opposite = True
        if hit:
            overlap_a += 1
        if same:
            same_dir_a += 1
        if opposite:
            opposite_dir_a += 1
    return {
        "a_trades": len(a_events),
        "b_trades": len(b_events),
        "a_overlap_trades": overlap_a,
        "a_overlap_pct": round(overlap_a / len(a_events) * 100.0, 2) if a_events else 0.0,
        "a_same_direction_overlap_pct": round(same_dir_a / len(a_events) * 100.0, 2) if a_events else 0.0,
        "a_opposite_direction_overlap_pct": round(opposite_dir_a / len(a_events) * 100.0, 2) if a_events else 0.0,
        "b_overlap_trades": len(touched_b),
        "b_overlap_pct": round(len(touched_b) / len(b_events) * 100.0, 2) if b_events else 0.0,
    }


def candidate_score(result: dict[str, Any], baseline_main: dict[str, Any]) -> float:
    full = float(result["overall"]["total_return_pct"])
    maxdd = float(result["overall"]["max_drawdown_pct"])
    year = float(result["windows"]["current_year"]["total_return_pct"])
    delta_full = full - float(baseline_main["overall"]["total_return_pct"])
    delta_dd = float(baseline_main["overall"]["max_drawdown_pct"]) - maxdd
    return round(delta_full + year * 8.0 + delta_dd * 120.0, 4)


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
    shadow_params = load_best_shadow_params(Path(args.pressure_params))
    main_shadow_report = Path(args.main_shadow_report)
    if main_shadow_report.exists():
        cached = json.loads(main_shadow_report.read_text())
        main_summary = dict(cached["summary"])
        main_events = normalize_event_list(cached["events"])
    else:
        metrics, engine = run_engine(payload, prepared, args.start_date)
        base_trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
        fixed = expansion_overlay(base_trades, float(metrics.get("initial_capital", 1000.0)), FIXED_STRUCTURE_PARAMS, include_events=True)
        main_shadow = replay_shadow_events(
            fixed["events"],
            float(metrics.get("initial_capital", 1000.0)),
            daily_loss_stop_pct=shadow_params["daily_loss_stop_pct"],
            equity_drawdown_stop_pct=shadow_params["equity_drawdown_stop_pct"],
            consecutive_loss_stop=shadow_params["consecutive_loss_stop"],
            equity_drawdown_cooldown_days=shadow_params["equity_drawdown_cooldown_days"],
        )
        main_summary = add_windows(dict(main_shadow), float(metrics.get("initial_capital", 1000.0)))
        main_events = normalize_event_list(main_shadow["events"])

    daily = daily_candles_from_4h(prepared.c4h)
    h4_highs, h4_lows = precompute_swings(prepared.c4h, n=2, lookback=80)
    d1_highs, d1_lows = precompute_swings(daily, n=2, lookback=20)
    event_cache: dict[float, list[Any]] = {}

    standalone_case_summaries: dict[str, Any] = {}
    combo_results: list[dict[str, Any]] = []
    main_end = max(
        max((event["exit_time"] for event in main_events), default=normalize_ts(args.start_date)),
        normalize_ts(args.start_date),
    )
    full_main_curve = build_equity_curve(main_events, normalize_ts(args.start_date), main_end)

    for case_name in parse_str_list(args.standalone_cases):
        case_params = standalone_case_params(case_name)
        target_rr = float(case_params["target_rr"])
        if target_rr not in event_cache:
            scan_namespace = argparse.Namespace(
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
                target_rr=target_rr,
            )
            event_cache[target_rr] = scan_events(prepared.c15m, build_event_scan_args(scan_namespace))

        runtime_args = argparse.Namespace(
            target_rr=target_rr,
            require_confirmed_retest=True,
            require_fvg_touch=bool(case_params["require_fvg_touch"]),
            allow_ote_only=bool(case_params["allow_ote_only"]),
            require_htf_bias_align=True,
            require_h4_bias_align=True,
            require_d1_bias_align=bool(case_params["require_d1_bias_align"]),
            allowed_time_buckets=str(case_params["allowed_time_buckets"]),
            position_risk_fraction=1.0,
            initial_capital=1000.0,
        )
        raw_rows = trade_rows_for_events(
            event_cache[target_rr],
            prepared,
            daily,
            h4_highs,
            h4_lows,
            d1_highs,
            d1_lows,
            runtime_args,
        )
        slot_rows, slot_skipped = apply_max_open_positions(raw_rows, args.standalone_max_open_positions)
        standalone_case_summaries[case_name] = {
            "params": case_params,
            "raw_trades": len(raw_rows),
            "slot_trades": len(slot_rows),
            "slot_skipped": slot_skipped,
        }

        for risk_fraction in parse_float_list(args.standalone_risk_fraction_values):
            scaled_rows = scale_rows(slot_rows, risk_fraction)
            smc_events = normalize_event_list(scaled_rows)
            combo_end = max(
                main_end,
                max((event["exit_time"] for event in smc_events), default=normalize_ts(args.start_date)),
            )
            windows = {
                "current_year": pd.Timestamp(f"{combo_end.year}-01-01", tz="UTC"),
                "last_60d": combo_end - pd.Timedelta(days=60),
                "last_30d": combo_end - pd.Timedelta(days=30),
            }
            smc_summary = {
                "overall": summarize_rows(scaled_rows, 1000.0),
                "by_direction": group_summary(scaled_rows, "direction", 1000.0),
                "yearly": yearly_summary(scaled_rows, 1000.0),
            }
            standalone_case_summaries[case_name][f"risk_{risk_fraction:g}"] = smc_summary
            overlap = overlap_stats(smc_events, main_events)
            full_main_curve_aligned = build_equity_curve(main_events, normalize_ts(args.start_date), combo_end)
            full_smc_curve = build_equity_curve(smc_events, normalize_ts(args.start_date), combo_end)
            daily_corr = full_main_curve_aligned.pct_change().fillna(0.0).corr(full_smc_curve.pct_change().fillna(0.0))

            for main_weight in parse_float_list(args.main_weight_values):
                smc_weight = 1.0 - float(main_weight)
                combo_curve = full_main_curve_aligned * float(main_weight) + full_smc_curve * smc_weight
                overall = curve_summary(combo_curve)
                candidate = {
                    "standalone_case": case_name,
                    "standalone_risk_fraction": risk_fraction,
                    "weights": {
                        "main": round(float(main_weight), 4),
                        "smc": round(float(smc_weight), 4),
                    },
                    "standalone_slot": {
                        "max_open_positions": args.standalone_max_open_positions,
                        "raw_trades": len(raw_rows),
                        "slot_trades": len(slot_rows),
                        "slot_skipped": slot_skipped,
                    },
                    "main_summary": {
                        "overall": {
                            "total_return_pct": main_summary["total_return_pct"],
                            "max_drawdown_pct": main_summary["max_drawdown_pct"],
                        },
                        "current_year": main_summary["windows"]["current_year"],
                    },
                    "smc_summary": smc_summary,
                    "overall": overall,
                    "windows": {
                        name: combined_window_summary(main_events, smc_events, float(main_weight), start, combo_end)
                        for name, start in windows.items()
                    },
                    "overlap": overlap,
                    "daily_return_corr": None if pd.isna(daily_corr) else round(float(daily_corr), 4),
                }
                candidate["score"] = candidate_score(candidate, {"overall": main_summary, "windows": main_summary["windows"]})
                combo_results.append(candidate)

    combo_results.sort(key=lambda item: item["score"], reverse=True)
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
            "main_shadow_report": str(main_shadow_report.resolve()),
        },
        "main_baseline": {
            "overall": {
                "total_return_pct": main_summary["total_return_pct"],
                "max_drawdown_pct": main_summary["max_drawdown_pct"],
                "accepted_trades": main_summary["accepted_trades"],
                "skipped_trades": main_summary["skipped_trades"],
            },
            "windows": main_summary["windows"],
        },
        "standalone_cases": standalone_case_summaries,
        "top": combo_results[: args.top],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    for idx, item in enumerate(combo_results[: args.top], start=1):
        print(
            f"{idx:02d} score={item['score']:.2f} case={item['standalone_case']} risk={item['standalone_risk_fraction']:.1f} "
            f"w_main={item['weights']['main']:.2f} full={item['overall']['total_return_pct']:.2f}%/"
            f"{item['overall']['max_drawdown_pct']:.2f}% 2026={item['windows']['current_year']['total_return_pct']:.2f}% "
            f"corr={item['daily_return_corr']} overlap={item['overlap']['a_overlap_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
