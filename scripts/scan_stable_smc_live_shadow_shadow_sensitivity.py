#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
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
from scripts.replay_stable_smc_live_shadow import (  # noqa: E402
    apply_trailing_rr_modes,
    build_stable_events_for_params,
    replay_live_shadow,
)
from scripts.report_smc_trade_context import daily_candles_from_4h  # noqa: E402
from scripts.reproduce_reverse_short_overlay_candidates import clean_for_json  # noqa: E402
from scripts.research_reverse_short_from_failed_longs import (  # noqa: E402
    event_stream_summary,
    standard_sota_event,
)
from scripts.research_stable_reverse_short_plus_smc_short import (  # noqa: E402
    SMC_CASES,
    build_smc_events,
    replay_base_priority_stable_first,
)
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import (  # noqa: E402
    FIXED_STRUCTURE_PARAMS,
    add_windows,
    replay_shadow_events,
)
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "stable_smc_live_shadow_shadow_sensitivity.json"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shadow sensitivity scan for the current Stable+SMC live-shadow candidate.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--smc-case", default="v2_medium_dispbody05_otherlag4_10x", choices=sorted(SMC_CASES))
    parser.add_argument("--smc-allocation", type=float, default=1.0)
    parser.add_argument("--stable-allocation", type=float, default=1.0)
    parser.add_argument("--stable-selector", default="guarded_weak_loss")
    parser.add_argument("--stable-target-rr", type=float, default=2.875)
    parser.add_argument("--stable-max-hold-bars", type=int, default=40)
    parser.add_argument("--stable-leverage", type=float, default=5.0)
    parser.add_argument("--stable-stop-multiplier", type=float, default=1.0)
    parser.add_argument("--stable-max-short-stop-pct", type=float, default=1.75)
    parser.add_argument("--stage-trigger-rr-mode", default="close", choices=("close", "extreme"))
    parser.add_argument("--time-trailing-rr-mode", default="extreme", choices=("close", "extreme"))
    parser.add_argument("--atr-activation-rr-mode", default="close", choices=("close", "extreme"))
    parser.add_argument("--daily-loss-values", default="0,4,6,8")
    parser.add_argument("--equity-dd-values", default="0,12,15,18")
    parser.add_argument("--equity-cooldown-values", default="0,1,2,3")
    parser.add_argument("--loss-streak-values", default="0,2,3,4")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--sample-trades", type=int, default=5)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def score_shadow_candidate(result: dict[str, Any], baseline: dict[str, Any]) -> float:
    year = result.get("windows", {}).get("current_year", {})
    baseline_year = baseline.get("windows", {}).get("current_year", {})
    dd_penalty = max(0.0, float(result.get("max_drawdown_pct", 0.0) or 0.0) - float(baseline.get("max_drawdown_pct", 0.0) or 0.0))
    reject_penalty = max(
        0,
        int(result.get("decision_counts", {}).get("by_decision", {}).get("rejected", 0) or 0)
        - int(baseline.get("decision_counts", {}).get("by_decision", {}).get("rejected", 0) or 0),
    )
    return round(
        float(result.get("total_return_pct", 0.0) or 0.0)
        + float(year.get("total_return_pct", 0.0) or 0.0) * 200.0
        - dd_penalty * 50000.0
        - reject_penalty * 15000.0
        - max(0.0, float(baseline_year.get("total_return_pct", 0.0) or 0.0) - float(year.get("total_return_pct", 0.0) or 0.0)) * 200.0,
        4,
    )


def main() -> None:
    args = parse_args()
    base_payload = load_config_payload(Path(args.config))
    payload, pressure_params = apply_pressure_params(base_payload, Path(args.pressure_params))
    payload, trailing_rr_modes = apply_trailing_rr_modes(
        payload,
        stage_trigger_rr_mode=str(args.stage_trigger_rr_mode),
        time_trailing_rr_mode=str(args.time_trailing_rr_mode),
        atr_activation_rr_mode=str(args.atr_activation_rr_mode),
    )
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

    daily = daily_candles_from_4h(prepared.c4h)
    h4_highs, h4_lows = precompute_swings(prepared.c4h, n=2, lookback=80)
    d1_highs, d1_lows = precompute_swings(daily, n=2, lookback=20)

    results: list[dict[str, Any]] = []
    baseline_live: dict[str, Any] | None = None

    for daily_loss, equity_dd, cooldown, loss_streak in itertools.product(
        parse_float_list(args.daily_loss_values),
        parse_float_list(args.equity_dd_values),
        parse_int_list(args.equity_cooldown_values),
        parse_int_list(args.loss_streak_values),
    ):
        if equity_dd <= 0 and cooldown > 0:
            continue
        if equity_dd > 0 and cooldown <= 0:
            continue

        shadow = replay_shadow_events(
            fixed["events"],
            initial_capital,
            daily_loss_stop_pct=float(daily_loss),
            equity_drawdown_stop_pct=float(equity_dd),
            consecutive_loss_stop=int(loss_streak),
            equity_drawdown_cooldown_days=int(cooldown),
        )
        shadow_events = shadow["events"]
        base_shadow_summary = event_stream_summary(shadow_events, initial_capital, prepared.end)
        base_events = [standard_sota_event(event) for event in shadow_events]

        stable_events, stable_summary = build_stable_events_for_params(
            payload,
            prepared,
            shadow_events,
            float(args.stable_allocation),
            float(args.stable_target_rr),
            int(args.stable_max_hold_bars),
            float(args.stable_leverage),
            float(args.stable_stop_multiplier),
            float(args.stable_max_short_stop_pct),
            selector=str(args.stable_selector),
        )
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

        reference = replay_base_priority_stable_first(
            base_events,
            stable_events,
            smc_events,
            initial_capital,
            prepared.end,
            base_shadow_summary,
        )
        live, _ = replay_live_shadow(
            base_events + stable_events + smc_events,
            initial_capital,
            prepared.end,
            base_shadow_summary,
        )
        live["reference_base_priority"] = {
            "total_return_pct": reference["total_return_pct"],
            "max_drawdown_pct": reference["max_drawdown_pct"],
            "current_year_return_pct": reference["windows"]["current_year"]["total_return_pct"],
        }
        live["shadow_params"] = {
            "daily_loss_stop_pct": float(daily_loss),
            "equity_drawdown_stop_pct": float(equity_dd),
            "equity_drawdown_cooldown_days": int(cooldown),
            "consecutive_loss_stop": int(loss_streak),
        }
        live["shadow_summary"] = add_windows(dict(shadow), initial_capital)
        live["stable_summary"] = stable_summary
        live["smc_summary"] = smc_summary
        live["trailing_rr_modes"] = trailing_rr_modes
        results.append(live)

        if (
            float(daily_loss) == 6.0
            and float(equity_dd) == 15.0
            and int(cooldown) == 2
            and int(loss_streak) == 0
        ):
            baseline_live = live

    if baseline_live is None:
        raise RuntimeError("Baseline shadow combination 6/15/2/0 was not included in the scan.")

    for item in results:
        item["score"] = score_shadow_candidate(item, baseline_live)
        item["delta_vs_baseline"] = {
            "total_return_pct": round(float(item.get("total_return_pct", 0.0) or 0.0) - float(baseline_live.get("total_return_pct", 0.0) or 0.0), 4),
            "max_drawdown_pct": round(float(item.get("max_drawdown_pct", 0.0) or 0.0) - float(baseline_live.get("max_drawdown_pct", 0.0) or 0.0), 4),
            "current_year_return_pct": round(
                float(item.get("windows", {}).get("current_year", {}).get("total_return_pct", 0.0) or 0.0)
                - float(baseline_live.get("windows", {}).get("current_year", {}).get("total_return_pct", 0.0) or 0.0),
                4,
            ),
        }

    by_return = sorted(results, key=lambda item: float(item.get("total_return_pct", 0.0) or 0.0), reverse=True)
    by_score = sorted(results, key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    stable = [
        item for item in by_score
        if float(item.get("max_drawdown_pct", 0.0) or 0.0) <= float(baseline_live.get("max_drawdown_pct", 0.0) or 0.0)
        and float(item.get("windows", {}).get("current_year", {}).get("total_return_pct", 0.0) or 0.0)
        >= float(baseline_live.get("windows", {}).get("current_year", {}).get("total_return_pct", 0.0) or 0.0)
        and int(item.get("decision_counts", {}).get("by_decision", {}).get("accepted", 0) or 0)
        >= int(baseline_live.get("decision_counts", {}).get("by_decision", {}).get("accepted", 0) or 0)
    ]

    report = {
        "metadata": {
            "config": str(Path(args.config).resolve()),
            "pressure_params": str(Path(args.pressure_params).resolve()),
            "pressure_params_applied": pressure_params,
            "trailing_rr_modes": trailing_rr_modes,
            "stable_params": {
                "stable_target_rr": float(args.stable_target_rr),
                "stable_max_hold_bars": int(args.stable_max_hold_bars),
                "stable_leverage": float(args.stable_leverage),
                "stable_stop_multiplier": float(args.stable_stop_multiplier),
                "stable_max_short_stop_pct": float(args.stable_max_short_stop_pct),
                "stable_allocation": float(args.stable_allocation),
                "smc_case": args.smc_case,
                "smc_allocation": float(args.smc_allocation),
            },
            "search_space": {
                "daily_loss_values": parse_float_list(args.daily_loss_values),
                "equity_dd_values": parse_float_list(args.equity_dd_values),
                "equity_cooldown_values": parse_int_list(args.equity_cooldown_values),
                "loss_streak_values": parse_int_list(args.loss_streak_values),
                "candidate_count": len(results),
            },
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
        },
        "baseline_shadow": clean_for_json(baseline_live),
        "top_by_return": [clean_for_json(item) for item in by_return[: int(args.top)]],
        "top_by_score": [clean_for_json(item) for item in by_score[: int(args.top)]],
        "stable_candidates": [clean_for_json(item) for item in stable[: int(args.top)]],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(output)
    print(
        "Baseline "
        f"full={baseline_live['total_return_pct']:.2f}%/{baseline_live['max_drawdown_pct']:.2f}% "
        f"2026={baseline_live['windows']['current_year']['total_return_pct']:.2f}% "
        f"shadow={baseline_live['shadow_params']}"
    )
    print("Top shadow candidates by score:")
    for idx, item in enumerate(by_score[: min(int(args.top), 10)], start=1):
        year = item["windows"]["current_year"]
        delta = item["delta_vs_baseline"]
        print(
            f"{idx:02d} full={item['total_return_pct']:.2f}%/{item['max_drawdown_pct']:.2f}% "
            f"2026={year['total_return_pct']:.2f}% "
            f"d_full={delta['total_return_pct']:+.2f}% d_dd={delta['max_drawdown_pct']:+.2f} "
            f"shadow={item['shadow_params']}"
        )


if __name__ == "__main__":
    main()
