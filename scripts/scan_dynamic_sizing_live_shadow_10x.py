#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.confirmed_multiframe_score_utils import (  # noqa: E402
    align_confirmed_mapping,
    passes_score_gate,
    resample_confirmed_1h,
    score_snapshot,
)
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.live_shadow_utils import (  # noqa: E402
    add_combo_deltas,
    add_standard_windows,
    clean_for_json,
    compact_combo_result,
    parse_float_list,
    standard_event_summary,
    standard_sota_event,
)
from scripts.replay_sota_smc_live_shadow import apply_trailing_rr_modes, replay_live_shadow  # noqa: E402
from scripts.report_smc_trade_context import daily_candles_from_4h  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402
from scripts.score_bucket_sizing_utils import apply_score_bucket_sizing_to_events  # noqa: E402
from scripts.smc_live_utils import SMC_CASES, build_smc_events  # noqa: E402
from strategy.scalp_robust_v2_core import precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "dynamic_sizing_live_shadow_10x_scan.json"
RR_MODE_CHOICES = ("close", "extreme")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan 10x dynamic effective-leverage sizing under SOTA+SMC live-shadow replay.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--stage-trigger-rr-mode", default="close", choices=RR_MODE_CHOICES)
    parser.add_argument("--time-trailing-rr-mode", default="extreme", choices=RR_MODE_CHOICES)
    parser.add_argument("--atr-activation-rr-mode", default="extreme", choices=RR_MODE_CHOICES)
    parser.add_argument("--smc-case", default="v2_medium_dispbody05_otherlag4_10x", choices=sorted(SMC_CASES))
    parser.add_argument("--smc-allocation", type=float, default=1.0)
    parser.add_argument("--sota-score-net-min", type=int, default=3)
    parser.add_argument("--sota-score-bull-min", type=int, default=8)
    parser.add_argument("--sota-score-bear-max", type=int, default=6)
    parser.add_argument("--sota-score-conflict-mode", default="any", choices=("any", "conflict", "clean"))
    parser.add_argument("--enable-long-score-bucket-sizing", action="store_true")
    parser.add_argument(
        "--long-score-bucket-sizing-rules-json",
        default="",
        help="Optional JSON array/dict for long score bucket sizing rules.",
    )
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=12.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=4)
    parser.add_argument("--base-leverage-values", default="3.5,4.0,4.5")
    parser.add_argument("--high-growth-leverage-values", default="6.5,7.5,8.0")
    parser.add_argument("--tight-stop-leverage-values", default="7.5,8.0")
    parser.add_argument("--max-effective-leverage-values", default="7.5,8.0")
    parser.add_argument("--defense-leverage-values", default="1.5,2.0,2.5")
    parser.add_argument("--drawdown-leverage-values", default="1.5,2.0")
    parser.add_argument("--unhealthy-leverage-values", default="1.5,2.0")
    parser.add_argument("--failed-breakout-guard-leverage-values", default="1.5,2.0,2.5")
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--sample-trades", type=int, default=0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def parse_score_bucket_rules(raw: str) -> Any:
    if not str(raw or "").strip():
        return None
    return json.loads(raw)


def dynamic_params_from_base(
    *,
    base: dict[str, Any],
    base_leverage: float,
    high_growth_leverage: float,
    tight_stop_leverage: float,
    max_effective_leverage: float,
    defense_leverage: float,
    drawdown_leverage: float,
    unhealthy_leverage: float,
    failed_breakout_guard_leverage: float,
) -> dict[str, Any]:
    params = dict(base)
    params.update(
        {
            "base_leverage": min(float(base_leverage), float(max_effective_leverage)),
            "high_growth_leverage": min(float(high_growth_leverage), float(max_effective_leverage)),
            "tight_stop_leverage": min(float(tight_stop_leverage), float(max_effective_leverage)),
            "max_effective_leverage": float(max_effective_leverage),
            "defense_leverage": min(float(defense_leverage), float(max_effective_leverage)),
            "drawdown_leverage": min(float(drawdown_leverage), float(max_effective_leverage)),
            "unhealthy_leverage": min(float(unhealthy_leverage), float(max_effective_leverage)),
            "recovery_leverage": min(float(drawdown_leverage), float(max_effective_leverage)),
            "failed_breakout_guard_leverage": min(float(failed_breakout_guard_leverage), float(max_effective_leverage)),
        }
    )
    return params


def params_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    values = product(
        parse_float_list(args.base_leverage_values),
        parse_float_list(args.high_growth_leverage_values),
        parse_float_list(args.tight_stop_leverage_values),
        parse_float_list(args.max_effective_leverage_values),
        parse_float_list(args.defense_leverage_values),
        parse_float_list(args.drawdown_leverage_values),
        parse_float_list(args.unhealthy_leverage_values),
        parse_float_list(args.failed_breakout_guard_leverage_values),
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for combo in values:
        params = dynamic_params_from_base(
            base=FIXED_STRUCTURE_PARAMS,
            base_leverage=combo[0],
            high_growth_leverage=combo[1],
            tight_stop_leverage=combo[2],
            max_effective_leverage=combo[3],
            defense_leverage=combo[4],
            drawdown_leverage=combo[5],
            unhealthy_leverage=combo[6],
            failed_breakout_guard_leverage=combo[7],
        )
        if params["base_leverage"] > params["max_effective_leverage"]:
            continue
        key = json.dumps({k: params[k] for k in PARAM_KEYS}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(params)
    return out


PARAM_KEYS = [
    "base_leverage",
    "high_growth_leverage",
    "tight_stop_leverage",
    "max_effective_leverage",
    "defense_leverage",
    "drawdown_leverage",
    "unhealthy_leverage",
    "recovery_leverage",
    "failed_breakout_guard_leverage",
]


def event_key(event: dict[str, Any]) -> str:
    return f"{event.get('event_type')}|{int(event.get('entry_idx', 0) or 0)}|{int(event.get('exit_idx', 0) or 0)}"


def score_cache_get(
    cache: dict[int, dict[str, Any]],
    prepared: Any,
    c1h: list[Any],
    mapping_1h: list[int],
    entry_idx: int,
) -> dict[str, Any]:
    if entry_idx not in cache:
        cache[entry_idx] = asdict(score_snapshot(prepared, c1h, mapping_1h, entry_idx))
    return cache[entry_idx]


def apply_cached_score_gate(
    prepared: Any,
    c1h: list[Any],
    mapping_1h: list[int],
    score_cache: dict[int, dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    net_min: int,
    bull_min: int,
    bear_max: int,
    conflict_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    removed = 0
    for event in events:
        entry_idx = int(event.get("entry_idx", 0) or 0)
        enriched = dict(event)
        enriched.update(score_cache_get(score_cache, prepared, c1h, mapping_1h, entry_idx))
        if passes_score_gate(
            enriched,
            net_min=int(net_min),
            bull_min=int(bull_min),
            bear_max=int(bear_max),
            conflict_mode=str(conflict_mode),
        ):
            filtered.append(enriched)
        else:
            removed += 1
    return filtered, {
        "enabled": True,
        "rule": {
            "net_min": int(net_min),
            "bull_min": int(bull_min),
            "bear_max": int(bear_max),
            "conflict_mode": str(conflict_mode),
        },
        "original_candidates": len(events),
        "filtered_candidates": len(filtered),
        "removed_candidates": removed,
    }


def compact(result: dict[str, Any], sample_trades: int) -> dict[str, Any]:
    payload = compact_combo_result(result, sample_trades)
    payload["decision_counts"] = result.get("decision_counts", {})
    return payload


def sort_by_2026(item: dict[str, Any]) -> tuple[float, float, float, float]:
    live = item["live_shadow"]
    current_year = live.get("windows", {}).get("current_year", {})
    return (
        float(current_year.get("total_return_pct", 0.0) or 0.0),
        -float(current_year.get("max_drawdown_pct", 0.0) or 0.0),
        float(live.get("total_return_pct", 0.0) or 0.0),
        -float(live.get("max_drawdown_pct", 0.0) or 0.0),
    )


def sort_by_return(item: dict[str, Any]) -> tuple[float, float, float]:
    live = item["live_shadow"]
    current_year = live.get("windows", {}).get("current_year", {})
    return (
        float(live.get("total_return_pct", 0.0) or 0.0),
        -float(live.get("max_drawdown_pct", 0.0) or 0.0),
        float(current_year.get("total_return_pct", 0.0) or 0.0),
    )


def summarize_overlay(overlay: dict[str, Any]) -> dict[str, Any]:
    current_year = overlay.get("windows", {}).get("current_year", {})
    return {
        "total_return_pct": overlay.get("total_return_pct"),
        "max_drawdown_pct": overlay.get("max_drawdown_pct"),
        "current_year_return_pct": current_year.get("total_return_pct"),
        "current_year_max_drawdown_pct": current_year.get("max_drawdown_pct"),
        "accepted_trades": overlay.get("accepted_trades"),
        "skipped_trades": overlay.get("skipped_trades"),
        "avg_effective_leverage": overlay.get("avg_effective_leverage"),
        "max_effective_leverage_seen": overlay.get("max_effective_leverage_seen"),
        "failure_counts": overlay.get("failure_counts", {}),
        "risk_mode_counts": overlay.get("risk_mode_counts", {}),
        "accepted_risk_mode_counts": overlay.get("accepted_risk_mode_counts", {}),
        "failed_breakout_guard_applied": overlay.get("failed_breakout_guard_applied"),
    }


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
    payload["replay_sync_entry_to_signal_price"] = True
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
    score_cache: dict[int, dict[str, Any]] = {}

    candidates: list[dict[str, Any]] = []
    grid = params_grid(args)
    for idx, params in enumerate(grid, start=1):
        if idx == 1 or idx % 100 == 0:
            print(f"[{idx}/{len(grid)}] scanning dynamic sizing", flush=True)
        overlay = expansion_overlay(trades, initial_capital, params, include_events=True)
        shadow = replay_shadow_events(
            overlay["events"],
            initial_capital,
            daily_loss_stop_pct=float(args.daily_loss_stop_pct),
            equity_drawdown_stop_pct=float(args.equity_drawdown_stop_pct),
            consecutive_loss_stop=int(args.consecutive_loss_stop),
            equity_drawdown_cooldown_days=int(args.equity_drawdown_cooldown_days),
        )
        raw_sota = [standard_sota_event(event) for event in shadow["events"]]
        base_events, score_gate = apply_cached_score_gate(
            prepared,
            c1h,
            mapping_1h,
            score_cache,
            raw_sota,
            net_min=int(args.sota_score_net_min),
            bull_min=int(args.sota_score_bull_min),
            bear_max=int(args.sota_score_bear_max),
            conflict_mode=str(args.sota_score_conflict_mode),
        )
        base_events, long_score_bucket_sizing = apply_score_bucket_sizing_to_events(
            base_events,
            enabled=bool(args.enable_long_score_bucket_sizing),
            rules=parse_score_bucket_rules(str(args.long_score_bucket_sizing_rules_json)),
        )
        baseline = standard_event_summary(base_events, initial_capital, "entry_idx")
        baseline = add_standard_windows(baseline, initial_capital, prepared.end, "entry_idx")
        live, decisions = replay_live_shadow(base_events + smc_events, initial_capital, prepared.end, baseline)
        live = add_combo_deltas(live, baseline)
        candidates.append(
            {
                "params": {key: params.get(key) for key in PARAM_KEYS},
                "dynamic_overlay": summarize_overlay(overlay),
                "shadow_risk_gate": {
                    "daily_loss_stop_pct": float(args.daily_loss_stop_pct),
                    "equity_drawdown_stop_pct": float(args.equity_drawdown_stop_pct),
                    "equity_drawdown_cooldown_days": int(args.equity_drawdown_cooldown_days),
                    "consecutive_loss_stop": int(args.consecutive_loss_stop),
                    "accepted_sota_before_score_gate": len(raw_sota),
                    "skipped_trades": int(shadow.get("skipped_trades", 0) or 0),
                    "trigger_counts": shadow.get("trigger_counts", {}),
                },
                "sota_score_gate": score_gate,
                "long_score_bucket_sizing": long_score_bucket_sizing,
                "smc_candidates": len(smc_events),
                "live_shadow": compact(live, int(args.sample_trades)),
                "decision_count_total": len(decisions),
            }
        )

    ranked_by_2026 = sorted(candidates, key=sort_by_2026, reverse=True)
    ranked_by_return = sorted(candidates, key=sort_by_return, reverse=True)
    report = {
        "metadata": {
            "config": str(Path(args.config).resolve()),
            "pressure_params": str(Path(args.pressure_params).resolve()),
            "pressure_params_applied": pressure_params,
            "trailing_rr_modes": trailing_rr_modes,
            "start_date": str(args.start_date),
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "confirmed_4h_only": True,
            "replay_sync_entry_to_signal_price": True,
            "base_engine": {
                "total_return_pct": round(float(metrics.get("total_return_pct", 0.0) or 0.0), 2),
                "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0) or 0.0), 2),
                "total_trades": int(metrics.get("total_trades", 0) or 0),
            },
            "sota_score_gate": {
                "net_min": int(args.sota_score_net_min),
                "bull_min": int(args.sota_score_bull_min),
                "bear_max": int(args.sota_score_bear_max),
                "conflict_mode": str(args.sota_score_conflict_mode),
            },
            "long_score_bucket_sizing": {
                "enabled": bool(args.enable_long_score_bucket_sizing),
                "rules": parse_score_bucket_rules(str(args.long_score_bucket_sizing_rules_json)),
            },
            "shadow_risk_gate": {
                "daily_loss_stop_pct": float(args.daily_loss_stop_pct),
                "equity_drawdown_stop_pct": float(args.equity_drawdown_stop_pct),
                "equity_drawdown_cooldown_days": int(args.equity_drawdown_cooldown_days),
                "consecutive_loss_stop": int(args.consecutive_loss_stop),
            },
            "smc_case": str(args.smc_case),
            "smc_allocation": float(args.smc_allocation),
            "smc_summary": smc_summary,
            "candidate_count": len(candidates),
        },
        "top_by_2026": ranked_by_2026[: int(args.top_n)],
        "top_by_return": ranked_by_return[: int(args.top_n)],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output)
    for idx, item in enumerate(ranked_by_2026[:8], start=1):
        live = item["live_shadow"]
        year = live.get("windows", {}).get("current_year", {})
        overlay = item["dynamic_overlay"]
        print(
            f"{idx:02d} 2026={float(year.get('total_return_pct', 0.0) or 0.0):.2f}%/"
            f"{float(year.get('max_drawdown_pct', 0.0) or 0.0):.2f}% "
            f"full={float(live.get('total_return_pct', 0.0) or 0.0):.2f}%/"
            f"{float(live.get('max_drawdown_pct', 0.0) or 0.0):.2f}% "
            f"avgLev={overlay.get('avg_effective_leverage')} params={item['params']}"
        )


if __name__ == "__main__":
    main()
