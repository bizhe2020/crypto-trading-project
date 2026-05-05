#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import subprocess
import sys
import tempfile
import time
from types import MethodType, SimpleNamespace
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.okx_executor import ExecutorConfig, OkxExecutionEngine  # noqa: E402
from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.replay_stable_live_shadow import (  # noqa: E402
    add_standard_windows,
    build_stable_events,
    clean_for_json,
    standard_event_summary,
    standard_sota_event,
    to_candidate,
)
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402
from scripts.smc_short_event_builder import (  # noqa: E402
    SMC_CASES,
    allowed_bucket,
    allowed_direction,
    atr_series,
    build_event_scan_args,
    build_smc_events,
    completed_d1_idx_for_entry,
    completed_4h_idx_for_entry,
    daily_candles_from_4h,
    htf_structure_bias,
    scan_events,
)
from strategy.scalp_robust_v2_core import ActionType, Direction, precompute_swings  # noqa: E402
from strategy.sota_overlay_state import replay_single_position_events  # noqa: E402


DEFAULT_REPLAY_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "stable_smc_live_shadow_replay.json"
DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "live_overlay_audit.json"
DEFAULT_SMC_IDXS = (1489, 6006, 10234, 150710)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit live-safe Stable/SMC overlay wiring with key-window single-bar checks.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-root", default=str(ROOT / "data" / "okx" / "futures"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--replay-output", default=str(DEFAULT_REPLAY_OUTPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--smc-case", default="v2_medium_dispbody05_otherlag4_10x")
    parser.add_argument("--smc-idxs", default=",".join(str(idx) for idx in DEFAULT_SMC_IDXS))
    parser.add_argument("--stable-target-rr", type=float, default=2.875)
    parser.add_argument("--stable-max-hold-bars", type=int, default=40)
    parser.add_argument("--stable-stop-multiplier", type=float, default=1.0)
    parser.add_argument("--stable-max-short-stop-pct", type=float, default=1.75)
    parser.add_argument("--stage-trigger-rr-mode", default="close", choices=("close", "extreme"))
    parser.add_argument("--time-trailing-rr-mode", default="extreme", choices=("close", "extreme"))
    parser.add_argument("--atr-activation-rr-mode", default="close", choices=("close", "extreme"))
    parser.add_argument("--shadow-daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--shadow-equity-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--shadow-equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--shadow-consecutive-loss-stop", type=int, default=0)
    parser.add_argument("--year", type=int, default=None, help="Optional UTC year for replay-vs-live window audit, e.g. 2026.")
    parser.add_argument("--skip-formal-rebuild", action="store_true", help="Use --replay-output as the formal replay source instead of rebuilding it.")
    parser.add_argument("--run-tests", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    base_payload = load_config_payload(Path(args.config))
    payload, _ = apply_pressure_params(base_payload, Path(args.pressure_params))
    payload["mode"] = "paper"
    payload["data_root"] = args.data_root
    payload["telegram_enabled"] = False
    payload["telegram_command_enabled"] = False
    payload["telegram_ob_status_enabled"] = False
    payload["telegram_drift_report_enabled"] = False
    payload["enable_exchange_brackets"] = False
    payload["enable_live_overlay_strategy"] = True
    payload["live_overlay_smc_case"] = args.smc_case
    payload["live_overlay_smc_allocation"] = 1.0
    payload["live_overlay_stable_allocation"] = 1.0
    payload["live_overlay_stable_target_rr"] = float(args.stable_target_rr)
    payload["live_overlay_stable_max_hold_bars"] = int(args.stable_max_hold_bars)
    payload["live_overlay_stable_stop_multiplier"] = float(args.stable_stop_multiplier)
    payload["live_overlay_stable_max_short_stop_pct"] = float(args.stable_max_short_stop_pct)
    payload["stage_trigger_rr_mode"] = args.stage_trigger_rr_mode
    payload["time_trailing_rr_mode"] = args.time_trailing_rr_mode
    payload["atr_activation_rr_mode"] = args.atr_activation_rr_mode
    payload["enable_shadow_risk_gate"] = True
    payload["shadow_daily_loss_stop_pct"] = float(args.shadow_daily_loss_stop_pct)
    payload["shadow_equity_drawdown_stop_pct"] = float(args.shadow_equity_drawdown_stop_pct)
    payload["shadow_equity_drawdown_cooldown_days"] = int(args.shadow_equity_drawdown_cooldown_days)
    payload["shadow_consecutive_loss_stop"] = int(args.shadow_consecutive_loss_stop)
    return payload


def parse_idxs(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def data_paths(data_root: str) -> tuple[Path, Path]:
    root = Path(data_root)
    return (
        root / "BTC_USDT_USDT-15m-futures.feather",
        root / "BTC_USDT_USDT-4h-futures.feather",
    )


def action_summary(action: Any) -> dict[str, Any]:
    metadata = action.metadata or {}
    return {
        "type": action.type.value,
        "reason": action.reason,
        "direction": action.direction,
        "entry_price": action.entry_price,
        "exit_price": action.exit_price,
        "stop_price": action.stop_price,
        "target_price": action.target_price,
        "overlay_event_type": metadata.get("overlay_event_type"),
        "entry_idx": metadata.get("entry_idx"),
        "smc_live_safe": metadata.get("smc_live_safe"),
        "smc_time_bucket": metadata.get("smc_time_bucket"),
        "smc_mss_lag_bars": metadata.get("smc_mss_lag_bars"),
    }


def audit_single_bar(payload: dict[str, Any], idx: int) -> dict[str, Any]:
    config = dict(payload)
    config["state_db_path"] = str(Path(tempfile.mkdtemp(prefix="live-overlay-audit-")) / "state.db")
    executor = OkxExecutionEngine(ExecutorConfig.from_dict(config))
    engine, _ = executor.load_engine()
    if idx < 0 or idx >= len(engine.c15m):
        return {
            "idx": idx,
            "status": "out_of_range",
            "actions": [],
        }
    start = time.perf_counter()
    actions = executor._evaluate_latest_with_live_overlay(engine, idx, idx)
    elapsed = time.perf_counter() - start
    open_actions = [action for action in actions if action.type in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}]
    return {
        "idx": idx,
        "timestamp": engine._timestamp_for_idx(idx),
        "elapsed_seconds": round(elapsed, 6),
        "actions": [action_summary(action) for action in actions],
        "open_event_types": [(action.metadata or {}).get("overlay_event_type") for action in open_actions],
        "smc_open_count": sum(1 for action in open_actions if (action.metadata or {}).get("overlay_event_type") == "smc_short"),
    }


def load_replay_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    payload = json.loads(path.read_text())
    return {
        "status": "loaded",
        "path": str(path),
        "baseline": {
            "total_return_pct": payload.get("baseline_shadow_sota", {}).get("total_return_pct"),
            "max_drawdown_pct": payload.get("baseline_shadow_sota", {}).get("max_drawdown_pct"),
        },
        "reference": {
            "total_return_pct": payload.get("reference_base_priority_stable_smc", {}).get("total_return_pct"),
            "max_drawdown_pct": payload.get("reference_base_priority_stable_smc", {}).get("max_drawdown_pct"),
        },
        "live_shadow": {
            "total_return_pct": payload.get("live_shadow", {}).get("total_return_pct"),
            "max_drawdown_pct": payload.get("live_shadow", {}).get("max_drawdown_pct"),
            "event_type_counts": payload.get("live_shadow", {}).get("event_type_counts"),
            "decision_counts": payload.get("live_shadow", {}).get("decision_counts"),
            "windows": payload.get("live_shadow", {}).get("windows"),
        },
    }


def load_formal_replay_from_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Replay output not found: {path}")
    payload = json.loads(path.read_text())
    live_shadow = payload.get("live_shadow", {})
    events = live_shadow.get("events") or []
    has_full_events = bool(events)
    summary = {key: value for key, value in live_shadow.items() if key != "events"}
    if not has_full_events:
        summary["events_unavailable"] = True
    return {
        "initial_capital": float(payload.get("initial_capital", 1000.0) or 1000.0),
        "prepared_end": str(payload.get("prepared_end") or ""),
        "source": "json",
        "events_unavailable": not has_full_events,
        "shadow_summary": payload.get("baseline_shadow_sota", {}) or {},
        "candidate_counts": {
            "base_events": int((summary.get("event_type_counts") or {}).get("sota_long", 0) or 0),
            "stable_events": int((summary.get("event_type_counts") or {}).get("stable_reverse_short", 0) or 0),
            "smc_events": int((summary.get("event_type_counts") or {}).get("smc_short", 0) or 0),
        },
        "stable_summary": {},
        "smc_summary": {},
        "live_shadow_summary": summary,
        "live_shadow_events": events,
        "decisions_sample": payload.get("live_shadow", {}).get("decision_samples", []) or [],
    }


def year_filter_events(events: list[dict[str, Any]], year: int) -> list[dict[str, Any]]:
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    selected: list[dict[str, Any]] = []
    for event in events:
        entry_time = pd.Timestamp(str(event.get("entry_time")))
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")
        else:
            entry_time = entry_time.tz_convert("UTC")
        if start <= entry_time < end:
            selected.append(dict(event))
    return selected


def event_identity(event: dict[str, Any]) -> str:
    return (
        f"{event.get('event_type')}"
        f"|{int(event.get('entry_idx', 0) or 0)}"
        f"|{int(event.get('exit_idx', event.get('entry_idx', 0)) or 0)}"
    )


def event_entry_identity(event: dict[str, Any]) -> str:
    return f"{event.get('event_type')}|{int(event.get('entry_idx', 0) or 0)}"


def event_compact(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": event.get("event_type"),
        "entry_idx": int(event.get("entry_idx", 0) or 0),
        "exit_idx": int(event.get("exit_idx", event.get("entry_idx", 0)) or 0),
        "entry_time": str(event.get("entry_time")),
        "exit_time": str(event.get("exit_time")),
        "direction": event.get("direction"),
        "return_pct": round(float(event.get("return", 0.0) or 0.0) * 100.0, 4),
        "exit_reason": str(event.get("exit_reason") or ""),
    }


def diff_event_streams(
    formal_events: list[dict[str, Any]],
    live_events: list[dict[str, Any]],
    *,
    sample_size: int = 20,
) -> dict[str, Any]:
    formal_exact = {event_identity(event): event for event in formal_events}
    live_exact = {event_identity(event): event for event in live_events}
    formal_entry = {event_entry_identity(event): event for event in formal_events}
    live_entry = {event_entry_identity(event): event for event in live_events}
    missing_exact_keys = sorted(set(formal_exact) - set(live_exact))
    extra_exact_keys = sorted(set(live_exact) - set(formal_exact))
    missing_entry_keys = sorted(set(formal_entry) - set(live_entry))
    extra_entry_keys = sorted(set(live_entry) - set(formal_entry))
    return {
        "formal_event_count": len(formal_events),
        "live_event_count": len(live_events),
        "exact_match_count": len(set(formal_exact) & set(live_exact)),
        "entry_match_count": len(set(formal_entry) & set(live_entry)),
        "formal_only_exact_count": len(missing_exact_keys),
        "live_only_exact_count": len(extra_exact_keys),
        "formal_only_entry_count": len(missing_entry_keys),
        "live_only_entry_count": len(extra_entry_keys),
        "formal_only_exact_sample": [event_compact(formal_exact[key]) for key in missing_exact_keys[:sample_size]],
        "live_only_exact_sample": [event_compact(live_exact[key]) for key in extra_exact_keys[:sample_size]],
        "formal_only_entry_sample": [event_compact(formal_entry[key]) for key in missing_entry_keys[:sample_size]],
        "live_only_entry_sample": [event_compact(live_entry[key]) for key in extra_entry_keys[:sample_size]],
    }


def build_formal_replay(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    data_15m, data_4h = data_paths(args.data_root)
    prepared = load_prepared_data(
        data_15m_path=data_15m,
        data_4h_path=data_4h,
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
        daily_loss_stop_pct=float(args.shadow_daily_loss_stop_pct),
        equity_drawdown_stop_pct=float(args.shadow_equity_drawdown_stop_pct),
        consecutive_loss_stop=int(args.shadow_consecutive_loss_stop),
        equity_drawdown_cooldown_days=int(args.shadow_equity_drawdown_cooldown_days),
    )
    shadow_events = shadow["events"]
    base_events = [standard_sota_event(event) for event in shadow_events]
    stable_events, stable_summary = build_stable_events(
        payload,
        prepared,
        shadow_events,
        allocation=float(payload.get("live_overlay_stable_allocation", 1.0) or 1.0),
        target_rr=float(payload.get("live_overlay_stable_target_rr", 2.875) or 2.875),
        max_hold_bars=int(payload.get("live_overlay_stable_max_hold_bars", 40) or 40),
        leverage=float(payload.get("live_overlay_stable_leverage", 5.0) or 5.0),
        stop_multiplier=float(payload.get("live_overlay_stable_stop_multiplier", 1.0) or 1.0),
        max_short_stop_pct=float(payload.get("live_overlay_stable_max_short_stop_pct", 1.75) or 1.75),
    )
    daily = daily_candles_from_4h(prepared.c4h)
    h4_highs, h4_lows = precompute_swings(prepared.c4h, n=2, lookback=80)
    d1_highs, d1_lows = precompute_swings(daily, n=2, lookback=20)
    smc_case = str(args.smc_case or "")
    if smc_case not in SMC_CASES:
        raise ValueError(f"Unknown SMC case: {smc_case}")
    smc_args = SimpleNamespace(
        data_15m=str(data_15m),
        data_4h=str(data_4h),
        start_date=str(args.start_date),
    )
    smc_events, smc_summary = build_smc_events(
        smc_case,
        SMC_CASES[smc_case],
        smc_args,
        prepared,
        daily,
        h4_highs,
        h4_lows,
        d1_highs,
        d1_lows,
        float(payload.get("live_overlay_smc_allocation", 1.0) or 1.0),
        taker_fee_rate=float(payload.get("taker_fee_rate", 0.0005) or 0.0),
        slippage_bps=float(payload.get("slippage_bps", 0.0) or 0.0),
    )
    accepted, decisions = replay_single_position_events([to_candidate(event) for event in (base_events + stable_events + smc_events)])
    live_events = [candidate.metadata["event"] for candidate in accepted]
    live_summary = standard_event_summary(live_events, initial_capital, "entry_idx")
    live_summary = add_standard_windows(live_summary, initial_capital, prepared.end, "entry_idx")
    return {
        "initial_capital": initial_capital,
        "prepared_end": str(prepared.end),
        "shadow_summary": {
            "total_return_pct": shadow.get("total_return_pct"),
            "max_drawdown_pct": shadow.get("max_drawdown_pct"),
            "accepted_trades": shadow.get("accepted_trades"),
            "skipped_trades": shadow.get("skipped_trades"),
        },
        "candidate_counts": {
            "base_events": len(base_events),
            "stable_events": len(stable_events),
            "smc_events": len(smc_events),
        },
        "stable_summary": stable_summary,
        "smc_summary": smc_summary,
        "live_shadow_summary": {key: value for key, value in live_summary.items() if key != "events"},
        "live_shadow_events": live_summary["events"],
        "decisions_sample": decisions[:40],
    }


def build_live_safe_smc_lookup(executor: OkxExecutionEngine, engine: Any) -> dict[int, dict[str, Any]]:
    case_args = executor._overlay_smc_case_args()
    scan_args = build_event_scan_args(case_args)
    scan_args.allow_incomplete_tail = True
    all_events = scan_events(list(engine.c15m), scan_args)
    atr_values = atr_series(engine.c15m, int(getattr(case_args, "atr_period", 14)))
    daily = daily_candles_from_4h(engine.c4h)
    h4_highs, h4_lows = precompute_swings(engine.c4h, n=2, lookback=80)
    d1_highs, d1_lows = precompute_swings(daily, n=2, lookback=20)
    daily_ts = [candle.ts for candle in daily]
    lookup: dict[int, dict[str, Any]] = {}
    for event in all_events:
        if event.direction != "BEAR" or event.retest is None:
            continue
        idx = int(event.retest.idx)
        entry_candle = engine.c15m[idx]
        if bool(getattr(case_args, "require_confirmed_retest", False)) and not bool(event.retest.confirmed):
            continue
        if bool(getattr(case_args, "require_fvg_touch", False)) and not bool(event.retest.fvg_touched):
            continue
        if not bool(getattr(case_args, "allow_ote_only", True)) and not bool(event.retest.fvg_touched):
            continue
        if bool(getattr(case_args, "require_ote_touch", False)) and not bool(event.retest.ote_touched):
            continue
        bucket, ny_time = executor._overlay_live_smc_candidate.__globals__["time_bucket"](entry_candle.ts)
        if not allowed_bucket(bucket, str(getattr(case_args, "allowed_time_buckets", "all"))):
            continue
        if not allowed_direction("BEAR", str(getattr(case_args, "allowed_directions", "all"))):
            continue
        if bool(getattr(case_args, "drop_asia_session", False)) and bucket == "asia_evening_ny":
            continue
        mss_lag_bars = int(event.mss_idx) - int(event.sweep_idx) if event.mss_idx is not None else None
        if int(getattr(case_args, "max_mss_lag_bars", 0)) > 0 and mss_lag_bars is not None and mss_lag_bars > int(case_args.max_mss_lag_bars):
            continue
        if int(getattr(case_args, "global_min_mss_lag_bars", 0)) > 0 and mss_lag_bars is not None and mss_lag_bars < int(case_args.global_min_mss_lag_bars):
            continue
        if int(getattr(case_args, "global_max_mss_lag_bars", 0)) > 0 and mss_lag_bars is not None and mss_lag_bars > int(case_args.global_max_mss_lag_bars):
            continue
        if bucket == "ny_am_killzone" and int(getattr(case_args, "ny_max_mss_lag_bars", 0)) > 0 and mss_lag_bars is not None and mss_lag_bars > int(case_args.ny_max_mss_lag_bars):
            continue
        if bucket == "other" and int(getattr(case_args, "other_min_mss_lag_bars", 0)) > 0 and mss_lag_bars is not None and mss_lag_bars < int(case_args.other_min_mss_lag_bars):
            continue
        if float(event.displacement_body_atr or 0.0) < float(getattr(case_args, "min_displacement_body_atr", 0.0) or 0.0):
            continue
        if float(event.displacement_range_atr or 0.0) < float(getattr(case_args, "min_displacement_range_atr", 0.0) or 0.0):
            continue
        if float(getattr(case_args, "bear_min_sweep_distance_pct", 0.0) or 0.0) > 0.0 and float(event.sweep_distance_pct or 0.0) < float(getattr(case_args, "bear_min_sweep_distance_pct", 0.0) or 0.0):
            continue
        h4_idx = completed_4h_idx_for_entry(engine.mapping, idx)
        h4_bias = htf_structure_bias(engine.c4h, h4_highs, h4_lows, h4_idx) if h4_idx >= 0 else "NONE"
        d1_idx = completed_d1_idx_for_entry(daily_ts, entry_candle.ts)
        d1_bias = htf_structure_bias(daily, d1_highs, d1_lows, d1_idx) if d1_idx >= 0 else "NONE"
        if bool(getattr(case_args, "require_h4_bias_align", False)) and bool(getattr(case_args, "require_htf_bias_align", False)) and h4_bias != "BEAR":
            continue
        if bool(getattr(case_args, "require_h4_bias_align", False)) and not bool(getattr(case_args, "require_htf_bias_align", False)) and h4_bias not in {"BEAR", "NONE"}:
            continue
        if bool(getattr(case_args, "require_d1_bias_align", False)) and bool(getattr(case_args, "require_htf_bias_align", False)) and d1_bias != "BEAR":
            continue
        if bool(getattr(case_args, "require_d1_bias_align", False)) and not bool(getattr(case_args, "require_htf_bias_align", False)) and d1_bias not in {"BEAR", "NONE"}:
            continue
        stop_buffer = atr_values[idx] * float(getattr(case_args, "stop_buffer_atr", 0.05)) if idx < len(atr_values) else 0.0
        stop_price = float(event.sweep_extreme) + stop_buffer
        entry_price = float(event.retest.close)
        risk_points = stop_price - entry_price
        if risk_points <= 0:
            continue
        lookup.setdefault(idx, {
            "entry_idx": idx,
            "entry_time": event.retest.timestamp,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": entry_price - risk_points * float(getattr(case_args, "target_rr", 2.0) or 2.0),
            "time_bucket": bucket,
            "ny_time": ny_time,
            "mss_lag_bars": mss_lag_bars,
            "h4_bias": h4_bias,
            "d1_bias": d1_bias,
            "fvg_touched": bool(event.retest.fvg_touched),
            "ote_touched": bool(event.retest.ote_touched),
        })
    return lookup


def patch_live_safe_smc_lookup(executor: OkxExecutionEngine, lookup: dict[int, dict[str, Any]]) -> None:
    case_args = executor._overlay_smc_case_args()

    def patched(self: OkxExecutionEngine, eng: Any, idx: int) -> Any | None:
        candidate = lookup.get(int(idx))
        if candidate is None:
            return None
        entry_price = float(candidate["entry_price"])
        stop_price = float(candidate["stop_price"])
        target_price = float(candidate["target_price"])
        risk_points = stop_price - entry_price
        if risk_points <= 0:
            return None
        diagnostics = self._overlay_live_smc_candidate.__globals__["high_leverage_trade_diagnostics"](
            pd.Series(
                {
                    "entry_time": candidate["entry_time"],
                    "direction": "BEAR",
                    "entry_price": entry_price,
                    "initial_stop_price": stop_price,
                    "notional": self._overlay_capital(eng) * float(getattr(case_args, "leverage", 10.0)) * float(getattr(case_args, "position_size_pct", 1.0)),
                }
            ),
            capital=self._overlay_capital(eng),
            leverage=float(getattr(case_args, "leverage", 10.0)),
            maintenance_margin_pct=float(getattr(case_args, "maintenance_margin_pct", 0.5)),
        )
        if float(diagnostics["liquidation_buffer_pct"]) < float(getattr(case_args, "min_liq_buffer_pct", 1.2)):
            return None
        target_rr = float(getattr(case_args, "target_rr", 2.0) or 2.0)
        action, _ = self._overlay_build_open_short_action(
            engine=eng,
            idx=idx,
            event_type="smc_short",
            signal_entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            target_rr=target_rr,
            max_hold_bars=int(getattr(case_args, "outcome_lookahead_bars", 96)),
            allocation=float(self.config.live_overlay_smc_allocation),
            leverage=float(getattr(case_args, "leverage", 10.0)),
            stop_reason="stop_loss",
            target_reason=f"target_{target_rr:.1f}r",
            metadata={
                "smc_case": self.config.live_overlay_smc_case,
                "smc_time_bucket": candidate.get("time_bucket"),
                "smc_ny_time": candidate.get("ny_time"),
                "smc_mss_lag_bars": candidate.get("mss_lag_bars"),
                "smc_h4_bias": candidate.get("h4_bias"),
                "smc_d1_bias": candidate.get("d1_bias"),
                "smc_fvg_touched": bool(candidate.get("fvg_touched", False)),
                "smc_ote_touched": bool(candidate.get("ote_touched", False)),
                "smc_stop_buffer_atr": float(getattr(case_args, "stop_buffer_atr", 0.05)),
                "smc_live_safe": True,
            },
        )
        return action

    executor._overlay_maybe_build_smc_candidate = MethodType(patched, executor)  # type: ignore[method-assign]


def trade_to_event(trade: Any) -> dict[str, Any]:
    regime_label = str(getattr(trade, "regime_label", "") or "")
    if regime_label in {"smc_short", "stable_reverse_short"}:
        event_type = regime_label
    elif getattr(trade, "direction", None) == Direction.BEAR:
        event_type = "overlay_short_unknown"
    else:
        event_type = "sota_long"
    return {
        "event_type": event_type,
        "entry_idx": int(getattr(trade, "entry_idx", 0) or 0),
        "exit_idx": int(getattr(trade, "exit_idx", getattr(trade, "entry_idx", 0)) or 0),
        "entry_time": str(getattr(trade, "entry_time")),
        "exit_time": str(getattr(trade, "exit_time")),
        "direction": getattr(trade, "direction", None),
        "return": float(getattr(trade, "pnl_pct", 0.0) or 0.0),
        "exit_reason": str(getattr(trade, "exit_reason", "") or ""),
    }


def execution_summary(executor: OkxExecutionEngine, action: Any, result: dict[str, Any]) -> dict[str, Any]:
    metadata = action.metadata or {}
    return {
        "timestamp": action.timestamp,
        "type": action.type.value,
        "reason": action.reason,
        "direction": action.direction,
        "status": result.get("status"),
        "overlay_event_type": metadata.get("overlay_event_type"),
        "entry_idx": metadata.get("entry_idx"),
        "index": metadata.get("index"),
        "pause_until": result.get("pause_until"),
        "decision_reason": result.get("reason"),
        "shadow_real_position_open": bool(executor._load_shadow_gate_state().get("real_position_open")),
        "shadow_real_position_direction": executor._load_shadow_gate_state().get("real_position_direction"),
    }


def year_live_audit(payload: dict[str, Any], year: int, args: argparse.Namespace) -> dict[str, Any]:
    config = dict(payload)
    config["state_db_path"] = str(Path(tempfile.mkdtemp(prefix=f"live-overlay-audit-{year}-")) / "state.db")
    executor = OkxExecutionEngine(ExecutorConfig.from_dict(config))
    engine, _ = executor.load_engine()
    lookup = build_live_safe_smc_lookup(executor, engine)
    patch_live_safe_smc_lookup(executor, lookup)
    ts_values = [float(candle.ts) for candle in engine.c15m]
    start = pd.Timestamp(f"{year}-01-01", tz="UTC").timestamp()
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC").timestamp()
    start_idx = bisect.bisect_left(ts_values, start)
    end_idx = bisect.bisect_left(ts_values, end) - 1
    loop_start_idx = start_idx
    started = time.perf_counter()
    action_count = 0
    open_action_count = 0
    execution_samples: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    event_type_counts: dict[str, int] = {}
    for idx in range(loop_start_idx, end_idx + 1):
        actions = executor._evaluate_latest_with_live_overlay(engine, idx, idx)
        for action in actions:
            result = executor.execute_action(action, engine)
            action_count += 1
            if action.type in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}:
                open_action_count += 1
            status = str(result.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            event_type = str((action.metadata or {}).get("overlay_event_type") or ("sota_long" if action.type == ActionType.OPEN_LONG else "other"))
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            if len(execution_samples) < 40:
                execution_samples.append(execution_summary(executor, action, result))
    elapsed = time.perf_counter() - started
    trades = [
        trade_to_event(trade)
        for trade in engine.trades
        if pd.Timestamp(str(getattr(trade, "entry_time")), tz="UTC") >= pd.Timestamp(f"{year}-01-01", tz="UTC")
    ]
    summary = standard_event_summary(trades, 1000.0, "entry_idx")
    return {
        "year": year,
        "range": {
            "loop_start_idx": loop_start_idx,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "loop_start_time": engine._timestamp_for_idx(loop_start_idx) if 0 <= loop_start_idx < len(engine.c15m) else None,
            "start_time": engine._timestamp_for_idx(start_idx) if start_idx < len(engine.c15m) else None,
            "end_time": engine._timestamp_for_idx(end_idx) if 0 <= end_idx < len(engine.c15m) else None,
            "live_warmup_from_start": False,
        },
        "elapsed_seconds": round(elapsed, 3),
        "smc_lookup_candidates": len(lookup),
        "live_actions": action_count,
        "live_open_actions": open_action_count,
        "execution_status_counts": status_counts,
        "action_event_type_counts": event_type_counts,
        "live_trade_summary": {key: value for key, value in summary.items() if key != "events"},
        "live_trade_events": summary["events"],
        "sample_executions": execution_samples,
        "shadow_state": executor._load_shadow_gate_state(engine),
    }


def run_command(command: list[str]) -> dict[str, Any]:
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def main() -> None:
    args = parse_args()
    payload = load_config(args)
    idxs = parse_idxs(args.smc_idxs)
    key_bar_results = [audit_single_bar(payload, idx) for idx in idxs]
    formal_replay = (
        load_formal_replay_from_json(Path(args.replay_output))
        if bool(args.skip_formal_rebuild)
        else build_formal_replay(payload, args)
    )
    report = {
        "metadata": {
            "config": str(Path(args.config).resolve()),
            "pressure_params": str(Path(args.pressure_params).resolve()),
            "data_root": args.data_root,
            "start_date": args.start_date,
            "smc_case": args.smc_case,
            "stable_params": {
                "target_rr": args.stable_target_rr,
                "max_hold_bars": args.stable_max_hold_bars,
                "stop_multiplier": args.stable_stop_multiplier,
                "max_short_stop_pct": args.stable_max_short_stop_pct,
            },
            "trailing_rr_modes": {
                "stage_trigger_rr_mode": args.stage_trigger_rr_mode,
                "time_trailing_rr_mode": args.time_trailing_rr_mode,
                "atr_activation_rr_mode": args.atr_activation_rr_mode,
            },
            "shadow_params": {
                "daily_loss_stop_pct": args.shadow_daily_loss_stop_pct,
                "equity_drawdown_stop_pct": args.shadow_equity_drawdown_stop_pct,
                "equity_drawdown_cooldown_days": args.shadow_equity_drawdown_cooldown_days,
                "consecutive_loss_stop": args.shadow_consecutive_loss_stop,
            },
            "live_warmup_from_start": False,
        },
        "key_bar_audit": key_bar_results,
        "replay_summary": load_replay_summary(Path(args.replay_output)),
        "formal_replay": {
            key: value
            for key, value in formal_replay.items()
            if key != "live_shadow_events"
        },
    }
    if args.year is not None:
        report["year_audit"] = year_live_audit(payload, args.year, args)
        formal_events_available = bool(formal_replay.get("live_shadow_events"))
        formal_year_events = year_filter_events(formal_replay["live_shadow_events"], args.year) if formal_events_available else []
        live_year_events = year_filter_events(report["year_audit"]["live_trade_events"], args.year)
        if formal_events_available:
            formal_year_summary = standard_event_summary(formal_year_events, 1000.0, "entry_idx")
        else:
            formal_year_summary = dict(
                ((formal_replay.get("live_shadow_summary") or {}).get("windows") or {}).get("current_year", {})
            )
            formal_year_summary["events_unavailable"] = True
        live_year_summary = standard_event_summary(live_year_events, 1000.0, "entry_idx")
        report["year_formal_replay"] = {key: value for key, value in formal_year_summary.items() if key != "events"}
        report["year_event_diff"] = (
            diff_event_streams(formal_year_events, live_year_events)
            if formal_events_available
            else {
                "status": "summary_only",
                "reason": "formal_replay_json_has_no_full_events",
                "live_event_count": len(live_year_events),
            }
        )
        replay_summary = report["replay_summary"]
        if replay_summary.get("status") == "loaded":
            replay_year = (replay_summary.get("live_shadow", {}).get("windows") or {}).get("current_year", {})
            report["year_replay_vs_live"] = {
                "year": args.year,
                "replay_current_year": replay_year,
                "live_main_loop": report["year_audit"]["live_trade_summary"],
                "return_gap_pct": round(
                    float(report["year_audit"]["live_trade_summary"].get("total_return_pct", 0.0) or 0.0)
                    - float(replay_year.get("total_return_pct", 0.0) or 0.0),
                    4,
                ),
                "trade_gap": int(report["year_audit"]["live_trade_summary"].get("trades", 0) or 0) - int(replay_year.get("trades", 0) or 0),
            }
        report["year_formal_vs_live"] = {
            "year": args.year,
            "formal_current_year": report["year_formal_replay"],
            "live_main_loop": report["year_audit"]["live_trade_summary"],
            "return_gap_pct": round(
                float(report["year_audit"]["live_trade_summary"].get("total_return_pct", 0.0) or 0.0)
                - float(report["year_formal_replay"].get("total_return_pct", 0.0) or 0.0),
                4,
            ),
            "trade_gap": int(report["year_audit"]["live_trade_summary"].get("trades", 0) or 0) - int(report["year_formal_replay"].get("trades", 0) or 0),
        }
        report["year_formal_event_samples"] = {
            "formal_events": formal_year_summary.get("events", [])[:40],
            "formal_events_unavailable": not formal_events_available,
            "live_events": live_year_summary["events"][:40],
        }
        report["year_audit"].pop("live_trade_events", None)
    if args.run_tests:
        report["test_commands"] = [
            run_command([sys.executable, "-m", "unittest", "tests.test_live_overlay_runtime"]),
            run_command([sys.executable, "-m", "py_compile", "bot/okx_executor.py", "tests/test_live_overlay_runtime.py"]),
        ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    for item in key_bar_results:
        print(
            f"idx={item['idx']} ts={item.get('timestamp')} elapsed={item.get('elapsed_seconds')}s "
            f"smc_open_count={item.get('smc_open_count')} events={item.get('open_event_types')}"
        )
    replay = report["replay_summary"]
    if replay.get("status") == "loaded":
        live = replay["live_shadow"]
        print(f"replay_live_shadow={live.get('total_return_pct')}%/{live.get('max_drawdown_pct')}%")
    if args.year is not None:
        live_year = report["year_audit"]["live_trade_summary"]
        formal_year = report.get("year_formal_replay", {})
        print(
            f"{args.year}_formal={formal_year.get('total_return_pct')}% "
            f"{args.year}_live={live_year.get('total_return_pct')}% "
            f"gap={report.get('year_formal_vs_live', {}).get('return_gap_pct')}%"
        )


if __name__ == "__main__":
    main()
