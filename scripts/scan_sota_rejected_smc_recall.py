#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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
    passes_score_gate,
    resample_confirmed_1h,
    score_snapshot,
)
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.live_shadow_utils import clean_for_json, standard_sota_event  # noqa: E402
from scripts.replay_sota_smc_live_shadow import (  # noqa: E402
    _apply_config_defaults,
    apply_auto_tit_override,
    apply_trailing_rr_modes,
    parse_score_bucket_rules,
    replay_live_shadow,
    resolve_sota_soft_stop_args,
)
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, replay_shadow_events  # noqa: E402
from scripts.score_bucket_sizing_utils import apply_score_bucket_sizing_to_events  # noqa: E402
from scripts.sota_liquidity_context import annotate_sota_events_with_liquidity_context  # noqa: E402


DEFAULT_FROZEN = ROOT / "var" / "high_leverage_expansion" / "frozen_live_core_20260515.json"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "sota_rejected_smc_recall_scan_20260517.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan whether SMC/liquidity context can recall SOTA long candidates rejected by score/structure gates."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.high-leverage-structure.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--frozen-report", default=str(DEFAULT_FROZEN))
    parser.add_argument("--target-leverages", default="2,3,5,8")
    parser.add_argument("--min-subgroup-trades", type=int, default=2)
    parser.add_argument("--sample-events", type=int, default=8)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def event_key(event: dict[str, Any]) -> str:
    return f"{event.get('event_type')}|{int(event.get('entry_idx', 0) or 0)}|{int(event.get('exit_idx', 0) or 0)}"


def compact_summary(result: dict[str, Any]) -> dict[str, Any]:
    current_year = (result.get("windows") or {}).get("current_year", {})
    return {
        "total_return_pct": round(float(result.get("total_return_pct", 0.0) or 0.0), 2),
        "max_drawdown_pct": round(float(result.get("max_drawdown_pct", 0.0) or 0.0), 2),
        "trades": int(result.get("trades", 0) or 0),
        "wins": int(result.get("wins", 0) or 0),
        "losses": int(result.get("losses", 0) or 0),
        "win_rate_pct": round(float(result.get("win_rate_pct", 0.0) or 0.0), 2),
        "avg_return_pct": round(float(result.get("avg_return_pct", 0.0) or 0.0), 4),
        "profit_factor": round(float(result.get("profit_factor", 0.0) or 0.0), 4),
        "event_type_counts": result.get("event_type_counts", {}),
        "exit_counts": result.get("exit_counts", {}),
        "current_year": {
            "total_return_pct": round(float(current_year.get("total_return_pct", 0.0) or 0.0), 2),
            "max_drawdown_pct": round(float(current_year.get("max_drawdown_pct", 0.0) or 0.0), 2),
            "trades": int(current_year.get("trades", 0) or 0),
            "win_rate_pct": round(float(current_year.get("win_rate_pct", 0.0) or 0.0), 2),
            "profit_factor": round(float(current_year.get("profit_factor", 0.0) or 0.0), 4),
        },
    }


def with_target_leverage(event: dict[str, Any], target_leverage: float) -> dict[str, Any]:
    updated = dict(event)
    source_leverage = float(updated.get("source_effective_leverage", 0.0) or 0.0)
    if source_leverage > 0:
        scale = float(target_leverage) / source_leverage
        updated["return"] = float(updated.get("return", 0.0) or 0.0) * scale
        updated["return_pct"] = round(float(updated["return"]) * 100.0, 4)
    updated["pre_recall_source_effective_leverage"] = round(source_leverage, 6)
    updated["source_effective_leverage"] = round(float(target_leverage), 6)
    return updated


def recall_conditions(event: dict[str, Any]) -> set[str]:
    status = str(event.get("feature_recent_sweep_status") or "")
    out: set[str] = set()
    if bool(event.get("feature_recent_fvg_near_entry")):
        out.add("recent_fvg_near_entry")
    if status == "mss_with_fvg":
        out.add("mss_with_fvg")
    if bool(event.get("feature_recent_fvg_near_entry")) and status == "mss_with_fvg":
        out.add("fvg_near_and_mss_with_fvg")
    if bool(event.get("feature_recent_sweep_has_fvg")):
        out.add("sweep_has_fvg")
    if bool(event.get("feature_recent_fvg_near_entry")) or status == "mss_with_fvg":
        out.add("fvg_near_or_mss_with_fvg")
    return out


def normalized_value(event: dict[str, Any], field: str) -> Any:
    if field == "entry_year":
        return pd.Timestamp(event.get("entry_time")).year
    value = event.get(field)
    if field in {"net_score", "bull_total", "bear_total"}:
        return int(value or 0)
    if field in {"conflict", "feature_bearish_structure", "feature_recent_fvg_near_entry"}:
        return bool(value)
    return value


def subgroup_values(event: dict[str, Any], dimensions: tuple[str, ...]) -> dict[str, Any]:
    return {field: normalized_value(event, field) for field in dimensions}


def subgroup_key(values: dict[str, Any]) -> str:
    return "|".join(f"{field}={values[field]}" for field in sorted(values))


def matches_values(event: dict[str, Any], values: dict[str, Any]) -> bool:
    for field, expected in values.items():
        if normalized_value(event, field) != expected:
            return False
    return True


def group_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [float(event.get("return_pct", 0.0) or 0.0) for event in events]
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value < 0]
    return {
        "trades": len(events),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(events) * 100.0, 2) if events else 0.0,
        "sum_return_pct": round(sum(returns), 4),
        "avg_return_pct": round(sum(returns) / len(events), 4) if events else 0.0,
        "best_return_pct": round(max(returns), 4) if returns else 0.0,
        "worst_return_pct": round(min(returns), 4) if returns else 0.0,
        "profit_factor": round(sum(wins) / abs(sum(losses)), 4) if losses else (999.0 if wins else 0.0),
        "exit_counts": dict(sorted(Counter(str(event.get("exit_reason") or "") for event in events).items())),
    }


def yearly_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        groups[str(pd.Timestamp(event.get("entry_time")).year)].append(event)
    return {year: group_stats(items) for year, items in sorted(groups.items())}


def rejected_stage(event: dict[str, Any], *, score_passed: bool, structure_passed: bool) -> str:
    if not score_passed:
        return "score_gate"
    if not structure_passed:
        return "structure_gate"
    return "accepted"


def main() -> None:
    args = parse_args()
    base_payload = load_config_payload(Path(args.config))
    replay_args = argparse.Namespace(
        config=str(args.config),
        pressure_params=str(args.pressure_params),
        data_15m=str(args.data_15m),
        data_4h=str(args.data_4h),
        start_date=str(args.start_date),
        informative_asof_from_15m=None,
        confirmed_4h_only=None,
        replay_sync_entry_to_signal_price=None,
        smc_case=None,
        smc_allocation=None,
        smc_min_entry_idx=None,
        enable_gap_smc_short=None,
        gap_smc_case=None,
        gap_smc_min_flat_days=None,
        gap_smc_leverage=None,
        gap_smc_max_stop_distance_pct=None,
        gap_smc_min_entry_idx=None,
        enable_sota_score_gate=None,
        sota_score_net_min=None,
        sota_score_bull_min=None,
        sota_score_bear_max=None,
        sota_score_conflict_mode=None,
        require_non_bearish_structure_for_long=None,
        enable_long_score_bucket_sizing=None,
        long_score_bucket_sizing_rules_json="",
        enable_sota_soft_stop_recovery_overlay=False,
        sota_soft_stop_net_min=None,
        sota_soft_stop_bear_max=None,
        sota_soft_stop_max_leverage=None,
        sota_soft_stop_buffer_r=None,
        sota_soft_stop_target_rr=None,
        sota_soft_stop_max_extension_bars=None,
        sota_soft_stop_exclude_score_buckets=None,
        stage_trigger_rr_mode=None,
        time_trailing_rr_mode=None,
        atr_activation_rr_mode=None,
        enable_auto_time_based_trailing=None,
        daily_loss_stop_pct=6.0,
        equity_drawdown_stop_pct=12.0,
        equity_drawdown_cooldown_days=2,
        consecutive_loss_stop=4,
        sample_trades=0,
        output=str(args.output),
        paper_log_output="",
    )
    replay_args = _apply_config_defaults(replay_args, base_payload)
    _, replay_args = resolve_sota_soft_stop_args(replay_args, base_payload)
    payload, pressure_params = apply_pressure_params(base_payload, Path(args.pressure_params))
    payload, trailing_rr_modes = apply_trailing_rr_modes(
        payload,
        stage_trigger_rr_mode=str(replay_args.stage_trigger_rr_mode),
        time_trailing_rr_mode=str(replay_args.time_trailing_rr_mode),
        atr_activation_rr_mode=str(replay_args.atr_activation_rr_mode),
    )
    payload = apply_auto_tit_override(payload, replay_args)
    payload["replay_sync_entry_to_signal_price"] = bool(replay_args.replay_sync_entry_to_signal_price)

    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=payload.get("regime_switcher_thresholds"),
        informative_asof_from_15m=bool(replay_args.informative_asof_from_15m),
        confirmed_4h_only=bool(replay_args.confirmed_4h_only),
    )
    metrics, engine = run_engine(payload, prepared, args.start_date)
    trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
    initial_capital = float(metrics.get("initial_capital", 1000.0))
    fixed = expansion_overlay(trades, initial_capital, FIXED_STRUCTURE_PARAMS, include_events=True)
    shadow = replay_shadow_events(
        fixed["events"],
        initial_capital,
        daily_loss_stop_pct=float(replay_args.daily_loss_stop_pct),
        equity_drawdown_stop_pct=float(replay_args.equity_drawdown_stop_pct),
        consecutive_loss_stop=int(replay_args.consecutive_loss_stop),
        equity_drawdown_cooldown_days=int(replay_args.equity_drawdown_cooldown_days),
    )
    raw_events = annotate_sota_events_with_liquidity_context(
        prepared.c15m,
        [standard_sota_event(event) for event in shadow["events"]],
    )

    c1h = resample_confirmed_1h(prepared.c15m)
    mapping_1h = align_confirmed_mapping(c1h, prepared.c15m)
    scored_events: list[dict[str, Any]] = []
    rejected_events: list[dict[str, Any]] = []
    accepted_events: list[dict[str, Any]] = []
    for event in raw_events:
        enriched = dict(event)
        enriched.update(asdict(score_snapshot(prepared, c1h, mapping_1h, int(event.get("entry_idx", 0) or 0))))
        score_passed = passes_score_gate(
            enriched,
            net_min=int(replay_args.sota_score_net_min),
            bull_min=int(replay_args.sota_score_bull_min),
            bear_max=int(replay_args.sota_score_bear_max),
            conflict_mode=str(replay_args.sota_score_conflict_mode),
        )
        structure_passed = not bool(enriched.get("feature_bearish_structure", False))
        stage = rejected_stage(enriched, score_passed=score_passed, structure_passed=structure_passed)
        enriched["sota_reject_stage"] = stage
        enriched["sota_score_gate_passed"] = bool(score_passed)
        enriched["sota_structure_gate_passed"] = bool(structure_passed)
        scored_events.append(enriched)
        if stage == "accepted":
            accepted_events.append(enriched)
        else:
            rejected_events.append(enriched)

    frozen = json.loads(Path(args.frozen_report).read_text())
    frozen_events = list(frozen["live_shadow"]["events"])
    frozen_summary = compact_summary(frozen["live_shadow"])
    frozen_keys = {event_key(event) for event in frozen_events}
    rejected_events = [event for event in rejected_events if event_key(event) not in frozen_keys]

    target_leverages = parse_float_list(args.target_leverages)
    condition_names = [
        "recent_fvg_near_entry",
        "mss_with_fvg",
        "fvg_near_and_mss_with_fvg",
        "sweep_has_fvg",
        "fvg_near_or_mss_with_fvg",
    ]
    candidates: list[dict[str, Any]] = []

    def evaluate_candidate(
        *,
        condition: str,
        dimensions: tuple[str, ...],
        values: dict[str, Any],
        matched: list[dict[str, Any]],
        target_leverage: float,
    ) -> dict[str, Any]:
        recalled = []
        for event in matched:
            updated = with_target_leverage(event, target_leverage)
            updated["event_type"] = f"sota_recall_{condition}"
            updated["recall_condition"] = condition
            updated["recall_dimensions"] = list(dimensions)
            updated["recall_values"] = values
            updated["recall_target_effective_leverage"] = target_leverage
            recalled.append(updated)
        combined, decisions = replay_live_shadow(
            frozen_events + recalled,
            initial_capital,
            prepared.end,
            frozen["live_shadow"],
        )
        accepted_recall = [
            event for event in combined.get("events", []) if str(event.get("event_type") or "").startswith("sota_recall_")
        ]
        return {
            "condition": condition,
            "dimensions": list(dimensions),
            "values": values,
            "target_effective_leverage": target_leverage,
            "matched_rejected_trades": len(matched),
            "accepted_recall_trades": len(accepted_recall),
            "candidate": compact_summary(combined),
            "delta_vs_frozen": {
                "total_return_pct": round(float(combined["total_return_pct"]) - float(frozen["live_shadow"]["total_return_pct"]), 2),
                "max_drawdown_pct": round(float(combined["max_drawdown_pct"]) - float(frozen["live_shadow"]["max_drawdown_pct"]), 2),
                "current_year_return_pct": round(
                    float(combined["windows"]["current_year"]["total_return_pct"])
                    - float(frozen["live_shadow"]["windows"]["current_year"]["total_return_pct"]),
                    2,
                ),
                "current_year_max_drawdown_pct": round(
                    float(combined["windows"]["current_year"]["max_drawdown_pct"])
                    - float(frozen["live_shadow"]["windows"]["current_year"]["max_drawdown_pct"]),
                    2,
                ),
            },
            "recalled_stats": group_stats(accepted_recall),
            "recalled_yearly_stats": yearly_stats(accepted_recall),
            "decision_counts": combined.get("decision_counts", {}),
            "sample_recalled_events": [
                {
                    "event_key": event_key(event),
                    "entry_time": event.get("entry_time"),
                    "exit_time": event.get("exit_time"),
                    "return_pct": event.get("return_pct"),
                    "reject_stage": event.get("sota_reject_stage"),
                    "net_score": event.get("net_score"),
                    "bull_total": event.get("bull_total"),
                    "bear_total": event.get("bear_total"),
                    "regime_label": event.get("regime_label"),
                    "conflict": bool(event.get("conflict")),
                    "recent_fvg_near_entry": bool(event.get("feature_recent_fvg_near_entry")),
                    "recent_sweep_status": event.get("feature_recent_sweep_status"),
                    "feature_bearish_structure": bool(event.get("feature_bearish_structure")),
                }
                for event in accepted_recall[: int(args.sample_events)]
            ],
        }

    for condition in condition_names:
        matched = [event for event in rejected_events if condition in recall_conditions(event)]
        for target_leverage in target_leverages:
            candidates.append(
                evaluate_candidate(
                    condition=condition,
                    dimensions=(),
                    values={},
                    matched=matched,
                    target_leverage=target_leverage,
                )
            )

    dimension_sets = [
        ("sota_reject_stage",),
        ("net_score", "bull_total", "bear_total"),
        ("net_score", "bull_total", "bear_total", "sota_reject_stage"),
        ("regime_label", "sota_reject_stage"),
        ("conflict", "sota_reject_stage"),
        ("bear_total", "sota_reject_stage"),
        ("net_score", "bear_total", "sota_reject_stage"),
        ("feature_bearish_structure", "sota_reject_stage"),
        ("entry_year", "sota_reject_stage"),
    ]
    subgroup_candidates: list[dict[str, Any]] = []
    for condition in condition_names:
        condition_events = [event for event in rejected_events if condition in recall_conditions(event)]
        for dimensions in dimension_sets:
            groups: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
            for event in condition_events:
                values = subgroup_values(event, dimensions)
                key = subgroup_key(values)
                if key not in groups:
                    groups[key] = (values, [])
                groups[key][1].append(event)
            for values, matched in groups.values():
                if len(matched) < int(args.min_subgroup_trades):
                    continue
                for target_leverage in target_leverages:
                    subgroup_candidates.append(
                        evaluate_candidate(
                            condition=condition,
                            dimensions=dimensions,
                            values=values,
                            matched=matched,
                            target_leverage=target_leverage,
                        )
                    )

    by_stage = defaultdict(list)
    by_condition = defaultdict(list)
    for event in rejected_events:
        by_stage[str(event.get("sota_reject_stage") or "")].append(event)
        for condition in recall_conditions(event):
            by_condition[condition].append(event)

    report = {
        "metadata": {
            "config": str(Path(args.config).resolve()),
            "frozen_report": str(Path(args.frozen_report).resolve()),
            "pressure_params": str(Path(args.pressure_params).resolve()),
            "pressure_params_applied": pressure_params,
            "trailing_rr_modes": trailing_rr_modes,
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "raw_sota_candidates": len(raw_events),
            "accepted_sota_after_gates": len(accepted_events),
            "rejected_sota_candidates": len(rejected_events),
            "score_gate": {
                "net_min": int(replay_args.sota_score_net_min),
                "bull_min": int(replay_args.sota_score_bull_min),
                "bear_max": int(replay_args.sota_score_bear_max),
                "conflict_mode": str(replay_args.sota_score_conflict_mode),
            },
            "long_score_bucket_sizing_rules": parse_score_bucket_rules(str(replay_args.long_score_bucket_sizing_rules_json)),
        },
        "frozen": frozen_summary,
        "rejected_breakdown": {
            "by_stage": {stage: group_stats(events) for stage, events in sorted(by_stage.items())},
            "by_recall_condition": {condition: group_stats(events) for condition, events in sorted(by_condition.items())},
            "score_bucket_counts": dict(
                sorted(Counter(f"n{e.get('net_score')}_b{e.get('bull_total')}_bear{e.get('bear_total')}" for e in rejected_events).items())
            ),
        },
        "candidates": sorted(
            candidates,
            key=lambda item: (
                -float(item["delta_vs_frozen"]["current_year_return_pct"]),
                float(item["delta_vs_frozen"]["max_drawdown_pct"]),
                -float(item["delta_vs_frozen"]["total_return_pct"]),
            ),
        ),
        "subgroup_candidates": sorted(
            subgroup_candidates,
            key=lambda item: (
                -float(item["delta_vs_frozen"]["current_year_return_pct"]),
                float(item["delta_vs_frozen"]["max_drawdown_pct"]),
                -float(item["delta_vs_frozen"]["total_return_pct"]),
            ),
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(clean_for_json(report), indent=2, ensure_ascii=False) + "\n")
    print(args.output)
    print("frozen", frozen_summary)
    print("rejected", report["metadata"]["rejected_sota_candidates"], report["rejected_breakdown"]["by_stage"])
    print("top")
    for item in report["candidates"][:10]:
        print(item["condition"], item["target_effective_leverage"], item["candidate"], item["delta_vs_frozen"], item["matched_rejected_trades"], item["accepted_recall_trades"])
    print("top_subgroups")
    for item in report["subgroup_candidates"][:10]:
        print(
            item["condition"],
            item["dimensions"],
            item["values"],
            item["target_effective_leverage"],
            item["candidate"],
            item["delta_vs_frozen"],
            item["matched_rejected_trades"],
            item["accepted_recall_trades"],
        )


if __name__ == "__main__":
    main()
