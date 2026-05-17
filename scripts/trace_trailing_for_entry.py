#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from bot.okx_executor import ExecutorConfig  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine  # noqa: E402
from scripts.replay_sota_smc_live_shadow import apply_trailing_rr_modes  # noqa: E402
from strategy.scalp_robust_v2_core import ActionType, Direction, PositionState, ScalpRobustEngine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace stop/trailing state for one engine trade entry.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--entry-time", required=True)
    parser.add_argument("--live-db", default=None)
    parser.add_argument("--use-live-anchor", action="store_true")
    parser.add_argument("--direct-live-anchor", action="store_true")
    parser.add_argument("--stage-trigger-rr-mode", default="close", choices=("close", "extreme"))
    parser.add_argument("--time-trailing-rr-mode", default="close", choices=("close", "extreme"))
    parser.add_argument("--atr-activation-rr-mode", default="close", choices=("close", "extreme"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-rows", type=int, default=400)
    return parser.parse_args()


def timestamp_text(engine: Any, idx: int) -> str:
    return str(engine._timestamp_for_idx(idx))


def to_utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def load_manual_sync_anchor(live_db_path: Path, entry_time: pd.Timestamp) -> dict[str, Any] | None:
    conn = sqlite3.connect(str(live_db_path))
    conn.row_factory = sqlite3.Row
    try:
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('actions', 'action_log')"
        ).fetchall()
        table_names = {str(row["name"]) for row in table_rows}
        table_name = "actions" if "actions" in table_names else "action_log" if "action_log" in table_names else None
        if table_name is None:
            raise ValueError(f"No actions/action_log table found in {live_db_path}")
        rows = conn.execute(
            f"""
            SELECT timestamp, action_type, payload
            FROM {table_name}
            WHERE action_type IN ('OPEN_LONG', 'OPEN_SHORT', 'MANUAL_POSITION_SYNC')
            ORDER BY timestamp ASC, id ASC
            """
        ).fetchall()
    finally:
        conn.close()

    target_entry = to_utc_timestamp(entry_time)
    pending_open: dict[str, Any] | None = None
    candidate_syncs: list[dict[str, Any]] = []
    for row in rows:
        action_type = str(row["action_type"] or "")
        raw_payload = row["payload"]
        try:
            payload = json.loads(raw_payload) if raw_payload else {}
        except json.JSONDecodeError:
            payload = {}
        timestamp = to_utc_timestamp(row["timestamp"])
        if action_type in {"OPEN_LONG", "OPEN_SHORT"}:
            if pending_open is not None and candidate_syncs:
                return select_manual_sync_anchor(pending_open, candidate_syncs)
            action_entry_time = payload.get("timestamp") or payload.get("entry_time") or row["timestamp"]
            try:
                normalized_entry = to_utc_timestamp(action_entry_time)
            except Exception:
                normalized_entry = timestamp
            if normalized_entry == target_entry:
                pending_open = {
                    "timestamp": str(timestamp),
                    "action_type": action_type,
                    "payload": payload,
                }
                candidate_syncs = []
                continue
            pending_open = None
            candidate_syncs = []
            continue
        if action_type == "MANUAL_POSITION_SYNC" and pending_open is not None:
            candidate_syncs.append({"timestamp": str(timestamp), "payload": payload})
    if pending_open is not None and candidate_syncs:
        return select_manual_sync_anchor(pending_open, candidate_syncs)
    return None


def manual_sync_anchor_from_payload(pending_open: dict[str, Any], manual_sync: dict[str, Any]) -> dict[str, Any] | None:
    payload = manual_sync["payload"]
    snapshot = payload.get("snapshot") if isinstance(payload, dict) else None
    if not isinstance(snapshot, dict):
        return None
    position = snapshot.get("position")
    if not isinstance(position, dict):
        position = snapshot
    return {
        "open_action": pending_open,
        "manual_sync": manual_sync,
        "anchor": {
            "entry_price": float(position.get("entry_price", 0.0) or 0.0),
            "sl_price": float(position.get("sl_price", 0.0) or 0.0),
            "initial_sl_price": float(position.get("initial_sl_price", 0.0) or 0.0),
            "target_price": float(position.get("target_price", 0.0) or 0.0),
            "quantity": abs(float(position.get("quantity", 0.0) or 0.0)),
            "notional": float(position.get("notional", 0.0) or 0.0),
            "capital_at_entry": float(position.get("capital_at_entry", 0.0) or 0.0),
            "entry_fee": float(position.get("entry_fee", 0.0) or 0.0),
            "entry_slippage_cost": float(position.get("entry_slippage_cost", 0.0) or 0.0),
        },
    }


def select_manual_sync_anchor(pending_open: dict[str, Any], syncs: list[dict[str, Any]]) -> dict[str, Any] | None:
    preferred = [
        sync
        for sync in syncs
        if isinstance(sync.get("payload"), dict) and sync["payload"].get("context") == "after_execute"
    ]
    for sync in [*(reversed(preferred)), *reversed(syncs)]:
        anchor = manual_sync_anchor_from_payload(pending_open, sync)
        if anchor is not None:
            return anchor
    return None


def apply_live_anchor(position: Any, live_anchor: dict[str, Any]) -> None:
    entry_price = float(live_anchor.get("entry_price", 0.0) or 0.0)
    stop_price = float(live_anchor.get("sl_price", 0.0) or live_anchor.get("initial_sl_price", 0.0) or 0.0)
    initial_stop = float(live_anchor.get("initial_sl_price", 0.0) or stop_price)
    target_price = float(live_anchor.get("target_price", 0.0) or 0.0)
    quantity = abs(float(live_anchor.get("quantity", 0.0) or 0.0))
    if entry_price > 0:
        setattr(position, "entry_price", entry_price)
    if stop_price > 0:
        setattr(position, "sl_price", stop_price)
    if initial_stop > 0:
        setattr(position, "initial_sl_price", initial_stop)
    if target_price > 0:
        setattr(position, "target_price", target_price)
    if quantity > 0:
        setattr(position, "quantity", quantity)
        risk_price = abs(float(getattr(position, "entry_price", 0.0) or 0.0) - float(getattr(position, "initial_sl_price", 0.0) or 0.0))
        setattr(position, "risk_amount", quantity * risk_price)
    for key in ("notional", "capital_at_entry", "entry_fee", "entry_slippage_cost"):
        value = float(live_anchor.get(key, 0.0) or 0.0)
        if value > 0:
            setattr(position, key, value)


def build_engine(payload: dict[str, Any], prepared: Any) -> ScalpRobustEngine:
    config = ExecutorConfig.from_dict(payload).to_scalp_strategy_config()
    engine = ScalpRobustEngine(
        prepared.c4h,
        prepared.c15m,
        prepared.mapping,
        prepared.precomputed,
        config,
    )
    engine._regime_switch_cache = {
        c4h_idx: (label, engine._config_for_regime(label))
        for c4h_idx, label in prepared.regime_labels.items()
    }
    engine._regime_feature_cache = dict(prepared.regime_features)
    return engine


def candle_idx_for_time(engine: Any, value: Any) -> int | None:
    target = to_utc_timestamp(value)
    for idx, candle in enumerate(engine.c15m):
        if pd.Timestamp(candle.ts, unit="s", tz="UTC") == target:
            return idx
    return None


def live_anchor_position(anchor_bundle: dict[str, Any], engine: Any, entry_time: pd.Timestamp) -> PositionState:
    open_payload = anchor_bundle["open_action"]["payload"]
    open_metadata = open_payload.get("metadata") if isinstance(open_payload.get("metadata"), dict) else {}
    sync_payload = anchor_bundle["manual_sync"]["payload"]
    snapshot = sync_payload.get("snapshot") if isinstance(sync_payload, dict) else {}
    snapshot_position = snapshot.get("position") if isinstance(snapshot, dict) else {}
    if not isinstance(snapshot_position, dict):
        snapshot_position = {}
    anchor = anchor_bundle["anchor"]
    mapped_idx = candle_idx_for_time(engine, entry_time)
    if mapped_idx is None:
        raise ValueError(f"Could not map live entry time to local candles: {entry_time}")
    entry_idx = mapped_idx
    direction = str(snapshot_position.get("direction") or open_payload.get("direction") or Direction.BULL)
    entry_price = float(anchor.get("entry_price", open_payload.get("entry_price", 0.0)) or 0.0)
    sl_price = float(anchor.get("sl_price", anchor.get("initial_sl_price", open_payload.get("stop_price", 0.0))) or 0.0)
    initial_sl_price = float(anchor.get("initial_sl_price", sl_price) or sl_price)
    target_price = float(anchor.get("target_price", open_payload.get("target_price", 0.0)) or 0.0)
    quantity = abs(float(anchor.get("quantity", snapshot_position.get("quantity", 0.0)) or 0.0))
    risk_amount = quantity * abs(entry_price - initial_sl_price)
    target_rr = float(snapshot_position.get("target_rr", open_metadata.get("target_rr", engine.config.rr_ratio)) or engine.config.rr_ratio)
    max_hold_bars_raw = snapshot_position.get("max_hold_bars", open_metadata.get("max_hold_bars"))
    max_hold_bars = int(max_hold_bars_raw) if max_hold_bars_raw is not None else None
    return PositionState(
        direction=direction,
        signal_entry_price=float(snapshot_position.get("signal_entry_price", open_metadata.get("signal_entry_price", entry_price)) or entry_price),
        entry_price=entry_price,
        sl_price=sl_price,
        initial_sl_price=initial_sl_price,
        target_price=target_price,
        entry_time=str(snapshot_position.get("entry_time") or open_payload.get("timestamp") or timestamp_text(engine, entry_idx)),
        capital_at_entry=float(anchor.get("capital_at_entry", snapshot_position.get("capital_at_entry", 0.0)) or 0.0),
        risk_amount=risk_amount,
        notional=float(anchor.get("notional", snapshot_position.get("notional", 0.0)) or 0.0),
        quantity=quantity,
        entry_fee=float(anchor.get("entry_fee", snapshot_position.get("entry_fee", 0.0)) or 0.0),
        entry_slippage_cost=float(anchor.get("entry_slippage_cost", snapshot_position.get("entry_slippage_cost", 0.0)) or 0.0),
        entry_idx=entry_idx,
        entry_regime_score=int(snapshot_position.get("entry_regime_score", open_metadata.get("entry_regime_score", 0)) or 0),
        target_rr=target_rr,
        max_hold_bars=max_hold_bars,
        trail_style=str(snapshot_position.get("trail_style", open_metadata.get("trail_style", "normal")) or "normal"),
        stage=int(snapshot_position.get("stage", -1) or -1),
        tit_stage=int(snapshot_position.get("tit_stage", 0) or 0),
        time_based_trailing_enabled=bool(snapshot_position.get("time_based_trailing_enabled", open_metadata.get("time_based_trailing_enabled", False))),
        auto_tit_reason=snapshot_position.get("auto_tit_reason", open_metadata.get("auto_tit_reason")),
        risk_regime=snapshot_position.get("risk_regime", open_metadata.get("risk_regime")),
        regime_label=snapshot_position.get("regime_label", open_metadata.get("regime_label")),
        exit_profile=snapshot_position.get("exit_profile", open_metadata.get("exit_profile")),
        exit_profile_reason=snapshot_position.get("exit_profile_reason", open_metadata.get("exit_profile_reason")),
        exit_profile_overrides=snapshot_position.get("exit_profile_overrides", open_metadata.get("exit_profile_overrides")),
    )


def trace_position(engine: Any, *, max_rows: int) -> list[dict[str, Any]]:
    if engine.position is None:
        return []
    entry_idx = int(getattr(engine.position, "entry_idx"))
    trace: list[dict[str, Any]] = []
    end_idx = min(len(engine.c15m) - 1, entry_idx + max_rows)
    for idx in range(entry_idx + 1, end_idx + 1):
        engine._apply_regime_switch_for_idx(idx)
        pos = engine.position
        if pos is None:
            break
        candle = engine.c15m[idx]
        stop_before = float(pos.sl_price)
        target_before = float(pos.target_price)
        highest, lowest = engine._price_extrema_since_entry(pos, idx)
        atr = engine._atr_for_idx(idx)
        stage_rr = engine._unrealized_rr_for_mode(pos, candle, engine._exit_str(pos, "stage_trigger_rr_mode", "close"))
        atr_rr = engine._unrealized_rr_for_mode(pos, candle, engine._exit_str(pos, "atr_activation_rr_mode", "close"))
        stopped_before_update = (
            candle.l <= stop_before if pos.direction == Direction.BULL else candle.h >= stop_before
        )
        actions = engine.manage_position(idx)
        action_payloads = [asdict(action) for action in actions]
        pos_after = engine.position
        trace.append(
            {
                "idx": idx,
                "time": timestamp_text(engine, idx),
                "open": candle.o,
                "high": candle.h,
                "low": candle.l,
                "close": candle.c,
                "stop_before": stop_before,
                "target_before": target_before,
                "highest_since_entry": highest,
                "lowest_since_entry": lowest,
                "atr": atr,
                "stage_rr": stage_rr,
                "atr_rr": atr_rr,
                "stopped_before_update": stopped_before_update,
                "actions": action_payloads,
                "stop_after": float(getattr(pos_after, "sl_price", 0.0) or 0.0) if pos_after is not None else None,
                "position_open_after": pos_after is not None,
            }
        )
        if any(action.type == ActionType.CLOSE_POSITION for action in actions):
            break
    return trace


def trace_trade(
    engine: Any,
    trade: Any,
    *,
    max_rows: int,
    live_anchor: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    direction = str(getattr(trade, "direction"))
    entry_idx = int(getattr(trade, "entry_idx"))
    exit_idx = int(getattr(trade, "exit_idx"))
    initial_stop = float(getattr(trade, "initial_stop_price", 0.0) or 0.0)
    entry_price = float(getattr(trade, "entry_price", 0.0) or 0.0)
    risk_regime = getattr(trade, "risk_regime", None)
    regime_label = getattr(trade, "regime_label", None)

    # Recreate a single-position engine state by replaying actions until the target entry opens.
    replay_engine = type(engine)(engine.c4h, engine.c15m, engine.mapping, engine.precomputed, engine._base_config)
    replay_engine._regime_switch_cache = dict(getattr(engine, "_regime_switch_cache", {}))
    replay_engine._regime_feature_cache = dict(getattr(engine, "_regime_feature_cache", {}))
    actions = replay_engine.evaluate_range(max(100, entry_idx - 500), entry_idx + 1)
    if replay_engine.position is None or int(getattr(replay_engine.position, "entry_idx", -1)) != entry_idx:
        # Fall back to the materialized trade fields; this keeps the trace useful even if a prior
        # stateful dependency prevents isolated replay from opening the exact trade.
        from strategy.scalp_robust_v2_core import PositionState

        quantity = abs(float(getattr(trade, "quantity", 0.0) or 0.0))
        target_price = float(getattr(trade, "target_price", 0.0) or 0.0)
        notional = float(getattr(trade, "notional", 0.0) or 0.0)
        capital_at_entry = float(getattr(trade, "capital_at_entry", 0.0) or 0.0)
        entry_fee = float(getattr(trade, "entry_fee", 0.0) or 0.0)
        entry_slippage_cost = float(getattr(trade, "slippage_cost", 0.0) or 0.0)
        signal_entry_price = float(getattr(trade, "signal_entry_price", entry_price) or entry_price)
        if live_anchor is not None:
            entry_price = float(live_anchor.get("entry_price", entry_price) or entry_price)
            initial_stop = float(live_anchor.get("initial_sl_price", initial_stop) or initial_stop)
            target_price = float(live_anchor.get("target_price", target_price) or target_price)
            quantity = abs(float(live_anchor.get("quantity", quantity) or quantity))
            notional = float(live_anchor.get("notional", notional) or notional)
            capital_at_entry = float(live_anchor.get("capital_at_entry", capital_at_entry) or capital_at_entry)
            entry_fee = float(live_anchor.get("entry_fee", entry_fee) or entry_fee)
            entry_slippage_cost = float(live_anchor.get("entry_slippage_cost", entry_slippage_cost) or entry_slippage_cost)
        initial_risk_price = abs(entry_price - initial_stop)
        replay_engine.position = PositionState(
            direction=direction,
            signal_entry_price=signal_entry_price,
            entry_price=entry_price,
            sl_price=initial_stop,
            initial_sl_price=initial_stop,
            target_price=target_price,
            entry_time=timestamp_text(engine, entry_idx),
            capital_at_entry=capital_at_entry,
            risk_amount=quantity * initial_risk_price,
            notional=notional,
            quantity=quantity,
            entry_fee=entry_fee,
            entry_slippage_cost=entry_slippage_cost,
            entry_idx=entry_idx,
            entry_regime_score=0,
            target_rr=float(getattr(trade, "target_rr", 0.0) or engine.config.rr_ratio),
            max_hold_bars=None,
            trail_style=str(getattr(trade, "trail_style", "normal") or "normal"),
            risk_regime=risk_regime,
            regime_label=regime_label,
            time_based_trailing_enabled=bool(getattr(trade, "time_based_trailing_enabled", False)),
            auto_tit_reason=getattr(trade, "auto_tit_reason", None),
            exit_profile=getattr(trade, "exit_profile", None),
            exit_profile_reason=getattr(trade, "exit_profile_reason", None),
            exit_profile_overrides=getattr(trade, "exit_profile_overrides", None),
        )
    if live_anchor is not None and replay_engine.position is not None:
        apply_live_anchor(replay_engine.position, live_anchor)

    trace: list[dict[str, Any]] = []
    end_idx = min(len(replay_engine.c15m) - 1, max(exit_idx, entry_idx) + 1, entry_idx + max_rows)
    for idx in range(entry_idx + 1, end_idx + 1):
        replay_engine._apply_regime_switch_for_idx(idx)
        pos = replay_engine.position
        if pos is None:
            break
        candle = replay_engine.c15m[idx]
        stop_before = float(pos.sl_price)
        target_before = float(pos.target_price)
        highest, lowest = replay_engine._price_extrema_since_entry(pos, idx)
        atr = replay_engine._atr_for_idx(idx)
        stage_rr = replay_engine._unrealized_rr_for_mode(pos, candle, replay_engine._exit_str(pos, "stage_trigger_rr_mode", "close"))
        atr_rr = replay_engine._unrealized_rr_for_mode(pos, candle, replay_engine._exit_str(pos, "atr_activation_rr_mode", "close"))
        stopped_before_update = (
            candle.l <= stop_before if pos.direction == Direction.BULL else candle.h >= stop_before
        )
        actions = replay_engine.manage_position(idx)
        action_payloads = [asdict(action) for action in actions]
        pos_after = replay_engine.position
        trace.append(
            {
                "idx": idx,
                "time": timestamp_text(replay_engine, idx),
                "open": candle.o,
                "high": candle.h,
                "low": candle.l,
                "close": candle.c,
                "stop_before": stop_before,
                "target_before": target_before,
                "highest_since_entry": highest,
                "lowest_since_entry": lowest,
                "atr": atr,
                "stage_rr": stage_rr,
                "atr_rr": atr_rr,
                "stopped_before_update": stopped_before_update,
                "actions": action_payloads,
                "stop_after": float(getattr(pos_after, "sl_price", 0.0) or 0.0) if pos_after is not None else None,
                "position_open_after": pos_after is not None,
            }
        )
        if any(action.type == ActionType.CLOSE_POSITION for action in actions):
            break
    return trace


def main() -> None:
    args = parse_args()
    payload = load_config_payload(Path(args.config))
    payload, pressure_params = apply_pressure_params(payload, Path(args.pressure_params))
    payload, trailing_modes = apply_trailing_rr_modes(
        payload,
        stage_trigger_rr_mode=args.stage_trigger_rr_mode,
        time_trailing_rr_mode=args.time_trailing_rr_mode,
        atr_activation_rr_mode=args.atr_activation_rr_mode,
    )
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=payload.get("regime_switcher_thresholds"),
    )
    target = to_utc_timestamp(args.entry_time)
    anchor_bundle = None
    if args.use_live_anchor:
        if not args.live_db:
            raise SystemExit("--use-live-anchor requires --live-db")
        anchor_bundle = load_manual_sync_anchor(Path(args.live_db), target)
        if anchor_bundle is None:
            raise SystemExit(f"No MANUAL_POSITION_SYNC anchor found in {args.live_db} for entry_time={target}")
    if args.direct_live_anchor:
        if anchor_bundle is None:
            raise SystemExit("--direct-live-anchor requires --use-live-anchor and --live-db")
        engine = build_engine(payload, prepared)
        engine.position = live_anchor_position(anchor_bundle, engine, target)
        trade = engine.position
        trace = trace_position(engine, max_rows=int(args.max_rows))
        engine_total_return_pct = None
    else:
        metrics, engine = run_engine(payload, prepared, args.start_date)
        matched = [
            trade
            for trade in engine.trades
            if to_utc_timestamp(getattr(trade, "entry_time")) == target
        ]
        if not matched:
            raise SystemExit(f"No engine trade found for entry_time={target}")
        trade = matched[0]
        trace = trace_trade(engine, trade, max_rows=int(args.max_rows), live_anchor=(anchor_bundle or {}).get("anchor"))
        engine_total_return_pct = metrics.get("total_return_pct")
    report = {
        "metadata": {
            "config": str(Path(args.config).resolve()),
            "pressure_params": str(Path(args.pressure_params).resolve()),
            "pressure_params_applied": pressure_params,
            "trailing_modes": trailing_modes,
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "engine_total_return_pct": engine_total_return_pct,
            "use_live_anchor": bool(args.use_live_anchor),
            "direct_live_anchor": bool(args.direct_live_anchor),
        },
        "trade": {key: value for key, value in trade.__dict__.items() if key not in {"metadata"}},
        "live_anchor": anchor_bundle,
        "trace": trace,
    }
    print(json.dumps({**report["metadata"], "trade": report["trade"], "trace_rows": len(trace)}, ensure_ascii=False, default=str, indent=2))
    for row in trace:
        if row["actions"] or row["stopped_before_update"] or row["time"] >= "2026-05-04 02:00":
            print(json.dumps(row, ensure_ascii=False, default=str, sort_keys=True))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, default=str, indent=2) + "\n")
        print(f"JSON written: {output}")


if __name__ == "__main__":
    main()
