#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.qqq_shadow_gate import QqqShadowGateProfile, QqqShadowGateStateMachine  # noqa: E402
from bot.state_store import StateStore  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "config.paper.qqq-usdt-aggressive-frozen.json"
DEFAULT_DB = ROOT / "var" / "tmp" / "qqq_shadow_gate_mock_trace.db"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_shadow_gate_mock_trace_20260530.json"


def state_summary(gate: QqqShadowGateStateMachine, state: dict[str, Any]) -> dict[str, Any]:
    normalized = gate.normalize_state(state)
    return {
        "enabled": bool(normalized.get("enabled", False)),
        "capital": round(float(normalized.get("capital", 0.0) or 0.0), 6),
        "equity_peak": round(float(normalized.get("equity_peak", 0.0) or 0.0), 6),
        "loss_streak": int(normalized.get("loss_streak", 0) or 0),
        "gate_remaining_bars": int(normalized.get("gate_remaining_bars", 0) or 0),
        "gate_reason": normalized.get("gate_reason"),
        "clear_streak": int(normalized.get("clear_streak", 0) or 0),
        "stopped_after_stop": bool(normalized.get("stopped_after_stop", False)),
        "last_bar_timestamp": normalized.get("last_bar_timestamp"),
        "position_open": isinstance(normalized.get("position"), dict),
    }


def profile_payload(profile: QqqShadowGateProfile) -> dict[str, Any]:
    return {
        "enabled": profile.enabled,
        "reentry_rule": profile.reentry_rule,
        "reentry_clear_bars": profile.reentry_clear_bars,
        "loss_streak_stop": profile.loss_streak_stop,
        "loss_streak_cooldown_bars": profile.loss_streak_cooldown_bars,
        "equity_dd_stop_pct": profile.equity_dd_stop_pct,
        "equity_dd_cooldown_bars": profile.equity_dd_cooldown_bars,
        "initial_capital": profile.initial_capital,
    }


def emit(
    *,
    store: StateStore,
    gate: QqqShadowGateStateMachine,
    state: dict[str, Any],
    scenario: str,
    event: dict[str, Any],
    decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "scenario": scenario,
        "event": event,
        "decision": decision,
        "state": state_summary(gate, state),
        "profile": profile_payload(gate.profile),
    }
    store.append_action(str(event.get("timestamp") or "mock"), "QQQ_SHADOW_GATE", payload)
    return payload


def run_clear_bars_scenario(store: StateStore, base_profile: QqqShadowGateProfile) -> list[dict[str, Any]]:
    profile = replace(base_profile, loss_streak_stop=0, equity_dd_stop_pct=0.0)
    gate = QqqShadowGateStateMachine(profile)
    state = gate.default_state()
    trace: list[dict[str, Any]] = []

    state, event = gate.observe_bar(state, timestamp="2026-05-01T00:00:00Z", allow_long=True, defense_state=False)
    trace.append(emit(store=store, gate=gate, state=state, scenario="clear_bars_after_stop", event=event or {"event": "observe_bar"}))
    decision = gate.entry_decision(state)
    trace.append(emit(store=store, gate=gate, state=state, scenario="clear_bars_after_stop", event={"event": "entry_decision", "timestamp": "2026-05-01T00:00:00Z"}, decision=decision))
    if decision["allow"]:
        state, event = gate.record_open(state, timestamp="2026-05-01T00:00:00Z", entry_price=100.0, leverage=10.0)
        trace.append(emit(store=store, gate=gate, state=state, scenario="clear_bars_after_stop", event=event or {"event": "open"}))

    state, event = gate.record_close(state, timestamp="2026-05-01T04:00:00Z", exit_price=99.0, reason="qqq_trailing_stop_hit", taker_fee_rate=0.0, slippage_bps=0.0)
    trace.append(emit(store=store, gate=gate, state=state, scenario="clear_bars_after_stop", event=event or {"event": "close"}))
    for timestamp in ["2026-05-01T08:00:00Z", "2026-05-01T12:00:00Z"]:
        state, event = gate.observe_bar(state, timestamp=timestamp, allow_long=True, defense_state=False)
        decision = gate.entry_decision(state)
        trace.append(emit(store=store, gate=gate, state=state, scenario="clear_bars_after_stop", event=event or {"event": "observe_bar", "timestamp": timestamp}, decision=decision))
    return trace


def run_loss_streak_scenario(store: StateStore, base_profile: QqqShadowGateProfile) -> list[dict[str, Any]]:
    profile = replace(base_profile, reentry_clear_bars=0, equity_dd_stop_pct=0.0)
    gate = QqqShadowGateStateMachine(profile)
    state = gate.default_state()
    trace: list[dict[str, Any]] = []
    timestamps = [
        ("2026-05-02T00:00:00Z", "2026-05-02T04:00:00Z"),
        ("2026-05-02T08:00:00Z", "2026-05-02T12:00:00Z"),
    ]
    for entry_ts, exit_ts in timestamps:
        state, event = gate.record_open(state, timestamp=entry_ts, entry_price=100.0, leverage=10.0)
        trace.append(emit(store=store, gate=gate, state=state, scenario="loss_streak_cooldown", event=event or {"event": "open"}))
        state, event = gate.record_close(state, timestamp=exit_ts, exit_price=99.5, reason="signal_off", taker_fee_rate=0.0, slippage_bps=0.0)
        decision = gate.entry_decision(state)
        trace.append(emit(store=store, gate=gate, state=state, scenario="loss_streak_cooldown", event=event or {"event": "close"}, decision=decision))

    state, event = gate.observe_bar(state, timestamp="2026-05-02T16:00:00Z", allow_long=True, defense_state=False)
    decision = gate.entry_decision(state)
    trace.append(emit(store=store, gate=gate, state=state, scenario="loss_streak_cooldown", event=event or {"event": "observe_bar"}, decision=decision))
    return trace


def run_equity_dd_scenario(store: StateStore, base_profile: QqqShadowGateProfile) -> list[dict[str, Any]]:
    profile = replace(base_profile, reentry_clear_bars=0, loss_streak_stop=0)
    gate = QqqShadowGateStateMachine(profile)
    state = gate.default_state()
    trace: list[dict[str, Any]] = []

    state, event = gate.record_open(state, timestamp="2026-05-03T00:00:00Z", entry_price=100.0, leverage=10.0)
    trace.append(emit(store=store, gate=gate, state=state, scenario="equity_dd_cooldown", event=event or {"event": "open"}))
    state, event = gate.record_close(state, timestamp="2026-05-03T04:00:00Z", exit_price=96.0, reason="qqq_trailing_stop_hit", taker_fee_rate=0.0, slippage_bps=0.0)
    decision = gate.entry_decision(state)
    trace.append(emit(store=store, gate=gate, state=state, scenario="equity_dd_cooldown", event=event or {"event": "close"}, decision=decision))
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a mock QQQ shadow gate trace and SQLite action_log.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    profile = QqqShadowGateProfile.from_config(config)
    if not profile.enabled:
        raise RuntimeError("shadow_gate_replay_profile is not runtime-enabled")

    db_path = Path(args.db).resolve()
    if db_path.exists():
        db_path.unlink()
    store = StateStore(db_path)

    scenarios = {
        "clear_bars_after_stop": run_clear_bars_scenario(store, profile),
        "loss_streak_cooldown": run_loss_streak_scenario(store, profile),
        "equity_dd_cooldown": run_equity_dd_scenario(store, profile),
    }
    report = {
        "config": str(config_path),
        "db": str(db_path),
        "profile": profile_payload(profile),
        "scenarios": scenarios,
        "action_log_tail": list(reversed(store.recent_actions(limit=200))),
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "db": str(db_path), "events": len(report["action_log_tail"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
