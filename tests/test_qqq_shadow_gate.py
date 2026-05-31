from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.qqq_shadow_gate import QqqShadowGateProfile, QqqShadowGateStateMachine
from bot.qqq_usdt_executor import QqqOrderContext, QqqUsdtExecutionEngine
from bot.strategy_router import RoutedSignalCandidate, StrategyRouterConfig


def test_shadow_gate_requires_clear_bars_after_stop() -> None:
    gate = QqqShadowGateStateMachine(
        QqqShadowGateProfile(
            enabled=True,
            reentry_rule="clear",
            reentry_clear_bars=2,
            initial_capital=1000.0,
        )
    )
    state = gate.default_state()
    state, _ = gate.record_open(state, timestamp="2026-05-01T00:00:00Z", entry_price=100.0, leverage=10.0)
    state, close_event = gate.record_close(
        state,
        timestamp="2026-05-01T04:00:00Z",
        exit_price=99.0,
        reason="qqq_trailing_stop_hit",
        taker_fee_rate=0.0,
        slippage_bps=0.0,
    )

    assert close_event is not None
    assert state["stopped_after_stop"] is True
    assert gate.entry_decision(state)["reason"] == "reentry_clear_bars"

    state, _ = gate.observe_bar(state, timestamp="2026-05-01T08:00:00Z", allow_long=True, defense_state=False)
    assert gate.entry_decision(state)["reason"] == "reentry_clear_bars"

    state, _ = gate.observe_bar(state, timestamp="2026-05-01T12:00:00Z", allow_long=True, defense_state=False)
    assert gate.entry_decision(state)["allow"] is True


def test_shadow_gate_loss_streak_triggers_cooldown() -> None:
    gate = QqqShadowGateStateMachine(
        QqqShadowGateProfile(
            enabled=True,
            loss_streak_stop=2,
            loss_streak_cooldown_bars=3,
            initial_capital=1000.0,
        )
    )
    state = gate.default_state()

    state, _ = gate.record_open(state, timestamp="t0", entry_price=100.0, leverage=10.0)
    state, _ = gate.record_close(state, timestamp="t1", exit_price=99.5, reason="signal_off", taker_fee_rate=0.0, slippage_bps=0.0)
    assert state["loss_streak"] == 1
    assert state["gate_remaining_bars"] == 0

    state, _ = gate.record_open(state, timestamp="t2", entry_price=100.0, leverage=10.0)
    state, close_event = gate.record_close(state, timestamp="t3", exit_price=99.5, reason="signal_off", taker_fee_rate=0.0, slippage_bps=0.0)
    assert close_event is not None
    assert close_event["triggers"] == ["loss_streak"]
    assert state["loss_streak"] == 0
    assert state["gate_reason"] == "loss_streak"
    assert gate.entry_decision(state)["allow"] is False

    state, _ = gate.observe_bar(state, timestamp="t4", allow_long=True, defense_state=False)
    assert state["gate_remaining_bars"] == 2


def test_shadow_gate_equity_dd_triggers_cooldown() -> None:
    gate = QqqShadowGateStateMachine(
        QqqShadowGateProfile(
            enabled=True,
            equity_dd_stop_pct=25.0,
            equity_dd_cooldown_bars=2,
            initial_capital=1000.0,
        )
    )
    state = gate.default_state()
    state, _ = gate.record_open(state, timestamp="t0", entry_price=100.0, leverage=10.0)
    state, close_event = gate.record_close(state, timestamp="t1", exit_price=96.0, reason="qqq_trailing_stop_hit", taker_fee_rate=0.0, slippage_bps=0.0)

    assert close_event is not None
    assert close_event["triggers"] == ["equity_dd"]
    assert state["gate_remaining_bars"] == 2
    assert state["gate_reason"] == "equity_dd"
    assert gate.entry_decision(state)["allow"] is False


def test_qqq_executor_writes_shadow_gate_action_log(tmp_path: Path) -> None:
    config_path = tmp_path / "qqq.json"
    config_path.write_text(
        json.dumps(
            {
                "execution_symbol": "QQQ/USDT:USDT",
                "base_leverage": 10.0,
                "offense_leverage": 10.0,
                "stop_loss_pct": 4.0,
                "initial_capital": 1000.0,
                "taker_fee_rate": 0.0,
                "slippage_bps": 0.0,
                "shadow_gate_replay_profile": {
                    "status": "runtime_enabled",
                    "runtime_enabled": True,
                    "reentry_rule": "clear",
                    "reentry_clear_bars": 2,
                    "loss_streak_stop": 2,
                    "loss_streak_cooldown_bars": 20,
                    "equity_dd_stop_pct": 25.0,
                    "equity_dd_cooldown_bars": 10,
                },
            }
        )
    )
    engine = QqqUsdtExecutionEngine(
        StrategyRouterConfig(
            mode="paper",
            state_path=str(tmp_path / "router.json"),
            btc_strategy_config="config/config.paper.high-leverage-structure.json",
            qqq_strategy_config=str(config_path),
            qqq_state_db_path=str(tmp_path / "qqq_state.db"),
        ),
        config_path,
    )
    context = QqqOrderContext(
        symbol="QQQ/USDT:USDT",
        margin_mode="isolated",
        leverage=10.0,
        stop_loss_pct=4.0,
        reference_price=100.0,
        latest_low=100.0,
        stop_price=96.0,
        stop_hit=False,
        route_score=100.0,
        candidate={
            "active": True,
            "timestamp": "2026-05-01T00:00:00Z",
            "metadata": {"defense_state": False},
        },
    )
    candidate = RoutedSignalCandidate(
        "qqq_usdt_aggressive",
        "QQQ/USDT:USDT",
        True,
        100.0,
        timestamp="2026-05-01T00:00:00Z",
        leverage=10.0,
    )
    engine._risk_on_window_status = lambda: {"enabled": False, "open": True}  # type: ignore[method-assign]
    engine._build_context = lambda _: context  # type: ignore[method-assign]

    opened = engine.evaluate_latest(candidate)
    assert opened["actions"][0]["status"] == "paper_opened"

    closed = engine.close_position(
        reason="qqq_trailing_stop_hit",
        exit_price=96.0,
        timestamp="2026-05-01T04:00:00Z",
    )
    assert closed["status"] == "paper_closed"

    actions = engine.store.recent_actions(limit=10)
    gate_actions = [item for item in actions if item["action_type"] == "QQQ_SHADOW_GATE"]
    events = [item["payload"]["event"]["event"] for item in gate_actions]
    assert "observe_bar" in events
    assert "open" in events
    assert "close" in events
