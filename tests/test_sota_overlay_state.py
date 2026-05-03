from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bot.okx_executor import ExecutorConfig, OkxExecutionEngine
from strategy.scalp_robust_v2_core import ActionType, StrategyAction
from strategy.sota_overlay_state import (
    OverlayCandidate,
    account_lock_decision,
    leveraged_net_return,
    replay_single_position_events,
)


class OverlayStateTest(unittest.TestCase):
    def test_smc_net_return_uses_roundtrip_fee_and_slippage(self) -> None:
        result = leveraged_net_return(
            signal_return_pct=3.0,
            leverage=10.0,
            position_size_pct=1.0,
            allocation=1.0,
            taker_fee_rate=0.0005,
            slippage_bps=5.0,
        )

        self.assertAlmostEqual(result["gross_unit_return"], 0.03)
        self.assertAlmostEqual(result["roundtrip_cost"], 0.002)
        self.assertAlmostEqual(result["net_unit_return"], 0.028)
        self.assertAlmostEqual(result["account_return"], 0.28)
        self.assertAlmostEqual(result["account_return_pct"], 28.0)

    def test_stable_preempts_overlapping_sota(self) -> None:
        accepted, decisions = replay_single_position_events(
            [
                OverlayCandidate(
                    event_type="stable_reverse_short",
                    direction="BEAR",
                    entry_idx=10,
                    exit_idx=20,
                    entry_time="2026-01-01 00:00",
                    exit_time="2026-01-01 02:30",
                ),
                OverlayCandidate(
                    event_type="sota_long",
                    direction="BULL",
                    entry_idx=12,
                    exit_idx=25,
                    entry_time="2026-01-01 00:30",
                    exit_time="2026-01-01 03:15",
                ),
            ]
        )

        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].event_type, "stable_reverse_short")
        self.assertEqual(decisions[1]["decision"], "rejected")
        self.assertEqual(decisions[1]["reason"], "position_lock_open")
        self.assertEqual(decisions[1]["paper_tag"], "stable_preempted_sota")

    def test_account_lock_rejects_when_exchange_has_position(self) -> None:
        decision = account_lock_decision(
            OverlayCandidate(event_type="smc_short", direction="BEAR", entry_time="2026-01-01 00:00"),
            local_position_open=False,
            exchange_long_contracts=0.0,
            exchange_short_contracts=5.0,
        )

        self.assertEqual(decision["decision"], "rejected")
        self.assertEqual(decision["reason"], "account_position_open")
        self.assertEqual(decision["paper_tag"], "account_lock_rejected")

    def test_account_lock_marks_stable_preempting_sota_from_open_state(self) -> None:
        decision = account_lock_decision(
            OverlayCandidate(event_type="sota_long", direction="BULL", entry_time="2026-01-01 00:15"),
            local_position_open=True,
            blocking_candidate=OverlayCandidate(
                event_type="stable_reverse_short",
                direction="BEAR",
                entry_idx=10,
                exit_idx=20,
                entry_time="2026-01-01 00:00",
            ),
        )

        self.assertEqual(decision["decision"], "rejected")
        self.assertEqual(decision["paper_tag"], "stable_preempted_sota")
        self.assertEqual(decision["blocking_event_type"], "stable_reverse_short")


class ExecutorOverlayLockTest(unittest.TestCase):
    def build_executor(self) -> OkxExecutionEngine:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        return OkxExecutionEngine(
            ExecutorConfig(
                mode="paper",
                symbol="BTC/USDT:USDT",
                timeframe="15m",
                informative_timeframe="4h",
                leverage=10,
                margin_mode="isolated",
                max_open_positions=1,
                risk_per_trade=0.035,
                state_db_path=str(Path(tmpdir.name) / "state.db"),
                telegram_enabled=False,
            )
        )

    def test_executor_allows_current_open_action_owned_position(self) -> None:
        executor = self.build_executor()
        action = StrategyAction(
            type=ActionType.OPEN_LONG,
            timestamp="2026-01-01 00:00",
            direction="BULL",
            entry_price=100000.0,
            stop_price=99000.0,
            target_price=103000.0,
            metadata={"index": 1},
        )
        engine = type(
            "Engine",
            (),
            {
                "position": type("Position", (), {"entry_time": "2026-01-01 00:00", "direction": "BULL"})(),
            },
        )()

        self.assertIsNone(executor._sota_overlay_account_lock_pre_open(action, engine))
        self.assertIsNone(executor._load_sota_overlay_open_candidate())

    def test_executor_rejects_conflicting_local_position(self) -> None:
        executor = self.build_executor()
        executor._save_sota_overlay_open_candidate(
            OverlayCandidate(
                event_type="stable_reverse_short",
                direction="BEAR",
                entry_idx=1,
                entry_time="2026-01-01 00:00",
            )
        )
        action = StrategyAction(
            type=ActionType.OPEN_LONG,
            timestamp="2026-01-01 01:00",
            direction="BULL",
            entry_price=100000.0,
            stop_price=99000.0,
            target_price=103000.0,
            metadata={"index": 2, "overlay_event_type": "sota_long"},
        )
        engine = type(
            "Engine",
            (),
            {
                "position": type("Position", (), {"entry_time": "2026-01-01 00:00", "direction": "BEAR"})(),
            },
        )()

        result = executor._sota_overlay_account_lock_pre_open(action, engine)

        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "sota_overlay_skipped_open")
        self.assertEqual(result["reason"], "local_position_open")
        self.assertEqual(result["decision"]["paper_tag"], "stable_preempted_sota")


if __name__ == "__main__":
    unittest.main()
