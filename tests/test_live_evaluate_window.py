from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import MethodType

from bot.okx_executor import OkxExecutionEngine
from bot.state_store import StateStore
from strategy.scalp_robust_v2_core import StrategySnapshot


class FakeEngine:
    def __init__(self) -> None:
        self.capital = 100.0
        self.position = None
        self.calls: list[tuple[int, int]] = []

    def evaluate_range(self, start_idx: int, end_idx: int) -> list:
        self.calls.append((start_idx, end_idx))
        return []

    def _timestamp_for_idx(self, idx: int) -> str:
        return f"t{idx}"

    def snapshot(self) -> StrategySnapshot:
        return StrategySnapshot(
            capital=self.capital,
            position=None,
            exit_reasons={},
            trade_count=0,
        )


class LiveEvaluateWindowTest(unittest.TestCase):
    def build_executor(self, *, start_idx: int, latest_closed_idx: int) -> tuple[OkxExecutionEngine, FakeEngine, StateStore]:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        store = StateStore(Path(tmpdir.name) / "state.db")
        store.set_value("last_processed_candle_time", f"t{start_idx - 1}")

        engine = FakeEngine()
        executor = object.__new__(OkxExecutionEngine)
        executor.store = store
        executor.config = type("Config", (), {"symbol": "BTC/USDT:USDT"})()
        executor.load_engine = MethodType(lambda self: (engine, start_idx), executor)
        executor._sync_live_capital = MethodType(lambda self, loaded: loaded.capital, executor)
        executor._latest_closed_index = MethodType(lambda self, loaded: latest_closed_idx, executor)
        executor._assert_live_state_synced = MethodType(lambda self, loaded, *, context: None, executor)
        return executor, engine, store

    def test_evaluate_latest_includes_latest_closed_candle(self) -> None:
        executor, engine, store = self.build_executor(start_idx=4, latest_closed_idx=5)

        status = executor.evaluate_latest()

        self.assertEqual(engine.calls, [(4, 6)])
        self.assertEqual(status["processed_candle_time"], "t5")
        self.assertEqual(store.get_value("last_processed_candle_time"), "t5")

    def test_single_candle_window_is_evaluated(self) -> None:
        executor, engine, store = self.build_executor(start_idx=5, latest_closed_idx=5)

        status = executor.evaluate_latest()

        self.assertEqual(status["status"], "ok")
        self.assertEqual(engine.calls, [(5, 6)])
        self.assertEqual(store.get_value("last_processed_candle_time"), "t5")


if __name__ == "__main__":
    unittest.main()
