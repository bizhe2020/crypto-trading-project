from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from bot.okx_executor import ExecutorConfig, OkxExecutionEngine
from strategy.scalp_robust_v2_core import Direction


class TelegramCommandTests(unittest.TestCase):
    def _engine(self) -> OkxExecutionEngine:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
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
                state_db_path=str(Path(tmp.name) / "state.db"),
                telegram_enabled=True,
                telegram_token="test-token",
                telegram_chat_id="123",
            )
        )

    def test_stop_and_start_toggle_open_pause(self) -> None:
        engine = self._engine()

        stop_reply = engine._telegram_command_reply("/stop")
        self.assertIn("已暂停新开仓", stop_reply)
        self.assertTrue(engine._telegram_open_paused())

        start_reply = engine._telegram_command_reply("/start")
        self.assertIn("已恢复开仓", start_reply)
        self.assertFalse(engine._telegram_open_paused())

    def test_help_and_status_reply(self) -> None:
        engine = self._engine()

        help_text = engine._telegram_command_reply("/help")
        self.assertIn("/balance", help_text)
        self.assertIn("/drift", help_text)
        self.assertIn("/ob", help_text)
        status = engine._telegram_command_reply("/status")
        self.assertIn("📡 状态雷达", status)
        self.assertIn("BTC/USDT:USDT", status)

    def test_status_includes_exchange_bracket_prices(self) -> None:
        engine = self._engine()
        engine.config.mode = "live"
        engine._fetch_position_state = lambda pos_side: (  # type: ignore[method-assign]
            {"contracts": 1.0, "notional_usdt": 1000.0}
            if pos_side == "long"
            else {"contracts": 0.0, "notional_usdt": 0.0}
        )
        engine._select_pending_algo_order = lambda pos_side: {  # type: ignore[method-assign]
            "algoId": "algo-123",
            "slTriggerPx": "90000",
            "tpTriggerPx": "110000",
        }

        status = engine._telegram_command_reply("/status")

        self.assertIn("🏛️ 交易所仓位：🟢 long", status)
        self.assertIn("🛡️ 交易所止损：90000.0", status)
        self.assertIn("🎯 交易所止盈：110000.0", status)
        self.assertIn("🔐 保护单ID：algo-123", status)

        table = engine._telegram_command_reply("/status table")
        self.assertIn("🧾 状态面板", table)
        self.assertIn("📦 仓位\n🏛️ 交易所：🟢 long", table)
        self.assertIn("🛡️ 止损：90000.0", table)
        self.assertNotIn("|", table)

    def test_drift_aliases_reply_with_drift_report(self) -> None:
        engine = self._engine()
        engine._build_drift_report_message = lambda: "DRIFT_REPORT"  # type: ignore[method-assign]

        self.assertEqual(engine._telegram_command_reply("/drift@mybot"), "DRIFT_REPORT")
        self.assertEqual(engine._telegram_command_reply("/health"), "DRIFT_REPORT")
        self.assertEqual(engine._telegram_command_reply("/体检"), "DRIFT_REPORT")

    def test_ob_aliases_reply_with_ob_report(self) -> None:
        engine = self._engine()
        engine._build_ob_status_message = lambda: "OB_REPORT"  # type: ignore[method-assign]

        self.assertEqual(engine._telegram_command_reply("/ob"), "OB_REPORT")
        self.assertEqual(engine._telegram_command_reply("/状态"), "OB_REPORT")

    def test_ob_stronger_bear_break_must_be_lower_than_primary(self) -> None:
        engine = object.__new__(OkxExecutionEngine)
        strategy_engine = SimpleNamespace(
            c15m=[
                SimpleNamespace(c=120.0, h=121.0, l=119.0),
                SimpleNamespace(c=118.0, h=120.0, l=115.0),
                SimpleNamespace(c=94.0, h=96.0, l=90.0),
                SimpleNamespace(c=106.0, h=108.0, l=105.0),
                SimpleNamespace(c=102.0, h=110.0, l=100.0),
                SimpleNamespace(c=104.0, h=109.0, l=101.0),
            ],
            precomputed=SimpleNamespace(highs_15m=[], lows_15m=[1, 2, 3, 4]),
            _timestamp_for_idx=lambda idx: f"t{idx}",
        )

        reference = engine._structure_reference(strategy_engine, 5, Direction.BEAR)

        self.assertEqual(reference["primary"]["break_price"], 100.0)
        self.assertEqual(reference["primary"]["strong_break_price"], 90.0)

    def test_ob_stronger_break_omitted_when_not_more_extreme(self) -> None:
        engine = object.__new__(OkxExecutionEngine)
        strategy_engine = SimpleNamespace(
            c15m=[
                SimpleNamespace(c=120.0, h=121.0, l=119.0),
                SimpleNamespace(c=118.0, h=120.0, l=115.0),
                SimpleNamespace(c=111.0, h=113.0, l=110.0),
                SimpleNamespace(c=106.0, h=108.0, l=105.0),
                SimpleNamespace(c=102.0, h=110.0, l=100.0),
                SimpleNamespace(c=104.0, h=109.0, l=101.0),
            ],
            precomputed=SimpleNamespace(highs_15m=[], lows_15m=[1, 2, 3, 4]),
            _timestamp_for_idx=lambda idx: f"t{idx}",
        )

        reference = engine._structure_reference(strategy_engine, 5, Direction.BEAR)

        self.assertEqual(reference["primary"]["break_price"], 100.0)
        self.assertNotIn("strong_break_price", reference["primary"])

    def test_ob_stronger_bull_break_must_be_higher_than_primary(self) -> None:
        engine = object.__new__(OkxExecutionEngine)
        strategy_engine = SimpleNamespace(
            c15m=[
                SimpleNamespace(c=80.0, h=81.0, l=79.0),
                SimpleNamespace(c=92.0, h=95.0, l=90.0),
                SimpleNamespace(c=108.0, h=110.0, l=106.0),
                SimpleNamespace(c=96.0, h=98.0, l=94.0),
                SimpleNamespace(c=102.0, h=100.0, l=90.0),
                SimpleNamespace(c=99.0, h=101.0, l=97.0),
            ],
            precomputed=SimpleNamespace(highs_15m=[1, 2, 3, 4], lows_15m=[]),
            _timestamp_for_idx=lambda idx: f"t{idx}",
        )

        reference = engine._structure_reference(strategy_engine, 5, Direction.BULL)

        self.assertEqual(reference["primary"]["break_price"], 100.0)
        self.assertEqual(reference["primary"]["strong_break_price"], 110.0)

    def test_ob_stronger_bull_break_omitted_when_not_more_extreme(self) -> None:
        engine = object.__new__(OkxExecutionEngine)
        strategy_engine = SimpleNamespace(
            c15m=[
                SimpleNamespace(c=80.0, h=81.0, l=79.0),
                SimpleNamespace(c=92.0, h=95.0, l=90.0),
                SimpleNamespace(c=96.0, h=98.0, l=94.0),
                SimpleNamespace(c=99.0, h=99.0, l=95.0),
                SimpleNamespace(c=102.0, h=100.0, l=90.0),
                SimpleNamespace(c=99.0, h=101.0, l=97.0),
            ],
            precomputed=SimpleNamespace(highs_15m=[1, 2, 3, 4], lows_15m=[]),
            _timestamp_for_idx=lambda idx: f"t{idx}",
        )

        reference = engine._structure_reference(strategy_engine, 5, Direction.BULL)

        self.assertEqual(reference["primary"]["break_price"], 100.0)
        self.assertNotIn("strong_break_price", reference["primary"])


if __name__ == "__main__":
    unittest.main()
