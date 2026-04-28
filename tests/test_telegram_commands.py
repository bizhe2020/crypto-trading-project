from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bot.okx_executor import ExecutorConfig, OkxExecutionEngine


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


if __name__ == "__main__":
    unittest.main()
