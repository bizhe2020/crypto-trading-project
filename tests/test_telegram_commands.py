from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

        self.assertIn("/balance", engine._telegram_command_reply("/help"))
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

    def test_sticker_ids_are_selected_by_mood(self) -> None:
        engine = self._engine()
        engine.config.telegram_profit_sticker_ids = ["profit-a", "profit-b"]
        engine.config.telegram_loss_sticker_ids = ["loss-a"]
        engine.config.telegram_neutral_sticker_ids = ["neutral-a"]

        self.assertEqual(engine._telegram_sticker_ids_for_mood("profit"), ["profit-a", "profit-b"])
        self.assertEqual(engine._telegram_sticker_ids_for_mood("loss"), ["loss-a"])
        self.assertEqual(engine._telegram_sticker_ids_for_mood("neutral"), ["neutral-a"])

    def test_send_sticker_uses_configured_pool(self) -> None:
        engine = self._engine()
        engine.config.telegram_profit_sticker_ids = ["profit-a"]

        with patch("bot.okx_executor.requests.post") as post:
            engine._send_telegram_sticker("profit", "123")

        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.kwargs["json"]["sticker"], "profit-a")


if __name__ == "__main__":
    unittest.main()
