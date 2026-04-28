from __future__ import annotations

import unittest
from types import MethodType

from bot.okx_executor import OkxExecutionEngine


class TelegramCommandTest(unittest.TestCase):
    def build_executor(self) -> tuple[OkxExecutionEngine, list[tuple[str, str | None]]]:
        sent: list[tuple[str, str | None]] = []
        executor = object.__new__(OkxExecutionEngine)
        executor.config = type(
            "Config",
            (),
            {
                "telegram_ob_status_interval_minutes": 60,
                "telegram_drift_report_interval_hours": 24,
            },
        )()
        executor._send_telegram = MethodType(lambda self, message, chat_id=None: sent.append((message, chat_id)), executor)
        executor._build_drift_report_message = MethodType(lambda self: "DRIFT_REPORT", executor)
        executor._build_ob_status_message = MethodType(lambda self: "OB_REPORT", executor)
        return executor, sent

    def test_drift_aliases_reply_with_drift_report(self) -> None:
        executor, sent = self.build_executor()

        self.assertTrue(executor._handle_telegram_command("/drift@mybot", chat_id="123"))
        self.assertTrue(executor._handle_telegram_command("/health", chat_id="123"))
        self.assertTrue(executor._handle_telegram_command("/体检", chat_id="123"))

        self.assertEqual(sent, [("DRIFT_REPORT", "123"), ("DRIFT_REPORT", "123"), ("DRIFT_REPORT", "123")])

    def test_ob_aliases_reply_with_ob_report(self) -> None:
        executor, sent = self.build_executor()

        self.assertTrue(executor._handle_telegram_command("/ob", chat_id="123"))
        self.assertTrue(executor._handle_telegram_command("/status", chat_id="123"))

        self.assertEqual(sent, [("OB_REPORT", "123"), ("OB_REPORT", "123")])

    def test_unknown_command_is_ignored(self) -> None:
        executor, sent = self.build_executor()

        self.assertFalse(executor._handle_telegram_command("/unknown", chat_id="123"))

        self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main()
