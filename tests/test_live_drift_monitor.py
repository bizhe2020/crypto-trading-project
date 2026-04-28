from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.live_drift_monitor import build_live_trades, load_action_log, trade_metrics


def insert_action(conn: sqlite3.Connection, timestamp: str, action_type: str, payload: dict) -> None:
    conn.execute(
        "INSERT INTO action_log(timestamp, action_type, payload) VALUES(?, ?, ?)",
        (timestamp, action_type, json.dumps(payload)),
    )


class LiveDriftMonitorTest(unittest.TestCase):
    def build_db(self) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        db_path = Path(tmpdir.name) / "state.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE action_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            insert_action(
                conn,
                "2026-04-01 00:00",
                "OPEN_LONG",
                {
                    "type": "OPEN_LONG",
                    "timestamp": "2026-04-01 00:00",
                    "direction": "BULL",
                    "entry_price": 10005.0,
                    "stop_price": 9800.0,
                    "target_price": 10600.0,
                    "metadata": {
                        "signal_entry_price": 10000.0,
                        "capital_at_entry": 1000.0,
                        "notional": 3000.0,
                        "risk_amount": 30.0,
                    },
                },
            )
            insert_action(
                conn,
                "2026-04-01 04:00",
                "UPDATE_STOP",
                {
                    "type": "UPDATE_STOP",
                    "timestamp": "2026-04-01 04:00",
                    "stop_price": 10050.0,
                },
            )
            insert_action(
                conn,
                "2026-04-01 06:00",
                "CLOSE_POSITION",
                {
                    "type": "CLOSE_POSITION",
                    "timestamp": "2026-04-01 06:00",
                    "direction": "BULL",
                    "exit_price": 10605.0,
                    "reason": "target_rr",
                    "metadata": {
                        "signal_exit_price": 10600.0,
                        "net_pnl": 60.0,
                    },
                },
            )
        return db_path

    def test_build_live_trades_pairs_open_update_and_close(self) -> None:
        actions = load_action_log(self.build_db())

        trades, diagnostics = build_live_trades(actions)

        self.assertEqual(len(trades), 1)
        self.assertEqual(diagnostics["orphan_closes"], 0)
        self.assertAlmostEqual(trades[0].pnl_pct or 0.0, 0.06)
        self.assertAlmostEqual(trades[0].entry_slippage_bps or 0.0, 5.0)
        self.assertAlmostEqual(trades[0].exit_slippage_bps or 0.0, 4.716981132075472)
        self.assertAlmostEqual(trades[0].stop_target_deviation_bps or 0.0, 4.716981132075472)

    def test_trade_metrics_uses_account_return_distribution(self) -> None:
        trades, _ = build_live_trades(load_action_log(self.build_db()))

        metrics = trade_metrics(trades, window_days=30)

        self.assertEqual(metrics["trade_count"], 1)
        self.assertEqual(metrics["win_rate_pct"], 100.0)
        self.assertEqual(metrics["avg_win_pct"], 6.0)
        self.assertEqual(metrics["expectancy_pct"], 6.0)


if __name__ == "__main__":
    unittest.main()
