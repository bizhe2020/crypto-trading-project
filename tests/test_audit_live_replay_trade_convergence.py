from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.audit_live_replay_trade_convergence import load_live_trades, load_replay_events, match_live_trade, parse_time


def insert_action(
    conn: sqlite3.Connection,
    timestamp: str,
    action_type: str,
    payload: dict,
    *,
    created_at: str | None = None,
) -> None:
    if created_at is not None:
        conn.execute(
            "INSERT INTO action_log(timestamp, action_type, payload, created_at) VALUES(?, ?, ?, ?)",
            (timestamp, action_type, json.dumps(payload), created_at),
        )
        return
    conn.execute(
        "INSERT INTO action_log(timestamp, action_type, payload) VALUES(?, ?, ?)",
        (timestamp, action_type, json.dumps(payload)),
    )


class AuditLiveReplayTradeConvergenceTest(unittest.TestCase):
    def build_live_db(self) -> Path:
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
                    "entry_price": 10050.0,
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
                "2026-04-01 00:15",
                "MANUAL_POSITION_SYNC",
                {
                    "context": "after_execute",
                    "snapshot": {
                        "position": {
                            "direction": "BULL",
                            "entry_time": "2026-04-01 00:00",
                            "signal_entry_price": 10000.0,
                            "entry_price": 10000.0,
                            "sl_price": 9800.0,
                            "target_price": 10600.0,
                            "capital_at_entry": 1000.0,
                            "notional": 2000.0,
                            "risk_amount": 40.0,
                        }
                    },
                },
            )
            insert_action(
                conn,
                "2026-04-01 01:00",
                "CLOSE_POSITION",
                {
                    "type": "CLOSE_POSITION",
                    "timestamp": "2026-04-01 01:00",
                    "direction": "BULL",
                    "exit_price": 10600.0,
                    "reason": "target_rr",
                    "metadata": {"signal_exit_price": 10600.0, "net_pnl": 80.0},
                },
            )
        return db_path

    def build_replay_json(self) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "replay.json"
        payload = {
            "live_shadow": {
                "events": [
                    {
                        "event_type": "sota_long",
                        "direction": "BULL",
                        "entry_time": "2026-04-01T00:00:00+00:00",
                        "exit_time": "2026-04-01T01:00:00+00:00",
                        "entry_idx": 1,
                        "exit_idx": 2,
                        "return_pct": 8.0,
                    }
                ]
            }
        }
        path.write_text(json.dumps(payload, ensure_ascii=False))
        return path

    def test_match_live_trade_tracks_signal_and_execution_time_separately(self) -> None:
        live_rows, diagnostics = load_live_trades(self.build_live_db(), None)
        replay_events = load_replay_events(self.build_replay_json(), "live_shadow")

        self.assertEqual(diagnostics["orphan_closes"], 0)
        self.assertEqual(len(live_rows), 1)

        live = live_rows[0]
        match = match_live_trade(live, replay_events, 3)

        self.assertEqual(live["signal_entry_time"], "2026-04-01T00:00:00+00:00")
        self.assertEqual(live["entry_execution_time"], "2026-04-01T00:15:00+00:00")
        self.assertEqual(live["entry_execution_delay_seconds"], 900.0)
        self.assertEqual(live["entry_execution_delay_bars"], 1.0)
        self.assertEqual(match["status"], "exact_entry_match")
        self.assertEqual(match["signal_entry_gap_seconds"], 0.0)
        self.assertEqual(match["execution_entry_gap_seconds"], 900.0)
        self.assertEqual(parse_time(live["entry_execution_time"]).isoformat(), "2026-04-01T00:15:00+00:00")

    def test_load_live_trades_keeps_exchange_fill_sync_negative_pnl(self) -> None:
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
                "2026-05-19 10:45:00",
                "OPEN_SHORT",
                {
                    "type": "OPEN_SHORT",
                    "timestamp": "2026-05-19 10:45:00",
                    "direction": "BEAR",
                    "entry_price": 77006.5,
                    "stop_price": 77297.340263,
                    "target_price": 76309.3097,
                    "metadata": {
                        "signal_entry_price": 77006.5,
                        "capital_at_entry": 1000.0,
                        "notional": 71470.52,
                        "risk_amount": 100.0,
                    },
                },
            )
            insert_action(
                conn,
                "2026-05-19 11:01:17",
                "MANUAL_POSITION_SYNC",
                {
                    "context": "after_execute",
                    "snapshot": {
                        "position": {
                            "direction": "BEAR",
                            "entry_time": "2026-05-19 11:01:17",
                            "signal_entry_price": 77006.5,
                            "entry_price": 77026.96357273609,
                            "sl_price": 76959.37946592011,
                            "target_price": 76309.3097,
                            "capital_at_entry": 1000.0,
                            "notional": 71470.52,
                            "risk_amount": 100.0,
                        }
                    },
                },
            )
            insert_action(
                conn,
                "2026-05-19 12:55:07",
                "CLOSE_POSITION",
                {
                    "type": "CLOSE_POSITION",
                    "timestamp": "2026-05-19 12:55:07",
                    "direction": "BEAR",
                    "exit_price": 76957.6,
                    "reason": "external_stop_loss",
                    "metadata": {
                        "source": "exchange_fill_sync",
                        "synthetic": False,
                        "signal_exit_price": 76957.6,
                        "net_pnl": -7.084782095,
                    },
                },
                created_at="2026-05-19 15:22:58",
            )

        live_rows, diagnostics = load_live_trades(db_path, None)

        self.assertEqual(diagnostics["orphan_closes"], 0)
        self.assertEqual(len(live_rows), 1)
        self.assertAlmostEqual(live_rows[0]["net_pnl"], -7.084782, places=6)
        self.assertAlmostEqual(live_rows[0]["pnl_pct"], -0.7085, places=4)
        self.assertEqual(live_rows[0]["signal_exit_time"], "2026-05-19T12:55:07+00:00")
        self.assertEqual(live_rows[0]["exit_execution_time"], "2026-05-19T15:22:58+00:00")


if __name__ == "__main__":
    unittest.main()
