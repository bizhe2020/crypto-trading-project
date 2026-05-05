from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.audit_live_replay_trade_convergence import load_live_trades, load_replay_events, match_live_trade, parse_time


def insert_action(conn: sqlite3.Connection, timestamp: str, action_type: str, payload: dict) -> None:
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


if __name__ == "__main__":
    unittest.main()
