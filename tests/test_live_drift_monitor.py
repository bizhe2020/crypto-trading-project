from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.live_drift_monitor import build_live_trades, format_report, load_action_log, trade_metrics


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

    def test_build_live_trades_uses_manual_sync_snapshot_as_execution_anchor(self) -> None:
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
                            "capital_at_entry": 990.0,
                            "notional": 2000.0,
                            "risk_amount": 40.0,
                        }
                    },
                },
            )
            insert_action(
                conn,
                "2026-04-01 03:00",
                "CLOSE_POSITION",
                {
                    "type": "CLOSE_POSITION",
                    "timestamp": "2026-04-01 03:00",
                    "direction": "BULL",
                    "exit_price": 10600.0,
                    "reason": "target_rr",
                        "metadata": {"signal_exit_price": 10600.0, "net_pnl": 80.0},
                },
                created_at="2026-04-01 03:00:00",
            )

        trades, diagnostics = build_live_trades(load_action_log(db_path))

        self.assertEqual(len(trades), 1)
        self.assertEqual(diagnostics["orphan_closes"], 0)
        self.assertAlmostEqual(trades[0].entry_price or 0.0, 10000.0)
        self.assertAlmostEqual(trades[0].capital_at_entry or 0.0, 990.0)
        self.assertAlmostEqual(trades[0].notional or 0.0, 2000.0)
        self.assertAlmostEqual(trades[0].risk_amount or 0.0, 40.0)
        self.assertAlmostEqual(trades[0].entry_slippage_bps or 0.0, 0.0)
        self.assertEqual(trades[0].entry_time.isoformat(), "2026-04-01T00:00:00+00:00")
        self.assertEqual(trades[0].entry_execution_time.isoformat(), "2026-04-01T00:15:00+00:00")
        self.assertEqual(trades[0].exit_execution_time.isoformat(), "2026-04-01T03:00:00+00:00")

    def test_trade_metrics_uses_account_return_distribution(self) -> None:
        trades, _ = build_live_trades(load_action_log(self.build_db()))

        metrics = trade_metrics(trades, window_days=30)

        self.assertEqual(metrics["trade_count"], 1)
        self.assertEqual(metrics["win_rate_pct"], 100.0)
        self.assertEqual(metrics["avg_win_pct"], 6.0)
        self.assertEqual(metrics["expectancy_pct"], 6.0)

    def test_format_report_includes_health_conclusion_and_capital_advice(self) -> None:
        report = {
            "status": "WATCH",
            "window": {"window_days": 30, "recent_trades_floor": 20},
            "recent": {
                "trade_count": 3,
                "total_return_pct": 2.0,
                "win_rate_pct": 66.7,
                "profit_factor": 1.8,
                "payoff_ratio": 1.4,
                "expectancy_pct": 0.7,
                "trades_per_month": 4.2,
                "trades_per_year": 50.4,
                "avg_entry_slippage_bps": 4.0,
                "avg_exit_slippage_bps": 5.0,
                "avg_stop_target_deviation_bps": None,
                "stop_target_reference_count": 0,
            },
            "all": {"trade_count": 3, "total_return_pct": 2.0},
            "baseline_comparison": {"trade_frequency_ratio": 0.77},
            "flags": [{"level": "watch", "message": "Small sample: 3 closed trades."}],
        }

        message = format_report(report)

        self.assertIn("🧭 体检结论", message)
        self.assertIn("💰 启动资金建议", message)
        self.assertIn("当前约 10,000U = 计划资金 20%", message)
        self.assertIn("暂时维持 20%", message)


if __name__ == "__main__":
    unittest.main()
