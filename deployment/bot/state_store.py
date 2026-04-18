from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from deployment.strategy.scalp_robust_v2_core import StrategySnapshot


class StateStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS action_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def save_snapshot(self, snapshot: StrategySnapshot) -> None:
        payload = json.dumps(asdict(snapshot), ensure_ascii=False)
        self.set_value("strategy_snapshot", payload)

    def load_snapshot(self) -> dict[str, Any] | None:
        value = self.get_value("strategy_snapshot")
        return json.loads(value) if value else None

    def set_value(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO bot_state(key, value, updated_at)
                VALUES(?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (key, value),
            )

    def get_value(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM bot_state WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def append_action(self, timestamp: str, action_type: str, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO action_log(timestamp, action_type, payload) VALUES(?, ?, ?)",
                (timestamp, action_type, json.dumps(payload, ensure_ascii=False)),
            )
