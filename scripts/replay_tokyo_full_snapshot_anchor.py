#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.market_data import MarketDataBundle  # noqa: E402
from bot.okx_executor import ExecutorConfig, OkxExecutionEngine  # noqa: E402
from bot.state_store import StateStore  # noqa: E402
from scripts.live_drift_monitor import build_live_trades, load_action_log, parse_timestamp  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "tokyo_audit" / "tokyo_full_snapshot_anchor_replay_20260505.json"
DEFAULT_TOKYO_15M = ROOT / "var" / "tokyo_audit" / "BTC_USDT_USDT-15m-futures.remote-tail.feather"
DEFAULT_TOKYO_4H = ROOT / "var" / "tokyo_audit" / "BTC_USDT_USDT-4h-futures.remote-tail.feather"


class FixedOhlcvRepository:
    def __init__(self, data_15m_path: Path, data_4h_path: Path) -> None:
        self.data_15m_path = data_15m_path
        self.data_4h_path = data_4h_path
        self.as_of_candle_time: str | None = None

    def load_pair(
        self,
        pair: str = "BTC/USDT:USDT",
        client: Any | None = None,
        timeframe: str = "15m",
        informative_timeframe: str = "4h",
    ) -> MarketDataBundle:
        if timeframe != "15m" or informative_timeframe != "4h":
            raise ValueError("Tokyo snapshot replay only supports 15m primary + 4h informative data")
        primary = self._read(self.data_15m_path)
        informative = self._as_of_4h(self._read(self.data_4h_path), primary)
        return MarketDataBundle(
            primary_candles=primary,
            informative_candles=informative,
            primary_timeframe=timeframe,
            informative_timeframe=informative_timeframe,
            candles_15m=primary,
            candles_4h=informative,
        )

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        df = pd.read_feather(path)
        if df.empty:
            raise ValueError(f"OHLCV file is empty: {path}")
        df = df.copy()
        df.columns = [str(column).lower() for column in df.columns]
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df.sort_values("date").reset_index(drop=True)

    def _as_of_4h(self, informative: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
        if not self.as_of_candle_time:
            return informative
        as_of = parse_time(self.as_of_candle_time)
        bucket_hour = (as_of.hour // 4) * 4
        bucket_start = as_of.replace(hour=bucket_hour, minute=0, second=0, microsecond=0)
        partial = primary[(primary["date"] >= bucket_start) & (primary["date"] <= as_of)]
        if partial.empty:
            return informative[informative["date"] <= bucket_start].reset_index(drop=True)
        replacement = dict(partial.iloc[0])
        replacement["date"] = bucket_start
        replacement["open"] = float(partial.iloc[0]["open"])
        replacement["high"] = float(partial["high"].max())
        replacement["low"] = float(partial["low"].min())
        replacement["close"] = float(partial.iloc[-1]["close"])
        replacement["volume"] = float(partial["volume"].sum()) if "volume" in partial.columns else 0.0
        history = informative[informative["date"] < bucket_start]
        current = pd.DataFrame([replacement])
        return pd.concat([history, current], ignore_index=True).reset_index(drop=True)


class DummyExchange:
    def amount_to_precision(self, symbol: str, amount: float) -> str:
        return f"{float(amount):.8f}"

    def price_to_precision(self, symbol: str, price: float) -> str:
        return f"{float(price):.1f}"


class DummyClient:
    def __init__(self, available_usdt: float) -> None:
        self.available_usdt = float(available_usdt)
        self.exchange = DummyExchange()

    def load_markets(self) -> dict[str, Any]:
        return {"BTC/USDT:USDT": {"contract": True, "contractSize": 0.01}}

    def fetch_balance(self) -> dict[str, Any]:
        return {"USDT": {"free": self.available_usdt}}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> list[list[float]]:
        return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Tokyo live DB from a full snapshot anchor in fixed/legacy resume modes.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.high-leverage-structure.template.json"))
    parser.add_argument("--live-db", default=str(ROOT / "var" / "tokyo_audit" / "runtime_high_leverage_structure_live.db"))
    parser.add_argument("--data-15m", default=str(DEFAULT_TOKYO_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_TOKYO_4H))
    parser.add_argument("--anchor-row-id", type=int, default=1641)
    parser.add_argument("--until-time", default="2026-05-04 10:30")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def parse_time(value: Any) -> datetime:
    parsed = parse_timestamp(value)
    if parsed is None:
        raise ValueError(f"Invalid timestamp: {value}")
    return parsed


def load_action_rows(db_path: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, timestamp, action_type, payload, created_at
            FROM action_log
            ORDER BY id ASC
            """
        ).fetchall()
    output: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            payload = {}
        output.append(
            {
                "id": int(row["id"]),
                "timestamp": row["timestamp"],
                "action_type": row["action_type"],
                "payload": payload if isinstance(payload, dict) else {},
                "created_at": row["created_at"],
            }
        )
    return output


def anchor_row(rows: list[dict[str, Any]], row_id: int) -> dict[str, Any]:
    for row in rows:
        if int(row["id"]) == int(row_id):
            return row
    raise ValueError(f"Anchor row not found: {row_id}")


def previous_processed_candle(rows: list[dict[str, Any]], anchor: dict[str, Any]) -> str:
    anchor_created = parse_time(anchor["created_at"] or anchor["timestamp"])
    candidates: list[tuple[datetime, str]] = []
    for row in rows:
        if row["action_type"] not in {"EVALUATE", "INITIALIZE"}:
            continue
        created = parse_timestamp(row.get("created_at"))
        if created is None or created > anchor_created:
            continue
        payload = row["payload"]
        processed = payload.get("processed_candle_time") or row.get("timestamp")
        if processed:
            candidates.append((created, str(processed)))
    if not candidates:
        raise ValueError("No processed candle before anchor")
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def live_evaluate_timestamps(rows: list[dict[str, Any]], anchor: dict[str, Any], until_time: datetime) -> list[str]:
    anchor_created = parse_time(anchor["created_at"] or anchor["timestamp"])
    output: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row["action_type"] != "EVALUATE":
            continue
        created = parse_timestamp(row.get("created_at"))
        if created is None or created <= anchor_created:
            continue
        processed = row["payload"].get("processed_candle_time") or row.get("timestamp")
        processed_dt = parse_timestamp(processed)
        if processed_dt is None or processed_dt > until_time:
            continue
        key = str(processed)
        if key not in seen:
            output.append(key)
            seen.add(key)
    return output


def build_shadow_state_from_trades(trades: list[Any], anchor_dt: datetime, capital: float) -> dict[str, Any]:
    state = {
        "mode": "shadow_risk_gate",
        "capital": capital,
        "drawdown_peak": capital,
        "pause_until_ts": 0.0,
        "real_position_open": False,
        "real_position_direction": None,
        "paper_entry_time": None,
        "day_start_capital": {},
        "day_pnl": {},
        "loss_streak": 0,
        "events": [],
    }
    running_capital = None
    peak = 0.0
    for trade in trades:
        if trade.exit_time > anchor_dt:
            continue
        before = float(trade.capital_at_entry or 0.0)
        pnl = float(trade.net_pnl)
        after = before + pnl
        running_capital = after
        peak = max(peak, after)
        day_key = trade.exit_time.strftime("%Y-%m-%d")
        state["day_start_capital"].setdefault(day_key, before)
        state["day_pnl"][day_key] = float(state["day_pnl"].get(day_key, 0.0) or 0.0) + pnl
        state["loss_streak"] = 0 if pnl > 0 else int(state["loss_streak"]) + 1
        state["events"].append(
            {
                "time": trade.exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "event": "mirror_close",
                "direction": trade.direction,
                "pnl": pnl,
                "capital": after,
                "triggers": [],
                "pause_until": "",
            }
        )
    if running_capital is not None:
        state["capital"] = running_capital
        state["drawdown_peak"] = max(peak, running_capital)
    return state


def build_dynamic_state_from_trades(trades: list[Any], anchor_dt: datetime, capital: float) -> dict[str, Any]:
    state = {
        "mode": "offense",
        "capital": capital,
        "drawdown_peak": capital,
        "unit_returns": [],
        "loss_streak": 0,
        "win_streak": 0,
        "last_update_time": None,
        "last_decision": None,
    }
    for trade in trades:
        if trade.exit_time > anchor_dt:
            continue
        pnl = float(trade.net_pnl)
        notional = float(trade.notional or 0.0)
        unit_return = pnl / notional if notional > 0 else 0.0
        state["unit_returns"].append(unit_return)
        if pnl > 0:
            state["win_streak"] = int(state["win_streak"]) + 1
            state["loss_streak"] = 0
        else:
            state["loss_streak"] = int(state["loss_streak"]) + 1
            state["win_streak"] = 0
        state["capital"] = float(trade.capital_at_entry or 0.0) + pnl
        state["drawdown_peak"] = max(float(state["drawdown_peak"] or 0.0), float(state["capital"] or 0.0))
        state["last_update_time"] = trade.exit_time.strftime("%Y-%m-%d %H:%M:%S")
        state["last_close"] = {
            "time": state["last_update_time"],
            "pnl": pnl,
            "notional": notional,
            "unit_return": unit_return,
            "capital": state["capital"],
        }
    state["unit_returns"] = state["unit_returns"][-100:]
    return state


def find_candle_index(executor: OkxExecutionEngine, engine: Any, timestamp: str) -> int:
    for idx, candle in enumerate(executor._engine_candles(engine)):
        if executor._timestamp_from_ts(candle.ts) == timestamp:
            return idx
    raise ValueError(f"Candle not found: {timestamp}")


def build_executor(
    config_path: Path,
    tmp_state_db: Path,
    available_usdt: float,
    data_15m_path: Path,
    data_4h_path: Path,
) -> OkxExecutionEngine:
    payload = json.loads(config_path.read_text())
    payload["mode"] = "paper"
    payload["state_db_path"] = str(tmp_state_db)
    payload["telegram_enabled"] = False
    payload["telegram_command_enabled"] = False
    payload["enable_exchange_brackets"] = False
    payload["api_key"] = None
    payload["api_secret"] = None
    payload["api_passphrase"] = None
    payload["enable_live_candidate_arbitration"] = False
    config = ExecutorConfig.from_dict(payload)
    executor = object.__new__(OkxExecutionEngine)
    executor.config = config
    executor.config_path = config_path.resolve()
    executor.client = DummyClient(available_usdt)
    executor.store = StateStore(tmp_state_db)
    executor.market_data = FixedOhlcvRepository(data_15m_path, data_4h_path)
    executor._markets_cache = None
    executor._sync_live_capital = MethodType(lambda self, loaded: float(getattr(loaded, "capital", 0.0) or 0.0), executor)
    executor._assert_live_state_synced = MethodType(
        lambda self, loaded, *, context, timestamp=None, exit_idx=None: None,
        executor,
    )
    return executor


def apply_legacy_resume(executor: OkxExecutionEngine) -> None:
    def legacy_find_resume_index(self: OkxExecutionEngine, candles: list[Any]) -> int:
        last_processed = self.store.get_value("last_processed_candle_time")
        min_start = self._minimum_start_index()
        if not last_processed:
            return min_start
        for idx, candle in enumerate(candles):
            candle_time = self._timestamp_from_ts(candle.ts)
            if candle_time > last_processed:
                return max(min_start, idx - 1)
        return max(min_start, len(candles) - 1)

    executor._find_resume_index = MethodType(legacy_find_resume_index, executor)


def run_mode(
    mode: str,
    config_path: Path,
    data_15m_path: Path,
    data_4h_path: Path,
    anchor_snapshot: dict[str, Any],
    prev_processed: str,
    target_timestamps: list[str],
    shadow_state: dict[str, Any],
    dynamic_state: dict[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_state_db = Path(tmpdir) / f"{mode}.db"
        executor = build_executor(
            config_path,
            tmp_state_db,
            float(anchor_snapshot.get("capital", 0.0) or 0.0),
            data_15m_path,
            data_4h_path,
        )
        if mode == "legacy":
            apply_legacy_resume(executor)
        executor.store.set_value("strategy_snapshot", json.dumps(anchor_snapshot, ensure_ascii=False))
        executor.store.set_value("last_processed_candle_time", prev_processed)
        executor.store.set_value("shadow_risk_gate_state", json.dumps(shadow_state, ensure_ascii=False))
        executor.store.set_value("dynamic_high_leverage_structure_state", json.dumps(dynamic_state, ensure_ascii=False))

        loaded_engine, _ = executor.load_engine()
        target_indices = {timestamp: find_candle_index(executor, loaded_engine, timestamp) for timestamp in target_timestamps}
        statuses: list[dict[str, Any]] = []
        for timestamp in target_timestamps:
            latest_idx = target_indices[timestamp]
            executor.market_data.as_of_candle_time = timestamp
            executor._latest_closed_index = MethodType(lambda self, loaded, idx=latest_idx: idx, executor)
            status = executor.evaluate_latest()
            statuses.append(status)

        action_rows = load_action_rows(tmp_state_db)
        opens = [row for row in action_rows if row["action_type"] in {"OPEN_LONG", "OPEN_SHORT"}]
        evaluates = [row for row in action_rows if row["action_type"] == "EVALUATE"]
        snapshots = [
            {
                "processed_candle_time": row["payload"].get("processed_candle_time"),
                "actions": [
                    {
                        "type": action.get("type"),
                        "timestamp": action.get("timestamp"),
                        "direction": action.get("direction"),
                        "entry_price": action.get("entry_price"),
                        "metadata_index": (action.get("metadata") or {}).get("index") if isinstance(action, dict) else None,
                    }
                    for action in row["payload"].get("actions", [])
                    if isinstance(action, dict)
                ],
                "pending_pullback_state": (row["payload"].get("snapshot") or {}).get("pending_pullback_state"),
                "position_open": row["payload"].get("position_open"),
            }
            for row in evaluates
        ]
        return {
            "mode": mode,
            "target_timestamps": target_timestamps,
            "open_actions": [
                {
                    "timestamp": row["timestamp"],
                    "type": row["action_type"],
                    "created_at": row["created_at"],
                    "entry_price": row["payload"].get("entry_price"),
                    "metadata_index": (row["payload"].get("metadata") or {}).get("index"),
                }
                for row in opens
            ],
            "evaluations": snapshots,
            "final_last_processed": StateStore(tmp_state_db).get_value("last_processed_candle_time"),
            "final_snapshot": StateStore(tmp_state_db).load_snapshot(),
        }


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    live_db = Path(args.live_db)
    data_15m_path = Path(args.data_15m)
    data_4h_path = Path(args.data_4h)
    rows = load_action_rows(live_db)
    anchor = anchor_row(rows, int(args.anchor_row_id))
    anchor_snapshot = anchor["payload"].get("snapshot")
    if not isinstance(anchor_snapshot, dict):
        raise ValueError(f"Anchor row has no snapshot: {args.anchor_row_id}")

    anchor_dt = parse_time(anchor["timestamp"])
    until_dt = parse_time(args.until_time)
    trades, trade_diagnostics = build_live_trades(load_action_log(live_db))
    prev_processed = previous_processed_candle(rows, anchor)
    target_timestamps = live_evaluate_timestamps(rows, anchor, until_dt)
    shadow_state = build_shadow_state_from_trades(trades, anchor_dt, float(anchor_snapshot.get("capital", 0.0) or 0.0))
    dynamic_state = build_dynamic_state_from_trades(trades, anchor_dt, float(anchor_snapshot.get("capital", 0.0) or 0.0))

    fixed = run_mode(
        "fixed",
        config_path,
        data_15m_path,
        data_4h_path,
        anchor_snapshot,
        prev_processed,
        target_timestamps,
        shadow_state,
        dynamic_state,
    )
    legacy = run_mode(
        "legacy",
        config_path,
        data_15m_path,
        data_4h_path,
        anchor_snapshot,
        prev_processed,
        target_timestamps,
        shadow_state,
        dynamic_state,
    )

    report = {
        "inputs": {
            "config": str(config_path.resolve()),
            "live_db": str(live_db.resolve()),
            "data_15m": str(data_15m_path.resolve()),
            "data_4h": str(data_4h_path.resolve()),
            "anchor_row_id": int(args.anchor_row_id),
            "anchor_timestamp": anchor["timestamp"],
            "anchor_created_at": anchor["created_at"],
            "until_time": args.until_time,
            "previous_processed_candle": prev_processed,
            "target_timestamps": target_timestamps,
        },
        "anchor_state": {
            "strategy_snapshot": anchor_snapshot,
            "shadow_risk_gate_state": shadow_state,
            "dynamic_high_leverage_structure_state": dynamic_state,
            "trade_diagnostics": trade_diagnostics,
        },
        "fixed": fixed,
        "legacy": legacy,
        "summary": {
            "fixed_open_count": len(fixed["open_actions"]),
            "legacy_open_count": len(legacy["open_actions"]),
            "fixed_open_actions": fixed["open_actions"],
            "legacy_open_actions": legacy["open_actions"],
            "legacy_reproduces_0915_reopen": any(
                row.get("timestamp") == "2026-05-04 09:15" for row in legacy["open_actions"]
            ),
            "fixed_blocks_0915_reopen": not any(
                row.get("timestamp") == "2026-05-04 09:15" for row in fixed["open_actions"]
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"JSON written: {output}")


if __name__ == "__main__":
    main()
