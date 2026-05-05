#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine  # noqa: E402


DEFAULT_CONFIG = ROOT / "var" / "tokyo_audit" / "config.live.high-leverage-structure.json"
DEFAULT_LIVE_DB = ROOT / "var" / "tokyo_audit" / "runtime_high_leverage_structure_live.db"
DEFAULT_15M = ROOT / "var" / "tokyo_audit" / "BTC_USDT_USDT-15m-futures.remote-tail.feather"
DEFAULT_4H = ROOT / "var" / "tokyo_audit" / "BTC_USDT_USDT-4h-futures.remote-tail.feather"
DEFAULT_OUTPUT = ROOT / "var" / "tokyo_audit" / "tokyo_replay_stop_gap_20260505.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Tokyo live/replay stop gaps around candle close and entry sync.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--live-db", default=str(DEFAULT_LIVE_DB))
    parser.add_argument("--data-15m", default=str(DEFAULT_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_4H))
    parser.add_argument("--start-date", default="2026-04-15 16:00")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def load_live_actions(db_path: Path) -> list[dict[str, Any]]:
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
            payload = json.loads(row["payload"] or "{}")
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


def live_open_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    opens = []
    for row in rows:
        if row["action_type"] not in {"OPEN_LONG", "OPEN_SHORT"}:
            continue
        payload = row["payload"]
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        opens.append(
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "created_at": row["created_at"],
                "direction": payload.get("direction"),
                "entry_price": payload.get("entry_price"),
                "stop_price": payload.get("stop_price"),
                "target_price": payload.get("target_price"),
                "metadata_index": metadata.get("index"),
                "signal_entry_price": metadata.get("signal_entry_price"),
                "regime_label": metadata.get("regime_label"),
            }
        )
    return opens


def live_stop_updates(rows: list[dict[str, Any]], entry_idx: int) -> list[dict[str, Any]]:
    updates = []
    for row in rows:
        if row["action_type"] != "EVALUATE":
            continue
        for action in row["payload"].get("actions", []) or []:
            if not isinstance(action, dict) or action.get("type") != "UPDATE_STOP":
                continue
            metadata = action.get("metadata") if isinstance(action.get("metadata"), dict) else {}
            idx = int(metadata.get("index", -1) or -1)
            if idx < entry_idx:
                continue
            updates.append(
                {
                    "row_id": row["id"],
                    "processed_candle_time": row["payload"].get("processed_candle_time") or row["timestamp"],
                    "created_at": row["created_at"],
                    "timestamp": action.get("timestamp"),
                    "reason": action.get("reason"),
                    "stop_price": action.get("stop_price"),
                    "target_price": action.get("target_price"),
                    "metadata": metadata,
                }
            )
    return updates


def replay_variant(
    payload: dict[str, Any],
    args: argparse.Namespace,
    *,
    slippage_bps: float,
    replay_sync_entry_to_signal_price: bool,
) -> dict[str, Any]:
    variant_payload = dict(payload)
    variant_payload["slippage_bps"] = float(slippage_bps)
    variant_payload["replay_sync_entry_to_signal_price"] = bool(replay_sync_entry_to_signal_price)
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=variant_payload.get("regime_switcher_thresholds"),
        informative_asof_from_15m=True,
    )
    metrics, engine = run_engine(variant_payload, prepared, args.start_date)
    trades = []
    for trade in engine.trades:
        if trade.entry_time >= "2026-04-29":
            trades.append(
                {
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "direction": trade.direction,
                    "entry_idx": trade.entry_idx,
                    "exit_idx": trade.exit_idx,
                    "entry_price": trade.entry_price,
                    "initial_stop_price": trade.initial_stop_price,
                    "final_stop_price": trade.final_stop_price,
                    "exit_price": trade.exit_price,
                    "exit_reason": trade.exit_reason,
                    "return_pct": round(float(trade.pnl_pct or 0.0) * 100.0, 4),
                    "last_stop_update_reason": trade.last_stop_update_reason,
                    "last_stop_update_idx": trade.last_stop_update_idx,
                }
            )
    return {
        "slippage_bps": slippage_bps,
        "replay_sync_entry_to_signal_price": replay_sync_entry_to_signal_price,
        "metrics": {
            "total_trades": metrics.get("total_trades"),
            "total_return_pct": round(float(metrics.get("total_return_pct", 0.0) or 0.0), 4),
            "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0) or 0.0), 4),
            "exit_reasons": metrics.get("exit_reasons", {}),
        },
        "recent_trades": trades,
    }


def candle_row(data_15m: Path, timestamp: str) -> dict[str, Any]:
    df = pd.read_feather(data_15m)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    target = pd.Timestamp(timestamp, tz="UTC")
    row = df[df["date"] == target]
    if row.empty:
        return {}
    item = row.iloc[0]
    return {
        "date": str(item["date"]),
        "open": float(item["open"]),
        "high": float(item["high"]),
        "low": float(item["low"]),
        "close": float(item["close"]),
    }


def main() -> None:
    args = parse_args()
    payload = load_config_payload(Path(args.config))
    rows = load_live_actions(Path(args.live_db))
    opens = live_open_rows(rows)
    open_0429 = [row for row in opens if row["timestamp"] == "2026-04-29 12:00"]
    open_0502 = [row for row in opens if row["timestamp"] == "2026-05-02 06:45"]
    report = {
        "inputs": {
            "config": str(Path(args.config).resolve()),
            "live_db": str(Path(args.live_db).resolve()),
            "data_15m": str(Path(args.data_15m).resolve()),
            "data_4h": str(Path(args.data_4h).resolve()),
            "start_date": args.start_date,
        },
        "partial_candle_gap_20260429_1200": {
            "live_open": open_0429[0] if open_0429 else None,
            "finalized_candle": candle_row(Path(args.data_15m), "2026-04-29 12:00"),
            "interpretation": (
                "Live opened on the 12:00 bar while the finalized candle closed below open; "
                "this is not reproducible by close-only replay."
            ),
        },
        "entry_sync_gap_20260502_0645": {
            "live_open": open_0502[0] if open_0502 else None,
            "live_stop_updates": live_stop_updates(rows, 1595),
            "modeled_slippage_5bps": replay_variant(
                payload,
                args,
                slippage_bps=5.0,
                replay_sync_entry_to_signal_price=False,
            ),
            "execution_sync_entry_price": replay_variant(
                payload,
                args,
                slippage_bps=5.0,
                replay_sync_entry_to_signal_price=True,
            ),
            "interpretation": (
                "Tokyo live reconciled entry to exchange fill price, shrinking R distance and enabling stage/ATR stop updates; "
                "modelled 5bps replay keeps the slipped entry and delays those updates."
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(
        {
            "partial_candle_live_open": bool(open_0429),
            "entry_sync_live_open": bool(open_0502),
            "modeled_5bps_recent": report["entry_sync_gap_20260502_0645"]["modeled_slippage_5bps"]["recent_trades"],
            "execution_sync_recent": report["entry_sync_gap_20260502_0645"]["execution_sync_entry_price"]["recent_trades"],
            "output": str(output),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
