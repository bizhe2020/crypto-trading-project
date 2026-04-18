from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.bot.okx_executor import ExecutorConfig
from deployment.bot.orderbook_feed import OkxOrderBookFeed

OPEN_ACTIONS = {"OPEN_LONG", "OPEN_SHORT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch next live trade, record books5, replay OBI exits, then recalculate performance")
    parser.add_argument("--state-db", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="var/forward_obi")
    parser.add_argument("--inst-id", default="BTC-USDT-SWAP")
    parser.add_argument("--channel", default="books5")
    parser.add_argument("--poll-interval-ms", type=int, default=100)
    parser.add_argument("--db-poll-seconds", type=float, default=1.0)
    parser.add_argument("--max-wait-seconds", type=int, default=0)
    parser.add_argument("--attach-current-open", action="store_true")
    parser.add_argument("--min-profit-rr", type=float, default=0.75)
    parser.add_argument("--tighten-dwell-seconds", type=float, default=1.5)
    parser.add_argument("--tighten-obi-threshold", type=float, default=0.12)
    parser.add_argument("--tighten-edge-bps", type=float, default=0.15)
    parser.add_argument("--low-lock-rr", type=float, default=0.25)
    parser.add_argument("--mid-lock-rr", type=float, default=0.5)
    parser.add_argument("--high-lock-rr", type=float, default=0.75)
    parser.add_argument("--force-exit-min-profit-rr", type=float, default=1.5)
    parser.add_argument("--force-exit-dwell-seconds", type=float, default=2.0)
    parser.add_argument("--force-exit-obi-threshold", type=float, default=0.2)
    parser.add_argument("--force-exit-edge-bps", type=float, default=0.25)
    return parser.parse_args()


def _parse_utc_timestamp(value: str) -> int:
    return int(datetime.strptime(value, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _supported_executor_config(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(ExecutorConfig)}
    return {key: value for key, value in payload.items() if key in allowed}


def _load_executor_config(path: str | Path) -> ExecutorConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExecutorConfig.from_dict(_supported_executor_config(payload))


def _connect(path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(path)


def _latest_action_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM action_log").fetchone()
    return int(row[0] or 0)


def _find_current_open(conn: sqlite3.Connection) -> tuple[int, dict[str, Any]] | None:
    rows = conn.execute(
        "SELECT id, action_type, payload FROM action_log WHERE action_type IN ('OPEN_LONG','OPEN_SHORT','CLOSE_POSITION') ORDER BY id"
    ).fetchall()
    current: tuple[int, dict[str, Any]] | None = None
    for action_id, action_type, payload_raw in rows:
        payload = json.loads(payload_raw)
        if action_type in OPEN_ACTIONS:
            current = (int(action_id), payload)
        else:
            current = None
    return current


def _wait_for_open(conn: sqlite3.Connection, args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    if args.attach_current_open:
        current = _find_current_open(conn)
        if current is not None:
            return current
    after_id = _latest_action_id(conn)
    started = time.time()
    while True:
        row = conn.execute(
            "SELECT id, payload FROM action_log WHERE id > ? AND action_type IN ('OPEN_LONG','OPEN_SHORT') ORDER BY id LIMIT 1",
            (after_id,),
        ).fetchone()
        if row is not None:
            return int(row[0]), json.loads(row[1])
        if args.max_wait_seconds > 0 and time.time() - started >= args.max_wait_seconds:
            raise TimeoutError("Timed out waiting for next open trade")
        time.sleep(max(args.db_poll_seconds, 0.2))


def _wait_for_close(conn: sqlite3.Connection, open_id: int, feed: OkxOrderBookFeed, output_jsonl: Path, poll_interval_ms: int, db_poll_seconds: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    last_snapshot_ts: int | None = None
    seen_stop_ids: set[int] = set()
    stop_updates: list[dict[str, Any]] = []
    with output_jsonl.open("a", encoding="utf-8") as handle:
        while True:
            snapshot = feed.latest_snapshot()
            if snapshot is not None and snapshot.timestamp_ms != last_snapshot_ts:
                handle.write(json.dumps({
                    "inst_id": snapshot.inst_id,
                    "timestamp_ms": snapshot.timestamp_ms,
                    "bids": [[level.price, level.size] for level in snapshot.bids],
                    "asks": [[level.price, level.size] for level in snapshot.asks],
                }, ensure_ascii=False) + "\n")
                handle.flush()
                last_snapshot_ts = snapshot.timestamp_ms

            rows = conn.execute(
                "SELECT id, timestamp, action_type, payload FROM action_log WHERE id > ? AND action_type IN ('UPDATE_STOP','CLOSE_POSITION') ORDER BY id",
                (open_id,),
            ).fetchall()
            for action_id, timestamp, action_type, payload_raw in rows:
                payload = json.loads(payload_raw)
                if action_type == "UPDATE_STOP" and int(action_id) not in seen_stop_ids:
                    seen_stop_ids.add(int(action_id))
                    stop_updates.append({
                        "timestamp": str(timestamp),
                        "timestamp_ms": _parse_utc_timestamp(str(timestamp)),
                        "stop_price": payload.get("stop_price"),
                        "target_price": payload.get("target_price"),
                        "reason": payload.get("reason"),
                        "metadata": payload.get("metadata") or {},
                    })
                elif action_type == "CLOSE_POSITION":
                    return {"timestamp": str(timestamp), "payload": payload}, stop_updates

            time.sleep(max(poll_interval_ms, 10) / 1000.0)
            time.sleep(max(db_poll_seconds - max(poll_interval_ms, 10) / 1000.0, 0.0))


def _write_single_trade_baseline(args: argparse.Namespace, config: ExecutorConfig, open_id: int, open_payload: dict[str, Any], close_info: dict[str, Any], stop_updates: list[dict[str, Any]], output_json: Path, output_csv: Path) -> None:
    open_meta = open_payload.get("metadata") or {}
    close_payload = close_info["payload"]
    close_meta = close_payload.get("metadata") or {}
    capital_at_entry = float(open_meta.get("capital_at_entry") or 0.0)
    risk_amount = float(open_meta.get("risk_amount") or 0.0)
    trade = {
        "trade_id": open_id,
        "direction": open_payload["direction"],
        "entry_time": open_payload["timestamp"],
        "entry_time_utc": datetime.strptime(open_payload["timestamp"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).isoformat(),
        "entry_timestamp_ms": _parse_utc_timestamp(open_payload["timestamp"]),
        "entry_price": float(open_payload["entry_price"]),
        "signal_entry_price": float(open_meta["signal_entry_price"]),
        "initial_stop_price": float(open_payload["stop_price"]),
        "current_stop_price": float(stop_updates[-1]["stop_price"]) if stop_updates else float(open_payload["stop_price"]),
        "target_price": float(open_payload["target_price"]) if open_payload.get("target_price") is not None else None,
        "entry_context": {
            "entry_regime_score": open_meta.get("entry_regime_score"),
            "target_rr": open_meta.get("target_rr"),
            "trail_style": open_meta.get("trail_style"),
            "max_hold_bars": open_meta.get("max_hold_bars"),
            "risk_per_trade": open_meta.get("risk_per_trade"),
            "risk_regime": open_meta.get("risk_regime"),
            "position_size_pct": open_meta.get("position_size_pct"),
            "notional": open_meta.get("notional"),
            "quantity": open_meta.get("quantity"),
            "risk_amount": open_meta.get("risk_amount"),
            "capital_at_entry": open_meta.get("capital_at_entry"),
            "max_notional": open_meta.get("max_notional"),
            "risk_based_notional": open_meta.get("risk_based_notional"),
        },
        "stop_updates": stop_updates,
        "obi_replay": {
            "status": "pending",
            "orderbook_inst_id": args.inst_id,
            "start_timestamp_ms": _parse_utc_timestamp(open_payload["timestamp"]),
            "end_timestamp_ms": _parse_utc_timestamp(close_info["timestamp"]),
            "baseline_exit_reason": close_payload.get("reason"),
            "baseline_exit_price": close_payload.get("exit_price"),
        },
        "exit_time": close_info["timestamp"],
        "exit_time_utc": datetime.strptime(close_info["timestamp"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).isoformat(),
        "exit_timestamp_ms": _parse_utc_timestamp(close_info["timestamp"]),
        "original_exit_reason": close_payload.get("reason"),
        "original_signal_exit_price": close_meta.get("signal_exit_price"),
        "original_exit_price": close_payload.get("exit_price"),
        "capital_at_entry": capital_at_entry,
        "gross_pnl": close_meta.get("gross_pnl"),
        "fees": close_meta.get("fees"),
        "slippage_cost": close_meta.get("slippage_cost"),
        "pnl": close_meta.get("net_pnl"),
        "pnl_pct": (float(close_meta.get("net_pnl") or 0.0) / capital_at_entry) if capital_at_entry else 0.0,
        "rr_ratio": (float(close_meta.get("net_pnl") or 0.0) / risk_amount) if risk_amount else 0.0,
        "final_stop_price_at_exit": float(stop_updates[-1]["stop_price"]) if stop_updates else float(open_payload["stop_price"]),
    }
    payload = {
        "metadata": {
            "strategy": "live_single_trade_monitor",
            "config_path": str(Path(args.config).resolve()),
            "symbol": config.symbol,
            "timeframe": config.timeframe,
            "informative_timeframe": config.informative_timeframe,
            "start_date": open_payload["timestamp"][:10],
            "orderbook_inst_id": args.inst_id,
            "trade_count": 1,
            "metrics_summary": {
                "initial_capital": capital_at_entry,
                "final_capital": capital_at_entry + float(close_meta.get("net_pnl") or 0.0),
                "total_return_pct": (float(close_meta.get("net_pnl") or 0.0) / capital_at_entry * 100.0) if capital_at_entry else 0.0,
                "profit_factor": 0.0,
                "win_rate": 100.0 if float(close_meta.get("net_pnl") or 0.0) > 0 else 0.0,
                "parameters": {
                    "leverage": config.leverage,
                    "fixed_notional_usdt": config.fixed_notional_usdt,
                    "taker_fee_rate": config.taker_fee_rate,
                    "slippage_bps": config.slippage_bps,
                },
            },
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "trades": [trade],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trade_id", "direction", "entry_time", "exit_time", "entry_price", "initial_stop_price", "target_price", "original_exit_reason", "original_exit_price", "pnl", "rr_ratio"])
        writer.writeheader()
        writer.writerow({
            "trade_id": trade["trade_id"],
            "direction": trade["direction"],
            "entry_time": trade["entry_time"],
            "exit_time": trade["exit_time"],
            "entry_price": trade["entry_price"],
            "initial_stop_price": trade["initial_stop_price"],
            "target_price": trade["target_price"],
            "original_exit_reason": trade["original_exit_reason"],
            "original_exit_price": trade["original_exit_price"],
            "pnl": trade["pnl"],
            "rr_ratio": trade["rr_ratio"],
        })


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    config = _load_executor_config(args.config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    conn = _connect(args.state_db)
    try:
        try:
            open_id, open_payload = _wait_for_open(conn, args)
        except TimeoutError as exc:
            print(json.dumps({"event": "timeout", "message": str(exc)}, ensure_ascii=False))
            return

        trade_tag = f"{open_payload['timestamp'].replace(' ', '_').replace(':', '')}_id{open_id}"
        trade_dir = output_root / trade_tag
        trade_dir.mkdir(parents=True, exist_ok=True)
        orderbook_path = trade_dir / "books5.jsonl"
        baseline_json = trade_dir / "baseline_trade.json"
        baseline_csv = trade_dir / "baseline_trade.csv"
        replay_json = trade_dir / "obi_replay.json"
        replay_csv = trade_dir / "obi_replay.csv"
        repriced_json = trade_dir / "obi_repriced_performance.json"
        repriced_csv = trade_dir / "obi_repriced_trades.csv"

        print(json.dumps({"event": "open_detected", "open_id": open_id, "timestamp": open_payload["timestamp"], "output_dir": str(trade_dir)}, ensure_ascii=False))

        feed = OkxOrderBookFeed(inst_id=args.inst_id, channel=args.channel)
        feed.start()
        try:
            close_info, stop_updates = _wait_for_close(conn, open_id, feed, orderbook_path, args.poll_interval_ms, args.db_poll_seconds)
        finally:
            feed.stop()

        _write_single_trade_baseline(args, config, open_id, open_payload, close_info, stop_updates, baseline_json, baseline_csv)

        _run([
            sys.executable,
            "deployment/bot/batch_replay_obi_overlay.py",
            "--trades-json", str(baseline_json),
            "--orderbook-input", str(orderbook_path),
            "--output-json", str(replay_json),
            "--output-csv", str(replay_csv),
            "--min-profit-rr", str(args.min_profit_rr),
            "--tighten-dwell-seconds", str(args.tighten_dwell_seconds),
            "--tighten-obi-threshold", str(args.tighten_obi_threshold),
            "--tighten-edge-bps", str(args.tighten_edge_bps),
            "--low-lock-rr", str(args.low_lock_rr),
            "--mid-lock-rr", str(args.mid_lock_rr),
            "--high-lock-rr", str(args.high_lock_rr),
            "--force-exit-min-profit-rr", str(args.force_exit_min_profit_rr),
            "--force-exit-dwell-seconds", str(args.force_exit_dwell_seconds),
            "--force-exit-obi-threshold", str(args.force_exit_obi_threshold),
            "--force-exit-edge-bps", str(args.force_exit_edge_bps),
        ])
        _run([
            sys.executable,
            "deployment/bot/recalculate_obi_replay_performance.py",
            "--baseline-trades-json", str(baseline_json),
            "--obi-replay-json", str(replay_json),
            "--output-json", str(repriced_json),
            "--output-csv", str(repriced_csv),
        ])

        print(json.dumps({
            "event": "pipeline_complete",
            "trade_id": open_id,
            "baseline_json": str(baseline_json),
            "orderbook_jsonl": str(orderbook_path),
            "replay_json": str(replay_json),
            "repriced_json": str(repriced_json),
        }, ensure_ascii=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
