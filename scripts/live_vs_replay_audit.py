from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.market_data import OhlcvRepository
from bot.okx_executor import OkxExecutionEngine
from strategy.scalp_robust_v2_core import (
    ActionType,
    ScalpRobustEngine,
    dataframe_to_candles,
)


OPEN_ACTION_TYPES = {ActionType.OPEN_LONG.value, ActionType.OPEN_SHORT.value}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit live EVALUATE actions against a continuous strategy replay. "
            "This is intended to detect candles where replay would open but live did not."
        )
    )
    parser.add_argument("--config", default="config/config.live.high-leverage-structure.json")
    parser.add_argument("--db", default=None, help="Live runtime sqlite DB. Defaults to state_db_path from config.")
    parser.add_argument("--bars", type=int, default=240, help="Number of recent live EVALUATE rows to compare.")
    parser.add_argument("--start-date", default="2022-01-01", help="Continuous replay start date, UTC.")
    parser.add_argument("--offline", action="store_true", help="Use only local OHLCV files; do not fetch remote tail.")
    parser.add_argument("--json-output", default=None, help="Optional path for full JSON audit output.")
    parser.add_argument("--show-matches", action="store_true", help="Include matching rows in stdout.")
    return parser.parse_args()


def load_live_evaluates(db_path: Path, bars: int) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, payload, created_at
            FROM action_log
            WHERE action_type = 'EVALUATE'
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(bars),),
        ).fetchall()
    result: list[dict[str, Any]] = []
    for row_id, timestamp, payload, created_at in reversed(rows):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            decoded = {"raw": payload}
        actions = decoded.get("actions") if isinstance(decoded, dict) else []
        result.append(
            {
                "id": row_id,
                "timestamp": str(timestamp),
                "processed_candle_time": str(decoded.get("processed_candle_time") or timestamp)
                if isinstance(decoded, dict)
                else str(timestamp),
                "created_at": str(created_at),
                "status": decoded.get("status") if isinstance(decoded, dict) else None,
                "actions": actions if isinstance(actions, list) else [],
            }
        )
    return result


def timestamp_to_utc_seconds(value: str) -> float:
    normalized = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).timestamp()


def find_start_idx(engine: ScalpRobustEngine, start_date: str) -> int:
    start_ts = timestamp_to_utc_seconds(start_date)
    idx = next((i for i, candle in enumerate(engine.c15m) if candle.ts >= start_ts), 0)
    return max(100, idx + 100)


def build_replay_engine(executor: OkxExecutionEngine, *, offline: bool) -> ScalpRobustEngine:
    client = None if offline else executor.client
    bundle = OhlcvRepository(executor.config.data_root).load_pair(
        executor.config.symbol,
        client=client,
        timeframe=executor.config.timeframe,
        informative_timeframe=executor.config.informative_timeframe,
    )
    primary_candles = dataframe_to_candles(bundle.primary_candles)
    informative_candles = dataframe_to_candles(bundle.informative_candles)
    return ScalpRobustEngine.from_candles(
        informative_candles,
        primary_candles,
        executor.config.to_scalp_strategy_config(),
    )


def timestamp_index(engine: ScalpRobustEngine) -> dict[str, int]:
    return {engine._timestamp_for_idx(idx): idx for idx in range(len(engine.c15m))}


def action_signature(action: dict[str, Any]) -> str:
    action_type = normalized_action_type(action)
    direction = str(action.get("direction") or "")
    reason = str(action.get("reason") or "")
    if action_type in OPEN_ACTION_TYPES:
        return f"{action_type}:{direction}"
    if action_type == ActionType.CLOSE_POSITION.value:
        return f"{action_type}:{direction}:{reason}"
    if action_type == ActionType.UPDATE_STOP.value:
        stop_price = action.get("stop_price")
        try:
            stop_text = f"{float(stop_price):.1f}"
        except (TypeError, ValueError):
            stop_text = "-"
        return f"{action_type}:{direction}:{reason}:{stop_text}"
    return f"{action_type}:{direction}:{reason}"


def summarize_action(action: dict[str, Any]) -> dict[str, Any]:
    metadata = action.get("metadata") if isinstance(action.get("metadata"), dict) else {}
    return {
        "type": normalized_action_type(action),
        "direction": action.get("direction"),
        "entry_price": action.get("entry_price"),
        "stop_price": action.get("stop_price"),
        "target_price": action.get("target_price"),
        "reason": action.get("reason"),
        "regime_label": metadata.get("regime_label"),
        "risk_regime": metadata.get("risk_regime"),
        "trail_style": metadata.get("trail_style"),
        "notional": metadata.get("notional"),
        "quantity": metadata.get("quantity"),
    }


def normalized_action_type(action: dict[str, Any]) -> str:
    raw = action.get("type")
    if isinstance(raw, ActionType):
        return raw.value
    text = str(raw or "")
    prefix = "ActionType."
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def group_replay_actions(actions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        timestamp = str(action.get("timestamp") or "")
        if timestamp:
            grouped[timestamp].append(action)
    return dict(grouped)


def compare_actions(live_actions: list[dict[str, Any]], replay_actions: list[dict[str, Any]]) -> dict[str, Any]:
    live_signatures = [action_signature(action) for action in live_actions]
    replay_signatures = [action_signature(action) for action in replay_actions]
    live_open = [action for action in live_actions if normalized_action_type(action) in OPEN_ACTION_TYPES]
    replay_open = [action for action in replay_actions if normalized_action_type(action) in OPEN_ACTION_TYPES]
    missed_open = bool(replay_open and not live_open)
    extra_live_open = bool(live_open and not replay_open)
    exact_match = live_signatures == replay_signatures
    return {
        "match": exact_match,
        "missed_open": missed_open,
        "extra_live_open": extra_live_open,
        "live_signatures": live_signatures,
        "replay_signatures": replay_signatures,
        "live_actions": [summarize_action(action) for action in live_actions],
        "replay_actions": [summarize_action(action) for action in replay_actions],
    }


def print_report(report: dict[str, Any], *, show_matches: bool) -> None:
    summary = report["summary"]
    print("LIVE VS REPLAY AUDIT")
    print("=" * 72)
    print(f"config: {report['config']}")
    print(f"db: {report['db']}")
    print(f"replay_start_date: {report['replay_start_date']}")
    print(f"live_rows: {summary['live_rows']}")
    print(f"matched: {summary['matched']}")
    print(f"mismatched: {summary['mismatched']}")
    print(f"missed_open: {summary['missed_open']}")
    print(f"extra_live_open: {summary['extra_live_open']}")
    print(f"replay_only_action_rows: {summary['replay_only_action_rows']}")
    print()

    rows = report["rows"] if show_matches else [row for row in report["rows"] if not row["match"]]
    if not rows:
        print("No mismatches in audited live EVALUATE rows.")
        return

    for row in rows:
        marker = "OK" if row["match"] else "DIFF"
        if row["missed_open"]:
            marker = "MISSED_OPEN"
        elif row["extra_live_open"]:
            marker = "EXTRA_LIVE_OPEN"
        print(f"[{marker}] {row['timestamp']} live={row['live_signatures']} replay={row['replay_signatures']}")
        if row["live_actions"]:
            print(f"  live:   {json.dumps(row['live_actions'], ensure_ascii=False)}")
        if row["replay_actions"]:
            print(f"  replay: {json.dumps(row['replay_actions'], ensure_ascii=False)}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    executor = OkxExecutionEngine.from_file(config_path)
    db_path = Path(args.db or executor.config.state_db_path).resolve()
    live_rows = load_live_evaluates(db_path, args.bars)
    if not live_rows:
        raise SystemExit(f"No EVALUATE rows found in {db_path}")

    engine = build_replay_engine(executor, offline=bool(args.offline))
    index_by_time = timestamp_index(engine)
    missing_times = [row["processed_candle_time"] for row in live_rows if row["processed_candle_time"] not in index_by_time]
    if missing_times:
        sample = ", ".join(missing_times[:5])
        raise SystemExit(f"Live timestamps not found in local replay candles: {sample}")

    replay_start_idx = find_start_idx(engine, args.start_date)
    latest_live_idx = max(index_by_time[row["processed_candle_time"]] for row in live_rows)
    replay_actions = [asdict(action) for action in engine.evaluate_range(replay_start_idx, latest_live_idx + 1)]
    replay_by_time = group_replay_actions(replay_actions)

    rows: list[dict[str, Any]] = []
    matched = mismatched = missed_open = extra_live_open = replay_only_action_rows = 0
    for live_row in live_rows:
        timestamp = live_row["processed_candle_time"]
        live_actions = live_row["actions"]
        replay_actions_at_time = replay_by_time.get(timestamp, [])
        comparison = compare_actions(live_actions, replay_actions_at_time)
        row = {
            "id": live_row["id"],
            "timestamp": timestamp,
            "created_at": live_row["created_at"],
            "live_status": live_row["status"],
            **comparison,
        }
        rows.append(row)
        if comparison["match"]:
            matched += 1
        else:
            mismatched += 1
        if comparison["missed_open"]:
            missed_open += 1
        if comparison["extra_live_open"]:
            extra_live_open += 1
        if replay_actions_at_time and not live_actions:
            replay_only_action_rows += 1

    report = {
        "config": str(config_path),
        "db": str(db_path),
        "bars": int(args.bars),
        "replay_start_date": args.start_date,
        "replay_start_idx": replay_start_idx,
        "latest_live_idx": latest_live_idx,
        "summary": {
            "live_rows": len(live_rows),
            "matched": matched,
            "mismatched": mismatched,
            "missed_open": missed_open,
            "extra_live_open": extra_live_open,
            "replay_only_action_rows": replay_only_action_rows,
        },
        "rows": rows,
    }
    print_report(report, show_matches=bool(args.show_matches))
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print()
        print(f"JSON written: {output_path}")


if __name__ == "__main__":
    main()
