#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_drift_monitor import build_live_trades, load_action_log  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-trade convergence audit for live DB trades vs replay event streams.")
    parser.add_argument("--live-db", required=True, help="Runtime sqlite DB copied from live/paper.")
    parser.add_argument("--replay-json", required=True, help="Replay JSON from replay_stable_smc_live_shadow.py.")
    parser.add_argument("--stream", default="live_shadow", help="Replay stream key. Default: live_shadow.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument("--start-time", default=None, help="Only include live trades with entry_time >= this UTC timestamp.")
    parser.add_argument("--max-nearby", type=int, default=6)
    return parser.parse_args()


def parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                parsed = None
        if parsed is None:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def event_return_pct(event: dict[str, Any]) -> float:
    if event.get("return_pct") is not None:
        return float(event.get("return_pct") or 0.0)
    return float(event.get("return", 0.0) or 0.0) * 100.0


def event_direction(event: dict[str, Any]) -> str:
    direction = str(event.get("direction") or "")
    if direction in {"BULL", "BEAR"}:
        return direction
    event_type = str(event.get("event_type") or "")
    if event_type.endswith("_short") or "short" in event_type:
        return "BEAR"
    if event_type.endswith("_long") or "long" in event_type:
        return "BULL"
    return direction


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    entry_time = parse_time(event.get("entry_time"))
    exit_time = parse_time(event.get("exit_time"))
    return {
        "event_key": f"{event.get('event_type')}|{event.get('entry_idx')}|{event.get('exit_idx')}",
        "event_type": event.get("event_type"),
        "direction": event_direction(event),
        "entry_time": entry_time.isoformat() if entry_time else None,
        "exit_time": exit_time.isoformat() if exit_time else None,
        "entry_idx": event.get("entry_idx"),
        "exit_idx": event.get("exit_idx"),
        "return_pct": round(event_return_pct(event), 4),
    }


def load_replay_events(path: Path, stream: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    section = payload.get(stream)
    if not isinstance(section, dict):
        raise ValueError(f"Replay stream not found or not an object: {stream}")
    events = section.get("events")
    if not isinstance(events, list):
        events = section.get("sample_events")
    if not isinstance(events, list):
        raise ValueError(f"Replay stream has no events/sample_events: {stream}")
    return [normalize_event(event) for event in events if isinstance(event, dict)]


def load_live_trades(path: Path, start_time: datetime | None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    trades, diagnostics = build_live_trades(load_action_log(path))
    rows: list[dict[str, Any]] = []
    for idx, trade in enumerate(trades, start=1):
        if start_time is not None and trade.entry_time < start_time:
            continue
        entry_execution_time = trade.entry_execution_time.isoformat() if trade.entry_execution_time else None
        exit_execution_time = trade.exit_execution_time.isoformat() if trade.exit_execution_time else None
        rows.append(
            {
                "trade_no": idx,
                "direction": trade.direction,
                "entry_time": trade.entry_time.isoformat(),
                "signal_entry_time": trade.entry_time.isoformat(),
                "entry_execution_time": entry_execution_time,
                "exit_time": trade.exit_time.isoformat(),
                "signal_exit_time": trade.exit_time.isoformat(),
                "exit_execution_time": exit_execution_time,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "signal_entry_price": trade.signal_entry_price,
                "signal_exit_price": trade.signal_exit_price,
                "stop_price": trade.stop_price,
                "target_price": trade.target_price,
                "exit_reason": trade.exit_reason,
                "pnl_pct": round((trade.pnl_pct or 0.0) * 100.0, 4) if trade.pnl_pct is not None else None,
                "net_pnl": round(float(trade.net_pnl), 6),
                "capital_at_entry": trade.capital_at_entry,
                "notional": trade.notional,
                "risk_amount": trade.risk_amount,
                "entry_slippage_bps": trade.entry_slippage_bps,
                "exit_slippage_bps": trade.exit_slippage_bps,
                "stop_target_deviation_bps": trade.stop_target_deviation_bps,
                "entry_execution_delay_seconds": seconds_between(entry_execution_time, trade.entry_time.isoformat()),
                "entry_execution_delay_bars": bars_from_seconds(seconds_between(entry_execution_time, trade.entry_time.isoformat())),
                "exit_execution_delay_seconds": seconds_between(exit_execution_time, trade.exit_time.isoformat()),
                "exit_execution_delay_bars": bars_from_seconds(seconds_between(exit_execution_time, trade.exit_time.isoformat())),
            }
        )
    return rows, diagnostics


def seconds_between(a: str | None, b: str | None) -> float | None:
    left = parse_time(a)
    right = parse_time(b)
    if left is None or right is None:
        return None
    return abs((left - right).total_seconds())


def bars_from_seconds(seconds: float | None, timeframe_seconds: int = 900) -> float | None:
    if seconds is None or timeframe_seconds <= 0:
        return None
    return round(float(seconds) / float(timeframe_seconds), 4)


def average(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def overlaps(live: dict[str, Any], event: dict[str, Any]) -> bool:
    live_entry = parse_time(live.get("entry_time"))
    live_exit = parse_time(live.get("exit_time"))
    event_entry = parse_time(event.get("entry_time"))
    event_exit = parse_time(event.get("exit_time"))
    if None in {live_entry, live_exit, event_entry, event_exit}:
        return False
    return bool(event_entry <= live_exit and event_exit >= live_entry)


def event_sort_key_for_live(live: dict[str, Any], event: dict[str, Any]) -> tuple[int, float, float]:
    exact = int(
        event.get("entry_time") == live.get("signal_entry_time") and event.get("direction") == live.get("direction")
    )
    same_direction = int(event.get("direction") == live.get("direction"))
    signal_entry_gap = seconds_between(live.get("signal_entry_time"), event.get("entry_time"))
    execution_entry_gap = seconds_between(live.get("entry_execution_time"), event.get("entry_time"))
    signal_exit_gap = seconds_between(live.get("signal_exit_time"), event.get("exit_time"))
    return (
        -exact,
        -same_direction,
        signal_entry_gap if signal_entry_gap is not None else float("inf"),
        execution_entry_gap if execution_entry_gap is not None else float("inf"),
        signal_exit_gap if signal_exit_gap is not None else float("inf"),
    )


def match_live_trade(live: dict[str, Any], events: list[dict[str, Any]], max_nearby: int) -> dict[str, Any]:
    exact = [
        event
        for event in events
        if event.get("entry_time") == live.get("signal_entry_time") and event.get("direction") == live.get("direction")
    ]
    covering = [event for event in events if overlaps(live, event)]
    nearby = sorted(events, key=lambda event: event_sort_key_for_live(live, event))[:max(1, max_nearby)]
    status = "exact_entry_match" if exact else "covered_by_replay_position" if covering else "no_overlap"
    best = (exact or covering or nearby)[0] if (exact or covering or nearby) else None
    signal_entry_gap_seconds = seconds_between(live.get("signal_entry_time"), (best or {}).get("entry_time"))
    execution_entry_gap_seconds = seconds_between(live.get("entry_execution_time"), (best or {}).get("entry_time"))
    signal_exit_gap_seconds = seconds_between(live.get("signal_exit_time"), (best or {}).get("exit_time"))
    execution_exit_gap_seconds = seconds_between(live.get("exit_execution_time"), (best or {}).get("exit_time"))
    return {
        "status": status,
        "best_replay_event": best,
        "exact_signal_entry_matches": exact,
        "exact_entry_matches": exact,
        "covering_events": covering,
        "nearby_events": nearby,
        "signal_entry_gap_seconds": signal_entry_gap_seconds,
        "execution_entry_gap_seconds": execution_entry_gap_seconds,
        "signal_exit_gap_seconds": signal_exit_gap_seconds,
        "execution_exit_gap_seconds": execution_exit_gap_seconds,
        "entry_execution_delay_seconds": live.get("entry_execution_delay_seconds"),
        "entry_execution_delay_bars": live.get("entry_execution_delay_bars"),
        "exit_execution_delay_seconds": live.get("exit_execution_delay_seconds"),
        "exit_execution_delay_bars": live.get("exit_execution_delay_bars"),
        "pnl_gap_pct": round(float(live.get("pnl_pct") or 0.0) - float((best or {}).get("return_pct") or 0.0), 4)
        if best
        else None,
    }


def compounded_return_pct(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    capital = 1.0
    for value in values:
        capital *= 1.0 + float(value) / 100.0
    return round((capital - 1.0) * 100.0, 4)


def main() -> None:
    args = parse_args()
    start_time = parse_time(args.start_time)
    live_rows, diagnostics = load_live_trades(Path(args.live_db), start_time)
    replay_events = load_replay_events(Path(args.replay_json), str(args.stream))
    audited = []
    for live in live_rows:
        audited.append({**live, "replay_match": match_live_trade(live, replay_events, int(args.max_nearby))})

    exact = sum(1 for row in audited if row["replay_match"]["status"] == "exact_entry_match")
    covered = sum(1 for row in audited if row["replay_match"]["status"] == "covered_by_replay_position")
    no_overlap = sum(1 for row in audited if row["replay_match"]["status"] == "no_overlap")
    matched_replay = [
        row["replay_match"]["best_replay_event"]
        for row in audited
        if isinstance(row.get("replay_match"), dict) and isinstance(row["replay_match"].get("best_replay_event"), dict)
    ]
    report = {
        "inputs": {
            "live_db": str(Path(args.live_db).resolve()),
            "replay_json": str(Path(args.replay_json).resolve()),
            "stream": args.stream,
            "start_time": args.start_time,
        },
        "diagnostics": diagnostics,
        "summary": {
            "live_trades": len(live_rows),
            "replay_events": len(replay_events),
            "exact_entry_matches": exact,
            "exact_signal_entry_matches": exact,
            "covered_by_replay_position": covered,
            "no_overlap": no_overlap,
            "live_compounded_return_pct": compounded_return_pct(live_rows, "pnl_pct"),
            "best_matched_replay_compounded_return_pct": compounded_return_pct(matched_replay, "return_pct"),
            "avg_entry_execution_delay_seconds": average([row.get("entry_execution_delay_seconds") for row in live_rows]),
            "avg_entry_execution_delay_bars": average([row.get("entry_execution_delay_bars") for row in live_rows]),
            "avg_matched_execution_entry_gap_seconds": average(
                [row["replay_match"].get("execution_entry_gap_seconds") for row in audited]
            ),
        },
        "rows": audited,
    }
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    for row in audited:
        match = row["replay_match"]
        best = match.get("best_replay_event") or {}
        print(
            f"{row['signal_entry_time']} {row['direction']} live={row.get('pnl_pct')}% "
            f"exec={row.get('entry_execution_time')} delay={row.get('entry_execution_delay_bars')} bars "
            f"match={match['status']} replay={best.get('event_key')} replay_ret={best.get('return_pct')}% "
            f"sig_gap={match.get('signal_entry_gap_seconds')}s exec_gap={match.get('execution_entry_gap_seconds')}s "
            f"gap={match.get('pnl_gap_pct')}%"
        )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"JSON written: {output}")


if __name__ == "__main__":
    main()
