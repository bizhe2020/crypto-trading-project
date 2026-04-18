from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.strategy.obi_trailing import OBIOverlay, OBIOverlayConfig, OrderBookLevel, OrderBookSnapshot


@dataclass
class ReplayTrade:
    open_action_id: int
    direction: str
    entry_time: str
    entry_timestamp_ms: int
    entry_price: float
    initial_stop_price: float
    stop_price: float
    target_price: float | None
    close_time: str | None = None
    close_timestamp_ms: int | None = None
    close_reason: str | None = None
    close_exit_price: float | None = None
    fallback_end_timestamp_ms: int | None = None
    end_timestamp_ms: int | None = None
    end_source: str = "eof"
    metadata: dict[str, Any] = field(default_factory=dict)
    snapshot_count: int = 0
    decision_count: int = 0
    tighten_count: int = 0
    exit_count: int = 0
    final_stop_price: float | None = None
    latest_metrics: dict[str, Any] | None = None
    first_snapshot_timestamp_ms: int | None = None
    last_snapshot_timestamp_ms: int | None = None
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def effective_end_timestamp_ms(self) -> int | None:
        return self.end_timestamp_ms if self.end_timestamp_ms is not None else self.fallback_end_timestamp_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch replay OBI overlay decisions from action_log or exported trade JSON + books5 JSONL")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--state-db")
    source_group.add_argument("--trades-json")
    parser.add_argument("--orderbook-input", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--limit", type=int, default=0)
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


def _parse_time_to_ms(value: str | None) -> int | None:
    if value is None:
        return None
    return int(__import__("datetime").datetime.strptime(value, "%Y-%m-%d %H:%M").replace(tzinfo=__import__("datetime").timezone.utc).timestamp() * 1000)


def _overlay_config(args: argparse.Namespace) -> OBIOverlayConfig:
    return OBIOverlayConfig(
        min_profit_rr=args.min_profit_rr,
        tighten_dwell_seconds=args.tighten_dwell_seconds,
        tighten_obi_threshold=args.tighten_obi_threshold,
        tighten_edge_bps=args.tighten_edge_bps,
        low_lock_rr=args.low_lock_rr,
        mid_lock_rr=args.mid_lock_rr,
        high_lock_rr=args.high_lock_rr,
        force_exit_min_profit_rr=args.force_exit_min_profit_rr,
        force_exit_dwell_seconds=args.force_exit_dwell_seconds,
        force_exit_obi_threshold=args.force_exit_obi_threshold,
        force_exit_edge_bps=args.force_exit_edge_bps,
    )


def _build_snapshot(payload: dict[str, Any]) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        inst_id=str(payload["inst_id"]),
        timestamp_ms=int(payload["timestamp_ms"]),
        bids=tuple(OrderBookLevel(price=float(level[0]), size=float(level[1])) for level in payload.get("bids", [])),
        asks=tuple(OrderBookLevel(price=float(level[0]), size=float(level[1])) for level in payload.get("asks", [])),
    )


def _iter_orderbook(path: Path) -> Iterator[OrderBookSnapshot]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield _build_snapshot(json.loads(line))


def _load_trades_from_export(args: argparse.Namespace) -> list[ReplayTrade]:
    payload = json.loads(Path(args.trades_json).read_text(encoding="utf-8"))
    exported_trades = payload.get("trades") or []
    start_filter_ms = _parse_time_to_ms(args.start_time)
    end_filter_ms = _parse_time_to_ms(args.end_time)
    trades: list[ReplayTrade] = []
    for item in exported_trades:
        entry_timestamp_ms = int(item["entry_timestamp_ms"])
        if start_filter_ms is not None and entry_timestamp_ms < start_filter_ms:
            continue
        if end_filter_ms is not None and entry_timestamp_ms > end_filter_ms:
            continue
        replay = item.get("obi_replay") or {}
        trades.append(
            ReplayTrade(
                open_action_id=int(item.get("trade_id", len(trades) + 1)),
                direction=str(item["direction"]),
                entry_time=str(item["entry_time"]),
                entry_timestamp_ms=entry_timestamp_ms,
                entry_price=float(item["entry_price"]),
                initial_stop_price=float(item["initial_stop_price"]),
                stop_price=float(item["initial_stop_price"]),
                target_price=float(item["target_price"]) if item.get("target_price") is not None else None,
                close_time=item.get("exit_time"),
                close_timestamp_ms=int(item["exit_timestamp_ms"]) if item.get("exit_timestamp_ms") is not None else None,
                close_reason=item.get("original_exit_reason"),
                close_exit_price=float(item["original_exit_price"]) if item.get("original_exit_price") is not None else None,
                end_timestamp_ms=int(replay["end_timestamp_ms"]) if replay.get("end_timestamp_ms") is not None else None,
                metadata={"entry_context": item.get("entry_context") or {}, "stop_updates": item.get("stop_updates") or []},
            )
        )
        if args.limit > 0 and len(trades) >= args.limit:
            break
    return trades


def _load_trades_from_state_db(args: argparse.Namespace) -> list[ReplayTrade]:
    start_filter_ms = _parse_time_to_ms(args.start_time)
    end_filter_ms = _parse_time_to_ms(args.end_time)
    rows = sqlite3.connect(args.state_db).execute(
        "SELECT id, timestamp, action_type, payload FROM action_log WHERE action_type IN ('OPEN_LONG','OPEN_SHORT','CLOSE_POSITION') ORDER BY timestamp, id"
    ).fetchall()
    trades: list[ReplayTrade] = []
    current: ReplayTrade | None = None
    for action_id, timestamp, action_type, payload_raw in rows:
        timestamp_ms = _parse_time_to_ms(timestamp)
        payload = json.loads(payload_raw)
        if action_type in {"OPEN_LONG", "OPEN_SHORT"}:
            if current is not None:
                current.fallback_end_timestamp_ms = timestamp_ms
                current.end_source = "next_open"
                trades.append(current)
            current = ReplayTrade(
                open_action_id=int(action_id),
                direction=str(payload["direction"]),
                entry_time=str(payload["timestamp"]),
                entry_timestamp_ms=int(timestamp_ms),
                entry_price=float(payload["entry_price"]),
                initial_stop_price=float(payload["stop_price"]),
                stop_price=float(payload["stop_price"]),
                target_price=float(payload["target_price"]) if payload.get("target_price") is not None else None,
                metadata=dict(payload.get("metadata") or {}),
                final_stop_price=float(payload["stop_price"]),
            )
        elif action_type == "CLOSE_POSITION" and current is not None:
            current.close_time = str(payload["timestamp"])
            current.close_timestamp_ms = int(timestamp_ms)
            current.close_reason = payload.get("reason")
            current.close_exit_price = float(payload["exit_price"]) if payload.get("exit_price") is not None else None
            current.end_timestamp_ms = int(timestamp_ms)
            current.end_source = "close"
            trades.append(current)
            current = None
    if current is not None:
        trades.append(current)
    filtered: list[ReplayTrade] = []
    for trade in trades:
        if start_filter_ms is not None and trade.entry_timestamp_ms < start_filter_ms:
            continue
        if end_filter_ms is not None and trade.entry_timestamp_ms > end_filter_ms:
            continue
        filtered.append(trade)
        if args.limit > 0 and len(filtered) >= args.limit:
            break
    return filtered


def _load_trades(args: argparse.Namespace) -> list[ReplayTrade]:
    if args.trades_json:
        return _load_trades_from_export(args)
    return _load_trades_from_state_db(args)


def main() -> None:
    args = parse_args()
    trades = _load_trades(args)
    orderbook_iter = iter(_iter_orderbook(Path(args.orderbook_input)))
    current_snapshot = next(orderbook_iter, None)
    overlays = [OBIOverlay(_overlay_config(args)) for _ in trades]
    positions = [
        SimpleNamespace(
            direction=trade.direction,
            entry_time=trade.entry_time,
            entry_price=trade.entry_price,
            initial_sl_price=trade.initial_stop_price,
            sl_price=trade.stop_price,
        )
        for trade in trades
    ]

    for idx, trade in enumerate(trades):
        overlay = overlays[idx]
        position = positions[idx]
        window_end_ms = trade.effective_end_timestamp_ms()
        while current_snapshot is not None and current_snapshot.timestamp_ms < trade.entry_timestamp_ms:
            current_snapshot = next(orderbook_iter, None)
        while current_snapshot is not None:
            if current_snapshot.timestamp_ms < trade.entry_timestamp_ms:
                current_snapshot = next(orderbook_iter, None)
                continue
            if window_end_ms is not None and current_snapshot.timestamp_ms > window_end_ms:
                break
            decision = overlay.evaluate(current_snapshot, position)
            trade.snapshot_count += 1
            if trade.first_snapshot_timestamp_ms is None:
                trade.first_snapshot_timestamp_ms = current_snapshot.timestamp_ms
            trade.last_snapshot_timestamp_ms = current_snapshot.timestamp_ms
            trade.latest_metrics = decision.metrics
            if decision.action == "tighten" and decision.stop_price is not None:
                position.sl_price = decision.stop_price
                trade.stop_price = decision.stop_price
                trade.final_stop_price = decision.stop_price
                trade.decision_count += 1
                trade.tighten_count += 1
                trade.decisions.append(
                    {
                        "timestamp_ms": current_snapshot.timestamp_ms,
                        "action": decision.action,
                        "reason": decision.reason,
                        "stop_price": decision.stop_price,
                        "metrics": decision.metrics,
                    }
                )
            elif decision.action == "exit":
                trade.decision_count += 1
                trade.exit_count += 1
                trade.decisions.append(
                    {
                        "timestamp_ms": current_snapshot.timestamp_ms,
                        "action": decision.action,
                        "reason": decision.reason,
                        "exit_price": decision.exit_price,
                        "metrics": decision.metrics,
                    }
                )
                current_snapshot = next(orderbook_iter, None)
                break
            current_snapshot = next(orderbook_iter, None)

    summary = {
        "state_db": args.state_db,
        "trades_json": args.trades_json,
        "orderbook_input": args.orderbook_input,
        "total_trades": len(trades),
        "covered_trades": sum(1 for trade in trades if trade.snapshot_count > 0),
        "trades_with_tighten": sum(1 for trade in trades if trade.tighten_count > 0),
        "trades_with_exit": sum(1 for trade in trades if trade.exit_count > 0),
        "total_tighten_events": sum(trade.tighten_count for trade in trades),
        "total_exit_events": sum(trade.exit_count for trade in trades),
        "config": asdict(_overlay_config(args)),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "trades": [asdict(trade) for trade in trades]}
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "open_action_id", "direction", "entry_time", "close_time", "entry_price", "initial_stop_price",
                "target_price", "snapshot_count", "decision_count", "tighten_count", "exit_count", "close_reason",
            ])
            writer.writeheader()
            for trade in trades:
                writer.writerow(
                    {
                        "open_action_id": trade.open_action_id,
                        "direction": trade.direction,
                        "entry_time": trade.entry_time,
                        "close_time": trade.close_time,
                        "entry_price": trade.entry_price,
                        "initial_stop_price": trade.initial_stop_price,
                        "target_price": trade.target_price,
                        "snapshot_count": trade.snapshot_count,
                        "decision_count": trade.decision_count,
                        "tighten_count": trade.tighten_count,
                        "exit_count": trade.exit_count,
                        "close_reason": trade.close_reason,
                        "final_stop_price": trade.final_stop_price,
                    }
                )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
