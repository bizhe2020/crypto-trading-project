from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.bot.orderbook_feed import OkxOrderBookFeed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record OKX order book snapshots to JSONL")
    parser.add_argument("--inst-id", default="BTC-USDT-SWAP")
    parser.add_argument("--channel", default="books5")
    parser.add_argument("--url")
    parser.add_argument("--output")
    parser.add_argument("--output-dir")
    parser.add_argument("--file-prefix", default="btc_books5")
    parser.add_argument("--rotate-utc", choices=["none", "daily"], default="none")
    parser.add_argument("--duration-seconds", type=int, default=0, help="0 means run until interrupted")
    parser.add_argument("--poll-interval-ms", type=int, default=100)
    parser.add_argument("--quiet-snapshots", action="store_true")
    return parser.parse_args()


def _resolve_output_path(args: argparse.Namespace, timestamp_ms: int | None) -> Path:
    if args.rotate_utc == "daily":
        if not args.output_dir:
            raise SystemExit("--output-dir is required when --rotate-utc=daily")
        ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
        stamp = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y%m%d")
        return Path(args.output_dir) / f"{args.file_prefix}_{stamp}.jsonl"
    if args.output:
        return Path(args.output)
    if args.output_dir:
        return Path(args.output_dir) / f"{args.file_prefix}.jsonl"
    raise SystemExit("Provide --output or --output-dir")


def main() -> None:
    args = parse_args()
    feed = OkxOrderBookFeed(inst_id=args.inst_id, channel=args.channel, url=args.url)
    feed.start()

    start = time.time()
    last_timestamp_ms: int | None = None
    current_output_path: Path | None = None
    last_status_at = 0.0
    handle = None
    try:
        while True:
            snapshot = feed.latest_snapshot()
            if snapshot is not None and snapshot.timestamp_ms != last_timestamp_ms:
                next_output_path = _resolve_output_path(args, snapshot.timestamp_ms)
                if handle is None or next_output_path != current_output_path:
                    if handle is not None:
                        handle.close()
                    next_output_path.parent.mkdir(parents=True, exist_ok=True)
                    handle = next_output_path.open("a", encoding="utf-8")
                    current_output_path = next_output_path
                    print(json.dumps({"event": "rotate", "output": str(current_output_path)}, ensure_ascii=False))

                payload = {
                    "inst_id": snapshot.inst_id,
                    "timestamp_ms": snapshot.timestamp_ms,
                    "bids": [[level.price, level.size] for level in snapshot.bids],
                    "asks": [[level.price, level.size] for level in snapshot.asks],
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
                last_timestamp_ms = snapshot.timestamp_ms
                if not args.quiet_snapshots:
                    print(
                        json.dumps(
                            {"event": "snapshot", "timestamp_ms": snapshot.timestamp_ms, "output": str(current_output_path)},
                            ensure_ascii=False,
                        )
                    )
            elif time.time() - last_status_at >= 5.0:
                print(
                    json.dumps(
                        {
                            "event": "waiting_for_snapshot",
                            "last_error": feed.last_error(),
                        },
                        ensure_ascii=False,
                    )
                )
                last_status_at = time.time()

            if args.duration_seconds > 0 and time.time() - start >= args.duration_seconds:
                break
            time.sleep(max(args.poll_interval_ms, 10) / 1000.0)
    except KeyboardInterrupt:
        pass
    finally:
        if handle is not None:
            handle.close()
        feed.stop()


if __name__ == "__main__":
    main()
