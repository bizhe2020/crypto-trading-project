from __future__ import annotations

import argparse
import json
import time
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-seconds", type=int, default=0, help="0 means run until interrupted")
    parser.add_argument("--poll-interval-ms", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feed = OkxOrderBookFeed(inst_id=args.inst_id, channel=args.channel)
    feed.start()

    start = time.time()
    last_timestamp_ms: int | None = None
    try:
        with output_path.open("a", encoding="utf-8") as handle:
            while True:
                snapshot = feed.latest_snapshot()
                if snapshot is not None and snapshot.timestamp_ms != last_timestamp_ms:
                    payload = {
                        "inst_id": snapshot.inst_id,
                        "timestamp_ms": snapshot.timestamp_ms,
                        "bids": [[level.price, level.size] for level in snapshot.bids],
                        "asks": [[level.price, level.size] for level in snapshot.asks],
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    handle.flush()
                    last_timestamp_ms = snapshot.timestamp_ms
                    print(json.dumps({"event": "snapshot", "timestamp_ms": snapshot.timestamp_ms}, ensure_ascii=False))

                if args.duration_seconds > 0 and time.time() - start >= args.duration_seconds:
                    break
                time.sleep(max(args.poll_interval_ms, 10) / 1000.0)
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop()


if __name__ == "__main__":
    main()
