#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.okx_client import OkxClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OKX market metadata cache for offline router sizing.")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT:USDT", "QQQ/USDT:USDT", "GOOGL/USDT:USDT"])
    parser.add_argument("--output", default="var/okx/markets_cache.json")
    parser.add_argument("--proxy", default=None)
    args = parser.parse_args()

    client = OkxClient(None, trading_mode="live", proxy=args.proxy)
    markets = client.load_markets()
    payload = {symbol: markets.get(symbol) for symbol in args.symbols}
    missing = [symbol for symbol, market in payload.items() if not market]
    if missing:
        raise SystemExit(f"Missing OKX market metadata: {', '.join(missing)}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(out), "symbols": list(payload)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
