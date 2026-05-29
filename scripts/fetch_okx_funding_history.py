#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import ccxt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch OKX funding rate history into a feather file.")
    parser.add_argument("--symbol", default="QQQ/USDT:USDT")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ex = ccxt.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    rows = ex.fetch_funding_rate_history(args.symbol, limit=int(args.limit))
    df = pd.DataFrame(rows)
    if df.empty:
        print("no_rows")
        return
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    elif "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], utc=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_feather(out)
    print(out)
    print(len(df), df["date"].min(), df["date"].max())


if __name__ == "__main__":
    main()
