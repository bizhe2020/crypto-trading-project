#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import ccxt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "okx" / "futures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch OKX swap OHLCV history into feather files.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--timeframe", action="append", default=None, help="Repeatable, e.g. --timeframe 15m --timeframe 4h")
    parser.add_argument("--start", default="2022-01-01T00:00:00Z")
    parser.add_argument("--end", default=None, help="UTC ISO timestamp. Defaults to exchange latest.")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument(
        "--proxy",
        default=None,
        help="HTTP/SOCKS proxy URL. Defaults to HTTPS_PROXY, HTTP_PROXY, or ALL_PROXY when set.",
    )
    return parser.parse_args()


def output_path_for(output_dir: Path, symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    return output_dir / f"{safe_symbol}-{timeframe}-futures.feather"


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.read_feather(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df[["date", "open", "high", "low", "close", "volume"]]


def normalize(rows: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[["date", "open", "high", "low", "close", "volume"]]


def merge_and_save(existing: pd.DataFrame, fetched: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    if existing.empty:
        combined = fetched.copy()
    elif fetched.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, fetched], ignore_index=True)
    if combined.empty:
        return combined
    combined["date"] = pd.to_datetime(combined["date"], utc=True)
    combined = combined.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_feather(output_path)
    return combined


def fetch_timeframe(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start: str,
    end: str | None,
    limit: int,
    output_path: Path,
    sleep_seconds: float,
) -> pd.DataFrame:
    existing = load_existing(output_path)
    since = exchange.parse8601(start)
    if not existing.empty:
        latest_ms = int(existing["date"].max().timestamp() * 1000)
        since = max(since, latest_ms + exchange.parse_timeframe(timeframe) * 1000)
    end_ms = exchange.parse8601(end) if end else None
    fetched_batches: list[pd.DataFrame] = []
    last_since = since

    while True:
        if end_ms is not None and since > end_ms:
            break
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not rows:
            break
        batch = normalize(rows)
        if end_ms is not None:
            batch = batch[batch["date"] <= pd.to_datetime(end_ms, unit="ms", utc=True)]
        if not batch.empty:
            fetched_batches.append(batch)
        next_since = int(rows[-1][0]) + exchange.parse_timeframe(timeframe) * 1000
        if next_since <= last_since:
            break
        since = next_since
        last_since = next_since
        print(f"{timeframe}: fetched through {pd.to_datetime(rows[-1][0], unit='ms', utc=True)}", flush=True)
        time.sleep(max(sleep_seconds, 0.0))

    fetched = pd.concat(fetched_batches, ignore_index=True) if fetched_batches else pd.DataFrame()
    return merge_and_save(existing, fetched, output_path)


def main() -> None:
    args = parse_args()
    timeframes = args.timeframe or ["15m", "4h"]
    proxy = args.proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("ALL_PROXY")
    exchange_config = {"enableRateLimit": True, "options": {"defaultType": "swap"}}
    if proxy:
        exchange_config["proxies"] = {"http": proxy, "https": proxy}
        exchange_config["aiohttp_proxy"] = proxy
        print(f"Using proxy: {proxy}")
    exchange = ccxt.okx(exchange_config)
    output_dir = Path(args.output_dir)
    for timeframe in timeframes:
        output_path = output_path_for(output_dir, args.symbol, timeframe)
        df = fetch_timeframe(
            exchange=exchange,
            symbol=args.symbol,
            timeframe=timeframe,
            start=args.start,
            end=args.end,
            limit=args.limit,
            output_path=output_path,
            sleep_seconds=args.sleep_seconds,
        )
        if df.empty:
            print(f"{timeframe}: no data written to {output_path}")
            continue
        print(f"{timeframe}: wrote {len(df)} rows to {output_path}")
        print(f"{timeframe}: range {df['date'].min()} -> {df['date'].max()}")


if __name__ == "__main__":
    main()
