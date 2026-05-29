#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import requests
from urllib.parse import urlencode


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "public" / "etf"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch public ETF OHLCV history from Yahoo Finance chart API.")
    parser.add_argument("--symbol", action="append", default=None, help="Repeatable, e.g. --symbol QQQ --symbol TQQQ")
    parser.add_argument("--timeframe", action="append", default=None, help="Repeatable. Use 1d for long history.")
    parser.add_argument("--start", default="2022-01-01T00:00:00Z")
    parser.add_argument("--end", default=None, help="UTC ISO timestamp. Defaults to now.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--proxy", default=None, help="HTTP proxy URL, e.g. http://127.0.0.1:6244")
    return parser.parse_args()


def output_path_for(output_dir: Path, symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.upper().replace("/", "_").replace(":", "_")
    return output_dir / f"{safe_symbol}-{timeframe}.feather"


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.read_feather(path)
    return normalize_dataframe(df)


def normalize_timestamp(value: str | None, default_now: bool = False) -> pd.Timestamp:
    timestamp = pd.Timestamp.now(tz="UTC") if default_now and not value else pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def normalize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    if "date" not in result.columns:
        time_column = next((column for column in ("date", "timestamp", "datetime", "open_time") if column in result.columns), None)
        if time_column is None:
            raise ValueError("ETF data must contain a date/timestamp column")
        result = result.rename(columns={time_column: "date"})
    required_columns = {"date", "open", "high", "low", "close", "volume"}
    missing = required_columns - set(result.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"ETF data is missing required columns: {missing_text}")
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["date", "open", "high", "low", "close"])
    result = result.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return result[["date", "open", "high", "low", "close", "volume"]]


def fetch_chunk(
    session: requests.Session,
    symbol: str,
    start_ms: int,
    end_ms: int,
    timeframe: str,
    proxy: str | None = None,
) -> pd.DataFrame:
    params = {
        "period1": int(start_ms / 1000),
        "period2": int(end_ms / 1000),
        "interval": timeframe,
        "includeAdjustedClose": "true",
        "events": "div,splits",
    }
    if proxy:
        url = f"{YAHOO_CHART_URL.format(symbol=symbol.upper())}?{urlencode(params)}"
        command = [
            "curl",
            "-sS",
            "-L",
            "--max-time",
            "60",
            "-A",
            "Mozilla/5.0",
            "--compressed",
            "--proxy",
            proxy,
            url,
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"curl failed for {symbol} {timeframe}: {result.stderr.strip() or result.stdout.strip()}")
        payload = json.loads(result.stdout)
    else:
        response = session.get(YAHOO_CHART_URL.format(symbol=symbol.upper()), params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError(f"Yahoo chart error for {symbol} {timeframe}: {error}")
    result = chart.get("result") or []
    if not result:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    result0 = result[0]
    timestamps = result0.get("timestamp") or []
    indicators = result0.get("indicators", {}).get("quote", [])
    if not timestamps or not indicators:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    quote = indicators[0]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True),
            "open": quote.get("open", []),
            "high": quote.get("high", []),
            "low": quote.get("low", []),
            "close": quote.get("close", []),
            "volume": quote.get("volume", []),
        }
    )
    return normalize_dataframe(frame)


def fetch_timeframe(
    session: requests.Session,
    symbol: str,
    timeframe: str,
    start: str,
    end: str | None,
    output_path: Path,
    sleep_seconds: float,
    proxy: str | None = None,
) -> pd.DataFrame:
    existing = load_existing(output_path)
    start_dt = normalize_timestamp(start)
    end_dt = normalize_timestamp(end, default_now=True)
    since = start_dt
    if not existing.empty:
        since = max(since, existing["date"].max() + pd.Timedelta(minutes=1))

    fetched_batches: list[pd.DataFrame] = []
    max_span = {
        "1d": pd.Timedelta(days=3650),
        "1h": pd.Timedelta(days=700),
        "15m": pd.Timedelta(days=59),
        "5m": pd.Timedelta(days=59),
    }.get(timeframe, pd.Timedelta(days=59))
    step = {
        "1d": pd.Timedelta(days=1),
        "1h": pd.Timedelta(hours=1),
        "15m": pd.Timedelta(minutes=15),
        "5m": pd.Timedelta(minutes=5),
    }.get(timeframe, pd.Timedelta(minutes=15))

    while since < end_dt:
        chunk_end = min(since + max_span, end_dt)
        batch = fetch_chunk(
            session,
            symbol,
            int(since.timestamp() * 1000),
            int(chunk_end.timestamp() * 1000),
            timeframe,
            proxy=proxy,
        )
        if not batch.empty:
            batch = batch[(batch["date"] >= since) & (batch["date"] <= end_dt)]
            if not batch.empty:
                fetched_batches.append(batch)
                since = batch["date"].max() + step
            else:
                since = chunk_end + step
        else:
            since = chunk_end + step
        print(f"{symbol.upper()} {timeframe}: fetched through {since.isoformat()}", flush=True)
        time.sleep(max(sleep_seconds, 0.0))

    fetched = pd.concat(fetched_batches, ignore_index=True) if fetched_batches else pd.DataFrame()
    combined = pd.concat([existing, fetched], ignore_index=True) if not existing.empty else fetched
    combined = normalize_dataframe(combined) if not combined.empty else combined
    if not combined.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_feather(output_path)
    return combined


def main() -> None:
    args = parse_args()
    symbols = args.symbol or ["QQQ"]
    timeframes = args.timeframe or ["1d"]
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    if args.proxy:
        session.proxies.update({"http": args.proxy, "https": args.proxy})
        print(f"Using proxy: {args.proxy}")

    output_dir = Path(args.output_dir)
    for symbol in symbols:
        for timeframe in timeframes:
            output_path = output_path_for(output_dir, symbol, timeframe)
            df = fetch_timeframe(
                session=session,
                symbol=symbol,
                timeframe=timeframe,
                start=args.start,
                end=args.end,
                output_path=output_path,
                sleep_seconds=args.sleep_seconds,
                proxy=args.proxy,
            )
            if df.empty:
                print(f"{symbol.upper()} {timeframe}: no data written to {output_path}")
                continue
            print(f"{symbol.upper()} {timeframe}: wrote {len(df)} rows to {output_path}")
            print(f"{symbol.upper()} {timeframe}: range {df['date'].min()} -> {df['date'].max()}")


if __name__ == "__main__":
    main()
