#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "data" / "public" / "qqq_constituents"
DEFAULT_OUTPUT = ROOT / "data" / "public" / "breadth" / "qqq_constituent_breadth-1d.feather"
DEFAULT_REPORT = ROOT / "var" / "reports" / "qqq_constituent_breadth_fetch.json"
NASDAQ100_URL = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
NASDAQ_HISTORICAL_URL = "https://api.nasdaq.com/api/quote/{symbol}/historical"
STOCKANALYSIS_HOLDINGS_URL = "https://stockanalysis.com/etf/qqq/holdings/"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch current Nasdaq-100/QQQ constituents and build daily constituent breadth features."
    )
    parser.add_argument("--start", default="2020-01-01T00:00:00Z")
    parser.add_argument("--end", default=None, help="UTC ISO timestamp. Defaults to now.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--symbols", default=None, help="Optional comma-separated symbol override.")
    parser.add_argument("--max-symbols", type=int, default=0, help="Debug cap. 0 means no cap.")
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--skip-fetch", action="store_true", help="Use cached raw constituent files only.")
    parser.add_argument("--holdings-csv", default=None, help="Optional local holdings CSV with symbol/weight columns.")
    parser.add_argument("--weights-url", default=STOCKANALYSIS_HOLDINGS_URL)
    return parser.parse_args()


def normalize_timestamp(value: str | None, default_now: bool = False) -> pd.Timestamp:
    timestamp = pd.Timestamp.now(tz="UTC") if default_now and not value else pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def yahoo_symbol(symbol: str) -> str:
    return normalize_symbol(symbol)


def output_path_for(raw_dir: Path, symbol: str) -> Path:
    safe_symbol = yahoo_symbol(symbol).replace("/", "_").replace(":", "_")
    return raw_dir / f"{safe_symbol}-1d.feather"


def request_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    return session


def fetch_nasdaq100_symbols(session: requests.Session) -> pd.DataFrame:
    response = session.get(NASDAQ100_URL, timeout=30)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", {}).get("data", {}).get("rows", [])
    records: list[dict[str, Any]] = []
    for row in rows:
        symbol = normalize_symbol(str(row.get("symbol", "")))
        if not symbol:
            continue
        records.append(
            {
                "symbol": symbol,
                "name": str(row.get("companyName", "")),
                "market_cap": parse_number(row.get("marketCap")),
                "source": "nasdaq100",
            }
        )
    return pd.DataFrame(records).drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    text = re.sub(r"[^0-9.\-]", "", str(value))
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fetch_json_with_curl(url: str) -> dict[str, Any]:
    command = [
        "curl",
        "-sS",
        "-L",
        "--max-time",
        "60",
        "-A",
        "Mozilla/5.0",
        "-H",
        "Accept: application/json, text/plain, */*",
        url,
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


def strip_tags(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value)
    text = re.sub(r"\s+", " ", html.unescape(text))
    return text.strip()


def parse_weight_pct(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).replace("%", "").replace(",", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fetch_stockanalysis_weights(session: requests.Session, url: str) -> pd.DataFrame:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    table_match = re.search(r"<table.*?</table>", response.text, flags=re.S | re.I)
    if not table_match:
        return pd.DataFrame(columns=["symbol", "weight_pct", "source"])
    records: list[dict[str, Any]] = []
    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", table_match.group(0), flags=re.S | re.I):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.S | re.I)
        if len(cells) < 4:
            continue
        symbol = normalize_symbol(strip_tags(cells[1]))
        weight = parse_weight_pct(strip_tags(cells[3]))
        if symbol and weight is not None:
            records.append({"symbol": symbol, "weight_pct": weight, "source": "stockanalysis"})
    return pd.DataFrame(records).drop_duplicates("symbol").reset_index(drop=True)


def load_holdings_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {str(column).strip().lower(): column for column in df.columns}
    symbol_col = lower.get("symbol") or lower.get("ticker") or lower.get("holding ticker")
    weight_col = lower.get("weight") or lower.get("weight_pct") or lower.get("% weight") or lower.get("allocation")
    if symbol_col is None:
        raise ValueError(f"{path} must contain a symbol/ticker column")
    out = pd.DataFrame({"symbol": df[symbol_col].map(normalize_symbol)})
    if weight_col is not None:
        out["weight_pct"] = df[weight_col].map(parse_weight_pct)
    else:
        out["weight_pct"] = None
    out["source"] = "csv"
    return out.dropna(subset=["symbol"]).drop_duplicates("symbol").reset_index(drop=True)


def build_universe(args: argparse.Namespace, session: requests.Session) -> tuple[pd.DataFrame, list[str]]:
    flags: list[str] = []
    if args.symbols:
        symbols = [normalize_symbol(item) for item in str(args.symbols).split(",") if item.strip()]
        universe = pd.DataFrame({"symbol": symbols, "source": "manual"})
    else:
        universe = fetch_nasdaq100_symbols(session)
    if args.holdings_csv:
        weights = load_holdings_csv(Path(args.holdings_csv))
    else:
        try:
            weights = fetch_stockanalysis_weights(session, str(args.weights_url))
        except Exception as exc:
            flags.append(f"weights_fetch_failed:{type(exc).__name__}:{exc}")
            weights = pd.DataFrame(columns=["symbol", "weight_pct", "source"])
    if not weights.empty and "weight_pct" in weights.columns:
        universe = universe.merge(weights[["symbol", "weight_pct"]], on="symbol", how="left")
    else:
        universe["weight_pct"] = None
    universe["weight_pct"] = pd.to_numeric(universe["weight_pct"], errors="coerce")
    if universe["weight_pct"].isna().all():
        flags.append("weights_unavailable_equal_weight_used")
    if int(args.max_symbols) > 0:
        universe = universe.head(int(args.max_symbols)).copy()
        flags.append(f"max_symbols_cap:{int(args.max_symbols)}")
    return universe.drop_duplicates("symbol").reset_index(drop=True), flags


def normalize_ohlcv(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df.columns = [str(column).strip().lower() for column in df.columns]
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def fetch_nasdaq_history(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    params = {
        "assetclass": "stocks",
        "fromdate": start.strftime("%Y-%m-%d"),
        "todate": end.strftime("%Y-%m-%d"),
        "limit": 9999,
    }
    url = f"{NASDAQ_HISTORICAL_URL.format(symbol=normalize_symbol(symbol))}?{urlencode(params)}"
    payload = fetch_json_with_curl(url)
    rows = payload.get("data", {}).get("tradesTable", {}).get("rows", [])
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(
        {
            "date": [row.get("date") for row in rows],
            "open": [parse_number(row.get("open")) for row in rows],
            "high": [parse_number(row.get("high")) for row in rows],
            "low": [parse_number(row.get("low")) for row in rows],
            "close": [parse_number(row.get("close")) for row in rows],
            "volume": [parse_number(row.get("volume")) for row in rows],
        }
    )
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y", utc=True, errors="coerce")
    return normalize_ohlcv(frame)


def fetch_yahoo_history(session: requests.Session, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    params = {
        "period1": int(start.timestamp()),
        "period2": int(end.timestamp()),
        "interval": "1d",
        "includeAdjustedClose": "true",
        "events": "div,splits",
    }
    url = f"{YAHOO_CHART_URL.format(symbol=yahoo_symbol(symbol))}?{urlencode(params)}"
    command = [
        "curl",
        "-sS",
        "-L",
        "--max-time",
        "60",
        "-A",
        "Mozilla/5.0",
        "--compressed",
        url,
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    payload = json.loads(result.stdout)
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart error for {symbol}: {chart['error']}")
    result = chart.get("result") or []
    if not result:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    result0 = result[0]
    timestamps = result0.get("timestamp") or []
    quotes = result0.get("indicators", {}).get("quote", [])
    if not timestamps or not quotes:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    quote = quotes[0]
    return normalize_ohlcv(
        pd.DataFrame(
            {
                "date": pd.to_datetime(timestamps, unit="s", utc=True),
                "open": quote.get("open", []),
                "high": quote.get("high", []),
                "low": quote.get("low", []),
                "close": quote.get("close", []),
                "volume": quote.get("volume", []),
            }
        )
    )


def load_or_fetch_symbol(
    session: requests.Session,
    raw_dir: Path,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    skip_fetch: bool,
) -> pd.DataFrame:
    path = output_path_for(raw_dir, symbol)
    if skip_fetch:
        if not path.exists():
            return pd.DataFrame()
        return normalize_ohlcv(pd.read_feather(path))
    try:
        df = fetch_nasdaq_history(symbol, start, end)
    except Exception:
        df = fetch_yahoo_history(session, symbol, start, end)
    if not df.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_feather(path)
    return df


def add_symbol_features(df: pd.DataFrame, symbol: str, weight_pct: float | None) -> pd.DataFrame:
    out = normalize_ohlcv(df)
    out["date"] = out["date"].dt.normalize()
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    out["symbol"] = symbol
    out["weight_pct"] = weight_pct
    out["ret_1d"] = out["close"].pct_change(fill_method=None)
    out["ret_5d"] = out["close"].pct_change(5, fill_method=None)
    out["ret_20d"] = out["close"].pct_change(20, fill_method=None)
    out["ma20"] = out["close"].rolling(20, min_periods=10).mean()
    out["ma50"] = out["close"].rolling(50, min_periods=25).mean()
    out["ma200"] = out["close"].rolling(200, min_periods=100).mean()
    out["high20"] = out["close"].rolling(20, min_periods=10).max()
    out["low20"] = out["close"].rolling(20, min_periods=10).min()
    out["high60"] = out["close"].rolling(60, min_periods=30).max()
    out["low60"] = out["close"].rolling(60, min_periods=30).min()
    out["above_ma20"] = (out["close"] > out["ma20"]).where(out["ma20"].notna(), pd.NA)
    out["above_ma50"] = (out["close"] > out["ma50"]).where(out["ma50"].notna(), pd.NA)
    out["above_ma200"] = (out["close"] > out["ma200"]).where(out["ma200"].notna(), pd.NA)
    out["new_high_20"] = (out["close"] >= out["high20"]).where(out["high20"].notna(), pd.NA)
    out["new_low_20"] = (out["close"] <= out["low20"]).where(out["low20"].notna(), pd.NA)
    out["new_high_60"] = (out["close"] >= out["high60"]).where(out["high60"].notna(), pd.NA)
    out["new_low_60"] = (out["close"] <= out["low60"]).where(out["low60"].notna(), pd.NA)
    return out


def weighted_pct(group: pd.DataFrame, column: str) -> float | None:
    weights = pd.to_numeric(group["weight_pct"], errors="coerce")
    valid = group[column].notna() & weights.notna() & (weights > 0.0)
    if not valid.any():
        return None
    return float((group.loc[valid, column].astype(float) * weights.loc[valid]).sum() / weights.loc[valid].sum() * 100.0)


def mean_pct(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return None
    return float(numeric.mean(skipna=True) * 100.0)


def safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def safe_median(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.median())


def safe_std(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 2:
        return None
    return float(numeric.std())


def aggregate_breadth(long_frame: pd.DataFrame, total_symbols: int) -> pd.DataFrame:
    if long_frame.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    top10_symbols = set(
        long_frame[["symbol", "weight_pct"]]
        .drop_duplicates("symbol")
        .dropna(subset=["weight_pct"])
        .sort_values("weight_pct", ascending=False)
        .head(10)["symbol"]
    )
    top10_weight = (
        long_frame[["symbol", "weight_pct"]]
        .drop_duplicates("symbol")
        .dropna(subset=["weight_pct"])
        .sort_values("weight_pct", ascending=False)
        .head(10)["weight_pct"]
        .sum()
    )
    for date, group in long_frame.groupby("date", sort=True):
        valid_close = group["close"].notna()
        count = int(valid_close.sum())
        weighted_abs_contribution = (group["ret_1d"].abs() * group["weight_pct"]).replace([pd.NA], 0).sum(skipna=True)
        top10 = group[group["symbol"].isin(top10_symbols)]
        top10_abs_contribution = (top10["ret_1d"].abs() * top10["weight_pct"]).replace([pd.NA], 0).sum(skipna=True)
        advancers = (group["ret_1d"] > 0.0).where(group["ret_1d"].notna(), pd.NA)
        records.append(
            {
                "date": date,
                "qqq_breadth_constituent_count": count,
                "qqq_breadth_data_coverage_pct": count / total_symbols * 100.0 if total_symbols else None,
                "qqq_breadth_advancers_pct": mean_pct(advancers),
                "qqq_breadth_above_ma20_pct": mean_pct(group["above_ma20"]),
                "qqq_breadth_above_ma50_pct": mean_pct(group["above_ma50"]),
                "qqq_breadth_above_ma200_pct": mean_pct(group["above_ma200"]),
                "qqq_breadth_new_high_20_pct": mean_pct(group["new_high_20"]),
                "qqq_breadth_new_low_20_pct": mean_pct(group["new_low_20"]),
                "qqq_breadth_new_high_60_pct": mean_pct(group["new_high_60"]),
                "qqq_breadth_new_low_60_pct": mean_pct(group["new_low_60"]),
                "qqq_breadth_median_ret_5d": safe_median(group["ret_5d"]),
                "qqq_breadth_ret20_dispersion": safe_std(group["ret_20d"]),
                "qqq_breadth_weighted_above_ma20_pct": weighted_pct(group, "above_ma20"),
                "qqq_breadth_weighted_above_ma50_pct": weighted_pct(group, "above_ma50"),
                "qqq_breadth_weighted_above_ma200_pct": weighted_pct(group, "above_ma200"),
                "qqq_breadth_top10_weight_pct": float(top10_weight) if pd.notna(top10_weight) else None,
                "qqq_breadth_top10_abs_contribution_share_pct": (
                    float(top10_abs_contribution / weighted_abs_contribution * 100.0)
                    if weighted_abs_contribution and weighted_abs_contribution > 0.0
                    else None
                ),
            }
        )
    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    session = request_session()
    start = normalize_timestamp(args.start)
    end = normalize_timestamp(args.end, default_now=True)
    raw_dir = Path(args.raw_dir)
    universe, flags = build_universe(args, session)
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for _, row in universe.iterrows():
        symbol = str(row["symbol"])
        try:
            df = load_or_fetch_symbol(session, raw_dir, symbol, start, end, bool(args.skip_fetch))
            if df.empty:
                failures.append(f"{symbol}:empty")
                continue
            frames.append(add_symbol_features(df, symbol, row.get("weight_pct")))
            print(f"{symbol}: {len(df)} rows", flush=True)
            time.sleep(max(float(args.sleep_seconds), 0.0))
        except Exception as exc:
            failures.append(f"{symbol}:{type(exc).__name__}:{exc}")
    long_frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    breadth = aggregate_breadth(long_frame, total_symbols=len(universe))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not breadth.empty:
        breadth.to_feather(output)
    report = {
        "generated_at_utc": str(pd.Timestamp.utcnow()),
        "source": {
            "universe": NASDAQ100_URL if not args.symbols else "manual",
            "weights": str(args.holdings_csv or args.weights_url),
            "prices": "Nasdaq historical API with Yahoo Finance chart API fallback",
        },
        "symbols_requested": int(len(universe)),
        "symbols_succeeded": int(len(frames)),
        "failures": failures,
        "flags": flags,
        "output": str(output),
        "rows": int(len(breadth)),
        "start": str(pd.Timestamp(breadth["date"].min()).date()) if not breadth.empty else None,
        "end": str(pd.Timestamp(breadth["date"].max()).date()) if not breadth.empty else None,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(report_path)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
