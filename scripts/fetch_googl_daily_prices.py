#!/usr/bin/env python3
"""GOOGL 正股日线拉取 — 东京服务器自闭环数据源。

用 Yahoo chart API（与 QQQ 风险刷新 scripts/fetch_public_etf_history.py 同源，
服务器已实盘验证可用）拉取 GOOGL 全量日线，写出 scan_googl_daily_signal.py
需要的 prices.csv（列 ticker,date,open,close，单 ticker GOOGL）。

替代原 scripts/sync_googl_value_data_to_tokyo.sh 的本地 scp 同步，
去除对本地 价值投资project 数据目录的依赖。服务器每日 cron 直接拉取。

用法:
    python scripts/fetch_googl_daily_prices.py \
        --out /root/projects/value_data/prices.csv \
        --start 2007-01-01

行为:
    - 从 --start 拉取到当前，覆盖写出 prices.csv（仅 GOOGL 行）。
    - 日期统一规范化为该交易日的 UTC 午夜（与信号层历史格式一致）。
    - 拉取失败时退出码非 0，不覆盖已有文件（refresh 脚本沿用旧信号，
      adapter 的 stale guard 兜底）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
SYMBOL = "GOOGL"
REQUIRED_COLUMNS = ["ticker", "date", "open", "close"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch GOOGL daily OHLC from Yahoo into scan_googl_daily_signal prices.csv format.")
    parser.add_argument("--out", default="var/runtime/googl/prices.csv", help="输出 prices.csv 路径")
    parser.add_argument("--start", default="2007-01-01", help="拉取起始日（UTC 日期，含）")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--user-agent", default="Mozilla/5.0 googl-signal-refresh")
    return parser.parse_args()


def normalize_daily_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """把 Yahoo 日线时间戳规范化为该交易日的 UTC 午夜。

    Yahoo 1d bar 的时间戳是美东开市时刻（夏令时 13:30Z / 冬令时 14:30Z），
    与信号层历史 prices.csv 的 UTC 午夜口径对齐。
    """
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="coerce")
    result = result.dropna(subset=["date"])
    result["date"] = result["date"].dt.normalize()
    return result


def fetch_symbol_daily(
    session: requests.Session,
    symbol: str,
    start: str,
    *,
    timeout_seconds: float,
) -> pd.DataFrame:
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    params = {
        "period1": int(start_ms / 1000),
        "period2": int(end_ms / 1000),
        "interval": "1d",
        "includeAdjustedClose": "true",
        "events": "div,splits",
    }
    url = f"{YAHOO_CHART_URL.format(symbol=symbol.upper())}?{urlencode(params)}"
    response = session.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError(f"Yahoo chart error for {symbol}: {error}")
    result = chart.get("result") or []
    if not result:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    result0 = result[0]
    timestamps = result0.get("timestamp") or []
    indicators = result0.get("indicators", {}).get("quote", [])
    if not timestamps or not indicators:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    quote = indicators[0]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True),
            "open": quote.get("open", []),
            "close": quote.get("close", []),
        }
    )
    frame = normalize_daily_dates(frame)
    frame = frame.dropna(subset=["open", "close"]).copy()
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["open", "close"])
    frame["ticker"] = symbol.upper()
    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return frame[REQUIRED_COLUMNS]


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (ROOT / args.out).resolve()

    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})

    fetched = fetch_symbol_daily(session, SYMBOL, args.start, timeout_seconds=args.timeout_seconds)
    if fetched.empty:
        raise RuntimeError(f"{SYMBOL} fetch returned no rows")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    fetched.to_csv(tmp, index=False)
    tmp.replace(out_path)

    print(
        f"{SYMBOL}: wrote {len(fetched)} rows ({fetched['date'].min().date()} -> {fetched['date'].max().date()}) "
        f"to {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
