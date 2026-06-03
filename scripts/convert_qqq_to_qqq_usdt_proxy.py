#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "public" / "etf_full" / "QQQ-1d.feather"
DEFAULT_FALLBACK_SOURCE = ROOT / "data" / "public" / "etf_long" / "QQQ-1d.feather"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "proxy" / "qqq_usdt"
DEFAULT_REPORT = ROOT / "var" / "reports" / "qqq_to_qqq_usdt_proxy_conversion.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert QQQ ETF daily OHLCV into OKX-style synthetic QQQ/USDT proxy files."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--fallback-source", default=str(DEFAULT_FALLBACK_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument(
        "--compat-timeframes",
        default="4h,1h",
        help="Comma-separated lower-timeframe compatibility files to write from daily bars. Empty disables them.",
    )
    parser.add_argument("--write-zero-funding", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def file_name(timeframe: str, suffix: str = "proxy-long") -> str:
    if timeframe == "funding":
        return f"QQQ_USDT_USDT-8h-funding_rate-zero-{suffix}.feather"
    return f"QQQ_USDT_USDT-{timeframe}-futures-{suffix}.feather"


def load_qqq_daily(source: Path, fallback_source: Path) -> tuple[pd.DataFrame, Path]:
    path = source if source.exists() else fallback_source
    if not path.exists():
        raise FileNotFoundError(f"No QQQ source found: {source} or {fallback_source}")
    df = pd.read_feather(path)
    df.columns = [str(column).strip().lower() for column in df.columns]
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    df["session_day"] = df["date"].dt.normalize()
    df = df.sort_values("date").drop_duplicates("session_day", keep="last")
    out = df[["session_day", "open", "high", "low", "close", "volume"]].rename(columns={"session_day": "date"})
    out = out.sort_values("date").reset_index(drop=True)
    return out[["date", "open", "high", "low", "close", "volume"]], path


def write_ohlcv(df: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out = out[["date", "open", "high", "low", "close", "volume"]]
    out.to_feather(output_path)
    return {
        "path": str(output_path),
        "rows": int(len(out)),
        "start": str(pd.Timestamp(out["date"].min()).date()) if not out.empty else None,
        "end": str(pd.Timestamp(out["date"].max()).date()) if not out.empty else None,
        "columns": out.columns.tolist(),
    }


def build_zero_funding(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["date", "funding_rate", "fundingRate"])
    start = pd.Timestamp(daily["date"].min()).floor("D")
    end = pd.Timestamp(daily["date"].max()).ceil("D")
    dates = pd.date_range(start, end, freq="8h", tz="UTC")
    return pd.DataFrame({"date": dates, "funding_rate": 0.0, "fundingRate": 0.0})


def json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return str(value)
    return str(value)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    daily, source_used = load_qqq_daily(Path(args.source), Path(args.fallback_source))

    outputs: dict[str, Any] = {}
    outputs["1d"] = write_ohlcv(daily, output_dir / file_name("1d"))

    for timeframe in parse_csv_list(args.compat_timeframes):
        outputs[timeframe] = write_ohlcv(daily, output_dir / file_name(timeframe, suffix="dailyproxy-long"))

    if args.write_zero_funding:
        funding = build_zero_funding(daily)
        funding_path = output_dir / file_name("funding")
        funding_path.parent.mkdir(parents=True, exist_ok=True)
        funding.to_feather(funding_path)
        outputs["zero_funding_8h"] = {
            "path": str(funding_path),
            "rows": int(len(funding)),
            "start": str(pd.Timestamp(funding["date"].min())) if not funding.empty else None,
            "end": str(pd.Timestamp(funding["date"].max())) if not funding.empty else None,
            "columns": funding.columns.tolist(),
        }

    payload = {
        "generated_at_utc": str(pd.Timestamp.utcnow()),
        "mode": "qqq_to_qqq_usdt_proxy_conversion",
        "source_used": str(source_used),
        "output_dir": str(output_dir),
        "assumptions": [
            "QQQ USD prices are treated as QQQ/USDT proxy prices at 1 USD = 1 USDT.",
            "Volume is ETF share volume, not QQQ/USDT contract volume.",
            "The 1d file preserves real ETF daily OHLCV after session-day normalization to 00:00 UTC.",
            "Compatibility 4h/1h files are one ETF daily bar per trading day, not real intraday bars.",
            "Zero funding is a neutral proxy and must not be interpreted as historical QQQ/USDT funding.",
        ],
        "outputs": outputs,
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default))
    print(report)
    print(json.dumps(outputs, ensure_ascii=False))


if __name__ == "__main__":
    main()
