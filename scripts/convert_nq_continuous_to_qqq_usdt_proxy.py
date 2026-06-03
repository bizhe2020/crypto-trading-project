#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NQ_SOURCE = ROOT / "data" / "public" / "futures_full" / "NQ=F-1d.feather"
DEFAULT_QQQ_ANCHOR = ROOT / "data" / "public" / "etf_full" / "QQQ-1d.feather"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "proxy" / "qqq_usdt_nq_continuous"
DEFAULT_REPORT = ROOT / "var" / "reports" / "nq_continuous_to_qqq_usdt_proxy_conversion.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Yahoo NQ=F continuous futures into a QQQ/USDT-scaled proxy dataset."
    )
    parser.add_argument("--nq-source", default=str(DEFAULT_NQ_SOURCE))
    parser.add_argument("--qqq-anchor", default=str(DEFAULT_QQQ_ANCHOR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--compat-timeframes", default="4h,1h")
    parser.add_argument("--write-zero-funding", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def output_name(timeframe: str, suffix: str = "nq-continuous-scaled-long") -> str:
    if timeframe == "funding":
        return f"QQQ_USDT_USDT-8h-funding_rate-zero-{suffix}.feather"
    return f"QQQ_USDT_USDT-{timeframe}-futures-{suffix}.feather"


def load_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
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
    df = df.sort_values("date").drop_duplicates("session_day", keep="last").reset_index(drop=True)
    return df[["session_day", "open", "high", "low", "close", "volume"]].rename(columns={"session_day": "date"})


def scale_nq_to_qqq(nq: pd.DataFrame, qqq: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    common = nq[["date", "close"]].rename(columns={"close": "nq_close"}).merge(
        qqq[["date", "close"]].rename(columns={"close": "qqq_close"}),
        on="date",
        how="inner",
    )
    if common.empty:
        raise ValueError("No common session days between NQ source and QQQ anchor")
    anchor = common.sort_values("date").iloc[-1]
    scale = float(anchor["qqq_close"]) / float(anchor["nq_close"])
    out = nq.copy()
    for column in ["open", "high", "low", "close"]:
        out[column] = out[column] * scale
    out = out[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)

    aligned = common.sort_values("date").copy()
    aligned["nq_ret"] = aligned["nq_close"].pct_change(fill_method=None)
    aligned["qqq_ret"] = aligned["qqq_close"].pct_change(fill_method=None)
    corr = aligned[["nq_ret", "qqq_ret"]].dropna().corr().iloc[0, 1]
    meta = {
        "anchor_date": str(pd.Timestamp(anchor["date"]).date()),
        "anchor_nq_close": round(float(anchor["nq_close"]), 6),
        "anchor_qqq_close": round(float(anchor["qqq_close"]), 6),
        "scale_factor": scale,
        "overlap_rows": int(len(common)),
        "daily_return_correlation_to_qqq": round(float(corr), 6) if pd.notna(corr) else None,
    }
    return out, meta


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
    nq = load_daily(Path(args.nq_source))
    qqq = load_daily(Path(args.qqq_anchor))
    scaled, scale_meta = scale_nq_to_qqq(nq, qqq)
    output_dir = Path(args.output_dir)

    outputs: dict[str, Any] = {}
    outputs["1d"] = write_ohlcv(scaled, output_dir / output_name("1d"))
    for timeframe in parse_csv_list(args.compat_timeframes):
        outputs[timeframe] = write_ohlcv(scaled, output_dir / output_name(timeframe, suffix="nq-continuous-dailyproxy-long"))

    if args.write_zero_funding:
        funding = build_zero_funding(scaled)
        funding_path = output_dir / output_name("funding")
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
        "mode": "nq_continuous_to_qqq_usdt_proxy_conversion",
        "nq_source": str(Path(args.nq_source)),
        "qqq_anchor_source": str(Path(args.qqq_anchor)),
        "output_dir": str(output_dir),
        "scale": scale_meta,
        "assumptions": [
            "Yahoo NQ=F is used as a front-month continuous Nasdaq-100 futures approximation.",
            "NQ OHLC prices are multiplied by a single scale factor so the latest common close matches QQQ.",
            "Scaled prices preserve NQ percentage returns but are not historical QQQ prices.",
            "Volume is NQ source volume, not QQQ/USDT contract volume.",
            "Compatibility 4h/1h files are one scaled daily bar per trading day, not real intraday bars.",
            "Zero funding is a neutral proxy and must not be interpreted as historical QQQ/USDT funding.",
        ],
        "outputs": outputs,
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default))
    print(report)
    print(json.dumps({"scale": scale_meta, "outputs": outputs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
