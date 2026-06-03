#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "public" / "macro" / "fred_macro-1d.feather"
DEFAULT_REPORT = ROOT / "var" / "reports" / "fred_macro_fetch.json"
FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"


@dataclass(frozen=True)
class FredSeries:
    series_id: str
    column: str
    release_lag_days: int
    description: str


SERIES: list[FredSeries] = [
    FredSeries("DFF", "macro_fed_funds_effective", 0, "Effective federal funds rate"),
    FredSeries("DGS2", "macro_treasury_2y", 0, "2-year Treasury yield"),
    FredSeries("DGS10", "macro_treasury_10y", 0, "10-year Treasury yield"),
    FredSeries("T10Y2Y", "macro_treasury_10y_2y_spread", 0, "10-year minus 2-year Treasury spread"),
    FredSeries("DFII10", "macro_real_yield_10y", 0, "10-year TIPS real yield"),
    FredSeries("SOFR", "macro_sofr", 0, "Secured Overnight Financing Rate"),
    FredSeries("VIXCLS", "macro_vix_fred", 0, "CBOE VIX close from FRED"),
    FredSeries("DTWEXBGS", "macro_broad_dollar_index", 0, "Trade weighted broad dollar index"),
    FredSeries("DCOILWTICO", "macro_wti_oil", 0, "WTI crude oil price"),
    FredSeries("BAMLH0A0HYM2", "macro_high_yield_oas", 0, "US high yield option-adjusted spread"),
    FredSeries("ICSA", "macro_initial_claims", 5, "Initial unemployment claims"),
    FredSeries("CPIAUCSL", "macro_cpi", 15, "CPI all urban consumers"),
    FredSeries("CPILFESL", "macro_core_cpi", 15, "Core CPI"),
    FredSeries("PCEPI", "macro_pce", 30, "PCE price index"),
    FredSeries("PCEPILFE", "macro_core_pce", 30, "Core PCE price index"),
    FredSeries("UNRATE", "macro_unemployment_rate", 7, "Unemployment rate"),
    FredSeries("PAYEMS", "macro_nonfarm_payrolls", 7, "Nonfarm payroll employment"),
    FredSeries("CES0500000003", "macro_avg_hourly_earnings", 7, "Average hourly earnings"),
    FredSeries("GDPC1", "macro_real_gdp", 35, "Real GDP"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch FRED macro indicators into a daily available-as-of feature table.")
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--proxy", default=None, help="HTTP proxy URL, e.g. http://127.0.0.1:7892")
    parser.add_argument("--api-key", default=os.environ.get("FRED_API_KEY"))
    return parser.parse_args()


def fred_csv_url(series_id: str, start: str) -> str:
    return f"{FRED_GRAPH_URL}?{urlencode({'id': series_id, 'cosd': start})}"


def fetch_series_graph(config: FredSeries, start: str, proxy: str | None = None) -> pd.DataFrame:
    url = fred_csv_url(config.series_id, start)
    if proxy:
        command = [
            "curl",
            "--http1.1",
            "-sS",
            "-L",
            "--max-time",
            "45",
            "-A",
            "Mozilla/5.0",
            "--proxy",
            proxy,
            url,
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        csv_text = result.stdout
    else:
        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        csv_text = response.text
    df = pd.read_csv(io.StringIO(csv_text), na_values=[".", ""])
    df.columns = [str(column).strip() for column in df.columns]
    if "observation_date" not in df.columns or config.series_id not in df.columns:
        raise ValueError(f"Unexpected FRED CSV columns for {config.series_id}: {df.columns.tolist()}")
    out = df[["observation_date", config.series_id]].copy()
    out["observation_date"] = pd.to_datetime(out["observation_date"], utc=True, errors="coerce")
    out[config.column] = pd.to_numeric(out[config.series_id], errors="coerce")
    out = out.drop(columns=[config.series_id]).dropna(subset=["observation_date"])
    out["date"] = out["observation_date"] + pd.to_timedelta(config.release_lag_days, unit="D")
    out["date"] = out["date"].dt.normalize()
    out = out.drop(columns=["observation_date"]).drop_duplicates("date", keep="last").sort_values("date")
    return out[["date", config.column]]


def fetch_series_api(config: FredSeries, start: str, api_key: str, proxy: str | None = None) -> pd.DataFrame:
    params = {
        "series_id": config.series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    }
    if proxy:
        command = [
            "curl",
            "--http1.1",
            "-sS",
            "-L",
            "--max-time",
            "45",
            "-A",
            "Mozilla/5.0",
            "--proxy",
            proxy,
            f"{FRED_API_URL}?{urlencode(params)}",
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        payload = json.loads(result.stdout)
    else:
        response = requests.get(
            FRED_API_URL,
            params=params,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        payload = response.json()
    if payload.get("error_code"):
        raise RuntimeError(f"FRED API error for {config.series_id}: {payload.get('error_message')}")
    observations = payload.get("observations") or []
    frame = pd.DataFrame(
        {
            "observation_date": [item.get("date") for item in observations],
            config.series_id: [item.get("value") for item in observations],
        }
    )
    frame[config.series_id] = frame[config.series_id].replace(".", pd.NA)
    frame.columns = [str(column).strip() for column in frame.columns]
    out = frame[["observation_date", config.series_id]].copy()
    out["observation_date"] = pd.to_datetime(out["observation_date"], utc=True, errors="coerce")
    out[config.column] = pd.to_numeric(out[config.series_id], errors="coerce")
    out = out.drop(columns=[config.series_id]).dropna(subset=["observation_date"])
    out["date"] = out["observation_date"] + pd.to_timedelta(config.release_lag_days, unit="D")
    out["date"] = out["date"].dt.normalize()
    out = out.drop(columns=["observation_date"]).drop_duplicates("date", keep="last").sort_values("date")
    return out[["date", config.column]]


def fetch_series(config: FredSeries, start: str, proxy: str | None = None, api_key: str | None = None) -> pd.DataFrame:
    if api_key:
        return fetch_series_api(config, start, api_key, proxy)
    return fetch_series_graph(config, start, proxy)


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if {"macro_treasury_10y", "macro_treasury_2y"}.issubset(out.columns):
        out["macro_treasury_10y_minus_2y_calc"] = out["macro_treasury_10y"] - out["macro_treasury_2y"]
    if {"macro_fed_funds_effective", "macro_treasury_10y"}.issubset(out.columns):
        out["macro_fed_funds_minus_10y"] = out["macro_fed_funds_effective"] - out["macro_treasury_10y"]
    if {"macro_high_yield_oas", "macro_treasury_10y"}.issubset(out.columns):
        out["macro_hy_oas_plus_10y"] = out["macro_high_yield_oas"] + out["macro_treasury_10y"]

    yoy_columns = [
        "macro_cpi",
        "macro_core_cpi",
        "macro_pce",
        "macro_core_pce",
        "macro_avg_hourly_earnings",
        "macro_real_gdp",
    ]
    for column in yoy_columns:
        if column in out.columns:
            periods = 365
            out[f"{column}_yoy"] = out[column].pct_change(periods=periods, fill_method=None) * 100.0
    if "macro_nonfarm_payrolls" in out.columns:
        out["macro_nonfarm_payrolls_chg"] = out["macro_nonfarm_payrolls"].diff()
    if "macro_initial_claims" in out.columns:
        out["macro_initial_claims_chg_4w"] = out["macro_initial_claims"].diff(28)
    return out


def build_daily_macro(series_frames: list[pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    calendar = pd.DataFrame({"date": pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")})
    out = calendar
    for frame in series_frames:
        out = out.merge(frame, on="date", how="left")
    value_columns = [column for column in out.columns if column != "date"]
    out[value_columns] = out[value_columns].ffill()
    out = add_derived_columns(out)
    out = out.replace([float("inf"), float("-inf")], pd.NA)
    return out


def json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return str(value)
    return str(value)


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC")
    frames: list[pd.DataFrame] = []
    loaded: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    source_mode = "fred_api" if args.api_key else "fred_graph"

    for config in SERIES:
        try:
            frame = fetch_series(config, args.start, args.proxy, args.api_key)
            frames.append(frame)
            loaded.append(
                {
                    "series_id": config.series_id,
                    "column": config.column,
                    "release_lag_days": str(config.release_lag_days),
                    "description": config.description,
                }
            )
            print(f"{config.series_id}: {len(frame)} observations", flush=True)
        except Exception as exc:
            failures.append({"series_id": config.series_id, "error": f"{type(exc).__name__}:{exc}"})

    if not frames:
        raise RuntimeError("No FRED series were loaded.")

    macro = build_daily_macro(frames, start, end)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    macro.to_feather(output)

    report = {
        "generated_at_utc": str(pd.Timestamp.utcnow()),
        "source": FRED_API_URL if args.api_key else FRED_GRAPH_URL,
        "source_mode": source_mode,
        "proxy": str(args.proxy) if args.proxy else None,
        "rows": int(len(macro)),
        "start": str(pd.Timestamp(macro["date"].min()).date()),
        "end": str(pd.Timestamp(macro["date"].max()).date()),
        "columns": [column for column in macro.columns if column != "date"],
        "loaded": loaded,
        "failures": failures,
        "output": str(output),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=json_default))
    print(report_path)
    print(json.dumps({"rows": report["rows"], "columns": len(report["columns"]), "failures": failures}, ensure_ascii=False))


if __name__ == "__main__":
    main()
