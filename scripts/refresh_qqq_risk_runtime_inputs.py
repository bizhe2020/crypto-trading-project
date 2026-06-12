#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests import HTTPError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_public_etf_history import fetch_timeframe, output_path_for  # noqa: E402
from scripts.qqq_risk_runtime_generation import write_summary  # noqa: E402


DEFAULT_ETF_OUTPUT_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_BREADTH_OUTPUT = ROOT / "data" / "public" / "breadth" / "qqq_constituent_breadth-1d.feather"
DEFAULT_MACRO_OUTPUT = ROOT / "data" / "public" / "macro" / "fred_macro-1d.feather"
DEFAULT_RUNTIME_DIR = ROOT / "var" / "runtime" / "qqq_risk"
DEFAULT_RECENT_CSV = DEFAULT_RUNTIME_DIR / "qqq_recent_risk_runtime_predictions.csv"
DEFAULT_LONG_CSV = DEFAULT_RUNTIME_DIR / "qqq_long_cycle_risk_runtime_predictions.csv"
DEFAULT_RECENT_REPORT = DEFAULT_RUNTIME_DIR / "qqq_recent_risk_runtime_report.json"
DEFAULT_LONG_REPORT = DEFAULT_RUNTIME_DIR / "qqq_long_cycle_risk_runtime_report.json"
DEFAULT_SUMMARY = DEFAULT_RUNTIME_DIR / "qqq_risk_runtime_refresh_summary.json"

RISK_ETF_SYMBOLS = [
    "QQQ",
    "SPY",
    "^IXIC",
    "^VIX",
    "^VIX3M",
    "^VVIX",
    "^SKEW",
    "QQEW",
    "RSP",
    "SMH",
    "XLY",
    "XLP",
    "HYG",
    "IEF",
    "LQD",
    "TLT",
    "GLD",
    "USO",
    "CPER",
    "UUP",
    "BTC-USD",
]

ETF_START_FALLBACKS = [
    "2006-01-01T00:00:00Z",
    "2010-01-01T00:00:00Z",
    "2014-01-01T00:00:00Z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh supporting data and regenerate live QQQ risk CSV inputs.")
    parser.add_argument("--etf-start", default="1999-01-01T00:00:00Z")
    parser.add_argument("--breadth-start", default="2020-01-01T00:00:00Z")
    parser.add_argument("--macro-start", default="2017-01-01")
    parser.add_argument("--end", default=None, help="Optional UTC ISO timestamp / YYYY-MM-DD limit.")
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--fred-api-key", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--etf-output-dir", default=str(DEFAULT_ETF_OUTPUT_DIR))
    parser.add_argument("--breadth-output", default=str(DEFAULT_BREADTH_OUTPUT))
    parser.add_argument("--macro-output", default=str(DEFAULT_MACRO_OUTPUT))
    parser.add_argument("--recent-csv", default=str(DEFAULT_RECENT_CSV))
    parser.add_argument("--long-csv", default=str(DEFAULT_LONG_CSV))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--max-output-lag-days", type=int, default=5)
    parser.add_argument("--macro-refresh-mode", choices=["auto", "market_proxy"], default="auto")
    parser.add_argument("--skip-etf", action="store_true")
    parser.add_argument("--skip-breadth", action="store_true")
    parser.add_argument("--skip-macro", action="store_true")
    parser.add_argument("--skip-recent", action="store_true")
    parser.add_argument("--skip-long-cycle", action="store_true")
    return parser.parse_args()


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def normalize_utc_timestamp(value: str | pd.Timestamp | None, *, default_now: bool = False) -> pd.Timestamp:
    if value is None:
        if default_now:
            return utc_now()
        raise ValueError("timestamp value is required when default_now is false")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def run_command(cmd: list[str]) -> dict[str, Any]:
    started = utc_now()
    subprocess.run(cmd, cwd=ROOT, check=True)
    finished = utc_now()
    return {
        "command": cmd,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "duration_seconds": round((finished - started).total_seconds(), 3),
    }


def refresh_etf_data(args: argparse.Namespace) -> dict[str, Any]:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 qqq-risk-runtime-refresh"})
    if args.proxy:
        session.proxies.update({"http": str(args.proxy), "https": str(args.proxy)})
    output_dir = Path(args.etf_output_dir)
    failures: list[dict[str, Any]] = []
    refreshed: list[dict[str, Any]] = []
    started = utc_now()
    for symbol in RISK_ETF_SYMBOLS:
        output_path = output_path_for(output_dir, symbol, "1d")
        try:
            frame, metadata = refresh_etf_symbol(
                session=session,
                symbol=symbol,
                start=str(args.etf_start),
                end=args.end,
                output_path=output_path,
                sleep_seconds=float(args.sleep_seconds),
                proxy=args.proxy,
                timeout_seconds=float(args.timeout_seconds),
            )
            refreshed.append(
                {
                    "symbol": symbol,
                    "rows": int(len(frame)),
                    "latest": str(frame["date"].max()) if not frame.empty else None,
                    "path": str(output_path),
                    **metadata,
                }
            )
        except Exception as exc:
            failures.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
    finished = utc_now()
    if failures:
        raise RuntimeError(f"ETF refresh failed for {len(failures)} symbols: {failures}")
    return {
        "symbols": refreshed,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "duration_seconds": round((finished - started).total_seconds(), 3),
    }


def _is_bad_request(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        response = getattr(exc, "response", None)
        return response is not None and int(response.status_code) == 400
    text = str(exc)
    return "400 Client Error" in text or "HTTP Error 400" in text or "Bad Request" in text


def refresh_etf_symbol(
    *,
    session: requests.Session,
    symbol: str,
    start: str,
    end: str | None,
    output_path: Path,
    sleep_seconds: float,
    proxy: str | None,
    timeout_seconds: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    attempted_starts: list[str] = []
    candidates = [str(start)] + [item for item in ETF_START_FALLBACKS if pd.Timestamp(item) > pd.Timestamp(start)]
    last_exc: Exception | None = None
    for candidate_start in candidates:
        attempted_starts.append(candidate_start)
        try:
            frame = fetch_timeframe(
                session=session,
                symbol=symbol,
                timeframe="1d",
                start=candidate_start,
                end=end,
                output_path=output_path,
                sleep_seconds=sleep_seconds,
                proxy=proxy,
                timeout_seconds=timeout_seconds,
            )
            metadata: dict[str, Any] = {"requested_start": str(start), "used_start": candidate_start}
            if candidate_start != str(start):
                metadata["start_fallback_applied"] = True
                metadata["attempted_starts"] = attempted_starts
            return frame, metadata
        except Exception as exc:
            last_exc = exc
            if candidate_start == candidates[-1] or not _is_bad_request(exc):
                raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"ETF refresh failed without attempting symbol fetch: {symbol}")


def latest_csv_date(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, usecols=["date"])
    if frame.empty:
        return None
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.max())


def validate_output_lag(
    path: Path,
    *,
    max_lag_days: int,
    reference_time: pd.Timestamp | str | None = None,
) -> dict[str, Any]:
    latest = latest_csv_date(path)
    current = normalize_utc_timestamp(reference_time, default_now=reference_time is None)
    lag_days = None if latest is None else int((current.normalize() - latest.normalize()).days)
    valid = latest is not None and lag_days is not None and lag_days <= int(max_lag_days)
    payload = {
        "path": str(path),
        "latest": str(latest) if latest is not None else None,
        "reference_time": str(current),
        "lag_days": lag_days,
        "max_lag_days": int(max_lag_days),
        "valid": bool(valid),
    }
    if not valid:
        raise RuntimeError(f"Refreshed risk CSV is still stale: {payload}")
    return payload


def run_step(summary: dict[str, Any], step_name: str, fn: Any) -> dict[str, Any]:
    started = utc_now()
    try:
        raw = fn()
    except Exception as exc:
        finished = utc_now()
        summary["steps"][step_name] = {
            "status": "error",
            "started_at_utc": started,
            "finished_at_utc": finished,
            "duration_seconds": round((finished - started).total_seconds(), 3),
            "error": f"{type(exc).__name__}: {exc}",
        }
        raise
    payload = dict(raw) if isinstance(raw, dict) else {"result": raw}
    payload.setdefault("status", "ok")
    payload.setdefault("started_at_utc", started)
    payload.setdefault("finished_at_utc", utc_now())
    payload.setdefault(
        "duration_seconds",
        round((pd.Timestamp(payload["finished_at_utc"]) - pd.Timestamp(payload["started_at_utc"])).total_seconds(), 3),
    )
    summary["steps"][step_name] = payload
    return payload


def build_market_proxy_macro_file(*, etf_dir: Path, output_path: Path) -> dict[str, Any]:
    uup_path = output_path_for(etf_dir, "UUP", "1d")
    uso_path = output_path_for(etf_dir, "USO", "1d")
    if not uup_path.exists():
        raise FileNotFoundError(f"UUP ETF proxy not found: {uup_path}")
    if not uso_path.exists():
        raise FileNotFoundError(f"USO ETF proxy not found: {uso_path}")

    uup = pd.read_feather(uup_path)[["date", "close"]].rename(columns={"close": "macro_broad_dollar_index"})
    uso = pd.read_feather(uso_path)[["date", "close"]].rename(columns={"close": "macro_wti_oil"})
    uup["date"] = pd.to_datetime(uup["date"], utc=True, errors="coerce")
    uso["date"] = pd.to_datetime(uso["date"], utc=True, errors="coerce")
    macro = uup.merge(uso, on="date", how="outer").sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    macro = macro.dropna(subset=["date"])
    macro[["macro_broad_dollar_index", "macro_wti_oil"]] = macro[
        ["macro_broad_dollar_index", "macro_wti_oil"]
    ].ffill()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    macro.to_feather(output_path)
    return {
        "source_mode": "market_proxy",
        "output": str(output_path),
        "rows": int(len(macro)),
        "start": str(pd.Timestamp(macro["date"].min()).date()) if not macro.empty else None,
        "end": str(pd.Timestamp(macro["date"].max()).date()) if not macro.empty else None,
        "proxy_columns": {
            "macro_broad_dollar_index": str(uup_path),
            "macro_wti_oil": str(uso_path),
        },
    }


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json)
    validation_reference = normalize_utc_timestamp(args.end) if args.end else None
    summary: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "mode": "qqq_risk_runtime_refresh",
        "project_root": str(ROOT),
        "status": "running",
        "config": {
            "etf_start": str(args.etf_start),
            "breadth_start": str(args.breadth_start),
            "macro_start": str(args.macro_start),
            "end": args.end,
            "proxy": args.proxy,
            "fred_api_key_configured": bool(args.fred_api_key),
            "macro_refresh_mode": str(args.macro_refresh_mode),
            "max_output_lag_days": int(args.max_output_lag_days),
            "recent_csv": str(args.recent_csv),
            "long_csv": str(args.long_csv),
            "summary_json": str(args.summary_json),
        },
        "steps": {},
    }
    try:
        if not args.skip_etf:
            run_step(summary, "etf_refresh", lambda: refresh_etf_data(args))

        if not args.skip_breadth:
            breadth_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "fetch_qqq_constituent_breadth.py"),
                "--start",
                str(args.breadth_start),
                "--output",
                str(args.breadth_output),
            ]
            if args.end:
                breadth_cmd.extend(["--end", str(args.end)])
            run_step(summary, "breadth_refresh", lambda: run_command(breadth_cmd))

        if not args.skip_macro:
            if str(args.macro_refresh_mode) == "market_proxy":
                run_step(
                    summary,
                    "macro_refresh_fallback",
                    lambda: build_market_proxy_macro_file(
                        etf_dir=Path(args.etf_output_dir),
                        output_path=Path(args.macro_output),
                    ),
                )
            else:
                macro_cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "fetch_fred_macro_indicators.py"),
                    "--start",
                    str(args.macro_start),
                    "--output",
                    str(args.macro_output),
                    "--report",
                    str(ROOT / "var" / "reports" / "fred_macro_fetch_runtime.json"),
                ]
                if args.end:
                    macro_cmd.extend(["--end", str(args.end)])
                if args.proxy:
                    macro_cmd.extend(["--proxy", str(args.proxy)])
                if args.fred_api_key:
                    macro_cmd.extend(["--api-key", str(args.fred_api_key)])
                try:
                    run_step(summary, "macro_refresh", lambda: run_command(macro_cmd))
                except Exception:
                    run_step(
                        summary,
                        "macro_refresh_fallback",
                        lambda: build_market_proxy_macro_file(
                            etf_dir=Path(args.etf_output_dir),
                            output_path=Path(args.macro_output),
                        ),
                    )

        if not args.skip_recent:
            run_step(
                summary,
                "recent_risk_csv",
                lambda: run_command(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "generate_qqq_recent_risk_csv.py"),
                        "--output-csv",
                        str(args.recent_csv),
                        "--output-json",
                        str(DEFAULT_RECENT_REPORT),
                    ]
                ),
            )
            run_step(
                summary,
                "recent_validation",
                lambda: validate_output_lag(
                    Path(args.recent_csv),
                    max_lag_days=int(args.max_output_lag_days),
                    reference_time=validation_reference,
                ),
            )

        if not args.skip_long_cycle:
            run_step(
                summary,
                "long_cycle_risk_csv",
                lambda: run_command(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "generate_qqq_long_cycle_risk_csv.py"),
                        "--output-csv",
                        str(args.long_csv),
                        "--output-json",
                        str(DEFAULT_LONG_REPORT),
                    ]
                ),
            )
            run_step(
                summary,
                "long_cycle_validation",
                lambda: validate_output_lag(
                    Path(args.long_csv),
                    max_lag_days=int(args.max_output_lag_days),
                    reference_time=validation_reference,
                ),
            )
        summary["status"] = "ok"
    except Exception as exc:
        summary["status"] = "error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["completed_at_utc"] = utc_now()
        write_summary(summary_path, summary)
        raise

    summary["completed_at_utc"] = utc_now()
    write_summary(summary_path, summary)
    print(summary_path)
    print(
        json.dumps(
            {
                "recent_latest": summary["steps"].get("recent_validation", {}).get("latest"),
                "long_cycle_latest": summary["steps"].get("long_cycle_validation", {}).get("latest"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
