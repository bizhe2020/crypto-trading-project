#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_REPORT_DIR = ROOT / "var" / "reports"


@dataclass
class SymbolAudit:
    symbol: str
    path: str
    rows: int
    start: str | None
    end: str | None
    missing_days: int
    yearly_returns: dict[str, float]
    max_drawdown_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit TQQQ/SQQQ/QQQ daily data coverage and baseline trend stats.")
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--public-dir", default=str(DEFAULT_PUBLIC_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    return parser.parse_args()


def load_feather(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.read_feather(path)
    df.columns = [str(column).strip().lower() for column in df.columns]
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)


def missing_days(index: pd.DatetimeIndex) -> int:
    if len(index) <= 1:
        return 0
    normalized = pd.to_datetime(index, utc=True).dt.normalize()
    full = pd.date_range(normalized.min(), normalized.max(), freq="B", tz="UTC")
    present = pd.to_datetime(normalized.unique(), utc=True)
    return int(len(full.difference(present)))


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0, pd.NA) * 100.0
    return float(dd.max(skipna=True) or 0.0)


def yearly_buy_and_hold(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    df = df.copy()
    df["year"] = df["date"].dt.year.astype(str)
    output: dict[str, float] = {}
    for year, group in df.groupby("year"):
        start = float(group.iloc[0]["close"])
        end = float(group.iloc[-1]["close"])
        output[year] = round((end / start - 1.0) * 100.0, 2) if start > 0 else 0.0
    return output


def run_audit(symbol: str, public_dir: Path) -> SymbolAudit:
    path = public_dir / f"{symbol.upper()}-1d.feather"
    df = load_feather(path)
    if df.empty:
        return SymbolAudit(symbol=symbol.upper(), path=str(path), rows=0, start=None, end=None, missing_days=0, yearly_returns={}, max_drawdown_pct=0.0)
    equity = df["close"] / float(df.iloc[0]["close"]) * 100.0
    return SymbolAudit(
        symbol=symbol.upper(),
        path=str(path),
        rows=int(len(df)),
        start=str(df["date"].min()),
        end=str(df["date"].max()),
        missing_days=missing_days(df["date"]),
        yearly_returns=yearly_buy_and_hold(df),
        max_drawdown_pct=round(max_drawdown_pct(equity), 2),
    )


def main() -> None:
    args = parse_args()
    symbols = args.symbol or ["QQQ", "TQQQ", "SQQQ"]
    public_dir = Path(args.public_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    audits = [run_audit(symbol, public_dir) for symbol in symbols]
    report = {
        "symbols": {audit.symbol: asdict(audit) for audit in audits},
    }
    output_path = report_dir / "tqqq_sqqq_data_health_audit.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(output_path)
    for audit in audits:
        print(f"{audit.symbol}: rows={audit.rows} range={audit.start} -> {audit.end} dd={audit.max_drawdown_pct}%")


if __name__ == "__main__":
    main()
