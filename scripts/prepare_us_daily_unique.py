#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
LOCAL_US = ZoneInfo("America/New_York")
DEFAULT_DATA_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "us_daily_unique_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate US daily ETF/index files by US local trade day.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--symbols", default="QQQ,SPY,^IXIC,^VIX,TQQQ,SQQQ")
    parser.add_argument("--write-clean", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    rows: list[dict[str, object]] = []

    for symbol in symbols:
        path = data_dir / f"{symbol}-1d.feather"
        if not path.exists():
            rows.append({"symbol": symbol, "exists": False})
            continue
        df = pd.read_feather(path)
        original_rows = int(len(df))
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["us_day"] = df["date"].dt.tz_convert(LOCAL_US).dt.date
        deduped = df.sort_values("date").drop_duplicates("us_day", keep="last").reset_index(drop=True)
        duplicate_rows = original_rows - int(len(deduped))
        if args.write_clean and duplicate_rows > 0:
            clean = deduped.drop(columns=["us_day"])
            clean.to_feather(path)
        rows.append(
            {
                "symbol": symbol,
                "exists": True,
                "original_rows": original_rows,
                "deduped_rows": int(len(deduped)),
                "duplicate_rows_removed": duplicate_rows,
                "latest_original_timestamp": str(df["date"].max()) if not df.empty else "",
                "latest_deduped_timestamp": str(deduped["date"].max()) if not deduped.empty else "",
            }
        )

    payload = {"symbols": rows}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
