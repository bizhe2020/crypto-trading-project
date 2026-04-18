from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check health of recorded Funding/OI files")
    parser.add_argument("--input-dir", default="var/funding_oi/recorded")
    parser.add_argument("--file-prefix", default="btc_funding_oi")
    parser.add_argument("--bucket-seconds", type=int, default=60)
    parser.add_argument("--warn-gap-minutes", type=float, default=5.0)
    parser.add_argument("--max-files", type=int, default=14)
    return parser.parse_args()


def _iso(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(f"{args.file_prefix}_*.jsonl"))[-max(args.max_files, 1) :]
    if not files:
        print(json.dumps({"status": "empty", "input_dir": str(input_dir)}, ensure_ascii=False, indent=2))
        return

    timestamps: list[int] = []
    duplicate_rows = 0
    missing_mark_price_rows = 0
    missing_index_price_rows = 0
    file_summaries: list[dict] = []
    for path in files:
        rows = _read_jsonl(path)
        seen: set[int] = set()
        file_timestamps: list[int] = []
        file_duplicates = 0
        for row in rows:
            timestamp_ms = int(row["timestamp_ms"])
            file_timestamps.append(timestamp_ms)
            if timestamp_ms in seen:
                file_duplicates += 1
            else:
                seen.add(timestamp_ms)
            if row.get("mark_price") in (None, ""):
                missing_mark_price_rows += 1
            if row.get("index_price") in (None, ""):
                missing_index_price_rows += 1
        duplicate_rows += file_duplicates
        timestamps.extend(file_timestamps)
        file_summaries.append(
            {
                "file": str(path),
                "rows": len(rows),
                "duplicate_rows": file_duplicates,
                "first_timestamp_ms": min(file_timestamps) if file_timestamps else None,
                "last_timestamp_ms": max(file_timestamps) if file_timestamps else None,
            }
        )

    timestamps.sort()
    bucket_seconds = max(args.bucket_seconds, 1)
    gap_count = 0
    max_gap_seconds = 0.0
    for previous, current in zip(timestamps, timestamps[1:]):
        gap_seconds = (current - previous) / 1000.0
        if gap_seconds > bucket_seconds * 1.5:
            gap_count += 1
            max_gap_seconds = max(max_gap_seconds, gap_seconds)

    last_timestamp_ms = timestamps[-1] if timestamps else None
    lag_seconds = None
    lag_warning = False
    if last_timestamp_ms is not None:
        lag_seconds = max(0.0, datetime.now(tz=timezone.utc).timestamp() - last_timestamp_ms / 1000.0)
        lag_warning = lag_seconds > max(args.warn_gap_minutes, 0.0) * 60.0

    print(
        json.dumps(
            {
                "status": "ok",
                "input_dir": str(input_dir),
                "files_checked": len(files),
                "rows_total": len(timestamps),
                "first_timestamp_ms": timestamps[0] if timestamps else None,
                "first_timestamp_iso": _iso(timestamps[0] if timestamps else None),
                "last_timestamp_ms": last_timestamp_ms,
                "last_timestamp_iso": _iso(last_timestamp_ms),
                "lag_seconds": lag_seconds,
                "lag_warning": lag_warning,
                "duplicate_rows": duplicate_rows,
                "gap_count": gap_count,
                "max_gap_seconds": max_gap_seconds,
                "missing_mark_price_rows": missing_mark_price_rows,
                "missing_index_price_rows": missing_index_price_rows,
                "files": file_summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
