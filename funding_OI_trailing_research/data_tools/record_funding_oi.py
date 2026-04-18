from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

OKX_PUBLIC_BASE_URL = "https://www.okx.com"


@dataclass
class RecorderState:
    last_bucket_start_ms: int | None = None
    consecutive_failures: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously record OKX funding/OI snapshots")
    parser.add_argument("--inst-id", default="BTC-USDT-SWAP")
    parser.add_argument("--inst-type", default="SWAP")
    parser.add_argument("--output-dir", default="var/funding_oi/recorded")
    parser.add_argument("--file-prefix", default="btc_funding_oi")
    parser.add_argument("--poll-interval-seconds", type=float, default=60.0)
    parser.add_argument("--bucket-seconds", type=int, default=60)
    parser.add_argument("--request-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--max-backoff-seconds", type=float, default=300.0)
    parser.add_argument("--include-ticker", action="store_true")
    parser.add_argument("--mark-price-fallback", action="store_true")
    parser.add_argument("--run-once", action="store_true")
    return parser.parse_args()


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "crypto-trading-project/funding-oi-recorder"})
    return session


def _request_json(
    session: requests.Session,
    path: str,
    params: dict[str, Any],
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    response = session.get(f"{OKX_PUBLIC_BASE_URL}{path}", params=params, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if str(payload.get("code")) != "0":
        raise RuntimeError(f"OKX API error for {path}: {payload}")
    return list(payload.get("data", []))


def _funding_snapshot(
    session: requests.Session,
    inst_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    rows = _request_json(
        session,
        "/api/v5/public/funding-rate",
        {"instId": inst_id},
        timeout_seconds,
    )
    if not rows:
        raise RuntimeError("Funding snapshot is empty")
    row = rows[0]
    return {
        "funding_rate": float(row.get("fundingRate") or 0.0),
        "next_funding_rate": float(row.get("nextFundingRate") or 0.0),
        "funding_time_ms": int(row.get("fundingTime") or 0),
        "next_funding_time_ms": int(row.get("nextFundingTime") or 0),
        "premium": float(row.get("premium") or 0.0),
    }


def _open_interest_snapshot(
    session: requests.Session,
    inst_id: str,
    inst_type: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    rows = _request_json(
        session,
        "/api/v5/public/open-interest",
        {"instId": inst_id, "instType": inst_type},
        timeout_seconds,
    )
    if not rows:
        raise RuntimeError("Open interest snapshot is empty")
    row = rows[0]
    return {
        "open_interest": float(row.get("oi") or 0.0),
        "open_interest_ccy": float(row.get("oiCcy") or 0.0),
        "open_interest_usd": float(row.get("oiUsd") or 0.0),
    }


def _mark_price_snapshot(
    session: requests.Session,
    inst_id: str,
    inst_type: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    rows = _request_json(
        session,
        "/api/v5/public/mark-price",
        {"instId": inst_id, "instType": inst_type},
        timeout_seconds,
    )
    if not rows:
        raise RuntimeError("Mark price snapshot is empty")
    row = rows[0]
    return {"mark_price": float(row.get("markPx") or 0.0)}


def _ticker_snapshot(
    session: requests.Session,
    inst_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    rows = _request_json(
        session,
        "/api/v5/market/ticker",
        {"instId": inst_id},
        timeout_seconds,
    )
    if not rows:
        raise RuntimeError("Ticker snapshot is empty")
    row = rows[0]
    return {
        "last_price": float(row.get("last") or 0.0),
        "index_price": float(row.get("idxPx") or 0.0),
    }


def _bucket_start_ms(timestamp_ms: int, bucket_seconds: int) -> int:
    bucket_ms = max(bucket_seconds, 1) * 1000
    return (timestamp_ms // bucket_ms) * bucket_ms


def _daily_output_path(output_dir: Path, prefix: str, timestamp_ms: int) -> Path:
    utc_day = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    return output_dir / f"{prefix}_{utc_day}.jsonl"


def _load_last_bucket(path: Path, bucket_seconds: int) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        if size <= 0:
            return None
        offset = min(size, 8192)
        handle.seek(-offset, 2)
        tail = handle.read().decode("utf-8", errors="ignore")
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        row = json.loads(lines[-1])
        timestamp_ms = int(row["timestamp_ms"])
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return _bucket_start_ms(timestamp_ms, bucket_seconds)


def _collect_snapshot(
    session: requests.Session,
    args: argparse.Namespace,
    timestamp_ms: int,
) -> dict[str, Any]:
    row = {
        "timestamp_ms": timestamp_ms,
        "recorded_at_iso": datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat(),
        "inst_id": args.inst_id,
        "inst_type": args.inst_type,
        "snapshot_source": "okx_public_api",
        "bucket_seconds": args.bucket_seconds,
    }
    row.update(_funding_snapshot(session, args.inst_id, args.request_timeout_seconds))
    row.update(_open_interest_snapshot(session, args.inst_id, args.inst_type, args.request_timeout_seconds))
    try:
        row.update(_mark_price_snapshot(session, args.inst_id, args.inst_type, args.request_timeout_seconds))
    except Exception:
        if not args.mark_price_fallback:
            raise
    if args.include_ticker or args.mark_price_fallback:
        ticker = _ticker_snapshot(session, args.inst_id, args.request_timeout_seconds)
        row.update(ticker)
        if "mark_price" not in row:
            row["mark_price"] = ticker["last_price"]
    return row


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    session = _session()
    state = RecorderState()

    while True:
        now_ms = int(time.time() * 1000)
        output_path = _daily_output_path(output_dir, args.file_prefix, now_ms)
        if state.last_bucket_start_ms is None:
            state.last_bucket_start_ms = _load_last_bucket(output_path, args.bucket_seconds)

        bucket_start_ms = _bucket_start_ms(now_ms, args.bucket_seconds)
        if state.last_bucket_start_ms == bucket_start_ms:
            if args.run_once:
                print(
                    json.dumps(
                        {
                            "status": "skipped_duplicate_bucket",
                            "bucket_start_ms": bucket_start_ms,
                            "output_file": str(output_path),
                        },
                        ensure_ascii=False,
                    )
                )
                return
            time.sleep(max(args.poll_interval_seconds, 1.0))
            continue

        try:
            row = _collect_snapshot(session, args, now_ms)
            _append_jsonl(output_path, row)
            state.last_bucket_start_ms = bucket_start_ms
            state.consecutive_failures = 0
            print(
                json.dumps(
                    {
                        "status": "recorded",
                        "timestamp_ms": now_ms,
                        "output_file": str(output_path),
                        "funding_rate": row["funding_rate"],
                        "open_interest": row["open_interest"],
                        "mark_price": row.get("mark_price"),
                    },
                    ensure_ascii=False,
                )
            )
            if args.run_once:
                return
            time.sleep(max(args.poll_interval_seconds, 1.0))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            state.consecutive_failures += 1
            backoff_seconds = min(
                max(args.poll_interval_seconds, 1.0) * (2 ** min(state.consecutive_failures - 1, 6)),
                max(args.max_backoff_seconds, 1.0),
            )
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error": str(exc),
                        "consecutive_failures": state.consecutive_failures,
                        "backoff_seconds": backoff_seconds,
                    },
                    ensure_ascii=False,
                )
            )
            if args.run_once:
                raise
            time.sleep(backoff_seconds)


if __name__ == "__main__":
    main()
