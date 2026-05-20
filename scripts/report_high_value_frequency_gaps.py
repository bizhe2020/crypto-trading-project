#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

import pandas as pd


def _normalize_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _gap_bucket(previous_event_type: str, next_event_type: str, gap_days: float) -> str:
    if next_event_type == "smc_short":
        return "smc_short_cluster"
    if previous_event_type == "smc_short":
        return "avoid_fill"
    if gap_days < 30.0:
        return "secondary_reentry"
    if gap_days < 40.0:
        return "low_priority"
    return "avoid_fill"


def build_gap_rows(reference_events: list[dict[str, Any]], min_gap_days: float = 0.0) -> list[dict[str, Any]]:
    ordered = sorted(
        [dict(event) for event in reference_events],
        key=lambda item: (
            _normalize_ts(item.get("entry_time")),
            _normalize_ts(item.get("exit_time")),
            int(item.get("entry_idx", 0) or 0),
        ),
    )
    gaps: list[dict[str, Any]] = []
    minimum = max(float(min_gap_days), 0.0)

    for previous, current in zip(ordered, ordered[1:]):
        gap_start = _normalize_ts(previous.get("exit_time"))
        gap_end = _normalize_ts(current.get("entry_time"))
        gap_days = (gap_end - gap_start).total_seconds() / 86400.0
        if gap_days <= minimum:
            continue

        previous_event_type = str(previous.get("event_type") or "unknown")
        next_event_type = str(current.get("event_type") or "unknown")
        gaps.append(
            {
                "gap_start": str(gap_start),
                "gap_end": str(gap_end),
                "gap_days": round(gap_days, 4),
                "bucket": _gap_bucket(previous_event_type, next_event_type, gap_days),
                "previous_event_type": previous_event_type,
                "next_event_type": next_event_type,
                "previous_entry_time": previous.get("entry_time"),
                "previous_exit_time": previous.get("exit_time"),
                "next_entry_time": current.get("entry_time"),
                "next_exit_time": current.get("exit_time"),
            }
        )
    return gaps
