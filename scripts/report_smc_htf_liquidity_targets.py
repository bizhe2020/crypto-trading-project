#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_readiness_report import load_prepared_data  # noqa: E402
from scripts.report_smc_trade_context import DEFAULT_OUTPUT as DEFAULT_SMC_REPORT  # noqa: E402
from scripts.report_smc_trade_context import (  # noqa: E402
    completed_4h_idx_for_entry,
    completed_d1_idx_for_entry,
    daily_candles_from_4h,
    summarize_rows,
    target_rr_bucket,
)
from strategy.scalp_robust_v2_core import Candle, precompute_swings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_context" / "smc_htf_liquidity_targets_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report HTF BSL/SSL liquidity target quality for SMC-tagged trades.")
    parser.add_argument("--smc-report", default=str(DEFAULT_SMC_REPORT))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--h4-swing-n", type=int, default=2)
    parser.add_argument("--h4-swing-lookback", type=int, default=80)
    parser.add_argument("--d1-swing-n", type=int, default=2)
    parser.add_argument("--d1-swing-lookback", type=int, default=20)
    parser.add_argument("--h4-lookback-bars", type=int, default=360)
    parser.add_argument("--d1-lookback-bars", type=int, default=180)
    parser.add_argument("--equal-tolerance-pct", type=float, default=0.35)
    parser.add_argument("--equal-min-touches", type=int, default=2)
    parser.add_argument("--min-target-rr", type=float, default=1.5)
    parser.add_argument("--post-exit-lookahead-bars", type=int, default=96)
    parser.add_argument("--top-samples", type=int, default=20)
    return parser.parse_args()


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def rr_to_level(row: dict[str, Any], level: float) -> float | None:
    entry = float(row.get("entry_price", 0.0) or 0.0)
    stop = row.get("initial_stop_price")
    if stop is None:
        return None
    risk = abs(entry - float(stop))
    if risk <= 0:
        return None
    direction = str(row.get("direction") or "")
    if direction == "BULL":
        return (float(level) - entry) / risk
    if direction == "BEAR":
        return (entry - float(level)) / risk
    return None


def target_distance_pct(row: dict[str, Any], level: float) -> float:
    entry = float(row.get("entry_price", 0.0) or 0.0)
    if entry <= 0:
        return 0.0
    direction = str(row.get("direction") or "")
    if direction == "BULL":
        return (float(level) - entry) / entry * 100.0
    if direction == "BEAR":
        return (entry - float(level)) / entry * 100.0
    return 0.0


def candidate_level_allowed(row: dict[str, Any], level: float) -> bool:
    entry = float(row.get("entry_price", 0.0) or 0.0)
    direction = str(row.get("direction") or "")
    if direction == "BULL":
        return level > entry
    if direction == "BEAR":
        return level < entry
    return False


def add_swing_candidates(
    candidates: list[dict[str, Any]],
    row: dict[str, Any],
    candles: list[Candle],
    swing_indices: list[int],
    end_idx: int,
    lookback: int,
    source: str,
    price_attr: str,
) -> None:
    start = max(0, end_idx - lookback)
    for idx in swing_indices:
        if not (start <= idx <= end_idx):
            continue
        level = float(getattr(candles[idx], price_attr))
        if not candidate_level_allowed(row, level):
            continue
        rr = rr_to_level(row, level)
        if rr is None:
            continue
        candidates.append(
            {
                "source": source,
                "level": level,
                "idx": idx,
                "touches": 1,
                "rr": rr,
                "distance_pct": target_distance_pct(row, level),
            }
        )


def add_equal_level_candidates(
    candidates: list[dict[str, Any]],
    row: dict[str, Any],
    candles: list[Candle],
    swing_indices: list[int],
    end_idx: int,
    lookback: int,
    source: str,
    price_attr: str,
    tolerance_pct: float,
    min_touches: int,
) -> None:
    start = max(0, end_idx - lookback)
    levels: list[tuple[int, float]] = [
        (idx, float(getattr(candles[idx], price_attr)))
        for idx in swing_indices
        if start <= idx <= end_idx
    ]
    levels.sort(key=lambda item: item[1])
    clusters: list[list[tuple[int, float]]] = []
    for item in levels:
        idx, level = item
        placed = False
        for cluster in clusters:
            ref = sum(value for _idx, value in cluster) / len(cluster)
            if ref > 0 and abs(level - ref) / ref * 100.0 <= tolerance_pct:
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue
        level = sum(value for _idx, value in cluster) / len(cluster)
        if not candidate_level_allowed(row, level):
            continue
        rr = rr_to_level(row, level)
        if rr is None:
            continue
        candidates.append(
            {
                "source": source,
                "level": level,
                "idx": max(idx for idx, _value in cluster),
                "touches": len(cluster),
                "rr": rr,
                "distance_pct": target_distance_pct(row, level),
            }
        )


def select_target(candidates: list[dict[str, Any]], min_rr: float) -> dict[str, Any] | None:
    qualified = [item for item in candidates if float(item["rr"]) >= min_rr]
    if not qualified:
        return None
    source_rank = {
        "d1_equal_bsl": 0,
        "d1_equal_ssl": 0,
        "h4_equal_bsl": 1,
        "h4_equal_ssl": 1,
        "d1_swing_bsl": 2,
        "d1_swing_ssl": 2,
        "h4_swing_bsl": 3,
        "h4_swing_ssl": 3,
    }
    return min(qualified, key=lambda item: (float(item["rr"]), source_rank.get(str(item["source"]), 9)))


def target_hit(row: dict[str, Any], prepared: Any, target: dict[str, Any] | None) -> dict[str, Any]:
    if target is None:
        return {"hit": False, "hit_idx": None, "hit_time": None, "bars_to_hit": None}
    entry_idx = int(row.get("entry_idx", -1))
    exit_idx = row.get("exit_idx")
    if exit_idx is None:
        exit_idx = entry_idx
    exit_idx = int(exit_idx)
    level = float(target["level"])
    direction = str(row.get("direction") or "")
    for idx in range(max(0, entry_idx), min(exit_idx, len(prepared.c15m) - 1) + 1):
        candle = prepared.c15m[idx]
        if direction == "BULL" and float(candle.h) >= level:
            return {
                "hit": True,
                "hit_idx": idx,
                "hit_time": str(pd.Timestamp(candle.ts, unit="s", tz="UTC")),
                "bars_to_hit": idx - entry_idx,
            }
        if direction == "BEAR" and float(candle.l) <= level:
            return {
                "hit": True,
                "hit_idx": idx,
                "hit_time": str(pd.Timestamp(candle.ts, unit="s", tz="UTC")),
                "bars_to_hit": idx - entry_idx,
            }
    return {"hit": False, "hit_idx": None, "hit_time": None, "bars_to_hit": None}


def target_hit_after_exit(
    row: dict[str, Any],
    prepared: Any,
    target: dict[str, Any] | None,
    lookahead_bars: int,
) -> dict[str, Any]:
    if target is None:
        return {"hit": False, "hit_idx": None, "hit_time": None, "bars_after_exit": None}
    entry_idx = int(row.get("entry_idx", -1))
    exit_idx = row.get("exit_idx")
    if exit_idx is None:
        exit_idx = entry_idx
    exit_idx = int(exit_idx)
    level = float(target["level"])
    direction = str(row.get("direction") or "")
    start = max(0, exit_idx + 1)
    end = min(len(prepared.c15m) - 1, exit_idx + max(int(lookahead_bars), 0))
    for idx in range(start, end + 1):
        candle = prepared.c15m[idx]
        if direction == "BULL" and float(candle.h) >= level:
            return {
                "hit": True,
                "hit_idx": idx,
                "hit_time": str(pd.Timestamp(candle.ts, unit="s", tz="UTC")),
                "bars_after_exit": idx - exit_idx,
            }
        if direction == "BEAR" and float(candle.l) <= level:
            return {
                "hit": True,
                "hit_idx": idx,
                "hit_time": str(pd.Timestamp(candle.ts, unit="s", tz="UTC")),
                "bars_after_exit": idx - exit_idx,
            }
    return {"hit": False, "hit_idx": None, "hit_time": None, "bars_after_exit": None}


def build_candidates(
    row: dict[str, Any],
    prepared: Any,
    daily: list[Candle],
    daily_ts: list[float],
    h4_highs: list[int],
    h4_lows: list[int],
    d1_highs: list[int],
    d1_lows: list[int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    entry_idx = int(row.get("entry_idx", -1))
    if entry_idx < 0:
        return []
    entry_ts = prepared.c15m[entry_idx].ts
    h4_idx = completed_4h_idx_for_entry(prepared.mapping, entry_idx)
    d1_idx = completed_d1_idx_for_entry(daily_ts, entry_ts)
    direction = str(row.get("direction") or "")
    candidates: list[dict[str, Any]] = []

    if direction == "BULL":
        add_equal_level_candidates(candidates, row, daily, d1_highs, d1_idx, args.d1_lookback_bars, "d1_equal_bsl", "h", args.equal_tolerance_pct, args.equal_min_touches)
        add_equal_level_candidates(candidates, row, prepared.c4h, h4_highs, h4_idx, args.h4_lookback_bars, "h4_equal_bsl", "h", args.equal_tolerance_pct, args.equal_min_touches)
        add_swing_candidates(candidates, row, daily, d1_highs, d1_idx, args.d1_lookback_bars, "d1_swing_bsl", "h")
        add_swing_candidates(candidates, row, prepared.c4h, h4_highs, h4_idx, args.h4_lookback_bars, "h4_swing_bsl", "h")
    elif direction == "BEAR":
        add_equal_level_candidates(candidates, row, daily, d1_lows, d1_idx, args.d1_lookback_bars, "d1_equal_ssl", "l", args.equal_tolerance_pct, args.equal_min_touches)
        add_equal_level_candidates(candidates, row, prepared.c4h, h4_lows, h4_idx, args.h4_lookback_bars, "h4_equal_ssl", "l", args.equal_tolerance_pct, args.equal_min_touches)
        add_swing_candidates(candidates, row, daily, d1_lows, d1_idx, args.d1_lookback_bars, "d1_swing_ssl", "l")
        add_swing_candidates(candidates, row, prepared.c4h, h4_lows, h4_idx, args.h4_lookback_bars, "h4_swing_ssl", "l")

    # Deduplicate very close levels, preferring equal-level / higher-timeframe sources.
    candidates.sort(key=lambda item: (float(item["rr"]), str(item["source"])))
    deduped: list[dict[str, Any]] = []
    for item in candidates:
        level = float(item["level"])
        if any(abs(level - float(prev["level"])) / level * 100.0 <= args.equal_tolerance_pct for prev in deduped):
            continue
        item = dict(item)
        item["level"] = round(level, 2)
        item["rr"] = round(float(item["rr"]), 3)
        item["distance_pct"] = round(float(item["distance_pct"]), 3)
        deduped.append(item)
    return deduped


def grouped_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(row)
    return {name: summarize_rows(bucket) for name, bucket in sorted(groups.items())}


def main() -> None:
    args = parse_args()
    report = json.loads(Path(args.smc_report).read_text())
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=None,
    )
    daily = daily_candles_from_4h(prepared.c4h)
    daily_ts = [candle.ts for candle in daily]
    h4_highs, h4_lows = precompute_swings(prepared.c4h, n=args.h4_swing_n, lookback=args.h4_swing_lookback)
    d1_highs, d1_lows = precompute_swings(daily, n=args.d1_swing_n, lookback=args.d1_swing_lookback)

    rows: list[dict[str, Any]] = []
    for row in report.get("rows", []):
        candidates = build_candidates(row, prepared, daily, daily_ts, h4_highs, h4_lows, d1_highs, d1_lows, args)
        nearest_any = candidates[0] if candidates else None
        selected = select_target(candidates, args.min_target_rr)
        hit = target_hit(row, prepared, selected)
        post_exit_hit = target_hit_after_exit(row, prepared, selected, args.post_exit_lookahead_bars)
        annotated = dict(row)
        annotated.update(
            {
                "htf_liquidity_candidate_count": len(candidates),
                "htf_nearest_target_level": None if nearest_any is None else nearest_any["level"],
                "htf_nearest_target_source": None if nearest_any is None else nearest_any["source"],
                "htf_nearest_target_rr": None if nearest_any is None else nearest_any["rr"],
                "htf_nearest_target_rr_bucket": target_rr_bucket(None if nearest_any is None else float(nearest_any["rr"])),
                "htf_selected_target_level": None if selected is None else selected["level"],
                "htf_selected_target_source": None if selected is None else selected["source"],
                "htf_selected_target_rr": None if selected is None else selected["rr"],
                "htf_selected_target_rr_bucket": target_rr_bucket(None if selected is None else float(selected["rr"])),
                "htf_selected_target_distance_pct": None if selected is None else selected["distance_pct"],
                "htf_selected_target_touches": None if selected is None else selected["touches"],
                "htf_selected_target_hit": hit["hit"],
                "htf_selected_target_hit_time": hit["hit_time"],
                "htf_selected_target_bars_to_hit": hit["bars_to_hit"],
                "htf_selected_target_post_exit_hit": post_exit_hit["hit"],
                "htf_selected_target_post_exit_hit_time": post_exit_hit["hit_time"],
                "htf_selected_target_bars_after_exit": post_exit_hit["bars_after_exit"],
                "htf_top_targets": candidates[:5],
            }
        )
        rows.append(annotated)

    selected_rows = [row for row in rows if row.get("htf_selected_target_level") is not None]
    hit_rows = [row for row in selected_rows if row.get("htf_selected_target_hit")]
    post_exit_hit_rows = [row for row in selected_rows if row.get("htf_selected_target_post_exit_hit")]
    output_payload = {
        "metadata": {
            "smc_report": str(Path(args.smc_report).resolve()),
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
            "daily_candles": len(daily),
            "h4_swings": {"highs": len(h4_highs), "lows": len(h4_lows)},
            "d1_swings": {"highs": len(d1_highs), "lows": len(d1_lows)},
            "params": {
                "h4_lookback_bars": args.h4_lookback_bars,
                "d1_lookback_bars": args.d1_lookback_bars,
                "equal_tolerance_pct": args.equal_tolerance_pct,
                "equal_min_touches": args.equal_min_touches,
                "min_target_rr": args.min_target_rr,
                "post_exit_lookahead_bars": args.post_exit_lookahead_bars,
            },
        },
        "summary": {
            "all": summarize_rows(rows),
            "with_selected_target": summarize_rows(selected_rows),
            "selected_target_hit": summarize_rows(hit_rows),
            "selected_target_hit_rate_pct": round(len(hit_rows) / len(selected_rows) * 100.0, 2) if selected_rows else 0.0,
            "selected_target_post_exit_hit": summarize_rows(post_exit_hit_rows),
            "selected_target_post_exit_hit_rate_pct": round(len(post_exit_hit_rows) / len(selected_rows) * 100.0, 2) if selected_rows else 0.0,
            "by_nearest_rr_bucket": grouped_summary(rows, "htf_nearest_target_rr_bucket"),
            "by_selected_source": grouped_summary(selected_rows, "htf_selected_target_source"),
            "by_selected_hit": grouped_summary(selected_rows, "htf_selected_target_hit"),
            "by_selected_post_exit_hit": grouped_summary(selected_rows, "htf_selected_target_post_exit_hit"),
            "by_direction": grouped_summary(selected_rows, "direction"),
            "by_regime_label": grouped_summary(selected_rows, "regime_label"),
        },
        "samples": {
            "highest_rr_targets": sorted(selected_rows, key=lambda item: float(item.get("htf_selected_target_rr") or 0.0), reverse=True)[: args.top_samples],
            "hit_targets": hit_rows[: args.top_samples],
            "post_exit_hit_targets": post_exit_hit_rows[: args.top_samples],
            "missed_targets": [row for row in selected_rows if not row.get("htf_selected_target_hit")][: args.top_samples],
        },
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(output_payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    compact = {
        "all": output_payload["summary"]["all"],
        "with_selected_target": output_payload["summary"]["with_selected_target"],
        "selected_target_hit_rate_pct": output_payload["summary"]["selected_target_hit_rate_pct"],
        "selected_target_post_exit_hit_rate_pct": output_payload["summary"]["selected_target_post_exit_hit_rate_pct"],
        "by_selected_source": output_payload["summary"]["by_selected_source"],
        "by_selected_hit": output_payload["summary"]["by_selected_hit"],
        "by_selected_post_exit_hit": output_payload["summary"]["by_selected_post_exit_hit"],
    }
    print(json.dumps(clean_for_json(compact), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
