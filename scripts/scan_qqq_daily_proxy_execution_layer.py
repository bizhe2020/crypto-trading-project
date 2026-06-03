#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_qqq_daily_proxy_high_stop import (  # noqa: E402
    STOP_MODES,
    build_signal_path,
    load_ohlcv,
    load_strict_config,
    merge_signal_with_qqq,
    replay_daily_proxy,
)


DEFAULT_SIGNAL_CONFIG = ROOT / "config" / "config.paper.tqqq-only-strict-recovery-frozen.json"
DEFAULT_DATA_ROOT = ROOT / "data" / "public" / "etf_long"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_daily_proxy_execution_layer_scan_20260530.json"


def parse_float_grid(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def build_profiles(leverage_values: list[float], defense_values: list[float]) -> dict[str, dict[str, float]]:
    profiles: dict[str, dict[str, float]] = {}
    for leverage in leverage_values:
        profiles[f"fixed{leverage:g}"] = {
            "base": float(leverage),
            "offense": float(leverage),
            "defense": float(leverage),
        }
    for base in leverage_values:
        for defense in defense_values:
            if defense > base:
                continue
            profiles[f"base{base:g}_off{base:g}_def{defense:g}"] = {
                "base": float(base),
                "offense": float(base),
                "defense": float(defense),
            }
    return profiles


def score_candidate(summary: dict[str, Any], *, dd_cap: float) -> float:
    total = float(summary["total_return_pct"])
    dd = float(summary["max_drawdown_pct"])
    y = summary.get("yearly_returns_pct", {})
    recent = float(y.get("2023", 0.0)) + float(y.get("2024", 0.0)) + float(y.get("2025", 0.0)) + float(y.get("2026", 0.0))
    penalty = max(0.0, dd - dd_cap) * 60.0
    return total + recent * 0.5 - dd * 8.0 - penalty


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan long-horizon QQQ daily proxy execution layer on frozen TQQQ signal.")
    parser.add_argument("--signal-config", default=str(DEFAULT_SIGNAL_CONFIG))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--leverages", default="1,1.5,2,3,4,5,6,8,10")
    parser.add_argument("--defense-leverages", default="0,0.5,1,2,3")
    parser.add_argument("--stop-loss-pcts", default="2,2.5,3,3.5,4,5,6,8,10")
    parser.add_argument("--stop-modes", default="close_same_bar,high_prev_strict")
    parser.add_argument("--taker-fee-rate", type=float, default=0.0002)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--daily-funding-rate", type=float, default=0.0)
    parser.add_argument("--dd-cap", type=float, default=60.0)
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    signal_config = load_strict_config(Path(args.signal_config))
    data_root = Path(args.data_root)
    signal_path, signal_summary = build_signal_path(signal_config, data_root)
    qqq = load_ohlcv(data_root / "QQQ-1d.feather")
    bars = merge_signal_with_qqq(signal_path, qqq)

    profiles = build_profiles(parse_float_grid(args.leverages), parse_float_grid(args.defense_leverages))
    stop_loss_pcts = parse_float_grid(args.stop_loss_pcts)
    stop_modes = [item.strip() for item in args.stop_modes.split(",") if item.strip()]
    for stop_mode in stop_modes:
        if stop_mode not in STOP_MODES:
            raise ValueError(f"Unsupported stop mode: {stop_mode}")

    summaries: list[dict[str, Any]] = []
    total_jobs = len(profiles) * len(stop_loss_pcts) * len(stop_modes)
    done = 0
    for profile_name, profile in profiles.items():
        for stop_loss_pct in stop_loss_pcts:
            for stop_mode in stop_modes:
                result = replay_daily_proxy(
                    bars,
                    profile_name=profile_name,
                    profile=profile,
                    stop_mode=stop_mode,
                    stop_loss_pct=float(stop_loss_pct),
                    taker_fee_rate=float(args.taker_fee_rate),
                    slippage_bps=float(args.slippage_bps),
                    initial_capital=float(signal_config.get("initial_capital", 1000.0)),
                    daily_funding_rate=float(args.daily_funding_rate),
                    rebalance_on_leverage_change=True,
                )
                summary = result["summary"]
                summary["profile"] = profile
                summary["score"] = round(score_candidate(summary, dd_cap=float(args.dd_cap)), 4)
                summaries.append(summary)
                done += 1
                if done % 100 == 0 or done == total_jobs:
                    print(f"scanned {done}/{total_jobs}", flush=True)

    ranked = sorted(summaries, key=lambda item: float(item["score"]), reverse=True)
    feasible = [item for item in ranked if float(item["max_drawdown_pct"]) <= float(args.dd_cap)]
    payload = {
        "metadata": {
            "signal_config": str(Path(args.signal_config)),
            "signal_frozen_label": signal_config.get("frozen_label"),
            "data_root": str(data_root),
            "taker_fee_rate": float(args.taker_fee_rate),
            "slippage_bps": float(args.slippage_bps),
            "daily_funding_rate": float(args.daily_funding_rate),
            "dd_cap": float(args.dd_cap),
            "jobs": int(total_jobs),
        },
        "coverage": {
            "rows": int(len(bars)),
            "start": str(bars["date"].min()) if not bars.empty else None,
            "end": str(bars["date"].max()) if not bars.empty else None,
        },
        "signal_summary": signal_summary,
        "top_by_score": ranked[: int(args.top)],
        "top_feasible_by_score": feasible[: int(args.top)],
        "all": summaries,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(out)
    print(json.dumps({"top_by_score": payload["top_by_score"][:10], "top_feasible_by_score": payload["top_feasible_by_score"][:10]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
