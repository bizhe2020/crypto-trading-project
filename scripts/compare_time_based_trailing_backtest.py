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

from scripts.backtest_config_report import (
    DEFAULT_DATA_15M,
    DEFAULT_DATA_4H,
    DEFAULT_OUTPUT_DIR,
    build_report_from_payload,
    load_dataframe,
    load_config_payload,
    parse_end_timestamp,
)


TIME_BASED_DEFAULTS = {
    "enable_time_based_trailing": True,
    "T1": 15,
    "T2": 40,
    "T_max": 96,
    "S0_trigger_rr": 0.5,
    "S1_trigger_rr": 1.0,
    "S3_trigger_rr": 3.0,
    "S4_close_rr": 0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare backtests with and without time-based trailing.")
    parser.add_argument("--config", required=True, help="Path to a config JSON file.")
    parser.add_argument("--start-date", default="2023-01-01", help="Backtest start date, default 2023-01-01")
    parser.add_argument("--end-date", help="Backtest end date. Defaults to latest shared data timestamp.")
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stdout", action="store_true", help="Print JSON summary to stdout.")
    return parser.parse_args()


def latest_shared_end_timestamp(data_15m_path: Path, data_4h_path: Path) -> pd.Timestamp:
    df15 = load_dataframe(data_15m_path)
    df4 = load_dataframe(data_4h_path)
    if df15.empty or df4.empty:
        raise ValueError("Missing market data for comparison backtest.")
    end = min(df15["date"].max(), df4["date"].max())
    return pd.Timestamp(end).tz_convert("UTC")


def build_case_payload(base_payload: dict[str, Any], enable_time_based_trailing: bool) -> dict[str, Any]:
    payload = dict(base_payload)
    for key, value in TIME_BASED_DEFAULTS.items():
        payload.setdefault(key, value)
    payload["enable_time_based_trailing"] = enable_time_based_trailing
    return payload


def compact_metrics(report: dict[str, Any]) -> dict[str, float | int]:
    overall = report["overall"]
    return {
        "total_return_pct": overall["total_return_pct"],
        "total_trades": overall["total_trades"],
        "win_rate": overall["win_rate"],
        "sharpe_ratio": overall["sharpe_ratio"],
        "win_loss_ratio": overall["win_loss_ratio"],
        "max_drawdown_pct": overall["max_drawdown_pct"],
    }


def print_summary(case_name: str, metrics: dict[str, float | int]) -> None:
    print(
        f"{case_name:<12} "
        f"Return={metrics['total_return_pct']:>8.2f}%  "
        f"Trades={metrics['total_trades']:>5d}  "
        f"WinRate={metrics['win_rate']:>6.2f}%  "
        f"Sharpe={metrics['sharpe_ratio']:>7.3f}  "
        f"WL={metrics['win_loss_ratio']:>6.3f}  "
        f"MDD={metrics['max_drawdown_pct']:>7.2f}%"
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    data_15m_path = Path(args.data_15m)
    data_4h_path = Path(args.data_4h)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start_date, tz="UTC")
    end = parse_end_timestamp(args.end_date) if args.end_date else latest_shared_end_timestamp(data_15m_path, data_4h_path)

    base_payload = load_config_payload(config_path)
    without_tit_payload = build_case_payload(base_payload, enable_time_based_trailing=False)
    with_tit_payload = build_case_payload(base_payload, enable_time_based_trailing=True)

    without_tit = build_report_from_payload(
        config_payload=without_tit_payload,
        config_label=f"{config_path.resolve()}#without_tit",
        data_15m_path=data_15m_path,
        data_4h_path=data_4h_path,
        start=start,
        end=end,
    )
    with_tit = build_report_from_payload(
        config_payload=with_tit_payload,
        config_label=f"{config_path.resolve()}#with_tit",
        data_15m_path=data_15m_path,
        data_4h_path=data_4h_path,
        start=start,
        end=end,
    )

    without_summary = compact_metrics(without_tit)
    with_summary = compact_metrics(with_tit)
    delta = {
        key: round(float(with_summary[key]) - float(without_summary[key]), 3)
        for key in ("total_return_pct", "win_rate", "sharpe_ratio", "win_loss_ratio", "max_drawdown_pct")
    }
    delta["total_trades"] = int(with_summary["total_trades"]) - int(without_summary["total_trades"])

    result = {
        "config": str(config_path.resolve()),
        "date_range": {
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
        },
        "cases": {
            "without_tit": without_tit,
            "with_tit": with_tit,
        },
        "summary": {
            "without_tit": without_summary,
            "with_tit": with_summary,
            "delta_with_minus_without": delta,
        },
    }

    output_path = output_dir / (
        f"compare_tit_{config_path.stem}_{start.strftime('%Y-%m-%d')}_to_{end.strftime('%Y-%m-%d')}.json"
    )
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    print(output_path)
    print_summary("without_tit", without_summary)
    print_summary("with_tit", with_summary)

    if args.stdout:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
