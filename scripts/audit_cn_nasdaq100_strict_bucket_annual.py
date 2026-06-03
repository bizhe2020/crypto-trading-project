#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cn_nasdaq100_strict_utils import load_config, load_strict_frame, run_strict_path  # noqa: E402


OUTPUT = ROOT / "var" / "reports" / "cn_nasdaq100_strict_bucket_annual_audit.json"


def candidate_config() -> dict[str, Any]:
    config = load_config(ROOT / "config" / "config.paper.cn-nasdaq100-etf.json")
    config["conditional_leverage_enabled"] = False
    config["conditional_leverage_value"] = 1.0
    config["tiered_leverage_enabled"] = True
    config["tiered_leverage_rules"] = [
        {
            "vix_label": "vix_normal",
            "rel_strength_label": "qqq_strong",
            "leverage": 2.0,
        },
        {
            "vix_label": "vix_normal",
            "rel_strength_label": "qqq_neutral",
            "leverage": 1.5,
        },
    ]
    config["entry_fast_window"] = 21
    config["entry_slow_window"] = 200
    config["regime_filter"] = "ixic_filter"
    config["max_hold_days"] = 120
    config["trailing_lookback_days"] = 4
    config["trailing_drawdown_pct"] = 4.0
    return config


def main() -> None:
    config = candidate_config()
    frame = load_strict_frame(config)
    path = run_strict_path(frame, config).reset_index(drop=True)
    hold = path[path["position"].eq("LONG")].copy()
    hold["year"] = hold["date"].dt.year.astype(str)

    def bucket_label(row: pd.Series) -> str:
        if float(row["leverage"]) >= 2.0:
            return "normal_strong_2x"
        if float(row["leverage"]) >= 1.5:
            return "normal_neutral_1.5x"
        return "base_1x"

    hold["bucket"] = hold.apply(bucket_label, axis=1)

    year_bucket_rows: list[dict[str, Any]] = []
    for (year, bucket), group in hold.groupby(["year", "bucket"]):
        compounded = float((1.0 + group["daily_return"].astype(float)).prod() - 1.0) * 100.0
        year_bucket_rows.append(
            {
                "year": str(year),
                "bucket": str(bucket),
                "days": int(len(group)),
                "mean_daily_return_pct": round(float(group["daily_return"].mean()) * 100.0, 4),
                "median_daily_return_pct": round(float(group["daily_return"].median()) * 100.0, 4),
                "compounded_bucket_return_pct": round(compounded, 2),
            }
        )

    trade_rows: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    prev_row: pd.Series | None = None
    for _, row in path.iterrows():
        if bool(row["entered_today"]):
            active = {
                "entry_date": str(pd.Timestamp(row["date"]).date()),
                "entry_year": str(pd.Timestamp(row["date"]).year),
                "entry_bucket": bucket_label(row),
                "entry_capital": float(prev_row["capital"]) if prev_row is not None else float(row["capital"]) / max(1e-12, 1.0 + float(row["daily_return"])),
            }
        if bool(row["exited_today"]) and active is not None and prev_row is not None:
            exit_capital = float(prev_row["capital"])
            trade_rows.append(
                {
                    "entry_year": active["entry_year"],
                    "entry_bucket": active["entry_bucket"],
                    "entry_date": active["entry_date"],
                    "exit_date": str(pd.Timestamp(row["date"]).date()),
                    "trade_return_pct": round((exit_capital / float(active["entry_capital"]) - 1.0) * 100.0, 2),
                }
            )
            active = None
        prev_row = row
    if active is not None and prev_row is not None:
        exit_capital = float(prev_row["capital"])
        trade_rows.append(
            {
                "entry_year": active["entry_year"],
                "entry_bucket": active["entry_bucket"],
                "entry_date": active["entry_date"],
                "exit_date": str(pd.Timestamp(prev_row["date"]).date()),
                "trade_return_pct": round((exit_capital / float(active["entry_capital"]) - 1.0) * 100.0, 2),
            }
        )

    trade_df = pd.DataFrame(trade_rows)
    trade_bucket_year: list[dict[str, Any]] = []
    if not trade_df.empty:
        for (year, bucket), group in trade_df.groupby(["entry_year", "entry_bucket"]):
            trade_bucket_year.append(
                {
                    "year": str(year),
                    "bucket": str(bucket),
                    "trade_count": int(len(group)),
                    "mean_trade_return_pct": round(float(group["trade_return_pct"].mean()), 2),
                    "median_trade_return_pct": round(float(group["trade_return_pct"].median()), 2),
                    "best_trade_return_pct": round(float(group["trade_return_pct"].max()), 2),
                    "worst_trade_return_pct": round(float(group["trade_return_pct"].min()), 2),
                    "positive_trade_ratio_pct": round(float((group["trade_return_pct"] > 0).mean() * 100.0), 2),
                }
            )

    payload = {
        "candidate": {
            "entry_fast_window": 21,
            "entry_slow_window": 200,
            "regime_filter": "ixic_filter",
            "max_hold_days": 120,
            "trailing_lookback_days": 4,
            "trailing_drawdown_pct": 4.0,
            "tiered_leverage_rules": config["tiered_leverage_rules"],
        },
        "holding_day_bucket_year": year_bucket_rows,
        "trade_bucket_year": trade_bucket_year,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(OUTPUT)
    print(json.dumps({"holding_day_bucket_year": year_bucket_rows[:6], "trade_bucket_year": trade_bucket_year[:6]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
