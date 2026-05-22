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

from scripts.nasdaq100_cn_strategy_utils import load_config, load_strategy_frame, run_full_strategy_path


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "cn_nasdaq100_conditional_leverage_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan conditional 2x leverage overlays for 513100 driven by QQQ signals.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.paper.cn-nasdaq100-etf.json"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def annual_returns(equity: pd.DataFrame) -> dict[str, float]:
    if equity.empty:
        return {}
    frame = equity.copy()
    frame["year"] = frame["date"].dt.year.astype(str)
    out: dict[str, float] = {}
    for year, group in frame.groupby("year"):
        start = float(group.iloc[0]["equity"])
        end = float(group.iloc[-1]["equity"])
        out[year] = round((end / start - 1.0) * 100.0, 2) if start > 0 else 0.0
    return out


def run_overlay(path: pd.DataFrame, leverage_mask: pd.Series) -> dict[str, Any]:
    capital = 1000.0
    rows: list[dict[str, Any]] = []
    for idx, row in path.iterrows():
        active = str(row["position"]) == "LONG"
        lev = 2.0 if active and bool(leverage_mask.iloc[idx]) else (1.0 if active else 0.0)
        daily_ret = float(row["daily_return"])
        if active and lev > 1.0:
            # Reconstruct base asset return from the baseline path then scale only the asset component.
            cost_component = 0.0
            if idx > 0 and str(path.iloc[idx - 1]["position"]) != "LONG":
                cost_component = daily_ret - ((float(row["asset_close"]) / float(path.iloc[idx - 1]["asset_close"])) - 1.0)
            asset_ret = daily_ret - cost_component
            daily_ret = asset_ret * lev + cost_component
        capital *= 1.0 + daily_ret
        rows.append({"date": row["date"], "equity": capital, "position": row["position"], "daily_return": daily_ret, "leverage": lev})
    equity = pd.DataFrame(rows)
    peak = equity["equity"].cummax()
    dd = ((peak - equity["equity"]) / peak.replace(0, pd.NA) * 100.0).max(skipna=True)
    return {
        "total_return_pct": round((capital / 1000.0 - 1.0) * 100.0, 2),
        "max_drawdown_pct": round(float(dd or 0.0), 2),
        "yearly_returns_pct": annual_returns(equity[["date", "equity"]]),
        "trades": int((equity["position"] != equity["position"].shift(1)).sum()),
        "leveraged_days": int((equity["leverage"] > 1.0).sum()),
    }


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    frame = load_strategy_frame(config).copy()
    path = run_full_strategy_path(frame, config).reset_index(drop=True)
    path_frame = frame.loc[frame.index[-len(path):]].reset_index(drop=True)

    candidates = {
        "base_1x": pd.Series([False] * len(path)),
        "vix_low_2x": path_frame["vix_label"].eq("vix_low"),
        "qqq_strong_2x": path_frame["rel_strength_label"].eq("qqq_strong"),
        "vix_low_and_qqq_strong_2x": path_frame["vix_label"].eq("vix_low") & path_frame["rel_strength_label"].eq("qqq_strong"),
        "ixic_up_and_qqq_strong_2x": path_frame["ixic_trend_label"].eq("ixic_up") & path_frame["rel_strength_label"].eq("qqq_strong"),
    }

    results: dict[str, Any] = {}
    for name, leverage_mask in candidates.items():
        results[name] = run_overlay(path, leverage_mask.reset_index(drop=True))

    payload = {"results": results}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    for name, result in results.items():
        print(name, result["total_return_pct"], result["max_drawdown_pct"], result["yearly_returns_pct"], "leveraged_days", result["leveraged_days"])


if __name__ == "__main__":
    main()
