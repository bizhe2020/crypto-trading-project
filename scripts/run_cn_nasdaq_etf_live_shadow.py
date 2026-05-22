#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nasdaq100_cn_strategy_utils import ROOT, build_allow_mask, load_config, load_strategy_frame, run_full_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the CN Nasdaq-100 ETF frozen config as a daily live-shadow log.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live-shadow.cn-nasdaq100-etf.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    frame = load_strategy_frame(config).reset_index(drop=True)
    allow_mask = build_allow_mask(frame, config).reset_index(drop=True)
    summary = run_full_strategy(frame, config)

    rows: list[dict[str, object]] = []
    for idx, row in frame.iterrows():
        planned = bool(int(row["planned_position"]) > 0)
        allowed = bool(allow_mask.iloc[idx])
        rows.append(
            {
                "date": str(pd.Timestamp(row["date"]).date()),
                "planned_signal": str(config.get("execution_symbol", "513100.SS")) if planned else "CASH",
                "allowed": allowed,
                "decision": str(config.get("execution_symbol", "513100.SS")) if planned and allowed else "CASH",
                "qqq_close": round(float(row["qqq_close"]), 4),
                "execution_close": round(float(row["asset_close"]), 4),
                "vix_label": str(row["vix_label"]),
                "ixic_trend_label": str(row["ixic_trend_label"]),
                "rel_strength_label": str(row["rel_strength_label"]),
            }
        )

    output_path = ROOT / str(config["shadow_log_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    payload = {
        "config": str(Path(args.config).resolve()),
        "summary": {
            "total_return_pct": summary["total_return_pct"],
            "max_drawdown_pct": summary["max_drawdown_pct"],
            "yearly_returns_pct": summary["yearly_returns_pct"],
            "trades": summary["trades"],
        },
        "log_path": str(output_path.resolve()),
        "days": len(rows),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
