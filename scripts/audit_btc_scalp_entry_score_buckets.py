#!/usr/bin/env python3
"""入场端的**多周期分数分桶期望值**审计（无前视口径）。

目的：出场/风控层已被证明没有真实 edge 空间（见
`docs/btc_scalp_optimization_20260921.md` §4.4/§5），剩下的空间在入场端。
实盘真正生效的入场门是 `bot/okx_executor.py` 的 SOTA score gate
（`net_min=3 / bull_min=8 / bear_max=6 / conflict_mode=any`，见
`_sota_score_gate_rule`），但 `live_readiness_report` 的引擎口径**不含**它。

本脚本把引擎跑出的每一笔入场，按同一套 `score_snapshot`（4h confirmed + 1h + 15m）
打分，然后：

1. 按 bull_total / net_score 分桶，给出每桶的**风险归一化净 R、毛 R、胜率**；
2. 复现实盘 score gate，比较「被接受」与「被拒绝」两组的期望值；
3. 报告 score 与 R 的相关性（Spearman），判断分数是否真有区分度。

用法::

    python3 scripts/audit_btc_scalp_entry_score_buckets.py --start-date 2023-01-01
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.confirmed_multiframe_score_utils import (  # noqa: E402
    align_confirmed_mapping,
    passes_score_gate,
    resample_confirmed_1h,
    score_snapshot,
)
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.scan_btc_scalp_optimization import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_DATA_15M,
    DEFAULT_DATA_4H,
    load_base_config,
)

DEFAULT_OUTPUT = ROOT / "var" / "tmp" / "btc_opt" / "entry_score_buckets.json"
DEFAULT_RULE = {"net_min": 3, "bull_min": 8, "bear_max": 6, "conflict_mode": "any"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="入场分数分桶期望值审计")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def bucket_stats(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(column):
        rows.append(
            {
                column: int(value),
                "trades": int(len(group)),
                "net_r_sum": round(float(group["net_r"].sum()), 2),
                "avg_net_r": round(float(group["net_r"].mean()), 4),
                "gross_r_sum": round(float(group["gross_r"].sum()), 2),
                "win_rate_pct": round(float((group["net_r"] > 0).mean() * 100.0), 1),
                "avg_fee_r": round(float(group["fee_r"].mean()), 4),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    payload = load_base_config(Path(args.config))
    prepared = load_prepared_data(
        Path(args.data_15m),
        Path(args.data_4h),
        pd.Timestamp(args.start_date, tz="UTC"),
        payload.get("regime_switcher_thresholds"),
    )
    metrics, engine = run_engine(payload, prepared, args.start_date)
    trades = trade_dataframe(engine)
    risk = (trades["quantity"] * (trades["entry_price"] - trades["initial_stop_price"]).abs()).replace(
        0.0, float("nan")
    )
    trades["net_r"] = trades["pnl"] / risk
    trades["gross_r"] = trades["gross_pnl"] / risk
    trades["fee_r"] = trades["fees"] / risk

    c1h = resample_confirmed_1h(prepared.c15m)
    mapping_1h = align_confirmed_mapping(c1h, prepared.c15m)
    snapshots: list[dict[str, Any]] = []
    for entry_idx in trades["entry_idx"].astype(int):
        snapshots.append(asdict(score_snapshot(prepared, c1h, mapping_1h, entry_idx)))
    scored = pd.concat([trades.reset_index(drop=True), pd.DataFrame(snapshots)], axis=1)
    scored["accepted_by_live_rule"] = scored.apply(
        lambda row: passes_score_gate(
            {
                "net_score": row["net_score"],
                "bull_total": row["bull_total"],
                "bear_total": row["bear_total"],
                "conflict": bool(row["conflict"]),
            },
            **DEFAULT_RULE,
        ),
        axis=1,
    )

    accepted = scored[scored["accepted_by_live_rule"]]
    rejected = scored[~scored["accepted_by_live_rule"]]

    report: dict[str, Any] = {
        "start_date": args.start_date,
        "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 2),
        "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0)), 2),
        "trades": int(len(scored)),
        "rule": DEFAULT_RULE,
        "accepted": {
            "trades": int(len(accepted)),
            "net_r_sum": round(float(accepted["net_r"].sum()), 2),
            "avg_net_r": round(float(accepted["net_r"].mean()), 4) if len(accepted) else None,
            "win_rate_pct": round(float((accepted["net_r"] > 0).mean() * 100.0), 1) if len(accepted) else None,
        },
        "rejected": {
            "trades": int(len(rejected)),
            "net_r_sum": round(float(rejected["net_r"].sum()), 2),
            "avg_net_r": round(float(rejected["net_r"].mean()), 4) if len(rejected) else None,
            "win_rate_pct": round(float((rejected["net_r"] > 0).mean() * 100.0), 1) if len(rejected) else None,
        },
        "bull_total_buckets": bucket_stats(scored, "bull_total"),
        "net_score_buckets": bucket_stats(scored, "net_score"),
        "spearman": {
            "bull_total_vs_net_r": round(float(scored["bull_total"].corr(scored["net_r"], method="spearman")), 4),
            "net_score_vs_net_r": round(float(scored["net_score"].corr(scored["net_r"], method="spearman")), 4),
        },
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"\n输出: {output_path}")


if __name__ == "__main__":
    main()
