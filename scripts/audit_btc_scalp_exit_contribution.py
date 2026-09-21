#!/usr/bin/env python3
"""出场改动的「自身 pnl 贡献」诊断（memory 要求的方法论）。

`memory/path_dependent_overfitting_detection.md` 的核心判据：

> 判定真实 edge 的核心指标：看改进是否直接改善目标组的「自身 pnl 贡献」，
> 而不是总收益。… 改动一个直接贡献 ≈0 的小参数组，总收益却大幅变化，
> 说明收益来自 capital 路径改变，不是该组参数的真实价值。

本脚本把「基准配置」与「候选配置」在同一窗口各跑一次，按
`(entry_time, direction)` 匹配同一笔入场，然后：

1. 用 **风险归一化的 R**（`pnl / risk_amount`）而不是复利收益比较 —— 去掉 capital 放大；
2. 把匹配交易分成：
   - **改变了出场原因的子集**（true edge 只能来自这里）
   - **出场原因不变、但 R 有变化的子集**
   - **未匹配的交易**（入场序列被改动导致，属路径效应）
3. 报告各子集的 R 贡献与总差异，判断改进是「自身贡献」还是「放大效应」。

用法::

    python3 scripts/audit_btc_scalp_exit_contribution.py \
        --candidate '{"enable_atr_trailing": false}' --start-date 2023-01-01
"""

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

from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.scan_btc_scalp_optimization import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_DATA_15M,
    DEFAULT_DATA_4H,
    load_base_config,
)

DEFAULT_OUTPUT = ROOT / "var" / "tmp" / "btc_opt" / "exit_contribution.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="出场改动的自身 pnl 贡献诊断")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--candidate", required=True, help="候选覆盖键的 JSON 字符串")
    parser.add_argument("--name", default="candidate")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def trades_with_r(trades: pd.DataFrame) -> pd.DataFrame:
    frame = trades.copy()
    risk = (frame["quantity"] * (frame["entry_price"] - frame["initial_stop_price"]).abs()).replace(
        0.0, float("nan")
    )
    frame["risk_amount_rebuilt"] = risk
    frame["net_r"] = frame["pnl"] / risk
    frame["gross_r"] = frame["gross_pnl"] / risk
    frame["fee_r"] = frame["fees"] / risk
    frame["key"] = frame["entry_time"].astype(str) + "|" + frame["direction"].astype(str)
    return frame


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"trades": 0, "net_r": 0.0, "gross_r": 0.0, "fee_r": 0.0}
    return {
        "trades": int(len(frame)),
        "net_r": round(float(frame["net_r"].sum()), 3),
        "gross_r": round(float(frame["gross_r"].sum()), 3),
        "fee_r": round(float(frame["fee_r"].sum()), 3),
    }


def main() -> None:
    args = parse_args()
    overrides = json.loads(args.candidate)
    base_payload = load_base_config(Path(args.config))
    candidate_payload = dict(base_payload)
    candidate_payload.update(overrides)

    prepared = load_prepared_data(
        Path(args.data_15m),
        Path(args.data_4h),
        pd.Timestamp(args.start_date, tz="UTC"),
        base_payload.get("regime_switcher_thresholds"),
    )

    base_metrics, base_engine = run_engine(base_payload, prepared, args.start_date)
    cand_metrics, cand_engine = run_engine(candidate_payload, prepared, args.start_date)

    base = trades_with_r(trade_dataframe(base_engine))
    cand = trades_with_r(trade_dataframe(cand_engine))
    merged = base.merge(
        cand,
        on="key",
        how="outer",
        suffixes=("_base", "_cand"),
        indicator=True,
    )

    matched = merged[merged["_merge"] == "both"].copy()
    matched["exit_changed"] = matched["exit_reason_base"] != matched["exit_reason_cand"]
    matched["r_delta"] = matched["net_r_cand"] - matched["net_r_base"]
    changed = matched[matched["exit_changed"]]
    unchanged = matched[~matched["exit_changed"]]
    only_base = merged[merged["_merge"] == "left_only"]
    only_cand = merged[merged["_merge"] == "right_only"]

    report: dict[str, Any] = {
        "start_date": args.start_date,
        "overrides": overrides,
        "base": {
            **summarize(base),
            "total_return_pct": round(float(base_metrics.get("total_return_pct", 0.0)), 2),
            "max_drawdown_pct": round(float(base_metrics.get("max_drawdown_pct", 0.0)), 2),
        },
        "candidate": {
            **summarize(cand),
            "total_return_pct": round(float(cand_metrics.get("total_return_pct", 0.0)), 2),
            "max_drawdown_pct": round(float(cand_metrics.get("max_drawdown_pct", 0.0)), 2),
        },
        "matched": {
            "count": int(len(matched)),
            "exit_changed_count": int(len(changed)),
            "exit_changed_net_r_delta": round(float(changed["r_delta"].sum()), 3),
            "unchanged_count": int(len(unchanged)),
            "unchanged_net_r_delta": round(float(unchanged["r_delta"].sum()), 3),
        },
        "unmatched": {
            "only_base_count": int(len(only_base)),
            "only_base_net_r": round(float(only_base["net_r_base"].fillna(0.0).sum()), 3),
            "only_cand_count": int(len(only_cand)),
            "only_cand_net_r": round(float(only_cand["net_r_cand"].fillna(0.0).sum()), 3),
        },
        "exit_reason_transitions": (
            changed.groupby(["exit_reason_base", "exit_reason_cand"])["r_delta"]
            .agg(["count", "sum"])
            .reset_index()
            .rename(columns={"sum": "net_r_delta"})
            .round(3)
            .to_dict(orient="records")
        ),
    }
    net_total_delta = report["candidate"]["net_r"] - report["base"]["net_r"]
    report["net_r_total_delta"] = round(net_total_delta, 3)
    report["share_from_exit_changed"] = (
        round(report["matched"]["exit_changed_net_r_delta"] / net_total_delta, 3)
        if net_total_delta not in (0.0,)
        else None
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"\n输出: {output_path}")


if __name__ == "__main__":
    main()
