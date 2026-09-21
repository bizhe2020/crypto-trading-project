#!/usr/bin/env python3
"""BTC scalp 参数 A/B 的**季度切片**分析（memory 方法论要求）。

背景：`memory/path_dependent_overfitting_detection.md` 与
`memory/shadow_gate_optimal_scan_20260827.md` 明确要求 —— 本引擎是
「加法累加 + 仓位随 capital 增长」，任何参数改动的全周期收益差异都可能来自
capital 累积路径依赖；只看全周期收益或「两半方向一致」会漏掉
**单季度集中型幻觉**（例：shadow gate 8-3 的 7521%、lock_r -0.10 的 3621%）。

本脚本对每个候选配置只跑**一次**全区间引擎，然后把权益按季度切片，
输出「哪些季度赢/输」，以及改进是否集中在单一季度。

用法::

    python3 scripts/scan_btc_scalp_quarters.py --set enginegate --start-date 2023-01-01
    python3 scripts/scan_btc_scalp_quarters.py --set final --start-date 2023-01-01
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
    VARIANT_SETS,
    load_base_config,
)

DEFAULT_OUTPUT = ROOT / "var" / "tmp" / "btc_opt" / "quarter_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTC scalp 季度切片 A/B（无前视口径）")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--set", default="enginegate", help="逗号分隔候选集，或 all")
    parser.add_argument("--only", default="", help="只跑名字包含该子串的候选（逗号分隔）")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def quarterly_curve(trades: pd.DataFrame, initial_capital: float) -> dict[str, dict[str, float]]:
    """把交易序列按平仓时间的季度分桶，返回每季收益与季内最大回撤。"""
    if trades.empty:
        return {}
    ordered = trades.sort_values("exit_time").reset_index(drop=True)
    capital = initial_capital
    buckets: dict[str, dict[str, float]] = {}
    for _, trade in ordered.iterrows():
        exit_time = pd.Timestamp(trade["exit_time"])
        quarter = f"{exit_time.year}Q{((exit_time.month - 1) // 3) + 1}"
        bucket = buckets.setdefault(
            quarter,
            {"start_capital": capital, "capital": capital, "peak": capital, "max_dd": 0.0, "pnl": 0.0, "trades": 0.0},
        )
        capital += float(trade["pnl"])
        bucket["capital"] = capital
        bucket["peak"] = max(bucket["peak"], capital)
        bucket["max_dd"] = max(bucket["max_dd"], (bucket["peak"] - capital) / bucket["peak"] * 100.0)
        bucket["pnl"] += float(trade["pnl"])
        bucket["trades"] += 1
    for bucket in buckets.values():
        start = float(bucket["start_capital"])
        bucket["return_pct"] = (float(bucket["capital"]) - start) / start * 100.0 if start > 0 else 0.0
    return buckets


def print_quarter_table(title: str, results: dict[str, dict[str, dict[str, float]]]) -> None:
    quarters = sorted({q for item in results.values() for q in item})
    print(f"\n== {title}（季度收益 %）==")
    header = f"{'variant':<34}" + "".join(f"{q:>10}" for q in quarters) + f"{'total%':>12}{'beat':>6}"
    print(header)
    print("-" * len(header))
    base = results.get("base(live)", {})
    for name, item in results.items():
        row = f"{name:<34}"
        total = 1.0
        beat = 0
        for quarter in quarters:
            value = item.get(quarter, {}).get("return_pct")
            if value is None:
                row += f"{'—':>10}"
                continue
            total *= 1.0 + value / 100.0
            row += f"{value:>10.1f}"
            base_value = base.get(quarter, {}).get("return_pct")
            if base_value is not None and value > base_value:
                beat += 1
        row += f"{(total - 1.0) * 100.0:>12.1f}{beat:>6d}"
        print(row)
    print(f"（beat = 该候选季度收益高于 base(live) 的季度数，共 {len(quarters)} 季）")


def main() -> None:
    args = parse_args()
    sets: list[str] = []
    for token in str(args.set).split(","):
        token = token.strip()
        if not token:
            continue
        if token == "all":
            sets = list(VARIANT_SETS)
            break
        if token not in VARIANT_SETS:
            raise SystemExit(f"未知候选集: {token}")
        sets.append(token)
    only = [t.strip() for t in str(args.only).split(",") if t.strip()]

    base_payload = load_base_config(Path(args.config))
    start = pd.Timestamp(args.start_date, tz="UTC")
    print(f"加载数据 start>={start} ...", flush=True)
    prepared = load_prepared_data(
        Path(args.data_15m),
        Path(args.data_4h),
        start,
        base_payload.get("regime_switcher_thresholds"),
    )
    print(f"范围={prepared.start} → {prepared.end}", flush=True)

    variants: list[tuple[str, dict[str, Any]]] = [("base(live)", dict(base_payload))]
    for set_name in sets:
        for name, overrides in VARIANT_SETS[set_name].items():
            if only and not any(token in name for token in only):
                continue
            payload = dict(base_payload)
            payload.update(overrides)
            variants.append((f"{set_name}.{name}", payload))

    results: dict[str, dict[str, dict[str, float]]] = {}
    raw: dict[str, Any] = {}
    for name, payload in variants:
        print(f"→ {name} ...", flush=True)
        metrics, engine = run_engine(payload, prepared, args.start_date)
        trades = trade_dataframe(engine)
        curve = quarterly_curve(trades, float(metrics.get("initial_capital", 1000.0)))
        results[name] = curve
        raw[name] = {
            "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 2),
            "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0)), 2),
            "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
            "total_trades": int(metrics.get("total_trades", 0)),
            "quarters": {q: {k: round(v, 2) for k, v in item.items()} for q, item in curve.items()},
        }

    print_quarter_table(f"{args.start_date} 起", results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
    print(f"\n输出: {output_path}")


if __name__ == "__main__":
    main()
