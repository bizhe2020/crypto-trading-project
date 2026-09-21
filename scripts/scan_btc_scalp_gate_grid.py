#!/usr/bin/env python3
"""影子风控闸门参数网格 —— 引擎级 + 季度切片（无前视口径）。

动机：`memory/shadow_gate_optimal_scan_20260827.md` 在**可能含前视的 harness**下
得出「12-2/6-4 就是最优」。该记忆自己标注了「待复核」。
本脚本在**已对齐的无前视口径**（`build_precomputed_state_confirmed_4h`）下，
把闸门真正接进引擎（`enable_shadow_risk_gate_backtest`）重跑同一张网格，
并按 memory 要求输出**季度切片**，用来判定「某个格子更优」是真 edge 还是单季度幻觉。

用法::

    python3 scripts/scan_btc_scalp_gate_grid.py --grid dd_cool --start-date 2023-01-01
    python3 scripts/scan_btc_scalp_gate_grid.py --grid daily_streak --start-date 2023-01-01
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
from scripts.scan_btc_scalp_quarters import quarterly_curve  # noqa: E402

DEFAULT_OUTPUT = ROOT / "var" / "tmp" / "btc_opt" / "gate_grid.json"

DD_VALUES = [8.0, 10.0, 12.0, 15.0, 20.0]
COOL_VALUES = [1, 2, 3, 5]
DAILY_VALUES = [4.0, 6.0, 8.0, 10.0]
STREAK_VALUES = [3, 4, 5, 6]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="闸门参数网格（引擎级 + 季度切片）")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--grid", default="dd_cool", choices=("dd_cool", "daily_streak"))
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def build_variants(grid: str, base_payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """生成网格候选。每条都显式打开引擎级闸门。"""
    common = {"enable_shadow_risk_gate": True, "enable_shadow_risk_gate_backtest": True}
    variants: list[tuple[str, dict[str, Any]]] = []
    if grid == "dd_cool":
        for dd in DD_VALUES:
            for cool in COOL_VALUES:
                payload = dict(base_payload)
                payload.update(common)
                payload["shadow_equity_drawdown_stop_pct"] = dd
                payload["shadow_equity_drawdown_cooldown_days"] = cool
                variants.append((f"dd{dd:g}_cool{cool}", payload))
    else:
        for daily in DAILY_VALUES:
            for streak in STREAK_VALUES:
                payload = dict(base_payload)
                payload.update(common)
                payload["shadow_daily_loss_stop_pct"] = daily
                payload["shadow_consecutive_loss_stop"] = streak
                variants.append((f"daily{daily:g}_streak{streak}", payload))
    return variants


def concentration(quarters: dict[str, dict[str, float]], base_quarters: dict[str, dict[str, float]]) -> dict[str, Any]:
    """改进是否集中在单一季度：给出每季超额与最大单季占比。"""
    diffs: dict[str, float] = {}
    for quarter, item in quarters.items():
        base_value = base_quarters.get(quarter, {}).get("return_pct")
        if base_value is None:
            continue
        diffs[quarter] = item.get("return_pct", 0.0) - base_value
    positive = {q: d for q, d in diffs.items() if d > 0}
    total_positive = sum(positive.values())
    best_quarter = max(diffs, key=lambda q: diffs[q]) if diffs else None
    best_value = diffs.get(best_quarter, 0.0) if best_quarter else 0.0
    return {
        "beat_quarters": len(positive),
        "total_quarters": len(diffs),
        "best_quarter": best_quarter,
        "best_quarter_excess_pct": round(best_value, 1),
        "best_share_of_positive": round(best_value / total_positive, 3) if total_positive > 0 else None,
    }


def main() -> None:
    args = parse_args()
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

    print("→ 基线（引擎不含闸门）...", flush=True)
    base_metrics, base_engine = run_engine(dict(base_payload), prepared, args.start_date)
    base_quarters = quarterly_curve(
        trade_dataframe(base_engine), float(base_metrics.get("initial_capital", 1000.0))
    )
    base_total = float(base_metrics.get("total_return_pct", 0.0))
    print(f"   base total={base_total:.1f}%", flush=True)

    rows: list[dict[str, Any]] = []
    for name, payload in build_variants(args.grid, base_payload):
        print(f"→ {name} ...", flush=True)
        metrics, engine = run_engine(payload, prepared, args.start_date)
        trades = trade_dataframe(engine)
        quarters = quarterly_curve(trades, float(metrics.get("initial_capital", 1000.0)))
        rows.append(
            {
                "name": name,
                "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 2),
                "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0)), 2),
                "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
                "total_trades": int(metrics.get("total_trades", 0)),
                "quarters": {q: round(v.get("return_pct", 0.0), 2) for q, v in quarters.items()},
                **concentration(quarters, base_quarters),
            }
        )

    print(f"\n== 网格 {args.grid}（{args.start_date} 起，base={base_total:.1f}%）==")
    header = f"{'variant':<18}{'total%':>11}{'dd%':>8}{'sharpe':>8}{'n':>6}{'beat':>7}{'best_q':>9}{'best%':>9}{'share':>7}"
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: -r["total_return_pct"]):
        print(
            f"{row['name']:<18}{row['total_return_pct']:>11.1f}{row['max_drawdown_pct']:>8.1f}"
            f"{row['sharpe_ratio']:>8.2f}{row['total_trades']:>6d}{row['beat_quarters']:>4d}/{row['total_quarters']:<2d}"
            f"{str(row['best_quarter']):>9}{row['best_quarter_excess_pct']:>9.1f}"
            f"{(row['best_share_of_positive'] if row['best_share_of_positive'] is not None else float('nan')):>7.2f}"
        )
    print("\n（share = 最大单季超额 / 全部正超额之和；越接近 1 越说明改进集中在单一季度）")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "grid": args.grid,
                "start_date": args.start_date,
                "base": {
                    "total_return_pct": round(base_total, 2),
                    "max_drawdown_pct": round(float(base_metrics.get("max_drawdown_pct", 0.0)), 2),
                    "sharpe_ratio": round(float(base_metrics.get("sharpe_ratio", 0.0)), 3),
                    "quarters": {q: round(v.get("return_pct", 0.0), 2) for q, v in base_quarters.items()},
                },
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"\n输出: {output_path}")


if __name__ == "__main__":
    main()
