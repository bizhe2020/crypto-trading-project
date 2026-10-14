#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组合层风控叠加实验（round 2）。

测试两类"权益路径"叠加（都与趋势/VIX 层正交）：
  D. 回撤降杠杆：权益自峰值回撤 >= X% 时把杠杆打 5 折，修复到 Y% 以内恢复
  R. 涨幅降杠杆：过去 60 日收益 >= Z% 时把杠杆打 5 折（防过热后回撤）
评估窗口：2020-2026（主）/ 2023-2026（样本外）/ 2011-2019（独立历史段）/ 2001-2004（压力）
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_qqq_spot_leverage import (  # noqa: E402
    Params,
    build_target_series,
    load_market_frame,
    metrics,
    prepare_indicators,
    run_engine,
)

OUT = Path(__file__).resolve().parent / "out"
WINDOWS = {
    "2020-2026": ("2020-01-02", "2026-08-14"),
    "test23-26": ("2023-01-01", "2026-08-14"),
    "oos11-19": ("2011-04-26", "2019-12-31"),
    "stress01-04": ("2001-01-02", "2004-12-31"),
}


def evaluate(frame: pd.DataFrame, p: Params) -> dict:
    ind = prepare_indicators(frame, p)
    out = {}
    for w, (a, b) in WINDOWS.items():
        sub = ind.loc[(ind.index >= pd.Timestamp(a)) & (ind.index <= pd.Timestamp(b))]
        if len(sub) < 60:
            continue
        m = metrics(run_engine(sub, build_target_series(sub, p), p, w)["path"])
        out[w] = m
    return out


def main() -> None:
    frame = load_market_frame("2001-01-02", "2026-08-14")
    base = Params()
    rows = []

    def add(label: str, p: Params) -> None:
        res = evaluate(frame, p)
        row = {"label": label}
        for w in WINDOWS:
            m = res.get(w)
            row[f"{w}|cagr"] = None if m is None else m["cagr_pct"]
            row[f"{w}|mdd"] = None if m is None else m["max_drawdown_pct"]
            row[f"{w}|calmar"] = None if m is None else m["calmar"]
            row[f"{w}|sharpe"] = None if m is None else m["sharpe"]
            row[f"{w}|avgl"] = None if m is None else m["avg_leverage"]
        rows.append(row)

    add("baseline v2", base)
    # D. 回撤降杠杆
    for start in [0.08, 0.10, 0.12, 0.15]:
        add(f"D dd>={int(start*100)}% x0.5", replace(base, dd_derisk_start=start, dd_derisk_factor=0.5, dd_rearm_level=0.05))
    add("D dd>=10% x0.0", replace(base, dd_derisk_start=0.10, dd_derisk_factor=0.0, dd_rearm_level=0.05))
    # R. 涨幅降杠杆
    for thr in [0.25, 0.35, 0.50]:
        add(f"R rr60>={int(thr*100)}% x0.5", replace(base, rr_derisk_60d=thr, rr_derisk_factor=0.5))
    # 组合
    add("D+R (dd10 + rr35)", replace(base, dd_derisk_start=0.10, dd_derisk_factor=0.5, dd_rearm_level=0.05, rr_derisk_60d=0.35, rr_derisk_factor=0.5))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "overlay_experiments.csv", index=False)
    pd.set_option("display.width", 260)
    print(df.to_string(index=False))
    print(f"\n输出: {OUT}/overlay_experiments.csv")


if __name__ == "__main__":
    main()
