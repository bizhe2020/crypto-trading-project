#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VIX 闸门参数在不同"VIX 使用口径"下的稳健性扫描（round 2）。

动机：round 2 发现 VIX gate 的收益高度依赖"当日 VIX 收盘"（15:00 ET 可观测），
一旦改成"前一日 VIX 收盘"（保守口径）CAGR 从 41% 掉到 24%。本脚本回答：
  (a) 是否存在在两种口径下都不差的参数？
  (b) 换用"期限结构 VIX/VIX3M 触发"(term_trigger) 是否比"VIX 绝对水平触发"更稳健？
  (c) 用 5 日平滑 VIX 是否降低口径敏感性？
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
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
WIN_MAIN = ("2020-01-02", "2026-08-14")
WIN_OOS = ("2011-04-26", "2019-12-31")


def scope(ind: pd.DataFrame, p: Params, win: tuple[str, str]) -> dict:
    sub = ind.loc[(ind.index >= pd.Timestamp(win[0])) & (ind.index <= pd.Timestamp(win[1]))]
    return metrics(run_engine(sub, build_target_series(sub, p), p, "x")["path"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_market_frame("2001-01-02", "2026-08-14")
    base = Params()
    rows = []
    for stress in [22, 26, 28, 30, 32, 36, 40]:
        for hyst in [0, 2, 4, 6]:
            for timing in ["same_close", "prior_close"]:
                p = replace(base, vix_stress=stress, vix_hysteresis=hyst, vix_timing=timing)
                ind = prepare_indicators(frame, p)
                m = scope(ind, p, WIN_MAIN)
                o = scope(ind, p, WIN_OOS)
                rows.append(
                    {
                        "stress": stress,
                        "hyst": hyst,
                        "timing": timing,
                        "cagr20": m["cagr_pct"],
                        "mdd20": m["max_drawdown_pct"],
                        "cal20": m["calmar"],
                        "cagr11": o["cagr_pct"],
                        "cal11": o["calmar"],
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "vix_gate_robustness.csv", index=False)

    piv = df.pivot_table(index=["stress", "hyst"], columns="timing", values=["cagr20", "cal20", "cagr11", "cal11"])
    piv.columns = ["_".join(c) for c in piv.columns]
    piv = piv.reset_index()
    piv["min_cal20"] = piv[["cal20_same_close", "cal20_prior_close"]].min(axis=1)
    piv["min_cagr20"] = piv[["cagr20_same_close", "cagr20_prior_close"]].min(axis=1)
    piv["min_cal11"] = piv[["cal11_same_close", "cal11_prior_close"]].min(axis=1)

    pd.set_option("display.width", 260)
    print("=== 按“两种 VIX 口径下最差 Calmar”排序（2020-2026）===")
    print(piv.sort_values("min_cal20", ascending=False).head(12).round(3).to_string(index=False))
    print()
    print("=== 按 same_close Calmar 排序（= 原来的选择标准）===")
    print(piv.sort_values("cal20_same_close", ascending=False).head(8).round(3).to_string(index=False))
    print()
    print("=== 按 prior_close Calmar 排序（= 保守口径下的最优）===")
    print(piv.sort_values("cal20_prior_close", ascending=False).head(8).round(3).to_string(index=False))
    print()

    # 平滑 VIX 对照
    print("=== 5 日平滑 VIX 作为闸门信号 ===")
    sm_rows = []
    for lag in [0, 1, 2]:
        v = frame["vix"].rolling(5).mean()
        f2 = frame.copy()
        f2["vix"] = v.shift(lag)
        for stress in [22, 24, 26, 28]:
            p = replace(base, vix_timing="same_close", vix_shift_days=1, vix_stress=stress)
            ind = prepare_indicators(f2, p)
            m = scope(ind, p, WIN_MAIN)
            sm_rows.append({"vix_smooth": 5, "lag": lag, "stress": stress, "cagr20": m["cagr_pct"], "mdd20": m["max_drawdown_pct"], "cal20": m["calmar"]})
    sm = pd.DataFrame(sm_rows)
    sm.to_csv(OUT / "vix_gate_smoothed.csv", index=False)
    print(sm.round(2).to_string(index=False))
    print(f"\n输出: {OUT}/vix_gate_robustness.csv, vix_gate_smoothed.csv")


if __name__ == "__main__":
    main()
