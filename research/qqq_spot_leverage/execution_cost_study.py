#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""执行摩擦与交易成本敏感性（round 4）。

引擎默认按"信号日次日开盘价"成交 + 1bp 固定成本。现实中：
  * 开盘挂单成交价与理论开盘价有偏差；
  * 越是波动日（往往就是翻转日）偏差越大。
本脚本用两个参数刻画：
  slip_vix_coef : 滑点 = coef × VIX（bp），随波动升高
  slip_move_frac: 滑点额外加上"成交日 |开盘→收盘| 幅度"的 frac 倍（最高 2%）

输出：主区间(2020-2026, CSV 复权) 与长样本(1999-2026, feather) 的收益衰减表，
以及三档执行假设下的"可置信区间"。
"""
from __future__ import annotations

import json
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

SCENARIOS = {
    "S0 理想 (1bp 固定)": dict(cost_bps=1.0, slip_vix_coef=0.0, slip_move_frac=0.0),
    "S0b 推荐现实 (7bp 固定)": dict(cost_bps=7.0, slip_vix_coef=0.0, slip_move_frac=0.0),
    "S1 乐观 (3bp 固定)": dict(cost_bps=3.0, slip_vix_coef=0.0, slip_move_frac=0.0),
    "S2 现实 (7bp 折算: VIX系数0.1 + 1bp)": dict(cost_bps=1.0, slip_vix_coef=0.1, slip_move_frac=0.0),
    "S3 保守 (0.1×当日振幅)": dict(cost_bps=1.0, slip_vix_coef=0.05, slip_move_frac=0.10),
    "S4 悲观 (0.3×当日振幅)": dict(cost_bps=2.0, slip_vix_coef=0.1, slip_move_frac=0.30),
}
MAIN = ("2020-01-02", "2026-08-14")
LONG = ("1999-03-10", "2026-05-29")
OOS = ("2011-04-26", "2019-12-31")


def eval_window(frame: pd.DataFrame, p: Params, win: tuple[str, str], source: str) -> dict:
    ind = prepare_indicators(frame, p)
    sub = ind.loc[(ind.index >= pd.Timestamp(win[0])) & (ind.index <= pd.Timestamp(win[1]))]
    path = run_engine(sub, build_target_series(sub, p), p, source)["path"]
    m = metrics(path)
    return {
        "cagr_pct": m["cagr_pct"],
        "mdd_pct": m["max_drawdown_pct"],
        "sharpe": m["sharpe"],
        "calmar": m["calmar"],
        "cost_yr_pct": m["cost_drag_annual_pct"],
        "rebalances": m["rebalances"],
        "years": m["years"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame_csv = load_market_frame("2001-01-02", MAIN[1], price_source="csv")
    frame_long = load_market_frame(LONG[0], LONG[1], price_source="feather")
    rows = []
    for name, kw in SCENARIOS.items():
        p = replace(Params(), **kw)
        r_main = eval_window(frame_csv, p, MAIN, name)
        r_oos = eval_window(frame_csv, p, OOS, name)
        r_long = eval_window(frame_long, p, LONG, name)
        rows.append(
            {
                "scenario": name,
                "main_2020_26_cagr": r_main["cagr_pct"],
                "main_2020_26_mdd": r_main["mdd_pct"],
                "main_2020_26_calmar": r_main["calmar"],
                "main_cost_yr": r_main["cost_yr_pct"],
                "oos_2011_19_cagr": r_oos["cagr_pct"],
                "long_1999_26_cagr": r_long["cagr_pct"],
                "long_cost_yr": r_long["cost_yr_pct"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "execution_cost_sensitivity.csv", index=False)

    # 参数扫描：VIX 系数 × 振幅系数
    sweep = []
    for coef in [0.0, 0.05, 0.1, 0.2]:
        for frac in [0.0, 0.05, 0.1, 0.2, 0.3]:
            p = replace(Params(), cost_bps=1.0, slip_vix_coef=coef, slip_move_frac=frac)
            m = eval_window(frame_csv, p, MAIN, "main")
            l = eval_window(frame_long, p, LONG, "long")
            sweep.append({"vix_coef": coef, "move_frac": frac, "main_cagr": m["cagr_pct"], "main_calmar": m["calmar"],
                          "long_cagr": l["cagr_pct"], "main_cost_yr": m["cost_yr_pct"]})
    sw = pd.DataFrame(sweep)
    sw.to_csv(OUT / "execution_cost_sweep.csv", index=False)

    (OUT / "execution_cost_sensitivity.json").write_text(
        json.dumps({"scenarios": SCENARIOS, "rows": json.loads(df.to_json(orient="records")),
                    "sweep": json.loads(sw.to_json(orient="records"))}, ensure_ascii=False, indent=2)
    )

    pd.set_option("display.width", 240)
    print("=== 执行假设情景（v3 参数）===")
    print(df.to_string(index=False))
    print()
    piv = sw.pivot(index="vix_coef", columns="move_frac", values="main_cagr").round(1)
    print("=== 主区间 CAGR：VIX 系数(行) × 振幅系数(列) ===")
    print(piv.to_string())
    print()
    piv2 = sw.pivot(index="vix_coef", columns="move_frac", values="long_cagr").round(1)
    print("=== 长样本 CAGR：VIX 系数(行) × 振幅系数(列) ===")
    print(piv2.to_string())
    print(f"\n输出: {OUT}/execution_cost_sensitivity.csv, execution_cost_sweep.csv")


if __name__ == "__main__":
    main()
