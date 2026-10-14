#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""趋势规则对比实验（round 2）。

在同一套 regime 闸门 / 修复再入场 / 执行口径下，对比不同趋势规则的样本内、
样本外、压力段表现，并检验 VIX 使用口径（当日 vs 前一日）是否影响结论。

    dual_ma      : 收盘>MA200 且 MA50>MA200 满杠杆；仅收盘>MA200 半杠杆（当前主策略）
    dual_ma_flat : 收盘>MA200 且 MA200 未显著下行
    donchian     : 收盘创 slow 日新高（突破）
    ema_dual     : EMA50 > EMA200

用法: python3 research/qqq_spot_leverage/compare_trend_models.py
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
    buy_hold,
    load_market_frame,
    metrics,
    prepare_indicators,
    run_engine,
)

OUT = Path(__file__).resolve().parent / "out"
END = "2026-08-14"
WINDOWS = {
    "2020-2026": ("2020-01-02", END),
    "train 2020-2022": ("2020-01-02", "2022-12-31"),
    "test 2023-2026": ("2023-01-01", END),
    "oos 2011-2019": ("2011-04-26", "2019-12-31"),
    "stress 2001-2004": ("2001-01-02", "2004-12-31"),
}
MODES = ["dual_ma", "dual_ma_flat", "donchian", "ema_dual"]


def evaluate(frame: pd.DataFrame, p: Params) -> dict:
    out = {}
    ind_all = prepare_indicators(frame, p)
    for name, (a, b) in WINDOWS.items():
        sub = ind_all.loc[(ind_all.index >= pd.Timestamp(a)) & (ind_all.index <= pd.Timestamp(b))]
        if len(sub) < 60:
            out[name] = None
            continue
        run = run_engine(sub, build_target_series(sub, p), p, name)
        m = metrics(run["path"])
        out[name] = {
            "cagr": m["cagr_pct"],
            "mdd": m["max_drawdown_pct"],
            "sharpe": m["sharpe"],
            "calmar": m["calmar"],
            "avg_lev": m["avg_leverage"],
            "rebalances": m["rebalances"],
            "total": m["total_return_pct"],
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_market_frame("2001-01-02", END)
    base = Params()

    rows = []
    # 1) 趋势规则 × slow_window
    for mode in MODES:
        for slow in [150, 200, 250]:
            p = replace(base, trend_mode=mode, slow_window=slow)
            res = evaluate(frame, p)
            row = {"trend_mode": mode, "slow": slow}
            for w, m in res.items():
                row[f"{w}|cagr"] = None if m is None else m["cagr"]
                row[f"{w}|mdd"] = None if m is None else m["mdd"]
                row[f"{w}|calmar"] = None if m is None else m["calmar"]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "trend_model_comparison.csv", index=False)

    # 2) VIX 口径：当日 vs 前一日
    vix_rows = []
    for mode in ["dual_ma", "donchian"]:
        for timing in ["same_close", "prior_close"]:
            p = replace(base, trend_mode=mode, vix_timing=timing)
            res = evaluate(frame, p)
            vix_rows.append(
                {
                    "trend_mode": mode,
                    "vix_timing": timing,
                    **{f"{w}|cagr": (None if res[w] is None else res[w]["cagr"]) for w in WINDOWS},
                    **{f"{w}|mdd": (None if res[w] is None else res[w]["mdd"]) for w in WINDOWS},
                    **{f"{w}|calmar": (None if res[w] is None else res[w]["calmar"]) for w in WINDOWS},
                }
            )
    vdf = pd.DataFrame(vix_rows)
    vdf.to_csv(OUT / "vix_timing_comparison.csv", index=False)

    # 3) 基准（买入持有）
    bench = {}
    for name, (a, b) in WINDOWS.items():
        sub = frame.loc[(frame.index >= pd.Timestamp(a)) & (frame.index <= pd.Timestamp(b))]
        if len(sub) < 60:
            bench[name] = None
            continue
        ind = prepare_indicators(frame, base).loc[sub.index]
        m = metrics(buy_hold(ind, 1.0, base, name)["path"])
        bench[name] = {"cagr": m["cagr_pct"], "mdd": m["max_drawdown_pct"], "sharpe": m["sharpe"], "calmar": m["calmar"]}

    (OUT / "trend_model_comparison.json").write_text(
        json.dumps({"windows": WINDOWS, "benchmark_1x": bench, "rows": json.loads(df.to_json(orient="records")),
                    "vix_timing": json.loads(vdf.to_json(orient="records"))}, ensure_ascii=False, indent=2)
    )

    pd.set_option("display.width", 240)
    print("基准买入持有 1x：")
    for w, m in bench.items():
        print(f"  {w:<18} " + ("n/a" if m is None else f"CAGR {m['cagr']:>6.2f}%  MDD {m['mdd']:>7.2f}%  Calmar {m['calmar']:.3f}"))
    print()
    show = df[["trend_mode", "slow",
               "2020-2026|cagr", "2020-2026|mdd", "2020-2026|calmar",
               "test 2023-2026|cagr", "test 2023-2026|calmar",
               "oos 2011-2019|cagr", "oos 2011-2019|calmar",
               "stress 2001-2004|cagr", "stress 2001-2004|mdd"]]
    print(show.to_string(index=False))
    print()
    print(vdf.to_string(index=False))
    print(f"\n输出: {OUT}/trend_model_comparison.csv, vix_timing_comparison.csv, trend_model_comparison.json")


if __name__ == "__main__":
    main()
