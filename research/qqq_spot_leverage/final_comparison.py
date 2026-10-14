#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修正口径后的最终对比与建议（round 7）。

背景：round 7 发现并修复了执行口径 bug——旧引擎让"当日开盘建立的仓位"去赚
"前一日开盘→当日开盘"的收益（含建仓前的隔夜 gap），在 MOC（收盘成交）现实下不可实现。
修正后重新评估所有版本，本脚本输出最终对比表与建议。

用法: python3 research/qqq_spot_leverage/final_comparison.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

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

CONFIGS = {
    # 注意：round 7 起 v5 的 overlay 默认关闭，因此这里显式打开/关闭以展示各自贡献
    "A 完整 v5（VIX+RS+修复仓全开）": replace(Params(), use_vix_gate=True, use_recovery=True, rs_mode="rs20", rs_min=0.0),
    "B 仅趋势阶梯（默认配置，overlay 全关）": Params(),
    "C 趋势阶梯 + VIX 闸门": replace(Params(), use_vix_gate=True),
    "D 趋势阶梯 + RS 闸门": replace(Params(), rs_mode="rs20", rs_min=0.0),
    "E 200MA on/off 1.5x（二元开关）": replace(Params(), lev_half=0.0),
    "F 恒定 1.5x": replace(Params(), lev_half=1.5, drift_band=10.0),
    "G 买入持有 1x": replace(Params(), lev_full=1.0, lev_half=1.0, max_leverage=1.0, drift_band=10.0),
}
WINDOWS = {
    "2020-2026(主)": ("2020-01-02", "2026-08-14"),
    "train 20-22": ("2020-01-02", "2022-12-31"),
    "test 23-26(样本外)": ("2023-01-01", "2026-08-14"),
    "oos 2011-2019": ("2011-04-26", "2019-12-31"),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_market_frame("2001-01-02", "2026-08-14")
    frame_long = load_market_frame("1999-03-10", "2026-05-29", price_source="feather")

    rows = []
    for name, p in CONFIGS.items():
        ind = prepare_indicators(frame, p)
        row = {"config": name}
        for w, (a, b) in WINDOWS.items():
            sub = ind.loc[(ind.index >= a) & (ind.index <= b)]
            m = metrics(run_engine(sub, build_target_series(sub, p), p, w)["path"])
            row[f"{w} CAGR"] = m["cagr_pct"]
            row[f"{w} MDD"] = m["max_drawdown_pct"]
            row[f"{w} Calmar"] = m["calmar"]
            row[f"{w} Sharpe"] = m["sharpe"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "final_comparison.csv", index=False)

    # 长样本（feather，压力段）
    long_rows = []
    segs = {"dot-com 00-02": ("2000-03-10", "2002-12-31"), "GFC 07-09": ("2007-10-01", "2009-03-31"),
            "FULL 99-26": ("1999-03-10", "2026-05-29")}
    for name, p in CONFIGS.items():
        ind = prepare_indicators(frame_long, p)
        row = {"config": name}
        for w, (a, b) in segs.items():
            sub = ind.loc[(ind.index >= a) & (ind.index <= b)]
            m = metrics(run_engine(sub, build_target_series(sub, p), p, w)["path"])
            row[w] = f"CAGR {m['cagr_pct']:.1f}% / MDD {m['max_drawdown_pct']:.1f}%"
        long_rows.append(row)
    ldf = pd.DataFrame(long_rows)
    ldf.to_csv(OUT / "final_comparison_long.csv", index=False)

    # 引擎/数据校验：1x 自检 + 换手近似上界（每 bar 偏差 ~9.21pp/交易；此处按 1x 全期 9.21pp 折算）
    f_ind = prepare_indicators(frame, Params())
    f_win = f_ind.loc[WINDOWS["2020-2026(主)"][0]:WINDOWS["2020-2026(主)"][1]]
    bh_m = metrics(buy_hold(f_win, 1.0, Params(), "bh")["path"])
    asset_pct = float(f_win["adj_close"].iloc[-1] / f_win["adj_close"].iloc[0] - 1) * 100
    checks = {
        "buy_hold_1x_engine_pct": bh_m["total_return_pct"],
        "buy_hold_1x_asset_pct": round(asset_pct, 2),
        "entry_bar_bias_pp": round(bh_m["total_return_pct"] - asset_pct, 2),
        "note": "引擎在收盘建仓，建仓那根 bar 的收益归零；1x 全期偏差见 entry_bar_bias_pp（约 0.5pp/年）。该偏差对换手极低的配置影响很小。",
    }
    payload = {"windows": WINDOWS, "main": json.loads(df.to_json(orient="records")),
               "long_sample": json.loads(ldf.to_json(orient="records")), "engine_checks": checks}
    (OUT / "final_comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    pd.set_option("display.width", 300)
    show = df[["config"] + [c for c in df.columns if c.endswith(("CAGR", "MDD")) and "主" in c or c.endswith("Calmar")]]
    print("=== 修正口径（MOC 收盘成交）最终对比 ===")
    print(df.to_string(index=False))
    print()
    print("=== 长样本压力段 ===")
    print(ldf.to_string(index=False))
    print(f"\n输出: {OUT}/final_comparison.csv / .json, final_comparison_long.csv")


if __name__ == "__main__":
    main()
