#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组合稳健性敏感性（round 22）：crypto 腿的构造方式会不会决定结论？

问题：如果"加 20% BTC 改善组合"只在某个特定 BTC 规则/参数下成立，那它就是挑出来的。
本脚本对 crypto 腿做三类扰动，看结论是否稳健：
  1. 规则：买入持有 / MA50 / MA100 / MA200 / MA300 / 波动率目标 40% / 60%
  2. 权重：10% / 20% / 30%
  3. MA 窗口细扫：30~300 共 9 档，并做前后半样本拆分

窗口：2014-09-17 ~ 2026-05-29（11.7 年）；QQQ 腿 = 阶梯 1.25x（overlay 全关）。

用法: python3 research/qqq_spot_leverage/portfolio_robustness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import portfolio_long_sample as PL  # noqa: E402

OUT = HERE / "out"


def btc_aligned(qidx: pd.DatetimeIndex) -> pd.Series:
    b = pd.read_feather(PL.REPO / "data" / "public" / "etf" / "BTC-USD-1d.feather")
    b["d"] = pd.to_datetime(b["date"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    b = b.sort_values("d").drop_duplicates("d", keep="last").set_index("d")["close"]
    return b.reindex(b.index.union(qidx)).ffill().reindex(qidx)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    q = PL.qqq_leg(1.25)
    qidx = q.loc[PL.START:PL.END].dropna().index
    al = btc_aligned(qidx)
    rb = al.pct_change()

    variants = {
        "买持": rb,
        "MA50": (al > al.rolling(50).mean()).shift(1).fillna(False).astype(float) * rb,
        "MA100": (al > al.rolling(100).mean()).shift(1).fillna(False).astype(float) * rb,
        "MA200": (al > al.rolling(200).mean()).shift(1).fillna(False).astype(float) * rb,
        "MA300": (al > al.rolling(300).mean()).shift(1).fillna(False).astype(float) * rb,
        "波动目标40%": (0.40 / (rb.rolling(20).std() * 252 ** 0.5)).clip(upper=1.0).shift(1) * rb,
        "波动目标60%": (0.60 / (rb.rolling(20).std() * 252 ** 0.5)).clip(upper=1.0).shift(1) * rb,
    }
    base = PL.stats(q.reindex(qidx).dropna())

    rows = []
    for name, c in variants.items():
        d = pd.DataFrame({"Q": q.reindex(qidx), "C": c}).dropna()
        for w in [0.1, 0.2, 0.3]:
            s = PL.stats((1 - w) * d["Q"] + w * d["C"])
            rows.append({"variant": name, "w_btc": w, **s, "sharpe_gain": round(s["sharpe"] - base["sharpe"], 2),
                         "mdd_gain": round(s["mdd_pct"] - base["mdd_pct"], 1)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "portfolio_variant_sensitivity.csv", index=False)

    # MA 窗口细扫 + 前后半样本
    ma_rows = []
    for win in [30, 50, 75, 100, 125, 150, 200, 250, 300]:
        c = (al > al.rolling(win).mean()).shift(1).fillna(False).astype(float) * rb
        d = pd.DataFrame({"Q": q.reindex(qidx), "C": c}).dropna()
        half = len(d) // 2
        s_full = PL.stats(0.8 * d["Q"] + 0.2 * d["C"])
        s_h1 = PL.stats(0.8 * d["Q"].iloc[:half] + 0.2 * d["C"].iloc[:half])
        s_h2 = PL.stats(0.8 * d["Q"].iloc[half:] + 0.2 * d["C"].iloc[half:])
        ma_rows.append({"ma_window": win, "cagr": s_full["cagr_pct"], "mdd": s_full["mdd_pct"],
                        "sharpe": s_full["sharpe"], "calmar": s_full["calmar"],
                        "sharpe_h1": s_h1["sharpe"], "sharpe_h2": s_h2["sharpe"]})
    ma = pd.DataFrame(ma_rows)
    # 前后半样本的单腿基准
    d_all = pd.DataFrame({"Q": q.reindex(qidx), "C": variants["MA200"]}).dropna()
    half = len(d_all) // 2
    base_h1 = PL.stats(d_all["Q"].iloc[:half])["sharpe"]
    base_h2 = PL.stats(d_all["Q"].iloc[half:])["sharpe"]
    ma.to_csv(OUT / "portfolio_ma_window_sensitivity.csv", index=False)

    payload = {
        "window": [str(qidx[0].date()), str(qidx[-1].date())],
        "base_single_leg": base,
        "variant_grid": json.loads(df.to_json(orient="records")),
        "ma_window": json.loads(ma.to_json(orient="records")),
        "base_sharpe_h1": base_h1, "base_sharpe_h2": base_h2,
        "verdict": {
            "sharpe_improves_all_variants_w20": bool((df[(df.w_btc == 0.2)]["sharpe"] > base["sharpe"]).all()),
            "mdd_improves_variants_w20": int((df[(df.w_btc == 0.2)]["mdd_pct"] > base["mdd_pct"]).sum()),
            "n_variants": len(variants),
            "ma_sharpe_min": float(ma["sharpe"].min()), "ma_sharpe_max": float(ma["sharpe"].max()),
            "ma_all_above_base": bool((ma["sharpe"] > base["sharpe"]).all()),
            "both_halves_improve": bool((ma["sharpe_h1"] > base_h1).all() and (ma["sharpe_h2"] > base_h2).all()),
        },
    }
    (OUT / "portfolio_robustness.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float))

    pd.set_option("display.width", 220)
    print(f"窗口 {payload['window'][0]} ~ {payload['window'][1]}")
    print(f"QQQ 单腿基准: CAGR {base['cagr_pct']}% MDD {base['mdd_pct']}% Sharpe {base['sharpe']} Calmar {base['calmar']}")
    print()
    print("=== 规则 × 权重（w=0.2 摘要）===")
    print(df[df.w_btc == 0.2][["variant", "cagr_pct", "mdd_pct", "sharpe", "calmar", "sharpe_gain", "mdd_gain"]].to_string(index=False))
    print()
    print("=== MA 窗口细扫（w=0.2，含前后半样本 Sharpe）===")
    print(ma.to_string(index=False))
    print()
    print(f"单腿前后半 Sharpe: {base_h1:.2f} / {base_h2:.2f}")
    print("判定:", json.dumps(payload["verdict"], ensure_ascii=False))
    print(f"\n输出: {OUT}/portfolio_robustness.json, portfolio_variant_sensitivity.csv, portfolio_ma_window_sensitivity.csv")


if __name__ == "__main__":
    main()
