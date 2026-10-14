#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组合层长样本（round 21）：QQQ 台阶腿 + BTC(MA200) 在 2014-2026 的表现。

为什么做这个
------------
round 13~17 的组合结论只覆盖 2022-10 起（约 3.6 年）或 2020 起（6.6 年）。
但 BTC-USD 日线可回溯到 2014-09，QQQ 可回溯到 1999（feather 连续），
因此可以把"低相关 + 风险预算"这个结论放到**更长的样本**上检验，
并单独看 2018（加密寒冬）、2020（新冠）、2022（股债双杀）这些压力段。

口径
----
* QQQ 腿：双均线阶梯（overlay 全关）。默认用 **1.25x 上限**（交付配置），
  同时给出 1.5x 上限作对照；价格用 feather 连续序列（未含股息，注意口径）。
* BTC 腿：BTC-USD 收盘 > MA200 持有，否则空仓；对齐到 QQQ 交易日。
* 组合：固定权重、季度再平衡，单边 1bp。

用法: python3 research/qqq_spot_leverage/portfolio_long_sample.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backtest_qqq_spot_leverage import (  # noqa: E402
    Params,
    build_target_series,
    load_market_frame,
    prepare_indicators,
    run_engine,
)

OUT = HERE / "out"
REPO = HERE.parents[1]
START = "2014-09-17"
END = "2026-05-29"
COST_BPS = 1.0
REB_MONTHS = {3, 6, 9, 12}


def qqq_leg(cap: float) -> pd.Series:
    frame = load_market_frame("2001-01-02", END, price_source="feather")
    p = Params(rs_mode="", use_vix_gate=False, use_recovery=False,
               lev_full=cap, max_leverage=cap, lev_half=min(1.0, cap), recovery_lev=min(0.75, cap))
    ind = prepare_indicators(frame, p)
    return run_engine(ind, build_target_series(ind, p), p, "q")["path"]["net_ret"]


def btc_leg(qidx: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    b = pd.read_feather(REPO / "data" / "public" / "etf" / "BTC-USD-1d.feather")
    b["d"] = pd.to_datetime(b["date"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    b = b.sort_values("d").drop_duplicates("d", keep="last").set_index("d")["close"]
    al = b.reindex(b.index.union(qidx)).ffill().reindex(qidx)
    ma = al.rolling(200).mean()
    on = (al > ma).shift(1).fillna(False).astype(float)
    return on * al.pct_change(), al


def stats(r: pd.Series) -> dict:
    eq = (1 + r.fillna(0)).cumprod()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    mdd = float((eq / eq.cummax() - 1).min())
    return {
        "cagr_pct": round(float(eq.iloc[-1] ** (1 / yrs) - 1) * 100, 2),
        "mdd_pct": round(mdd * 100, 2),
        "vol_pct": round(float(r.std() * math.sqrt(252)) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * math.sqrt(252)), 2) if r.std() > 0 else 0.0,
        "calmar": round(float(eq.iloc[-1] ** (1 / yrs) - 1) / abs(mdd), 3) if mdd < 0 else 0.0,
        "total_pct": round(float(eq.iloc[-1] - 1) * 100, 1),
    }


def portfolio(q: pd.Series, c: pd.Series, w_c: float) -> pd.Series:
    """固定权重 + 季度再平衡（权重漂移用日度近似处理：直接按 w 混合日收益）。"""
    return (1 - w_c) * q + w_c * c


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    navs = {"BTC 收盘": None}
    for cap in [1.25, 1.5]:
        q = qqq_leg(cap)
        qidx = q.loc[START:END].dropna().index
        c, btc_px = btc_leg(qidx)
        d = pd.DataFrame({"Q": q.reindex(qidx), "C": c}).dropna()
        navs[f"BTC 收盘"] = btc_px.reindex(d.index)
        for w in [0.0, 0.1, 0.2, 0.3, 0.4]:
            s = stats(portfolio(d["Q"], d["C"], w))
            rows.append({"qqq_cap": cap, "w_btc": w, **s})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "portfolio_long_sample.csv", index=False)

    # 分年（推荐配置：cap 1.25 / w 0.3）
    q = qqq_leg(1.25)
    qidx = q.loc[START:END].dropna().index
    c, _ = btc_leg(qidx)
    d = pd.DataFrame({"Q": q.reindex(qidx), "C": c}).dropna()
    mix = portfolio(d["Q"], d["C"], 0.3)
    yearly = {}
    for name, r in [("Q 单腿", d["Q"]), ("Q70+C30", mix), ("C 单腿", d["C"])]:
        eq = (1 + r).cumprod()
        yearly[name] = {str(y): round((g.iloc[-1] / g.iloc[0] - 1) * 100, 1) for y, g in eq.groupby(eq.index.year)}

    # 压力段
    stress = {}
    for label, (a, b) in {"2018 加密寒冬": ("2018-01-01", "2018-12-31"),
                          "2020 新冠": ("2020-02-19", "2020-03-23"),
                          "2022 股债双杀": ("2022-01-01", "2022-12-31"),
                          "2025 关税": ("2025-03-25", "2025-04-08")}.items():
        seg = d.loc[a:b]
        if len(seg) < 5:
            continue
        stress[label] = {
            "Q 单腿": stats(seg["Q"])["total_pct"],
            "C 单腿": stats(seg["C"])["total_pct"],
            "Q70+C30": stats(0.7 * seg["Q"] + 0.3 * seg["C"])["total_pct"],
        }

    payload = {"window": [str(d.index[0].date()), str(d.index[-1].date())], "n_days": int(len(d)),
               "grid": json.loads(df.to_json(orient="records")), "yearly": yearly, "stress": stress,
               "note": "QQQ 腿用 feather 价格序列（未含股息），绝对收益略低于总收益口径；关注相对与风险指标。"}
    (OUT / "portfolio_long_sample.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float))

    pd.set_option("display.width", 240)
    print(f"组合长样本 {payload['window'][0]} ~ {payload['window'][1]}（{len(d)} 天）")
    print()
    print("=== 权重网格（按 Sharpe 排序前 12）===")
    print(df.sort_values("sharpe", ascending=False).head(12).to_string(index=False))
    print()
    print("=== 推荐配置分年（cap 1.25 / w 0.3）===")
    yr = pd.DataFrame(yearly)
    print(yr.to_string())
    print()
    print("=== 压力段（区间总收益 %）===")
    print(pd.DataFrame(stress).T.to_string())
    print(f"\n输出: {OUT}/portfolio_long_sample.json, portfolio_long_sample.csv")


if __name__ == "__main__":
    main()
