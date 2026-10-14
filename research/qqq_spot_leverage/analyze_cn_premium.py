#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A 股纳指 QDII ETF 的溢价/折价分解与跟踪误差分析（round 29）。

为什么需要这个
--------------
跨境 QDII ETF 的二级市场价格 = 基金净值(NAV) × (1 + 溢价)。
其中 NAV 跟随纳斯达克100（叠加汇率），而**溢价是纯粹的资金面噪音**：
它不提供任何长期收益，却会在你买入的那一刻直接变成你的成本。

因此真实收益可以精确分解为：
    价格收益  ≈  净值收益(真实跟踪)  +  溢价变动
我们在这里量化两项各自的量级，为"是否需要对溢价做交易纪律"提供依据。

净值口径与时点
-------------
A 股在 D 日 15:00 收盘，而此时美股 D 日尚未开盘，所以 D 日盘后公布的净值
对应的是**美东 D-1 日收盘**的持仓估值。因此：
    premium_D = close_D / nav_D - 1
其中包含了 A 股投资者对"美东 D 日隔夜涨跌"的预期。
为把纯溢价从"预期"里剥出来，同时给出：
    premium_excess_D = premium_D - (QQQ_D / QQQ_{D-1} - 1)
即扣掉隔夜预期后的剩余部分（更接近真实资金溢价）。

用法：
    python3 research/qqq_spot_leverage/analyze_cn_premium.py
    python3 research/qqq_spot_leverage/analyze_cn_premium.py --start 2020-01-01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ETF_DIR = REPO / "data" / "public" / "etf"
OUT = HERE / "out"
sys.path.insert(0, str(HERE))

NAMES = {
    "159501": "纳指ETF嘉实", "513100": "纳指ETF国泰", "159941": "纳指ETF广发",
    "513300": "纳斯达克ETF华夏", "159632": "纳斯达克ETF华安", "513390": "纳指100ETF博时",
    "513110": "纳指ETF招商", "159696": "纳指ETF易方达", "513870": "纳指ETF富国",
}

CODES = ["513100", "159941", "513300", "159632", "159501", "513110", "513390", "159696", "513870"]


def suffix(code: str) -> str:
    return f"{code}.{'SH' if code.startswith('5') else 'SZ'}"


def load_price(code: str) -> pd.DataFrame:
    p = ETF_DIR / f"{suffix(code)}-1d.feather"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_feather(p)
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").drop_duplicates("date").set_index("date")


def load_nav(code: str) -> pd.Series:
    p = ETF_DIR / f"{suffix(code)}-nav.feather"
    if not p.exists():
        return pd.Series(dtype=float)
    d = pd.read_feather(p)
    d["date"] = pd.to_datetime(d["date"])
    s = d.sort_values("date").drop_duplicates("date").set_index("date")["nav"]
    return s.astype(float)


def _naive_day(idx: pd.Index) -> pd.DatetimeIndex:
    """把（可能带时区/带美东盘中时刻的）日期索引统一成"美东当地自然日"。

    `etf/QQQ-1d.feather` 的 date 是 tz-aware UTC 且带 13:30 UTC 时刻，
    若不转换会与 A 股的自然日索引完全对不上（表现为 corr=NaN）。
    """
    s = pd.Series(1.0, index=pd.DatetimeIndex(idx))
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_convert("America/New_York")
    s.index = s.index.normalize().tz_localize(None)
    return pd.DatetimeIndex(s.index)


def load_qqq() -> pd.Series:
    """QQQ 收盘价：多来源取并集（长样本优先作基准，短样本补充最新段）。"""
    parts = []
    for cand in [REPO / "data" / "public" / "etf_full" / "QQQ-1d.feather", ETF_DIR / "QQQ-1d.feather"]:
        if not cand.exists():
            continue
        d = pd.read_feather(cand)
        s = pd.Series(d["close"].to_numpy(float), index=_naive_day(pd.to_datetime(d["date"])), name="qqq")
        parts.append(s[~s.index.duplicated(keep="last")])
    if not parts:
        raise SystemExit("找不到 QQQ 数据")
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def analyze(code: str, qqq: pd.Series, start: str, end: str) -> dict | None:
    px = load_price(code)
    nav = load_nav(code)
    if px.empty or nav.empty:
        return None
    j = pd.DataFrame({"close": px["close"], "open": px["open"], "volume": px.get("volume")}).join(nav, how="inner")
    j = j.loc[(j.index >= start) & (j.index <= end)].dropna()
    if len(j) < 60:
        return None

    # NAV 收益 vs QQQ 收益（同一"估值基准"，NAV 对应美东 T-1 收盘）
    j["nav_ret"] = j["nav"].pct_change()
    j["px_ret"] = j["close"].pct_change()
    # QQQ 按自然日重采样到 A 股日历：取"当日或之前最后一个"美股收盘
    q = qqq.sort_index()
    q = q.reindex(q.index.union(j.index)).ffill().reindex(j.index)
    j["qqq_ret"] = q.pct_change()
    # NAV_D 对应美东 D-1 收盘 -> 与 qqq 的 D-1 收益对齐
    j["qqq_ret_lag1"] = j["qqq_ret"].shift(1)

    j["premium"] = j["close"] / j["nav"] - 1.0
    j["premium_excess"] = j["premium"] - j["qqq_ret_lag1"]

    v = j.dropna(subset=["nav_ret", "px_ret", "premium"])
    if len(v) < 60:
        return None

    nav_tr = float((1 + v["nav_ret"]).prod() - 1)
    px_tr = float((1 + v["px_ret"]).prod() - 1)

    yrs = len(v) / 252.0
    nav_cagr = (1 + nav_tr) ** (1 / yrs) - 1
    px_cagr = (1 + px_tr) ** (1 / yrs) - 1

    qa = v["qqq_ret_lag1"].dropna()
    cc = v[["nav_ret", "qqq_ret_lag1", "px_ret", "premium"]].replace([np.inf, -np.inf], np.nan).dropna(
        subset=["nav_ret", "qqq_ret_lag1"])
    if len(cc) > 10 and cc["qqq_ret_lag1"].std() > 0 and cc["nav_ret"].std() > 0:
        corr = float(cc["nav_ret"].corr(cc["qqq_ret_lag1"]))
        beta = float(np.polyfit(cc["qqq_ret_lag1"].to_numpy(float),
                                cc["nav_ret"].to_numpy(float), 1)[0])
    else:
        corr = beta = float("nan")

    cp = cc.dropna(subset=["px_ret"])
    px_corr = float(cp["px_ret"].corr(cp["qqq_ret_lag1"])) if len(cp) > 10 and cp["px_ret"].std() > 0 else float("nan")

    return {
        "code": code, "name": NAMES.get(code, code),
        "start": str(v.index.min().date()), "end": str(v.index.max().date()), "days": int(len(v)),
        "nav_total": nav_tr, "nav_cagr": float(nav_cagr),
        "px_total": px_tr, "px_cagr": float(px_cagr),
        "premium_ann_drag": float(px_cagr - nav_cagr),
        "premium_mean": float(v["premium"].mean()), "premium_std": float(v["premium"].std()),
        "premium_min": float(v["premium"].min()), "premium_max": float(v["premium"].max()),
        "premium_p05": float(v["premium"].quantile(0.05)), "premium_p95": float(v["premium"].quantile(0.95)),
        "premium_last": float(v["premium"].iloc[-1]),
        "premium_ac1": float(v["premium"].autocorr(1)),
        "nav_vs_qqq_corr": corr, "nav_beta_vs_qqq": beta,
        "px_vs_qqq_corr": px_corr,
        "nav_vol_ann": float(v["nav_ret"].std() * np.sqrt(252)),
        "px_vol_ann": float(v["px_ret"].std() * np.sqrt(252)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--end", default="2026-12-31")
    a = ap.parse_args()

    qqq = load_qqq()
    rows, series = [], {}
    for c in CODES:
        r = analyze(c, qqq, a.start, a.end)
        if r:
            rows.append(r)
            px = load_price(c); nav = load_nav(c)
            j = pd.DataFrame({"close": px["close"]}).join(nav, how="inner").dropna()
            series[c] = (j["close"] / j["nav"] - 1.0).loc[a.start:a.end]
    if not rows:
        print("没有可用数据；请先运行 tools/fetch_cn_etf.py")
        return 1

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)

    show = df[["code", "name", "start", "end", "days", "nav_cagr", "px_cagr", "premium_ann_drag",
               "premium_mean", "premium_std", "premium_min", "premium_max", "premium_last",
               "nav_vs_qqq_corr", "px_vs_qqq_corr", "nav_beta_vs_qqq"]].copy()
    for c in ["nav_cagr", "px_cagr", "premium_ann_drag", "premium_mean", "premium_std",
              "premium_min", "premium_max", "premium_last"]:
        show[c] = (show[c] * 100).round(2)
    for c in ["nav_vs_qqq_corr", "px_vs_qqq_corr", "nav_beta_vs_qqq"]:
        show[c] = show[c].round(3)
    print("\n=== A 股纳指 QDII ETF：净值跟踪 vs 价格（含溢价） ===")
    print("单位：% (除 corr/beta)；premium 为 收盘价/净值-1")
    print(show.to_string(index=False))

    if series:
        pr = pd.DataFrame(series)
        pr.to_csv(OUT / "cn_premium_panel.csv", float_format="%.6f")
        print("\n=== 最新溢价（%） ===")
        last = pr.ffill().iloc[-1].dropna().sort_values(ascending=False) * 100
        for k, v in last.items():
            print(f"  {k} {NAMES.get(k,''):16s} {v:7.2f}%")
        print("\n=== 溢价分位（%） ===")
        q = pd.DataFrame({
            "p05": pr.quantile(0.05), "p50": pr.quantile(0.50), "p95": pr.quantile(0.95),
            "max": pr.max(), "min": pr.min(),
        }) * 100
        print(q.round(2).to_string())

    df.to_json(OUT / "cn_premium_stats.json", orient="records", indent=2, force_ascii=False)
    print(f"\n已写出 {OUT/'cn_premium_stats.json'} 与 cn_premium_panel.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
