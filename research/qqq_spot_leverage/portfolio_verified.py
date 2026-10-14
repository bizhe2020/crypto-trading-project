#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组合层（round 14）：用**可独立验证**的两条腿做风险预算。

为什么不用 var/reports/strategy_equity_return_comparison.csv 里的 tqqq_sqqq 曲线？
  该曲线在 2025-04（QQQ −9.6%）期间显示 +44%、2024-08（QQQ −1.6%）显示 +13.9%、
  三年 CAGR 194%/MDD −16%/Sharpe 3.0，且与真实 TQQQ 日收益相关仅 0.44——
  在没有其信号与执行逻辑的情况下无法复核，**因此不作为组合输入**。

替代方案：用真实 TQQQ/SQQQ/QQQ/BTC 价格自建两条可复现的腿：
  N 腿（纳指杠杆腿）：TQQQ 二元趋势开关（TQQQ 收盘 > TQQQ MA100 -> 持有 TQQQ，否则空仓/持 SQQQ）
  Q 腿（QQQ 现货腿）：本项目已验证的双均线阶梯（1.25x 上限）
  C 腿（crypto 腿）：BTC + MA200 过滤（对齐到 QQQ 交易日）
组合：按权重混合，季度再平衡；给出相关性、权重扫描与训练/测试双窗口。

用法: python3 research/qqq_spot_leverage/portfolio_verified.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_qqq_spot_leverage import (  # noqa: E402
    Params,
    build_target_series,
    load_market_frame,
    prepare_indicators,
    run_engine,
)

OUT = Path(__file__).resolve().parent / "out"
REPO = Path(__file__).resolve().parents[2]
START = "2022-10-18"
END = "2026-05-29"


def _feather_close(rel: str) -> pd.Series:
    df = pd.read_feather(REPO / rel)
    df["d"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    df = df.sort_values("d").drop_duplicates("d", keep="last").set_index("d")
    return df["close"]


def nasdaq_leg(mode: str = "tqqq") -> pd.Series:
    """纳指杠杆腿：真实 TQQQ（可选 SQQQ 反向）二元趋势开关。"""
    tqqq = _feather_close("data/public/etf_long/TQQQ-1d.feather")
    sqqq = _feather_close("data/public/etf/SQQQ-1d.feather")
    idx = tqqq.index.intersection(sqqq.index)
    tqqq, sqqq = tqqq.loc[idx], sqqq.loc[idx]
    rt, rs = tqqq.pct_change(), sqqq.pct_change()
    ma = tqqq.rolling(100).mean()
    on = (tqqq > ma).shift(1).fillna(False)
    if mode == "tqqq":
        return rt * on.astype(float)
    if mode == "tqqq_sqqq":
        return rt * on.astype(float) + rs * (~on).astype(float) * 0.5   # 空仓期用一半 SQQQ 对冲
    raise ValueError(mode)


def qqq_leg(cap: float = 1.25) -> pd.Series:
    frame = load_market_frame("2001-01-02", "2026-08-14")
    p = Params(rs_mode="", use_vix_gate=False, use_recovery=False,
               lev_full=cap, max_leverage=cap, lev_half=min(1.0, cap), recovery_lev=min(0.75, cap))
    ind = prepare_indicators(frame, p)
    return run_engine(ind, build_target_series(ind, p), p, "q")["path"]["net_ret"]


def btc_leg(qqq_index) -> pd.Series:
    btc = _feather_close("data/public/etf/BTC-USD-1d.feather")
    idx = btc.index.union(qqq_index)
    btc = btc.reindex(idx).ffill().reindex(qqq_index)
    r = btc.pct_change()
    ma = btc.rolling(200).mean()
    return (btc > ma).shift(1).fillna(False).astype(float) * r


def stats(r: pd.Series) -> dict:
    eq = (1 + r.fillna(0)).cumprod()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    mdd = float((eq / eq.cummax() - 1).min())
    return {
        "cagr_pct": round(float(eq.iloc[-1] ** (1 / years) - 1) * 100, 2),
        "mdd_pct": round(mdd * 100, 2),
        "vol_pct": round(float(r.std() * math.sqrt(252)) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * math.sqrt(252)), 2) if r.std() > 0 else 0.0,
        "total_pct": round(float(eq.iloc[-1] - 1) * 100, 1),
        "calmar": round(float(eq.iloc[-1] ** (1 / years) - 1) / abs(mdd), 3) if mdd < 0 else 0.0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    q = qqq_leg(1.25)
    q_idx = q.loc[START:END].dropna().index
    legs = {
        "N TQQQ趋势开关": nasdaq_leg("tqqq").reindex(q_idx),
        "N2 TQQQ+SQQQ开关": nasdaq_leg("tqqq_sqqq").reindex(q_idx),
        "Q QQQ阶梯1.25x": q.reindex(q_idx),
        "C BTC+MA200": btc_leg(q_idx),
    }
    df = pd.DataFrame(legs).dropna()
    window = [str(df.index[0].date()), str(df.index[-1].date())]

    single = pd.DataFrame({k: stats(df[k]) for k in df.columns}).T.sort_values("sharpe", ascending=False)
    single.to_csv(OUT / "portfolio_verified_legs.csv")
    corr = df.corr().round(3)
    corr.to_csv(OUT / "portfolio_verified_corr.csv")

    # 三条腿等权/风险预算组合
    combos = {
        "Q 单腿": {"Q QQQ阶梯1.25x": 1.0},
        "N 单腿": {"N TQQQ趋势开关": 1.0},
        "Q60 + N20 + C20": {"Q QQQ阶梯1.25x": 0.6, "N TQQQ趋势开关": 0.2, "C BTC+MA200": 0.2},
        "Q50 + N30 + C20": {"Q QQQ阶梯1.25x": 0.5, "N TQQQ趋势开关": 0.3, "C BTC+MA200": 0.2},
        "Q40 + N30 + C30": {"Q QQQ阶梯1.25x": 0.4, "N TQQQ趋势开关": 0.3, "C BTC+MA200": 0.3},
        # 注：只给 NASDAQ 腿留 10% 给 BTC（crypto 腿在此窗口自身回撤更大）
        "Q60 + N30 + C10": {"Q QQQ阶梯1.25x": 0.6, "N TQQQ趋势开关": 0.3, "C BTC+MA200": 0.1},
        "Q50 + N2_20 + C30": {"Q QQQ阶梯1.25x": 0.5, "N2 TQQQ+SQQQ开关": 0.2, "C BTC+MA200": 0.3},
        "等风险(Q/N/C)": {"Q QQQ阶梯1.25x": 1 / 3, "N TQQQ趋势开关": 1 / 3, "C BTC+MA200": 1 / 3},
        # 推荐落地方案（paper_two_leg.py 的默认权重）
        "Q70 + C30": {"Q QQQ阶梯1.25x": 0.7, "C BTC+MA200": 0.3},
        "Q80 + C20": {"Q QQQ阶梯1.25x": 0.8, "C BTC+MA200": 0.2},
        "Q75 + C25": {"Q QQQ阶梯1.25x": 0.75, "C BTC+MA200": 0.25},
    }
    rows = []
    for name, w in combos.items():
        r = sum(wi * df[k] for k, wi in w.items())
        rows.append({"combo": name, **stats(r)})
    res = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    res.to_csv(OUT / "portfolio_verified_combos.csv", index=False)

    # 训练/测试
    splits = {}
    for label, (a, b) in {"train 22-10~24-12": ("2022-10-18", "2024-12-31"), "test 25-01~26-05": ("2025-01-01", "2026-05-29")}.items():
        s = df.loc[a:b]
        splits[label] = {name: stats(sum(wi * s[k] for k, wi in w.items())) for name, w in combos.items()}

    payload = {"window": window, "n_days": int(len(df)), "single_leg": json.loads(single.to_json(orient="index")),
               "correlation": json.loads(corr.to_json()), "combos": json.loads(res.to_json(orient="records")),
               "splits": splits,
               "note": "所有腿均用真实 ETF/加密价格自建，可独立复现；未使用 var/reports 里无法复核的 tqqq_sqqq 曲线。"}
    (OUT / "portfolio_verified.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float))

    pd.set_option("display.width", 240)
    print(f"区间 {window[0]} ~ {window[1]}（{len(df)} 天）")
    print()
    print("=== 单腿（真实价格自建）===")
    print(single.to_string())
    print()
    print("=== 相关性 ===")
    print(corr.to_string())
    print()
    print("=== 组合 ===")
    print(res.to_string(index=False))
    print()
    for label, d in splits.items():
        print(f"--- {label} ---")
        print(pd.DataFrame(d).T[["cagr_pct", "mdd_pct", "vol_pct", "sharpe", "calmar"]].to_string())
        print()
    print(f"输出: {OUT}/portfolio_verified.json, _legs.csv, _corr.csv, _combos.csv")


if __name__ == "__main__":
    main()
