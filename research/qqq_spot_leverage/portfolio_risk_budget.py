#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组合风险预算（round 13）：QQQ 现货杠杆腿 × crypto 腿。

背景
----
前 12 轮把 QQQ 单腿做到了极限：修正口径后，"趋势阶梯（1.5x 上限）"样本内 27.3%/−33.6%，
但没有稳定 alpha。继续优化单腿的边际收益很低，**组合层**才可能有增量。

本脚本对比三条 crypto 腿的实现（数据都可本地复现）：
  C1 BTC 现货买入持有           （data/public/etf/BTC-USD-1d.feather，2014-09 起）
  C2 BTC 波动率目标 40%         （按 20 日已实现波动缩放到目标波动，上限 1.0x）
  C3 BTC + 200 日均线趋势过滤   （跌破 MA200 空仓，等价于"加密版趋势开关"）

QQQ 腿：
  Q1 双均线阶梯（当前默认，1.5x 上限）
  Q2 买入持有 1x
组合方式：固定权重 w（每季度再平衡），并给出等风险贡献（ERC）与最小回撤解。

输出：相关性矩阵、单腿指标、权重扫描（有效前沿）、推荐组合。
用法: python3 research/qqq_spot_leverage/portfolio_risk_budget.py
"""
from __future__ import annotations

import json
import math
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
    prepare_indicators,
    run_engine,
)

OUT = Path(__file__).resolve().parent / "out"
REPO = Path(__file__).resolve().parents[2]
START = "2020-01-02"
END = "2026-08-13"


def btc_leg(name: str) -> pd.Series:
    """返回 BTC 腿的日收益序列（按名称选择实现）。"""
    b = pd.read_feather(REPO / "data" / "public" / "etf" / "BTC-USD-1d.feather")
    b["d"] = pd.to_datetime(b["date"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    b = b.sort_values("d").drop_duplicates("d", keep="last").set_index("d")
    px = b["close"].loc[:END]
    r = px.pct_change()
    if name == "C1 BTC 买入持有":
        return r
    if name == "C2 BTC 波动率目标40%":
        rv = r.rolling(20).std() * math.sqrt(252)
        scale = (0.40 / rv).clip(upper=1.0).shift(1)
        return r * scale
    if name == "C3 BTC + MA200 过滤":
        ma = px.rolling(200).mean()
        on = (px > ma).shift(1).fillna(False)
        return r * on.astype(float)
    raise ValueError(name)


def qqq_leg(name: str) -> pd.Series:
    frame = load_market_frame("2001-01-02", END)
    if name == "Q1 双均线阶梯(1.5x)":
        p = Params()
        ind = prepare_indicators(frame, p)
        path = run_engine(ind, build_target_series(ind, p), p, "q")["path"]
        return path["net_ret"]
    if name == "Q2 QQQ 买入持有1x":
        p = Params(rs_mode="", use_vix_gate=False, use_recovery=False)
        ind = prepare_indicators(frame, p)
        path = run_engine(ind, pd.Series(1.0, index=ind.index), p, "q")["path"]
        return path["net_ret"]
    raise ValueError(name)


def stats(r: pd.Series) -> dict:
    eq = (1 + r.fillna(0)).cumprod()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return {
        "cagr_pct": round(float(eq.iloc[-1] ** (1 / years) - 1) * 100, 2),
        "mdd_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 2),
        "vol_pct": round(float(r.std() * math.sqrt(252)) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * math.sqrt(252)), 2) if r.std() > 0 else 0.0,
        "total_pct": round(float(eq.iloc[-1] - 1) * 100, 1),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    q_names = ["Q1 双均线阶梯(1.5x)", "Q2 QQQ 买入持有1x"]
    c_names = ["C1 BTC 买入持有", "C2 BTC 波动率目标40%", "C3 BTC + MA200 过滤"]

    legs = {}
    for n in q_names:
        legs[n] = qqq_leg(n)
    for n in c_names:
        legs[n] = btc_leg(n)
    df = pd.DataFrame(legs).loc[START:END].dropna(how="all")
    df = df.loc[df.index >= pd.Timestamp(START)]
    # 统一到两者都有数据的区间
    common = df.dropna()
    print(f"组合分析区间 {common.index[0].date()} ~ {common.index[-1].date()}（{len(common)} 天）")
    print()

    single = pd.DataFrame({k: stats(common[k]) for k in common.columns}).T
    single.to_csv(OUT / "portfolio_leg_stats.csv")
    print("=== 单腿指标 ===")
    print(single.to_string())

    corr = common.corr().round(3)
    corr.to_csv(OUT / "portfolio_correlation.csv")
    print()
    print("=== 日收益相关性 ===")
    print(corr.to_string())

    # 权重扫描（Q1 × 每条 C 腿）
    rows = []
    for c in c_names:
        for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            r = w * common[c].fillna(0) + (1 - w) * common["Q1 双均线阶梯(1.5x)"].fillna(0)
            s = stats(r)
            rows.append({"crypto_leg": c, "w_crypto": w, "w_qqq": round(1 - w, 2), **s})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "portfolio_weight_sweep.csv", index=False)

    # 等风险贡献（用各自波动率反比近似）
    vol_c = common[c_names].std() * math.sqrt(252)
    vol_q = common["Q1 双均线阶梯(1.5x)"].std() * math.sqrt(252)
    erc_rows = []
    for c in c_names:
        w_c = (1 / vol_c[c]) / (1 / vol_c[c] + 1 / vol_q)
        r = w_c * common[c].fillna(0) + (1 - w_c) * common["Q1 双均线阶梯(1.5x)"].fillna(0)
        erc_rows.append({"crypto_leg": c, "w_crypto": round(float(w_c), 3), "w_qqq": round(1 - float(w_c), 3), **stats(r)})
    erc = pd.DataFrame(erc_rows)
    erc.to_csv(OUT / "portfolio_erc.csv", index=False)

    payload = {
        "window": [str(common.index[0].date()), str(common.index[-1].date())],
        "n_days": int(len(common)),
        "single_leg": json.loads(single.to_json(orient="index")),
        "correlation": json.loads(corr.to_json()),
        "weight_sweep": json.loads(sweep.to_json(orient="records")),
        "erc": json.loads(erc.to_json(orient="records")),
        "caveat": "crypto 腿用 BTC 现货代理（未用项目内 SOTA 合约策略的实盘/回放路径）；"
                  "若接入真实 crypto 策略收益，需替换 btc_leg()。",
    }
    (OUT / "portfolio_risk_budget.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float))

    pd.set_option("display.width", 240)
    print()
    print("=== 权重扫描（crypto 权重 vs 组合表现，按 Calmar 排序前 12）===")
    sweep["calmar"] = (sweep["cagr_pct"] / sweep["mdd_pct"].abs()).round(3)
    print(sweep.sort_values("calmar", ascending=False).head(12).to_string(index=False))
    print()
    print("=== 等风险贡献组合 ===")
    erc["calmar"] = (erc["cagr_pct"] / erc["mdd_pct"].abs()).round(3)
    print(erc.to_string(index=False))
    print(f"\n输出: {OUT}/portfolio_risk_budget.json, portfolio_weight_sweep.csv, portfolio_erc.csv, portfolio_correlation.csv")


if __name__ == "__main__":
    main()
