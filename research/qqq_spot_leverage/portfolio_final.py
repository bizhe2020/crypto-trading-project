#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组合层最终研究（round 13 终版）：QQQ 杠杆腿 + crypto 腿的风险预算。

要点
----
* QQQ 腿：双均线阶梯（默认配置，1.5x 上限），来自已验证的回测引擎；
* crypto 腿：本脚本**不**伪造项目内 SOTA 合约策略的收益，而是用可独立复现的 BTC 现货
  三种实现（买入持有 / 波动率目标 / MA200 趋势过滤）作为"crypto beta"代理，
  并在报告里明确标注这一点；
* 独立校验：所有组合指标都用 **BTC-USD 与 QQQ 原始日收益**重新计算，
  不依赖任何中间回测状态；
* 关键统计：滚动 252 日相关性（检验"低相关"是否只是样本期巧合）。

用法: python3 research/qqq_spot_leverage/portfolio_final.py
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
START = "2020-01-02"
END = "2026-08-13"


def load_btc() -> pd.Series:
    b = pd.read_feather(REPO / "data" / "public" / "etf" / "BTC-USD-1d.feather")
    b["d"] = pd.to_datetime(b["date"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    b = b.sort_values("d").drop_duplicates("d", keep="last").set_index("d")
    return b["close"].loc[:END]


def btc_aligned(px: pd.Series, qqq_index) -> pd.Series:
    """把 BTC 对齐到 QQQ 交易日：两个 QQQ 交易日之间的 BTC 价格变化（把周末跳空并进一个区间）。"""
    idx = px.index.union(qqq_index)
    return px.reindex(idx).ffill().reindex(qqq_index)


def btc_variants(px: pd.Series) -> dict[str, pd.Series]:
    r = px.pct_change()
    rv = r.rolling(20).std() * math.sqrt(252)
    vol_target = (0.40 / rv).clip(upper=1.0).shift(1) * r
    ma = px.rolling(200).mean()
    trend = (px > ma).shift(1).fillna(False).astype(float) * r
    ma100 = px.rolling(100).mean()
    trend100 = (px > ma100).shift(1).fillna(False).astype(float) * r
    return {
        "BTC 买入持有": r,
        "BTC 波动率目标40%": vol_target,
        "BTC MA200 过滤": trend,
        "BTC MA100 过滤": trend100,
    }


def qqq_legs() -> dict[str, pd.Series]:
    frame = load_market_frame("2001-01-02", END)
    out = {}
    p = Params()
    ind = prepare_indicators(frame, p)
    out["QQQ 阶梯1.5x"] = run_engine(ind, build_target_series(ind, p), p, "q")["path"]["net_ret"]
    p1 = Params(rs_mode="", use_vix_gate=False, use_recovery=False)
    ind1 = prepare_indicators(frame, p1)
    out["QQQ 买入持有1x"] = run_engine(ind1, pd.Series(1.0, index=ind1.index), p1, "q")["path"]["net_ret"]
    p12 = Params(rs_mode="", use_vix_gate=False, use_recovery=False, lev_full=1.25, max_leverage=1.25, lev_half=1.0, recovery_lev=1.0)
    ind12 = prepare_indicators(frame, p12)
    out["QQQ 阶梯1.25x"] = run_engine(ind12, build_target_series(ind12, p12), p12, "q")["path"]["net_ret"]
    return out


def stats(r: pd.Series) -> dict:
    eq = (1 + r.fillna(0)).cumprod()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return {
        "cagr_pct": round(float(eq.iloc[-1] ** (1 / years) - 1) * 100, 2),
        "mdd_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 2),
        "vol_pct": round(float(r.std() * math.sqrt(252)) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * math.sqrt(252)), 2) if r.std() > 0 else 0.0,
        "total_pct": round(float(eq.iloc[-1] - 1) * 100, 1),
        "calmar": round(float(eq.iloc[-1] ** (1 / years) - 1) / abs(float((eq / eq.cummax() - 1).min())), 3)
        if (eq / eq.cummax() - 1).min() < 0 else 0.0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    btc = load_btc()
    qqq_v = qqq_legs()
    # 关键：BTC 7x24 交易、QQQ 5x5，必须把 BTC 对齐到 QQQ 交易日（周末跳空并入相邻区间）
    qqq_index = pd.DataFrame(qqq_v).loc[START:END].dropna().index
    btc_v = btc_variants(btc_aligned(btc, qqq_index))

    data = pd.DataFrame({**qqq_v, **btc_v})
    data = data.loc[(data.index >= pd.Timestamp(START)) & (data.index <= pd.Timestamp(END))]
    data = data.dropna(how="all")
    common = data.dropna()
    window = [str(common.index[0].date()), str(common.index[-1].date())]

    single = pd.DataFrame({k: stats(common[k]) for k in common.columns}).T.sort_values("sharpe", ascending=False)
    single.to_csv(OUT / "portfolio_leg_stats.csv")

    corr = common.corr().round(3)
    corr.to_csv(OUT / "portfolio_correlation.csv")

    # 滚动相关性（QQQ 阶梯 vs BTC 各实现）
    roll = {}
    for name in btc_v:
        roll[name] = common["QQQ 阶梯1.5x"].rolling(252).corr(common[name])
    roll_df = pd.DataFrame(roll).dropna()
    roll_summary = {k: {"mean": round(float(v.mean()), 3), "min": round(float(v.min()), 3),
                        "max": round(float(v.max()), 3), "pct_negative": round(float((v < 0).mean()) * 100, 1)}
                    for k, v in roll_df.items()}
    roll_df.to_csv(OUT / "portfolio_rolling_corr.csv")

    # 权重网格（QQQ 阶梯1.5x × 每条 crypto 腿）
    rows = []
    for name in btc_v:
        for w in np.arange(0, 0.55, 0.05):
            r = w * common[name].fillna(0) + (1 - w) * common["QQQ 阶梯1.5x"].fillna(0)
            s = stats(r)
            rows.append({"crypto_leg": name, "w_crypto": round(float(w), 2), "w_qqq": round(1 - float(w), 2), **s})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "portfolio_weight_sweep.csv", index=False)

    best_sharpe = sweep.loc[sweep["sharpe"].idxmax()].to_dict()
    best_calmar = sweep.loc[sweep["calmar"].idxmax()].to_dict()
    qqq_only = sweep[(sweep.w_crypto == 0.0)].iloc[0].to_dict()

    # 也评估 "1.25x QQQ + BTC MA200 20%"
    extra = {}
    r_mix = 0.20 * common["BTC MA200 过滤"].fillna(0) + 0.80 * common["QQQ 阶梯1.25x"].fillna(0)
    extra["QQQ阶梯1.25x + BTC(MA200) 20%"] = stats(r_mix)

    payload = {
        "window": window, "n_days": int(len(common)),
        "single_leg": json.loads(single.to_json(orient="index")),
        "correlation": json.loads(corr.to_json()),
        "rolling_corr_252d": roll_summary,
        "weight_sweep": json.loads(sweep.to_json(orient="records")),
        "grid_best_sharpe": best_sharpe, "grid_best_calmar": best_calmar, "qqq_only": qqq_only,
        "extra_mix": extra,
        "caveat": "crypto 腿为 BTC 现货代理（买入持有/波动率目标/MA 过滤），"
                  "不是项目内 SOTA 合约策略的实盘收益；接入真实策略收益需替换 btc_variants()。",
    }
    (OUT / "portfolio_final.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float))

    pd.set_option("display.width", 240)
    print(f"组合区间 {window[0]} ~ {window[1]}（{len(common)} 天）")
    print()
    print("=== 单腿指标（按 Sharpe 排序）===")
    print(single.to_string())
    print()
    print("=== 日收益相关性 ===")
    print(corr.to_string())
    print()
    print("=== 滚动 252 日相关性（QQQ阶梯 vs BTC 腿）===")
    for k, v in roll_summary.items():
        print(f"  {k:<20} 均值 {v['mean']:+.3f}  区间 [{v['min']:+.3f}, {v['max']:+.3f}]  负相关占比 {v['pct_negative']:.1f}%")
    print()
    print("=== 最优组合 ===")
    for label, d in [("最高 Sharpe", best_sharpe), ("最高 Calmar", best_calmar), ("QQQ 单腿(w=0)", qqq_only)]:
        print(f"  {label:<12} {d['crypto_leg']:<18} w_crypto={d['w_crypto']:.2f}  "
              f"CAGR {d['cagr_pct']:.2f}%  MDD {d['mdd_pct']:.2f}%  vol {d['vol_pct']:.2f}%  Sharpe {d['sharpe']:.2f}  Calmar {d['calmar']:.3f}")
    for k, v in extra.items():
        print(f"  {k:<34} CAGR {v['cagr_pct']:.2f}%  MDD {v['mdd_pct']:.2f}%  Sharpe {v['sharpe']:.2f}  Calmar {v['calmar']:.3f}")
    print(f"\n输出: {OUT}/portfolio_final.json, portfolio_weight_sweep.csv, portfolio_correlation.csv, portfolio_rolling_corr.csv")


if __name__ == "__main__":
    main()
