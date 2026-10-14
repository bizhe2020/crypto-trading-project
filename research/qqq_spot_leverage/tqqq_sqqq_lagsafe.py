#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TQQQ/SQQQ 策略的"口径安全"复刻 + A/B 对照（round 16）。

背景（round 15 审计）
--------------------
项目里 `scripts/plot_strategy_comparison.py::tqqq_sqqq_curve()` 产出的净值曲线
（CAGR 201%、MDD −16%）在用"当日 VIX/IXIC 标签"做当日持仓决策；把闸门滞后一日后
CAGR 掉到 2.7%、MDD 扩大到 −56%。本脚本按 `docs/backtest_lookahead_policy.md` 的标准
复刻同一策略，并同时给出 A（当日闸门）/ B（全部滞后）两版，量化口径差。

复刻的规则（来自 scripts/scan_tqqq_context_bucket_overlays.py 与 plot_strategy_comparison.py）
---------------------------------------------------------------------------------------------
* 方向：QQQ fast_ma(25) vs slow_ma(200)
* 多头闸门：vix_label ∈ {vix_low, vix_normal} 且 ixic_trend_label == ixic_up
* 空头闸门：short_context_score ≥ 5 且 vix_label ∈ {vix_high, vix_extreme}
    score = (ixic_down) + (vix_high/extreme) + (qqq_weak) + (qqq_mom_20 < −5%) + (qqq_mom_60 < −10%)
* 多头持仓管理：max_hold 90 天、trailing 30 天 / 12%
* 空头持仓管理：max_hold 20 天、trailing 5 天 / 6%
* 换手成本 10bp/次；VIX 分档 30/22/16，相对强弱 ±3%

用法: python3 research/qqq_spot_leverage/tqqq_sqqq_lagsafe.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
REPO = Path(__file__).resolve().parents[2]
ETF = REPO / "data" / "public" / "etf"
ETF_LONG = REPO / "data" / "public" / "etf_long"
COST_BPS = 10.0


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_feather(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def label_vix(v: float) -> str:
    if v >= 30:
        return "vix_extreme"
    if v >= 22:
        return "vix_high"
    if v >= 16:
        return "vix_normal"
    return "vix_low"


def label_rel(v: float) -> str:
    if v >= 0.03:
        return "qqq_strong"
    if v <= -0.03:
        return "qqq_weak"
    return "qqq_neutral"


def build_frame(fast: int = 25, slow: int = 200) -> pd.DataFrame:
    qqq = load_df(ETF / "QQQ-1d.feather")
    tqqq = load_df(ETF_LONG / "TQQQ-1d.feather")
    sqqq = load_df(ETF / "SQQQ-1d.feather")
    spy = load_df(ETF / "SPY-1d.feather")
    ixic = load_df(ETF / "^IXIC-1d.feather")
    vix = load_df(ETF / "^VIX-1d.feather")

    def sess(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
        out = df[["date", col]].copy()
        out["day"] = out["date"].dt.tz_convert("America/New_York").dt.normalize()
        out = out.sort_values("day").drop_duplicates("day", keep="last")
        return out[["day", col]].rename(columns={col: name})

    f = qqq[["date", "close"]].rename(columns={"close": "qqq_close"})
    f["day"] = f["date"].dt.tz_convert("America/New_York").dt.normalize()
    f = f.merge(sess(tqqq, "close", "tqqq_close"), on="day", how="inner")
    f = f.merge(sess(sqqq, "close", "sqqq_close"), on="day", how="inner")
    f = f.merge(sess(spy, "close", "spy_close"), on="day", how="left")
    f = f.merge(sess(ixic, "close", "ixic_close"), on="day", how="left")
    f = f.merge(sess(vix, "close", "vix_close"), on="day", how="left")
    f = f.dropna(subset=["qqq_close", "tqqq_close", "sqqq_close"]).sort_values("day").reset_index(drop=True)

    f["fast_ma"] = f["qqq_close"].rolling(fast).mean()
    f["slow_ma"] = f["qqq_close"].rolling(slow).mean()
    f["trend"] = 0
    f.loc[f["fast_ma"] > f["slow_ma"], "trend"] = 1
    f.loc[f["fast_ma"] < f["slow_ma"], "trend"] = -1
    f["planned_trend"] = f["trend"].shift(1).fillna(0).astype(int)

    f["qqq_mom_20"] = f["qqq_close"].pct_change(20)
    f["qqq_mom_60"] = f["qqq_close"].pct_change(60)
    f["rel_20"] = f["qqq_close"].pct_change(20) - f["spy_close"].pct_change(20)
    f["ixic_ma_50"] = f["ixic_close"].rolling(50).mean()
    f["ixic_up"] = (f["ixic_close"] > f["ixic_ma_50"]).fillna(False)
    f["vix_label"] = f["vix_close"].map(lambda v: label_vix(float(v)) if pd.notna(v) else "unknown")
    f["rel_label"] = f["rel_20"].map(lambda v: label_rel(float(v)) if pd.notna(v) else "unknown")
    f["long_gate"] = f["vix_label"].isin(["vix_low", "vix_normal"]) & f["ixic_up"]
    f["short_score"] = (
        f["ixic_up"].eq(False).astype(int)
        + f["vix_label"].isin(["vix_high", "vix_extreme"]).astype(int)
        + f["rel_label"].eq("qqq_weak").astype(int)
        + (f["qqq_mom_20"] < -0.05).astype(int)
        + (f["qqq_mom_60"] < -0.10).astype(int)
    )
    f["short_gate"] = (f["short_score"] >= 5) & f["vix_label"].isin(["vix_high", "vix_extreme"])
    return f


def run(frame: pd.DataFrame, lag_all: bool, initial: float = 1000.0) -> pd.DataFrame:
    """lag_all=False -> A 版（当日闸门，复刻原口径）；True -> B 版（全部滞后一日，口径安全）。"""
    f = frame.copy()
    if lag_all:
        for col in ["long_gate", "short_gate", "planned_trend"]:
            f[col] = f[col].shift(1)
        f["planned_trend"] = f["planned_trend"].fillna(0).astype(int)
        f["long_gate"] = f["long_gate"].fillna(False).astype(bool)
        f["short_gate"] = f["short_gate"].fillna(False).astype(bool)

    capital = initial
    asset = "CASH"
    weight = 0.0
    hold_days = 0
    peak = 0.0
    max_hold = 0
    trail_lb = 0
    trail_dd = 0.0
    prev_asset = "CASH"
    rows = []
    for idx, row in f.iterrows():
        # 目标
        want = "CASH"
        if int(row["planned_trend"]) > 0 and bool(row["long_gate"]):
            want = "TQQQ"
        elif int(row["planned_trend"]) < 0 and bool(row["short_gate"]):
            want = "SQQQ"

        if asset != "CASH" and want != asset:
            asset, weight, hold_days, peak = "CASH", 0.0, 0, 0.0
        if asset == "CASH" and want != "CASH":
            asset, weight, hold_days = want, 1.0, 0
            px_col = "tqqq_close" if asset == "TQQQ" else "sqqq_close"
            peak = float(row[px_col])
            if asset == "TQQQ":
                max_hold, trail_lb, trail_dd = 90, 30, 12.0
            else:
                max_hold, trail_lb, trail_dd = 20, 5, 6.0

        position = asset
        ret = 0.0
        if idx > 0 and position != "CASH":
            px_col = "tqqq_close" if position == "TQQQ" else "sqqq_close"
            prev = float(f.iloc[idx - 1][px_col])
            cur = float(row[px_col])
            ret = (cur / prev - 1.0) * weight if prev > 0 else 0.0
            hold_days += 1
            peak = max(peak, cur)
            trail_hit = (trail_lb > 0 and trail_dd > 0 and hold_days >= trail_lb and peak > 0
                         and (peak - cur) / peak * 100.0 >= trail_dd)
            time_hit = max_hold > 0 and hold_days >= max_hold
            if trail_hit or time_hit:
                asset, weight, hold_days, peak = "CASH", 0.0, 0, 0.0
        if idx > 0 and position != prev_asset:
            ret -= COST_BPS / 1e4
        prev_asset = position
        capital *= 1 + ret
        rows.append({"date": row["date"], "daily_return": ret, "equity": capital, "position": position})
    return pd.DataFrame(rows)


def stats(curve: pd.DataFrame) -> dict:
    eq = curve["equity"]
    dates = pd.to_datetime(curve["date"])
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    r = curve["daily_return"]
    return {
        "final_equity": round(float(eq.iloc[-1]), 1),
        "total_pct": round(float(eq.iloc[-1] / eq.iloc[0] - 1) * 100, 1),
        "cagr_pct": round(float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100, 2),
        "mdd_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 2),
        "vol_pct": round(float(r.std() * math.sqrt(252)) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * math.sqrt(252)), 2) if r.std() > 0 else 0.0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = build_frame()
    a = run(frame, lag_all=False)
    b = run(frame, lag_all=True)

    rows = [{"variant": "A 当日闸门（复刻原口径）", **stats(a)},
            {"variant": "B 全部滞后（口径安全）", **stats(b)}]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tqqq_sqqq_lagsafe_summary.csv", index=False)
    a.to_csv(OUT / "tqqq_sqqq_lagsafe_A.csv", index=False)
    b.to_csv(OUT / "tqqq_sqqq_lagsafe_B.csv", index=False)

    # 分年（B 版）
    b["year"] = pd.to_datetime(b["date"]).dt.year
    yr = b.groupby("year").apply(lambda g: round((g["equity"].iloc[-1] / g["equity"].iloc[0] - 1) * 100, 1)).to_dict()
    pos_mix = b["position"].value_counts(normalize=True).round(3).to_dict()

    payload = {"range": [str(pd.to_datetime(a['date'].iloc[0]).date()), str(pd.to_datetime(a['date'].iloc[-1]).date())],
               "A_same_day_gate": stats(a), "B_lagged": stats(b),
               "B_yearly_pct": {str(k): v for k, v in yr.items()}, "B_position_mix": pos_mix,
               "note": "A 用于对齐项目原口径；B 按 docs/backtest_lookahead_policy.md 全部滞后，是可用于实盘的版本。"}
    (OUT / "tqqq_sqqq_lagsafe.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float))

    pd.set_option("display.width", 200)
    print("=== TQQQ/SQQQ 口径安全复刻（同一规则，只差闸门滞后）===")
    print(df.to_string(index=False))
    print()
    print("B 版分年收益:", json.dumps({str(k): v for k, v in yr.items()}, ensure_ascii=False))
    print("B 版持仓占比:", json.dumps(pos_mix, ensure_ascii=False))
    print(f"\n输出: {OUT}/tqqq_sqqq_lagsafe.json, _summary.csv, _A.csv, _B.csv")


if __name__ == "__main__":
    main()
