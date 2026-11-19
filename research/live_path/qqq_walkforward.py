"""QQQ SMA200 buffer + 黄金投票门槛的走步验证（walk-forward）。

规则选择不使用未来数据：
  * 测试年 y：只用 y-5 ~ y-1 的 Calmar 选 (QQQ buffer, gold min_vote)；
  * 选中的参数在 y 年逐日模拟（信号仍然 t 收盘 → t+1 开盘）；
  * 把 2012-2026 每年的样本外收益串起来，得到 OOS 曲线。

对照：固定 0%/0.0（现状）、1%/0.4、5%/0.4，以及 QQQ 买入持有。
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.time_in_market.explore as E  # noqa: E402
import research.time_in_market.sim_basket_liquidation as S  # noqa: E402
from strategy.trend_vote import vote_target  # noqa: E402
from research.live_path.qqq_buffer_study import (  # noqa: E402
    gold_threshold,
    period_stats,
    run_basket,
    sma200_buffer_target,
)

BUFFERS = (0.0, 0.01, 0.02, 0.03, 0.05)
GOLD_THRESHOLDS = (0.0, 0.4, 0.6)
TEST_YEARS = range(2012, 2027)
TRAIN_YEARS = 5


def slice_window(eq: pd.Series, start: str, end: str | None) -> pd.Series:
    s = eq[eq.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        s = s[s.index < pd.Timestamp(end, tz="UTC")]
    return s


def curve_stats(eq: pd.Series) -> dict:
    s = eq.dropna()
    if len(s) < 30:
        return {"cagr": None, "mdd": None, "calmar": None, "ret": None}
    s = s / s.iloc[0]
    yrs = max((s.index[-1] - s.index[0]).days / 365.0, 1e-9)
    cagr = float(s.iloc[-1] ** (1 / yrs) - 1) if s.iloc[-1] > 0 else -1.0
    mdd = float((1 - s / s.cummax()).max())
    return {
        "cagr": round(cagr * 100, 1),
        "mdd": round(mdd * 100, 1),
        "calmar": round(cagr / mdd, 2) if mdd > 0 else None,
        "ret": round(float(s.iloc[-1] - 1) * 100, 1),
    }


def chain_years(eq: pd.Series, years: range) -> pd.Series:
    pieces = []
    level = 1.0
    for y in years:
        seg = slice_window(eq, f"{y}-01-01", f"{y + 1}-01-01")
        if len(seg) < 30:
            continue
        seg = seg / seg.iloc[0] * level
        pieces.append(seg)
        level = float(seg.iloc[-1])
    return pd.concat(pieces) if pieces else pd.Series(dtype=float)


def main() -> None:
    E.LONG_FUNDING_PER_DAY = 0.06 / 365.0
    legs = {
        "GLD": E.load_daily(ROOT / "var/tmp/time_in_market/etf_long/GLD-1d.feather", False),
        "QQQ": E.load_daily(ROOT / "var/tmp/time_in_market/etf_long/QQQ-1d.feather", False),
    }
    end = min(d.index[-1] for d in legs.values())
    cal = pd.date_range(pd.Timestamp("2005-01-01", tz="UTC"), end, freq="1D")
    qqq_close = legs["QQQ"]["close"]
    gld_close = legs["GLD"]["close"]
    gold_raw = vote_target(gld_close).shift(1).fillna(0.0)

    variants: dict[tuple[float, float], pd.Series] = {}
    for buf in BUFFERS:
        q = sma200_buffer_target(qqq_close, buffer=buf).shift(1).fillna(0.0)
        for gthr in GOLD_THRESHOLDS:
            gv = gold_threshold(gold_raw, gthr) if gthr else gold_raw
            res = run_basket(legs, {"GLD": gv, "QQQ": q}, cal, k=2.5, lev=5.0, weight=0.5)
            variants[(buf, gthr)] = res["equity"]

    print("=== walk-forward: train 5y, select by Calmar, test next year ===", flush=True)
    oos_parts = []
    level = 1.0
    picks = []
    for y in TEST_YEARS:
        train_start = f"{y - TRAIN_YEARS}-01-01"
        train_end = f"{y}-01-01"
        scored = []
        for key, eq in variants.items():
            st = period_stats(eq, train_start, train_end)
            if st and st.get("calmar") is not None:
                scored.append((st["calmar"], st["cagr"], key))
        if not scored:
            continue
        scored.sort(reverse=True)
        best = scored[0][2]
        seg = slice_window(variants[best], f"{y}-01-01", f"{y + 1}-01-01")
        if len(seg) < 30:
            continue
        seg = seg / seg.iloc[0] * level
        oos_parts.append(seg)
        level = float(seg.iloc[-1])
        picks.append((y, best[0], best[1], round(seg.iloc[-1] / seg.iloc[0] - 1, 4)))
        print(f"{y}: buffer {best[0]*100:.0f}% / gold>={best[1]:.1f} | train calmar {scored[0][0]} | year ret {picks[-1][3]*100:+.1f}%", flush=True)

    oos = pd.concat(oos_parts)
    print("\n=== OOS comparison (2012-2026, chained test years) ===", flush=True)
    rows = {
        "walk-forward": oos,
        "fixed 0% / gold 0.0": chain_years(variants[(0.0, 0.0)], TEST_YEARS),
        "fixed 1% / gold 0.0": chain_years(variants[(0.01, 0.0)], TEST_YEARS),
        "fixed 1% / gold 0.4": chain_years(variants[(0.01, 0.4)], TEST_YEARS),
        "fixed 0% / gold 0.4": chain_years(variants[(0.0, 0.4)], TEST_YEARS),
        "fixed 5% / gold 0.4": chain_years(variants[(0.05, 0.4)], TEST_YEARS),
    }
    qqq_curve = chain_years(qqq_close, TEST_YEARS)
    rows["QQQ buy&hold"] = qqq_curve
    out = {}
    for name, eq in rows.items():
        st = curve_stats(eq)
        out[name] = st
        print(f"{name:22s} CAGR {st['cagr']:6.1f}% | MDD {st['mdd']:5.1f}% | Calmar {st['calmar']} | total {st['ret']:6.1f}%", flush=True)

    print("\nselection counts:", flush=True)
    print("buffers", Counter(p[1] for p in picks), flush=True)
    print("gold", Counter(p[2] for p in picks), flush=True)
    print("picks", picks, flush=True)


if __name__ == "__main__":
    main()
