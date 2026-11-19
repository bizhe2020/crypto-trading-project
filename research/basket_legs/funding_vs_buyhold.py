"""篮子（永续实现）的资金费敏感度，以及与不加杠杆的买入持有对照。

问题：qqq_walkforward 显示 2012-2026 现状篮子（k=2.5、5x、资金费 6%/年）CAGR 15.4% / MDD 36%，
而 QQQ 1 倍买入持有（只算价格）CAGR 18.9% / MDD 36%。篮子的择时 + 2.5 倍名义到底被什么吃掉了？

口径：
  * 篮子 = 实盘机制：GLD V5 + QQQ SMA200，t 收盘信号 → t+1 开盘，偏离 10% 才调仓，逐仓强平，单边 8bp
    （sim_basket_liquidation.simulate，与 qqq_walkforward 同一入口）。资金费年化 f ∈ {0, 3, 6, 9}%。
  * 对照（不加杠杆、无资金费、只算价格，不含分红 → 对买入持有略偏保守）：
      QQQ 1x 买入持有；GLD/QQQ 50/50 每月再平衡；同一择时信号但 1 倍名义（k=1、0 资金费，等价于现货/ETF 实现）。
  * 区间：2007+（含 2008）、2012+、2021+。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.time_in_market.explore as E  # noqa: E402
import research.time_in_market.sim_basket_liquidation as S  # noqa: E402
from strategy.trend_vote import signal_target, vote_target  # noqa: E402

PERIODS = (("2007+", "2007-01-01"), ("2012+", "2012-01-01"), ("2021+", "2021-01-01"))


def stats(eq: pd.Series, start: str) -> str:
    s = eq[eq.index >= pd.Timestamp(start, tz="UTC")].dropna()
    s = s / s.iloc[0]
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = s.iloc[-1] ** (1 / yrs) - 1
    mdd = float((1 - s / s.cummax()).max())
    return f"{cagr*100:5.1f}%/{mdd*100:4.1f}%/{cagr/mdd:4.2f}"


def main() -> None:
    legs = {
        "GLD": E.load_daily(ROOT / "var/tmp/time_in_market/etf_long/GLD-1d.feather", False),
        "QQQ": E.load_daily(ROOT / "var/tmp/time_in_market/etf_long/QQQ-1d.feather", False),
    }
    end = min(d.index[-1] for d in legs.values())
    cal = pd.date_range(pd.Timestamp("2005-01-01", tz="UTC"), end, freq="1D")
    q, g = legs["QQQ"]["close"], legs["GLD"]["close"]
    votes = {
        "GLD": vote_target(g).shift(1).fillna(0.0),
        "QQQ": signal_target(q, "sma200").shift(1).fillna(0.0),
    }
    rows: dict[str, pd.Series] = {}
    for f in (0.0, 0.03, 0.06, 0.09):
        E.LONG_FUNDING_PER_DAY = f / 365.0
        rows[f"basket k2.5 5x fund{int(f*100)}%"] = S.simulate(legs, votes, None, cal, k=2.5, lev=5.0, weight=0.5)["equity"]
    E.LONG_FUNDING_PER_DAY = 0.0
    rows["basket k1 (1x, no funding)"] = S.simulate(legs, votes, None, cal, k=1.0, lev=5.0, weight=0.5)["equity"]
    rows["basket k2 (1x/leg, no funding)"] = S.simulate(legs, votes, None, cal, k=2.0, lev=5.0, weight=0.5)["equity"]
    idx = cal
    qd = q.reindex(idx).ffill()
    gd = g.reindex(idx).ffill()
    rows["QQQ 1x buy&hold"] = qd / qd.dropna().iloc[0]
    rq, rg = qd.pct_change().fillna(0), gd.pct_change().fillna(0)
    # 50/50 每月再平衡
    eq, w, vals = 1.0, None, []
    month = None
    wq = wg = 0.5
    for d in idx:
        eq_q, eq_g = wq * (1 + rq[d]), wg * (1 + rg[d])
        eq = eq_q + eq_g
        wq, wg = eq_q, eq_g
        if month != d.month:
            wq, wg, month = eq / 2, eq / 2, d.month
        vals.append(eq)
    rows["QQQ/GLD 50/50 1x"] = pd.Series(vals, index=idx)

    print(f"{'variant':34s}" + "".join(f"{name:>22s}" for name, _ in PERIODS))
    print(f"{'':34s}" + "".join(f"{'CAGR/MDD/Calmar':>22s}" for _ in PERIODS))
    for name, eq in rows.items():
        eq = eq.dropna()
        print(f"{name:34s}" + "".join(f"{stats(eq, start):>22s}" for _, start in PERIODS))
    qi = votes["QQQ"].reindex(idx).fillna(0)
    gi = votes["GLD"].reindex(idx).fillna(0)
    for name, start in PERIODS:
        m = idx >= pd.Timestamp(start, tz="UTC")
        expo = 2.5 * 0.5 * (qi[m] + gi[m]).mean()
        print(f"{name}: 平均多头名义 {expo:.2f}x 权益 → 6% 资金费约 {expo*6:.1f}%/年权益")


if __name__ == "__main__":
    main()
