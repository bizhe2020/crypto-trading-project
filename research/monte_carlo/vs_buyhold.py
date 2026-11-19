"""现有策略 vs 纳指 / 标普买入持有（及常见增强）的对比。

口径：
  * ETF 买入持有用日线收盘价 + 近似分红（QQQ 0.7%/年、SPY 1.6%/年、GLD 0，按日均摊），无资金费、无杠杆。
  * 2x 增强：每日再平衡 2 倍（类似 QLD/SSO），借款成本按年化 --borrow（默认 6%，与永续资金费同口径）。
  * QQQ SMA200 1x：收盘在 200 日均线上方持有 QQQ，否则空仓（空仓不计利息，偏保守）。
  * 篮子：实盘机制（GLD V5 + QQQ SMA200，k=2，5x，资金费 6%），sim_basket_liquidation.simulate。
  * 策略（篮子 + scalp 0.1 倍）只有 2022 起的数据。
  * 蒙特卡洛：日收益块自助（30 天），3 年期，跑赢 3% 存款的概率；买入持有同样给 h=0 / h=0.5 两档折扣。
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
from research.live_path.split_haircut_mc import scalp_daily  # noqa: E402
from research.monte_carlo.own_capital_mc import block_bootstrap  # noqa: E402
from strategy.trend_vote import sma200_target, vote_target  # noqa: E402

DIV = {"QQQ": 0.007, "SPY": 0.016, "GLD": 0.0}
BORROW = 0.06


def daily_close(path: Path) -> pd.Series:
    d = E.load_daily(path, False)["close"]
    d.index = d.index.normalize()
    return d


def bh_returns(close: pd.Series, div: float) -> pd.Series:
    r = close.pct_change().dropna()
    gap = r.index.to_series().diff().dt.days.fillna(1).to_numpy()
    return r + div / 365.0 * gap


def to_calendar(r: pd.Series, cal: pd.DatetimeIndex) -> pd.Series:
    return r.reindex(cal).fillna(0.0)


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / 365.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    mdd = float((1 - eq / eq.cummax()).max())
    yr = eq.resample("YE").last().pct_change()
    yr.iloc[0] = eq.resample("YE").last().iloc[0] - 1
    return {"cagr": round(cagr * 100, 1), "mdd": round(mdd * 100, 1), "calmar": round(cagr / mdd, 2),
            "worst_year": round(float(yr.min()) * 100, 1), "neg_years": f"{int((yr < 0).sum())}/{len(yr)}"}


def mc(r: np.ndarray, h: float, w: float, rng, years=3, rate=0.03, paths=6000) -> dict:
    b = block_bootstrap(r - h * r.mean(), paths, 365 * years, 30.0, rng)
    eq = np.cumprod(np.clip(1 + b, 0, None), axis=1)[:, -1]
    tot = w * eq + (1 - w) * (1 + rate) ** years
    c = tot ** (1 / years) - 1
    return {"p_beat": round(float((tot > (1 + rate) ** years).mean()), 2), "p_loss": round(float((tot < 1).mean()), 2),
            "cagr_med": round(float(np.median(c)) * 100, 1), "cagr_p5": round(float(np.percentile(c, 5)) * 100, 1)}


def main() -> None:
    E.LONG_FUNDING_PER_DAY = BORROW / 365.0
    gld_p, qqq_p = Path("var/tmp/time_in_market/etf_long/GLD-1d.feather"), Path("var/tmp/time_in_market/etf_long/QQQ-1d.feather")
    spy_p = Path("var/tmp/basket_legs/etf/SPY-1d.feather")
    q, s, g = daily_close(qqq_p), daily_close(spy_p), daily_close(gld_p)
    end = min(q.index[-1], g.index[-1])
    cal06 = pd.date_range(pd.Timestamp("2006-01-01", tz="UTC"), end, freq="1D")
    cal22 = pd.date_range(pd.Timestamp("2022-01-01", tz="UTC"), end, freq="1D")

    rq, rs_, rg = bh_returns(q, DIV["QQQ"]), bh_returns(s, DIV["SPY"]), bh_returns(g, 0.0)
    gap = lambda r: r.index.to_series().diff().dt.days.fillna(1).to_numpy()  # noqa: E731
    lev2 = lambda r: 2 * r - BORROW / 365.0 * gap(r)  # noqa: E731
    sig = sma200_target(q).shift(1).reindex(rq.index).fillna(0.0)
    series = {
        "QQQ 买入持有": rq,
        "SPY 买入持有": rs_,
        "QQQ 2x 每日再平衡": lev2(rq),
        "SPY 2x 每日再平衡": lev2(rs_),
        "QQQ SMA200 1x": rq * sig,
        "QQQ SMA200 2x": lev2(rq) * sig,
        "50% QQQ + 50% GLD 买入持有": 0.5 * rq.reindex(rg.index).fillna(0) + 0.5 * rg,
    }
    legs = {"GLD": E.load_daily(gld_p, False), "QQQ": E.load_daily(qqq_p, False)}
    votes = {"GLD": vote_target(legs["GLD"]["close"]).shift(1).fillna(0.0),
             "QQQ": sma200_target(legs["QQQ"]["close"]).shift(1).fillna(0.0)}
    basket06 = S.simulate(legs, votes, None, cal06, k=2, lev=5)["equity"].pct_change().fillna(0.0)
    cal_all = {k: to_calendar(v, cal06) for k, v in series.items()}
    cal_all["篮子（k=2, 5x, 资金费 6%）"] = basket06

    rng = np.random.default_rng(3)
    print("== 2006 起")
    for k, r in cal_all.items():
        print(f"{k:28s} {stats(r)}")
    print("== 2022 起")
    for k, r in cal_all.items():
        print(f"{k:28s} {stats(r[r.index >= cal22[0]])}")
    tr = S.load_scalp_trades(Path("var/tmp/struct/nogate.csv"))
    tr["ret"] *= 0.1
    tr["eff_lev"] *= 0.1
    strat22 = S.simulate(legs, votes, tr, cal22, k=2, lev=5)["equity"].pct_change().fillna(0.0)
    print(f"{'策略（篮子 + scalp 0.1x）':28s} {stats(strat22)}")

    print("== 3 年跑赢 3% 存款（蒙特卡洛，历史 2006 起；策略 = 篮子 2006 起 + scalp 2022 起）")
    sc = scalp_daily(Path("var/tmp/struct/nogate.csv"), 0.1, "2022-01-01")
    for name in ["QQQ 买入持有", "SPY 买入持有", "QQQ 2x 每日再平衡", "QQQ SMA200 1x", "50% QQQ + 50% GLD 买入持有", "篮子（k=2, 5x, 资金费 6%）"]:
        r = cal_all[name].to_numpy()
        row = {f"h{int(h * 100)}_w{w}": mc(r, h, w, rng) for h in (0.0, 0.5) for w in (1.0, 0.3)}
        print(name, row)
    # 策略：两腿分开自助后相加
    rb = basket06.to_numpy()
    for h in (0.0, 0.5):
        for w in (1.0, 0.3):
            b = block_bootstrap(rb - h * rb.mean(), 6000, 365 * 3, 30.0, rng) + block_bootstrap(sc - 0.5 * sc.mean(), 6000, 365 * 3, 30.0, rng)
            eq = np.cumprod(np.clip(1 + b, 0, None), axis=1)[:, -1]
            tot = w * eq + (1 - w) * 1.03 ** 3
            c = tot ** (1 / 3) - 1
            print(f"策略 basket_h{int(h * 100)} scalp_h50 w{w}: p_beat {(tot > 1.03 ** 3).mean():.2f} p_loss {(tot < 1).mean():.2f} "
                  f"cagr_med {np.median(c) * 100:.1f} cagr_p5 {np.percentile(c, 5) * 100:.1f}")


if __name__ == "__main__":
    main()
