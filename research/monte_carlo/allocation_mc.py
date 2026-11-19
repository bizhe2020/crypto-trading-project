"""存款 + 策略的资金配比：N 年期跑赢存款的概率、最大亏损、期望超额。

* 策略日收益 = 篮子（2006 年起历史，k × 交易所杠杆 L）+ scalp（2022 年起不带闸门逐笔 × m），各自块自助后相加
  （同 research/live_path/split_haircut_mc.py），各自打 edge 折扣。
* 组合 = w × 策略 + (1 − w) × 存款（按 rate 复利），期初配好后不再再平衡。
  因此「整体跑赢存款」⇔「策略那一块跑赢存款」，概率与 w 无关；w 只改变亏损和超额的幅度。
* 先在 (m, k) 网格上找策略本身最优的组合，再对选出的策略算各个 w。
"""
from __future__ import annotations

import argparse
import json
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

SCEN = {"optimistic": (0.5, 0.0), "middle": (0.5, 0.5), "pessimistic": (1.0, 0.5)}  # (scalp 折扣, 篮子折扣)


def sleeve_paths(rb, rs, hs, hb, n_paths, n_days, rng):
    b = block_bootstrap(rb - hb * rb.mean(), n_paths, n_days, 30.0, rng)
    if rs is not None:
        b = b + block_bootstrap(rs - hs * rs.mean(), n_paths, n_days, 30.0, rng)
    return np.cumprod(np.clip(1 + b, 0, None), axis=1)  # 策略净值路径


def total_stats(eq, w, rate, years):
    n_days = eq.shape[1]
    dep = (1 + rate) ** (np.arange(1, n_days + 1) / 365.0)
    tot = w * eq + (1 - w) * dep
    final, dfinal = tot[:, -1], dep[-1]
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(tot), 1)), tot], axis=1), axis=1)[:, 1:]
    mdd = (1 - tot / peak).max(axis=1)
    excess_ann = (final / dfinal) ** (1 / years) - 1  # 相对存款的年化超额
    pct = lambda a, q: round(float(np.percentile(a, q)) * 100, 1)  # noqa: E731
    return {
        "p_beat": round(float((final > dfinal).mean()), 3),
        "p_loss_principal": round(float((final < 1).mean()), 3),
        "excess_total_mean_pct": round(float((final - dfinal).mean()) * 100, 1),
        "excess_total_median_pct": round(float(np.median(final - dfinal)) * 100, 1),
        "excess_ann_median_pct": pct(excess_ann, 50),
        "final_p5_pct": pct(final - 1, 5), "final_p1_pct": pct(final - 1, 1),
        "mdd_median_pct": pct(mdd, 50), "mdd_p95_pct": pct(mdd, 95),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gld-1d", required=True)
    p.add_argument("--qqq-1d", required=True)
    p.add_argument("--scalp-nogate", required=True)
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--rate", type=float, default=0.03)
    p.add_argument("--lev", type=float, default=5.0)
    p.add_argument("--m", default="0,0.1,0.25")
    p.add_argument("--k", default="0.5,1,1.5,2,2.5,3")
    p.add_argument("--w", default="0.1,0.2,0.3,0.5,1.0")
    p.add_argument("--paths", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    E.LONG_FUNDING_PER_DAY = 0.06 / 365.0
    legs = {"GLD": E.load_daily(Path(args.gld_1d), False), "QQQ": E.load_daily(Path(args.qqq_1d), False)}
    votes = {"GLD": vote_target(legs["GLD"]["close"]).shift(1).fillna(0.0),
             "QQQ": sma200_target(legs["QQQ"]["close"]).shift(1).fillna(0.0)}
    end = min(d.index[-1] for d in legs.values())
    cal = pd.date_range(pd.Timestamp("2006-01-01", tz="UTC"), end, freq="1D")
    n_days = 365 * args.years
    out: dict = {"params": vars(args), "grid": {}, "alloc": {}}

    rbs = {k: S.simulate(legs, votes, None, cal, k=k, lev=args.lev)["equity"].pct_change().dropna().to_numpy()
           for k in (float(x) for x in args.k.split(","))}
    rss = {m: (scalp_daily(Path(args.scalp_nogate), m, "2022-01-01") if m > 0 else None)
           for m in (float(x) for x in args.m.split(","))}
    for m, rs in rss.items():
        for k, rb in rbs.items():
            row = {}
            for sc, (hs, hb) in SCEN.items():
                eq = sleeve_paths(rb, rs, hs, hb, args.paths, n_days, np.random.default_rng(args.seed))  # 同种子：网格之间可比
                row[sc] = total_stats(eq, 1.0, args.rate, args.years)
            out["grid"][f"m{m}|k{k}"] = row
            print(f"m={m} k={k}: " + " | ".join(f"{sc} beat {v['p_beat']} exAnn50 {v['excess_ann_median_pct']} p5 {v['final_p5_pct']}"
                                              for sc, v in row.items()), flush=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
