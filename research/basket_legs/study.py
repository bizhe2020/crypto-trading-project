"""篮子加腿研究：在 GLD(V5) + QQQ(SMA200) 之外，加入 OKX 上有永续合约的传统资产，看收益/回撤怎么变。

事先固定的规则（避免挑结果）：
  * 新腿信号一律用 SMA200 单规则（与 QQQ 腿同一条文献规则）；另跑一遍 V5 做稳健性对照。GLD 保持 V5、QQQ 保持 SMA200。
  * 等权：每条腿 weight = 1/n；篮子倍数 k 固定 2（满仓总名义 = 2 × 权益，和现在一样）。
  * 交易所杠杆 5x 逐仓，含强平；成本单边 8bp；多头资金费年化 --funding（默认 6%，所有腿相同）。
  * 回测从 2007-01-01 起（SMA200 需要约 1 年预热；SLV/USO/XBI 2006 年才上市，CPER 2011 年起，单独标注）。
  * 用 sim_basket_liquidation.simulate（与实盘篮子同一套机制：开盘按前一日收盘信号再平衡、偏离 10% 才调仓）。

输出：
  1. 单腿加入：{GLD, QQQ, X} 三腿等权 vs 两腿现状；全期 + 三段（2007-13 / 2014-20 / 2021+）。
  2. 预定义组合：按资产类别各取一条（股票 QQQ、黄金 GLD、长债 TLT、原油 USO、海外股 EWJ），以及全部候选等权。
  3. 各腿策略日收益的相关矩阵。
  4. 在相同最大回撤下比较：把 k 调到与两腿现状的全期回撤相同，看年化。
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
from strategy.trend_vote import sma200_target, vote_target  # noqa: E402

CANDIDATES = ["TLT", "IWM", "SPY", "EWJ", "EWZ", "EWY", "EWT", "USO", "XLE", "SLV", "CPER", "SMH", "XBI"]
PERIODS = {"2007-13": ("2007-01-01", "2014-01-01"), "2014-20": ("2014-01-01", "2021-01-01"), "2021+": ("2021-01-01", "2100-01-01")}


def stats(eq: pd.Series) -> dict:
    eq = eq / eq.iloc[0]
    r = eq.pct_change().dropna()
    yrs = len(eq) / 365.0
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else -1.0
    mdd = float((1 - eq / eq.cummax()).max())
    vol = float(r.std() * np.sqrt(365))
    return {"cagr": round(cagr * 100, 1), "mdd": round(mdd * 100, 1), "calmar": round(cagr / mdd, 2) if mdd else None,
            "sharpe": round(float(r.mean() * 365 / vol), 2) if vol else None}


def window(eq: pd.Series, a: str, b: str) -> pd.Series:
    return eq[(eq.index >= pd.Timestamp(a, tz="UTC")) & (eq.index < pd.Timestamp(b, tz="UTC"))]


def run(legs_all, names, rules, cal, k, lev):
    legs = {n: legs_all[n] for n in names}
    votes = {}
    for n in names:
        rule = rules.get(n, "sma200")
        f = vote_target if rule == "v5" else sma200_target
        votes[n] = f(legs[n]["close"]).shift(1).fillna(0.0)
    # simulate 的 weight 是统一标量 → 等权 1/n
    res = S.simulate(legs, votes, None, cal, k=k, lev=lev, weight=1.0 / len(names))
    return res


def summarize(res, start) -> dict:
    eq = res["equity"]
    out = {"all": stats(window(eq, start, "2100-01-01")),
           "liq": len(res["liquidations"])}
    for p, (a, b) in PERIODS.items():
        w = window(eq, max(a, start), b)
        out[p] = stats(w) if len(w) > 30 else None
    return out


def k_for_mdd(legs_all, names, rules, cal, lev, target_mdd, start) -> tuple[float, dict]:
    lo, hi = 0.5, 6.0
    best = None
    for _ in range(12):
        mid = (lo + hi) / 2
        s = summarize(run(legs_all, names, rules, cal, mid, lev), start)
        if s["all"]["mdd"] > target_mdd:
            hi = mid
        else:
            lo, best = mid, s
    return round(lo, 2), best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--etf-dir", required=True)
    p.add_argument("--gld-1d", required=True)
    p.add_argument("--qqq-1d", required=True)
    p.add_argument("--funding", type=float, default=0.06)
    p.add_argument("--k", type=float, default=2.0)
    p.add_argument("--lev", type=float, default=5.0)
    p.add_argument("--start", default="2007-01-01")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    E.LONG_FUNDING_PER_DAY = args.funding / 365.0

    legs_all = {"GLD": E.load_daily(Path(args.gld_1d), False), "QQQ": E.load_daily(Path(args.qqq_1d), False)}
    for s in CANDIDATES:
        legs_all[s] = E.load_daily(Path(args.etf_dir) / f"{s}-1d.feather", False)
    end = min(legs_all["GLD"].index[-1], legs_all["QQQ"].index[-1])
    cal = pd.date_range(pd.Timestamp("2005-01-01", tz="UTC"), end, freq="1D")
    start = args.start
    base_rules = {"GLD": "v5", "QQQ": "sma200"}
    out: dict = {"params": vars(args)}

    base = summarize(run(legs_all, ["GLD", "QQQ"], base_rules, cal, args.k, args.lev), start)
    out["base"] = base
    print("base GLD+QQQ", json.dumps(base, ensure_ascii=False), flush=True)

    single = {}
    for rule in ("sma200", "v5"):
        for s in CANDIDATES:
            st = max(start, str(legs_all[s].index[0] + pd.Timedelta(days=300))[:10])
            rules = dict(base_rules, **{s: rule})
            r = summarize(run(legs_all, ["GLD", "QQQ", s], rules, cal, args.k, args.lev), st)
            b = summarize(run(legs_all, ["GLD", "QQQ"], base_rules, cal, args.k, args.lev), st) if st != start else base
            beat = sum(1 for pp in PERIODS if r[pp] and b[pp] and r[pp]["calmar"] is not None and b[pp]["calmar"] is not None and r[pp]["calmar"] > b[pp]["calmar"])
            single[f"{s}|{rule}"] = {"start": st, "with": r, "base_same_start": b, "periods_calmar_beat": beat}
            print(f"+{s:5s} {rule:6s} from {st}: all {r['all']} vs base {b['all']} | calmar beat {beat}/3 | liq {r['liq']}", flush=True)
    out["single"] = single

    combos = {
        "class5": ["QQQ", "GLD", "TLT", "USO", "EWJ"],
        "class5_xle": ["QQQ", "GLD", "TLT", "XLE", "EWJ"],
        "all15": ["GLD", "QQQ"] + [c for c in CANDIDATES if c != "CPER"],
    }
    target_mdd = base["all"]["mdd"]
    out["combos"] = {}
    for name, names in combos.items():
        for rule in ("sma200", "v5"):
            rules = {n: rule for n in names}
            rules.update(base_rules)
            r = summarize(run(legs_all, names, rules, cal, args.k, args.lev), start)
            kk, rk = k_for_mdd(legs_all, names, rules, cal, args.lev, target_mdd, start)
            out["combos"][f"{name}|{rule}"] = {"legs": names, "k2": r, "k_match_mdd": kk, "at_k_match": rk}
            print(f"{name:10s} {rule:6s} k=2: {r['all']} | 07-13 {r['2007-13']} | 14-20 {r['2014-20']} | 21+ {r['2021+']} | liq {r['liq']}", flush=True)
            print(f"           matched MDD {target_mdd}%: k={kk} -> {rk['all'] if rk else None}", flush=True)
    kb, rb = k_for_mdd(legs_all, ["GLD", "QQQ"], base_rules, cal, args.lev, target_mdd, start)
    out["base_k_match"] = {"k": kb, "res": rb}

    # 各腿单独策略（信号 × 1 倍名义）的日收益相关
    rets = {}
    for n in ["GLD", "QQQ"] + CANDIDATES:
        rule = base_rules.get(n, "sma200")
        rr = run(legs_all, [n], {n: rule}, cal, 1.0, args.lev)["equity"].pct_change()
        rets[n] = rr[rr.index >= pd.Timestamp(start, tz="UTC")]
    corr = pd.DataFrame(rets).corr().round(2)
    out["corr"] = corr.to_dict()
    print(corr.to_string())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
