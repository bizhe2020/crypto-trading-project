"""QQQ SMA200 缓冲/确认 + 黄金腿投票门槛研究。

背景：2026-11 实盘 QQQ 腿在 647.6 清仓、690.8 追回，一次 whipsaw 损失约 590 USDT。
本脚本只回答两个问题：
  1. 给 QQQ 的 SMA200 加 buffer / 连续确认，长期能不能降低 whipsaw、改善 Calmar？
  2. 黄金腿要求 vote >= 0.4 / 0.6，或者直接去掉，组合会不会更好？

口径与实盘一致：
  * 日线收盘出信号，次日开盘执行（votes 已 shift(1)）。
  * 篮子 k=2.5、交易所杠杆 5x、逐仓强平、单边 8bp、多头资金费 6%/年。
  * 2022+ 组合叠加 scalp 无闸门逐笔 ×0.25（与当前实盘风险预算一致）。
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
from strategy.trend_vote import vote_target  # noqa: E402


def sma200_buffer_target(close: pd.Series, buffer: float = 0.0, confirm: int = 1) -> pd.Series:
    """带缓冲/连续确认的 SMA200 状态机。

    buffer=0, confirm=1 时与 strategy.trend_vote.sma200_target 等价（严格大于）。
    状态在收盘决定，调用方负责 shift(1) 后执行。
    """
    sma = close.rolling(200).mean()
    c = close.to_numpy(dtype=float)
    s = sma.to_numpy(dtype=float)
    state = 0
    above = 0
    below = 0
    out = np.zeros(len(c), dtype=float)
    for i in range(len(c)):
        if np.isnan(s[i]):
            out[i] = 0.0
            continue
        if state == 0:
            if c[i] > s[i] * (1.0 + buffer):
                above += 1
                if above >= confirm:
                    state = 1
                    above = 0
                    below = 0
            else:
                above = 0
        else:
            if c[i] < s[i] * (1.0 - buffer):
                below += 1
                if below >= confirm:
                    state = 0
                    above = 0
                    below = 0
            else:
                below = 0
        out[i] = float(state)
    return pd.Series(out, index=close.index)


def gold_threshold(vote: pd.Series, threshold: float) -> pd.Series:
    if threshold <= 0:
        return vote
    return vote.where(vote >= threshold - 1e-9, 0.0)


def period_stats(eq: pd.Series, start: str, end: str | None = None) -> dict | None:
    s = eq[eq.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        s = s[s.index < pd.Timestamp(end, tz="UTC")]
    if len(s) < 30:
        return None
    s = s / s.iloc[0]
    yrs = len(s) / 365.0
    cagr = float(s.iloc[-1] ** (1 / yrs) - 1) if s.iloc[-1] > 0 else -1.0
    mdd = float((1 - s / s.cummax()).max())
    return {
        "cagr": round(cagr * 100, 1),
        "mdd": round(mdd * 100, 1),
        "calmar": round(cagr / mdd, 2) if mdd > 0 else None,
        "ret": round(float(s.iloc[-1] - 1) * 100, 1),
    }


def summarize(eq: pd.Series, start: str = "2007-01-01") -> dict:
    windows = {
        "all": (start, None),
        "2007-13": ("2007-01-01", "2014-01-01"),
        "2014-20": ("2014-01-01", "2021-01-01"),
        "2021+": ("2021-01-01", None),
        "2022+": ("2022-01-01", None),
        "2025+": ("2025-01-01", None),
    }
    out = {k: period_stats(eq, a, b) for k, (a, b) in windows.items()}
    out["all"] = out["all"] or period_stats(eq, start)
    return out


def run_basket(legs, votes, cal, k=2.5, lev=5.0, weight=0.5):
    res = S.simulate(legs, votes, None, cal, k=k, lev=lev, weight=weight)
    return res


def run_combo(legs, votes, trades, cal, k=2.5, lev=5.0, weight=0.5):
    first = S.simulate(legs, votes, trades, cal, k=k, lev=lev, weight=weight)
    mf = first["margin_frac"]
    skip = np.array([
        (mf.get(d - pd.Timedelta(days=1), 0.0) + e / S.SCALP_LEVERAGE) > S.MARGIN_SAFETY
        for d, e in zip(trades["entry_day"], trades["eff_lev"])
    ])
    res = S.simulate(legs, votes, trades, cal, k=k, lev=lev, weight=weight, skip_mask=skip)
    res["rejected"] = int(skip.sum())
    return res


def transitions(target: pd.Series) -> int:
    t = target.fillna(0.0).to_numpy()
    return int(((t[1:] > t[:-1]).sum()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gld-1d", default="var/tmp/time_in_market/etf_long/GLD-1d.feather")
    p.add_argument("--qqq-1d", default="var/tmp/time_in_market/etf_long/QQQ-1d.feather")
    p.add_argument("--scalp-nogate", default="var/tmp/struct/nogate.csv")
    p.add_argument("--funding", type=float, default=0.06)
    p.add_argument("--k", type=float, default=2.5)
    p.add_argument("--lev", type=float, default=5.0)
    p.add_argument("--output", default="var/tmp/live_path/qqq_buffer_study.json")
    args = p.parse_args()

    E.LONG_FUNDING_PER_DAY = args.funding / 365.0
    legs = {
        "GLD": E.load_daily(Path(args.gld_1d), False),
        "QQQ": E.load_daily(Path(args.qqq_1d), False),
    }
    end = min(d.index[-1] for d in legs.values())
    cal_long = pd.date_range(pd.Timestamp("2005-01-01", tz="UTC"), end, freq="1D")
    cal_22 = pd.date_range(pd.Timestamp("2022-01-01", tz="UTC"), end, freq="1D")

    gold_vote = vote_target(legs["GLD"]["close"]).shift(1).fillna(0.0)

    qqq_variants = {
        "sma200": (0.0, 1),
        "buf1": (0.01, 1),
        "buf2": (0.02, 1),
        "buf3": (0.03, 1),
        "buf5": (0.05, 1),
        "conf2": (0.0, 2),
        "buf2_conf2": (0.02, 2),
        "buf3_conf2": (0.03, 2),
    }

    # scalp x0.25 for 2022+ combo
    base_trades = S.load_scalp_trades(Path(args.scalp_nogate))
    trades = base_trades.copy()
    trades["ret"] *= 0.25
    trades["eff_lev"] *= 0.25

    out = {"params": vars(args), "qqq": {}, "gold": {}, "combo": {}}

    print("=== A. QQQ SMA200 buffer / confirm (GLD V5 + QQQ variant, 2007+, k2.5 L5) ===", flush=True)
    for name, (buf, conf) in qqq_variants.items():
        qqq_raw = sma200_buffer_target(legs["QQQ"]["close"], buffer=buf, confirm=conf)
        votes = {"GLD": gold_vote, "QQQ": qqq_raw.shift(1).fillna(0.0)}
        res = run_basket(legs, votes, cal_long, k=args.k, lev=args.lev, weight=0.5)
        st = summarize(res["equity"], "2007-01-01")
        st["liq"] = len(res["liquidations"])
        st["qqq_trips"] = transitions(qqq_raw)
        st["qqq_in_market_pct"] = round(float(qqq_raw.mean()) * 100, 1)
        out["qqq"][name] = st
        a = st["all"]
        print(f"{name:10s} all {a['cagr']:6.1f}%/{a['mdd']:5.1f}% cal {a['calmar']} | "
              f"07-13 {st['2007-13']['cagr'] if st['2007-13'] else '-':>6}% | "
              f"14-20 {st['2014-20']['cagr'] if st['2014-20'] else '-':>6}% | "
              f"21+ {st['2021+']['cagr'] if st['2021+'] else '-':>6}% | "
              f"liq {st['liq']} trips {st['qqq_trips']} in {st['qqq_in_market_pct']}%", flush=True)

    # pick best QQQ variant by full-period calmar, tie-break by 2014-20/2021+ calmar
    def score(n):
        s = out["qqq"][n]
        c = [s["all"]["calmar"] or 0]
        c += [s[p]["calmar"] if s[p] and s[p]["calmar"] is not None else 0 for p in ("2007-13", "2014-20", "2021+")]
        return (c[0], sum(c[1:]))

    best_qqq = max(qqq_variants, key=score)
    print(f"best qqq by calmar: {best_qqq}", flush=True)

    # B. gold threshold with best qqq + current qqq
    print("=== B. Gold vote threshold (best QQQ variant, 2007+) ===", flush=True)
    gold_variants = {"v5": 0.0, "thr04": 0.4, "thr06": 0.6, "none": None}
    for qname in qqq_variants:
        buf, conf = qqq_variants[qname]
        qqq_raw = sma200_buffer_target(legs["QQQ"]["close"], buffer=buf, confirm=conf)
        for gname in ("v5", "thr04", "thr06"):
            thr = gold_variants[gname]
            if gname == "none":
                use_legs = {"QQQ": legs["QQQ"]}
                votes = {"QQQ": qqq_raw.shift(1).fillna(0.0)}
                weight = 1.0
            else:
                use_legs = legs
                gv = gold_threshold(gold_vote, thr) if thr else gold_vote
                votes = {"GLD": gv, "QQQ": qqq_raw.shift(1).fillna(0.0)}
                weight = 0.5
            res = run_basket(use_legs, votes, cal_long, k=args.k, lev=args.lev, weight=weight)
            st = summarize(res["equity"], "2007-01-01")
            st["liq"] = len(res["liquidations"])
            out["gold"][f"{qname}|{gname}"] = st
            a = st["all"]
            print(f"{qname:10s} gold {gname:6s}: all {a['cagr']:6.1f}%/{a['mdd']:5.1f}% cal {a['calmar']} | "
                  f"07-13 {st['2007-13']['cagr'] if st['2007-13'] else '-':>6}% | "
                  f"14-20 {st['2014-20']['cagr'] if st['2014-20'] else '-':>6}% | "
                  f"21+ {st['2021+']['cagr'] if st['2021+'] else '-':>6}% | liq {st['liq']}", flush=True)

    # C. 2022+ combo with scalp x0.25
    print("=== C. 2022+ combo with scalp x0.25 ===", flush=True)
    combos = {}
    for qname in qqq_variants:
        buf, conf = qqq_variants[qname]
        qqq_raw = sma200_buffer_target(legs["QQQ"]["close"], buffer=buf, confirm=conf)
        for gname in ("v5", "thr04", "thr06"):
            thr = gold_variants[gname]
            if gname == "none":
                use_legs = {"QQQ": legs["QQQ"]}
                votes = {"QQQ": qqq_raw.shift(1).fillna(0.0)}
                weight = 1.0
            else:
                use_legs = legs
                gv = gold_threshold(gold_vote, thr) if thr else gold_vote
                votes = {"GLD": gv, "QQQ": qqq_raw.shift(1).fillna(0.0)}
                weight = 0.5
            res = run_combo(use_legs, votes, trades, cal_22, k=args.k, lev=args.lev, weight=weight)
            st = summarize(res["equity"], "2022-01-01")
            st["rejected"] = res.get("rejected", 0)
            combos[f"{qname}|{gname}"] = st
            a = st["all"]
            print(f"{qname:10s} gold {gname:6s}: 22+ {a['cagr']:6.1f}%/{a['mdd']:5.1f}% cal {a['calmar']} | "
                  f"2025+ {st['2025+']['cagr'] if st['2025+'] else '-':>6}% | rej {st['rejected']}", flush=True)
    out["combo"] = combos

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
