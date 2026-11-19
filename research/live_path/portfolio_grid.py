"""组合结构网格：scalp 仓位倍数 m × 篮子倍数 k × 篮子交易所杠杆 L（同账户逐日模拟 + 块自助蒙特卡洛）。

scalp 逐笔来自不带执行层闸门的完整链路重放：交易集合与仓位无关，按 m 线性缩放 ret 与有效杠杆即为精确结果。
另外可传入若干「实测」逐笔（例如带闸门的 1.0 倍现状）做对照，这些不再缩放。
篮子：GLD 腿 V5、QQQ 腿 SMA200，资金费年化 --funding，含逐仓强平和 scalp 拒单（sim_basket_liquidation.simulate）。
蒙特卡洛：日收益平稳块自助（块长 30 天），edge 折扣 h ∈ {0, 0.5}：r' = r − h·mean(r)。
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
from research.monte_carlo.own_capital_mc import block_bootstrap, path_stats  # noqa: E402
from strategy.trend_vote import sma200_target, vote_target  # noqa: E402


def run_combo(legs, votes, trades, cal, k, lev):
    if trades is None or k == 0:
        return S.simulate(legs, votes, trades, cal, k=k, lev=lev)
    first = S.simulate(legs, votes, trades, cal, k=k, lev=lev)
    mf = first["margin_frac"]
    skip = np.array([(mf.get(d - pd.Timedelta(days=1), 0.0) + e / S.SCALP_LEVERAGE) > S.MARGIN_SAFETY
                     for d, e in zip(trades["entry_day"], trades["eff_lev"])])
    res = S.simulate(legs, votes, trades, cal, k=k, lev=lev, skip_mask=skip)
    res["rejected"] = int(skip.sum())
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gld-1d", required=True)
    p.add_argument("--qqq-1d", required=True)
    p.add_argument("--scalp-nogate", required=True)
    p.add_argument("--extra", action="append", default=[], metavar="NAME=CSV")
    p.add_argument("--m", default="0,0.25,0.33,0.4,0.5,0.75,1.0")
    p.add_argument("--k", default="0,1,1.5,2,2.5,3")
    p.add_argument("--lev", default="5,10")
    p.add_argument("--funding", type=float, default=0.06)
    p.add_argument("--paths", type=int, default=4000)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    E.LONG_FUNDING_PER_DAY = args.funding / 365.0
    legs = {"GLD": E.load_daily(Path(args.gld_1d), False), "QQQ": E.load_daily(Path(args.qqq_1d), False)}
    votes = {"GLD": vote_target(legs["GLD"]["close"]).shift(1).fillna(0.0),
             "QQQ": sma200_target(legs["QQQ"]["close"]).shift(1).fillna(0.0)}
    end = min(d.index[-1] for d in legs.values())
    cal = pd.date_range(pd.Timestamp("2022-01-01", tz="UTC"), end, freq="1D")
    w25 = pd.Timestamp("2025-01-01", tz="UTC")

    base = S.load_scalp_trades(Path(args.scalp_nogate))
    scalp_sets: dict[str, pd.DataFrame | None] = {}
    for m in (float(x) for x in args.m.split(",")):
        if m == 0:
            scalp_sets["m0"] = None
            continue
        t = base.copy()
        t["ret"] *= m
        t["eff_lev"] *= m
        scalp_sets[f"m{m}"] = t
    for item in args.extra:
        name, path = item.split("=", 1)
        scalp_sets[name] = S.load_scalp_trades(Path(path))

    rng = np.random.default_rng(args.seed)
    deposit = {"dep1.5": 0.015}
    rows = {}
    for sname, trades in scalp_sets.items():
        for k in (float(x) for x in args.k.split(",")):
            for lev in (float(x) for x in args.lev.split(",")):
                if k == 0 and lev != float(args.lev.split(",")[0]):
                    continue
                if trades is None and k == 0:
                    continue
                res = run_combo(legs, votes, trades, cal, k, lev)
                r = res["equity"].pct_change().dropna().to_numpy()
                mc = {}
                for h in (0.0, 0.5):
                    sims = block_bootstrap(r - h * r.mean(), args.paths, 365 * 5, 30.0, rng)
                    st = path_stats(sims, 5.0, deposit)
                    mc[f"h{int(h * 100)}"] = {k2: st[k2] for k2 in ("p_beat_dep1.5", "p_loss", "p_half", "final_p50", "mdd_p50_pct")}
                rows[f"{sname}|k{k}|L{int(lev)}"] = {
                    "all": S.summarize(res, cal[0]), "from2025": S.summarize(res, w25),
                    "rejected": res.get("rejected", 0), "mc5y": mc,
                }
                a, b = rows[f"{sname}|k{k}|L{int(lev)}"]["all"], rows[f"{sname}|k{k}|L{int(lev)}"]["from2025"]
                print(f"{sname:10s} k={k:<4} L={int(lev):<3} all {a['cagr_pct']:6.1f}%/{a['mdd_pct']:5.1f}% cal {a['calmar']} liq {a['liquidations']} "
                      f"| 25+ {b['cagr_pct']:6.1f}%/{b['mdd_pct']:5.1f}% | rej {rows[f'{sname}|k{k}|L{int(lev)}']['rejected']} "
                      f"| h0 beat {mc['h0']['p_beat_dep1.5']} | h50 beat {mc['h50']['p_beat_dep1.5']} half {mc['h50']['p_half']} mdd50 {mc['h50']['mdd_p50_pct']}",
                      flush=True)
    Path(args.output).write_text(json.dumps({"params": vars(args), "rows": rows}, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
