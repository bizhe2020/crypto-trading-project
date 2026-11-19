"""分腿打折的蒙特卡洛：scalp 和篮子各自打不同的 edge 折扣，再按日相加。

为什么分开：scalp 参数是在 2022 年后的数据上反复调出来的，2020-08~2021-12 的样本外重放 48 笔合计 −1.8R，
edge 的可信度远低于篮子（文献规则，2006 年起 20 年历史）。把两者混在一起打同一个折扣会高估 scalp。

做法：
  * 篮子：sim_basket_liquidation.simulate 只跑篮子（k, L），2006 年起逐日收益。
  * scalp：不带闸门的完整链路逐笔 × m，按平仓日记到日收益（相对当日权益），2022 年起。
  * 两条日收益序列各自做平稳块自助（块长 30 天）后逐日相加（实测日相关约 0.01~0.03，按独立处理）。
    组合日收益 = r_basket + r_scalp（同一账户、按总权益定仓的一阶近似；不含保证金挤占，m ≤ 0.5 时不触发）。
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


def scalp_daily(trades_csv: Path, m: float, start: str) -> np.ndarray:
    t = pd.read_csv(trades_csv, parse_dates=["entry_time", "exit_time"]).sort_values("entry_time")
    ret = t["pnl"] / t["capital_at_entry"] * m
    cal = pd.date_range(pd.Timestamp(start, tz="UTC"), t["exit_time"].max().normalize(), freq="1D")
    daily = pd.Series(0.0, index=cal)
    for d, r in zip(t["exit_time"].dt.normalize(), ret):
        daily[d] = (1 + daily[d]) * (1 + r) - 1
    return daily.to_numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gld-1d", required=True)
    p.add_argument("--qqq-1d", required=True)
    p.add_argument("--scalp-nogate", required=True)
    p.add_argument("--m", default="0,0.25,0.33,0.5,1.0")
    p.add_argument("--k", default="1.5,2,2.5")
    p.add_argument("--lev", type=float, default=5.0)
    p.add_argument("--funding", type=float, default=0.06)
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--paths", type=int, default=6000)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    E.LONG_FUNDING_PER_DAY = args.funding / 365.0
    legs = {"GLD": E.load_daily(Path(args.gld_1d), False), "QQQ": E.load_daily(Path(args.qqq_1d), False)}
    votes = {"GLD": vote_target(legs["GLD"]["close"]).shift(1).fillna(0.0),
             "QQQ": sma200_target(legs["QQQ"]["close"]).shift(1).fillna(0.0)}
    end = min(d.index[-1] for d in legs.values())
    cal = pd.date_range(pd.Timestamp("2006-01-01", tz="UTC"), end, freq="1D")
    n_days = 365 * args.years
    rng = np.random.default_rng(args.seed)
    deposit = {"dep1.5": 0.015}
    scenarios = {"scalp_h50_basket_h0": (0.5, 0.0), "scalp_h100_basket_h0": (1.0, 0.0),
                 "scalp_h50_basket_h50": (0.5, 0.5), "scalp_h100_basket_h50": (1.0, 0.5)}
    out: dict = {"params": vars(args), "rows": {}}
    for k in (float(x) for x in args.k.split(",")):
        rb = S.simulate(legs, votes, None, cal, k=k, lev=args.lev)["equity"].pct_change().dropna().to_numpy()
        for m in (float(x) for x in args.m.split(",")):
            rs = scalp_daily(Path(args.scalp_nogate), m, "2022-01-01") if m > 0 else None
            row = {}
            for sc, (hs, hb) in scenarios.items():
                b = block_bootstrap(rb - hb * rb.mean(), args.paths, n_days, 30.0, rng)
                if rs is not None:
                    b = b + block_bootstrap(rs - hs * rs.mean(), args.paths, n_days, 30.0, rng)
                st = path_stats(b, float(args.years), deposit)
                row[sc] = {x: st[x] for x in ("p_beat_dep1.5", "p_loss", "p_half", "final_p5", "final_p50", "cagr_p50_pct", "mdd_p50_pct")}
            out["rows"][f"k{k}|m{m}"] = row
            print(f"k={k} m={m}: " + " | ".join(
                f"{sc}: beat {v['p_beat_dep1.5']} half {v['p_half']} cagr50 {v['cagr_p50_pct']} mdd50 {v['mdd_p50_pct']}"
                for sc, v in row.items()), flush=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
