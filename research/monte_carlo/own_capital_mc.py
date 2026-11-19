"""只用自有资金（不借钱）的蒙特卡洛：实盘组合（BTC scalp + 趋势篮子）能否长期跑赢存款利率。

1. 用 sim_basket_liquidation.simulate 复刻实盘口径的同账户逐日权益（2022 起）：
   GLD 腿 V5、QQQ 腿 SMA200、k=2、交易所 10x、含强平与 scalp 拒单、资金费 --funding。
2. 对日收益做平稳块自助法（stationary block bootstrap，平均块长 --block 天），保留回撤的聚集性。
3. edge 折扣：r' = r − h·mean(r)，h ∈ {0, 0.5, 0.75, 1}。h=1 表示「策略没有 edge、只剩波动」。
   回测是样本内结果（scalp 参数、QQQ 规则都看过历史），实盘也只有半年，所以 h=0 是乐观上限。
4. 输出各期限下：跑赢存款的概率、亏损概率、终值分位、最大回撤分布。
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


def live_equity(gld: Path, qqq: Path, trades_csv: Path, funding: float, k: float, lev: float) -> dict[str, pd.Series]:
    E.LONG_FUNDING_PER_DAY = funding / 365.0
    legs = {"GLD": E.load_daily(gld, False), "QQQ": E.load_daily(qqq, False)}
    votes = {
        "GLD": vote_target(legs["GLD"]["close"]).shift(1).fillna(0.0),
        "QQQ": sma200_target(legs["QQQ"]["close"]).shift(1).fillna(0.0),
    }
    end = min(d.index[-1] for d in legs.values())
    trades = S.load_scalp_trades(trades_csv)
    cal = pd.date_range(pd.Timestamp("2022-01-01", tz="UTC"), end, freq="1D")
    first = S.simulate(legs, votes, trades, cal, k=k, lev=lev)
    mf = first["margin_frac"]
    skip = np.array([
        (mf.get(d - pd.Timedelta(days=1), 0.0) + e / S.SCALP_LEVERAGE) > S.MARGIN_SAFETY
        for d, e in zip(trades["entry_day"], trades["eff_lev"])
    ])
    combo = S.simulate(legs, votes, trades, cal, k=k, lev=lev, skip_mask=skip)["equity"]
    scalp = S.simulate(legs, votes, trades, cal, k=0.0, lev=1.0)["equity"]
    return {"combo": combo, "scalp_only": scalp}


def block_bootstrap(r: np.ndarray, n_paths: int, n_days: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """平稳块自助法：每天以 1/block 概率跳到随机起点，否则顺延（循环取）。"""
    n = len(r)
    jump = rng.random((n_paths, n_days)) < 1.0 / block
    jump[:, 0] = True
    starts = rng.integers(0, n, size=(n_paths, n_days))
    idx = np.empty((n_paths, n_days), dtype=np.int64)
    cur = starts[:, 0]
    for t in range(n_days):
        cur = np.where(jump[:, t], starts[:, t], (cur + 1) % n)
        idx[:, t] = cur
    return r[idx]


def path_stats(rets: np.ndarray, years: float, deposit: dict[str, float]) -> dict:
    growth = np.clip(1.0 + rets, 0.0, None)
    eq = np.cumprod(growth, axis=1)
    final = eq[:, -1]
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(eq), 1)), eq], axis=1), axis=1)[:, 1:]
    mdd = (1.0 - eq / peak).max(axis=1)
    cagr = np.where(final > 0, final ** (1.0 / years) - 1.0, -1.0)
    q = lambda a, p: round(float(np.percentile(a, p)), 3)  # noqa: E731
    out = {
        "p_loss": round(float((final < 1.0).mean()), 3),
        "p_half": round(float((final < 0.5).mean()), 3),
        "p_mdd_ge_50": round(float((mdd >= 0.5).mean()), 3),
        "p_mdd_ge_70": round(float((mdd >= 0.7).mean()), 3),
        "final_p5": q(final, 5), "final_p25": q(final, 25), "final_p50": q(final, 50),
        "final_p75": q(final, 75), "final_p95": q(final, 95),
        "cagr_p50_pct": round(float(np.median(cagr)) * 100, 1),
        "mdd_p50_pct": round(float(np.median(mdd)) * 100, 1),
    }
    for name, rate in deposit.items():
        out[f"p_beat_{name}"] = round(float((final > (1 + rate) ** years).mean()), 3)
    return out


def hist_stats(eq: pd.Series) -> dict:
    eq = eq / eq.iloc[0]
    yrs = len(eq) / 365.0
    r = eq.pct_change().dropna()
    return {
        "days": len(eq), "total_pct": round(float(eq.iloc[-1] - 1) * 100, 1),
        "cagr_pct": round(float(eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1),
        "mdd_pct": round(float((1 - eq / eq.cummax()).max()) * 100, 1),
        "worst_day_pct": round(float(r.min()) * 100, 1), "best_day_pct": round(float(r.max()) * 100, 1),
        "ann_vol_pct": round(float(r.std() * np.sqrt(365)) * 100, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gld-1d", required=True)
    p.add_argument("--qqq-1d", required=True)
    p.add_argument("--scalp-trades", required=True)
    p.add_argument("--funding", type=float, default=0.06, help="多头资金费年化")
    p.add_argument("--k", type=float, default=2.0)
    p.add_argument("--lev", type=float, default=10.0)
    p.add_argument("--paths", type=int, default=10000)
    p.add_argument("--block", type=float, default=30.0, help="平均块长（天）")
    p.add_argument("--years", default="1,3,5,10")
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    deposit = {"dep1.0": 0.010, "dep1.5": 0.015, "dep3.0": 0.030}
    eqs = live_equity(Path(args.gld_1d), Path(args.qqq_1d), Path(args.scalp_trades), args.funding, args.k, args.lev)
    rng = np.random.default_rng(args.seed)
    out: dict = {"params": vars(args), "deposit_rates": deposit, "history": {}, "mc": {}}
    for name, eq in eqs.items():
        out["history"][name] = hist_stats(eq)
        r = eq.pct_change().dropna().to_numpy()
        mu = r.mean()
        out["mc"][name] = {}
        for h in (0.0, 0.5, 0.75, 1.0):
            rr = r - h * mu
            implied_cagr = float(np.prod(1 + rr) ** (365 / len(rr)) - 1)
            row = {"implied_hist_cagr_pct": round(implied_cagr * 100, 1)}
            for y in (int(x) for x in args.years.split(",")):
                sims = block_bootstrap(rr, args.paths, 365 * y, args.block, rng)
                row[f"{y}y"] = path_stats(sims, float(y), deposit)
            out["mc"][name][f"haircut{int(h * 100)}"] = row
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps(out["history"], ensure_ascii=False, indent=2))
    print(path)


if __name__ == "__main__":
    main()
