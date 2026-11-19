"""scalp 仓位倍数的阈值：在不同 edge 保留率下，增长最快的倍数 m*、增长归零的倍数 m0，以及组合（加篮子）下的最优倍数。

口径：
  * 逐笔收益 r_i = pnl / capital_at_entry（不带闸门的完整链路，m=1 即现状仓位，含手续费）。
  * edge 保留率 keep：r_i - (1 - keep) * mean(r)，即只扣均值、保留波动。
  * 解析部分：每笔对数增长 G(m) = mean(log(1 + m r))，按每年笔数换算成年化。
  * 组合部分：篮子 k=2、5 倍（2006 起日线），scalp 2022 起，两腿独立块自助，5 年。
"""
from __future__ import annotations

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

KEEPS = (1.0, 0.75, 0.5, 0.25)
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def trades(paths: list[str], start: str) -> tuple[np.ndarray, float]:
    t = pd.concat([pd.read_csv(p, parse_dates=["entry_time", "exit_time"]) for p in paths]).sort_values("entry_time")
    t = t[t["entry_time"] >= pd.Timestamp(start, tz="UTC")]
    yrs = (t["exit_time"].max() - t["entry_time"].min()).days / 365.0
    return (t["pnl"] / t["capital_at_entry"]).to_numpy(), len(t) / yrs


def analytic(r: np.ndarray, per_year: float) -> dict:
    out = {}
    fine = np.linspace(0.0, 2.0, 401)
    for keep in KEEPS:
        rr = r - (1 - keep) * r.mean()
        g = np.array([np.mean(np.log(np.clip(1 + m * rr, 1e-9, None))) for m in fine])
        i = int(np.argmax(g))
        m_star = float(fine[i])
        zero = fine[(fine > m_star) & (g <= 0)]
        m0 = float(zero[0]) if len(zero) else None
        cagr = lambda m: float(np.exp(per_year * np.mean(np.log(np.clip(1 + m * rr, 1e-9, None)))) - 1) * 100  # noqa: E731
        out[f"keep{keep}"] = {"m_star": m_star, "m_zero": m0, "cagr_at_m_star": round(cagr(m_star), 1),
                              "cagr_by_m": {str(m): round(cagr(m), 1) for m in (0.1, 0.25, 0.5, 0.75, 1.0)}}
    return out


def main() -> None:
    s22, s20 = "var/tmp/struct/nogate.csv", "var/tmp/struct/oos_nogate.csv"
    res: dict = {}
    for lab, ps, st in (("2022起", [s22], "2022-01-01"), ("含样本外2020-08起", [s20, s22], "2020-08-01")):
        r, py = trades(ps, st)
        res[f"analytic_{lab}"] = {"n": len(r), "per_year": round(py, 1), "mean_r": round(float(r.mean()) * 100, 3), **analytic(r, py)}
        print(lab, json.dumps(res[f"analytic_{lab}"], ensure_ascii=False))

    # 组合：篮子 + scalp，5 年蒙特卡洛
    E.LONG_FUNDING_PER_DAY = 0.06 / 365.0
    legs = {"GLD": E.load_daily(Path("var/tmp/time_in_market/etf_long/GLD-1d.feather"), False),
            "QQQ": E.load_daily(Path("var/tmp/time_in_market/etf_long/QQQ-1d.feather"), False)}
    votes = {"GLD": vote_target(legs["GLD"]["close"]).shift(1).fillna(0.0),
             "QQQ": sma200_target(legs["QQQ"]["close"]).shift(1).fillna(0.0)}
    cal = pd.date_range(pd.Timestamp("2006-01-01", tz="UTC"), min(d.index[-1] for d in legs.values()), freq="1D")
    rb = S.simulate(legs, votes, None, cal, k=2, lev=5)["equity"].pct_change().dropna().to_numpy()
    rng = np.random.default_rng(20261016)
    n_days, paths = 365 * 5, 3000
    combo = {}
    for hb in (1.0, 0.5):
        bb = block_bootstrap(rb - (1 - hb) * rb.mean(), paths, n_days, 30.0, rng)
        for keep in KEEPS:
            rows = {}
            for m in GRID:
                b = bb.copy()
                if m > 0:
                    rs = scalp_daily(Path(s22), float(m), "2022-01-01")
                    b = b + block_bootstrap(rs - (1 - keep) * rs.mean(), paths, n_days, 30.0, rng)
                eq = np.cumprod(np.clip(1 + b, 0, None), axis=1)
                f = eq[:, -1]
                peak = np.maximum.accumulate(eq, axis=1)
                rows[str(m)] = {"cagr_med": round(float(np.median(f) ** 0.2 - 1) * 100, 1),
                                "p_half": round(float((f < 0.5).mean()), 3), "p_loss": round(float((f < 1).mean()), 3),
                                "mdd_med": round(float(np.median((1 - eq / np.maximum(peak, 1)).max(axis=1))) * 100, 1)}
            best = max(rows, key=lambda k: rows[k]["cagr_med"])
            combo[f"basket{hb}|scalp{keep}"] = {"best_m": best, "rows": rows}
            print(f"篮子保留{hb} scalp保留{keep}: 最优 m={best} " +
                  " ".join(f"{m}:{v['cagr_med']}/{v['p_half']}" for m, v in rows.items()), flush=True)
    res["combo"] = combo
    Path("var/tmp/struct/scalp_threshold.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
