"""家庭总资产视角：国泰纳指 ETF + 黄金 ETF + 交易所纯 scalp（篮子停掉）。

口径：
  * 纳指 ETF 用 QQQ 日线 + 分红 0.7%/年 − 国内 ETF 费率 0.75%/年；黄金 ETF 用 GLD 日线 − 额外 0.2%/年
    （GLD 价格已含 0.4% 费率，国内黄金 ETF 约 0.6%）。不计汇率、QDII 溢价。
  * ETF 历史 2006 起；scalp 逐笔 = 不带闸门的完整链路（nogate.csv 2022 起，oos_nogate.csv 2020-08 到 2021-12）。
  * scalp 账户独立（逐仓/单独账户），最坏亏光账户，不会拖累 ETF；两腿独立自助（实测相关性约 0）。
  * ETF 每年再平衡一次回目标权重。
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

from research.monte_carlo.own_capital_mc import block_bootstrap  # noqa: E402
from research.monte_carlo.vs_buyhold import bh_returns, daily_close  # noqa: E402
from strategy.trend_vote import sma200_target  # noqa: E402

NDX_RMB, GOLD_RMB = 300_000.0, 200_000.0
SCALP_USDT, USDCNY = 8122.0, 7.1
YEARS, PATHS, RATE = (3, 5), 4000, 0.03


def scalp_daily_from(csvs: list[Path], m: float, start: str) -> np.ndarray:
    t = pd.concat([pd.read_csv(p, parse_dates=["entry_time", "exit_time"]) for p in csvs]).sort_values("entry_time")
    t = t[t["entry_time"] >= pd.Timestamp(start, tz="UTC")]
    ret = (t["pnl"] / t["capital_at_entry"] * m).to_numpy()
    cal = pd.date_range(pd.Timestamp(start, tz="UTC"), t["exit_time"].max().normalize(), freq="1D")
    daily = pd.Series(0.0, index=cal)
    for d, r in zip(t["exit_time"].dt.normalize(), ret):
        daily[d] = (1 + daily[d]) * (1 + r) - 1
    return daily.to_numpy()


def etf_paths(rq: np.ndarray, rg: np.ndarray, sig: np.ndarray, w_ndx: float, mode: str, n_days: int, rng) -> np.ndarray:
    """返回 (paths, n_days) 的 ETF 组合权益。纳指/黄金用同一组块索引自助，保留两者相关性。"""
    n = len(rq)
    jump = rng.random((PATHS, n_days)) < 1 / 30.0
    jump[:, 0] = True
    starts = rng.integers(0, n, size=(PATHS, n_days))
    idx = np.empty((PATHS, n_days), dtype=np.int64)
    cur = starts[:, 0]
    for t in range(n_days):
        cur = np.where(jump[:, t], starts[:, t], (cur + 1) % n)
        idx[:, t] = cur
    q, g, s = rq[idx], rg[idx], sig[idx]
    if mode == "sma200":  # 纳指那部分跌破 200 日线时换成黄金（不卖出 ETF 体系外，仍在两只 ETF 里切）
        q = s * q + (1 - s) * g
    eq_q, eq_g = np.full(PATHS, w_ndx), np.full(PATHS, 1 - w_ndx)
    out = np.empty((PATHS, n_days))
    for t in range(n_days):
        eq_q, eq_g = eq_q * (1 + q[:, t]), eq_g * (1 + g[:, t])
        tot = eq_q + eq_g
        if (t + 1) % 365 == 0:
            eq_q, eq_g = tot * w_ndx, tot * (1 - w_ndx)
        out[:, t] = tot
    return out


def summarize(total0: float, final: np.ndarray, years: int, mdd: np.ndarray) -> dict:
    c = (final / total0) ** (1 / years) - 1
    dep = total0 * (1 + RATE) ** years
    return {"p_beat3": round(float((final > dep).mean()), 2), "p_loss": round(float((final < total0).mean()), 2),
            "cagr_med": round(float(np.median(c)) * 100, 1), "cagr_mean_wealth": round(float((final.mean() / total0) ** (1 / years) - 1) * 100, 1),
            "cagr_p5": round(float(np.percentile(c, 5)) * 100, 1), "cagr_p95": round(float(np.percentile(c, 95)) * 100, 1),
            "mdd_med": round(float(np.median(mdd)) * 100, 1), "mdd_p95": round(float(np.percentile(mdd, 95)) * 100, 1),
            "gain_med_rmb": round(float(np.median(final) - total0)), "gain_p5_rmb": round(float(np.percentile(final, 5) - total0))}


def main() -> None:
    q = daily_close(Path("var/tmp/time_in_market/etf_long/QQQ-1d.feather"))
    g = daily_close(Path("var/tmp/time_in_market/etf_long/GLD-1d.feather"))
    cal = pd.date_range(pd.Timestamp("2006-01-01", tz="UTC"), min(q.index[-1], g.index[-1]), freq="1D")
    rq = bh_returns(q, 0.007 - 0.0075).reindex(cal).fillna(0.0)
    rg = bh_returns(g, -0.002).reindex(cal).fillna(0.0)
    sig = sma200_target(q).shift(1).reindex(cal).ffill().fillna(0.0)
    rq_a, rg_a, sig_a = rq.to_numpy(), rg.to_numpy(), sig.to_numpy()

    scalp_rmb = SCALP_USDT * USDCNY
    total0 = NDX_RMB + GOLD_RMB + scalp_rmb
    s22, s20 = Path("var/tmp/struct/nogate.csv"), Path("var/tmp/struct/oos_nogate.csv")
    scalp_scen = {  # 名称 -> (csv 列表, 起点, edge 保留率)
        "2022起全保": ([s22], "2022-01-01", 1.0),
        "2022起打五折": ([s22], "2022-01-01", 0.5),
        "含样本外2020-08起": ([s20, s22], "2020-08-01", 1.0),
        "无edge": ([s22], "2022-01-01", 0.0),
    }
    etf_scen = [("现状 60/40", 0.6, "bh"), ("100% 纳指", 1.0, "bh"), ("80/20", 0.8, "bh"), ("50/50", 0.5, "bh"),
                ("纳指SMA200切黄金 60/40", 0.6, "sma200"), ("纳指SMA200切黄金 100%", 1.0, "sma200")]
    ms = (0.0, 0.25, 0.5, 0.75, 1.0)
    rng = np.random.default_rng(20261016)
    res: dict = {"total0_rmb": round(total0), "scalp_rmb": round(scalp_rmb)}

    # 1) 历史回测：ETF 部分（2006 起、2022 起）
    hist = {}
    for name, w, mode in etf_scen:
        qq = rq_a if mode == "bh" else sig_a * rq_a + (1 - sig_a) * rg_a
        eq_q, eq_g, eq = w, 1 - w, []
        for t in range(len(cal)):
            eq_q, eq_g = eq_q * (1 + qq[t]), eq_g * (1 + rg_a[t])
            if cal[t].month == 1 and cal[t].day == 1:
                tot = eq_q + eq_g
                eq_q, eq_g = tot * w, tot * (1 - w)
            eq.append(eq_q + eq_g)
        e = pd.Series(eq, index=cal)
        row = {}
        for lab, st in (("2006起", "2006-01-01"), ("2022起", "2022-01-01")):
            ee = e[e.index >= pd.Timestamp(st, tz="UTC")]
            ee = ee / ee.iloc[0]
            yrs = len(ee) / 365.0
            row[lab] = {"cagr": round((ee.iloc[-1] ** (1 / yrs) - 1) * 100, 1), "mdd": round(float((1 - ee / ee.cummax()).max()) * 100, 1)}
        hist[name] = row
    res["etf_hist"] = hist

    # 2) scalp 单独（账户内）：各倍数、各情景 3 年
    sc_cache = {}
    scalp_only = {}
    for sname, (csvs, st, keep) in scalp_scen.items():
        for m in ms[1:]:
            r = scalp_daily_from(csvs, m, st)
            r = r - (1 - keep) * r.mean()
            sc_cache[(sname, m)] = r
            b = block_bootstrap(r, PATHS, 365 * 5, 30.0, rng)
            eq = np.cumprod(np.clip(1 + b, 0, None), axis=1)
            out = {}
            for y in YEARS:
                f = eq[:, 365 * y - 1]
                out[f"{y}y"] = {"cagr_med": round(float(np.median(f ** (1 / y) - 1)) * 100, 1),
                                "p_half_loss": round(float((f < 0.5).mean()), 2), "p_loss": round(float((f < 1).mean()), 2)}
            scalp_only[f"{sname} m={m}"] = out
    res["scalp_only"] = scalp_only

    # 3) 家庭总资产：ETF 情景 × scalp 倍数 × scalp 情景，ETF 用历史全保 / 打五折
    grid = {}
    n_days = 365 * max(YEARS)
    for name, w, mode in etf_scen:
        base = etf_paths(rq_a, rg_a, sig_a, w, mode, n_days, rng)
        for etf_keep in (1.0, 0.5):
            if etf_keep < 1.0:  # 打折：按日均收益扣掉一半（近似），对数口径
                mu = np.log(base[:, -1]).mean() / n_days
                etf_eq = base * np.exp(-0.5 * mu * np.arange(1, n_days + 1))
            else:
                etf_eq = base
            etf_eq = etf_eq * (NDX_RMB + GOLD_RMB)
            for sname in scalp_scen:
                for m in ms:
                    if m == 0:
                        sc_eq = np.full((PATHS, n_days), scalp_rmb)  # 不跑 scalp：账户里 USDT 放着（不计息）
                    else:
                        b = block_bootstrap(sc_cache[(sname, m)], PATHS, n_days, 30.0, rng)
                        sc_eq = scalp_rmb * np.cumprod(np.clip(1 + b, 0, None), axis=1)
                    tot = etf_eq + sc_eq
                    for y in YEARS:
                        tt = tot[:, : 365 * y]
                        peak = np.maximum.accumulate(np.concatenate([np.full((PATHS, 1), total0), tt], axis=1), axis=1)[:, 1:]
                        mdd = (1 - tt / peak).max(axis=1)
                        grid[f"{name}|etf{int(etf_keep * 100)}|{sname}|m{m}|{y}y"] = summarize(total0, tt[:, -1], y, mdd)
    res["grid"] = grid
    out = Path("var/tmp/monte_carlo/household.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({"total0": res["total0_rmb"], "etf_hist": hist}, ensure_ascii=False, indent=1))
    for k, v in scalp_only.items():
        print("scalp", k, v)
    for k, v in grid.items():
        if k.endswith("3y") and ("|m0.0|" in k or "|m0.5|" in k or "|m1.0|" in k):
            print(k, v)


if __name__ == "__main__":
    main()
