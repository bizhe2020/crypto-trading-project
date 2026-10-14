#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1.5x 现货策略 + collar 叠加回测（round 10）。

动机：round 9 证明 collar 对 1x QQQ 能压波动/回撤并提升 Sharpe（真实期权链）。
对**加杠杆**的组合，尾部保护的价值应更大。这里把两者接起来：

  组合日收益 = L_t × r_QQQ(t) − 融资 + 期权腿日收益
    * L_t       = 主策略目标杠杆（趋势阶梯，默认 overlay 全关；可按需打开 VIX/RS）
    * 期权腿    = 每 21 个交易日滚动 1 月期 collar（买 put 付 ask / 卖 call 收 bid），
                  到期按内在价值结算，持有期用 Black-Scholes（入场 IV）估值
    * 期权仅按 **裸 1x 名义** 购买（未按 L_t 放大），因此对冲比率随杠杆下降——这更贴近
      "先做尾部保险，再决定仓位" 的现实，也便于看清纯保护效果。

数据：/tmp/qqq_chain_cache.csv（firstrate 原始 QQQ 链抽取，19MB）
窗口：2023-07-03 ~ 2026-08-07（真实期权报价的完整覆盖期）

用法: python3 research/qqq_spot_leverage/levered_collar_backtest.py [--sweep]
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_qqq_spot_leverage import (  # noqa: E402
    Params,
    build_target_series,
    load_market_frame,
    prepare_indicators,
)

OUT = Path(__file__).resolve().parent / "out"
CACHE = Path("/tmp/qqq_put_cache.csv")
RF = 0.045
ROLL_DAYS = 21
TARGET_DTE = 35
COST_BPS = 1.0
_IDX = None
_EXPS = None


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put(s: float, k: float, t: float, vol: float, r: float = RF) -> float:
    if t <= 0 or vol <= 0:
        return max(k - s, 0.0)
    d1 = (math.log(s / k) + (r + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    return k * math.exp(-r * t) * norm_cdf(-d2) - s * norm_cdf(-d1)


def bs_call(s: float, k: float, t: float, vol: float, r: float = RF) -> float:
    if t <= 0 or vol <= 0:
        return max(s - k, 0.0)
    d1 = (math.log(s / k) + (r + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    return s * norm_cdf(d1) - k * math.exp(-r * t) * norm_cdf(d2)


def stats(nav: pd.Series) -> dict:
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    r = nav.pct_change().dropna()
    return {
        "total_pct": round(float(nav.iloc[-1] - 1) * 100, 2),
        "cagr_pct": round(float(nav.iloc[-1] ** (1 / years) - 1) * 100, 2),
        "mdd_pct": round(float((nav / nav.cummax() - 1).min()) * 100, 2),
        "vol_pct": round(float(r.std() * math.sqrt(252)) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * math.sqrt(252)), 2) if r.std() > 0 else 0.0,
    }


def build_index(opt: pd.DataFrame) -> tuple[dict, dict]:
    """返回 ({(date_str, expiry_str): {strike: (bid, ask, iv)}}, {date_str: sorted expiries})。"""
    idx: dict = {}
    exps_by_date: dict = {}
    for t, exp, k, bid, ask, iv in zip(opt["date"], opt["expiry"], opt["strike"], opt["bid"], opt["ask"], opt["iv"]):
        ts, es = str(t)[:10], str(exp)[:10]
        idx.setdefault((ts, es), {})[float(k)] = (float(bid), float(ask), float(iv or 0.25))
        exps_by_date.setdefault(ts, set()).add(es)
    return idx, {d: sorted(v) for d, v in exps_by_date.items()}


def pick_strike(chain: dict, target: float) -> float:
    return min(chain.keys(), key=lambda k: abs(k - target))


def option_legs(px: pd.Series, dates: list, opt: pd.DataFrame, put_m: float, call_m: float, idx: dict | None = None, exps_by_date: dict | None = None):
    """返回 (逐日市值 mtm, 逐日现金流 cashflow, 事件)。

    滚动规则（无保护缺口）：
      * 每个滚动日：先按内在价值结算手上剩余期权（现金入账），再买新 put（付 ask），
        可选卖新 call（收 bid）；因此任何时刻都持有保护。
      * 若在到期日之前自然到期，也按内在价值结算。
    记账分离：权利金/结算进入 cashflow，mtm 只记市值；期权腿 P&L = Δmtm − cashflow。
    """
    mtm = pd.Series(0.0, index=dates)
    cash = pd.Series(0.0, index=dates)
    events = []
    put_leg = call_leg = None
    roll_dates = set(dates[::ROLL_DAYS])
    if idx is None:
        idx, exps_by_date = build_index(opt)

    def chain_for(t, exp_str: str) -> dict:
        return idx.get((str(t)[:10], exp_str), {})

    def settle(s: float, t) -> None:
        nonlocal put_leg, call_leg
        if put_leg is not None:
            pay = max(put_leg[0] - s, 0.0)
            cash.loc[t] += pay
            events.append({"date": str(t.date()), "type": "put_settle", "strike": put_leg[0], "spot": round(s, 2), "payoff": round(pay, 2)})
            put_leg = None
        if call_leg is not None:
            pay = max(s - call_leg[0], 0.0)
            cash.loc[t] -= pay
            events.append({"date": str(t.date()), "type": "call_settle", "strike": call_leg[0], "spot": round(s, 2), "payoff": round(-pay, 2)})
            call_leg = None

    for t in dates:
        s = float(px.loc[t])
        # 自然到期结算
        if (put_leg is not None and t >= put_leg[1]) or (call_leg is not None and t >= call_leg[1]):
            settle(s, t)
        # 滚动：先结算未到期部分，再开新仓（保证无缺口）
        if t in roll_dates:
            if put_leg is not None or call_leg is not None:
                settle(s, t)
            ts = str(t)[:10]
            exps = [e for e in (exps_by_date or {}).get(ts, []) if e > ts]
            if exps:
                exp = min(exps, key=lambda e: abs((pd.Timestamp(e) - t).days - TARGET_DTE))
                chain = chain_for(t, exp)
                if chain:
                    pk = pick_strike(chain, s * put_m)
                    pb, pa, piv = chain[pk]
                    put_leg = (pk, pd.Timestamp(exp), piv, t, pa)
                    cash.loc[t] -= pa
                    ev = {"date": str(t.date()), "type": "open", "spot": round(s, 2),
                          "put": pk, "put_ask": round(pa, 2), "expiry": str(pd.Timestamp(exp).date())}
                    if call_m > 0:
                        ck = pick_strike(chain, s * call_m)
                        cb, ca, civ = chain[ck]
                        call_leg = (ck, pd.Timestamp(exp), civ, t, cb)
                        cash.loc[t] += cb
                        ev.update({"call": ck, "call_bid": round(cb, 2)})
                    ev["net"] = round(-pa + (cb if call_leg is not None and "call" in ev else 0.0), 2)
                    events.append(ev)
        # 收盘 MTM：优先用"当日该合约的市场中价"，否则用当日 IV 重估，最后回退 entry IV
        v = 0.0
        if put_leg is not None:
            tau = max((put_leg[1] - t).days / 365.0, 0.0)
            if t == put_leg[3]:          # 建仓日：按付出的 ask 记账（保守）
                v += put_leg[4]
            else:
                q = chain_for(t, put_leg[1].strftime("%Y-%m-%d")).get(put_leg[0])
                if q is not None:
                    v += q[0]                              # 多头保守按 bid 估
                else:
                    v += bs_put(s, put_leg[0], tau, put_leg[2]) if tau > 0 else max(put_leg[0] - s, 0.0)
        if call_leg is not None:
            tau = max((call_leg[1] - t).days / 365.0, 0.0)
            if t == call_leg[3]:
                v -= call_leg[4]
            else:
                q = opt[(opt["date"] == t) & (opt["expiry"] == call_leg[1].strftime("%Y-%m-%d"))
                        & (opt["cp"] == "c") & (opt["strike"] == call_leg[0])]
                if not q.empty:
                    v -= float(q.iloc[0]["ask"])         # 空头保守按 ask 估
                else:
                    v -= bs_call(s, call_leg[0], tau, call_leg[2]) if tau > 0 else max(s - call_leg[0], 0.0)
        mtm.loc[t] = v
    return mtm, cash, events


def run(px: pd.Series, target: pd.Series, dates: list, opt: pd.DataFrame,
        put_m: float, call_m: float, use_collar: bool, hedge_ratio: float = 1.0):
    """美元账本：股票腿 = L × 净值；期权腿 = hedge_ratio × 期初净值 的保险（按点数计价）。

    组合每日 P&L 增加：units × (Δmtm) − units × cashflow
    其中 cashflow 是当日权利金/结算现金（买 put 为负、卖 call 为正）。
    """
    s0 = float(px.loc[dates[0]])
    global _IDX, _EXPS
    if use_collar and _IDX is None:
        _IDX, _EXPS = build_index(opt)
    mtm, cf, events = (option_legs(px, dates, opt, put_m, call_m, _IDX, _EXPS) if use_collar
                       else (pd.Series(0.0, index=dates), pd.Series(0.0, index=dates), []))
    units = (hedge_ratio / s0) if use_collar else 0.0   # 以"期初净值 1"为基准的保险份数
    prev_mtm = float(mtm.loc[dates[0]])
    nav = 1.0
    lev = 0.0
    lev_target = target.reindex(dates).ffill().fillna(0.0)
    navs = []
    for i, t in enumerate(dates):
        s = float(px.loc[t])
        s_prev = float(px.loc[dates[i - 1]]) if i > 0 else s
        r = s / s_prev - 1.0 if i > 0 else 0.0
        fin = max(lev - 1.0, 0.0) * 0.045 / 252.0
        nav *= 1.0 + lev * r - fin
        # 期权腿：市值变化 − 当日现金流
        nav += units * (float(mtm.loc[t]) - prev_mtm) - units * float(cf.loc[t])
        prev_mtm = float(mtm.loc[t])
        want = float(lev_target.loc[t])
        if abs(want - lev) >= 0.25 or (want == 0 and lev != 0):
            nav *= 1.0 - abs(want - lev) * COST_BPS / 1e4
            lev = want
        navs.append(nav)
    return pd.Series(navs, index=dates), events


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    opt = pd.read_csv(CACHE)
    opt["date"] = pd.to_datetime(opt["date"])
    opt = opt[(opt["bid"] > 0) & (opt["ask"] > 0)].copy()
    frame = load_market_frame("2001-01-02", "2026-08-14")
    px = frame["adj_close"]
    vix = frame["vix"]
    dates = [d for d in px.index if pd.Timestamp("2023-07-03") <= d <= pd.Timestamp("2026-08-07")]

    p = Params()
    ind = prepare_indicators(frame, p)
    target = build_target_series(ind, p)

    rows = []
    navs_out = {}
    configs = [("1.5x 不对冲", False, 0.0, 0.0, 0.0)]
    for put_m in [0.90, 0.93, 0.95]:
        for hr in [1.0, 1.5]:
            configs.append((f"1.5x put-only {int(put_m*100)} x{hr}", True, put_m, 0.0, hr))
    configs.append(("1.5x collar 91/103 x1.5", True, 0.91, 1.03, 1.5))
    configs.append(("1.2x put-only 93 x1.2", True, 0.93, 0.0, 1.2))
    configs.append(("1.0x 不对冲", False, 0.0, 0.0, 0.0))
    configs.append(("1.0x put-only 93 x1.0", True, 0.93, 0.0, 1.0))
    for name, use_collar, put_m, call_m, hr in configs:
        lev_cap = 1.2 if name.startswith("1.2x") else (1.0 if name.startswith("1.0x") else 1.5)
        tgt = target.clip(upper=lev_cap) if lev_cap < 1.5 else target
        if name.startswith("1.0x"):
            tgt = pd.Series(1.0, index=target.index)
        nav, ev = run(px, tgt, dates, opt, put_m, call_m, use_collar, hr)
        st = stats(nav)
        rows.append({"config": name, **st,
                     "rolls": sum(1 for e in ev if e["type"] == "open"),
                     "net_credit_pts": round(sum(e["net"] for e in ev if e["type"] == "open"), 2)})
        navs_out[name] = nav

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "levered_collar_comparison.csv", index=False)
    pd.DataFrame(navs_out).to_csv(OUT / "levered_collar_nav.csv")
    (OUT / "levered_collar_comparison.json").write_text(json.dumps(json.loads(df.to_json(orient="records")), ensure_ascii=False, indent=2))

    pd.set_option("display.width", 240)
    print(f"窗口 {dates[0].date()} ~ {dates[-1].date()}（{len(dates)} 天，真实期权报价覆盖期）")
    print(df.to_string(index=False))
    print(f"\n输出: {OUT}/levered_collar_comparison.csv / _nav.csv")


if __name__ == "__main__":
    main()
