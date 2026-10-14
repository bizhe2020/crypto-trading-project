#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""连续 collar 保护回测（round 9）：用真实 QQQ 期权链（firstrate 原始包，含 put 侧）。

每 21 个交易日滚动一次 1 个月期 collar：
    买 ~91% put（付 ask）+ 卖 ~103% call（收 bid）
期权到期按内在价值结算；期间按 Black-Scholes 用入场 IV 估值（便于逐日看效果）。
对比：不对冲的 QQQ、以及"仅在 VIX>=阈值 时才买保护"的条件版本。

数据：/tmp/qqq_chain_cache.csv（由 firstrate QQQ_*.txt 抽取的 38 个滚动日报价）
      窗口 2023-07-03 ~ 2026-08-07（约 3.1 年，含 2024-08 / 2025-04 / 2026-06 三次回撤）

用法: python3 research/qqq_spot_leverage/collar_backtest.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_qqq_spot_leverage import load_market_frame  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
CACHE = Path("/tmp/qqq_chain_cache.csv")
RF = 0.045
ROLL_DAYS = 21
PUT_MONEYNESS = 0.91
CALL_MONEYNESS = 1.03
TARGET_DTE = 35
VIX_TRIGGER = 0.0   # >0 时只在 VIX >= 该值时开 collar（条件保护）


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


def run_one(px, vix, dates, opt, put_m=PUT_MONEYNESS, call_m=CALL_MONEYNESS, vix_trigger=VIX_TRIGGER) -> dict:
    """跑一条 collar 路径，返回 {res, hedged, unhedged, events}。"""
    if not CACHE.exists():
        raise SystemExit("missing cache; run the extraction step first")
    opt = pd.read_csv(CACHE)
    frame = load_market_frame("2001-01-02", "2026-08-14")
    px = frame["adj_close"]
    vix = frame["vix"]

    dates = [d for d in px.index if pd.Timestamp("2023-07-03") <= d <= pd.Timestamp("2026-08-07")]
    roll_dates = dates[::ROLL_DAYS]
    opt["date"] = pd.to_datetime(opt["date"])

    # 状态：现金（期权现金流）与在手上的期权腿
    cash = 0.0
    put_leg = None   # dict(k, qty, cost, expiry, vol)
    call_leg = None
    legs = []        # 用于估值
    nav_hedge, nav_unhedged = [], []  # noqa
    prev_s = None
    events = []

    def mark_legs(leg, s, t):
        if leg is None:
            return 0.0
        k, expiry, vol, side = leg
        tau = max((expiry - t).days / 365.0, 0.0)
        if tau == 0:
            intrinsic = max(k - s, 0.0) if side == "p" else max(s - k, 0.0)
            return intrinsic * (1 if side == "p" else -1)
        if side == "p":
            return bs_put(s, k, tau, vol)
        return -bs_call(s, k, tau, vol)

    for i, t in enumerate(dates):
        s = float(px.loc[t])
        # --- 到期结算 ---
        if put_leg is not None and t >= put_leg[1]:
            k = put_leg[0]
            cash += max(k - s, 0.0)
            events.append({"date": str(t.date()), "type": "put_settle", "strike": k, "spot": round(s, 2), "payoff": round(max(k - s, 0.0), 2)})
            put_leg = None
        if call_leg is not None and t >= call_leg[1]:
            k = call_leg[0]
            cash -= max(s - k, 0.0)
            events.append({"date": str(t.date()), "type": "call_settle", "strike": k, "spot": round(s, 2), "payoff": round(-max(s - k, 0.0), 2)})
            call_leg = None
        # --- 滚动建仓 ---
        vix_now = float(vix.loc[t]) if pd.notna(vix.loc[t]) else None
        allow = (vix_trigger <= 0) or (vix_now is not None and vix_now >= vix_trigger)
        if t in roll_dates and allow and (put_leg is None or call_leg is None):
            day = opt[opt["date"] == t]
            if not day.empty:
                exps = sorted({e for e in day["expiry"] if pd.Timestamp(e) > t})
                if exps:
                    exp = min(exps, key=lambda e: abs((pd.Timestamp(e) - t).days - TARGET_DTE))
                    chain = day[day["expiry"] == exp]
                    puts = chain[chain["cp"] == "p"]
                    calls = chain[chain["cp"] == "c"]
                    if not puts.empty and not calls.empty:
                        pk = float(puts.iloc[(puts["strike"] - s * put_m).abs().argsort().iloc[0]]["strike"])
                        ck = float(calls.iloc[(calls["strike"] - s * call_m).abs().argsort().iloc[0]]["strike"])
                        pr = puts[puts["strike"] == pk].iloc[0]
                        cr = calls[calls["strike"] == ck].iloc[0]
                        put_leg = (pk, pd.Timestamp(exp), float(pr["iv"] or 0.25), "p")
                        call_leg = (ck, pd.Timestamp(exp), float(cr["iv"] or 0.25), "c")
                        cash += float(cr["bid"]) - float(pr["ask"])   # 卖 call 收 bid、买 put 付 ask
                        events.append({
                            "date": str(t.date()), "type": "open_collar", "spot": round(s, 2),
                            "put_strike": pk, "put_ask": round(float(pr["ask"]), 2),
                            "call_strike": ck, "call_bid": round(float(cr["bid"]), 2),
                            "net_credit": round(float(cr["bid"]) - float(pr["ask"]), 2),
                            "expiry": str(pd.Timestamp(exp).date()), "vix": round(float(vix.loc[t]), 1) if pd.notna(vix.loc[t]) else None,
                        })
        # --- 估值 ---
        opt_val = 0.0
        if put_leg is not None:
            opt_val += mark_legs(put_leg, s, t)
        if call_leg is not None:
            opt_val += mark_legs(call_leg, s, t)
        nav_hedge.append(s + cash + opt_val)
        nav_unhedged.append(s)
        prev_s = s

    # 归一化到起点 1.0
    hedged = pd.Series(nav_hedge, index=dates)
    unhedged = pd.Series(nav_unhedged, index=dates)
    hedged = hedged / hedged.iloc[0]
    unhedged = unhedged / unhedged.iloc[0]

    def stats(s: pd.Series) -> dict:
        years = (s.index[-1] - s.index[0]).days / 365.25
        r = s.pct_change().dropna()
        return {
            "total_pct": round(float(s.iloc[-1] - 1) * 100, 2),
            "cagr_pct": round(float(s.iloc[-1] ** (1 / years) - 1) * 100, 2),
            "mdd_pct": round(float((s / s.cummax() - 1).min()) * 100, 2),
            "vol_pct": round(float(r.std() * math.sqrt(252)) * 100, 2),
            "sharpe": round(float(r.mean() / r.std() * math.sqrt(252)), 2) if r.std() > 0 else 0.0,
        }

    res = {"window": [str(dates[0].date()), str(dates[-1].date())], "unhedged": stats(unhedged), "collar": stats(hedged),
           "n_rolls": sum(1 for e in events if e["type"] == "open_collar"),
           "total_net_credit_pct_of_spot": round(sum(e.get("net_credit", 0) for e in events if e["type"] == "open_collar") / float(px.loc[dates[0]]) * 100, 2)}
    return {"res": res, "hedged": hedged, "unhedged": unhedged, "events": events}


def sweep_main() -> None:
    """扫描 put/call 货币性与 VIX 触发条件，找保护效率前沿。"""
    import itertools

    opt = pd.read_csv(CACHE)
    opt["date"] = pd.to_datetime(opt["date"])
    frame = load_market_frame("2001-01-02", "2026-08-14")
    px = frame["adj_close"]
    vix = frame["vix"]
    dates = [d for d in px.index if pd.Timestamp("2023-07-03") <= d <= pd.Timestamp("2026-08-07")]

    rows = []
    base = None
    for put_m, call_m, trig in itertools.product(
        [0.90, 0.93], [1.03, 1.05, 1.08], [0.0, 20.0, 25.0]
    ):
        out = run_one(px, vix, dates, opt, put_m, call_m, trig)
        if base is None:
            base = out["res"]["unhedged"]
        c = out["res"]["collar"]
        rows.append({
            "put_mny": put_m, "call_mny": call_m, "vix_trigger": trig, "rolls": out["res"]["n_rolls"],
            "net_credit_pct": out["res"]["total_net_credit_pct_of_spot"],
            "cagr": c["cagr_pct"], "mdd": c["mdd_pct"], "sharpe": c["sharpe"], "vol": c["vol_pct"],
            "d_cagr_vs_unhedged": round(c["cagr_pct"] - base["cagr_pct"], 2),
            "d_mdd_vs_unhedged": round(c["mdd_pct"] - base["mdd_pct"], 2),
        })
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    df.to_csv(OUT / "collar_sweep.csv", index=False)
    (OUT / "collar_sweep.json").write_text(json.dumps({"unhedged": base, "rows": json.loads(df.to_json(orient="records"))}, ensure_ascii=False, indent=2))
    pd.set_option("display.width", 240)
    print("不对冲基准:", json.dumps(base, ensure_ascii=False))
    print()
    print("=== collar 结构扫描（按 Sharpe 排序）===")
    print(df.to_string(index=False))
    print(f"\n输出: {OUT}/collar_sweep.csv")


def single_main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    opt = pd.read_csv(CACHE)
    opt["date"] = pd.to_datetime(opt["date"])
    frame = load_market_frame("2001-01-02", "2026-08-14")
    px = frame["adj_close"]
    vix = frame["vix"]
    dates = [d for d in px.index if pd.Timestamp("2023-07-03") <= d <= pd.Timestamp("2026-08-07")]
    out = run_one(px, vix, dates, opt)
    res, events = out["res"], out["events"]
    pd.DataFrame({"hedged": out["hedged"], "unhedged": out["unhedged"]}).to_csv(OUT / "collar_backtest_nav.csv")
    pd.DataFrame(events).to_csv(OUT / "collar_backtest_events.csv", index=False)
    (OUT / "collar_backtest.json").write_text(json.dumps({"summary": res, "events": events}, ensure_ascii=False, indent=2))
    print("=== 连续 1 月期 collar vs 不对冲（真实期权链）===")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    ev = pd.DataFrame(out["events"])
    if not ev.empty:
        print("\n最近 8 次滚动：")
        print(ev[ev["type"] == "open_collar"].tail(8).to_string(index=False))
    print(f"\n输出: {OUT}/collar_backtest.json")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    if args.sweep:
        sweep_main()
    else:
        single_main()


if __name__ == "__main__":
    main()
