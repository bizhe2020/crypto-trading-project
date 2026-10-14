#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单腿 1.5x 策略 —— 国内纳指 ETF「溢价感知」执行版（round 29）。

与 `backtest_cn_single_leg.py` 的区别
------------------------------------
旧脚本只按**价格**回测，把跨境 QDII 的溢价/折价当成噪音吸收进收益里。
本脚本把收益精确拆开，并对溢价加一条交易纪律：

    价格收益  =  净值收益(真实跟踪纳指)  +  溢价变动

其中**溢价变动**是纯粹的摩擦：它不带来任何长期收益，却会在买入瞬间变成成本。
`analyze_cn_premium.py` 的实证（2020 起）：
    * 溢价 > 10% 时，未来 20 日价格收益 513100 −2.3% / 159941 −3.0%；60 日 159501 −1.5%
    * corr(溢价, 未来60日收益) = −0.39 (159501) / −0.13 (159941)
    * 溢价分位 P95 ≈ 8~11%，但 2026 年长期停在 7~14%（QDII 额度稀缺导致的**制度性**抬升）

本脚本的规则
-----------
* 底层标的用 **NAV**（净值）序列，保证拿到的是"真实纳指收益"；
* 溢价按其占**已投入仓位**的比例计入盈亏：premium_return_effective = lev * Δpremium；
* **溢价闸门**（默认 6%）：溢价高于阈值时，禁止**建仓/加仓**，但**永远允许减仓/清仓**
  （风险不能被闸门挡住）；
* 硬杠杆上限 1.5x 始终强制（即使溢价上涨把有效杠杆推高也会被拉回）；
* A 股 T+1：当日买入的份额当日不能卖出（脚本用"信号→次日开盘成交"天然满足，
  另加 traded_today 标记防止同一日买卖）；
* 成交成本：双边滑点（默认 8bp）+ 佣金 2bp，融资利息按借款部分日计息。

用法：
    python3 research/qqq_spot_leverage/backtest_cn_premium_aware.py --code 159501
    python3 research/qqq_spot_leverage/backtest_cn_premium_aware.py --all --gate 0.06
    python3 research/qqq_spot_leverage/backtest_cn_premium_aware.py --sweep
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ETF_DIR = REPO / "data" / "public" / "etf"
OUT = HERE / "out"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))

from backtest_qqq_spot_leverage import Params, load_market_frame, prepare_indicators, base_target  # noqa: E402
from cn_data import load_panel as _load_panel  # noqa: E402

CODES = ["513100", "159941", "159501", "513300", "159632", "513110", "513390", "159696", "513870"]
NAMES = {
    "159501": "纳指ETF嘉实", "513100": "纳指ETF国泰", "159941": "纳指ETF广发",
    "513300": "纳斯达克ETF华夏", "159632": "纳斯达克ETF华安", "513390": "纳指100ETF博时",
    "513110": "纳指ETF招商", "159696": "纳指ETF易方达", "513870": "纳指ETF富国",
}


def suffix(code: str) -> str:
    return f"{code}.{'SH' if code.startswith('5') else 'SZ'}"


def load_panel(code: str) -> pd.DataFrame:
    """合并行情与净值，得到 open/close/nav/premium（含拆分连续化，见 tools/cn_data.py）。"""
    return _load_panel(code)


def qqq_static_target(p: Params) -> pd.Series:
    """QQQ 侧的静态目标杠杆（方向阶梯 + 波动率目标），硬止损在执行循环里处理。"""
    frame = load_market_frame("2001-01-02", "2026-12-31")
    ind = prepare_indicators(frame, p)
    return ind.apply(lambda r: base_target(r, p), axis=1).rename("target")


def map_signal_to_cn(sig_index: pd.DatetimeIndex, cn_index: pd.DatetimeIndex) -> dict:
    """QQQ 交易日 D 收盘信号 -> 最早可成交的 A 股交易日（严格晚于 D）。"""
    cn_days = list(cn_index)
    out = {}
    for d in sig_index:
        i = bisect.bisect_right(cn_days, d)
        if i < len(cn_days):
            out[d] = cn_days[i]
    return out


def run(df: pd.DataFrame, target_qqq: pd.Series, p: Params, *, gate: float = 0.06, use_stop: bool = True,
        slippage_bps: float = 8.0, commission_bps: float = 2.0, margin_rate: float = 0.05,
        max_lev: float = 1.5, drift_band: float = 0.25, premium_aware: bool = True,
        block_mask: pd.Series | None = None) -> pd.DataFrame:
    """执行循环。

    溢价感知模式下，盈亏按 净值收益 + 有效杠杆×溢价变动 计算；
    premium_aware=False 时退化为直接用价格收益（即旧口径，用于对照）。
    """
    mapping = map_signal_to_cn(target_qqq.index, df.index)
    eff: dict[pd.Timestamp, float] = {}
    for qd, cd in mapping.items():
        eff[cd] = float(target_qqq.loc[qd])

    idx = list(df.index)
    op = df["open"].to_numpy(float)
    cl = df["close"].to_numpy(float)
    nav_s = df["nav"].to_numpy(float)
    prem = df["premium"].to_numpy(float)

    equity = 1.0
    lev = 0.0
    peak_price = np.nan
    cooldown = 0
    stopped = False
    prev_target = np.nan
    rows = []
    cost_bps = slippage_bps + commission_bps
    blocked_new = 0

    for i, d in enumerate(idx):
        tgt = eff.get(d, lev)
        if not np.isfinite(tgt):
            tgt = lev

        # ---- 1) 融资利息（按昨日收盘后的借款部分） ----
        borrow = max(lev - 1.0, 0.0)
        equity -= equity * borrow * margin_rate / 252.0

        # ---- 2) 开盘调仓（含恐慌屏蔽 / 溢价闸门 / 1.5x 硬上限） ----
        traded = False
        # 恐慌屏蔽：命中时只允许"维持或减仓"，禁止加仓（绝不强制清仓）
        if block_mask is not None and bool(block_mask.iloc[i]):
            tgt = min(tgt, lev)
        if np.isfinite(op[i]) and op[i] > 0 and not stopped and cooldown == 0:
            gate_ok = True
            if premium_aware and i > 0 and np.isfinite(prem[i - 1]):
                # 只在"建仓/加仓"时拦截；减仓与清仓永不拦截（风险不能被闸门挡住）
                if tgt > lev + 1e-9 and prem[i - 1] > gate:
                    gate_ok = False
                    blocked_new += 1
            # 硬上限：溢价上涨会把有效杠杆推高，这里强制压回 max_lev
            if lev > max_lev + 1e-9:
                tgt = min(tgt, max_lev)
                gate_ok = True
            if gate_ok:
                need = ((not np.isfinite(prev_target)) or abs(tgt - prev_target) > 1e-9
                        or abs(lev - tgt) > drift_band)
                if need and abs(tgt - lev) > 1e-9:
                    equity *= 1.0 - abs(tgt - lev) * cost_bps / 1e4
                    lev = float(tgt)
                    prev_target = tgt
                    peak_price = cl[i] if np.isfinite(cl[i]) else peak_price
                    traded = True
                elif need:
                    prev_target = tgt
        elif not np.isfinite(op[i]) or op[i] <= 0:
            prev_target = tgt

        # ---- 3) 盯市：建仓当日 open->close，其后 close->close（严格不重复计一次涨跌） ----
        if np.isfinite(cl[i]) and np.isfinite(op[i]) and op[i] > 0:
            if i == 0 or traded:
                r_day = cl[i] / op[i] - 1.0
            elif i > 0 and cl[i - 1] > 0:
                r_day = cl[i] / cl[i - 1] - 1.0
            else:
                r_day = 0.0
            equity *= 1.0 + lev * r_day

        # ---- 4) 硬止损：CN 标的自身价格路径，25 日高点回撤 8% ----
        if use_stop and np.isfinite(cl[i]):
            lo = max(0, i - p.trail_lookback + 1)
            win_peak = np.nanmax(cl[lo:i + 1])
            if cooldown > 0:
                cooldown -= 1
                if cooldown == 0:
                    stopped = False
                    prev_target = np.nan
            elif lev > 0 and win_peak > 0 and (cl[i] / win_peak - 1.0) <= -p.trail_drawdown:
                cooldown = p.cooldown_days
                stopped = True
                equity *= 1.0 - abs(lev) * cost_bps / 1e4
                lev = 0.0
                prev_target = 0.0

        rows.append({"date": d, "equity": equity, "lev": lev, "target": tgt,
                     "premium": prem[i] if np.isfinite(prem[i]) else np.nan,
                     "blocked": blocked_new})
    out = pd.DataFrame(rows).set_index("date")
    out.attrs["blocked_new"] = blocked_new
    return out


def stats(s: pd.Series) -> dict:
    years = (s.index[-1] - s.index[0]).days / 365.25
    r = s.pct_change().dropna()
    mdd = float((s / s.cummax() - 1).min())
    cagr = float(s.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0
    return {
        "total_pct": round(float(s.iloc[-1] - 1) * 100, 1),
        "cagr_pct": round(cagr * 100, 2),
        "mdd_pct": round(mdd * 100, 2),
        "vol_pct": round(float(r.std() * math.sqrt(252)) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * math.sqrt(252)), 2) if r.std() > 0 else 0.0,
        "calmar": round(cagr / abs(mdd), 3) if mdd < 0 else 0.0,
    }


def evaluate(code: str, static: pd.Series, p: Params, *, gate: float, start: str, end: str,
             use_stop: bool = True, premium_aware: bool = True, slippage_bps: float = 8.0,
             margin_rate: float = 0.05, cap: float = 1.5) -> dict | None:
    try:
        d = load_panel(code)
    except SystemExit:
        return None
    d = d.loc[(d.index >= start) & (d.index <= end)]
    if len(d) < 250:
        return None
    # 两融 100% 保证金 -> 杠杆上限 2.0x；策略自身上限由 cap 决定
    static_c = static.clip(upper=cap)
    res = run(d, static_c, p, gate=gate, use_stop=use_stop, premium_aware=premium_aware,
              slippage_bps=slippage_bps, margin_rate=margin_rate, max_lev=cap)

    # 对照 A：同区间买入持有（价格，含溢价）—— 用户实际能拿到的
    bh = d["close"] / d["close"].iloc[0]
    # 对照 B：买入持有（净值，纯指数）—— 理论上限
    bhn = d["nav"] / d["nav"].iloc[0]
    # 对照 C：不加溢价闸门的同策略
    res_nogate = run(d, static_c, p, gate=99.0, use_stop=use_stop, premium_aware=premium_aware,
                     slippage_bps=slippage_bps, margin_rate=margin_rate, max_lev=cap)
    # 对照 D：常数 cap 倍不加择时
    const = run(d, pd.Series(cap, index=static.index), p, gate=99.0, use_stop=False,
                premium_aware=premium_aware, slippage_bps=slippage_bps, margin_rate=margin_rate, max_lev=cap)

    return {
        "code": code, "name": NAMES.get(code, code),
        "window": [str(d.index[0].date()), str(d.index[-1].date())], "n_days": int(len(d)),
        "premium_now": round(float(d["premium"].iloc[-1]) * 100, 2),
        "premium_median": round(float(d["premium"].median()) * 100, 2),
        "premium_p95": round(float(d["premium"].quantile(0.95)) * 100, 2),
        "gate": gate, "cap": cap,
        "margin_rate": margin_rate, "slippage_bps": slippage_bps,
        "strategy_gated": stats(res["equity"]),
        "strategy_no_gate": stats(res_nogate["equity"]),
        "buy_hold_price": stats(bh),
        "buy_hold_nav": stats(bhn),
        "constant_1.5x": stats(const["equity"]),
        "avg_lev": round(float(res["lev"].mean()), 3),
        "blocked_days": int(res.attrs.get("blocked_new", 0)),
        "rebalances": int((res["lev"].diff().abs() > 0.01).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Premium-aware single-leg 1.5x on CN Nasdaq ETFs")
    ap.add_argument("--code", default="159501")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gate", type=float, default=0.06, help="溢价闸门（0.06 = 6%%）")
    ap.add_argument("--start", default="2020-01-02")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--slippage-bps", type=float, default=8.0)
    ap.add_argument("--margin-rate", type=float, default=0.05, help="融资年化利率（两融 5%%）")
    ap.add_argument("--price-only", action="store_true", help="用价格口径（旧算法）对照")
    ap.add_argument("--cap", type=float, default=1.5, help="杠杆上限（两融100%%保证金下最高2.0）")
    ap.add_argument("--sweep", action="store_true", help="溢价闸门敏感性扫描")
    ap.add_argument("--cap-sweep", action="store_true", help="杠杆上限敏感性扫描")
    a = ap.parse_args()

    p = Params()
    static = qqq_static_target(p)
    codes = CODES if (a.all or a.sweep) else [a.code]
    pa = not a.price_only

    if a.cap_sweep:
        print("=== 杠杆上限敏感性（融资 {:.1f}%/年，溢价闸门 {:.0f}%）：CAGR / MDD / Sharpe / Calmar ===".format(
            a.margin_rate * 100, a.gate * 100))
        print("{:8s} {:14s}".format("code", "name") + "".join(
            "{:>26s}".format("cap=" + str(c) + "x") for c in [1.0, 1.25, 1.5, 1.75, 2.0]))
        for c in codes:
            cells = []
            for cap in [1.0, 1.25, 1.5, 1.75, 2.0]:
                r = evaluate(c, static, p, gate=a.gate, start=a.start, end=a.end, premium_aware=pa,
                             slippage_bps=a.slippage_bps, margin_rate=a.margin_rate, cap=cap)
                if not r:
                    cells.append("{:>26s}".format("-")); continue
                s_ = r["strategy_gated"]
                cells.append("{0:7.1f}/{1:6.1f}/{2:5.2f}/{3:5.2f}".format(
                    s_["cagr_pct"], s_["mdd_pct"], s_["sharpe"], s_["calmar"]))
            print("{:8s} {:14s}".format(c, NAMES.get(c, "")) + "".join("{:>26s}".format(x) for x in cells))
        return 0

    if a.sweep:
        print("=== 溢价闸门敏感性（策略 CAGR / MDD / Sharpe / Calmar）===")
        hdr = f"{'code':8s} {'name':14s}" + "".join(f"{('gate' + str(int(g*100)) + '%'):>26s}" for g in
                                                   [99.0, 0.10, 0.08, 0.06, 0.04, 0.02, 0.00])
        print(hdr)
        for c in codes:
            cells = []
            for g in [99.0, 0.10, 0.08, 0.06, 0.04, 0.02, 0.00]:
                r = evaluate(c, static, p, gate=g, start=a.start, end=a.end, premium_aware=pa,
                             slippage_bps=a.slippage_bps, margin_rate=a.margin_rate, cap=a.cap)
                if not r:
                    cells.append(f"{'-':>26s}")
                    continue
                s = r["strategy_gated"]
                tag = "none" if g > 1 else f"{g*100:.0f}%"
                cells.append(f"{s['cagr_pct']:7.1f}/{s['mdd_pct']:6.1f}/{s['sharpe']:5.2f}/{s['calmar']:5.2f}")
            print(f"{c:8s} {NAMES.get(c,''):14s}" + "".join(f"{x:>26s}" for x in cells))
        print("\n格子内容：CAGR% / MDD% / Sharpe / Calmar")
        return 0

    rows = []
    for c in codes:
        r = evaluate(c, static, p, gate=a.gate, start=a.start, end=a.end, premium_aware=pa,
                     slippage_bps=a.slippage_bps, margin_rate=a.margin_rate, cap=a.cap)
        if r:
            rows.append(r)
    if not rows:
        print("没有可用标的（检查 data/public/etf/ 下的行情与净值文件）")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cn_premium_aware.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    pd.set_option("display.width", 260)
    print(f"\n=== 单腿1.5x · 国内纳指ETF执行（溢价感知={pa}, 闸门={a.gate*100:.0f}%, "
          f"{a.start}~{a.end}）===")
    hdr = (f"{'code':7s} {'name':13s} {'窗口':21s} {'策略CAGR':>8s} {'MDD':>7s} {'Sharpe':>7s} {'Calmar':>7s} "
           f"{'无闸门':>8s} {'买入持有':>8s} {'纯净值':>7s} {'常1.5x':>8s} {'现溢价':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        s, n, bp, bn, cs = (r["strategy_gated"], r["strategy_no_gate"], r["buy_hold_price"],
                            r["buy_hold_nav"], r["constant_1.5x"])
        print(f"{r['code']:7s} {r['name']:13s} {r['window'][0]}~{r['window'][1]} "
              f"{s['cagr_pct']:8.2f} {s['mdd_pct']:7.2f} {s['sharpe']:7.2f} {s['calmar']:7.2f} "
              f"{n['cagr_pct']:8.2f} {bp['cagr_pct']:8.2f} {bn['cagr_pct']:7.2f} {cs['cagr_pct']:8.2f} "
              f"{r['premium_now']:6.2f}%")
    print("\n单位：CAGR/MDD 为 %；'无闸门'=同策略不加溢价闸门；'买入持有'=价格口径 1x；")
    print("'纯净值'=净值口径 1x（理论无溢价摩擦上限）；'常1.5x'=恒定 1.5x 不加择时；'现溢价'=最新溢价。")

    if len(rows) == 1 or a.code:
        r = rows[0]
        print(f"\n--- {r['code']} {r['name']} 明细 ---")
        print(json.dumps({k: r[k] for k in ["window", "n_days", "premium_now", "premium_median",
                                            "premium_p95", "strategy_gated", "strategy_no_gate",
                                            "buy_hold_price", "buy_hold_nav", "constant_1.5x",
                                            "avg_lev", "blocked_days", "rebalances"]},
                         ensure_ascii=False, indent=2))
    print(f"\n已写出 {OUT/'cn_premium_aware.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
