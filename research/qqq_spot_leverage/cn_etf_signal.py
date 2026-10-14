#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""国内纳指 ETF 单腿信号（round 25）——把 QQQ 信号翻译成"A 股今天该买多少"。

为什么单独做
------------
* 信号仍来自 QQQ / VIX / SPY / IXIC（美东收盘后可得）；
* 但 A 股 15:00 收盘时纽约还没开盘，所以你在 A 股看到的 CN ETF 当日涨跌，
  反映的是**前一夜**的纳指。因此**信号必须等到下一个 A 股交易日执行**；
* 输出直接给：目标仓位（占自有资金倍数，上限 1.5x）、对应 ETF 的**股数**（按你的资金）、
  以及信号与执行的日期对照，避免"用错日子"；
* **溢价闸门（round 29 新增）**：跨境 QDII 的市价 = 净值 ×(1+溢价)，溢价是纯噪音且高溢价
  显著预测低未来收益。因此当溢价高于阈值时，脚本会把目标杠杆强制降到 0（只拦建仓/加仓，
  减仓永不被拦），并明确告诉你"现在不该买"以及原因。

用法：
    python3 research/qqq_spot_leverage/cn_etf_signal.py --capital 100000
    python3 research/qqq_spot_leverage/cn_etf_signal.py --capital 100000 --symbol 159501
    python3 research/qqq_spot_leverage/cn_etf_signal.py --capital 100000 --symbol 513100 --gate 0.02
    python3 research/qqq_spot_leverage/cn_etf_signal.py --json
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PANIC_DIR = REPO / "data" / "public" / "etf"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))
from backtest_qqq_spot_leverage import Params, load_market_frame, prepare_indicators  # noqa: E402
from cn_data import load_panel  # noqa: E402


def vvix_spike(lookback: int = 5, threshold: float = 0.15) -> dict:
    """VVIX（波动率的波动率）急升检测。

    round 30 实测：VVIX 5 日涨 >15% 时把杠杆减半，在 513100(2020-26) 与
    159501(2022-26) 上同时改善 Sharpe 与 Calmar，且与硬止损**相互独立**
    （关掉硬止损后单独加它依然有效）。详见 test_panic_filters.py。

    注意：VVIX 是"VIX 的隐含波动率"，衡量的是波动率本身失控，属于**危机前兆**，
    和 VIX 绝对水平（已测过，无效）不是一回事。
    """
    p = PANIC_DIR / "^VVIX-1d.feather"
    if not p.exists():
        return {"available": False}
    d = pd.read_feather(p)
    dt = pd.to_datetime(d["date"])
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    else:
        dt = dt.dt.normalize()
    v = pd.Series(d["close"].to_numpy(float), index=dt).sort_index()
    v = v[~v.index.duplicated(keep="last")]
    chg = float(v.iloc[-1] / v.iloc[-1 - lookback] - 1.0)
    return {"available": True, "last": round(float(v.iloc[-1]), 2),
            "last_date": str(v.index[-1].date()),
            "chg_5d": round(chg, 4), "spike": bool(chg > threshold),
            "lookback": lookback, "threshold": threshold}

REPO = HERE.parents[1]


def normalize_symbol(symbol: str) -> str:
    """接受 159501 / 159501.SZ / 513100.SS / 513100.SH，统一成 6 位代码。"""
    return symbol.split(".")[0].strip()


def cn_prices(symbol: str) -> pd.DataFrame:
    """行情+净值+溢价（自动处理 .SS/.SH/.SZ 后缀与份额拆分连续化）。"""
    code = normalize_symbol(symbol)
    try:
        d = load_panel(code)
    except SystemExit:
        raise
    d = d.rename_axis("day")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description="CN Nasdaq-ETF single-leg signal (1.5x cap)")
    ap.add_argument("--symbol", default="159501", help="A 股纳指 ETF 代码，如 159501(嘉实) / 513100(国泰)")
    ap.add_argument("--gate", type=float, default=0.02,
                    help="溢价闸门：溢价高于此值时禁止建仓/加仓（默认 0.02=2%%；设 99 关闭）")
    ap.add_argument("--no-gate", action="store_true", help="关闭溢价闸门（仅看纯信号）")
    ap.add_argument("--vvix-half", action="store_true",
                    help="启用 VVIX 急升减半（VVIX 5日涨>15%% 时目标杠杆减半，round 30 实证有效）")
    ap.add_argument("--vvix-threshold", type=float, default=0.15)
    ap.add_argument("--capital", type=float, default=100_000.0, help="自有资金（元）")
    ap.add_argument("--cap", type=float, default=1.5, help="杠杆上限（默认 1.5x）")
    ap.add_argument("--lot", type=int, default=100, help="最小交易单位（ETF 为 100 份/手）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    p = Params(lev_full=args.cap, max_leverage=args.cap, lev_half=min(1.0, args.cap),
               recovery_lev=min(0.75, args.cap))
    frame = load_market_frame("2001-01-02", "2026-08-14")
    ind = prepare_indicators(frame, p)
    # 用引擎的静态目标（方向阶梯 + 波动率目标）；硬止损需要持仓路径，在 paper/实盘里跟踪
    from backtest_qqq_spot_leverage import base_target  # noqa: E402
    static = ind.apply(lambda r: base_target(r, p), axis=1)

    q_last = static.index[-1]
    target = float(static.loc[q_last])

    # 找"下一个 A 股交易日"
    px = cn_prices(args.symbol)
    cn_days = list(px.index)
    i = bisect.bisect_right(cn_days, q_last)
    exec_day = cn_days[i] if i < len(cn_days) else None
    last_cn = px.index[-1]
    ref_px = float(px["close"].iloc[-1])

    # ---- 溢价闸门 ----
    gate = 99.0 if args.no_gate else args.gate
    premium = float(px["premium"].iloc[-1]) if "premium" in px.columns else float("nan")
    nav = float(px["nav"].iloc[-1]) if "nav" in px.columns else float("nan")
    premium_median = float(px["premium"].median()) if "premium" in px.columns else float("nan")
    premium_p95 = float(px["premium"].quantile(0.95)) if "premium" in px.columns else float("nan")
    gate_blocked = bool(premium_median == premium_median and premium > gate)
    target_raw = target
    if gate_blocked:
        target = 0.0          # 只拦建仓/加仓；实际减仓/清仓永远允许

    # ---- VVIX 急升减半（可选，round 30）----
    vv = vvix_spike(threshold=args.vvix_threshold) if args.vvix_half else {"available": False}
    vvix_applied = False
    if vv.get("available") and vv.get("spike") and target > 0:
        target *= 0.5
        vvix_applied = True

    # 股数：目标市值 = 自有资金 × 目标杠杆
    notional = args.capital * target
    shares = int(notional / ref_px // args.lot * args.lot) if ref_px > 0 else 0
    actual_notional = shares * ref_px
    actual_lev = actual_notional / args.capital if args.capital else 0.0
    borrow = max(actual_notional - args.capital, 0.0)

    # 输入新鲜度（VIX/SPY 常落后于 QQQ）
    def last_valid(c):
        s = ind[c].dropna()
        return s.index[-1] if len(s) else None
    complete = min([d for d in (last_valid("vix"), last_valid("spy"), last_valid("ixic_close") if "ixic_close" in ind else None, q_last) if d is not None])
    stale = bool(complete < q_last)
    target_complete = float(static.loc[complete])

    payload = {
        "symbol": args.symbol,
        "signal_date": str(q_last.date()),
        "earliest_execution_date": str(exec_day.date()) if exec_day is not None else None,
        "cn_last_data_date": str(last_cn.date()),
        "premium_now": None if premium != premium else round(premium, 4),
        "premium_median": None if premium_median != premium_median else round(premium_median, 4),
        "premium_p95": None if premium_p95 != premium_p95 else round(premium_p95, 4),
        "nav": None if nav != nav else round(nav, 4),
        "premium_gate": round(gate, 4) if gate < 90 else None,
        "premium_gate_blocked": gate_blocked,
        "target_leverage_before_gate": round(target_raw, 3),
        "vvix": vv if vv.get("available") else None,
        "vvix_half_applied": vvix_applied,
        "target_leverage": round(target, 3),
        "target_leverage_complete_inputs": round(target_complete, 3),
        "stale_inputs": stale,
        "inputs_complete_upto": str(complete.date()),
        "reference_price": round(ref_px, 3),
        "capital": args.capital,
        "shares_to_hold": shares,
        "actual_notional": round(actual_notional, 2),
        "actual_leverage": round(actual_lev, 3),
        "borrow_amount": round(borrow, 2),
        "notes": [
            "A 股 T 日 15:00 收盘时纽约未开盘，CN ETF 当日涨跌反映前一夜纳指；",
            "因此本信号在 QQQ 收盘后计算，最早在【下一个 A 股交易日】执行；",
            "若 earliest_execution_date 为空，说明该 ETF 数据未覆盖到最新（需补数据）；",
            "1.5x 杠杆在国内需两融/收益互换等工具，融资成本按你的实际利率另计。",
            "溢价=收盘价/净值-1；高溢价显著预测低未来收益，故高于阈值时目标杠杆置 0。",
        ],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(f"交易所标的    : {payload['symbol']}（最新价 {payload['reference_price']}，数据到 {payload['cn_last_data_date']}）")
    print(f"信号日        : {payload['signal_date']}（美东 QQQ 收盘后）")
    print(f"最早可执行日  : {payload['earliest_execution_date']}")
    if payload["premium_now"] is not None:
        print(f"当前溢价      : {payload['premium_now']*100:+.2f}%"
              f"（历史中位 {payload['premium_median']*100:+.2f}%，P95 {payload['premium_p95']*100:+.2f}%；"
              f"净值 {payload['nav']}）")
    if payload["premium_gate_blocked"]:
        print(f"🚫 溢价闸门    : 溢价 {payload['premium_now']*100:+.2f}% > 阈值 {payload['premium_gate']*100:.1f}%"
              f" → 目标杠杆被压到 0（信号本身是 {payload['target_leverage_before_gate']}x）")
        print(f"   理由        : 溢价是纯噪音成本，高溢价买入未来收益为负；等溢价回落到阈值以下再建仓。")
    if payload.get("vvix"):
        v = payload["vvix"]
        mark = "⚠️ 急升，已减半" if payload["vvix_half_applied"] else "正常"
        print(f"VVIX          : {v['last']}（{v['last_date']}），5日 {v['chg_5d']*100:+.1f}% -> {mark}")
    print(f"目标杠杆      : {payload['target_leverage']}x")
    if payload["stale_inputs"]:
        print(f"⚠️ 输入不全   : VIX/SPY 只到 {payload['inputs_complete_upto']}；"
              f"用全部输入可得日算 = {payload['target_leverage_complete_inputs']}x")
    print(f"自有资金      : {payload['capital']:,.0f}")
    print(f"应持有        : {payload['shares_to_hold']:,} 份（市值 {payload['actual_notional']:,.0f}，"
          f"实际杠杆 {payload['actual_leverage']:.2f}x，借款 {payload['borrow_amount']:,.0f}）")


if __name__ == "__main__":
    main()
