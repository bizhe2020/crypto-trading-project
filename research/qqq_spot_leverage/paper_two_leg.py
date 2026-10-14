#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双腿 paper 跟踪器（round 17）：QQQ 阶梯腿 + BTC(MA200 过滤) 腿。

设计（与 `final/最终结论与建议.md` 第 8d 节一致）：
  * QQQ 腿：双均线阶梯，默认 1.25x 上限（overlay 全关），信号来自已验证的
    `backtest_qqq_spot_leverage.prepare_indicators/build_target_series`；
  * BTC 腿：BTC 现货，收盘 > MA200 则持有，否则空仓；BTC 7×24 交易，
    因此它的信号每天都能更新，但组合再平衡只在 QQQ 交易日进行（与回测口径一致）；
  * 组合：按目标权重（默认 Q 70% / BTC 30%）在 QQQ 交易日调仓，双边各计 1bp 成本；
  * 幂等：同一天重复运行只刷新、不重复记账；状态与决策日志落盘。

用法：
    python3 research/qqq_spot_leverage/paper_two_leg.py                # 运行
    python3 research/qqq_spot_leverage/paper_two_leg.py --reset        # 重置
    python3 research/qqq_spot_leverage/paper_two_leg.py --status       # 查看状态
    python3 research/qqq_spot_leverage/paper_two_leg.py --w-btc 0.2 --qqq-cap 1.25
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backtest_qqq_spot_leverage import (  # noqa: E402
    Params,
    QQQ_LAST_KNOWN,
    build_target_series,
    load_market_frame,
    prepare_indicators,
)

REPO = HERE.parents[1]
STATE_DIR = HERE / "state"
STATE_PATH = STATE_DIR / "two_leg_paper_state.json"
DECISIONS_PATH = STATE_DIR / "two_leg_paper_decisions.jsonl"
INITIAL_CAPITAL = 100_000.0
MIN_TRADE_DELTA = 0.02      # 目标权重变化 < 2% 不交易
COST_BPS = 1.0              # 单边成本
REBALANCE_MONTHS = {3, 6, 9, 12}   # 季度再平衡月份（与组合回测口径一致）


def btc_series() -> pd.Series:
    b = pd.read_feather(REPO / "data" / "public" / "etf" / "BTC-USD-1d.feather")
    b["d"] = pd.to_datetime(b["date"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    b = b.sort_values("d").drop_duplicates("d", keep="last").set_index("d")
    return b["close"]


def btc_signal(as_of: pd.Timestamp) -> tuple[float, pd.Series, pd.Series]:
    """返回 (目标权重: 0/1, BTC 收盘价序列, MA200)。"""
    px = btc_series()
    px = px.loc[:as_of]
    ma = px.rolling(200).mean()
    if len(px) < 2 or pd.isna(ma.iloc[-1]):
        return 0.0, px, ma
    on = 1.0 if float(px.iloc[-1]) > float(ma.iloc[-1]) else 0.0
    return on, px, ma


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "initial_capital": INITIAL_CAPITAL,
        "nav": INITIAL_CAPITAL,
        "w_qqq": 0.0,
        "lev_qqq": 0.0,
        "w_btc": 0.0,
        "last_mark_date": None,
        "last_signal_date": None,
        "days": 0,
        "trades": 0,
        "params": {},
    }


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def append_decision(rec: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with DECISIONS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run(as_of: str | None, reset: bool, status_only: bool, w_btc: float, qqq_cap: float) -> None:
    if reset:
        for p in (STATE_PATH, DECISIONS_PATH):
            if p.exists():
                p.unlink()
        print("[two-leg] 状态已重置")

    p = Params(rs_mode="", use_vix_gate=False, use_recovery=False,
               lev_full=qqq_cap, max_leverage=qqq_cap,
               lev_half=min(1.0, qqq_cap), recovery_lev=min(0.75, qqq_cap))
    end = as_of or QQQ_LAST_KNOWN
    frame = load_market_frame("2001-01-02", end)
    ind = prepare_indicators(frame, p)
    target = build_target_series(ind, p)
    ts = ind.index[-1]
    row = ind.loc[ts]
    lev_signal = float(target.loc[ts])
    btc_on, btc_px, btc_ma = btc_signal(ts)

    state = load_state()
    state["params"] = {"w_btc": w_btc, "w_qqq": round(1 - w_btc, 2), "qqq_cap": qqq_cap}
    same_day = state.get("last_signal_date") == str(ts.date())

    if status_only:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        print(f"最新 {ts.date()}: QQQ 信号 {lev_signal:.2f}x（上限 {qqq_cap}x），BTC MA200 开关 {btc_on:.0f}")
        return

    # 目标权重（按名义资金分配）
    w_btc_target = w_btc * btc_on
    w_qqq_target = (1 - w_btc) * lev_signal

    # 估值：用上一交易日的收盘价推进两条腿
    if not same_day and state["last_mark_date"]:
        prev_ts = pd.Timestamp(state["last_mark_date"])
        if prev_ts in ind.index and prev_ts != ts:
            q_ret = float(row["adj_close"]) / float(ind.loc[prev_ts, "adj_close"]) - 1.0
            btc_prev = btc_px.loc[:prev_ts]
            b_ret = (float(btc_px.loc[:ts].iloc[-1]) / float(btc_prev.iloc[-1]) - 1.0) if len(btc_prev) else 0.0
            state["nav"] *= 1.0 + state["w_qqq"] * q_ret + state["w_btc"] * b_ret
    state["last_mark_date"] = str(ts.date())

    # 调仓触发：① 信号/目标变化 >= 阈值；② 季度再平衡月（把漂移拉回目标）
    quarter_rebalance = ts.month in REBALANCE_MONTHS and state.get("last_rebalance_month") != ts.month
    turnover = 0.0
    signal_change = (abs(w_qqq_target - state["w_qqq"]) >= MIN_TRADE_DELTA
                     or abs(w_btc_target - state["w_btc"]) >= MIN_TRADE_DELTA)
    if signal_change or quarter_rebalance:
        turnover = abs(w_qqq_target - state["w_qqq"]) + abs(w_btc_target - state["w_btc"])
        state["nav"] *= 1.0 - turnover * COST_BPS / 1e4
        state["trades"] += 1
    if quarter_rebalance:
        state["last_rebalance_month"] = ts.month
    prev_wq, prev_wb = state["w_qqq"], state["w_btc"]
    state["w_qqq"], state["w_btc"] = w_qqq_target, w_btc_target
    state["lev_qqq"] = lev_signal
    if not same_day:
        state["days"] += 1
        state["last_signal_date"] = str(ts.date())

    rec = {
        "date": str(ts.date()),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "qqq_close": round(float(row["adj_close"]), 2),
        "qqq_ma50": round(float(row["ma_fast"]), 2) if pd.notna(row["ma_fast"]) else None,
        "qqq_ma200": round(float(row["ma_slow"]), 2) if pd.notna(row["ma_slow"]) else None,
        "qqq_lev_signal": lev_signal,
        "btc_close": round(float(btc_px.loc[:ts].iloc[-1]), 2),
        "btc_ma200": round(float(btc_ma.iloc[-1]), 2) if pd.notna(btc_ma.iloc[-1]) else None,
        "btc_on": btc_on,
        "w_qqq_target": round(w_qqq_target, 4),
        "w_btc_target": round(w_btc_target, 4),
        "prev_w_qqq": round(prev_wq, 4),
        "prev_w_btc": round(prev_wb, 4),
        "turnover": round(turnover, 4),
        "quarter_rebalance": bool(quarter_rebalance),
        "nav": round(float(state["nav"]), 2),
        "repeat_run": same_day,
    }
    if not same_day:
        append_decision(rec)
    save_state(state)

    print(f"[two-leg] {ts.date()}  QQQ {rec['qqq_close']} (信号 {lev_signal:.2f}x)  "
          f"BTC {rec['btc_close']} (MA200 {rec['btc_ma200']}, 开关 {btc_on:.0f})")
    if turnover > 0:
        print(f"[two-leg] 调仓：QQQ {prev_wq:.2f}→{w_qqq_target:.2f}，BTC {prev_wb:.2f}→{w_btc_target:.2f}"
              f"（换手 {turnover*100:.1f}%）")
    else:
        print(f"[two-leg] 维持：QQQ {state['w_qqq']:.2f}，BTC {state['w_btc']:.2f}")
    print(f"[two-leg] 纸面净值 {state['nav']:,.2f}"
          f"（累计 {(state['nav']/state['initial_capital']-1)*100:+.2f}%，调仓 {state['trades']} 次）")
    print(f"[two-leg] 状态 {STATE_PATH}")
    print(f"[two-leg] 日志 {DECISIONS_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-leg paper tracker: QQQ ladder + BTC MA200")
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--status", action="store_true")
    # 默认 0.20：11.7 年长样本里 Sharpe 1.41 / 回撤 −29.3%，是"收益明显改善、回撤温和"的折中
    # （0.10 回撤最小 −26.9%；0.30 Sharpe 最高 1.47 但回撤 −32.1%）
    ap.add_argument("--w-btc", type=float, default=0.20, help="BTC 腿目标权重（默认 0.20）")
    ap.add_argument("--qqq-cap", type=float, default=1.25, help="QQQ 腿杠杆上限（默认 1.25）")
    # 权重速查（2014-09~2026-05 长样本）：w=0.10 → CAGR 23.2%/MDD −26.9%/Sharpe 1.21；
    # w=0.20 → 28.7%/−29.3%/1.41；w=0.30 → 34.1%/−32.1%/1.47
    args = ap.parse_args()
    run(args.as_of, args.reset, args.status, args.w_btc, args.qqq_cap)


if __name__ == "__main__":
    main()
