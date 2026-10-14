#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QQQ 现货杠杆策略 — paper 跟踪运行器。

设计原则（避免"回测一套、实盘一套"的漂移）：
  * 信号只来自 backtest_qqq_spot_leverage.py 的 prepare_indicators / build_target_series；
  * 状态文件记录持仓、净值、历史决策，每次运行幂等（同一天重复运行不会重复记账）；
  * 每日决策输出到 JSONL，便于事后与回测逐日对账。

流程（每个交易日收盘后运行一次）：
  1. 读入最新行情，算出当日 target；
  2. 用"上一交易日收盘价"给已有持仓 mark-to-market，更新净值；
  3. 若 target 与当前杠杆差异 >= 阈值，记录一笔调仓（按当日收盘价成交的纸面价）；
  4. 写回状态 + 追加决策日志。

用法：
    python3 research/qqq_spot_leverage/paper_qqq_spot_leverage.py            # 运行并打印
    python3 research/qqq_spot_leverage/paper_qqq_spot_leverage.py --reset    # 清空状态重新开始
    python3 research/qqq_spot_leverage/paper_qqq_spot_leverage.py --status   # 只看当前状态
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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

STATE_DIR = HERE / "state"
STATE_PATH = STATE_DIR / "qqq_spot_leverage_paper_state.json"
DECISIONS_PATH = STATE_DIR / "qqq_spot_leverage_paper_decisions.jsonl"
INITIAL_CAPITAL = 100_000.0  # 单腿 1.5x 上限 + 波动率目标 22%（默认 Params）
MIN_TRADE_DELTA = 0.25  # 与回测的 drift_band / 档位吸附一致


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "initial_capital": INITIAL_CAPITAL,
        "nav": INITIAL_CAPITAL,
        "leverage": 0.0,
        "last_mark_date": None,
        "last_signal_date": None,
        "days": 0,
        "trades": 0,
    }


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def append_decision(record: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with DECISIONS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_frame(as_of: str | None) -> tuple[pd.DataFrame, Params]:
    p = Params()
    frame = load_market_frame("2001-01-02", as_of or QQQ_LAST_KNOWN)
    if as_of:
        frame = frame.loc[frame.index <= pd.Timestamp(as_of)]
    if frame.empty:
        raise SystemExit("no market data")
    ind = prepare_indicators(frame, p)
    return ind, p


def run(as_of: str | None, reset: bool, status_only: bool) -> None:
    if reset:
        if STATE_PATH.exists():
            STATE_PATH.unlink()
        if DECISIONS_PATH.exists():
            DECISIONS_PATH.unlink()
        print("[paper] 状态已重置")

    ind, p = build_frame(as_of)
    target = build_target_series(ind, p)
    ts = ind.index[-1]
    row = ind.loc[ts]
    signal = float(target.loc[ts])

    state = load_state()

    if status_only:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        print(f"最新行情 {ts.date()}：目标杠杆 {signal:.2f}x")
        return

    # ---- 0) 幂等：同一天重复运行只更新不重复记账 ----
    same_day = state.get("last_signal_date") == str(ts.date())
    if same_day:
        print(f"[paper] {ts.date()} 已处理过，本次只刷新行情/信号，不重复记账")

    # ---- 1) 用最新收盘价给已有持仓 mark-to-market ----
    if (not same_day) and state["last_mark_date"] and state["leverage"] > 0:
        prev_ts = pd.Timestamp(state["last_mark_date"])
        if prev_ts in ind.index and prev_ts != ts:
            prev_close = float(ind.loc[prev_ts, "adj_close"])
            cur_close = float(row["adj_close"])
            state["nav"] *= 1.0 + state["leverage"] * (cur_close / prev_close - 1.0)
    if not same_day:
        state["last_mark_date"] = str(ts.date())

    # ---- 2) 判断是否需要调仓 ----
    current = float(state["leverage"])
    trade = abs(signal - current) >= MIN_TRADE_DELTA or (signal == 0.0 and current != 0.0)
    traded = 0.0
    if trade:
        traded = signal - current
        state["leverage"] = signal
        state["trades"] += 1

    if not same_day:
        state["days"] += 1
        state["last_signal_date"] = str(ts.date())

    record = {
        "date": str(ts.date()),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "repeat_run": bool(same_day),
        "qqq_close": round(float(row["adj_close"]), 2),
        "ma_fast": round(float(row["ma_fast"]), 2) if pd.notna(row["ma_fast"]) else None,
        "ma_slow": round(float(row["ma_slow"]), 2) if pd.notna(row["ma_slow"]) else None,
        "rs20": round(float(row["rs20"]), 5) if pd.notna(row.get("rs20")) else None,
        "vix": round(float(row["vix"]), 2) if pd.notna(row["vix"]) else None,
        "target_leverage": signal,
        "previous_leverage": current,
        "traded_leverage": traded,
        "traded": bool(trade),
        "nav": round(float(state["nav"]), 2),
        "params": asdict(p),
    }
    if not same_day:
        append_decision(record)
    save_state(state)

    print(f"[paper] {ts.date()}  QQQ {row['adj_close']:.2f}  "
          f"MA{p.fast_window} {row['ma_fast']:.2f}  MA{p.slow_window} {row['ma_slow']:.2f}  "
          f"RS20 {record['rs20']}  VIX {record['vix']}")
    if trade:
        print(f"[paper] 调仓：{current:.2f}x → {signal:.2f}x（{'加' if traded > 0 else '减'} {abs(traded):.2f}）")
    else:
        print(f"[paper] 维持 {current:.2f}x（目标 {signal:.2f}x，差异 < {MIN_TRADE_DELTA}）")
    print(f"[paper] 纸面净值 {state['nav']:,.2f}（初始 {state['initial_capital']:,.0f}，"
          f"累计 {(state['nav'] / state['initial_capital'] - 1) * 100:+.2f}%，调仓 {state['trades']} 次）")
    print(f"[paper] 状态: {STATE_PATH}")
    print(f"[paper] 日志: {DECISIONS_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description="QQQ spot leverage paper runner")
    ap.add_argument("--as-of", default=None, help="截止日期（默认数据最后一行）")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    run(args.as_of, args.reset, args.status)


if __name__ == "__main__":
    main()
