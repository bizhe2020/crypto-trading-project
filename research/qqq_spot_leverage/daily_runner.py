#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日运行器（round 18）：数据新鲜度检查 + 双腿 paper 信号 + 决策落盘。

为什么需要它
------------
`paper_two_leg.py` 只看"数据最后一行"，如果某个输入（VIX/SPY/BTC）比 QQQ 落后若干天，
它会**静默**用旧值算信号。本运行器显式检查每个输入的最后有效日期，超过阈值就标 STALE，
并在状态里记录，避免"过期数据驱动交易"这种低级错误（这类问题我在 round 7/15 都遇到过）。

输出：
  * 控制台摘要（每个输入的新鲜度 + 双腿目标权重）
  * state/two_leg_daily_status.jsonl（每日一行，便于事后对账）
  * 复用 paper_two_leg.py 的状态与决策日志

用法：
    python3 research/qqq_spot_leverage/daily_runner.py
    python3 research/qqq_spot_leverage/daily_runner.py --max-stale-days 5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backtest_qqq_spot_leverage import QQQ_LAST_KNOWN, load_market_frame  # noqa: E402
from paper_two_leg import btc_series  # noqa: E402

STATE_DIR = HERE / "state"
STATUS_PATH = STATE_DIR / "two_leg_daily_status.jsonl"


def trading_days_between(a: pd.Timestamp, b: pd.Timestamp, index: pd.DatetimeIndex) -> int:
    return int(((index > a) & (index <= b)).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily runner with data-freshness guard")
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--max-stale-days", type=int, default=3, help="允许落后的交易日数，超过则标 STALE")
    ap.add_argument("--w-btc", type=float, default=0.30)
    ap.add_argument("--qqq-cap", type=float, default=1.25)
    ap.add_argument("--no-paper", action="store_true", help="只做新鲜度检查，不跑 paper")
    ap.add_argument("--strict", action="store_true", help="任一输入过期则退出码非 0（供自动化阻断）")
    args = ap.parse_args()

    end = args.as_of or QQQ_LAST_KNOWN
    frame = load_market_frame("2001-01-02", end)
    idx = frame.index
    last_q = idx[-1]
    btc = btc_series().loc[:last_q]

    inputs = {
        "QQQ": (frame["adj_close"].dropna().index[-1], "QQQ 收盘"),
        "VIX": (frame["vix"].dropna().index[-1] if frame["vix"].notna().any() else None, "VIX 收盘"),
        "VIX3M": (frame["vix3m"].dropna().index[-1] if frame["vix3m"].notna().any() else None, "VIX3M 收盘"),
        "SPY": (frame["spy"].dropna().index[-1] if frame["spy"].notna().any() else None, "SPY 收盘（RS/相对强弱用）"),
        "BTC": (btc.index[-1] if len(btc) else None, "BTC 收盘（MA200 开关用）"),
    }

    rows = []
    any_stale = False
    for name, (last, desc) in inputs.items():
        if last is None:
            lag = None
            stale = True
        else:
            lag = trading_days_between(pd.Timestamp(last), last_q, idx)
            stale = lag > args.max_stale_days
        any_stale = any_stale or stale
        rows.append({"input": name, "desc": desc, "last_valid": str(last.date()) if last is not None else None,
                     "lag_trading_days": lag, "stale": bool(stale)})

    print(f"=== 数据新鲜度（基准日 {last_q.date()}，阈值 {args.max_stale_days} 个交易日）===")
    for r in rows:
        flag = "STALE" if r["stale"] else "ok"
        print(f"  {r['input']:<6} 最后 {str(r['last_valid']):<12} 落后 {str(r['lag_trading_days']):>3} 交易日  [{flag}]  {r['desc']}")
    if any_stale:
        print("  ⚠️ 有输入过期：信号仍会生成，但请在状态文件里核对后再交易。")
        if args.strict:
            print("  ⛔ --strict 已开启：退出码 2（不视为可交易日）")
    print()

    paper_out = None
    if not args.no_paper:
        cmd = [sys.executable, "-W", "ignore", str(HERE / "paper_two_leg.py"),
               "--as-of", str(last_q.date()), "--w-btc", str(args.w_btc), "--qqq-cap", str(args.qqq_cap)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        paper_out = proc.stdout.strip()
        print(paper_out)

    # 读回 paper 状态
    state_path = STATE_DIR / "two_leg_paper_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    rec = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "as_of": str(last_q.date()),
        "max_stale_days": args.max_stale_days,
        "inputs": rows,
        "any_stale": bool(any_stale),
        "paper": {k: state.get(k) for k in ("nav", "w_qqq", "w_btc", "lev_qqq", "trades", "days")},
        "params": {"w_btc": args.w_btc, "qqq_cap": args.qqq_cap},
        "paper_stdout_tail": (paper_out or "").splitlines()[-3:] if paper_out else None,
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with STATUS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n状态已追加: {STATUS_PATH}")
    if any_stale and args.strict:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
