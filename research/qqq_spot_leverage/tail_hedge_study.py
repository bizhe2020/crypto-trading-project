#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""尾部保护研究（round 9，修正版）：用 firstrate 原始链的真实 put/call 报价评估 collar。

数据：~/projects/期权量化/data/firstrate/<quarter>_option_chain_*.zip 里的 QQQ_*.txt
      列：trade_date, strike, expiry, c/p, last, bid, ask, iv, delta, volume, oi, ...
      （qqq_call_daily.csv 只保留了 call 侧；put 侧只在原始包里）

方法：对每个回撤事件，在起点前一日选"约 1 个月后到期"的期权链，构建
      买 91% put / 卖 103% call 的 zero-cost-ish collar（按真实 ask 买、bid 卖），
      计算成本、谷底收益、以及对上行（反弹到事件结束日）的封顶代价。
对照：纯多头（不对冲）。

用法: python3 research/qqq_spot_leverage/tail_hedge_study.py
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_qqq_spot_leverage import load_market_frame  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
CHAIN_DIR = Path("/tmp/qqq_chain")
FR_DIR = Path.home() / "projects" / "期权量化" / "data" / "firstrate"

EVENTS = [
    ("E1 2024-07~10 套息平仓", "2024-07-11", "2024-08-05", "2024-10-30"),
    ("E2 2025-03~05 关税冲击", "2025-03-25", "2025-04-08", "2025-05-12"),
    ("E3 2026-06~08 调整", "2026-06-03", "2026-07-30", "2026-08-07"),
]
TARGET_DTE = 35


def chain_files() -> list[Path]:
    return sorted(CHAIN_DIR.glob("QQQ_*_option_chain.txt"))


def load_event_quotes(date: str) -> dict[tuple[str, str], list[dict]]:
    """读取该交易日的全部 put/call 报价（按 expiry 分组）。"""
    byexp: dict[str, list[dict]] = defaultdict(list)
    for path in chain_files():
        with path.open() as fh:
            for line in fh:
                p = line.rstrip("\n").split(",")
                if len(p) < 9 or p[0] != date:
                    continue
                byexp[p[2]].append(
                    {
                        "strike": float(p[1]),
                        "cp": p[3],
                        "last": float(p[4] or 0),
                        "bid": float(p[5] or 0),
                        "ask": float(p[6] or 0),
                        "iv": float(p[7] or 0),
                        "delta": float(p[8] or 0),
                        "oi": float(p[10] or 0) if len(p) > 10 else 0.0,
                    }
                )
    return byexp


def pick_expiry(byexp: dict[str, list[dict]], trade_date: str, target_dte: int) -> str | None:
    td = pd.Timestamp(trade_date)
    exps = [e for e in byexp if e > trade_date]
    if not exps:
        return None
    return min(exps, key=lambda e: abs((pd.Timestamp(e) - td).days - target_dte))


def quote_at(rows: list[dict], strike: float, cp: str) -> dict | None:
    exact = [r for r in rows if r["cp"] == cp and abs(r["strike"] - strike) < 1e-9]
    if exact:
        return exact[0]
    same = [r for r in rows if r["cp"] == cp and r["ask"] > 0]
    if not same:
        return None
    return min(same, key=lambda r: abs(r["strike"] - strike))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not chain_files():
        # 自动解压
        for z in sorted(FR_DIR.glob("*.zip")):
            q = z.name.split("_option_chain")[0]
            import subprocess

            subprocess.run(["unzip", "-o", "-j", str(z), f"QQQ_{q}_option_chain.txt", "-d", str(CHAIN_DIR)],
                           capture_output=True, check=False)
    frame = load_market_frame("2001-01-02", "2026-08-14")
    spot = frame["adj_close"]

    rows_out = []
    for name, entry, trough, end in EVENTS:
        byexp = load_event_quotes(entry)
        if not byexp:
            print(f"[skip] {name}: 无 {entry} 原始链（可能缺该季度包）")
            continue
        exp = pick_expiry(byexp, entry, TARGET_DTE)
        chain = byexp.get(exp, [])
        s0 = float(spot.loc[pd.Timestamp(entry)])
        s_tr = float(spot.loc[pd.Timestamp(trough)]) if pd.Timestamp(trough) in spot.index else None
        s_end = float(spot.loc[pd.Timestamp(end)]) if pd.Timestamp(end) in spot.index else None

        put_k, call_k = round(s0 * 0.91), round(s0 * 1.03)
        put = quote_at(chain, put_k, "p")
        call = quote_at(chain, call_k, "c")
        if put is None or call is None:
            print(f"[skip] {name}: 缺 {put_k}P/{call_k}C 报价（exp {exp}）")
            continue

        put_cost = put["ask"]          # 买 put 付 ask
        call_credit = call["bid"]      # 卖 call 收 bid
        net_cost = put_cost - call_credit

        def collar_pnl(sT: float) -> float:
            return max(put["strike"] - sT, 0.0) - max(sT - call["strike"], 0.0) - net_cost

        rows_out.append(
            {
                "event": name,
                "entry": entry,
                "expiry": exp,
                "dte": (pd.Timestamp(exp) - pd.Timestamp(entry)).days,
                "spot_entry": round(s0, 2),
                "put_strike": put["strike"],
                "put_ask": round(put_cost, 2),
                "put_iv": round(put["iv"], 4),
                "put_delta": round(put["delta"], 3),
                "call_strike": call["strike"],
                "call_bid": round(call_credit, 2),
                "call_iv": round(call["iv"], 4),
                "call_delta": round(call["delta"], 3),
                "collar_net_cost_pct": round(net_cost / s0 * 100, 3),
                "unhedged_at_trough_pct": round((s_tr / s0 - 1) * 100, 2) if s_tr else None,
                "collar_pnl_at_trough_pct": round(collar_pnl(s_tr) / s0 * 100, 2) if s_tr else None,
                "hedged_at_trough_pct": round(((s_tr + collar_pnl(s_tr)) / s0 - 1) * 100, 2) if s_tr else None,
                "unhedged_at_end_pct": round((s_end / s0 - 1) * 100, 2) if s_end else None,
                "hedged_at_end_pct": round(((s_end + collar_pnl(s_end)) / s0 - 1) * 100, 2) if s_end else None,
            }
        )

    df = pd.DataFrame(rows_out)
    df.to_csv(OUT / "tail_hedge_events.csv", index=False)
    (OUT / "tail_hedge_events.json").write_text(json.dumps(json.loads(df.to_json(orient="records")), ensure_ascii=False, indent=2))

    pd.set_option("display.width", 300)
    print("=== 真实期权链下的 collar（买 91% put / 卖 103% call，ask买 bid卖）===")
    print(df.to_string(index=False))
    print()
    print("读法：collar_net_cost_pct=期初净成本（负=净收权利金）；hedged_* = 该时点组合收益（含期权）。")
    print(f"\n输出: {OUT}/tail_hedge_events.csv")


if __name__ == "__main__":
    main()
