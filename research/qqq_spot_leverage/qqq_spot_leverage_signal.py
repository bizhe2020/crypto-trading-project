#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QQQ 现货杠杆策略 — 每日信号（无前视，收盘后运行）。

输出的 target_leverage 是「明天开盘」应持有的杠杆：

    1.50x  趋势确认（收盘>MA200 且 MA50>MA200）且 VIX 平静
    1.00x  收盘>MA200 但 MA50<=MA200（趋势未确认），或 VIX 中性(22~28)
    0.25x  VIX 高压(>=28) 或 VIX/VIX3M>=1.0（近月恐慌）
    0.00x  收盘<MA200 或 VIX>=35

用法:
    python3 research/qqq_spot_leverage/qqq_spot_leverage_signal.py
    python3 research/qqq_spot_leverage/qqq_spot_leverage_signal.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_qqq_spot_leverage import (  # noqa: E402
    Params,
    QQQ_LAST_KNOWN,
    add_indicators,
    prepare_indicators,
    build_target_series,
    load_market_frame,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="QQQ spot leverage daily signal")
    ap.add_argument("--as-of", default=None, help="截止日期 (默认取数据最后一行)")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    p = Params()
    frame = load_market_frame("2001-01-02", QQQ_LAST_KNOWN)
    ind_all = prepare_indicators(frame, p)
    if args.as_of:
        ind_all = ind_all.loc[ind_all.index <= pd.Timestamp(args.as_of)]
    target = build_target_series(ind_all, p)

    # 输入新鲜度：所有依赖项都有效的最晚日期（VIX/VIX3M/SPY 常落后于 QQQ）
    def _last_valid(col: str):
        s = ind_all[col].dropna() if col in ind_all.columns else pd.Series(dtype=float)
        return s.index[-1] if len(s) else None

    complete_upto = min([d for d in (_last_valid("vix"), _last_valid("vix3m"), _last_valid("spy"), ind_all.index[-1]) if d is not None])
    ind_complete = ind_all.loc[:complete_upto]
    target_complete = build_target_series(ind_complete, p)

    ind = ind_all
    row = ind.iloc[-1]
    ts = ind.index[-1]
    vix_fresh = pd.notna(row["vix"]) and str(ind["vix"].dropna().index[-1].date()) == str(ts.date())

    if bool(row["above_slow"]) and bool(row["fast_above_slow"]):
        trend = "uptrend_confirmed"
    elif bool(row["above_slow"]):
        trend = "above_ma_slow_only"
    elif pd.notna(row.get("ma_slow_slope")) and float(row["ma_slow_slope"]) < p.recovery_ma_slope_min:
        trend = "bear_below_ma_slow_steep(no_recovery)"
    else:
        trend = "below_ma_slow_recovery_watch"

    slope = float(row["ma_slow_slope"]) if pd.notna(row.get("ma_slow_slope")) else 0.0
    payload = {
        "date": str(ts.date()),
        "inputs_complete_upto": str(complete_upto.date()),
        "target_using_complete_inputs": round(float(target_complete.iloc[-1]), 3),
        "stale_inputs": bool(complete_upto < ts),
        "close": round(float(row["adj_close"]), 2),
        f"ma{p.fast_window}": round(float(row["ma_fast"]), 2),
        f"ma{p.slow_window}": round(float(row["ma_slow"]), 2),
        "ma_slow_slope_20d": round(slope, 4),
        "trend": trend,
        "recovery_armed": bool(
            p.use_recovery
            and not bool(row["above_slow"])
            and slope >= p.recovery_ma_slope_min
            and pd.notna(row["ret_20"])
            and float(row["ret_20"]) >= p.recovery_ret20_min
            and (not p.recovery_require_above_fast_ma or bool(row["above_recovery_ma"]))
            and float(row["adj_close"]) >= float(row["ma_slow"]) * (1.0 - p.recovery_floor_pct)
        ),
        "vix": round(float(row["vix"]), 2) if pd.notna(row["vix"]) else None,
        "vix3m": round(float(row["vix3m"]), 2) if pd.notna(row["vix3m"]) else None,
        "vix_term": round(float(row["vix_term"]), 3) if pd.notna(row["vix_term"]) else None,
        "vix_is_fresh": bool(vix_fresh),
        "target_leverage": float(target.loc[ts]),
        "action": {
            0.0: "空仓（现金/货基）",
            0.25: "减到 0.25x（保留底仓）",
            0.5: "持有 0.5x（修复仓）",
            0.75: "持有 0.75x（修复仓）",
            1.0: "持有 1.0x（不加杠杆）",
            1.5: "持有 1.5x（满杠杆，上限）",
        }.get(float(target.loc[ts]), "按目标杠杆调仓"),
        "note": "信号在收盘后计算，次日开盘执行。若 stale_inputs=true，说明 VIX/SPY 等输入落后于 QQQ，"
                "此时 target_leverage 会因波动率窗口变化而失真，请改用 target_using_complete_inputs，"
                "或先补齐数据（import_market_data.py）再交易。",
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"日期        : {payload['date']}")
    print(f"QQQ 收盘    : {payload['close']}")
    print(f"MA{p.fast_window}/MA{p.slow_window}      : {payload[f'ma{p.fast_window}']} / {payload[f'ma{p.slow_window}']}  ({payload['trend']})")
    print(f"VIX/VIX3M   : {payload['vix']} / {payload['vix3m']}  (term={payload['vix_term']}, fresh={payload['vix_is_fresh']})")
    print(f"目标杠杆    : {payload['target_leverage']}x  ->  {payload['action']}")
    if payload["stale_inputs"]:
        print(f"⚠️ 输入不全 : VIX/SPY 等只到 {payload['inputs_complete_upto']}")
        print(f"   建议口径 : 用全部输入可得日计算的目标 = {payload['target_using_complete_inputs']}x")


if __name__ == "__main__":
    main()
