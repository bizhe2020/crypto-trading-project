#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据新鲜度检查（round 19）——可直接用于线上信号服务的"发信号前闸门"。

为什么需要
----------
`config/tqqq_sqqq_signal_email.json` 里 `refresh_fail_open: true`，且日志显示
`refresh_proxy` (127.0.0.1:7892) 未启动、直连 Yahoo 全部 403，导致服务**静默使用旧数据**。
本脚本把"数据有多旧"变成可检查、可阻断的信号。

它只读取已落盘的数据文件，不联网、不改任何东西：

    python3 research/qqq_spot_leverage/check_freshness.py                    # 人类可读
    python3 research/qqq_spot_leverage/check_freshness.py --json             # 机器可读
    python3 research/qqq_spot_leverage/check_freshness.py --strict           # 过期则退出码 2
    python3 research/qqq_spot_leverage/check_freshness.py --max-lag 5

接入线上脚本（示例，只加 3 行，不改原有逻辑）：

    from research.qqq_spot_leverage.check_freshness import check_freshness
    ok, detail = check_freshness(Path("data/public/etf"), max_lag_trading_days=3)
    if not ok and config.get("refresh_fail_open") is not True:
        raise SystemExit(f"数据过期，拒绝出信号: {detail}")
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO / "data" / "public" / "etf"


def _last_date(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    try:
        df = pd.read_feather(path)
    except Exception:
        return None
    cols = {str(c).strip().lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("timestamp") or cols.get("datetime")
    if date_col is None:
        return None
    s = pd.to_datetime(df[date_col], utc=True, errors="coerce").dropna()
    return s.max() if len(s) else None


def check_freshness(
    data_root: Path = DEFAULT_DATA_ROOT,
    symbols: tuple[str, ...] = ("QQQ", "TQQQ", "SQQQ", "SPY", "^IXIC", "^VIX"),
    reference: str = "QQQ",
    max_lag_trading_days: int = 3,
    as_of: str | None = None,
) -> tuple[bool, dict]:
    """返回 (是否全部新鲜, 明细)。

    基准日优先取 **as_of（默认今天）**——因为"所有文件都很旧"这种系统性停摆，
    用文件之间互相比是看不出来的（这正是本次线上服务静默 7 周的原因）。
    """
    last = {sym: _last_date(data_root / f"{sym}-1d.feather") for sym in symbols}
    ref_last = pd.Timestamp(as_of or pd.Timestamp.now(tz="UTC").normalize())
    if ref_last.tzinfo is None:
        ref_last = ref_last.tz_localize("UTC")

    # 交易日历：用"data_root 里所有符号日期 + 今天"的并集，再补工作日（近似交易日）
    try:
        ref_df = pd.read_feather(data_root / f"{reference}-1d.feather")
        date_col = {str(c).strip().lower(): c for c in ref_df.columns}["date"]
        have = pd.DatetimeIndex(pd.to_datetime(ref_df[date_col], utc=True, errors="coerce").dropna().unique()).sort_values()
    except Exception:
        have = pd.DatetimeIndex([])
    cal = pd.bdate_range(min(have.min(), ref_last) if len(have) else ref_last, ref_last, tz="UTC")
    ref_days = cal.union(have).sort_values()

    details = {}
    all_ok = True
    for sym, ts in last.items():
        if ts is None:
            details[sym] = {"last": None, "lag_trading_days": None, "state": "MISSING"}
            all_ok = False
            continue
        lag = int(((ref_days > ts) & (ref_days <= ref_last)).sum())
        state = "ok" if lag <= max_lag_trading_days else "STALE"
        all_ok = all_ok and state == "ok"
        details[sym] = {"last": str(ts.date()), "lag_trading_days": lag, "state": state}

    return all_ok, {
        "reference": f"{reference} / as_of",
        "reference_last": str(ref_last.date()),
        "max_lag_trading_days": max_lag_trading_days,
        "all_fresh": all_ok,
        "details": details,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Data freshness gate for the TQQQ/SQQQ signal service")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--max-lag", type=int, default=3, help="允许落后的交易日数")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict", action="store_true", help="不新鲜则退出码 2")
    ap.add_argument("--as-of", default=None, help="基准日（默认今天）")
    args = ap.parse_args()

    ok, detail = check_freshness(Path(args.data_root), max_lag_trading_days=args.max_lag, as_of=args.as_of)
    if args.json:
        print(json.dumps(detail, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"基准日 {detail.get('reference_last')}；阈值 {detail.get('max_lag_trading_days')} 个交易日")
        for sym, d in (detail.get("details") or {}).items():
            print(f"  {sym:<6} 最后 {str(d['last']):<12} 落后 {str(d['lag_trading_days']):>3} 交易日  [{d['state']}]")
        print(f"结论: {'全部新鲜 ✅' if ok else '存在过期输入 ⚠️'}")
    if args.strict and not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
