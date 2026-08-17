#!/usr/bin/env python3
"""GOOGL 财报降档 A/B：持仓硬扛财报 vs 财报日强制空仓（再重入）。

研究问题：财报跳空是单票高倍止损无法防御的离散尾部风险（±7-12%）。机构行为
（CME / Franzoni）是财报前系统性降险。本脚本验证：财报日前 N 天强制 FLAT、
财报后由信号自然重入，是否改善 return / maxDD。

财报日（Alphabet，公告均美股盘后，窗口 2024-01→2026-08，10 次）：
  2024-04-25 / 07-23 / 10-29
  2025-02-04 / 04-24 / 07-23 / 10-29
  2026-02-04 / 04-29 / 07-22
数据确认 gap 落在公告日当天（E-1 收 → E 收）：2025-02-04 -6.3%、2024-04-25
+12.6%、2026-04-29 +7.2%。

模型：对每个财报日 E，把 [E-N+1, E] 共 N 个 UTC 日的 allow_long 强制 False →
持仓在窗口首日开盘平仓（signal_flat），窗口内禁入，E+1 起按信号 + shadow gate
reentry_clear=2 决定重入。N=1 = 只空仓财报日当天。

评估口径与 run_googl_vol_target_ab 一致（shadow gate 15%/20bar + ramp 0.5/1.0 +
funding + capture_open_gaps）。全窗口 + 信念前/后子窗口。

用法:
    python scripts/run_googl_earnings_derisk_ab.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_googl_usdt_4h import (  # noqa: E402
    attach_googl_daily_state,
    load_funding,
    load_okx_4h,
    max_drawdown_pct,
    run_googl_4h_replay,
)

DEFAULT_CONFIG = ROOT / "config" / "config.paper.googl-high-leverage-runtime.json"
DEFAULT_SIGNAL = ROOT / "var" / "runtime" / "googl" / "googl_daily_signal.csv"
DEFAULT_OKX_4H = ROOT / "data" / "okx" / "futures" / "GOOGL_USDT_USDT-4h-futures.feather"
DEFAULT_FUNDING = ROOT / "data" / "okx" / "futures" / "GOOGL_USDT_USDT-8h-funding_rate.feather"

PRE_START = pd.Timestamp("2024-01-02", tz="UTC")
CONVICTION_START = pd.Timestamp("2025-11-14", tz="UTC")
END = pd.Timestamp("2026-08-07", tz="UTC")

TIERS = {"offense": 11.2, "base": 7.5, "defense": 3.8, "flat": 0.0}

EARNINGS_DATES = [
    "2024-04-25", "2024-07-23", "2024-10-29",
    "2025-02-04", "2025-04-24", "2025-07-23", "2025-10-29",
    "2026-02-04", "2026-04-29", "2026-07-22",
]


def apply_derisk(merged: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """把财报日前 n_days 个 UTC 日强制 FLAT，返回改好 allow_long 的副本。"""
    out = merged.copy()
    day = out["date"].dt.floor("D")
    mask_days = set()
    for e in EARNINGS_DATES:
        e = pd.Timestamp(e, tz="UTC")
        for k in range(n_days):
            mask_days.add(e - pd.Timedelta(days=k))
    out["allow_long"] = out["allow_long"] & ~day.isin(list(mask_days))
    return out


def daily_equity(path: pd.DataFrame) -> pd.Series:
    p = path.copy()
    p["day"] = p["date"].dt.floor("D")
    daily_ret = p.groupby("day")["daily_return"].apply(lambda s: float((1.0 + s).prod() - 1.0))
    return (1.0 + daily_ret).cumprod() * 1000.0


def win_metrics(path: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, float]:
    eq = daily_equity(path)
    w = eq[(eq.index >= start) & (eq.index <= end)]
    if len(w) < 2:
        return 0.0, 0.0
    total = (float(w.iloc[-1]) / float(w.iloc[0]) - 1.0) * 100.0
    return total, max_drawdown_pct(w)


def run_one(merged: pd.DataFrame, funding, config: dict) -> dict:
    return run_googl_4h_replay(
        merged, funding,
        leverage_tiers=TIERS,
        stop_loss_pct=float(config.get("stop_loss_pct", 4.0)),
        taker_fee_rate=float(config.get("taker_fee_rate", 0.0005)),
        slippage_bps=float(config.get("slippage_bps", 5.0)),
        initial_capital=float(config.get("initial_capital", 1000.0)),
        equity_dd_stop_pct=15.0,
        equity_dd_cooldown_bars=20,
        reentry_rule="clear",
        reentry_clear_bars=2,
        include_funding=True,
        capture_open_gaps=True,
        entry_price_col=None,
        ramp_confirm_pct=float(config.get("ramp_confirm_pct", 0.5)),
        ramp_pre_stop_pct=float(config.get("ramp_pre_stop_pct", 1.0)),
        ramp_stop_pct=0.0,
        be_lock_pct=0.0,
    )


def main() -> None:
    config = json.loads(DEFAULT_CONFIG.read_text())
    bars = load_okx_4h(DEFAULT_OKX_4H)
    signal = pd.read_csv(DEFAULT_SIGNAL)
    merged = attach_googl_daily_state(bars, signal)
    funding = load_funding(DEFAULT_FUNDING)

    base = run_one(merged, funding, config)
    b_full = win_metrics(base["path"], PRE_START, END)
    b_pre = win_metrics(base["path"], PRE_START, CONVICTION_START - pd.Timedelta(days=1))
    b_conv = win_metrics(base["path"], CONVICTION_START, END)
    print("=" * 100)
    print("基线（硬扛财报）")
    print(f"  全窗口: 收益 {b_full[0]:+9.2f}%  maxDD {b_full[1]:6.2f}%  |  "
          f"信念前 {b_pre[0]:+8.2f}%/{b_pre[1]:.2f}%  |  信念后 {b_conv[0]:+8.2f}%/{b_conv[1]:.2f}%")
    print(f"  交易 {base['summary']['trades']} 笔  胜率 {base['summary']['win_rate_pct']}%")
    print("=" * 100)

    print("\n财报降档 A/B（强制 FLAT 财报日前 N 天，N=0 即基线）")
    print(f"{'N天':>4} {'全窗口收益%':>12} {'全maxDD%':>9} {'CAGR%':>8} "
          f"{'信念前%':>10} {'前DD%':>7} {'信念后%':>10} {'后DD%':>7} {'交易':>4}")
    for n in (1, 2, 3, 5):
        m2 = apply_derisk(merged, n)
        r = run_one(m2, funding, config)
        f = win_metrics(r["path"], PRE_START, END)
        pre = win_metrics(r["path"], PRE_START, CONVICTION_START - pd.Timedelta(days=1))
        conv = win_metrics(r["path"], CONVICTION_START, END)
        print(f"{n:>4} {f[0]:>12.2f} {f[1]:>9.2f} {r['summary']['cagr_pct']:>8} "
              f"{pre[0]:>10.2f} {pre[1]:>7.2f} {conv[0]:>10.2f} {conv[1]:>7.2f} {r['summary']['trades']:>4}")
    print("=" * 100)


if __name__ == "__main__":
    main()
