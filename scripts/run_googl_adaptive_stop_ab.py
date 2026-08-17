#!/usr/bin/env python3
"""GOOGL 自适应止损 A/B：固定 4% 止损 vs 波动率缩放止损宽度。

研究问题：固定 4% trailing stop 在高波动期可能过紧（趋势中途被打掉）、低波动期
可能过松（回吐过多）。Fonseca 2026 证明 ATR 乘数 [3.5,7.0] 是平坦稳健平台——
自适应止损是结构性稳健参数，而非点优化。本脚本验证：把固定 4% 换成
「4% × (前20日波动率/扩展中位波动率)」的逐 bar 缩放，是否改善 return/maxDD。

方向：vol_prior 高 → 止损更宽（给趋势呼吸空间）；vol_prior 低 → 止损更紧（锁利）。
无前视：vol_prior = 前20日年化波动率 shift(1)，中位 = 扩展中位数。

评估口径与 run_googl_vol_target_ab 一致。全窗口 + 信念前/后子窗口。

用法:
    python scripts/run_googl_adaptive_stop_ab.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
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


def build_daily_stop_pct(bars: pd.DataFrame, base_pct: float, vol_window: int = 20,
                         lo: float = 0.5, hi: float = 1.5) -> pd.DataFrame:
    """逐 bar 止损宽度（%）：base_pct × clip(前20日波动率/扩展中位波动率, lo, hi)。"""
    daily = bars.set_index("date")["close"].resample("1D").last().dropna()
    logret = np.log(daily / daily.shift(1))
    vol_20d = logret.rolling(vol_window).std() * np.sqrt(252)
    vol_prior = vol_20d.shift(1)
    anchor = vol_prior.expanding().median()
    ratio = (vol_prior / anchor).clip(lower=lo, upper=hi)
    stop_pct = base_pct * ratio
    stop_pct = stop_pct.fillna(base_pct)
    return pd.DataFrame({"date": stop_pct.index, "stop_pct": stop_pct.values})


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
    prev = eq[eq.index < start]
    base = 1000.0 if prev.empty else float(prev.iloc[-1])
    total = (float(w.iloc[-1]) / base - 1.0) * 100.0
    return total, max_drawdown_pct(w)


def run_one(merged: pd.DataFrame, funding, config: dict, stop_pct_col: str | None) -> dict:
    return run_googl_4h_replay(
        merged, funding,
        leverage_tiers=TIERS,
        stop_pct_col=stop_pct_col,
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
    base_pct = float(config.get("stop_loss_pct", 4.0))
    bars = load_okx_4h(DEFAULT_OKX_4H)
    signal = pd.read_csv(DEFAULT_SIGNAL)
    merged = attach_googl_daily_state(bars, signal)
    funding = load_funding(DEFAULT_FUNDING)

    base = run_one(merged, funding, config, stop_pct_col=None)
    b_full = win_metrics(base["path"], PRE_START, END)
    b_pre = win_metrics(base["path"], PRE_START, CONVICTION_START - pd.Timedelta(days=1))
    b_conv = win_metrics(base["path"], CONVICTION_START, END)
    print("=" * 100)
    print(f"基线（固定 {base_pct}% 止损）")
    print(f"  全窗口: 收益 {b_full[0]:+9.2f}%  maxDD {b_full[1]:6.2f}%  |  "
          f"信念前 {b_pre[0]:+8.2f}%/{b_pre[1]:.2f}%  |  信念后 {b_conv[0]:+8.2f}%/{b_conv[1]:.2f}%")
    print(f"  交易 {base['summary']['trades']} 笔  胜率 {base['summary']['win_rate_pct']}%")
    print("=" * 100)

    print("\n自适应止损 A/B（stop_pct = %.1f%% × clip(前20日波动/中位, lo, hi)）" % base_pct)
    print(f"{'lo/hi':>10} {'全窗口收益%':>12} {'全maxDD%':>9} {'CAGR%':>8} "
          f"{'信念前%':>10} {'前DD%':>7} {'信念后%':>10} {'后DD%':>7} {'交易':>4}")
    combos = [(0.5, 1.5), (0.6, 1.5), (0.7, 1.5), (0.5, 2.0), (0.75, 1.25), (0.75, 2.0)]
    for lo, hi in combos:
        sp = build_daily_stop_pct(bars, base_pct, vol_window=20, lo=lo, hi=hi)
        sp["date"] = pd.to_datetime(sp["date"], utc=True)
        m2 = pd.merge_asof(
            merged.sort_values("date"),
            sp.sort_values("date"),
            on="date",
            direction="backward",
            allow_exact_matches=True,
        )
        r = run_one(m2, funding, config, stop_pct_col="stop_pct")
        f = win_metrics(r["path"], PRE_START, END)
        pre = win_metrics(r["path"], PRE_START, CONVICTION_START - pd.Timedelta(days=1))
        conv = win_metrics(r["path"], CONVICTION_START, END)
        mean_stop = float(m2["stop_pct"].mean())
        print(f"{lo:.2f}/{hi:.2f}   {f[0]:>12.2f} {f[1]:>9.2f} {r['summary']['cagr_pct']:>8} "
              f"{pre[0]:>10.2f} {pre[1]:>7.2f} {conv[0]:>10.2f} {conv[1]:>7.2f} {r['summary']['trades']:>4}")
    print("=" * 100)
    print(f"mean stop_pct 约 {base_pct}% 时 = 与基线平均止损宽度可比；hi 越大 = 高波动期止损越宽。")


if __name__ == "__main__":
    main()
