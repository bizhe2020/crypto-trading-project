#!/usr/bin/env python3
"""GOOGL TradingView 指标 A/B：把 TV 常用指标作为入场过滤器 / 止损替代，实测是否提升收益。

研究问题：TradingView 上最常被用来交易单票趋势的指标——ADX（趋势强度）、RSI、
MACD、SuperTrend、Donchian 通道——作为「额外入场门」或「止损替代」，能否提升
当前 GOOGL 策略的收益。

无前视纪律：所有指标都在 4h close 重采样的日线上计算，且 **shift(1)** 后才作为
当日入场门（即「昨日指标状态 → 今日是否允许入场」）。指标只 AND 进 allow_long
（禁止入场，不提前平仓），避免把平仓也变成前视。

基线 = 部署 config（0.75x + ramp 0.5% + pre_stop 1.0% + fixed 4%，= +2051%/54%，
17 笔 / 47% 胜）。评估口径与 §5.6 一致（canonical bar 级 summary）。

用法:
    python scripts/run_googl_tradingview_ab.py
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
    run_googl_4h_replay,
)

DEFAULT_CONFIG = ROOT / "config" / "config.paper.googl-high-leverage-runtime.json"
DEFAULT_SIGNAL = ROOT / "var" / "runtime" / "googl" / "googl_daily_signal.csv"
DEFAULT_OKX_4H = ROOT / "data" / "okx" / "futures" / "GOOGL_USDT_USDT-4h-futures.feather"
DEFAULT_FUNDING = ROOT / "data" / "okx" / "futures" / "GOOGL_USDT_USDT-8h-funding_rate.feather"

TIERS = {"offense": 11.2, "base": 7.5, "defense": 3.8, "flat": 0.0}


# --------------------------------------------------------------------------- #
# 指标计算（日线，Wilder 平滑）
# --------------------------------------------------------------------------- #

def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return _wilder(tr, n).bfill()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n)
    pdi = 100.0 * _wilder(pd.Series(plus_dm, index=df.index), n) / a
    mdi = 100.0 * _wilder(pd.Series(minus_dm, index=df.index), n) / a
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return _wilder(dx, n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0.0)
    loss = -d.clip(upper=0.0)
    avg_gain = _wilder(gain, n)
    avg_loss = _wilder(loss, n)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.DataFrame:
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal = line.ewm(span=sig, adjust=False).mean()
    return pd.DataFrame({"macd": line, "macd_signal": signal, "macd_hist": line - signal})


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    a = atr(df, n)
    hl2 = (df["high"] + df["low"]) / 2.0
    basic_ub = hl2 + mult * a
    basic_lb = hl2 - mult * a
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    for i in range(1, len(df)):
        if basic_ub.iloc[i] < final_ub.iloc[i - 1] or df["close"].iloc[i - 1] > final_ub.iloc[i - 1]:
            final_ub.iloc[i] = basic_ub.iloc[i]
        else:
            final_ub.iloc[i] = final_ub.iloc[i - 1]
        if basic_lb.iloc[i] > final_lb.iloc[i - 1] or df["close"].iloc[i - 1] < final_lb.iloc[i - 1]:
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            final_lb.iloc[i] = final_lb.iloc[i - 1]
    trend = pd.Series(np.nan, index=df.index)
    for i in range(1, len(df)):
        c = df["close"].iloc[i]
        prev_trend = trend.iloc[i - 1]
        prev_ub = final_ub.iloc[i - 1]
        prev_lb = final_lb.iloc[i - 1]
        if prev_trend == prev_ub:
            trend.iloc[i] = prev_ub if c <= final_ub.iloc[i] else final_lb.iloc[i]
        else:
            trend.iloc[i] = final_ub.iloc[i] if c > final_lb.iloc[i] else prev_lb
    return pd.DataFrame({"st_trend": trend, "st_ub": final_ub, "st_lb": final_lb})


def build_daily(bars: pd.DataFrame) -> pd.DataFrame:
    b = bars.copy()
    b["day"] = b["date"].dt.floor("D")
    return (
        b.groupby("day")
        .agg(open=("open", "first"), high=("high", "max"),
             low=("low", "min"), close=("close", "last"))
        .reset_index()
        .rename(columns={"day": "date"})
    )


# --------------------------------------------------------------------------- #
# 过滤注入：把 per-day 布尔 mask（shift(1) 无前视）AND 进 allow_long
# --------------------------------------------------------------------------- #

def apply_filter(merged: pd.DataFrame, daily: pd.DataFrame, mask: pd.Series, name: str) -> pd.DataFrame:
    """mask 是逐日布尔（已 shift(1)），merge_asof 到 bars 后 AND allow_long。"""
    f = pd.DataFrame({"date": daily["date"], "pass": mask.astype(bool)})
    f["date"] = pd.to_datetime(f["date"], utc=True)
    out = pd.merge_asof(
        merged.sort_values("date"), f.sort_values("date"),
        on="date", direction="backward", allow_exact_matches=True,
    )
    out["pass"] = out["pass"].fillna(False)
    out["allow_long"] = out["allow_long"] & out["pass"]
    return out


def run_one(merged: pd.DataFrame, funding, config: dict) -> dict:
    return run_googl_4h_replay(
        merged, funding, leverage_tiers=TIERS,
        stop_loss_pct=float(config.get("stop_loss_pct", 4.0)),
        taker_fee_rate=float(config.get("taker_fee_rate", 0.0005)),
        slippage_bps=float(config.get("slippage_bps", 5.0)),
        initial_capital=float(config.get("initial_capital", 1000.0)),
        equity_dd_stop_pct=15.0, equity_dd_cooldown_bars=20,
        reentry_rule="clear", reentry_clear_bars=2,
        include_funding=True, capture_open_gaps=True, entry_price_col=None,
        ramp_confirm_pct=float(config.get("ramp_confirm_pct", 0.5)),
        ramp_pre_stop_pct=float(config.get("ramp_pre_stop_pct", 1.0)),
        ramp_stop_pct=0.0, be_lock_pct=0.0,
    )


def main() -> None:
    config = json.loads(DEFAULT_CONFIG.read_text())
    bars = load_okx_4h(DEFAULT_OKX_4H)
    signal = pd.read_csv(DEFAULT_SIGNAL)
    merged = attach_googl_daily_state(bars, signal)
    funding = load_funding(DEFAULT_FUNDING)

    daily = build_daily(bars)
    adx_ = adx(daily, 14)
    rsi_ = rsi(daily["close"], 14)
    m = macd(daily["close"])
    st = supertrend(daily, 10, 3.0)
    donch_hi = daily["high"].rolling(20).max().shift(1)  # 20 日新高（前视已 shift）
    donch_lo = daily["low"].rolling(20).min().shift(1)

    # 基线
    base = run_one(merged, funding, config)
    print("=" * 110)
    print(f"基线: 收益 {base['summary']['total_return_pct']:+9.2f}%  maxDD {base['summary']['max_drawdown_pct']:6.2f}%  "
          f"交易 {base['summary']['trades']} 笔  胜率 {base['summary']['win_rate_pct']}%")
    print("=" * 110)

    print(f"\n{'过滤/止损':<34} {'收益%':>10} {'maxDD%':>7} {'交易':>4} {'胜率%':>6}")

    def report(name: str, m2: pd.DataFrame):
        r = run_one(m2, funding, config)
        s = r["summary"]
        print(f"{name:<34} {s['total_return_pct']:+10.2f} {s['max_drawdown_pct']:>7.2f} {s['trades']:>4} {s['win_rate_pct']:>6.1f}")

    # --- ADX 趋势强度过滤（只做 ADX>thr 的强趋势） ---
    for thr in (15, 20, 25, 30):
        mask = (adx_.shift(1) > thr)
        report(f"ADX14 > {thr}", apply_filter(merged, daily, mask, f"adx{thr}"))

    # --- RSI 过滤 ---
    for thr in (45, 50, 55):
        report(f"RSI14 > {thr}", apply_filter(merged, daily, rsi_.shift(1) > thr, f"rsi{thr}"))
    report("RSI14 < 70 (不追高)", apply_filter(merged, daily, rsi_.shift(1) < 70, "rsi70"))
    report("RSI14 in [40,65]", apply_filter(merged, daily, (rsi_.shift(1) >= 40) & (rsi_.shift(1) <= 65), "rsiband"))

    # --- MACD 过滤 ---
    report("MACD hist > 0", apply_filter(merged, daily, m["macd_hist"].shift(1) > 0, "macdhist"))
    report("MACD > signal", apply_filter(merged, daily, (m["macd"] - m["macd_signal"]).shift(1) > 0, "macdx"))

    # --- Donchian 突破过滤（入场须在 20 日新高附近） ---
    report("close > 20日高(-1) 且 < +5%", apply_filter(
        merged, daily, (daily["close"].shift(1) > donch_hi) & (daily["close"].shift(1) < donch_hi * 1.05), "donch"))

    # --- SuperTrend 方向过滤（price 在 SuperTrend 线上方 = 多头） ---
    st_up = daily["close"].shift(1) > st["st_trend"].shift(1)
    report("SuperTrend 多头", apply_filter(merged, daily, st_up, "stup"))

    # --- 组合过滤 ---
    report("ADX>20 且 MACD>0", apply_filter(
        merged, daily, (adx_.shift(1) > 20) & (m["macd_hist"].shift(1) > 0), "combo1"))
    report("ADX>20 且 RSI>50", apply_filter(
        merged, daily, (adx_.shift(1) > 20) & (rsi_.shift(1) > 50), "combo2"))

    print("=" * 110)
    print("所有过滤都只 AND allow_long（禁入场、不提前平仓），shift(1) 无前视。")


if __name__ == "__main__":
    main()
