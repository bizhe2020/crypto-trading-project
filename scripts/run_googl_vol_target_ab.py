#!/usr/bin/env python3
"""GOOGL 波动率目标 A/B：固定杠杆 vs 波动率缩放杠杆。

研究问题（对应「提升 GOOGL 收益空间」目标）：当前 0.75x 是固定乘数，完全
不考虑波动率状态。学术证据（Moreira-Muir 2017 JF；Cederburg 2020）表明
momentum/趋势是唯一 OOS 稳健受益于波动率管理的类别。本脚本在真实 4h 执行
回测框架上验证：把固定杠杆换成「逆波动率缩放」是否改善 return / maxDD。

缩放规则（无前视）：
    vol_20d     = 20 日年化 realized vol（4h close 重采样成日线）
    vol_prior   = vol_20d.shift(1)          # 入场日 D 只用 D-1 及之前数据
    anchor      = vol_prior.expanding().median()  # 扩展中位数，无前视锚点
    vol_scale   = clip(anchor / vol_prior, lo, hi)
    有效杠杆     = leverage_tiers[tier] × vol_scale

    vol_scale 中位数 ≈ 1.0（锚点=中位数），故平均杠杆与固定 0.75x 基准可比，
    只是把杠杆从高波动期「搬」到低波动期。lo 控制降险下限，hi 控制加杠杆上限。

评估口径：与 sweep_googl_knobs / compare_qqq_googl_router 一致（shadow gate
15%/20bar + reentry clear 2 + ramp_confirm 0.5 + ramp_pre_stop 1.0，含 funding、
含隔夜跳空 capture_open_gaps）。全窗口 2024-01→2026-08 + 信念前/后两子窗口
（2025-11-14 为 conviction 切换点）。

用法:
    python scripts/run_googl_vol_target_ab.py
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
CONVICTION_START = pd.Timestamp("2025-11-14", tz="UTC")  # 伯克希尔 13F 信念切换点
END = pd.Timestamp("2026-08-07", tz="UTC")

TIERS = {"offense": 11.2, "base": 7.5, "defense": 3.8, "flat": 0.0}


def build_daily_vol_scale(bars: pd.DataFrame, vol_window: int = 20, lo: float = 0.5, hi: float = 1.5) -> pd.DataFrame:
    """从 4h close 重采样日线，算逆波动率缩放列（无前视）。返回 date + vol_scale。"""
    daily = bars.set_index("date")["close"].resample("1D").last().dropna()
    logret = np.log(daily / daily.shift(1))
    vol_20d = logret.rolling(vol_window).std() * np.sqrt(252)
    vol_prior = vol_20d.shift(1)  # 入场日 D 只用 D-1 及之前的波动率
    anchor = vol_prior.expanding().median()  # 扩展中位数，无前视锚点
    scale = (anchor / vol_prior).clip(lower=lo, upper=hi)
    scale = scale.fillna(1.0)
    return pd.DataFrame({"date": scale.index, "vol_scale": scale.values})


def daily_equity(path: pd.DataFrame) -> pd.Series:
    """把 4h-bar 级 daily_return 聚合到日频，从 1000 复合，返回按日索引的权益曲线。"""
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
    mdd = max_drawdown_pct(w)
    return total, mdd


def run_one(merged: pd.DataFrame, funding, config: dict, lev_scale_col: str | None) -> dict:
    return run_googl_4h_replay(
        merged,
        funding,
        leverage_tiers=TIERS,
        lev_scale_col=lev_scale_col,
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

    # 基线（固定 0.75x）
    base = run_one(merged, funding, config, lev_scale_col=None)
    b_full = win_metrics(base["path"], PRE_START, END)
    b_pre = win_metrics(base["path"], PRE_START, CONVICTION_START - pd.Timedelta(days=1))
    b_conv = win_metrics(base["path"], CONVICTION_START, END)
    b_s = base["summary"]
    print("=" * 100)
    print("基线（固定 0.75x，offense 11.2 / base 7.5 / defense 3.8 未用）")
    print(f"  全窗口 {PRE_START.date()}→{END.date()}:  收益 {b_full[0]:+9.2f}%  maxDD {b_full[1]:6.2f}%")
    print(f"  信念前 {PRE_START.date()}→2025-11-13:  收益 {b_pre[0]:+9.2f}%  maxDD {b_pre[1]:6.2f}%")
    print(f"  信念后 2025-11-14→{END.date()}:  收益 {b_conv[0]:+9.2f}%  maxDD {b_conv[1]:6.2f}%")
    print(f"  交易 {b_s['trades']} 笔  胜率 {b_s['win_rate_pct']}%  CAGR {b_s['cagr_pct']}%")
    print("=" * 100)

    # 波动率目标 sweep（clamp 边界）
    print("\n波动率目标 A/B（vol_scale = clip(扩展中位/前20日波动率, lo, hi)）")
    print(f"{'lo/hi':>12} {'mean_scale':>10} {'全窗口收益%':>12} {'全maxDD%':>9} {'CAGR%':>8} "
          f"{'信念前%':>10} {'前DD%':>7} {'信念后%':>10} {'后DD%':>7}")
    combos = [(0.5, 1.5), (0.6, 1.4), (0.7, 1.3), (0.75, 1.25), (0.5, 1.25), (0.5, 1.0), (0.6, 1.0)]
    for lo, hi in combos:
        vol_grid = build_daily_vol_scale(bars, vol_window=20, lo=lo, hi=hi)
        vol_grid["date"] = pd.to_datetime(vol_grid["date"], utc=True)
        m2 = pd.merge_asof(
            merged.sort_values("date"),
            vol_grid.sort_values("date"),
            on="date",
            direction="backward",
            allow_exact_matches=True,
        )
        r = run_one(m2, funding, config, lev_scale_col="vol_scale")
        f = win_metrics(r["path"], PRE_START, END)
        pre = win_metrics(r["path"], PRE_START, CONVICTION_START - pd.Timedelta(days=1))
        conv = win_metrics(r["path"], CONVICTION_START, END)
        mean_scale = float(m2["vol_scale"].mean())
        cagr = r["summary"]["cagr_pct"]
        print(f"{lo:.2f}/{hi:.2f}   {mean_scale:>10.3f} {f[0]:>12.2f} {f[1]:>9.2f} {cagr:>8} "
              f"{pre[0]:>10.2f} {pre[1]:>7.2f} {conv[0]:>10.2f} {conv[1]:>7.2f}")
    print("=" * 100)
    print("mean_scale ≈ 1.0 = 与基线平均杠杆可比；<1.0 = 波动率目标整体降了杠杆，")
    print("        需要区分「收益提升来自择时」还是「收益降低因平均杠杆下降」。")
    print("信念后窗口（2025-11-14→）含全部 conviction offense 单，是收益来源最集中的段。")


if __name__ == "__main__":
    main()
