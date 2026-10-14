#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单腿 1.5x 策略 —— 国内纳指 ETF 执行版（round 25）。

执行标的：`data/public/etf/513100.SS-1d.feather`（国泰纳指100ETF，2020-01-02 ~ 2026-05-22）
        也可通过 --cn-symbol 换成别的 A 股纳指 ETF（嘉实等），只要放入同目录同名 feather。

时间线（关键，决定无前视口径）
----------------------------
* **信号**：用 QQQ / VIX / SPY / IXIC 的收盘数据，在美东 D 日收盘后确定目标杠杆；
* **执行**：A 股在 D 日 15:00 已收盘，因此最早只能在**下一个 A 股交易日**成交 ——
  本回测按"下一个 A 股交易日的**开盘价**"执行（与仓库 `execution_price_mode: next_open` 一致）；
* **收益**：持仓按 A 股标的的收盘价 mark-to-market。

成本口径（A 股现实）
-------------------
* 佣金+过户：双边约 2bp；场内 ETF 无印花税；
* 冲击/滑点：默认 8bp/次（可用 --slippage-bps 调；跨境 ETF 点差较大）；
* **融资成本**：A 股散户通常拿不到 1.5x 场内杠杆，本回测按年化 6% 对借款部分计息（保守），
  可用 --margin-rate 调；若你实际用收益互换/两融，请按真实利率改。

用法：
    python3 research/qqq_spot_leverage/backtest_cn_single_leg.py
    python3 research/qqq_spot_leverage/backtest_cn_single_leg.py --slippage-bps 15 --margin-rate 0.08
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backtest_qqq_spot_leverage import Params, load_market_frame, prepare_indicators  # noqa: E402

OUT = HERE / "out"
REPO = HERE.parents[1]


def target_leverage_from_signal() -> pd.Series:
    """QQQ 信号（含波动率目标与硬止损的静态部分）：返回按 QQQ 交易日索引的目标杠杆。"""
    p = Params()
    frame = load_market_frame("2001-01-02", "2026-08-14")
    ind = prepare_indicators(frame, p)
    # 复用引擎的规则：方向阶梯 + 波动率目标 + 硬止损需要路径依赖，这里先生成"静态目标"，
    # 硬止损在执行循环里按 CN 标的的价格路径处理（更贴近国内实际持仓）。
    from backtest_qqq_spot_leverage import base_target  # noqa: E402

    static = ind.apply(lambda r: base_target(r, p), axis=1).rename("target")
    return static, p


def load_cn(symbol: str) -> pd.DataFrame:
    path = REPO / "data" / "public" / "etf" / f"{symbol}-1d.feather"
    if not path.exists():
        raise SystemExit(f"找不到 {path}；请先把该 ETF 的日线放到同目录（可用 import_market_data.py）")
    d = pd.read_feather(path)
    d["day"] = pd.to_datetime(d["date"], utc=True).dt.tz_convert("Asia/Shanghai").dt.normalize().dt.tz_localize(None)
    d = d.sort_values("day").drop_duplicates("day", keep="last").set_index("day")
    return d[["open", "high", "low", "close", "volume"]]


def map_signal_to_cn(sig_index: pd.DatetimeIndex, cn_index: pd.DatetimeIndex) -> dict:
    """QQQ 交易日 D 的信号 -> 最早的 A 股交易日（严格晚于 D）。"""
    cn_days = list(cn_index)
    out = {}
    for d in sig_index:
        i = bisect.bisect_right(cn_days, d)
        if i < len(cn_days):
            out[d] = cn_days[i]
    return out


def run(df: pd.DataFrame, target_qqq: pd.Series, p: Params, slippage_bps: float, margin_rate: float,
        use_stop: bool = True) -> pd.DataFrame:
    """执行循环：A 股开盘价成交，收盘价估值，借款按 margin_rate 日计息。"""
    mapping = map_signal_to_cn(target_qqq.index, df.index)
    # CN 交易日 -> 该日应执行的目标（来自前一个 QQQ 交易日的信号）
    eff: dict[pd.Timestamp, float] = {}
    for qd, cd in mapping.items():
        eff[cd] = float(target_qqq.loc[qd])

    idx = list(df.index)
    open_px = df["open"].to_numpy(float)
    close_px = df["close"].to_numpy(float)
    nav = 1.0
    lev = 0.0
    peak_price = np.nan
    cooldown = 0
    stopped = False
    prev_target = np.nan
    rows = []
    for i, d in enumerate(idx):
        tgt = eff.get(d, lev)          # 没有新信号的日子沿用当前
        target_changed = (not np.isfinite(prev_target)) or abs(tgt - prev_target) > 1e-9
        # 1) 昨日仓位吃 昨收->今开 的跳空
        if i > 0 and lev != 0 and np.isfinite(open_px[i]) and np.isfinite(close_px[i - 1]) and close_px[i - 1] > 0:
            nav *= 1.0 + lev * (open_px[i] / close_px[i - 1] - 1.0)
        # 2) 融资利息（按上日收盘后的借款）
        borrow = max(lev - 1.0, 0.0)
        nav -= nav * borrow * margin_rate / 252.0
        # 3) 开盘调仓
        do_trade = False
        if np.isfinite(open_px[i]) and open_px[i] > 0:
            if stopped:
                do_trade = False
            elif cooldown > 0:
                do_trade = False
            elif target_changed or abs(lev - tgt) > 0.25:
                do_trade = True
        if do_trade:
            turn = abs(tgt - lev)
            nav *= 1.0 - turn * slippage_bps / 1e4
            lev = tgt
            prev_target = tgt
            peak_price = close_px[i] if np.isfinite(close_px[i]) else peak_price
        # 4) 日内 open->close
        if np.isfinite(open_px[i]) and open_px[i] > 0 and np.isfinite(close_px[i]):
            nav *= 1.0 + lev * (close_px[i] / open_px[i] - 1.0)
        # 5) 硬止损（用 CN 标的自身价格路径，25 日高点回撤 8%）
        if use_stop and np.isfinite(close_px[i]):
            lo = max(0, i - p.trail_lookback + 1)
            win_peak = np.nanmax(close_px[lo:i + 1])
            if cooldown > 0:
                cooldown -= 1
                if cooldown == 0:
                    stopped = False
                    prev_target = np.nan
            elif lev > 0 and win_peak > 0 and (close_px[i] / win_peak - 1.0) <= -p.trail_drawdown:
                # 次日开盘清仓并冷却
                cooldown = p.cooldown_days
                stopped = True
                turn = abs(lev)
                nav *= 1.0 - turn * slippage_bps / 1e4
                lev = 0.0
                prev_target = 0.0
        rows.append({"date": d, "nav": nav, "lev": lev, "target": tgt})
    return pd.DataFrame(rows).set_index("date")


def stats(s: pd.Series) -> dict:
    years = (s.index[-1] - s.index[0]).days / 365.25
    r = s.pct_change().dropna()
    mdd = float((s / s.cummax() - 1).min())
    return {
        "total_pct": round(float(s.iloc[-1] - 1) * 100, 1),
        "cagr_pct": round(float(s.iloc[-1] ** (1 / years) - 1) * 100, 2),
        "mdd_pct": round(mdd * 100, 2),
        "vol_pct": round(float(r.std() * math.sqrt(252)) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * math.sqrt(252)), 2) if r.std() > 0 else 0.0,
        "calmar": round(float(s.iloc[-1] ** (1 / years) - 1) / abs(mdd), 3) if mdd < 0 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-leg 1.5x strategy executed on a CN Nasdaq ETF")
    ap.add_argument("--cn-symbol", default="513100.SS")
    ap.add_argument("--slippage-bps", type=float, default=8.0)
    ap.add_argument("--margin-rate", type=float, default=0.06)
    ap.add_argument("--no-stop", action="store_true")
    args = ap.parse_args()

    static, p = target_leverage_from_signal()
    df = load_cn(args.cn_symbol)

    # 只保留信号与 CN 都有数据的区间
    first_q = static.first_valid_index()
    df = df.loc[df.index >= pd.Timestamp("2020-01-02")]
    res = run(df, static, p, args.slippage_bps, args.margin_rate, use_stop=not args.no_stop)
    nav = res["nav"]
    m = stats(nav)

    # 对照：同区间 CN ETF 买入持有；以及常量 1.5x
    bh = df["close"] / df["close"].iloc[0]
    dd = (df["close"] / df["close"].cummax() - 1).min()
    const = run(df, pd.Series(1.5, index=static.index), p, args.slippage_bps, args.margin_rate, use_stop=False)["nav"]

    out = {
        "symbol": args.cn_symbol,
        "window": [str(nav.index[0].date()), str(nav.index[-1].date())],
        "n_days": int(len(nav)),
        "assumptions": {"slippage_bps_per_trade": args.slippage_bps, "margin_rate": args.margin_rate,
                        "execution": "next A-share open", "stop": not args.no_stop,
                        "stop_params": {"lookback": p.trail_lookback, "drawdown": p.trail_drawdown, "cooldown": p.cooldown_days}},
        "strategy": m,
        "buy_hold_1x": {"total_pct": round(float(bh.iloc[-1] - 1) * 100, 1), "mdd_pct": round(float(dd) * 100, 2)},
        "constant_1.5x": stats(const),
        "avg_leverage": round(float(res["lev"].mean()), 3),
        "rebalances": int((res["lev"].diff().abs() > 0.01).sum()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cn_single_leg.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    res.to_csv(OUT / "cn_single_leg_nav.csv")

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
