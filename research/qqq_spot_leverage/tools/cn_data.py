#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""国内纳指 ETF 数据加载（含拆分/分红连续化），供各回测脚本复用。

两个必须处理的坑
----------------
1. **份额拆分**：513100 在 2022-01-13、159941 在 2022-07-04 做过拆分，
   东财原始接口（fqt=0）会留下 −80%/−75% 的假跌幅，直接回测会瞬间爆仓。
   → 行情统一改用 **fqt=1（前复权）**；净值接口没有复权参数，
     这里用"检测跳变 + 按整数比回补"的方式把净值也连续化。
2. **溢价**：跨境 QDII 的二级市场价格 = 净值 ×(1+溢价)，溢价是纯噪音，
   必须与净值分开保存，否则会把溢价波动误当成指数收益。

对外接口
--------
    load_panel(code)   -> DataFrame[open, high, low, close, nav, premium]（连续、无重复日）
    load_nav(code)     -> Series（已连续化）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
ETF_DIR = REPO / "data" / "public" / "etf"

# 单日跌幅超过该阈值即视为"拆分/份额变更"而非真实行情（A 股有涨跌幅限制，
# 场内 ETF 单日真实跌幅不会接近 -40%；QDII 极端日也很少超过 -15%）
SPLIT_JUMP = -0.35
# 候选整数比（份额变更通常是整数倍）
SPLIT_RATIOS = [2, 3, 4, 5, 6, 8, 10, 20]


def suffix(code: str) -> str:
    return f"{code}.{'SH' if code.startswith('5') else 'SZ'}"


def _fix_splits(s: pd.Series, label: str = "") -> pd.Series:
    """把由份额拆分造成的跳变回补成连续序列（向后调整历史值）。

    做法：从最早到最晚扫描，每发现一次 <= SPLIT_JUMP 的跳变，
    就认为该日发生了份额拆分，比值 r = 前值/后值，取最接近的整数比，
    并把该日之前的全部历史值除以 r。
    """
    if s.empty:
        return s
    v = s.to_numpy(float).copy()
    idx = s.index
    n_events = 0
    for i in range(1, len(v)):
        if not (np.isfinite(v[i]) and np.isfinite(v[i - 1])) or v[i] <= 0 or v[i - 1] <= 0:
            continue
        chg = v[i] / v[i - 1] - 1.0
        if chg > SPLIT_JUMP:
            continue
        raw = v[i - 1] / v[i]
        r = min(SPLIT_RATIOS, key=lambda x: abs(x - raw))
        if abs(r - raw) / raw > 0.06:   # 不像整数比 -> 可能真是暴跌，不动
            continue
        v[:i] = v[:i] / r
        n_events += 1
        print(f"    [split] {label} {idx[i].date()}: 比值≈{raw:.3f} -> 按 1:{r} 连续化")
    return pd.Series(v, index=idx, name=s.name)


def load_nav(code: str) -> pd.Series:
    """基金单位净值（已做拆分连续化）。"""
    p = ETF_DIR / f"{suffix(code)}-nav.feather"
    if not p.exists():
        return pd.Series(dtype=float)
    d = pd.read_feather(p)
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
    s = d.sort_values("date").drop_duplicates("date").set_index("date")["nav"].astype(float)
    return _fix_splits(s, f"{code}-nav")


def load_panel(code: str, *, verbose: bool = False) -> pd.DataFrame:
    """行情 + 净值 + 溢价。行情需为前复权（fqt=1）。"""
    sp = ETF_DIR / f"{suffix(code)}-1d.feather"
    sn = ETF_DIR / f"{suffix(code)}-nav.feather"
    if not sp.exists():
        raise SystemExit(f"缺少行情 {sp}；请先运行 tools/fetch_cn_etf.py --codes {code}")
    if not sn.exists():
        raise SystemExit(f"缺少净值 {sn}；请先运行 tools/fetch_cn_etf.py --codes {code}")
    px = pd.read_feather(sp)
    px["date"] = pd.to_datetime(px["date"]).dt.tz_localize(None).dt.normalize()
    px = px.sort_values("date").drop_duplicates("date").set_index("date")
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in px.columns]
    px = px[keep].astype(float)

    if verbose:
        r = px["close"].pct_change()
        if r.min() < SPLIT_JUMP:
            print(f"    [warn] {code} 行情仍有 {r.min():.1%} 的跳变（{r.idxmin().date()}），"
                  f"该 feather 可能是 fqt=0 的旧文件")

    nav = load_nav(code)
    d = px.join(nav.rename("nav"), how="inner").dropna(subset=["close", "nav"])
    d = d[(d["close"] > 0) & (d["nav"] > 0)]
    d["premium"] = d["close"] / d["nav"] - 1.0
    if "open" not in d.columns:
        d["open"] = d["close"]
    return d


def load_qqq_daily() -> pd.Series:
    """QQQ 收盘价（统一成美东自然日，多来源取并集）。"""
    parts = []
    for cand in [REPO / "data" / "public" / "etf_full" / "QQQ-1d.feather",
                 ETF_DIR / "QQQ-1d.feather"]:
        if not cand.exists():
            continue
        d = pd.read_feather(cand)
        s = pd.Series(d["close"].to_numpy(float), index=pd.to_datetime(d["date"]), name="qqq")
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_convert("America/New_York").normalize().tz_localize(None)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        parts.append(s)
    if not parts:
        raise SystemExit("找不到 QQQ 数据")
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="first")].sort_index()
