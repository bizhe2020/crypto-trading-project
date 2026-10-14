#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""恐慌指标（VIX / VIX3M / VVIX / SKEW）能否给单腿 1.5x 加分？（round 30）

背景
----
数据目录里有 4 个恐慌/波动率指标，历史长度：

| 指标 | 含义 | 本地覆盖 |
|---|---|---|
| ^VIX   | 30 日隐含波动率（"恐慌指数"） | 2022-01 起（短） |
| ^VIX3M | 3 个月隐含波动率 | 2006-07 起 |
| ^VVIX  | VIX 的隐含波动率（波动率的波动率） | 2007-01 起 |
| ^SKEW  | 尾部风险偏度（黑天鹅定价） | 1999-01 起 |

此前只测过 **VIX 水平闸门**，结论为负。本脚本把其余 3 个 + 它们的经典用法一起测：

* **期限结构倒挂**：`VIX / VIX3M > 1`（近月比远月贵 = 市场急panic，经典的危机信号）；
* **VIX 高位**：`VIX > 30 / 35`（绝对水平）；
* **VVIX 高位**：`VVIX > 110/120`（波动率本身失控）；
* **SKEW 高位**：`SKEW > 145/150`（尾部风险定价高）。

实验设计（避免自欺）
------------------
对每个过滤器测 **4 种用法**，全部叠加在「国内 ETF + 溢价闸门」的策略之上：

1. **`delev`（降杠杆）**：命中时把目标杠杆 **× 0.5**（不清仓，只减半）；
2. **`block`（禁新仓）**：命中时禁止建仓/加仓（同溢价闸门逻辑）；
3. **`half`（禁新仓且现有仓位减半）**：更激进；
4. **`cash`（命中即空仓）**：最激进 —— 同时也测它是否只是"错过反弹"。

**无前视**：恐慌指标在美东 D 日收盘后与 QQQ 收盘同时可得；
引擎把"QQQ D 日收盘的目标"映射到**下一个 A 股交易日**执行，
所以指标只需对齐 QQQ 日历，**不需要额外 shift**（否则会双重滞后、白白错过信号）。

用法：
    python3 research/qqq_spot_leverage/test_panic_filters.py
    python3 research/qqq_spot_leverage/test_panic_filters.py --code 513100 --mode all
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "out"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))

import backtest_cn_premium_aware as B  # noqa: E402
from backtest_qqq_spot_leverage import Params  # noqa: E402

PANIC = {"^VIX": "vix", "^VIX3M": "vix3m", "^VVIX": "vvix", "^SKEW": "skew"}


def load_panic() -> pd.DataFrame:
    """读取 4 个恐慌指标，统一成美东自然日索引。"""
    cols = {}
    for fname, name in PANIC.items():
        p = REPO / "data" / "public" / "etf" / f"{fname}-1d.feather"
        if not p.exists():
            continue
        d = pd.read_feather(p)
        d["date"] = pd.to_datetime(d["date"])
        if getattr(d["date"].dt, "tz", None) is not None:
            d["date"] = d["date"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        else:
            d["date"] = d["date"].dt.normalize()
        s = d.sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float)
        cols[name] = s
    return pd.DataFrame(cols)


def build_filters(pan: pd.DataFrame) -> dict[str, pd.Series]:
    """返回 {过滤名: bool Series}，True = 命中恐慌。"""
    f: dict[str, pd.Series] = {}
    if "vix" in pan:
        f["VIX>25"] = pan["vix"] > 25
        f["VIX>30"] = pan["vix"] > 30
        f["VIX>35"] = pan["vix"] > 35
    if "vix3m" in pan:
        f["VIX3M>25"] = pan["vix3m"] > 25
    if "vix" in pan and "vix3m" in pan:
        ratio = pan["vix"] / pan["vix3m"]
        f["倒挂 VIX/VIX3M>1.0"] = ratio > 1.0
        f["倒挂 VIX/VIX3M>1.05"] = ratio > 1.05
    if "vvix" in pan:
        f["VVIX>100"] = pan["vvix"] > 100
        f["VVIX>110"] = pan["vvix"] > 110
        f["VVIX>120"] = pan["vvix"] > 120
    if "skew" in pan:
        f["SKEW>140"] = pan["skew"] > 140
        f["SKEW>145"] = pan["skew"] > 145
        f["SKEW>150"] = pan["skew"] > 150
    # VIX 5 日急升（恐慌加速）
    if "vix" in pan:
        f["VIX 5日涨>30%"] = pan["vix"].pct_change(5) > 0.30
    if "vvix" in pan:
        f["VVIX 5日涨>15%"] = pan["vvix"].pct_change(5) > 0.15
    return f


def apply_filter(mode: str, base: pd.Series, hit: pd.Series) -> tuple[pd.Series, pd.Series | None]:
    """把恐慌过滤叠加到目标杠杆序列上。

    返回 (调整后的目标序列, block_mask)。
    * delev / half / cash  -> 直接改目标序列
    * block                -> 返回 block_mask，由引擎保证"只许减不许加"（不强制清仓）
    """
    t = base.copy()
    h = hit.reindex(t.index).fillna(False)
    if mode == "delev":
        return t.where(~h, t * 0.5), None
    if mode == "block":
        return t, h
    if mode == "half":
        return t.where(~h, t * 0.5), h          # 先减半，且命中期间不许加回
    if mode == "cash":
        return t.where(~h, 0.0), None
    raise ValueError(mode)


def run_arm(d: pd.DataFrame, tgt: pd.Series, p: Params, *, gate: float,
            margin_rate: float, cap: float, block_mask: pd.Series | None = None,
            use_stop: bool = True) -> dict:
    """跑一"臂"并返回绩效指标。"""
    r = B.run(d, tgt, p, gate=gate, use_stop=use_stop, block_mask=block_mask,
              slippage_bps=8.0, margin_rate=margin_rate, max_lev=cap)
    return B.stats(r["equity"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="159501")
    ap.add_argument("--start", default="2022-01-03")
    ap.add_argument("--end", default="2026-09-21")
    ap.add_argument("--gate", type=float, default=0.02)
    ap.add_argument("--margin-rate", type=float, default=0.05)
    ap.add_argument("--cap", type=float, default=1.5)
    ap.add_argument("--modes", nargs="*", default=["delev", "block", "half", "cash"])
    a = ap.parse_args()

    p = Params()
    tgt = B.qqq_static_target(p)
    d = B.load_panel(a.code).loc[a.start:a.end]
    pan = load_panic()
    print(f"标的 {a.code}  窗口 {d.index.min().date()} ~ {d.index.max().date()}  ({len(d)} 天)")
    print("恐慌指标覆盖：" + ", ".join(f"{k}({v.dropna().index.min().date()}~{v.dropna().index.max().date()}, n={v.notna().sum()})"
                                     for k, v in pan.items()))

    # 基准臂
    base = run_arm(d, tgt, p, gate=a.gate, margin_rate=a.margin_rate, cap=a.cap)
    # 注意：目标序列必须保持 **QQQ 日历**。引擎内部 map_signal_to_cn 会把
    # "QQQ D 日收盘信号"映射到"下一个 A 股交易日"，这本身就实现了 S 日收盘后 -> S+1 执行的时序。
    # 因此恐慌指标也必须对齐到 QQQ 日历，**不需要再 shift(1)**，否则会双重滞后；
    # 更不能把它 reindex 到 CN 日历（两个日历大部分日期不重合，会污染信号）。
    print(f"\n基准（阶梯+波动率目标+硬止损+溢价闸门{a.gate*100:.0f}%）："
          f"CAGR {base['cagr_pct']:.2f}% / MDD {base['mdd_pct']:.2f}% / Sharpe {base['sharpe']:.2f} / Calmar {base['calmar']:.2f}")

    # 恐慌指标对齐到 **QQQ 日历**（与 target 同一日历），执行滞后交给引擎的 CN 映射
    pan_q = pan.reindex(pan.index.union(tgt.index)).ffill().reindex(tgt.index)
    filters = build_filters(pan_q)

    rows = []
    for name, hit in filters.items():
        hit_q = hit.fillna(False).astype(bool)
        n_hit = int(hit_q.sum())
        if n_hit == 0:
            continue
        for mode in a.modes:
            t2, bm = apply_filter(mode, tgt, hit_q)
            st = run_arm(d, t2, p, gate=a.gate, margin_rate=a.margin_rate, cap=a.cap, block_mask=bm)
            rows.append({
                "filter": name, "mode": mode, "days_hit": n_hit, "pct_hit": round(n_hit / len(tgt) * 100, 1),
                **{k: st[k] for k in ["cagr_pct", "mdd_pct", "sharpe", "calmar"]},
                "d_cagr": round(st["cagr_pct"] - base["cagr_pct"], 2),
                "d_mdd": round(st["mdd_pct"] - base["mdd_pct"], 2),
                "d_sharpe": round(st["sharpe"] - base["sharpe"], 2),
                "d_calmar": round(st["calmar"] - base["calmar"], 2),
            })

    df = pd.DataFrame(rows).sort_values("d_calmar", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "panic_filter_tests.csv", index=False)
    (OUT / "panic_filter_tests.json").write_text(json.dumps(
        {"code": a.code, "window": [str(d.index.min().date()), str(d.index.max().date())],
         "base": base, "gate": a.gate, "margin_rate": a.margin_rate, "cap": a.cap,
         "results": rows}, ensure_ascii=False, indent=2))

    pd.set_option("display.width", 220)
    print("\n=== 恐慌过滤叠加效果（按 Calmar 改善排序）===")
    show = df[["filter", "mode", "pct_hit", "cagr_pct", "mdd_pct", "sharpe", "calmar",
               "d_cagr", "d_mdd", "d_sharpe", "d_calmar"]]
    print(show.to_string(index=False))

    improved = df[(df["d_calmar"] > 0) & (df["d_sharpe"] > 0)]
    print(f"\n同时改善 Sharpe 与 Calmar 的组合：{len(improved)} / {len(df)}")
    if len(improved):
        print(improved[["filter", "mode", "pct_hit", "cagr_pct", "mdd_pct", "sharpe", "calmar"]].to_string(index=False))
    print(f"\n已写出 {OUT/'panic_filter_tests.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
