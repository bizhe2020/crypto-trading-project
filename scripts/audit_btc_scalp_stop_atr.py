#!/usr/bin/env python3
"""止损宽度校准分析：每笔的止损距离相当于多少倍 ATR。

动机
----
外部证据（arXiv:2602.11708，150+ 加密永续、36 个月样本外、含成本）显示
**ATR 止损倍数 α∈[2.0,3.5] 是宽阔高原，最优 ≈2.5**；Volatility Box 的
200 组 SP500 设置显示噪音触发率：**<1.0×ATR → >65%**，1.5× → 38%，2.0× → 21%。
且「止损越宽 → 胜率越高、盈亏比越低，但**期望值越高**」。

本脚本用项目自身的 ATR 定义（`report_pa_ict_liquidity_features.atr_series`，
Wilder ATR，周期取配置的 `atr_period`）计算每笔真实交易的
`stop_distance / ATR(entry)`，用来回答：

    「0/21 止盈、几乎全部止损出场」是否只是止损设置在噪音区内造成的？

用法:
    python3 scripts/audit_btc_scalp_stop_atr.py \
        --trades var/tokyo_audit/btc_scalp_20260920/trades_audit.csv \
        --klines var/tokyo_audit/btc_scalp_20260920/BTC_USDT_USDT-15m-2026.csv \
        --atr-period 14 \
        --out-md var/reports/btc_scalp_stop_atr.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_pa_ict_liquidity_features import atr_series  # noqa: E402
from strategy.scalp_robust_v2_core import dataframe_to_candles  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--klines", required=True)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--out-md", default="var/reports/btc_scalp_stop_atr.md")
    parser.add_argument("--out-csv", default="var/tokyo_audit/btc_scalp_20260920/stop_atr.csv")
    args = parser.parse_args()

    frame = pd.read_csv(args.trades)
    frame["open_ts"] = pd.to_datetime(frame["open_ts"], utc=True)
    frame = frame[frame["stop"].notna()].reset_index(drop=True)

    klines = pd.read_csv(args.klines, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    candles = dataframe_to_candles(klines)
    atr = np.array(atr_series(candles, args.atr_period), dtype=float)

    ts_to_idx = {ts: i for i, ts in enumerate(klines["date"])}
    rows = []
    for _, trade in frame.iterrows():
        idx = ts_to_idx.get(trade["open_ts"])
        if idx is None or idx >= len(atr) or not np.isfinite(atr[idx]) or atr[idx] <= 0:
            rows.append({"open_ts": trade["open_ts"], "atr_multiple": np.nan})
            continue
        entry = float(trade["entry_signal"])
        stop = float(trade["stop"])
        stop_dist = abs(entry - stop)
        rows.append({
            "open_ts": trade["open_ts"],
            "direction": trade["direction"],
            "entry": entry,
            "stop_dist_usdt": stop_dist,
            "stop_dist_pct": stop_dist / entry * 100.0,
            "atr_usdt": atr[idx],
            "atr_pct": atr[idx] / entry * 100.0,
            "atr_multiple": stop_dist / atr[idx],
            "realized_R": trade["realized_R"],
            "mfe_R": trade["mfe_R"],
            "mae_R": trade["mae_R"],
            "net_pnl": trade["net_pnl"],
            "regime_label": trade["regime_label"],
            "target_rr": trade["target_rr"],
        })
    result = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)

    valid = result.dropna(subset=["atr_multiple"])

    lines: list[str] = []
    add = lines.append
    add("# BTC Scalp 止损宽度校准：止损 = 多少倍 ATR")
    add("")
    add(f"- 样本：**{len(result)} 笔**，其中可计算 ATR 倍数 **{len(valid)} 笔**")
    add(f"- ATR 定义：Wilder ATR({args.atr_period})，作用于 15m（项目自身实现）")
    add(f"- 入场 bar 收盘时 ATR 已知，**无前视**")
    add("")
    add("## 分布")
    add("")
    if len(valid):
        q = valid["atr_multiple"]
        add("| 统计量 | 值 |")
        add("|---|---:|")
        add(f"| 最小 | {q.min():.2f}× |")
        add(f"| 25 分位 | {q.quantile(0.25):.2f}× |")
        add(f"| **中位数** | **{q.median():.2f}×** |")
        add(f"| 75 分位 | {q.quantile(0.75):.2f}× |")
        add(f"| 最大 | {q.max():.2f}× |")
        add(f"| 平均 | {q.mean():.2f}× |")
        add("")
        add(f"- 止损距离：中位 {valid['stop_dist_pct'].median():.2f}% ；"
            f"ATR 本身：中位 {valid['atr_pct'].median():.2f}%")
        add("")

        add("## 与外部证据对照")
        add("")
        add("| 区间 | 笔数 | 占比 | 外部参考（噪音触发率） |")
        add("|---|---:|---:|---|")
        buckets = [
            ("< 1.0×（噪音区）", valid["atr_multiple"] < 1.0, "**>65%** 被噪音打掉"),
            ("1.0 – 1.5×", (valid["atr_multiple"] >= 1.0) & (valid["atr_multiple"] < 1.5), "~50%"),
            ("1.5 – 2.0×", (valid["atr_multiple"] >= 1.5) & (valid["atr_multiple"] < 2.0), "~38%"),
            ("2.0 – 3.5×（高原）", (valid["atr_multiple"] >= 2.0) & (valid["atr_multiple"] <= 3.5), "~21%，期望值最高区"),
            ("> 3.5×", valid["atr_multiple"] > 3.5, "更宽，盈亏比下降"),
        ]
        for label, mask, ref in buckets:
            count = int(mask.sum())
            add(f"| {label} | {count} | {count/len(valid)*100:.0f}% | {ref} |")
        add("")

        in_plateau = valid[(valid["atr_multiple"] >= 2.0) & (valid["atr_multiple"] <= 3.5)]
        below_1 = valid[valid["atr_multiple"] < 1.0]
        add("## 分层结果")
        add("")
        add("| 分组 | 笔数 | 平均实现 R | 总 R | 平均 MFE | 净盈亏 |")
        add("|---|---:|---:|---:|---:|---:|")
        groups = [
            ("全部", valid),
            ("< 1.0×（噪音区）", below_1),
            ("2.0 – 3.5×（外部推荐高原）", in_plateau),
            ("> 3.5×", valid[valid["atr_multiple"] > 3.5]),
        ]
        for label, group in groups:
            if group.empty:
                add(f"| {label} | 0 | – | – | – | – |")
                continue
            add(f"| {label} | {len(group)} | {group['realized_R'].mean():.2f} | "
                f"{group['realized_R'].sum():.2f} | {group['mfe_R'].mean():.2f} | "
                f"{group['net_pnl'].sum():,.0f} |")
        add("")
        corr = valid[["atr_multiple", "realized_R", "mfe_R", "mae_R"]].apply(
            pd.to_numeric, errors="coerce").corr(method="spearman")["atr_multiple"]
        add("Spearman 相关（与止损 ATR 倍数）：")
        add("")
        for name, value in corr.items():
            if name != "atr_multiple":
                add(f"- {name}: **{value:.3f}**")
        add("")

        # 目标距离：止损宽度 × 目标 R 倍数 —— 直接说明「止盈为什么够不到」
        valid_t = valid.copy()
        valid_t["target_rr"] = pd.to_numeric(valid_t["target_rr"], errors="coerce")
        valid_t = valid_t.dropna(subset=["target_rr"])
        if len(valid_t):
            valid_t["target_dist_pct"] = valid_t["stop_dist_pct"] * valid_t["target_rr"]
            valid_t["target_dist_atr"] = valid_t["atr_multiple"] * valid_t["target_rr"]
            add("## 目标距离：止盈到底要多远")
            add("")
            add("| 项目 | 中位数 | 最小 | 最大 |")
            add("|---|---:|---:|---:|")
            add(f"| 止损距离 (%价格) | {valid_t['stop_dist_pct'].median():.2f}% | "
                f"{valid_t['stop_dist_pct'].min():.2f}% | {valid_t['stop_dist_pct'].max():.2f}% |")
            add(f"| 目标 R 倍数 | {valid_t['target_rr'].median():.2f}R | "
                f"{valid_t['target_rr'].min():.2f}R | {valid_t['target_rr'].max():.2f}R |")
            add(f"| **目标距离 (%价格)** | **{valid_t['target_dist_pct'].median():.2f}%** | "
                f"{valid_t['target_dist_pct'].min():.2f}% | {valid_t['target_dist_pct'].max():.2f}% |")
            add(f"| **目标距离 (×ATR)** | **{valid_t['target_dist_atr'].median():.1f}×** | "
                f"{valid_t['target_dist_atr'].min():.1f}× | {valid_t['target_dist_atr'].max():.1f}× |")
            add("")
            typical_move = valid["mfe_R"].median() * valid_t["stop_dist_pct"].median()
            add(f"- 实测**典型有利波动**（中位 MFE {valid['mfe_R'].median():.2f}R × 中位止损 "
                f"{valid_t['stop_dist_pct'].median():.2f}%）≈ **{typical_move:.2f}% 价格移动**")
            add(f"- 配置的**目标要求**中位 **{valid_t['target_dist_pct'].median():.2f}% 价格移动** "
                f"→ 是典型波动的 **{valid_t['target_dist_pct'].median()/typical_move:.1f} 倍**")
            add("")
        add("## 判读")
        add("")
        median_mult = q.median()
        if median_mult < 1.0:
            add(f"止损中位数 **{median_mult:.2f}×ATR 落在噪音区（<1.0×）**，"
                "按外部证据该区间 >65% 会被噪音打掉 —— 与「0/21 止盈、几乎全部止损出场」一致。")
        elif median_mult < 2.0:
            add(f"止损中位数 **{median_mult:.2f}×ATR 低于外部推荐高原（2.0–3.5×）**，"
                "噪音触发率约 38–50%，仍有相当比例属噪音打掉。")
        elif median_mult <= 3.5:
            add(f"止损中位数 **{median_mult:.2f}×ATR 落在外部推荐高原（2.0–3.5×）内**，"
                "止损宽度**不是**主要问题，问题应在入场质量或目标设置。")
        else:
            add(f"止损中位数 **{median_mult:.2f}×ATR 宽于外部推荐高原**，"
                "盈亏比被压缩，需检查是否止损过宽。")
        add("")
        add("> 外部证据来源见 `research/smc-btc-perp-strategy-review.md`：")
        add("> arXiv:2602.11708（加密永续 ATR 止损高原）、Volatility Box（噪音触发率）。")
        add("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
