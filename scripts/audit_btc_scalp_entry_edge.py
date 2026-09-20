#!/usr/bin/env python3
"""入场边际检验：真实入场 vs 随机对照组。

外部文献（Northmark 2026，995,550 笔预注册检验）用「随机入场 + 匹配止损距离」作对照，
发现 SMC/ICT 形态的胜率与随机几乎相同甚至更差。本脚本在**你自己的 21 个真实入场**上
做同样的检验：

    对每个真实入场，在同一时间区间内随机抽取 K 个「假入场」，
    方向随机、止损宽度从真实分布中重采样，然后用**完全相同的出场规则**结算。
    比较真实入场的平均 R 与随机分布，得到经验 p 值。

如果真实入场的表现落在随机分布的中间，说明入场信号没有可测量的边际优势。

用法:
    python3 scripts/audit_btc_scalp_entry_edge.py \
        --trades var/tokyo_audit/btc_scalp_20260920/trades_audit.csv \
        --klines var/tokyo_audit/btc_scalp_20260920/BTC_USDT_USDT-15m-2026.csv \
        --simulations 300 \
        --out-md var/reports/btc_scalp_entry_edge.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audit_btc_scalp_exit_ablation import Rule, simulate_trade

# 用于对照的中性出场规则（真实与随机样本完全一致）
CONTROL_RULE = Rule(
    "对照规则：1R 启动 0.5R 追踪",
    trail_start_r=1.0,
    trail_dist_r=0.5,
)


def simulate_real(frame: pd.DataFrame, klines: pd.DataFrame, rule: Rule,
                  max_bars: int) -> np.ndarray:
    return np.array([
        simulate_trade(row, klines, rule, max_bars, pessimistic=True)["R"]
        for _, row in frame.iterrows()
    ], dtype=float)


def simulate_random(frame: pd.DataFrame, klines: pd.DataFrame, rule: Rule,
                    max_bars: int, rng: np.random.Generator) -> np.ndarray:
    """随机对照组：随机入场时间 + 随机方向 + 重采样止损宽度。"""
    results = []
    bar_index = {ts: i for i, ts in enumerate(klines["date"])}
    times = klines["date"].to_numpy()
    n_bars = len(klines)
    stop_pcts = pd.to_numeric(frame["risk_pct"], errors="coerce").dropna().to_numpy()

    for _ in range(len(frame)):
        idx = int(rng.integers(0, n_bars - max_bars - 1))
        entry_bar = klines.iloc[idx]
        stop_pct = float(rng.choice(stop_pcts)) / 100.0
        direction = "BULL" if rng.random() < 0.5 else "BEAR"
        entry = float(entry_bar["close"])
        stop = entry * (1 - stop_pct) if direction == "BULL" else entry * (1 + stop_pct)
        row = pd.Series({
            "entry_signal": entry,
            "stop": stop,
            "direction": direction,
            "open_ts": pd.Timestamp(times[idx]),
        })
        results.append(simulate_trade(row, klines, rule, max_bars, pessimistic=True)["R"])
    return np.array(results, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--klines", required=True)
    parser.add_argument("--simulations", type=int, default=300)
    parser.add_argument("--max-bars", type=int, default=144)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--out-md", default="var/reports/btc_scalp_entry_edge.md")
    args = parser.parse_args()

    frame = pd.read_csv(args.trades)
    frame["open_ts"] = pd.to_datetime(frame["open_ts"], utc=True)
    frame = frame[frame["stop"].notna()].reset_index(drop=True)
    klines = pd.read_csv(args.klines, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    real_r = simulate_real(frame, klines, CONTROL_RULE, args.max_bars)
    rng = np.random.default_rng(args.seed)
    random_means = np.empty(args.simulations, dtype=float)
    random_winrates = np.empty(args.simulations, dtype=float)
    for i in range(args.simulations):
        draw = simulate_random(frame, klines, CONTROL_RULE, args.max_bars, rng)
        random_means[i] = np.nanmean(draw)
        random_winrates[i] = np.nanmean(draw > 0) * 100

    real_mean = float(np.nanmean(real_r))
    real_win = float(np.nanmean(real_r > 0) * 100)
    p_value = float((random_means >= real_mean).mean())

    lines: list[str] = []
    add = lines.append
    add("# BTC Scalp 入场边际检验：真实入场 vs 随机对照组")
    add("")
    add(f"- 真实入场：**{len(frame)} 笔**（来自实盘 action_log）")
    add(f"- 对照规则（真实与随机完全一致）：{CONTROL_RULE.name}")
    add(f"- 随机对照：**{args.simulations:,} 轮**，每轮 {len(frame)} 个随机入场"
        "（随机时间 + 随机方向 + 从真实分布重采样止损宽度）")
    add(f"- 最长持仓：{args.max_bars} 根 15m K")
    add("")
    add("## 结果")
    add("")
    add("| 指标 | 真实入场 | 随机对照（均值） | 随机对照（5%~95%） |")
    add("|---|---:|---:|---:|")
    add(f"| 平均 R | **{real_mean:.3f}** | {random_means.mean():.3f} | "
        f"{np.percentile(random_means, 5):.3f} ~ {np.percentile(random_means, 95):.3f} |")
    add(f"| 胜率 | **{real_win:.1f}%** | {random_winrates.mean():.1f}% | "
        f"{np.percentile(random_winrates, 5):.1f}% ~ {np.percentile(random_winrates, 95):.1f}% |")
    add("")
    add(f"**经验 p 值（随机对照组平均 R ≥ 真实平均 R 的比例）= {p_value:.3f}**")
    add("")
    if p_value > 0.5:
        add("解读：真实入场的表现**差于或等于随机入场**，入场信号没有可测量的边际优势。")
    elif p_value > 0.1:
        add("解读：真实入场与随机入场**无法区分**（p > 0.1），样本量下看不出入场优势。")
    else:
        add("解读：真实入场**显著优于**随机对照，入场信号存在可测量的优势。")
    add("")
    add("## 说明")
    add("")
    add("- 随机对照与真实样本使用同一段行情、同一出场规则、同一止损宽度分布，"
        "因此差异只能归因于「入场时机与方向的选择」。")
    add("- 样本量仅 21 笔，检验功效有限；p 值 > 0.1 应读作「没有证据表明有优势」，"
        "而不是「证明没有优势」。")
    add("- 结果未扣手续费；真实交易每笔另有约 0.116R 成本，随机对照不含成本。")
    add("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
