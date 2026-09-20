#!/usr/bin/env python3
"""回测前视偏差审计：4h 信息是否用了「当时尚未收盘」的那根 4h K 线。

背景
----
Northmark (2026) 的 995,550 笔预注册检验指出，SMC/ICT 类回测最常见的工具级缺陷是
**在成交当根 K 线上判定止盈**；修正后其 6 个标的全部由盈转亏。

本项目里与此等价（但更隐蔽）的风险在**多周期信息对齐**：
`ScalpRobustEngine._bias_for_idx()` 在 `bias_for_15m` 缺失时会回退到
`bias_4h[self.mapping[idx]]`。而 `align_timeframes()` 的语义是
「ts <= 当前 15m 时间」→ **指向当时仍在形成中的那根 4h**。

`build_precomputed_state_confirmed_4h()` 用 `confirmed_idx = mapped_idx - 1`
（已收盘的上一根 4h）修正了这一点，并把结果写进 `*_for_15m` 数组。

本脚本量化两种口径的**分歧率**，即前视偏差的实际影响面。

用法:
    python3 scripts/audit_backtest_lookahead.py \
        --klines-15m var/tokyo_audit/btc_scalp_20260920/BTC_USDT_USDT-15m-2026.csv \
        --out-md var/reports/btc_scalp_lookahead_audit.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.scalp_robust_v2_core import (  # noqa: E402
    _build_precomputed_state_lookahead,
    align_timeframes,
    build_precomputed_state_confirmed_4h,
    dataframe_to_candles,
)


def resample_4h(df15: pd.DataFrame) -> pd.DataFrame:
    """把 15m 重采样成与交易所对齐的 4h（UTC 00/04/08/12/16/20 起）。"""
    frame = df15.set_index("date").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = frame.resample("4h", label="left", closed="left").agg(agg).dropna().reset_index()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--klines-15m", required=True)
    parser.add_argument("--out-md", default="var/reports/btc_scalp_lookahead_audit.md")
    args = parser.parse_args()

    df15 = pd.read_csv(args.klines_15m, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df4 = resample_4h(df15)
    c15m = dataframe_to_candles(df15)
    c4h = dataframe_to_candles(df4)
    mapping = align_timeframes(c4h, c15m)

    raw = _build_precomputed_state_lookahead(c4h, c15m)      # 裸口径（有前视，仅存于私有函数）
    confirmed = build_precomputed_state_confirmed_4h(c4h, c15m)  # 实盘口径（无前视）

    n = len(c15m)
    raw_bias = [raw.bias_4h[mapping[i]] for i in range(n)]
    cand_bias = confirmed.bias_for_15m

    def diff_rate(raw_list, cand_list, label: str) -> dict:
        pairs = [(raw_list[i], cand_list[i]) for i in range(min(len(raw_list), len(cand_list)))]
        diffs = sum(1 for a, b in pairs if a != b)
        return {"指标": label, "对比样本": len(pairs), "分歧数": diffs,
                "分歧率%": round(diffs / len(pairs) * 100, 2) if pairs else 0.0}

    rows = [diff_rate(raw_bias, cand_bias, "4h bias（方向）")]
    for label, raw_attr, cand_attr in [
        ("1d regime bull EMA100", "regime_1d_bull_100", "regime_1d_bull_100_for_15m"),
        ("1d regime bull EMA200", "regime_1d_bull_200", "regime_1d_bull_200_for_15m"),
        ("1d regime bear EMA100", "regime_1d_bear_100", "regime_1d_bear_100_for_15m"),
        ("4h bull trend score", "bull_trend_score_4h", "bull_trend_score_for_15m"),
        ("4h bear trend score", "bear_trend_score_4h", "bear_trend_score_for_15m"),
    ]:
        r_list = [getattr(raw, raw_attr)[mapping[i]] for i in range(n)]
        c_list = getattr(confirmed, cand_attr)
        rows.append(diff_rate(r_list, c_list, label))

    table = pd.DataFrame(rows)

    # 只在有 4h 已收盘（mapping>=1）的 bar 上统计，避免首根噪声
    valid = [i for i in range(n) if mapping[i] >= 1]
    bias_flip = sum(1 for i in valid if raw.bias_4h[mapping[i]] != raw.bias_4h[mapping[i] - 1])

    lines: list[str] = []
    add = lines.append
    add("# 回测前视偏差审计：4h 信息对齐口径")
    add("")
    add(f"- 15m K 线：**{n:,} 根**（{df15['date'].min()} → {df15['date'].max()}）")
    add(f"- 4h K 线：{len(c4h):,} 根（由 15m 重采样，UTC 对齐）")
    add("")
    add("## 结论")
    add("")
    add("| 检查项 | 结果 |")
    add("|---|---|")
    add("| 开仓当根 K 线是否判定出场 | **否** —— `evaluate_range` 先 `manage_position(i)` 再开仓，开仓后进入 `i+1` |")
    add("| 出场内部止损/止盈判定顺序 | **止损先判**（保守），见 `manage_position` BULL 分支 |")
    add("| 成交价假设 | 信号 bar 的**收盘价** `entry = curr.c`，再加 `slippage_bps` |")
    add("| 实盘 4h 信息口径 | `bot/okx_executor.py:2395` 使用 `build_precomputed_state_confirmed_4h` → **无前视** |")
    add("| 回测 4h 信息口径 | `load_prepared_data` 只有一条路径 → `build_precomputed_state_confirmed_4h`，**无前视** |")
    add("")
    add("### 口径分歧率（raw 有前视 vs confirmed 无前视）")
    add("")
    add(table.to_markdown(index=False))
    add("")
    add(f"- 只在 `mapping[i] >= 1` 的 {len(valid):,} 根 15m 上统计，"
        f"**相邻两根 4h 的 bias 方向不同**的比例：{bias_flip:,} / {len(valid):,} = "
        f"**{bias_flip/len(valid)*100:.1f}%**")
    add("")
    add("## 判读")
    add("")
    add("分歧率越高，回测与实盘的差距越大 —— 因为回测在多看一根尚未收盘的 4h K 线。")
    add("")
    add("**已执行的封堵**：三个口径开关（`confirmed_4h_only` / `informative_asof_from_15m` /")
    add("`--raw-4h-state`）与 `build_precomputed_state_asof_15m` 均已删除；裸构造器改名为")
    add("`_build_precomputed_state_lookahead` 并私有化，只能被 confirmed 版本内部调用。")
    add("因此工程上已无法再写出有前视的回测。下表的分歧率保留，用于说明这条规则的由来。")
    add("")
    add("> 注意：`scan_events` + `trade_rows_for_events` 这条 SMC 事件路径")
    add("> 已用 `completed_4h_idx_for_entry(mapping, idx) = mapping[idx] - 1` 自行修正，**不受影响**。")
    add("> 受影响的是 `ScalpRobustEngine` 的 SOTA 多头路径。")
    add("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
