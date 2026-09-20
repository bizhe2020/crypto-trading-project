#!/usr/bin/env python3
"""BTC scalp 出场规则 / 风险预算反事实消融。

思路：**固定 21 个真实入场点不变**，只替换出场规则与仓位风险预算，看结果如何变化。
这样可以把「入场质量」与「出场/风控政策」的贡献彻底分开 —— 如果换出场规则就能显著改善，
说明问题在政策而不在信号。

数据：scripts/audit_btc_scalp_live_history.py 产出的 trades_audit.csv + 15m K 线。

路径假设（重要）：
  15m OHLC 无法确定 bar 内价格先后顺序。当同一根 bar 同时触及止损与止盈时：
    - 基准（保守）：判定为止损先成交
    - 乐观边界：判定为止盈先成交
  两个结果一起看，避免把结论建立在路径假设上。

用法:
    python3 scripts/audit_btc_scalp_exit_ablation.py \
        --trades var/tokyo_audit/btc_scalp_20260920/trades_audit.csv \
        --klines var/tokyo_audit/btc_scalp_20260920/BTC_USDT_USDT-15m-2026.csv \
        --out-md var/reports/btc_scalp_exit_ablation.md \
        --out-csv var/tokyo_audit/btc_scalp_20260920/exit_ablation.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule:
    """一条出场规则。R 单位均相对初始止损距离。"""

    name: str
    tp_r: float | None = None          # 固定止盈 R；None = 不止盈
    be_trigger_r: float | None = None  # 达到该浮盈后把止损移到保本
    trail_start_r: float | None = None # 达到该浮盈后启动追踪
    trail_dist_r: float | None = None  # 追踪止损距离（R）
    time_stop_bars: int | None = None  # 最长持仓 bar 数
    stall_bars: int | None = None      # 停滞判定窗口
    stall_min_r: float | None = None   # 窗口结束时浮盈低于该值则离场
    description: str = ""


def simulate_trade(row: pd.Series, klines: pd.DataFrame, rule: Rule,
                   max_bars: int, pessimistic: bool) -> dict:
    entry = float(row["entry_signal"])
    stop = float(row["stop"])
    if not np.isfinite(entry) or not np.isfinite(stop) or entry == stop:
        return {"R": np.nan, "bars": np.nan, "exit_kind": "invalid"}
    is_long = str(row["direction"]) == "BULL"
    risk = abs(entry - stop)
    sign = 1.0 if is_long else -1.0

    window = klines[klines["date"] >= row["open_ts"]].head(max_bars)
    if window.empty:
        return {"R": np.nan, "bars": np.nan, "exit_kind": "no_data"}

    current_stop = stop
    best_r = 0.0
    for idx, bar in enumerate(window.itertuples(index=False)):
        high, low, close = float(bar.high), float(bar.low), float(bar.close)

        # 1) 先判断止损（保守假设下优先）
        stop_hit = (low <= current_stop) if is_long else (high >= current_stop)

        # 2) 止盈
        tp_price = entry + sign * rule.tp_r * risk if rule.tp_r is not None else None
        tp_hit = False
        if tp_price is not None:
            tp_hit = (high >= tp_price) if is_long else (low <= tp_price)

        if stop_hit and tp_hit:
            if pessimistic:
                return {"R": (current_stop - entry) * sign / risk, "bars": idx + 1,
                        "exit_kind": "stop_and_tp_same_bar"}
            return {"R": rule.tp_r, "bars": idx + 1, "exit_kind": "target"}
        if stop_hit:
            return {"R": (current_stop - entry) * sign / risk, "bars": idx + 1,
                    "exit_kind": "breakeven" if abs(current_stop - entry) < 1e-9 else "stop"}
        if tp_hit:
            return {"R": rule.tp_r, "bars": idx + 1, "exit_kind": "target"}

        # 3) 停滞离场：窗口结束时浮盈仍不足，说明这笔交易没走出来
        if (rule.stall_bars is not None and rule.stall_min_r is not None
                and idx + 1 == rule.stall_bars):
            current_r = (close - entry) * sign / risk
            if current_r < rule.stall_min_r:
                return {"R": current_r, "bars": idx + 1, "exit_kind": "stall_exit"}

        # 4) 更新浮盈与追踪止损
        excursion_r = ((high - entry) if is_long else (entry - low)) / risk
        best_r = max(best_r, excursion_r)
        if rule.be_trigger_r is not None and best_r >= rule.be_trigger_r:
            candidate = entry
            current_stop = max(current_stop, candidate) if is_long else min(current_stop, candidate)
        if (rule.trail_start_r is not None and rule.trail_dist_r is not None
                and best_r >= rule.trail_start_r):
            if is_long:
                current_stop = max(current_stop, entry + (best_r - rule.trail_dist_r) * risk)
            else:
                current_stop = min(current_stop, entry - (best_r - rule.trail_dist_r) * risk)

        if rule.time_stop_bars is not None and idx + 1 >= rule.time_stop_bars:
            return {"R": (close - entry) * sign / risk, "bars": idx + 1, "exit_kind": "time_stop"}

    last_close = float(window.iloc[-1]["close"])
    return {"R": (last_close - entry) * sign / risk, "bars": len(window), "exit_kind": "max_bars"}


def build_rules() -> list[Rule]:
    return [
        Rule("实际历史（基准）", description="机器人真实出场结果，用于对照"),
        Rule("固定 1R 止盈", tp_r=1.0, description="到 +1R 就走，止损不动"),
        Rule("固定 1.5R 止盈", tp_r=1.5),
        Rule("固定 2R 止盈", tp_r=2.0),
        Rule("固定 2.5R 止盈", tp_r=2.5),
        Rule("固定 3R 止盈", tp_r=3.0),
        Rule("固定 5.5R 止盈（当前配置）", tp_r=5.5),
        Rule("2R 止盈 + 1R 后保本", tp_r=2.0, be_trigger_r=1.0),
        Rule("3R 止盈 + 1R 后保本", tp_r=3.0, be_trigger_r=1.0),
        Rule("1.5R 启动 0.75R 追踪", trail_start_r=1.5, trail_dist_r=0.75),
        Rule("1R 启动 0.5R 追踪", trail_start_r=1.0, trail_dist_r=0.5),
        Rule("2R 启动 1R 追踪", trail_start_r=2.0, trail_dist_r=1.0),
        Rule("2R 止盈 + 48bar 时间止损", tp_r=2.0, time_stop_bars=48),
        Rule("1.5R 止盈 + 24bar 时间止损", tp_r=1.5, time_stop_bars=24),
        Rule("保本 + 1.5R 追踪（无固定止盈）", be_trigger_r=0.5, trail_start_r=1.0, trail_dist_r=0.5),
        Rule("停滞离场：12bar 未达 0.5R 则走", stall_bars=12, stall_min_r=0.5),
        Rule("停滞离场 + 1R 启动 0.5R 追踪", stall_bars=12, stall_min_r=0.5,
             trail_start_r=1.0, trail_dist_r=0.5),
        Rule("停滞离场(24bar) + 2R 止盈", stall_bars=24, stall_min_r=0.5, tp_r=2.0),
        Rule("停滞离场(12bar) + 1.5R 止盈", stall_bars=12, stall_min_r=0.5, tp_r=1.5),
    ]


def summarize(frame: pd.DataFrame, rules: list[Rule], klines: pd.DataFrame,
              max_bars: int, pessimistic: bool) -> pd.DataFrame:
    rows = []
    fee_r = pd.to_numeric(frame["fee_R"], errors="coerce").fillna(0.0).to_numpy()
    for rule in rules:
        if rule.name.startswith("实际历史"):
            r_values = pd.to_numeric(frame["realized_R"], errors="coerce").to_numpy()
            bars = pd.to_numeric(frame["bars_held"], errors="coerce").to_numpy()
            kinds = frame["reason"].fillna("actual").to_numpy()
        else:
            results = [simulate_trade(row, klines, rule, max_bars, pessimistic)
                       for _, row in frame.iterrows()]
            r_values = np.array([r["R"] for r in results], dtype=float)
            bars = np.array([r["bars"] for r in results], dtype=float)
            kinds = np.array([r["exit_kind"] for r in results])
        net_r = r_values - fee_r
        risk_amount = pd.to_numeric(frame["risk_amount"], errors="coerce").to_numpy()
        rows.append({
            "规则": rule.name,
            "总R": np.nansum(r_values),
            "平均R": np.nanmean(r_values),
            "扣费后平均R": np.nanmean(net_r),
            "扣费后总R": np.nansum(net_r),
            "胜率%": float(np.nanmean(r_values > 0) * 100) if len(r_values) else np.nan,
            "真实风险净盈亏USDT": np.nansum(r_values * risk_amount),
            "等风险复利终值": equal_risk_equity(net_r, 0.035),
            "等风险最大回撤%": equal_risk_drawdown(net_r, 0.035),
            "平均持仓bar": np.nanmean(bars),
            "触及止盈数": int(np.sum(kinds == "target")),
            "止损数": int(np.sum(np.isin(kinds, ["stop", "breakeven", "stop_and_tp_same_bar"]))),
            "时间/超时数": int(np.sum(np.isin(kinds, ["time_stop", "max_bars"]))),
            "停滞离场数": int(np.sum(kinds == "stall_exit")),
        })
    return pd.DataFrame(rows)


def equal_risk_equity(net_r: np.ndarray, risk_pct: float, start: float = 10000.0) -> float:
    equity = start
    for r in net_r:
        if np.isfinite(r):
            equity *= (1.0 + risk_pct * r)
    return equity


def equal_risk_drawdown(net_r: np.ndarray, risk_pct: float, start: float = 10000.0) -> float:
    equity = start
    peak = equity
    max_dd = 0.0
    for r in net_r:
        if not np.isfinite(r):
            continue
        equity *= (1.0 + risk_pct * r)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100.0)
    return max_dd


def flat_risk_equity(frame: pd.DataFrame, risk_pct: float, start_equity: float = 10000.0) -> tuple[float, float, pd.Series]:
    """用真实实现 R 序列 + 固定风险比例，复利推演权益曲线。"""
    r_values = pd.to_numeric(frame["realized_R"], errors="coerce").fillna(0.0).to_numpy()
    equity = start_equity
    curve = []
    peak = equity
    max_dd = 0.0
    for r in r_values:
        equity *= (1.0 + risk_pct * r)
        curve.append(equity)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100.0)
    return equity, max_dd, pd.Series(curve)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--klines", required=True)
    parser.add_argument("--out-md", default="var/reports/btc_scalp_exit_ablation.md")
    parser.add_argument("--out-csv", default="var/tokyo_audit/btc_scalp_20260920/exit_ablation.csv")
    parser.add_argument("--max-bars", type=int, default=144,
                        help="最长持仓 bar 数（默认 144 = 36 小时，与配置 T_max 一致）")
    args = parser.parse_args()

    frame = pd.read_csv(args.trades)
    frame["open_ts"] = pd.to_datetime(frame["open_ts"], utc=True)
    frame = frame[frame["stop"].notna()].reset_index(drop=True)
    klines = pd.read_csv(args.klines, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    # 风险金额：优先用机器人开仓时记录的实际风险金额，缺失时用 风险比例 × 入场资金
    risk_amount = pd.to_numeric(frame.get("risk_amount"), errors="coerce")
    if risk_amount is None:
        risk_amount = pd.Series(np.nan, index=frame.index)
    fallback = pd.to_numeric(frame["risk_per_trade"], errors="coerce") * pd.to_numeric(
        frame["capital_at_entry"], errors="coerce")
    frame["risk_amount"] = risk_amount.fillna(fallback).fillna(0.0)
    frame["fee_R"] = pd.to_numeric(frame.get("fees"), errors="coerce").abs() / frame["risk_amount"]

    rules = build_rules()
    pessimistic = summarize(frame, rules, klines, args.max_bars, pessimistic=True)
    optimistic = summarize(frame, rules, klines, args.max_bars, pessimistic=False)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged = pessimistic.merge(
        optimistic[["规则", "总R", "扣费后总R"]], on="规则", suffixes=("", "_乐观"))
    merged.to_csv(out_csv, index=False)

    fee_r_mean = float(pd.to_numeric(frame["fee_R"], errors="coerce").mean())
    fee_r_total = float(pd.to_numeric(frame["fee_R"], errors="coerce").sum())

    lines: list[str] = []
    add = lines.append
    add("# BTC Scalp 出场规则 / 风险预算反事实消融")
    add("")
    add(f"- 固定入场：**{len(frame)} 个真实入场点**（来自实盘 action_log）")
    add(f"- 最长持仓：{args.max_bars} 根 15m K（= {args.max_bars*15/60:.0f} 小时）")
    add("- 同一根 K 线同时触及止损与止盈时，保守列判定为止损先成交，乐观列判定为止盈先成交")
    add(f"- **手续费拖累：平均每笔 {fee_r_mean:.3f} R，21 笔合计 {fee_r_total:.2f} R**"
        "（这是判断任何改进是否够大的标尺）")
    add(f"- 对比基准「实际历史」：总 R = {pessimistic.iloc[0]['总R']:.2f}，"
        f"扣费后总 R = {pessimistic.iloc[0]['扣费后总R']:.2f}")
    add("")
    add("## 出场规则对比（保守路径假设）")
    add("")
    add(pessimistic.to_markdown(index=False, floatfmt=",.3f"))
    add("")
    add("## 路径假设敏感性（乐观 − 保守，总 R）")
    add("")
    sens = merged[["规则", "总R", "总R_乐观"]].copy()
    sens["差异"] = sens["总R_乐观"] - sens["总R"]
    add(sens.to_markdown(index=False, floatfmt=",.3f"))
    add("")

    add("## 风险预算：真实实现 R 序列 × 固定风险比例")
    add("")
    add("| 每笔风险 | 终值(起始10000) | 总收益% | 最大回撤% |")
    add("|---|---:|---:|---:|")
    for risk_pct in (0.01, 0.02, 0.035, 0.05, 0.06, 0.12):
        equity, max_dd, _ = flat_risk_equity(frame, risk_pct)
        add(f"| {risk_pct:.1%} | {equity:,.0f} | {(equity/10000-1)*100:+.1f}% | {max_dd:.1f}% |")
    add("")
    add("对照：实际历史按各笔真实 risk_per_trade 与真实 R 复利推演")
    equity, max_dd, _ = flat_risk_equity(frame, 1.0)
    add("")
    add(f"- 用真实逐笔风险比例复利：终值 {equity:,.0f}，最大回撤 {max_dd:.1f}%")
    add("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(pessimistic.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print()
    print(f"CSV:  {out_csv}")
    print(f"报告: {out_md}")


if __name__ == "__main__":
    main()
