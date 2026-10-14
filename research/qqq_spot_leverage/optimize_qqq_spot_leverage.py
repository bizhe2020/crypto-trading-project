#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QQQ 现货杠杆策略 — 迭代优化实验台。

围绕三个已定位的弱点做受控实验（全部沿用 backtest 模块的无前视口径）：

  A. 再入场：MA200 下方出现修复信号时，用小仓位提前参与（针对 2020 V 型反转错过）
  B. VIX 闸门迟滞：降档需要阈值缓冲 / 连续确认（针对 VIX 在阈值附近抖动）
  C. 仓位管理：0.5 死区、recovery 档位、防御档位

筛选纪律：
  1. 全样本指标（累计/回撤/Sharpe/Calmar/换手）；
  2. 训练/测试拆分：train = 2020-01-02 ~ 2022-12-31，test = 2023-01-01 ~ 2026-08-14；
     只在 train 上调参，用 test 做样本外校验；
  3. 参数邻域稳定性（±20% 邻域内不应剧烈变差）。

输出：out/optimization_results.csv + out/optimization_report.json + 控制台对比表
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_qqq_spot_leverage import (  # noqa: E402
    Params,
    add_indicators,
    prepare_indicators,
    build_target_series,
    buy_hold,
    load_market_frame,
    metrics,
    run_engine,
)

OUT_DIR = Path(__file__).resolve().parent / "out"
FULL_END = "2026-08-14"
TRAIN_END = "2022-12-31"
WARMUP_START = "2001-01-02"


def _json_default(obj: object) -> object:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    raise TypeError(f"not JSON serializable: {type(obj)}")


def evaluate(base_ind: pd.DataFrame, p: Params, window: tuple[str, str] | None = None) -> dict:
    ind = base_ind
    if window is not None:
        ind = ind.loc[(ind.index >= pd.Timestamp(window[0])) & (ind.index <= pd.Timestamp(window[1]))]
    target = build_target_series(ind, p)
    run = run_engine(ind, target, p, "eval")
    m = metrics(run["path"])
    return m, run["path"]


def evaluate_config(frame: pd.DataFrame, p: Params, label: str) -> dict:
    """在 warmup 框上算指标；用（fast,slow）分档缓存 add_indicators 结果。"""
    ind = prepare_indicators(frame, p)
    row: dict = {"label": label, **{f"p_{k}": v for k, v in p.__dict__.items()}}
    full = ind.loc[ind.index >= pd.Timestamp("2020-01-02")]
    m, path = evaluate(full, p)
    row.update({f"full_{k}": v for k, v in m.items()})
    train = ind.loc[(ind.index >= pd.Timestamp("2020-01-02")) & (ind.index <= pd.Timestamp(TRAIN_END))]
    mtr, _ = evaluate(train, p)
    row.update({f"train_{k}": v for k, v in mtr.items()})
    test = ind.loc[(ind.index > pd.Timestamp(TRAIN_END)) & (ind.index <= pd.Timestamp(FULL_END))]
    mte, _ = evaluate(test, p)
    row.update({f"test_{k}": v for k, v in mte.items()})
    return row


def build_configs() -> list[tuple[str, Params]]:
    base = Params()
    configs: list[tuple[str, Params]] = [("baseline", base)]

    # A. 再入场
    for lev in [0.25, 0.5, 1.0]:
        for ret20 in [0.0, 0.02, 0.04]:
            configs.append(
                (
                    f"A_recov lev={lev} ret20={ret20}",
                    replace(base, use_recovery=True, recovery_lev=lev, recovery_ret20_min=ret20),
                )
            )
    configs.append(
        ("A_recov lev=0.5 ret20=0.02 floor=0.05", replace(base, use_recovery=True, recovery_lev=0.5, recovery_ret20_min=0.02, recovery_floor_pct=0.05))
    )
    configs.append(
        ("A_recov lev=0.5 ret20=0.02 noFastFilter", replace(base, use_recovery=True, recovery_lev=0.5, recovery_ret20_min=0.02, recovery_require_above_fast_ma=False))
    )
    # A2. 再入场 + MA200 斜率护栏（避免持续熊市里接刀）
    for slope in [0.0, -0.02, -0.04]:
        configs.append(
            (
                f"A2_recov0.5+slope{slope:+.2f}",
                replace(base, use_recovery=True, recovery_lev=0.5, recovery_ret20_min=0.02, recovery_ma_slope_min=slope),
            )
        )

    # B. VIX 迟滞
    for hyst in [2.0, 4.0]:
        configs.append((f"B_vixHyst +{hyst:.0f}", replace(base, vix_hysteresis=hyst)))
    for cd in [2, 3]:
        configs.append((f"B_vixConfirm {cd}d", replace(base, vix_confirm_days=cd)))
    configs.append(("B_termBypass 40", replace(base, vix_term_bypass_above=40.0)))
    configs.append(("B_vixHyst+4 & termBypass40", replace(base, vix_hysteresis=4.0, vix_term_bypass_above=40.0)))

    # C. 仓位管理
    configs.append(("C_deadband0.5", replace(base, min_trade_delta=0.5)))
    configs.append(("C_defense0.0", replace(base, lev_defense=0.0)))
    configs.append(("C_defense0.5", replace(base, lev_defense=0.5)))
    configs.append(("C_vixNeutral25", replace(base, vix_neutral=25.0)))
    configs.append(("C_vixStress30", replace(base, vix_stress=30.0)))
    configs.append(("C_vixStress26", replace(base, vix_stress=26.0)))

    # 组合（A+B / A+C / A+B+C）
    configs.append(("AB_recov0.5+hyst4", replace(base, use_recovery=True, recovery_lev=0.5, recovery_ret20_min=0.02, vix_hysteresis=4.0)))
    configs.append(("AB_recov0.5+confirm2", replace(base, use_recovery=True, recovery_lev=0.5, recovery_ret20_min=0.02, vix_confirm_days=2)))
    configs.append(("AC_recov0.5+deadband0.5", replace(base, use_recovery=True, recovery_lev=0.5, recovery_ret20_min=0.02, min_trade_delta=0.5)))
    configs.append(("ABC_recov0.5+hyst4+deadband0.5", replace(base, use_recovery=True, recovery_lev=0.5, recovery_ret20_min=0.02, vix_hysteresis=4.0, min_trade_delta=0.5)))
    configs.append(("ABC_recov0.5+confirm2+deadband0.5", replace(base, use_recovery=True, recovery_lev=0.5, recovery_ret20_min=0.02, vix_confirm_days=2, min_trade_delta=0.5)))

    # 最终候选：VIX 迟滞 + 斜率护栏再入场 + 死区
    for lev in [0.25, 0.5, 0.75, 1.0]:
        configs.append(
            (
                f"FINAL recov{lev}+hyst4+slope-2%",
                replace(base, use_recovery=True, recovery_lev=lev, recovery_ret20_min=0.02, recovery_ma_slope_min=-0.02, vix_hysteresis=4.0, min_trade_delta=0.5),
            )
        )
    configs.append(
        ("FINAL recov0.5+hyst4+slope-2%+volTarget20", replace(base, use_recovery=True, recovery_lev=0.5, recovery_ret20_min=0.02, recovery_ma_slope_min=-0.02, vix_hysteresis=4.0, min_trade_delta=0.5, vol_target=0.20))
    )
    # 斜率护栏单独作用（不加 recovery）：检验它是否自身有价值
    configs.append(("slopeGuard-only (no recovery)", replace(base, vix_hysteresis=4.0, min_trade_delta=0.5)))
    return configs


def main() -> None:
    ap = argparse.ArgumentParser(description="QQQ spot leverage optimization lab")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_market_frame(WARMUP_START, FULL_END)
    configs = build_configs()

    # (fast,slow) 组合不多，add_indicators 结果按组合缓存
    rows = []
    for label, p in configs:
        rows.append(evaluate_config(frame, p, label))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "optimization_results.csv", index=False)

    base = df[df["label"] == "baseline"].iloc[0]
    bench_ind = prepare_indicators(frame, Params()).loc[lambda d: d.index >= pd.Timestamp("2020-01-02")]
    bench = metrics(buy_hold(bench_ind, 1.0, Params(), "bh")["path"])

    rank_cols = [
        "label", "full_total_return_pct", "full_cagr_pct", "full_max_drawdown_pct",
        "full_sharpe", "full_calmar", "full_avg_leverage", "full_rebalances",
        "train_cagr_pct", "train_max_drawdown_pct", "train_calmar",
        "test_cagr_pct", "test_max_drawdown_pct", "test_calmar", "test_sharpe",
    ]
    ranked = df[rank_cols].copy()
    ranked["delta_calmar_vs_base"] = ranked["full_calmar"] - float(base["full_calmar"])
    ranked["delta_cagr_vs_base"] = ranked["full_cagr_pct"] - float(base["full_cagr_pct"])
    ranked["delta_mdd_vs_base"] = ranked["full_max_drawdown_pct"] - float(base["full_max_drawdown_pct"])
    ranked = ranked.sort_values("full_calmar", ascending=False)
    ranked.to_csv(out_dir / "optimization_ranked.csv", index=False)

    payload = {
        "baseline": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in base.items()},
        "benchmark_1x": bench,
        "configs": json.loads(df.to_json(orient="records")),
    }
    (out_dir / "optimization_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    )

    pd.set_option("display.width", 200)
    print("基准（买入持有 1x）:", {k: bench[k] for k in ["cagr_pct", "max_drawdown_pct", "sharpe", "calmar"]})
    print()
    print("按全样本 Calmar 排序（前 20）：")
    print(ranked.head(20).to_string(index=False))
    print()
    print("按样本外(test 2023+) Calmar 排序（前 12）：")
    print(ranked.sort_values("test_calmar", ascending=False).head(12).to_string(index=False))
    print(f"\n输出: {out_dir}/optimization_results.csv, optimization_ranked.csv, optimization_report.json")


if __name__ == "__main__":
    main()
