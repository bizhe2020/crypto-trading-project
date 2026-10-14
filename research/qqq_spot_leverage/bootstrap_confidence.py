#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计置信区间（round 5）：stationary bootstrap（Politis & Romano 1994）。

6.6 年样本只有一个点估计（CAGR 44.3% / Calmar 2.75），无法判断这是运气还是能力。
本脚本对**日收益序列**做 stationary bootstrap（几何块长，默认均值 20 天），
重采样 2000 条路径，得到 CAGR / 最大回撤 / Sharpe / Calmar 的分布，
并与"买入持有 1.5x"做配对比较（同一条重采样路径上对比）。

注意（口径）：
  * 重采样的是"策略日收益"，因此 **回撤/波动/Calmar 的分布有意义**；
  * 但趋势策略的收益有强自相关，bootstrap 会破坏长期趋势结构，
    所以 **CAGR 分布偏保守**（把趋势打散后复利会变差）——报告里据此给出区间而非点估计。
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_qqq_spot_leverage import (  # noqa: E402
    Params,
    build_target_series,
    load_market_frame,
    metrics,
    prepare_indicators,
    run_engine,
)

OUT = Path(__file__).resolve().parent / "out"
WINDOW = ("2020-01-02", "2026-08-14")
N_PATHS = 2000
MEAN_BLOCK = 20
SEED = 20260922


def stationary_bootstrap_indices(n: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap：几何块长、环绕采样。"""
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(0, n)
    for i in range(1, n):
        if rng.random() < p:
            idx[i] = rng.integers(0, n)
        else:
            idx[i] = (idx[i - 1] + 1) % n
    return idx


def path_metrics_from_returns(rets: np.ndarray) -> dict:
    eq = np.cumprod(1.0 + rets)
    n = len(rets)
    years = n / 252.0
    total = float(eq[-1] - 1.0)
    cagr = float(eq[-1] ** (1.0 / years) - 1.0) if years > 0 and eq[-1] > 0 else -1.0
    vol = float(rets.std(ddof=1) * np.sqrt(252)) if n > 2 else 0.0
    sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(252)) if n > 2 and rets.std(ddof=1) > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    return {"total": total, "cagr": cagr, "vol": vol, "sharpe": sharpe, "mdd": mdd, "calmar": calmar}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_market_frame("2001-01-02", WINDOW[1])
    p = Params()
    ind = prepare_indicators(frame, p).loc[WINDOW[0]:WINDOW[1]]

    # 策略收益 & 基准收益
    strat_path = run_engine(ind, build_target_series(ind, p), p, "main")["path"]
    strat_ret = strat_path["net_ret"].to_numpy(dtype=float)
    bench_target = pd.Series(1.5, index=ind.index)
    bench_path = run_engine(ind, bench_target, replace(p, use_vix_gate=False, use_recovery=False, trail_drawdown=0.0), "bench")["path"]
    bench_ret = bench_path["net_ret"].to_numpy(dtype=float)

    n = len(strat_ret)
    point = {
        "strategy": metrics(strat_path),
        "benchmark_1.5x": metrics(bench_path),
    }

    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(N_PATHS):
        idx = stationary_bootstrap_indices(n, MEAN_BLOCK, rng)
        ms = path_metrics_from_returns(strat_ret[idx])
        mb = path_metrics_from_returns(bench_ret[idx])
        rows.append(
            {
                "s_total": ms["total"], "s_cagr": ms["cagr"], "s_vol": ms["vol"],
                "s_sharpe": ms["sharpe"], "s_mdd": ms["mdd"], "s_calmar": ms["calmar"],
                "b_total": mb["total"], "b_cagr": mb["cagr"], "b_mdd": mb["mdd"], "b_calmar": mb["calmar"],
                "win": ms["calmar"] > mb["calmar"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bootstrap_paths.csv", index=False)

    def q(s: pd.Series) -> dict:
        return {
            "p05": round(float(s.quantile(0.05)), 4),
            "p25": round(float(s.quantile(0.25)), 4),
            "median": round(float(s.quantile(0.50)), 4),
            "p75": round(float(s.quantile(0.75)), 4),
            "p95": round(float(s.quantile(0.95)), 4),
            "mean": round(float(s.mean()), 4),
        }

    summary = {
        "n_days": n,
        "n_paths": N_PATHS,
        "mean_block": MEAN_BLOCK,
        "seed": SEED,
        "point_estimate": {
            "strategy_cagr_pct": point["strategy"]["cagr_pct"],
            "strategy_mdd_pct": point["strategy"]["max_drawdown_pct"],
            "strategy_sharpe": point["strategy"]["sharpe"],
            "strategy_calmar": point["strategy"]["calmar"],
            "bench_cagr_pct": point["benchmark_1.5x"]["cagr_pct"],
            "bench_mdd_pct": point["benchmark_1.5x"]["max_drawdown_pct"],
            "bench_sharpe": point["benchmark_1.5x"]["sharpe"],
            "bench_calmar": point["benchmark_1.5x"]["calmar"],
        },
        "strategy": {
            "cagr": q(df["s_cagr"]), "mdd": q(df["s_mdd"]),
            "sharpe": q(df["s_sharpe"]), "calmar": q(df["s_calmar"]), "vol": q(df["s_vol"]),
        },
        "benchmark_1.5x": {
            "cagr": q(df["b_cagr"]), "mdd": q(df["b_mdd"]), "calmar": q(df["b_calmar"]),
        },
        "prob_strategy_calmar_gt_bench": round(float(df["win"].mean()), 4),
        "prob_strategy_cagr_gt_bench": round(float((df["s_cagr"] > df["b_cagr"]).mean()), 4),
        "prob_cagr_positive": round(float((df["s_cagr"] > 0).mean()), 4),
        "prob_calmar_gt_1": round(float((df["s_calmar"] > 1.0).mean()), 4),
    }
    (OUT / "bootstrap_confidence.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    pd.set_option("display.width", 220)
    print(f"窗口 {WINDOW[0]} ~ {WINDOW[1]}（{n} 个交易日），stationary bootstrap {N_PATHS} 条路径，平均块长 {MEAN_BLOCK}")
    print()
    print("点估计：")
    pe = summary["point_estimate"]
    print(f"  策略   CAGR {pe['strategy_cagr_pct']:>6.2f}%  MDD {pe['strategy_mdd_pct']:>7.2f}%  Sharpe {pe['strategy_sharpe']:.3f}  Calmar {pe['strategy_calmar']:.3f}")
    print(f"  基准1.5 CAGR {pe['bench_cagr_pct']:>6.2f}%  MDD {pe['bench_mdd_pct']:>7.2f}%  Sharpe {pe['bench_sharpe']:.3f}  Calmar {pe['bench_calmar']:.3f}")
    print()
    for key, label, scale in [("cagr", "CAGR %", 100), ("mdd", "MDD %", 100), ("sharpe", "Sharpe", 1), ("calmar", "Calmar", 1)]:
        s = summary["strategy"][key]
        print(f"  策略 {label:<8} p05 {s['p05']*scale:>7.2f} | p25 {s['p25']*scale:>7.2f} | 中位 {s['median']*scale:>7.2f} | p75 {s['p75']*scale:>7.2f} | p95 {s['p95']*scale:>7.2f}")
    print()
    b = summary["benchmark_1.5x"]
    print(f"  基准1.5 CAGR   p05 {b['cagr']['p05']*100:>7.2f} | 中位 {b['cagr']['median']*100:>7.2f} | p95 {b['cagr']['p95']*100:>7.2f}")
    print(f"  基准1.5 MDD    p05 {b['mdd']['p05']*100:>7.2f} | 中位 {b['mdd']['median']*100:>7.2f} | p95 {b['mdd']['p95']*100:>7.2f}")
    print()
    print(f"  P(策略 Calmar > 基准1.5x) = {summary['prob_strategy_calmar_gt_bench']:.1%}")
    print(f"  P(策略 CAGR   > 基准1.5x) = {summary['prob_strategy_cagr_gt_bench']:.1%}")
    print(f"  P(策略 CAGR > 0)          = {summary['prob_cagr_positive']:.1%}")
    print(f"  P(策略 Calmar > 1.0)      = {summary['prob_calmar_gt_1']:.1%}")
    print(f"\n输出: {OUT}/bootstrap_confidence.json, bootstrap_paths.csv")


if __name__ == "__main__":
    main()
