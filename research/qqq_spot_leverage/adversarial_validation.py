#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对抗性验证 + 相对强弱实验（round 6）。

A. 对抗性验证（确认 v4 不是伪信号）
   A1 信号错位测试：把 target 序列整体平移 k 天（k=-3..3），k=0 应显著最好；
      若 k≠0 也好，说明信号位置不重要（伪信号）。
   A2 随机择时对照：用同分布（同样的仓位天数与档位分布）的随机 target 跑 200 次，
      看真实策略在分布中的分位。
   A3 打乱收益测试：把个股日收益顺序随机打乱（保留仓位序列），策略优势应消失。

B. 相对强弱（QQQ vs SPY）闸门实验
   数据源 etf_full 自带 SPY，覆盖 1999 起（价格口径，两侧一致）。
   候选：RS20 = QQQ 20日收益 − SPY 20日收益；RS_gate = 只在 RS 不弱时给满杠杆。
   评估：主区间/训练/测试/2011-2019/长样本 1999-2026。
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
MAIN = ("2020-01-02", "2026-08-14")
TRAIN = ("2020-01-02", "2022-12-31")
TEST = ("2023-01-01", "2026-08-14")
OOS = ("2011-04-26", "2019-12-31")
LONG = ("1999-03-10", "2026-05-29")


def window_metrics(ind: pd.DataFrame, p: Params, win: tuple[str, str]) -> dict:
    sub = ind.loc[(ind.index >= pd.Timestamp(win[0])) & (ind.index <= pd.Timestamp(win[1]))]
    m = metrics(run_engine(sub, build_target_series(sub, p), p, "w")["path"])
    return {"cagr": m["cagr_pct"], "mdd": m["max_drawdown_pct"], "sharpe": m["sharpe"], "calmar": m["calmar"],
            "reb": m["rebalances"]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_market_frame("2001-01-02", MAIN[1])
    frame_long = load_market_frame(LONG[0], LONG[1], price_source="feather")
    base = Params()
    ind = prepare_indicators(frame, base)
    ind_main = ind.loc[MAIN[0]:MAIN[1]]
    target = build_target_series(ind_main, base)

    result: dict = {"A_adversarial": {}, "B_relative_strength": {}}

    # ---------- A1 信号错位 ----------
    a1 = []
    for k in range(-3, 4):
        shifted = target.shift(k).fillna(0.0)
        m = metrics(run_engine(ind_main, shifted, base, f"shift{k}")["path"])
        a1.append({"shift_days": k, "cagr": m["cagr_pct"], "mdd": m["max_drawdown_pct"], "calmar": m["calmar"]})
    result["A_adversarial"]["shift_test"] = a1
    pd.DataFrame(a1).to_csv(OUT / "adversarial_shift.csv", index=False)

    # ---------- A2 随机择时对照 ----------
    rng = np.random.default_rng(42)
    levels, counts = np.unique(target.values, return_counts=True)
    real = metrics(run_engine(ind_main, target, base, "real")["path"])
    rand_calmar, rand_cagr = [], []
    for _ in range(300):
        perm = rng.permutation(target.values)
        mp = metrics(run_engine(ind_main, pd.Series(perm, index=ind_main.index), base, "rand")["path"])
        rand_calmar.append(mp["calmar"])
        rand_cagr.append(mp["cagr_pct"])
    rand_calmar = np.array(rand_calmar)
    rand_cagr = np.array(rand_cagr)
    pct_calmar = float((rand_calmar < real["calmar"]).mean())
    result["A_adversarial"]["random_timing"] = {
        "real_calmar": real["calmar"], "real_cagr": real["cagr_pct"],
        "random_calmar_mean": round(float(rand_calmar.mean()), 3),
        "random_calmar_p95": round(float(np.quantile(rand_calmar, 0.95)), 3),
        "random_cagr_mean": round(float(rand_cagr.mean()), 2),
        "percentile_vs_random": round(pct_calmar, 4),
        "n_random": len(rand_calmar),
    }
    pd.DataFrame({"random_calmar": rand_calmar, "random_cagr": rand_cagr}).to_csv(OUT / "adversarial_random.csv", index=False)

    # ---------- A3 收益打乱（保留仓位） ----------
    o = ind_main["adj_open"].to_numpy(float)
    c = ind_main["adj_close"].to_numpy(float)
    r_asset = c[1:] / c[:-1] - 1.0
    shuffled_calmar = []
    for _ in range(300):
        rs = rng.permutation(r_asset)
        # 用同一仓位序列、打乱后的收益重建净值
        w = target.shift(1).fillna(0.0).to_numpy(float)[1:]
        eq = np.cumprod(1.0 + w * rs[: len(w)])
        peak = np.maximum.accumulate(eq)
        mdd = float((eq / peak - 1.0).min())
        yrs = len(eq) / 252
        cagr = float(eq[-1] ** (1 / yrs) - 1) if eq[-1] > 0 else -1.0
        shuffled_calmar.append(cagr / abs(mdd) if mdd < 0 else 0.0)
    shuffled_calmar = np.array(shuffled_calmar)
    result["A_adversarial"]["shuffled_returns"] = {
        "real_calmar": real["calmar"],
        "shuffled_mean": round(float(shuffled_calmar.mean()), 3),
        "shuffled_p95": round(float(np.quantile(shuffled_calmar, 0.95)), 3),
        "percentile": round(float((shuffled_calmar < real["calmar"]).mean()), 4),
    }

    # ---------- B 相对强弱 ----------
    def with_rs(fr: pd.DataFrame) -> pd.DataFrame:
        f2 = fr.copy()
        f2["rs20"] = f2["adj_close"].pct_change(20) - f2["spy"].pct_change(20)
        f2["rs60"] = f2["adj_close"].pct_change(60) - f2["spy"].pct_change(60)
        return f2

    frame_rs = with_rs(frame)
    frame_long_rs = with_rs(frame_long)
    b_rows = []
    variants = [("baseline (no RS)", {})]
    for thr in [-0.03, -0.01, 0.0, 0.01, 0.03]:
        variants.append((f"RS20 >= {thr:+.2f}", {"rs_mode": "rs20", "rs_min": thr}))
    for thr in [-0.03, 0.0, 0.03]:
        variants.append((f"RS60 >= {thr:+.2f}", {"rs_mode": "rs60", "rs_min": thr}))
    for name, kw in variants:
        p = replace(base, **kw)
        r_main = window_metrics(prepare_indicators(frame_rs, p), p, MAIN)
        r_train = window_metrics(prepare_indicators(frame_rs, p), p, TRAIN)
        r_test = window_metrics(prepare_indicators(frame_rs, p), p, TEST)
        r_oos = window_metrics(prepare_indicators(frame_rs, p), p, OOS)
        r_long = window_metrics(prepare_indicators(frame_long_rs, p), p, LONG)
        b_rows.append({
            "variant": name,
            "main_cagr": r_main["cagr"], "main_mdd": r_main["mdd"], "main_calmar": r_main["calmar"],
            "train_calmar": r_train["calmar"], "test_calmar": r_test["calmar"],
            "oos_calmar": r_oos["calmar"], "long_calmar": r_long["calmar"],
            "main_reb": r_main["reb"],
        })
    bdf = pd.DataFrame(b_rows)
    bdf.to_csv(OUT / "relative_strength_experiment.csv", index=False)
    result["B_relative_strength"] = json.loads(bdf.to_json(orient="records"))

    (OUT / "adversarial_and_rs.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float))

    pd.set_option("display.width", 240)
    print("=== A1 信号错位测试（k=0 应最好）===")
    print(pd.DataFrame(a1).to_string(index=False))
    print()
    print("=== A2/A3 对抗性检验 ===")
    print(json.dumps(result["A_adversarial"]["random_timing"], ensure_ascii=False, indent=1))
    print(json.dumps(result["A_adversarial"]["shuffled_returns"], ensure_ascii=False, indent=1))
    print()
    print("=== B 相对强弱闸门 ===")
    print(bdf.to_string(index=False))
    print(f"\n输出: {OUT}/adversarial_shift.csv, adversarial_random.csv, relative_strength_experiment.csv")


if __name__ == "__main__":
    main()
