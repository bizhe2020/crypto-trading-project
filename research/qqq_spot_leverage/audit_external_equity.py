#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对其他策略净值的口径审计（round 15，只读，不修改任何原文件）。

对象：`scripts/plot_strategy_comparison.py::tqqq_sqqq_curve()` 产出的
      `var/reports/strategy_equity_return_comparison.csv` 中的 `tqqq_equity` 曲线。

方法（控制变量 A/B）：
  A 原口径：把 `long_mask` 用**当日** `vix_label`/`ixic_trend_label` 计算，
             并把该 mask 作用于**当日**的 TQQQ/TQQQ 日收益（close[t-1]→close[t]）。
  B 滞后一日：把 mask 与 short 判据改为只用 t-1 可得的信息。
两者共用同一份原始逻辑（仅替换这两处），因此差异完全来自"当日闸门"。

结论（可复现）：
  A 期末 67,816（+6681%，CAGR 201.4%，MDD −16.3%）
  B 期末  1,108（  +10.8%，CAGR   2.7%，MDD −55.8%）
  → 该曲线的超额收益依赖"当日收盘才能知道的闸门"作用于当日收益，属前视口径。

用法: python3 research/qqq_spot_leverage/audit_external_equity.py
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent / "out"


def _load_variant(src: str, name: str, module_globals: dict):
    src = src.replace(
        '    assert round(float(curve.iloc[-1]["equity"]) / 1000.0 * 100.0 - 100.0, 2) == weighted["summary"]["total_return_pct"]\n',
        "")
    mod = types.ModuleType(name)
    mod.__dict__.update(module_globals)
    exec(compile(src, f"{name}.py", "exec"), mod.__dict__)
    return mod.tqqq_sqqq_curve()


def curve_stats(curve: pd.DataFrame, label: str) -> dict:
    eq = curve["equity"]
    dates = pd.to_datetime(curve["date"])
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    return {
        "label": label,
        "final_equity": round(float(eq.iloc[-1]), 1),
        "total_pct": round(float(eq.iloc[-1] / 1000.0 - 1) * 100, 1),
        "cagr_pct": round(float((eq.iloc[-1] / 1000.0) ** (1 / years) - 1) * 100, 2),
        "mdd_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 2),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    import scripts.plot_strategy_comparison as P

    src = (REPO / "scripts" / "plot_strategy_comparison.py").read_text()

    base = _load_variant(src, "plot_base", P.__dict__)

    src_lag = src.replace(
        'long_mask = frame["vix_label"].isin(["vix_low", "vix_normal"]) & frame["ixic_trend_label"].eq("ixic_up")',
        'long_mask = frame["vix_label"].shift(1).isin(["vix_low", "vix_normal"]) & frame["ixic_trend_label"].shift(1).eq("ixic_up")',
    ).replace(
        'desired_short = int(row["planned_trend"]) < 0 and allow_short(row, "bearish_score5")',
        'desired_short = int(row["planned_trend"]) < 0 and allow_short(frame.shift(1).loc[idx], "bearish_score5")',
    )
    lagged = _load_variant(src_lag, "plot_lag", P.__dict__)

    a = curve_stats(base, "A 原口径（当日闸门）")
    b = curve_stats(lagged, "B 闸门滞后1日")
    m = base.merge(lagged, on="date", suffixes=("_a", "_b"))
    m["diff"] = m["daily_return_a"] - m["daily_return_b"]
    top = m.reindex(m["diff"].abs().nlargest(8).index)[["date", "daily_return_a", "daily_return_b", "diff"]]
    top.to_csv(OUT / "audit_external_top_days.csv", index=False)

    payload = {
        "target": "var/reports/strategy_equity_return_comparison.csv :: tqqq_equity (tqqq_sqqq strategy)",
        "generator": "scripts/plot_strategy_comparison.py::tqqq_sqqq_curve()",
        "variant_A": a, "variant_B": b,
        "diff_days_gt_50bp": int((m["diff"].abs() > 0.005).sum()),
        "n_days": int(len(m)),
        "interpretation": "该曲线把'当日收盘才能确定的 vix_label/ixic_trend_label 闸门'用于当日的 TQQQ/SQQQ 收益，"
                          "因此含有前视；把闸门滞后一日后 CAGR 从 201% 掉到 2.7%、回撤从 −16% 扩大到 −56%。",
    }
    (OUT / "audit_external_equity.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float))

    pd.set_option("display.width", 200)
    print("=== 外部策略净值口径审计（只读）===")
    print(json.dumps(a, ensure_ascii=False))
    print(json.dumps(b, ensure_ascii=False))
    print()
    print(f"差异 >0.5pp 的天数: {payload['diff_days_gt_50bp']} / {payload['n_days']}")
    print("差异最大的 8 天：")
    print(top.to_string(index=False))
    print(f"\n输出: {OUT}/audit_external_equity.json, audit_external_top_days.csv")


if __name__ == "__main__":
    main()
