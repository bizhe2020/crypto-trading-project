"""scalp 仓位结构的离线研究（基于不带闸门的完整链路逐笔）。

不带闸门时，交易集合与仓位大小无关（同一组 236 笔），每笔的 R（净盈亏 / 计划风险金额）是固定的，
所以任意 sizing 规则都可以离线精确复利：ret_i = f_i × R_i。（名义上限触发的笔数单独报告；
这些笔的实际风险小于计划风险，按 R_eff = ret / 计划风险 处理，等价于保留原有封顶。）

输出：
  1. 每个 risk_regime × 方向 的 R 统计（全样本 / 2025 前 / 2025 起），用来判断 regime 倾斜有没有依据。
  2. 按年份的 R 统计（edge 是否衰减）。
  3. 离线 sizing 对比：现行倾斜 × m、等风险 flat f、半倾斜（sqrt）、以及回撤减仓（DD 线性缩放）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LIVE_RISK = {
    ("bull_strong", "BULL"): 0.12, ("bull_strong", "BEAR"): 0.01,
    ("bull_weak", "BULL"): 0.05, ("bull_weak", "BEAR"): 0.05,
    ("bear_weak", "BULL"): 0.03, ("bear_weak", "BEAR"): 0.05,
    ("bear_strong", "BULL"): 0.0225, ("bear_strong", "BEAR"): 0.10,
}
BASE_RISK = 0.035


def load(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path, parse_dates=["entry_time", "exit_time"]).sort_values("entry_time").reset_index(drop=True)
    t["ret"] = t["pnl"] / t["capital_at_entry"]
    t["plan_risk"] = [LIVE_RISK.get((r, d), BASE_RISK) for r, d in zip(t["risk_regime"], t["direction"])]
    t["R"] = t["ret"] / t["plan_risk"]
    stop_dist = (t["entry_price"] - t["initial_stop_price"]).abs()
    t["R_price"] = np.where(t["direction"] == "BULL", t["exit_price"] - t["entry_price"],
                            t["entry_price"] - t["exit_price"]) / stop_dist
    t["capped"] = (t["R"].abs() < 0.8 * t["R_price"].abs()) & (t["R_price"].abs() > 0.3)
    t["year"] = t["entry_time"].dt.year
    return t


def rstats(x: pd.Series) -> dict:
    n = len(x)
    if n == 0:
        return {"n": 0}
    m, s = float(x.mean()), float(x.std(ddof=1)) if n > 1 else 0.0
    return {"n": n, "mean_R": round(m, 3), "sum_R": round(float(x.sum()), 2),
            "t": round(m / (s / np.sqrt(n)), 2) if s > 0 else None, "wr": round(float((x > 0).mean()), 3)}


def compound(ret: np.ndarray, days: np.ndarray) -> dict:
    eq = np.cumprod(1 + np.clip(ret, -0.99, None))
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    mdd = float((1 - eq / peak).max())
    years = float((days[-1] - days[0]) / np.timedelta64(365, "D"))
    cagr = float(eq[-1] ** (1 / years) - 1) if eq[-1] > 0 else -1.0
    return {"final_x": round(float(eq[-1]), 2), "cagr_pct": round(cagr * 100, 1), "mdd_pct": round(mdd * 100, 1),
            "calmar": round(cagr / mdd, 2) if mdd > 0 else None}


def dd_scaled(R: np.ndarray, f: np.ndarray, dd_full: float, floor: float) -> np.ndarray:
    """回撤减仓：风险 = f × max(floor, 1 − DD/dd_full)，DD 按逐笔权益峰值算（连续，不是开关）。"""
    eq, peak, out = 1.0, 1.0, np.empty_like(R)
    for i in range(len(R)):
        dd = 1 - eq / peak
        scale = max(floor, 1 - dd / dd_full)
        out[i] = f[i] * scale * R[i]
        eq *= 1 + out[i]
        peak = max(peak, eq)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trades", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    t = load(Path(args.trades))
    pre, post = t["entry_time"] < pd.Timestamp("2025-01-01", tz="UTC"), t["entry_time"] >= pd.Timestamp("2025-01-01", tz="UTC")
    out: dict = {"n_trades": len(t), "capped_trades": int(t["capped"].sum())}

    by = {}
    for (rg, d), g in t.groupby(["risk_regime", "direction"]):
        by[f"{rg}|{d}"] = {"live_risk": LIVE_RISK.get((rg, d), BASE_RISK), "all": rstats(g["R"]),
                           "pre2025": rstats(g.loc[pre[g.index], "R"]), "from2025": rstats(g.loc[post[g.index], "R"])}
    out["by_regime"] = by
    out["by_year"] = {int(y): rstats(g["R"]) for y, g in t.groupby("year")}
    out["all"] = {"all": rstats(t["R"]), "pre2025": rstats(t.loc[pre, "R"]), "from2025": rstats(t.loc[post, "R"])}

    R = t["R"].to_numpy()
    tilt = t["plan_risk"].to_numpy()
    days = t["exit_time"].to_numpy()
    mean_tilt = float(tilt.mean())
    sqrt_tilt = np.sqrt(tilt / BASE_RISK) * BASE_RISK
    sqrt_tilt *= mean_tilt / sqrt_tilt.mean()  # 与现行倾斜同均值风险
    variants = {}
    for m in (1.0, 0.75, 0.5, 0.4, 0.33, 0.25):
        variants[f"tilt_x{m}"] = tilt * m
        variants[f"sqrt_tilt_x{m}"] = sqrt_tilt * m
    for f in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
        variants[f"flat_{f}"] = np.full(len(R), f)
    res = {}
    for name, f in variants.items():
        ret = f * R
        row = {"mean_risk_pct": round(float(f.mean()) * 100, 2), "all": compound(ret, days),
               "pre2025": compound(ret[pre.values], days[pre.values]), "from2025": compound(ret[post.values], days[post.values])}
        res[name] = row
    for name in ("tilt_x0.5", "tilt_x0.4", "flat_0.025", "flat_0.03"):
        f = variants[name]
        for dd_full, floor in ((0.3, 0.25), (0.4, 0.25), (0.5, 0.33)):
            ret = dd_scaled(R, f, dd_full, floor)
            res[f"{name}_dd{int(dd_full * 100)}f{int(floor * 100)}"] = {
                "mean_risk_pct": None, "all": compound(ret, days),
                "pre2025": compound(dd_scaled(R[pre.values], f[pre.values], dd_full, floor), days[pre.values]),
                "from2025": compound(dd_scaled(R[post.values], f[post.values], dd_full, floor), days[post.values])}
    out["sizing"] = res
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("n_trades", "capped_trades", "all", "by_year", "by_regime")}, ensure_ascii=False, indent=1))
    for k, v in res.items():
        print(f"{k:28s} risk%={v['mean_risk_pct']}  all={v['all']}  pre={v['pre2025']['cagr_pct']}/{v['pre2025']['mdd_pct']}  post={v['from2025']['cagr_pct']}/{v['from2025']['mdd_pct']}")


if __name__ == "__main__":
    main()
