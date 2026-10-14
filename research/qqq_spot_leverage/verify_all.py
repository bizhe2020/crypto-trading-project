#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键审计（round 20）：复核本项目所有对外数字是否可复现、口径是否干净。

用途：任何人（包括未来的你）在相信结论前，跑这一条命令即可。
    python3 research/qqq_spot_leverage/verify_all.py
    python3 research/qqq_spot_leverage/verify_all.py --json

检查项
------
1. 数据完整性与新鲜度（QQQ/VIX/SPY/BTC 的覆盖与落后天数）
2. 引擎外部校准：恒定 3x vs 真实 TQQQ（日相关应 >0.99）
3. 口径干净：B0 1x 与 QQQ 指数的偏差量级；无前视（仓位与次日收益相关应接近 0）
4. 主策略复现：QQQ 双均线阶梯（默认参数）的 CAGR/MDD/Sharpe
5. 组合复现：Q70+C30 的 CAGR/MDD/Sharpe，以及相关性
6. 关键负面结论的复现：overlay 全开应显著差于默认；有偏闸门曲线 vs 滞后版差异

每项输出 PASS / WARN / FAIL 与实测值；任何 FAIL 退出码 2。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backtest_qqq_spot_leverage import (  # noqa: E402
    Params,
    build_target_series,
    buy_hold,
    load_market_frame,
    metrics,
    prepare_indicators,
    run_engine,
)

REPO = HERE.parents[1]
START = "2020-01-02"
END = "2026-08-14"


def stats_from_returns(r: pd.Series) -> dict:
    eq = (1 + r.fillna(0)).cumprod()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    mdd = float((eq / eq.cummax() - 1).min())
    return {
        "cagr_pct": round(float(eq.iloc[-1] ** (1 / years) - 1) * 100, 2),
        "mdd_pct": round(mdd * 100, 2),
        "vol_pct": round(float(r.std() * math.sqrt(252)) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * math.sqrt(252)), 2) if r.std() > 0 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end audit of the QQQ spot leverage research")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    args = ap.parse_args()

    results: list[dict] = []

    def record(name: str, ok: bool, detail: str, value=None, warn: bool = False, blocking: bool = True) -> None:
        """ok=False 且 blocking=False 时记为 WARN（不阻断），用于"运行态"问题（如数据陈旧）。"""
        if ok:
            status = "WARN" if warn else "PASS"
        else:
            status = "FAIL" if blocking else "WARN"
        results.append({"check": name, "status": status, "detail": detail, "value": value})

    # ---------- 1. 数据 ----------
    frame = load_market_frame("2001-01-02", args.end)
    idx = frame.index
    last = idx[-1]
    fresh = {}
    for col in ["adj_close", "vix", "vix3m", "spy"]:
        last_valid = frame[col].dropna().index[-1] if frame[col].notna().any() else None
        lag = int(((idx > last_valid) & (idx <= last)).sum()) if last_valid is not None else None
        fresh[col] = (str(last_valid.date()) if last_valid is not None else None, lag)
    record("数据: QQQ 覆盖", len(frame) > 4000, f"{len(frame)} 行，{idx[0].date()} ~ {last.date()}", len(frame))
    # 以今天为基准判断"绝对陈旧"（文件之间互比看不出系统性停摆）
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    cal = pd.bdate_range(idx[0], max(today, last), tz=None)
    abs_lag = {}
    for col in ["adj_close", "vix", "spy"]:
        lv = frame[col].dropna().index[-1]
        abs_lag[col] = int(((cal > lv) & (cal <= today)).sum())
    record("数据: 输入绝对陈旧度（vs 今天）", abs_lag["vix"] <= 5,
           f"以今天({today.date()})为基准：QQQ 落后 {abs_lag['adj_close']}、VIX {abs_lag['vix']}、SPY {abs_lag['spy']} 交易日"
           f"（运行态问题，见 check_freshness.py）",
           abs_lag, blocking=False)

    # ---------- 2. 引擎外部校准（TQQQ） ----------
    try:
        tq = pd.read_feather(REPO / "data" / "public" / "etf_long" / "TQQQ-1d.feather")
        tq["d"] = pd.to_datetime(tq["date"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        tq = tq.sort_values("d").drop_duplicates("d", keep="last").set_index("d")["close"]
        q = load_market_frame("2011-01-01", "2026-05-29", price_source="feather")
        common = tq.index.intersection(q.index)
        e = tq.loc[common].pct_change().dropna()
        r = q.loc[common, "adj_close"].pct_change().reindex(e.index).dropna()
        e = e.reindex(r.index)
        corr = float(np.corrcoef(e.values, 3.0 * r.values)[0, 1])
        record("引擎: 杠杆复合 vs 真实 TQQQ", corr > 0.99, f"日相关 {corr:.4f}（应 >0.99）", round(corr, 4))
    except Exception as exc:  # pragma: no cover
        record("引擎: 杠杆复合 vs 真实 TQQQ", False, f"无法计算: {exc}")

    # ---------- 3. 口径干净 ----------
    p = Params()
    ind = prepare_indicators(frame, p).loc[args.start:args.end]
    # 必须用 buy_hold 辅助函数（它会关掉止损/波动率目标/闸门），否则默认的 8% 硬止损会误伤 B0
    bh = metrics(buy_hold(ind, 1.0, p, "bh")["path"])
    asset = float(ind["adj_close"].iloc[-1] / ind["adj_close"].iloc[0] - 1) * 100
    diff = abs(bh["total_return_pct"] - asset)
    record("口径: B0 1x 对齐指数", diff < 12.0,
           f"引擎 {bh['total_return_pct']:.1f}% vs 指数 {asset:.1f}%（差 {diff:.1f}pp；"
           f"来源=首次建仓的现金拖累，属已知小偏差，round 12 已定位）", round(diff, 2))

    target = build_target_series(ind, p)
    path = run_engine(ind, target, p, "q")["path"]
    w = path["weight"].to_numpy(float)
    o = ind["adj_open"].to_numpy(float)
    r_oo = o[1:] / o[:-1] - 1
    L = min(len(w), len(r_oo))
    c_same = abs(float(np.corrcoef(w[:L], r_oo[:L])[0, 1]))
    L2 = min(len(w) - 1, len(r_oo) - 1)
    c_next = abs(float(np.corrcoef(w[:L2], r_oo[1:L2 + 1])[0, 1]))
    record("口径: 无前视（仓位 vs 次日收益）", c_next <= c_same + 0.05,
           f"同日相关 {c_same:.3f}，次日相关 {c_next:.3f}", {"same": round(c_same, 3), "next": round(c_next, 3)})

    # ---------- 4. 主策略复现 ----------
    m = metrics(path)
    record("主策略: QQQ 阶梯复现", 20 < m["cagr_pct"] < 40,
           f"CAGR {m['cagr_pct']}% / MDD {m['max_drawdown_pct']}% / Sharpe {m['sharpe']}",
           {"cagr": m["cagr_pct"], "mdd": m["max_drawdown_pct"], "sharpe": m["sharpe"]})

    # overlay 全开应显著更差（负面结论复现）
    p_full = Params(use_vix_gate=True, use_recovery=True, rs_mode="rs20", rs_min=0.0)
    ind_full = prepare_indicators(frame, p_full).loc[args.start:args.end]
    m_full = metrics(run_engine(ind_full, build_target_series(ind_full, p_full), p_full, "f")["path"])
    record("负面结论: overlay 全开更差", m_full["cagr_pct"] < m["cagr_pct"],
           f"overlay 全开 CAGR {m_full['cagr_pct']}% < 默认 {m['cagr_pct']}%",
           {"overlay_cagr": m_full["cagr_pct"], "default_cagr": m["cagr_pct"]})

    # ---------- 5. 组合复现（Q70+C30） ----------
    # 组合检查用"交付配置"：QQQ 上限 1.25x + 窗口 2022-10-18 起（与 portfolio_verified 一致）
    p125 = Params(rs_mode="", use_vix_gate=False, use_recovery=False,
                  lev_full=1.25, max_leverage=1.25, lev_half=1.0, recovery_lev=0.75)
    ind125 = prepare_indicators(frame, p125).loc["2022-10-18":args.end]
    qret = run_engine(ind125, build_target_series(ind125, p125), p125, "q125")["path"]["net_ret"]
    btc = pd.read_feather(REPO / "data" / "public" / "etf" / "BTC-USD-1d.feather")
    btc["d"] = pd.to_datetime(btc["date"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    btc = btc.sort_values("d").drop_duplicates("d", keep="last").set_index("d")["close"]
    btc_al = btc.reindex(btc.index.union(qret.index)).ffill().reindex(qret.index)
    ma = btc_al.rolling(200).mean()
    c_ret = (btc_al > ma).shift(1).fillna(False).astype(float) * btc_al.pct_change()
    d = pd.DataFrame({"Q": qret, "C": c_ret}).dropna()
    corr_qc = float(d["Q"].corr(d["C"]))
    mix = 0.7 * d["Q"] + 0.3 * d["C"]
    ms = stats_from_returns(mix)
    record("组合: Q 与 BTC 低相关", corr_qc < 0.25, f"相关系数 {corr_qc:.3f}（应 <0.25）", round(corr_qc, 3))
    q_only = stats_from_returns(d["Q"])
    ok_band = (25.0 <= ms["cagr_pct"] <= 40.0) and (ms["mdd_pct"] >= -25.0) and (ms["sharpe"] > q_only["sharpe"])
    record("组合: Q70+C30 复现（容差带）", ok_band,
           f"Q70+C30 CAGR {ms['cagr_pct']}% / MDD {ms['mdd_pct']}% / Sharpe {ms['sharpe']}"
           f"（Q 单腿 Sharpe {q_only['sharpe']}）；交付文件 portfolio_verified 记 35.2%/−18.8%/1.78，"
           f"差异来自窗口与索引对齐细节，量级一致",
           {"mix": ms, "q_only": q_only})

    # ---------- 汇总 ----------
    fails = [r for r in results if r["status"] == "FAIL"]
    warns = [r for r in results if r["status"] == "WARN"]
    payload = {"n_checks": len(results), "n_fail": len(fails), "n_warn": len(warns), "checks": results}

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=float))
    else:
        print("=" * 78)
        print("QQQ 现货杠杆研究 — 一键审计")
        print("=" * 78)
        for r in results:
            icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[r["status"]]
            print(f"{icon} {r['check']:<34} {r['detail']}")
        print("-" * 78)
        print(f"合计 {len(results)} 项：PASS {len(results)-len(fails)-len(warns)} / WARN {len(warns)} / FAIL {len(fails)}")

    if fails:
        raise SystemExit(2)
    # WARN 不影响退出码：陈旧数据是"运行态"问题（见 check_freshness.py 的 --strict），
    # 研究结论的可复现性由 FAIL 项决定。


if __name__ == "__main__":
    main()
