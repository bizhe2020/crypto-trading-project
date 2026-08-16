#!/usr/bin/env python3
"""QQQ↔GOOGL router 超参网格扫描。

在 compare_qqq_googl_router 的日频帧上扫描:
- qqq_min_route_score  : QQQ 作为 challenger 的最低分（GOOGL flat 时是否切去 QQQ 的门槛）
- googl_min_route_score: GOOGL 作为 challenger 的最低分（默认 0）
- switch_advantage     : 切换滞回

已知约束（窗口 2026-03-04 → 08-07，157 天）:
- GOOGL 激活时恒 107.8，QQQ 最高 110 → GOOGL 在市时 QQQ 永远抢不走（需 +6），
  router 的 GOOGL 天数是固定的（41），可调维度只有 GOOGL flat 时 QQQ vs cash。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_qqq_googl_router import (  # noqa: E402
    build_googl_daily,
    build_qqq_daily,
    run_router,
    max_drawdown_pct,
    START,
    END,
    GOOGL_RUNTIME,
    QQQ_RUNTIME,
)


def build_merged() -> pd.DataFrame:
    gcfg = json.loads(GOOGL_RUNTIME.read_text())
    qcfg = json.loads(QQQ_RUNTIME.read_text())
    g = build_googl_daily(gcfg)
    q = build_qqq_daily(qcfg)
    full_dates = pd.date_range(START.normalize(), END.normalize(), freq="D", tz="UTC")
    m = (
        pd.DataFrame({"date": full_dates})
        .merge(q, on="date", how="left")
        .merge(g, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )
    for c in ("qqq_return", "googl_return"):
        m[c] = m[c].fillna(0.0)
    for c in ("qqq_active", "googl_active"):
        m[c] = m[c].fillna(False).astype(bool)
    m["qqq_score"] = m["qqq_score"].fillna(0.0)
    m["googl_score"] = m["googl_score"].fillna(0.0)
    return m


def main() -> None:
    pd.set_option("future.no_silent_downcasting", True)
    m = build_merged()
    googl_eq = (1.0 + m["googl_return"]).cumprod() * 1000.0
    qqq_eq = (1.0 + m["qqq_return"]).cumprod() * 1000.0
    googl_total = (googl_eq.iloc[-1] / 1000.0 - 1.0) * 100.0
    qqq_total = (qqq_eq.iloc[-1] / 1000.0 - 1.0) * 100.0
    print(f"基线: 单 GOOGL {googl_total:+.2f}% 单 QQQ {qqq_total:+.2f}%")
    print("=" * 84)

    rows: list[dict] = []
    for qqq_min in (0.0, 90.0, 94.0, 96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0):
        for adv in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
            r = run_router(m, switch_cost_bps=0.0, qqq_min_score=qqq_min, switch_advantage=adv)
            eq = r["router_equity"]
            total = (eq.iloc[-1] / 1000.0 - 1.0) * 100.0
            rows.append(
                {
                    "qqq_min": qqq_min,
                    "adv": adv,
                    "total_pct": round(total, 1),
                    "maxdd_pct": round(max_drawdown_pct(eq), 1),
                    "switches": int(r["switched"].sum()),
                    "qqq_days": int((r["selected"] == "QQQ").sum()),
                    "googl_days": int((r["selected"] == "GOOGL").sum()),
                }
            )

    out = pd.DataFrame(rows).sort_values("total_pct", ascending=False).reset_index(drop=True)
    print(f"{'排名':>4} {'qqq_min':>8} {'adv':>5} {'总收益%':>10} {'最大回撤%':>10} {'切换':>5} {'QQQ天':>6} {'GOOGL天':>7}")
    print("-" * 84)
    for i, r in out.head(15).iterrows():
        print(f"{i + 1:>4} {r['qqq_min']:>8.0f} {r['adv']:>5.0f} {r['total_pct']:>10.2f} {r['maxdd_pct']:>10.2f} {r['switches']:>5} {r['qqq_days']:>6} {r['googl_days']:>7}")

    print("\n===== 最优组合明细 =====")
    best = out.iloc[0]
    r = run_router(m, switch_cost_bps=0.0, qqq_min_score=float(best["qqq_min"]), switch_advantage=float(best["adv"]))
    prev = None
    for _, row in r.iterrows():
        if row["selected"] != prev:
            print(f"  {row['date'].date()} -> {row['selected']:<5} eq={row['router_equity']:.0f} (qqq={row['qqq_score']:.0f}, googl={row['googl_score']:.0f})")
            prev = row["selected"]

    print("\n===== qqq_min 的一维影响（adv=6 固定）=====")
    rows1 = [x for x in rows if x["adv"] == 6.0]
    for r in sorted(rows1, key=lambda x: x["qqq_min"]):
        print(f"  qqq_min={r['qqq_min']:>5.0f}  adv=6  total={r['total_pct']:>8.2f}%  maxdd={r['maxdd_pct']:>6.2f}%  qqq_days={r['qqq_days']:>3}")


if __name__ == "__main__":
    main()
