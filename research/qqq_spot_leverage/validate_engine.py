#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""引擎验证：用真实杠杆 ETF 净值校准杠杆复合（round 8）。

杠杆 ETF 的净值 ≈ cumprod(1 + L × r_index)（日重置）。这是对"杠杆复利"最硬的外部检验：
  * TQQQ (3x QQQ, 2011-2026, 3873 天)
  * SQQQ (-3x QQQ, 2022-2026, 1104 天)
若引擎/公式能复现它们的净值（差异≈ETF 费率+掉期成本），说明杠杆记账方向正确。

另外做 1x 自检：引擎恒定 1x 应等于指数价格收益。

用法: python3 research/qqq_spot_leverage/validate_engine.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_qqq_spot_leverage import (  # noqa: E402
    Params,
    load_market_frame,
    metrics,
    prepare_indicators,
    run_engine,
)

OUT = Path(__file__).resolve().parent / "out"
ETF_FILES = {
    "TQQQ": ("data/public/etf_long/TQQQ-1d.feather", 3.0),
    "SQQQ": ("data/public/etf/SQQQ-1d.feather", -3.0),
}
REPO = Path(__file__).resolve().parents[2]


def load_etf(rel: str) -> pd.Series:
    fe = pd.read_feather(REPO / rel)
    fe["d"] = pd.to_datetime(fe["date"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    fe = fe.sort_values("d").drop_duplicates("d", keep="last").set_index("d")
    return fe["close"].rename(rel)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qqq = load_market_frame("1999-03-10", "2026-05-29", price_source="feather")
    rows = []
    for name, (rel, lev) in ETF_FILES.items():
        etf = load_etf(rel)
        common = etf.index.intersection(qqq.index)
        if len(common) < 100:
            continue
        e = etf.loc[common].pct_change().dropna()
        r = qqq.loc[common, "adj_close"].pct_change().reindex(e.index).dropna()
        e = e.reindex(r.index)
        theo = float(np.prod(1.0 + lev * r.values) - 1.0)
        act = float(etf.loc[e.index].iloc[-1] / etf.loc[e.index].iloc[0] - 1.0)
        years = len(r) / 252.0
        rows.append(
            {
                "etf": name,
                "target_leverage": lev,
                "days": int(len(r)),
                "start": str(r.index[0].date()),
                "end": str(r.index[-1].date()),
                "etf_actual_pct": round(act * 100, 1),
                "theory_cumprod_pct": round(theo * 100, 1),
                "daily_corr": round(float(np.corrcoef(e.values, lev * r.values)[0, 1]), 4),
                "annual_gap_pp": round(((1 + act) ** (1 / years) - 1) * 100 - ((1 + theo) ** (1 / years) - 1) * 100, 2),
            }
        )

    # 1x 自检：引擎恒定 1x vs 指数
    f = load_market_frame("2001-01-02", "2026-08-14")
    p = Params()
    ind = prepare_indicators(f, p).loc["2020-01-02":"2026-08-14"]
    m = metrics(run_engine(ind, pd.Series(1.0, index=ind.index), Params(rs_mode="", use_vix_gate=False, use_recovery=False, drift_band=10.0), "bh")["path"])
    asset = float(ind["adj_close"].iloc[-1] / ind["adj_close"].iloc[0] - 1.0) * 100
    sanity = {"engine_1x_pct": m["total_return_pct"], "asset_pct": round(asset, 2),
              "diff_pp": round(m["total_return_pct"] - asset, 2)}

    payload = {"leverage_etf_validation": rows, "buy_hold_sanity": sanity,
               "note": "ETF 净值 ≈ cumprod(1+L×r)；annual_gap 反映 ETF 管理费+掉期成本+跟踪误差"}
    (OUT / "engine_validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    pd.set_option("display.width", 200)
    print("=== 杠杆 ETF 校准 ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("=== 1x 自检 ===")
    print(json.dumps(sanity, ensure_ascii=False))
    print(f"\n输出: {OUT}/engine_validation.json")


if __name__ == "__main__":
    main()
