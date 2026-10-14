#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""长样本压力测试（round 3）：用连续的 feather 序列跑 1999-2026。

背景（见报告第 2 节）：
  * 期权量化的 CSV 缺 2005-2011，且 2001-2004 复权因子损坏（2003-03-24 复权比 0.061）
  * data/public/etf_full/QQQ-1d.feather 连续覆盖 1999-03 ~ 2026-05（未做股息复权）
因此用 feather 序列可以真正评估"互联网泡沫尾声 + 2008 金融危机"这两段长熊。

注意：feather 序列不含股息（≈价格指数），总收益会比真实 QQQ 总收益低
约 0.5~1.5pp/年。这里关注的是**回撤控制与相对表现**，不是绝对收益。

用法: python3 research/qqq_spot_leverage/long_sample_stress.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_qqq_spot_leverage import (  # noqa: E402
    Params,
    build_target_series,
    buy_hold,
    load_market_frame,
    metrics,
    prepare_indicators,
    run_engine,
)

OUT = Path(__file__).resolve().parent / "out"
SEGMENTS = {
    "dot-com 2000-2002": ("2000-03-10", "2002-12-31"),
    "recovery 2003-2007": ("2003-01-01", "2007-10-31"),
    "GFC 2007-10~2009-03": ("2007-10-01", "2009-03-31"),
    "post-GFC 2009-2013": ("2009-04-01", "2013-12-31"),
    "2011-2019": ("2011-04-26", "2019-12-31"),
    "2020-2026(with div-free)": ("2020-01-02", "2026-05-29"),
}
FULL = ("1999-03-10", "2026-05-29")


def run(frame: pd.DataFrame, p: Params, label: str) -> pd.DataFrame:
    ind = prepare_indicators(frame, p)
    return run_engine(ind, build_target_series(ind, p), p, label)["path"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_market_frame(FULL[0], FULL[1], price_source="feather")
    base = Params()
    variants = {
        "v3 (default)": base,
        "v2 (no decline-confirm, hyst4)": replace(base, vix_hysteresis=4.0, vix_gate_confirm_decline=False),
        "no VIX gate": replace(base, use_vix_gate=False),
        "1.0x cap": replace(base, lev_full=1.0, max_leverage=1.0, recovery_lev=0.5),
    }

    rows = []
    for name, p in variants.items():
        path_all = run(frame, p, name)
        for seg, (a, b) in list(SEGMENTS.items()) + [("FULL 1999-2026", FULL)]:
            eq = path_all["equity"].loc[a:b]
            if len(eq) < 60:
                continue
            seg_ret = eq / eq.iloc[0] - 1.0
            dd = (eq / eq.cummax() - 1.0).min()
            w = path_all["weight"].loc[a:b]
            rets = path_all["net_ret"].loc[a:b]
            yrs = (eq.index[-1] - eq.index[0]).days / 365.25
            cagr = (1 + float(seg_ret.iloc[-1])) ** (1 / yrs) - 1 if yrs > 0 and seg_ret.iloc[-1] > -1 else -1.0
            rows.append(
                {
                    "variant": name,
                    "segment": seg,
                    "total_pct": round(float(seg_ret.iloc[-1]) * 100, 1),
                    "cagr_pct": round(cagr * 100, 2),
                    "mdd_pct": round(float(dd) * 100, 2),
                    "sharpe": round(float(rets.mean() / rets.std() * (252 ** 0.5)), 2) if rets.std() > 0 else 0.0,
                    "avg_lev": round(float(w.mean()), 2),
                    "exposure_pct": round(float((w > 0.1).mean()) * 100, 1),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "long_sample_stress.csv", index=False)

    # 买入持有基准
    ind = prepare_indicators(frame, base)
    bh_rows = []
    for seg, (a, b) in list(SEGMENTS.items()) + [("FULL 1999-2026", FULL)]:
        sub = ind.loc[a:b]
        if len(sub) < 60:
            continue
        m = metrics(buy_hold(sub, 1.0, base, seg)["path"])
        bh_rows.append({"segment": seg, "bh_cagr_pct": m["cagr_pct"], "bh_mdd_pct": m["max_drawdown_pct"], "bh_calmar": m["calmar"]})
    bh = pd.DataFrame(bh_rows)

    payload = {"note": "feather price series is dividend-unadjusted; treat as price index",
               "segments": SEGMENTS, "results": json.loads(df.to_json(orient="records")),
               "buy_hold": json.loads(bh.to_json(orient="records"))}
    (OUT / "long_sample_stress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    pd.set_option("display.width", 220)
    print("=== 买入持有 1x（feather，未含股息）===")
    print(bh.to_string(index=False))
    print()
    for name in variants:
        print(f"=== {name} ===")
        print(df[df["variant"] == name].drop(columns=["variant"]).to_string(index=False))
        print()
    print(f"输出: {OUT}/long_sample_stress.csv, long_sample_stress.json")


if __name__ == "__main__":
    main()
