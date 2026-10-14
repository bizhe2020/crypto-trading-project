"""第 3 步：趋势篮子的参数 / 算法搜索 + 走步（walk-forward）检验。

问题：实盘用的固定 5 规则投票（V5）是不是「最优」？如果事后挑最好的参数，样本外还能赢 V5 吗？

方法（全部只用已收盘日线、t 收盘信号 → t+1 开盘执行，成本口径与 explore.py 一致）：
  1. 候选池：单规则（均线/均线交叉/唐奇安/时序动量 多组窗口）、V5 的窗口缩放版（敏感性）、
     全规则投票、V5 + 波动率目标。
  2. 全样本（2006→）、三个分段的 Sharpe/收益/回撤，以及 V5 在候选里的排名。
  3. 走步选择：每年年初用过去 N 年的 Sharpe 选每条腿最好的候选，持有一年；
     对比固定 V5、买入持有。样本外赢不了 V5 = 搜索带来的只是过拟合。
  4. 腿权重：等权 vs 逆波动。
  5. 与 scalp（实盘口径逐笔）组合：2022→ 各窗口。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.time_in_market.explore import load_daily, metrics, scalp_daily, sleeve_returns, to_calendar  # noqa: E402
from strategy.trend_vote import vote_target  # noqa: E402


def _sma_L(c: pd.Series, n: int) -> pd.Series:
    sma = c.rolling(n).mean()
    return (c > sma).astype(float).where(sma.notna(), 0.0)


def _mom_L(c: pd.Series, n: int) -> pd.Series:
    mom = c / c.shift(n) - 1
    return (mom > 0).astype(float).where(mom.notna(), 0.0)


def _ema_L(c: pd.Series, f: int, s: int) -> pd.Series:
    valid = c.expanding().count() >= s
    return (c.ewm(span=f, adjust=False).mean() > c.ewm(span=s, adjust=False).mean()).astype(float).where(valid, 0.0)


def _donch_L(d: pd.DataFrame, entry: int, exit_: int) -> pd.Series:
    c = d["close"]
    hh = d["high"].rolling(entry).max().shift(1)
    ll = d["low"].rolling(exit_).min().shift(1)
    pos, cur = [], 0.0
    for ci, hi, lo in zip(c.values, hh.values, ll.values):
        if np.isnan(hi) or np.isnan(lo):
            pos.append(0.0)
            continue
        if cur == 0.0 and ci > hi:
            cur = 1.0
        elif cur == 1.0 and ci < lo:
            cur = 0.0
        pos.append(cur)
    return pd.Series(pos, index=c.index)


def _vote_scaled(c: pd.Series, scale: float) -> pd.Series:
    smas = [max(5, round(n * scale)) for n in (50, 100, 200)]
    moms = [max(5, round(n * scale)) for n in (30, 90)]
    parts = [_sma_L(c, n) for n in smas] + [_mom_L(c, n) for n in moms]
    return sum(parts) / len(parts)


def _vol_target(c: pd.Series, base: pd.Series, target_vol: float, cap: float = 1.0) -> pd.Series:
    rv = np.log(c).diff().rolling(20).std() * np.sqrt(252)
    scale = (target_vol / rv).clip(upper=cap).fillna(0.0)
    return base * scale


def candidates(d: pd.DataFrame) -> dict[str, pd.Series]:
    c = d["close"]
    out: dict[str, pd.Series] = {"buyhold": pd.Series(1.0, index=c.index)}
    for n in (20, 50, 100, 150, 200):
        out[f"sma{n}"] = _sma_L(c, n)
    for f, s in ((10, 30), (20, 50), (50, 200)):
        out[f"ema{f}_{s}"] = _ema_L(c, f, s)
    for e, x in ((20, 10), (55, 20)):
        out[f"donch{e}_{x}"] = _donch_L(d, e, x)
    for n in (20, 30, 60, 90, 120, 250):
        out[f"tsmom{n}"] = _mom_L(c, n)
    out["V5"] = vote_target(c)
    for s in (0.5, 0.75, 1.25, 1.5, 2.0):
        out[f"V5x{s}"] = _vote_scaled(c, s)
    singles = [k for k in out if k not in ("buyhold", "V5") and not k.startswith("V5x")]
    out["Vall"] = sum(out[k] for k in singles) / len(singles)
    for tv in (0.10, 0.15, 0.20):
        out[f"V5_vt{int(tv * 100)}"] = _vol_target(c, out["V5"], tv)
    return out


def sharpe(r: pd.Series) -> float:
    v = r.std()
    return float(r.mean() / v * np.sqrt(365)) if v > 0 else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gld-1d", required=True)
    p.add_argument("--qqq-1d", required=True)
    p.add_argument("--scalp-trades", required=True)
    p.add_argument("--start", default="2006-01-01")
    p.add_argument("--lookback-years", type=int, default=5)
    p.add_argument("--long-funding-annual", type=float, default=None,
                   help="覆盖多头资金费年化（如 0.06）；默认沿用 explore.py 的 0.01%%/8h≈11%%/年")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.long_funding_annual is not None:
        import research.time_in_market.explore as explore_mod

        explore_mod.LONG_FUNDING_PER_DAY = args.long_funding_annual / 365.0

    legs = {
        "GLD": load_daily(Path(args.gld_1d), resample_from_intraday=False),
        "QQQ": load_daily(Path(args.qqq_1d), resample_from_intraday=False),
    }
    end = min(d.index[-1] for d in legs.values())
    cal = pd.date_range(pd.Timestamp(args.start, tz="UTC"), end, freq="1D")

    sleeves: dict[str, dict[str, pd.DataFrame]] = {}
    for leg, d in legs.items():
        sleeves[leg] = {name: to_calendar(sleeve_returns(d, tgt, 1.0), cal) for name, tgt in candidates(d).items()}
    names = list(sleeves["GLD"])

    periods = {
        "全样本": (cal[0], end),
        "2006-2013": (pd.Timestamp("2006-01-01", tz="UTC"), pd.Timestamp("2013-12-31", tz="UTC")),
        "2014-2020": (pd.Timestamp("2014-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")),
        "2021-": (pd.Timestamp("2021-01-01", tz="UTC"), end),
    }

    def basket_ret(name: str, weights: dict[str, float] | None = None) -> pd.Series:
        w = weights or {leg: 1 / len(legs) for leg in legs}
        return sum(w[leg] * sleeves[leg][name]["ret"] for leg in legs)

    # 1) 候选全表（篮子层面，两腿同一规则等权）
    rows = []
    for name in names:
        r = basket_ret(name)
        pos = sum(sleeves[leg][name]["pos"] for leg in legs) / len(legs)
        row = {"cand": name}
        for pn, (a, b) in periods.items():
            m = (r.index >= a) & (r.index <= b)
            mm = metrics(r[m], pos[m])
            row.update({f"{pn}|sharpe": mm["sharpe"], f"{pn}|cagr": mm["cagr_pct"], f"{pn}|mdd": mm["mdd_pct"]})
        rows.append(row)
    table = pd.DataFrame(rows)
    ranks = {pn: int(table[f"{pn}|sharpe"].rank(ascending=False)[table.cand == "V5"].iloc[0]) for pn in periods}

    # 2) 走步：每年用过去 N 年 Sharpe 给每条腿选最佳候选
    years = range(cal[0].year + args.lookback_years, end.year + 1)
    wf_parts, picks = [], []
    for y in years:
        ys, ye = pd.Timestamp(f"{y}-01-01", tz="UTC"), min(pd.Timestamp(f"{y}-12-31", tz="UTC"), end)
        ls = ys - pd.DateOffset(years=args.lookback_years)
        leg_rets = []
        for leg in legs:
            scores = {n: sharpe(sleeves[leg][n]["ret"][(cal >= ls) & (cal < ys)]) for n in names}
            best = max(scores, key=lambda k: -np.inf if np.isnan(scores[k]) else scores[k])
            picks.append({"year": y, "leg": leg, "pick": best, "is_sharpe": round(scores[best], 2)})
            leg_rets.append(sleeves[leg][best]["ret"][(cal >= ys) & (cal <= ye)])
        wf_parts.append(sum(leg_rets) / len(leg_rets))
    wf = pd.concat(wf_parts)
    oos_start = wf.index[0]
    oos = {
        "walk_forward_pick": metrics(wf),
        "fixed_V5": metrics(basket_ret("V5")[basket_ret("V5").index >= oos_start]),
        "fixed_Vall": metrics(basket_ret("Vall")[basket_ret("Vall").index >= oos_start]),
        "buyhold": metrics(basket_ret("buyhold")[basket_ret("buyhold").index >= oos_start]),
    }

    # 3) 腿权重：逆波动（用过去 252 日已实现波动，月度更新，无前视）
    def inv_vol_ret(name: str) -> pd.Series:
        vols = {}
        for leg, d in legs.items():
            rv = np.log(d["close"]).diff().rolling(252).std().shift(1)
            vols[leg] = rv.reindex(cal).ffill()
        inv = pd.DataFrame({leg: 1 / v for leg, v in vols.items()})
        w = inv.div(inv.sum(axis=1), axis=0).resample("MS").first().reindex(cal).ffill().fillna(0.5)
        return sum(w[leg] * sleeves[leg][name]["ret"] for leg in legs)

    weighting = {}
    for name in ("V5", "Vall"):
        for label, r in (("equal", basket_ret(name)), ("inv_vol", inv_vol_ret(name))):
            weighting[f"{name}|{label}"] = {pn: metrics(r[(r.index >= a) & (r.index <= b)]) for pn, (a, b) in periods.items()}

    # 4) 与 scalp 组合（2022→）
    cal22 = cal[cal >= pd.Timestamp("2022-01-01", tz="UTC")]
    sc = scalp_daily(Path(args.scalp_trades), cal22)
    windows = {
        "2022起": cal22[0],
        "2025起": pd.Timestamp("2025-01-01", tz="UTC"),
        "近12月": end - pd.Timedelta(days=365),
        "近6月": end - pd.Timedelta(days=182),
        "近3月": end - pd.Timedelta(days=91),
    }
    top = table.sort_values("全样本|sharpe", ascending=False).cand.head(6).tolist()
    combo_names = list(dict.fromkeys(["V5", "Vall", "buyhold", *top]))
    combo = {"scalp_only": {w: metrics(sc["ret"][sc.index >= ws]) for w, ws in windows.items()}}
    for name in combo_names:
        r = sc["ret"] + basket_ret(name).reindex(cal22).fillna(0.0)
        combo[f"scalp+1x{name}"] = {w: metrics(r[r.index >= ws]) for w, ws in windows.items()}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out.with_name(out.stem + "_candidates.csv"), index=False)
    pd.DataFrame(picks).to_csv(out.with_name(out.stem + "_wf_picks.csv"), index=False)
    out.write_text(json.dumps({
        "end": str(end.date()), "n_candidates": len(names), "V5_sharpe_rank": ranks,
        "walk_forward_oos_from": str(oos_start.date()), "walk_forward": oos,
        "weighting": weighting, "combo_with_scalp": combo,
    }, ensure_ascii=False, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
