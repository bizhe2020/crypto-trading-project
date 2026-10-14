"""QQQ 杠杆腿迭代研究：在 V5 投票之上叠加宏观/波动率/估值信号。

候选全部在看结果之前按外部文献声明（见 README §5「QQQ 腿研究」）：
  * 趋势：V5（现行）、SMA200 单规则（Gayed & Bilello 2016 "Leverage for the Long Run"）。
  * 波动率目标：已实现波动（Moreira & Muir 2017；Cederburg et al. 2020 质疑样本外）、
    下行波动（Wang & Yan 2021 "Downside risk and the performance of volatility-managed portfolios"）、
    隐含波动 VXN。允许上限 cap>1 即「平静时加杠杆」。
  * 恐慌：VIX 水平阈值、VIX/VIX3M 期限结构倒挂（backwardation → 降仓）。
  * 利率：10Y（^TNX）60 日变动 > 阈值（利率冲击）、10Y-2Y 倒挂、10Y 实际利率水平。
  * 估值：Shiller CAPE（S&P500，月度，滞后 1 个月使用）滚动 20 年分位。
口径：只用 t 日收盘（FRED 序列再多滞后 1 日）→ t+1 开盘执行（open→open），单边 8bp，
多头资金费按年化 --funding（默认 6%）。仓位 = 腿名义 / 账户权益（现行 QQQ 腿 = 1.0 × 投票）。
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

from strategy.trend_vote import vote_target  # noqa: E402

NY = "America/New_York"
FEE = 0.0008


def _ny_daily(path: Path) -> pd.DataFrame:
    d = pd.read_feather(path)
    d["day"] = pd.to_datetime(d["date"], utc=True).dt.tz_convert(NY).dt.normalize().dt.tz_localize(None)
    return d.drop_duplicates("day", keep="last").set_index("day")[["open", "high", "low", "close"]]


def _fred(path: Path, lag_days: int = 1) -> pd.Series:
    s = pd.read_csv(path)
    s.columns = ["day", "v"]
    s["day"] = pd.to_datetime(s["day"])
    s["v"] = pd.to_numeric(s["v"], errors="coerce")
    s = s.dropna().set_index("day")["v"]
    s.index = s.index + pd.Timedelta(days=lag_days)  # 发布滞后
    return s


def _cape(path: Path) -> pd.Series:
    # 文件行形如 "Sep 1, 2026,40.90"（日期本身含逗号）
    recs = [line.rsplit(",", 1) for line in path.read_text().splitlines()[1:] if line.strip()]
    c = pd.DataFrame(recs, columns=["date", "cape"])
    c["cape"] = c["cape"].astype(float)
    c["day"] = pd.to_datetime(c["date"], format="%b %d, %Y")
    c = c[c["day"].dt.day == 1].set_index("day")["cape"].sort_index()  # 只用月初定值（丢掉当月实时估计）
    c.index = c.index + pd.offsets.MonthBegin(1)  # 月度数据滞后 1 个月才可用
    return c


def load(raw: Path, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = _ny_daily(raw / "QQQ-1d.feather")
    q = q[q.index < end]
    f = pd.DataFrame(index=q.index)
    for name, file in (("vix", "^VIX"), ("vix3m", "^VIX3M"), ("vxn", "^VXN"), ("tnx", "^TNX")):
        f[name] = _ny_daily(raw / f"{file}-1d.feather")["close"].reindex(q.index).ffill(limit=3)
    for name, file in (("t10y2y", "T10Y2Y"), ("real10", "DFII10")):
        f[name] = _fred(raw / f"{file}.csv").reindex(q.index, method="ffill")
    f["cape"] = _cape(raw / "cape.csv").reindex(q.index, method="ffill")
    return q, f


def sleeve(q: pd.DataFrame, target: pd.Series, funding_annual: float) -> pd.Series:
    """target 在 t 收盘决定，t+1 开盘生效；返回日收益（相对权益）。"""
    pos = target.shift(1).fillna(0.0)
    o2o = q["open"].shift(-1) / q["open"] - 1
    turnover = pos.diff().abs().fillna(pos.abs())
    gap_days = pd.Series(q.index, index=q.index).diff().dt.days.shift(-1).fillna(1.0)  # 周末/假日按自然日计资金费
    funding = pos.clip(lower=0) * funding_annual / 365.0 * gap_days
    return (pos * o2o - turnover * FEE - funding).fillna(0.0)


def metrics(r: pd.Series, pos: pd.Series | None = None) -> dict:
    eq = (1 + r).cumprod()
    years = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    mdd = float((1 - eq / eq.cummax()).max())
    sh = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")
    out = {"cagr": round(cagr * 100, 1), "mdd": round(mdd * 100, 1), "sharpe": round(sh, 2),
           "calmar": round(cagr / mdd, 2) if mdd > 0 else None}
    if pos is not None:
        out["avg_expo"] = round(float(pos.mean()), 2)
    return out


def candidates(q: pd.DataFrame, f: pd.DataFrame) -> dict[str, pd.Series]:
    c = q["close"]
    lr = np.log(c).diff()
    rv20 = lr.rolling(20).std() * np.sqrt(252)
    dv20 = lr.clip(upper=0).rolling(20).apply(lambda x: np.sqrt((x ** 2).mean()), raw=True) * np.sqrt(252) * np.sqrt(2)
    v5 = vote_target(c)
    sma200 = (c > c.rolling(200).mean()).astype(float).where(c.rolling(200).mean().notna(), 0.0)
    out: dict[str, pd.Series] = {"V5": v5, "SMA200": sma200, "buyhold": pd.Series(1.0, index=c.index)}

    def scale(base, vol, tv, cap):
        return (base * (tv / vol).clip(upper=cap)).where(vol.notna(), base)

    for tv in (0.15, 0.20, 0.25):
        for cap in (1.0, 1.5, 2.0):
            out[f"V5_rv{int(tv*100)}_cap{cap}"] = scale(v5, rv20, tv, cap)
            out[f"V5_dv{int(tv*100)}_cap{cap}"] = scale(v5, dv20, tv, cap)
            out[f"V5_vxn{int(tv*100)}_cap{cap}"] = scale(v5, f["vxn"] / 100, tv, cap)
    for x in (20, 25, 30):
        out[f"V5_vix>{x}_off"] = v5.where(~(f["vix"] > x), 0.0)
        out[f"V5_vix>{x}_half"] = v5.where(~(f["vix"] > x), v5 * 0.5)
    ts = f["vix"] / f["vix3m"]
    out["V5_vixTS>1_off"] = v5.where(~(ts > 1.0), 0.0)
    out["V5_vixTS>1_half"] = v5.where(~(ts > 1.0), v5 * 0.5)
    out["V5_vixTS>0.95_half"] = v5.where(~(ts > 0.95), v5 * 0.5)
    d60 = f["tnx"] - f["tnx"].shift(60)
    for bp in (0.5, 0.75, 1.0):
        out[f"V5_y10up{int(bp*100)}bp_half"] = v5.where(~(d60 > bp), v5 * 0.5)
    out["V5_curveinv_half"] = v5.where(~(f["t10y2y"] < 0), v5 * 0.5)
    for lvl in (2.0, 2.5):
        out[f"V5_real10>{lvl}_half"] = v5.where(~(f["real10"] > lvl), v5 * 0.5)
    pct = f["cape"].rolling(252 * 20, min_periods=252 * 10).rank(pct=True)
    for p in (0.8, 0.9):
        out[f"V5_cape>{int(p*100)}pct_x0.75"] = v5.where(~(pct > p), v5 * 0.75)
    # 预先声明的组合：期限结构过滤 + 已实现波动目标（文献里最常见的两件套）
    for cap in (1.0, 1.5):
        out[f"V5_vixTS_half+rv20_cap{cap}"] = scale(out["V5_vixTS>1_half"], rv20, 0.20, cap)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--funding", type=float, default=0.06)
    p.add_argument("--end", default="2026-10-14")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    q, f = load(Path(args.raw), pd.Timestamp(args.end))
    cands = candidates(q, f)
    periods = {"2007-": ("2007-01-01", None), "2007-2013": ("2007-01-01", "2013-12-31"),
               "2014-2020": ("2014-01-01", "2020-12-31"), "2021-": ("2021-01-01", None),
               "2022-": ("2022-01-01", None), "近12月": (str((q.index[-1] - pd.Timedelta(days=365)).date()), None)}
    rows = []
    for name, tgt in cands.items():
        tgt = tgt.fillna(0.0)
        for fund_label, fund in (("base", args.funding), ("fund2x", args.funding * 2)):
            r = sleeve(q, tgt, fund)
            row = {"cand": name, "funding": fund_label}
            for pn, (a, b) in periods.items():
                m = (r.index >= a) & ((r.index <= b) if b else True)
                for k, v in metrics(r[m], tgt.shift(1).fillna(0)[m]).items():
                    row[f"{pn}|{k}"] = v
            rows.append(row)
    table = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    # 当前读数（t=最后一个已收盘日）
    last = q.index[-1]
    snap = {k: (None if pd.isna(v) else round(float(v), 3)) for k, v in f.loc[last].items()}
    snap["vix_ts"] = round(float(f.loc[last, "vix"] / f.loc[last, "vix3m"]), 3)
    snap["tnx_60d_change"] = round(float(f["tnx"].iloc[-1] - f["tnx"].iloc[-61]), 3)
    snap["cape_20y_pct"] = round(float(f["cape"].rolling(252 * 20, min_periods=252 * 10).rank(pct=True).iloc[-1]), 3)
    snap["rv20"] = round(float(np.log(q["close"]).diff().rolling(20).std().iloc[-1] * np.sqrt(252)), 3)
    snap["targets_now"] = {k: round(float(v.iloc[-1]), 3) for k, v in cands.items()}
    out.with_suffix(".snapshot.json").write_text(json.dumps({"last_day": str(last.date()), **snap}, ensure_ascii=False, indent=2))
    print(out)


if __name__ == "__main__":
    main()
