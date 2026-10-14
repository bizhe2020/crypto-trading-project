"""第 4 步：篮子倍数 k、交易所杠杆（强平风险）、乖离 cap 机制。

A. k（篮子名义 / 账户权益）网格 → 与 scalp 组合的收益/回撤/Calmar；k 受保证金约束：
   篮子保证金 = k / 交易所杠杆 × 权益，scalp 最坏占用约 80%（实盘口径重放最高 16x 有效杠杆 / 20x）。
B. 交易所杠杆 L 的历史强平检验：持仓期内从「建仓/再平衡参考价」到之后最低价的最大跌幅，
   与逐仓强平距离 ≈ 1/L − 维持保证金率 比较（不补保证金的最坏情形）。
C. cap 机制（类比 scalp cap12）：收盘相对 SMA200 乖离 > X% 时，投票清零（hard）或减半（soft）。
口径：已收盘日线、t 收盘信号 → t+1 开盘执行；成本单边 8bp；资金费 --long-funding-annual（默认 6%）。
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

import research.time_in_market.explore as E  # noqa: E402
from strategy.trend_vote import vote_target  # noqa: E402

MMR = 0.005


def leg_sleeve(d: pd.DataFrame, target: pd.Series, cal: pd.DatetimeIndex) -> pd.DataFrame:
    return E.to_calendar(E.sleeve_returns(d, target, 1.0), cal)


def capped_vote(close: pd.Series, cap_pct: float | None, mode: str) -> pd.Series:
    v = vote_target(close)
    if cap_pct is None:
        return v
    sma = close.rolling(200).mean()
    dev = (close / sma - 1) * 100
    over = (dev > cap_pct) & sma.notna()
    return v.where(~over, 0.0 if mode == "hard" else v * 0.5)


def worst_adverse_moves(d: pd.DataFrame, target: pd.Series, band: float = 0.1) -> pd.Series:
    """模拟持仓：仓位 >0 期间，从最近一次（再）建仓的执行价到之后最低价的跌幅（%）。

    再平衡规则与实盘一致：目标变化超过 band（按满仓名义计）才调仓；调仓后参考价重置为
    加权平均成本的近似 —— 这里取保守口径：只有「新开仓」才重置参考价，加仓不摊薄。
    """
    pos = target.shift(1).fillna(0.0)  # 当日开盘生效
    o, low = d["open"], d["low"]
    ref = np.nan
    out = []
    held = 0.0
    for t in d.index:
        p = pos.loc[t]
        if held == 0.0 and p > 0:
            ref = o.loc[t]
        if p == 0:
            ref = np.nan
        held = p
        out.append((ref - low.loc[t]) / ref * 100 if p > 0 and not np.isnan(ref) else np.nan)
    s = pd.Series(out, index=d.index)
    return s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gld-1d", required=True)
    p.add_argument("--qqq-1d", required=True)
    p.add_argument("--scalp-trades", required=True)
    p.add_argument("--long-funding-annual", type=float, default=0.06)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    E.LONG_FUNDING_PER_DAY = args.long_funding_annual / 365.0

    legs = {k: E.load_daily(Path(getattr(args, f"{k.lower()}_1d")), False) for k in ("GLD", "QQQ")}
    end = min(d.index[-1] for d in legs.values())
    cal = pd.date_range(pd.Timestamp("2006-01-01", tz="UTC"), end, freq="1D")
    cal22 = cal[cal >= pd.Timestamp("2022-01-01", tz="UTC")]
    sc = E.scalp_daily(Path(args.scalp_trades), cal22)
    W = {
        "2022起": cal22[0], "2025起": pd.Timestamp("2025-01-01", tz="UTC"),
        "近12月": end - pd.Timedelta(days=365), "近6月": end - pd.Timedelta(days=182), "近3月": end - pd.Timedelta(days=91),
    }
    P = {
        "2006-": (cal[0], end), "2006-2013": (cal[0], pd.Timestamp("2013-12-31", tz="UTC")),
        "2014-2020": (pd.Timestamp("2014-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC")),
        "2021-": (pd.Timestamp("2021-01-01", tz="UTC"), end),
    }

    def basket(cap_pct=None, mode="hard") -> pd.Series:
        return sum(0.5 * leg_sleeve(d, capped_vote(d["close"], cap_pct, mode), cal)["ret"] for d in legs.values())

    def win(r: pd.Series, windows: dict) -> dict:
        return {w: E.metrics(r[r.index >= ws]) for w, ws in windows.items()}

    def per(r: pd.Series) -> dict:
        return {n: E.metrics(r[(r.index >= a) & (r.index <= b)]) for n, (a, b) in P.items()}

    base = basket()
    res: dict = {"scalp_only": win(sc["ret"], W)}

    # A. k 网格
    res["k_grid"] = {}
    for k in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        r = sc["ret"] + k * base.reindex(cal22).fillna(0.0)
        res["k_grid"][str(k)] = {
            "combo": win(r, W),
            "min_exchange_leverage_for_margin_le_20pct": round(k / 0.2, 1),
        }
    # 成本 / 资金费翻倍压力
    E.FEE_SLIP *= 2
    E.LONG_FUNDING_PER_DAY *= 2
    stressed = basket()
    E.FEE_SLIP /= 2
    E.LONG_FUNDING_PER_DAY /= 2
    res["k_grid_stress2x"] = {
        str(k): win(sc["ret"] + k * stressed.reindex(cal22).fillna(0.0), W) for k in (0.5, 1.0, 1.5, 2.0, 3.0)
    }

    # B. 强平检验
    res["liquidation"] = {}
    for name, d in legs.items():
        adv = worst_adverse_moves(d, vote_target(d["close"]))
        res["liquidation"][name] = {
            "max_adverse_pct": round(float(adv.max()), 2),
            "max_adverse_date": str(adv.idxmax().date()),
            **{f"lev{L}_events": int((adv >= (1 / L - MMR) * 100).sum()) for L in (3, 4, 5, 6, 8, 10, 15, 20)},
            "top5": {str(k.date()): round(float(v), 2) for k, v in adv.nlargest(5).items()},
        }

    # C. cap 机制
    res["cap"] = {"none": {"basket": per(base), "combo_k1": win(sc["ret"] + base.reindex(cal22).fillna(0.0), W)}}
    for mode in ("hard", "soft"):
        for cap in (8, 10, 12, 15, 20, 25, 30):
            r = basket(cap, mode)
            res["cap"][f"{mode}{cap}"] = {
                "basket": per(r), "combo_k1": win(sc["ret"] + r.reindex(cal22).fillna(0.0), W),
            }
    # cap 触发频率
    res["cap_trigger_days_pct"] = {
        name: {cap: round(float(((d["close"] / d["close"].rolling(200).mean() - 1) * 100 > cap).mean() * 100), 1)
               for cap in (10, 15, 20, 30)}
        for name, d in legs.items()
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str) + "\n")
    print(out)


if __name__ == "__main__":
    main()
