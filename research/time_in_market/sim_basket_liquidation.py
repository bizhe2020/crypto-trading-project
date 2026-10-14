"""第 5 步：接受强平时，篮子倍数 k × 交易所杠杆 L 的「同账户」逐日模拟。

与 explore.py 的「日收益线性叠加」不同，这里按实盘机制模拟一个账户：
  * 账户权益 E = 现金 + 篮子浮盈；scalp 按 E 复利（实盘 scalp 用 totalEq 定仓），日收益来自实盘口径逐笔。
  * 篮子每腿逐仓：持仓 q、均价 A、保证金 M。每个 ETF 交易日开盘按前一日收盘投票再平衡
    （偏离 > band × 满仓名义才调，和实盘一致）；加仓追加保证金 Δnotional / L，减仓按比例释放。
  * 强平：当日最低价 ≤ A ×(1 − M/(q·A) + mmr) → 损失全部保证金 M，仓位归零，次日按信号重新开。
  * scalp 抢占：scalp 开仓日若 篮子保证金/E + scalp 所需保证金(有效杠杆/20) > 98%，该笔 scalp 视为被拒单。
  * 成本：单边 8bp；多头资金费 --long-funding-annual（默认 6%）。信号只用已收盘日线。
局限：ETF 日线看不到永续夜盘/周末插针（强平次数是下限）；被拒的 scalp 单不回灌 shadow gate 状态。
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
SCALP_LEVERAGE = 20.0
MARGIN_SAFETY = 0.98


def load_scalp_trades(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path, parse_dates=["entry_time", "exit_time"]).sort_values("entry_time").reset_index(drop=True)
    cap = 1000 + t["pnl"].cumsum().shift(1).fillna(0.0)
    t["eff_lev"] = t["quantity"] * t["entry_price"] / cap
    t["ret"] = t["pnl"] / cap  # 相对开仓时权益的收益率（scalp 按总权益定仓）
    t["entry_day"] = t["entry_time"].dt.normalize()
    t["exit_day"] = t["exit_time"].dt.normalize()
    return t


def simulate(
    legs: dict[str, pd.DataFrame],
    votes: dict[str, pd.Series],
    trades: pd.DataFrame | None,
    cal: pd.DatetimeIndex,
    *,
    k: float,
    lev: float,
    band: float = 0.1,
    weight: float = 0.5,
    skip_mask: np.ndarray | None = None,
) -> dict:
    fee = E.FEE_SLIP
    fund = E.LONG_FUNDING_PER_DAY
    cash = 1.0
    st = {n: {"q": 0.0, "A": 0.0, "M": 0.0} for n in legs}
    last_close = {n: np.nan for n in legs}
    eq_path, margin_frac_path, daily_pnl = [], [], []
    liqs: list[tuple[str, str]] = []
    exits_by_day: dict[pd.Timestamp, list[int]] = {}
    if trades is not None:
        for i, d in enumerate(trades["exit_day"]):
            exits_by_day.setdefault(d, []).append(i)

    def upl() -> float:
        return sum(s["q"] * (last_close[n] - s["A"]) for n, s in st.items() if s["q"] > 0 and not np.isnan(last_close[n]))

    prev_eq = 1.0
    for day in cal:
        # 1) 篮子：开盘再平衡 → 盘中强平检查 → 收盘资金费
        for n, d in legs.items():
            if day not in d.index:
                continue
            o, lo, c = d.at[day, "open"], d.at[day, "low"], d.at[day, "close"]
            s = st[n]
            v = votes[n].get(day, 0.0)  # 已 shift：当日开盘生效的是前一日收盘的投票
            equity_open = cash + sum(
                ss["q"] * ((o if nn == n else last_close[nn]) - ss["A"])
                for nn, ss in st.items() if ss["q"] > 0 and not np.isnan(last_close[nn])
            )
            target = k * max(equity_open, 0.0) * weight * v
            cur = s["q"] * o
            if v <= 0 and s["q"] > 0:
                cash += s["q"] * (o - s["A"]) - cur * fee
                s.update(q=0.0, A=0.0, M=0.0)
            elif v > 0 and (s["q"] == 0 or abs(target - cur) > band * k * equity_open * weight):
                dn = target - cur
                if dn == 0:
                    pass
                elif dn > 0:
                    add_q = dn / o
                    s["A"] = (s["A"] * s["q"] + o * add_q) / (s["q"] + add_q)
                    s["q"] += add_q
                    s["M"] += dn / lev
                else:
                    red_q = -dn / o
                    cash += red_q * (o - s["A"])
                    s["M"] *= (s["q"] - red_q) / s["q"]
                    s["q"] -= red_q
                cash -= abs(dn) * fee
            if s["q"] > 0:
                liq_px = s["A"] * (1 - s["M"] / (s["q"] * s["A"]) + MMR)
                if lo <= liq_px:
                    cash -= s["M"]
                    liqs.append((n, str(day.date())))
                    s.update(q=0.0, A=0.0, M=0.0)
                else:
                    cash -= s["q"] * c * fund
            last_close[n] = c
        eq_basket_close = cash + upl()
        # 2) scalp：当日平仓的逐笔按开仓时权益复利（被拒单的跳过）
        if trades is not None:
            for i in exits_by_day.get(day, []):
                if skip_mask is not None and skip_mask[i]:
                    continue
                cash += trades.at[i, "ret"] * max(prev_eq, 0.0)
        eq = cash + upl()
        bm = sum(s["M"] for s in st.values())
        margin_frac_path.append(bm / eq if eq > 0 else 1.0)
        daily_pnl.append(eq - prev_eq)
        eq_path.append(eq)
        prev_eq = eq
        _ = eq_basket_close
    eqs = pd.Series(eq_path, index=cal)
    return {"equity": eqs, "margin_frac": pd.Series(margin_frac_path, index=cal),
            "pnl": pd.Series(daily_pnl, index=cal), "liquidations": liqs}


def summarize(res: dict, start: pd.Timestamp) -> dict:
    eq = res["equity"][res["equity"].index >= start]
    pnl = res["pnl"][res["pnl"].index >= start]
    eq = eq / eq.iloc[0]
    years = len(eq) / 365.0
    total = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0
    mdd = float((1 - eq / eq.cummax()).max())
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "ret_pct": round(total * 100, 1), "cagr_pct": round(cagr * 100, 1), "mdd_pct": round(mdd * 100, 1),
        "calmar": round(cagr / mdd, 2) if mdd > 0 else None,
        "profit_factor_daily": round(float(gains / losses), 3) if losses > 0 else None,
        "liquidations": sum(1 for _, d in res["liquidations"] if pd.Timestamp(d, tz="UTC") >= start),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gld-1d", required=True)
    p.add_argument("--qqq-1d", required=True)
    p.add_argument("--scalp-trades", required=True)
    p.add_argument("--long-funding-annual", type=float, default=0.06)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    E.LONG_FUNDING_PER_DAY = args.long_funding_annual / 365.0

    legs = {"GLD": E.load_daily(Path(args.gld_1d), False), "QQQ": E.load_daily(Path(args.qqq_1d), False)}
    votes = {n: vote_target(d["close"]).shift(1).fillna(0.0) for n, d in legs.items()}
    end = min(d.index[-1] for d in legs.values())
    trades = load_scalp_trades(Path(args.scalp_trades))
    cal22 = pd.date_range(pd.Timestamp("2022-01-01", tz="UTC"), end, freq="1D")
    cal06 = pd.date_range(pd.Timestamp("2006-01-01", tz="UTC"), end, freq="1D")
    W = {"2022起": cal22[0], "2025起": pd.Timestamp("2025-01-01", tz="UTC"),
         "近12月": end - pd.Timedelta(days=365), "近6月": end - pd.Timedelta(days=182)}

    out: dict = {"scalp_only": {w: summarize(simulate(legs, votes, trades, cal22, k=0.0, lev=1.0), s) for w, s in W.items()}}
    grid = {}
    for k in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
        for lev in (6, 10, 15, 20, 30):
            first = simulate(legs, votes, trades, cal22, k=k, lev=lev)
            mf = first["margin_frac"]
            skip = np.array([
                (mf.get(d - pd.Timedelta(days=1), 0.0) + e / SCALP_LEVERAGE) > MARGIN_SAFETY
                for d, e in zip(trades["entry_day"], trades["eff_lev"])
            ])
            res = simulate(legs, votes, trades, cal22, k=k, lev=lev, skip_mask=skip)
            alone = simulate(legs, votes, None, cal06, k=k, lev=lev)
            grid[f"k{k}_L{lev}"] = {
                "combo": {w: summarize(res, s) for w, s in W.items()},
                "scalp_rejected": int(skip[(trades["entry_day"] >= cal22[0]).values].sum()),
                "basket_margin_max_pct": round(float(mf.max() * 100), 1),
                "basket_alone_2006": summarize(alone, cal06[0]),
                "basket_alone_liq_dates": alone["liquidations"],
            }
    out["grid"] = grid
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n")
    print(path)


if __name__ == "__main__":
    main()
