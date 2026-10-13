"""第 2 步：在场率探索 —— BTC scalp 空仓期能不能用「趋势袖子」把闲置资金用起来。

从零设计，不复用历史 router 打分：
  * 每条腿都只用「已收盘日线」生成信号，次日开盘执行（open→open 收益），不含前视。
  * 成本：单边 fee 5bp + 滑点 3bp；多头资金费按 0.01%/8h（≈11%/年 × 杠杆）计；空头资金费按 0 计（保守）。
  * scalp 用实盘配置无前视重放的逐笔成交（scalp_trades.py），按平仓日记入日收益。
  * 组合 = scalp 日收益 + k × 袖子日收益（袖子挂在不同合约上，用 scalp 闲置保证金，
    每日按权益再平衡）。

输出：每条候选在多个窗口的收益/回撤/在场率，以及 scalp+袖子 组合。
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEE_SLIP = 0.0008  # 单边
LONG_FUNDING_PER_DAY = 0.0003  # 0.01% × 3


def load_daily(path: Path, resample_from_intraday: bool) -> pd.DataFrame:
    df = pd.read_feather(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").drop_duplicates("date")
    if resample_from_intraday:
        df = (
            df.set_index("date")
            .resample("1D", label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
        )
        # 最后一根日线若未收盘则丢掉（严格只用已收盘信息）。
        last_intraday = pd.read_feather(path, columns=["date"])["date"].max()
        if df.index[-1] + pd.Timedelta(days=1) > pd.Timestamp(last_intraday) + pd.Timedelta(minutes=15):
            df = df.iloc[:-1]
    else:
        df["date"] = df["date"].dt.normalize()
        df = df.set_index("date")[["open", "high", "low", "close"]]
    return df


def signal_family(d: pd.DataFrame) -> dict[str, pd.Series]:
    """返回 {名字: 目标仓位(-1/0/1)}，仓位在 t 收盘决定，t+1 开盘生效。"""
    c, h, l = d["close"], d["high"], d["low"]
    out: dict[str, pd.Series] = {}
    for n in (20, 50, 100, 200):
        sma = c.rolling(n).mean()
        out[f"sma{n}_L"] = (c > sma).astype(float).where(sma.notna(), 0.0)
        out[f"sma{n}_LS"] = np.sign(c - sma).where(sma.notna(), 0.0)
    for f, s in ((10, 30), (20, 50), (50, 200)):
        ef, es = c.ewm(span=f, adjust=False).mean(), c.ewm(span=s, adjust=False).mean()
        valid = c.expanding().count() >= s
        out[f"ema{f}_{s}_L"] = (ef > es).astype(float).where(valid, 0.0)
        out[f"ema{f}_{s}_LS"] = np.sign(ef - es).where(valid, 0.0)
    for entry, exit_ in ((20, 10), (55, 20)):
        hh = h.rolling(entry).max().shift(1)
        ll = l.rolling(exit_).min().shift(1)
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
        out[f"donch{entry}_{exit_}_L"] = pd.Series(pos, index=c.index)
    for n in (30, 90):
        mom = c / c.shift(n) - 1
        out[f"tsmom{n}_L"] = (mom > 0).astype(float).where(mom.notna(), 0.0)
    return out


def sleeve_returns(d: pd.DataFrame, target: pd.Series, leverage: float) -> pd.DataFrame:
    """open→open 收益；t 收盘信号 → t+1 开盘建仓 → 持有到 t+2 开盘。"""
    pos = (target.shift(1).fillna(0.0) * leverage)  # 在 t 当天 open 生效的仓位（来自 t-1 收盘）
    o2o = d["open"].shift(-1) / d["open"] - 1  # 当天 open → 次日 open
    turnover = pos.diff().abs().fillna(pos.abs())
    funding = pos.clip(lower=0.0) * LONG_FUNDING_PER_DAY
    ret = pos * o2o - turnover * FEE_SLIP - funding
    return pd.DataFrame({"ret": ret.fillna(0.0), "pos": pos})


def to_calendar(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    # ETF 周末/假日：收益 0，仓位沿用（仍占用资金、计入在场）。
    out = df.reindex(index)
    out["ret"] = out["ret"].fillna(0.0)
    out["pos"] = out["pos"].ffill(limit=4).fillna(0.0)  # 数据截止后不再虚计在场
    return out


def scalp_daily(trades_csv: Path, index: pd.DatetimeIndex, initial_capital: float = 1000.0) -> pd.DataFrame:
    t = pd.read_csv(trades_csv, parse_dates=["entry_time", "exit_time"])
    t = t.sort_values("exit_time")
    equity_before = initial_capital + t["pnl"].cumsum().shift(1).fillna(0.0)
    t["ret"] = t["pnl"] / equity_before
    t["day"] = t["exit_time"].dt.normalize()
    daily = (1 + t.groupby("day")["ret"].apply(lambda r: float(np.prod(1 + r) - 1))).sub(1)
    ret = daily.reindex(index).fillna(0.0)
    inmkt = pd.Series(0.0, index=index)
    for e, x in zip(t["entry_time"], t["exit_time"]):
        # 某日有任意持仓即视为当日在场（日粒度近似）。
        inmkt.loc[e.normalize(): x.normalize()] = 1.0
    hours = pd.Series(0.0, index=index)
    for e, x in zip(t["entry_time"], t["exit_time"]):
        hours.loc[e.normalize()] += (x - e).total_seconds() / 3600.0  # 只做总量统计
    return pd.DataFrame({"ret": ret, "pos": inmkt, "hours": hours})


def metrics(ret: pd.Series, pos: pd.Series | None = None) -> dict[str, float]:
    eq = (1 + ret).cumprod()
    dd = 1 - eq / eq.cummax()
    years = max(len(ret) / 365.0, 1e-9)
    total = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0
    mdd = float(dd.max())
    vol = float(ret.std() * np.sqrt(365))
    out = {
        "ret_pct": round(total * 100, 1),
        "cagr_pct": round(cagr * 100, 1),
        "mdd_pct": round(mdd * 100, 1),
        "calmar": round(cagr / mdd, 2) if mdd > 0 else float("nan"),
        "sharpe": round(float(ret.mean() * 365 / vol), 2) if vol > 0 else float("nan"),
    }
    if pos is not None:
        out["inmkt_pct"] = round(float((pos.abs() > 0).mean() * 100), 1)
    return out


def out_path_votes(output: str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out.with_name(out.stem + "_votes.csv")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--btc-15m", required=True)
    p.add_argument("--gld-1d", required=True)
    p.add_argument("--qqq-1d", required=True)
    p.add_argument("--scalp-trades", required=True)
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    btc = load_daily(Path(args.btc_15m), resample_from_intraday=True)
    legs = {
        "BTC": btc,
        "GLD": load_daily(Path(args.gld_1d), resample_from_intraday=False),
        "QQQ": load_daily(Path(args.qqq_1d), resample_from_intraday=False),
    }
    end = pd.Timestamp(args.end, tz="UTC") if args.end else btc.index[-1]
    cal = pd.date_range(pd.Timestamp(args.start, tz="UTC"), end, freq="1D")
    windows = {
        "2022起": cal[0],
        "2023起": pd.Timestamp("2023-01-01", tz="UTC"),
        "2025起": pd.Timestamp("2025-01-01", tz="UTC"),
        "近12月": end - pd.Timedelta(days=365),
        "近6月": end - pd.Timedelta(days=182),
        "近3月": end - pd.Timedelta(days=91),
    }

    def windowed(ret: pd.Series, pos: pd.Series | None) -> dict[str, dict[str, float]]:
        res = {}
        for name, ws in windows.items():
            m = ret.index >= ws
            res[name] = metrics(ret[m], None if pos is None else pos[m])
        return res

    scalp = scalp_daily(Path(args.scalp_trades), cal)
    rows = [{"leg": "SCALP", "rule": "live", "lev": 1.0, **{f"{w}|{k}": v for w, mm in windowed(scalp["ret"], scalp["pos"]).items() for k, v in mm.items()}}]

    sleeves: dict[tuple[str, str, float], pd.DataFrame] = {}
    for leg, d in legs.items():
        # ETF 数据可能早于窗口：信号用完整历史预热，再截到日历窗口。
        fam = signal_family(d)
        fam["buyhold_L"] = pd.Series(1.0, index=d.index)
        for rule, lev in itertools.product(fam, (1.0, 2.0, 3.0)):
            s = to_calendar(sleeve_returns(d, fam[rule], lev), cal)
            sleeves[(leg, rule, lev)] = s
            rows.append({"leg": leg, "rule": rule, "lev": lev,
                         **{f"{w}|{k}": v for w, mm in windowed(s["ret"], s["pos"]).items() for k, v in mm.items()}})

    table = pd.DataFrame(rows)

    # 组合：scalp + k × 袖子（袖子在独立合约，用闲置保证金）。
    combo_rows = []
    for (leg, rule, lev), s in sleeves.items():
        if lev != 1.0:
            continue
        for k in (0.5, 1.0, 2.0):
            r = scalp["ret"] + k * s["ret"]
            pos = ((scalp["pos"] > 0) | (s["pos"] != 0)).astype(float)
            combo_rows.append({"leg": leg, "rule": rule, "k": k,
                               **{f"{w}|{kk}": v for w, mm in windowed(r, pos).items() for kk, v in mm.items()}})
    combos = pd.DataFrame(combo_rows)

    # 防过拟合：不挑单一参数，每条腿用预先声明的「趋势投票」（5 条规则仓位的平均，0~1），
    # 三腿等权成篮子。规则集在看结果之前固定，不做二次筛选。
    vote_rules = ("sma50_L", "sma100_L", "sma200_L", "tsmom30_L", "tsmom90_L")
    votes: dict[str, pd.DataFrame] = {}
    for leg, d in legs.items():
        fam = signal_family(d)
        target = sum(fam[r] for r in vote_rules) / len(vote_rules)
        votes[leg] = to_calendar(sleeve_returns(d, target, 1.0), cal)
    basket_ret = sum(v["ret"] for v in votes.values()) / len(votes)
    basket_pos = sum(v["pos"] for v in votes.values()) / len(votes)
    vote_rows = []
    for name, (r, pos) in {**{f"vote_{leg}": (v["ret"], v["pos"]) for leg, v in votes.items()},
                           "basket3": (basket_ret, basket_pos)}.items():
        vote_rows.append({"name": name, "k": 0.0,
                          **{f"{w}|{kk}": vv for w, mm in windowed(r, pos).items() for kk, vv in mm.items()}})
        for k in (0.5, 1.0, 2.0, 3.0):
            rr = scalp["ret"] + k * r
            pp = ((scalp["pos"] > 0) | (pos != 0)).astype(float)
            vote_rows.append({"name": f"scalp+{k}x{name}", "k": k,
                              **{f"{w}|{kk}": vv for w, mm in windowed(rr, pp).items() for kk, vv in mm.items()}})
    pd.DataFrame(vote_rows).to_csv(out_path_votes(args.output), index=False)
    exposure = pd.DataFrame({leg: v["pos"] for leg, v in votes.items()})
    exposure.to_csv(Path(args.output).with_name(Path(args.output).stem + "_vote_exposure.csv"))

    corr = {leg: round(float(scalp["ret"].corr(sleeves[(leg, "buyhold_L", 1.0)]["ret"])), 3) for leg in legs}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out.with_name(out.stem + "_sleeves.csv"), index=False)
    combos.to_csv(out.with_name(out.stem + "_combos.csv"), index=False)
    out.write_text(json.dumps({
        "window_starts": {k: str(v.date()) for k, v in windows.items()},
        "end": str(end.date()),
        "scalp_vs_buyhold_daily_corr": corr,
        "scalp_hours_in_market_pct": round(float(scalp["hours"].sum() / (len(cal) * 24) * 100), 2),
        "cost_model": {"fee_slip_per_side": FEE_SLIP, "long_funding_per_day": LONG_FUNDING_PER_DAY},
    }, ensure_ascii=False, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
