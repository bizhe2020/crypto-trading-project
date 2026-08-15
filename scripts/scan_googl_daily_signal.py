"""GOOGL 高倍合约策略 — 日线信号生成器（两段式）。

复用现有 TQQQ strict-candidate 框架（tqqq_cash_strict_utils.run_strict_candidate），
把 GOOGL 日线价格 + SPY 市场 regime + 伯克希尔 13F 信念映射成它需要的列。

两段式设计（conviction 边界 = 伯克希尔首次披露 ALPHABET 持仓的 filing_date）:
    段 1 pre-conviction: 趋势跟随 — GOOGL 收盘 > ma(60) 入场（慢 MA，强复利资产
                          捕获上行 + 避开深跌）+ 10% trailing hard exit
                          + 无 SPY regime 过滤 + 无时间限制。熊市保护模块。
    段 2 conviction:     信念做多 — close>ma(20) 快速入场（信念段快速重入场）
                          + regime 放宽（conviction 代替 SPY 过滤）
                          + trailing / max_hold 关闭（信念穿透回调）。
资本从段 1 连续滚入段 2。

2026-08-15 优化（v0.2）：回测 2007-2026 发现 fast(20)>slow(60) 交叉入场是最大收益拖累。
改为 close>ma60 + 10% trailing、去掉 SPY regime、去掉 max_hold，信号层收益 342.6% → 1967.0%，
maxDD 33.4% → 36.9%（+3.5pp），conviction 段行为不变（close>ma20，在市 55%）。

用法:
    python scripts/scan_googl_daily_signal.py \
        --prices-csv /path/to/价值投资project/data/prices.csv \
        --holdings-csv /path/to/价值投资project/data/berkshire_13f_holdings.csv \
        --out var/runtime/googl/googl_daily_signal.csv

数据源:
    prices.csv       ticker,date,open,close,basis（前复权）
    holdings.csv     filing_date,report_date,issuer_name,shares,value_usd
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.tqqq_cash_strict_utils import run_strict_candidate  # noqa: E402

# --- 段 1 pre-conviction 默认参数（趋势跟随 / 熊市保护） ---
PRE_ENTRY_MA_WINDOW = 60          # GOOGL 收盘 > 60日均线 入场（慢 MA）
PRE_TRAILING_DRAWDOWN_PCT = 10.0  # 10% trailing hard exit
PRE_MAX_HOLD_DAYS = 0             # 0 = 无时间限制
PRE_SLOW_WINDOW = 60              # 保留列（slow MA 不再参与入场，供试验）
PRE_SPY_REGIME_ENABLED = False    # 不使用 SPY regime 过滤（慢 MA 自含回撤管理）

# --- 段 2 conviction 默认参数（信念做多） ---
CONV_TRAILING_DRAWDOWN_PCT = 0.0  # 0 = 关闭 trailing
CONV_MAX_HOLD_DAYS = 0            # 0 = 无时间限制
CONV_FAST_WINDOW = 20

DEFAULT_SWITCH_COST_BPS = 10.0
DEFAULT_INITIAL_CAPITAL = 1000.0
DEFAULT_SPY_REGIME_WINDOW = 200
ALPHABET_KEYWORDS = ("ALPHABET", "GOOGLE")

# 杠杆档位（执行层使用，信号层只产出 desired position）
# 2026-08-16 真实 4h 执行回测定档（0.75x 乘数）：offense 11.2x / base 7.5x / defense 3.8x。
# 原 15x/10x 在真实数据上被严格支配（收益更低 +397% vs +535%、maxDD 更高 90% vs 79%）。
LEVERAGE_OFFENSE = 11.2
LEVERAGE_BASE = 7.5


def load_value_prices(csv_path: Path) -> pd.DataFrame:
    """Load GOOGL + SPY daily rows (ticker, date, open, close) sorted by date."""
    frame = pd.read_csv(csv_path)
    needed = {"ticker", "date", "open", "close"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"prices.csv missing columns: {sorted(missing)}")
    frame = frame[frame["ticker"].isin(["GOOGL", "GOOG", "SPY"])].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise ValueError("no GOOGL/GOOG/SPY rows in prices.csv")
    return frame


def build_conviction_series(
    holdings_csv: Path,
    dates: pd.DatetimeIndex,
) -> pd.Series:
    """Berkshire 13F 信念序列：某日之前最新已披露的 13F 是否持有 ALPHABET。

    用 filing_date（披露日）而非 report_date，避免前视偏差。
    返回与 dates 对齐的 bool Series。
    """
    if not holdings_csv.exists():
        return pd.Series(False, index=dates)
    holdings = pd.read_csv(holdings_csv)
    if "filing_date" not in holdings.columns or "issuer_name" not in holdings.columns:
        return pd.Series(False, index=dates)
    alphabet = holdings[
        holdings["issuer_name"].str.upper().str.contains("|".join(ALPHABET_KEYWORDS), na=False)
        & holdings["shares"].fillna(0).astype(float).gt(0)
    ].copy()
    if alphabet.empty:
        return pd.Series(False, index=dates)
    alphabet["filing_date"] = pd.to_datetime(alphabet["filing_date"], utc=True, errors="coerce")
    alphabet = alphabet.dropna(subset=["filing_date"]).sort_values("filing_date")
    filings = alphabet["filing_date"].unique()
    # conviction_on[t] = latest filing as of t is an ALPHABET holding
    index = pd.DatetimeIndex(pd.to_datetime(dates, utc=True)).sort_values()
    positions = pd.Series(False, index=index)
    active = False
    fidx = 0
    for ts in index:
        while fidx < len(filings) and filings[fidx] <= ts:
            active = True
            fidx += 1
        positions[ts] = active
    return positions.reindex(index).fillna(False)


def build_googl_frame(
    prices: pd.DataFrame,
    conviction: pd.Series,
    *,
    pre_entry_ma_window: int = PRE_ENTRY_MA_WINDOW,
    slow_window: int = PRE_SLOW_WINDOW,
    conv_fast_window: int = CONV_FAST_WINDOW,
    spy_regime_enabled: bool = PRE_SPY_REGIME_ENABLED,
    spy_regime_window: int = DEFAULT_SPY_REGIME_WINDOW,
) -> pd.DataFrame:
    """构建 run_strict_candidate 需要的 GOOGL 日线 frame。

    列:
        tqqq_open / tqqq_close  = GOOGL 开/收（复用 run_strict_candidate 的复利引擎）
        entry_signal            = pre 段 close>ma(pre_entry_ma_window)
                                  （段 2 在 run_googl_signal 里覆写为 close>ma）
        ixic_trend_label        = "ixic_up"（无 regime）或 SPY>spy_regime_window / 信念开
        vix_label / rel_strength_label = 常量（v1 简化）
        berkshire_conviction    = 信念序列
    """
    googl = prices[prices["ticker"].eq("GOOGL")][["date", "open", "close"]].rename(
        columns={"open": "googl_open", "close": "googl_close"}
    )
    spy = prices[prices["ticker"].eq("SPY")][["date", "close"]].rename(
        columns={"close": "spy_close"}
    )
    # GOOGL 缺失时用 GOOG（Alphabet C 类）补齐
    if googl.empty:
        googl = prices[prices["ticker"].eq("GOOG")][["date", "open", "close"]].rename(
            columns={"open": "googl_open", "close": "googl_close"}
        )
    frame = googl.sort_values("date").reset_index(drop=True)
    if not spy.empty:
        frame = pd.merge_asof(frame, spy.sort_values("date"), on="date", direction="backward")
        frame["spy_200ma"] = frame["spy_close"].rolling(spy_regime_window).mean()
    conviction_reindexed = conviction.reindex(frame["date"]).fillna(False).to_numpy()

    frame["fast_ma"] = frame["googl_close"].rolling(pre_entry_ma_window).mean()
    frame["slow_ma"] = frame["googl_close"].rolling(slow_window).mean()
    frame["conv_fast_ma"] = frame["googl_close"].rolling(conv_fast_window).mean()
    # pre 段入场：GOOGL 收盘 > 入场均线（慢 MA，强复利资产捕获上行 + 避开深跌）
    frame["entry_signal"] = (frame["googl_close"] > frame["fast_ma"]).astype(int)
    if spy_regime_enabled and not spy.empty:
        # regime: SPY>window 或 conviction → ixic_up（conviction 放宽 SPY 过滤）
        spy_up = frame["spy_close"].gt(frame["spy_200ma"]).fillna(False)
        frame["ixic_trend_label"] = pd.Series(
            ["ixic_up" if (bool(spy_up.iloc[i]) or bool(conviction_reindexed[i])) else "ixic_down" for i in range(len(frame))]
        )
    else:
        # 无 regime 过滤：全程 ixic_up（慢 MA 自含回撤管理，conviction 段同向）
        frame["ixic_trend_label"] = "ixic_up"
    frame["vix_label"] = "vix_low"
    frame["rel_strength_label"] = "qqq_neutral"
    frame["tqqq_open"] = frame["googl_open"]
    frame["tqqq_close"] = frame["googl_close"]
    frame["qqq_close"] = frame["googl_close"]
    frame["berkshire_conviction"] = conviction_reindexed
    frame = frame.dropna(subset=["fast_ma", "slow_ma", "conv_fast_ma", "tqqq_open", "tqqq_close"]).reset_index(drop=True)
    return frame


def _run_segment(
    frame: pd.DataFrame,
    *,
    entry_signal_col: str,
    trailing_drawdown_pct: float,
    max_hold_days: int,
    initial_capital: float,
    regime_filter: str,
    force_regime_up: bool,
    switch_cost_bps: float,
) -> pd.DataFrame:
    """Run run_strict_candidate on a segment, returning its path."""
    seg = frame.reset_index(drop=True).copy()
    seg["entry_signal"] = seg[entry_signal_col].astype(int)
    if force_regime_up:
        seg["ixic_trend_label"] = "ixic_up"
    result = run_strict_candidate(
        seg,
        regime_filter=regime_filter,
        max_hold_days=max_hold_days,
        trailing_lookback_days=10 if trailing_drawdown_pct > 0 else 0,
        trailing_drawdown_pct=trailing_drawdown_pct,
        switch_cost_bps=switch_cost_bps,
        initial_capital=initial_capital,
        de_risk_signal_name="off",
        recovery_reentry_rule="off",
        hard_exit_reset_signal="main_desired",
        drawdown_ladder_enabled=False,
    )
    return result["path"]


def run_googl_signal(
    prices_csv: Path,
    holdings_csv: Path,
    *,
    regime_filter: str = "ixic_filter",
    pre_entry_ma_window: int = PRE_ENTRY_MA_WINDOW,
    pre_slow_window: int = PRE_SLOW_WINDOW,
    conv_fast_window: int = CONV_FAST_WINDOW,
    pre_trailing_drawdown_pct: float = PRE_TRAILING_DRAWDOWN_PCT,
    pre_max_hold_days: int = PRE_MAX_HOLD_DAYS,
    conv_trailing_drawdown_pct: float = CONV_TRAILING_DRAWDOWN_PCT,
    conv_max_hold_days: int = CONV_MAX_HOLD_DAYS,
    pre_spy_regime_enabled: bool = PRE_SPY_REGIME_ENABLED,
    pre_spy_regime_window: int = DEFAULT_SPY_REGIME_WINDOW,
    switch_cost_bps: float = DEFAULT_SWITCH_COST_BPS,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> dict[str, Any]:
    prices = load_value_prices(prices_csv)
    conviction = build_conviction_series(holdings_csv, pd.DatetimeIndex(prices["date"].unique()))
    frame = build_googl_frame(
        prices,
        conviction,
        pre_entry_ma_window=pre_entry_ma_window,
        slow_window=pre_slow_window,
        conv_fast_window=conv_fast_window,
        spy_regime_enabled=pre_spy_regime_enabled,
        spy_regime_window=pre_spy_regime_window,
    )
    conviction_dates = frame.loc[frame["berkshire_conviction"], "date"]
    if conviction_dates.empty:
        # 无信念 → 纯 pre-conviction 趋势跟随
        path = _run_segment(
            frame,
            entry_signal_col="entry_signal",
            trailing_drawdown_pct=pre_trailing_drawdown_pct,
            max_hold_days=pre_max_hold_days,
            initial_capital=initial_capital,
            regime_filter=regime_filter,
            force_regime_up=False,
            switch_cost_bps=switch_cost_bps,
        )
        first_conv = None
    else:
        first_conv = conviction_dates.min()
        pre = frame[frame["date"] < first_conv]
        post = frame[frame["date"] >= first_conv].reset_index(drop=True)
        # 段 2 入场条件 = close > conv_fast_ma（快重入场，捕获信念段上行）
        post["conv_entry_signal"] = (post["googl_close"] > post["conv_fast_ma"]).astype(int)
        r1 = _run_segment(
            pre,
            entry_signal_col="entry_signal",
            trailing_drawdown_pct=pre_trailing_drawdown_pct,
            max_hold_days=pre_max_hold_days,
            initial_capital=initial_capital,
            regime_filter=regime_filter,
            force_regime_up=False,
            switch_cost_bps=switch_cost_bps,
        )
        cap1 = float(r1["capital"].iloc[-1])
    r2 = _run_segment(
        post,
        entry_signal_col="conv_entry_signal",
        trailing_drawdown_pct=conv_trailing_drawdown_pct,
        max_hold_days=conv_max_hold_days,
        initial_capital=cap1,
        regime_filter=regime_filter,
        force_regime_up=True,
        switch_cost_bps=switch_cost_bps,
    )
    path = pd.concat([r1, r2], ignore_index=True)
    path["position"] = path["position"].replace({"TQQQ": "GOOGL", "CASH": "FLAT"})
    path["date"] = pd.to_datetime(path["date"], utc=True)
    conviction_reindexed = conviction.reindex(path["date"]).fillna(False)
    path["berkshire_conviction"] = conviction_reindexed.to_numpy()

    # 杠杆档位（执行层参考）：conviction 在市 → offense，非信念在市 → base
    in_market = path["position"].eq("GOOGL")
    path["leverage_tier"] = "flat"
    path.loc[in_market & path["berkshire_conviction"], "leverage_tier"] = "offense"
    path.loc[in_market & ~path["berkshire_conviction"], "leverage_tier"] = "base"
    path["target_leverage"] = 0.0
    path.loc[path["leverage_tier"].eq("offense"), "target_leverage"] = LEVERAGE_OFFENSE
    path.loc[path["leverage_tier"].eq("base"), "target_leverage"] = LEVERAGE_BASE
    return {"frame": frame, "path": path, "conviction_start": first_conv}


def summarize(path: pd.DataFrame) -> dict[str, Any]:
    if path.empty:
        return {}
    capital = path["capital"]
    total_return_pct = (capital.iloc[-1] / capital.iloc[0] - 1.0) * 100.0
    peak = capital.cummax()
    max_dd_pct = ((peak - capital) / peak.replace(0, pd.NA) * 100.0).max(skipna=True) or 0.0
    in_market = path["position"].eq("GOOGL")
    days_in = int(in_market.sum())
    trades = int(path["entered_today"].sum())
    conviction_days = int(path["berkshire_conviction"].astype(bool).sum())
    conviction_in = int((in_market & path["berkshire_conviction"].astype(bool)).sum())
    leverage_tiers = path.groupby("leverage_tier")["date"].count().to_dict()
    return {
        "start": str(path["date"].iloc[0].date()),
        "end": str(path["date"].iloc[-1].date()),
        "total_return_pct": round(float(total_return_pct), 2),
        "max_drawdown_pct": round(float(max_dd_pct), 2),
        "days_in_market": days_in,
        "trades": trades,
        "conviction_days_total": conviction_days,
        "conviction_days_in_market": conviction_in,
        "conviction_in_market_pct": round(conviction_in / conviction_days * 100.0, 1) if conviction_days else 0.0,
        "leverage_tier_days": {k: int(v) for k, v in leverage_tiers.items()},
        "final_capital": round(float(capital.iloc[-1]), 2),
        "latest_position": str(path["position"].iloc[-1]),
        "latest_date": str(path["date"].iloc[-1].date()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GOOGL 高倍合约策略日线信号生成（两段式）")
    parser.add_argument("--prices-csv", required=True, help="价值项目 prices.csv 路径")
    parser.add_argument("--holdings-csv", required=True, help="伯克希尔 13F holdings.csv 路径")
    parser.add_argument("--out", default="var/runtime/googl/googl_daily_signal.csv", help="输出信号 CSV")
    parser.add_argument("--pre-trailing-drawdown-pct", type=float, default=PRE_TRAILING_DRAWDOWN_PCT)
    parser.add_argument("--conv-max-hold-days", type=int, default=CONV_MAX_HOLD_DAYS)
    args = parser.parse_args()

    payload = run_googl_signal(
        Path(args.prices_csv),
        Path(args.holdings_csv),
        pre_trailing_drawdown_pct=args.pre_trailing_drawdown_pct,
        conv_max_hold_days=args.conv_max_hold_days,
    )
    path = payload["path"]
    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    path[
        ["date", "position", "berkshire_conviction", "leverage_tier", "target_leverage"]
    ].to_csv(out_path, index=False)

    summary = summarize(path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("signal written to:", out_path)


if __name__ == "__main__":
    main()
