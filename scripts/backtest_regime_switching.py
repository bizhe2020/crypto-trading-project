#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot.okx_executor import ExecutorConfig
from strategy.scalp_robust_v2_core import ScalpRobustEngine, dataframe_to_candles


CONFIG_PATH = ROOT / "config" / "config.live.5x-3pct.json"
DATA_ROOT = Path("/Users/laoji/projects/crypto-trading-project/deployment/data/okx/futures")
DATA_15M = DATA_ROOT / "BTC_USDT_USDT-15m-futures.feather"
DATA_4H = DATA_ROOT / "BTC_USDT_USDT-4h-futures.feather"
START_15M = pd.Timestamp("2023-01-01", tz="UTC")
END = pd.Timestamp("2026-04-18 23:59:59", tz="UTC")
SPLIT_START = pd.Timestamp("2025-01-01", tz="UTC")
INITIAL_CAPITAL = 1000.0


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_feather(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def build_base_config_dict() -> dict[str, Any]:
    payload = load_config()
    payload.update(
        {
            "initial_capital": INITIAL_CAPITAL,
            "pullback_window": 40,
            "sl_buffer_pct": 1.25,
            "bull_strong_long_risk_per_trade": 0.06,
            "bear_strong_short_risk_per_trade": 0.00001,
            "enable_target_rr_cap": True,
            "short_strong_rr_ratio": 5.0,
            "enable_atr_trailing": True,
            "atr_activation_rr": 2.06,
            "atr_loose_multiplier": 2.7,
            "atr_normal_multiplier": 2.25,
            "atr_tight_multiplier": 1.8,
            "atr_regime_filter": "tight_style_off",
            "disable_fixed_target_exit": False,
        }
    )
    return payload


def make_strategy_config(overrides: dict[str, Any] | None = None):
    payload = build_base_config_dict()
    if overrides:
        payload.update(overrides)
    return ExecutorConfig.from_dict(payload).to_scalp_strategy_config()


def run_engine(config) -> tuple[ScalpRobustEngine, pd.DataFrame]:
    df15 = load_dataframe(DATA_15M)
    df4 = load_dataframe(DATA_4H)
    df15 = df15[(df15["date"] >= START_15M) & (df15["date"] <= END)].reset_index(drop=True)
    df4 = df4[df4["date"] <= END].reset_index(drop=True)

    engine = ScalpRobustEngine.from_candles(
        dataframe_to_candles(df4),
        dataframe_to_candles(df15),
        config,
    )
    engine.run_backtest(start_date="2023-01-01")
    trades = pd.DataFrame([trade.__dict__ for trade in engine.trades])
    if trades.empty:
        return engine, trades
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
    trades = trades.sort_values("exit_time").reset_index(drop=True)
    return engine, trades


def summarize_period(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    if trades.empty:
        return {
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "start_capital": INITIAL_CAPITAL,
            "end_capital": INITIAL_CAPITAL,
        }

    before = trades[trades["exit_time"] < start]
    during = trades[(trades["exit_time"] >= start) & (trades["exit_time"] <= end)].copy()

    start_capital = INITIAL_CAPITAL + float(before["pnl"].sum())
    capital = start_capital
    peak = start_capital
    max_drawdown_pct = 0.0

    for _, trade in during.iterrows():
        capital += float(trade["pnl"])
        if capital > peak:
            peak = capital
        if peak > 0:
            drawdown = (peak - capital) / peak * 100.0
            if drawdown > max_drawdown_pct:
                max_drawdown_pct = drawdown

    win_rate = float((during["pnl"] > 0).mean() * 100.0) if not during.empty else 0.0
    trade_returns = during["pnl"] / start_capital if start_capital > 0 else pd.Series(dtype=float)
    sharpe_ratio = 0.0
    if len(trade_returns) > 1 and float(trade_returns.std(ddof=1)) > 0:
        sharpe_ratio = float((trade_returns.mean() / trade_returns.std(ddof=1)) * (len(trade_returns) ** 0.5))

    total_return_pct = 0.0
    if start_capital > 0:
        total_return_pct = (capital / start_capital - 1.0) * 100.0

    return {
        "total_return_pct": round(total_return_pct, 2),
        "sharpe_ratio": round(sharpe_ratio, 3),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "total_trades": int(len(during)),
        "win_rate": round(win_rate, 2),
        "start_capital": round(start_capital, 2),
        "end_capital": round(capital, 2),
    }


def build_monthly_returns(trades: pd.DataFrame) -> pd.DataFrame:
    capital = INITIAL_CAPITAL
    rows: list[dict[str, Any]] = []
    for month in pd.period_range("2023-01", "2026-04", freq="M"):
        mask = trades["exit_time"].dt.to_period("M") == month if not trades.empty else []
        grp = trades[mask] if not trades.empty else pd.DataFrame()
        start_capital = capital
        pnl = float(grp["pnl"].sum()) if not grp.empty else 0.0
        capital += pnl
        ret = (pnl / start_capital * 100.0) if start_capital > 0 else 0.0
        rows.append(
            {
                "month": str(month),
                "return_pct": round(ret, 2),
                "pnl": round(pnl, 2),
                "trades": int(len(grp)),
                "end_capital": round(capital, 2),
            }
        )
    return pd.DataFrame(rows)


def regime_labels_for_months(engine: ScalpRobustEngine) -> dict[str, str]:
    labels: dict[str, str] = {}
    month_starts = pd.date_range("2023-01-01", "2026-04-01", freq="MS", tz="UTC")
    candle_times = pd.Series([pd.Timestamp(candle.ts, unit="s", tz="UTC") for candle in engine.c15m])
    for month_start in month_starts:
        matches = candle_times[candle_times >= month_start]
        if matches.empty:
            continue
        idx = int(matches.index[0])
        labels[month_start.strftime("%Y-%m")] = engine._apply_regime_switch_for_idx(idx)
    return labels


def main() -> None:
    thresholds = {
        "high_growth_score_min": 3,
        "compression_growth_score_min": 4,
        "compression_growth_adx_max": 22.0,
        "compression_growth_atr_ratio_min": 0.75,
        "compression_growth_atr_ratio_max": 1.05,
        "compression_growth_momentum_min": -0.02,
        "compression_growth_ema_gap_min": -0.005,
        "normal_score_min": 3,
        "normal_atr_ratio_min": 0.8,
        "normal_momentum_max": 0.04,
        "normal_ema_gap_max": 0.005,
        "normal_adx_min": 18.0,
    }
    variants = {
        "baseline": {},
        "hg_params": {
            "atr_loose_multiplier": 2.0,
            "pullback_window": 40,
        },
        "normal_params": {
            "atr_loose_multiplier": 2.8,
            "pullback_window": 30,
        },
        "switched": {
            "enable_regime_switching": True,
            "regime_switcher_thresholds": thresholds,
            "regime_switcher_hg_overrides": {
                "atr_loose_multiplier": 2.0,
                "pullback_window": 40,
            },
            "regime_switcher_normal_overrides": {
                "atr_loose_multiplier": 2.8,
                "pullback_window": 30,
            },
            "regime_switcher_flat_overrides": {
                "atr_loose_multiplier": 2.7,
                "pullback_window": 40,
            },
        },
    }

    results: dict[str, Any] = {}
    monthly_frames: dict[str, pd.DataFrame] = {}
    switched_regimes: dict[str, str] = {}

    for name, overrides in variants.items():
        config = make_strategy_config(overrides)
        engine, trades = run_engine(config)
        results[name] = {
            "full_period": summarize_period(trades, START_15M, END),
            "period_2025_2026": summarize_period(trades, SPLIT_START, END),
        }
        monthly_frames[name] = build_monthly_returns(trades)
        if name == "switched":
            switched_regimes = regime_labels_for_months(engine)

    monthly_comparison = monthly_frames["baseline"][["month", "return_pct"]].rename(
        columns={"return_pct": "baseline_return_pct"}
    )
    for name in ("hg_params", "normal_params", "switched"):
        monthly_comparison = monthly_comparison.merge(
            monthly_frames[name][["month", "return_pct"]].rename(columns={"return_pct": f"{name}_return_pct"}),
            on="month",
            how="left",
        )
    monthly_comparison["regime"] = monthly_comparison["month"].map(switched_regimes)

    print(
        json.dumps(
            {
                "results": results,
                "monthly_comparison": monthly_comparison.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
