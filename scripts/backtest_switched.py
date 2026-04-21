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
DATA_15M = Path(
    "/Users/laoji/projects/crypto-trading-project/deployment/data/okx/futures/BTC_USDT_USDT-15m-futures.feather"
)
DATA_4H = Path(
    "/Users/laoji/projects/crypto-trading-project/deployment/data/okx/futures/BTC_USDT_USDT-4h-futures.feather"
)
FULL_START = pd.Timestamp("2023-01-01", tz="UTC")
SPLIT_START = pd.Timestamp("2025-01-01", tz="UTC")
END = pd.Timestamp("2026-04-18 23:59:59", tz="UTC")
INITIAL_CAPITAL = 1000.0


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_feather(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def load_base_payload() -> dict[str, Any]:
    payload = json.loads(CONFIG_PATH.read_text())
    payload.update(
        {
            "initial_capital": INITIAL_CAPITAL,
            "atr_activation_rr": 2.06,
            "atr_loose_multiplier": 2.7,
            "atr_normal_multiplier": 2.25,
            "atr_tight_multiplier": 1.8,
            "atr_regime_filter": "tight_style_off",
            "short_strong_rr_ratio": 5.0,
            "bear_strong_short_risk_per_trade": 0.00001,
            "enable_atr_trailing": True,
            "disable_fixed_target_exit": False,
            "enable_target_rr_cap": True,
            "pullback_window": 40,
            "sl_buffer_pct": 1.25,
            "bull_strong_long_risk_per_trade": 0.06,
        }
    )
    return payload


def make_strategy_config(overrides: dict[str, Any] | None = None):
    payload = load_base_payload()
    if overrides:
        payload.update(overrides)
    return ExecutorConfig.from_dict(payload).to_scalp_strategy_config()


def load_candles() -> tuple[list, list]:
    df15 = load_dataframe(DATA_15M)
    df4h = load_dataframe(DATA_4H)
    df15 = df15[(df15["date"] >= FULL_START) & (df15["date"] <= END)].reset_index(drop=True)
    df4h = df4h[df4h["date"] <= END].reset_index(drop=True)
    return dataframe_to_candles(df4h), dataframe_to_candles(df15)


def run_variant(
    c4h_candles: list,
    c15m_candles: list,
    overrides: dict[str, Any] | None = None,
) -> tuple[ScalpRobustEngine, dict[str, Any], pd.DataFrame]:
    strategy_config = make_strategy_config(overrides)
    engine = ScalpRobustEngine.from_candles(c4h_candles, c15m_candles, strategy_config)
    metrics = engine.run_backtest(start_date="2023-01-01")
    trades = pd.DataFrame([trade.__dict__ for trade in engine.trades])
    if trades.empty:
        return engine, metrics, trades
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
    trades["exit_month"] = trades["exit_time"].dt.tz_localize(None).dt.to_period("M")
    trades = trades.sort_values("exit_time").reset_index(drop=True)
    return engine, metrics, trades


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
            max_drawdown_pct = max(max_drawdown_pct, (peak - capital) / peak * 100.0)

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
        "win_rate": round(float((during["pnl"] > 0).mean() * 100.0) if len(during) else 0.0, 2),
        "start_capital": round(start_capital, 2),
        "end_capital": round(capital, 2),
    }


def build_monthly_returns(trades: pd.DataFrame) -> pd.DataFrame:
    capital = INITIAL_CAPITAL
    rows: list[dict[str, Any]] = []
    month_range = pd.period_range("2023-01", "2026-04", freq="M")
    for month in month_range:
        grp = trades[trades["exit_month"] == month] if not trades.empty else pd.DataFrame()
        start_capital = capital
        pnl = float(grp["pnl"].sum()) if not grp.empty else 0.0
        capital += pnl
        return_pct = (pnl / start_capital * 100.0) if start_capital > 0 else 0.0
        rows.append(
            {
                "month": str(month),
                "return_pct": round(return_pct, 2),
                "pnl": round(pnl, 2),
                "trades": int(len(grp)),
                "end_capital": round(capital, 2),
            }
        )
    return pd.DataFrame(rows)


def switched_regime_labels(engine: ScalpRobustEngine) -> dict[str, str]:
    labels: dict[str, str] = {}
    candle_times = pd.Series([pd.Timestamp(candle.ts, unit="s", tz="UTC") for candle in engine.c15m])
    for month_start in pd.date_range("2023-01-01", "2026-04-01", freq="MS", tz="UTC"):
        matches = candle_times[candle_times >= month_start]
        if matches.empty:
            continue
        idx = int(matches.index[0])
        labels[month_start.strftime("%Y-%m")] = engine._apply_regime_switch_for_idx(idx)
    return labels


def main() -> None:
    c4h_candles, c15m_candles = load_candles()
    switched_thresholds = {
        "high_growth_score_min": 3,
        "compression_growth_score_min": 4,
        "flat_score_min": 3,
        "normal_score_min": 1,
        "strong_high_growth_adx_min": 35.0,
        "strong_high_growth_momentum_min": 0.04,
        "high_growth_adx_min": 20.0,
        "high_growth_momentum_min": -0.01,
        "high_growth_ema_gap_min": 0.01,
        "compression_growth_adx_max": 18.5,
        "compression_growth_atr_ratio_min": 0.75,
        "compression_growth_atr_ratio_max": 0.95,
        "compression_growth_momentum_min": -0.02,
        "compression_growth_ema_gap_min": -0.002,
        "flat_adx_max": 20.0,
        "flat_momentum_abs_max": 0.03,
        "flat_momentum_min": -0.005,
        "flat_ema_gap_min": 0.0,
        "flat_atr_ratio_max": 0.9,
        "normal_adx_min": 20.0,
        "normal_momentum_max": 0.0,
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
            "regime_switcher_thresholds": switched_thresholds,
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
    month_regimes: dict[str, str] = {}

    for name, overrides in variants.items():
        engine, full_metrics, trades = run_variant(c4h_candles, c15m_candles, overrides)
        results[name] = {
            "full_period": {
                "total_return_pct": round(full_metrics.get("total_return_pct", 0.0), 2),
                "sharpe_ratio": round(full_metrics.get("sharpe_ratio", 0.0), 3),
                "max_drawdown_pct": round(full_metrics.get("max_drawdown_pct", 0.0), 2),
                "total_trades": int(full_metrics.get("total_trades", 0)),
                "win_rate": round(full_metrics.get("win_rate", 0.0), 2),
            },
            "period_2025_2026": summarize_period(trades, SPLIT_START, END),
        }
        monthly_frames[name] = build_monthly_returns(trades)
        if name == "switched":
            month_regimes = switched_regime_labels(engine)

    monthly = monthly_frames["baseline"][["month", "return_pct"]].rename(columns={"return_pct": "baseline_return_pct"})
    for name in ("hg_params", "normal_params", "switched"):
        monthly = monthly.merge(
            monthly_frames[name][["month", "return_pct"]].rename(columns={"return_pct": f"{name}_return_pct"}),
            on="month",
            how="left",
        )
    monthly["regime"] = monthly["month"].map(month_regimes)

    print(
        json.dumps(
            {
                "results": results,
                "monthly_comparison": monthly.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
