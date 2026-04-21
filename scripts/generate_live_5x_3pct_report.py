#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.okx_executor import ExecutorConfig
from strategy.scalp_robust_v2_core import (
    ScalpRobustEngine,
    align_timeframes,
    build_precomputed_state,
    dataframe_to_candles,
)


COMMON_OVERRIDES = {
    "atr_regime_filter": "tight_style_off",
    "atr_activation_rr": 2.0,
    "atr_loose_multiplier": 2.7,
    "atr_normal_multiplier": 2.25,
    "atr_tight_multiplier": 1.8,
    "disable_fixed_target_exit": False,
    "enable_atr_trailing": True,
}

BEAR_STRONG_OPTIMIZED_OVERRIDES = {
    "allow_bear_strong_short": True,
    "bear_strong_short_pullback_window": 10,
    "bear_strong_short_sl_buffer_pct": 0.5,
    "bear_strong_short_retrace_min_ob_fill_pct": 0.8,
    "bear_strong_short_entry_min_ob_fill_pct": 0.55,
    "bear_strong_short_rr_ratio_override": 2.5,
    "bear_strong_short_trail_style_override": "tight",
    "bear_strong_short_max_hold_bars": 96,
    "bear_strong_short_atr_activation_rr": 1.5,
    "bear_strong_short_atr_loose_multiplier": 1.5,
}

BEAR_STRONG_RESET_OVERRIDES = {
    "allow_bear_strong_short": True,
    "bear_strong_short_pullback_window": None,
    "bear_strong_short_sl_buffer_pct": None,
    "bear_strong_short_retrace_min_ob_fill_pct": None,
    "bear_strong_short_entry_min_ob_fill_pct": None,
    "bear_strong_short_rr_ratio_override": None,
    "bear_strong_short_trail_style_override": None,
    "bear_strong_short_max_hold_bars": None,
    "bear_strong_short_atr_activation_rr": None,
    "bear_strong_short_atr_loose_multiplier": None,
    "bear_strong_short_atr_normal_multiplier": None,
    "bear_strong_short_atr_tight_multiplier": None,
}

REGIME_ORDER = {
    "bull_strong": 0,
    "bull_weak": 1,
    "bear_strong": 2,
    "bear_weak": 3,
    "unknown": 4,
}

TRAIL_STYLE_ORDER = {
    "B": 0,
    "M": 1,
    "S": 2,
    "ATR": 3,
    "unknown": 4,
}

DIRECTION_LABEL = {
    "BULL": "LONG",
    "BEAR": "SHORT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate live 5x 3pct attribution markdown report")
    parser.add_argument("--config", default="config/config.live.5x-3pct.json")
    parser.add_argument(
        "--data-15m",
        default="/Users/laoji/projects/crypto-trading-project/deployment/data/okx/futures/BTC_USDT_USDT-15m-futures.feather",
    )
    parser.add_argument(
        "--data-4h",
        default="/Users/laoji/projects/crypto-trading-project/deployment/data/okx/futures/BTC_USDT_USDT-4h-futures.feather",
    )
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-04-18")
    parser.add_argument(
        "--output-md",
        default="var/reports/live_5x_3pct_optimized_attribution_2023-01-01_to_2026-04-18.md",
    )
    parser.add_argument(
        "--output-json",
        default="var/reports/live_5x_3pct_optimized_attribution_2023-01-01_to_2026-04-18.json",
    )
    return parser.parse_args()


def load_filtered_feather(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    df = pd.read_feather(path)
    if "date" in df.columns:
        ts = pd.to_datetime(df["date"], utc=True)
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        raise ValueError(f"Unsupported dataframe columns for {path}")

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    return df[(ts >= start_ts) & (ts < end_ts)].reset_index(drop=True)


def load_head_payload(config_path: Path) -> dict[str, Any]:
    rel_path = config_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    result = subprocess.run(
        ["git", "show", f"HEAD:{rel_path}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def normalize_trail_style(trail_style: str | None) -> str:
    if trail_style is None:
        return "unknown"
    normalized = trail_style.strip()
    if not normalized:
        return "unknown"
    upper = normalized.upper()
    if upper in {"B", "M", "S", "ATR"}:
        return upper
    mapping = {
        "loose": "B",
        "normal": "M",
        "tight": "S",
        "atr": "ATR",
    }
    return mapping.get(normalized.lower(), normalized)


def load_engine_inputs(
    data_15m_path: Path,
    data_4h_path: Path,
    start_date: str,
    end_date: str,
) -> tuple[list[Any], list[Any], list[int], Any]:
    c15m_df = load_filtered_feather(data_15m_path, start_date, end_date)
    c4h_df = load_filtered_feather(data_4h_path, start_date, end_date)
    c15m = dataframe_to_candles(c15m_df)
    c4h = dataframe_to_candles(c4h_df)
    mapping = align_timeframes(c4h, c15m)
    precomputed = build_precomputed_state(c4h, c15m)
    return c4h, c15m, mapping, precomputed


def run_case(
    name: str,
    payload: dict[str, Any],
    c4h: list[Any],
    c15m: list[Any],
    mapping: list[int],
    precomputed: Any,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    strategy_config = ExecutorConfig.from_dict(payload).to_scalp_strategy_config()
    start_dt = pd.Timestamp(start_date, tz="UTC")
    start_idx = next((i for i, candle in enumerate(c15m) if candle.ts >= start_dt.timestamp()), 0)

    engine = ScalpRobustEngine(c4h, c15m, mapping, precomputed, strategy_config)
    engine.evaluate_range(start_idx + 100, len(c15m) - 1)
    if engine.position:
        engine.close_position(
            len(c15m) - 1,
            "end_of_data",
            entry_risk_regime=engine.position.entry_risk_regime,
            trail_style=engine.position.trail_style,
        )

    metrics = engine.compute_metrics()
    trades = pd.DataFrame([trade.__dict__ for trade in engine.trades])
    if not trades.empty:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
        trades["hold_hours"] = (
            trades["exit_time"] - trades["entry_time"]
        ).dt.total_seconds() / 3600.0
        trades["direction_label"] = trades["direction"].map(DIRECTION_LABEL).fillna(trades["direction"])
        trades["trail_style_label"] = trades["trail_style"].map(normalize_trail_style)
        trades["exit_month"] = trades["exit_time"].dt.strftime("%Y-%m")
        trades["exit_year"] = trades["exit_time"].dt.year

    initial_capital = float(metrics["initial_capital"])
    return {
        "name": name,
        "payload": payload,
        "metrics": metrics,
        "trades": trades,
        "overall": summarize_overall(metrics),
        "direction_breakdown": summarize_by_direction(trades, initial_capital),
        "regime_direction_breakdown": summarize_by_regime_direction(trades, initial_capital),
        "trail_style_distribution": summarize_by_trail_style(trades, initial_capital),
        "exit_reason_distribution": summarize_by_exit_reason(trades, initial_capital),
        "monthly_breakdown": summarize_monthly(trades, initial_capital, start_date, end_date),
        "annual_returns": summarize_annual_returns(trades, initial_capital, start_date, end_date),
        "bear_strong_short": summarize_bucket(
            trades,
            initial_capital,
            entry_risk_regime="bear_strong",
            direction="BEAR",
        ),
    }


def summarize_overall(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 2),
        "sharpe": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
        "max_drawdown": round(float(metrics.get("max_drawdown_pct", 0.0)), 2),
        "total_trades": int(metrics.get("total_trades", 0)),
        "win_rate": round(float(metrics.get("win_rate", 0.0)), 2),
        "profit_factor": round(float(metrics.get("profit_factor", 0.0)), 3),
        "final_capital": round(float(metrics.get("final_capital", 0.0)), 2),
    }


def summarize_trade_frame(trades: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    if trades.empty:
        return {
            "trades": 0,
            "pnl": 0.0,
            "total_return_pct": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_pnl": 0.0,
            "avg_rr": 0.0,
        }

    gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    gross_loss = abs(float(trades.loc[trades["pnl"] <= 0, "pnl"].sum()))
    total_pnl = float(trades["pnl"].sum())
    return {
        "trades": int(len(trades)),
        "pnl": round(total_pnl, 2),
        "total_return_pct": round(total_pnl / initial_capital * 100.0, 2) if initial_capital > 0 else 0.0,
        "win_rate": round(float((trades["pnl"] > 0).mean() * 100.0), 2),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else 0.0,
        "avg_pnl": round(float(trades["pnl"].mean()), 2),
        "avg_rr": round(float(trades["rr_ratio"].mean()), 3),
    }


def summarize_bucket(
    trades: pd.DataFrame,
    initial_capital: float,
    *,
    entry_risk_regime: str,
    direction: str,
) -> dict[str, Any]:
    if trades.empty:
        base = summarize_trade_frame(trades, initial_capital)
        base.update(
            {
                "stop_loss_pct": 0.0,
                "target_pct": 0.0,
                "avg_hold_hours": 0.0,
                "median_hold_hours": 0.0,
            }
        )
        return base

    bucket = trades[
        (trades["entry_risk_regime"] == entry_risk_regime)
        & (trades["direction"] == direction)
    ].copy()
    base = summarize_trade_frame(bucket, initial_capital)
    if bucket.empty:
        base.update(
            {
                "stop_loss_pct": 0.0,
                "target_pct": 0.0,
                "avg_hold_hours": 0.0,
                "median_hold_hours": 0.0,
            }
        )
        return base

    base.update(
        {
            "stop_loss_pct": round(float((bucket["exit_reason"] == "stop_loss").mean() * 100.0), 2),
            "target_pct": round(
                float(bucket["exit_reason"].astype(str).str.contains("target").mean() * 100.0),
                2,
            ),
            "avg_hold_hours": round(float(bucket["hold_hours"].mean()), 2),
            "median_hold_hours": round(float(bucket["hold_hours"].median()), 2),
        }
    )
    return base


def summarize_by_direction(trades: pd.DataFrame, initial_capital: float) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    rows: list[dict[str, Any]] = []
    for direction in ("BULL", "BEAR"):
        bucket = trades[trades["direction"] == direction]
        stats = summarize_trade_frame(bucket, initial_capital)
        stats["direction"] = DIRECTION_LABEL.get(direction, direction)
        rows.append(stats)
    return rows


def summarize_by_regime_direction(trades: pd.DataFrame, initial_capital: float) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = trades.groupby(["entry_risk_regime", "direction"], dropna=False)
    for (regime, direction), bucket in grouped:
        stats = summarize_trade_frame(bucket, initial_capital)
        stats["entry_risk_regime"] = regime or "unknown"
        stats["direction"] = DIRECTION_LABEL.get(direction, direction)
        rows.append(stats)
    rows.sort(
        key=lambda row: (
            REGIME_ORDER.get(str(row["entry_risk_regime"]), len(REGIME_ORDER)),
            0 if row["direction"] == "LONG" else 1,
        )
    )
    return rows


def summarize_by_trail_style(trades: pd.DataFrame, initial_capital: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_trades = int(len(trades))
    for style in ("B", "M", "S"):
        bucket = trades[trades["trail_style_label"] == style] if not trades.empty else trades
        stats = summarize_trade_frame(bucket, initial_capital)
        stats["trail_style"] = style
        stats["share_pct"] = round((stats["trades"] / total_trades * 100.0), 2) if total_trades > 0 else 0.0
        rows.append(stats)
    rows.sort(key=lambda row: TRAIL_STYLE_ORDER.get(str(row["trail_style"]), len(TRAIL_STYLE_ORDER)))
    return rows


def summarize_by_exit_reason(trades: pd.DataFrame, initial_capital: float) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    rows: list[dict[str, Any]] = []
    for exit_reason, bucket in trades.groupby("exit_reason", dropna=False):
        stats = summarize_trade_frame(bucket, initial_capital)
        stats["exit_reason"] = str(exit_reason)
        stats["avg_hold_hours"] = round(float(bucket["hold_hours"].mean()), 2)
        rows.append(stats)
    rows.sort(key=lambda row: (-row["trades"], row["exit_reason"]))
    return rows


def summarize_monthly(
    trades: pd.DataFrame,
    initial_capital: float,
    start_date: str,
    end_date: str | None,
) -> list[dict[str, Any]]:
    if end_date is None:
        raise ValueError("end_date is required for monthly summaries")

    months = pd.period_range(start=start_date, end=end_date, freq="M")
    if trades.empty:
        running_capital = initial_capital
        return [
            {
                "month": str(month),
                "trades": 0,
                "pnl": 0.0,
                "return_pct": 0.0,
                "start_capital": round(running_capital, 2),
                "end_capital": round(running_capital, 2),
            }
            for month in months
        ]

    grouped = trades.groupby("exit_month", dropna=False).agg(
        trades=("pnl", "size"),
        pnl=("pnl", "sum"),
    )
    rows: list[dict[str, Any]] = []
    running_capital = initial_capital
    for month in months:
        key = str(month)
        pnl = float(grouped.loc[key, "pnl"]) if key in grouped.index else 0.0
        trade_count = int(grouped.loc[key, "trades"]) if key in grouped.index else 0
        month_start = running_capital
        month_end = running_capital + pnl
        rows.append(
            {
                "month": key,
                "trades": trade_count,
                "pnl": round(pnl, 2),
                "return_pct": round((pnl / month_start * 100.0), 2) if month_start > 0 else 0.0,
                "start_capital": round(month_start, 2),
                "end_capital": round(month_end, 2),
            }
        )
        running_capital = month_end
    return rows


def summarize_annual_returns(
    trades: pd.DataFrame,
    initial_capital: float,
    start_date: str,
    end_date: str | None,
) -> list[dict[str, Any]]:
    if end_date is None:
        raise ValueError("end_date is required for annual summaries")

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts_exclusive = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    years = list(range(start_ts.year, end_ts_exclusive.year + 1))
    grouped = (
        trades.groupby("exit_year", dropna=False).agg(pnl=("pnl", "sum"))
        if not trades.empty
        else pd.DataFrame(columns=["pnl"])
    )

    rows: list[dict[str, Any]] = []
    running_capital = initial_capital
    for year in years:
        year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        next_year_start = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        period_start = max(year_start, start_ts)
        period_end = min(next_year_start, end_ts_exclusive)
        if period_start >= period_end:
            continue

        pnl = float(grouped.loc[year, "pnl"]) if year in grouped.index else 0.0
        year_start_capital = running_capital
        year_end_capital = running_capital + pnl
        return_pct = (pnl / year_start_capital * 100.0) if year_start_capital > 0 else 0.0
        days = max((period_end - period_start).days, 1)
        is_partial = period_end < next_year_start
        if is_partial and year_start_capital > 0 and year_end_capital > 0:
            annualized_return_pct = ((year_end_capital / year_start_capital) ** (365.0 / days) - 1.0) * 100.0
        else:
            annualized_return_pct = return_pct
        rows.append(
            {
                "year": year,
                "period": "YTD" if is_partial else "Full",
                "days": days,
                "pnl": round(pnl, 2),
                "return_pct": round(return_pct, 2),
                "annualized_return_pct": round(annualized_return_pct, 2),
                "start_capital": round(year_start_capital, 2),
                "end_capital": round(year_end_capital, 2),
            }
        )
        running_capital = year_end_capital
    return rows


def diff_metric(next_value: float | int, base_value: float | int, digits: int = 2) -> float | int:
    delta = float(next_value) - float(base_value)
    return round(delta, digits)


def build_delta_summary(base_case: dict[str, Any], next_case: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall": {
            "total_return_pct": diff_metric(next_case["overall"]["total_return_pct"], base_case["overall"]["total_return_pct"]),
            "sharpe": diff_metric(next_case["overall"]["sharpe"], base_case["overall"]["sharpe"], 3),
            "max_drawdown": diff_metric(next_case["overall"]["max_drawdown"], base_case["overall"]["max_drawdown"]),
            "total_trades": int(next_case["overall"]["total_trades"]) - int(base_case["overall"]["total_trades"]),
            "win_rate": diff_metric(next_case["overall"]["win_rate"], base_case["overall"]["win_rate"]),
            "profit_factor": diff_metric(next_case["overall"]["profit_factor"], base_case["overall"]["profit_factor"], 3),
        },
        "bear_strong_short": {
            "trades": int(next_case["bear_strong_short"]["trades"]) - int(base_case["bear_strong_short"]["trades"]),
            "pnl": diff_metric(next_case["bear_strong_short"]["pnl"], base_case["bear_strong_short"]["pnl"]),
            "total_return_pct": diff_metric(
                next_case["bear_strong_short"]["total_return_pct"],
                base_case["bear_strong_short"]["total_return_pct"],
            ),
            "win_rate": diff_metric(next_case["bear_strong_short"]["win_rate"], base_case["bear_strong_short"]["win_rate"]),
            "profit_factor": diff_metric(
                next_case["bear_strong_short"]["profit_factor"],
                base_case["bear_strong_short"]["profit_factor"],
                3,
            ),
            "stop_loss_pct": diff_metric(
                next_case["bear_strong_short"]["stop_loss_pct"],
                base_case["bear_strong_short"]["stop_loss_pct"],
            ),
        },
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def format_signed(value: float, digits: int = 2, suffix: str = "") -> str:
    return f"{value:+.{digits}f}{suffix}"


def format_unsigned(value: float, digits: int = 2, suffix: str = "") -> str:
    return f"{value:.{digits}f}{suffix}"


def build_report(
    optimized_case: dict[str, Any],
    baseline_case: dict[str, Any],
    raw_head_case: dict[str, Any],
    start_date: str,
    end_date: str,
) -> str:
    delta_vs_baseline = build_delta_summary(baseline_case, optimized_case)
    delta_vs_raw_head = build_delta_summary(raw_head_case, optimized_case)

    overall_rows = []
    for case in (baseline_case, optimized_case):
        overall = case["overall"]
        overall_rows.append(
            [
                case["name"],
                format_unsigned(overall["total_return_pct"], 2, "%"),
                format_unsigned(overall["sharpe"], 3),
                format_unsigned(overall["max_drawdown"], 2, "%"),
                str(overall["total_trades"]),
                format_unsigned(overall["win_rate"], 2, "%"),
                format_unsigned(overall["profit_factor"], 3),
            ]
        )

    annual_rows = [
        [
            str(row["year"]),
            row["period"],
            format_unsigned(row["pnl"], 2),
            format_unsigned(row["return_pct"], 2, "%"),
            format_unsigned(row["annualized_return_pct"], 2, "%"),
            format_unsigned(row["start_capital"], 2),
            format_unsigned(row["end_capital"], 2),
        ]
        for row in optimized_case["annual_returns"]
    ]

    direction_rows = [
        [
            row["direction"],
            str(row["trades"]),
            format_unsigned(row["pnl"], 2),
            format_unsigned(row["total_return_pct"], 2, "%"),
            format_unsigned(row["win_rate"], 2, "%"),
            format_unsigned(row["profit_factor"], 3),
        ]
        for row in optimized_case["direction_breakdown"]
    ]

    regime_direction_rows = [
        [
            str(row["entry_risk_regime"]),
            row["direction"],
            str(row["trades"]),
            format_unsigned(row["pnl"], 2),
            format_unsigned(row["total_return_pct"], 2, "%"),
            format_unsigned(row["win_rate"], 2, "%"),
            format_unsigned(row["profit_factor"], 3),
            format_unsigned(row["avg_rr"], 3),
        ]
        for row in optimized_case["regime_direction_breakdown"]
    ]

    bear_short_rows = []
    for case in (baseline_case, optimized_case):
        row = case["bear_strong_short"]
        bear_short_rows.append(
            [
                case["name"],
                str(row["trades"]),
                format_unsigned(row["pnl"], 2),
                format_unsigned(row["total_return_pct"], 2, "%"),
                format_unsigned(row["win_rate"], 2, "%"),
                format_unsigned(row["profit_factor"], 3),
                format_unsigned(row["stop_loss_pct"], 2, "%"),
                format_unsigned(row["avg_hold_hours"], 2),
            ]
        )

    trail_rows = [
        [
            row["trail_style"],
            str(row["trades"]),
            format_unsigned(row["share_pct"], 2, "%"),
            format_unsigned(row["pnl"], 2),
            format_unsigned(row["total_return_pct"], 2, "%"),
            format_unsigned(row["win_rate"], 2, "%"),
            format_unsigned(row["profit_factor"], 3),
        ]
        for row in optimized_case["trail_style_distribution"]
    ]

    exit_rows = [
        [
            row["exit_reason"],
            str(row["trades"]),
            format_unsigned(row["pnl"], 2),
            format_unsigned(row["total_return_pct"], 2, "%"),
            format_unsigned(row["win_rate"], 2, "%"),
            format_unsigned(row["avg_hold_hours"], 2),
        ]
        for row in optimized_case["exit_reason_distribution"]
    ]

    monthly_breakdown = optimized_case["monthly_breakdown"]
    top_months = sorted(monthly_breakdown, key=lambda row: (row["pnl"], row["month"]), reverse=True)[:5]
    bottom_months = sorted(monthly_breakdown, key=lambda row: (row["pnl"], row["month"]))[:5]
    top_month_rows = [
        [
            row["month"],
            str(row["trades"]),
            format_unsigned(row["pnl"], 2),
            format_unsigned(row["return_pct"], 2, "%"),
            format_unsigned(row["start_capital"], 2),
            format_unsigned(row["end_capital"], 2),
        ]
        for row in top_months
    ]
    bottom_month_rows = [
        [
            row["month"],
            str(row["trades"]),
            format_unsigned(row["pnl"], 2),
            format_unsigned(row["return_pct"], 2, "%"),
            format_unsigned(row["start_capital"], 2),
            format_unsigned(row["end_capital"], 2),
        ]
        for row in bottom_months
    ]

    delta_pre_rows = [
        [
            "overall",
            format_signed(delta_vs_baseline["overall"]["total_return_pct"], 2, "pp"),
            format_signed(delta_vs_baseline["overall"]["sharpe"], 3),
            format_signed(delta_vs_baseline["overall"]["max_drawdown"], 2, "pp"),
            f"{delta_vs_baseline['overall']['total_trades']:+d}",
            format_signed(delta_vs_baseline["overall"]["win_rate"], 2, "pp"),
            format_signed(delta_vs_baseline["overall"]["profit_factor"], 3),
        ],
        [
            "bear_strong_short",
            format_signed(delta_vs_baseline["bear_strong_short"]["total_return_pct"], 2, "pp"),
            "-",
            "-",
            f"{delta_vs_baseline['bear_strong_short']['trades']:+d}",
            format_signed(delta_vs_baseline["bear_strong_short"]["win_rate"], 2, "pp"),
            format_signed(delta_vs_baseline["bear_strong_short"]["profit_factor"], 3),
        ],
    ]

    delta_baseline_rows = [
        [
            "optimized_full vs baseline_3247_rerun",
            format_signed(delta_vs_baseline["overall"]["total_return_pct"], 2, "pp"),
            format_signed(delta_vs_baseline["overall"]["sharpe"], 3),
            format_signed(delta_vs_baseline["overall"]["max_drawdown"], 2, "pp"),
            f"{delta_vs_baseline['overall']['total_trades']:+d}",
            format_signed(delta_vs_baseline["overall"]["win_rate"], 2, "pp"),
            format_signed(delta_vs_baseline["overall"]["profit_factor"], 3),
        ]
    ]

    raw_head_rows = [
        [
            "optimized_full vs raw_head_config",
            format_signed(delta_vs_raw_head["overall"]["total_return_pct"], 2, "pp"),
            format_signed(delta_vs_raw_head["overall"]["sharpe"], 3),
            format_signed(delta_vs_raw_head["overall"]["max_drawdown"], 2, "pp"),
            f"{delta_vs_raw_head['overall']['total_trades']:+d}",
            format_signed(delta_vs_raw_head["overall"]["win_rate"], 2, "pp"),
            format_signed(delta_vs_raw_head["overall"]["profit_factor"], 3),
        ]
    ]

    lines = [
        "# Live 5x 3pct Optimized Strategy Attribution Report",
        "",
        "## Scope",
        f"- Data: 15m=`BTC_USDT_USDT-15m-futures.feather`, 4h=`BTC_USDT_USDT-4h-futures.feather`",
        f"- Range: `{start_date}` to `{end_date}` inclusive, UTC filter",
        "- Cases:",
        "  - `baseline_3247_rerun`: current config plus ATR and fixed-target overrides, but bear_strong_short optimization reset",
        "  - `optimized_full`: current config plus ATR/fixed-target overrides and full bear_strong_short optimized parameters",
        "  - `raw_head_config`: `git HEAD` version of `config.live.5x-3pct.json`, kept only as extra reference",
        f"- Note: the historically referenced `3247%` baseline reruns to `{format_unsigned(baseline_case['overall']['total_return_pct'], 2, '%')}` under the current workspace code and the `{start_date}` to `{end_date}` data window.",
        "",
        "## Applied Optimized Overrides",
        "```json",
        json.dumps({**COMMON_OVERRIDES, **BEAR_STRONG_OPTIMIZED_OVERRIDES}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 1. Overall",
        markdown_table(
            ["Case", "Return", "Sharpe", "MaxDD", "Trades", "WinRate", "PF"],
            overall_rows,
        ),
        "",
        "## 2. Annual Returns",
        "2026 is YTD through 2026-04-18; `Annualized` is only annualized for partial-year rows.",
        markdown_table(
            ["Year", "Period", "PnL", "Return", "Annualized", "StartCap", "EndCap"],
            annual_rows,
        ),
        "",
        "## 3. Long vs Short Breakdown",
        markdown_table(
            ["Direction", "Trades", "PnL", "Return", "WinRate", "PF"],
            direction_rows,
        ),
        "",
        "## 4. Regime Breakdown by LONG / SHORT",
        markdown_table(
            ["Regime", "Direction", "Trades", "PnL", "Return", "WinRate", "PF", "AvgRR"],
            regime_direction_rows,
        ),
        "",
        "## 5. Bear_strong Short Before vs After",
        markdown_table(
            ["Case", "Trades", "PnL", "Return", "WinRate", "PF", "StopLoss", "AvgHoldH"],
            bear_short_rows,
        ),
        "",
        "Bear_strong short delta vs `baseline_3247_rerun`:",
        markdown_table(
            ["Bucket", "ReturnDelta", "SharpeDelta", "MaxDDDelta", "TradesDelta", "WinRateDelta", "PFDelta"],
            delta_pre_rows,
        ),
        "",
        f"- Bear_strong short pnl delta: `{format_signed(delta_vs_baseline['bear_strong_short']['pnl'], 2)}`",
        f"- Bear_strong short stop-loss delta: `{format_signed(delta_vs_baseline['bear_strong_short']['stop_loss_pct'], 2, 'pp')}`",
        "",
        "## 6. Trail Style Distribution (B / M / S)",
        markdown_table(
            ["Style", "Trades", "Share", "PnL", "Return", "WinRate", "PF"],
            trail_rows,
        ),
        "",
        "## 7. Exit Reason Distribution",
        markdown_table(
            ["ExitReason", "Trades", "PnL", "Return", "WinRate", "AvgHoldH"],
            exit_rows,
        ),
        "",
        "## 8. Monthly PnL Top 5",
        markdown_table(
            ["Month", "Trades", "PnL", "Return", "StartCap", "EndCap"],
            top_month_rows,
        ),
        "",
        "## 9. Monthly PnL Bottom 5",
        markdown_table(
            ["Month", "Trades", "PnL", "Return", "StartCap", "EndCap"],
            bottom_month_rows,
        ),
        "",
        "## 10. Comparison vs Original 3247 Configuration",
        markdown_table(
            ["Comparison", "ReturnDelta", "SharpeDelta", "MaxDDDelta", "TradesDelta", "WinRateDelta", "PFDelta"],
            delta_baseline_rows,
        ),
        "",
        "## Appendix. Extra Reference vs Raw Head Config",
        markdown_table(
            ["Comparison", "ReturnDelta", "SharpeDelta", "MaxDDDelta", "TradesDelta", "WinRateDelta", "PFDelta"],
            raw_head_rows,
        ),
    ]
    return "\n".join(lines) + "\n"


def build_serializable_payload(
    optimized_case: dict[str, Any],
    baseline_case: dict[str, Any],
    raw_head_case: dict[str, Any],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    return {
        "date_range": {
            "start": start_date,
            "end": end_date,
        },
        "applied_optimized_overrides": {
            **COMMON_OVERRIDES,
            **BEAR_STRONG_OPTIMIZED_OVERRIDES,
        },
        "cases": {
            "baseline_3247_rerun": {
                "overall": baseline_case["overall"],
                "annual_returns": baseline_case["annual_returns"],
                "direction_breakdown": baseline_case["direction_breakdown"],
                "regime_direction_breakdown": baseline_case["regime_direction_breakdown"],
                "bear_strong_short": baseline_case["bear_strong_short"],
            },
            "optimized_full": {
                "overall": optimized_case["overall"],
                "annual_returns": optimized_case["annual_returns"],
                "direction_breakdown": optimized_case["direction_breakdown"],
                "regime_direction_breakdown": optimized_case["regime_direction_breakdown"],
                "trail_style_distribution": optimized_case["trail_style_distribution"],
                "exit_reason_distribution": optimized_case["exit_reason_distribution"],
                "monthly_breakdown": optimized_case["monthly_breakdown"],
                "bear_strong_short": optimized_case["bear_strong_short"],
            },
            "raw_head_config_reference": {
                "overall": raw_head_case["overall"],
                "annual_returns": raw_head_case["annual_returns"],
                "direction_breakdown": raw_head_case["direction_breakdown"],
                "regime_direction_breakdown": raw_head_case["regime_direction_breakdown"],
                "bear_strong_short": raw_head_case["bear_strong_short"],
            },
        },
        "delta_vs_baseline_3247_rerun": build_delta_summary(baseline_case, optimized_case),
        "delta_vs_raw_head_config": build_delta_summary(raw_head_case, optimized_case),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    data_15m_path = Path(args.data_15m)
    data_4h_path = Path(args.data_4h)
    output_md_path = Path(args.output_md)
    output_json_path = Path(args.output_json)

    live_payload = json.loads(config_path.read_text())
    original_payload = load_head_payload(config_path)

    optimized_payload = dict(live_payload)
    optimized_payload.update(COMMON_OVERRIDES)
    optimized_payload.update(BEAR_STRONG_OPTIMIZED_OVERRIDES)

    pre_opt_payload = dict(live_payload)
    pre_opt_payload.update(COMMON_OVERRIDES)
    pre_opt_payload.update(BEAR_STRONG_RESET_OVERRIDES)

    c4h, c15m, mapping, precomputed = load_engine_inputs(
        data_15m_path,
        data_4h_path,
        args.start_date,
        args.end_date,
    )

    original_case = run_case(
        "raw_head_config",
        original_payload,
        c4h,
        c15m,
        mapping,
        precomputed,
        args.start_date,
        args.end_date,
    )
    pre_opt_case = run_case(
        "baseline_3247_rerun",
        pre_opt_payload,
        c4h,
        c15m,
        mapping,
        precomputed,
        args.start_date,
        args.end_date,
    )
    optimized_case = run_case(
        "optimized_full",
        optimized_payload,
        c4h,
        c15m,
        mapping,
        precomputed,
        args.start_date,
        args.end_date,
    )

    report = build_report(
        optimized_case=optimized_case,
        baseline_case=pre_opt_case,
        raw_head_case=original_case,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    serializable_payload = build_serializable_payload(
        optimized_case=optimized_case,
        baseline_case=pre_opt_case,
        raw_head_case=original_case,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(report)
    output_json_path.write_text(json.dumps(serializable_payload, ensure_ascii=False, indent=2))

    print(f"Markdown report written to: {output_md_path}")
    print(f"JSON summary written to: {output_json_path}")


if __name__ == "__main__":
    main()
