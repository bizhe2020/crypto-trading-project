#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.okx_executor import ExecutorConfig
from strategy.scalp_robust_v2_core import ScalpRobustEngine, dataframe_to_candles


DEFAULT_DATA_ROOT = Path("/Users/laoji/projects/crypto-trading-project/deployment/data/okx/futures")
DEFAULT_DATA_15M = DEFAULT_DATA_ROOT / "BTC_USDT_USDT-15m-futures.feather"
DEFAULT_DATA_4H = DEFAULT_DATA_ROOT / "BTC_USDT_USDT-4h-futures.feather"
DEFAULT_OUTPUT_DIR = ROOT / "var" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scalp_robust_v2_core backtests for one or more configs.")
    parser.add_argument("--config", action="append", required=True, help="Path to a config JSON file. Repeatable.")
    parser.add_argument("--start-date", required=True, help="Backtest start date, e.g. 2025-01-01")
    parser.add_argument("--end-date", required=True, help="Backtest end date, e.g. 2026-04-18")
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stdout", action="store_true", help="Print combined JSON summary to stdout.")
    return parser.parse_args()


def parse_end_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.nanosecond == 0:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return ts


def load_dataframe(path: Path, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
    df = pd.read_feather(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    elif "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        raise ValueError(f"Unsupported dataframe format for {path}")

    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    return df.sort_values("date").reset_index(drop=True)


def summarize_trade_bucket(trades: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    base = {
        "trades": 0,
        "pnl": 0.0,
        "return_pct": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_pnl": 0.0,
        "avg_hold_hours": 0.0,
    }
    if trades.empty:
        return base

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    gross_profit = float(wins["pnl"].sum())
    gross_loss = abs(float(losses["pnl"].sum()))
    pnl = float(trades["pnl"].sum())

    base.update(
        {
            "trades": int(len(trades)),
            "pnl": round(pnl, 2),
            "return_pct": round((pnl / initial_capital) * 100.0, 2) if initial_capital > 0 else 0.0,
            "win_rate": round(float((trades["pnl"] > 0).mean() * 100.0), 2),
            "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else 0.0,
            "avg_pnl": round(float(trades["pnl"].mean()), 2),
            "avg_hold_hours": round(float(trades["hold_hours"].mean()), 2),
        }
    )
    return base


def load_config_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_report_from_payload(
    config_payload: dict[str, Any],
    config_label: str,
    data_15m_path: Path,
    data_4h_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    config = ExecutorConfig.from_dict(config_payload).to_scalp_strategy_config()

    df15 = load_dataframe(data_15m_path, start=start, end=end)
    df4 = load_dataframe(data_4h_path, end=end)

    engine = ScalpRobustEngine.from_candles(
        dataframe_to_candles(df4),
        dataframe_to_candles(df15),
        config,
    )
    metrics = engine.run_backtest(start_date=start.strftime("%Y-%m-%d"))

    trades = pd.DataFrame([trade.__dict__ for trade in engine.trades])
    if not trades.empty:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
        trades["hold_hours"] = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 3600.0
    else:
        trades["hold_hours"] = pd.Series(dtype=float)

    by_direction = {}
    for direction in ("BULL", "BEAR"):
        by_direction[direction] = summarize_trade_bucket(
            trades[trades["direction"] == direction].copy() if not trades.empty else trades,
            initial_capital=float(config.initial_capital),
        )

    return {
        "config": config_label,
        "date_range": {
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
        },
        "data_points": {
            "candles_15m": int(len(df15)),
            "candles_4h": int(len(df4)),
        },
        "overall": {
            "initial_capital": round(float(metrics.get("initial_capital", 0.0)), 2),
            "final_capital": round(float(metrics.get("final_capital", 0.0)), 2),
            "net_pnl": round(float(metrics.get("final_capital", 0.0) - metrics.get("initial_capital", 0.0)), 2),
            "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 2),
            "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
            "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0)), 2),
            "risk_adjusted_return": round(float(metrics.get("risk_adjusted_return", 0.0)), 3),
            "total_trades": int(metrics.get("total_trades", 0)),
            "winning_trades": int(metrics.get("winning_trades", 0)),
            "losing_trades": int(metrics.get("losing_trades", 0)),
            "win_rate": round(float(metrics.get("win_rate", 0.0)), 2),
            "profit_factor": round(float(metrics.get("profit_factor", 0.0)), 3),
            "win_loss_ratio": round(float(metrics.get("wl_ratio", 0.0)), 3),
            "target_hit_rate": round(float(metrics.get("target_hit_rate", 0.0)), 2),
            "gross_pnl_before_fees": round(float(metrics.get("gross_pnl_before_fees", 0.0)), 2),
            "total_fees_paid": round(float(metrics.get("total_fees_paid", 0.0)), 2),
            "total_slippage_cost": round(float(metrics.get("total_slippage_cost", 0.0)), 2),
            "exit_reasons": metrics.get("exit_reasons", {}),
        },
        "by_direction": by_direction,
    }


def build_report(
    config_path: Path,
    data_15m_path: Path,
    data_4h_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    payload = load_config_payload(config_path)
    return build_report_from_payload(
        config_payload=payload,
        config_label=str(config_path.resolve()),
        data_15m_path=data_15m_path,
        data_4h_path=data_4h_path,
        start=start,
        end=end,
    )


def output_path_for(output_dir: Path, config_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    return output_dir / f"backtest_{config_path.stem}_{start.strftime('%Y-%m-%d')}_to_{end.strftime('%Y-%m-%d')}.json"


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start_date, tz="UTC")
    end = parse_end_timestamp(args.end_date)
    data_15m_path = Path(args.data_15m)
    data_4h_path = Path(args.data_4h)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []
    for config_arg in args.config:
        config_path = Path(config_arg)
        report = build_report(config_path, data_15m_path, data_4h_path, start, end)
        reports.append(report)
        output_path = output_path_for(output_dir, config_path, start, end)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(output_path)

    if args.stdout:
        print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
