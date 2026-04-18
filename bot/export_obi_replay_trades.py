from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.market_data import OhlcvRepository
from strategy.scalp_robust_v2_core import (
    ActionType,
    ScalpRobustEngine,
    StrategyConfig,
    StrategyAction,
    Trade,
    dataframe_to_candles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export backtest trades into standard OBI replay input format")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default="data/okx/futures")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--orderbook-inst-id", default="BTC-USDT-SWAP")
    return parser.parse_args()


class BacktestExportConfig:
    def __init__(
        self,
        *,
        symbol: str,
        timeframe: str,
        informative_timeframe: str,
        data_root: str,
        strategy: StrategyConfig,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.informative_timeframe = informative_timeframe
        self.data_root = data_root
        self.strategy = strategy


def _supported_strategy_config(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(StrategyConfig)}
    return {key: value for key, value in payload.items() if key in allowed}


def _load_backtest_config(config_path: str | Path, data_root: str) -> BacktestExportConfig:
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return BacktestExportConfig(
        symbol=str(raw.get("symbol", "BTC/USDT:USDT")),
        timeframe=str(raw.get("timeframe", "15m")),
        informative_timeframe=str(raw.get("informative_timeframe", "4h")),
        data_root=data_root,
        strategy=StrategyConfig(**_supported_strategy_config(raw)),
    )


def _load_engine(config: BacktestExportConfig) -> ScalpRobustEngine:
    repo = OhlcvRepository(config.data_root)
    bundle = repo.load_pair(
        config.symbol,
        client=None,
        timeframe=config.timeframe,
        informative_timeframe=config.informative_timeframe,
    )
    primary_candles = dataframe_to_candles(bundle.primary_candles)
    informative_candles = dataframe_to_candles(bundle.informative_candles)
    return ScalpRobustEngine.from_candles(
        informative_candles,
        primary_candles,
        config.strategy,
    )


def _parse_utc_timestamp(value: str) -> int:
    return int(datetime.strptime(value, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _iso_utc(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).isoformat()


def _run_backtest_with_actions(engine: ScalpRobustEngine, start_date: str) -> tuple[list[StrategyAction], dict[str, Any]]:
    start_dt = datetime.fromisoformat(start_date)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    else:
        start_dt = start_dt.astimezone(timezone.utc)
    start_ts = start_dt.timestamp()
    start_idx = next((i for i, candle in enumerate(engine.c15m) if candle.ts >= start_ts), 0)

    actions = engine.evaluate_range(start_idx + 100, len(engine.c15m) - 1)
    if engine.position:
        actions.append(engine.close_position(len(engine.c15m) - 1, "end_of_data"))
    return actions, engine.compute_metrics()


def _build_standard_trades(actions: list[StrategyAction], trades: list[Trade], orderbook_inst_id: str) -> list[dict[str, Any]]:
    standard_trades: list[dict[str, Any]] = []
    open_trade: dict[str, Any] | None = None
    closed_trade_idx = 0

    for action in actions:
        if action.type in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}:
            open_trade = {
                "trade_id": len(standard_trades) + 1,
                "direction": action.direction,
                "entry_time": action.timestamp,
                "entry_time_utc": _iso_utc(action.timestamp),
                "entry_timestamp_ms": _parse_utc_timestamp(action.timestamp),
                "entry_price": action.entry_price,
                "signal_entry_price": action.metadata.get("signal_entry_price") if action.metadata else None,
                "initial_stop_price": action.stop_price,
                "current_stop_price": action.stop_price,
                "target_price": action.target_price,
                "entry_context": {
                    "entry_regime_score": action.metadata.get("entry_regime_score") if action.metadata else None,
                    "target_rr": action.metadata.get("target_rr") if action.metadata else None,
                    "trail_style": action.metadata.get("trail_style") if action.metadata else None,
                    "max_hold_bars": action.metadata.get("max_hold_bars") if action.metadata else None,
                    "risk_per_trade": action.metadata.get("risk_per_trade") if action.metadata else None,
                    "risk_regime": action.metadata.get("risk_regime") if action.metadata else None,
                    "position_size_pct": action.metadata.get("position_size_pct") if action.metadata else None,
                    "notional": action.metadata.get("notional") if action.metadata else None,
                    "max_notional": action.metadata.get("max_notional") if action.metadata else None,
                    "risk_based_notional": action.metadata.get("risk_based_notional") if action.metadata else None,
                },
                "stop_updates": [],
                "obi_replay": {
                    "status": "pending",
                    "orderbook_inst_id": orderbook_inst_id,
                    "start_timestamp_ms": _parse_utc_timestamp(action.timestamp),
                    "end_timestamp_ms": None,
                },
            }
            continue

        if action.type == ActionType.UPDATE_STOP and open_trade is not None:
            open_trade["current_stop_price"] = action.stop_price
            open_trade["target_price"] = action.target_price or open_trade["target_price"]
            open_trade["stop_updates"].append(
                {
                    "timestamp": action.timestamp,
                    "timestamp_ms": _parse_utc_timestamp(action.timestamp),
                    "stop_price": action.stop_price,
                    "target_price": action.target_price,
                    "reason": action.reason,
                    "metadata": action.metadata or {},
                }
            )
            continue

        if action.type == ActionType.CLOSE_POSITION and open_trade is not None:
            realized_trade = trades[closed_trade_idx]
            closed_trade_idx += 1
            standard_trades.append(
                {
                    **open_trade,
                    "exit_time": realized_trade.exit_time,
                    "exit_time_utc": _iso_utc(realized_trade.exit_time),
                    "exit_timestamp_ms": _parse_utc_timestamp(realized_trade.exit_time),
                    "original_exit_reason": realized_trade.exit_reason,
                    "original_signal_exit_price": realized_trade.signal_exit_price,
                    "original_exit_price": realized_trade.exit_price,
                    "capital_at_entry": realized_trade.capital_at_entry,
                    "gross_pnl": realized_trade.gross_pnl,
                    "fees": realized_trade.fees,
                    "slippage_cost": realized_trade.slippage_cost,
                    "pnl": realized_trade.pnl,
                    "pnl_pct": realized_trade.pnl_pct,
                    "rr_ratio": realized_trade.rr_ratio,
                    "final_stop_price_at_exit": open_trade["current_stop_price"],
                    "obi_replay": {
                        **open_trade["obi_replay"],
                        "end_timestamp_ms": _parse_utc_timestamp(realized_trade.exit_time),
                        "baseline_exit_reason": realized_trade.exit_reason,
                        "baseline_exit_price": realized_trade.exit_price,
                    },
                }
            )
            open_trade = None

    if closed_trade_idx != len(trades):
        raise RuntimeError("Not all realized trades were paired")
    return standard_trades


def _write_csv(path: Path, trades: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trade_id", "direction", "entry_time", "exit_time", "entry_price", "signal_entry_price",
        "initial_stop_price", "final_stop_price_at_exit", "target_price", "entry_regime_score",
        "target_rr", "trail_style", "risk_regime", "risk_per_trade", "notional", "original_exit_reason",
        "original_signal_exit_price", "original_exit_price", "gross_pnl", "fees", "slippage_cost", "pnl",
        "pnl_pct", "rr_ratio", "obi_start_timestamp_ms", "obi_end_timestamp_ms", "stop_update_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            context = trade.get("entry_context") or {}
            replay = trade.get("obi_replay") or {}
            writer.writerow(
                {
                    "trade_id": trade["trade_id"],
                    "direction": trade["direction"],
                    "entry_time": trade["entry_time"],
                    "exit_time": trade["exit_time"],
                    "entry_price": trade["entry_price"],
                    "signal_entry_price": trade.get("signal_entry_price"),
                    "initial_stop_price": trade["initial_stop_price"],
                    "final_stop_price_at_exit": trade.get("final_stop_price_at_exit"),
                    "target_price": trade.get("target_price"),
                    "entry_regime_score": context.get("entry_regime_score"),
                    "target_rr": context.get("target_rr"),
                    "trail_style": context.get("trail_style"),
                    "risk_regime": context.get("risk_regime"),
                    "risk_per_trade": context.get("risk_per_trade"),
                    "notional": context.get("notional"),
                    "original_exit_reason": trade.get("original_exit_reason"),
                    "original_signal_exit_price": trade.get("original_signal_exit_price"),
                    "original_exit_price": trade.get("original_exit_price"),
                    "gross_pnl": trade.get("gross_pnl"),
                    "fees": trade.get("fees"),
                    "slippage_cost": trade.get("slippage_cost"),
                    "pnl": trade.get("pnl"),
                    "pnl_pct": trade.get("pnl_pct"),
                    "rr_ratio": trade.get("rr_ratio"),
                    "obi_start_timestamp_ms": replay.get("start_timestamp_ms"),
                    "obi_end_timestamp_ms": replay.get("end_timestamp_ms"),
                    "stop_update_count": len(trade.get("stop_updates") or []),
                }
            )


def main() -> None:
    args = parse_args()
    config = _load_backtest_config(args.config, args.data_root)
    engine = _load_engine(config)
    actions, metrics = _run_backtest_with_actions(engine, args.start_date)
    standard_trades = _build_standard_trades(actions, engine.trades, args.orderbook_inst_id)

    payload = {
        "metadata": {
            "strategy": "scalp_robust_v2",
            "config_path": str(Path(args.config).resolve()),
            "data_root": str(Path(args.data_root).resolve()),
            "symbol": config.symbol,
            "timeframe": config.timeframe,
            "informative_timeframe": config.informative_timeframe,
            "start_date": args.start_date,
            "orderbook_inst_id": args.orderbook_inst_id,
            "trade_count": len(standard_trades),
            "metrics_summary": {
                **metrics,
                "parameters": {
                    **asdict(config.strategy),
                    "data_root": config.data_root,
                },
            },
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "trades": standard_trades,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(Path(args.output_csv), standard_trades)
    print(json.dumps({"trade_count": len(standard_trades), **{k: metrics[k] for k in ("total_return_pct", "profit_factor", "win_rate")}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
