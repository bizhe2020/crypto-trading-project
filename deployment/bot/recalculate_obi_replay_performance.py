from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


STOPLIKE_REASONS = {
    "stop_loss",
    "initial_stop_loss",
    "trailing_stop_loss",
    "trailing_stop_profit",
}


@dataclass
class RepricedTrade:
    trade_id: int
    direction: str
    entry_time: str
    exit_time: str
    entry_price: float
    signal_entry_price: float
    signal_exit_price: float
    exit_price: float
    exit_reason: str
    capital_at_entry: float
    notional: float
    quantity: float
    risk_amount: float
    gross_pnl: float
    fees: float
    slippage_cost: float
    pnl: float
    pnl_pct: float
    rr_ratio: float
    changed: bool
    replacement_type: str
    baseline_exit_reason: str
    baseline_signal_exit_price: float
    baseline_exit_price: float
    delta_pnl: float
    delta_rr: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recalculate compounded strategy performance after OBI exit replacement")
    parser.add_argument("--baseline-trades-json", required=True)
    parser.add_argument("--obi-replay-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def _apply_exit_slippage(signal_exit_price: float, direction: str, slippage_bps: float) -> float:
    slip = slippage_bps / 10_000.0
    if direction == "BULL":
        return signal_exit_price * (1 - slip)
    return signal_exit_price * (1 + slip)


def _timestamp_ms_to_utc_string(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _compute_metrics(trades: list[RepricedTrade], initial_capital: float) -> dict[str, Any]:
    if not trades:
        return {
            "initial_capital": initial_capital,
            "final_capital": initial_capital,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "risk_adjusted_return": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "wl_ratio": 0.0,
            "target_hit_rate": 0.0,
            "gross_pnl_before_fees": 0.0,
            "total_fees_paid": 0.0,
            "total_slippage_cost": 0.0,
            "exit_reasons": {},
            "strategy_cagr_pct": 0.0,
        }

    wins = [trade for trade in trades if trade.pnl > 0]
    losses = [trade for trade in trades if trade.pnl <= 0]
    gross_profit = sum(trade.pnl for trade in wins)
    gross_loss = abs(sum(trade.pnl for trade in losses))
    final_capital = initial_capital + sum(trade.pnl for trade in trades)
    total_return = (final_capital - initial_capital) / initial_capital * 100 if initial_capital > 0 else 0.0
    returns = [trade.pnl_pct for trade in trades]
    mean_r = sum(returns) / len(returns)
    std_r = math.sqrt(sum((value - mean_r) ** 2 for value in returns) / len(returns)) if returns else 0.0
    sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
    peak = initial_capital
    run_cap = initial_capital
    max_dd = 0.0
    exit_reasons: dict[str, int] = {}
    for trade in trades:
        run_cap += trade.pnl
        peak = max(peak, run_cap)
        max_dd = max(max_dd, (peak - run_cap) / peak * 100 if peak > 0 else 0.0)
        exit_reasons[trade.exit_reason] = exit_reasons.get(trade.exit_reason, 0) + 1
    target_hits = sum(1 for trade in trades if "target" in trade.exit_reason)
    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0
    start_dt = datetime.strptime(trades[0].entry_time, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(trades[-1].exit_time, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    years = max((end_dt - start_dt).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    cagr = ((final_capital / initial_capital) ** (1 / years) - 1) * 100 if initial_capital > 0 and final_capital > 0 else 0.0
    return {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "risk_adjusted_return": total_return / max_dd if max_dd > 0 else 0.0,
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0.0,
        "wl_ratio": avg_win / avg_loss if avg_loss > 0 else 0.0,
        "target_hit_rate": target_hits / len(trades) * 100 if trades else 0.0,
        "gross_pnl_before_fees": sum(trade.gross_pnl for trade in trades),
        "total_fees_paid": sum(trade.fees for trade in trades),
        "total_slippage_cost": sum(trade.slippage_cost for trade in trades),
        "exit_reasons": exit_reasons,
        "strategy_cagr_pct": cagr,
    }


def _replacement_for_trade(baseline_trade: dict[str, Any], replay_trade: dict[str, Any] | None) -> tuple[str, float, str, bool]:
    baseline_reason = str(baseline_trade["original_exit_reason"])
    baseline_signal_exit_price = float(baseline_trade["original_signal_exit_price"])
    baseline_exit_time = str(baseline_trade["exit_time"])
    if replay_trade is None:
        return baseline_exit_time, baseline_signal_exit_price, baseline_reason, False

    decisions = replay_trade.get("decisions") or []
    exit_decision = next((decision for decision in decisions if decision.get("action") == "exit"), None)
    if exit_decision is not None and exit_decision.get("exit_price") is not None:
        return (
            _timestamp_ms_to_utc_string(int(exit_decision["timestamp_ms"])),
            float(exit_decision["exit_price"]),
            str(exit_decision.get("reason") or "obi_force_exit"),
            True,
        )

    if baseline_reason in STOPLIKE_REASONS:
        final_stop_price = replay_trade.get("final_stop_price")
        if final_stop_price is not None:
            final_stop_price = float(final_stop_price)
            direction = str(baseline_trade["direction"])
            improved = (
                final_stop_price > baseline_signal_exit_price
                if direction == "BULL"
                else final_stop_price < baseline_signal_exit_price
            )
            if improved:
                return baseline_exit_time, final_stop_price, "obi_tightened_stop", True

    return baseline_exit_time, baseline_signal_exit_price, baseline_reason, False


def _reprice_trade(
    baseline_trade: dict[str, Any],
    replay_trade: dict[str, Any] | None,
    capital_at_entry: float,
    leverage: float,
    fixed_notional_usdt: float | None,
    taker_fee_rate: float,
    slippage_bps: float,
) -> RepricedTrade:
    direction = str(baseline_trade["direction"])
    entry_context = baseline_trade.get("entry_context") or {}
    signal_entry_price = float(baseline_trade["signal_entry_price"])
    entry_price = float(baseline_trade["entry_price"])
    initial_stop_price = float(baseline_trade["initial_stop_price"])
    stop_distance = abs(entry_price - initial_stop_price)
    risk_per_trade = float(entry_context["risk_per_trade"])
    position_size_pct = float(entry_context["position_size_pct"])
    risk_amount = capital_at_entry * risk_per_trade
    max_notional = fixed_notional_usdt if fixed_notional_usdt is not None else capital_at_entry * position_size_pct * leverage
    risk_based_notional = (risk_amount / stop_distance) * entry_price if stop_distance > 0 else max_notional
    notional = min(max_notional, risk_based_notional)
    quantity = notional / entry_price if entry_price > 0 else 0.0
    entry_fee = notional * taker_fee_rate
    entry_slippage_cost = quantity * abs(entry_price - signal_entry_price)

    exit_time, signal_exit_price, exit_reason, changed = _replacement_for_trade(baseline_trade, replay_trade)
    exit_price = _apply_exit_slippage(signal_exit_price, direction, slippage_bps)
    if direction == "BULL":
        gross_pnl = quantity * (exit_price - entry_price)
    else:
        gross_pnl = quantity * (entry_price - exit_price)
    exit_fee = quantity * exit_price * taker_fee_rate
    exit_slippage_cost = quantity * abs(exit_price - signal_exit_price)
    fees = entry_fee + exit_fee
    slippage_cost = entry_slippage_cost + exit_slippage_cost
    pnl = gross_pnl - fees
    pnl_pct = pnl / capital_at_entry if capital_at_entry > 0 else 0.0
    rr_ratio = pnl / risk_amount if risk_amount > 0 else 0.0
    baseline_pnl = float(baseline_trade["pnl"])
    baseline_rr = float(baseline_trade["rr_ratio"])
    replacement_type = "none"
    if changed:
        replacement_type = "obi_force_exit" if exit_reason == "obi_force_exit" else "obi_tightened_stop"

    return RepricedTrade(
        trade_id=int(baseline_trade["trade_id"]),
        direction=direction,
        entry_time=str(baseline_trade["entry_time"]),
        exit_time=exit_time,
        entry_price=entry_price,
        signal_entry_price=signal_entry_price,
        signal_exit_price=signal_exit_price,
        exit_price=exit_price,
        exit_reason=exit_reason,
        capital_at_entry=capital_at_entry,
        notional=notional,
        quantity=quantity,
        risk_amount=risk_amount,
        gross_pnl=gross_pnl,
        fees=fees,
        slippage_cost=slippage_cost,
        pnl=pnl,
        pnl_pct=pnl_pct,
        rr_ratio=rr_ratio,
        changed=changed,
        replacement_type=replacement_type,
        baseline_exit_reason=str(baseline_trade["original_exit_reason"]),
        baseline_signal_exit_price=float(baseline_trade["original_signal_exit_price"]),
        baseline_exit_price=float(baseline_trade["original_exit_price"]),
        delta_pnl=pnl - baseline_pnl,
        delta_rr=rr_ratio - baseline_rr,
    )


def _write_csv(path: Path, trades: list[RepricedTrade]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trade_id",
                "direction",
                "entry_time",
                "exit_time",
                "baseline_exit_reason",
                "exit_reason",
                "replacement_type",
                "changed",
                "capital_at_entry",
                "notional",
                "quantity",
                "signal_exit_price",
                "exit_price",
                "pnl",
                "rr_ratio",
                "delta_pnl",
                "delta_rr",
            ],
        )
        writer.writeheader()
        for trade in trades:
            writer.writerow({
                "trade_id": trade.trade_id,
                "direction": trade.direction,
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "baseline_exit_reason": trade.baseline_exit_reason,
                "exit_reason": trade.exit_reason,
                "replacement_type": trade.replacement_type,
                "changed": trade.changed,
                "capital_at_entry": trade.capital_at_entry,
                "notional": trade.notional,
                "quantity": trade.quantity,
                "signal_exit_price": trade.signal_exit_price,
                "exit_price": trade.exit_price,
                "pnl": trade.pnl,
                "rr_ratio": trade.rr_ratio,
                "delta_pnl": trade.delta_pnl,
                "delta_rr": trade.delta_rr,
            })


def main() -> None:
    args = parse_args()
    baseline_payload = json.loads(Path(args.baseline_trades_json).read_text(encoding="utf-8"))
    replay_payload = json.loads(Path(args.obi_replay_json).read_text(encoding="utf-8"))
    baseline_trades = baseline_payload.get("trades") or []
    replay_map = {int(trade["open_action_id"]): trade for trade in replay_payload.get("trades") or []}
    baseline_metrics = baseline_payload["metadata"]["metrics_summary"]
    params = baseline_metrics.get("parameters") or {}
    initial_capital = float(baseline_metrics["initial_capital"])
    leverage = float(params["leverage"])
    fixed_notional_usdt = params.get("fixed_notional_usdt")
    if fixed_notional_usdt is not None:
        fixed_notional_usdt = float(fixed_notional_usdt)
    taker_fee_rate = float(params["taker_fee_rate"])
    slippage_bps = float(params["slippage_bps"])

    repriced_trades: list[RepricedTrade] = []
    capital = initial_capital
    for baseline_trade in baseline_trades:
        replay_trade = replay_map.get(int(baseline_trade["trade_id"]))
        repriced = _reprice_trade(
            baseline_trade=baseline_trade,
            replay_trade=replay_trade,
            capital_at_entry=capital,
            leverage=leverage,
            fixed_notional_usdt=fixed_notional_usdt,
            taker_fee_rate=taker_fee_rate,
            slippage_bps=slippage_bps,
        )
        repriced_trades.append(repriced)
        capital += repriced.pnl

    replaced_metrics = _compute_metrics(repriced_trades, initial_capital)
    replacement_counts = {
        "changed_trades": sum(1 for trade in repriced_trades if trade.changed),
        "obi_force_exit_trades": sum(1 for trade in repriced_trades if trade.replacement_type == "obi_force_exit"),
        "obi_tightened_stop_trades": sum(1 for trade in repriced_trades if trade.replacement_type == "obi_tightened_stop"),
    }
    payload = {
        "baseline": baseline_metrics,
        "replaced": replaced_metrics,
        "delta": {
            "total_return_pct": replaced_metrics["total_return_pct"] - baseline_metrics["total_return_pct"],
            "profit_factor": replaced_metrics["profit_factor"] - baseline_metrics["profit_factor"],
            "win_rate": replaced_metrics["win_rate"] - baseline_metrics["win_rate"],
            "max_drawdown_pct": replaced_metrics["max_drawdown_pct"] - baseline_metrics["max_drawdown_pct"],
            "strategy_cagr_pct": replaced_metrics["strategy_cagr_pct"] - baseline_metrics.get("strategy_cagr_pct", 0.0),
        },
        "replacement_counts": replacement_counts,
        "replay_summary": replay_payload.get("summary") or {},
        "trades": [trade.__dict__ for trade in repriced_trades],
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(Path(args.output_csv), repriced_trades)
    print(json.dumps({
        "changed_trades": replacement_counts["changed_trades"],
        "baseline_total_return_pct": baseline_metrics["total_return_pct"],
        "replaced_total_return_pct": replaced_metrics["total_return_pct"],
        "delta_total_return_pct": payload["delta"]["total_return_pct"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
