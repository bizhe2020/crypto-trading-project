#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_proxy_strategy_router import qqq_replay_risk_on_allowed, qqq_replay_signal_leverage  # noqa: E402
from scripts.replay_qqq_usdt_10x import load_funding, max_drawdown_pct  # noqa: E402
from scripts.scan_qqq_usdt_4h_triggers import attach_daily_state, load_okx_4h, load_signal_path  # noqa: E402
from scripts.scan_qqq_usdt_btc_logic_overlays import enrich_bars  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "config.paper.qqq-usdt-aggressive-frozen.json"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_usdt_profit_roll_scan.json"
DEFAULT_OUTPUT_CSV = ROOT / "var" / "reports" / "qqq_usdt_profit_roll_scan.csv"


def _safe_bool(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def prepare_bars(config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    signal_config, signal_path = load_signal_path(ROOT / str(config["signal_source"]))
    bars = enrich_bars(attach_daily_state(load_okx_4h(ROOT / str(config["data_4h"])), signal_path))
    funding = load_funding(ROOT / str(config["funding_history_path"]))
    merged = pd.merge_asof(
        bars.sort_values("date"),
        funding.sort_values("date"),
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["funding_rate_value"] = merged["funding_rate_value"].fillna(0.0)
    merged = merged.sort_values("date").reset_index(drop=True)
    for column in ("allow_long", "high_growth", "defense_state", "breakout_12"):
        merged[f"signal_{column}"] = merged[column].shift(1)
    merged["signal_date"] = merged["date"].shift(1)
    return signal_config, merged


def profit_trigger_ok(
    *,
    trigger: str,
    open_price: float,
    entry_price: float,
    stop_price: float,
    stop_loss_pct: float,
) -> bool:
    if entry_price <= 0:
        return False
    current_r = (open_price / entry_price - 1.0) / (stop_loss_pct / 100.0)
    if trigger == "any":
        return current_r > 0.0
    if trigger == "0.5r":
        return current_r >= 0.5
    if trigger == "1.0r":
        return current_r >= 1.0
    if trigger == "1.5r":
        return current_r >= 1.5
    if trigger == "breakeven_stop":
        return stop_price >= entry_price
    if trigger == "breakeven_or_1r":
        return stop_price >= entry_price or current_r >= 1.0
    raise ValueError(f"Unsupported profit trigger: {trigger}")


def simulate_profit_roll(
    merged: pd.DataFrame,
    config: dict[str, Any],
    *,
    roll_enabled: bool,
    min_actual_leverage: float,
    profit_trigger: str,
    max_rolls_per_trade: int,
    cooldown_bars: int,
    skip_defense_roll: bool,
) -> dict[str, Any]:
    initial_capital = float(config.get("initial_capital", 1000.0))
    stop_loss_pct = float(config["stop_loss_pct"])
    per_side_cost = float(config["taker_fee_rate"]) + float(config["slippage_bps"]) / 10000.0
    lev_profile = {
        "base": float(config["base_leverage"]),
        "offense": float(config["offense_leverage"]),
        "defense": float(config["defense_leverage"]),
    }
    target_leverage = max(float(lev_profile["base"]), float(lev_profile["offense"]), float(lev_profile["defense"]))

    capital = initial_capital
    holding = False
    qty = 0.0
    entry_price = 0.0
    stop_price = 0.0
    peak_close = 0.0
    current_trade: dict[str, Any] | None = None
    rolls_this_trade = 0
    cooldown_remaining = 0

    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    roll_count = 0
    blocked_rolls = 0
    fee_cost_total = 0.0
    funding_cost_total = 0.0
    notional_added_total = 0.0

    for row in merged.itertuples(index=False):
        start_capital = capital
        signal = {
            "allow_long": row.signal_allow_long,
            "high_growth": row.signal_high_growth,
            "defense_state": row.signal_defense_state,
            "breakout_12": row.signal_breakout_12,
        }
        allow_now = _safe_bool(signal["allow_long"])
        defense_now = _safe_bool(signal["defense_state"])
        signal_leverage = qqq_replay_signal_leverage(signal, lev_profile)
        open_price = float(row.open)
        low_price = float(row.low)
        close_price = float(row.close)
        entered = False
        exited = False
        stop_hit = False
        rolled = False
        roll_notional = 0.0
        fee_cost = 0.0
        funding_cost = 0.0

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        if holding and not allow_now:
            exit_notional = abs(qty * open_price)
            fee_cost += exit_notional * per_side_cost
            capital -= fee_cost
            fee_cost_total += fee_cost
            if current_trade is not None:
                trades.append(
                    {
                        "entry_date": current_trade["entry_date"],
                        "exit_date": str(pd.Timestamp(row.date)),
                        "trade_return_pct": round((capital / float(current_trade["entry_capital"]) - 1.0) * 100.0, 2),
                        "rolls": rolls_this_trade,
                    }
                )
            holding = False
            exited = True
            qty = 0.0
            current_trade = None
            rolls_this_trade = 0
            cooldown_remaining = 0

        if allow_now and not holding and signal_leverage > 0.0:
            risk_on_open, _ = qqq_replay_risk_on_allowed(config, row.date)
            if risk_on_open and capital > 0:
                entry_notional = capital * target_leverage
                fee_cost += entry_notional * per_side_cost
                capital -= entry_notional * per_side_cost
                fee_cost_total += entry_notional * per_side_cost
                qty = entry_notional / open_price if open_price > 0 else 0.0
                holding = True
                entered = True
                entry_price = open_price
                stop_price = entry_price * (1.0 - stop_loss_pct / 100.0)
                peak_close = open_price
                current_trade = {"entry_date": str(pd.Timestamp(row.date)), "entry_capital": capital}
                rolls_this_trade = 0
                cooldown_remaining = 0

        actual_leverage_before = abs(qty * open_price) / capital if holding and capital > 0 else 0.0
        if (
            roll_enabled
            and holding
            and allow_now
            and capital > 0
            and actual_leverage_before < float(min_actual_leverage)
            and rolls_this_trade < int(max_rolls_per_trade)
            and cooldown_remaining <= 0
            and not (skip_defense_roll and defense_now)
            and profit_trigger_ok(
                trigger=profit_trigger,
                open_price=open_price,
                entry_price=entry_price,
                stop_price=stop_price,
                stop_loss_pct=stop_loss_pct,
            )
        ):
            risk_on_open, _ = qqq_replay_risk_on_allowed(config, row.date)
            if risk_on_open:
                target_notional = capital * target_leverage
                current_notional = abs(qty * open_price)
                add_notional = max(target_notional - current_notional, 0.0)
                if add_notional > 0:
                    roll_fee = add_notional * per_side_cost
                    capital -= roll_fee
                    fee_cost += roll_fee
                    fee_cost_total += roll_fee
                    qty += add_notional / open_price if open_price > 0 else 0.0
                    roll_notional = add_notional
                    notional_added_total += add_notional
                    roll_count += 1
                    rolls_this_trade += 1
                    cooldown_remaining = int(cooldown_bars)
                    rolled = True
            else:
                blocked_rolls += 1

        actual_leverage_after = abs(qty * open_price) / capital if holding and capital > 0 else 0.0
        if holding:
            previous_stop = stop_price
            if low_price <= previous_stop:
                stop_hit = True
                exit_price = previous_stop
                capital += qty * (exit_price - open_price)
                exit_notional = abs(qty * exit_price)
                exit_fee = exit_notional * per_side_cost
                capital -= exit_fee
                fee_cost += exit_fee
                fee_cost_total += exit_fee
                if current_trade is not None:
                    trades.append(
                        {
                            "entry_date": current_trade["entry_date"],
                            "exit_date": str(pd.Timestamp(row.date)),
                            "trade_return_pct": round((capital / float(current_trade["entry_capital"]) - 1.0) * 100.0, 2),
                            "rolls": rolls_this_trade,
                        }
                    )
                holding = False
                exited = True
                qty = 0.0
                current_trade = None
                rolls_this_trade = 0
                cooldown_remaining = 0
            else:
                capital += qty * (close_price - open_price)
                funding_cost = max(float(row.funding_rate_value), 0.0) * abs(qty * close_price)
                capital -= funding_cost
                funding_cost_total += funding_cost
                peak_close = max(peak_close, close_price)
                stop_price = max(stop_price, peak_close * (1.0 - stop_loss_pct / 100.0))

        rows.append(
            {
                "date": pd.Timestamp(row.date),
                "capital": float(capital),
                "bar_return": float(capital / start_capital - 1.0 if start_capital > 0 else 0.0),
                "holding": bool(holding),
                "allow_long": bool(allow_now),
                "defense_state": bool(defense_now),
                "entered": bool(entered),
                "exited": bool(exited),
                "stop_hit": bool(stop_hit),
                "rolled": bool(rolled),
                "roll_notional": float(roll_notional),
                "actual_leverage_before": float(actual_leverage_before),
                "actual_leverage_after": float(actual_leverage_after),
                "fee_cost": float(fee_cost),
                "funding_cost": float(funding_cost),
            }
        )

        if capital <= 0:
            break

    path = pd.DataFrame(rows)
    trades_df = pd.DataFrame(trades)
    end_capital = float(path.iloc[-1]["capital"]) if not path.empty else initial_capital
    invested = path[path["holding"] | path["entered"] | path["exited"]].copy() if not path.empty else pd.DataFrame()
    return {
        "roll_enabled": bool(roll_enabled),
        "min_actual_leverage": float(min_actual_leverage),
        "profit_trigger": str(profit_trigger),
        "max_rolls_per_trade": int(max_rolls_per_trade),
        "cooldown_bars": int(cooldown_bars),
        "skip_defense_roll": bool(skip_defense_roll),
        "summary": {
            "total_return_pct": round((end_capital / initial_capital - 1.0) * 100.0, 2),
            "max_drawdown_pct": round(max_drawdown_pct(path["capital"]), 2) if not path.empty else 0.0,
            "score": round((end_capital / initial_capital - 1.0) * 100.0 - max_drawdown_pct(path["capital"]) * 2.0, 2) if not path.empty else 0.0,
            "bars": int(len(path)),
            "invested_bars": int(path["holding"].sum()) if not path.empty else 0,
            "trades": int(len(trades_df)),
            "win_rate_pct": round(float((trades_df["trade_return_pct"] > 0).mean() * 100.0), 2) if not trades_df.empty else 0.0,
            "avg_trade_return_pct": round(float(trades_df["trade_return_pct"].mean()), 2) if not trades_df.empty else 0.0,
            "roll_count": int(roll_count),
            "blocked_rolls": int(blocked_rolls),
            "avg_rolls_per_trade": round(float(trades_df["rolls"].mean()), 2) if not trades_df.empty else 0.0,
            "fee_cost_pct_of_initial": round(float(fee_cost_total / initial_capital * 100.0), 2),
            "funding_cost_pct_of_initial": round(float(funding_cost_total / initial_capital * 100.0), 2),
            "notional_added_pct_of_initial": round(float(notional_added_total / initial_capital * 100.0), 2),
            "avg_actual_leverage_when_in": round(float(invested["actual_leverage_after"].replace(0.0, pd.NA).mean()), 2) if not invested.empty else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan profit-only roll-to-10x policies for QQQ/USDT.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    signal_config, merged = prepare_bars(config)

    results = [
        simulate_profit_roll(
            merged,
            config,
            roll_enabled=False,
            min_actual_leverage=0.0,
            profit_trigger="none",
            max_rolls_per_trade=0,
            cooldown_bars=0,
            skip_defense_roll=True,
        )
    ]

    for threshold, trigger, max_rolls, cooldown, skip_defense in itertools.product(
        [8.0, 8.5, 9.0, 9.5],
        ["any", "0.5r", "1.0r", "1.5r", "breakeven_stop", "breakeven_or_1r"],
        [1, 2, 4, 99],
        [0, 1, 2],
        [True, False],
    ):
        results.append(
            simulate_profit_roll(
                merged,
                config,
                roll_enabled=True,
                min_actual_leverage=float(threshold),
                profit_trigger=str(trigger),
                max_rolls_per_trade=int(max_rolls),
                cooldown_bars=int(cooldown),
                skip_defense_roll=bool(skip_defense),
            )
        )

    by_return = sorted(
        results,
        key=lambda item: (
            item["summary"]["total_return_pct"],
            -item["summary"]["max_drawdown_pct"],
            -item["summary"]["roll_count"],
        ),
        reverse=True,
    )
    by_score = sorted(results, key=lambda item: item["summary"]["score"], reverse=True)
    payload = {
        "mode": "qqq_usdt_profit_roll_scan",
        "config": {
            "source": str(Path(args.config)),
            "signal_frozen_label": signal_config.get("frozen_label"),
            "execution_policy": {
                "signal_lag_4h_bars": 1,
                "uses_prior_closed_4h_signal": True,
                "risk_on_changes_market_window_only": bool(config.get("qqq_rebalance_risk_on_market_hours_only", False)),
                "market_hours_timezone": str(config.get("qqq_market_hours_timezone", "America/New_York")),
                "market_hours_start": str(config.get("qqq_market_hours_start", "09:30")),
                "market_hours_end": str(config.get("qqq_market_hours_end", "16:00")),
                "market_calendar": str(config.get("qqq_market_calendar", "NYSE")),
            },
            "target_leverage": max(float(config["base_leverage"]), float(config["offense_leverage"]), float(config["defense_leverage"])),
            "stop_loss_pct": float(config["stop_loss_pct"]),
            "per_side_cost": float(config["taker_fee_rate"]) + float(config["slippage_bps"]) / 10000.0,
        },
        "baseline": results[0],
        "best_by_return": by_return[:20],
        "best_by_score": by_score[:20],
        "results": results,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    csv_rows = []
    for item in results:
        csv_rows.append({key: value for key, value in item.items() if key != "summary"} | item["summary"])
    pd.DataFrame(csv_rows).sort_values(
        ["total_return_pct", "max_drawdown_pct"],
        ascending=[False, True],
    ).to_csv(args.output_csv, index=False)

    print(out)
    print(args.output_csv)
    print(json.dumps({"baseline": results[0], "best_by_return": by_return[:10], "best_by_score": by_score[:10]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
