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

from scripts.replay_qqq_usdt_10x import load_funding, max_drawdown_pct  # noqa: E402
from scripts.scan_qqq_usdt_4h_triggers import attach_daily_state, load_okx_4h, load_signal_path  # noqa: E402
from scripts.scan_qqq_usdt_btc_logic_overlays import enrich_bars  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "config.paper.tqqq-only-strict-recovery-frozen.json"
DEFAULT_OKX_4H = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-4h-futures.feather"
DEFAULT_FUNDING = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-8h-funding_rate.feather"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_usdt_take_profit_scan.json"

TAKE_PROFIT_PROFILES = {
    "none": {"tp_pct": None, "touch_trigger_r": None, "touch_buffer_pct": None},
    "tp_12": {"tp_pct": 12.0, "touch_trigger_r": None, "touch_buffer_pct": None},
    "tp_20": {"tp_pct": 20.0, "touch_trigger_r": None, "touch_buffer_pct": None},
    "tp_30": {"tp_pct": 30.0, "touch_trigger_r": None, "touch_buffer_pct": None},
    "touch_1r_0.8": {"tp_pct": None, "touch_trigger_r": 1.0, "touch_buffer_pct": 0.8},
    "touch_1.5r_1.0": {"tp_pct": None, "touch_trigger_r": 1.5, "touch_buffer_pct": 1.0},
    "tp20_touch1r": {"tp_pct": 20.0, "touch_trigger_r": 1.0, "touch_buffer_pct": 0.8},
}


def simulate(
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    stop_loss_pct: float,
    taker_fee_rate: float,
    slippage_bps: float,
    initial_capital: float,
    take_profit_profile_name: str,
) -> dict[str, Any]:
    profile = TAKE_PROFIT_PROFILES[take_profit_profile_name]
    merged = pd.merge_asof(
        bars.sort_values("date"),
        funding.sort_values("date"),
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["funding_rate_value"] = merged["funding_rate_value"].fillna(0.0)

    capital = float(initial_capital)
    holding = False
    entry_price = 0.0
    stop_price = 0.0
    peak_close = 0.0
    prev_allow = False
    current_trade: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    per_side_cost = float(taker_fee_rate) + float(slippage_bps) / 10000.0

    for row in merged.itertuples(index=False):
        start_capital = capital
        allow_now = bool(row.allow_long)
        entered_today = False
        exited_today = False
        stop_hit = False
        tp_hit = False
        fee_cost = 0.0
        funding_cost = 0.0

        if holding and not allow_now:
            leverage_now = 8.0 if not bool(row.high_growth) and not bool(row.defense_state) else (10.0 if bool(row.high_growth) else 2.0)
            fee_cost = per_side_cost
            capital *= 1.0 - fee_cost * leverage_now
            holding = False
            exited_today = True
            if current_trade is not None:
                trades.append(
                    {
                        "entry_date": current_trade["entry_date"],
                        "exit_date": str(pd.Timestamp(row.date)),
                        "trade_return_pct": round((capital / float(current_trade["entry_capital"]) - 1.0) * 100.0, 2),
                    }
                )
            current_trade = None

        if allow_now and not holding and not prev_allow:
            leverage_now = 8.0 if not bool(row.high_growth) and not bool(row.defense_state) else (10.0 if bool(row.high_growth) else 2.0)
            fee_cost = per_side_cost
            capital *= 1.0 - fee_cost * leverage_now
            holding = True
            entered_today = True
            entry_price = float(row.open)
            stop_price = entry_price * (1.0 - float(stop_loss_pct) / 100.0)
            peak_close = float(row.open)
            current_trade = {"entry_date": str(pd.Timestamp(row.date)), "entry_capital": capital}

        leverage_now = 0.0
        if holding:
            if bool(row.high_growth):
                leverage_now = 10.0
            elif bool(row.defense_state):
                leverage_now = 2.0
            else:
                leverage_now = 8.0

            open_price = float(row.open)
            high_price = float(row.high)
            low_price = float(row.low)
            close_price = float(row.close)
            peak_close = max(peak_close, close_price)

            if profile["touch_trigger_r"] is not None and entry_price > 0:
                r_multiple = (peak_close / entry_price - 1.0) / (float(stop_loss_pct) / 100.0)
                if r_multiple >= float(profile["touch_trigger_r"]):
                    stop_price = max(stop_price, peak_close * (1.0 - float(profile["touch_buffer_pct"]) / 100.0))

            tp_price = entry_price * (1.0 + float(profile["tp_pct"]) / 100.0) if profile["tp_pct"] is not None else None
            if low_price <= stop_price:
                stop_hit = True
                exit_price = stop_price
            elif tp_price is not None and high_price >= tp_price:
                tp_hit = True
                exit_price = tp_price
            else:
                exit_price = None

            if exit_price is not None:
                bar_ret = exit_price / open_price - 1.0 if open_price > 0 else 0.0
                capital *= 1.0 + leverage_now * bar_ret
                capital *= 1.0 - per_side_cost * leverage_now
                holding = False
                exited_today = True
                if current_trade is not None:
                    trades.append(
                        {
                            "entry_date": current_trade["entry_date"],
                            "exit_date": str(pd.Timestamp(row.date)),
                            "trade_return_pct": round((capital / float(current_trade["entry_capital"]) - 1.0) * 100.0, 2),
                        }
                    )
                current_trade = None
            else:
                bar_ret = close_price / open_price - 1.0 if open_price > 0 else 0.0
                capital *= 1.0 + leverage_now * bar_ret
                funding_cost = max(float(row.funding_rate_value), 0.0) * leverage_now
                capital *= 1.0 - funding_cost

        rows.append(
            {
                "date": pd.Timestamp(row.date),
                "holding": holding,
                "entered_today": entered_today,
                "exited_today": exited_today,
                "stop_hit": stop_hit,
                "tp_hit": tp_hit,
                "capital": float(capital),
                "daily_return": capital / start_capital - 1.0 if start_capital > 0 else 0.0,
                "funding_cost": float(funding_cost),
            }
        )
        prev_allow = allow_now

    path = pd.DataFrame(rows)
    trades_df = pd.DataFrame(trades)
    return {
        "take_profit_profile": take_profit_profile_name,
        "summary": {
            "total_return_pct": round((float(path.iloc[-1]["capital"]) / float(initial_capital) - 1.0) * 100.0, 2) if not path.empty else 0.0,
            "max_drawdown_pct": round(max_drawdown_pct(path["capital"]), 2) if not path.empty else 0.0,
            "trades": int(len(trades_df)),
            "win_rate_pct": round(float((trades_df["trade_return_pct"] > 0).mean() * 100.0), 2) if not trades_df.empty else 0.0,
            "avg_trade_return_pct": round(float(trades_df["trade_return_pct"].mean()), 2) if not trades_df.empty else 0.0,
            "funding_cost_pct_est": round(float(path["funding_cost"].sum() * 100.0), 2) if not path.empty else 0.0,
            "invested_bars": int(path["holding"].sum()) if not path.empty else 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan take-profit profiles on QQQ/USDT dynamic leverage candidate.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--okx-4h", default=str(DEFAULT_OKX_4H))
    parser.add_argument("--funding", default=str(DEFAULT_FUNDING))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--stop-loss-pct", type=float, default=2.0)
    parser.add_argument("--taker-fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    args = parser.parse_args()

    config, signal_path = load_signal_path(Path(args.config))
    bars = enrich_bars(attach_daily_state(load_okx_4h(Path(args.okx_4h)), signal_path))
    funding = load_funding(Path(args.funding))

    results = []
    for take_profit_profile_name in TAKE_PROFIT_PROFILES:
        results.append(
            simulate(
                bars,
                funding,
                stop_loss_pct=float(args.stop_loss_pct),
                taker_fee_rate=float(args.taker_fee_rate),
                slippage_bps=float(args.slippage_bps),
                initial_capital=float(config["initial_capital"]),
                take_profit_profile_name=take_profit_profile_name,
            )
        )

    results = sorted(
        results,
        key=lambda x: (
            x["summary"]["total_return_pct"],
            -x["summary"]["max_drawdown_pct"],
        ),
        reverse=True,
    )
    payload = {
        "config": {
            "signal_frozen_label": config.get("frozen_label"),
            "stop_loss_pct": float(args.stop_loss_pct),
            "taker_fee_rate": float(args.taker_fee_rate),
            "slippage_bps": float(args.slippage_bps),
            "dynamic_leverage_profile": "base8_off10_def2",
        },
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(out)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
