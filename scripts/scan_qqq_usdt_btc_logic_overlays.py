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


DEFAULT_CONFIG = ROOT / "config" / "config.paper.tqqq-only-strict-recovery-frozen.json"
DEFAULT_OKX_4H = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-4h-futures.feather"
DEFAULT_FUNDING = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-8h-funding_rate.feather"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_usdt_btc_logic_overlay_scan.json"

LEVERAGE_PROFILES = {
    "fixed10": {"base": 10.0, "offense": 10.0, "defense": 10.0},
    "base8_off10_def2": {"base": 8.0, "offense": 10.0, "defense": 2.0},
    "base6_off10_def3": {"base": 6.0, "offense": 10.0, "defense": 3.0},
    "base4_off8_def2": {"base": 4.0, "offense": 8.0, "defense": 2.0},
}

FAILED_BREAKOUT_PROFILES = {
    "off": "off",
    "reduce2": "reduce2",
    "flat": "flat",
}

TOUCH_LOCK_PROFILES = {
    "off": None,
    "lock_1r_0.8": {"trigger_r": 1.0, "buffer_pct": 0.8},
    "lock_1.5r_1.0": {"trigger_r": 1.5, "buffer_pct": 1.0},
}


def enrich_bars(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    frame["mom_3"] = frame["close"] / frame["close"].shift(3) - 1.0
    frame["ema_gap_pct"] = frame["ema20"] / frame["ema50"] - 1.0
    frame["breakout_fail_12"] = (frame["high"] > frame["prev_high_12"]) & (frame["close"] < frame["prev_high_12"])
    frame["high_growth"] = (
        (frame["close"] > frame["prev_high_12"])
        & (frame["ema20"] > frame["ema50"])
        & (frame["mom_3"] > 0.015)
        & (frame["ema_gap_pct"] > 0.005)
    )
    frame["defense_state"] = frame["breakout_fail_12"] | (frame["close"] < frame["ema20"])
    return frame


def simulate(
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    leverage_profile_name: str,
    failed_breakout_profile_name: str,
    touch_lock_profile_name: str,
    stop_loss_pct: float,
    taker_fee_rate: float,
    slippage_bps: float,
    initial_capital: float,
) -> dict[str, Any]:
    lev_profile = LEVERAGE_PROFILES[leverage_profile_name]
    fb_profile = FAILED_BREAKOUT_PROFILES[failed_breakout_profile_name]
    touch_profile = TOUCH_LOCK_PROFILES[touch_lock_profile_name]
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
        guard_hit = False
        fee_cost = 0.0
        funding_cost = 0.0

        if holding and not allow_now:
            fee_cost = per_side_cost
            capital *= 1.0 - fee_cost * lev_profile["base"]
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
            fee_cost = per_side_cost
            capital *= 1.0 - fee_cost * lev_profile["base"]
            holding = True
            entered_today = True
            entry_price = float(row.open)
            stop_price = entry_price * (1.0 - float(stop_loss_pct) / 100.0)
            peak_close = float(row.open)
            current_trade = {"entry_date": str(pd.Timestamp(row.date)), "entry_capital": capital}

        leverage_now = 0.0
        if holding:
            if bool(row.high_growth):
                leverage_now = lev_profile["offense"]
            elif bool(row.defense_state):
                leverage_now = lev_profile["defense"]
            else:
                leverage_now = lev_profile["base"]

            if bool(row.breakout_fail_12):
                if fb_profile == "flat":
                    guard_hit = True
                    exit_price = float(row.open)
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
                    leverage_now = 0.0
                elif fb_profile == "reduce2":
                    leverage_now = min(leverage_now, 2.0)

        if holding:
            open_price = float(row.open)
            low_price = float(row.low)
            close_price = float(row.close)
            peak_close = max(peak_close, close_price)

            if touch_profile is not None and entry_price > 0:
                r_multiple = (peak_close / entry_price - 1.0) / (float(stop_loss_pct) / 100.0)
                if r_multiple >= float(touch_profile["trigger_r"]):
                    stop_price = max(stop_price, peak_close * (1.0 - float(touch_profile["buffer_pct"]) / 100.0))

            if low_price <= stop_price:
                stop_hit = True
                exit_price = stop_price
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
                "guard_hit": guard_hit,
                "capital": float(capital),
                "daily_return": capital / start_capital - 1.0 if start_capital > 0 else 0.0,
                "funding_cost": float(funding_cost),
                "leverage_now": float(leverage_now),
            }
        )
        prev_allow = allow_now

    path = pd.DataFrame(rows)
    trades_df = pd.DataFrame(trades)
    return {
        "leverage_profile": leverage_profile_name,
        "failed_breakout_profile": failed_breakout_profile_name,
        "touch_lock_profile": touch_lock_profile_name,
        "summary": {
            "total_return_pct": round((float(path.iloc[-1]["capital"]) / float(initial_capital) - 1.0) * 100.0, 2) if not path.empty else 0.0,
            "max_drawdown_pct": round(max_drawdown_pct(path["capital"]), 2) if not path.empty else 0.0,
            "trades": int(len(trades_df)),
            "win_rate_pct": round(float((trades_df["trade_return_pct"] > 0).mean() * 100.0), 2) if not trades_df.empty else 0.0,
            "avg_trade_return_pct": round(float(trades_df["trade_return_pct"].mean()), 2) if not trades_df.empty else 0.0,
            "funding_cost_pct_est": round(float(path["funding_cost"].sum() * 100.0), 2) if not path.empty else 0.0,
            "invested_bars": int(path["holding"].sum()) if not path.empty else 0,
            "avg_leverage_when_in": round(float(path.loc[path["holding"], "leverage_now"].mean()), 2) if (not path.empty and (path["holding"]).any()) else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan portable BTC frozen logic overlays on QQQ/USDT 10x replay.")
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
    for lev_name in LEVERAGE_PROFILES:
        for fb_name in FAILED_BREAKOUT_PROFILES:
            for tl_name in TOUCH_LOCK_PROFILES:
                results.append(
                    simulate(
                        bars,
                        funding,
                        leverage_profile_name=lev_name,
                        failed_breakout_profile_name=fb_name,
                        touch_lock_profile_name=tl_name,
                        stop_loss_pct=float(args.stop_loss_pct),
                        taker_fee_rate=float(args.taker_fee_rate),
                        slippage_bps=float(args.slippage_bps),
                        initial_capital=float(config["initial_capital"]),
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
        },
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(out)
    print(json.dumps(results[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
