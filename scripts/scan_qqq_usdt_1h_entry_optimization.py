#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_qqq_usdt_10x import load_funding, max_drawdown_pct  # noqa: E402
from scripts.scan_qqq_usdt_4h_triggers import attach_daily_state, load_okx_4h, load_signal_path  # noqa: E402
from scripts.scan_qqq_usdt_btc_logic_overlays import enrich_bars  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "config.paper.tqqq-only-strict-recovery-frozen.json"
DEFAULT_OKX_4H = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-4h-futures.feather"
DEFAULT_OKX_1H = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-1h-futures.feather"
DEFAULT_FUNDING = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-8h-funding_rate.feather"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_usdt_1h_entry_optimization.json"

ENTRY_PROFILES = [
    "4h_open",
    "1h_immediate",
    "1h_pullback_ema20",
    "1h_breakout_6",
    "1h_pullback_or_breakout",
]


def load_okx_1h(path: Path) -> pd.DataFrame:
    df = pd.read_feather(path).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["prev_high_6"] = df["high"].shift(1).rolling(6).max()
    df["prev_high_3"] = df["high"].shift(1).rolling(3).max()
    df["pullback_ema20"] = (df["low"] <= df["ema20"]) & (df["close"] > df["ema20"]) & (df["close"] > df["open"])
    df["breakout_6"] = df["close"] > df["prev_high_6"]
    df["pullback_or_breakout"] = df["pullback_ema20"] | df["breakout_6"]
    return df


def assign_4h_signal_windows(bars4h: pd.DataFrame) -> pd.DataFrame:
    frame = bars4h.copy()
    frame["signal_start"] = frame["date"]
    frame["signal_end"] = frame["date"] + pd.Timedelta(hours=4)
    return frame


def choose_entry_bar(window_1h: pd.DataFrame, profile: str) -> pd.Series | None:
    if window_1h.empty:
        return None
    if profile in {"4h_open", "1h_immediate"}:
        return window_1h.iloc[0]
    if profile == "1h_pullback_ema20":
        matched = window_1h[window_1h["pullback_ema20"]]
        return matched.iloc[0] if not matched.empty else window_1h.iloc[0]
    if profile == "1h_breakout_6":
        matched = window_1h[window_1h["breakout_6"]]
        return matched.iloc[0] if not matched.empty else window_1h.iloc[0]
    if profile == "1h_pullback_or_breakout":
        matched = window_1h[window_1h["pullback_or_breakout"]]
        return matched.iloc[0] if not matched.empty else window_1h.iloc[0]
    raise ValueError(profile)


def simulate(
    bars4h: pd.DataFrame,
    bars1h: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    entry_profile: str,
    stop_loss_pct: float,
    taker_fee_rate: float,
    slippage_bps: float,
    initial_capital: float,
) -> dict:
    merged = pd.merge_asof(
        bars4h.sort_values("date"),
        funding.sort_values("date"),
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["funding_rate_value"] = merged["funding_rate_value"].fillna(0.0)
    merged = assign_4h_signal_windows(merged)

    capital = float(initial_capital)
    holding = False
    prev_allow = False
    entry_price = 0.0
    stop_price = 0.0
    peak_close = 0.0
    current_trade = None
    trades = []
    rows = []
    per_side_cost = float(taker_fee_rate) + float(slippage_bps) / 10000.0

    for row in merged.itertuples(index=False):
        start_capital = capital
        allow_now = bool(row.allow_long)
        entered_today = False
        exited_today = False
        stop_hit = False
        funding_cost = 0.0
        fee_cost = 0.0
        leverage_now = 0.0

        if holding:
            if bool(row.high_growth):
                leverage_now = 10.0
            elif bool(row.defense_state):
                leverage_now = 2.0
            else:
                leverage_now = 10.0

        if holding and not allow_now:
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
            window = bars1h[(bars1h["date"] >= row.signal_start) & (bars1h["date"] < row.signal_end)]
            chosen = choose_entry_bar(window, entry_profile)
            if chosen is None:
                chosen_open = float(row.open)
                chosen_close = float(row.close)
            else:
                chosen_open = float(chosen["open"])
                chosen_close = float(chosen["close"])
            leverage_now = 10.0 if bool(row.high_growth) else (2.0 if bool(row.defense_state) else 10.0)
            fee_cost = per_side_cost
            capital *= 1.0 - fee_cost * leverage_now
            holding = True
            entered_today = True
            entry_price = chosen_open
            stop_price = entry_price * (1.0 - float(stop_loss_pct) / 100.0)
            peak_close = chosen_close
            current_trade = {"entry_date": str(pd.Timestamp(row.date)), "entry_capital": capital}

        if holding:
            if bool(row.high_growth):
                leverage_now = 10.0
            elif bool(row.defense_state):
                leverage_now = 2.0
            else:
                leverage_now = 10.0
            open_price = float(row.open)
            low_price = float(row.low)
            close_price = float(row.close)
            peak_close = max(peak_close, close_price)
            stop_price = max(stop_price, peak_close * (1.0 - float(stop_loss_pct) / 100.0))

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
                "capital": float(capital),
                "daily_return": capital / start_capital - 1.0 if start_capital > 0 else 0.0,
                "funding_cost": float(funding_cost),
            }
        )
        prev_allow = allow_now

    path = pd.DataFrame(rows)
    trades_df = pd.DataFrame(trades)
    total_return_pct = round((float(path.iloc[-1]["capital"]) / float(initial_capital) - 1.0) * 100.0, 2) if not path.empty else 0.0
    max_dd = round(max_drawdown_pct(path["capital"]), 2) if not path.empty else 0.0
    score = round(total_return_pct - max_dd * 2.0, 2)
    return {
        "entry_profile": entry_profile,
        "summary": {
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd,
            "score": score,
            "trades": int(len(trades_df)),
            "win_rate_pct": round(float((trades_df["trade_return_pct"] > 0).mean() * 100.0), 2) if not trades_df.empty else 0.0,
            "avg_trade_return_pct": round(float(trades_df["trade_return_pct"].mean()), 2) if not trades_df.empty else 0.0,
            "funding_cost_pct_est": round(float(path["funding_cost"].sum() * 100.0), 2) if not path.empty else 0.0,
            "invested_bars": int(path["holding"].sum()) if not path.empty else 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize 1h entry timing on aggressive QQQ/USDT structure.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--okx-4h", default=str(DEFAULT_OKX_4H))
    parser.add_argument("--okx-1h", default=str(DEFAULT_OKX_1H))
    parser.add_argument("--funding", default=str(DEFAULT_FUNDING))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--stop-loss-pct", type=float, default=3.5)
    parser.add_argument("--taker-fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    args = parser.parse_args()

    config, signal_path = load_signal_path(Path(args.config))
    bars4h = enrich_bars(attach_daily_state(load_okx_4h(Path(args.okx_4h)), signal_path))
    bars1h = load_okx_1h(Path(args.okx_1h))
    funding = load_funding(Path(args.funding))

    coverage_start = bars1h["date"].min()
    bars4h = bars4h[bars4h["date"] >= coverage_start].reset_index(drop=True)

    results = []
    for profile in ENTRY_PROFILES:
        results.append(
            simulate(
                bars4h,
                bars1h,
                funding,
                entry_profile=profile,
                stop_loss_pct=float(args.stop_loss_pct),
                taker_fee_rate=float(args.taker_fee_rate),
                slippage_bps=float(args.slippage_bps),
                initial_capital=float(config["initial_capital"]),
            )
        )

    by_score = sorted(results, key=lambda x: x["summary"]["score"], reverse=True)
    payload = {
        "config": {
            "signal_frozen_label": config.get("frozen_label"),
            "base_structure": "fixed10",
            "stop_loss_pct": float(args.stop_loss_pct),
            "coverage_start_1h": str(coverage_start),
            "coverage_end_4h": str(bars4h["date"].max()) if not bars4h.empty else None,
        },
        "results": results,
        "top_by_score": by_score,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(out)
    print(json.dumps(by_score, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
