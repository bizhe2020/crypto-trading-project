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

from scripts.tqqq_cash_strict_utils import (  # noqa: E402
    annual_returns,
    load_strict_config,
    load_strict_frame_with_overlay_context,
    max_drawdown_pct,
    run_strict_candidate,
)


DEFAULT_SIGNAL_CONFIG = ROOT / "config" / "config.paper.tqqq-only-strict-recovery-frozen.json"
DEFAULT_EXEC_CONFIG = ROOT / "config" / "config.paper.qqq-usdt-aggressive-frozen.json"
DEFAULT_DATA_ROOT = ROOT / "data" / "public" / "etf_long"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_daily_proxy_high_stop_replay_20260530.json"
DEFAULT_PATH_OUTPUT = ROOT / "var" / "reports" / "qqq_daily_proxy_high_stop_replay_20260530.csv"


STOP_MODES = {
    "close_same_bar": "current-code style: update trailing stop with same-day close before low check",
    "close_prev_strict": "strict: update trailing stop with previous completed daily close only",
    "high_prev_strict": "strict: update trailing stop with previous completed daily high only",
    "high_same_day_optimistic": "optimistic: assume same-day high occurs before same-day low",
}


def load_ohlcv(path: Path) -> pd.DataFrame:
    frame = pd.read_feather(path).copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    frame["session_day"] = frame["date"].dt.normalize()
    return frame


def build_signal_path(config: dict[str, Any], data_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = load_strict_frame_with_overlay_context(
        data_root=data_root,
        entry_fast_window=int(config["entry_fast_window"]),
        entry_slow_window=int(config["entry_slow_window"]),
    )
    result = run_strict_candidate(
        frame,
        regime_filter=str(config["regime_filter"]),
        max_hold_days=int(config["max_hold_days"]),
        trailing_lookback_days=int(config["trailing_lookback_days"]),
        trailing_drawdown_pct=float(config["trailing_drawdown_pct"]),
        switch_cost_bps=float(config["switch_cost_bps"]),
        initial_capital=float(config["initial_capital"]),
        de_risk_signal_name=str(config.get("de_risk_signal_name", "off")),
        recovery_reentry_rule=str(config.get("recovery_reentry_rule", "off")),
        recovery_reentry_cooldown_days=int(config.get("recovery_reentry_cooldown_days", 0)),
        drawdown_ladder_enabled=bool(config.get("drawdown_ladder_enabled", False)),
        drawdown_ladder_source=str(config.get("drawdown_ladder_source", "tqqq")),
        drawdown_ladder_threshold_pct=float(config.get("drawdown_ladder_threshold_pct", 0.0)),
        drawdown_ladder_peak_lookback_days=int(config.get("drawdown_ladder_peak_lookback_days", 90)),
        drawdown_ladder_scheme=str(config.get("drawdown_ladder_scheme", "two_equal")),
        drawdown_ladder_vix_rule=str(config.get("drawdown_ladder_vix_rule", "all")),
        drawdown_ladder_rebound_exit_pct=float(config.get("drawdown_ladder_rebound_exit_pct", 10.0)),
        drawdown_ladder_max_hold_days=int(config.get("drawdown_ladder_max_hold_days", 15)),
    )
    return result["path"].copy(), result["summary"]


def enrich_daily_execution_state(qqq: pd.DataFrame) -> pd.DataFrame:
    frame = qqq.copy()
    frame["ema20"] = frame["close"].ewm(span=20, adjust=False).mean()
    frame["ema50"] = frame["close"].ewm(span=50, adjust=False).mean()
    frame["prev_high_12"] = frame["high"].shift(1).rolling(12).max()
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


def annual_return_from_path(path: pd.DataFrame) -> dict[str, float]:
    if path.empty:
        return {}
    equity = path[["date", "capital"]].rename(columns={"capital": "equity"}).copy()
    return annual_returns(equity)


def buy_hold_return_pct(frame: pd.DataFrame, price_column: str = "close") -> float:
    if frame.empty:
        return 0.0
    start = float(frame.iloc[0][price_column])
    end = float(frame.iloc[-1][price_column])
    return round((end / start - 1.0) * 100.0, 2) if start > 0 else 0.0


def choose_leverage(row: Any, profile_name: str, profile: dict[str, float], holding: bool) -> float:
    if not holding:
        return 0.0
    if profile_name == "fixed10":
        return float(profile["base"])
    if bool(row.exec_high_growth):
        return float(profile["offense"])
    if bool(row.exec_defense_state):
        return float(profile["defense"])
    return float(profile["base"])


def stop_exit_price(open_price: float, stop_price: float) -> float:
    return open_price if open_price <= stop_price else stop_price


def finalize_trade(trades: list[dict[str, Any]], current_trade: dict[str, Any] | None, exit_date: pd.Timestamp, capital: float, reason: str) -> None:
    if current_trade is None:
        return
    entry_capital = float(current_trade["entry_capital"])
    trades.append(
        {
            "entry_date": current_trade["entry_date"],
            "exit_date": str(exit_date),
            "exit_reason": reason,
            "hold_days": int(current_trade["hold_days"]),
            "trade_return_pct": round((float(capital) / entry_capital - 1.0) * 100.0, 2) if entry_capital > 0 else 0.0,
        }
    )


def replay_daily_proxy(
    bars: pd.DataFrame,
    *,
    profile_name: str,
    profile: dict[str, float],
    stop_mode: str,
    stop_loss_pct: float,
    taker_fee_rate: float,
    slippage_bps: float,
    initial_capital: float,
    daily_funding_rate: float,
    rebalance_on_leverage_change: bool = True,
) -> dict[str, Any]:
    if stop_mode not in STOP_MODES:
        raise ValueError(f"Unsupported stop_mode: {stop_mode}")

    per_side_cost = float(taker_fee_rate) + float(slippage_bps) / 10000.0
    capital = float(initial_capital)
    holding = False
    prev_allow = False
    prev_close = 0.0
    last_leverage = 0.0
    stop_price = 0.0
    peak_value = 0.0
    current_trade: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    for row in bars.itertuples(index=False):
        start_capital = capital
        allow_now = bool(row.allow_long)
        entered = False
        exited = False
        stop_hit = False
        signal_exit = False
        fee_cost_pct = 0.0
        funding_cost_pct = 0.0
        gap_return = 0.0
        intraday_return = 0.0
        exit_reason = ""
        open_price = float(row.open)
        high_price = float(row.high)
        low_price = float(row.low)
        close_price = float(row.close)

        leverage_now = choose_leverage(row, profile_name, profile, holding)

        if holding and prev_close > 0 and open_price > 0:
            gap_return = open_price / prev_close - 1.0
            capital *= 1.0 + last_leverage * gap_return

        if holding and open_price <= stop_price:
            stop_hit = True
            capital *= 1.0 - per_side_cost * last_leverage
            fee_cost_pct += per_side_cost * last_leverage
            exited = True
            exit_reason = f"gap_stop_{stop_mode}"
            finalize_trade(trades, current_trade, pd.Timestamp(row.date), capital, exit_reason)
            holding = False
            current_trade = None
            stop_price = 0.0
            peak_value = 0.0
            leverage_now = 0.0

        if holding and not allow_now:
            exit_leverage = last_leverage if last_leverage > 0 else leverage_now
            capital *= 1.0 - per_side_cost * exit_leverage
            fee_cost_pct += per_side_cost * exit_leverage
            exited = True
            signal_exit = True
            exit_reason = "signal_off"
            finalize_trade(trades, current_trade, pd.Timestamp(row.date), capital, exit_reason)
            holding = False
            current_trade = None
            stop_price = 0.0
            peak_value = 0.0
            leverage_now = 0.0

        if allow_now and not holding and not prev_allow:
            leverage_now = float(profile["base"])
            capital *= 1.0 - per_side_cost * leverage_now
            fee_cost_pct += per_side_cost * leverage_now
            holding = True
            entered = True
            peak_value = open_price
            stop_price = open_price * (1.0 - float(stop_loss_pct) / 100.0)
            current_trade = {
                "entry_date": str(pd.Timestamp(row.date)),
                "entry_capital": float(capital),
                "hold_days": 0,
            }

        if holding:
            leverage_now = choose_leverage(row, profile_name, profile, holding)
            if current_trade is not None:
                current_trade["hold_days"] += 1
            if rebalance_on_leverage_change and last_leverage > 0 and abs(leverage_now - last_leverage) > 1e-9:
                rebalance_cost_pct = abs(leverage_now - last_leverage) * per_side_cost
                capital *= 1.0 - rebalance_cost_pct
                fee_cost_pct += rebalance_cost_pct

            if stop_mode == "close_same_bar":
                peak_value = max(peak_value, close_price)
                stop_price = max(stop_price, peak_value * (1.0 - float(stop_loss_pct) / 100.0))
                if low_price <= stop_price:
                    stop_hit = True
                    exit_price = stop_exit_price(open_price, stop_price)
                    intraday_return = exit_price / open_price - 1.0 if open_price > 0 else 0.0
                    capital *= 1.0 + leverage_now * intraday_return
                    capital *= 1.0 - per_side_cost * leverage_now
                    fee_cost_pct += per_side_cost * leverage_now
                    exited = True
                    exit_reason = "stop_close_same_bar"
                    finalize_trade(trades, current_trade, pd.Timestamp(row.date), capital, exit_reason)
                    holding = False
                    current_trade = None
                    stop_price = 0.0
                    peak_value = 0.0
                else:
                    intraday_return = close_price / open_price - 1.0 if open_price > 0 else 0.0
                    capital *= 1.0 + leverage_now * intraday_return

            elif stop_mode == "close_prev_strict":
                if low_price <= stop_price:
                    stop_hit = True
                    exit_price = stop_exit_price(open_price, stop_price)
                    intraday_return = exit_price / open_price - 1.0 if open_price > 0 else 0.0
                    capital *= 1.0 + leverage_now * intraday_return
                    capital *= 1.0 - per_side_cost * leverage_now
                    fee_cost_pct += per_side_cost * leverage_now
                    exited = True
                    exit_reason = "stop_close_prev_strict"
                    finalize_trade(trades, current_trade, pd.Timestamp(row.date), capital, exit_reason)
                    holding = False
                    current_trade = None
                    stop_price = 0.0
                    peak_value = 0.0
                else:
                    intraday_return = close_price / open_price - 1.0 if open_price > 0 else 0.0
                    capital *= 1.0 + leverage_now * intraday_return
                    peak_value = max(peak_value, close_price)
                    stop_price = max(stop_price, peak_value * (1.0 - float(stop_loss_pct) / 100.0))

            elif stop_mode == "high_prev_strict":
                if low_price <= stop_price:
                    stop_hit = True
                    exit_price = stop_exit_price(open_price, stop_price)
                    intraday_return = exit_price / open_price - 1.0 if open_price > 0 else 0.0
                    capital *= 1.0 + leverage_now * intraday_return
                    capital *= 1.0 - per_side_cost * leverage_now
                    fee_cost_pct += per_side_cost * leverage_now
                    exited = True
                    exit_reason = "stop_high_prev_strict"
                    finalize_trade(trades, current_trade, pd.Timestamp(row.date), capital, exit_reason)
                    holding = False
                    current_trade = None
                    stop_price = 0.0
                    peak_value = 0.0
                else:
                    intraday_return = close_price / open_price - 1.0 if open_price > 0 else 0.0
                    capital *= 1.0 + leverage_now * intraday_return
                    peak_value = max(peak_value, high_price)
                    stop_price = max(stop_price, peak_value * (1.0 - float(stop_loss_pct) / 100.0))

            elif stop_mode == "high_same_day_optimistic":
                if open_price <= stop_price:
                    stop_hit = True
                    exit_price = open_price
                else:
                    peak_value = max(peak_value, high_price)
                    stop_price = max(stop_price, peak_value * (1.0 - float(stop_loss_pct) / 100.0))
                    stop_hit = low_price <= stop_price
                    exit_price = stop_price
                if stop_hit:
                    intraday_return = exit_price / open_price - 1.0 if open_price > 0 else 0.0
                    capital *= 1.0 + leverage_now * intraday_return
                    capital *= 1.0 - per_side_cost * leverage_now
                    fee_cost_pct += per_side_cost * leverage_now
                    exited = True
                    exit_reason = "stop_high_same_day_optimistic"
                    finalize_trade(trades, current_trade, pd.Timestamp(row.date), capital, exit_reason)
                    holding = False
                    current_trade = None
                    stop_price = 0.0
                    peak_value = 0.0
                else:
                    intraday_return = close_price / open_price - 1.0 if open_price > 0 else 0.0
                    capital *= 1.0 + leverage_now * intraday_return

            if holding and daily_funding_rate:
                funding_cost_pct = max(float(daily_funding_rate), 0.0) * leverage_now
                capital *= 1.0 - funding_cost_pct

        rows.append(
            {
                "date": pd.Timestamp(row.date),
                "session_day": pd.Timestamp(row.session_day),
                "profile_name": profile_name,
                "stop_mode": stop_mode,
                "allow_long": allow_now,
                "holding": bool(holding),
                "entered": bool(entered),
                "exited": bool(exited),
                "stop_hit": bool(stop_hit),
                "signal_exit": bool(signal_exit),
                "exit_reason": exit_reason,
                "leverage": float(leverage_now if holding or entered else 0.0),
                "stop_price": float(stop_price),
                "capital": float(capital),
                "daily_return": float(capital / start_capital - 1.0 if start_capital > 0 else 0.0),
                "gap_return": float(gap_return),
                "intraday_return": float(intraday_return),
                "fee_cost_pct": float(fee_cost_pct),
                "funding_cost_pct": float(funding_cost_pct),
                "high_growth": bool(row.high_growth),
                "defense_state": bool(row.defense_state),
                "exec_high_growth": bool(row.exec_high_growth),
                "exec_defense_state": bool(row.exec_defense_state),
            }
        )
        prev_close = close_price
        last_leverage = float(leverage_now if holding else 0.0)
        prev_allow = allow_now

    path = pd.DataFrame(rows)
    trades_df = pd.DataFrame(trades)
    invested_days = int(path["holding"].sum()) if not path.empty else 0
    closed_trades = int(len(trades_df))
    win_rate = round(float((trades_df["trade_return_pct"] > 0).mean() * 100.0), 2) if closed_trades else 0.0
    avg_hold = round(float(trades_df["hold_days"].mean()), 2) if closed_trades else 0.0
    avg_trade = round(float(trades_df["trade_return_pct"].mean()), 2) if closed_trades else 0.0
    summary = {
        "profile_name": profile_name,
        "stop_mode": stop_mode,
        "stop_mode_description": STOP_MODES[stop_mode],
        "stop_loss_pct": float(stop_loss_pct),
        "total_return_pct": round((float(path.iloc[-1]["capital"]) / float(initial_capital) - 1.0) * 100.0, 2) if not path.empty else 0.0,
        "max_drawdown_pct": round(max_drawdown_pct(path["capital"]), 2) if not path.empty else 0.0,
        "yearly_returns_pct": annual_return_from_path(path),
        "bars": int(len(path)),
        "invested_days": invested_days,
        "invested_ratio_pct": round(invested_days / len(path) * 100.0, 2) if len(path) else 0.0,
        "closed_trades": closed_trades,
        "win_rate_pct": win_rate,
        "avg_hold_days": avg_hold,
        "avg_trade_return_pct": avg_trade,
        "stop_hits": int(path["stop_hit"].sum()) if not path.empty else 0,
        "signal_exits": int(path["signal_exit"].sum()) if not path.empty else 0,
        "total_fee_cost_pct_est": round(float(path["fee_cost_pct"].sum() * 100.0), 2) if not path.empty else 0.0,
        "total_funding_cost_pct_est": round(float(path["funding_cost_pct"].sum() * 100.0), 2) if not path.empty else 0.0,
        "latest_holding": bool(path.iloc[-1]["holding"]) if not path.empty else False,
        "start": str(path.iloc[0]["date"]) if not path.empty else None,
        "end": str(path.iloc[-1]["date"]) if not path.empty else None,
    }
    return {"summary": summary, "path": path, "trades": trades_df}


def merge_signal_with_qqq(signal_path: pd.DataFrame, qqq: pd.DataFrame) -> pd.DataFrame:
    signal = signal_path[["date", "position"]].copy()
    signal["session_day"] = pd.to_datetime(signal["date"], utc=True).dt.normalize()
    signal["allow_long"] = signal["position"].eq("TQQQ")
    bars = enrich_daily_execution_state(qqq)
    bars["exec_high_growth"] = bars["high_growth"].shift(1).astype("boolean").fillna(False).astype(bool)
    bars["exec_defense_state"] = bars["defense_state"].shift(1).astype("boolean").fillna(False).astype(bool)
    merged = bars.merge(signal[["session_day", "allow_long"]], on="session_day", how="inner")
    return merged.sort_values("date").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay frozen TQQQ signal on long-horizon QQQ daily proxy with high-based stop mocks.")
    parser.add_argument("--signal-config", default=str(DEFAULT_SIGNAL_CONFIG))
    parser.add_argument("--exec-config", default=str(DEFAULT_EXEC_CONFIG))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--path-output", default=str(DEFAULT_PATH_OUTPUT))
    parser.add_argument("--daily-funding-rate", type=float, default=0.0, help="Optional daily funding rate charged on notional, e.g. 0.0001.")
    parser.add_argument("--stop-loss-pct", type=float, default=None)
    parser.add_argument("--taker-fee-rate", type=float, default=None)
    parser.add_argument("--slippage-bps", type=float, default=None)
    parser.add_argument("--no-rebalance-cost", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal_config = load_strict_config(Path(args.signal_config))
    exec_config = json.loads(Path(args.exec_config).read_text())
    data_root = Path(args.data_root)

    signal_path, tqqq_signal_summary = build_signal_path(signal_config, data_root)
    qqq = load_ohlcv(data_root / "QQQ-1d.feather")
    tqqq = load_ohlcv(data_root / "TQQQ-1d.feather")
    merged = merge_signal_with_qqq(signal_path, qqq)
    stop_loss_pct = float(args.stop_loss_pct) if args.stop_loss_pct is not None else float(exec_config.get("stop_loss_pct", 3.5))
    taker_fee_rate = float(args.taker_fee_rate) if args.taker_fee_rate is not None else float(exec_config.get("taker_fee_rate", 0.0005))
    slippage_bps = float(args.slippage_bps) if args.slippage_bps is not None else float(exec_config.get("slippage_bps", 5.0))

    profiles = {
        "fixed10": {
            "base": 10.0,
            "offense": 10.0,
            "defense": 10.0,
        },
        "config_base_off_def": {
            "base": float(exec_config.get("base_leverage", 10.0)),
            "offense": float(exec_config.get("offense_leverage", 10.0)),
            "defense": float(exec_config.get("defense_leverage", 1.0)),
        },
    }
    summaries: list[dict[str, Any]] = []
    all_paths: list[pd.DataFrame] = []
    trade_summaries: dict[str, list[dict[str, Any]]] = {}
    for profile_name, profile in profiles.items():
        for stop_mode in STOP_MODES:
            result = replay_daily_proxy(
                merged,
                profile_name=profile_name,
                profile=profile,
                stop_mode=stop_mode,
                stop_loss_pct=stop_loss_pct,
                taker_fee_rate=taker_fee_rate,
                slippage_bps=slippage_bps,
                initial_capital=float(exec_config.get("initial_capital", signal_config.get("initial_capital", 1000.0))),
                daily_funding_rate=float(args.daily_funding_rate),
                rebalance_on_leverage_change=not bool(args.no_rebalance_cost),
            )
            summaries.append(result["summary"])
            all_paths.append(result["path"])
            key = f"{profile_name}:{stop_mode}"
            trade_summaries[key] = result["trades"].to_dict(orient="records")

    out_payload = {
        "metadata": {
            "signal_config": str(Path(args.signal_config)),
            "signal_frozen_label": signal_config.get("frozen_label"),
            "exec_config": str(Path(args.exec_config)),
            "exec_frozen_label": exec_config.get("frozen_label"),
            "data_root": str(data_root),
            "data_source": "Yahoo Finance chart API via scripts/fetch_public_etf_history.py",
            "important_limitations": [
                "QQQ ETF daily OHLC is used as a proxy for long-horizon QQQ/USDT execution because OKX contract history is short.",
                "Funding is not modeled unless --daily-funding-rate is provided.",
                "high_same_day_optimistic assumes the daily high occurs before the daily low and can overstate stop performance.",
                "config_base_off_def uses daily proxy high_growth/defense states; live QQQ/USDT uses lower timeframe execution state.",
            ],
        },
        "coverage": {
            "signal_path": {
                "rows": int(len(signal_path)),
                "start": str(signal_path["date"].min()) if not signal_path.empty else None,
                "end": str(signal_path["date"].max()) if not signal_path.empty else None,
            },
            "merged_execution": {
                "rows": int(len(merged)),
                "start": str(merged["date"].min()) if not merged.empty else None,
                "end": str(merged["date"].max()) if not merged.empty else None,
            },
            "qqq": {
                "rows": int(len(qqq)),
                "start": str(qqq["date"].min()) if not qqq.empty else None,
                "end": str(qqq["date"].max()) if not qqq.empty else None,
                "buy_hold_return_pct": buy_hold_return_pct(qqq),
            },
            "tqqq": {
                "rows": int(len(tqqq)),
                "start": str(tqqq["date"].min()) if not tqqq.empty else None,
                "end": str(tqqq["date"].max()) if not tqqq.empty else None,
                "buy_hold_return_pct": buy_hold_return_pct(tqqq),
            },
        },
        "config": {
            "stop_loss_pct": stop_loss_pct,
            "taker_fee_rate": taker_fee_rate,
            "slippage_bps": slippage_bps,
            "daily_funding_rate": float(args.daily_funding_rate),
            "rebalance_on_leverage_change": not bool(args.no_rebalance_cost),
            "profiles": profiles,
        },
        "tqqq_signal_source_summary": tqqq_signal_summary,
        "summaries": sorted(summaries, key=lambda item: float(item["total_return_pct"]), reverse=True),
        "trades": trade_summaries,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False))

    if all_paths:
        path_frame = pd.concat(all_paths, ignore_index=True)
        path_out = Path(args.path_output)
        path_out.parent.mkdir(parents=True, exist_ok=True)
        path_frame.to_csv(path_out, index=False)

    print(out)
    print(json.dumps({"coverage": out_payload["coverage"], "summaries": out_payload["summaries"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
