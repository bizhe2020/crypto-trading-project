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

from scripts.scan_qqq_usdt_4h_triggers import attach_daily_state, load_okx_4h, load_signal_path  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "config.paper.tqqq-only-strict-recovery-frozen.json"
DEFAULT_OKX_4H = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-4h-futures.feather"
DEFAULT_FUNDING = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-8h-funding_rate.feather"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_usdt_10x_replay.json"


def load_funding(path: Path) -> pd.DataFrame:
    df = pd.read_feather(path).copy()
    if "date" not in df.columns:
        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            df["date"] = pd.to_datetime(df["datetime"], utc=True)
    else:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    rate_col = "fundingRate" if "fundingRate" in df.columns else "funding_rate"
    df["funding_rate_value"] = pd.to_numeric(df[rate_col], errors="coerce").fillna(0.0)
    df["funding_event_time"] = df["date"]
    return df[["date", "funding_event_time", "funding_rate_value"]].sort_values("date").reset_index(drop=True)


def is_funding_settlement_bar(bar_date: Any, funding_event_time: Any) -> bool:
    if pd.isna(funding_event_time):
        return False
    bar_ts = pd.Timestamp(bar_date)
    event_ts = pd.Timestamp(funding_event_time)
    bar_ts = bar_ts.tz_localize("UTC") if bar_ts.tzinfo is None else bar_ts.tz_convert("UTC")
    event_ts = event_ts.tz_localize("UTC") if event_ts.tzinfo is None else event_ts.tz_convert("UTC")
    return bar_ts == event_ts


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0, pd.NA) * 100.0
    return float(dd.max(skipna=True) or 0.0)


def run_10x_replay(
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    leverage: float,
    taker_fee_rate: float,
    slippage_bps: float,
    stop_loss_pct: float,
    initial_capital: float,
) -> dict[str, Any]:
    merged = pd.merge_asof(
        bars.sort_values("date"),
        funding.sort_values("date"),
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["funding_rate_value"] = merged["funding_rate_value"].fillna(0.0)
    merged["funding_event_time"] = merged["funding_event_time"].where(merged["funding_event_time"].notna(), pd.NaT)

    capital = float(initial_capital)
    holding = False
    stop_price = 0.0
    entry_price = 0.0
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    current_trade: dict[str, Any] | None = None
    prev_allow = False

    per_side_cost = float(taker_fee_rate) + float(slippage_bps) / 10000.0

    for row in merged.itertuples(index=False):
        start_capital = capital
        allow_now = bool(row.allow_long)
        entered_today = False
        exited_today = False
        fee_cost = 0.0
        funding_cost = 0.0
        funding_settled = False
        stop_hit = False

        if holding and not allow_now:
            fee_cost = per_side_cost
            capital *= 1.0 - fee_cost * leverage
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
            capital *= 1.0 - fee_cost * leverage
            holding = True
            entered_today = True
            entry_price = float(row.open)
            stop_price = entry_price * (1.0 - float(stop_loss_pct) / 100.0)
            current_trade = {"entry_date": str(pd.Timestamp(row.date)), "entry_capital": capital}

        if holding:
            open_price = float(row.open)
            high_price = float(row.high)
            low_price = float(row.low)
            close_price = float(row.close)
            if low_price <= stop_price:
                stop_hit = True
                exit_price = stop_price
                bar_ret = exit_price / open_price - 1.0 if open_price > 0 else 0.0
                capital *= 1.0 + leverage * bar_ret
                capital *= 1.0 - per_side_cost * leverage
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
                stop_price = 0.0
                entry_price = 0.0
            else:
                bar_ret = close_price / open_price - 1.0 if open_price > 0 else 0.0
                capital *= 1.0 + leverage * bar_ret
                funding_settled = is_funding_settlement_bar(row.date, row.funding_event_time)
                if funding_settled:
                    funding_cost = float(row.funding_rate_value) * leverage
                    capital *= 1.0 - funding_cost
                stop_price = max(stop_price, close_price * (1.0 - float(stop_loss_pct) / 100.0))

        rows.append(
            {
                "date": pd.Timestamp(row.date),
                "holding": holding,
                "allow_long": allow_now,
                "entered_today": entered_today,
                "exited_today": exited_today,
                "stop_hit": stop_hit,
                "capital": float(capital),
                "daily_return": capital / start_capital - 1.0 if start_capital > 0 else 0.0,
                "funding_rate_value": float(row.funding_rate_value),
                "funding_settled": bool(funding_settled),
                "fee_cost": float(fee_cost),
                "funding_cost": float(funding_cost),
            }
        )
        prev_allow = allow_now

    path = pd.DataFrame(rows)
    trades_df = pd.DataFrame(trades)
    payload = {
        "summary": {
            "total_return_pct": round((float(path.iloc[-1]['capital']) / float(initial_capital) - 1.0) * 100.0, 2) if not path.empty else 0.0,
            "max_drawdown_pct": round(max_drawdown_pct(path["capital"]), 2) if not path.empty else 0.0,
            "bars": int(len(path)),
            "invested_bars": int(path["holding"].sum()) if not path.empty else 0,
            "trades": int(len(trades_df)),
            "win_rate_pct": round(float((trades_df["trade_return_pct"] > 0).mean() * 100.0), 2) if not trades_df.empty else 0.0,
            "avg_trade_return_pct": round(float(trades_df["trade_return_pct"].mean()), 2) if not trades_df.empty else 0.0,
            "total_funding_cost_pct_est": round(float(path["funding_cost"].sum() * 100.0), 2) if not path.empty else 0.0,
            "positive_funding_cost_pct_est": round(float(path.loc[path["funding_cost"] > 0, "funding_cost"].sum() * 100.0), 2) if not path.empty else 0.0,
            "negative_funding_credit_pct_est": round(float(-path.loc[path["funding_cost"] < 0, "funding_cost"].sum() * 100.0), 2) if not path.empty else 0.0,
            "funding_settlement_events": int(path["funding_settled"].sum()) if not path.empty else 0,
            "start": str(path.iloc[0]["date"]) if not path.empty else None,
            "end": str(path.iloc[-1]["date"]) if not path.empty else None,
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay ETF-derived QQQ signals on QQQ/USDT 4h with 10x fee/funding/stop modeling.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--okx-4h", default=str(DEFAULT_OKX_4H))
    parser.add_argument("--funding", default=str(DEFAULT_FUNDING))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--taker-fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--stop-loss-pct", type=float, default=2.0)
    args = parser.parse_args()

    config, signal_path = load_signal_path(Path(args.config))
    bars = attach_daily_state(load_okx_4h(Path(args.okx_4h)), signal_path)
    funding = load_funding(Path(args.funding))
    result = run_10x_replay(
        bars,
        funding,
        leverage=float(args.leverage),
        taker_fee_rate=float(args.taker_fee_rate),
        slippage_bps=float(args.slippage_bps),
        stop_loss_pct=float(args.stop_loss_pct),
        initial_capital=float(config["initial_capital"]),
    )
    payload = {
        "config": {
            "signal_frozen_label": config.get("frozen_label"),
            "leverage": float(args.leverage),
            "taker_fee_rate": float(args.taker_fee_rate),
            "slippage_bps": float(args.slippage_bps),
            "stop_loss_pct": float(args.stop_loss_pct),
        },
        **result,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(out)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
