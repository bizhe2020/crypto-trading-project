#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.client_email_utils import load_json, write_json


@dataclass
class ClientPaperState:
    anchor_trade_day: str
    principal: float
    anchor_asset: str = ""
    anchor_fill_price: float = 0.0
    anchor_units: int = 0


def load_or_init_state(state_path: Path, anchor_trade_day: date, principal: float, *, persist: bool = True) -> ClientPaperState:
    payload = load_json(state_path)
    stored_anchor = str(payload.get("anchor_trade_day") or "").strip()
    stored_principal = float(payload.get("principal") or 0.0)
    if stored_anchor and abs(stored_principal - principal) < 1e-9:
        return ClientPaperState(
            anchor_trade_day=stored_anchor,
            principal=stored_principal,
            anchor_asset=str(payload.get("anchor_asset") or "").strip(),
            anchor_fill_price=float(payload.get("anchor_fill_price") or 0.0),
            anchor_units=int(payload.get("anchor_units") or 0),
        )
    state = ClientPaperState(anchor_trade_day=str(anchor_trade_day), principal=float(principal))
    if persist:
        write_json(
            state_path,
            {
                "anchor_trade_day": state.anchor_trade_day,
                "principal": state.principal,
                "anchor_asset": state.anchor_asset,
                "anchor_fill_price": state.anchor_fill_price,
                "anchor_units": state.anchor_units,
            },
        )
    return state


def build_client_paper_curve(
    *,
    trade_days: pd.DataFrame,
    price_by_asset: dict[str, pd.DataFrame],
    targets_by_day: dict[str, dict[str, Any]],
    anchor_trade_day: str,
    principal: float,
    anchor_fill_override: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if trade_days.empty:
        return pd.DataFrame(columns=["trade_day", "equity_close", "asset", "leverage", "daily_return"])

    anchor_day = pd.Timestamp(anchor_trade_day).date()
    day_strings = [str(pd.Timestamp(item).date()) for item in trade_days["trade_day"]]
    if anchor_trade_day not in day_strings:
        return pd.DataFrame(columns=["trade_day", "equity_close", "asset", "leverage", "daily_return"])

    price_lookup: dict[str, dict[str, dict[str, float]]] = {}
    for asset, frame in price_by_asset.items():
        lookup: dict[str, dict[str, float]] = {}
        for _, row in frame.iterrows():
            lookup[str(pd.Timestamp(row["trade_day"]).date())] = {
                "open": float(row["open"]),
                "close": float(row["close"]),
            }
        price_lookup[asset] = lookup

    equity = float(principal)
    previous_asset = "CASH"
    previous_leverage = 0.0
    previous_closes: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    first_anchor_applied = False
    anchor_fill_override = dict(anchor_fill_override or {})
    override_asset = str(anchor_fill_override.get("asset") or "").strip()
    override_fill_price = float(anchor_fill_override.get("fill_price") or 0.0)
    override_units = int(anchor_fill_override.get("units") or 0)

    for _, row in trade_days.iterrows():
        trade_day = pd.Timestamp(row["trade_day"]).date()
        if trade_day < anchor_day:
            for asset, lookup in price_lookup.items():
                today = lookup.get(str(trade_day))
                if today:
                    previous_closes[asset] = float(today["close"])
            continue

        key = str(trade_day)
        target = dict(targets_by_day.get(key) or {"asset": "CASH", "leverage": 0.0, "signal_source_day": ""})
        target_asset = str(target.get("asset") or "CASH")
        target_leverage = float(target.get("leverage") or 0.0)

        if (
            not first_anchor_applied
            and trade_day == anchor_day
            and target_asset != "CASH"
            and override_asset == target_asset
            and override_fill_price > 0
        ):
            today_target = price_lookup.get(target_asset, {}).get(key)
            if not today_target:
                continue
            if override_units > 0:
                invested_notional = float(override_units) * override_fill_price
                cash_buffer = principal - invested_notional
                equity_close = cash_buffer + float(override_units) * float(today_target["close"])
            else:
                equity_close = equity * (float(today_target["close"]) / override_fill_price)
            daily_return = equity_close / equity - 1.0 if equity > 0 else 0.0
            equity = equity_close
            rows.append(
                {
                    "trade_day": pd.Timestamp(trade_day),
                    "equity_close": float(equity),
                    "asset": target_asset,
                    "leverage": float(target_leverage),
                    "daily_return": float(daily_return),
                    "signal_source_day": str(target.get("signal_source_day") or ""),
                }
            )
            for asset, lookup in price_lookup.items():
                today = lookup.get(key)
                if today:
                    previous_closes[asset] = float(today["close"])
            previous_asset = target_asset
            previous_leverage = target_leverage
            first_anchor_applied = True
            continue

        equity_open = equity
        if previous_asset != "CASH":
            today_prev_asset = price_lookup.get(previous_asset, {}).get(key)
            previous_close = previous_closes.get(previous_asset)
            if today_prev_asset and previous_close and previous_close > 0:
                equity_open = equity * (1.0 + previous_leverage * (float(today_prev_asset["open"]) / previous_close - 1.0))

        equity_close = equity_open
        if target_asset != "CASH":
            today_target = price_lookup.get(target_asset, {}).get(key)
            if today_target and float(today_target["open"]) > 0:
                equity_close = equity_open * (
                    1.0 + target_leverage * (float(today_target["close"]) / float(today_target["open"]) - 1.0)
                )

        daily_return = equity_close / equity - 1.0 if equity > 0 else 0.0
        equity = equity_close
        rows.append(
            {
                "trade_day": pd.Timestamp(trade_day),
                "equity_close": float(equity),
                "asset": target_asset,
                "leverage": float(target_leverage),
                "daily_return": float(daily_return),
                "signal_source_day": str(target.get("signal_source_day") or ""),
            }
        )

        for asset, lookup in price_lookup.items():
            today = lookup.get(key)
            if today:
                previous_closes[asset] = float(today["close"])
        previous_asset = target_asset
        previous_leverage = target_leverage

    return pd.DataFrame(rows)
