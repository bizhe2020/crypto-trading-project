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


DEFAULT_CONFIG = ROOT / "config" / "config.paper.qqq-usdt-aggressive-frozen.json"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_usdt_defense_leverage_profile_scan.json"


DEFENSE_PROFILES: dict[str, dict[str, Any]] = {
    "defense0_flat": {"mode": "fixed", "defense": 0.0},
    "defense1_hard": {"mode": "fixed", "defense": 1.0},
    "defense2_current": {"mode": "fixed", "defense": 2.0},
    "defense3": {"mode": "fixed", "defense": 3.0},
    "defense5_half": {"mode": "fixed", "defense": 5.0},
    "defense8_soft": {"mode": "fixed", "defense": 8.0},
    "two_step_5_then_1_after_2": {"mode": "two_step", "first": 5.0, "then": 1.0, "after_bars": 2},
    "two_step_5_then_1_after_3": {"mode": "two_step", "first": 5.0, "then": 1.0, "after_bars": 3},
    "two_step_5_then_0_after_2": {"mode": "two_step", "first": 5.0, "then": 0.0, "after_bars": 2},
}


def target_leverage(profile: dict[str, Any], row: Any, defense_streak: int) -> float:
    if bool(row.high_growth) and not bool(row.defense_state):
        return 10.0
    if not bool(row.defense_state):
        return 10.0
    if str(profile["mode"]) == "fixed":
        return float(profile["defense"])
    if str(profile["mode"]) == "two_step":
        if defense_streak >= int(profile["after_bars"]):
            return float(profile["then"])
        return float(profile["first"])
    raise ValueError(f"Unsupported defense profile: {profile}")


def simulate(
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    profile_name: str,
    rebalance_cost_multiplier: float,
    stop_loss_pct: float,
    taker_fee_rate: float,
    slippage_bps: float,
    initial_capital: float,
) -> dict[str, Any]:
    profile = DEFENSE_PROFILES[profile_name]
    merged = pd.merge_asof(
        bars.sort_values("date"),
        funding.sort_values("date"),
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["funding_rate_value"] = merged["funding_rate_value"].fillna(0.0)

    capital = float(initial_capital)
    signal_active = False
    stopped_until_signal_reset = False
    prev_allow = False
    prev_leverage = 0.0
    defense_streak = 0
    entry_price = 0.0
    stop_price = 0.0
    peak_close = 0.0
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    current_trade: dict[str, Any] | None = None
    per_side_cost = float(taker_fee_rate) + float(slippage_bps) / 10000.0

    for row in merged.itertuples(index=False):
        start_capital = capital
        allow_now = bool(row.allow_long)
        entered_signal = False
        exited_signal = False
        stop_hit = False
        funding_cost = 0.0
        entry_exit_cost = 0.0
        rebalance_cost = 0.0
        leverage_now = 0.0

        if not allow_now:
            defense_streak = 0
            stopped_until_signal_reset = False
            if signal_active:
                entry_exit_cost = per_side_cost * prev_leverage
                capital *= 1.0 - entry_exit_cost
                exited_signal = True
                signal_active = False
                if current_trade is not None:
                    trades.append(
                        {
                            "entry_date": current_trade["entry_date"],
                            "exit_date": str(pd.Timestamp(row.date)),
                            "trade_return_pct": round((capital / float(current_trade["entry_capital"]) - 1.0) * 100.0, 2),
                        }
                    )
                current_trade = None
                prev_leverage = 0.0

        if allow_now and not prev_allow:
            signal_active = True
            stopped_until_signal_reset = False
            entered_signal = True
            entry_price = float(row.open)
            stop_price = entry_price * (1.0 - float(stop_loss_pct) / 100.0)
            peak_close = float(row.open)
            current_trade = {"entry_date": str(pd.Timestamp(row.date)), "entry_capital": capital}

        if signal_active and allow_now and not stopped_until_signal_reset:
            defense_streak = defense_streak + 1 if bool(row.defense_state) else 0
            leverage_now = target_leverage(profile, row, defense_streak)
            leverage_gap = abs(leverage_now - prev_leverage)
            if leverage_gap > 1e-9:
                if entered_signal:
                    entry_exit_cost += per_side_cost * leverage_gap
                    capital *= 1.0 - per_side_cost * leverage_gap
                else:
                    rebalance_cost = per_side_cost * leverage_gap * float(rebalance_cost_multiplier)
                    capital *= 1.0 - rebalance_cost

            if leverage_now > 0:
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
                    entry_exit_cost += per_side_cost * leverage_now
                    capital *= 1.0 - per_side_cost * leverage_now
                    stopped_until_signal_reset = True
                    signal_active = False
                    exited_signal = True
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
                else:
                    bar_ret = close_price / open_price - 1.0 if open_price > 0 else 0.0
                    capital *= 1.0 + leverage_now * bar_ret
                    funding_cost = max(float(row.funding_rate_value), 0.0) * leverage_now
                    capital *= 1.0 - funding_cost
            prev_leverage = leverage_now

        rows.append(
            {
                "date": pd.Timestamp(row.date),
                "allow_long": allow_now,
                "signal_active": signal_active,
                "entered_signal": entered_signal,
                "exited_signal": exited_signal,
                "stop_hit": stop_hit,
                "high_growth": bool(row.high_growth),
                "defense_state": bool(row.defense_state),
                "defense_streak": int(defense_streak),
                "leverage_now": float(leverage_now),
                "capital": float(capital),
                "bar_return": capital / start_capital - 1.0 if start_capital > 0 else 0.0,
                "funding_cost": float(funding_cost),
                "entry_exit_cost": float(entry_exit_cost),
                "rebalance_cost": float(rebalance_cost),
            }
        )
        prev_allow = allow_now

    path = pd.DataFrame(rows)
    trades_df = pd.DataFrame(trades)
    total_return_pct = round((float(path.iloc[-1]["capital"]) / float(initial_capital) - 1.0) * 100.0, 2) if not path.empty else 0.0
    max_dd = round(max_drawdown_pct(path["capital"]), 2) if not path.empty else 0.0
    in_path = path.loc[path["leverage_now"] > 0]
    return {
        "profile": profile_name,
        "profile_spec": profile,
        "rebalance_cost_multiplier": float(rebalance_cost_multiplier),
        "summary": {
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd,
            "score": round(total_return_pct - max_dd * 2.0, 2),
            "trades": int(len(trades_df)),
            "win_rate_pct": round(float((trades_df["trade_return_pct"] > 0).mean() * 100.0), 2) if not trades_df.empty else 0.0,
            "avg_trade_return_pct": round(float(trades_df["trade_return_pct"].mean()), 2) if not trades_df.empty else 0.0,
            "invested_bars": int((path["leverage_now"] > 0).sum()) if not path.empty else 0,
            "zero_exposure_bars": int(((path["allow_long"]) & (path["leverage_now"] <= 0)).sum()) if not path.empty else 0,
            "defense_exposure_bars": int(((path["defense_state"]) & (path["leverage_now"] > 0)).sum()) if not path.empty else 0,
            "stop_hits": int(path["stop_hit"].sum()) if not path.empty else 0,
            "funding_cost_pct_est": round(float(path["funding_cost"].sum() * 100.0), 2) if not path.empty else 0.0,
            "entry_exit_cost_pct_est": round(float(path["entry_exit_cost"].sum() * 100.0), 2) if not path.empty else 0.0,
            "rebalance_cost_pct_est": round(float(path["rebalance_cost"].sum() * 100.0), 2) if not path.empty else 0.0,
            "avg_leverage_when_in": round(float(in_path["leverage_now"].mean()), 2) if not in_path.empty else 0.0,
            "max_leverage_when_in": round(float(in_path["leverage_now"].max()), 2) if not in_path.empty else 0.0,
            "leverage_changes": int((path["leverage_now"].diff().abs() > 1e-9).sum()) if not path.empty else 0,
        },
    }


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan QQQ/USDT defense exposure profiles with optional rebalance costs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--rebalance-cost-multipliers", default="0,1")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    _, signal_path = load_signal_path(ROOT / str(config["signal_source"]))
    bars = enrich_bars(attach_daily_state(load_okx_4h(ROOT / str(config["data_4h"])), signal_path))
    funding = load_funding(ROOT / str(config["funding_history_path"]))

    results: list[dict[str, Any]] = []
    for cost_multiplier in parse_float_list(str(args.rebalance_cost_multipliers)):
        for profile_name in DEFENSE_PROFILES:
            results.append(
                simulate(
                    bars,
                    funding,
                    profile_name=profile_name,
                    rebalance_cost_multiplier=cost_multiplier,
                    stop_loss_pct=float(config["stop_loss_pct"]),
                    taker_fee_rate=float(config["taker_fee_rate"]),
                    slippage_bps=float(config["slippage_bps"]),
                    initial_capital=float(config["initial_capital"]),
                )
            )

    def by_return_key(item: dict[str, Any]) -> tuple[float, float, float]:
        summary = item["summary"]
        return (
            float(summary["total_return_pct"]),
            -float(summary["max_drawdown_pct"]),
            -float(summary["rebalance_cost_pct_est"]),
        )

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        groups.setdefault(str(item["rebalance_cost_multiplier"]), []).append(item)

    ranked = {
        cost: {
            "top_by_return": sorted(items, key=by_return_key, reverse=True),
            "top_by_score": sorted(items, key=lambda item: item["summary"]["score"], reverse=True),
        }
        for cost, items in groups.items()
    }

    payload = {
        "config": {
            "source_config": str(config_path),
            "frozen_label": config.get("frozen_label"),
            "stop_loss_pct": float(config["stop_loss_pct"]),
            "taker_fee_rate": float(config["taker_fee_rate"]),
            "slippage_bps": float(config["slippage_bps"]),
            "profiles": DEFENSE_PROFILES,
            "coverage": {
                "bars": int(len(bars)),
                "start": str(bars["date"].min()) if not bars.empty else None,
                "end": str(bars["date"].max()) if not bars.empty else None,
                "allow_long_bars": int(bars["allow_long"].sum()) if not bars.empty else 0,
                "high_growth_bars": int(bars["high_growth"].sum()) if not bars.empty else 0,
                "defense_bars": int(bars["defense_state"].sum()) if not bars.empty else 0,
                "allow_and_defense_bars": int((bars["allow_long"] & bars["defense_state"]).sum()) if not bars.empty else 0,
            },
        },
        "ranked": ranked,
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(out)
    compact = {
        cost: {
            "top_by_return": value["top_by_return"][:8],
            "top_by_score": value["top_by_score"][:8],
        }
        for cost, value in ranked.items()
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
