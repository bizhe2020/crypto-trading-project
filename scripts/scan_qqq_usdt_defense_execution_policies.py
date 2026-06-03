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
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_usdt_defense_execution_policy_scan.json"

RESTORE_RULES = ["immediate", "clear2", "clear3", "high_growth_or_clear3"]


def build_policies() -> dict[str, dict[str, Any]]:
    policies: dict[str, dict[str, Any]] = {}
    base_specs = {
        "fixed_def1": {"mode": "fixed", "defense": 1.0, "confirm_bars": 1},
        "fixed_def2": {"mode": "fixed", "defense": 2.0, "confirm_bars": 1},
        "confirm_def1_after2": {"mode": "fixed", "defense": 1.0, "confirm_bars": 2},
        "confirm_def1_after3": {"mode": "fixed", "defense": 1.0, "confirm_bars": 3},
        "confirm_def0_after2": {"mode": "fixed", "defense": 0.0, "confirm_bars": 2},
        "soft5_then1_after2": {"mode": "two_step", "soft": 5.0, "hard": 1.0, "hard_after_bars": 2},
        "soft5_then1_after3": {"mode": "two_step", "soft": 5.0, "hard": 1.0, "hard_after_bars": 3},
        "soft5_then1_after4": {"mode": "two_step", "soft": 5.0, "hard": 1.0, "hard_after_bars": 4},
        "soft5_then0_after3": {"mode": "two_step", "soft": 5.0, "hard": 0.0, "hard_after_bars": 3},
    }
    for base_name, spec in base_specs.items():
        for restore_rule in RESTORE_RULES:
            for cooldown_bars in (0, 1, 2, 3):
                name = f"{base_name}_restore_{restore_rule}_cool{cooldown_bars}"
                policies[name] = {**spec, "restore_rule": restore_rule, "cooldown_bars": cooldown_bars}
    return policies


POLICIES = build_policies()


def restore_ready(rule: str, row: Any, clear_streak: int) -> bool:
    if rule == "immediate":
        return True
    if rule == "clear2":
        return clear_streak >= 2
    if rule == "clear3":
        return clear_streak >= 3
    if rule == "high_growth_or_clear3":
        return bool(row.high_growth) or clear_streak >= 3
    raise ValueError(f"Unsupported restore rule: {rule}")


def desired_leverage(policy: dict[str, Any], row: Any, *, defense_streak: int, clear_streak: int, current_leverage: float) -> float:
    if bool(row.defense_state):
        mode = str(policy["mode"])
        if mode == "fixed":
            if defense_streak >= int(policy["confirm_bars"]):
                return float(policy["defense"])
            return current_leverage if current_leverage > 0 else 10.0
        if mode == "two_step":
            if defense_streak >= int(policy["hard_after_bars"]):
                return float(policy["hard"])
            return float(policy["soft"])
        raise ValueError(f"Unsupported policy mode: {mode}")
    if restore_ready(str(policy["restore_rule"]), row, clear_streak):
        return 10.0
    return current_leverage if current_leverage > 0 else 10.0


def simulate(
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    policy_name: str,
    rebalance_cost_multiplier: float,
    stop_loss_pct: float,
    taker_fee_rate: float,
    slippage_bps: float,
    initial_capital: float,
) -> dict[str, Any]:
    policy = POLICIES[policy_name]
    merged = pd.merge_asof(
        bars.sort_values("date"),
        funding.sort_values("date"),
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["funding_rate_value"] = merged["funding_rate_value"].fillna(0.0)

    capital = float(initial_capital)
    active_signal = False
    stopped_until_reset = False
    prev_allow = False
    current_leverage = 0.0
    bars_since_rebalance = 1000000
    defense_streak = 0
    clear_streak = 0
    entry_price = 0.0
    stop_price = 0.0
    peak_close = 0.0
    current_trade: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    per_side_cost = float(taker_fee_rate) + float(slippage_bps) / 10000.0

    for row in merged.itertuples(index=False):
        start_capital = capital
        allow_now = bool(row.allow_long)
        entered = False
        exited = False
        stop_hit = False
        funding_cost = 0.0
        entry_exit_cost = 0.0
        rebalance_cost = 0.0
        target_leverage = 0.0

        if not allow_now:
            defense_streak = 0
            clear_streak = 0
            stopped_until_reset = False
            if active_signal:
                entry_exit_cost = per_side_cost * current_leverage
                capital *= 1.0 - entry_exit_cost
                active_signal = False
                exited = True
                if current_trade is not None:
                    trades.append(
                        {
                            "entry_date": current_trade["entry_date"],
                            "exit_date": str(pd.Timestamp(row.date)),
                            "trade_return_pct": round((capital / float(current_trade["entry_capital"]) - 1.0) * 100.0, 2),
                        }
                    )
                current_trade = None
                current_leverage = 0.0

        if allow_now and not prev_allow:
            active_signal = True
            stopped_until_reset = False
            entered = True
            bars_since_rebalance = 1000000
            entry_price = float(row.open)
            stop_price = entry_price * (1.0 - float(stop_loss_pct) / 100.0)
            peak_close = float(row.open)
            current_trade = {"entry_date": str(pd.Timestamp(row.date)), "entry_capital": capital}

        if active_signal and allow_now and not stopped_until_reset:
            if bool(row.defense_state):
                defense_streak += 1
                clear_streak = 0
            else:
                clear_streak += 1
                defense_streak = 0

            target_leverage = desired_leverage(
                policy,
                row,
                defense_streak=defense_streak,
                clear_streak=clear_streak,
                current_leverage=current_leverage,
            )
            gap = abs(target_leverage - current_leverage)
            if gap > 1e-9:
                can_rebalance = entered or bars_since_rebalance >= int(policy["cooldown_bars"])
                if can_rebalance:
                    if entered or current_leverage <= 0:
                        entry_exit_cost += per_side_cost * gap
                        capital *= 1.0 - per_side_cost * gap
                    else:
                        rebalance_cost = per_side_cost * gap * float(rebalance_cost_multiplier)
                        capital *= 1.0 - rebalance_cost
                    current_leverage = target_leverage
                    bars_since_rebalance = 0
            else:
                bars_since_rebalance += 1

            if current_leverage > 0:
                open_price = float(row.open)
                low_price = float(row.low)
                close_price = float(row.close)
                peak_close = max(peak_close, close_price)
                stop_price = max(stop_price, peak_close * (1.0 - float(stop_loss_pct) / 100.0))
                if low_price <= stop_price:
                    stop_hit = True
                    exit_price = stop_price
                    bar_ret = exit_price / open_price - 1.0 if open_price > 0 else 0.0
                    capital *= 1.0 + current_leverage * bar_ret
                    entry_exit_cost += per_side_cost * current_leverage
                    capital *= 1.0 - per_side_cost * current_leverage
                    current_leverage = 0.0
                    active_signal = False
                    stopped_until_reset = True
                    exited = True
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
                    capital *= 1.0 + current_leverage * bar_ret
                    funding_cost = max(float(row.funding_rate_value), 0.0) * current_leverage
                    capital *= 1.0 - funding_cost

        rows.append(
            {
                "date": pd.Timestamp(row.date),
                "allow_long": allow_now,
                "active_signal": active_signal,
                "entered": entered,
                "exited": exited,
                "stop_hit": stop_hit,
                "defense_state": bool(row.defense_state),
                "high_growth": bool(row.high_growth),
                "defense_streak": int(defense_streak),
                "clear_streak": int(clear_streak),
                "target_leverage": float(target_leverage),
                "leverage_now": float(current_leverage),
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
        "policy": policy_name,
        "policy_spec": policy,
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
    parser = argparse.ArgumentParser(description="Scan execution-aware QQQ/USDT defense rebalance policies.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--rebalance-cost-multipliers", default="0,0.25,0.5,1")
    parser.add_argument("--taker-fee-rate", type=float, default=None)
    parser.add_argument("--slippage-bps", type=float, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    taker_fee_rate = float(config["taker_fee_rate"] if args.taker_fee_rate is None else args.taker_fee_rate)
    slippage_bps = float(config["slippage_bps"] if args.slippage_bps is None else args.slippage_bps)
    _, signal_path = load_signal_path(ROOT / str(config["signal_source"]))
    bars = enrich_bars(attach_daily_state(load_okx_4h(ROOT / str(config["data_4h"])), signal_path))
    funding = load_funding(ROOT / str(config["funding_history_path"]))

    results: list[dict[str, Any]] = []
    for cost_multiplier in parse_float_list(str(args.rebalance_cost_multipliers)):
        for policy_name in POLICIES:
            results.append(
                simulate(
                    bars,
                    funding,
                    policy_name=policy_name,
                    rebalance_cost_multiplier=cost_multiplier,
                    stop_loss_pct=float(config["stop_loss_pct"]),
                    taker_fee_rate=taker_fee_rate,
                    slippage_bps=slippage_bps,
                    initial_capital=float(config["initial_capital"]),
                )
            )

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        groups.setdefault(str(item["rebalance_cost_multiplier"]), []).append(item)

    ranked = {}
    for cost, items in groups.items():
        ranked[cost] = {
            "top_by_return": sorted(
                items,
                key=lambda item: (
                    item["summary"]["total_return_pct"],
                    -item["summary"]["max_drawdown_pct"],
                    -item["summary"]["rebalance_cost_pct_est"],
                ),
                reverse=True,
            ),
            "top_by_score": sorted(items, key=lambda item: item["summary"]["score"], reverse=True),
        }

    payload = {
        "config": {
            "source_config": str(config_path),
            "frozen_label": config.get("frozen_label"),
            "stop_loss_pct": float(config["stop_loss_pct"]),
            "taker_fee_rate": taker_fee_rate,
            "slippage_bps": slippage_bps,
            "policies": POLICIES,
            "coverage": {
                "bars": int(len(bars)),
                "start": str(bars["date"].min()) if not bars.empty else None,
                "end": str(bars["date"].max()) if not bars.empty else None,
                "allow_long_bars": int(bars["allow_long"].sum()) if not bars.empty else 0,
                "allow_and_defense_bars": int((bars["allow_long"] & bars["defense_state"]).sum()) if not bars.empty else 0,
                "high_growth_bars": int(bars["high_growth"].sum()) if not bars.empty else 0,
            },
        },
        "ranked": ranked,
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    compact = {
        cost: {
            "top_by_return": value["top_by_return"][:10],
            "top_by_score": value["top_by_score"][:10],
        }
        for cost, value in ranked.items()
    }
    print(out)
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
