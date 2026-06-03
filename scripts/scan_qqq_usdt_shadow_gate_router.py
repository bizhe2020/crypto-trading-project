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

from scripts.replay_proxy_strategy_router import (  # noqa: E402
    DEFAULT_BTC_FROZEN,
    DEFAULT_QQQ_USDT_CONFIG,
    build_btc_path_from_frozen_artifact,
    equity_from_returns,
    max_drawdown_pct,
    run_router,
)
from scripts.replay_qqq_usdt_10x import is_funding_settlement_bar, load_funding  # noqa: E402
from scripts.scan_qqq_usdt_4h_triggers import attach_daily_state, load_okx_4h, load_signal_path  # noqa: E402
from scripts.scan_qqq_usdt_btc_logic_overlays import enrich_bars  # noqa: E402


DEFAULT_NQ_4H = ROOT / "data" / "proxy" / "qqq_usdt_nq_continuous" / "QQQ_USDT_USDT-4h-futures-nq-continuous-dailyproxy-long.feather"
DEFAULT_NQ_FUNDING = ROOT / "data" / "proxy" / "qqq_usdt_nq_continuous" / "QQQ_USDT_USDT-8h-funding_rate-zero-nq-continuous-scaled-long.feather"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_usdt_shadow_gate_router_scan_20220101_20260529.json"


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [str(item.strip()) for item in str(raw).split(",") if item.strip()]


def summarize_equity(equity: pd.Series) -> dict[str, Any]:
    if equity.empty:
        return {"total_return_pct": 0.0, "max_drawdown_pct": 0.0, "daily_cvar5_pct": 0.0, "calmar_like": None}
    equity = pd.to_numeric(equity, errors="coerce").ffill()
    returns = equity.pct_change().fillna(0.0)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if float(equity.iloc[0]) > 0 else 0.0
    dd = max_drawdown_pct(equity)
    cvar_count = max(1, int(len(returns) * 0.05))
    cvar5 = float(returns.nsmallest(cvar_count).mean() * 100.0) if len(returns) else 0.0
    return {
        "total_return_pct": round(total * 100.0, 2),
        "max_drawdown_pct": round(dd, 2),
        "daily_cvar5_pct": round(cvar5, 4),
        "calmar_like": round((total * 100.0) / dd, 4) if dd > 0 else None,
    }


def reentry_ready(
    *,
    rule: str,
    bars_since_stop: int | None,
    cooldown_bars: int,
    clear_streak: int,
    clear_bars: int,
    high_growth: bool,
) -> bool:
    if bars_since_stop is None:
        return True
    if rule == "signal_reset":
        return False
    if rule == "cooldown":
        return bars_since_stop >= int(cooldown_bars)
    if rule == "clear":
        return clear_streak >= int(clear_bars)
    if rule == "high_growth_or_clear":
        return bool(high_growth) or clear_streak >= int(clear_bars)
    raise ValueError(f"Unsupported reentry rule: {rule}")


def trigger_shadow_gate(
    *,
    idx: int,
    cooldown_bars: int,
    current_gate_until: int,
) -> int:
    if cooldown_bars <= 0:
        return current_gate_until
    return max(current_gate_until, idx + int(cooldown_bars))


def simulate_qqq_path(
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    initial_capital: float,
    leverage: float,
    stop_loss_pct: float,
    taker_fee_rate: float,
    slippage_bps: float,
    reentry_rule: str,
    reentry_cooldown_bars: int,
    reentry_clear_bars: int,
    loss_streak_stop: int,
    loss_streak_cooldown_bars: int,
    equity_dd_stop_pct: float,
    equity_dd_cooldown_bars: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    equity_peak = capital
    holding = False
    stopped_after_stop = False
    bars_since_stop: int | None = None
    clear_streak = 0
    loss_streak = 0
    gate_until_idx = -1
    entry_price = 0.0
    stop_price = 0.0
    peak_close = 0.0
    current_trade: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    trigger_counts: dict[str, int] = {}
    per_side_cost = float(taker_fee_rate) + float(slippage_bps) / 10000.0

    for idx, row in enumerate(merged.itertuples(index=False)):
        start_capital = capital
        allow_now = bool(row.allow_long)
        gate_active = idx < gate_until_idx
        entered = False
        exited = False
        stop_hit = False
        funding_cost = 0.0
        fee_cost = 0.0
        leverage_now = float(leverage) if holding else 0.0

        if allow_now:
            clear_streak = clear_streak + 1 if not bool(row.defense_state) else 0
        else:
            clear_streak = 0
            stopped_after_stop = False
            bars_since_stop = None

        if bars_since_stop is not None:
            bars_since_stop += 1

        if holding and not allow_now:
            fee_cost += per_side_cost * float(leverage)
            capital *= 1.0 - fee_cost
            holding = False
            exited = True
            if current_trade is not None:
                trade_return = capital / float(current_trade["entry_capital"]) - 1.0
                trades.append({"entry_date": current_trade["entry_date"], "exit_date": str(pd.Timestamp(row.date)), "trade_return_pct": round(trade_return * 100.0, 2), "exit_reason": "signal_off"})
                loss_streak = loss_streak + 1 if trade_return <= 0 else 0
                if loss_streak_stop > 0 and loss_streak >= loss_streak_stop:
                    gate_until_idx = trigger_shadow_gate(idx=idx, cooldown_bars=loss_streak_cooldown_bars, current_gate_until=gate_until_idx)
                    trigger_counts["loss_streak"] = trigger_counts.get("loss_streak", 0) + 1
                    loss_streak = 0
            current_trade = None

        can_reenter = reentry_ready(
            rule=reentry_rule,
            bars_since_stop=bars_since_stop if stopped_after_stop else None,
            cooldown_bars=reentry_cooldown_bars,
            clear_streak=clear_streak,
            clear_bars=reentry_clear_bars,
            high_growth=bool(row.high_growth),
        )
        can_open = bool(allow_now and not holding and not gate_active and can_reenter)
        if can_open:
            fee_cost += per_side_cost * float(leverage)
            capital *= 1.0 - per_side_cost * float(leverage)
            holding = True
            entered = True
            stopped_after_stop = False
            bars_since_stop = None
            entry_price = float(row.open)
            stop_price = entry_price * (1.0 - float(stop_loss_pct) / 100.0)
            peak_close = float(row.open)
            current_trade = {"entry_date": str(pd.Timestamp(row.date)), "entry_capital": capital}
            leverage_now = float(leverage)

        if holding:
            open_price = float(row.open)
            low_price = float(row.low)
            close_price = float(row.close)
            peak_close = max(peak_close, close_price)
            stop_price = max(stop_price, peak_close * (1.0 - float(stop_loss_pct) / 100.0))

            if low_price <= stop_price:
                stop_hit = True
                exit_price = stop_price
                bar_ret = exit_price / open_price - 1.0 if open_price > 0 else 0.0
                capital *= 1.0 + float(leverage) * bar_ret
                fee_cost += per_side_cost * float(leverage)
                capital *= 1.0 - per_side_cost * float(leverage)
                holding = False
                exited = True
                stopped_after_stop = True
                bars_since_stop = 0
                if current_trade is not None:
                    trade_return = capital / float(current_trade["entry_capital"]) - 1.0
                    trades.append({"entry_date": current_trade["entry_date"], "exit_date": str(pd.Timestamp(row.date)), "trade_return_pct": round(trade_return * 100.0, 2), "exit_reason": "trailing_stop"})
                    loss_streak = loss_streak + 1 if trade_return <= 0 else 0
                    if loss_streak_stop > 0 and loss_streak >= loss_streak_stop:
                        gate_until_idx = trigger_shadow_gate(idx=idx, cooldown_bars=loss_streak_cooldown_bars, current_gate_until=gate_until_idx)
                        trigger_counts["loss_streak"] = trigger_counts.get("loss_streak", 0) + 1
                        loss_streak = 0
                current_trade = None
            else:
                bar_ret = close_price / open_price - 1.0 if open_price > 0 else 0.0
                capital *= 1.0 + float(leverage) * bar_ret
                if is_funding_settlement_bar(row.date, row.funding_event_time):
                    funding_cost = float(row.funding_rate_value) * float(leverage)
                    capital *= 1.0 - funding_cost

        equity_peak = max(equity_peak, capital)
        if equity_dd_stop_pct > 0 and equity_peak > 0:
            dd_pct = (equity_peak - capital) / equity_peak * 100.0
            if dd_pct >= float(equity_dd_stop_pct):
                gate_until_idx = trigger_shadow_gate(idx=idx, cooldown_bars=equity_dd_cooldown_bars, current_gate_until=gate_until_idx)
                trigger_counts["equity_dd"] = trigger_counts.get("equity_dd", 0) + 1
                equity_peak = capital

        rows.append(
            {
                "date": pd.Timestamp(row.date),
                "session_day": pd.Timestamp(row.date).floor("D"),
                "bar_return": float(capital / start_capital - 1.0 if start_capital > 0 else 0.0),
                "capital": float(capital),
                "holding": bool(holding),
                "allow_long": bool(allow_now),
                "entered": bool(entered),
                "exited": bool(exited),
                "stop_hit": bool(stop_hit),
                "gate_active": bool(gate_active),
                "stopped_after_stop": bool(stopped_after_stop),
                "high_growth": bool(row.high_growth),
                "defense_state": bool(row.defense_state),
                "leverage_now": float(leverage_now),
                "funding_cost": float(funding_cost),
                "fee_cost": float(fee_cost),
            }
        )

    path_4h = pd.DataFrame(rows)
    trades_df = pd.DataFrame(trades)
    daily_rows: list[dict[str, Any]] = []
    for day, group in path_4h.groupby("session_day", sort=True):
        effective_signal = group["allow_long"] & ~group["gate_active"]
        active = bool((group["holding"] | group["entered"] | group["exited"] | effective_signal).any())
        avg_leverage = float(group.loc[group["leverage_now"] > 0, "leverage_now"].mean()) if bool((group["leverage_now"] > 0).any()) else 0.0
        qqq_score = 0.0
        if active:
            qqq_score = 72.0 + avg_leverage * 4.0
            if bool(group["high_growth"].any()):
                qqq_score += 10.0
            if bool(group["defense_state"].any()):
                qqq_score -= 6.0
        daily_rows.append(
            {
                "date": pd.Timestamp(day),
                "qqq_equity_raw": float(group["capital"].iloc[-1]),
                "qqq_return": float((1.0 + group["bar_return"]).prod() - 1.0),
                "qqq_active": active,
                "qqq_score": round(float(qqq_score), 2),
                "position": "QQQ_USDT_LONG" if active else "CASH",
                "entry_type": "shadow_gate_scan",
                "vix_label": "",
                "ixic_trend_label": "",
                "rel_strength_label": "",
                "risk_cash_day": False,
                "risk_capped_day": False,
                "avg_leverage_when_active": avg_leverage,
                "shadow_gate_day": bool(group["gate_active"].any()),
                "stop_hit_day": bool(group["stop_hit"].any()),
            }
        )

    daily_path = pd.DataFrame(daily_rows)
    summary = {
        "total_return_pct": round((float(path_4h["capital"].iloc[-1]) / float(initial_capital) - 1.0) * 100.0, 2) if not path_4h.empty else 0.0,
        "max_drawdown_pct": round(max_drawdown_pct(path_4h["capital"]), 2) if not path_4h.empty else 0.0,
        "bars": int(len(path_4h)),
        "days": int(len(daily_path)),
        "invested_bars": int(path_4h["holding"].sum()) if not path_4h.empty else 0,
        "invested_days": int(daily_path["qqq_active"].sum()) if not daily_path.empty else 0,
        "trades": int(len(trades_df)),
        "stop_hits": int(path_4h["stop_hit"].sum()) if not path_4h.empty else 0,
        "gate_days": int(daily_path["shadow_gate_day"].sum()) if not daily_path.empty else 0,
        "trigger_counts": trigger_counts,
        "funding_cost_pct_est": round(float(path_4h["funding_cost"].sum() * 100.0), 2) if not path_4h.empty else 0.0,
        "fee_cost_pct_est": round(float(path_4h["fee_cost"].sum() * 100.0), 2) if not path_4h.empty else 0.0,
        "avg_leverage_when_in": round(float(path_4h.loc[path_4h["holding"], "leverage_now"].mean()), 2) if bool(path_4h["holding"].any()) else 0.0,
    }
    return daily_path, summary


def route_candidate(
    *,
    btc_path: pd.DataFrame,
    qqq_path: pd.DataFrame,
    initial_capital: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = pd.merge(btc_path, qqq_path, on="date", how="inner").sort_values("date").reset_index(drop=True)
    merged["btc_equity"] = equity_from_returns(merged["btc_return"], initial_capital)
    merged["qqq_equity"] = equity_from_returns(merged["qqq_return"], initial_capital)
    router_path = run_router(
        merged,
        initial_capital=initial_capital,
        btc_min_score=35.0,
        qqq_min_score=60.0,
        switch_advantage=8.0,
        btc_takeover_advantage=None,
        qqq_takeover_advantage=None,
        switch_cost_bps=10.0,
    )
    full = pd.concat([merged, router_path.drop(columns=["date"])], axis=1)
    return full, {
        "router": summarize_equity(full["router_equity"]),
        "qqq": summarize_equity(full["qqq_equity"]),
        "selection": {
            "btc_days": int((full["selected_strategy"] == "BTC").sum()),
            "qqq_proxy_days": int((full["selected_strategy"] == "QQQ_PROXY").sum()),
            "cash_days": int((full["selected_strategy"] == "CASH").sum()),
            "switches": int(full["switched"].sum()),
        },
    }


def candidate_key(result: dict[str, Any], metric: str) -> Any:
    router = result["router"]
    if metric == "dd":
        return (router["max_drawdown_pct"], -router["total_return_pct"], -router["calmar_like"])
    if metric == "calmar":
        return (-(router["calmar_like"] or -999999.0), router["max_drawdown_pct"])
    if metric == "cvar":
        return (-router["daily_cvar5_pct"], router["max_drawdown_pct"])
    if metric == "balanced":
        return (
            router["max_drawdown_pct"] * 2.0
            - router["total_return_pct"] / 100.0
            - (router["calmar_like"] or 0.0) / 10.0,
            router["max_drawdown_pct"],
        )
    raise ValueError(metric)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan QQQ-independent shadow gate plus trailing/re-entry rules inside the 2022-2026 BTC+QQQ NQ mock router.")
    parser.add_argument("--config", default=str(DEFAULT_QQQ_USDT_CONFIG))
    parser.add_argument("--data-4h", default=str(DEFAULT_NQ_4H))
    parser.add_argument("--funding", default=str(DEFAULT_NQ_FUNDING))
    parser.add_argument("--btc-frozen", default=str(DEFAULT_BTC_FROZEN))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2026-05-29")
    parser.add_argument("--stop-loss-values", default="2.0,2.5,3.0,3.5,4.0,5.0")
    parser.add_argument("--reentry-rules", default="signal_reset,cooldown,clear,high_growth_or_clear")
    parser.add_argument("--reentry-cooldown-bars", default="0,2,5,10")
    parser.add_argument("--reentry-clear-bars", default="1,2,3")
    parser.add_argument("--loss-streak-values", default="0,2,3")
    parser.add_argument("--loss-streak-cooldown-bars", default="0,5,10")
    parser.add_argument("--equity-dd-values", default="0,20,30,40")
    parser.add_argument("--equity-dd-cooldown-bars", default="0,10,20")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    _, signal_path = load_signal_path(ROOT / str(config["signal_source"]))
    bars = enrich_bars(attach_daily_state(load_okx_4h(Path(args.data_4h)), signal_path))
    start = pd.Timestamp(args.start_date, tz="UTC")
    end = pd.Timestamp(args.end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    bars = bars.loc[(bars["date"] >= start) & (bars["date"] <= end)].copy()
    funding = load_funding(Path(args.funding))
    initial_capital = float(config["initial_capital"])
    leverage = float(config["base_leverage"])
    taker_fee_rate = float(config["taker_fee_rate"])
    slippage_bps = float(config["slippage_bps"])
    btc_path, btc_meta = build_btc_path_from_frozen_artifact(
        frozen_path=Path(args.btc_frozen),
        start=start,
        end=end,
        initial_capital=initial_capital,
    )

    stop_values = parse_float_list(args.stop_loss_values)
    reentry_rules = parse_str_list(args.reentry_rules)
    reentry_cooldowns = parse_int_list(args.reentry_cooldown_bars)
    clear_values = parse_int_list(args.reentry_clear_bars)
    loss_streak_values = parse_int_list(args.loss_streak_values)
    loss_streak_cooldowns = parse_int_list(args.loss_streak_cooldown_bars)
    equity_dd_values = parse_float_list(args.equity_dd_values)
    equity_dd_cooldowns = parse_int_list(args.equity_dd_cooldown_bars)

    reentry_specs: list[tuple[str, int, int]] = []
    for rule in reentry_rules:
        if rule == "signal_reset":
            reentry_specs.append((rule, 0, 0))
        elif rule == "cooldown":
            reentry_specs.extend((rule, cooldown, 0) for cooldown in reentry_cooldowns if cooldown > 0)
        elif rule in {"clear", "high_growth_or_clear"}:
            reentry_specs.extend((rule, 0, clear_bars) for clear_bars in clear_values if clear_bars > 0)
        else:
            raise ValueError(f"Unsupported reentry rule: {rule}")

    loss_specs = [(0, 0)] + [(streak, cooldown) for streak, cooldown in itertools.product(loss_streak_values, loss_streak_cooldowns) if streak > 0 and cooldown > 0]
    equity_specs = [(0.0, 0)] + [(dd, cooldown) for dd, cooldown in itertools.product(equity_dd_values, equity_dd_cooldowns) if dd > 0 and cooldown > 0]

    results: list[dict[str, Any]] = []
    for stop_loss_pct, reentry_spec, loss_spec, equity_spec in itertools.product(stop_values, reentry_specs, loss_specs, equity_specs):
        reentry_rule, reentry_cooldown, reentry_clear = reentry_spec
        loss_streak, loss_cooldown = loss_spec
        equity_dd, equity_cooldown = equity_spec
        qqq_path, qqq_summary = simulate_qqq_path(
            bars,
            funding,
            initial_capital=initial_capital,
            leverage=leverage,
            stop_loss_pct=stop_loss_pct,
            taker_fee_rate=taker_fee_rate,
            slippage_bps=slippage_bps,
            reentry_rule=reentry_rule,
            reentry_cooldown_bars=reentry_cooldown,
            reentry_clear_bars=reentry_clear,
            loss_streak_stop=loss_streak,
            loss_streak_cooldown_bars=loss_cooldown,
            equity_dd_stop_pct=equity_dd,
            equity_dd_cooldown_bars=equity_cooldown,
        )
        _, routed = route_candidate(btc_path=btc_path, qqq_path=qqq_path, initial_capital=initial_capital)
        results.append(
            {
                "params": {
                    "stop_loss_pct": stop_loss_pct,
                    "reentry_rule": reentry_rule,
                    "reentry_cooldown_bars": reentry_cooldown,
                    "reentry_clear_bars": reentry_clear,
                    "loss_streak_stop": loss_streak,
                    "loss_streak_cooldown_bars": loss_cooldown,
                    "equity_dd_stop_pct": equity_dd,
                    "equity_dd_cooldown_bars": equity_cooldown,
                },
                "router": routed["router"],
                "qqq": routed["qqq"],
                "selection": routed["selection"],
                "qqq_path_summary": qqq_summary,
            }
        )

    baseline = next(
        (
            item
            for item in results
            if item["params"] == {
                "stop_loss_pct": 3.5,
                "reentry_rule": "signal_reset",
                "reentry_cooldown_bars": 0,
                "reentry_clear_bars": 0,
                "loss_streak_stop": 0,
                "loss_streak_cooldown_bars": 0,
                "equity_dd_stop_pct": 0.0,
                "equity_dd_cooldown_bars": 0,
            }
        ),
        None,
    )
    ranked = {
        "top_by_router_dd": sorted(results, key=lambda item: candidate_key(item, "dd"))[: args.top],
        "top_by_router_calmar": sorted(results, key=lambda item: candidate_key(item, "calmar"))[: args.top],
        "top_by_router_cvar": sorted(results, key=lambda item: candidate_key(item, "cvar"))[: args.top],
        "top_balanced": sorted(results, key=lambda item: candidate_key(item, "balanced"))[: args.top],
    }
    report = {
        "period": {"start": args.start_date, "end": args.end_date},
        "config": {
            "source_config": str(Path(args.config).resolve()),
            "data_4h": str(Path(args.data_4h).resolve()),
            "funding": str(Path(args.funding).resolve()),
            "btc_frozen": str(Path(args.btc_frozen).resolve()),
            "btc_meta": btc_meta,
            "candidate_count": len(results),
            "grid": {
                "stop_loss_values": stop_values,
                "reentry_specs": reentry_specs,
                "loss_specs": loss_specs,
                "equity_specs": equity_specs,
            },
        },
        "baseline": baseline,
        "ranked": ranked,
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    compact = {
        key: [
            {
                "params": item["params"],
                "router": item["router"],
                "qqq": item["qqq"],
                "selection": item["selection"],
                "qqq_path_summary": item["qqq_path_summary"],
            }
            for item in value[:5]
        ]
        for key, value in ranked.items()
    }
    print(out)
    print(json.dumps({"baseline": baseline, "top": compact}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
