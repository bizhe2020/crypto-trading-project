#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_qqq_usdt_10x import load_funding, max_drawdown_pct  # noqa: E402
from scripts.scan_qqq_usdt_4h_triggers import attach_daily_state, load_okx_4h, load_signal_path  # noqa: E402
from scripts.scan_qqq_usdt_btc_logic_overlays import enrich_bars  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "config.paper.tqqq-only-strict-recovery-frozen.json"
DEFAULT_OKX_4H = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-4h-futures.feather"
DEFAULT_FUNDING = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-8h-funding_rate.feather"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_usdt_funding_aware_leverage_scan.json"

THRESHOLD_BPS = [0.0, 1.0, 2.0, 4.0, 6.0]
REDUCE_BASE_TO = [8.0, 6.0, 4.0, 2.0]
REDUCE_OFFENSE_TO = [10.0, 8.0, 6.0, 4.0, 2.0]


def simulate(
    bars,
    funding,
    *,
    stop_loss_pct: float,
    taker_fee_rate: float,
    slippage_bps: float,
    initial_capital: float,
    funding_threshold_bps: float,
    reduce_base_to: float,
    reduce_offense_to: float,
) -> dict:
    import pandas as pd

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
        funding_cost = 0.0
        fee_cost = 0.0
        stop_hit = False

        positive_funding_bps = max(float(row.funding_rate_value), 0.0) * 10000.0
        funding_hot = positive_funding_bps >= float(funding_threshold_bps) if funding_threshold_bps > 0 else False

        leverage_now = 0.0
        if holding:
            if bool(row.high_growth):
                leverage_now = 10.0
            elif bool(row.defense_state):
                leverage_now = 2.0
            else:
                leverage_now = 8.0
            if funding_hot:
                if leverage_now >= 10.0:
                    leverage_now = min(leverage_now, float(reduce_offense_to))
                elif leverage_now >= 8.0:
                    leverage_now = min(leverage_now, float(reduce_base_to))

        if holding and not allow_now:
            fee_cost = per_side_cost
            capital *= 1.0 - fee_cost * leverage_now
            holding = False
            exited_today = True
            if current_trade is not None:
                trades.append(
                    {
                        "entry_date": current_trade["entry_date"],
                        "exit_date": str(row.date),
                        "trade_return_pct": round((capital / float(current_trade["entry_capital"]) - 1.0) * 100.0, 2),
                    }
                )
            current_trade = None

        if allow_now and not holding and not prev_allow:
            leverage_now = 8.0
            if funding_hot:
                leverage_now = min(leverage_now, float(reduce_base_to))
            fee_cost = per_side_cost
            capital *= 1.0 - fee_cost * leverage_now
            holding = True
            entered_today = True
            entry_price = float(row.open)
            stop_price = entry_price * (1.0 - float(stop_loss_pct) / 100.0)
            peak_close = float(row.open)
            current_trade = {"entry_date": str(row.date), "entry_capital": capital}

        if holding:
            if bool(row.high_growth):
                leverage_now = 10.0
            elif bool(row.defense_state):
                leverage_now = 2.0
            else:
                leverage_now = 8.0
            if funding_hot:
                if leverage_now >= 10.0:
                    leverage_now = min(leverage_now, float(reduce_offense_to))
                elif leverage_now >= 8.0:
                    leverage_now = min(leverage_now, float(reduce_base_to))

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
                            "exit_date": str(row.date),
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
                "date": row.date,
                "holding": holding,
                "entered_today": entered_today,
                "exited_today": exited_today,
                "stop_hit": stop_hit,
                "capital": float(capital),
                "daily_return": capital / start_capital - 1.0 if start_capital > 0 else 0.0,
                "funding_cost": float(funding_cost),
                "funding_hot": funding_hot,
                "leverage_now": float(leverage_now),
            }
        )
        prev_allow = allow_now

    path = pd.DataFrame(rows)
    trades_df = pd.DataFrame(trades)
    total_return_pct = round((float(path.iloc[-1]["capital"]) / float(initial_capital) - 1.0) * 100.0, 2) if not path.empty else 0.0
    max_dd = round(max_drawdown_pct(path["capital"]), 2) if not path.empty else 0.0
    score = round(total_return_pct - max_dd * 2.0, 2)
    return {
        "funding_threshold_bps": float(funding_threshold_bps),
        "reduce_base_to": float(reduce_base_to),
        "reduce_offense_to": float(reduce_offense_to),
        "summary": {
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd,
            "score": score,
            "trades": int(len(trades_df)),
            "win_rate_pct": round(float((trades_df["trade_return_pct"] > 0).mean() * 100.0), 2) if not trades_df.empty else 0.0,
            "avg_trade_return_pct": round(float(trades_df["trade_return_pct"].mean()), 2) if not trades_df.empty else 0.0,
            "funding_cost_pct_est": round(float(path["funding_cost"].sum() * 100.0), 2) if not path.empty else 0.0,
            "invested_bars": int(path["holding"].sum()) if not path.empty else 0,
            "funding_hot_bars": int(path["funding_hot"].sum()) if not path.empty else 0,
            "avg_leverage_when_in": round(float(path.loc[path["holding"], "leverage_now"].mean()), 2) if (not path.empty and (path["holding"]).any()) else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan funding-aware leverage downshift rules for QQQ/USDT.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--okx-4h", default=str(DEFAULT_OKX_4H))
    parser.add_argument("--funding", default=str(DEFAULT_FUNDING))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--stop-loss-pct", type=float, default=2.5)
    parser.add_argument("--taker-fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    args = parser.parse_args()

    config, signal_path = load_signal_path(Path(args.config))
    bars = enrich_bars(attach_daily_state(load_okx_4h(Path(args.okx_4h)), signal_path))
    funding = load_funding(Path(args.funding))

    results = []
    baseline = simulate(
        bars,
        funding,
        stop_loss_pct=float(args.stop_loss_pct),
        taker_fee_rate=float(args.taker_fee_rate),
        slippage_bps=float(args.slippage_bps),
        initial_capital=float(config["initial_capital"]),
        funding_threshold_bps=999.0,
        reduce_base_to=8.0,
        reduce_offense_to=10.0,
    )
    baseline["mode"] = "baseline_no_funding_gate"
    results.append(baseline)
    for threshold_bps in THRESHOLD_BPS:
        if threshold_bps == 0.0:
            continue
        for reduce_base_to in REDUCE_BASE_TO:
            for reduce_offense_to in REDUCE_OFFENSE_TO:
                if reduce_offense_to < reduce_base_to:
                    continue
                item = simulate(
                    bars,
                    funding,
                    stop_loss_pct=float(args.stop_loss_pct),
                    taker_fee_rate=float(args.taker_fee_rate),
                    slippage_bps=float(args.slippage_bps),
                    initial_capital=float(config["initial_capital"]),
                    funding_threshold_bps=float(threshold_bps),
                    reduce_base_to=float(reduce_base_to),
                    reduce_offense_to=float(reduce_offense_to),
                )
                item["mode"] = "funding_gate"
                results.append(item)

    baseline_summary = baseline["summary"]
    for item in results:
        item["delta_vs_baseline"] = {
            "total_return_pct": round(item["summary"]["total_return_pct"] - baseline_summary["total_return_pct"], 2),
            "max_drawdown_pct": round(item["summary"]["max_drawdown_pct"] - baseline_summary["max_drawdown_pct"], 2),
            "funding_cost_pct_est": round(item["summary"]["funding_cost_pct_est"] - baseline_summary["funding_cost_pct_est"], 2),
        }

    by_score = sorted(results, key=lambda x: x["summary"]["score"], reverse=True)
    payload = {
        "candidate": {
            "signal_frozen_label": config.get("frozen_label"),
            "stop_loss_pct": float(args.stop_loss_pct),
            "base_profile": "dyn_cap10",
        },
        "baseline": baseline,
        "top_by_score": by_score[:12],
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(out)
    print(json.dumps({"baseline": baseline, "top_by_score": by_score[:8]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
