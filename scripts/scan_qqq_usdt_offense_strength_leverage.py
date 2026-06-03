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
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_usdt_offense_strength_leverage_scan.json"


PROFILES: dict[str, dict[str, Any]] = {
    "baseline_fixed10": {
        "base": 10.0,
        "defense": 10.0,
        "mode": "discrete",
        "offense": 10.0,
    },
    "candidate_base10_off10_def1": {
        "base": 10.0,
        "defense": 1.0,
        "mode": "discrete",
        "offense": 10.0,
    },
    "score_linear_def1_base4_cap10": {
        "base": 4.0,
        "defense": 1.0,
        "mode": "linear",
        "cap": 10.0,
        "floor": 1.0,
    },
    "score_linear_def1_base6_cap10": {
        "base": 6.0,
        "defense": 1.0,
        "mode": "linear",
        "cap": 10.0,
        "floor": 1.0,
    },
    "score_linear_def1_base8_cap10": {
        "base": 8.0,
        "defense": 1.0,
        "mode": "linear",
        "cap": 10.0,
        "floor": 1.0,
    },
    "score_bucket_def1_1_5_8_10": {
        "base": 5.0,
        "defense": 1.0,
        "mode": "bucket",
        "buckets": [(0.35, 1.0), (0.55, 5.0), (0.75, 8.0), (1.01, 10.0)],
    },
    "score_bucket_def1_1_6_8_10": {
        "base": 6.0,
        "defense": 1.0,
        "mode": "bucket",
        "buckets": [(0.35, 1.0), (0.55, 6.0), (0.75, 8.0), (1.01, 10.0)],
    },
    "score_bucket_def1_1_4_7_10": {
        "base": 4.0,
        "defense": 1.0,
        "mode": "bucket",
        "buckets": [(0.35, 1.0), (0.55, 4.0), (0.75, 7.0), (1.01, 10.0)],
    },
    "score_bucket_def1_1_3_6_10": {
        "base": 3.0,
        "defense": 1.0,
        "mode": "bucket",
        "buckets": [(0.35, 1.0), (0.55, 3.0), (0.75, 6.0), (1.01, 10.0)],
    },
    "score_bucket_def1_6_8_10": {
        "base": 6.0,
        "defense": 1.0,
        "mode": "bucket",
        "buckets": [(0.55, 6.0), (0.75, 8.0), (1.01, 10.0)],
    },
    "score_bucket_def1_8_9_10": {
        "base": 8.0,
        "defense": 1.0,
        "mode": "bucket",
        "buckets": [(0.55, 8.0), (0.75, 9.0), (1.01, 10.0)],
    },
    "score_bucket_def1_9_10_10": {
        "base": 9.0,
        "defense": 1.0,
        "mode": "bucket",
        "buckets": [(0.55, 9.0), (0.75, 10.0), (1.01, 10.0)],
    },
    "offense_bucket_def1_base10_6_8_10": {
        "base": 10.0,
        "defense": 1.0,
        "mode": "offense_bucket",
        "buckets": [(0.55, 6.0), (0.75, 8.0), (1.01, 10.0)],
    },
    "offense_bucket_def1_base10_8_9_10": {
        "base": 10.0,
        "defense": 1.0,
        "mode": "offense_bucket",
        "buckets": [(0.55, 8.0), (0.75, 9.0), (1.01, 10.0)],
    },
    "offense_linear_def1_base10_floor6_cap10": {
        "base": 10.0,
        "defense": 1.0,
        "mode": "offense_linear",
        "floor": 6.0,
        "cap": 10.0,
    },
    "offense_linear_def1_base10_floor8_cap10": {
        "base": 10.0,
        "defense": 1.0,
        "mode": "offense_linear",
        "floor": 8.0,
        "cap": 10.0,
    },
}


def normalize(series: pd.Series) -> pd.Series:
    low = float(series.quantile(0.05))
    high = float(series.quantile(0.95))
    if high <= low:
        return pd.Series(0.0, index=series.index)
    return ((series - low) / (high - low)).clip(lower=0.0, upper=1.0).fillna(0.0)


def attach_offense_strength(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    frame["breakout_dist"] = (frame["close"] / frame["prev_high_12"] - 1.0).replace([float("inf"), -float("inf")], 0.0)
    frame["offense_strength"] = (
        normalize(frame["mom_3"]) * 0.35
        + normalize(frame["ema_gap_pct"]) * 0.25
        + normalize(frame["breakout_dist"]) * 0.25
        + normalize(frame["close"] / frame["ema20"] - 1.0) * 0.15
    ).clip(lower=0.0, upper=1.0)
    frame.loc[~frame["high_growth"].astype(bool), "offense_strength"] *= 0.55
    frame.loc[frame["defense_state"].astype(bool), "offense_strength"] = frame.loc[
        frame["defense_state"].astype(bool), "offense_strength"
    ].clip(upper=0.25)
    return frame


def leverage_for(profile: dict[str, Any], row: Any) -> float:
    if bool(row.defense_state):
        return float(profile["defense"])
    strength = float(row.offense_strength)
    mode = str(profile["mode"])
    if mode == "discrete":
        return float(profile["offense"]) if bool(row.high_growth) else float(profile["base"])
    if mode == "linear":
        floor = float(profile["floor"])
        cap = float(profile["cap"])
        raw = floor + strength * (cap - floor)
        return max(floor, min(cap, raw))
    if mode == "bucket":
        for upper, leverage in profile["buckets"]:
            if strength < float(upper):
                return float(leverage)
        return float(profile["buckets"][-1][1])
    if mode == "offense_bucket":
        if not bool(row.high_growth):
            return float(profile["base"])
        for upper, leverage in profile["buckets"]:
            if strength < float(upper):
                return float(leverage)
        return float(profile["buckets"][-1][1])
    if mode == "offense_linear":
        if not bool(row.high_growth):
            return float(profile["base"])
        floor = float(profile["floor"])
        cap = float(profile["cap"])
        raw = floor + strength * (cap - floor)
        return max(floor, min(cap, raw))
    raise ValueError(f"Unsupported profile mode: {mode}")


def simulate(
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    profile_name: str,
    stop_loss_pct: float,
    taker_fee_rate: float,
    slippage_bps: float,
    rebalance_cost_multiplier: float,
    initial_capital: float,
) -> dict[str, Any]:
    profile = PROFILES[profile_name]
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
    prev_leverage = 0.0
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
        fee_cost = 0.0
        rebalance_cost = 0.0
        leverage_now = 0.0

        if holding and not allow_now:
            fee_cost = per_side_cost * prev_leverage
            capital *= 1.0 - fee_cost
            holding = False
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
            prev_leverage = 0.0

        if allow_now and not holding and not prev_allow:
            leverage_now = leverage_for(profile, row)
            fee_cost = per_side_cost * leverage_now
            capital *= 1.0 - fee_cost
            holding = True
            entered = True
            entry_price = float(row.open)
            stop_price = entry_price * (1.0 - float(stop_loss_pct) / 100.0)
            peak_close = float(row.open)
            current_trade = {"entry_date": str(pd.Timestamp(row.date)), "entry_capital": capital}
            prev_leverage = leverage_now

        if holding:
            leverage_now = leverage_for(profile, row)
            if not entered:
                rebalance_notional = abs(leverage_now - prev_leverage)
                if rebalance_notional > 1e-9:
                    rebalance_cost = per_side_cost * rebalance_notional * float(rebalance_cost_multiplier)
                    capital *= 1.0 - rebalance_cost

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
                fee_cost += per_side_cost * leverage_now
                capital *= 1.0 - per_side_cost * leverage_now
                holding = False
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
                prev_leverage = 0.0
            else:
                bar_ret = close_price / open_price - 1.0 if open_price > 0 else 0.0
                capital *= 1.0 + leverage_now * bar_ret
                funding_cost = max(float(row.funding_rate_value), 0.0) * leverage_now
                capital *= 1.0 - funding_cost
                prev_leverage = leverage_now

        rows.append(
            {
                "date": pd.Timestamp(row.date),
                "holding": holding,
                "entered": entered,
                "exited": exited,
                "stop_hit": stop_hit,
                "capital": float(capital),
                "daily_return": capital / start_capital - 1.0 if start_capital > 0 else 0.0,
                "funding_cost": float(funding_cost),
                "fee_cost": float(fee_cost),
                "rebalance_cost": float(rebalance_cost),
                "leverage_now": float(leverage_now),
                "offense_strength": float(row.offense_strength),
            }
        )
        prev_allow = allow_now

    path = pd.DataFrame(rows)
    trades_df = pd.DataFrame(trades)
    total_return_pct = round((float(path.iloc[-1]["capital"]) / float(initial_capital) - 1.0) * 100.0, 2) if not path.empty else 0.0
    max_dd = round(max_drawdown_pct(path["capital"]), 2) if not path.empty else 0.0
    score = round(total_return_pct - max_dd * 2.0, 2)
    in_path = path.loc[path["holding"] | path["entered"]]
    return {
        "profile": profile_name,
        "profile_spec": profile,
        "summary": {
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd,
            "score": score,
            "trades": int(len(trades_df)),
            "win_rate_pct": round(float((trades_df["trade_return_pct"] > 0).mean() * 100.0), 2) if not trades_df.empty else 0.0,
            "avg_trade_return_pct": round(float(trades_df["trade_return_pct"].mean()), 2) if not trades_df.empty else 0.0,
            "invested_bars": int(path["holding"].sum()) if not path.empty else 0,
            "stop_hits": int(path["stop_hit"].sum()) if not path.empty else 0,
            "funding_cost_pct_est": round(float(path["funding_cost"].sum() * 100.0), 2) if not path.empty else 0.0,
            "fee_cost_pct_est": round(float(path["fee_cost"].sum() * 100.0), 2) if not path.empty else 0.0,
            "rebalance_cost_pct_est": round(float(path["rebalance_cost"].sum() * 100.0), 2) if not path.empty else 0.0,
            "avg_leverage_when_in": round(float(in_path["leverage_now"].mean()), 2) if not in_path.empty else 0.0,
            "max_leverage_when_in": round(float(in_path["leverage_now"].max()), 2) if not in_path.empty else 0.0,
            "leverage_changes": int((path["leverage_now"].diff().abs() > 1e-9).sum()) if not path.empty else 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan QQQ/USDT offense-strength 0-1 leverage mapping.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--rebalance-cost-multiplier", type=float, default=1.0)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    _, signal_path = load_signal_path(ROOT / str(config["signal_source"]))
    bars = enrich_bars(attach_daily_state(load_okx_4h(ROOT / str(config["data_4h"])), signal_path))
    bars = attach_offense_strength(bars)
    funding = load_funding(ROOT / str(config["funding_history_path"]))

    results = [
        simulate(
            bars,
            funding,
            profile_name=name,
            stop_loss_pct=float(config["stop_loss_pct"]),
            taker_fee_rate=float(config["taker_fee_rate"]),
            slippage_bps=float(config["slippage_bps"]),
            rebalance_cost_multiplier=float(args.rebalance_cost_multiplier),
            initial_capital=float(config["initial_capital"]),
        )
        for name in PROFILES
    ]
    by_return = sorted(
        results,
        key=lambda item: (
            item["summary"]["total_return_pct"],
            -item["summary"]["max_drawdown_pct"],
            -item["summary"]["rebalance_cost_pct_est"],
        ),
        reverse=True,
    )
    by_score = sorted(results, key=lambda item: item["summary"]["score"], reverse=True)
    baseline = next(item for item in results if item["profile"] == "baseline_fixed10")

    payload = {
        "config": {
            "source_config": str(config_path),
            "frozen_label": config.get("frozen_label"),
            "stop_loss_pct": float(config["stop_loss_pct"]),
            "taker_fee_rate": float(config["taker_fee_rate"]),
            "slippage_bps": float(config["slippage_bps"]),
            "rebalance_cost_multiplier": float(args.rebalance_cost_multiplier),
            "profiles": PROFILES,
            "coverage": {
                "bars": int(len(bars)),
                "start": str(bars["date"].min()) if not bars.empty else None,
                "end": str(bars["date"].max()) if not bars.empty else None,
                "allow_long_bars": int(bars["allow_long"].sum()) if not bars.empty else 0,
                "high_growth_bars": int(bars["high_growth"].sum()) if not bars.empty else 0,
                "defense_bars": int(bars["defense_state"].sum()) if not bars.empty else 0,
            },
        },
        "baseline": baseline,
        "top_by_return": by_return,
        "top_by_score": by_score,
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(out)
    print(json.dumps({"baseline": baseline, "top_by_return": by_return, "top_by_score": by_score}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
