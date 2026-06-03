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

from scripts.audit_tqqq_cash_regime_context import load_df  # noqa: E402
from scripts.scan_tqqq_context_bucket_overlays import (  # noqa: E402
    allow_short,
    prepare_frame,
    select_long_profile,
)
from scripts.replay_tqqq_sqqq_trend_baseline import max_drawdown_pct  # noqa: E402


DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "tqqq_kelly_bucket_weight_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Kelly-style bucket weights for Stable TQQQ + bearish SQQQ.")
    parser.add_argument("--tqqq", default=str(DEFAULT_PUBLIC_DIR / "TQQQ-1d.feather"))
    parser.add_argument("--sqqq", default=str(DEFAULT_PUBLIC_DIR / "SQQQ-1d.feather"))
    parser.add_argument("--qqq", default=str(DEFAULT_PUBLIC_DIR / "QQQ-1d.feather"))
    parser.add_argument("--spy", default=str(DEFAULT_PUBLIC_DIR / "SPY-1d.feather"))
    parser.add_argument("--ixic", default=str(DEFAULT_PUBLIC_DIR / "^IXIC-1d.feather"))
    parser.add_argument("--vix", default=str(DEFAULT_PUBLIC_DIR / "^VIX-1d.feather"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--entry-fast-window", type=int, default=25)
    parser.add_argument("--entry-slow-window", type=int, default=200)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--switch-cost-bps", type=float, default=10.0)
    parser.add_argument("--long-profile-name", default="stable_base")
    parser.add_argument("--short-rule-name", default="bearish_score5")
    parser.add_argument("--short-max-hold-days", type=int, default=20)
    parser.add_argument("--short-trailing-lookback-days", type=int, default=5)
    parser.add_argument("--short-trailing-drawdown-pct", type=float, default=6.0)
    parser.add_argument("--long-weight-values", default="0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--short-weight-values", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--max-dd-delta-pct", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def annual_returns(equity: pd.DataFrame) -> dict[str, float]:
    if equity.empty:
        return {}
    frame = equity.copy()
    frame["year"] = frame["date"].dt.year.astype(str)
    out: dict[str, float] = {}
    for year, group in frame.groupby("year"):
        start = float(group.iloc[0]["equity"])
        end = float(group.iloc[-1]["equity"])
        out[year] = round((end / start - 1.0) * 100.0, 2) if start > 0 else 0.0
    return out


def asset_price_column(asset: str) -> str:
    if asset == "TQQQ":
        return "tqqq_close"
    if asset == "SQQQ":
        return "sqqq_close"
    raise ValueError(f"Unsupported asset: {asset}")


def run_weighted_candidate(
    frame: pd.DataFrame,
    *,
    long_profile_name: str,
    short_rule_name: str,
    short_exit_profile: tuple[int, int, float],
    long_weight: float,
    short_weight: float,
    initial_capital: float,
    switch_cost_bps: float,
) -> dict[str, Any]:
    long_mask = frame["vix_label"].isin(["vix_low", "vix_normal"]) & frame["ixic_trend_label"].eq("ixic_up")
    capital = initial_capital
    previous_position = "CASH"
    active_asset = "CASH"
    active_weight = 0.0
    active_max_hold_days = 0
    active_trailing_lookback_days = 0
    active_trailing_drawdown_pct = 0.0
    hold_days = 0
    rolling_peak = 0.0
    rows: list[dict[str, Any]] = []

    for idx, row in frame.iterrows():
        raw_desired_asset = "CASH"
        raw_desired_weight = 0.0
        desired_long = int(row["planned_trend"]) > 0 and bool(long_mask.iloc[idx])
        desired_short = int(row["planned_trend"]) < 0 and allow_short(row, short_rule_name)
        candidate_long_profile = select_long_profile(row, long_profile_name)

        if desired_long and long_weight > 0:
            raw_desired_asset = "TQQQ"
            raw_desired_weight = float(long_weight)
        elif desired_short and short_weight > 0:
            raw_desired_asset = "SQQQ"
            raw_desired_weight = float(short_weight)

        if active_asset != "CASH" and raw_desired_asset != active_asset:
            active_asset = "CASH"
            active_weight = 0.0
            hold_days = 0
            rolling_peak = 0.0

        if active_asset == "CASH" and raw_desired_asset != "CASH":
            active_asset = raw_desired_asset
            active_weight = raw_desired_weight
            hold_days = 0
            rolling_peak = float(row[asset_price_column(active_asset)])
            if active_asset == "TQQQ":
                active_max_hold_days, active_trailing_lookback_days, active_trailing_drawdown_pct = candidate_long_profile
            else:
                active_max_hold_days, active_trailing_lookback_days, active_trailing_drawdown_pct = short_exit_profile

        position = active_asset
        daily_ret = 0.0
        trailing_exit = False
        time_exit = False
        if idx > 0 and position != "CASH":
            price_col = asset_price_column(position)
            prev_close = float(frame.iloc[idx - 1][price_col])
            cur_close = float(row[price_col])
            asset_ret = cur_close / prev_close - 1.0 if prev_close > 0 else 0.0
            daily_ret = asset_ret * active_weight
            hold_days += 1
            rolling_peak = max(rolling_peak, cur_close)
            if (
                active_trailing_lookback_days > 0
                and active_trailing_drawdown_pct > 0
                and hold_days >= active_trailing_lookback_days
                and rolling_peak > 0
            ):
                drawdown_from_peak = (rolling_peak - cur_close) / rolling_peak * 100.0
                trailing_exit = drawdown_from_peak >= active_trailing_drawdown_pct
            if active_max_hold_days > 0 and hold_days >= active_max_hold_days:
                time_exit = True
            if trailing_exit or time_exit:
                active_asset = "CASH"
                active_weight = 0.0
                hold_days = 0
                rolling_peak = 0.0

        if idx > 0 and position != previous_position:
            daily_ret -= float(switch_cost_bps) / 10000.0
        previous_position = position
        capital *= 1.0 + daily_ret
        rows.append(
            {
                "date": row["date"],
                "equity": capital,
                "position": position,
                "weight": active_weight if position != "CASH" else 0.0,
                "daily_return": daily_ret,
            }
        )

    equity = pd.DataFrame(rows)
    yearly = annual_returns(equity)
    position_counts = equity["position"].value_counts().to_dict()
    total_return_pct = round((capital / initial_capital - 1.0) * 100.0, 2)
    max_dd = round(max_drawdown_pct(equity["equity"]), 2)
    weighted_days = float((equity["weight"]).sum())
    score = total_return_pct - max_dd * 2.0 + float(yearly.get("2026", 0.0) or 0.0) * 0.8
    return {
        "params": {
            "long_weight": float(long_weight),
            "short_weight": float(short_weight),
        },
        "summary": {
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd,
            "yearly_returns_pct": yearly,
            "position_counts": position_counts,
            "weighted_days": round(weighted_days, 2),
            "score": round(score, 4),
        },
    }


def candidate_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    summary = item["summary"]
    yearly = summary.get("yearly_returns_pct", {})
    return (
        float(summary.get("score", 0.0)),
        float(summary.get("total_return_pct", 0.0)),
        float(yearly.get("2026", 0.0)),
        -float(summary.get("max_drawdown_pct", 0.0)),
    )


def main() -> None:
    args = parse_args()
    qqq = load_df(Path(args.qqq))
    tqqq = load_df(Path(args.tqqq))
    sqqq = load_df(Path(args.sqqq))
    spy = load_df(Path(args.spy))
    ixic = load_df(Path(args.ixic))
    vix = load_df(Path(args.vix))
    frame = prepare_frame(
        qqq,
        tqqq,
        sqqq,
        spy,
        ixic,
        vix,
        int(args.entry_fast_window),
        int(args.entry_slow_window),
    )

    baseline = run_weighted_candidate(
        frame,
        long_profile_name=str(args.long_profile_name),
        short_rule_name=str(args.short_rule_name),
        short_exit_profile=(
            int(args.short_max_hold_days),
            int(args.short_trailing_lookback_days),
            float(args.short_trailing_drawdown_pct),
        ),
        long_weight=1.0,
        short_weight=1.0,
        initial_capital=float(args.initial_capital),
        switch_cost_bps=float(args.switch_cost_bps),
    )
    baseline_dd = float(baseline["summary"]["max_drawdown_pct"])

    results: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for long_weight in parse_float_list(args.long_weight_values):
        for short_weight in parse_float_list(args.short_weight_values):
            item = run_weighted_candidate(
                frame,
                long_profile_name=str(args.long_profile_name),
                short_rule_name=str(args.short_rule_name),
                short_exit_profile=(
                    int(args.short_max_hold_days),
                    int(args.short_trailing_lookback_days),
                    float(args.short_trailing_drawdown_pct),
                ),
                long_weight=float(long_weight),
                short_weight=float(short_weight),
                initial_capital=float(args.initial_capital),
                switch_cost_bps=float(args.switch_cost_bps),
            )
            summary = item["summary"]
            dd_delta = round(float(summary["max_drawdown_pct"]) - baseline_dd, 2)
            item["summary"]["dd_delta_pct"] = dd_delta
            results.append(item)
            if dd_delta <= float(args.max_dd_delta_pct):
                accepted.append(item)

    ranked = sorted(accepted, key=candidate_sort_key, reverse=True)
    payload = {
        "reference": {
            "description": "Stable + bearish_score5 full-weight baseline",
            "baseline": baseline,
            "max_dd_delta_pct": float(args.max_dd_delta_pct),
        },
        "scan_size": len(results),
        "accepted_size": len(accepted),
        "top_candidates": ranked[: int(args.top)],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    print("baseline", baseline["summary"])
    for item in ranked[: min(int(args.top), 12)]:
        params = item["params"]
        summary = item["summary"]
        yearly = summary["yearly_returns_pct"]
        print(
            f"long_w={params['long_weight']:.2f} short_w={params['short_weight']:.2f} "
            f"full={summary['total_return_pct']:.2f}% dd={summary['max_drawdown_pct']:.2f}% dd_delta={summary['dd_delta_pct']:.2f}% "
            f"2026={float(yearly.get('2026', 0.0)):.2f}% score={summary['score']:.2f}"
        )


if __name__ == "__main__":
    main()
