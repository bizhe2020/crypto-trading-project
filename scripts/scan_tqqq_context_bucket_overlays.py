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

from scripts.audit_tqqq_cash_regime_context import build_regime_frame, load_df  # noqa: E402
from scripts.replay_tqqq_sqqq_trend_baseline import max_drawdown_pct  # noqa: E402


DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "tqqq_context_bucket_overlay_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan BTC-style context/bucket overlays for Stable TQQQ + selective SQQQ."
    )
    parser.add_argument("--tqqq", default=str(DEFAULT_PUBLIC_DIR / "TQQQ-1d.feather"))
    parser.add_argument("--sqqq", default=str(DEFAULT_PUBLIC_DIR / "SQQQ-1d.feather"))
    parser.add_argument("--qqq", default=str(DEFAULT_PUBLIC_DIR / "QQQ-1d.feather"))
    parser.add_argument("--spy", default=str(DEFAULT_PUBLIC_DIR / "SPY-1d.feather"))
    parser.add_argument("--ixic", default=str(DEFAULT_PUBLIC_DIR / "^IXIC-1d.feather"))
    parser.add_argument("--vix", default=str(DEFAULT_PUBLIC_DIR / "^VIX-1d.feather"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--entry-fast-window", type=int, default=25)
    parser.add_argument("--entry-slow-window", type=int, default=200)
    parser.add_argument("--switch-cost-bps", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


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


def prepare_frame(
    qqq: pd.DataFrame,
    tqqq: pd.DataFrame,
    sqqq: pd.DataFrame,
    spy: pd.DataFrame,
    ixic: pd.DataFrame,
    vix: pd.DataFrame,
    fast_window: int,
    slow_window: int,
) -> pd.DataFrame:
    frame = build_regime_frame(qqq, tqqq, spy, ixic, vix, fast_window, slow_window)
    qqq_extra = qqq[["date", "open", "high", "low", "close", "volume"]].rename(
        columns={
            "open": "qqq_open",
            "high": "qqq_high",
            "low": "qqq_low",
            "close": "qqq_close_raw",
            "volume": "qqq_volume",
        }
    )
    sqqq_frame = sqqq[["date", "open", "high", "low", "close"]].rename(
        columns={
            "open": "sqqq_open",
            "high": "sqqq_high",
            "low": "sqqq_low",
            "close": "sqqq_close",
        }
    )
    frame = frame.merge(qqq_extra, on="date", how="inner")
    frame = frame.merge(sqqq_frame, on="date", how="inner")
    frame = frame.sort_values("date").reset_index(drop=True)

    frame["trend_state"] = 0
    frame.loc[frame["fast_ma"] > frame["slow_ma"], "trend_state"] = 1
    frame.loc[frame["fast_ma"] < frame["slow_ma"], "trend_state"] = -1
    frame["planned_trend"] = frame["trend_state"].shift(1).fillna(0).astype(int)

    prev_close = frame["qqq_close_raw"].shift(1)
    tr = pd.concat(
        [
            frame["qqq_high"] - frame["qqq_low"],
            (frame["qqq_high"] - prev_close).abs(),
            (frame["qqq_low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["qqq_atr_14"] = tr.rolling(14).mean()
    frame["qqq_atr_pct_14"] = frame["qqq_atr_14"] / frame["qqq_close_raw"]
    frame["qqq_volume_ratio_20"] = frame["qqq_volume"] / frame["qqq_volume"].rolling(20).mean()
    frame["qqq_mom_10"] = frame["qqq_close_raw"] / frame["qqq_close_raw"].shift(10) - 1.0
    frame["qqq_mom_20"] = frame["qqq_close_raw"] / frame["qqq_close_raw"].shift(20) - 1.0
    frame["qqq_dist_slow"] = frame["qqq_close_raw"] / frame["slow_ma"] - 1.0
    frame["qqq_prev20_high"] = frame["qqq_high"].shift(1).rolling(20).max()
    frame["qqq_prev20_low"] = frame["qqq_low"].shift(1).rolling(20).min()
    frame["qqq_pullback_20"] = frame["qqq_close_raw"] / frame["qqq_prev20_high"] - 1.0
    frame["qqq_breakout_20"] = frame["qqq_close_raw"] > frame["qqq_prev20_high"]
    frame["qqq_breakdown_20"] = frame["qqq_close_raw"] < frame["qqq_prev20_low"]
    frame["qqq_sweep_reclaim_20"] = (frame["qqq_low"] < frame["qqq_prev20_low"]) & (
        frame["qqq_close_raw"] > frame["qqq_prev20_low"]
    )
    frame["qqq_breakout_fail_20"] = (frame["qqq_high"] > frame["qqq_prev20_high"]) & (
        frame["qqq_close_raw"] < frame["qqq_prev20_high"]
    )
    frame["qqq_compression_60"] = frame["qqq_atr_pct_14"] < (frame["qqq_atr_pct_14"].rolling(60).mean() * 0.9)

    frame["long_context_score"] = 0
    frame.loc[frame["vix_label"].eq("vix_low"), "long_context_score"] += 1
    frame.loc[frame["rel_strength_label"].eq("qqq_strong"), "long_context_score"] += 1
    frame.loc[frame["qqq_mom_20"] > 0.06, "long_context_score"] += 1
    frame.loc[frame["qqq_dist_slow"] > 0.04, "long_context_score"] += 1
    frame.loc[frame["qqq_volume_ratio_20"] > 1.05, "long_context_score"] += 1
    frame.loc[frame["qqq_breakout_20"] | frame["qqq_sweep_reclaim_20"], "long_context_score"] += 1

    frame["short_context_score"] = 0
    frame.loc[frame["ixic_trend_label"].eq("ixic_down"), "short_context_score"] += 1
    frame.loc[frame["vix_label"].isin(["vix_high", "vix_extreme"]), "short_context_score"] += 1
    frame.loc[frame["rel_strength_label"].eq("qqq_weak"), "short_context_score"] += 1
    frame.loc[frame["qqq_mom_20"] < -0.05, "short_context_score"] += 1
    frame.loc[frame["qqq_dist_slow"] < -0.04, "short_context_score"] += 1
    frame.loc[frame["qqq_volume_ratio_20"] > 1.05, "short_context_score"] += 1
    frame.loc[frame["qqq_breakdown_20"] | frame["qqq_breakout_fail_20"], "short_context_score"] += 1
    return frame


def select_long_profile(row: pd.Series, profile_name: str) -> tuple[int, int, float]:
    score = int(row.get("long_context_score", 0) or 0)
    if profile_name == "stable_base":
        return 90, 30, 12.0
    if profile_name == "weak_tight_score2":
        return (90, 10, 8.0) if score <= 2 else (90, 30, 12.0)
    if profile_name == "weak_tight_score3":
        return (90, 10, 8.0) if score <= 3 else (90, 30, 12.0)
    if profile_name == "three_bucket_score":
        if score <= 2:
            return 90, 10, 8.0
        if score <= 4:
            return 90, 20, 10.0
        return 90, 30, 12.0
    if profile_name == "breakout_loose_else_mid":
        if bool(row.get("qqq_breakout_20")) or bool(row.get("qqq_sweep_reclaim_20")) or score >= 5:
            return 90, 30, 12.0
        return 90, 20, 10.0
    raise ValueError(f"Unsupported long profile: {profile_name}")


def allow_short(row: pd.Series, rule_name: str) -> bool:
    score = int(row.get("short_context_score", 0) or 0)
    rel_weak_vix_high = bool(row.get("rel_strength_label") == "qqq_weak") and bool(
        row.get("vix_label") in ["vix_high", "vix_extreme"]
    )
    if rule_name == "off":
        return False
    if rule_name == "narrow_base":
        return rel_weak_vix_high
    if rule_name == "narrow_score4":
        return rel_weak_vix_high and score >= 4
    if rule_name == "narrow_score5":
        return rel_weak_vix_high and score >= 5
    if rule_name == "bearish_score5":
        return score >= 5 and bool(row.get("vix_label") in ["vix_high", "vix_extreme"])
    if rule_name == "breakdown_confirm":
        return rel_weak_vix_high and bool(row.get("qqq_breakdown_20") or row.get("qqq_breakout_fail_20"))
    raise ValueError(f"Unsupported short rule: {rule_name}")


def run_candidate(
    frame: pd.DataFrame,
    *,
    long_profile_name: str,
    short_rule_name: str,
    short_exit_profile: tuple[int, int, float],
    initial_capital: float,
    switch_cost_bps: float,
) -> dict[str, Any]:
    long_mask = frame["vix_label"].isin(["vix_low", "vix_normal"]) & frame["ixic_trend_label"].eq("ixic_up")
    capital = initial_capital
    previous_position = "CASH"
    active_asset = "CASH"
    active_max_hold_days = 0
    active_trailing_lookback_days = 0
    active_trailing_drawdown_pct = 0.0
    hold_days = 0
    rolling_peak = 0.0
    entry_equity = capital
    entry_date: pd.Timestamp | None = None
    entry_price: float | None = None
    exit_override_asset: str | None = None
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    for idx, row in frame.iterrows():
        raw_desired_asset = "CASH"
        desired_long = int(row["planned_trend"]) > 0 and bool(long_mask.iloc[idx])
        desired_short = int(row["planned_trend"]) < 0 and allow_short(row, short_rule_name)
        candidate_long_profile = select_long_profile(row, long_profile_name)
        if desired_long:
            raw_desired_asset = "TQQQ"
        elif desired_short:
            raw_desired_asset = "SQQQ"

        if active_asset != "CASH" and raw_desired_asset != active_asset:
            exit_row = frame.iloc[max(idx - 1, 0)]
            exit_price = float(exit_row[asset_price_column(active_asset)])
            if entry_date is not None and entry_price is not None:
                trades.append(
                    {
                        "asset": active_asset,
                        "entry_date": str(entry_date.date()),
                        "exit_date": str(pd.Timestamp(exit_row["date"]).date()),
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(exit_price, 4),
                        "trade_return_pct": round((capital / entry_equity - 1.0) * 100.0, 2),
                        "exit_reason": "signal_flip",
                        "hold_days": int(hold_days),
                    }
                )
            active_asset = "CASH"
            hold_days = 0
            rolling_peak = 0.0
            if exit_override_asset == raw_desired_asset:
                raw_desired_asset = "CASH"

        desired_asset = raw_desired_asset
        if desired_asset != "CASH" and exit_override_asset == desired_asset:
            desired_asset = "CASH"

        if active_asset == "CASH" and desired_asset != "CASH":
            active_asset = desired_asset
            hold_days = 0
            rolling_peak = float(row[asset_price_column(active_asset)])
            entry_equity = capital
            entry_date = pd.Timestamp(row["date"])
            entry_price = float(row[asset_price_column(active_asset)])
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
            daily_ret = cur_close / prev_close - 1.0 if prev_close > 0 else 0.0
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
                capital_after_today = capital * (1.0 + daily_ret)
                if entry_date is not None and entry_price is not None:
                    trades.append(
                        {
                            "asset": position,
                            "entry_date": str(entry_date.date()),
                            "exit_date": str(pd.Timestamp(row["date"]).date()),
                            "entry_price": round(entry_price, 4),
                            "exit_price": round(float(row[price_col]), 4),
                            "trade_return_pct": round((capital_after_today / entry_equity - 1.0) * 100.0, 2),
                            "exit_reason": "trailing" if trailing_exit else "time",
                            "hold_days": int(hold_days),
                        }
                    )
                active_asset = "CASH"
                exit_override_asset = position
                hold_days = 0
                rolling_peak = 0.0
                entry_date = None
                entry_price = None
                entry_equity = capital_after_today
        if idx > 0 and position != previous_position:
            daily_ret -= float(switch_cost_bps) / 10000.0
        previous_position = position
        capital *= 1.0 + daily_ret

        if position == "CASH" and raw_desired_asset != "CASH" and exit_override_asset == raw_desired_asset:
            exit_override_asset = None
        elif raw_desired_asset == "CASH":
            exit_override_asset = None

        rows.append({"date": row["date"], "equity": capital, "position": position, "daily_return": daily_ret})

    if active_asset != "CASH" and entry_date is not None and entry_price is not None:
        exit_row = frame.iloc[-1]
        trades.append(
            {
                "asset": active_asset,
                "entry_date": str(entry_date.date()),
                "exit_date": str(pd.Timestamp(exit_row["date"]).date()),
                "entry_price": round(entry_price, 4),
                "exit_price": round(float(exit_row[asset_price_column(active_asset)]), 4),
                "trade_return_pct": round((capital / entry_equity - 1.0) * 100.0, 2),
                "exit_reason": "end_of_data",
                "hold_days": int(hold_days),
            }
        )

    equity = pd.DataFrame(rows)
    yearly = annual_returns(equity)
    trade_returns = pd.Series([float(item["trade_return_pct"]) for item in trades], dtype=float) if trades else pd.Series(dtype=float)
    tqqq_trades = [item for item in trades if item["asset"] == "TQQQ"]
    sqqq_trades = [item for item in trades if item["asset"] == "SQQQ"]
    total_return_pct = round((capital / initial_capital - 1.0) * 100.0, 2)
    max_dd = round(max_drawdown_pct(equity["equity"]), 2)
    score = total_return_pct - max_dd * 2.0 + len(trades) * 30.0 + float(yearly.get("2026", 0.0)) * 0.8
    return {
        "candidate": {
            "long_profile_name": long_profile_name,
            "short_rule_name": short_rule_name,
            "short_exit_profile": {
                "max_hold_days": int(short_exit_profile[0]),
                "trailing_lookback_days": int(short_exit_profile[1]),
                "trailing_drawdown_pct": float(short_exit_profile[2]),
            },
        },
        "summary": {
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd,
            "yearly_returns_pct": yearly,
            "trades": int(len(trades)),
            "tqqq_trades": int(len(tqqq_trades)),
            "sqqq_trades": int(len(sqqq_trades)),
            "mean_trade_return_pct": round(float(trade_returns.mean()), 2) if not trade_returns.empty else 0.0,
            "median_trade_return_pct": round(float(trade_returns.median()), 2) if not trade_returns.empty else 0.0,
            "positive_trade_ratio_pct": round(float((trade_returns > 0).mean() * 100.0), 2) if not trade_returns.empty else 0.0,
            "score": round(score, 4),
        },
        "trades": trades,
        "sample_trades": trades[:12],
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

    long_profiles = [
        "stable_base",
        "weak_tight_score2",
        "weak_tight_score3",
        "three_bucket_score",
        "breakout_loose_else_mid",
    ]
    short_rules = [
        "off",
        "narrow_base",
        "narrow_score4",
        "narrow_score5",
        "bearish_score5",
        "breakdown_confirm",
    ]
    short_exit_profiles = [
        (20, 5, 6.0),
        (20, 5, 8.0),
        (20, 10, 8.0),
        (40, 10, 8.0),
    ]

    results: list[dict[str, Any]] = []
    for long_profile_name in long_profiles:
        for short_rule_name in short_rules:
            for short_exit_profile in short_exit_profiles:
                item = run_candidate(
                    frame,
                    long_profile_name=long_profile_name,
                    short_rule_name=short_rule_name,
                    short_exit_profile=short_exit_profile,
                    initial_capital=float(args.initial_capital),
                    switch_cost_bps=float(args.switch_cost_bps),
                )
                results.append(item)

    ranked = sorted(results, key=candidate_sort_key, reverse=True)
    payload = {
        "reference": {
            "entry_fast_window": int(args.entry_fast_window),
            "entry_slow_window": int(args.entry_slow_window),
            "switch_cost_bps": float(args.switch_cost_bps),
            "description": "Stable TQQQ base with BTC-style context score / bucket exits / selective SQQQ.",
        },
        "scan_size": len(results),
        "top_candidates": ranked[: int(args.top)],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    for item in ranked[: min(int(args.top), 12)]:
        summary = item["summary"]
        candidate = item["candidate"]
        yearly = summary["yearly_returns_pct"]
        print(
            f"score={summary['score']:.2f} full={summary['total_return_pct']:.2f}% dd={summary['max_drawdown_pct']:.2f}% "
            f"trades={summary['trades']} tqqq={summary['tqqq_trades']} sqqq={summary['sqqq_trades']} "
            f"2026={float(yearly.get('2026', 0.0)):.2f}% long={candidate['long_profile_name']} "
            f"short={candidate['short_rule_name']} s_exit={candidate['short_exit_profile']}"
        )


if __name__ == "__main__":
    main()
