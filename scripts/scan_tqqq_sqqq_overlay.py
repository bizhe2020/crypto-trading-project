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
from scripts.scan_tqqq_cash_exit_profiles import parse_float_list, parse_int_list  # noqa: E402


DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "tqqq_sqqq_overlay_scan.json"


LONG_PRESETS: dict[str, dict[str, Any]] = {
    "stable": {
        "mask": "vix_ixic",
        "max_hold_days": 90,
        "trailing_lookback_days": 30,
        "trailing_drawdown_pct": 12.0,
    },
    "middle": {
        "mask": "vix_ixic",
        "max_hold_days": 90,
        "trailing_lookback_days": 10,
        "trailing_drawdown_pct": 10.0,
    },
    "broader": {
        "mask": "ixic_filter",
        "max_hold_days": 90,
        "trailing_lookback_days": 10,
        "trailing_drawdown_pct": 8.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan selective SQQQ overlays on top of TQQQ trend presets.")
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
    parser.add_argument("--long-preset-values", default="stable,middle,broader")
    parser.add_argument("--short-mask-values", default="off,trend_down,ixic_down,rel_weak,vix_high,ixic_down_rel_weak,ixic_down_vix_high,rel_weak_vix_high,all_bearish")
    parser.add_argument("--short-max-hold-days-values", default="20,40,60")
    parser.add_argument("--short-trailing-lookback-days-values", default="5,10")
    parser.add_argument("--short-trailing-drawdown-pct-values", default="6,8,10")
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


def build_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    vix_allow = frame["vix_label"].isin(["vix_low", "vix_normal"])
    vix_bear = frame["vix_label"].isin(["vix_high", "vix_extreme"])
    ixic_up = frame["ixic_trend_label"].eq("ixic_up")
    ixic_down = frame["ixic_trend_label"].eq("ixic_down")
    rel_weak = frame["rel_strength_label"].eq("qqq_weak")
    return {
        "off": pd.Series([False] * len(frame), index=frame.index),
        "trend_down": pd.Series([True] * len(frame), index=frame.index),
        "ixic_down": ixic_down,
        "rel_weak": rel_weak,
        "vix_high": vix_bear,
        "ixic_down_rel_weak": ixic_down & rel_weak,
        "ixic_down_vix_high": ixic_down & vix_bear,
        "rel_weak_vix_high": rel_weak & vix_bear,
        "all_bearish": ixic_down & rel_weak & vix_bear,
        "base": pd.Series([True] * len(frame), index=frame.index),
        "vix_filter": vix_allow,
        "ixic_filter": ixic_up,
        "vix_ixic": vix_allow & ixic_up,
    }


def parse_name_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


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
    sqqq_frame = sqqq[["date", "open", "high", "low", "close"]].rename(
        columns={
            "open": "sqqq_open",
            "high": "sqqq_high",
            "low": "sqqq_low",
            "close": "sqqq_close",
        }
    )
    frame = frame.merge(sqqq_frame, on="date", how="inner")
    frame["trend_state"] = 0
    frame.loc[frame["fast_ma"] > frame["slow_ma"], "trend_state"] = 1
    frame.loc[frame["fast_ma"] < frame["slow_ma"], "trend_state"] = -1
    frame["planned_trend"] = frame["trend_state"].shift(1).fillna(0).astype(int)
    return frame.reset_index(drop=True)


def annualized_trade_stats(trades: list[dict[str, Any]]) -> dict[str, float]:
    if not trades:
        return {
            "mean_trade_return_pct": 0.0,
            "median_trade_return_pct": 0.0,
            "positive_trade_ratio_pct": 0.0,
            "best_trade_return_pct": 0.0,
            "worst_trade_return_pct": 0.0,
        }
    series = pd.Series([float(item["trade_return_pct"]) for item in trades], dtype=float)
    return {
        "mean_trade_return_pct": round(float(series.mean()), 2),
        "median_trade_return_pct": round(float(series.median()), 2),
        "positive_trade_ratio_pct": round(float((series > 0).mean() * 100.0), 2),
        "best_trade_return_pct": round(float(series.max()), 2),
        "worst_trade_return_pct": round(float(series.min()), 2),
    }


def asset_price_column(asset: str) -> str:
    if asset == "TQQQ":
        return "tqqq_close"
    if asset == "SQQQ":
        return "sqqq_close"
    raise ValueError(f"Unsupported asset: {asset}")


def summarize_trade_sides(trades: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "tqqq_trades": int(sum(1 for item in trades if item.get("asset") == "TQQQ")),
        "sqqq_trades": int(sum(1 for item in trades if item.get("asset") == "SQQQ")),
    }


def run_overlay_candidate(
    frame: pd.DataFrame,
    *,
    long_mask: pd.Series,
    short_mask: pd.Series,
    long_max_hold_days: int,
    long_trailing_lookback_days: int,
    long_trailing_drawdown_pct: float,
    short_max_hold_days: int,
    short_trailing_lookback_days: int,
    short_trailing_drawdown_pct: float,
    initial_capital: float,
    switch_cost_bps: float,
) -> dict[str, Any]:
    long_mask = long_mask.reset_index(drop=True)
    short_mask = short_mask.reset_index(drop=True)
    capital = initial_capital
    previous_position = "CASH"
    active_asset = "CASH"
    hold_days = 0
    rolling_peak = 0.0
    entry_equity = capital
    entry_date: pd.Timestamp | None = None
    entry_price: float | None = None
    exit_override_asset: str | None = None
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    def close_open_trade(exit_idx: int, exit_asset: str, exit_reason: str) -> None:
        nonlocal entry_date, entry_price, entry_equity, trades
        if entry_date is None or entry_price is None:
            return
        exit_row = frame.iloc[exit_idx]
        exit_price = float(exit_row[asset_price_column(exit_asset)])
        trades.append(
            {
                "asset": exit_asset,
                "entry_date": str(entry_date.date()),
                "exit_date": str(pd.Timestamp(exit_row["date"]).date()),
                "entry_price": round(float(entry_price), 4),
                "exit_price": round(exit_price, 4),
                "trade_return_pct": round((capital / entry_equity - 1.0) * 100.0, 2),
                "exit_reason": exit_reason,
                "hold_days": int(hold_days),
            }
        )
        entry_date = None
        entry_price = None
        entry_equity = capital

    for idx, row in frame.iterrows():
        raw_desired_asset = "CASH"
        if int(row["planned_trend"]) > 0 and bool(long_mask.iloc[idx]):
            raw_desired_asset = "TQQQ"
        elif int(row["planned_trend"]) < 0 and bool(short_mask.iloc[idx]):
            raw_desired_asset = "SQQQ"

        if active_asset != "CASH" and raw_desired_asset != active_asset:
            close_open_trade(idx - 1, active_asset, "signal_flip")
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
            if position == "TQQQ":
                max_hold_days = long_max_hold_days
                trailing_lookback_days = long_trailing_lookback_days
                trailing_drawdown_pct = long_trailing_drawdown_pct
            else:
                max_hold_days = short_max_hold_days
                trailing_lookback_days = short_trailing_lookback_days
                trailing_drawdown_pct = short_trailing_drawdown_pct
            if trailing_lookback_days > 0 and trailing_drawdown_pct > 0 and hold_days >= trailing_lookback_days and rolling_peak > 0:
                drawdown_from_peak = (rolling_peak - cur_close) / rolling_peak * 100.0
                trailing_exit = drawdown_from_peak >= trailing_drawdown_pct
            if max_hold_days > 0 and hold_days >= max_hold_days:
                time_exit = True
            if trailing_exit or time_exit:
                if idx > 0:
                    capital_after_today = capital * (1.0 + daily_ret)
                else:
                    capital_after_today = capital
                if entry_date is not None and entry_price is not None:
                    trades.append(
                        {
                            "asset": position,
                            "entry_date": str(entry_date.date()),
                            "exit_date": str(pd.Timestamp(row["date"]).date()),
                            "entry_price": round(float(entry_price), 4),
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

        rows.append(
            {
                "date": row["date"],
                "equity": capital,
                "position": position,
                "daily_return": daily_ret,
                "vix_label": row["vix_label"],
                "rel_strength_label": row["rel_strength_label"],
                "ixic_trend_label": row["ixic_trend_label"],
            }
        )

    if active_asset != "CASH":
        close_open_trade(len(frame) - 1, active_asset, "end_of_data")

    equity = pd.DataFrame(rows)
    yearly = annual_returns(equity)
    total_return_pct = round((capital / initial_capital - 1.0) * 100.0, 2)
    max_dd = round(max_drawdown_pct(equity["equity"]), 2)
    invested_days = int((equity["position"] != "CASH").sum())
    trades_count = int(len(trades))
    side_counts = summarize_trade_sides(trades)
    score = (
        total_return_pct
        - max_dd * 2.0
        + trades_count * 35.0
        + float(yearly.get("2022", 0.0)) * 1.2
        + float(yearly.get("2026", 0.0)) * 0.8
    )
    return {
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd,
        "yearly_returns_pct": yearly,
        "trades": trades_count,
        "invested_days": invested_days,
        "invested_ratio_pct": round(invested_days / len(equity) * 100.0, 2) if len(equity) else 0.0,
        "position_counts": equity["position"].value_counts().to_dict(),
        "trade_sides": side_counts,
        "trade_stats": annualized_trade_stats(trades),
        "score": round(score, 4),
        "sample_trades": trades[:10],
    }


def candidate_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    yearly = item.get("yearly_returns_pct", {})
    return (
        float(item.get("score", 0.0)),
        float(item.get("total_return_pct", 0.0)),
        float(yearly.get("2026", 0.0)),
        -float(item.get("max_drawdown_pct", 0.0)),
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
    masks = build_masks(frame)

    candidates: list[dict[str, Any]] = []
    long_preset_names = parse_name_list(args.long_preset_values)
    short_mask_names = parse_name_list(args.short_mask_values)
    for long_preset_name in long_preset_names:
        long_preset = LONG_PRESETS[long_preset_name]
        long_mask_name = str(long_preset["mask"])
        long_mask = masks[long_mask_name]
        for short_mask_name in short_mask_names:
            short_mask = masks[short_mask_name]
            for short_max_hold_days in parse_int_list(args.short_max_hold_days_values):
                for short_trailing_lookback_days in parse_int_list(args.short_trailing_lookback_days_values):
                    for short_trailing_drawdown_pct in parse_float_list(args.short_trailing_drawdown_pct_values):
                        item = run_overlay_candidate(
                            frame,
                            long_mask=long_mask,
                            short_mask=short_mask,
                            long_max_hold_days=int(long_preset["max_hold_days"]),
                            long_trailing_lookback_days=int(long_preset["trailing_lookback_days"]),
                            long_trailing_drawdown_pct=float(long_preset["trailing_drawdown_pct"]),
                            short_max_hold_days=int(short_max_hold_days),
                            short_trailing_lookback_days=int(short_trailing_lookback_days),
                            short_trailing_drawdown_pct=float(short_trailing_drawdown_pct),
                            initial_capital=float(args.initial_capital),
                            switch_cost_bps=float(args.switch_cost_bps),
                        )
                        item["candidate"] = {
                            "long_preset": long_preset_name,
                            "long_mask": long_mask_name,
                            "long_max_hold_days": int(long_preset["max_hold_days"]),
                            "long_trailing_lookback_days": int(long_preset["trailing_lookback_days"]),
                            "long_trailing_drawdown_pct": float(long_preset["trailing_drawdown_pct"]),
                            "short_mask": short_mask_name,
                            "short_max_hold_days": int(short_max_hold_days),
                            "short_trailing_lookback_days": int(short_trailing_lookback_days),
                            "short_trailing_drawdown_pct": float(short_trailing_drawdown_pct),
                        }
                        candidates.append(item)

    ranked = sorted(candidates, key=candidate_sort_key, reverse=True)
    payload = {
        "entry_reference": {
            "fast_window": int(args.entry_fast_window),
            "slow_window": int(args.entry_slow_window),
            "switch_cost_bps": float(args.switch_cost_bps),
            "description": "QQQ MA trend, TQQQ/SQQQ/CASH selective overlay scan",
        },
        "long_presets": {name: LONG_PRESETS[name] for name in long_preset_names},
        "scan_size": len(candidates),
        "top_candidates": ranked[: int(args.top)],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    for item in ranked[: min(int(args.top), 10)]:
        candidate = item["candidate"]
        yearly = item["yearly_returns_pct"]
        print(
            f"score={item['score']:.2f} full={item['total_return_pct']:.2f}% dd={item['max_drawdown_pct']:.2f}% "
            f"trades={item['trades']} tqqq={item['trade_sides']['tqqq_trades']} sqqq={item['trade_sides']['sqqq_trades']} "
            f"2022={float(yearly.get('2022', 0.0)):.2f}% 2026={float(yearly.get('2026', 0.0)):.2f}% "
            f"long={candidate['long_preset']} short={candidate['short_mask']} "
            f"s_hold={candidate['short_max_hold_days']} s_tlb={candidate['short_trailing_lookback_days']} s_tdd={candidate['short_trailing_drawdown_pct']}"
        )


if __name__ == "__main__":
    main()
