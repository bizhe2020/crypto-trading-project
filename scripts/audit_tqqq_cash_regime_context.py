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

from scripts.replay_tqqq_sqqq_trend_baseline import load_df, max_drawdown_pct
from scripts.scan_tqqq_cash_exit_profiles import build_entry_frame


DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "tqqq_cash_regime_context_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit regime context for the current TQQQ/CASH main candidate.")
    parser.add_argument("--tqqq", default=str(DEFAULT_PUBLIC_DIR / "TQQQ-1d.feather"))
    parser.add_argument("--qqq", default=str(DEFAULT_PUBLIC_DIR / "QQQ-1d.feather"))
    parser.add_argument("--spy", default=str(DEFAULT_PUBLIC_DIR / "SPY-1d.feather"))
    parser.add_argument("--ixic", default=str(DEFAULT_PUBLIC_DIR / "^IXIC-1d.feather"))
    parser.add_argument("--vix", default=str(DEFAULT_PUBLIC_DIR / "^VIX-1d.feather"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--entry-fast-window", type=int, default=25)
    parser.add_argument("--entry-slow-window", type=int, default=150)
    parser.add_argument("--max-hold-days", type=int, default=90)
    parser.add_argument("--trailing-lookback-days", type=int, default=10)
    parser.add_argument("--trailing-drawdown-pct", type=float, default=12.0)
    parser.add_argument("--switch-cost-bps", type=float, default=5.0)
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


def label_vix(vix_close: float) -> str:
    if vix_close >= 30:
        return "vix_extreme"
    if vix_close >= 22:
        return "vix_high"
    if vix_close >= 16:
        return "vix_normal"
    return "vix_low"


def label_rel_strength(value: float) -> str:
    if value >= 0.03:
        return "qqq_strong"
    if value <= -0.03:
        return "qqq_weak"
    return "qqq_neutral"


def with_session_day(df: pd.DataFrame, close_column: str) -> pd.DataFrame:
    frame = df[["date", close_column]].copy()
    frame["session_day"] = pd.to_datetime(frame["date"], utc=True).dt.normalize()
    frame = frame.sort_values("session_day").drop_duplicates("session_day", keep="last").reset_index(drop=True)
    return frame


def build_regime_frame(qqq: pd.DataFrame, tqqq: pd.DataFrame, spy: pd.DataFrame, ixic: pd.DataFrame, vix: pd.DataFrame, fast_window: int, slow_window: int) -> pd.DataFrame:
    frame = build_entry_frame(qqq, tqqq, fast_window, slow_window)
    frame["session_day"] = pd.to_datetime(frame["date"], utc=True).dt.normalize()
    spy_frame = with_session_day(spy, "close").rename(columns={"close": "spy_close"})
    ixic_frame = with_session_day(ixic, "close").rename(columns={"close": "ixic_close"})
    vix_frame = with_session_day(vix, "close").rename(columns={"close": "vix_close"})
    merged = frame.merge(spy_frame[["session_day", "spy_close"]], on="session_day", how="left")
    merged = merged.merge(ixic_frame[["session_day", "ixic_close"]], on="session_day", how="left")
    merged = merged.merge(vix_frame[["session_day", "vix_close"]], on="session_day", how="left")
    merged["qqq_spy_rel_20"] = (merged["qqq_close"] / merged["qqq_close"].shift(20)) - (merged["spy_close"] / merged["spy_close"].shift(20))
    merged["ixic_ma_50"] = merged["ixic_close"].rolling(50).mean()
    merged["ixic_trend_label"] = "unknown"
    ixic_ready = merged["ixic_close"].notna() & merged["ixic_ma_50"].notna()
    merged.loc[ixic_ready & (merged["ixic_close"] > merged["ixic_ma_50"]), "ixic_trend_label"] = "ixic_up"
    merged.loc[ixic_ready & (merged["ixic_close"] <= merged["ixic_ma_50"]), "ixic_trend_label"] = "ixic_down"
    merged["vix_label"] = merged["vix_close"].map(lambda value: label_vix(float(value)) if pd.notna(value) else "unknown")
    merged["rel_strength_label"] = merged["qqq_spy_rel_20"].map(lambda value: label_rel_strength(float(value)) if pd.notna(value) else "unknown")
    return merged


def run_main_candidate(frame: pd.DataFrame, initial_capital: float, switch_cost_bps: float, max_hold_days: int, trailing_lookback_days: int, trailing_drawdown_pct: float) -> pd.DataFrame:
    capital = initial_capital
    previous_position = "CASH"
    active = False
    hold_days = 0
    rolling_peak = 0.0
    exit_override = False
    rows: list[dict[str, Any]] = []

    for idx, row in frame.iterrows():
        desired_active = int(row["planned_position"]) > 0
        if not desired_active:
            active = False
            hold_days = 0
            rolling_peak = 0.0
            exit_override = False
        else:
            if not active and not exit_override:
                active = True
                hold_days = 0
                rolling_peak = float(row["tqqq_close"])
            elif exit_override:
                active = False

        position = "TQQQ" if active else "CASH"
        daily_ret = 0.0
        trailing_exit = False
        time_exit = False
        if idx > 0 and position == "TQQQ":
            prev_close = float(frame.iloc[idx - 1]["tqqq_close"])
            cur_close = float(row["tqqq_close"])
            daily_ret = cur_close / prev_close - 1.0 if prev_close > 0 else 0.0
            hold_days += 1
            rolling_peak = max(rolling_peak, cur_close)
            if trailing_lookback_days > 0 and trailing_drawdown_pct > 0 and hold_days >= trailing_lookback_days and rolling_peak > 0:
                drawdown_from_peak = (rolling_peak - cur_close) / rolling_peak * 100.0
                trailing_exit = drawdown_from_peak >= trailing_drawdown_pct
            if max_hold_days > 0 and hold_days >= max_hold_days:
                time_exit = True
            if trailing_exit or time_exit:
                active = False
                exit_override = True
                hold_days = 0
                rolling_peak = 0.0
        elif position == "CASH" and desired_active:
            exit_override = False

        if idx > 0 and position != previous_position:
            daily_ret -= float(switch_cost_bps) / 10000.0
        previous_position = position
        capital *= 1.0 + daily_ret
        rows.append(
            {
                "date": row["date"],
                "position": position,
                "daily_return": daily_ret,
                "capital": capital,
                "vix_label": row["vix_label"],
                "rel_strength_label": row["rel_strength_label"],
                "ixic_trend_label": row["ixic_trend_label"],
                "trailing_exit": trailing_exit,
                "time_exit": time_exit,
            }
        )
    return pd.DataFrame(rows)


def summarize_subset(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "days": 0,
            "invested_days": 0,
            "subset_compounded_return_pct": 0.0,
            "sum_return_pct": 0.0,
            "avg_daily_return_bps": 0.0,
            "avg_invested_day_return_bps": 0.0,
            "positive_day_rate_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "annual_returns_pct": {},
        }
    base = 1000.0
    equity = base * (1.0 + df["daily_return"]).cumprod()
    frame = pd.DataFrame({"date": df["date"], "equity": equity})
    invested = df[df["position"] == "TQQQ"].copy()
    positive_rate = float((df["daily_return"] > 0).mean() * 100.0) if len(df) else 0.0
    avg_daily_return_bps = float(df["daily_return"].mean() * 10000.0) if len(df) else 0.0
    avg_invested_day_return_bps = float(invested["daily_return"].mean() * 10000.0) if len(invested) else 0.0
    return {
        "days": int(len(df)),
        "invested_days": int((df["position"] == "TQQQ").sum()),
        "subset_compounded_return_pct": round((float(equity.iloc[-1]) / base - 1.0) * 100.0, 2),
        "sum_return_pct": round(float(df["daily_return"].sum() * 100.0), 2),
        "avg_daily_return_bps": round(avg_daily_return_bps, 2),
        "avg_invested_day_return_bps": round(avg_invested_day_return_bps, 2),
        "positive_day_rate_pct": round(positive_rate, 2),
        "max_drawdown_pct": round(max_drawdown_pct(equity), 2),
        "annual_returns_pct": annual_returns(frame),
    }


def group_summary(df: pd.DataFrame, column: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, group in df.groupby(column):
        output[str(key)] = summarize_subset(group.reset_index(drop=True))
    return output


def main() -> None:
    args = parse_args()
    qqq = load_df(Path(args.qqq))
    tqqq = load_df(Path(args.tqqq))
    spy = load_df(Path(args.spy))
    ixic = load_df(Path(args.ixic))
    vix = load_df(Path(args.vix))
    frame = build_regime_frame(qqq, tqqq, spy, ixic, vix, int(args.entry_fast_window), int(args.entry_slow_window))
    result = run_main_candidate(
        frame,
        initial_capital=float(args.initial_capital),
        switch_cost_bps=float(args.switch_cost_bps),
        max_hold_days=int(args.max_hold_days),
        trailing_lookback_days=int(args.trailing_lookback_days),
        trailing_drawdown_pct=float(args.trailing_drawdown_pct),
    )
    payload = {
        "main_candidate": {
            "entry_fast_window": int(args.entry_fast_window),
            "entry_slow_window": int(args.entry_slow_window),
            "max_hold_days": int(args.max_hold_days),
            "trailing_lookback_days": int(args.trailing_lookback_days),
            "trailing_drawdown_pct": float(args.trailing_drawdown_pct),
            "switch_cost_bps": float(args.switch_cost_bps),
        },
        "overall": summarize_subset(result),
        "by_vix_label": group_summary(result, "vix_label"),
        "by_rel_strength_label": group_summary(result, "rel_strength_label"),
        "by_ixic_trend_label": group_summary(result, "ixic_trend_label"),
        "exit_counts": {
            "trailing_exit_days": int(result["trailing_exit"].sum()),
            "time_exit_days": int(result["time_exit"].sum()),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    print(json.dumps(payload["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
