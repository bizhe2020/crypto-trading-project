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

from scripts.tqqq_cash_strict_utils import load_strict_config, load_strict_frame_with_overlay_context, run_strict_candidate  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "config.paper.tqqq-only-strict-recovery-frozen.json"
DEFAULT_OKX_4H = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-4h-futures.feather"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_usdt_4h_trigger_scan.json"


ENTRY_PROFILES = [
    "immediate",
    "breakout_12",
    "breakout_20",
    "pullback_ema20",
    "pullback_break3",
]

EXIT_PROFILES = {
    "daily_only": {"atr_mult": 0.0, "trail_pct": 0.0},
    "atr2.5_trail5": {"atr_mult": 2.5, "trail_pct": 5.0},
    "atr2.0_trail4": {"atr_mult": 2.0, "trail_pct": 4.0},
    "atr1.5_trail3": {"atr_mult": 1.5, "trail_pct": 3.0},
}


def load_signal_path(config_path: Path, overrides: dict[str, Any] | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    config = load_strict_config(config_path)
    if overrides:
        allowed_overrides = {
            "regime_filter",
            "max_hold_days",
            "trailing_lookback_days",
            "trailing_drawdown_pct",
            "switch_cost_bps",
            "initial_capital",
            "de_risk_signal_name",
            "recovery_reentry_rule",
            "recovery_reentry_cooldown_days",
            "drawdown_ladder_enabled",
            "drawdown_ladder_source",
            "drawdown_ladder_threshold_pct",
            "drawdown_ladder_peak_lookback_days",
            "drawdown_ladder_scheme",
            "drawdown_ladder_vix_rule",
            "drawdown_ladder_rebound_exit_pct",
            "drawdown_ladder_max_hold_days",
        }
        unknown = sorted(set(overrides) - allowed_overrides)
        if unknown:
            raise ValueError(f"Unsupported signal override keys: {unknown}")
        config.update(overrides)
    frame = load_strict_frame_with_overlay_context(
        data_root=ROOT / str(config["data_root"]),
        entry_fast_window=int(config["entry_fast_window"]),
        entry_slow_window=int(config["entry_slow_window"]),
    )
    result = run_strict_candidate(
        frame,
        regime_filter=str(config["regime_filter"]),
        max_hold_days=int(config["max_hold_days"]),
        trailing_lookback_days=int(config["trailing_lookback_days"]),
        trailing_drawdown_pct=float(config["trailing_drawdown_pct"]),
        switch_cost_bps=float(config["switch_cost_bps"]),
        initial_capital=float(config["initial_capital"]),
        de_risk_signal_name=str(config.get("de_risk_signal_name", "off")),
        recovery_reentry_rule=str(config.get("recovery_reentry_rule", "off")),
        recovery_reentry_cooldown_days=int(config.get("recovery_reentry_cooldown_days", 0)),
        drawdown_ladder_enabled=bool(config.get("drawdown_ladder_enabled", False)),
        drawdown_ladder_source=str(config.get("drawdown_ladder_source", "tqqq")),
        drawdown_ladder_threshold_pct=float(config.get("drawdown_ladder_threshold_pct", 0.0)),
        drawdown_ladder_peak_lookback_days=int(config.get("drawdown_ladder_peak_lookback_days", 90)),
        drawdown_ladder_scheme=str(config.get("drawdown_ladder_scheme", "two_equal")),
        drawdown_ladder_vix_rule=str(config.get("drawdown_ladder_vix_rule", "all")),
        drawdown_ladder_rebound_exit_pct=float(config.get("drawdown_ladder_rebound_exit_pct", 10.0)),
        drawdown_ladder_max_hold_days=int(config.get("drawdown_ladder_max_hold_days", 15)),
    )
    return config, result["path"].copy()


def load_okx_4h(path: Path) -> pd.DataFrame:
    df = pd.read_feather(path).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["prev_high_3"] = df["high"].shift(1).rolling(3).max()
    df["prev_high_12"] = df["high"].shift(1).rolling(12).max()
    df["prev_high_20"] = df["high"].shift(1).rolling(20).max()
    df["dd_from_high12_pct"] = (df["close"] / df["prev_high_12"] - 1.0) * 100.0
    df["recent_pullback_5"] = df["dd_from_high12_pct"].rolling(5).min() <= -2.0
    df["breakout_12"] = df["close"] > df["prev_high_12"]
    df["breakout_20"] = df["close"] > df["prev_high_20"]
    df["pullback_ema20"] = df["recent_pullback_5"] & (df["close"] > df["ema20"]) & (df["close"] > df["open"])
    df["pullback_break3"] = df["recent_pullback_5"] & (df["close"] > df["prev_high_3"]) & (df["ema20"] > df["ema50"])
    return df


def attach_daily_state(
    okx_4h: pd.DataFrame,
    signal_path: pd.DataFrame,
    *,
    trim_to_signal_end: bool = True,
) -> pd.DataFrame:
    daily = signal_path[["date", "position"]].copy().sort_values("date").reset_index(drop=True)
    merged = pd.merge_asof(
        okx_4h.sort_values("date"),
        daily,
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    if trim_to_signal_end:
        merged = merged[merged["date"] <= daily["date"].max()].copy()
    merged["allow_long"] = merged["position"].eq("TQQQ")
    return merged.reset_index(drop=True)


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0, pd.NA) * 100.0
    return float(dd.max(skipna=True) or 0.0)


def simulate(
    bars: pd.DataFrame,
    *,
    entry_profile: str,
    exit_profile_name: str,
    initial_capital: float,
    switch_cost_bps: float,
) -> dict[str, Any]:
    exit_profile = EXIT_PROFILES[exit_profile_name]
    capital = float(initial_capital)
    holding = False
    exit_pending = False
    entry_price = 0.0
    entry_atr = 0.0
    peak_close = 0.0
    hold_bars = 0
    prev_allow = False
    prev_capital = capital
    trades: list[dict[str, Any]] = []
    current_trade: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []

    for idx, row in bars.iterrows():
        start_capital = capital
        allow_now = bool(row["allow_long"])
        entered_today = False
        exited_today = False
        action_cost = 0.0

        if holding and (not allow_now or exit_pending):
            action_cost += float(switch_cost_bps) / 10000.0
            capital *= 1.0 - float(switch_cost_bps) / 10000.0
            holding = False
            exited_today = True
            if current_trade is not None:
                trades.append(
                    {
                        "entry_date": str(current_trade["entry_date"]),
                        "exit_date": str(pd.Timestamp(row["date"])),
                        "trade_return_pct": round((capital / float(current_trade["entry_capital"]) - 1.0) * 100.0, 2),
                        "hold_bars": int(hold_bars),
                        "entry_profile": entry_profile,
                        "exit_profile": exit_profile_name,
                    }
                )
            current_trade = None
            exit_pending = False
            hold_bars = 0
            peak_close = 0.0
            entry_price = 0.0
            entry_atr = 0.0

        enter_signal = False
        if allow_now and not holding:
            if entry_profile == "immediate":
                enter_signal = not prev_allow
            elif idx > 0 and prev_allow:
                enter_signal = bool(bars.iloc[idx - 1][entry_profile])
        if enter_signal:
            action_cost += float(switch_cost_bps) / 10000.0
            capital *= 1.0 - float(switch_cost_bps) / 10000.0
            holding = True
            entered_today = True
            entry_price = float(row["open"])
            entry_atr = float(bars.iloc[idx - 1]["atr14"]) if idx > 0 and pd.notna(bars.iloc[idx - 1]["atr14"]) else float(row["atr14"] or 0.0)
            peak_close = float(row["open"])
            hold_bars = 0
            current_trade = {
                "entry_date": str(pd.Timestamp(row["date"])),
                "entry_capital": capital,
            }

        trail_signal = False
        atr_signal = False
        if holding:
            open_price = float(row["open"])
            close_price = float(row["close"])
            if open_price > 0:
                capital *= 1.0 + (close_price / open_price - 1.0)
            hold_bars += 1
            peak_close = max(peak_close, close_price)
            if exit_profile["trail_pct"] > 0 and peak_close > 0:
                trail_signal = close_price <= peak_close * (1.0 - exit_profile["trail_pct"] / 100.0)
            if exit_profile["atr_mult"] > 0 and entry_atr > 0:
                atr_signal = close_price <= (entry_price - exit_profile["atr_mult"] * entry_atr)
            if trail_signal or atr_signal:
                exit_pending = True

        prev_allow = allow_now
        rows.append(
            {
                "date": pd.Timestamp(row["date"]),
                "allow_long": allow_now,
                "holding": holding,
                "entered_today": entered_today,
                "exited_today": exited_today,
                "daily_return": capital / start_capital - 1.0 if start_capital > 0 else 0.0,
                "capital": float(capital),
                "trail_signal": trail_signal,
                "atr_signal": atr_signal,
                "action_cost": float(action_cost),
            }
        )
        prev_capital = capital

    path = pd.DataFrame(rows)
    total_return_pct = round((float(path.iloc[-1]["capital"]) / float(initial_capital) - 1.0) * 100.0, 2) if not path.empty else 0.0
    max_dd = round(max_drawdown_pct(path["capital"]), 2) if not path.empty else 0.0
    trades_df = pd.DataFrame(trades)
    win_rate = round(float((trades_df["trade_return_pct"] > 0).mean() * 100.0), 2) if not trades_df.empty else 0.0
    avg_hold_bars = round(float(trades_df["hold_bars"].mean()), 2) if not trades_df.empty else 0.0
    avg_trade_return = round(float(trades_df["trade_return_pct"].mean()), 2) if not trades_df.empty else 0.0
    return {
        "entry_profile": entry_profile,
        "exit_profile": exit_profile_name,
        "summary": {
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd,
            "trades": int(len(trades_df)),
            "win_rate_pct": win_rate,
            "avg_hold_bars": avg_hold_bars,
            "avg_trade_return_pct": avg_trade_return,
            "invested_bars": int(path["holding"].sum()) if not path.empty else 0,
            "latest_holding": bool(path.iloc[-1]["holding"]) if not path.empty else False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan QQQ/USDT 4h trigger overlays under ETF-derived daily direction.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--okx-4h", default=str(DEFAULT_OKX_4H))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    config, signal_path = load_signal_path(Path(args.config))
    okx_4h = load_okx_4h(Path(args.okx_4h))
    bars = attach_daily_state(okx_4h, signal_path)

    results = []
    for entry_profile in ENTRY_PROFILES:
        for exit_profile_name in EXIT_PROFILES:
            results.append(
                simulate(
                    bars,
                    entry_profile=entry_profile,
                    exit_profile_name=exit_profile_name,
                    initial_capital=float(config["initial_capital"]),
                    switch_cost_bps=float(config["switch_cost_bps"]),
                )
            )

    results = sorted(
        results,
        key=lambda x: (
            x["summary"]["total_return_pct"],
            -x["summary"]["max_drawdown_pct"],
            x["summary"]["win_rate_pct"],
        ),
        reverse=True,
    )

    payload = {
        "config": config,
        "coverage": {
            "bars": int(len(bars)),
            "start": str(bars["date"].min()) if not bars.empty else None,
            "end": str(bars["date"].max()) if not bars.empty else None,
            "allowed_bars": int(bars["allow_long"].sum()) if not bars.empty else 0,
        },
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(out)
    print(json.dumps(results[:8], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
