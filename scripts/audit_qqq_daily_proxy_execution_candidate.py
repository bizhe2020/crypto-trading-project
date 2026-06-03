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

from scripts.replay_qqq_daily_proxy_high_stop import (  # noqa: E402
    build_signal_path,
    load_ohlcv,
    load_strict_config,
    max_drawdown_pct,
    merge_signal_with_qqq,
    replay_daily_proxy,
)


DEFAULT_SIGNAL_CONFIG = ROOT / "config" / "config.paper.tqqq-only-strict-recovery-frozen.json"
DEFAULT_DATA_ROOT = ROOT / "data" / "public" / "etf_long"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_daily_proxy_execution_candidate_audit_20260530.json"


def equity_summary(path: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    if path.empty:
        return {"total_return_pct": 0.0, "max_drawdown_pct": 0.0}
    return {
        "total_return_pct": round((float(path.iloc[-1]["capital"]) / float(initial_capital) - 1.0) * 100.0, 2),
        "max_drawdown_pct": round(max_drawdown_pct(path["capital"]), 2),
    }


def segment_summary(path: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    part = path[(path["date"] >= start_ts) & (path["date"] <= end_ts)].copy()
    if part.empty:
        return {"start": start, "end": end, "rows": 0, "total_return_pct": 0.0, "max_drawdown_pct": 0.0}
    base = 1000.0
    equity = base * (1.0 + part["daily_return"]).cumprod()
    return {
        "start": start,
        "end": end,
        "rows": int(len(part)),
        "total_return_pct": round((float(equity.iloc[-1]) / base - 1.0) * 100.0, 2),
        "max_drawdown_pct": round(max_drawdown_pct(equity), 2),
        "invested_days": int(part["holding"].sum()),
    }


def remove_top_trade_proxy(path: pd.DataFrame, trades: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    if trades.empty:
        return {"removed": None, "summary": equity_summary(path, initial_capital)}
    top = trades.sort_values("trade_return_pct", ascending=False).iloc[0]
    start = pd.Timestamp(top["entry_date"])
    end = pd.Timestamp(top["exit_date"])
    adjusted = path.copy()
    mask = (adjusted["date"] >= start) & (adjusted["date"] <= end)
    adjusted.loc[mask, "daily_return"] = 0.0
    adjusted["capital"] = float(initial_capital) * (1.0 + adjusted["daily_return"]).cumprod()
    return {
        "removed": top.to_dict(),
        "summary": equity_summary(adjusted, initial_capital),
    }


def audit_candidate(
    bars: pd.DataFrame,
    *,
    profile_name: str,
    profile: dict[str, float],
    stop_loss_pct: float,
    stop_mode: str,
    taker_fee_rate: float,
    slippage_bps: float,
    initial_capital: float,
) -> dict[str, Any]:
    result = replay_daily_proxy(
        bars,
        profile_name=profile_name,
        profile=profile,
        stop_mode=stop_mode,
        stop_loss_pct=stop_loss_pct,
        taker_fee_rate=taker_fee_rate,
        slippage_bps=slippage_bps,
        initial_capital=initial_capital,
        daily_funding_rate=0.0,
        rebalance_on_leverage_change=True,
    )
    path = result["path"]
    trades = result["trades"]
    segments = [
        segment_summary(path, "2010-01-01", "2014-12-31"),
        segment_summary(path, "2015-01-01", "2019-12-31"),
        segment_summary(path, "2020-01-01", "2022-12-31"),
        segment_summary(path, "2023-01-01", "2026-12-31"),
    ]
    top_trades = trades.sort_values("trade_return_pct", ascending=False).head(10).to_dict(orient="records") if not trades.empty else []
    bottom_trades = trades.sort_values("trade_return_pct", ascending=True).head(10).to_dict(orient="records") if not trades.empty else []
    return {
        "candidate": {
            "profile_name": profile_name,
            "profile": profile,
            "stop_loss_pct": stop_loss_pct,
            "stop_mode": stop_mode,
        },
        "summary": result["summary"],
        "segments": segments,
        "remove_top_trade": remove_top_trade_proxy(path, trades, initial_capital),
        "top_trades": top_trades,
        "bottom_trades": bottom_trades,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit selected QQQ daily proxy execution candidates.")
    parser.add_argument("--signal-config", default=str(DEFAULT_SIGNAL_CONFIG))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--taker-fee-rate", type=float, default=0.0002)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    args = parser.parse_args()

    signal_config = load_strict_config(Path(args.signal_config))
    signal_path, signal_summary = build_signal_path(signal_config, Path(args.data_root))
    qqq = load_ohlcv(Path(args.data_root) / "QQQ-1d.feather")
    bars = merge_signal_with_qqq(signal_path, qqq)
    initial_capital = float(signal_config.get("initial_capital", 1000.0))

    candidates = [
        ("conservative_fixed3_stop5_close_prev", "fixed3", {"base": 3.0, "offense": 3.0, "defense": 3.0}, 5.0, "close_prev_strict"),
        ("balanced_fixed4_stop5_close_prev", "fixed4", {"base": 4.0, "offense": 4.0, "defense": 4.0}, 5.0, "close_prev_strict"),
        ("aggressive_fixed5_stop5_close_prev", "fixed5", {"base": 5.0, "offense": 5.0, "defense": 5.0}, 5.0, "close_prev_strict"),
        ("strict_high_fixed4_stop65", "fixed4", {"base": 4.0, "offense": 4.0, "defense": 4.0}, 6.5, "high_prev_strict"),
    ]
    audits = {
        label: audit_candidate(
            bars,
            profile_name=profile_name,
            profile=profile,
            stop_loss_pct=stop_loss_pct,
            stop_mode=stop_mode,
            taker_fee_rate=float(args.taker_fee_rate),
            slippage_bps=float(args.slippage_bps),
            initial_capital=initial_capital,
        )
        for label, profile_name, profile, stop_loss_pct, stop_mode in candidates
    }
    payload = {
        "metadata": {
            "signal_config": str(Path(args.signal_config)),
            "signal_frozen_label": signal_config.get("frozen_label"),
            "data_root": str(Path(args.data_root)),
            "taker_fee_rate": float(args.taker_fee_rate),
            "slippage_bps": float(args.slippage_bps),
        },
        "signal_summary": signal_summary,
        "audits": audits,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(out)
    for label, audit in audits.items():
        s = audit["summary"]
        print(label, s["total_return_pct"], s["max_drawdown_pct"], s["closed_trades"], s["win_rate_pct"], s["avg_hold_days"])
        print("  remove_top", audit["remove_top_trade"]["summary"])
        print("  segments", audit["segments"])


if __name__ == "__main__":
    main()
