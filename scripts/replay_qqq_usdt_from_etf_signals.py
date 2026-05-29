#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tqqq_cash_strict_utils import load_strict_config, load_strict_frame_with_overlay_context, run_strict_candidate  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "config.paper.tqqq-only-strict-recovery-frozen.json"
DEFAULT_OKX_DIR = ROOT / "data" / "okx" / "futures"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_usdt_from_etf_signals_replay.json"


def load_okx_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_feather(path).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def build_signal_frame(config: dict) -> pd.DataFrame:
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
    return result["path"].copy()


def replay_on_okx_daily(signal_path: pd.DataFrame, okx_daily: pd.DataFrame, initial_capital: float, switch_cost_bps: float) -> dict:
    signal = signal_path[["date", "position", "entered_today", "exited_today"]].copy()
    signal["session_day"] = signal["date"].dt.normalize()
    okx = okx_daily.copy()
    okx["session_day"] = okx["date"].dt.normalize()
    merged = okx.merge(signal[["session_day", "position"]], on="session_day", how="inner")
    merged = merged.sort_values("session_day").reset_index(drop=True)
    if merged.empty:
        return {
            "summary": {
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "days": 0,
                "invested_days": 0,
                "latest_position": "CASH",
            },
            "path": merged,
        }

    capital = float(initial_capital)
    prev_position = "CASH"
    rows: list[dict] = []
    for row in merged.itertuples(index=False):
        position = str(row.position)
        daily_ret = 0.0
        if position == "TQQQ":
            open_price = float(row.open)
            close_price = float(row.close)
            if open_price > 0:
                daily_ret = close_price / open_price - 1.0
        if position != prev_position:
            daily_ret -= float(switch_cost_bps) / 10000.0
        capital *= 1.0 + daily_ret
        rows.append(
            {
                "date": pd.Timestamp(row.date),
                "session_day": pd.Timestamp(row.session_day),
                "position": position,
                "open": float(row.open),
                "close": float(row.close),
                "daily_return": float(daily_ret),
                "capital": float(capital),
            }
        )
        prev_position = position

    path = pd.DataFrame(rows)
    peak = path["capital"].cummax()
    dd = ((peak - path["capital"]) / peak.replace(0, pd.NA) * 100.0).fillna(0.0)
    return {
        "summary": {
            "total_return_pct": round((float(path.iloc[-1]["capital"]) / float(initial_capital) - 1.0) * 100.0, 2),
            "max_drawdown_pct": round(float(dd.max()), 2),
            "days": int(len(path)),
            "invested_days": int((path["position"] == "TQQQ").sum()),
            "latest_position": str(path.iloc[-1]["position"]),
            "start": str(path.iloc[0]["date"]),
            "end": str(path.iloc[-1]["date"]),
        },
        "path": path,
    }


def timeframe_coverage(path: Path) -> dict:
    df = load_okx_ohlcv(path)
    return {
        "rows": int(len(df)),
        "start": str(df["date"].min()),
        "end": str(df["date"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay ETF-derived TQQQ frozen signals on OKX QQQ/USDT data.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--okx-dir", default=str(DEFAULT_OKX_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    config = load_strict_config(Path(args.config))
    signal_path = build_signal_frame(config)

    okx_dir = Path(args.okx_dir)
    okx_1d = load_okx_ohlcv(okx_dir / "QQQ_USDT_USDT-1d-futures.feather")
    replay = replay_on_okx_daily(
        signal_path,
        okx_1d,
        initial_capital=float(config["initial_capital"]),
        switch_cost_bps=float(config["switch_cost_bps"]),
    )

    payload = {
        "config": config,
        "okx_replay_1d": replay["summary"],
        "coverage": {
            "signal_daily": {
                "rows": int(len(signal_path)),
                "start": str(signal_path["date"].min()),
                "end": str(signal_path["date"].max()),
            },
            "okx_1h": timeframe_coverage(okx_dir / "QQQ_USDT_USDT-1h-futures.feather"),
            "okx_4h": timeframe_coverage(okx_dir / "QQQ_USDT_USDT-4h-futures.feather"),
            "okx_1d": timeframe_coverage(okx_dir / "QQQ_USDT_USDT-1d-futures.feather"),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(out)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
