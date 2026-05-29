#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_REPORT_DIR = ROOT / "var" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple daily trend baseline for TQQQ/SQQQ.")
    parser.add_argument("--tqqq", default=str(DEFAULT_PUBLIC_DIR / "TQQQ-1d.feather"))
    parser.add_argument("--sqqq", default=str(DEFAULT_PUBLIC_DIR / "SQQQ-1d.feather"))
    parser.add_argument("--qqq", default=str(DEFAULT_PUBLIC_DIR / "QQQ-1d.feather"))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--fast-window", type=int, default=20)
    parser.add_argument("--slow-window", type=int, default=100)
    parser.add_argument("--mode", choices=["tqqq_cash", "switch_tqqq_sqqq"], default="tqqq_cash")
    parser.add_argument("--switch-cost-bps", type=float, default=5.0)
    return parser.parse_args()


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_feather(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0, pd.NA) * 100.0
    return float(dd.max(skipna=True) or 0.0)


def annual_returns(equity: pd.DataFrame) -> dict[str, float]:
    if equity.empty:
        return {}
    equity = equity.copy()
    equity["year"] = equity["date"].dt.year.astype(str)
    out: dict[str, float] = {}
    for year, group in equity.groupby("year"):
        start = float(group.iloc[0]["equity"])
        end = float(group.iloc[-1]["equity"])
        out[year] = round((end / start - 1.0) * 100.0, 2) if start > 0 else 0.0
    return out


def build_signal(df: pd.DataFrame, fast_window: int, slow_window: int) -> pd.DataFrame:
    out = df.copy()
    out["fast_ma"] = out["qqq_close"].rolling(fast_window).mean()
    out["slow_ma"] = out["qqq_close"].rolling(slow_window).mean()
    out["trend"] = 0
    out.loc[out["fast_ma"] > out["slow_ma"], "trend"] = 1
    out.loc[out["fast_ma"] < out["slow_ma"], "trend"] = -1
    return out


def signal_to_position(signal: float, mode: str) -> str:
    if signal > 0:
        return "TQQQ"
    if signal < 0 and mode == "switch_tqqq_sqqq":
        return "SQQQ"
    return "CASH"


def run_baseline(
    tqqq: pd.DataFrame,
    sqqq: pd.DataFrame,
    qqq: pd.DataFrame,
    initial_capital: float,
    fast_window: int,
    slow_window: int,
    mode: str,
    switch_cost_bps: float,
) -> dict[str, Any]:
    merged = qqq[["date", "close"]].rename(columns={"close": "qqq_close"})
    merged = merged.merge(tqqq[["date", "close"]].rename(columns={"close": "tqqq_close"}), on="date", how="inner")
    merged = merged.merge(sqqq[["date", "close"]].rename(columns={"close": "sqqq_close"}), on="date", how="inner")
    merged = build_signal(merged, fast_window, slow_window)
    merged = merged.dropna(subset=["fast_ma", "slow_ma"]).reset_index(drop=True)
    merged["effective_position"] = merged["trend"].shift(1).fillna(0).map(lambda value: signal_to_position(float(value), mode))

    capital = initial_capital
    capitals = []
    positions = []
    returns = []
    previous_asset = "CASH"
    for idx, row in merged.iterrows():
        asset = str(row["effective_position"])
        positions.append(asset)
        daily_ret = 0.0
        if idx > 0 and asset == "TQQQ":
            prev_idx = idx - 1
            prev = float(merged.iloc[prev_idx]["tqqq_close"])
            cur = float(row["tqqq_close"])
            daily_ret = cur / prev - 1.0 if prev > 0 else 0.0
        elif idx > 0 and asset == "SQQQ":
            prev_idx = idx - 1
            prev = float(merged.iloc[prev_idx]["sqqq_close"])
            cur = float(row["sqqq_close"])
            daily_ret = cur / prev - 1.0 if prev > 0 else 0.0
        if idx > 0 and asset != previous_asset:
            daily_ret -= float(switch_cost_bps) / 10000.0
        previous_asset = asset
        capital *= 1.0 + daily_ret
        returns.append(daily_ret)
        capitals.append(capital)

    equity = pd.DataFrame({"date": merged["date"], "equity": capitals, "position": positions})
    result = {
        "initial_capital": initial_capital,
        "final_capital": round(capital, 2),
        "total_return_pct": round((capital / initial_capital - 1.0) * 100.0, 2),
        "max_drawdown_pct": round(max_drawdown_pct(equity["equity"]), 2),
        "sharpe_like": round((pd.Series(returns).mean() / pd.Series(returns).std()) if len(returns) > 1 and pd.Series(returns).std() > 0 else 0.0, 3),
        "trades": int((pd.Series(positions) != pd.Series(positions).shift(1)).sum()),
        "annual_returns_pct": annual_returns(equity),
        "position_counts": pd.Series(positions).value_counts().to_dict(),
        "data_range": {
            "start": str(merged["date"].min()),
            "end": str(merged["date"].max()),
            "rows": int(len(merged)),
        },
        "params": {
            "fast_window": fast_window,
            "slow_window": slow_window,
            "mode": mode,
            "switch_cost_bps": switch_cost_bps,
        },
    }
    return result


def main() -> None:
    args = parse_args()
    tqqq = load_df(Path(args.tqqq))
    sqqq = load_df(Path(args.sqqq))
    qqq = load_df(Path(args.qqq))
    report = run_baseline(
        tqqq,
        sqqq,
        qqq,
        args.initial_capital,
        args.fast_window,
        args.slow_window,
        args.mode,
        args.switch_cost_bps,
    )
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / f"tqqq_sqqq_trend_baseline_{args.mode}.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(output_path)
    print(report["total_return_pct"], report["max_drawdown_pct"], report["annual_returns_pct"])


if __name__ == "__main__":
    main()
