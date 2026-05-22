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

from scripts.nasdaq100_cn_strategy_utils import load_config, load_strategy_frame


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "cn_nasdaq100_execution_slippage_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit execution/slippage sensitivity for CN Nasdaq-100 ETF baseline.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.paper.cn-nasdaq100-etf.json"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
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


def max_drawdown_pct(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = ((peak - equity) / peak.replace(0, pd.NA) * 100.0).max(skipna=True)
    return float(drawdown or 0.0)


def load_volume_frame(config: dict[str, Any]) -> pd.DataFrame:
    data_root = ROOT / str(config["data_root"])
    execution_symbol = str(config.get("execution_symbol", "513100.SS"))
    raw = pd.read_feather(data_root / f"{execution_symbol}-1d.feather")
    frame = raw[["date", "close", "volume"]].copy()
    frame["session_day"] = pd.to_datetime(frame["date"], utc=True).dt.normalize()
    frame = frame.sort_values("session_day").drop_duplicates("session_day", keep="last").reset_index(drop=True)
    frame["notional"] = frame["close"] * frame["volume"]
    return frame[["session_day", "volume", "notional"]]


def build_execution_prices(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame.copy().reset_index(drop=True)
    local["exec_asset_close"] = local["asset_close"]
    local["exec_same_close"] = local["asset_close"]
    local["exec_next_open"] = local["asset_open"].shift(-1)
    local["exec_next_close"] = local["asset_close"].shift(-1)
    local["exec_next_mid"] = ((local["asset_high"].shift(-1) + local["asset_low"].shift(-1)) / 2.0)
    return local


def leverage_for_row(row: pd.Series, config: dict[str, Any]) -> float:
    if str(row["position"]) != "LONG":
        return 0.0
    if bool(config.get("tiered_leverage_enabled", False)):
        for rule in list(config.get("tiered_leverage_rules", [])):
            if not isinstance(rule, dict):
                continue
            vix_label = str(rule.get("vix_label", "") or "").strip()
            rel_strength_label = str(rule.get("rel_strength_label", "") or "").strip()
            ixic_trend_label = str(rule.get("ixic_trend_label", "") or "").strip()
            if vix_label and str(row["vix_label"]) != vix_label:
                continue
            if rel_strength_label and str(row["rel_strength_label"]) != rel_strength_label:
                continue
            if ixic_trend_label and str(row["ixic_trend_label"]) != ixic_trend_label:
                continue
            return float(rule.get("leverage", 1.0) or 1.0)
        return 1.0
    if bool(config.get("conditional_leverage_enabled", False)):
        trigger = str(config.get("conditional_leverage_trigger", "none"))
        value = float(config.get("conditional_leverage_value", 1.0))
        if trigger == "vix_low" and str(row["vix_label"]) == "vix_low":
            return value
        if trigger == "qqq_strong" and str(row["rel_strength_label"]) == "qqq_strong":
            return value
        if trigger == "vix_low_and_qqq_strong" and str(row["vix_label"]) == "vix_low" and str(row["rel_strength_label"]) == "qqq_strong":
            return value
    return 1.0


def simulate_execution_mode(frame: pd.DataFrame, config: dict[str, Any], *, price_column: str) -> dict[str, Any]:
    local = frame.copy().reset_index(drop=True)
    capital = float(config.get("initial_capital", 1000.0))
    switch_cost = float(config.get("switch_cost_bps", 10.0)) / 10000.0
    rows: list[dict[str, Any]] = []
    executed_trade_rows: list[dict[str, Any]] = []

    for idx, row in local.iterrows():
        position = str(row["position"])
        daily_ret = 0.0
        leverage = leverage_for_row(row, config)
        if idx > 0 and position == "LONG":
            prev_px = float(local.iloc[idx - 1][price_column]) if pd.notna(local.iloc[idx - 1][price_column]) else float(local.iloc[idx - 1]["exec_asset_close"])
            cur_px = float(row[price_column]) if pd.notna(row[price_column]) else float(row["exec_asset_close"])
            if prev_px > 0:
                daily_ret = (cur_px / prev_px - 1.0) * leverage
        prev_position = str(local.iloc[idx - 1]["position"]) if idx > 0 else "CASH"
        if idx > 0 and position != prev_position:
            daily_ret -= switch_cost
            if position == "LONG" or prev_position == "LONG":
                executed_trade_rows.append(
                    {
                        "date": str(pd.Timestamp(row["date"]).date()),
                        "event": f"{prev_position}_to_{position}",
                        "price_mode": price_column,
                        "execution_price": round(float(row[price_column]) if pd.notna(row[price_column]) else float(row["exec_asset_close"]), 4),
                        "volume": float(row.get("volume", 0.0) or 0.0),
                        "notional": float(row.get("notional", 0.0) or 0.0),
                    }
                )
        capital *= 1.0 + daily_ret
        rows.append({"date": row["date"], "equity": capital, "daily_return": daily_ret, "position": position})

    equity = pd.DataFrame(rows)
    yearly = annual_returns(equity)
    total_return_pct = round((capital / float(config.get("initial_capital", 1000.0)) - 1.0) * 100.0, 2)
    max_dd = round(max_drawdown_pct(equity["equity"]), 2)
    notionals = pd.Series([row["notional"] for row in executed_trade_rows], dtype=float)
    return {
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd,
        "yearly_returns_pct": yearly,
        "trades": int((equity["position"] != equity["position"].shift(1)).sum()),
        "executions": int(len(executed_trade_rows)),
        "median_trade_day_notional": round(float(notionals.median()), 2) if not notionals.empty else 0.0,
        "p10_trade_day_notional": round(float(notionals.quantile(0.1)), 2) if not notionals.empty else 0.0,
        "p90_trade_day_notional": round(float(notionals.quantile(0.9)), 2) if not notionals.empty else 0.0,
    }


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    frame = load_strategy_frame(config)
    path_config = dict(config)
    path_config["tiered_leverage_enabled"] = False
    path_config["conditional_leverage_enabled"] = False

    from scripts.nasdaq100_cn_strategy_utils import run_full_strategy_path

    path = run_full_strategy_path(frame, path_config).reset_index(drop=True)
    aligned_frame = frame.loc[frame.index[-len(path):]].reset_index(drop=True)
    exec_frame = build_execution_prices(aligned_frame)
    volume_frame = load_volume_frame(config)
    exec_frame["session_day"] = pd.to_datetime(exec_frame["date"], utc=True).dt.normalize()
    exec_frame = exec_frame.merge(volume_frame, on="session_day", how="left")
    exec_columns = [
        "session_day",
        "asset_open",
        "asset_high",
        "asset_low",
        "exec_asset_close",
        "exec_same_close",
        "exec_next_open",
        "exec_next_close",
        "exec_next_mid",
        "volume",
        "notional",
    ]
    merged = pd.concat([path.reset_index(drop=True), exec_frame[exec_columns].reset_index(drop=True)], axis=1)

    modes = {
        "same_close_reference": "exec_same_close",
        "next_open": "exec_next_open",
        "next_close": "exec_next_close",
        "next_mid": "exec_next_mid",
    }
    results = {name: simulate_execution_mode(merged, config, price_column=column) for name, column in modes.items()}
    reference = results["same_close_reference"]
    deltas = {}
    for name, result in results.items():
        deltas[name] = {
            "return_delta_pct": round(float(result["total_return_pct"]) - float(reference["total_return_pct"]), 2),
            "dd_delta_pct": round(float(result["max_drawdown_pct"]) - float(reference["max_drawdown_pct"]), 2),
            "y2026_delta_pct": round(float(result["yearly_returns_pct"].get("2026", 0.0)) - float(reference["yearly_returns_pct"].get("2026", 0.0)), 2),
        }

    payload = {
        "baseline_execution_assumption": "same_close_reference",
        "results": results,
        "deltas_vs_reference": deltas,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
