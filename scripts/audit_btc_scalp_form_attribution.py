#!/usr/bin/env python3
"""BTC scalp **形态级入场归因**（无前视口径）。

背景：`docs/btc_scalp_optimization_20260921.md` 已证明
- 出场层 / 风控层没有真实 edge 空间（§4.4、§5）；
- 现行 SOTA 多周期分数与结果秩相关 ≈ 0（§6）。

唯一还没被证伪的优化方向是：**哪一类结构性入场有正期望**。本脚本把引擎在
live 配置下跑出的每一笔入场，按「入场时刻可观测」的特征分桶，给出
风险归一化净 R / 毛 R / 胜率 / MFE-MAE，并对「最佳阈值切分」做
**max-statistic 置换检验**（含对阈值搜索的多重比较惩罚），避免把噪音当发现。

特征（全部在 entry bar 收盘时可得，无前视）：
  - 止损宽度（ATR 倍数、%价格）
  - 日线乖离（alpha 门用的同一个数）
  - regime 特征（adx / momentum / ema_gap）
  - regime_label / trail_style / direction / risk_regime
  - 事后的 MFE/MAE 分布（诊断用，不可作为过滤器）

用法::

    python3 scripts/audit_btc_scalp_form_attribution.py --start-date 2023-01-01
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.scan_btc_scalp_optimization import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_DATA_15M,
    DEFAULT_DATA_4H,
    load_base_config,
)

DEFAULT_OUTPUT = ROOT / "var" / "tmp" / "btc_opt" / "form_attribution.json"
PERMUTATIONS = 2000
RNG_SEED = 20260921


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTC scalp 形态级入场归因")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def excursion_r(engine: Any, trade: Any) -> tuple[float, float]:
    """从 entry_idx+1 到 exit_idx 逐根算 MFE/MAE（以 R 为单位）。"""
    entry_idx = int(trade.entry_idx) if trade.entry_idx is not None else None
    exit_idx = int(trade.exit_idx) if trade.exit_idx is not None else None
    if entry_idx is None or exit_idx is None or exit_idx <= entry_idx:
        return (float("nan"), float("nan"))
    risk = abs(trade.entry_price - (trade.initial_stop_price or trade.entry_price))
    if risk <= 0 or not trade.quantity:
        return (float("nan"), float("nan"))
    bull = trade.direction == "BULL"
    mfe = mae = 0.0
    for idx in range(entry_idx + 1, exit_idx + 1):
        candle = engine.c15m[idx]
        if bull:
            fav = (candle.h - trade.entry_price) * trade.quantity
            adv = (candle.l - trade.entry_price) * trade.quantity
        else:
            fav = (trade.entry_price - candle.l) * trade.quantity
            adv = (trade.entry_price - candle.h) * trade.quantity
        mfe = max(mfe, fav / (risk * trade.quantity))
        mae = min(mae, adv / (risk * trade.quantity))
    return (mfe, mae)


def build_features(engine: Any, trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade in engine.trades:
        entry_idx = int(trade.entry_idx) if trade.entry_idx is not None else -1
        atr = engine._atr_for_idx(entry_idx) if entry_idx >= 0 else 0.0
        stop_dist = abs(trade.entry_price - (trade.initial_stop_price or trade.entry_price))
        features = engine._regime_features_for_idx(entry_idx) if entry_idx >= 0 else {}
        dev: float | None = None
        values = getattr(engine.precomputed, "daily_dev_pct_4h", None)
        if values and 0 <= entry_idx < len(engine.mapping):
            mapped = engine.mapping[entry_idx]
            if 0 <= mapped < len(values) and values[mapped] is not None and not math.isnan(values[mapped]):
                dev = float(values[mapped])
        mfe, mae = excursion_r(engine, trade)
        risk = trade.quantity * stop_dist if trade.quantity else 0.0
        rows.append(
            {
                "entry_time": trade.entry_time,
                "direction": trade.direction,
                "regime_label": trade.regime_label,
                "trail_style": trade.trail_style,
                "risk_regime": trade.risk_regime,
                "exit_reason": trade.exit_reason,
                "net_r": trade.pnl / risk if risk > 0 else float("nan"),
                "gross_r": trade.gross_pnl / risk if risk > 0 else float("nan"),
                "fee_r": trade.fees / risk if risk > 0 else float("nan"),
                "atr_mult": stop_dist / atr if atr > 0 else float("nan"),
                "stop_dist_pct": stop_dist / trade.entry_price * 100.0 if trade.entry_price else float("nan"),
                "daily_dev_pct": dev,
                "adx": float(features.get("adx", float("nan")) or float("nan")),
                "momentum": float(features.get("momentum", float("nan")) or float("nan")),
                "ema_gap": float(features.get("ema_gap", float("nan")) or float("nan")),
                "bars_held": (int(trade.exit_idx) - int(trade.entry_idx))
                if trade.exit_idx is not None and trade.entry_idx is not None
                else None,
                "mfe_r": mfe,
                "mae_r": mae,
            }
        )
    return pd.DataFrame(rows)


def bucket_table(frame: pd.DataFrame, column: str, bins: list[float], labels: list[str]) -> list[dict[str, Any]]:
    data = frame.dropna(subset=[column]).copy()
    if data.empty:
        return []
    data["bucket"] = pd.cut(data[column], bins=bins, labels=labels, include_lowest=True)
    rows: list[dict[str, Any]] = []
    for label, group in data.groupby("bucket", observed=True):
        rows.append(
            {
                "bucket": str(label),
                "trades": int(len(group)),
                "net_r_sum": round(float(group["net_r"].sum()), 2),
                "avg_net_r": round(float(group["net_r"].mean()), 4),
                "gross_r_sum": round(float(group["gross_r"].sum()), 2),
                "avg_gross_r": round(float(group["gross_r"].mean()), 4),
                "win_rate_pct": round(float((group["net_r"] > 0).mean() * 100.0), 1),
                "avg_fee_r": round(float(group["fee_r"].mean()), 4),
            }
        )
    return rows


def categorical_table(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    data = frame.dropna(subset=[column])
    rows: list[dict[str, Any]] = []
    for value, group in data.groupby(column):
        rows.append(
            {
                column: str(value),
                "trades": int(len(group)),
                "net_r_sum": round(float(group["net_r"].sum()), 2),
                "avg_net_r": round(float(group["net_r"].mean()), 4),
                "gross_r_sum": round(float(group["gross_r"].sum()), 2),
                "win_rate_pct": round(float((group["net_r"] > 0).mean() * 100.0), 1),
            }
        )
    return rows


def max_statistic_permutation(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    """对「该特征的某个阈值切分」做 max-statistic 置换检验。

    统计量 = 高低两组的平均净 R 之差（绝对值最大者）；置换时对**同一组阈值**
    重新取最大，从而把「搜索阈值」的多重比较惩罚算进去。
    """
    data = frame.dropna(subset=[column, "net_r"])
    if len(data) < 40:
        return {"feature": column, "note": "样本不足"}
    values = data[column].to_numpy(dtype=float)
    net_r = data["net_r"].to_numpy(dtype=float)
    thresholds = np.quantile(values, [0.25, 0.4, 0.5, 0.6, 0.75])

    def statistic(r: np.ndarray) -> tuple[float, float]:
        best = 0.0
        best_threshold = float("nan")
        for threshold in thresholds:
            high = r[values > threshold]
            low = r[values <= threshold]
            if len(high) < 10 or len(low) < 10:
                continue
            diff = float(high.mean() - low.mean())
            if abs(diff) > abs(best):
                best = diff
                best_threshold = float(threshold)
        return best, best_threshold

    observed, observed_threshold = statistic(net_r)
    rng = np.random.default_rng(RNG_SEED)
    exceed = 0
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(net_r)
        stat, _ = statistic(shuffled)
        if abs(stat) >= abs(observed):
            exceed += 1
    high_group = data[data[column] > observed_threshold]
    low_group = data[data[column] <= observed_threshold]
    return {
        "feature": column,
        "best_threshold": round(observed_threshold, 4),
        "high_trades": int(len(high_group)),
        "high_avg_net_r": round(float(high_group["net_r"].mean()), 4),
        "low_trades": int(len(low_group)),
        "low_avg_net_r": round(float(low_group["net_r"].mean()), 4),
        "diff": round(observed, 4),
        "permutation_p": round((exceed + 1) / (PERMUTATIONS + 1), 4),
        "note": "p 已包含阈值搜索的多重比较惩罚",
    }


def main() -> None:
    args = parse_args()
    payload = load_base_config(Path(args.config))
    prepared = load_prepared_data(
        Path(args.data_15m),
        Path(args.data_4h),
        pd.Timestamp(args.start_date, tz="UTC"),
        payload.get("regime_switcher_thresholds"),
    )
    metrics, engine = run_engine(payload, prepared, args.start_date)
    frame = build_features(engine, trade_dataframe(engine))

    report: dict[str, Any] = {
        "start_date": args.start_date,
        "trades": int(len(frame)),
        "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 2),
        "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0)), 2),
        "overall": {
            "net_r_sum": round(float(frame["net_r"].sum()), 2),
            "gross_r_sum": round(float(frame["gross_r"].sum()), 2),
            "avg_net_r": round(float(frame["net_r"].mean()), 4),
            "avg_gross_r": round(float(frame["gross_r"].mean()), 4),
            "avg_fee_r": round(float(frame["fee_r"].mean()), 4),
            "win_rate_pct": round(float((frame["net_r"] > 0).mean() * 100.0), 1),
            "median_mfe_r": round(float(frame["mfe_r"].median()), 3),
            "median_mae_r": round(float(frame["mae_r"].median()), 3),
            "pct_touch_2r": round(float((frame["mfe_r"] >= 2.0).mean() * 100.0), 1),
        },
        "buckets": {
            "atr_mult": bucket_table(frame, "atr_mult", [0, 2, 3, 4, 5, 99], ["<2", "2-3", "3-4", "4-5", ">5"]),
            "daily_dev_pct": bucket_table(
                frame, "daily_dev_pct", [-99, -5, 0, 5, 10, 12, 99], ["<-5", "-5..0", "0..5", "5..10", "10..12", ">12"]
            ),
            "adx": bucket_table(frame, "adx", [0, 20, 30, 40, 999], ["<20", "20-30", "30-40", ">40"]),
            "bars_held": bucket_table(frame, "bars_held", [0, 8, 16, 32, 64, 9999], ["<8", "8-16", "16-32", "32-64", ">64"]),
        },
        "categorical": {
            "regime_label": categorical_table(frame, "regime_label"),
            "trail_style": categorical_table(frame, "trail_style"),
            "direction": categorical_table(frame, "direction"),
            "exit_reason": categorical_table(frame, "exit_reason"),
        },
        "permutation_tests": [
            max_statistic_permutation(frame, column)
            for column in ("atr_mult", "daily_dev_pct", "adx", "momentum", "ema_gap", "stop_dist_pct")
        ],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"\n输出: {output_path}")


if __name__ == "__main__":
    main()
