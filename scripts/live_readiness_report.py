#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.okx_executor import ExecutorConfig  # noqa: E402
from scripts.backtest_config_report import (  # noqa: E402
    DEFAULT_DATA_15M,
    DEFAULT_DATA_4H,
    load_config_payload,
    load_dataframe,
)
from strategy.scalp_robust_v2_core import (  # noqa: E402
    Candle,
    PrecomputedState,
    ScalpRobustEngine,
    align_timeframes,
    build_precomputed_state,
    dataframe_to_candles,
)


DEFAULT_OUTPUT_DIR = ROOT / "var" / "live_readiness"


@dataclass
class PreparedData:
    c4h: list[Candle]
    c15m: list[Candle]
    mapping: list[int]
    precomputed: PrecomputedState
    start: pd.Timestamp
    end: pd.Timestamp
    regime_labels: dict[int, str]
    regime_features: dict[int, dict[str, Any]]


def regime_label_from_features(features: dict[str, Any], thresholds: Any) -> str:
    if (
        features["adx"] >= thresholds.strong_high_growth_adx_min
        and features["momentum"] >= thresholds.strong_high_growth_momentum_min
    ):
        return "high_growth"
    if features["compression_growth_score"] >= thresholds.compression_growth_score_min:
        return "high_growth"
    if (
        features["adx"] >= thresholds.high_growth_adx_min
        and features["momentum"] >= thresholds.high_growth_momentum_min
        and features["ema_gap"] >= thresholds.high_growth_ema_gap_min
        and not features["bearish_structure"]
    ):
        return "high_growth"
    if (
        features["flat_score"] >= thresholds.flat_score_min
        and features["momentum"] >= thresholds.flat_momentum_min
        and features["ema_gap"] >= thresholds.flat_ema_gap_min
    ):
        return "flat"
    if features["normal_score"] >= thresholds.normal_score_min:
        return "normal"
    return "normal"


def structure_flags_for_idx(highs: list[float], lows: list[float], end_idx: int, window: int) -> dict[str, bool]:
    window = max(int(window), 2)
    if end_idx + 1 < window * 2:
        return {"higher_high": False, "higher_low": False, "lower_high": False, "lower_low": False}
    recent_start = end_idx - window + 1
    prev_start = end_idx - window * 2 + 1
    recent_high = max(highs[recent_start : end_idx + 1])
    prev_high = max(highs[prev_start:recent_start])
    recent_low = min(lows[recent_start : end_idx + 1])
    prev_low = min(lows[prev_start:recent_start])
    return {
        "higher_high": recent_high > prev_high,
        "higher_low": recent_low > prev_low,
        "lower_high": recent_high < prev_high,
        "lower_low": recent_low < prev_low,
    }


def precompute_regime_state(
    c4h: list[Candle],
    c4h_indices: list[int],
    threshold_payload: dict[str, Any] | None,
) -> tuple[dict[int, str], dict[int, dict[str, Any]]]:
    try:
        from scripts.regime_detector import _adx_series, _atr_series, _ema, _to_thresholds
    except Exception:
        return ({idx: "flat" for idx in c4h_indices}, {idx: {} for idx in c4h_indices})

    thresholds = _to_thresholds(threshold_payload)
    highs = [float(candle.h) for candle in c4h]
    lows = [float(candle.l) for candle in c4h]
    closes = [float(candle.c) for candle in c4h]
    atr_values = _atr_series(highs, lows, closes, thresholds.atr_period)
    adx_values = _adx_series(highs, lows, closes, thresholds.adx_period)
    ema_fast = _ema(closes, thresholds.ema_fast_period)
    ema_slow = _ema(closes, thresholds.ema_slow_period)
    min_history = max(
        thresholds.atr_baseline_window,
        thresholds.momentum_window,
        thresholds.ema_slow_period,
        thresholds.structure_window * 2,
    )
    atr_prefix = [0.0]
    for value in atr_values:
        atr_prefix.append(atr_prefix[-1] + value)

    labels: dict[int, str] = {}
    features_by_idx: dict[int, dict[str, Any]] = {}
    for c4h_idx in c4h_indices:
        history_len = c4h_idx
        if history_len < min_history:
            labels[c4h_idx] = "flat"
            features_by_idx[c4h_idx] = {}
            continue

        end_idx = c4h_idx - 1
        atr_start = end_idx - thresholds.atr_baseline_window + 1
        atr_baseline = (atr_prefix[end_idx + 1] - atr_prefix[atr_start]) / thresholds.atr_baseline_window
        atr_now = atr_values[end_idx]
        momentum = closes[end_idx] / closes[end_idx - thresholds.momentum_window] - 1.0
        ema_gap = ema_fast[end_idx] / ema_slow[end_idx] - 1.0 if ema_slow[end_idx] != 0 else 0.0
        structure = structure_flags_for_idx(highs, lows, end_idx, thresholds.structure_window)
        trend_conflict = (momentum > 0 and ema_gap < 0) or (momentum < 0 and ema_gap > 0)
        bearish_structure = structure["lower_high"] and structure["lower_low"]
        bullish_structure = structure["higher_high"] and structure["higher_low"]
        atr_ratio = atr_now / atr_baseline if atr_baseline > 0 else 1.0
        adx = adx_values[end_idx]
        strong_growth_score = sum(
            [
                adx >= thresholds.high_growth_adx_min,
                momentum >= thresholds.high_growth_momentum_min,
                ema_gap >= thresholds.high_growth_ema_gap_min,
                bullish_structure,
            ]
        )
        compression_growth_score = sum(
            [
                adx <= thresholds.compression_growth_adx_max,
                thresholds.compression_growth_atr_ratio_min <= atr_ratio <= thresholds.compression_growth_atr_ratio_max,
                momentum >= thresholds.compression_growth_momentum_min,
                ema_gap >= thresholds.compression_growth_ema_gap_min,
            ]
        )
        flat_score = sum(
            [
                adx <= thresholds.flat_adx_max,
                atr_ratio <= thresholds.flat_atr_ratio_max,
                abs(momentum) <= thresholds.flat_momentum_abs_max,
                ema_gap >= thresholds.flat_ema_gap_min,
            ]
        )
        normal_score = sum(
            [
                momentum < thresholds.normal_momentum_max,
                adx >= thresholds.normal_adx_min,
                bearish_structure or trend_conflict,
            ]
        )
        features = {
            "atr": atr_now,
            "atr_ratio": atr_ratio,
            "adx": adx,
            "momentum": momentum,
            "ema_gap": ema_gap,
            "structure": structure,
            "bullish_structure": bullish_structure,
            "bearish_structure": bearish_structure,
            "trend_conflict": trend_conflict,
            "strong_growth_score": strong_growth_score,
            "compression_growth_score": compression_growth_score,
            "high_growth_score": max(strong_growth_score, compression_growth_score),
            "flat_score": flat_score,
            "normal_score": normal_score,
        }
        labels[c4h_idx] = regime_label_from_features(features, thresholds)
        features_by_idx[c4h_idx] = features

    return labels, features_by_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live-readiness checks for the scalp autoTIT strategy.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--include-fee-sensitivity", action="store_true", help="Run slower full-window fee stress cases.")
    parser.add_argument("--include-yearly", action="store_true", help="Run slower per-year baseline/autoTIT checks.")
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args()


def date_string(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def load_prepared_data(
    data_15m_path: Path,
    data_4h_path: Path,
    start: pd.Timestamp,
    threshold_payload: dict[str, Any] | None,
) -> PreparedData:
    df15 = load_dataframe(data_15m_path, start=start)
    df4 = load_dataframe(data_4h_path)
    if df15.empty:
        raise ValueError(f"No 15m data loaded from {data_15m_path}")
    if df4.empty:
        raise ValueError(f"No 4h data loaded from {data_4h_path}")
    end = pd.Timestamp(df15["date"].max()).tz_convert("UTC")
    df4 = df4[df4["date"] <= end].sort_values("date").reset_index(drop=True)
    c4h = dataframe_to_candles(df4)
    c15m = dataframe_to_candles(df15)
    mapping = align_timeframes(c4h, c15m)
    precomputed = build_precomputed_state(c4h, c15m)
    regime_labels, regime_features = precompute_regime_state(c4h, sorted(set(mapping)), threshold_payload)
    return PreparedData(
        c4h=c4h,
        c15m=c15m,
        mapping=mapping,
        precomputed=precomputed,
        start=pd.Timestamp(df15["date"].min()).tz_convert("UTC"),
        end=end,
        regime_labels=regime_labels,
        regime_features=regime_features,
    )


def payload_without_autotit(payload: dict[str, Any]) -> dict[str, Any]:
    baseline = dict(payload)
    baseline["enable_time_based_trailing"] = False
    baseline["enable_auto_time_based_trailing"] = False
    return baseline


def payload_with_fee(payload: dict[str, Any], taker_fee_rate: float) -> dict[str, Any]:
    updated = dict(payload)
    updated["taker_fee_rate"] = taker_fee_rate
    return updated


def run_engine(
    payload: dict[str, Any],
    prepared: PreparedData,
    start_date: str,
) -> tuple[dict[str, Any], ScalpRobustEngine]:
    config = ExecutorConfig.from_dict(payload).to_scalp_strategy_config()
    engine = ScalpRobustEngine(
        prepared.c4h,
        prepared.c15m,
        prepared.mapping,
        prepared.precomputed,
        config,
    )
    engine._regime_switch_cache = {
        c4h_idx: (label, engine._config_for_regime(label))
        for c4h_idx, label in prepared.regime_labels.items()
    }
    engine._regime_feature_cache = dict(prepared.regime_features)
    metrics = engine.run_backtest(start_date=start_date)
    return metrics, engine


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 2),
        "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
        "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0)), 2),
        "profit_factor": round(float(metrics.get("profit_factor", 0.0)), 3),
        "win_rate": round(float(metrics.get("win_rate", 0.0)), 2),
        "total_trades": int(metrics.get("total_trades", 0)),
        "target_hit_rate": round(float(metrics.get("target_hit_rate", 0.0)), 2),
        "total_fees_paid": round(float(metrics.get("total_fees_paid", 0.0)), 2),
        "total_slippage_cost": round(float(metrics.get("total_slippage_cost", 0.0)), 2),
        "exit_reasons": metrics.get("exit_reasons", {}),
    }


def trade_dataframe(engine: ScalpRobustEngine) -> pd.DataFrame:
    df = pd.DataFrame([trade.__dict__ for trade in engine.trades])
    if df.empty:
        return df
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["hold_hours"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 3600.0
    return df


def monthly_summary(trades: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    if trades.empty:
        return {}
    out: dict[str, Any] = {}
    for month, bucket in trades.groupby(trades["exit_time"].dt.strftime("%Y-%m")):
        pnl = float(bucket["pnl"].sum())
        wins = int((bucket["pnl"] > 0).sum())
        losses = int((bucket["pnl"] <= 0).sum())
        out[month] = {
            "return_pct_on_initial": round(pnl / initial_capital * 100.0, 2),
            "pnl": round(pnl, 2),
            "trades": int(len(bucket)),
            "wins": wins,
            "losses": losses,
        }
    return out


def worst_trade_streak(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 0
    streak = 0
    worst = 0
    for pnl in trades["pnl"]:
        if float(pnl) <= 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    return worst


def trade_return_sharpe(returns: list[float]) -> float:
    if len(returns) <= 1:
        return 0.0
    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / len(returns)
    std_return = variance ** 0.5
    return (mean_return / std_return * (252 ** 0.5)) if std_return > 0 else 0.0


def max_drawdown_from_capitals(capitals: list[float], initial_capital: float) -> float:
    peak = initial_capital
    max_drawdown = 0.0
    for capital in capitals:
        peak = max(peak, capital)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - capital) / peak * 100.0)
    return max_drawdown


def shadow_risk_gate_overlay(
    trades: pd.DataFrame,
    initial_capital: float,
    daily_loss_stop_pct: float = 6.0,
    equity_drawdown_stop_pct: float = 20.0,
    consecutive_loss_stop: int = 4,
    equity_drawdown_cooldown_days: int = 7,
) -> dict[str, Any]:
    guard = {
        "mode": "shadow_risk_gate_overlay",
        "daily_loss_stop_pct": daily_loss_stop_pct,
        "equity_drawdown_stop_pct": equity_drawdown_stop_pct,
        "consecutive_loss_stop": consecutive_loss_stop,
        "equity_drawdown_cooldown_days": equity_drawdown_cooldown_days,
        "total_return_pct": 0.0,
        "final_capital": round(initial_capital, 2),
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "total_trades": 0,
        "skipped_trades": 0,
        "trigger_count": 0,
        "trigger_counts": {},
        "first_trigger": None,
    }
    if trades.empty:
        return guard

    ordered = trades.sort_values("entry_time").reset_index(drop=True)
    capital = initial_capital
    drawdown_peak = initial_capital
    loss_streak = 0
    pause_until = pd.Timestamp.min.tz_localize("UTC")
    day_start_capital: dict[pd.Timestamp, float] = {}
    day_pnl: dict[pd.Timestamp, float] = {}
    capitals: list[float] = []
    returns: list[float] = []
    trigger_counts: dict[str, int] = {}
    first_trigger: dict[str, Any] | None = None
    accepted_trades = 0
    skipped_trades = 0

    for _, trade in ordered.iterrows():
        entry_time = pd.Timestamp(trade["entry_time"]).tz_convert("UTC")
        exit_time = pd.Timestamp(trade["exit_time"]).tz_convert("UTC")
        if entry_time < pause_until:
            skipped_trades += 1
            continue

        capital_before = capital
        trade_return = float(trade["pnl_pct"])
        pnl = capital_before * trade_return
        capital += pnl
        accepted_trades += 1
        returns.append(trade_return)
        capitals.append(capital)
        drawdown_peak = max(drawdown_peak, capital)

        exit_day = exit_time.normalize()
        if exit_day not in day_start_capital:
            day_start_capital[exit_day] = capital_before
            day_pnl[exit_day] = 0.0
        day_pnl[exit_day] += pnl

        if pnl > 0:
            loss_streak = 0
        else:
            loss_streak += 1

        triggered_reasons: list[str] = []
        if daily_loss_stop_pct > 0 and day_start_capital[exit_day] > 0:
            daily_loss_pct = -day_pnl[exit_day] / day_start_capital[exit_day] * 100.0
            if daily_loss_pct >= daily_loss_stop_pct:
                triggered_reasons.append("daily_loss")
                pause_until = max(pause_until, exit_day + pd.Timedelta(days=1))

        if consecutive_loss_stop > 0 and loss_streak >= consecutive_loss_stop:
            triggered_reasons.append("consecutive_loss")
            pause_until = max(pause_until, exit_day + pd.Timedelta(days=1))
            loss_streak = 0

        if equity_drawdown_stop_pct > 0 and drawdown_peak > 0:
            drawdown_pct = (drawdown_peak - capital) / drawdown_peak * 100.0
            if drawdown_pct >= equity_drawdown_stop_pct:
                triggered_reasons.append("equity_drawdown")
                pause_until = max(pause_until, exit_day + pd.Timedelta(days=equity_drawdown_cooldown_days))
                drawdown_peak = capital
                loss_streak = 0

        if triggered_reasons:
            for reason in triggered_reasons:
                trigger_counts[reason] = trigger_counts.get(reason, 0) + 1
            if first_trigger is None:
                first_trigger = {
                    "time": str(exit_time),
                    "capital": round(capital, 2),
                    "reasons": triggered_reasons,
                    "pause_until": str(pause_until),
                }

    guard.update(
        {
            "total_return_pct": round((capital - initial_capital) / initial_capital * 100.0, 2),
            "final_capital": round(capital, 2),
            "sharpe_ratio": round(trade_return_sharpe(returns), 3),
            "max_drawdown_pct": round(max_drawdown_from_capitals(capitals, initial_capital), 2),
            "total_trades": accepted_trades,
            "skipped_trades": skipped_trades,
            "trigger_count": int(sum(trigger_counts.values())),
            "trigger_counts": trigger_counts,
            "first_trigger": first_trigger,
        }
    )
    return guard


def run_case(
    name: str,
    payload: dict[str, Any],
    prepared: PreparedData,
    start_date: str,
) -> dict[str, Any]:
    metrics, engine = run_engine(payload, prepared, start_date)
    trades = trade_dataframe(engine)
    compact = compact_metrics(metrics)
    compact["worst_loss_streak"] = worst_trade_streak(trades)
    compact["shadow_risk_gate_overlay"] = shadow_risk_gate_overlay(
        trades=trades,
        initial_capital=float(metrics.get("initial_capital", 1000.0)),
    )
    compact["monthly"] = monthly_summary(trades, float(metrics.get("initial_capital", 1000.0)))
    compact["name"] = name
    return compact


def build_report(
    config_path: Path,
    data_15m_path: Path,
    data_4h_path: Path,
    start_date: str,
    include_fee_sensitivity: bool,
    include_yearly: bool,
) -> dict[str, Any]:
    base_payload = load_config_payload(config_path)
    start = pd.Timestamp(start_date, tz="UTC")
    prepared = load_prepared_data(
        data_15m_path,
        data_4h_path,
        start,
        base_payload.get("regime_switcher_thresholds"),
    )
    end_date = date_string(prepared.end)
    current_year_start = f"{prepared.end.year}-01-01"
    last_60_start = date_string(max(prepared.start, prepared.end - pd.Timedelta(days=60)))
    last_30_start = date_string(max(prepared.start, prepared.end - pd.Timedelta(days=30)))

    baseline_payload = payload_without_autotit(base_payload)
    cases = {
        "baseline_full": (baseline_payload, start_date),
        "autotit_full": (base_payload, start_date),
        "baseline_current_year": (baseline_payload, current_year_start),
        "autotit_current_year": (base_payload, current_year_start),
        "autotit_last_60d": (base_payload, last_60_start),
        "autotit_last_30d": (base_payload, last_30_start),
    }
    if include_fee_sensitivity:
        base_fee = float(base_payload.get("taker_fee_rate", 0.0005))
        cases.update(
            {
                "autotit_fee_1x": (payload_with_fee(base_payload, base_fee * 1.0), start_date),
                "autotit_fee_1_5x": (payload_with_fee(base_payload, base_fee * 1.5), start_date),
                "autotit_fee_2x": (payload_with_fee(base_payload, base_fee * 2.0), start_date),
            }
        )

    case_results = {
        name: run_case(name, payload, prepared, case_start)
        for name, (payload, case_start) in cases.items()
    }

    years: dict[str, Any] = {}
    if include_yearly:
        for year in range(prepared.start.year, prepared.end.year + 1):
            year_start = f"{year}-01-01"
            year_end = pd.Timestamp(f"{year}-12-31", tz="UTC")
            if pd.Timestamp(year_start, tz="UTC") > prepared.end:
                continue
            if year_end < prepared.start:
                continue
            years[str(year)] = {
                "baseline": run_case(f"baseline_{year}", baseline_payload, prepared, year_start),
                "autotit": run_case(f"autotit_{year}", base_payload, prepared, year_start),
            }

    recommendations = {
        "suggested_daily_loss_stop_pct": 6.0,
        "suggested_equity_drawdown_stop_pct": 20.0,
        "suggested_equity_drawdown_cooldown_days": 7,
        "suggested_consecutive_loss_stop": 4,
        "suggested_min_live_probe_days": 14,
        "notes": [
            "The risk overlay is a shadow execution gate: the strategy path keeps running while real entries can be skipped.",
            "For live trading, require manual review before resuming after a 20% equity drawdown cooldown.",
            "Use small notional until live slippage and stop placement match backtest assumptions.",
            "Stop trading if local position state diverges from exchange position state.",
            "Re-run this report after every data refresh or strategy/config change.",
        ],
    }

    return {
        "config": str(config_path.resolve()),
        "data": {
            "data_15m": str(data_15m_path.resolve()),
            "data_4h": str(data_4h_path.resolve()),
            "start": str(prepared.start),
            "end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
        },
        "ranges": {
            "full_start": start_date,
            "current_year_start": current_year_start,
            "last_60d_start": last_60_start,
            "last_30d_start": last_30_start,
            "end": end_date,
        },
        "options": {
            "include_fee_sensitivity": include_fee_sensitivity,
            "include_yearly": include_yearly,
        },
        "cases": case_results,
        "yearly": years,
        "recommendations": recommendations,
    }


def output_path_for(output_dir: Path, end: str) -> Path:
    return output_dir / f"live_readiness_{end}.json"


def main() -> None:
    args = parse_args()
    report = build_report(
        config_path=Path(args.config),
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start_date=args.start_date,
        include_fee_sensitivity=args.include_fee_sensitivity,
        include_yearly=args.include_yearly,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_path_for(output_dir, report["ranges"]["end"])
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output_path)
    if args.stdout:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
