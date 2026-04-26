#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_config_report import DEFAULT_DATA_15M, DEFAULT_DATA_4H, load_config_payload  # noqa: E402
from scripts.live_readiness_report import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    date_string,
    load_prepared_data,
    shadow_risk_gate_overlay,
    run_engine,
    trade_dataframe,
)


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan shadow risk-gate cooldown parameters on autoTIT trade sequences.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--daily-loss-values", default="0,3,4,5,6,7,8,10,12")
    parser.add_argument("--equity-dd-values", default="0,10,12,15,18,20,25,30,35")
    parser.add_argument("--equity-cooldown-values", default="0,1,3,5,7,10,14")
    parser.add_argument("--loss-streak-values", default="0,2,3,4,5,6,8")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stdout-json", action="store_true")
    return parser.parse_args()


def compact_raw(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 2),
        "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
        "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0)), 2),
        "total_trades": int(metrics.get("total_trades", 0)),
    }


def run_window(payload: dict[str, Any], prepared: Any, start_date: str) -> dict[str, Any]:
    metrics, engine = run_engine(payload, prepared, start_date)
    trades = trade_dataframe(engine)
    return {
        "raw": compact_raw(metrics),
        "trades": trades,
        "initial_capital": float(metrics.get("initial_capital", 1000.0)),
    }


def score_candidate(result: dict[str, Any], baselines: dict[str, dict[str, Any]]) -> float:
    full = result["windows"]["full"]
    current_year = result["windows"]["current_year"]
    recent_60d = result["windows"]["recent_60d"]
    recent_30d = result["windows"]["recent_30d"]
    full_base = baselines["full"]["raw"]
    year_base = baselines["current_year"]["raw"]
    recent_60d_base = baselines["recent_60d"]["raw"]
    recent_30d_base = baselines["recent_30d"]["raw"]

    score = 0.0
    score += (full["total_return_pct"] - full_base["total_return_pct"]) / 1000.0
    score += (current_year["total_return_pct"] - year_base["total_return_pct"]) * 1.8
    score += (recent_60d["total_return_pct"] - recent_60d_base["total_return_pct"]) * 1.2
    score += (recent_30d["total_return_pct"] - recent_30d_base["total_return_pct"]) * 0.8
    score += (full["sharpe_ratio"] - full_base["sharpe_ratio"]) * 6.0
    score += (current_year["sharpe_ratio"] - year_base["sharpe_ratio"]) * 8.0
    score += (recent_60d["sharpe_ratio"] - recent_60d_base["sharpe_ratio"]) * 5.0
    score += (full_base["max_drawdown_pct"] - full["max_drawdown_pct"]) * 0.25
    score += (year_base["max_drawdown_pct"] - current_year["max_drawdown_pct"]) * 0.4
    score -= max(0.0, recent_30d_base["total_return_pct"] - recent_30d["total_return_pct"]) * 2.0
    score -= result["skip_ratio_full"] * 15.0
    return round(score, 6)


def candidate_passes(result: dict[str, Any], baselines: dict[str, dict[str, Any]]) -> bool:
    for window in ("full", "current_year", "recent_60d"):
        candidate = result["windows"][window]
        baseline = baselines[window]["raw"]
        if candidate["total_return_pct"] <= baseline["total_return_pct"]:
            return False
        if candidate["sharpe_ratio"] < baseline["sharpe_ratio"]:
            return False
        if candidate["max_drawdown_pct"] > baseline["max_drawdown_pct"]:
            return False
    recent_30d = result["windows"]["recent_30d"]
    recent_30d_base = baselines["recent_30d"]["raw"]
    if recent_30d["total_return_pct"] < recent_30d_base["total_return_pct"]:
        return False
    return True


def build_candidate(
    daily_loss: float,
    equity_dd: float,
    equity_cooldown: int,
    loss_streak: int,
    windows: dict[str, dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    params = {
        "daily_loss_stop_pct": daily_loss,
        "equity_drawdown_stop_pct": equity_dd,
        "equity_drawdown_cooldown_days": equity_cooldown,
        "consecutive_loss_stop": loss_streak,
    }
    replayed: dict[str, Any] = {}
    for name, window in windows.items():
        replayed[name] = shadow_risk_gate_overlay(
            trades=window["trades"],
            initial_capital=window["initial_capital"],
            daily_loss_stop_pct=daily_loss,
            equity_drawdown_stop_pct=equity_dd,
            consecutive_loss_stop=loss_streak,
            equity_drawdown_cooldown_days=equity_cooldown,
        )
    full_trades = max(1, int(baselines["full"]["raw"]["total_trades"]))
    result = {
        "params": params,
        "windows": replayed,
        "skip_ratio_full": round(replayed["full"]["skipped_trades"] / full_trades, 4),
    }
    result["passes_stability_filter"] = candidate_passes(result, baselines)
    result["score"] = score_candidate(result, baselines)
    return result


def output_path_for(output_dir: Path, end_date: str) -> Path:
    return output_dir / f"shadow_risk_gate_scan_{end_date}.json"


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    payload = load_config_payload(config_path)
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=payload.get("regime_switcher_thresholds"),
    )
    end_date = date_string(prepared.end)
    current_year_start = f"{prepared.end.year}-01-01"
    last_60_start = date_string(max(prepared.start, prepared.end - pd.Timedelta(days=60)))
    last_30_start = date_string(max(prepared.start, prepared.end - pd.Timedelta(days=30)))

    window_starts = {
        "full": args.start_date,
        "current_year": current_year_start,
        "recent_60d": last_60_start,
        "recent_30d": last_30_start,
    }
    windows = {
        name: run_window(payload, prepared, start_date)
        for name, start_date in window_starts.items()
    }
    baselines = {name: {"raw": data["raw"]} for name, data in windows.items()}

    daily_values = parse_float_list(args.daily_loss_values)
    equity_dd_values = parse_float_list(args.equity_dd_values)
    equity_cooldown_values = parse_int_list(args.equity_cooldown_values)
    loss_streak_values = parse_int_list(args.loss_streak_values)

    candidates: list[dict[str, Any]] = []
    for daily_loss, equity_dd, equity_cooldown, loss_streak in itertools.product(
        daily_values,
        equity_dd_values,
        equity_cooldown_values,
        loss_streak_values,
    ):
        if equity_dd <= 0 and equity_cooldown > 0:
            continue
        if equity_dd > 0 and equity_cooldown <= 0:
            continue
        candidate = build_candidate(
            daily_loss=daily_loss,
            equity_dd=equity_dd,
            equity_cooldown=equity_cooldown,
            loss_streak=loss_streak,
            windows=windows,
            baselines=baselines,
        )
        candidates.append(candidate)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    stable = [item for item in candidates if item["passes_stability_filter"]]
    stable.sort(key=lambda item: item["score"], reverse=True)

    report = {
        "config": str(config_path.resolve()),
        "data": {
            "start": str(prepared.start),
            "end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
        },
        "window_starts": window_starts,
        "raw_baselines": {name: data["raw"] for name, data in baselines.items()},
        "search_space": {
            "daily_loss_values": daily_values,
            "equity_dd_values": equity_dd_values,
            "equity_cooldown_values": equity_cooldown_values,
            "loss_streak_values": loss_streak_values,
            "candidate_count": len(candidates),
        },
        "top_stable": stable[: args.top_n],
        "top_scored": candidates[: args.top_n],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_path_for(output_dir, end_date)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output_path)
    print(f"candidate_count={len(candidates)} stable_count={len(stable)}")
    print("raw_baselines:")
    for name, raw in report["raw_baselines"].items():
        print(
            f"  {name:12s} return={raw['total_return_pct']:8.2f}% "
            f"sharpe={raw['sharpe_ratio']:6.3f} maxdd={raw['max_drawdown_pct']:6.2f}% trades={raw['total_trades']:3d}"
        )
    print("top_stable:")
    for idx, item in enumerate(stable[: args.top_n], start=1):
        params = item["params"]
        full = item["windows"]["full"]
        year = item["windows"]["current_year"]
        recent_60d = item["windows"]["recent_60d"]
        recent_30d = item["windows"]["recent_30d"]
        print(
            f"{idx:02d} score={item['score']:8.3f} "
            f"daily={params['daily_loss_stop_pct']:>4g} dd={params['equity_drawdown_stop_pct']:>4g} "
            f"cool={params['equity_drawdown_cooldown_days']:>2d} streak={params['consecutive_loss_stop']:>2d} | "
            f"full={full['total_return_pct']:8.2f}%/{full['sharpe_ratio']:.3f}/{full['max_drawdown_pct']:.2f}% "
            f"ytd={year['total_return_pct']:7.2f}%/{year['sharpe_ratio']:.3f}/{year['max_drawdown_pct']:.2f}% "
            f"60d={recent_60d['total_return_pct']:7.2f}% "
            f"30d={recent_30d['total_return_pct']:7.2f}% "
            f"skip={item['skip_ratio_full']:.2%}"
        )
    if args.stdout_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
