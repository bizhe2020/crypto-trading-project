#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay, parse_float_list, parse_int_list  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, add_windows, replay_shadow_events  # noqa: E402


def parse_bool_list(value: str) -> list[bool]:
    out: list[bool] = []
    for item in [part.strip().lower() for part in value.split(",") if part.strip()]:
        if item in {"1", "true", "yes", "on"}:
            out.append(True)
        elif item in {"0", "false", "no", "off"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean value: {item}")
    return out


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan failed-breakout offense leverage guard on the fixed high-leverage overlay.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument(
        "--pressure-params",
        default=str(DEFAULT_PRESSURE_PARAMS_PATH),
        help="JSON reproduction file with pressure_level_target_cap_params. Use 'none' to skip.",
    )
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--enabled-values", default="false,true")
    parser.add_argument("--guard-leverage-values", default="2.0,4.0")
    parser.add_argument("--min-leverage-values", default="7.5")
    parser.add_argument("--min-quality-score-values", default="1,2,3")
    parser.add_argument("--min-momentum-pct-values", default="0.5,1.0,1.5")
    parser.add_argument("--min-ema-gap-pct-values", default="0.0,0.25,0.35")
    parser.add_argument("--min-adx-values", default="0.0,18.0,22.0")
    parser.add_argument("--regime-label-sets", default="high_growth")
    parser.add_argument("--risk-mode-sets", default="offense")
    parser.add_argument("--direction-sets", default="BULL")
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", default=str(ROOT / "var" / "high_leverage_expansion" / "failed_breakout_offense_guard_scan.json"))
    return parser.parse_args()


def split_set(value: str) -> list[str] | None:
    if value == "all":
        return None
    return value.split("+")


def guard_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for enabled in parse_bool_list(args.enabled_values):
        if not enabled:
            candidate = {"failed_breakout_guard_enabled": False}
            key = json.dumps(candidate, sort_keys=True)
            if key not in seen:
                candidates.append(candidate)
                seen.add(key)
            continue
        for values in itertools.product(
            parse_float_list(args.guard_leverage_values),
            parse_float_list(args.min_leverage_values),
            parse_int_list(args.min_quality_score_values),
            parse_float_list(args.min_momentum_pct_values),
            parse_float_list(args.min_ema_gap_pct_values),
            parse_float_list(args.min_adx_values),
            parse_str_list(args.regime_label_sets),
            parse_str_list(args.risk_mode_sets),
            parse_str_list(args.direction_sets),
        ):
            (
                guard_leverage,
                min_leverage,
                min_quality_score,
                min_momentum_pct,
                min_ema_gap_pct,
                min_adx,
                regime_label_set,
                risk_mode_set,
                direction_set,
            ) = values
            candidate = {
                "failed_breakout_guard_enabled": True,
                "failed_breakout_guard_leverage": guard_leverage,
                "failed_breakout_guard_min_leverage": min_leverage,
                "failed_breakout_guard_min_quality_score": min_quality_score,
                "failed_breakout_guard_min_momentum_pct": min_momentum_pct,
                "failed_breakout_guard_min_ema_gap_pct": min_ema_gap_pct,
                "failed_breakout_guard_min_adx": min_adx,
                "failed_breakout_guard_regime_labels": split_set(regime_label_set),
                "failed_breakout_guard_risk_modes": split_set(risk_mode_set),
                "failed_breakout_guard_directions": split_set(direction_set),
            }
            key = json.dumps(candidate, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    return candidates


def score_result(shadow: dict[str, Any]) -> float:
    year = shadow.get("windows", {}).get("current_year", {})
    recent_60d = shadow.get("windows", {}).get("last_60d", {})
    recent_30d = shadow.get("windows", {}).get("last_30d", {})
    return round(
        float(shadow["total_return_pct"])
        + float(year.get("total_return_pct", 0.0)) * 150.0
        + float(recent_60d.get("total_return_pct", 0.0)) * 80.0
        + float(recent_30d.get("total_return_pct", 0.0)) * 30.0
        - float(shadow["max_drawdown_pct"]) * 30.0
        - float(year.get("max_drawdown_pct", 0.0)) * 40.0,
        4,
    )


def main() -> None:
    args = parse_args()
    payload = load_config_payload(Path(args.config))
    pressure_params: dict[str, Any] = {}
    pressure_params_path: str | None = None
    if str(args.pressure_params).strip().lower() != "none":
        path = Path(args.pressure_params)
        payload, pressure_params = apply_pressure_params(payload, path)
        pressure_params_path = str(path.resolve())
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=payload.get("regime_switcher_thresholds"),
    )
    metrics, engine = run_engine(payload, prepared, args.start_date)
    trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
    initial_capital = float(metrics.get("initial_capital", 1000.0))

    results: list[dict[str, Any]] = []
    candidates = guard_candidates(args)
    for index, guard_params in enumerate(candidates, start=1):
        overlay_params = deepcopy(FIXED_STRUCTURE_PARAMS)
        overlay_params.update(guard_params)
        fixed = expansion_overlay(trades, initial_capital, overlay_params, include_events=True)
        shadow = replay_shadow_events(
            fixed["events"],
            initial_capital,
            daily_loss_stop_pct=float(args.daily_loss_stop_pct),
            equity_drawdown_stop_pct=float(args.equity_drawdown_stop_pct),
            consecutive_loss_stop=int(args.consecutive_loss_stop),
            equity_drawdown_cooldown_days=int(args.equity_drawdown_cooldown_days),
        )
        shadow = add_windows(shadow, initial_capital)
        result = {
            "score": score_result(shadow),
            "guard_params": guard_params,
            "fixed_structure_overlay": {key: value for key, value in fixed.items() if key != "events"},
            "shadow": shadow,
        }
        results.append(result)
        year = shadow.get("windows", {}).get("current_year", {})
        recent_60d = shadow.get("windows", {}).get("last_60d", {})
        print(
            f"{index:03d}/{len(candidates):03d} score={result['score']:.2f} "
            f"full={shadow['total_return_pct']:.2f}%/{shadow['max_drawdown_pct']:.2f}% "
            f"year={year.get('total_return_pct', 0.0):.2f}% "
            f"60d={recent_60d.get('total_return_pct', 0.0):.2f}% "
            f"guarded={fixed.get('failed_breakout_guard_applied', 0)} params={guard_params}",
            flush=True,
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    report = {
        "config": str(Path(args.config).resolve()),
        "pressure_params_path": pressure_params_path,
        "pressure_params": pressure_params,
        "data": {
            "start": str(prepared.start),
            "end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
        },
        "shadow_params": {
            "daily_loss_stop_pct": args.daily_loss_stop_pct,
            "equity_drawdown_stop_pct": args.equity_drawdown_stop_pct,
            "equity_drawdown_cooldown_days": args.equity_drawdown_cooldown_days,
            "consecutive_loss_stop": args.consecutive_loss_stop,
        },
        "candidate_count": len(results),
        "top": results[: args.top],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
