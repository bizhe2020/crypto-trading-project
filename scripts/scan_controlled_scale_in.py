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
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay, parse_float_list, parse_int_list, parse_str_list  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, add_windows, replay_shadow_events  # noqa: E402


DEFAULT_BEST_PARAMS = ROOT / "config" / "high_leverage_pressure_target_cap_best.params.json"


def parse_bool_list(value: str) -> list[bool]:
    out: list[bool] = []
    for item in parse_str_list(value):
        normalized = item.lower()
        if normalized in {"1", "true", "yes", "on"}:
            out.append(True)
        elif normalized in {"0", "false", "no", "off"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean value: {item}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan controlled scale-in on the current best pressure target-cap strategy.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--best-params", default=str(DEFAULT_BEST_PARAMS))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--include-disabled-baseline", action="store_true")
    parser.add_argument("--scale-in-trigger-rr-values", default="0.75,1.0,1.25,1.5")
    parser.add_argument("--scale-in-min-bars-held-values", default="2,4,8")
    parser.add_argument("--scale-in-min-interval-bars-values", default="8")
    parser.add_argument("--scale-in-risk-fraction-values", default="0.25,0.5,0.75")
    parser.add_argument("--scale-in-total-risk-multiplier-values", default="1.0")
    parser.add_argument("--scale-in-max-total-notional-multiplier-values", default="1.0")
    parser.add_argument("--scale-in-min-target-rr-values", default="1.5,2.0,2.5")
    parser.add_argument("--scale-in-min-price-move-pct-values", default="0.0,0.5")
    parser.add_argument("--scale-in-max-stop-distance-pct-values", default="1.5,2.0")
    parser.add_argument("--scale-in-require-stop-at-breakeven-values", default="true")
    parser.add_argument(
        "--scale-in-regime-label-sets",
        default="high_growth",
        help="Comma-separated label sets. Use + inside a set, e.g. high_growth+normal,high_growth.",
    )
    parser.add_argument(
        "--scale-in-trail-style-sets",
        default="loose,all",
        help="Comma-separated style sets. Use + inside a set, e.g. loose+normal,all.",
    )
    parser.add_argument("--daily-loss-stop-pct", type=float, default=6.0)
    parser.add_argument("--equity-drawdown-stop-pct", type=float, default=15.0)
    parser.add_argument("--equity-drawdown-cooldown-days", type=int, default=2)
    parser.add_argument("--consecutive-loss-stop", type=int, default=0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", default=str(ROOT / "var" / "high_leverage_expansion" / "controlled_scale_in_scan.json"))
    return parser.parse_args()


def load_best_pressure_params(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    params = payload.get("pressure_level_target_cap_params")
    if not isinstance(params, dict):
        raise ValueError(f"Missing pressure_level_target_cap_params in {path}")
    return dict(params)


def scale_in_param_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if args.include_disabled_baseline:
        candidates.append({"enable_controlled_scale_in": False})

    for (
        trigger_rr,
        min_bars,
        min_interval,
        risk_fraction,
        total_risk_multiplier,
        max_notional_multiplier,
        min_target_rr,
        min_price_move_pct,
        max_stop_distance_pct,
        require_be,
        regime_label_set,
        trail_style_set,
    ) in itertools.product(
        parse_float_list(args.scale_in_trigger_rr_values),
        parse_int_list(args.scale_in_min_bars_held_values),
        parse_int_list(args.scale_in_min_interval_bars_values),
        parse_float_list(args.scale_in_risk_fraction_values),
        parse_float_list(args.scale_in_total_risk_multiplier_values),
        parse_float_list(args.scale_in_max_total_notional_multiplier_values),
        parse_float_list(args.scale_in_min_target_rr_values),
        parse_float_list(args.scale_in_min_price_move_pct_values),
        parse_float_list(args.scale_in_max_stop_distance_pct_values),
        parse_bool_list(args.scale_in_require_stop_at_breakeven_values),
        parse_str_list(args.scale_in_regime_label_sets),
        parse_str_list(args.scale_in_trail_style_sets),
    ):
        candidates.append(
            {
                "enable_controlled_scale_in": True,
                "scale_in_max_slots": 2,
                "scale_in_trigger_rr": trigger_rr,
                "scale_in_min_bars_held": min_bars,
                "scale_in_min_interval_bars": min_interval,
                "scale_in_risk_fraction": risk_fraction,
                "scale_in_total_risk_multiplier": total_risk_multiplier,
                "scale_in_max_total_notional_multiplier": max_notional_multiplier,
                "scale_in_min_target_rr": min_target_rr,
                "scale_in_min_price_move_pct": min_price_move_pct,
                "scale_in_max_stop_distance_pct": max_stop_distance_pct,
                "scale_in_require_stop_at_breakeven": require_be,
                "scale_in_regime_labels": None if regime_label_set == "all" else regime_label_set.split("+"),
                "scale_in_trail_styles": None if trail_style_set == "all" else trail_style_set.split("+"),
            }
        )
    return candidates


def summarize_scale_in(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty or "scale_in_count" not in trades.columns:
        return {"trades_with_scale_in": 0, "scale_in_events": 0, "scale_in_notional": 0.0}
    counts = pd.to_numeric(trades["scale_in_count"], errors="coerce").fillna(0)
    notionals = pd.to_numeric(trades.get("scale_in_notional", 0.0), errors="coerce").fillna(0.0)
    return {
        "trades_with_scale_in": int((counts > 0).sum()),
        "scale_in_events": int(counts.sum()),
        "scale_in_notional": round(float(notionals.sum()), 2),
    }


def evaluate_candidate(
    base_payload: dict[str, Any],
    pressure_params: dict[str, Any],
    scale_in_params: dict[str, Any],
    prepared: Any,
    start_date: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = deepcopy(base_payload)
    payload.update(pressure_params)
    payload.update(scale_in_params)
    metrics, engine = run_engine(payload, prepared, start_date)
    trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
    initial_capital = float(metrics.get("initial_capital", 1000.0))
    fixed = expansion_overlay(trades, initial_capital, FIXED_STRUCTURE_PARAMS, include_events=True)
    shadow = replay_shadow_events(
        fixed["events"],
        initial_capital,
        daily_loss_stop_pct=float(args.daily_loss_stop_pct),
        equity_drawdown_stop_pct=float(args.equity_drawdown_stop_pct),
        consecutive_loss_stop=int(args.consecutive_loss_stop),
        equity_drawdown_cooldown_days=int(args.equity_drawdown_cooldown_days),
    )
    shadow = add_windows(shadow, initial_capital)
    year = shadow.get("windows", {}).get("current_year", {})
    recent_60d = shadow.get("windows", {}).get("last_60d", {})
    recent_30d = shadow.get("windows", {}).get("last_30d", {})
    score = round(
        float(shadow["total_return_pct"])
        + float(year.get("total_return_pct", 0.0)) * 150.0
        + float(recent_60d.get("total_return_pct", 0.0)) * 80.0
        + float(recent_30d.get("total_return_pct", 0.0)) * 30.0
        - float(shadow["max_drawdown_pct"]) * 30.0
        - float(year.get("max_drawdown_pct", 0.0)) * 40.0,
        4,
    )
    return {
        "score": score,
        "scale_in_params": scale_in_params,
        "scale_in_summary": summarize_scale_in(trades),
        "engine": {
            "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 2),
            "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0)), 2),
            "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
            "total_trades": int(metrics.get("total_trades", 0)),
            "exit_reasons": metrics.get("exit_reasons", {}),
        },
        "fixed_structure_overlay": {key: value for key, value in fixed.items() if key != "events"},
        "shadow": shadow,
    }


def main() -> None:
    args = parse_args()
    base_payload = load_config_payload(Path(args.config))
    pressure_params = load_best_pressure_params(Path(args.best_params))
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=base_payload.get("regime_switcher_thresholds"),
    )

    candidates = scale_in_param_grid(args)
    results: list[dict[str, Any]] = []
    for index, params in enumerate(candidates, start=1):
        result = evaluate_candidate(base_payload, pressure_params, params, prepared, args.start_date, args)
        results.append(result)
        shadow = result["shadow"]
        year = shadow.get("windows", {}).get("current_year", {})
        summary = result["scale_in_summary"]
        print(
            f"{index:03d}/{len(candidates):03d} score={result['score']:.2f} "
            f"full={shadow['total_return_pct']:.2f}%/{shadow['max_drawdown_pct']:.2f}% "
            f"year={year.get('total_return_pct', 0.0):.2f}% "
            f"scale_events={summary['scale_in_events']} params={params}",
            flush=True,
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "config": str(Path(args.config).resolve()),
        "best_params": str(Path(args.best_params).resolve()),
        "data": {
            "start": str(prepared.start),
            "end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
        },
        "pressure_params": pressure_params,
        "shadow_params": {
            "daily_loss_stop_pct": args.daily_loss_stop_pct,
            "equity_drawdown_stop_pct": args.equity_drawdown_stop_pct,
            "equity_drawdown_cooldown_days": args.equity_drawdown_cooldown_days,
            "consecutive_loss_stop": args.consecutive_loss_stop,
        },
        "candidate_count": len(results),
        "top": results[: args.top],
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)
    for idx, item in enumerate(results[: args.top], start=1):
        shadow = item["shadow"]
        year = shadow.get("windows", {}).get("current_year", {})
        recent_60d = shadow.get("windows", {}).get("last_60d", {})
        summary = item["scale_in_summary"]
        print(
            f"{idx:02d} score={item['score']:.2f} full={shadow['total_return_pct']:.2f}%/"
            f"{shadow['max_drawdown_pct']:.2f}% year={year.get('total_return_pct', 0.0):.2f}% "
            f"60d={recent_60d.get('total_return_pct', 0.0):.2f}% "
            f"scale_events={summary['scale_in_events']} params={item['scale_in_params']}"
        )


if __name__ == "__main__":
    main()
