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

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.report_smc_trade_context import DEFAULT_OUTPUT as DEFAULT_SMC_REPORT  # noqa: E402
from scripts.report_smc_trade_context import load_best_shadow_params  # noqa: E402
from scripts.reproduce_smc_context_overlay import attach_smc_tags, load_smc_tags, summary_without_events  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay, parse_float_list  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, add_windows, replay_shadow_events  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_context" / "smc_context_overlay_formal_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal state-feedback scan for SMC context overlay multipliers.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--smc-report", default=str(DEFAULT_SMC_REPORT))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--h4-favorable-multipliers", default="1.0,1.03,1.05,1.08,1.10,1.12,1.15")
    parser.add_argument("--h4-adverse-multipliers", default="1.0,0.98,0.95")
    parser.add_argument("--low-score-multipliers", default="1.0,0.98,0.95")
    parser.add_argument("--london-multipliers", default="1.0,1.03,1.05,1.08,1.10,1.12")
    parser.add_argument("--recent-sweep-mss-multipliers", default="1.0,0.98,0.95")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def score_candidate(result: dict[str, Any]) -> float:
    year = result.get("windows", {}).get("current_year", {})
    recent_60d = result.get("windows", {}).get("last_60d", {})
    recent_30d = result.get("windows", {}).get("last_30d", {})
    return round(
        float(result["total_return_pct"])
        + float(year.get("total_return_pct", 0.0)) * 200.0
        + float(recent_60d.get("total_return_pct", 0.0)) * 120.0
        + float(recent_30d.get("total_return_pct", 0.0)) * 80.0
        - float(result["max_drawdown_pct"]) * 60.0
        - float(year.get("max_drawdown_pct", 0.0)) * 50.0,
        4,
    )


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    if pd.isna(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    smc_report = Path(args.smc_report)
    if not smc_report.exists():
        raise SystemExit(f"SMC report missing: {smc_report}. Run scripts/report_smc_trade_context.py first.")

    base_payload = load_config_payload(Path(args.config))
    payload, pressure_params = apply_pressure_params(base_payload, Path(args.pressure_params))
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=base_payload.get("regime_switcher_thresholds"),
    )
    metrics, engine = run_engine(payload, prepared, args.start_date)
    trades = enrich_trades_with_regime_features(trade_dataframe(engine), prepared)
    trades = attach_smc_tags(trades, load_smc_tags(smc_report))
    initial_capital = float(metrics.get("initial_capital", 1000.0))
    shadow_params = load_best_shadow_params(Path(args.pressure_params))

    candidates: list[dict[str, Any]] = []
    for h4_favorable, h4_adverse, low_score, london, recent_sweep_mss in itertools.product(
        parse_float_list(args.h4_favorable_multipliers),
        parse_float_list(args.h4_adverse_multipliers),
        parse_float_list(args.low_score_multipliers),
        parse_float_list(args.london_multipliers),
        parse_float_list(args.recent_sweep_mss_multipliers),
    ):
        params = dict(FIXED_STRUCTURE_PARAMS)
        params.update(
            {
                "smc_context_overlay_enabled": True,
                "smc_h4_favorable_multiplier": h4_favorable,
                "smc_h4_adverse_multiplier": h4_adverse,
                "smc_low_score_multiplier": low_score,
                "smc_london_multiplier": london,
                "smc_recent_sweep_mss_multiplier": recent_sweep_mss,
            }
        )
        fixed = expansion_overlay(trades, initial_capital, params, include_events=True)
        shadow = replay_shadow_events(
            fixed["events"],
            initial_capital,
            daily_loss_stop_pct=shadow_params["daily_loss_stop_pct"],
            equity_drawdown_stop_pct=shadow_params["equity_drawdown_stop_pct"],
            consecutive_loss_stop=shadow_params["consecutive_loss_stop"],
            equity_drawdown_cooldown_days=shadow_params["equity_drawdown_cooldown_days"],
        )
        result = add_windows(dict(shadow), initial_capital)
        result["params"] = {
            "smc_h4_favorable_multiplier": h4_favorable,
            "smc_h4_adverse_multiplier": h4_adverse,
            "smc_low_score_multiplier": low_score,
            "smc_london_multiplier": london,
            "smc_recent_sweep_mss_multiplier": recent_sweep_mss,
        }
        result["fixed_structure_overlay"] = summary_without_events(fixed)
        result["score"] = score_candidate(result)
        candidates.append(result)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    report = {
        "metadata": {
            "config": str(Path(args.config).resolve()),
            "pressure_params": str(Path(args.pressure_params).resolve()),
            "smc_report": str(smc_report.resolve()),
            "start_date": args.start_date,
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "candles_15m": len(prepared.c15m),
            "candles_4h": len(prepared.c4h),
            "pressure_params_applied": pressure_params,
            "shadow_params": shadow_params,
            "candidate_count": len(candidates),
        },
        "top": candidates[: args.top],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    for idx, item in enumerate(candidates[: args.top], start=1):
        year = item.get("windows", {}).get("current_year", {})
        recent_60d = item.get("windows", {}).get("last_60d", {})
        recent_30d = item.get("windows", {}).get("last_30d", {})
        print(
            f"{idx:02d} score={item['score']:.2f} full={item['total_return_pct']:.2f}%/"
            f"{item['max_drawdown_pct']:.2f}% 2026={year.get('total_return_pct', 0.0):.2f}%/"
            f"{year.get('max_drawdown_pct', 0.0):.2f}% 60d={recent_60d.get('total_return_pct', 0.0):.2f}% "
            f"30d={recent_30d.get('total_return_pct', 0.0):.2f}% "
            f"accepted={item['accepted_trades']} skipped={item['skipped_trades']} "
            f"smc={item['fixed_structure_overlay'].get('smc_context_overlay_applied')} params={item['params']}"
        )


if __name__ == "__main__":
    main()
