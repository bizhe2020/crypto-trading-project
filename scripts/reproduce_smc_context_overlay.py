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

from scripts.backtest_config_report import load_config_payload  # noqa: E402
from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH, apply_pressure_params  # noqa: E402
from scripts.live_readiness_report import load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.report_smc_trade_context import DEFAULT_OUTPUT as DEFAULT_SMC_REPORT  # noqa: E402
from scripts.report_smc_trade_context import load_best_shadow_params  # noqa: E402
from scripts.scan_high_leverage_expansion import enrich_trades_with_regime_features, expansion_overlay  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import FIXED_STRUCTURE_PARAMS, add_windows, replay_shadow_events  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_context" / "smc_context_overlay_formal_reproduction.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce formal high-leverage overlay with optional SMC context risk multipliers."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.5x-3pct.json"))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--smc-report", default=str(DEFAULT_SMC_REPORT))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--cases", default="baseline,conservative,boundary")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def parse_case_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def case_params(name: str) -> dict[str, Any]:
    cases: dict[str, dict[str, Any]] = {
        "baseline": {
            "smc_context_overlay_enabled": False,
            "smc_h4_favorable_multiplier": 1.0,
            "smc_h4_adverse_multiplier": 1.0,
            "smc_low_score_multiplier": 1.0,
            "smc_london_multiplier": 1.0,
            "smc_recent_sweep_mss_multiplier": 1.0,
        },
        "conservative": {
            "smc_context_overlay_enabled": True,
            "smc_h4_favorable_multiplier": 1.15,
            "smc_h4_adverse_multiplier": 1.0,
            "smc_low_score_multiplier": 1.0,
            "smc_london_multiplier": 1.10,
            "smc_recent_sweep_mss_multiplier": 1.0,
        },
        "boundary": {
            "smc_context_overlay_enabled": True,
            "smc_h4_favorable_multiplier": 1.30,
            "smc_h4_adverse_multiplier": 1.0,
            "smc_low_score_multiplier": 1.0,
            "smc_london_multiplier": 1.20,
            "smc_recent_sweep_mss_multiplier": 1.0,
        },
        "h4_favorable_108": {
            "smc_context_overlay_enabled": True,
            "smc_h4_favorable_multiplier": 1.08,
            "smc_h4_adverse_multiplier": 1.0,
            "smc_low_score_multiplier": 1.0,
            "smc_london_multiplier": 1.0,
            "smc_recent_sweep_mss_multiplier": 1.0,
        },
        "saturation": {
            "smc_context_overlay_enabled": True,
            "smc_h4_favorable_multiplier": 1.60,
            "smc_h4_adverse_multiplier": 1.0,
            "smc_low_score_multiplier": 1.0,
            "smc_london_multiplier": 1.50,
            "smc_recent_sweep_mss_multiplier": 1.0,
        },
    }
    if name not in cases:
        raise ValueError(f"unknown case {name!r}; choose one of {', '.join(sorted(cases))}")
    return dict(cases[name])


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


def normalized_time(value: Any) -> str:
    return str(pd.Timestamp(value).tz_convert("UTC"))


def load_smc_tags(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("fixed_rows") or []
    tags: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry_time = row.get("entry_time")
        if not entry_time:
            continue
        tags[normalized_time(entry_time)] = {
            "smc_h4_pd_side": row.get("h4_pd_side"),
            "smc_session_bucket": row.get("session_bucket"),
            "smc_score": row.get("smc_score"),
            "smc_recent_sweep_mss": bool(row.get("recent_sweep_mss")),
        }
    return tags


def attach_smc_tags(trades: pd.DataFrame, tags: dict[str, dict[str, Any]]) -> pd.DataFrame:
    tagged = trades.copy()
    defaults = {
        "smc_h4_pd_side": "unknown",
        "smc_session_bucket": "unknown",
        "smc_score": 0,
        "smc_recent_sweep_mss": False,
    }
    for column, default in defaults.items():
        tagged[column] = default

    for idx, trade in tagged.iterrows():
        entry_time = trade.get("entry_time")
        if pd.isna(entry_time):
            continue
        smc = tags.get(normalized_time(entry_time))
        if not smc:
            continue
        for column, value in smc.items():
            tagged.at[idx, column] = value
    return tagged


def summary_without_events(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "events"}


def yearly_windows(events: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    if not events:
        return {}
    years = sorted({pd.Timestamp(event["entry_time"]).tz_convert("UTC").year for event in events})
    output: dict[str, Any] = {}
    for year in years:
        selected = [
            event
            for event in events
            if pd.Timestamp(event["entry_time"]).tz_convert("UTC").year == year
        ]
        replayed = replay_shadow_events(
            selected,
            initial_capital,
            daily_loss_stop_pct=0.0,
            equity_drawdown_stop_pct=0.0,
            consecutive_loss_stop=0,
            equity_drawdown_cooldown_days=0,
        )
        output[str(year)] = summary_without_events(replayed)
    return output


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

    cases: dict[str, Any] = {}
    for name in parse_case_names(args.cases):
        params = dict(FIXED_STRUCTURE_PARAMS)
        params.update(case_params(name))
        fixed = expansion_overlay(trades, initial_capital, params, include_events=True)
        shadow = replay_shadow_events(
            fixed["events"],
            initial_capital,
            daily_loss_stop_pct=shadow_params["daily_loss_stop_pct"],
            equity_drawdown_stop_pct=shadow_params["equity_drawdown_stop_pct"],
            consecutive_loss_stop=shadow_params["consecutive_loss_stop"],
            equity_drawdown_cooldown_days=shadow_params["equity_drawdown_cooldown_days"],
        )
        shadow_with_events = dict(shadow)
        shadow_summary = add_windows(dict(shadow), initial_capital)
        cases[name] = {
            "params": params,
            "fixed_structure_overlay": summary_without_events(fixed),
            "shadow": shadow_summary,
            "yearly": yearly_windows(shadow_with_events["events"], initial_capital),
        }

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
        },
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    for name, item in cases.items():
        shadow = item["shadow"]
        windows = shadow.get("windows", {})
        year = windows.get("current_year", {})
        recent_60d = windows.get("last_60d", {})
        recent_30d = windows.get("last_30d", {})
        print(
            f"{name}: full={shadow['total_return_pct']:.2f}% maxdd={shadow['max_drawdown_pct']:.2f}% "
            f"2026={year.get('total_return_pct', 0.0):.2f}% "
            f"60d={recent_60d.get('total_return_pct', 0.0):.2f}% "
            f"30d={recent_30d.get('total_return_pct', 0.0):.2f}% "
            f"accepted={shadow['accepted_trades']} skipped={shadow['skipped_trades']} "
            f"smc_applied={item['fixed_structure_overlay'].get('smc_context_overlay_applied')}"
        )


if __name__ == "__main__":
    main()
