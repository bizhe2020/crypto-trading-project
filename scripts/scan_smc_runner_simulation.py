#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.high_leverage_repro_params import DEFAULT_PRESSURE_PARAMS_PATH  # noqa: E402
from scripts.live_readiness_report import load_prepared_data  # noqa: E402
from scripts.report_smc_trade_context import load_best_shadow_params  # noqa: E402
from scripts.scan_high_leverage_expansion import parse_float_list, parse_int_list  # noqa: E402
from scripts.scan_shadow_on_fixed_high_leverage import add_windows, replay_shadow_events  # noqa: E402


DEFAULT_INPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_context" / "smc_htf_liquidity_targets_report_min1r_lookahead192.json"
DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_context" / "smc_runner_simulation_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate partial runner exits to HTF BSL/SSL liquidity targets.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--pressure-params", default=str(DEFAULT_PRESSURE_PARAMS_PATH))
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--runner-fractions", default="0.05,0.10,0.15,0.20")
    parser.add_argument("--min-target-rr-values", default="1.0,1.25,1.5,2.0")
    parser.add_argument("--lookahead-bars-values", default="48,96,192")
    parser.add_argument("--timeout-modes", default="original,close")
    parser.add_argument("--stop-modes", default="none")
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--atr-multipliers", default="1.5,2.0,2.5")
    parser.add_argument("--trail-activation-rr-values", default="0.0")
    parser.add_argument("--allowed-exit-reasons", default="all")
    parser.add_argument("--allowed-h4-pd-sides", default="all")
    parser.add_argument("--allowed-regime-labels", default="all")
    parser.add_argument("--allowed-target-sources", default="all")
    parser.add_argument("--only-positive-original-values", default="true,false")
    parser.add_argument("--accounting-modes", default="accounting,timed")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def parse_bool_list(value: str) -> list[bool]:
    output: list[bool] = []
    for item in value.split(","):
        normalized = item.strip().lower()
        if not normalized:
            continue
        if normalized in {"true", "1", "yes", "on"}:
            output.append(True)
        elif normalized in {"false", "0", "no", "off"}:
            output.append(False)
        else:
            raise ValueError(f"invalid boolean: {item}")
    return output


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def timestamp_to_idx(prepared: Any) -> dict[str, int]:
    return {str(pd.Timestamp(candle.ts, unit="s", tz="UTC")): idx for idx, candle in enumerate(prepared.c15m)}


def price_signal_return(direction: str, entry: float, exit_price: float) -> float:
    if entry <= 0:
        return 0.0
    if direction == "BULL":
        return (exit_price - entry) / entry
    if direction == "BEAR":
        return (entry - exit_price) / entry
    return 0.0


def allowed_exit_reason(row: dict[str, Any], allowed: str) -> bool:
    if allowed == "all":
        return True
    values = {item.strip() for item in allowed.split("+") if item.strip()}
    return str(row.get("exit_reason") or "") in values


def allowed_field_value(row: dict[str, Any], field: str, allowed: str) -> bool:
    if allowed == "all":
        return True
    values = {item.strip() for item in allowed.split("+") if item.strip()}
    return str(row.get(field) or "") in values


def atr_values(prepared: Any, period: int) -> list[float | None]:
    trs: list[float] = []
    output: list[float | None] = []
    prev_close: float | None = None
    for candle in prepared.c15m:
        high = float(candle.h)
        low = float(candle.l)
        if prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        if len(trs) >= period:
            output.append(sum(trs[-period:]) / period)
        else:
            output.append(None)
        prev_close = float(candle.c)
    return output


def stop_price_for_mode(
    row: dict[str, Any],
    direction: str,
    mode: str,
    atr: float | None,
    atr_multiplier: float,
    high_water: float,
    low_water: float,
) -> float | None:
    entry = float(row.get("entry_price", 0.0) or 0.0)
    if mode == "none":
        return None
    if mode == "breakeven":
        return entry
    if mode == "original_exit":
        original_return = float(row.get("return", 0.0) or 0.0)
        leverage = float(row.get("effective_leverage", 0.0) or 0.0)
        if leverage <= 0:
            return entry
        signal_return = original_return / leverage
        if direction == "BULL":
            return entry * (1.0 + signal_return)
        if direction == "BEAR":
            return entry * (1.0 - signal_return)
    if mode in {"atr", "chandelier"} and atr is not None:
        if mode == "atr":
            if direction == "BULL":
                return entry - atr * atr_multiplier
            if direction == "BEAR":
                return entry + atr * atr_multiplier
        if mode == "chandelier":
            if direction == "BULL":
                return high_water - atr * atr_multiplier
            if direction == "BEAR":
                return low_water + atr * atr_multiplier
    return None


def find_runner_exit(
    row: dict[str, Any],
    prepared: Any,
    atrs: list[float | None],
    exit_idx: int,
    lookahead_bars: int,
    timeout_mode: str,
    stop_mode: str,
    atr_multiplier: float,
    trail_activation_rr: float,
) -> dict[str, Any]:
    target = row.get("htf_selected_target_level")
    if target is None:
        return {"reason": "no_target", "idx": exit_idx, "price": None, "time": row.get("exit_time")}
    direction = str(row.get("direction") or "")
    level = float(target)
    entry = float(row.get("entry_price", 0.0) or 0.0)
    risk = abs(entry - float(row.get("initial_stop_price", entry) or entry))
    start = min(len(prepared.c15m) - 1, max(0, exit_idx + 1))
    end = min(len(prepared.c15m) - 1, exit_idx + max(int(lookahead_bars), 0))
    high_water = entry
    low_water = entry
    for idx in range(start, end + 1):
        candle = prepared.c15m[idx]
        high_water = max(high_water, float(candle.h))
        low_water = min(low_water, float(candle.l))
        if direction == "BULL" and float(candle.h) >= level:
            return {
                "reason": "target",
                "idx": idx,
                "price": level,
                "time": str(pd.Timestamp(candle.ts, unit="s", tz="UTC")),
            }
        if direction == "BEAR" and float(candle.l) <= level:
            return {
                "reason": "target",
                "idx": idx,
                "price": level,
                "time": str(pd.Timestamp(candle.ts, unit="s", tz="UTC")),
            }
        if risk > 0:
            if direction == "BULL":
                current_rr = (high_water - entry) / risk
            elif direction == "BEAR":
                current_rr = (entry - low_water) / risk
            else:
                current_rr = 0.0
        else:
            current_rr = 0.0
        if current_rr >= trail_activation_rr:
            stop_price = stop_price_for_mode(
                row=row,
                direction=direction,
                mode=stop_mode,
                atr=atrs[idx],
                atr_multiplier=atr_multiplier,
                high_water=high_water,
                low_water=low_water,
            )
            if stop_price is not None:
                if direction == "BULL" and float(candle.l) <= stop_price:
                    return {
                        "reason": f"stop_{stop_mode}",
                        "idx": idx,
                        "price": stop_price,
                        "time": str(pd.Timestamp(candle.ts, unit="s", tz="UTC")),
                    }
                if direction == "BEAR" and float(candle.h) >= stop_price:
                    return {
                        "reason": f"stop_{stop_mode}",
                        "idx": idx,
                        "price": stop_price,
                        "time": str(pd.Timestamp(candle.ts, unit="s", tz="UTC")),
                    }
    if timeout_mode == "close" and end >= 0:
        candle = prepared.c15m[end]
        return {
            "reason": "timeout_close",
            "idx": end,
            "price": float(candle.c),
            "time": str(pd.Timestamp(candle.ts, unit="s", tz="UTC")),
        }
    return {"reason": "timeout_original", "idx": exit_idx, "price": None, "time": row.get("exit_time")}


def build_adjusted_events(
    rows: list[dict[str, Any]],
    prepared: Any,
    atrs: list[float | None],
    ts_to_idx: dict[str, int],
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runner_fraction = float(params["runner_fraction"])
    min_target_rr = float(params["min_target_rr"])
    lookahead_bars = int(params["lookahead_bars"])
    timeout_mode = str(params["timeout_mode"])
    stop_mode = str(params["stop_mode"])
    atr_multiplier = float(params["atr_multiplier"])
    trail_activation_rr = float(params["trail_activation_rr"])
    allowed_exit_reasons = str(params["allowed_exit_reasons"])
    allowed_h4_pd_sides = str(params["allowed_h4_pd_sides"])
    allowed_regime_labels = str(params["allowed_regime_labels"])
    allowed_target_sources = str(params["allowed_target_sources"])
    only_positive_original = bool(params["only_positive_original"])
    accounting_mode = str(params["accounting_mode"])

    events: list[dict[str, Any]] = []
    adjusted_count = 0
    target_count = 0
    timeout_close_count = 0
    timeout_original_count = 0
    stop_count = 0
    stop_counts: dict[str, int] = {}
    delta_sum = 0.0
    for row in rows:
        original_return = float(row.get("return", 0.0) or 0.0)
        event = {
            "entry_time": row.get("entry_time"),
            "exit_time": row.get("exit_time"),
            "return": original_return,
            "direction": row.get("direction"),
            "entry_price": row.get("entry_price"),
            "exit_reason": row.get("exit_reason"),
            "runner_applied": False,
        }
        target_rr = row.get("htf_selected_target_rr")
        if target_rr is None or float(target_rr) < min_target_rr:
            events.append(event)
            continue
        if not allowed_exit_reason(row, allowed_exit_reasons):
            events.append(event)
            continue
        if not allowed_field_value(row, "h4_pd_side", allowed_h4_pd_sides):
            events.append(event)
            continue
        if not allowed_field_value(row, "regime_label", allowed_regime_labels):
            events.append(event)
            continue
        if not allowed_field_value(row, "htf_selected_target_source", allowed_target_sources):
            events.append(event)
            continue
        if only_positive_original and original_return <= 0:
            events.append(event)
            continue
        if runner_fraction <= 0:
            events.append(event)
            continue

        exit_time = str(pd.Timestamp(row["exit_time"]).tz_convert("UTC"))
        exit_idx = ts_to_idx.get(exit_time)
        if exit_idx is None:
            events.append(event)
            continue
        runner_exit = find_runner_exit(
            row=row,
            prepared=prepared,
            atrs=atrs,
            exit_idx=exit_idx,
            lookahead_bars=lookahead_bars,
            timeout_mode=timeout_mode,
            stop_mode=stop_mode,
            atr_multiplier=atr_multiplier,
            trail_activation_rr=trail_activation_rr,
        )
        reason = str(runner_exit["reason"])
        if reason == "no_target":
            events.append(event)
            continue
        if reason == "target":
            entry = float(row.get("entry_price", 0.0) or 0.0)
            leverage = float(row.get("effective_leverage", 0.0) or 0.0)
            runner_return = price_signal_return(str(row.get("direction") or ""), entry, float(runner_exit["price"])) * leverage
            target_count += 1
        elif reason == "timeout_close":
            entry = float(row.get("entry_price", 0.0) or 0.0)
            leverage = float(row.get("effective_leverage", 0.0) or 0.0)
            runner_return = price_signal_return(str(row.get("direction") or ""), entry, float(runner_exit["price"])) * leverage
            timeout_close_count += 1
        elif reason.startswith("stop_"):
            entry = float(row.get("entry_price", 0.0) or 0.0)
            leverage = float(row.get("effective_leverage", 0.0) or 0.0)
            runner_return = price_signal_return(str(row.get("direction") or ""), entry, float(runner_exit["price"])) * leverage
            stop_count += 1
            stop_counts[reason] = stop_counts.get(reason, 0) + 1
        else:
            runner_return = original_return
            timeout_original_count += 1

        adjusted_return = (1.0 - runner_fraction) * original_return + runner_fraction * runner_return
        adjusted_count += 1
        delta_sum += adjusted_return - original_return
        event.update(
            {
                "return": adjusted_return,
                "exit_time": runner_exit["time"] if accounting_mode == "timed" else row.get("exit_time"),
                "runner_applied": True,
                "runner_fraction": runner_fraction,
                "runner_exit_reason": reason,
                "runner_target_rr": float(target_rr),
                "runner_target_level": row.get("htf_selected_target_level"),
                "runner_return": runner_return,
                "original_return": original_return,
                "runner_delta": adjusted_return - original_return,
            }
        )
        events.append(event)

    diagnostics = {
        "adjusted_trades": adjusted_count,
        "target_hits": target_count,
        "timeout_close": timeout_close_count,
        "timeout_original": timeout_original_count,
        "stops": stop_count,
        "stop_counts": stop_counts,
        "runner_delta_sum_pct": round(delta_sum * 100.0, 4),
    }
    return events, diagnostics


def score_result(result: dict[str, Any]) -> float:
    year = result.get("windows", {}).get("current_year", {})
    recent_60d = result.get("windows", {}).get("last_60d", {})
    recent_30d = result.get("windows", {}).get("last_30d", {})
    return round(
        float(result["total_return_pct"])
        + float(year.get("total_return_pct", 0.0)) * 180.0
        + float(recent_60d.get("total_return_pct", 0.0)) * 100.0
        + float(recent_30d.get("total_return_pct", 0.0)) * 80.0
        - float(result["max_drawdown_pct"]) * 60.0
        - float(year.get("max_drawdown_pct", 0.0)) * 50.0,
        4,
    )


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text())
    rows = payload.get("rows", [])
    prepared = load_prepared_data(
        data_15m_path=Path(args.data_15m),
        data_4h_path=Path(args.data_4h),
        start=pd.Timestamp(args.start_date, tz="UTC"),
        threshold_payload=None,
    )
    ts_to_idx = timestamp_to_idx(prepared)
    atrs = atr_values(prepared, args.atr_period)
    shadow_params = load_best_shadow_params(Path(args.pressure_params))
    initial_capital = 1000.0

    results: list[dict[str, Any]] = []
    for runner_fraction in parse_float_list(args.runner_fractions):
        for min_target_rr in parse_float_list(args.min_target_rr_values):
            for lookahead_bars in parse_int_list(args.lookahead_bars_values):
                for timeout_mode in [item.strip() for item in args.timeout_modes.split(",") if item.strip()]:
                    for stop_mode in [item.strip() for item in args.stop_modes.split(",") if item.strip()]:
                        for atr_multiplier in parse_float_list(args.atr_multipliers):
                            if stop_mode not in {"atr", "chandelier"} and atr_multiplier != parse_float_list(args.atr_multipliers)[0]:
                                continue
                            for trail_activation_rr in parse_float_list(args.trail_activation_rr_values):
                                for allowed_exit_reasons in [item.strip() for item in args.allowed_exit_reasons.split(",") if item.strip()]:
                                    for allowed_h4_pd_sides in [item.strip() for item in args.allowed_h4_pd_sides.split(",") if item.strip()]:
                                        for allowed_regime_labels in [item.strip() for item in args.allowed_regime_labels.split(",") if item.strip()]:
                                            for allowed_target_sources in [item.strip() for item in args.allowed_target_sources.split(",") if item.strip()]:
                                                for only_positive in parse_bool_list(args.only_positive_original_values):
                                                    for accounting_mode in [item.strip() for item in args.accounting_modes.split(",") if item.strip()]:
                                                        params = {
                                                            "runner_fraction": runner_fraction,
                                                            "min_target_rr": min_target_rr,
                                                            "lookahead_bars": lookahead_bars,
                                                            "timeout_mode": timeout_mode,
                                                            "stop_mode": stop_mode,
                                                            "atr_multiplier": atr_multiplier,
                                                            "trail_activation_rr": trail_activation_rr,
                                                            "allowed_exit_reasons": allowed_exit_reasons,
                                                            "allowed_h4_pd_sides": allowed_h4_pd_sides,
                                                            "allowed_regime_labels": allowed_regime_labels,
                                                            "allowed_target_sources": allowed_target_sources,
                                                            "only_positive_original": only_positive,
                                                            "accounting_mode": accounting_mode,
                                                        }
                                                        events, diagnostics = build_adjusted_events(rows, prepared, atrs, ts_to_idx, params)
                                                        shadow = replay_shadow_events(
                                                            events,
                                                            initial_capital,
                                                            daily_loss_stop_pct=shadow_params["daily_loss_stop_pct"],
                                                            equity_drawdown_stop_pct=shadow_params["equity_drawdown_stop_pct"],
                                                            consecutive_loss_stop=shadow_params["consecutive_loss_stop"],
                                                            equity_drawdown_cooldown_days=shadow_params["equity_drawdown_cooldown_days"],
                                                        )
                                                        result = add_windows(dict(shadow), initial_capital)
                                                        result["params"] = params
                                                        result["diagnostics"] = diagnostics
                                                        result["score"] = score_result(result)
                                                        results.append(result)

    results.sort(key=lambda item: item["score"], reverse=True)
    report = {
        "metadata": {
            "input": str(Path(args.input).resolve()),
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "rows": len(rows),
            "shadow_params": shadow_params,
            "candidate_count": len(results),
            "baseline": {
                "total_return_pct": 88481.28,
                "max_drawdown_pct": 33.87,
                "current_year_2026_return_pct": 29.87,
                "last_60d_return_pct": 7.85,
                "last_30d_return_pct": 8.47,
            },
        },
        "top": results[: args.top],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_for_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    for idx, item in enumerate(results[: args.top], start=1):
        year = item.get("windows", {}).get("current_year", {})
        recent_60d = item.get("windows", {}).get("last_60d", {})
        recent_30d = item.get("windows", {}).get("last_30d", {})
        print(
            f"{idx:02d} score={item['score']:.2f} full={item['total_return_pct']:.2f}%/"
            f"{item['max_drawdown_pct']:.2f}% 2026={year.get('total_return_pct', 0.0):.2f}%/"
            f"{year.get('max_drawdown_pct', 0.0):.2f}% 60d={recent_60d.get('total_return_pct', 0.0):.2f}% "
            f"30d={recent_30d.get('total_return_pct', 0.0):.2f}% "
            f"accepted={item['accepted_trades']} skipped={item['skipped_trades']} "
            f"diag={item['diagnostics']} params={item['params']}"
        )


if __name__ == "__main__":
    main()
