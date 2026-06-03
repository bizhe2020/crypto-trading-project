#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.scan_qqq_usdt_stop_leverage_grid as base_grid  # noqa: E402
from scripts.replay_qqq_usdt_10x import load_funding  # noqa: E402
from scripts.scan_qqq_usdt_4h_triggers import attach_daily_state, load_okx_4h, load_signal_path  # noqa: E402
from scripts.scan_qqq_usdt_btc_logic_overlays import enrich_bars  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "config.paper.tqqq-only-strict-recovery-frozen.json"
DEFAULT_OKX_4H = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-4h-futures.feather"
DEFAULT_FUNDING = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-8h-funding_rate.feather"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_usdt_leverage_state_candidate_audit.json"


def simulate_profile(
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    name: str,
    profile: dict[str, float],
    stop_loss_pct: float,
    taker_fee_rate: float,
    slippage_bps: float,
    initial_capital: float,
) -> dict:
    original_profiles = dict(base_grid.LEVERAGE_PROFILES)
    base_grid.LEVERAGE_PROFILES = {name: profile}
    try:
        return base_grid.simulate(
            bars,
            funding,
            leverage_profile_name=name,
            stop_loss_pct=stop_loss_pct,
            taker_fee_rate=taker_fee_rate,
            slippage_bps=slippage_bps,
            initial_capital=initial_capital,
        )
    finally:
        base_grid.LEVERAGE_PROFILES = original_profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness audit for a QQQ/USDT leverage-state candidate.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--okx-4h", default=str(DEFAULT_OKX_4H))
    parser.add_argument("--funding", default=str(DEFAULT_FUNDING))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--candidate-name", default="base10_off12_def1")
    parser.add_argument("--candidate-profile", default="10,12,1")
    parser.add_argument("--baseline-name", default="fixed10")
    parser.add_argument("--baseline-profile", default="10,10,10")
    parser.add_argument("--stop-loss-pct", type=float, default=3.5)
    args = parser.parse_args()

    config, signal_path = load_signal_path(Path(args.config))
    bars = enrich_bars(attach_daily_state(load_okx_4h(Path(args.okx_4h)), signal_path))
    funding = load_funding(Path(args.funding))
    initial_capital = float(config["initial_capital"])
    candidate_values = [float(item) for item in args.candidate_profile.split(",")]
    baseline_values = [float(item) for item in args.baseline_profile.split(",")]
    candidate = {"base": candidate_values[0], "offense": candidate_values[1], "defense": candidate_values[2]}
    baseline = {"base": baseline_values[0], "offense": baseline_values[1], "defense": baseline_values[2]}

    scenarios = []
    date_min = pd.to_datetime(bars["date"], utc=True).min()
    date_max = pd.to_datetime(bars["date"], utc=True).max()
    midpoint = date_min + (date_max - date_min) / 2
    scenario_defs = [
        ("full", bars),
        ("first_half", bars[bars["date"] <= midpoint].copy()),
        ("second_half", bars[bars["date"] > midpoint].copy()),
        ("drop_last_7d", bars[bars["date"] <= date_max - pd.Timedelta(days=7)].copy()),
        ("drop_last_14d", bars[bars["date"] <= date_max - pd.Timedelta(days=14)].copy()),
    ]
    cost_defs = [
        ("normal_cost", 0.0005, 5.0),
        ("stress_cost", 0.0008, 10.0),
    ]
    for window_name, window_bars in scenario_defs:
        if window_bars.empty:
            continue
        for cost_name, fee, slip in cost_defs:
            base_result = simulate_profile(
                window_bars,
                funding,
                name=args.baseline_name,
                profile=baseline,
                stop_loss_pct=float(args.stop_loss_pct),
                taker_fee_rate=fee,
                slippage_bps=slip,
                initial_capital=initial_capital,
            )
            cand_result = simulate_profile(
                window_bars,
                funding,
                name=args.candidate_name,
                profile=candidate,
                stop_loss_pct=float(args.stop_loss_pct),
                taker_fee_rate=fee,
                slippage_bps=slip,
                initial_capital=initial_capital,
            )
            scenarios.append(
                {
                    "window": window_name,
                    "cost": cost_name,
                    "start": str(pd.to_datetime(window_bars["date"], utc=True).min()),
                    "end": str(pd.to_datetime(window_bars["date"], utc=True).max()),
                    "bars": int(len(window_bars)),
                    "baseline": base_result["summary"],
                    "candidate": cand_result["summary"],
                    "delta_return_pct": round(
                        cand_result["summary"]["total_return_pct"] - base_result["summary"]["total_return_pct"],
                        2,
                    ),
                    "delta_dd_pct": round(
                        cand_result["summary"]["max_drawdown_pct"] - base_result["summary"]["max_drawdown_pct"],
                        2,
                    ),
                }
            )

    payload = {
        "config": {
            "signal_frozen_label": config.get("frozen_label"),
            "candidate_name": args.candidate_name,
            "candidate_profile": candidate,
            "baseline_name": args.baseline_name,
            "baseline_profile": baseline,
            "stop_loss_pct": float(args.stop_loss_pct),
        },
        "scenarios": scenarios,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(out)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
