#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_usdt_leverage_state_scan.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused QQQ/USDT base/offense/defense leverage-state scan.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--okx-4h", default=str(DEFAULT_OKX_4H))
    parser.add_argument("--funding", default=str(DEFAULT_FUNDING))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--stop-loss-pct", type=float, default=3.5)
    parser.add_argument("--base-values", default="6,8,10")
    parser.add_argument("--offense-values", default="10,12")
    parser.add_argument("--defense-values", default="1,2,3,4,10")
    parser.add_argument("--taker-fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    args = parser.parse_args()

    config, signal_path = load_signal_path(Path(args.config))
    bars = enrich_bars(attach_daily_state(load_okx_4h(Path(args.okx_4h)), signal_path))
    funding = load_funding(Path(args.funding))

    base_values = [float(item) for item in str(args.base_values).split(",") if item]
    offense_values = [float(item) for item in str(args.offense_values).split(",") if item]
    defense_values = [float(item) for item in str(args.defense_values).split(",") if item]
    profiles = {}
    for base in base_values:
        for offense in offense_values:
            if offense < base:
                continue
            for defense in defense_values:
                if defense > base:
                    continue
                name = f"base{base:g}_off{offense:g}_def{defense:g}"
                profiles[name] = {"base": base, "offense": offense, "defense": defense}
    profiles["fixed10"] = {"base": 10.0, "offense": 10.0, "defense": 10.0}

    original_profiles = dict(base_grid.LEVERAGE_PROFILES)
    base_grid.LEVERAGE_PROFILES = profiles
    try:
        results = [
            base_grid.simulate(
                bars,
                funding,
                leverage_profile_name=name,
                stop_loss_pct=float(args.stop_loss_pct),
                taker_fee_rate=float(args.taker_fee_rate),
                slippage_bps=float(args.slippage_bps),
                initial_capital=float(config["initial_capital"]),
            )
            for name in profiles
        ]
    finally:
        base_grid.LEVERAGE_PROFILES = original_profiles

    def sort_key(item: dict) -> tuple[float, float, float]:
        summary = item["summary"]
        return (
            float(summary["total_return_pct"]),
            -float(summary["max_drawdown_pct"]),
            -float(summary["funding_cost_pct_est"]),
        )

    by_return = sorted(results, key=sort_key, reverse=True)
    by_score = sorted(results, key=lambda item: item["summary"]["score"], reverse=True)
    baseline = next((item for item in results if item["leverage_profile"] == "fixed10"), None)
    payload = {
        "config": {
            "signal_frozen_label": config.get("frozen_label"),
            "stop_loss_pct": float(args.stop_loss_pct),
            "base_values": base_values,
            "offense_values": offense_values,
            "defense_values": defense_values,
            "taker_fee_rate": float(args.taker_fee_rate),
            "slippage_bps": float(args.slippage_bps),
            "profiles": profiles,
        },
        "baseline": baseline,
        "top_by_return": by_return[:12],
        "top_by_score": by_score[:12],
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(out)
    print(json.dumps({"baseline": baseline, "top_by_return": by_return[:12], "top_by_score": by_score[:12]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
