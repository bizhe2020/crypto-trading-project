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
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_usdt_aggressive_grid.json"

AGGRESSIVE_LEVERAGE_PROFILES = {
    "dyn_cap10": {"base": 8.0, "offense": 10.0, "defense": 2.0},
    "base9_off10_def2": {"base": 9.0, "offense": 10.0, "defense": 2.0},
    "base10_off10_def1": {"base": 10.0, "offense": 10.0, "defense": 1.0},
    "base10_off10_def2": {"base": 10.0, "offense": 10.0, "defense": 2.0},
    "base8_off10_def4": {"base": 8.0, "offense": 10.0, "defense": 4.0},
    "base9_off10_def4": {"base": 9.0, "offense": 10.0, "defense": 4.0},
    "base10_off10_def4": {"base": 10.0, "offense": 10.0, "defense": 4.0},
    "base10_off10_def6": {"base": 10.0, "offense": 10.0, "defense": 6.0},
    "fixed10": {"base": 10.0, "offense": 10.0, "defense": 10.0},
}

AGGRESSIVE_STOP_VALUES = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggressive stop/leverage scan for QQQ/USDT with reward priority.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--okx-4h", default=str(DEFAULT_OKX_4H))
    parser.add_argument("--funding", default=str(DEFAULT_FUNDING))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--taker-fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    args = parser.parse_args()

    config, signal_path = load_signal_path(Path(args.config))
    bars = enrich_bars(attach_daily_state(load_okx_4h(Path(args.okx_4h)), signal_path))
    funding = load_funding(Path(args.funding))

    original_profiles = dict(base_grid.LEVERAGE_PROFILES)
    base_grid.LEVERAGE_PROFILES = AGGRESSIVE_LEVERAGE_PROFILES
    try:
        results = []
        for lev_name in AGGRESSIVE_LEVERAGE_PROFILES:
            for stop_loss_pct in AGGRESSIVE_STOP_VALUES:
                results.append(
                    base_grid.simulate(
                        bars,
                        funding,
                        leverage_profile_name=lev_name,
                        stop_loss_pct=float(stop_loss_pct),
                        taker_fee_rate=float(args.taker_fee_rate),
                        slippage_bps=float(args.slippage_bps),
                        initial_capital=float(config["initial_capital"]),
                    )
                )
    finally:
        base_grid.LEVERAGE_PROFILES = original_profiles

    by_return = sorted(
        results,
        key=lambda x: (
            x["summary"]["total_return_pct"],
            -x["summary"]["max_drawdown_pct"],
            x["summary"]["avg_leverage_when_in"],
        ),
        reverse=True,
    )
    by_score = sorted(results, key=lambda x: x["summary"]["score"], reverse=True)
    payload = {
        "config": {
            "signal_frozen_label": config.get("frozen_label"),
            "taker_fee_rate": float(args.taker_fee_rate),
            "slippage_bps": float(args.slippage_bps),
            "stop_values": AGGRESSIVE_STOP_VALUES,
            "leverage_profiles": AGGRESSIVE_LEVERAGE_PROFILES,
        },
        "top_by_return": by_return[:12],
        "top_by_score": by_score[:12],
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(out)
    print(json.dumps({"top_by_return": by_return[:12], "top_by_score": by_score[:12]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
