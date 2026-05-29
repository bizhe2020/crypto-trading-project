#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_qqq_usdt_10x import load_funding  # noqa: E402
from scripts.scan_qqq_usdt_4h_triggers import attach_daily_state, load_okx_4h, load_signal_path  # noqa: E402
from scripts.scan_qqq_usdt_btc_logic_overlays import enrich_bars  # noqa: E402
from scripts.scan_qqq_usdt_stop_leverage_grid import simulate  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "config.paper.tqqq-only-strict-recovery-frozen.json"
DEFAULT_OKX_4H = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-4h-futures.feather"
DEFAULT_FUNDING = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-8h-funding_rate.feather"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_usdt_cost_sensitivity_audit.json"

FEE_RATES = [0.0005, 0.0007, 0.0010]
SLIPPAGE_BPS = [5.0, 7.5, 10.0, 15.0]
FUNDING_MULTIPLIERS = [1.0, 1.5, 2.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit fee/funding sensitivity for the current QQQ/USDT main candidate.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--okx-4h", default=str(DEFAULT_OKX_4H))
    parser.add_argument("--funding", default=str(DEFAULT_FUNDING))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--leverage-profile", default="dyn_cap10")
    parser.add_argument("--stop-loss-pct", type=float, default=2.5)
    args = parser.parse_args()

    config, signal_path = load_signal_path(Path(args.config))
    bars = enrich_bars(attach_daily_state(load_okx_4h(Path(args.okx_4h)), signal_path))
    funding_base = load_funding(Path(args.funding))

    results = []
    baseline = None
    for fee_rate in FEE_RATES:
        for slippage_bps in SLIPPAGE_BPS:
            for funding_multiplier in FUNDING_MULTIPLIERS:
                funding = funding_base.copy()
                funding["funding_rate_value"] = funding["funding_rate_value"] * float(funding_multiplier)
                result = simulate(
                    bars,
                    funding,
                    leverage_profile_name=str(args.leverage_profile),
                    stop_loss_pct=float(args.stop_loss_pct),
                    taker_fee_rate=float(fee_rate),
                    slippage_bps=float(slippage_bps),
                    initial_capital=float(config["initial_capital"]),
                )
                summary = result["summary"]
                item = {
                    "fee_rate": float(fee_rate),
                    "fee_bps": round(float(fee_rate) * 10000.0, 2),
                    "slippage_bps": float(slippage_bps),
                    "funding_multiplier": float(funding_multiplier),
                    "summary": summary,
                }
                if fee_rate == 0.0005 and slippage_bps == 5.0 and funding_multiplier == 1.0:
                    baseline = item
                results.append(item)

    if baseline is None:
        raise RuntimeError("Failed to compute baseline scenario")

    for item in results:
        item["delta_vs_baseline"] = {
            "total_return_pct": round(item["summary"]["total_return_pct"] - baseline["summary"]["total_return_pct"], 2),
            "max_drawdown_pct": round(item["summary"]["max_drawdown_pct"] - baseline["summary"]["max_drawdown_pct"], 2),
            "funding_cost_pct_est": round(item["summary"]["funding_cost_pct_est"] - baseline["summary"]["funding_cost_pct_est"], 2),
        }

    top_by_score = sorted(results, key=lambda x: x["summary"]["score"], reverse=True)[:10]
    worst_by_score = sorted(results, key=lambda x: x["summary"]["score"])[:10]
    worst_by_return = sorted(results, key=lambda x: x["summary"]["total_return_pct"])[:10]
    payload = {
        "candidate": {
            "signal_frozen_label": config.get("frozen_label"),
            "leverage_profile": str(args.leverage_profile),
            "stop_loss_pct": float(args.stop_loss_pct),
        },
        "baseline": baseline,
        "top_by_score": top_by_score,
        "worst_by_score": worst_by_score,
        "worst_by_return": worst_by_return,
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(out)
    print(json.dumps({"baseline": baseline, "top_by_score": top_by_score[:5], "worst_by_return": worst_by_return[:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
