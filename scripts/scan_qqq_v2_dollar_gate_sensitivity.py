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

from scripts.audit_qqq_shadow_gate_v2_combined import (  # noqa: E402
    DEFAULT_BTC_FROZEN,
    DEFAULT_NQ_4H,
    DEFAULT_NQ_FUNDING,
    DEFAULT_QQQ_USDT_CONFIG,
    DEFAULT_ROUTER_CONFIG,
    annual_metrics,
    load_enriched_bars,
    parse_end_timestamp,
)
from scripts.audit_qqq_v2_macro_proxy_overlay import (  # noqa: E402
    attach_macro_context,
    load_daily_macro_proxy_context,
    run_policy,
)
from scripts.replay_proxy_strategy_router import build_btc_path_from_frozen_artifact  # noqa: E402
from scripts.replay_qqq_usdt_10x import load_funding  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_v2_dollar_gate_sensitivity_20220101_20260529.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan dollar-gate sensitivity on top of QQQ V2 macro proxy overlay replay.")
    parser.add_argument("--config", default=str(DEFAULT_QQQ_USDT_CONFIG))
    parser.add_argument("--router-config", default=str(DEFAULT_ROUTER_CONFIG))
    parser.add_argument("--nq-data-4h", default=str(DEFAULT_NQ_4H))
    parser.add_argument("--nq-funding", default=str(DEFAULT_NQ_FUNDING))
    parser.add_argument("--btc-frozen", default=str(DEFAULT_BTC_FROZEN))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2026-05-29")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def build_scan_policies() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for threshold in [1.0, 1.25, 1.5, 1.75, 2.0]:
        policies.append(
            {
                "name": f"dollar_flat_z{str(threshold).replace('.', '_')}",
                "kind": "cash",
                "rule": "dollar",
                "dollar_z_threshold": threshold,
                "label": f"Dollar stress flat (z >= {threshold})",
            }
        )
        policies.append(
            {
                "name": f"dollar_cap50_z{str(threshold).replace('.', '_')}",
                "kind": "tiered",
                "rule": "dollar",
                "dollar_z_threshold": threshold,
                "multiplier": 0.5,
                "label": f"Dollar stress cap 50% (z >= {threshold})",
            }
        )
    for dollar_threshold in [1.25, 1.5]:
        for credit_oas_threshold in [0.5, 1.0]:
            policies.append(
                {
                    "name": f"dollar_credit_flat_d{str(dollar_threshold).replace('.', '_')}_c{str(credit_oas_threshold).replace('.', '_')}",
                    "kind": "cash",
                    "rule": "dollar_and_credit",
                    "dollar_z_threshold": dollar_threshold,
                    "credit_oas_z_threshold": credit_oas_threshold,
                    "credit_rel_z_threshold": -1.0,
                    "label": f"Dollar + credit flat (d >= {dollar_threshold}, credit >= {credit_oas_threshold})",
                }
            )
            policies.append(
                {
                    "name": f"dollar_credit_cap50_d{str(dollar_threshold).replace('.', '_')}_c{str(credit_oas_threshold).replace('.', '_')}",
                    "kind": "tiered",
                    "rule": "dollar_and_credit",
                    "dollar_z_threshold": dollar_threshold,
                    "credit_oas_z_threshold": credit_oas_threshold,
                    "credit_rel_z_threshold": -1.0,
                    "multiplier": 0.5,
                    "label": f"Dollar + credit cap 50% (d >= {dollar_threshold}, credit >= {credit_oas_threshold})",
                }
            )
    return policies


def summarize_candidate(name: str, policy: dict[str, Any], summary: dict[str, Any], annual: dict[str, Any]) -> dict[str, Any]:
    router = summary["router"]
    qqq = summary["qqq_path_summary"]
    return {
        "name": name,
        "policy": policy,
        "router": router,
        "qqq_path_summary": qqq,
        "annual": annual,
        "key_metrics": {
            "total_return_pct": router["total_return_pct"],
            "max_drawdown_pct": router["max_drawdown_pct"],
            "daily_cvar5_pct": router["daily_cvar5_pct"],
            "calmar_like": router["calmar_like"],
            "macro_trigger_bars": qqq["macro_trigger_bars"],
            "macro_cash_bars": qqq["macro_cash_bars"],
            "macro_cap_bars": qqq["macro_cap_bars"],
            "risk_exit_events": qqq["risk_exit_events"],
        },
    }


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    router_config = json.loads(Path(args.router_config).read_text())
    start = pd.Timestamp(args.start_date, tz="UTC")
    end = parse_end_timestamp(args.end_date)
    initial_capital = float(config["initial_capital"])

    daily_macro, missing_flags = load_daily_macro_proxy_context()
    bars = attach_macro_context(load_enriched_bars(config, Path(args.nq_data_4h), start=start, end=end), daily_macro)
    funding = load_funding(Path(args.nq_funding))
    btc_path, _ = build_btc_path_from_frozen_artifact(
        frozen_path=Path(args.btc_frozen),
        start=start,
        end=end,
        initial_capital=initial_capital,
    )

    baseline_policy = {"kind": "none", "label": "Current V2 baseline"}
    baseline_path, baseline_summary = run_policy(
        policy_name="baseline_v2",
        policy=baseline_policy,
        bars=bars,
        funding=funding,
        btc_path=btc_path,
        config=config,
        router_config=router_config,
    )
    baseline_annual = annual_metrics(baseline_path)
    baseline_router = baseline_summary["router"]

    results: list[dict[str, Any]] = []
    for policy in build_scan_policies():
        path, summary = run_policy(
            policy_name=policy["name"],
            policy=policy,
            bars=bars,
            funding=funding,
            btc_path=btc_path,
            config=config,
            router_config=router_config,
        )
        annual = annual_metrics(path)
        item = summarize_candidate(policy["name"], policy, summary, annual)
        item["delta_vs_baseline"] = {
            "return_pct_points": round(summary["router"]["total_return_pct"] - baseline_router["total_return_pct"], 2),
            "max_drawdown_pct_points": round(summary["router"]["max_drawdown_pct"] - baseline_router["max_drawdown_pct"], 2),
            "daily_cvar5_pct_points": round(summary["router"]["daily_cvar5_pct"] - baseline_router["daily_cvar5_pct"], 4),
            "calmar_like_delta": round(summary["router"]["calmar_like"] - baseline_router["calmar_like"], 4),
            "2024_return_pct_points": round(annual["2024"]["router"]["total_return_pct"] - baseline_annual["2024"]["router"]["total_return_pct"], 2),
            "2024_max_drawdown_pct_points": round(annual["2024"]["router"]["max_drawdown_pct"] - baseline_annual["2024"]["router"]["max_drawdown_pct"], 2),
            "2025_return_pct_points": round(annual["2025"]["router"]["total_return_pct"] - baseline_annual["2025"]["router"]["total_return_pct"], 2),
        }
        results.append(item)

    top_return = sorted(results, key=lambda item: item["router"]["total_return_pct"], reverse=True)[:8]
    top_drawdown = sorted(results, key=lambda item: item["router"]["max_drawdown_pct"], reverse=False)[:8]
    top_calmar = sorted(results, key=lambda item: item["router"]["calmar_like"], reverse=True)[:8]
    top_cvar = sorted(results, key=lambda item: item["router"]["daily_cvar5_pct"], reverse=True)[:8]
    top_balanced = sorted(
        results,
        key=lambda item: (
            item["router"]["total_return_pct"] >= baseline_router["total_return_pct"],
            -abs(item["router"]["max_drawdown_pct"]),
            item["router"]["calmar_like"],
        ),
        reverse=True,
    )[:8]

    payload = {
        "mode": "qqq_v2_dollar_gate_sensitivity",
        "period": {"start": args.start_date, "end": args.end_date},
        "macro_context": {
            "rows": int(len(daily_macro)),
            "start": str(pd.Timestamp(daily_macro["date"].min()).date()) if not daily_macro.empty else None,
            "end": str(pd.Timestamp(daily_macro["date"].max()).date()) if not daily_macro.empty else None,
            "missing_data_flags": missing_flags,
        },
        "baseline": summarize_candidate("baseline_v2", baseline_policy, baseline_summary, baseline_annual),
        "results": results,
        "top_lists": {
            "top_return": top_return,
            "top_drawdown": top_drawdown,
            "top_calmar": top_calmar,
            "top_cvar": top_cvar,
            "top_balanced": top_balanced,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print(output)
    print(
        json.dumps(
            {
                "baseline": baseline_router,
                "top_return": [item["name"] for item in top_return[:5]],
                "top_drawdown": [item["name"] for item in top_drawdown[:5]],
                "top_calmar": [item["name"] for item in top_calmar[:5]],
                "top_balanced": [item["name"] for item in top_balanced[:5]],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
