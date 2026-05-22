#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nasdaq100_cn_strategy_utils import load_config, load_strategy_frame, run_full_strategy


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "cn_nasdaq100_next_open_tiered_only_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiered sizing scan on top of the best next-open exit profile.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.paper.cn-nasdaq100-etf.json"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--vix-low-weak-values", default="1.5,1.75,2.0")
    parser.add_argument("--vix-normal-strong-values", default="1.0,1.25,1.5,1.75")
    return parser.parse_args()


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def build_tiered_rules(vix_low_weak: float, vix_normal_strong: float) -> list[dict[str, float | str]]:
    return [
        {"vix_label": "vix_low", "rel_strength_label": "qqq_strong", "leverage": 2.0},
        {"vix_label": "vix_low", "leverage": float(vix_low_weak)},
        {"vix_label": "vix_normal", "rel_strength_label": "qqq_strong", "leverage": float(vix_normal_strong)},
    ]


def score(summary: dict[str, object]) -> float:
    yearly = dict(summary.get("yearly_returns_pct", {}))
    return float(summary["total_return_pct"]) - 2.0 * float(summary["max_drawdown_pct"]) + float(yearly.get("2026", 0.0))


def main() -> None:
    args = parse_args()
    base_config = load_config(Path(args.config))
    # Lock to the best exit profile found in next-open mode.
    base_config["max_hold_days"] = 90
    base_config["trailing_lookback_days"] = 5
    base_config["trailing_drawdown_pct"] = 8.0

    results = []
    for vix_low_weak in parse_floats(args.vix_low_weak_values):
        for vix_normal_strong in parse_floats(args.vix_normal_strong_values):
            config = dict(base_config)
            config["tiered_leverage_enabled"] = True
            config["conditional_leverage_enabled"] = False
            config["tiered_leverage_rules"] = build_tiered_rules(vix_low_weak, vix_normal_strong)
            frame = load_strategy_frame(config)
            summary = run_full_strategy(frame, config)
            results.append(
                {
                    "vix_low_weak_leverage": float(vix_low_weak),
                    "vix_normal_qqq_strong_leverage": float(vix_normal_strong),
                    "summary": {
                        "total_return_pct": summary["total_return_pct"],
                        "max_drawdown_pct": summary["max_drawdown_pct"],
                        "yearly_returns_pct": summary["yearly_returns_pct"],
                        "trades": summary["trades"],
                    },
                    "score": round(score(summary), 4),
                }
            )

    ranked = sorted(results, key=lambda item: (float(item["score"]), float(item["summary"]["total_return_pct"])), reverse=True)
    payload = {
        "fixed_exit_profile": {
            "max_hold_days": 90,
            "trailing_lookback_days": 5,
            "trailing_drawdown_pct": 8.0,
            "exit_only_best": {
                "total_return_pct": 594.23,
                "max_drawdown_pct": 13.21,
                "yearly_returns_pct": {"2022": 0.0, "2023": 100.68, "2024": 132.42, "2025": 33.33, "2026": 13.88},
            },
        },
        "top_candidates": ranked[:20],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    for item in ranked[:10]:
        s = item["summary"]
        print(item["vix_low_weak_leverage"], item["vix_normal_qqq_strong_leverage"], s["total_return_pct"], s["max_drawdown_pct"], s["yearly_returns_pct"])


if __name__ == "__main__":
    main()
