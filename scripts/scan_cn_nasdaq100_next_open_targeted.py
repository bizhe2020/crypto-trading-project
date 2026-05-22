#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nasdaq100_cn_strategy_utils import load_config, load_strategy_frame, run_full_strategy


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "cn_nasdaq100_next_open_targeted_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted scan around the formal next-open CN Nasdaq-100 ETF baseline.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.paper.cn-nasdaq100-etf.json"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-hold-values", default="60,90,120")
    parser.add_argument("--trailing-lookback-values", default="5,10,15")
    parser.add_argument("--trailing-drawdown-values", default="4,5,6,8")
    parser.add_argument("--vix-low-weak-values", default="1.5,1.75,2.0")
    parser.add_argument("--vix-normal-strong-values", default="1.0,1.25,1.5,1.75")
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def build_tiered_rules(vix_low_weak: float, vix_normal_strong: float) -> list[dict[str, Any]]:
    return [
        {"vix_label": "vix_low", "rel_strength_label": "qqq_strong", "leverage": 2.0},
        {"vix_label": "vix_low", "leverage": float(vix_low_weak)},
        {"vix_label": "vix_normal", "rel_strength_label": "qqq_strong", "leverage": float(vix_normal_strong)},
    ]


def score(summary: dict[str, Any]) -> float:
    return float(summary["total_return_pct"]) - 2.0 * float(summary["max_drawdown_pct"]) + float(summary["yearly_returns_pct"].get("2026", 0.0))


def main() -> None:
    args = parse_args()
    base_config = load_config(Path(args.config))
    results: list[dict[str, Any]] = []

    for max_hold in parse_ints(args.max_hold_values):
        for lookback in parse_ints(args.trailing_lookback_values):
            for drawdown in parse_floats(args.trailing_drawdown_values):
                for vix_low_weak in parse_floats(args.vix_low_weak_values):
                    for vix_normal_strong in parse_floats(args.vix_normal_strong_values):
                        config = dict(base_config)
                        config["max_hold_days"] = int(max_hold)
                        config["trailing_lookback_days"] = int(lookback)
                        config["trailing_drawdown_pct"] = float(drawdown)
                        config["tiered_leverage_enabled"] = True
                        config["conditional_leverage_enabled"] = False
                        config["tiered_leverage_rules"] = build_tiered_rules(vix_low_weak, vix_normal_strong)
                        frame = load_strategy_frame(config)
                        summary = run_full_strategy(frame, config)
                        results.append(
                            {
                                "max_hold_days": int(max_hold),
                                "trailing_lookback_days": int(lookback),
                                "trailing_drawdown_pct": float(drawdown),
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
        "current_formal_baseline": {
            "total_return_pct": 571.66,
            "max_drawdown_pct": 15.29,
            "yearly_returns_pct": {"2022": 0.0, "2023": 100.68, "2024": 124.86, "2025": 33.33, "2026": 13.88},
        },
        "top_candidates": ranked[:20],
        "scan_size": len(results),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    for item in ranked[:10]:
        s = item["summary"]
        print(
            item["max_hold_days"],
            item["trailing_lookback_days"],
            item["trailing_drawdown_pct"],
            item["vix_low_weak_leverage"],
            item["vix_normal_qqq_strong_leverage"],
            s["total_return_pct"],
            s["max_drawdown_pct"],
            s["yearly_returns_pct"],
        )


if __name__ == "__main__":
    main()
