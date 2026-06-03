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

from scripts.cn_nasdaq100_strict_utils import load_config, load_strict_frame, run_strict_path, summarize_path  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "config.paper.cn-nasdaq100-etf.json"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "cn_nasdaq100_strict_leverage_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan strict CN Nasdaq100 ETF leverage tiers on a fixed signal baseline.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--low-leverages", default="1.0,1.25,1.5,1.75,2.0")
    parser.add_argument("--normal-strong-leverages", default="1.0,1.25,1.5,1.75,2.0")
    parser.add_argument("--top", type=int, default=15)
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def candidate_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    summary = item["summary"]
    yearly = summary.get("yearly_returns_pct", {})
    return (
        float(item.get("score", 0.0)),
        float(summary.get("total_return_pct", 0.0)),
        float(yearly.get("2026", 0.0)),
        -float(summary.get("max_drawdown_pct", 0.0)),
    )


def main() -> None:
    args = parse_args()
    base_config = load_config(Path(args.config))
    base_config["conditional_leverage_enabled"] = False
    base_config["conditional_leverage_value"] = 1.0
    base_config["tiered_leverage_enabled"] = False
    base_config["tiered_leverage_rules"] = []

    # Freeze the current strict signal baseline before scanning leverage.
    base_config["entry_fast_window"] = 21
    base_config["entry_slow_window"] = 200
    base_config["regime_filter"] = "ixic_filter"
    base_config["max_hold_days"] = 120
    base_config["trailing_lookback_days"] = 4
    base_config["trailing_drawdown_pct"] = 4.0

    frame = load_strict_frame(base_config)
    candidates: list[dict[str, Any]] = []

    for low_leverage in parse_float_list(args.low_leverages):
        for normal_strong_leverage in parse_float_list(args.normal_strong_leverages):
            local_config = dict(base_config)
            if low_leverage <= 1.0 and normal_strong_leverage <= 1.0:
                local_config["tiered_leverage_enabled"] = False
                local_config["tiered_leverage_rules"] = []
            else:
                local_config["tiered_leverage_enabled"] = True
                local_config["tiered_leverage_rules"] = [
                    {
                        "vix_label": "vix_normal",
                        "rel_strength_label": "qqq_strong",
                        "leverage": float(normal_strong_leverage),
                    },
                    {
                        "vix_label": "vix_low",
                        "leverage": float(low_leverage),
                    },
                ]

            path = run_strict_path(frame, local_config)
            summary = summarize_path(path, initial_capital=float(local_config.get("initial_capital", 1000.0)))
            yearly = summary.get("yearly_returns_pct", {})
            score = round(
                float(summary.get("total_return_pct", 0.0))
                - 1.4 * float(summary.get("max_drawdown_pct", 0.0))
                + 1.0 * float(yearly.get("2026", 0.0))
                + 0.7 * float(yearly.get("2025", 0.0)),
                4,
            )
            candidates.append(
                {
                    "candidate": {
                        "low_leverage": float(low_leverage),
                        "normal_strong_leverage": float(normal_strong_leverage),
                    },
                    "summary": summary,
                    "score": score,
                }
            )

    ranked = sorted(candidates, key=candidate_sort_key, reverse=True)
    payload = {
        "signal_baseline": {
            "fast_window": 21,
            "slow_window": 200,
            "regime_filter": "ixic_filter",
            "max_hold_days": 120,
            "trailing_lookback_days": 4,
            "trailing_drawdown_pct": 4.0,
            "execution_mode": "strict_us_close_to_next_cn_session",
        },
        "scan_size": len(candidates),
        "top_candidates": ranked[: int(args.top)],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    for item in ranked[: min(int(args.top), 12)]:
        c = item["candidate"]
        s = item["summary"]
        y = s["yearly_returns_pct"]
        print(
            f"score={item['score']:.2f} full={s['total_return_pct']:.2f}% dd={s['max_drawdown_pct']:.2f}% "
            f"2025={float(y.get('2025', 0.0)):.2f}% 2026={float(y.get('2026', 0.0)):.2f}% "
            f"low={c['low_leverage']} normal_strong={c['normal_strong_leverage']}"
        )


if __name__ == "__main__":
    main()
