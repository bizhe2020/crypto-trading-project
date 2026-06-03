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
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "cn_nasdaq100_strict_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict scan for CN Nasdaq100 ETF driven by QQQ.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--fast-windows", default="20,25,30")
    parser.add_argument("--slow-windows", default="150,180,200,220")
    parser.add_argument("--regime-filters", default="base,ixic_filter,vix_ixic,all_three")
    parser.add_argument("--max-hold-days-values", default="60,90,120")
    parser.add_argument("--trailing-lookback-days-values", default="0,5,10,15")
    parser.add_argument("--trailing-drawdown-pct-values", default="0,5,8,10,12")
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [str(item.strip()) for item in str(value).split(",") if item.strip()]


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
    # Strict signal scan should not inherit any live leverage overlays.
    base_config["conditional_leverage_enabled"] = False
    base_config["conditional_leverage_value"] = 1.0
    base_config["tiered_leverage_enabled"] = False
    base_config["tiered_leverage_rules"] = []
    candidates: list[dict[str, Any]] = []

    for fast_window in parse_int_list(args.fast_windows):
        for slow_window in parse_int_list(args.slow_windows):
            if fast_window >= slow_window:
                continue
            config = dict(base_config)
            config["entry_fast_window"] = int(fast_window)
            config["entry_slow_window"] = int(slow_window)
            frame = load_strict_frame(config)
            for regime_filter in parse_str_list(args.regime_filters):
                for max_hold_days in parse_int_list(args.max_hold_days_values):
                    for trailing_lookback_days in parse_int_list(args.trailing_lookback_days_values):
                        for trailing_drawdown_pct in parse_float_list(args.trailing_drawdown_pct_values):
                            if trailing_lookback_days == 0 and trailing_drawdown_pct > 0:
                                continue
                            if trailing_lookback_days > 0 and trailing_drawdown_pct == 0:
                                continue
                            local_config = dict(config)
                            local_config["regime_filter"] = regime_filter
                            local_config["max_hold_days"] = int(max_hold_days)
                            local_config["trailing_lookback_days"] = int(trailing_lookback_days)
                            local_config["trailing_drawdown_pct"] = float(trailing_drawdown_pct)
                            result = run_strict_path(frame, local_config)
                            summary = summarize_path(result, initial_capital=float(local_config.get("initial_capital", 1000.0)))
                            yearly = summary.get("yearly_returns_pct", {})
                            score = round(
                                float(summary.get("total_return_pct", 0.0))
                                - 1.5 * float(summary.get("max_drawdown_pct", 0.0))
                                + 1.0 * float(yearly.get("2026", 0.0))
                                + 0.7 * float(yearly.get("2025", 0.0))
                                + 0.5 * float(yearly.get("2024", 0.0)),
                                4,
                            )
                            candidates.append(
                                {
                                    "candidate": {
                                        "fast_window": int(fast_window),
                                        "slow_window": int(slow_window),
                                        "regime_filter": regime_filter,
                                        "max_hold_days": int(max_hold_days),
                                        "trailing_lookback_days": int(trailing_lookback_days),
                                        "trailing_drawdown_pct": float(trailing_drawdown_pct),
                                    },
                                    "summary": summary,
                                    "score": score,
                                }
                            )

    ranked = sorted(candidates, key=candidate_sort_key, reverse=True)
    payload = {
        "reference": {
            "execution_mode": "strict_us_close_to_next_cn_session",
            "description": "US close signal mapped to next CN trade day, then CN open/close execution.",
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
            f"trades={s['trades']} 2024={float(y.get('2024', 0.0)):.2f}% 2025={float(y.get('2025', 0.0)):.2f}% "
            f"2026={float(y.get('2026', 0.0)):.2f}% fast={c['fast_window']} slow={c['slow_window']} "
            f"regime={c['regime_filter']} hold={c['max_hold_days']} tlb={c['trailing_lookback_days']} tdd={c['trailing_drawdown_pct']}"
        )


if __name__ == "__main__":
    main()
