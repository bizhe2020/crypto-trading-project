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

from scripts.tqqq_cash_strict_utils import load_strict_frame, run_strict_candidate  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "tqqq_cash_strict_scan.json"
DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict TQQQ/CASH combined scan.")
    parser.add_argument("--data-root", default=str(DEFAULT_PUBLIC_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--fast-windows", default="10,15,20,25,30,40")
    parser.add_argument("--slow-windows", default="100,120,150,180,200,250")
    parser.add_argument("--regime-filters", default="base,vix_filter,ixic_filter,vix_ixic,all_three")
    parser.add_argument("--max-hold-days-values", default="0,40,60,90,120")
    parser.add_argument("--trailing-lookback-days-values", default="0,5,10,20,30")
    parser.add_argument("--trailing-drawdown-pct-values", default="0,5,8,10,12,15")
    parser.add_argument("--switch-cost-bps", type=float, default=10.0)
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
        float(summary.get("score", 0.0)),
        float(summary.get("total_return_pct", 0.0)),
        float(yearly.get("2026", 0.0)),
        -float(summary.get("max_drawdown_pct", 0.0)),
    )


def main() -> None:
    args = parse_args()
    candidates: list[dict[str, Any]] = []
    for fast_window in parse_int_list(args.fast_windows):
        for slow_window in parse_int_list(args.slow_windows):
            if fast_window >= slow_window:
                continue
            frame = load_strict_frame(
                data_root=Path(args.data_root),
                entry_fast_window=int(fast_window),
                entry_slow_window=int(slow_window),
            )
            for regime_filter in parse_str_list(args.regime_filters):
                for max_hold_days in parse_int_list(args.max_hold_days_values):
                    for trailing_lookback_days in parse_int_list(args.trailing_lookback_days_values):
                        for trailing_drawdown_pct in parse_float_list(args.trailing_drawdown_pct_values):
                            if trailing_lookback_days == 0 and trailing_drawdown_pct > 0:
                                continue
                            if trailing_lookback_days > 0 and trailing_drawdown_pct == 0:
                                continue
                            result = run_strict_candidate(
                                frame,
                                regime_filter=regime_filter,
                                max_hold_days=int(max_hold_days),
                                trailing_lookback_days=int(trailing_lookback_days),
                                trailing_drawdown_pct=float(trailing_drawdown_pct),
                                switch_cost_bps=float(args.switch_cost_bps),
                                initial_capital=float(args.initial_capital),
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
                                        "switch_cost_bps": float(args.switch_cost_bps),
                                    },
                                    "summary": result["summary"],
                                }
                            )

    ranked = sorted(candidates, key=candidate_sort_key, reverse=True)
    payload = {
        "reference": {
            "execution_mode": "strict_t_plus_1_open",
            "description": "TQQQ/CASH strict scan with overnight+intraday split and next-open execution.",
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
            f"score={s['score']:.2f} full={s['total_return_pct']:.2f}% dd={s['max_drawdown_pct']:.2f}% "
            f"trades={s['trades']} 2022={float(y.get('2022', 0.0)):.2f}% 2023={float(y.get('2023', 0.0)):.2f}% "
            f"2024={float(y.get('2024', 0.0)):.2f}% 2025={float(y.get('2025', 0.0)):.2f}% 2026={float(y.get('2026', 0.0)):.2f}% "
            f"fast={c['fast_window']} slow={c['slow_window']} regime={c['regime_filter']} "
            f"hold={c['max_hold_days']} tlb={c['trailing_lookback_days']} tdd={c['trailing_drawdown_pct']}"
        )


if __name__ == "__main__":
    main()
