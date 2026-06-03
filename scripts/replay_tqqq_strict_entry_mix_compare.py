#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tqqq_cash_strict_utils import load_strict_frame_with_overlay_context, run_strict_candidate  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "reports" / "tqqq_strict_entry_mix_compare_20260526.json"


def main() -> None:
    frame = load_strict_frame_with_overlay_context(
        data_root=ROOT / "data" / "public" / "etf",
        entry_fast_window=25,
        entry_slow_window=150,
    )

    common = dict(
        regime_filter="ixic_filter",
        max_hold_days=90,
        trailing_lookback_days=10,
        trailing_drawdown_pct=12.0,
        switch_cost_bps=10.0,
        initial_capital=1000.0,
        de_risk_signal_name="breakout_fail_score_le3_flat",
    )

    variants = {
        "baseline": dict(**common),
        "recovery_only": dict(
            **common,
            recovery_reentry_rule="score_ge3",
            recovery_reentry_cooldown_days=0,
        ),
        "drawdown_only": dict(
            **common,
            drawdown_ladder_enabled=True,
            drawdown_ladder_source="tqqq",
            drawdown_ladder_threshold_pct=12.0,
            drawdown_ladder_peak_lookback_days=90,
            drawdown_ladder_scheme="three_40_30_30",
            drawdown_ladder_vix_rule="vix_low_normal",
            drawdown_ladder_rebound_exit_pct=10.0,
            drawdown_ladder_max_hold_days=15,
        ),
        "mixed_entry": dict(
            **common,
            recovery_reentry_rule="score_ge3",
            recovery_reentry_cooldown_days=0,
            drawdown_ladder_enabled=True,
            drawdown_ladder_source="tqqq",
            drawdown_ladder_threshold_pct=12.0,
            drawdown_ladder_peak_lookback_days=90,
            drawdown_ladder_scheme="three_40_30_30",
            drawdown_ladder_vix_rule="vix_low_normal",
            drawdown_ladder_rebound_exit_pct=10.0,
            drawdown_ladder_max_hold_days=15,
        ),
    }

    results = {}
    for name, cfg in variants.items():
        result = run_strict_candidate(frame, **cfg)
        results[name] = {
            "config": cfg,
            "summary": result["summary"],
        }

    payload = {"variants": results}
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(DEFAULT_OUTPUT)
    for name, item in results.items():
        s = item["summary"]
        print(
            f"{name}: full={s['total_return_pct']:.2f}% dd={s['max_drawdown_pct']:.2f}% "
            f"win={s['win_rate_pct']:.2f}% hold={s['avg_hold_days']:.2f}d overlays={s['overlay_entries']}"
        )


if __name__ == "__main__":
    main()
