#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "trailing_rr_modes_live_shadow_scan.json"
RR_MODE_CHOICES = ("close", "extreme")


def parse_mode_list(value: str) -> list[str]:
    modes = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in modes if item not in RR_MODE_CHOICES]
    if invalid:
        raise ValueError(f"Unsupported RR modes: {invalid}")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan trailing RR observation modes under SOTA+SMC live-shadow replay.")
    parser.add_argument("--replay-script", default=str(ROOT / "scripts" / "replay_sota_smc_live_shadow.py"))
    parser.add_argument("--stage-trigger-rr-modes", default="close,extreme")
    parser.add_argument("--time-trailing-rr-modes", default="close,extreme")
    parser.add_argument("--atr-activation-rr-modes", default="close,extreme")
    parser.add_argument("--sample-trades", type=int, default=20)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args, replay_args = parser.parse_known_args()
    if replay_args and replay_args[0] == "--":
        replay_args = replay_args[1:]
    args.replay_args = replay_args
    return args


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_result(report: dict[str, Any]) -> dict[str, Any]:
    live = report["live_shadow"]
    decisions = live.get("decision_counts", {})
    trailing_modes = report.get("metadata", {}).get("trailing_rr_modes", {})
    return {
        "trailing_rr_modes": trailing_modes,
        "total_return_pct": live.get("total_return_pct"),
        "max_drawdown_pct": live.get("max_drawdown_pct"),
        "current_year_return_pct": live.get("windows", {}).get("current_year", {}).get("total_return_pct"),
        "trade_count": live.get("trade_count"),
        "decision_counts": decisions,
        "reference_gap": report.get("reference_gap", {}),
    }


def main() -> None:
    args = parse_args()
    replay_script = Path(args.replay_script).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage_modes = parse_mode_list(args.stage_trigger_rr_modes)
    time_modes = parse_mode_list(args.time_trailing_rr_modes)
    atr_modes = parse_mode_list(args.atr_activation_rr_modes)

    results: list[dict[str, Any]] = []
    baseline_key = ("close", "close", "close")

    for stage_mode in stage_modes:
        for time_mode in time_modes:
            for atr_mode in atr_modes:
                run_output = output_path.with_name(
                    f"{output_path.stem}.{stage_mode}.{time_mode}.{atr_mode}{output_path.suffix}"
                )
                cmd = [
                    sys.executable,
                    str(replay_script),
                    "--stage-trigger-rr-mode",
                    stage_mode,
                    "--time-trailing-rr-mode",
                    time_mode,
                    "--atr-activation-rr-mode",
                    atr_mode,
                    "--sample-trades",
                    str(args.sample_trades),
                    "--output",
                    str(run_output),
                ]
                if args.replay_args:
                    cmd.extend(args.replay_args)
                subprocess.run(cmd, check=True, cwd=str(ROOT))
                report = load_report(run_output)
                result = compact_result(report)
                result["mode_key"] = {
                    "stage_trigger_rr_mode": stage_mode,
                    "time_trailing_rr_mode": time_mode,
                    "atr_activation_rr_mode": atr_mode,
                }
                results.append(result)

    baseline = next(
        item for item in results
        if (
            item["mode_key"]["stage_trigger_rr_mode"],
            item["mode_key"]["time_trailing_rr_mode"],
            item["mode_key"]["atr_activation_rr_mode"],
        ) == baseline_key
    )
    baseline_return = float(baseline.get("total_return_pct", 0.0) or 0.0)
    baseline_dd = float(baseline.get("max_drawdown_pct", 0.0) or 0.0)
    baseline_year = float(baseline.get("current_year_return_pct", 0.0) or 0.0)

    for item in results:
        item["delta_vs_close_baseline"] = {
            "total_return_pct": round(float(item.get("total_return_pct", 0.0) or 0.0) - baseline_return, 4),
            "max_drawdown_pct": round(float(item.get("max_drawdown_pct", 0.0) or 0.0) - baseline_dd, 4),
            "current_year_return_pct": round(float(item.get("current_year_return_pct", 0.0) or 0.0) - baseline_year, 4),
        }

    results.sort(key=lambda item: float(item.get("total_return_pct", 0.0) or 0.0), reverse=True)
    report = {
        "metadata": {
            "replay_script": str(replay_script),
            "output": str(output_path),
            "stage_trigger_rr_modes": stage_modes,
            "time_trailing_rr_modes": time_modes,
            "atr_activation_rr_modes": atr_modes,
            "candidate_count": len(results),
        },
        "baseline_close_close_close": baseline,
        "top_by_return": results,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(output_path)
    print(
        "Baseline "
        f"full={baseline_return:.2f}%/{baseline_dd:.2f}% "
        f"2026={baseline_year:.2f}%"
    )
    for idx, item in enumerate(results[:8], start=1):
        modes = item["mode_key"]
        delta = item["delta_vs_close_baseline"]
        print(
            f"{idx:02d} full={float(item['total_return_pct']):.2f}%/{float(item['max_drawdown_pct']):.2f}% "
            f"2026={float(item['current_year_return_pct']):.2f}% "
            f"stage={modes['stage_trigger_rr_mode']} time={modes['time_trailing_rr_mode']} atr={modes['atr_activation_rr_mode']} "
            f"d_full={delta['total_return_pct']:+.2f}% d_dd={delta['max_drawdown_pct']:+.2f}"
        )


if __name__ == "__main__":
    main()
