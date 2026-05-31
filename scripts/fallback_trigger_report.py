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

from scripts.live_drift_monitor import build_live_trades, load_action_log, resolve_state_db  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "config.live.high-leverage-structure.template.json"
DEFAULT_BASELINE = ROOT / "config" / "live_drift_baseline.high_leverage.json"
DEFAULT_OUTPUT = ROOT / "var" / "high_leverage_expansion" / "fallback_trigger_report.json"

RECOMMENDATION_LABELS = {
    "stay_main": "主配置",
    "fallback_a_short_off": "Fallback A / 关 SMC short",
    "fallback_b_short_off_conflict_boost_defense": "Fallback B / 关 short + 冲突 boost 防守",
    "fallback_c_long_core_only": "Fallback C / 只留 long core",
    "insufficient_data": "样本不足",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fallback A/B/C recommendation from recent runtime trade quality.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--state-db", default=None)
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--recent-trades", type=int, default=20)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def recent_live_trades(db_path: Path, limit: int) -> list[Any]:
    actions = load_action_log(db_path)
    trades, _ = build_live_trades(actions)
    trades = sorted(trades, key=lambda item: item.exit_time)
    return trades[-int(limit) :] if limit > 0 else trades


def safe_pct(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) * 100.0


def direction_bucket(direction: str | None) -> str:
    raw = str(direction or "").upper()
    if raw == "BEAR":
        return "short"
    if raw == "BULL":
        return "long"
    return "unknown"


def summarize(trades: list[Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "trade_count": len(trades),
        "long_trade_count": 0,
        "short_trade_count": 0,
        "long_win_rate_pct": None,
        "short_win_rate_pct": None,
        "long_expectancy_pct": None,
        "short_expectancy_pct": None,
        "short_loss_streak": 0,
        "stop_like_share_pct": 0.0,
        "overall_expectancy_pct": None,
    }
    if not trades:
        return summary

    groups = {"long": [], "short": []}
    stop_like = 0
    current_short_loss_streak = 0
    worst_short_loss_streak = 0

    for trade in trades:
        bucket = direction_bucket(trade.direction)
        pnl_pct = safe_pct(trade.pnl_pct)
        if bucket in groups and pnl_pct is not None:
            groups[bucket].append(pnl_pct)
            if bucket == "short":
                if pnl_pct <= 0.0:
                    current_short_loss_streak += 1
                    worst_short_loss_streak = max(worst_short_loss_streak, current_short_loss_streak)
                else:
                    current_short_loss_streak = 0
        reason = str(trade.exit_reason or "").lower()
        if "stop" in reason or "sl" in reason:
            stop_like += 1

    all_pcts = [safe_pct(trade.pnl_pct) for trade in trades if safe_pct(trade.pnl_pct) is not None]
    summary["long_trade_count"] = len(groups["long"])
    summary["short_trade_count"] = len(groups["short"])
    summary["stop_like_share_pct"] = round(stop_like / len(trades) * 100.0, 2)
    summary["short_loss_streak"] = worst_short_loss_streak
    if all_pcts:
        summary["overall_expectancy_pct"] = round(sum(all_pcts) / len(all_pcts), 4)
    for bucket in ("long", "short"):
        values = groups[bucket]
        if not values:
            continue
        win_rate = sum(1 for value in values if value > 0.0) / len(values) * 100.0
        expectancy = sum(values) / len(values)
        summary[f"{bucket}_win_rate_pct"] = round(win_rate, 2)
        summary[f"{bucket}_expectancy_pct"] = round(expectancy, 4)
    return summary


def evaluate_fallback(summary: dict[str, Any], baseline_payload: dict[str, Any]) -> dict[str, Any]:
    expected = baseline_payload.get("expected", {})
    thresholds = baseline_payload.get("thresholds", {})
    min_trades = int(thresholds.get("min_trades_for_quality", 8) or 8)
    overall_expectancy_floor = float(thresholds.get("alert_expectancy_below_pct", 0.0) or 0.0)
    warn_win_rate = float(thresholds.get("warn_win_rate_below_pct", 36.0) or 36.0)

    trade_count = int(summary["trade_count"])
    short_trades = int(summary["short_trade_count"])
    long_trades = int(summary["long_trade_count"])

    reasons: list[str] = []
    recommendation = "stay_main"

    if trade_count < min_trades:
        return {
            "recommendation": "insufficient_data",
            "reasons": [f"recent trades {trade_count} < min_trades_for_quality {min_trades}"],
        }

    short_win_rate = summary.get("short_win_rate_pct")
    short_expectancy = summary.get("short_expectancy_pct")
    long_win_rate = summary.get("long_win_rate_pct")
    long_expectancy = summary.get("long_expectancy_pct")
    overall_expectancy = summary.get("overall_expectancy_pct")
    stop_share = float(summary.get("stop_like_share_pct", 0.0) or 0.0)
    short_loss_streak = int(summary.get("short_loss_streak", 0) or 0)

    if short_trades >= 3:
        if short_win_rate is not None and short_win_rate <= 10.0:
            reasons.append(f"short win rate very low: {short_win_rate:.2f}%")
        if short_expectancy is not None and short_expectancy < 0.0:
            reasons.append(f"short expectancy negative: {short_expectancy:.4f}%")
        if short_loss_streak >= 3:
            reasons.append(f"short loss streak high: {short_loss_streak}")
        if reasons:
            recommendation = "fallback_a_short_off"

    if recommendation == "fallback_a_short_off":
        if stop_share >= 70.0 and long_trades >= 6 and long_expectancy is not None and long_expectancy <= 0.0:
            recommendation = "fallback_c_long_core_only"
            reasons.append(f"broad stop-like share elevated: {stop_share:.2f}%")
            reasons.append(f"long expectancy non-positive: {long_expectancy:.4f}%")
        elif long_trades >= 6 and long_win_rate is not None and long_win_rate < warn_win_rate and long_expectancy is not None and long_expectancy <= 0.2:
            recommendation = "fallback_b_short_off_conflict_boost_defense"
            reasons.append(f"long win rate weak: {long_win_rate:.2f}%")
            reasons.append(f"long expectancy weak: {long_expectancy:.4f}%")

    if recommendation == "stay_main":
        if overall_expectancy is not None and overall_expectancy <= overall_expectancy_floor:
            recommendation = "fallback_a_short_off"
            reasons.append(f"overall expectancy below alert floor: {overall_expectancy:.4f}%")

    expected_snapshot = {
        "expected_win_rate_pct": expected.get("win_rate_pct"),
        "expected_expectancy_pct": expected.get("expectancy_pct"),
        "expected_trades_per_month": expected.get("trades_per_month"),
    }
    return {
        "recommendation": recommendation,
        "reasons": reasons or ["current recent trade quality does not trigger fallback"],
        "expected_snapshot": expected_snapshot,
    }


def build_report_from_paths(
    *,
    config_path: Path,
    state_db: Path | None,
    baseline_path: Path,
    recent_trades: int,
) -> dict[str, Any]:
    db_path = resolve_state_db(config_path, str(state_db) if state_db is not None else None)
    baseline_payload = load_json(baseline_path)
    trades = recent_live_trades(db_path, int(recent_trades))
    summary = summarize(trades)
    decision = evaluate_fallback(summary, baseline_payload)
    return {
        "config": str(config_path.resolve()),
        "state_db": str(db_path.resolve()),
        "baseline": str(baseline_path.resolve()),
        "recent_trades_requested": int(recent_trades),
        "summary": summary,
        "decision": decision,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    return build_report_from_paths(
        config_path=Path(args.config),
        state_db=Path(args.state_db) if args.state_db else None,
        baseline_path=Path(args.baseline),
        recent_trades=int(args.recent_trades),
    )


def recommendation_label(code: str) -> str:
    return RECOMMENDATION_LABELS.get(code, code)


def format_summary_lines(report: dict[str, Any]) -> list[str]:
    decision = report.get("decision", {})
    summary = report.get("summary", {})
    recommendation = str(decision.get("recommendation") or "stay_main")
    reasons = decision.get("reasons") or []

    lines = [
        "🛟 Fallback 建议",
        f"建议: {recommendation_label(recommendation)}",
        (
            "样本: "
            f"总 {int(summary.get('trade_count') or 0)} / "
            f"多 {int(summary.get('long_trade_count') or 0)} / "
            f"空 {int(summary.get('short_trade_count') or 0)}"
        ),
    ]

    stop_like_share = summary.get("stop_like_share_pct")
    if stop_like_share is not None:
        lines.append(f"止损占比: {float(stop_like_share):.2f}%")

    if reasons:
        lines.extend(f"- {reason}" for reason in reasons[:3])
    return lines


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if bool(args.json):
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print(output_path)
    print("recommendation", report["decision"]["recommendation"])
    for reason in report["decision"]["reasons"]:
        print("-", reason)


if __name__ == "__main__":
    main()
