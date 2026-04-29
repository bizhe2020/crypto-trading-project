#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OPEN_ACTIONS = {"OPEN_LONG", "OPEN_SHORT"}
STOP_REASON_TOKENS = ("stop", "sl")
TARGET_REASON_TOKENS = ("target", "tp")


@dataclass
class ActionLogRow:
    row_id: int
    timestamp: str
    action_type: str
    payload: dict[str, Any]
    created_at: str | None


@dataclass
class LiveTrade:
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float | None
    exit_price: float | None
    signal_entry_price: float | None
    signal_exit_price: float | None
    stop_price: float | None
    target_price: float | None
    exit_reason: str
    net_pnl: float
    capital_at_entry: float | None
    notional: float | None
    risk_amount: float | None
    entry_slippage_bps: float | None
    exit_slippage_bps: float | None
    stop_target_deviation_bps: float | None

    @property
    def pnl_pct(self) -> float | None:
        if self.capital_at_entry is None or self.capital_at_entry <= 0:
            return None
        return self.net_pnl / self.capital_at_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare live bot trades against the promoted high-leverage baseline.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.live.high-leverage-structure.template.json"))
    parser.add_argument("--state-db", default=None, help="Override state_db_path from config.")
    parser.add_argument("--baseline", default=str(ROOT / "config" / "live_drift_baseline.high_leverage.json"))
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--recent-trades", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--send-telegram", action="store_true")
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"bootstrap", "runtime"}:
        return None
    text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                parsed = None
        if parsed is None:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_state_db(config_path: Path, override: str | None) -> Path:
    if override:
        path = Path(override)
    else:
        config = load_json(config_path)
        raw = config.get("state_db_path")
        if not raw:
            raise ValueError(f"state_db_path missing in {config_path}")
        path = Path(str(raw))
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_action_log(db_path: Path) -> list[ActionLogRow]:
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, timestamp, action_type, payload, created_at
            FROM action_log
            ORDER BY id ASC
            """
        ).fetchall()
    actions: list[ActionLogRow] = []
    for row in rows:
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            payload = {}
        actions.append(
            ActionLogRow(
                row_id=int(row["id"]),
                timestamp=str(row["timestamp"]),
                action_type=str(row["action_type"]),
                payload=payload if isinstance(payload, dict) else {},
                created_at=row["created_at"],
            )
        )
    return actions


def price_diff_bps(actual: float | None, reference: float | None) -> float | None:
    if actual is None or reference is None or reference <= 0:
        return None
    return abs(actual - reference) / reference * 10000.0


def reference_price_for_close(reason: str, open_trade: dict[str, Any]) -> float | None:
    lowered = reason.lower()
    if any(token in lowered for token in STOP_REASON_TOKENS):
        return safe_float(open_trade.get("stop_price"))
    if any(token in lowered for token in TARGET_REASON_TOKENS):
        return safe_float(open_trade.get("target_price"))
    return None


def build_live_trades(actions: list[ActionLogRow]) -> tuple[list[LiveTrade], dict[str, int]]:
    open_trade: dict[str, Any] | None = None
    trades: list[LiveTrade] = []
    diagnostics = {"orphan_closes": 0, "overwritten_opens": 0, "open_without_close": 0}

    for row in actions:
        payload = row.payload
        action_type = row.action_type
        action_time = parse_timestamp(payload.get("timestamp")) or parse_timestamp(row.timestamp)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

        if action_type in OPEN_ACTIONS:
            if open_trade is not None:
                diagnostics["overwritten_opens"] += 1
            open_trade = {
                "entry_time": action_time,
                "direction": payload.get("direction"),
                "entry_price": safe_float(payload.get("entry_price")),
                "signal_entry_price": safe_float(metadata.get("signal_entry_price")),
                "stop_price": safe_float(payload.get("stop_price")),
                "target_price": safe_float(payload.get("target_price")),
                "capital_at_entry": safe_float(metadata.get("capital_at_entry")),
                "notional": safe_float(metadata.get("notional")),
                "risk_amount": safe_float(metadata.get("risk_amount")),
            }
            continue

        if action_type == "UPDATE_STOP" and open_trade is not None:
            stop_price = safe_float(payload.get("stop_price"))
            if stop_price is not None:
                open_trade["stop_price"] = stop_price
            continue

        if action_type != "CLOSE_POSITION":
            continue

        if open_trade is None:
            diagnostics["orphan_closes"] += 1
            continue

        exit_time = action_time
        entry_time = open_trade.get("entry_time")
        if entry_time is None or exit_time is None:
            diagnostics["orphan_closes"] += 1
            open_trade = None
            continue

        exit_price = safe_float(payload.get("exit_price"))
        signal_exit_price = safe_float(metadata.get("signal_exit_price"))
        reason = str(payload.get("reason") or "")
        reference = reference_price_for_close(reason, open_trade)
        trade = LiveTrade(
            entry_time=entry_time,
            exit_time=exit_time,
            direction=str(open_trade.get("direction") or payload.get("direction") or ""),
            entry_price=safe_float(open_trade.get("entry_price")),
            exit_price=exit_price,
            signal_entry_price=safe_float(open_trade.get("signal_entry_price")),
            signal_exit_price=signal_exit_price,
            stop_price=safe_float(open_trade.get("stop_price")),
            target_price=safe_float(open_trade.get("target_price")),
            exit_reason=reason,
            net_pnl=safe_float(metadata.get("net_pnl")) or 0.0,
            capital_at_entry=safe_float(open_trade.get("capital_at_entry")),
            notional=safe_float(open_trade.get("notional")),
            risk_amount=safe_float(open_trade.get("risk_amount")),
            entry_slippage_bps=price_diff_bps(
                safe_float(open_trade.get("entry_price")),
                safe_float(open_trade.get("signal_entry_price")),
            ),
            exit_slippage_bps=price_diff_bps(exit_price, signal_exit_price),
            stop_target_deviation_bps=price_diff_bps(exit_price, reference),
        )
        trades.append(trade)
        open_trade = None

    if open_trade is not None:
        diagnostics["open_without_close"] += 1
    return trades, diagnostics


def average(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def trade_metrics(trades: list[LiveTrade], *, window_days: int | None = None) -> dict[str, Any]:
    pnl_pcts = [trade.pnl_pct for trade in trades if trade.pnl_pct is not None]
    wins = [value for value in pnl_pcts if value > 0]
    losses = [value for value in pnl_pcts if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    if trades:
        start = min(trade.entry_time for trade in trades)
        end = max(trade.exit_time for trade in trades)
        span_days = max((end - start).total_seconds() / 86400.0, 1.0)
    else:
        start = None
        end = None
        span_days = float(window_days or 1)

    if window_days is not None and window_days > 0:
        frequency_days = float(window_days)
    else:
        frequency_days = span_days

    compounded = 1.0
    for value in pnl_pcts:
        compounded *= 1.0 + value

    entry_slippage = [value for trade in trades if (value := trade.entry_slippage_bps) is not None]
    exit_slippage = [value for trade in trades if (value := trade.exit_slippage_bps) is not None]
    stop_target_deviation = [
        value for trade in trades if (value := trade.stop_target_deviation_bps) is not None
    ]

    return {
        "trade_count": len(trades),
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "total_return_pct": round((compounded - 1.0) * 100.0, 3) if pnl_pcts else None,
        "win_rate_pct": round(len(wins) / len(pnl_pcts) * 100.0, 3) if pnl_pcts else None,
        "avg_win_pct": round(average(wins) * 100.0, 3) if wins else None,
        "avg_loss_pct": round(average(losses) * 100.0, 3) if losses else None,
        "payoff_ratio": round((average(wins) or 0.0) / abs(average(losses) or 1.0), 3) if wins and losses else None,
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else None,
        "expectancy_pct": round(average(pnl_pcts) * 100.0, 3) if pnl_pcts else None,
        "trades_per_month": round(len(trades) / frequency_days * 30.4375, 3) if frequency_days > 0 else None,
        "trades_per_year": round(len(trades) / frequency_days * 365.25, 3) if frequency_days > 0 else None,
        "avg_entry_slippage_bps": round(average(entry_slippage), 3) if entry_slippage else None,
        "avg_exit_slippage_bps": round(average(exit_slippage), 3) if exit_slippage else None,
        "avg_stop_target_deviation_bps": round(average(stop_target_deviation), 3) if stop_target_deviation else None,
        "stop_target_reference_count": len(stop_target_deviation),
    }


def select_recent_trades(trades: list[LiveTrade], *, window_days: int, recent_trades: int) -> list[LiveTrade]:
    if not trades:
        return []
    ordered = sorted(trades, key=lambda trade: trade.exit_time)
    end = ordered[-1].exit_time
    start = end - timedelta(days=window_days)
    by_days = [trade for trade in ordered if trade.exit_time >= start]
    if len(by_days) >= recent_trades:
        return by_days
    return ordered[-recent_trades:]


def compare_to_baseline(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    expected = baseline.get("expected", {})
    comparisons: dict[str, Any] = {}
    for key in ("win_rate_pct", "profit_factor", "payoff_ratio", "expectancy_pct", "trades_per_month"):
        live_value = metrics.get(key)
        expected_value = expected.get(key)
        if live_value is None or expected_value is None:
            continue
        comparisons[key] = {
            "live": live_value,
            "expected": expected_value,
            "delta": round(float(live_value) - float(expected_value), 3),
        }
    if metrics.get("trades_per_month") is not None and expected.get("trades_per_month"):
        comparisons["trade_frequency_ratio"] = round(
            float(metrics["trades_per_month"]) / float(expected["trades_per_month"]),
            3,
        )
    return comparisons


def assess_status(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    thresholds = baseline.get("thresholds", {})
    flags: list[dict[str, str]] = []
    alerts = 0

    trade_count = int(metrics.get("trade_count") or 0)
    if trade_count == 0:
        return {"status": "NO_TRADES", "flags": [{"level": "info", "message": "No closed live trades found."}]}
    if trade_count < int(thresholds.get("min_trades_for_quality", 8)):
        flags.append({"level": "watch", "message": f"Small sample: {trade_count} closed trades."})

    checks = [
        ("win_rate_pct", "warn_win_rate_below_pct", "watch", "Win rate below baseline guard."),
        ("profit_factor", "warn_profit_factor_below", "watch", "Profit factor below watch level."),
        ("profit_factor", "alert_profit_factor_below", "alert", "Profit factor below 1.0 alert level."),
        ("payoff_ratio", "warn_payoff_below", "watch", "Payoff ratio below watch level."),
        ("expectancy_pct", "warn_expectancy_below_pct", "watch", "Expectancy below watch level."),
        ("expectancy_pct", "alert_expectancy_below_pct", "alert", "Expectancy is negative."),
        ("avg_entry_slippage_bps", "warn_avg_entry_slippage_bps_above", "watch", "Entry slippage above modeled guard."),
        ("avg_exit_slippage_bps", "warn_avg_exit_slippage_bps_above", "watch", "Exit slippage above modeled guard."),
        (
            "avg_stop_target_deviation_bps",
            "warn_stop_target_deviation_bps_above",
            "watch",
            "Stop/target execution deviation above guard.",
        ),
    ]
    for metric_key, threshold_key, level, message in checks:
        value = metrics.get(metric_key)
        threshold = thresholds.get(threshold_key)
        if value is None or threshold is None:
            continue
        breached = value > threshold if "above" in threshold_key else value < threshold
        if breached:
            flags.append({"level": level, "message": message})
            alerts += 1 if level == "alert" else 0

    frequency = metrics.get("trades_per_month")
    expected_frequency = baseline.get("expected", {}).get("trades_per_month")
    if frequency is not None and expected_frequency:
        ratio = float(frequency) / float(expected_frequency)
        if ratio < float(thresholds.get("warn_trade_frequency_ratio_below", 0.5)):
            flags.append({"level": "watch", "message": "Trade frequency is materially below baseline."})
        if ratio > float(thresholds.get("warn_trade_frequency_ratio_above", 1.8)):
            flags.append({"level": "watch", "message": "Trade frequency is materially above baseline."})

    status = "ALERT" if alerts else "WATCH" if flags else "OK"
    return {"status": status, "flags": flags}


def build_report(
    *,
    config_path: Path,
    state_db: Path,
    baseline: dict[str, Any],
    actions: list[ActionLogRow],
    trades: list[LiveTrade],
    diagnostics: dict[str, int],
    window_days: int,
    recent_trades: int,
) -> dict[str, Any]:
    recent = select_recent_trades(trades, window_days=window_days, recent_trades=recent_trades)
    all_metrics = trade_metrics(trades)
    recent_metrics = trade_metrics(recent, window_days=window_days)
    comparison = compare_to_baseline(recent_metrics, baseline)
    assessment = assess_status(recent_metrics, baseline)
    return {
        "config": str(config_path),
        "state_db": str(state_db),
        "baseline": {
            "name": baseline.get("name"),
            "source": baseline.get("source"),
        },
        "action_rows": len(actions),
        "diagnostics": diagnostics,
        "window": {
            "window_days": window_days,
            "recent_trades_floor": recent_trades,
            "selected_trades": len(recent),
        },
        "status": assessment["status"],
        "flags": assessment["flags"],
        "recent": recent_metrics,
        "all": all_metrics,
        "baseline_comparison": comparison,
    }


def format_value(value: Any, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def health_conclusion(report: dict[str, Any]) -> list[str]:
    status = str(report.get("status") or "")
    recent = report.get("recent", {})
    baseline = report.get("baseline_comparison", {})
    flags = report.get("flags", [])
    trade_count = int(recent.get("trade_count") or 0)
    profit_factor = recent.get("profit_factor")
    expectancy = recent.get("expectancy_pct")
    frequency_ratio = baseline.get("trade_frequency_ratio")

    if status == "NO_TRADES":
        verdict = "🕒 暂无足够实盘样本，先看执行链路是否稳定。"
    elif status == "ALERT":
        verdict = "🚨 实盘质量明显偏离，暂停加资金，优先排查执行/滑点/信号漂移。"
    elif status == "WATCH":
        verdict = "🟡 实盘进入观察区，暂不加仓，等样本和质量恢复。"
    else:
        verdict = "✅ 实盘质量暂时贴近基准，可以继续小资金验证。"

    details = [verdict]
    if trade_count < 8:
        details.append(f"样本: {trade_count} 笔，低于 8 笔，结论只看方向，不做重仓依据。")
    elif profit_factor is not None and expectancy is not None:
        details.append(f"质量: PF {profit_factor:.2f}，单笔期望 {expectancy:.2f}%。")
    if isinstance(frequency_ratio, (int, float)):
        details.append(f"频率: 当前约为基准的 {frequency_ratio:.2f}x。")
    if flags:
        details.append("动作: 先处理体检提示，再考虑扩大资金。")
    return details


def launch_capital_advice(report: dict[str, Any]) -> list[str]:
    status = str(report.get("status") or "")
    recent = report.get("recent", {})
    trade_count = int(recent.get("trade_count") or 0)
    profit_factor = recent.get("profit_factor")
    expectancy = recent.get("expectancy_pct")

    lines = [
        "资金假设: 当前约 10,000U = 计划资金 20%。",
        "回测压力: 最优策略 MaxDD 约 33.87%，20% 仓位下对应账户总资金压力约 6.8%。",
    ]
    if status == "OK" and trade_count >= 8 and (profit_factor or 0.0) >= 1.5 and (expectancy or 0.0) > 0.5:
        lines.append("建议: 维持 20% 运行；连续 20-30 笔仍贴近基准后，再考虑提高到 25%-35%。")
    elif status in {"NO_TRADES", "WATCH"} or trade_count < 8:
        lines.append("建议: 暂时维持 20%，不要因为短期没开仓而加资金；先累计 8-20 笔闭环样本。")
    else:
        lines.append("建议: 不加资金；若 ALERT 持续，优先降到 10%-15% 或暂停新开仓排查。")
    lines.append("红线: PF < 1.0、期望转负、滑点明显扩大或连续异常亏损时，不扩仓。")
    return lines


def format_report(report: dict[str, Any]) -> str:
    recent = report["recent"]
    all_metrics = report["all"]
    comparison = report["baseline_comparison"]
    flags = report["flags"]
    status_icon = {
        "OK": "✅",
        "WATCH": "🟡",
        "ALERT": "🚨",
        "NO_TRADES": "🕒",
    }.get(str(report["status"]), "ℹ️")
    lines = [
        "🩺 <Live Drift Monitor>",
        f"状态: {status_icon} {report['status']}",
        f"窗口: 最近 {report['window']['window_days']}d / 至少 {report['window']['recent_trades_floor']} 笔",
        "",
        "📊 近期质量",
        f"交易: {recent['trade_count']} 笔 | 收益: {format_value(recent['total_return_pct'], '%')}",
        f"胜率: {format_value(recent['win_rate_pct'], '%')} | PF: {format_value(recent['profit_factor'])}",
        f"盈亏比: {format_value(recent['payoff_ratio'])} | 单笔期望: {format_value(recent['expectancy_pct'], '%')}",
        f"频率: {format_value(recent['trades_per_month'])}/月 | {format_value(recent['trades_per_year'])}/年",
        "",
        "⚙️ 执行偏差",
        f"入场滑点: {format_value(recent['avg_entry_slippage_bps'], ' bps')}",
        f"出场滑点: {format_value(recent['avg_exit_slippage_bps'], ' bps')}",
        f"止盈/止损偏离: {format_value(recent['avg_stop_target_deviation_bps'], ' bps')} ({recent['stop_target_reference_count']} refs)",
        "",
        "📚 全部已平仓",
        f"交易: {all_metrics['trade_count']} 笔 | 收益: {format_value(all_metrics['total_return_pct'], '%')}",
    ]
    if comparison:
        lines.extend(["", "🧬 基准漂移"])
        for key, item in comparison.items():
            if isinstance(item, dict):
                lines.append(f"{key}: live {item['live']} vs expected {item['expected']} (delta {item['delta']})")
            else:
                lines.append(f"{key}: {item}")
    if flags:
        lines.extend(["", "🚦 体检提示"])
        lines.extend(f"- {flag['level']}: {flag['message']}" for flag in flags)
    lines.extend(["", "🧭 体检结论"])
    lines.extend(health_conclusion(report))
    lines.extend(["", "💰 启动资金建议"])
    lines.extend(launch_capital_advice(report))
    return "\n".join(lines)


def send_telegram(config_path: Path, message: str) -> None:
    import requests

    config = load_json(config_path)
    token = config.get("telegram_token")
    chat_id = config.get("telegram_chat_id")
    if not token or not chat_id:
        raise ValueError("telegram_token or telegram_chat_id missing in config")
    requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": message},
        timeout=10,
    ).raise_for_status()


def main() -> dict[str, Any]:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = ROOT / baseline_path

    state_db = resolve_state_db(config_path, args.state_db)
    baseline = load_json(baseline_path)
    actions = load_action_log(state_db)
    trades, diagnostics = build_live_trades(actions)
    report = build_report(
        config_path=config_path,
        state_db=state_db,
        baseline=baseline,
        actions=actions,
        trades=trades,
        diagnostics=diagnostics,
        window_days=args.window_days,
        recent_trades=args.recent_trades,
    )

    if args.json:
        output = json.dumps(report, ensure_ascii=False, indent=2)
    else:
        output = format_report(report)
    print(output)
    if args.send_telegram:
        send_telegram(config_path, format_report(report))
    return report


if __name__ == "__main__":
    main()
