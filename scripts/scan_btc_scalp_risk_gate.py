#!/usr/bin/env python3
"""影子风控闸门（shadow risk gate）语义 A/B。

现状语义（引擎与执行层一致）：
  - 单日亏损 >= 6%      → 暂停到「下一个 UTC 日」
  - 连续亏损 >= 4 笔    → 暂停到「下一个 UTC 日」
  - 权益回撤 >= 12%     → 冷却 N 天，并把 drawdown_peak 重置为当前权益

后两条被研究文档点名：交易间隔本身就有数天，「暂停到下一 UTC 日」等于没停；
回撤触发后重置 peak，形成「亏 12% → 停 2 天 → 再亏 12%」的阶梯式放血。

本脚本在**固定入场序列**上做反事实：不改变任何入场/出场逻辑，只改闸门语义，
比较不同语义下的终值、最大回撤与触发次数。

用法::

    python3 scripts/scan_btc_scalp_risk_gate.py --windows 2022-01-01,2025-01-01
    python3 scripts/scan_btc_scalp_risk_gate.py --live-trades var/tokyo_audit/btc_scalp_20260920/trades_audit.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_readiness_report import (  # noqa: E402
    load_prepared_data,
    run_engine,
    trade_dataframe,
)

DEFAULT_CONFIG = ROOT / "var" / "tokyo_audit" / "btc_scalp_20260920" / "config.live.high-leverage-structure.json"
DEFAULT_DATA_15M = ROOT / "var" / "tokyo_audit" / "btc_scalp_20260920" / "BTC_USDT_USDT-15m-futures.feather"
DEFAULT_DATA_4H = ROOT / "var" / "tokyo_audit" / "btc_scalp_20260920" / "BTC_USDT_USDT-4h-futures.feather"
DEFAULT_LIVE_TRADES = ROOT / "var" / "tokyo_audit" / "btc_scalp_20260920" / "trades_audit.csv"
DEFAULT_OUTPUT = ROOT / "var" / "tmp" / "btc_opt" / "risk_gate_scan.json"


@dataclass(frozen=True)
class GatePolicy:
    """风控闸门语义。"""

    name: str
    daily_loss_stop_pct: float = 6.0
    daily_loss_pause_days: float = 0.0  # 0 = 暂停到下一 UTC 日（现状）
    consecutive_loss_stop: int = 4
    consecutive_loss_pause_days: float = 0.0
    equity_drawdown_stop_pct: float = 12.0
    equity_drawdown_cooldown_days: float = 2.0
    reset_peak_on_drawdown: bool = True
    halt_drawdown_pct: float = 0.0  # >0：触发后永久停牌


POLICIES: list[GatePolicy] = [
    GatePolicy(
        name="off",
        daily_loss_stop_pct=0.0,
        consecutive_loss_stop=0,
        equity_drawdown_stop_pct=0.0,
        halt_drawdown_pct=0.0,
    ),
    GatePolicy(name="current"),
    GatePolicy(name="no_peak_reset", reset_peak_on_drawdown=False),
    GatePolicy(name="streak_pause_3d", consecutive_loss_pause_days=3.0),
    GatePolicy(name="streak_pause_3d_no_reset", consecutive_loss_pause_days=3.0, reset_peak_on_drawdown=False),
    GatePolicy(name="dd12_cool5_no_reset", equity_drawdown_cooldown_days=5.0, reset_peak_on_drawdown=False),
    GatePolicy(name="dd12_cool7_no_reset", equity_drawdown_cooldown_days=7.0, reset_peak_on_drawdown=False),
    GatePolicy(name="dd10_cool7_no_reset", equity_drawdown_stop_pct=10.0, equity_drawdown_cooldown_days=7.0, reset_peak_on_drawdown=False),
    GatePolicy(name="dd8_cool7_no_reset", equity_drawdown_stop_pct=8.0, equity_drawdown_cooldown_days=7.0, reset_peak_on_drawdown=False),
    GatePolicy(
        name="strict_combo",
        consecutive_loss_stop=3,
        consecutive_loss_pause_days=3.0,
        equity_drawdown_stop_pct=12.0,
        equity_drawdown_cooldown_days=5.0,
        reset_peak_on_drawdown=False,
    ),
    GatePolicy(
        name="rec_dd12_cool7_streak3",
        consecutive_loss_stop=3,
        consecutive_loss_pause_days=2.0,
        daily_loss_pause_days=1.0,
        equity_drawdown_stop_pct=12.0,
        equity_drawdown_cooldown_days=7.0,
        reset_peak_on_drawdown=False,
    ),
    GatePolicy(name="only_halt25", halt_drawdown_pct=25.0),
    GatePolicy(
        name="strict_combo_halt25",
        consecutive_loss_stop=3,
        consecutive_loss_pause_days=3.0,
        equity_drawdown_stop_pct=12.0,
        equity_drawdown_cooldown_days=5.0,
        reset_peak_on_drawdown=False,
        halt_drawdown_pct=25.0,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="影子风控闸门语义 A/B（无前视口径）")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--windows", default="2022-01-01,2025-01-01")
    parser.add_argument("--live-trades", default=str(DEFAULT_LIVE_TRADES))
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument(
        "--trade-cache",
        default=str(ROOT / "var" / "tmp" / "btc_opt" / "trade_cache"),
        help="回测入场序列缓存目录；存在缓存时跳过引擎重跑",
    )
    parser.add_argument("--refresh", action="store_true", help="忽略缓存，强制重跑引擎")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def load_base_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    for key in ("api_key", "api_secret", "api_passphrase", "telegram_token", "telegram_chat_id"):
        payload.pop(key, None)
    return payload


def simulate_gate(trades: pd.DataFrame, policy: GatePolicy, initial_capital: float) -> dict[str, Any]:
    """在固定交易序列上模拟闸门：只跳过入场，不改变单笔盈亏。"""
    ordered = trades.sort_values("entry_time").reset_index(drop=True)
    capital = initial_capital
    peak = initial_capital
    loss_streak = 0
    pause_until = pd.Timestamp.min.tz_localize("UTC")
    halted = False
    day_start: dict[str, float] = {}
    day_pnl: dict[str, float] = {}
    accepted = 0
    skipped = 0
    triggers: dict[str, int] = {}
    capitals: list[float] = []
    returns: list[float] = []

    for _, trade in ordered.iterrows():
        entry_time = pd.Timestamp(trade["entry_time"]).tz_convert("UTC")
        exit_time = pd.Timestamp(trade["exit_time"]).tz_convert("UTC")
        if halted or entry_time < pause_until:
            skipped += 1
            continue
        capital_before = capital
        trade_return = float(trade["pnl_pct"])
        capital *= 1.0 + trade_return
        accepted += 1
        returns.append(trade_return)
        capitals.append(capital)
        peak = max(peak, capital)

        day_key = exit_time.strftime("%Y-%m-%d")
        day_start.setdefault(day_key, capital_before)
        day_pnl[day_key] = day_pnl.get(day_key, 0.0) + (capital - capital_before)

        loss_streak = 0 if trade_return > 0 else loss_streak + 1

        fired: list[str] = []
        if policy.daily_loss_stop_pct > 0 and day_start[day_key] > 0:
            daily_loss_pct = -day_pnl[day_key] / day_start[day_key] * 100.0
            if daily_loss_pct >= policy.daily_loss_stop_pct:
                fired.append("daily_loss")
                base = exit_time.normalize() + pd.Timedelta(days=1)
                if policy.daily_loss_pause_days > 0:
                    base = exit_time + pd.Timedelta(days=policy.daily_loss_pause_days)
                pause_until = max(pause_until, base)

        if policy.consecutive_loss_stop > 0 and loss_streak >= policy.consecutive_loss_stop:
            fired.append("consecutive_loss")
            base = exit_time.normalize() + pd.Timedelta(days=1)
            if policy.consecutive_loss_pause_days > 0:
                base = exit_time + pd.Timedelta(days=policy.consecutive_loss_pause_days)
            pause_until = max(pause_until, base)
            loss_streak = 0

        if policy.equity_drawdown_stop_pct > 0 and peak > 0:
            drawdown_pct = (peak - capital) / peak * 100.0
            if drawdown_pct >= policy.equity_drawdown_stop_pct:
                fired.append("equity_drawdown")
                pause_until = max(
                    pause_until,
                    exit_time + pd.Timedelta(days=policy.equity_drawdown_cooldown_days),
                )
                if policy.reset_peak_on_drawdown:
                    peak = capital
                loss_streak = 0

        if policy.halt_drawdown_pct > 0 and peak > 0:
            drawdown_pct = (peak - capital) / peak * 100.0
            if drawdown_pct >= policy.halt_drawdown_pct:
                fired.append("halt")
                halted = True

        for reason in fired:
            triggers[reason] = triggers.get(reason, 0) + 1

    max_dd = 0.0
    run_peak = initial_capital
    run_cap = initial_capital
    for value in capitals:
        run_cap = value
        run_peak = max(run_peak, run_cap)
        max_dd = max(max_dd, (run_peak - run_cap) / run_peak * 100.0)
    if returns:
        mean = sum(returns) / len(returns)
        std = (sum((r - mean) ** 2 for r in returns) / len(returns)) ** 0.5
        sharpe = mean / std * (252 ** 0.5) if std > 0 else 0.0
    else:
        sharpe = 0.0
    return {
        "policy": policy.name,
        "final_capital": round(capital, 2),
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100.0, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 3),
        "accepted": accepted,
        "skipped": skipped,
        "halted": halted,
        "trigger_counts": triggers,
    }


def print_policy_table(title: str, results: list[dict[str, Any]]) -> None:
    print(f"\n== {title} ==")
    header = f"{'policy':<26}{'return%':>10}{'dd%':>8}{'sharpe':>8}{'taken':>7}{'skip':>6}  triggers"
    print(header)
    print("-" * len(header))
    for item in results:
        print(
            f"{item['policy']:<26}{item['total_return_pct']:>10.1f}{item['max_drawdown_pct']:>8.1f}"
            f"{item['sharpe_ratio']:>8.2f}{item['accepted']:>7d}{item['skipped']:>6d}  {item['trigger_counts']}"
        )


def load_or_run_trades(
    window: str,
    base_payload: dict[str, Any],
    prepared: Any,
    cache_dir: Path,
    refresh: bool,
) -> tuple[pd.DataFrame, float]:
    """取某窗口的入场序列；有缓存则直接复用，避免重复跑引擎。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"trades_{window}.csv"
    initial_path = cache_dir / f"initial_{window}.json"
    if cache_path.exists() and initial_path.exists() and not refresh:
        trades = pd.read_csv(cache_path)
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
        initial = float(json.loads(initial_path.read_text())["initial_capital"])
        return trades, initial
    metrics, engine = run_engine(base_payload, prepared, window)
    trades = trade_dataframe(engine)
    initial = float(metrics.get("initial_capital", 1000.0))
    trades.to_csv(cache_path, index=False)
    initial_path.write_text(json.dumps({"initial_capital": initial, "window": window}) + "\n")
    return trades, initial


def main() -> None:
    args = parse_args()
    output: dict[str, Any] = {"policies": [asdict(p) for p in POLICIES], "windows": {}}

    windows = [w.strip() for w in str(args.windows).split(",") if w.strip()]
    base_payload = load_base_config(Path(args.config))
    prepared = load_prepared_data(
        Path(args.data_15m),
        Path(args.data_4h),
        pd.Timestamp(min(windows), tz="UTC"),
        base_payload.get("regime_switcher_thresholds"),
    )

    for window in windows:
        trades, initial = load_or_run_trades(
            window,
            base_payload,
            prepared,
            Path(args.trade_cache),
            bool(args.refresh),
        )
        results = [simulate_gate(trades, policy, initial) for policy in POLICIES]
        output["windows"][window] = {"initial_capital": initial, "trades": int(len(trades)), "results": results}
        print_policy_table(f"回测入场序列 {window}（{len(trades)} 笔原始入场）", results)

    if not args.skip_live:
        live_path = Path(args.live_trades)
        if live_path.exists():
            live = pd.read_csv(live_path)
            live = live.rename(columns={"open_ts": "entry_time", "close_ts": "exit_time"})
            live["entry_time"] = pd.to_datetime(live["entry_time"], utc=True)
            live["exit_time"] = pd.to_datetime(live["exit_time"], utc=True)
            live["pnl_pct"] = live["net_pnl"] / live["capital_at_entry"]
            initial = float(live["capital_at_entry"].iloc[0])
            results = [simulate_gate(live, policy, initial) for policy in POLICIES]
            output["live_trades"] = {"initial_capital": initial, "trades": int(len(live)), "results": results}
            print_policy_table(f"实盘 21 笔真实入场（起始 {initial:.0f} USDT）", results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"\n输出: {output_path}")


if __name__ == "__main__":
    main()
