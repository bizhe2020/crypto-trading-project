"""BTC scalp 入场漏斗审计（scripts/audit_btc_scalp_entry_funnel.py）与事件流季度工具
（scripts/audit_btc_scalp_replay_quarters.py）里纯函数的单元测试。

覆盖：第一个拒绝门的归因、触发窗口与触发扫描（含引擎的「吞没不算触碰」语义）、
R 空间汇总、置换检验 / bootstrap、实盘高杠杆保护与 shadow 风控状态机、
季度复利 / beat / share、两条事件流的 R 空间拆分。全部使用合成输入，不依赖外部数据。
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_btc_scalp_entry_funnel import (  # noqa: E402
    LiveGates,
    bootstrap_mean_ci,
    first_rejecting_gate,
    mirror_score_snapshot,
    permutation_pvalue,
    r_space_summary,
    scan_trigger,
    summarize_by_key,
    trigger_window,
)
from scripts.audit_btc_scalp_replay_quarters import (  # noqa: E402
    compound_by_quarter,
    compound_stats,
    quarter_concentration,
    r_space_decomposition,
    window_stats,
)


# ---------------------------------------------------------------- 归因 / 触发


def test_first_rejecting_gate_returns_first_failure_in_order():
    checks = [("bias_4h", True), ("daily_alpha_cap", False), ("ema_regime", False)]
    assert first_rejecting_gate(checks) == "daily_alpha_cap"


def test_first_rejecting_gate_all_pass_uses_passed_label():
    assert first_rejecting_gate([("a", True), ("b", True)]) == "executed"
    assert first_rejecting_gate([("a", True)], passed_label="") == ""
    assert first_rejecting_gate([]) == "executed"


def test_trigger_window_matches_engine_expiry_semantics():
    # 第一次尝试在下一根；最后一次尝试在 bos + window + 1
    assert trigger_window(detect_idx=100, bos_idx=90, window=25, last_eval_idx=10_000) == (101, 116)
    # 「出生即过期」的 pending 仍会在下一根尝试一次
    assert trigger_window(detect_idx=100, bos_idx=60, window=25, last_eval_idx=10_000) == (101, 101)
    # 截断到最后一根可评估的 bar
    assert trigger_window(detect_idx=100, bos_idx=95, window=40, last_eval_idx=110) == (101, 110)


def _bars(rows):
    opens = [r[0] for r in rows]
    highs = [r[1] for r in rows]
    lows = [r[2] for r in rows]
    closes = [r[3] for r in rows]
    return opens, highs, lows, closes


def test_scan_trigger_long_needs_touch_and_bullish_close():
    rows = [
        (105, 106, 103, 104),    # 0 不触碰（low 103 在区上方）
        (104, 104.5, 101.5, 101.8),  # 1 触碰但收阴
        (101, 103.0, 100.5, 102.5),  # 2 触碰且收阳 -> 触发
    ]
    opens, highs, lows, closes = _bars(rows)
    assert scan_trigger(opens, highs, lows, closes, direction="BULL", ob_top=102.0, ob_bottom=100.0,
                        start=0, last=2) == (2, "triggered")


def test_scan_trigger_engulfing_bar_is_not_a_touch():
    # low 低于区底、high 高于区顶：引擎的 in_ob 判定不算触碰
    rows = [(99.0, 103.0, 99.0, 102.9)]
    opens, highs, lows, closes = _bars(rows)
    assert scan_trigger(opens, highs, lows, closes, direction="BULL", ob_top=102.0, ob_bottom=100.0,
                        start=0, last=0) == (None, "no_touch")


def test_scan_trigger_reason_priority_and_allowed_callback():
    rows = [
        (104, 104.5, 101.5, 101.8),  # 0 许可内触碰但收阴 -> wrong_colour
        (101, 103.0, 100.5, 102.5),  # 1 触碰收阳，但被方向许可挡掉
    ]
    opens, highs, lows, closes = _bars(rows)
    blocked_last = lambda k: k != 1  # noqa: E731
    assert scan_trigger(opens, highs, lows, closes, direction="BULL", ob_top=102.0, ob_bottom=100.0,
                        start=0, last=1, allowed=blocked_last) == (None, "wrong_colour")
    never = lambda k: False  # noqa: E731
    assert scan_trigger(opens, highs, lows, closes, direction="BULL", ob_top=102.0, ob_bottom=100.0,
                        start=0, last=1, allowed=never) == (None, "touch_blocked")


def test_scan_trigger_short_needs_bearish_close():
    rows = [
        (100.5, 101.5, 100.2, 101.0),  # 触碰收阳 -> 空头不触发
        (101.2, 101.6, 100.4, 100.6),  # 触碰收阴 -> 触发
    ]
    opens, highs, lows, closes = _bars(rows)
    assert scan_trigger(opens, highs, lows, closes, direction="BEAR", ob_top=101.8, ob_bottom=101.0,
                        start=0, last=1) == (1, "triggered")


# ---------------------------------------------------------------- R 空间汇总 / 统计


def test_r_space_summary_aggregates_only_triggered_rows():
    rows = [
        {"net_r": 2.0, "gross_r": 2.1, "fee_r": 0.1, "exit_reason": "target_rr", "mfe_r": 3.0, "mae_r": 0.2},
        {"net_r": -1.1, "gross_r": -1.0, "fee_r": 0.1, "exit_reason": "stop_loss", "mfe_r": 0.4, "mae_r": 1.0},
        {"net_r": None},  # 没触发：只计入 setups
    ]
    s = r_space_summary(rows)
    assert s["setups"] == 3 and s["triggered"] == 2
    assert s["net_r_sum"] == pytest.approx(0.9)
    assert s["gross_r_sum"] == pytest.approx(1.1)
    assert s["net_r_mean"] == pytest.approx(0.45)
    assert s["fee_r_mean"] == pytest.approx(0.1)
    assert s["win_rate_pct"] == pytest.approx(50.0)
    assert s["tp_rate_pct"] == pytest.approx(50.0)
    assert s["mfe_r_median"] == pytest.approx(1.7)


def test_r_space_summary_without_triggers():
    s = r_space_summary([{"net_r": None}, {"net_r": None}])
    assert s["setups"] == 2 and s["triggered"] == 0 and s["net_r_mean"] is None and s["net_r_sum"] == 0.0


def test_summarize_by_key_orders_gates_and_computes_share():
    rows = [{"first_gate": "executed", "net_r": 1.0, "gross_r": 1.1, "fee_r": 0.1}] + [
        {"first_gate": "bias_4h", "net_r": -0.5, "gross_r": -0.4, "fee_r": 0.1} for _ in range(3)
    ]
    out = summarize_by_key(rows, "first_gate")
    assert list(out) == ["bias_4h", "executed"]
    assert out["bias_4h"]["share_pct"] == pytest.approx(75.0)
    assert out["bias_4h"]["net_r_sum"] == pytest.approx(-1.5)


def test_permutation_pvalue_detects_separation_and_handles_empty():
    assert permutation_pvalue([1.0] * 20, [-1.0] * 20, n_perm=500) < 0.01
    same = permutation_pvalue([0.0, 1.0] * 10, [0.0, 1.0] * 10, n_perm=500)
    assert same > 0.9
    assert permutation_pvalue([], [1.0]) is None


def test_bootstrap_mean_ci_degenerate_and_empty():
    assert bootstrap_mean_ci([0.5] * 10, n_boot=200) == (0.5, 0.5)
    assert bootstrap_mean_ci([]) == (None, None)
    lo, hi = bootstrap_mean_ci([-1.0, 0.0, 1.0, 2.0] * 10, n_boot=500)
    assert lo < 0.5 < hi


def test_mirror_score_snapshot_swaps_totals_and_negates_net():
    snap = {"bull_total": 2, "bear_total": 14, "net_score": -12, "conflict": False}
    mirrored = mirror_score_snapshot(snap)
    assert (mirrored["bull_total"], mirrored["bear_total"], mirrored["net_score"]) == (14, 2, 12)
    assert snap["bull_total"] == 2  # 不改原对象


# ---------------------------------------------------------------- 实盘执行层门


LIVE_PAYLOAD = {
    # ExecutorConfig 的必填字段
    "mode": "paper",
    "symbol": "BTC/USDT:USDT",
    "timeframe": "15m",
    "informative_timeframe": "4h",
    "margin_mode": "isolated",
    "max_open_positions": 1,
    "risk_per_trade": 0.035,
    "state_db_path": "state/unused_test.db",
    # 被测的执行层门
    "leverage": 20.0,
    "enable_high_leverage_guard": True,
    "high_leverage_guard_min_leverage": 8.0,
    "high_leverage_maintenance_margin_pct": 0.5,
    "high_leverage_min_liquidation_buffer_pct": 1.2,
    "high_leverage_max_stop_distance_pct": 2.0,
    "high_leverage_max_account_effective_leverage": 20.0,
    "enable_shadow_risk_gate": True,
    "shadow_daily_loss_stop_pct": 6.0,
    "shadow_equity_drawdown_stop_pct": 12.0,
    "shadow_equity_drawdown_cooldown_days": 2,
    "shadow_consecutive_loss_stop": 4,
}


def _gates() -> LiveGates:
    return LiveGates(dict(LIVE_PAYLOAD), [])


def test_hl_guard_stop_cap_and_liquidation_buffer_at_20x():
    gates = _gates()
    failures, diag = gates.hl_failures(True, 100.0, 98.0)  # 止损 2.0%：恰好不超
    assert failures == []
    assert diag["liquidation_buffer_pct"] == pytest.approx(2.5)  # 20x、MMR 0.5% -> 4.5% - 2.0%
    failures, _ = gates.hl_failures(True, 100.0, 97.9)
    assert failures == ["stop_distance_too_wide"]
    failures, _ = gates.hl_failures(True, 100.0, 96.5)  # 止损 3.5% -> 缓冲 1.0% < 1.2%
    assert set(failures) == {"liquidation_buffer_too_small", "stop_distance_too_wide"}
    failures, _ = gates.hl_failures(False, 100.0, 101.5, notional=21_000.0, capital=1_000.0)
    assert failures == ["account_effective_leverage_too_high"]


def _ts(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp()


def test_shadow_daily_loss_pauses_until_next_utc_day():
    gates = _gates()
    assert not gates.shadow_paused(_ts("2026-01-05 10:00"), 1000.0)
    gates.after_close(capital_after=930.0, pnl=-70.0, exit_time="2026-01-05 12:00")  # -7% 当日
    assert gates.shadow_paused(_ts("2026-01-05 23:45"), 930.0)
    assert not gates.shadow_paused(_ts("2026-01-06 00:00"), 930.0)


def test_shadow_drawdown_cooldown_and_peak_reset():
    gates = _gates()
    gates.shadow_paused(_ts("2026-01-01 00:00"), 1000.0)  # 用初始资金建状态
    gates.after_close(capital_after=1100.0, pnl=100.0, exit_time="2026-01-02 10:00")
    gates.after_close(capital_after=1050.0, pnl=-50.0, exit_time="2026-01-03 10:00")
    gates.after_close(capital_after=960.0, pnl=-90.0, exit_time="2026-01-04 10:00")  # 回撤 12.7%（当日 -8.6%）
    assert gates.state["drawdown_peak"] == pytest.approx(960.0)  # 触发后 peak 重置
    assert gates.shadow_paused(_ts("2026-01-05 23:45"), 960.0)  # 冷却到 1/4 当日 0 点 + 2 天
    assert not gates.shadow_paused(_ts("2026-01-06 00:00"), 960.0)


def test_shadow_consecutive_losses_reset_streak():
    gates = _gates()
    gates.shadow_paused(_ts("2026-02-01 00:00"), 10_000.0)
    for day in range(1, 5):  # 每笔 -0.5%：不触发日亏 / 回撤，只触发连亏 4
        gates.after_close(capital_after=10_000.0 - 50.0 * day, pnl=-50.0, exit_time=f"2026-02-0{day} 08:00")
    assert gates.trigger_counts.get("consecutive_loss") == 1
    assert gates.state["loss_streak"] == 0
    assert gates.shadow_paused(_ts("2026-02-04 20:00"), 9800.0)
    assert not gates.shadow_paused(_ts("2026-02-05 00:00"), 9800.0)


# ---------------------------------------------------------------- 季度 / 窗口 / 拆分


def test_compound_stats_and_quarters():
    stats = compound_stats([10.0, -10.0])
    assert stats["return_pct"] == pytest.approx(-1.0)
    assert stats["max_dd_pct"] == pytest.approx(10.0)
    events = [
        {"exit_time": "2025-01-10 00:00", "return_pct": 10.0},
        {"exit_time": "2025-02-10 00:00", "return_pct": -10.0},
        {"exit_time": "2025-04-01 00:00", "return_pct": 5.0},
    ]
    q = compound_by_quarter(events)
    assert list(q) == ["2025Q1", "2025Q2"]
    assert q["2025Q1"]["return_pct"] == pytest.approx(-1.0)
    assert q["2025Q2"]["trades"] == 1


def test_window_stats_filters_by_entry_time():
    events = [
        {"entry_time": "2025-12-31 23:45:00+00:00", "return_pct": 50.0},
        {"entry_time": "2026-01-01 00:00:00+00:00", "return_pct": 10.0},
    ]
    assert window_stats(events, start="2026-01-01")["return_pct"] == pytest.approx(10.0)
    assert window_stats(events, end="2026-01-01")["return_pct"] == pytest.approx(50.0)


def test_quarter_concentration_counts_beats_ties_and_new_quarters():
    base = {"2023Q1": 10.0, "2023Q2": 5.0, "2023Q3": -4.0}
    cand = {"2023Q1": 30.0, "2023Q2": 5.0, "2023Q3": -6.0, "2024Q1": 4.0}  # 2024Q1 是候选新增的季度
    out = quarter_concentration(cand, base)
    assert out["beat_quarters"] == 2  # 2023Q1 与 2024Q1
    assert out["total_quarters"] == 4
    assert out["best_quarter"] == "2023Q1"
    assert out["best_share_of_positive"] == pytest.approx(20.0 / 24.0, abs=1e-3)
    same = quarter_concentration(dict(base), dict(base))
    assert same["beat_quarters"] == 0 and same["best_share_of_positive"] is None


def test_r_space_decomposition_separates_matched_removed_added():
    base = [
        {"entry_idx": 1, "direction": "BULL", "net_r": 1.0, "gross_r": 1.1, "fee_r": 0.1},
        {"entry_idx": 2, "direction": "BULL", "net_r": -1.0, "gross_r": -0.9, "fee_r": 0.1},
    ]
    cand = [
        {"entry_idx": 1, "direction": "BULL", "net_r": 1.0, "gross_r": 1.1, "fee_r": 0.1},
        {"entry_idx": 3, "direction": "BEAR", "net_r": 2.0, "gross_r": 2.1, "fee_r": 0.1},
    ]
    d = r_space_decomposition(base, cand)
    assert d["net_r_delta"] == pytest.approx(3.0)
    assert d["matched"] == {"n": 1, "exit_changed": 0, "net_r_delta": 0.0}
    assert d["removed"]["net_r"] == pytest.approx(-1.0)
    assert d["added"]["net_r"] == pytest.approx(2.0)
    assert d["added_by_direction"]["BEAR"]["n"] == 1.0
