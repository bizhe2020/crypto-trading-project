#!/usr/bin/env python3
"""BTC scalp 入场漏斗审计（唯一无前视口径 + 实盘执行层语义）。

回答两个问题：
1. 实盘为什么交易这么少：每个原始结构 setup（swing 上的 MSS + BOS + OB）第一个被哪道门挡掉；
2. 被挡掉的 setup 如果照实盘出场规则做了，会赚多少 R（固定风险、不复利、含手续费）。

两层，都只用 `load_prepared_data`（内部是 build_precomputed_state_confirmed_4h，见
docs/backtest_lookahead_policy.md），不改引擎行为：

A. 路径无关的 census：每根 swing K 线上，用「检测门全放开」的配置调用引擎自己的
   `_build_pending_pullback`，枚举两个方向的原始 setup；再按引擎顺序检查检测门
   （4h bias -> regime 方向许可 -> 日线乖离 cap -> 方向 EMA）。
B. live 语义的路径回放（FunnelEngine）：逐根复刻 `evaluate_range` 的双 pending 分支；开仓时按
   bot/okx_executor.py 的顺序套 score gate -> structure gate(+recall) -> shadow gate -> 高杠杆保护，
   被拒的开仓像实盘一样回滚（释放仓位、两个 pending 都已清空）。由此得到 position_open /
   slot_occupied / cleared_by_other / expired 等路径门，以及实盘仓位口径的复利曲线。

对每个没执行的 setup，按引擎触发规则（OB 区内触碰 + 收阳/收阴）找到反事实触发 bar，
用同一引擎「强制开仓」跑到出场，得到 R。已执行的 setup 用路径回放里的真实成交 R，
并用强制开仓复算一次做一致性校验。

用法::

    python3 scripts/audit_btc_scalp_entry_funnel.py \\
        --config var/tokyo_audit/btc_scalp_20261013/config.live.high-leverage-structure.json \\
        --data-15m var/tokyo_audit/btc_scalp_20261013/cut_20261013/BTC_USDT_USDT-15m-futures.feather \\
        --data-4h  var/tokyo_audit/btc_scalp_20261013/cut_20261013/BTC_USDT_USDT-4h-futures.feather \\
        --start-date 2023-01-01 --output-json var/tmp/btc_headroom/entry_funnel_2023.json

    # A/B：只跑 live 语义路径回放（跳过反事实模拟），在 live 配置上覆盖若干键
    python3 scripts/audit_btc_scalp_entry_funnel.py ... --path-only \\
        --override '{"high_leverage_max_stop_distance_pct": 2.5}' \\
        --events-out var/tmp/btc_headroom/path_hl25.jsonl

`--events-out` 写出与 replay paper log 同格式的 JSONL（decision=accepted、return_pct 为实盘仓位口径），
可直接喂给 scripts/audit_btc_scalp_replay_quarters.py 做季度切片 / beat / share。
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.okx_executor import ExecutorConfig  # noqa: E402
from scripts.audit_btc_scalp_replay_quarters import compound_by_quarter, quarter_label, window_stats  # noqa: E402
from scripts.confirmed_multiframe_score_utils import (  # noqa: E402
    align_confirmed_mapping,
    passes_score_gate,
    resample_confirmed_1h,
    score_snapshot,
)
from scripts.live_readiness_report import load_prepared_data  # noqa: E402
from scripts.scan_btc_scalp_optimization import load_base_config  # noqa: E402
from scripts.sota_liquidity_context import flatten_context_features, liquidity_context_for_entry  # noqa: E402
from strategy.scalp_robust_v2_core import ActionType, Direction, ScalpRobustEngine  # noqa: E402

BULL = Direction.BULL
BEAR = Direction.BEAR

# 门的顺序 = 实盘里一个 setup 依次经过的位置
DETECTION_GATES = ("bias_4h", "regime_direction", "daily_alpha_cap", "ema_regime")
PATH_GATES = ("position_open", "slot_occupied", "cleared_by_other", "expired")
EXECUTOR_GATES = ("score_gate", "structure_gate", "shadow_pause", "hl_guard")
TERMINAL = ("end_of_data", "executed")
GATE_ORDER = DETECTION_GATES + PATH_GATES + EXECUTOR_GATES + TERMINAL
FIXED_CAPITAL = 1000.0


# --------------------------------------------------------------------------------------
# 纯函数（单元测试覆盖）
# --------------------------------------------------------------------------------------


def first_rejecting_gate(checks: Iterable[tuple[str, bool]], passed_label: str = "executed") -> str:
    """按顺序返回第一个没通过的门；全部通过返回 passed_label。"""
    for name, ok in checks:
        if not ok:
            return name
    return passed_label


def trigger_window(detect_idx: int, bos_idx: int, window: int, last_eval_idx: int) -> tuple[int, int]:
    """pending 在 detect_idx 建立后可尝试触发的 bar 区间（含两端），与引擎语义一致。

    第一次尝试在下一根；每次尝试失败后若 k - bos_idx > window 就过期，
    所以最后一次尝试在 max(detect_idx + 1, bos_idx + window + 1)。
    """
    start = detect_idx + 1
    last = max(start, bos_idx + window + 1)
    return start, min(last, last_eval_idx)


def scan_trigger(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    *,
    direction: str,
    ob_top: float,
    ob_bottom: float,
    start: int,
    last: int,
    allowed: Callable[[int], bool] | None = None,
) -> tuple[int | None, str]:
    """复刻 `_open_action_from_pending` 的触发条件，返回 (触发 bar, 原因)。

    触碰：bar 的 low 或 high 落在 OB 区内；方向许可（allowed）不通过则这根不算；
    颜色：多头收阳（close > open），空头收阴（close < open）。
    未触发时原因优先级：wrong_colour（许可内触碰过但颜色不对）> touch_blocked > no_touch。
    """
    touched_blocked = False
    touched_wrong_colour = False
    for k in range(start, last + 1):
        in_zone = ob_bottom <= lows[k] <= ob_top or ob_bottom <= highs[k] <= ob_top
        if not in_zone:
            continue
        if allowed is not None and not allowed(k):
            touched_blocked = True
            continue
        if direction == BULL and closes[k] <= opens[k]:
            touched_wrong_colour = True
            continue
        if direction == BEAR and closes[k] >= opens[k]:
            touched_wrong_colour = True
            continue
        return k, "triggered"
    if touched_wrong_colour:
        return None, "wrong_colour"
    if touched_blocked:
        return None, "touch_blocked"
    return None, "no_touch"


def mirror_score_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """空头用的镜像打分：多空总分互换、净分取反，便于直接复用多头的 passes_score_gate 规则。"""
    return {
        **snapshot,
        "bull_total": snapshot["bear_total"],
        "bear_total": snapshot["bull_total"],
        "net_score": -int(snapshot["net_score"]),
    }


def r_space_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """R 空间汇总：只统计有模拟结果（net_r 非空）的行；R 已含手续费，固定风险、不复利。"""
    sims = [row for row in rows if row.get("net_r") is not None]
    out: dict[str, Any] = {"setups": len(rows), "triggered": len(sims)}
    if not sims:
        out.update({
            "net_r_sum": 0.0, "net_r_mean": None, "net_r_median": None, "gross_r_sum": 0.0,
            "gross_r_mean": None, "fee_r_mean": None, "win_rate_pct": None, "tp_rate_pct": None,
            "mfe_r_median": None, "mae_r_median": None,
        })
        return out
    net = np.array([float(row["net_r"]) for row in sims])
    gross = np.array([float(row["gross_r"]) for row in sims])
    fee = np.array([float(row["fee_r"]) for row in sims])
    mfe = np.array([float(row.get("mfe_r") or 0.0) for row in sims])
    mae = np.array([float(row.get("mae_r") or 0.0) for row in sims])
    tp = np.array([str(row.get("exit_reason") or "") == "target_rr" for row in sims])
    out.update({
        "net_r_sum": round(float(net.sum()), 3),
        "net_r_mean": round(float(net.mean()), 4),
        "net_r_median": round(float(np.median(net)), 4),
        "gross_r_sum": round(float(gross.sum()), 3),
        "gross_r_mean": round(float(gross.mean()), 4),
        "fee_r_mean": round(float(fee.mean()), 4),
        "win_rate_pct": round(float((net > 0).mean() * 100.0), 1),
        "tp_rate_pct": round(float(tp.mean() * 100.0), 1),
        "mfe_r_median": round(float(np.median(mfe)), 3),
        "mae_r_median": round(float(np.median(mae)), 3),
    })
    return out


def summarize_by_key(rows: Sequence[dict[str, Any]], key: str, order: Sequence[str] = GATE_ORDER) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key)), []).append(row)
    total = len(rows)
    ordered = [g for g in order if g in groups] + sorted(g for g in groups if g not in order)
    out: dict[str, Any] = {}
    for name in ordered:
        summary = r_space_summary(groups[name])
        summary["share_pct"] = round(len(groups[name]) / total * 100.0, 2) if total else 0.0
        out[name] = summary
    return out


def summarize_by_key_quarter(rows: Sequence[dict[str, Any]], key: str, time_key: str = "detect_time") -> dict[str, Any]:
    """{gate: {quarter: {setups, triggered, net_r_sum, net_r_mean}}}"""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        quarter = quarter_label(row[time_key])
        bucket = out.setdefault(str(row.get(key)), {}).setdefault(quarter, {"setups": 0, "triggered": 0, "net_r_sum": 0.0})
        bucket["setups"] += 1
        if row.get("net_r") is not None:
            bucket["triggered"] += 1
            bucket["net_r_sum"] += float(row["net_r"])
    for quarters in out.values():
        for bucket in quarters.values():
            bucket["net_r_sum"] = round(bucket["net_r_sum"], 3)
            bucket["net_r_mean"] = round(bucket["net_r_sum"] / bucket["triggered"], 4) if bucket["triggered"] else None
    return {gate: dict(sorted(quarters.items())) for gate, quarters in out.items()}


def permutation_pvalue(a: Sequence[float], b: Sequence[float], n_perm: int = 10000, seed: int = 20261013) -> float | None:
    """两样本均值差的双侧置换检验 p 值（含 +1 校正）。"""
    xa = np.asarray(a, dtype=float)
    xb = np.asarray(b, dtype=float)
    if len(xa) == 0 or len(xb) == 0:
        return None
    observed = abs(xa.mean() - xb.mean())
    pooled = np.concatenate([xa, xb])
    rng = np.random.default_rng(seed)
    n_a = len(xa)
    hits = 0
    for _ in range(int(n_perm)):
        rng.shuffle(pooled)
        if abs(pooled[:n_a].mean() - pooled[n_a:].mean()) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (int(n_perm) + 1)


def bootstrap_mean_ci(
    values: Sequence[float], n_boot: int = 5000, seed: int = 20261013, alpha: float = 0.05
) -> tuple[float | None, float | None]:
    data = np.asarray(values, dtype=float)
    if len(data) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    means: list[float] = []
    chunk = 500
    remaining = int(n_boot)
    while remaining > 0:
        size = min(chunk, remaining)
        idx = rng.integers(0, len(data), size=(size, len(data)))
        means.extend(data[idx].mean(axis=1).tolist())
        remaining -= size
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return round(float(lo), 4), round(float(hi), 4)


# --------------------------------------------------------------------------------------
# 实盘执行层门（bot/okx_executor.py 的顺序与语义）
# --------------------------------------------------------------------------------------


class LiveGates:
    """引擎产出开仓动作之后，实盘执行层依次套用的门：

    score gate（所有引擎开仓都按 sota_long 打分，含空头）-> structure gate（只限多，带 recall）
    -> shadow 风控暂停 -> 高杠杆保护（止损宽度 / 强平缓冲 / 账户有效杠杆）。
    shadow 状态只由真实平仓驱动（与 `_shadow_gate_after_close` 相同）。
    """

    def __init__(self, payload: dict[str, Any], c15m: list[Any], *, score_short_mode: str = "long_rule"):
        cfg = ExecutorConfig.from_dict(payload)
        self.score_enabled = bool(cfg.enable_sota_score_gate_live)
        # 实盘把所有引擎开仓都当 sota_long 打分（空头也要 bull_total>=8）= "long_rule"。
        # A/B：exempt = 空头不过 score gate；mirror = 空头用镜像规则（bear>=bull_min, net<=-net_min, bull<=bear_max）
        if score_short_mode not in {"long_rule", "exempt", "mirror"}:
            raise ValueError(f"unknown score_short_mode: {score_short_mode}")
        self.score_short_mode = score_short_mode
        self.score_rule = {
            "net_min": int(cfg.sota_score_net_min),
            "bull_min": int(cfg.sota_score_bull_min),
            "bear_max": int(cfg.sota_score_bear_max),
            "conflict_mode": str(cfg.sota_score_conflict_mode or "any"),
        }
        self.structure_enabled = bool(cfg.require_non_bearish_structure_for_long_live)
        self.recall_enabled = bool(cfg.enable_sota_rejected_smc_recall_long_live)
        self.recall_condition = str(cfg.sota_rejected_smc_recall_long_condition or "")
        self.recall_stage = str(cfg.sota_rejected_smc_recall_long_reject_stage or "")
        self.recall_regime = str(cfg.sota_rejected_smc_recall_long_regime_label or "")
        self.recall_target_leverage = float(cfg.sota_rejected_smc_recall_long_target_leverage or 0.0)
        self.leverage = float(cfg.leverage)
        self.hl_enabled = bool(cfg.enable_high_leverage_guard) and self.leverage >= float(cfg.high_leverage_guard_min_leverage)
        self.hl_mmr = max(float(cfg.high_leverage_maintenance_margin_pct), 0.0) / 100.0
        self.hl_min_buffer = float(cfg.high_leverage_min_liquidation_buffer_pct)
        self.hl_max_stop = float(cfg.high_leverage_max_stop_distance_pct)
        self.hl_max_account = float(cfg.high_leverage_max_account_effective_leverage)
        self.shadow_daily = float(cfg.shadow_daily_loss_stop_pct or 0.0)
        self.shadow_dd = float(cfg.shadow_equity_drawdown_stop_pct or 0.0)
        self.shadow_cool_days = int(cfg.shadow_equity_drawdown_cooldown_days or 0)
        self.shadow_streak = int(cfg.shadow_consecutive_loss_stop or 0)
        self.shadow_daily_pause_days = float(cfg.shadow_daily_loss_pause_days or 0.0)
        self.shadow_streak_pause_days = float(cfg.shadow_consecutive_loss_pause_days or 0.0)
        self.shadow_peak_reset = bool(cfg.shadow_drawdown_peak_reset)
        self.shadow_enabled = bool(cfg.enable_shadow_risk_gate) and (
            self.shadow_daily > 0 or self.shadow_dd > 0 or self.shadow_streak > 0
        )
        self.c1h = resample_confirmed_1h(c15m)
        self.mapping_1h = align_confirmed_mapping(self.c1h, c15m)
        self.state: dict[str, Any] | None = None
        self.trigger_counts: dict[str, int] = {}

    # ---- 单门判定 ----
    def score(self, engine: Any, idx: int) -> dict[str, Any]:
        return asdict(score_snapshot(engine, self.c1h, self.mapping_1h, int(idx)))

    def score_ok(self, snapshot: dict[str, Any]) -> bool:
        return (not self.score_enabled) or passes_score_gate(snapshot, **self.score_rule)

    @staticmethod
    def liquidity_features(engine: Any, idx: int, direction: str) -> dict[str, Any]:
        start = max(0, int(idx) - 360)
        candles = engine.c15m[start : int(idx) + 1]
        return flatten_context_features(liquidity_context_for_entry(candles, len(candles) - 1, str(direction)))

    def recall_ok(self, features: dict[str, Any], regime_label: str) -> bool:
        if not self.recall_enabled or self.recall_target_leverage <= 0:
            return False
        if self.recall_stage and self.recall_stage != "structure_gate":
            return False
        if self.recall_regime and str(regime_label) != self.recall_regime:
            return False
        condition = self.recall_condition
        if condition == "sweep_has_fvg":
            return bool(features.get("feature_recent_sweep_has_fvg"))
        if condition == "recent_fvg_near_entry":
            return bool(features.get("feature_recent_fvg_near_entry"))
        if condition == "mss_with_fvg":
            return str(features.get("feature_recent_sweep_status") or "") == "mss_with_fvg"
        if condition == "fvg_near_or_mss_with_fvg":
            return bool(features.get("feature_recent_fvg_near_entry")) or (
                str(features.get("feature_recent_sweep_status") or "") == "mss_with_fvg"
            )
        return False

    def hl_failures(
        self, is_long: bool, entry: float, stop: float, notional: float | None = None, capital: float | None = None
    ) -> tuple[list[str], dict[str, float]]:
        if not self.hl_enabled:
            return [], {}
        failures: list[str] = []
        if entry <= 0 or stop <= 0:
            return ["missing_entry_or_stop"], {}
        stop_pct = abs(entry - stop) / entry * 100.0
        if is_long:
            liq = entry * (1.0 - 1.0 / self.leverage + self.hl_mmr)
            buffer = (stop - liq) / entry * 100.0
        else:
            liq = entry * (1.0 + 1.0 / self.leverage - self.hl_mmr)
            buffer = (liq - stop) / entry * 100.0
        if self.hl_min_buffer > 0 and buffer < self.hl_min_buffer:
            failures.append("liquidation_buffer_too_small")
        if self.hl_max_stop > 0 and stop_pct > self.hl_max_stop:
            failures.append("stop_distance_too_wide")
        account = (notional / capital) if notional and capital else 0.0
        if self.hl_max_account > 0 and account > self.hl_max_account:
            failures.append("account_effective_leverage_too_high")
        return failures, {"stop_distance_pct": round(stop_pct, 4), "liquidation_buffer_pct": round(buffer, 4),
                          "account_effective_leverage": round(account, 4)}

    # ---- shadow 风控 ----
    def _ensure_state(self, capital: float) -> dict[str, Any]:
        if self.state is None:
            self.state = {"capital": capital, "drawdown_peak": capital, "loss_streak": 0,
                          "pause_until_ts": 0.0, "day_start_capital": {}, "day_pnl": {}}
        return self.state

    @staticmethod
    def _next_utc_day_ts(dt: datetime) -> float:
        return (datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc) + timedelta(days=1)).timestamp()

    @staticmethod
    def _cooldown_until_ts(dt: datetime, days: int) -> float:
        return (datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc) + timedelta(days=max(1, int(days)))).timestamp()

    def shadow_paused(self, ts: float, capital: float) -> bool:
        if not self.shadow_enabled:
            return False
        return ts < float(self._ensure_state(capital)["pause_until_ts"])

    def after_close(self, capital_after: float, pnl: float, exit_time: str) -> None:
        if not self.shadow_enabled:
            return
        state = self._ensure_state(capital_after - pnl)
        capital = float(capital_after)
        capital_before = capital - float(pnl)
        state["capital"] = capital
        state["drawdown_peak"] = max(float(state["drawdown_peak"]), capital)
        action_dt = datetime.strptime(str(exit_time)[:16], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        day_key = action_dt.strftime("%Y-%m-%d")
        if day_key not in state["day_start_capital"]:
            state["day_start_capital"][day_key] = capital_before
            state["day_pnl"][day_key] = 0.0
        state["day_pnl"][day_key] += float(pnl)
        state["loss_streak"] = 0 if pnl > 0 else int(state["loss_streak"]) + 1
        daily_pause = (action_dt + timedelta(days=self.shadow_daily_pause_days)).timestamp() if self.shadow_daily_pause_days > 0 else self._next_utc_day_ts(action_dt)
        streak_pause = (action_dt + timedelta(days=self.shadow_streak_pause_days)).timestamp() if self.shadow_streak_pause_days > 0 else self._next_utc_day_ts(action_dt)
        start_capital = float(state["day_start_capital"][day_key])
        if self.shadow_daily > 0 and start_capital > 0:
            if -float(state["day_pnl"][day_key]) / start_capital * 100.0 >= self.shadow_daily:
                state["pause_until_ts"] = max(float(state["pause_until_ts"]), daily_pause)
                self.trigger_counts["daily_loss"] = self.trigger_counts.get("daily_loss", 0) + 1
        if self.shadow_streak > 0 and int(state["loss_streak"]) >= self.shadow_streak:
            state["pause_until_ts"] = max(float(state["pause_until_ts"]), streak_pause)
            state["loss_streak"] = 0
            self.trigger_counts["consecutive_loss"] = self.trigger_counts.get("consecutive_loss", 0) + 1
        peak = float(state["drawdown_peak"])
        if self.shadow_dd > 0 and peak > 0 and (peak - capital) / peak * 100.0 >= self.shadow_dd:
            state["pause_until_ts"] = max(float(state["pause_until_ts"]), self._cooldown_until_ts(action_dt, self.shadow_cool_days))
            if self.shadow_peak_reset:
                state["drawdown_peak"] = capital
            state["loss_streak"] = 0
            self.trigger_counts["equity_drawdown"] = self.trigger_counts.get("equity_drawdown", 0) + 1

    # ---- 组合 ----
    def downstream_gate(
        self,
        engine: Any,
        idx: int,
        direction: str,
        entry: float,
        stop: float,
        *,
        bearish_structure: bool,
        regime_label: str,
        notional: float | None = None,
        capital: float | None = None,
        shadow_ts: float | None = None,
    ) -> tuple[str | None, dict[str, Any]]:
        """返回第一个拒绝的执行层门（None = 全部通过）。shadow_ts 为 None 时不检查 shadow。"""
        detail: dict[str, Any] = {}
        short_exempt = direction == BEAR and self.score_short_mode == "exempt"
        if self.score_enabled and not short_exempt:
            snapshot = self.score(engine, idx)
            detail["score"] = {k: snapshot[k] for k in ("bull_total", "bear_total", "net_score", "conflict")}
            if direction == BEAR and self.score_short_mode == "mirror":
                ok = self.score_ok(mirror_score_snapshot(snapshot))
            else:
                ok = self.score_ok(snapshot)
            if not ok:
                return "score_gate", detail
        if self.structure_enabled and direction == BULL and bearish_structure:
            features = self.liquidity_features(engine, idx, direction)
            recalled = self.recall_ok(features, regime_label)
            detail["structure"] = {"bearish": True, "recalled": recalled,
                                   "sweep_has_fvg": bool(features.get("feature_recent_sweep_has_fvg"))}
            if not recalled:
                return "structure_gate", detail
        if shadow_ts is not None and self.shadow_paused(shadow_ts, capital or 0.0):
            detail["shadow_pause_until"] = float(self.state["pause_until_ts"]) if self.state else 0.0
            return "shadow_pause", detail
        failures, diag = self.hl_failures(direction == BULL, entry, stop, notional, capital)
        if diag:
            detail["hl"] = diag
        if failures:
            detail["hl_failures"] = failures
            return "hl_guard", detail
        return None, detail

    def check_open(self, engine: Any, action: Any, idx: int) -> tuple[str, dict[str, Any]]:
        meta = action.metadata or {}
        gate, detail = self.downstream_gate(
            engine,
            idx,
            str(action.direction),
            float(action.entry_price or 0.0),
            float(action.stop_price or 0.0),
            bearish_structure=bool(meta.get("feature_bearish_structure", False)),
            regime_label=str(meta.get("regime_label") or ""),
            notional=float(meta.get("notional") or 0.0),
            capital=float(meta.get("capital_at_entry") or engine.capital or 0.0),
            shadow_ts=float(engine.c15m[idx].ts),
        )
        return gate or "executed", detail


# --------------------------------------------------------------------------------------
# 引擎：census + live 语义路径回放
# --------------------------------------------------------------------------------------


@dataclass
class Setup:
    sid: int
    detect_idx: int
    detect_time: str
    direction: str
    bias: str
    regime_label: str
    bos_idx: int
    ob_top: float
    ob_bottom: float
    window: int
    daily_dev_pct: float | None
    first_gate: str | None = None
    stored: bool = False
    trigger_idx: int | None = None
    gate_detail: dict[str, Any] = field(default_factory=dict)


class FunnelEngine(ScalpRobustEngine):
    """复刻 evaluate_range 双 pending 分支 + 实盘执行层门；不改父类任何行为。"""

    def init_funnel(self, gates: LiveGates) -> None:
        if not self._base_config.enable_dual_pending_state:
            raise NotImplementedError("只支持 enable_dual_pending_state=true（live 配置）")
        if self._base_config.enable_bos_direct_entry:
            raise NotImplementedError("不支持 enable_bos_direct_entry")
        self.gates = gates
        self.setups: list[Setup] = []
        self.open_log: list[dict[str, Any]] = []
        self._pending_sid: dict[int, int] = {}
        self._swing_bars = self.precomputed.highs_set | self.precomputed.lows_set
        self._permissive_cache: dict[str, Any] = {}

    def _permissive_config(self, label: str) -> Any:
        cfg = self._permissive_cache.get(label)
        if cfg is None:
            cfg = deepcopy(self.config)
            cfg.allow_long = True
            cfg.allow_short = True
            cfg.enable_daily_alpha_gate = False
            cfg.long_regime_filter_1d_ema_period = None
            cfg.short_regime_filter_1d_ema_period = None
            cfg.regime_filter_1d_ema_period = None
            self._permissive_cache[label] = cfg
        return cfg

    def _ema_regime_ok(self, idx: int, direction: str) -> bool:
        cfg = self.config
        if not cfg.enable_directional_regime_switch:
            return True if cfg.regime_filter_1d_ema_period is None else self._regime_ok_for_idx(idx)
        period = cfg.long_regime_filter_1d_ema_period if direction == BULL else cfg.short_regime_filter_1d_ema_period
        if period is None:
            period = cfg.regime_filter_1d_ema_period
        if period is None:
            return True
        return self._regime_state_for_idx(idx, int(period), "bull" if direction == BULL else "bear")

    def daily_dev_for_idx(self, idx: int) -> float | None:
        values = self.precomputed.daily_dev_pct_4h
        if not values or idx < 0 or idx >= len(self.mapping):
            return None
        mapped = self.mapping[idx]
        if mapped < 0 or mapped >= len(values):
            return None
        value = values[mapped]
        return None if value is None or value != value else round(float(value), 3)

    def _census(self, idx: int, label: str, bias: str) -> list[Setup]:
        real = self.config
        permissive = self._permissive_config(label)
        found: list[Setup] = []
        for direction in (BULL, BEAR):
            self.config = permissive
            try:
                pending = self._build_pending_pullback(idx, direction)
            finally:
                self.config = real
            if pending is None:
                continue
            allow_flag = real.allow_long if direction == BULL else real.allow_short
            setup = Setup(
                sid=len(self.setups),
                detect_idx=idx,
                detect_time=self._timestamp_for_idx(idx),
                direction=direction,
                bias=str(bias),
                regime_label=str(label),
                bos_idx=int(pending.bos_idx),
                ob_top=float(pending.ob_zone["top"]),
                ob_bottom=float(pending.ob_zone["bottom"]),
                window=int(pending.pullback_window),
                daily_dev_pct=self.daily_dev_for_idx(idx),
            )
            gate = first_rejecting_gate(
                [
                    ("bias_4h", bias == direction),
                    ("regime_direction", bool(allow_flag)),
                    ("daily_alpha_cap", self._daily_alpha_gate_ok_for_idx(idx, direction)),
                    ("ema_regime", self._ema_regime_ok(idx, direction)),
                ],
                passed_label="",
            )
            setup.first_gate = gate or None
            self.setups.append(setup)
            found.append(setup)
        return found

    @staticmethod
    def _mark(setups: list[Setup], gate: str) -> None:
        for setup in setups:
            if setup.first_gate is None:
                setup.first_gate = gate

    def run_funnel(self, start_idx: int, end_idx: int) -> None:
        for i in range(start_idx, end_idx):
            label = self._apply_regime_switch_for_idx(i)
            exited = False
            if self.position is not None:
                self.manage_position(i)
                if self.position is None:
                    exited = True
                    trade = self.trades[-1]
                    self.gates.after_close(self.capital, trade.pnl, trade.exit_time)
            bias = self._bias_for_idx(i)
            census = self._census(i, label, bias) if i in self._swing_bars else []
            if exited or self.position is not None:
                self._mark(census, "position_open")
                continue
            active = any(p is not None for p in self.pending_by_direction.values())
            if self.config.use_hfvf_filter and bias == Direction.NONE and not active:
                continue
            rejected_open = False
            for direction in (BULL, BEAR):
                pending = self.pending_by_direction[direction]
                if pending is None:
                    continue
                action = self._open_action_from_pending(i, pending)
                if action is not None:
                    sid = self._pending_sid.pop(id(pending), None)
                    other = self.pending_by_direction[BEAR if direction == BULL else BULL]
                    if other is not None:
                        other_sid = self._pending_sid.pop(id(other), None)
                        if other_sid is not None:
                            self.setups[other_sid].first_gate = "cleared_by_other"
                            self.setups[other_sid].gate_detail["cleared_at_idx"] = i
                    self.pending_by_direction[BULL] = None
                    self.pending_by_direction[BEAR] = None
                    gate, detail = self.gates.check_open(self, action, i)
                    if sid is not None:
                        setup = self.setups[sid]
                        setup.trigger_idx = i
                        setup.first_gate = gate
                        setup.gate_detail.update(detail)
                    self.open_log.append({"idx": i, "time": self._timestamp_for_idx(i), "direction": direction,
                                          "decision": gate, "sid": sid})
                    if gate != "executed":
                        # 实盘：被拒的开仓回滚本地仓位；该 bar 内引擎已认为有仓，不再检测新 setup
                        self.position = None
                        rejected_open = True
                    break
                if self._pending_expired(i, pending):
                    sid = self._pending_sid.pop(id(pending), None)
                    if sid is not None:
                        self.setups[sid].first_gate = "expired"
                        self.setups[sid].gate_detail["expired_at_idx"] = i
                    self.pending_by_direction[direction] = None
            if self.position is not None or rejected_open:
                self._mark(census, "position_open")
                continue
            for setup in census:
                if setup.first_gate is not None:
                    continue
                if self.pending_by_direction[setup.direction] is not None:
                    setup.first_gate = "slot_occupied"
                    continue
                pending = self._build_pending_pullback(i, bias)
                if pending is None or pending.direction != setup.direction or int(pending.bos_idx) != setup.bos_idx:
                    setup.first_gate = "census_mismatch"
                    continue
                self.pending_by_direction[setup.direction] = pending
                self._pending_sid[id(pending)] = setup.sid
                setup.stored = True
        for pending in list(self.pending_by_direction.values()):
            if pending is not None:
                sid = self._pending_sid.pop(id(pending), None)
                if sid is not None:
                    self.setups[sid].first_gate = "end_of_data"
        if self.position is not None:
            self.close_position(len(self.c15m) - 1, "end_of_data")


def build_engine(cls: type, payload: dict[str, Any], prepared: Any) -> Any:
    """与 scripts/live_readiness_report.run_engine 完全相同的构造方式。"""
    config = ExecutorConfig.from_dict(payload).to_scalp_strategy_config()
    engine = cls(prepared.c4h, prepared.c15m, prepared.mapping, prepared.precomputed, config)
    engine._regime_switch_cache = {
        c4h_idx: (label, engine._config_for_regime(label)) for c4h_idx, label in prepared.regime_labels.items()
    }
    engine._regime_feature_cache = dict(prepared.regime_features)
    return engine


def start_index(c15m: list[Any], start_date: str) -> int:
    start_ts = pd.Timestamp(start_date, tz="UTC").timestamp()
    first = next((i for i, candle in enumerate(c15m) if candle.ts >= start_ts), 0)
    return first + 100  # 与 run_backtest 相同的 warmup


# --------------------------------------------------------------------------------------
# 强制开仓模拟（固定风险、不复利）
# --------------------------------------------------------------------------------------


def trade_r_metrics(trade: Any, c15m: list[Any]) -> dict[str, Any] | None:
    risk_px = abs(float(trade.entry_price) - float(trade.initial_stop_price))
    risk = float(trade.quantity) * risk_px
    if risk <= 0:
        return None
    is_long = trade.direction == BULL
    lo, hi = int(trade.entry_idx) + 1, int(trade.exit_idx)
    if hi >= lo:
        top = max(c.h for c in c15m[lo : hi + 1])
        bottom = min(c.l for c in c15m[lo : hi + 1])
        mfe = (top - trade.entry_price) / risk_px if is_long else (trade.entry_price - bottom) / risk_px
        mae = (trade.entry_price - bottom) / risk_px if is_long else (top - trade.entry_price) / risk_px
    else:
        mfe = mae = 0.0
    return {
        "net_r": round(float(trade.pnl) / risk, 5),
        "gross_r": round(float(trade.gross_pnl) / risk, 5),
        "fee_r": round(float(trade.fees) / risk, 5),
        "exit_reason": str(trade.exit_reason),
        "entry_time": str(trade.entry_time),
        "exit_time": str(trade.exit_time),
        "bars_held": int(trade.exit_idx) - int(trade.entry_idx),
        "mfe_r": round(float(mfe), 4),
        "mae_r": round(float(mae), 4),
        "stop_pct": round(risk_px / float(trade.entry_price) * 100.0, 4),
        "trail_style": str(trade.trail_style or ""),
        "risk_regime": str(trade.risk_regime or ""),
    }


def forced_entry_prices(sim: Any, setup: Setup, trigger_idx: int) -> tuple[float, float, float]:
    """与 `_open_action_from_pending` 相同的入场 / 止损 / 目标（需先切到 trigger bar 的 regime 配置）。"""
    sim._apply_regime_switch_for_idx(trigger_idx)
    entry = float(sim.c15m[trigger_idx].c)
    raw = setup.ob_bottom if setup.direction == BULL else setup.ob_top
    buffer = sim._sl_buffer_pct_for_direction(setup.direction)
    stop = raw * (1 - buffer / 100.0) if setup.direction == BULL else raw * (1 + buffer / 100.0)
    _, target_rr, _, _ = sim._exit_template_for_idx(trigger_idx, setup.direction)
    risk = abs(entry - stop)
    target = entry + risk * target_rr if setup.direction == BULL else entry - risk * target_rr
    return entry, stop, target


def simulate_forced(sim: Any, setup: Setup, trigger_idx: int, seed_trades: list[Any]) -> dict[str, Any] | None:
    sim.position = None
    sim.trades = list(seed_trades)  # auto-TIT 看历史连亏，用路径回放里此前已平仓的交易做种子
    sim.capital = FIXED_CAPITAL
    entry, stop, target = forced_entry_prices(sim, setup, trigger_idx)
    action = sim.open_position(trigger_idx, setup.direction, entry, stop, target)
    if action is None or sim.position is None:
        return None
    last_eval = len(sim.c15m) - 2
    for k in range(trigger_idx + 1, last_eval + 1):
        sim._apply_regime_switch_for_idx(k)
        sim.manage_position(k)
        if sim.position is None:
            break
    if sim.position is not None:
        sim.close_position(len(sim.c15m) - 1, "end_of_data")
    return trade_r_metrics(sim.trades[-1], sim.c15m)


def make_live_allowed(sim: Any, direction: str) -> Callable[[int], bool]:
    def allowed(k: int) -> bool:
        sim._apply_regime_switch_for_idx(k)
        flag = sim.config.allow_long if direction == BULL else sim.config.allow_short
        return bool(flag) and sim._regime_ok_for_direction_idx(k, direction)

    return allowed


# --------------------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------------------


def path_performance(engine: Any, windows: Sequence[str]) -> dict[str, Any]:
    trades = list(engine.trades)
    metrics = engine.compute_metrics()
    per_trade = [trade_r_metrics(t, engine.c15m) for t in trades]
    r_rows = [row for row in per_trade if row]
    events = []
    for t, r in zip(trades, per_trade):
        event = {"entry_time": str(t.entry_time), "exit_time": str(t.exit_time), "entry_idx": int(t.entry_idx),
                 "exit_idx": int(t.exit_idx), "return_pct": float(t.pnl_pct) * 100.0,
                 "event_type": "sota_long" if t.direction == BULL else "engine_short",
                 "direction": t.direction, "decision": "accepted"}
        if r:
            event.update({k: r[k] for k in ("net_r", "gross_r", "fee_r", "exit_reason", "mfe_r", "mae_r", "stop_pct",
                                            "trail_style", "risk_regime")})
        events.append(event)
    out = {
        "total_return_pct": round(float(metrics.get("total_return_pct", 0.0)), 2),
        "max_drawdown_pct": round(float(metrics.get("max_drawdown_pct", 0.0)), 2),
        "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
        "trades": len(trades),
        "win_rate_pct": round(float(metrics.get("win_rate", 0.0)), 2),
        "target_hit_rate_pct": round(float(metrics.get("target_hit_rate", 0.0)), 2),
        "r_space": r_space_summary(r_rows),
        "quarters": {q: {k: round(v, 2) for k, v in s.items()} for q, s in compound_by_quarter(events).items()},
        "windows": {w: {k: round(v, 2) for k, v in window_stats(events, start=w).items()} for w in windows},
        "halves": {
            "< 2025-01-01": {k: round(v, 2) for k, v in window_stats(events, end="2025-01-01").items()},
            ">= 2025-01-01": {k: round(v, 2) for k, v in window_stats(events, start="2025-01-01").items()},
        },
    }
    return {"summary": out, "events": events}


def monthly_table(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        month = str(row["detect_time"])[:7]
        bucket = out.setdefault(month, {"raw_setups": 0, "passed_detection": 0, "stored": 0,
                                        "path_triggers": 0, "executed": 0})
        bucket["raw_setups"] += 1
        if row["first_gate"] not in DETECTION_GATES:
            bucket["passed_detection"] += 1
        if row.get("stored"):
            bucket["stored"] += 1
        if row.get("trigger_idx") is not None:
            bucket["path_triggers"] += 1
        if row["first_gate"] == "executed":
            bucket["executed"] += 1
    return dict(sorted(out.items()))


def significance_table(rows: Sequence[dict[str, Any]], min_n: int = 30, n_perm: int = 10000) -> dict[str, Any]:
    executed = [float(r["net_r"]) for r in rows if r["first_gate"] == "executed" and r.get("net_r") is not None]
    out: dict[str, Any] = {"executed_n": len(executed),
                           "executed_mean_net_r": round(float(np.mean(executed)), 4) if executed else None,
                           "executed_ci95": bootstrap_mean_ci(executed)}
    gates: dict[str, Any] = {}
    for gate in GATE_ORDER:
        if gate == "executed":
            continue
        values = [float(r["net_r"]) for r in rows if r["first_gate"] == gate and r.get("net_r") is not None]
        if len(values) < min_n:
            continue
        gates[gate] = {
            "n": len(values),
            "mean_net_r": round(float(np.mean(values)), 4),
            "ci95": bootstrap_mean_ci(values),
            "p_vs_executed": round(permutation_pvalue(values, executed, n_perm=n_perm), 4) if executed else None,
        }
        passing = [float(r["net_r"]) for r in rows if r["first_gate"] == gate and r.get("net_r") is not None
                   and r.get("downstream_gate") is None]
        if len(passing) >= min_n:
            gates[gate]["downstream_pass"] = {
                "n": len(passing),
                "mean_net_r": round(float(np.mean(passing)), 4),
                "ci95": bootstrap_mean_ci(passing),
                "p_vs_executed": round(permutation_pvalue(passing, executed, n_perm=n_perm), 4) if executed else None,
            }
    out["gates"] = gates
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-15m", required=True)
    parser.add_argument("--data-4h", required=True)
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--override", default="", help="覆盖 live 配置键的 JSON（A/B 用）")
    parser.add_argument("--label", default="base")
    parser.add_argument("--path-only", action="store_true", help="只跑 live 语义路径回放，跳过反事实模拟")
    parser.add_argument("--score-short-mode", default="long_rule", choices=("long_rule", "exempt", "mirror"),
                        help="A/B：空头开仓的 score gate 口径。long_rule=实盘现状（空头也按多头规则打分）；"
                             "exempt=空头不过 score gate；mirror=空头用镜像规则")
    parser.add_argument("--recent-start", default="2026-08-01")
    parser.add_argument("--windows", default="2025-01-01,2026-01-01,2026-07-15,2026-09-10")
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--output-json", default="var/tmp/btc_headroom/entry_funnel.json")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--events-out", default="", help="路径回放成交的 JSONL（replay paper log 格式）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    payload = load_base_config(Path(args.config))
    overrides = json.loads(args.override) if args.override.strip() else {}
    payload.update(overrides)
    windows = [w.strip() for w in str(args.windows).split(",") if w.strip()]

    prepared = load_prepared_data(Path(args.data_15m), Path(args.data_4h), pd.Timestamp(args.start_date, tz="UTC"),
                                  payload.get("regime_switcher_thresholds"))
    print(f"数据 {prepared.start} → {prepared.end}  15m={len(prepared.c15m)}  ({time.time() - t0:.1f}s)", flush=True)

    engine = build_engine(FunnelEngine, payload, prepared)
    gates = LiveGates(payload, prepared.c15m, score_short_mode=str(args.score_short_mode))
    engine.init_funnel(gates)
    first = start_index(prepared.c15m, args.start_date)
    engine.run_funnel(first, len(prepared.c15m) - 1)
    print(f"路径回放完成：setups={len(engine.setups)} 开仓尝试={len(engine.open_log)} 成交={len(engine.trades)} "
          f"({time.time() - t0:.1f}s)", flush=True)

    perf = path_performance(engine, windows)
    if args.events_out:
        out_events = Path(args.events_out)
        out_events.parent.mkdir(parents=True, exist_ok=True)
        with out_events.open("w", encoding="utf-8") as handle:
            for event in perf["events"]:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    open_decisions: dict[str, int] = {}
    for item in engine.open_log:
        open_decisions[item["decision"]] = open_decisions.get(item["decision"], 0) + 1

    result: dict[str, Any] = {
        "metadata": {
            "label": args.label,
            "config": str(Path(args.config).resolve()),
            "overrides": overrides,
            "score_short_mode": str(args.score_short_mode),
            "start_date": args.start_date,
            "data_start": str(prepared.start),
            "data_end": str(prepared.end),
            "path_only": bool(args.path_only),
            "gate_order": list(GATE_ORDER),
            "fixed_capital_for_r": FIXED_CAPITAL,
        },
        "path_run": {**perf["summary"], "open_attempt_decisions": open_decisions,
                     "shadow_trigger_counts": dict(gates.trigger_counts)},
    }

    if not args.path_only:
        sim = build_engine(ScalpRobustEngine, payload, prepared)
        c = prepared.c15m
        opens = [x.o for x in c]
        highs = [x.h for x in c]
        lows = [x.l for x in c]
        closes = [x.c for x in c]
        last_eval = len(c) - 2
        path_trades = sorted(engine.trades, key=lambda t: int(t.exit_idx))
        exit_idx_list = [int(t.exit_idx) for t in path_trades]
        trade_by_entry = {(int(t.entry_idx), t.direction): t for t in engine.trades}
        rows: list[dict[str, Any]] = []
        mismatches: list[float] = []
        for setup in engine.setups:
            row = {k: v for k, v in asdict(setup).items() if k != "gate_detail"}
            row["gate_detail"] = setup.gate_detail
            gate = setup.first_gate or "unknown"
            trigger: int | None = None
            reason = ""
            if gate in EXECUTOR_GATES or gate == "executed":
                trigger, reason = setup.trigger_idx, "path_trigger"
            else:
                start, last = trigger_window(setup.detect_idx, setup.bos_idx, setup.window, last_eval)
                allowed = None if gate in DETECTION_GATES else make_live_allowed(sim, setup.direction)
                trigger, reason = scan_trigger(opens, highs, lows, closes, direction=setup.direction,
                                               ob_top=setup.ob_top, ob_bottom=setup.ob_bottom,
                                               start=start, last=last, allowed=allowed)
            row["cf_trigger_idx"] = trigger
            row["cf_trigger_time"] = sim._timestamp_for_idx(trigger) if trigger is not None else None
            row["cf_reason"] = reason
            metrics = None
            if trigger is not None:
                seeds = path_trades[: bisect.bisect_left(exit_idx_list, int(trigger))]
                metrics = simulate_forced(sim, setup, int(trigger), seeds)
                if gate not in EXECUTOR_GATES and gate != "executed":
                    entry, stop, _ = forced_entry_prices(sim, setup, int(trigger))
                    features = sim._regime_features_for_idx(int(trigger))
                    down, detail = gates.downstream_gate(
                        sim, int(trigger), setup.direction, entry, stop,
                        bearish_structure=bool(features.get("bearish_structure", False)),
                        regime_label=str(sim._regime_switch_label_for_idx(int(trigger))),
                    )
                    row["downstream_gate"] = down
                    row["downstream_detail"] = detail
            if gate == "executed":
                actual = trade_by_entry.get((int(setup.trigger_idx), setup.direction))
                actual_metrics = trade_r_metrics(actual, engine.c15m) if actual is not None else None
                if actual_metrics and metrics:
                    mismatches.append(abs(actual_metrics["net_r"] - metrics["net_r"]))
                metrics = actual_metrics or metrics
            if metrics:
                row.update(metrics)
            else:
                row.update({"net_r": None, "gross_r": None, "fee_r": None})
            rows.append(row)

        recent = [r for r in rows if r["detect_time"] >= args.recent_start]
        result["funnel"] = {
            "all": summarize_by_key(rows, "first_gate"),
            "by_direction": {d: summarize_by_key([r for r in rows if r["direction"] == d], "first_gate")
                             for d in (BULL, BEAR)},
            "by_quarter": summarize_by_key_quarter(rows, "first_gate"),
            "recent": summarize_by_key(recent, "first_gate"),
            "recent_start": args.recent_start,
        }
        # 检测门 / 路径门挡掉的 setup：如果放开它，后面的执行层门（score/structure/高杠杆）还会不会拦
        down_rows = [r for r in rows if r["first_gate"] not in EXECUTOR_GATES and r["first_gate"] != "executed"
                     and r.get("net_r") is not None]
        for r in down_rows:
            r["downstream_gate_label"] = r.get("downstream_gate") or "pass"
        result["downstream"] = {
            gate: summarize_by_key([r for r in down_rows if r["first_gate"] == gate],
                                   "downstream_gate_label", order=("pass",) + EXECUTOR_GATES)
            for gate in DETECTION_GATES + PATH_GATES + ("end_of_data",)
            if any(r["first_gate"] == gate for r in down_rows)
        }
        result["monthly"] = monthly_table(rows)
        result["significance"] = significance_table(rows, n_perm=int(args.n_perm))
        result["validation"] = {
            "executed_rechecked": len(mismatches),
            "max_abs_net_r_diff": round(max(mismatches), 6) if mismatches else None,
            "n_diff_gt_1e-6": int(sum(1 for m in mismatches if m > 1e-6)),
            "census_mismatch": int(sum(1 for r in rows if r["first_gate"] == "census_mismatch")),
        }
        result["setups"] = rows
        if args.output_csv:
            out_csv = Path(args.output_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            flat = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows])
            flat.to_csv(out_csv, index=False)

        print("\n== 入场漏斗（第一个拒绝的门；R = 固定风险强制开仓、含手续费）==")
        print(f"{'gate':<18}{'setups':>8}{'share%':>8}{'trig':>6}{'netR_sum':>10}{'netR_avg':>10}{'win%':>7}{'TP%':>6}")
        for name, s in result["funnel"]["all"].items():
            avg = f"{s['net_r_mean']:.3f}" if s["net_r_mean"] is not None else "-"
            win = f"{s['win_rate_pct']:.0f}" if s["win_rate_pct"] is not None else "-"
            tp = f"{s['tp_rate_pct']:.0f}" if s["tp_rate_pct"] is not None else "-"
            print(f"{name:<18}{s['setups']:>8}{s['share_pct']:>8.1f}{s['triggered']:>6}{s['net_r_sum']:>10.2f}{avg:>10}{win:>7}{tp:>6}")
        print(f"\n校验：{result['validation']}")

    summary = result["path_run"]
    print(f"\n== live 语义路径回放 [{args.label}] ==")
    print(f"总收益 {summary['total_return_pct']:+.2f}%  最大回撤 {summary['max_drawdown_pct']:.2f}%  "
          f"Sharpe {summary['sharpe_ratio']:.3f}  成交 {summary['trades']}  开仓尝试 {summary['open_attempt_decisions']}")
    print(f"R 空间：{summary['r_space']}")
    print(f"窗口：{summary['windows']}")
    print(f"两半：{summary['halves']}")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
    print(f"\n输出: {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
