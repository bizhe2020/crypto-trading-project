#!/usr/bin/env python3
"""QQQ↔GOOGL router vs 单 GOOGL / 单 QQQ 回测比较。

窗口: 2026-03-04 → 2026-08-07
  - QQQ-USDT-SWAP 上市 2026-03-04（数据起点）
  - GOOGL 4h 本地数据末端 2026-08-07

方法:
- GOOGL-only : run_googl_4h_replay（shadow gate 15%/20bar + reentry clear 2 + 杠杆爬坡
  ramp_confirm 0.5% / pre-stop 2%，与实盘执行器一致）。
- QQQ-only   : build_qqq_usdt_leveraged_path（固定 10x + macro overlay，无 shadow gate）。
- Router     : 日频组合两条策略的日收益流。每日用 live adapter 的 route_score 选边：
  qqq_min_route_score=96 / googl_min_route_score=0 / switch_advantage=6，QQQ↔GOOGL 切换
  收 switch_cost（敏感性），24h 冷却在日频下天然满足（相邻决策间隔 ≥ 1 天）。

已知限制（诚实标注）:
1. 两条策略独立回测，再按日收益流拼接；切换瞬间强制平旧开新的边际手续费没有内嵌在
   策略自身收益里，故用 switch_cost 参数模拟。真实单边 ≈ taker(5bps)+slippage(5bps)
   × 杠杆(≈10x) ≈ 100bps/边，往返 ≈ 200bps。
2. GOOGL 路径带 equity_dd shadow gate（观测 GOOGL 自身权益回撤）；QQQ 路径无 shadow
   gate。实盘两者都过 router 的 risk gate，回测里存在不对称。
3. 日频路由按"当日实际持仓"作为 active 判定（与既有 run_router 约定一致），
   存在轻微 day-internal look-ahead，对日频比较影响很小。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_proxy_strategy_router import build_qqq_usdt_leveraged_path, max_drawdown_pct  # noqa: E402
from scripts.replay_googl_usdt_4h import run_googl_4h_replay, load_funding as load_googl_funding, attach_googl_daily_state  # noqa: E402
from scripts.scan_qqq_usdt_4h_triggers import load_signal_path, load_okx_4h, attach_daily_state  # noqa: E402
from scripts.scan_qqq_usdt_btc_logic_overlays import enrich_bars  # noqa: E402
from scripts.replay_qqq_usdt_10x import load_funding as load_qqq_funding  # noqa: E402
from bot.qqq_usdt_signal_adapter import QqqUsdtSignalAdapter  # noqa: E402
from bot.googl_usdt_signal_adapter import GooglUsdtSignalAdapter  # noqa: E402

QQQ_RUNTIME = ROOT / "config" / "config.paper.qqq-usdt-aggressive-runtime.json"
QQQ_FROZEN = ROOT / "config" / "config.paper.tqqq-only-strict-recovery-frozen.json"
GOOGL_RUNTIME = ROOT / "config" / "config.paper.googl-high-leverage-runtime.json"
GOOGL_SIGNAL_CSV = ROOT / "var" / "runtime" / "googl" / "googl_daily_signal.csv"
GOOGL_4H = ROOT / "data" / "okx" / "futures" / "GOOGL_USDT_USDT-4h-futures.feather"

QQQ_4H = Path("/tmp/QQQ_merged_4h.feather")
QQQ_FUNDING = Path("/tmp/QQQ_combined_funding.feather")
GOOGL_FUNDING = Path("/tmp/GOOGL_combined_funding.feather")

START = pd.Timestamp("2026-03-04", tz="UTC")
END = pd.Timestamp("2026-08-07", tz="UTC")
INITIAL_CAPITAL = 1000.0

GOOGL_TIERS = {"offense": 11.2, "base": 7.5, "defense": 3.8, "flat": 0.0}


def build_googl_daily(gcfg: dict[str, Any], replay_overrides: dict[str, Any] | None = None) -> pd.DataFrame:
    """跑 GOOGL 4h 执行层回测，输出日频帧（return / active / score）。

    replay_overrides: 透传给 run_googl_4h_replay 的参数覆盖（如
    {"ramp_pre_stop_pct": 1.0}），用于策略旋钮扫描；None = 全部走 gcfg 基线。
    """
    googl_4h = pd.read_feather(GOOGL_4H)
    googl_4h["date"] = pd.to_datetime(googl_4h["date"], utc=True)
    googl_4h = googl_4h.sort_values("date").reset_index(drop=True)
    funding = load_googl_funding(GOOGL_FUNDING)
    gsig_raw = pd.read_csv(GOOGL_SIGNAL_CSV)
    gsig_raw["date"] = pd.to_datetime(gsig_raw["date"], utc=True)
    merged = attach_googl_daily_state(googl_4h, gsig_raw)

    replay_kwargs: dict[str, Any] = dict(
        stop_loss_pct=float(gcfg.get("stop_loss_pct", 4.0)),
        taker_fee_rate=float(gcfg.get("taker_fee_rate", 0.0005)),
        slippage_bps=float(gcfg.get("slippage_bps", 5.0)),
        initial_capital=float(gcfg.get("initial_capital", 1000.0)),
        equity_dd_stop_pct=float(gcfg.get("shadow_gate_replay_profile", {}).get("equity_dd_stop_pct", 15.0)),
        equity_dd_cooldown_bars=int(gcfg.get("shadow_gate_replay_profile", {}).get("equity_dd_cooldown_bars", 20)),
        reentry_rule=str(gcfg.get("shadow_gate_replay_profile", {}).get("reentry_rule", "clear")),
        reentry_clear_bars=int(gcfg.get("shadow_gate_replay_profile", {}).get("reentry_clear_bars", 2)),
        include_funding=True,
        capture_open_gaps=True,
        entry_price_col=None,
        ramp_confirm_pct=float(gcfg.get("ramp_confirm_pct", 0.5)),
        ramp_stop_pct=0.0,
        be_lock_pct=0.0,
        ramp_pre_stop_pct=float(gcfg.get("ramp_pre_stop_pct", 2.0)),
    )
    if replay_overrides:
        replay_kwargs.update(replay_overrides)
    result = run_googl_4h_replay(merged, funding, leverage_tiers=GOOGL_TIERS, **replay_kwargs)
    path = result["path"].copy()
    path["day"] = path["date"].dt.floor("D")

    daily = pd.DataFrame(
        {
            "date": path.groupby("day")["date"].last().dt.floor("D"),
            "googl_return": path.groupby("day")["daily_return"].apply(lambda s: float((1.0 + s).prod() - 1.0)),
            # 当日是否持有过仓位（含入场/离场当天）。纯 shadow-gate 冷却日（无持仓）为 False。
            "googl_active": (
                path.groupby("day")["holding"].any()
                | path.groupby("day")["entered_today"].any()
                | path.groupby("day")["exited_today"].any()
            ),
            "googl_capital": path.groupby("day")["capital"].last(),
        }
    ).reset_index(drop=True)

    # 信号侧（live adapter 评分输入）
    gsig = pd.read_csv(GOOGL_SIGNAL_CSV)
    gsig["date"] = pd.to_datetime(gsig["date"], utc=True)
    gsig = gsig[["date", "position", "berkshire_conviction", "leverage_tier", "target_leverage"]].sort_values("date")
    daily = daily.merge(gsig, on="date", how="left")

    def _score(row: pd.Series) -> float:
        # live adapter 只在候选激活（position=="GOOGL"）时给分；FLAT 行记 0，
        # 由 ffill 把上一段 GOOGL score 带到离场日。
        if str(row["position"]) != "GOOGL":
            return 0.0
        tier = str(row["leverage_tier"] or "base")
        lev = float(row["target_leverage"] or GOOGL_TIERS.get(tier, GOOGL_TIERS["base"]))
        score = 40.0 + lev * 4.0
        if tier == "offense":
            score += 15.0
        elif tier == "base":
            score += 5.0
        if bool(row["berkshire_conviction"]):
            score += 8.0
        return round(score, 2)

    # 只对"当日信号为 GOOGL"的日子给 live score；离场日（信号翻 FLAT）前向继承上一段的
    # GOOGL score，使 router 的 incumbent-hold 在离场日成立（当日起决策时 GOOGL 仍可持有）。
    sig_score = daily.apply(_score, axis=1)
    daily["googl_score"] = sig_score.where(sig_score > 0.0).ffill().fillna(0.0)
    daily.loc[~daily["googl_active"], "googl_score"] = 0.0
    daily = daily[(daily["date"] >= START) & (daily["date"] <= END)].reset_index(drop=True)
    return daily


def build_qqq_daily(qcfg: dict[str, Any]) -> pd.DataFrame:
    """跑 QQQ/USDT 杠杆路径回测，用 live adapter 的 route_score 覆盖 proxy 分。"""
    qqq_path, meta = build_qqq_usdt_leveraged_path(
        config_path=QQQ_RUNTIME,
        initial_capital=INITIAL_CAPITAL,
        start=START,
        end=END,
        data_4h_path=QQQ_4H,
        funding_path=QQQ_FUNDING,
        risk_overlay_enabled=False,
    )
    daily = qqq_path[["date", "qqq_return", "qqq_active", "avg_leverage_when_active"]].copy()

    # live adapter 评分：最后一根已收盘 4h bar 的 enrich 状态
    _, qqq_signal = load_signal_path(QQQ_FROZEN)
    qqq_bars = QqqUsdtSignalAdapter._attach_daily_columns(
        enrich_bars(attach_daily_state(load_okx_4h(QQQ_4H), qqq_signal, trim_to_signal_end=False)),
        qqq_signal,
        trim_to_signal_end=False,
    )
    qqq_bars = qqq_bars[(qqq_bars["date"] >= START) & (qqq_bars["date"] <= END)].copy()
    qqq_bars["day"] = qqq_bars["date"].dt.floor("D")
    lev_profile = QqqUsdtSignalAdapter._leverage_profile(qcfg)

    scores: dict[pd.Timestamp, float] = {}
    for day, group in qqq_bars.groupby("day"):
        # 取当日最后一根 allow_long 的 bar（持有日=全天，离场日=翻转前的最后一根），
        # 纯 CASH 日无 allow_long bar → score 0。
        hold_bars = group[group["allow_long"]]
        if hold_bars.empty:
            scores[day] = 0.0
        else:
            latest = hold_bars.iloc[-1]
            lev_now, _ = QqqUsdtSignalAdapter._current_leverage(lev_profile, latest)
            scores[day] = QqqUsdtSignalAdapter._route_score(latest, lev_now)

    daily["qqq_score"] = daily["date"].map(scores).fillna(0.0)
    return daily.reset_index(drop=True)


def run_router(
    merged: pd.DataFrame,
    *,
    switch_cost_bps: float,
    qqq_min_score: float = 96.0,
    googl_min_score: float = 0.0,
    switch_advantage: float = 6.0,
) -> pd.DataFrame:
    """日频 QQQ↔GOOGL 路由。镜像既有 run_router 的选择语义 + 24h 切换冷却。"""
    capital = float(INITIAL_CAPITAL)
    current = "CASH"
    last_switch_at: pd.Timestamp | None = None
    rows: list[dict[str, Any]] = []

    for row in merged.itertuples(index=False):
        date = pd.Timestamp(row.date)
        previous = current
        selected = current
        reason = "hold"

        # 候选: active + 分数
        qqq_active = bool(row.qqq_active) and float(row.qqq_score) > 0.0
        googl_active = bool(row.googl_active) and float(row.googl_score) > 0.0

        # 24h 冷却：QQQ↔GOOGL 切换后 86400s 内不允许再切
        cooldown_blocked = last_switch_at is not None and (date - last_switch_at) < pd.Timedelta(seconds=86400)

        active_scores: dict[str, float] = {}
        if qqq_active:
            active_scores["QQQ"] = float(row.qqq_score)
        if googl_active:
            active_scores["GOOGL"] = float(row.googl_score)

        if not cooldown_blocked:
            current_score = active_scores.get(current)
            if current_score is not None:
                # incumbent 在市：只允许挑战者超过 switch_advantage 时切换
                challengers: list[tuple[str, float]] = []
                if qqq_active and current != "QQQ" and float(row.qqq_score) >= qqq_min_score:
                    challengers.append(("QQQ", float(row.qqq_score)))
                if googl_active and current != "GOOGL" and float(row.googl_score) >= googl_min_score:
                    challengers.append(("GOOGL", float(row.googl_score)))
                if challengers:
                    challengers.sort(key=lambda x: x[1], reverse=True)
                    best, best_score = challengers[0]
                    if best_score - current_score >= switch_advantage:
                        selected = best
                        reason = "best_route_score"
                    else:
                        reason = "hold_current_hysteresis"
                else:
                    reason = "hold_current_no_challenger"
            else:
                # 当前不在市（或已不在 active）→ 从合格候选中选最高分
                candidates: list[tuple[str, float]] = []
                if qqq_active and float(row.qqq_score) >= qqq_min_score:
                    candidates.append(("QQQ", float(row.qqq_score)))
                if googl_active and float(row.googl_score) >= googl_min_score:
                    candidates.append(("GOOGL", float(row.googl_score)))
                if candidates:
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    selected = candidates[0][0]
                    reason = "best_route_score"
                else:
                    selected, reason = "CASH", "no_eligible_candidates"
        else:
            reason = "hold_switch_cooldown"

        # 收益
        if selected == "QQQ":
            sel_return = float(row.qqq_return)
        elif selected == "GOOGL":
            sel_return = float(row.googl_return)
        else:
            sel_return = 0.0

        switched = selected != previous and selected != "CASH" and previous != "CASH"
        if switched and switch_cost_bps > 0.0:
            capital *= 1.0 - switch_cost_bps / 10000.0
            last_switch_at = date
        capital *= 1.0 + sel_return
        current = selected

        rows.append(
            {
                "date": date,
                "selected": selected,
                "reason": reason,
                "router_return": sel_return,
                "router_equity": capital,
                "switched": switched,
                "qqq_score": float(row.qqq_score),
                "googl_score": float(row.googl_score),
            }
        )
    return pd.DataFrame(rows)


def summarize(label: str, equity: pd.Series) -> dict[str, Any]:
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0 if len(equity) > 1 else 0.0
    n_days = max(int(len(equity)), 1)
    cagr = ((1.0 + total / 100.0) ** (365.25 / n_days) - 1.0) * 100.0 if total > -100.0 else float("nan")
    return {
        "label": label,
        "total_return_pct": round(total, 2),
        "cagr_pct": round(cagr, 2),
        "max_dd_pct": round(max_drawdown_pct(equity), 2),
        "days": n_days,
    }


def main() -> None:
    pd.set_option("future.no_silent_downcasting", True)
    gcfg = json.loads(GOOGL_RUNTIME.read_text())
    qcfg = json.loads(QQQ_RUNTIME.read_text())

    print(f"窗口: {START.date()} → {END.date()}（{int((END - START).days)} 天）")
    print("=" * 78)

    googl_daily = build_googl_daily(gcfg)
    qqq_daily = build_qqq_daily(qcfg)
    # 全日期轴 outer 对齐：GOOGL 只交易日有 bar，周末/假日补 0 收益 + 不活跃
    full_dates = pd.date_range(START.normalize(), END.normalize(), freq="D", tz="UTC")
    merged = (
        pd.DataFrame({"date": full_dates})
        .merge(qqq_daily, on="date", how="left")
        .merge(googl_daily, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )
    for col in ("qqq_return", "googl_return"):
        merged[col] = merged[col].fillna(0.0)
    for col in ("qqq_active", "googl_active"):
        merged[col] = merged[col].fillna(False).astype(bool)
    merged["qqq_score"] = merged["qqq_score"].fillna(0.0)
    merged["googl_score"] = merged["googl_score"].fillna(0.0)
    print(f"日频合并帧: {len(merged)} 天")
    print(f"GOOGL active 天数: {int(merged['googl_active'].sum())} | QQQ active 天数: {int(merged['qqq_active'].sum())}")
    print(f"GOOGL score 范围: {merged.loc[merged['googl_active'],'googl_score'].min()}-{merged.loc[merged['googl_active'],'googl_score'].max()}")
    print(f"QQQ score 范围: {merged.loc[merged['qqq_active'],'qqq_score'].min()}-{merged.loc[merged['qqq_active'],'qqq_score'].max()}")
    print("-" * 78)

    # 单策略
    googl_eq = (1.0 + merged["googl_return"]).cumprod() * INITIAL_CAPITAL
    qqq_eq = (1.0 + merged["qqq_return"]).cumprod() * INITIAL_CAPITAL
    rows = [
        summarize("单 GOOGL", googl_eq),
        summarize("单 QQQ", qqq_eq),
    ]

    # Router 敏感性
    for cost in (0.0, 100.0, 200.0):
        r = run_router(merged, switch_cost_bps=cost)
        eq = r["router_equity"]
        s = summarize(f"Router (switch_cost={cost:g}bps)", eq)
        s["switches"] = int(r["switched"].sum())
        s["days_in_qqq"] = int((r["selected"] == "QQQ").sum())
        s["days_in_googl"] = int((r["selected"] == "GOOGL").sum())
        s["days_cash"] = int((r["selected"] == "CASH").sum())
        rows.append(s)

    # 打印
    header = f"{'策略':<28}{'总收益%':>10}{'年化%':>10}{'最大回撤%':>10}{'切换次数':>8}{'QQQ天':>7}{'GOOGL天':>8}"
    print(header)
    print("-" * len(header))
    for s in rows:
        if "switches" in s:
            print(
                f"{s['label']:<28}{s['total_return_pct']:>10.2f}{s['cagr_pct']:>10.2f}"
                f"{s['max_dd_pct']:>10.2f}{s['switches']:>8}{s['days_in_qqq']:>7}{s['days_in_googl']:>8}"
            )
        else:
            print(
                f"{s['label']:<28}{s['total_return_pct']:>10.2f}{s['cagr_pct']:>10.2f}{s['max_dd_pct']:>10.2f}"
            )

    # GOOGL-only 汇总 + 单策略摘要
    print("-" * 78)
    sig_counts = merged.loc[merged["position"].isin(["GOOGL", "FLAT"]), "position"].value_counts().to_dict()
    print(f"GOOGL 信号日分布: {sig_counts}")


if __name__ == "__main__":
    main()
