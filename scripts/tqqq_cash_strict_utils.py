#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.audit_tqqq_cash_regime_context import build_regime_frame, load_df
from scripts.scan_tqqq_context_bucket_overlays import prepare_frame


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"


def load_strict_frame(
    *,
    data_root: Path = DEFAULT_PUBLIC_DIR,
    entry_fast_window: int,
    entry_slow_window: int,
) -> pd.DataFrame:
    qqq = load_df(data_root / "QQQ-1d.feather")
    tqqq = load_df(data_root / "TQQQ-1d.feather")
    spy = load_df(data_root / "SPY-1d.feather")
    ixic = load_df(data_root / "^IXIC-1d.feather")
    vix = load_df(data_root / "^VIX-1d.feather")
    frame = build_regime_frame(qqq, tqqq, spy, ixic, vix, entry_fast_window, entry_slow_window).copy()
    frame = frame.dropna(subset=["fast_ma", "slow_ma", "tqqq_open", "tqqq_close"]).reset_index(drop=True)
    frame["entry_signal"] = (frame["fast_ma"] > frame["slow_ma"]).astype(int)
    rich = prepare_frame(qqq, tqqq, tqqq, spy, ixic, vix, entry_fast_window, entry_slow_window)
    rich = rich[
        [
            "date",
            "qqq_breakout_fail_20",
            "long_context_score",
            "qqq_dist_slow",
            "qqq_mom_20",
        ]
    ].copy()
    frame = frame.merge(rich, on="date", how="left")
    return frame


def load_strict_frame_with_overlay_context(
    *,
    data_root: Path = DEFAULT_PUBLIC_DIR,
    entry_fast_window: int,
    entry_slow_window: int,
) -> pd.DataFrame:
    frame = load_strict_frame(
        data_root=data_root,
        entry_fast_window=entry_fast_window,
        entry_slow_window=entry_slow_window,
    )
    qqq = load_df(data_root / "QQQ-1d.feather")
    tqqq = load_df(data_root / "TQQQ-1d.feather")
    spy = load_df(data_root / "SPY-1d.feather")
    ixic = load_df(data_root / "^IXIC-1d.feather")
    vix = load_df(data_root / "^VIX-1d.feather")
    rich = prepare_frame(qqq, tqqq, tqqq, spy, ixic, vix, entry_fast_window, entry_slow_window)
    rich = rich[
        [
            "date",
            "qqq_breakout_20",
            "qqq_sweep_reclaim_20",
            "qqq_volume_ratio_20",
            "qqq_compression_60",
        ]
    ].copy()
    merged = frame.merge(rich, on="date", how="left")
    for column in ["qqq_breakout_20", "qqq_sweep_reclaim_20", "qqq_compression_60"]:
        merged[column] = merged[column].fillna(False).astype(bool)
    merged["qqq_volume_ratio_20"] = merged["qqq_volume_ratio_20"].fillna(0.0)
    return merged


def de_risk_fraction(row: pd.Series, signal_name: str) -> float:
    name = str(signal_name or "off").strip()
    if name in {"", "off", "none"}:
        return 1.0
    breakout_fail = bool(row.get("qqq_breakout_fail_20"))
    long_context_score = int(row.get("long_context_score", 0) or 0)
    qqq_dist_slow = float(row.get("qqq_dist_slow", 0.0) or 0.0)
    qqq_mom_20 = float(row.get("qqq_mom_20", 0.0) or 0.0)
    vix_label = str(row.get("vix_label", ""))

    if name == "breakout_fail_score_le3_flat":
        return 0.0 if breakout_fail and long_context_score <= 3 else 1.0
    if name == "breakout_fail_score_le3_half":
        return 0.5 if breakout_fail and long_context_score <= 3 else 1.0
    if name == "breakout_fail_score_le4_flat":
        return 0.0 if breakout_fail and long_context_score <= 4 else 1.0
    if name == "breakout_fail_score_le4_half":
        return 0.5 if breakout_fail and long_context_score <= 4 else 1.0
    if name == "mom20_negative_flat":
        return 0.0 if qqq_mom_20 < 0 else 1.0
    if name == "mom20_negative_half":
        return 0.5 if qqq_mom_20 < 0 else 1.0
    if name == "stretched_vix_normal_or_high_flat":
        return 0.0 if qqq_dist_slow > 0.08 and vix_label in {"vix_normal", "vix_high", "vix_extreme"} else 1.0
    if name == "stretched_vix_normal_or_high_half":
        return 0.5 if qqq_dist_slow > 0.08 and vix_label in {"vix_normal", "vix_high", "vix_extreme"} else 1.0
    raise ValueError(f"Unsupported de-risk signal: {signal_name}")


def build_allow_mask(frame: pd.DataFrame, regime_filter: str) -> pd.Series:
    vix_allow = frame["vix_label"].isin(["vix_low", "vix_normal"])
    ixic_allow = frame["ixic_trend_label"].eq("ixic_up")
    rel_allow = frame["rel_strength_label"].ne("qqq_weak")

    if regime_filter == "base":
        return pd.Series(True, index=frame.index)
    if regime_filter == "vix_filter":
        return vix_allow
    if regime_filter == "ixic_filter":
        return ixic_allow
    if regime_filter == "rel_filter":
        return rel_allow
    if regime_filter == "vix_ixic":
        return vix_allow & ixic_allow
    if regime_filter == "vix_rel":
        return vix_allow & rel_allow
    if regime_filter == "ixic_rel":
        return ixic_allow & rel_allow
    if regime_filter == "all_three":
        return vix_allow & ixic_allow & rel_allow
    raise ValueError(f"Unsupported regime_filter: {regime_filter}")


def build_recovery_mask(frame: pd.DataFrame, rule_name: str) -> pd.Series:
    if str(rule_name or "off").strip() in {"", "off", "none"}:
        return pd.Series(False, index=frame.index)
    score = frame["long_context_score"].fillna(0).astype(int)
    breakout = frame.get("qqq_breakout_20", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    reclaim = frame.get("qqq_sweep_reclaim_20", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    rel_strong = frame["rel_strength_label"].eq("qqq_strong")
    vix_not_high = frame["vix_label"].isin(["vix_low", "vix_normal"])
    mom_pos = frame["qqq_mom_20"].fillna(0.0) > 0

    mapping: dict[str, pd.Series] = {
        "score_ge3": score >= 3,
        "score_ge4": score >= 4,
        "score_ge5": score >= 5,
        "score_ge3_breakout20": (score >= 3) & breakout,
        "score_ge4_breakout20": (score >= 4) & breakout,
        "score_ge5_breakout20": (score >= 5) & breakout,
        "score_ge3_breakout_or_reclaim20": (score >= 3) & (breakout | reclaim),
        "score_ge4_breakout_or_reclaim20": (score >= 4) & (breakout | reclaim),
        "score_ge5_breakout_or_reclaim20": (score >= 5) & (breakout | reclaim),
        "rel_strong_breakout20": rel_strong & breakout,
        "rel_strong_breakout_or_reclaim20": rel_strong & (breakout | reclaim),
        "vix_not_high_rel_strong_breakout_or_reclaim20": vix_not_high & rel_strong & (breakout | reclaim),
        "score_ge4_breakout_or_reclaim20_mom_pos": (score >= 4) & (breakout | reclaim) & mom_pos,
    }
    if rule_name not in mapping:
        raise ValueError(f"Unsupported recovery re-entry rule: {rule_name}")
    return mapping[rule_name].fillna(False)


def _ladder_steps(name: str) -> list[tuple[float, float]]:
    if name == "two_equal":
        return [(0.0, 0.5), (5.0, 0.5)]
    if name == "three_40_30_30":
        return [(0.0, 0.4), (5.0, 0.3), (10.0, 0.3)]
    if name == "three_50_30_20":
        return [(0.0, 0.5), (5.0, 0.3), (10.0, 0.2)]
    raise ValueError(f"Unsupported drawdown ladder scheme: {name}")


def _vix_ok(label: str, rule: str) -> bool:
    if rule == "all":
        return True
    if rule == "vix_low_normal":
        return label in {"vix_low", "vix_normal"}
    if rule == "not_extreme":
        return label != "vix_extreme"
    raise ValueError(f"Unsupported drawdown ladder vix rule: {rule}")


def recovery_overlay_trigger(
    frame: pd.DataFrame,
    idx: int,
    *,
    recovery_rule_name: str,
    recovery_cooldown_days: int,
    recovery_cooldown_left: int,
    allow_mask: pd.Series,
    recovery_mask: pd.Series | None = None,
) -> bool:
    if idx <= 0 or recovery_cooldown_left > 0:
        return False
    if str(recovery_rule_name or "off").strip() in {"", "off", "none"}:
        return False
    prev_signal_on = bool(int(frame.iloc[idx - 1]["entry_signal"]) > 0)
    prev_allow = bool(allow_mask.iloc[idx - 1])
    if not prev_signal_on or prev_allow:
        return False
    mask = recovery_mask if recovery_mask is not None else build_recovery_mask(frame, recovery_rule_name).reset_index(drop=True)
    return bool(mask.iloc[idx - 1])


def drawdown_overlay_trigger(
    frame: pd.DataFrame,
    idx: int,
    *,
    allow_mask: pd.Series,
    drawdown_source: str,
    drawdown_threshold_pct: float,
    peak_lookback_days: int,
    ladder_scheme: str,
    vix_rule: str,
    drawdown_series: pd.Series | None = None,
) -> tuple[bool, float]:
    if idx <= 0 or float(drawdown_threshold_pct) <= 0:
        return False, 0.0
    prev_signal_on = bool(int(frame.iloc[idx - 1]["entry_signal"]) > 0)
    prev_allow = bool(allow_mask.iloc[idx - 1])
    if not prev_signal_on or prev_allow:
        return False, 0.0
    if drawdown_series is None:
        source_column = "tqqq_close" if drawdown_source == "tqqq" else "qqq_close"
        rolling_peak_ref = frame[source_column].rolling(int(peak_lookback_days)).max().shift(1)
        drawdown_series = (rolling_peak_ref - frame[source_column]) / rolling_peak_ref * 100.0
    value = drawdown_series.iloc[idx - 1]
    if pd.isna(value):
        return False, 0.0
    peak_dd = float(value)
    label = str(frame.iloc[idx - 1]["vix_label"])
    if peak_dd < float(drawdown_threshold_pct) or not _vix_ok(label, vix_rule):
        return False, 0.0
    extra_dd = peak_dd - float(drawdown_threshold_pct)
    allocation = 0.0
    candidate_step_idx = -1
    for i, (step_extra_dd, step_alloc) in enumerate(_ladder_steps(ladder_scheme)):
        if extra_dd >= float(step_extra_dd):
            candidate_step_idx = i
            allocation += float(step_alloc)
    return candidate_step_idx >= 0, allocation


def strict_overlay_trade_stats(path: pd.DataFrame) -> dict[str, Any]:
    if path.empty:
        return {
            "trades_closed": 0,
            "win_rate_pct": 0.0,
            "avg_hold_days": 0.0,
            "median_hold_days": 0.0,
            "avg_trade_return_pct": 0.0,
            "median_trade_return_pct": 0.0,
            "overlay_entries": 0,
        }
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    prev_capital = None
    for row in path.itertuples(index=False):
        if bool(row.entered_today):
            current = {
                "entry_date": pd.Timestamp(row.date),
                "entry_type": getattr(row, "entry_type", "base"),
                "entry_capital": float(prev_capital if prev_capital is not None else row.capital),
                "hold_days": 0,
            }
        if current is not None and str(row.position) == "TQQQ":
            current["hold_days"] += 1
        if bool(row.exited_today) and current is not None:
            exit_capital = float(row.capital)
            trade_return_pct = (exit_capital / float(current["entry_capital"]) - 1.0) * 100.0 if float(current["entry_capital"]) > 0 else 0.0
            entries.append(
                {
                    "entry_date": str(current["entry_date"].date()),
                    "exit_date": str(pd.Timestamp(row.date).date()),
                    "entry_type": current["entry_type"],
                    "trade_return_pct": round(trade_return_pct, 2),
                    "hold_days": int(current["hold_days"]),
                }
            )
            current = None
        prev_capital = float(row.capital)
    if not entries:
        return {
            "trades_closed": 0,
            "win_rate_pct": 0.0,
            "avg_hold_days": 0.0,
            "median_hold_days": 0.0,
            "avg_trade_return_pct": 0.0,
            "median_trade_return_pct": 0.0,
            "overlay_entries": 0,
        }
    trades = pd.DataFrame(entries)
    wins = (trades["trade_return_pct"] > 0).sum()
    return {
        "trades_closed": int(len(trades)),
        "win_rate_pct": round(float(wins) / len(trades) * 100.0, 2),
        "avg_hold_days": round(float(trades["hold_days"].mean()), 2),
        "median_hold_days": round(float(trades["hold_days"].median()), 2),
        "avg_trade_return_pct": round(float(trades["trade_return_pct"].mean()), 2),
        "median_trade_return_pct": round(float(trades["trade_return_pct"].median()), 2),
        "overlay_entries": int((trades["entry_type"] != "base").sum()),
    }


def annual_returns(equity: pd.DataFrame) -> dict[str, float]:
    if equity.empty:
        return {}
    frame = equity.copy()
    frame["year"] = frame["date"].dt.year.astype(str)
    out: dict[str, float] = {}
    for year, group in frame.groupby("year"):
        start = float(group.iloc[0]["equity"])
        end = float(group.iloc[-1]["equity"])
        out[year] = round((end / start - 1.0) * 100.0, 2) if start > 0 else 0.0
    return out


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0, pd.NA) * 100.0
    return float(dd.max(skipna=True) or 0.0)


def run_strict_candidate(
    frame: pd.DataFrame,
    *,
    regime_filter: str,
    max_hold_days: int,
    trailing_lookback_days: int,
    trailing_drawdown_pct: float,
    switch_cost_bps: float,
    initial_capital: float,
    hold_mode: str = "hard_exit",
    de_risk_signal_name: str = "off",
    recovery_reentry_rule: str = "off",
    recovery_reentry_cooldown_days: int = 0,
    drawdown_ladder_enabled: bool = False,
    drawdown_ladder_source: str = "tqqq",
    drawdown_ladder_threshold_pct: float = 0.0,
    drawdown_ladder_peak_lookback_days: int = 90,
    drawdown_ladder_scheme: str = "two_equal",
    drawdown_ladder_vix_rule: str = "all",
    drawdown_ladder_rebound_exit_pct: float = 10.0,
    drawdown_ladder_max_hold_days: int = 15,
) -> dict[str, Any]:
    allow_mask = build_allow_mask(frame, regime_filter)
    capital = float(initial_capital)
    holding = False
    pending_exit = False
    exit_override = False
    hold_days = 0
    rolling_peak = 0.0
    previous_close = 0.0
    recovery_mask = build_recovery_mask(frame, recovery_reentry_rule).reset_index(drop=True)
    recovery_cooldown_left = 0
    overlay_mode = False
    overlay_entry_type = "base"
    overlay_allocation = 1.0
    overlay_days = 0
    overlay_entry_close = 0.0
    overlay_exit_reason = ""
    drawdown_series = None
    if drawdown_ladder_enabled and float(drawdown_ladder_threshold_pct) > 0:
        source_column = "tqqq_close" if drawdown_ladder_source == "tqqq" else "qqq_close"
        rolling_peak_ref = frame[source_column].rolling(int(drawdown_ladder_peak_lookback_days)).max().shift(1)
        drawdown_series = (rolling_peak_ref - frame[source_column]) / rolling_peak_ref * 100.0
    rows: list[dict[str, Any]] = []

    for idx, row in frame.iterrows():
        start_capital = capital
        desired_today = False
        recovery_trigger = recovery_overlay_trigger(
            frame,
            idx,
            recovery_rule_name=recovery_reentry_rule,
            recovery_cooldown_days=recovery_reentry_cooldown_days,
            recovery_cooldown_left=recovery_cooldown_left,
            allow_mask=allow_mask,
            recovery_mask=recovery_mask,
        )
        ladder_trigger, ladder_allocation = drawdown_overlay_trigger(
            frame,
            idx,
            allow_mask=allow_mask,
            drawdown_source=drawdown_ladder_source,
            drawdown_threshold_pct=float(drawdown_ladder_threshold_pct),
            peak_lookback_days=int(drawdown_ladder_peak_lookback_days),
            ladder_scheme=drawdown_ladder_scheme,
            vix_rule=drawdown_ladder_vix_rule,
            drawdown_series=drawdown_series,
        )
        if idx > 0:
            base_desired_today = bool(int(frame.iloc[idx - 1]["entry_signal"]) > 0 and bool(allow_mask.iloc[idx - 1]))
            desired_today = base_desired_today or recovery_trigger or ladder_trigger or (overlay_mode and bool(int(frame.iloc[idx - 1]["entry_signal"]) > 0))
        else:
            base_desired_today = False

        if exit_override and not desired_today:
            exit_override = False

        if holding and previous_close > 0:
            capital *= float(row["tqqq_open"]) / previous_close

        action_cost = 0.0
        entered_today = False
        exited_today = False

        if holding and (pending_exit or not desired_today):
            action_cost += float(switch_cost_bps) / 10000.0
            capital *= 1.0 - float(switch_cost_bps) / 10000.0
            holding = False
            pending_exit = False
            hold_days = 0
            rolling_peak = 0.0
            exited_today = True
            if overlay_mode and recovery_reentry_cooldown_days > 0:
                recovery_cooldown_left = int(recovery_reentry_cooldown_days)
            overlay_mode = False
            overlay_entry_type = "base"
            overlay_allocation = 1.0
            overlay_days = 0
            overlay_entry_close = 0.0
            overlay_exit_reason = ""

        if (not holding) and desired_today and not exit_override:
            action_cost += float(switch_cost_bps) / 10000.0
            capital *= 1.0 - float(switch_cost_bps) / 10000.0
            holding = True
            hold_days = 0
            rolling_peak = float(row["tqqq_open"])
            entered_today = True
            if ladder_trigger and not base_desired_today:
                overlay_mode = True
                overlay_entry_type = "drawdown_ladder"
                overlay_allocation = float(ladder_allocation)
                overlay_days = 0
                overlay_entry_close = float(frame.iloc[idx - 1]["qqq_close"]) if idx > 0 else float(row["qqq_close"])
            elif recovery_trigger and not base_desired_today:
                overlay_mode = True
                overlay_entry_type = "recovery_reentry"
                overlay_allocation = 1.0
                overlay_days = 0
                overlay_entry_close = 0.0
            else:
                overlay_mode = False
                overlay_entry_type = "base"
                overlay_allocation = 1.0
                overlay_days = 0
                overlay_entry_close = 0.0
        elif (not holding) and recovery_cooldown_left > 0:
            recovery_cooldown_left -= 1

        trailing_exit = False
        time_exit = False
        leverage = 0.0
        if holding:
            base_fraction = de_risk_fraction(frame.iloc[idx - 1], de_risk_signal_name) if idx > 0 else 1.0
            leverage = float(base_fraction) * float(overlay_allocation if overlay_mode else 1.0)
            open_price = float(row["tqqq_open"])
            close_price = float(row["tqqq_close"])
            if open_price > 0:
                capital *= 1.0 + leverage * (close_price / open_price - 1.0)
            hold_days += 1
            rolling_peak = max(rolling_peak, close_price)
            if base_desired_today and overlay_mode and overlay_entry_type != "drawdown_ladder":
                overlay_mode = False
                overlay_entry_type = "base"
                overlay_allocation = 1.0
                overlay_days = 0
                overlay_entry_close = 0.0
            if trailing_lookback_days > 0 and trailing_drawdown_pct > 0 and hold_days >= trailing_lookback_days and rolling_peak > 0:
                drawdown_from_peak = (rolling_peak - close_price) / rolling_peak * 100.0
                trailing_exit = drawdown_from_peak >= trailing_drawdown_pct
            if max_hold_days > 0 and hold_days >= max_hold_days:
                time_exit = True
            if overlay_mode and overlay_entry_type == "drawdown_ladder":
                overlay_days += 1
                if overlay_entry_close > 0:
                    rebound_pct = (float(row["qqq_close"]) / overlay_entry_close - 1.0) * 100.0
                    if rebound_pct >= float(drawdown_ladder_rebound_exit_pct):
                        pending_exit = True
                        overlay_exit_reason = "drawdown_rebound"
                        if hold_mode == "hard_exit":
                            exit_override = True
                if not pending_exit and overlay_days >= int(drawdown_ladder_max_hold_days):
                    pending_exit = True
                    overlay_exit_reason = "drawdown_time"
                    if hold_mode == "hard_exit":
                        exit_override = True
            if trailing_exit or time_exit:
                pending_exit = True
                overlay_exit_reason = "trailing" if trailing_exit else "time"
                if hold_mode == "hard_exit":
                    exit_override = True

        previous_close = float(row["tqqq_close"])
        position = "TQQQ" if holding else "CASH"
        daily_return = capital / start_capital - 1.0 if start_capital > 0 else 0.0
        rows.append(
            {
                "date": pd.Timestamp(row["date"]),
                "position": position,
                "daily_return": float(daily_return),
                "capital": float(capital),
                "entered_today": bool(entered_today),
                "exited_today": bool(exited_today),
                "pending_exit": bool(pending_exit),
                "trailing_exit": bool(trailing_exit),
                "time_exit": bool(time_exit),
                "allow_today": bool(allow_mask.iloc[idx]),
                "vix_label": str(row["vix_label"]),
                "ixic_trend_label": str(row["ixic_trend_label"]),
                "rel_strength_label": str(row["rel_strength_label"]),
                "entry_type": overlay_entry_type if holding else "none",
                "overlay_mode": bool(overlay_mode),
                "overlay_allocation": float(overlay_allocation),
                "overlay_days": int(overlay_days),
                "overlay_exit_reason": overlay_exit_reason,
                "leverage": float(leverage),
                "action_cost": float(action_cost),
            }
        )

    path = pd.DataFrame(rows)
    equity = path[["date", "capital"]].rename(columns={"capital": "equity"})
    yearly = annual_returns(equity)
    total_return_pct = round((float(path.iloc[-1]["capital"]) / float(initial_capital) - 1.0) * 100.0, 2) if not path.empty else 0.0
    max_dd = round(max_drawdown_pct(path["capital"]), 2) if not path.empty else 0.0
    trades = int(path["entered_today"].sum()) if not path.empty else 0
    invested_days = int((path["position"] == "TQQQ").sum()) if not path.empty else 0
    sharpe_like = round((path["daily_return"].mean() / path["daily_return"].std()) if len(path) > 1 and path["daily_return"].std() > 0 else 0.0, 3)
    trade_stats = strict_overlay_trade_stats(path)
    score = round(
        total_return_pct
        - max_dd * 1.75
        + float(yearly.get("2022", 0.0)) * 1.2
        + float(yearly.get("2023", 0.0)) * 0.6
        + float(yearly.get("2024", 0.0)) * 0.7
        + float(yearly.get("2025", 0.0)) * 0.8
        + float(yearly.get("2026", 0.0)) * 1.2,
        4,
    )
    return {
        "summary": {
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd,
            "yearly_returns_pct": yearly,
            "trades": trades,
            "invested_days": invested_days,
            "invested_ratio_pct": round(invested_days / len(path) * 100.0, 2) if len(path) else 0.0,
            "sharpe_like": sharpe_like,
            "score": score,
            "latest_position": str(path.iloc[-1]["position"]) if not path.empty else "CASH",
            "closed_trades": int(trade_stats["trades_closed"]),
            "win_rate_pct": float(trade_stats["win_rate_pct"]),
            "avg_hold_days": float(trade_stats["avg_hold_days"]),
            "median_hold_days": float(trade_stats["median_hold_days"]),
            "avg_trade_return_pct": float(trade_stats["avg_trade_return_pct"]),
            "median_trade_return_pct": float(trade_stats["median_trade_return_pct"]),
            "overlay_entries": int(trade_stats["overlay_entries"]),
        },
        "path": path,
    }


def load_strict_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text())
