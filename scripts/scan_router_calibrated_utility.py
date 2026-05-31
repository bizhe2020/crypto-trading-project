#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_proxy_strategy_router import max_drawdown_pct  # noqa: E402


DEFAULT_LONG_CSV = ROOT / "var" / "reports" / "router_fair_trend_qmin96_switch6_nq_continuous_20260531.csv"
DEFAULT_REAL_CSV = ROOT / "var" / "reports" / "router_fair_trend_qmin96_switch6_real_overlap_20260531.csv"
DEFAULT_OUTPUT_JSON = ROOT / "var" / "reports" / "router_calibrated_utility_scan_20260531.json"
DEFAULT_OUTPUT_CSV = ROOT / "var" / "reports" / "router_calibrated_utility_scan_20260531.csv"
DEFAULT_OUTPUT_MD = ROOT / "var" / "reports" / "router_calibrated_utility_scan_20260531.md"
DEFAULT_BEST_PATH_CSV = ROOT / "var" / "reports" / "router_calibrated_utility_best_path_20260531.csv"


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def load_router_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    for leg in ["btc", "qqq"]:
        for suffix in ["return", "score"]:
            column = f"{leg}_{suffix}"
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        active_column = f"{leg}_active"
        frame[active_column] = frame[active_column].astype(bool)
    if "risk_cash_day" not in frame.columns:
        frame["risk_cash_day"] = False
    if "risk_capped_day" not in frame.columns:
        frame["risk_capped_day"] = False
    frame["risk_cash_day"] = frame["risk_cash_day"].fillna(False).astype(bool)
    frame["risk_capped_day"] = frame["risk_capped_day"].fillna(False).astype(bool)
    return frame.sort_values("date").reset_index(drop=True)


def cvar_abs(values: pd.Series, q: float) -> float:
    series = pd.Series(values).dropna()
    if series.empty:
        return 0.0
    count = max(1, int(math.ceil(len(series) * float(q))))
    worst_mean = float(series.nsmallest(count).mean())
    return abs(min(0.0, worst_mean))


def attach_rolling_stats(frame: pd.DataFrame, *, lookback_days: int, cvar_q: float) -> pd.DataFrame:
    out = frame.copy()
    for leg in ["btc", "qqq"]:
        hist = out[f"{leg}_return"].where(out[f"{leg}_active"]).shift(1)
        rolling = hist.rolling(int(lookback_days), min_periods=1)
        out[f"{leg}_hist_count"] = rolling.count().fillna(0.0)
        out[f"{leg}_hist_mean"] = rolling.mean().fillna(0.0)
        out[f"{leg}_hist_vol"] = rolling.std(ddof=0).fillna(0.0)
        out[f"{leg}_hist_cvar_abs"] = rolling.apply(lambda values: cvar_abs(values, cvar_q), raw=False).fillna(0.0)
    return out


def leg_utility(
    row: Any,
    *,
    leg: str,
    min_score: float,
    cvar_weight: float,
    vol_weight: float,
    raw_score_weight: float,
    shrinkage_samples: float,
) -> float:
    count = float(getattr(row, f"{leg}_hist_count", 0.0) or 0.0)
    shrink = count / (count + float(shrinkage_samples)) if count > 0 else 0.0
    mean_pct = float(getattr(row, f"{leg}_hist_mean", 0.0) or 0.0) * 100.0 * shrink
    cvar_pct = float(getattr(row, f"{leg}_hist_cvar_abs", 0.0) or 0.0) * 100.0
    vol_pct = float(getattr(row, f"{leg}_hist_vol", 0.0) or 0.0) * 100.0
    score = float(getattr(row, f"{leg}_score", 0.0) or 0.0)
    raw_bonus = max(0.0, score - float(min_score)) / 50.0 * float(raw_score_weight)
    return mean_pct - float(cvar_weight) * cvar_pct - float(vol_weight) * vol_pct + raw_bonus


def summarize_equity(path: pd.DataFrame, equity_column: str = "router_equity") -> dict[str, Any]:
    if path.empty:
        return {
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "daily_cvar5_pct": 0.0,
            "annual_returns_pct": {},
            "days": 0,
        }
    equity = pd.to_numeric(path[equity_column], errors="coerce").ffill()
    returns = equity.pct_change().fillna(0.0)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if float(equity.iloc[0]) > 0 else 0.0
    frame = path.copy()
    frame["_equity"] = equity
    frame["year"] = pd.to_datetime(frame["date"]).dt.year.astype(str)
    annual: dict[str, float] = {}
    for year, group in frame.groupby("year"):
        start_equity = float(group["_equity"].iloc[0])
        end_equity = float(group["_equity"].iloc[-1])
        annual[year] = round((end_equity / start_equity - 1.0) * 100.0, 2) if start_equity > 0 else 0.0
    cvar_count = max(1, int(len(returns) * 0.05))
    return {
        "total_return_pct": round(total * 100.0, 2),
        "max_drawdown_pct": round(max_drawdown_pct(equity), 2),
        "daily_cvar5_pct": round(float(returns.nsmallest(cvar_count).mean() * 100.0), 4),
        "annual_returns_pct": annual,
        "days": int(len(path)),
        "start": str(pd.Timestamp(path["date"].iloc[0]).date()),
        "end": str(pd.Timestamp(path["date"].iloc[-1]).date()),
    }


def summarize_focus(path: pd.DataFrame, focus_start: str) -> dict[str, Any]:
    focus = path.loc[path["date"] >= pd.Timestamp(focus_start, tz="UTC")].copy()
    return summarize_equity(focus) if not focus.empty else summarize_equity(path.iloc[0:0].copy())


def baseline_summary(frame: pd.DataFrame, *, initial_capital: float, switch_cost_bps: float, focus_start: str) -> dict[str, Any]:
    if "router_equity" in frame.columns:
        path = frame[["date", "router_equity"]].copy()
    else:
        path = run_raw_score_router(
            frame,
            initial_capital=initial_capital,
            btc_min_score=35.0,
            qqq_min_score=60.0,
            switch_advantage=8.0,
            switch_cost_bps=switch_cost_bps,
        )
    summary = summarize_equity(path)
    summary["focus"] = summarize_focus(path, focus_start)
    return summary


def choose_raw_score(
    row: Any,
    *,
    current: str,
    btc_min_score: float,
    qqq_min_score: float,
    switch_advantage: float,
) -> tuple[str, str]:
    candidates: list[tuple[str, float]] = []
    if bool(row.btc_active) and float(row.btc_score) >= float(btc_min_score):
        candidates.append(("BTC", float(row.btc_score)))
    if bool(row.qqq_active) and float(row.qqq_score) >= float(qqq_min_score):
        candidates.append(("QQQ_PROXY", float(row.qqq_score)))
    if not candidates:
        return "CASH", "no_eligible_candidates"
    candidates.sort(key=lambda item: item[1], reverse=True)
    best_strategy, best_score = candidates[0]
    current_score = next((score for strategy, score in candidates if strategy == current), None)
    if current_score is not None and current != best_strategy and best_score - current_score < float(switch_advantage):
        return current, "hold_current_hysteresis"
    return best_strategy, "best_route_score"


def run_raw_score_router(
    frame: pd.DataFrame,
    *,
    initial_capital: float,
    btc_min_score: float,
    qqq_min_score: float,
    switch_advantage: float,
    switch_cost_bps: float,
) -> pd.DataFrame:
    capital = float(initial_capital)
    current = "CASH"
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        previous = current
        selected, reason = choose_raw_score(
            row,
            current=current,
            btc_min_score=btc_min_score,
            qqq_min_score=qqq_min_score,
            switch_advantage=switch_advantage,
        )
        ret = float(row.btc_return) if selected == "BTC" else float(row.qqq_return) if selected == "QQQ_PROXY" else 0.0
        switched = selected != previous
        if switched and (selected != "CASH" or previous != "CASH"):
            capital *= 1.0 - float(switch_cost_bps) / 10000.0
        capital *= 1.0 + ret
        current = selected
        rows.append(
            {
                "date": pd.Timestamp(row.date),
                "selected_strategy": selected,
                "decision_reason": reason,
                "router_return": ret,
                "router_equity": capital,
                "switched": switched,
                "btc_utility": float(row.btc_score),
                "qqq_utility": float(row.qqq_score),
            }
        )
    return pd.DataFrame(rows)


def run_calibrated_router(
    frame: pd.DataFrame,
    *,
    initial_capital: float,
    btc_min_score: float,
    qqq_min_score: float,
    min_samples: int,
    cvar_weight: float,
    vol_weight: float,
    raw_score_weight: float,
    shrinkage_samples: float,
    flat_min_utility: float,
    qqq_flat_min_utility: float,
    hold_min_utility: float,
    switch_margin_utility: float,
    switch_cost_bps: float,
    block_qqq_risk_capped_entry: bool,
) -> pd.DataFrame:
    capital = float(initial_capital)
    current = "CASH"
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        btc_utility = leg_utility(
            row,
            leg="btc",
            min_score=btc_min_score,
            cvar_weight=cvar_weight,
            vol_weight=vol_weight,
            raw_score_weight=raw_score_weight,
            shrinkage_samples=shrinkage_samples,
        )
        qqq_utility = leg_utility(
            row,
            leg="qqq",
            min_score=qqq_min_score,
            cvar_weight=cvar_weight,
            vol_weight=vol_weight,
            raw_score_weight=raw_score_weight,
            shrinkage_samples=shrinkage_samples,
        )
        utilities = {"BTC": btc_utility, "QQQ_PROXY": qqq_utility}
        active = {
            "BTC": bool(row.btc_active),
            "QQQ_PROXY": bool(row.qqq_active),
        }
        raw_gate = {
            "BTC": float(row.btc_score) >= float(btc_min_score),
            "QQQ_PROXY": float(row.qqq_score) >= float(qqq_min_score),
        }
        enough_samples = {
            "BTC": float(row.btc_hist_count) >= float(min_samples),
            "QQQ_PROXY": float(row.qqq_hist_count) >= float(min_samples),
        }
        flat_gate = {
            "BTC": float(flat_min_utility),
            "QQQ_PROXY": float(qqq_flat_min_utility),
        }
        if block_qqq_risk_capped_entry and current != "QQQ_PROXY" and bool(row.risk_capped_day):
            raw_gate["QQQ_PROXY"] = False

        eligible = [
            strategy
            for strategy in ["BTC", "QQQ_PROXY"]
            if active[strategy]
            and raw_gate[strategy]
            and enough_samples[strategy]
            and utilities[strategy] >= flat_gate[strategy]
        ]
        previous = current
        reason = "no_eligible_candidates"
        selected = "CASH"

        current_active = current in active and active[current] and enough_samples[current]
        if current_active:
            challengers = [strategy for strategy in eligible if strategy != current]
            if challengers:
                best_challenger = max(challengers, key=lambda strategy: utilities[strategy])
                if utilities[best_challenger] >= utilities[current] + float(switch_margin_utility):
                    selected = best_challenger
                    reason = "utility_switch"
                elif utilities[current] >= float(hold_min_utility):
                    selected = current
                    reason = "hold_current_utility_hysteresis"
                else:
                    selected = current if utilities[current] >= utilities[best_challenger] else best_challenger
                    reason = "current_below_hold_min_best_available"
            elif utilities[current] >= float(hold_min_utility):
                selected = current
                reason = "hold_current_no_challenger"
            else:
                reason = "current_utility_below_hold_min"
        elif eligible:
            selected = max(eligible, key=lambda strategy: utilities[strategy])
            reason = "flat_best_utility"

        ret = float(row.btc_return) if selected == "BTC" else float(row.qqq_return) if selected == "QQQ_PROXY" else 0.0
        switched = selected != previous
        if switched and (selected != "CASH" or previous != "CASH"):
            capital *= 1.0 - float(switch_cost_bps) / 10000.0
        capital *= 1.0 + ret
        current = selected
        rows.append(
            {
                "date": pd.Timestamp(row.date),
                "selected_strategy": selected,
                "decision_reason": reason,
                "router_return": ret,
                "router_equity": capital,
                "switched": switched,
                "btc_utility": round(float(btc_utility), 6),
                "qqq_utility": round(float(qqq_utility), 6),
                "btc_hist_count": float(row.btc_hist_count),
                "qqq_hist_count": float(row.qqq_hist_count),
                "btc_score": float(row.btc_score),
                "qqq_score": float(row.qqq_score),
                "btc_active": bool(row.btc_active),
                "qqq_active": bool(row.qqq_active),
                "risk_cash_day": bool(row.risk_cash_day),
                "risk_capped_day": bool(row.risk_capped_day),
            }
        )
    return pd.DataFrame(rows)


def path_metrics(path: pd.DataFrame, *, focus_start: str) -> dict[str, Any]:
    summary = summarize_equity(path)
    focus = summarize_focus(path, focus_start)
    selection = path["selected_strategy"].value_counts().to_dict() if not path.empty else {}
    summary.update(
        {
            "focus": focus,
            "switches": int(path["switched"].sum()) if "switched" in path.columns else 0,
            "selection": {str(key): int(value) for key, value in selection.items()},
        }
    )
    return summary


def rank_score(long_metrics: dict[str, Any], real_metrics: dict[str, Any]) -> float:
    long_return = max(-0.99, float(long_metrics["total_return_pct"]) / 100.0)
    real_return = max(-0.99, float(real_metrics["total_return_pct"]) / 100.0)
    long_2026 = float(long_metrics["annual_returns_pct"].get("2026", 0.0))
    real_2026 = float(real_metrics["annual_returns_pct"].get("2026", 0.0))
    return (
        math.log1p(long_return) * 100.0
        + math.log1p(real_return) * 40.0
        + long_2026 * 0.25
        + real_2026 * 0.4
        - float(long_metrics["max_drawdown_pct"]) * 1.2
        - abs(float(long_metrics["daily_cvar5_pct"])) * 2.0
        - float(real_metrics["max_drawdown_pct"]) * 1.0
        - abs(float(real_metrics["daily_cvar5_pct"])) * 1.5
        - float(long_metrics["switches"]) * 0.03
    )


def run_one(
    stats_frame: pd.DataFrame,
    *,
    params: dict[str, Any],
    initial_capital: float,
    switch_cost_bps: float,
    focus_start: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = run_calibrated_router(
        stats_frame,
        initial_capital=initial_capital,
        btc_min_score=float(params["btc_min_score"]),
        qqq_min_score=float(params["qqq_min_score"]),
        min_samples=int(params["min_samples"]),
        cvar_weight=float(params["cvar_weight"]),
        vol_weight=float(params["vol_weight"]),
        raw_score_weight=float(params["raw_score_weight"]),
        shrinkage_samples=float(params["shrinkage_samples"]),
        flat_min_utility=float(params["flat_min_utility"]),
        qqq_flat_min_utility=float(params["qqq_flat_min_utility"]),
        hold_min_utility=float(params["hold_min_utility"]),
        switch_margin_utility=float(params["switch_margin_utility"]),
        switch_cost_bps=switch_cost_bps,
        block_qqq_risk_capped_entry=bool(params["block_qqq_risk_capped_entry"]),
    )
    return path, path_metrics(path, focus_start=focus_start)


def row_from_result(params: dict[str, Any], long_metrics: dict[str, Any], real_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        **params,
        "rank_score": round(rank_score(long_metrics, real_metrics), 4),
        "long_return_pct": long_metrics["total_return_pct"],
        "long_dd_pct": long_metrics["max_drawdown_pct"],
        "long_cvar5_pct": long_metrics["daily_cvar5_pct"],
        "long_2026_pct": long_metrics["annual_returns_pct"].get("2026", 0.0),
        "long_switches": long_metrics["switches"],
        "long_btc_days": long_metrics["selection"].get("BTC", 0),
        "long_qqq_days": long_metrics["selection"].get("QQQ_PROXY", 0),
        "long_cash_days": long_metrics["selection"].get("CASH", 0),
        "real_return_pct": real_metrics["total_return_pct"],
        "real_dd_pct": real_metrics["max_drawdown_pct"],
        "real_cvar5_pct": real_metrics["daily_cvar5_pct"],
        "real_2026_pct": real_metrics["annual_returns_pct"].get("2026", 0.0),
        "real_switches": real_metrics["switches"],
        "real_btc_days": real_metrics["selection"].get("BTC", 0),
        "real_qqq_days": real_metrics["selection"].get("QQQ_PROXY", 0),
        "real_cash_days": real_metrics["selection"].get("CASH", 0),
    }


def write_markdown(
    path: Path,
    *,
    baseline_long: dict[str, Any],
    baseline_real: dict[str, Any],
    rows: list[dict[str, Any]],
    best_path_csv: Path,
) -> None:
    top = rows[:12]
    lines = [
        "# Router Calibrated Utility Scan",
        "",
        "## Baseline Raw-Score Router",
        "",
        f"- Long proxy: return `{baseline_long['total_return_pct']}%`, DD `{baseline_long['max_drawdown_pct']}%`, 2026 `{baseline_long['annual_returns_pct'].get('2026', 0.0)}%`",
        f"- Real overlap: return `{baseline_real['total_return_pct']}%`, DD `{baseline_real['max_drawdown_pct']}%`, 2026 `{baseline_real['annual_returns_pct'].get('2026', 0.0)}%`",
        "",
        "## Design",
        "",
        "- BTC/QQQ raw route score is only an entry-quality gate.",
        "- Cross-asset competition uses trailing, no-lookahead utility: `mean_return - cvar_weight * cvar - vol_weight * volatility + small raw-score bonus`.",
        "- Flat entry uses an absolute utility floor; holding uses a lower `hold_min_utility`; switching requires `switch_margin_utility` extra utility.",
        "- Long proxy and real OKX overlap are both scored; ranking penalizes drawdown, CVaR, excessive switches, and explicitly rewards 2026 return.",
        "",
        "## Top Candidates",
        "",
        "| rank | lookback | min_samples | qqq_min | cvar_w | vol_w | qqq_flat_min | switch_u | long ret | long DD | 2026 | real ret | real DD | real 2026 | switches |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(top, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row["lookback_days"]),
                    str(row["min_samples"]),
                    str(row["qqq_min_score"]),
                    str(row["cvar_weight"]),
                    str(row["vol_weight"]),
                    str(row["qqq_flat_min_utility"]),
                    str(row["switch_margin_utility"]),
                    str(row["long_return_pct"]),
                    str(row["long_dd_pct"]),
                    str(row["long_2026_pct"]),
                    str(row["real_return_pct"]),
                    str(row["real_dd_pct"]),
                    str(row["real_2026_pct"]),
                    str(row["long_switches"]),
                ]
            )
            + " |"
        )
    if top:
        best = top[0]
        lines.extend(
            [
                "",
                "## Recommended Research Candidate",
                "",
                f"- Params: `lookback={best['lookback_days']}`, `min_samples={best['min_samples']}`, `btc_min={best['btc_min_score']}`, `qqq_min={best['qqq_min_score']}`, `cvar_weight={best['cvar_weight']}`, `vol_weight={best['vol_weight']}`, `flat_min={best['flat_min_utility']}`, `qqq_flat_min={best['qqq_flat_min_utility']}`, `hold_min={best['hold_min_utility']}`, `switch_margin={best['switch_margin_utility']}`.",
                f"- Long proxy: `{best['long_return_pct']}% / DD {best['long_dd_pct']}% / 2026 {best['long_2026_pct']}%`.",
                f"- Real overlap: `{best['real_return_pct']}% / DD {best['real_dd_pct']}% / 2026 {best['real_2026_pct']}%`.",
                f"- Best path CSV: `{best_path_csv}`.",
                "",
                "## Notes",
                "",
                "- This is a research router, not live config. Runtime implementation needs the same trailing-stat state persisted online.",
                "- The long proxy still depends on NQ-continuous synthetic QQQ/USDT history; the real overlap is short but kept as a sanity check.",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a calibrated BTC/QQQ router that compares trailing return-risk utility instead of raw route-score scale.")
    parser.add_argument("--long-csv", default=str(DEFAULT_LONG_CSV))
    parser.add_argument("--real-csv", default=str(DEFAULT_REAL_CSV))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--best-path-csv", default=str(DEFAULT_BEST_PATH_CSV))
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--switch-cost-bps", type=float, default=10.0)
    parser.add_argument("--focus-start", default="2026-01-01")
    parser.add_argument("--lookback-days", default="60,120")
    parser.add_argument("--min-samples", default="10,20")
    parser.add_argument("--btc-min-scores", default="35")
    parser.add_argument("--qqq-min-scores", default="90,100")
    parser.add_argument("--cvar-weights", default="0.5,1.0")
    parser.add_argument("--vol-weights", default="0.0,0.5")
    parser.add_argument("--raw-score-weights", default="0.0,0.25")
    parser.add_argument("--shrinkage-samples", default="20")
    parser.add_argument("--flat-min-utilities", default="-1.0,0.0")
    parser.add_argument("--qqq-flat-min-utilities", default="0.0,0.5")
    parser.add_argument("--hold-min-utilities", default="-2.0")
    parser.add_argument("--switch-margin-utilities", default="0.25,0.5")
    parser.add_argument("--cvar-q", type=float, default=0.2)
    parser.add_argument("--block-qqq-risk-capped-entry", action="store_true")
    args = parser.parse_args()

    long_frame = load_router_frame(Path(args.long_csv))
    real_frame = load_router_frame(Path(args.real_csv))
    baseline_long = baseline_summary(long_frame, initial_capital=float(args.initial_capital), switch_cost_bps=float(args.switch_cost_bps), focus_start=str(args.focus_start))
    baseline_real = baseline_summary(real_frame, initial_capital=float(args.initial_capital), switch_cost_bps=float(args.switch_cost_bps), focus_start=str(args.focus_start))
    lookbacks = parse_int_list(args.lookback_days)
    long_stats_by_lookback = {
        lookback: attach_rolling_stats(long_frame, lookback_days=lookback, cvar_q=float(args.cvar_q))
        for lookback in lookbacks
    }
    real_stats_by_lookback = {
        lookback: attach_rolling_stats(real_frame, lookback_days=lookback, cvar_q=float(args.cvar_q))
        for lookback in lookbacks
    }

    param_grid = itertools.product(
        lookbacks,
        parse_int_list(args.min_samples),
        parse_float_list(args.btc_min_scores),
        parse_float_list(args.qqq_min_scores),
        parse_float_list(args.cvar_weights),
        parse_float_list(args.vol_weights),
        parse_float_list(args.raw_score_weights),
        parse_float_list(args.shrinkage_samples),
        parse_float_list(args.flat_min_utilities),
        parse_float_list(args.qqq_flat_min_utilities),
        parse_float_list(args.hold_min_utilities),
        parse_float_list(args.switch_margin_utilities),
    )

    results: list[dict[str, Any]] = []
    best_path: pd.DataFrame | None = None
    best_row: dict[str, Any] | None = None
    for (
        lookback_days,
        min_samples,
        btc_min_score,
        qqq_min_score,
        cvar_weight,
        vol_weight,
        raw_score_weight,
        shrinkage_samples,
        flat_min_utility,
        qqq_flat_min_utility,
        hold_min_utility,
        switch_margin_utility,
    ) in param_grid:
        params = {
            "lookback_days": int(lookback_days),
            "min_samples": int(min_samples),
            "btc_min_score": float(btc_min_score),
            "qqq_min_score": float(qqq_min_score),
            "cvar_weight": float(cvar_weight),
            "vol_weight": float(vol_weight),
            "raw_score_weight": float(raw_score_weight),
            "shrinkage_samples": float(shrinkage_samples),
            "flat_min_utility": float(flat_min_utility),
            "qqq_flat_min_utility": float(qqq_flat_min_utility),
            "hold_min_utility": float(hold_min_utility),
            "switch_margin_utility": float(switch_margin_utility),
            "cvar_q": float(args.cvar_q),
            "block_qqq_risk_capped_entry": bool(args.block_qqq_risk_capped_entry),
        }
        long_path, long_metrics = run_one(
            long_stats_by_lookback[int(lookback_days)],
            params=params,
            initial_capital=float(args.initial_capital),
            switch_cost_bps=float(args.switch_cost_bps),
            focus_start=str(args.focus_start),
        )
        real_path, real_metrics = run_one(
            real_stats_by_lookback[int(lookback_days)],
            params=params,
            initial_capital=float(args.initial_capital),
            switch_cost_bps=float(args.switch_cost_bps),
            focus_start=str(args.focus_start),
        )
        row = row_from_result(params, long_metrics, real_metrics)
        results.append(row)
        if best_row is None or float(row["rank_score"]) > float(best_row["rank_score"]):
            best_row = row
            best_path = long_path

    results.sort(key=lambda item: float(item["rank_score"]), reverse=True)
    if best_path is not None:
        Path(args.best_path_csv).parent.mkdir(parents=True, exist_ok=True)
        best_path.to_csv(Path(args.best_path_csv), index=False)

    output = {
        "mode": "router_calibrated_utility_scan",
        "inputs": {
            "long_csv": str(Path(args.long_csv)),
            "real_csv": str(Path(args.real_csv)),
            "focus_start": str(args.focus_start),
            "switch_cost_bps": float(args.switch_cost_bps),
        },
        "baseline": {
            "long_proxy": baseline_long,
            "real_overlap": baseline_real,
        },
        "top": results[:50],
        "count": int(len(results)),
    }
    for path in [Path(args.output_json), Path(args.output_csv), Path(args.output_md)]:
        path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    pd.DataFrame(results).to_csv(Path(args.output_csv), index=False)
    write_markdown(
        Path(args.output_md),
        baseline_long=baseline_long,
        baseline_real=baseline_real,
        rows=results,
        best_path_csv=Path(args.best_path_csv),
    )
    print(json.dumps({"output_json": str(args.output_json), "output_csv": str(args.output_csv), "output_md": str(args.output_md), "best": results[0] if results else None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
