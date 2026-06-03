#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_proxy_strategy_router import max_drawdown_pct  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "config.paper.qqq-usdt-aggressive-frozen.json"
DEFAULT_NQ_4H = ROOT / "data" / "proxy" / "qqq_usdt_nq_continuous" / "QQQ_USDT_USDT-4h-futures-nq-continuous-dailyproxy-long.feather"
DEFAULT_NQ_FUNDING = ROOT / "data" / "proxy" / "qqq_usdt_nq_continuous" / "QQQ_USDT_USDT-8h-funding_rate-zero-nq-continuous-scaled-long.feather"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "router_qqq_risk_attribution_20220101_20260529.json"
DEFAULT_WORK_DIR = ROOT / "var" / "tmp" / "router_qqq_risk_attribution"
EPSILON = 1e-12


def run_replay(
    *,
    label: str,
    config_path: Path,
    start_date: str,
    end_date: str,
    data_4h: Path,
    funding: Path,
    work_dir: Path,
    disable_risk: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Path]]:
    out_json = work_dir / f"{label}.json"
    out_md = work_dir / f"{label}.md"
    out_csv = work_dir / f"{label}.csv"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "replay_proxy_strategy_router.py"),
        "--qqq-source",
        "usdt_leveraged",
        "--qqq-usdt-config",
        str(config_path),
        "--qqq-usdt-data-4h",
        str(data_4h),
        "--qqq-usdt-funding",
        str(funding),
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--output-json",
        str(out_json),
        "--output-md",
        str(out_md),
        "--output-csv",
        str(out_csv),
    ]
    if disable_risk:
        cmd.append("--disable-qqq-risk-overlay")
    subprocess.run(cmd, cwd=ROOT, check=True)
    return json.loads(out_json.read_text()), pd.read_csv(out_csv, parse_dates=["date"]), {"json": out_json, "md": out_md, "csv": out_csv}


def write_variant_config(base: dict[str, Any], path: Path, *, recent: bool, long_cycle: bool) -> Path:
    payload = dict(base)
    payload["risk_overlay_enabled"] = True
    if not recent:
        payload.pop("recent_risk_predictions_csv", None)
        payload.pop("recent_risk_score_column", None)
    if not long_cycle:
        payload.pop("long_cycle_risk_predictions_csv", None)
        payload.pop("long_cycle_risk_score_column", None)
        payload.pop("long_cycle_risk_cap_rules", None)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def summarize_equity(equity: pd.Series) -> dict[str, Any]:
    if equity.empty:
        return {"total_return_pct": 0.0, "max_drawdown_pct": 0.0, "calmar_like": None, "daily_cvar5_pct": 0.0}
    equity = pd.to_numeric(equity, errors="coerce").ffill()
    returns = equity.pct_change().fillna(0.0)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if float(equity.iloc[0]) > 0 else 0.0
    dd = max_drawdown_pct(equity)
    cvar5 = float(returns.nsmallest(max(1, int(len(returns) * 0.05))).mean() * 100.0) if len(returns) else 0.0
    return {
        "total_return_pct": round(total * 100.0, 2),
        "max_drawdown_pct": round(dd, 2),
        "calmar_like": round((total * 100.0) / dd, 4) if dd > 0 else None,
        "daily_cvar5_pct": round(cvar5, 4),
    }


def frame_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "router": summarize_equity(frame["router_equity"]),
        "qqq": summarize_equity(frame["qqq_equity"]),
        "btc": summarize_equity(frame["btc_equity"]),
    }


def pct(value: float) -> float:
    return round(float(value) * 100.0, 4)


def event_layer(row: Any) -> str:
    cash = bool(getattr(row, "risk_cash_day_policy", False))
    cap = bool(getattr(row, "risk_capped_day_policy", False))
    if cash and cap:
        return "cash_and_cap"
    if cash:
        return "cash"
    if cap:
        return "cap"
    return "return_changed_without_policy_flag"


def attribution(policy: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    merged = baseline.merge(policy, on="date", suffixes=("_base", "_policy"), how="inner")
    selected = merged[(merged["selected_strategy_base"] == "QQQ_PROXY") & (merged["qqq_active_base"])]
    delta = selected["qqq_return_policy"].fillna(0.0) - selected["qqq_return_base"].fillna(0.0)
    helped = selected[delta > EPSILON]
    hurt = selected[delta < -EPSILON]
    loss_days = selected["qqq_return_base"] < 0
    gain_days = selected["qqq_return_base"] > 0
    policy_risk_day = selected[["risk_cash_day_policy", "risk_capped_day_policy"]].any(axis=1)
    event_rows = selected.loc[policy_risk_day | (delta.abs() > EPSILON)].copy()
    event_delta = delta.loc[event_rows.index]
    events: list[dict[str, Any]] = []
    for row, day_delta in zip(event_rows.itertuples(index=False), event_delta):
        base_return = float(getattr(row, "qqq_return_base", 0.0) or 0.0)
        policy_return = float(getattr(row, "qqq_return_policy", 0.0) or 0.0)
        if day_delta > EPSILON and base_return < 0:
            outcome = "avoided_loss"
        elif day_delta < -EPSILON and base_return > 0:
            outcome = "missed_gain"
        elif day_delta > EPSILON:
            outcome = "helped_other"
        elif day_delta < -EPSILON:
            outcome = "hurt_other"
        else:
            outcome = "no_return_change"
        events.append(
            {
                "date": str(pd.Timestamp(getattr(row, "date")).date()),
                "layer": event_layer(row),
                "outcome": outcome,
                "base_qqq_return_pct": pct(base_return),
                "policy_qqq_return_pct": pct(policy_return),
                "delta_pct_points": pct(float(day_delta)),
                "base_avg_leverage": round(float(getattr(row, "avg_leverage_when_active_base", 0.0) or 0.0), 4),
                "policy_avg_leverage": round(float(getattr(row, "avg_leverage_when_active_policy", 0.0) or 0.0), 4),
                "policy_selected_strategy": str(getattr(row, "selected_strategy_policy", "")),
            }
        )
    return {
        "qqq_selected_days": int(len(selected)),
        "risk_trigger_days": int(policy_risk_day.sum()),
        "changed_return_days": int((delta.abs() > EPSILON).sum()),
        "compounded_policy_vs_baseline_pct": round(((1.0 + selected["qqq_return_policy"].fillna(0.0)).prod() / (1.0 + selected["qqq_return_base"].fillna(0.0)).prod() - 1.0) * 100.0, 2) if len(selected) else 0.0,
        "sum_daily_delta_pct_points": round(float(delta.sum() * 100.0), 2),
        "helped_days": int(len(helped)),
        "hurt_days": int(len(hurt)),
        "loss_days_helped": int(((delta > 1e-12) & loss_days).sum()),
        "gain_days_hurt": int(((delta < -1e-12) & gain_days).sum()),
        "avoided_loss_pct_points": round(float(delta[(delta > 0) & loss_days].sum() * 100.0), 2),
        "missed_gain_pct_points": round(float((-delta[(delta < 0) & gain_days]).sum() * 100.0), 2),
        "event_counts": pd.Series([event["outcome"] for event in events]).value_counts().to_dict() if events else {},
        "events": events,
    }


def choose_strategy(row: Any, *, current: str, qqq_min_score: float, risk_flag: bool, policy: str, penalty: float) -> tuple[str, str]:
    btc_active = bool(row.btc_active)
    btc_score = float(row.btc_score)
    qqq_active = bool(row.qqq_active)
    qqq_score = float(row.qqq_score)
    if risk_flag and policy == "no_new_qqq" and current != "QQQ_PROXY":
        qqq_active = False
    if risk_flag and policy == "block_btc_to_qqq" and current == "BTC":
        qqq_active = False
    if risk_flag and policy == "threshold_penalty":
        qqq_min_score += float(penalty)
    candidates: list[tuple[str, float]] = []
    if btc_active and btc_score >= 35.0:
        candidates.append(("BTC", btc_score))
    if qqq_active and qqq_score >= qqq_min_score:
        candidates.append(("QQQ_PROXY", qqq_score))
    if not candidates:
        return "CASH", "no_eligible_candidates"
    candidates.sort(key=lambda item: item[1], reverse=True)
    best_strategy, best_score = candidates[0]
    current_score = next((score for strategy, score in candidates if strategy == current), None)
    if current_score is not None and current != best_strategy and (best_score - current_score) < 8.0:
        return current, "hold_current_hysteresis"
    return best_strategy, "best_route_score"


def run_weak_router(base: pd.DataFrame, risk_flags: pd.Series, *, policy: str, penalty: float = 0.0, switch_cost_bps: float = 10.0) -> pd.DataFrame:
    capital = 1000.0
    current = "CASH"
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(base.itertuples(index=False)):
        previous = current
        selected, reason = choose_strategy(
            row,
            current=current,
            qqq_min_score=60.0,
            risk_flag=bool(risk_flags.iloc[idx]),
            policy=policy,
            penalty=penalty,
        )
        selected_return = 0.0
        if selected == "BTC":
            selected_return = float(row.btc_return)
        elif selected == "QQQ_PROXY":
            selected_return = float(row.qqq_return)
        switched = selected != previous
        if switched and (selected != "CASH" or previous != "CASH"):
            capital *= 1.0 - float(switch_cost_bps) / 10000.0
        capital *= 1.0 + selected_return
        current = selected
        rows.append(
            {
                "date": row.date,
                "selected_strategy": selected,
                "router_return": selected_return,
                "router_equity": capital,
                "decision_reason": reason,
                "switched": switched,
            }
        )
    return pd.DataFrame(rows)


def aligned_risk_flags(base: pd.DataFrame, policy: pd.DataFrame, columns: list[str]) -> pd.Series:
    flag_frame = policy[["date", *columns]].copy()
    flag_frame["risk_flag"] = flag_frame[columns].any(axis=1)
    merged = base[["date"]].merge(flag_frame[["date", "risk_flag"]], on="date", how="left")
    return merged["risk_flag"].fillna(False).astype(bool)


def weak_result_summary(result: pd.DataFrame, base: pd.DataFrame) -> dict[str, Any]:
    metrics = summarize_equity(result["router_equity"])
    base_end = float(base["router_equity"].iloc[-1])
    result_end = float(result["router_equity"].iloc[-1])
    metrics.update(
        {
            "relative_final_equity_vs_no_risk_pct": round((result_end / base_end - 1.0) * 100.0, 2) if base_end > 0 else 0.0,
            "selection": result["selected_strategy"].value_counts().to_dict(),
            "changed_selection_days_vs_no_risk": int((result["selected_strategy"].reset_index(drop=True) != base["selected_strategy"].reset_index(drop=True)).sum()),
            "switches": int(result["switched"].sum()),
        }
    )
    return metrics


def weak_action_sweeps(base: pd.DataFrame, risk_flags_by_label: dict[str, pd.Series]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    penalties = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70]
    for flag_label, flags in risk_flags_by_label.items():
        policy_results: dict[str, Any] = {}
        for label, policy, penalty in [
            ("no_new_qqq", "no_new_qqq", 0.0),
            ("block_btc_to_qqq", "block_btc_to_qqq", 0.0),
        ]:
            result = run_weak_router(base, flags, policy=policy, penalty=penalty)
            policy_results[label] = weak_result_summary(result, base)

        threshold_results: dict[str, Any] = {}
        for penalty in penalties:
            result = run_weak_router(base, flags, policy="threshold_penalty", penalty=float(penalty))
            threshold_results[f"penalty_{penalty}"] = weak_result_summary(result, base)
        policy_results["threshold_penalty"] = threshold_results
        output[flag_label] = policy_results
    return output


def replay_parity_check(base: pd.DataFrame) -> dict[str, Any]:
    simulated = run_weak_router(
        base.reset_index(drop=True),
        pd.Series([False] * len(base)),
        policy="no_action",
    )
    diff = (pd.to_numeric(simulated["router_equity"]) - pd.to_numeric(base["router_equity"].reset_index(drop=True))).abs()
    return {
        "max_abs_router_equity_diff": round(float(diff.max() if len(diff) else 0.0), 10),
        "final_abs_router_equity_diff": round(float(diff.iloc[-1] if len(diff) else 0.0), 10),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Attribute QQQ risk overlay only when router would select QQQ, then test weaker router-level actions.")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2026-05-29")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-4h", default=str(DEFAULT_NQ_4H))
    parser.add_argument("--funding", default=str(DEFAULT_NQ_FUNDING))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    base_config = json.loads(Path(args.config).read_text())
    recent_config = write_variant_config(base_config, work_dir / "recent_only_config.json", recent=True, long_cycle=False)
    long_config = write_variant_config(base_config, work_dir / "long_only_config.json", recent=False, long_cycle=True)

    runs: dict[str, dict[str, Any]] = {}
    frames: dict[str, pd.DataFrame] = {}
    paths: dict[str, dict[str, str]] = {}
    for label, cfg, disabled in [
        ("no_risk", Path(args.config), True),
        ("current", Path(args.config), False),
        ("recent_only", recent_config, False),
        ("long_only", long_config, False),
    ]:
        summary, frame, out_paths = run_replay(
            label=label,
            config_path=cfg,
            start_date=str(args.start_date),
            end_date=str(args.end_date),
            data_4h=Path(args.data_4h),
            funding=Path(args.funding),
            work_dir=work_dir,
            disable_risk=disabled,
        )
        runs[label] = summary
        frames[label] = frame
        paths[label] = {key: str(value) for key, value in out_paths.items()}

    base = frames["no_risk"].reset_index(drop=True)
    risk_flags_by_label = {
        "current_any": aligned_risk_flags(base, frames["current"], ["risk_cash_day", "risk_capped_day"]),
        "recent_cash_only": aligned_risk_flags(base, frames["recent_only"], ["risk_cash_day"]),
        "long_cap_only": aligned_risk_flags(base, frames["long_only"], ["risk_capped_day"]),
    }

    output = {
        "period": {"start": args.start_date, "end": args.end_date},
        "runs": {
            label: {
                "router": runs[label]["router"],
                "qqq": runs[label]["qqq_proxy_only"],
                "cost_inclusive_metrics": frame_metrics(frames[label]),
                "selection": runs[label]["selection"],
                "risk_counts": {key: runs[label]["source_summaries"]["qqq_proxy_summary"].get(key) for key in ["risk_cash_days", "risk_capped_days", "risk_exit_events", "avg_leverage_when_in"]},
                "paths": paths[label],
            }
            for label in runs
        },
        "attribution_vs_no_risk_on_no_risk_qqq_selected_days": {
            label: attribution(frames[label], frames["no_risk"])
            for label in ["current", "recent_only", "long_only"]
        },
        "weak_router_action_notes": {
            "basis": "Weak router actions are simulated from the no-risk replay daily BTC/QQQ returns and apply the same 10 bps route-switch cost to router_equity.",
            "parity_vs_no_risk_replay_before_policy": replay_parity_check(base),
        },
        "weak_router_actions": weak_action_sweeps(base, risk_flags_by_label),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(out)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
