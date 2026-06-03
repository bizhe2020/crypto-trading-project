#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_proxy_strategy_router import DEFAULT_BTC_FROZEN, DEFAULT_QQQ_USDT_CONFIG, build_btc_path_from_frozen_artifact  # noqa: E402
from scripts.replay_qqq_usdt_10x import load_funding  # noqa: E402
from scripts.scan_qqq_usdt_4h_triggers import attach_daily_state, load_okx_4h, load_signal_path  # noqa: E402
from scripts.scan_qqq_usdt_btc_logic_overlays import enrich_bars  # noqa: E402
from scripts.scan_qqq_usdt_shadow_gate_router import route_candidate, simulate_qqq_path, summarize_equity  # noqa: E402


DEFAULT_NQ_4H = ROOT / "data" / "proxy" / "qqq_usdt_nq_continuous" / "QQQ_USDT_USDT-4h-futures-nq-continuous-dailyproxy-long.feather"
DEFAULT_NQ_FUNDING = ROOT / "data" / "proxy" / "qqq_usdt_nq_continuous" / "QQQ_USDT_USDT-8h-funding_rate-zero-nq-continuous-scaled-long.feather"
DEFAULT_REAL_4H = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-4h-futures.feather"
DEFAULT_REAL_FUNDING = ROOT / "data" / "okx" / "futures" / "QQQ_USDT_USDT-8h-funding_rate.feather"
DEFAULT_OUTPUT = ROOT / "var" / "reports" / "qqq_shadow_gate_robustness_20220101_20260529.json"


BASELINE = {
    "stop_loss_pct": 3.5,
    "reentry_rule": "signal_reset",
    "reentry_cooldown_bars": 0,
    "reentry_clear_bars": 0,
    "loss_streak_stop": 0,
    "loss_streak_cooldown_bars": 0,
    "equity_dd_stop_pct": 0.0,
    "equity_dd_cooldown_bars": 0,
}

CANDIDATES: dict[str, dict[str, Any]] = {
    "baseline": BASELINE,
    "trailing_stop4_clear2": {
        **BASELINE,
        "stop_loss_pct": 4.0,
        "reentry_rule": "clear",
        "reentry_clear_bars": 2,
    },
    "shadow_balanced": {
        **BASELINE,
        "stop_loss_pct": 4.0,
        "reentry_rule": "clear",
        "reentry_clear_bars": 2,
        "loss_streak_stop": 2,
        "loss_streak_cooldown_bars": 20,
        "equity_dd_stop_pct": 25.0,
        "equity_dd_cooldown_bars": 10,
    },
    "shadow_low_dd": {
        **BASELINE,
        "stop_loss_pct": 4.0,
        "reentry_rule": "clear",
        "reentry_clear_bars": 2,
        "equity_dd_stop_pct": 15.0,
        "equity_dd_cooldown_bars": 20,
    },
    "shadow_cvar": {
        **BASELINE,
        "stop_loss_pct": 4.0,
        "reentry_rule": "clear",
        "reentry_clear_bars": 2,
        "equity_dd_stop_pct": 15.0,
        "equity_dd_cooldown_bars": 40,
    },
}


def parse_end_timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)


def load_enriched_bars(config: dict[str, Any], data_path: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    _, signal_path = load_signal_path(ROOT / str(config["signal_source"]))
    bars = enrich_bars(attach_daily_state(load_okx_4h(data_path), signal_path))
    return bars.loc[(bars["date"] >= start) & (bars["date"] <= end)].copy()


def bar_closure_audit(bars: pd.DataFrame, reference_now: pd.Timestamp) -> dict[str, Any]:
    dates = pd.to_datetime(bars["date"], utc=True).sort_values()
    deltas = dates.diff().dropna()
    median_delta = deltas.median() if not deltas.empty else pd.Timedelta(0)
    last_open = dates.iloc[-1] if not dates.empty else pd.NaT
    last_close = last_open + median_delta if pd.notna(last_open) else pd.NaT
    return {
        "rows": int(len(dates)),
        "start": str(dates.iloc[0]) if len(dates) else None,
        "end_open": str(last_open) if pd.notna(last_open) else None,
        "median_bar_delta": str(median_delta),
        "estimated_last_close": str(last_close) if pd.notna(last_close) else None,
        "reference_now": str(reference_now),
        "last_bar_closed_by_reference": bool(pd.notna(last_close) and last_close <= reference_now),
    }


def closed_only_bars(bars: pd.DataFrame, reference_now: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(bars["date"], utc=True).sort_values()
    deltas = dates.diff().dropna()
    if deltas.empty:
        return bars.copy()
    median_delta = deltas.median()
    cutoff = reference_now - median_delta
    return bars.loc[bars["date"] <= cutoff].copy()


def run_candidate(
    *,
    params: dict[str, Any],
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    btc_path: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    qqq_path, qqq_summary = simulate_qqq_path(
        bars,
        funding,
        initial_capital=float(config["initial_capital"]),
        leverage=float(config["base_leverage"]),
        stop_loss_pct=float(params["stop_loss_pct"]),
        taker_fee_rate=float(config["taker_fee_rate"]),
        slippage_bps=float(config["slippage_bps"]),
        reentry_rule=str(params["reentry_rule"]),
        reentry_cooldown_bars=int(params["reentry_cooldown_bars"]),
        reentry_clear_bars=int(params["reentry_clear_bars"]),
        loss_streak_stop=int(params["loss_streak_stop"]),
        loss_streak_cooldown_bars=int(params["loss_streak_cooldown_bars"]),
        equity_dd_stop_pct=float(params["equity_dd_stop_pct"]),
        equity_dd_cooldown_bars=int(params["equity_dd_cooldown_bars"]),
    )
    full, routed = route_candidate(btc_path=btc_path, qqq_path=qqq_path, initial_capital=float(config["initial_capital"]))
    return full, {
        "params": params,
        "router": routed["router"],
        "qqq": routed["qqq"],
        "selection": routed["selection"],
        "qqq_path_summary": qqq_summary,
    }


def annual_metrics(path: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    frame = path.copy()
    frame["year"] = pd.to_datetime(frame["date"], utc=True).dt.year.astype(str)
    for year, group in frame.groupby("year"):
        out[year] = {
            "router": summarize_equity(group["router_equity"]),
            "qqq": summarize_equity(group["qqq_equity"]),
        }
    return out


def rolling_compare(candidate: pd.DataFrame, baseline: pd.DataFrame, *, window: int, step: int) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    n = min(len(candidate), len(baseline))
    for start in range(0, max(0, n - window + 1), step):
        cand_slice = candidate.iloc[start : start + window]
        base_slice = baseline.iloc[start : start + window]
        cand = summarize_equity(cand_slice["router_equity"])
        base = summarize_equity(base_slice["router_equity"])
        windows.append(
            {
                "start": str(pd.Timestamp(cand_slice["date"].iloc[0]).date()),
                "end": str(pd.Timestamp(cand_slice["date"].iloc[-1]).date()),
                "candidate": cand,
                "baseline": base,
                "dd_improved": cand["max_drawdown_pct"] < base["max_drawdown_pct"],
                "cvar_improved": cand["daily_cvar5_pct"] > base["daily_cvar5_pct"],
                "calmar_improved": (cand["calmar_like"] or -999999) > (base["calmar_like"] or -999999),
                "return_improved": cand["total_return_pct"] > base["total_return_pct"],
            }
        )
    return {
        "window_days": window,
        "step_days": step,
        "count": len(windows),
        "dd_improved_pct": round(sum(item["dd_improved"] for item in windows) / len(windows) * 100.0, 2) if windows else 0.0,
        "cvar_improved_pct": round(sum(item["cvar_improved"] for item in windows) / len(windows) * 100.0, 2) if windows else 0.0,
        "calmar_improved_pct": round(sum(item["calmar_improved"] for item in windows) / len(windows) * 100.0, 2) if windows else 0.0,
        "return_improved_pct": round(sum(item["return_improved"] for item in windows) / len(windows) * 100.0, 2) if windows else 0.0,
        "worst_candidate_dd": max((item["candidate"]["max_drawdown_pct"] for item in windows), default=0.0),
        "windows": windows,
    }


def overlap_consistency(nq_path: pd.DataFrame, real_path: pd.DataFrame) -> dict[str, Any]:
    merged = nq_path.merge(real_path, on="date", suffixes=("_nq", "_real"), how="inner")
    if merged.empty:
        return {"days": 0}

    def col(name: str) -> pd.Series:
        value = merged[name]
        if isinstance(value, pd.DataFrame):
            return value.iloc[:, 0]
        return value

    return {
        "days": int(len(merged)),
        "start": str(pd.Timestamp(merged["date"].iloc[0]).date()),
        "end": str(pd.Timestamp(merged["date"].iloc[-1]).date()),
        "selected_match_pct": round(float((col("selected_strategy_nq") == col("selected_strategy_real")).mean() * 100.0), 2),
        "qqq_active_match_pct": round(float((col("qqq_active_nq") == col("qqq_active_real")).mean() * 100.0), 2),
        "router_return_corr": round(float(col("router_return_nq").corr(col("router_return_real"))), 4),
        "qqq_return_corr": round(float(col("qqq_return_nq").corr(col("qqq_return_real"))), 4),
    }


def neighborhood_params() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for stop in [3.5, 4.0, 4.5]:
        for clear in [1, 2, 3]:
            for equity_dd in [20.0, 25.0, 30.0]:
                for loss_cooldown in [10, 20]:
                    name = f"stop{stop}_clear{clear}_edd{int(equity_dd)}_lc{loss_cooldown}"
                    out[name] = {
                        **BASELINE,
                        "stop_loss_pct": stop,
                        "reentry_rule": "clear",
                        "reentry_clear_bars": clear,
                        "loss_streak_stop": 2,
                        "loss_streak_cooldown_bars": loss_cooldown,
                        "equity_dd_stop_pct": equity_dd,
                        "equity_dd_cooldown_bars": 10,
                    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness audit for selected QQQ shadow gate + trailing candidates.")
    parser.add_argument("--config", default=str(DEFAULT_QQQ_USDT_CONFIG))
    parser.add_argument("--nq-data-4h", default=str(DEFAULT_NQ_4H))
    parser.add_argument("--nq-funding", default=str(DEFAULT_NQ_FUNDING))
    parser.add_argument("--real-data-4h", default=str(DEFAULT_REAL_4H))
    parser.add_argument("--real-funding", default=str(DEFAULT_REAL_FUNDING))
    parser.add_argument("--btc-frozen", default=str(DEFAULT_BTC_FROZEN))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2026-05-29")
    parser.add_argument("--real-start-date", default="2026-03-04")
    parser.add_argument("--reference-now", default="2026-05-30T00:00:00+08:00")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    reference_now = pd.Timestamp(args.reference_now).tz_convert("UTC")
    start = pd.Timestamp(args.start_date, tz="UTC")
    end = parse_end_timestamp(args.end_date)
    real_start = pd.Timestamp(args.real_start_date, tz="UTC")
    nq_bars = load_enriched_bars(config, Path(args.nq_data_4h), start=start, end=end)
    real_bars = load_enriched_bars(config, Path(args.real_data_4h), start=real_start, end=end)
    nq_closed = closed_only_bars(nq_bars, reference_now)
    real_closed = closed_only_bars(real_bars, reference_now)
    nq_funding = load_funding(Path(args.nq_funding))
    real_funding = load_funding(Path(args.real_funding))
    btc_full, _ = build_btc_path_from_frozen_artifact(frozen_path=Path(args.btc_frozen), start=start, end=end, initial_capital=float(config["initial_capital"]))
    btc_real, _ = build_btc_path_from_frozen_artifact(frozen_path=Path(args.btc_frozen), start=real_start, end=end, initial_capital=float(config["initial_capital"]))

    full_paths: dict[str, pd.DataFrame] = {}
    selected: dict[str, Any] = {}
    for name, params in CANDIDATES.items():
        path, summary = run_candidate(params=params, bars=nq_bars, funding=nq_funding, btc_path=btc_full, config=config)
        full_paths[name] = path
        selected[name] = summary | {"annual": annual_metrics(path)}

    baseline_path = full_paths["baseline"]
    for name, path in full_paths.items():
        if name == "baseline":
            continue
        selected[name]["rolling"] = {
            "126d": rolling_compare(path, baseline_path, window=126, step=21),
            "252d": rolling_compare(path, baseline_path, window=252, step=21),
        }

    closed_only: dict[str, Any] = {}
    for name, params in CANDIDATES.items():
        path, summary = run_candidate(params=params, bars=nq_closed, funding=nq_funding, btc_path=btc_full, config=config)
        closed_only[name] = summary

    real_overlap: dict[str, Any] = {}
    nq_overlap_paths: dict[str, pd.DataFrame] = {}
    real_paths: dict[str, pd.DataFrame] = {}
    for name, params in CANDIDATES.items():
        nq_overlap, nq_summary = run_candidate(params=params, bars=nq_bars.loc[nq_bars["date"] >= real_start].copy(), funding=nq_funding, btc_path=btc_real, config=config)
        real_path, real_summary = run_candidate(params=params, bars=real_bars, funding=real_funding, btc_path=btc_real, config=config)
        nq_overlap_paths[name] = nq_overlap
        real_paths[name] = real_path
        real_overlap[name] = {
            "nq_overlap": nq_summary,
            "real": real_summary,
            "consistency": overlap_consistency(nq_overlap, real_path),
        }

    neighborhood: dict[str, Any] = {}
    for name, params in neighborhood_params().items():
        _, summary = run_candidate(params=params, bars=nq_bars, funding=nq_funding, btc_path=btc_full, config=config)
        neighborhood[name] = summary
    ranked_neighborhood = {
        "top_by_dd": sorted(neighborhood.items(), key=lambda item: (item[1]["router"]["max_drawdown_pct"], -item[1]["router"]["total_return_pct"]))[:15],
        "top_by_calmar": sorted(neighborhood.items(), key=lambda item: item[1]["router"]["calmar_like"] or -999999, reverse=True)[:15],
        "top_by_cvar": sorted(neighborhood.items(), key=lambda item: item[1]["router"]["daily_cvar5_pct"], reverse=True)[:15],
    }

    report = {
        "period": {"start": args.start_date, "end": args.end_date},
        "bar_closure_audit": {
            "reference_now": str(reference_now),
            "nq": bar_closure_audit(nq_bars, reference_now),
            "nq_closed_only_rows": int(len(nq_closed)),
            "real": bar_closure_audit(real_bars, reference_now),
            "real_closed_only_rows": int(len(real_closed)),
        },
        "selected_candidates": selected,
        "closed_only_check": closed_only,
        "real_overlap": real_overlap,
        "neighborhood": {
            "candidate_count": len(neighborhood),
            "ranked": ranked_neighborhood,
            "results": neighborhood,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(out)
    print(json.dumps({k: report[k] for k in ["bar_closure_audit", "selected_candidates", "closed_only_check"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
