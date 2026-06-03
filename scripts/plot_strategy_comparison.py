#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_tqqq_cash_regime_context import load_df  # noqa: E402
from scripts.nasdaq100_cn_strategy_utils import load_config, load_strategy_frame, run_full_strategy_path  # noqa: E402
from scripts.scan_tqqq_context_bucket_overlays import prepare_frame, run_candidate  # noqa: E402


DEFAULT_PUBLIC_DIR = ROOT / "data" / "public" / "etf"
DEFAULT_REPORT_DIR = ROOT / "var" / "reports"


def qqq_buyhold_curve(path: Path) -> pd.DataFrame:
    qqq = load_df(path)
    qqq = qqq.sort_values("date").reset_index(drop=True)
    qqq["daily_return"] = qqq["close"].pct_change().fillna(0.0)
    qqq["equity"] = 1000.0 * (1.0 + qqq["daily_return"]).cumprod()
    return qqq[["date", "daily_return", "equity"]].copy()


def buyhold_curve(path: Path) -> pd.DataFrame:
    df = load_df(path)
    df = df.sort_values("date").reset_index(drop=True)
    df["daily_return"] = df["close"].pct_change().fillna(0.0)
    df["equity"] = 1000.0 * (1.0 + df["daily_return"]).cumprod()
    return df[["date", "daily_return", "equity"]].copy()


def tqqq_sqqq_curve() -> pd.DataFrame:
    qqq = load_df(DEFAULT_PUBLIC_DIR / "QQQ-1d.feather")
    tqqq = load_df(DEFAULT_PUBLIC_DIR / "TQQQ-1d.feather")
    sqqq = load_df(DEFAULT_PUBLIC_DIR / "SQQQ-1d.feather")
    spy = load_df(DEFAULT_PUBLIC_DIR / "SPY-1d.feather")
    ixic = load_df(DEFAULT_PUBLIC_DIR / "^IXIC-1d.feather")
    vix = load_df(DEFAULT_PUBLIC_DIR / "^VIX-1d.feather")
    frame = prepare_frame(qqq, tqqq, sqqq, spy, ixic, vix, 25, 200)
    candidate = run_candidate(
        frame,
        long_profile_name="stable_base",
        short_rule_name="bearish_score5",
        short_exit_profile=(20, 5, 6.0),
        initial_capital=1000.0,
        switch_cost_bps=10.0,
    )
    # Rebuild the full daily path using the same engine output assumptions.
    # The engine itself does not currently return the path, so reconstruct here from trades is avoided;
    # instead use the audit summary path from the weighted scanner-compatible execution.
    from scripts.scan_tqqq_kelly_bucket_weights import run_weighted_candidate  # noqa: E402

    weighted = run_weighted_candidate(
        frame,
        long_profile_name="stable_base",
        short_rule_name="bearish_score5",
        short_exit_profile=(20, 5, 6.0),
        long_weight=1.0,
        short_weight=1.0,
        initial_capital=1000.0,
        switch_cost_bps=10.0,
    )
    # Recompute the path inline for plotting.
    long_mask = frame["vix_label"].isin(["vix_low", "vix_normal"]) & frame["ixic_trend_label"].eq("ixic_up")
    capital = 1000.0
    previous_position = "CASH"
    active_asset = "CASH"
    active_weight = 0.0
    active_max_hold_days = 0
    active_trailing_lookback_days = 0
    active_trailing_drawdown_pct = 0.0
    hold_days = 0
    rolling_peak = 0.0
    rows: list[dict[str, Any]] = []
    from scripts.scan_tqqq_context_bucket_overlays import allow_short, select_long_profile  # noqa: E402

    def asset_price_column(asset: str) -> str:
        return "tqqq_close" if asset == "TQQQ" else "sqqq_close"

    for idx, row in frame.iterrows():
        raw_desired_asset = "CASH"
        desired_long = int(row["planned_trend"]) > 0 and bool(long_mask.iloc[idx])
        desired_short = int(row["planned_trend"]) < 0 and allow_short(row, "bearish_score5")
        candidate_long_profile = select_long_profile(row, "stable_base")
        if desired_long:
            raw_desired_asset = "TQQQ"
            raw_desired_weight = 1.0
        elif desired_short:
            raw_desired_asset = "SQQQ"
            raw_desired_weight = 1.0
        else:
            raw_desired_weight = 0.0

        if active_asset != "CASH" and raw_desired_asset != active_asset:
            active_asset = "CASH"
            active_weight = 0.0
            hold_days = 0
            rolling_peak = 0.0

        if active_asset == "CASH" and raw_desired_asset != "CASH":
            active_asset = raw_desired_asset
            active_weight = raw_desired_weight
            hold_days = 0
            rolling_peak = float(row[asset_price_column(active_asset)])
            if active_asset == "TQQQ":
                active_max_hold_days, active_trailing_lookback_days, active_trailing_drawdown_pct = candidate_long_profile
            else:
                active_max_hold_days, active_trailing_lookback_days, active_trailing_drawdown_pct = (20, 5, 6.0)

        position = active_asset
        daily_ret = 0.0
        if idx > 0 and position != "CASH":
            price_col = asset_price_column(position)
            prev_close = float(frame.iloc[idx - 1][price_col])
            cur_close = float(row[price_col])
            asset_ret = cur_close / prev_close - 1.0 if prev_close > 0 else 0.0
            daily_ret = asset_ret * active_weight
            hold_days += 1
            rolling_peak = max(rolling_peak, cur_close)
            trailing_exit = False
            time_exit = False
            if (
                active_trailing_lookback_days > 0
                and active_trailing_drawdown_pct > 0
                and hold_days >= active_trailing_lookback_days
                and rolling_peak > 0
            ):
                drawdown_from_peak = (rolling_peak - cur_close) / rolling_peak * 100.0
                trailing_exit = drawdown_from_peak >= active_trailing_drawdown_pct
            if active_max_hold_days > 0 and hold_days >= active_max_hold_days:
                time_exit = True
            if trailing_exit or time_exit:
                active_asset = "CASH"
                active_weight = 0.0
                hold_days = 0
                rolling_peak = 0.0
        if idx > 0 and position != previous_position:
            daily_ret -= 10.0 / 10000.0
        previous_position = position
        capital *= 1.0 + daily_ret
        rows.append({"date": row["date"], "daily_return": daily_ret, "equity": capital})
    curve = pd.DataFrame(rows)
    assert round(float(curve.iloc[-1]["equity"]) / 1000.0 * 100.0 - 100.0, 2) == weighted["summary"]["total_return_pct"]
    return curve


def cn_etf_curve(config_path: Path) -> pd.DataFrame:
    config = load_config(config_path)
    frame = load_strategy_frame(config)
    path = run_full_strategy_path(frame, config)
    return path[["date", "daily_return", "capital"]].rename(columns={"capital": "equity"}).copy()


def normalize_curve(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    out = out.rename(columns={"equity": f"{name}_equity", "daily_return": f"{name}_daily_return"})
    return out


def add_drawdown_columns(merged: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame:
    out = merged.copy()
    for prefix in prefixes:
        equity_col = f"{prefix}_equity"
        dd_col = f"{prefix}_drawdown"
        peak = out[equity_col].cummax()
        out[dd_col] = ((peak - out[equity_col]) / peak.replace(0, pd.NA) * 100.0).fillna(0.0)
    return out


def annual_return_table(merged: pd.DataFrame, prefixes: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = merged.copy()
    frame["year"] = pd.to_datetime(frame["date"], utc=True).dt.year.astype(str)
    for year, group in frame.groupby("year"):
        row: dict[str, Any] = {"year": year}
        for prefix, label in prefixes.items():
            col = f"{prefix}_equity"
            start = float(group.iloc[0][col])
            end = float(group.iloc[-1][col])
            row[label] = round((end / start - 1.0) * 100.0, 2) if start > 0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def write_html_chart(merged: pd.DataFrame, output_html: Path) -> None:
    points = [
        {
            "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            "qqq_equity": float(row["qqq_equity"]),
            "tqqq_buyhold_equity": float(row["tqqq_buyhold_equity"]),
            "tqqq_equity": float(row["tqqq_equity"]),
            "cn_etf_equity": float(row["cn_etf_equity"]),
            "qqq_daily_return": float(row["qqq_daily_return"]) * 100.0,
            "tqqq_buyhold_daily_return": float(row["tqqq_buyhold_daily_return"]) * 100.0,
            "tqqq_daily_return": float(row["tqqq_daily_return"]) * 100.0,
            "cn_etf_daily_return": float(row["cn_etf_daily_return"]) * 100.0,
            "qqq_drawdown": float(row["qqq_drawdown"]),
            "tqqq_buyhold_drawdown": float(row["tqqq_buyhold_drawdown"]),
            "tqqq_drawdown": float(row["tqqq_drawdown"]),
            "cn_etf_drawdown": float(row["cn_etf_drawdown"]),
        }
        for _, row in merged.iterrows()
    ]
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Strategy Comparison</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      color: #1a1a1a;
      background: #f6f4ef;
    }}
    h1 {{ margin-bottom: 8px; }}
    p {{ color: #555; margin-top: 0; }}
    .card {{
      background: #fffef8;
      border: 1px solid #ddd4be;
      border-radius: 16px;
      padding: 16px;
      margin-bottom: 20px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.04);
    }}
    canvas {{ width: 100%; height: 420px; }}
  </style>
</head>
<body>
  <h1>Strategy Comparison</h1>
  <p>QQQ Buy&Hold vs TQQQ Buy&Hold vs TQQQ/SQQQ Current Strategy vs China ETF Strategy</p>
  <div class="card"><canvas id="equityChart"></canvas></div>
  <div class="card"><canvas id="returnChart"></canvas></div>
  <div class="card"><canvas id="drawdownChart"></canvas></div>
  <script>
    const points = {json.dumps(points, ensure_ascii=False)};
    const labels = points.map(p => p.date);
    new Chart(document.getElementById('equityChart'), {{
      type: 'line',
      data: {{
        labels,
        datasets: [
          {{ label: 'QQQ Buy&Hold', data: points.map(p => p.qqq_equity), borderColor: '#0f766e', borderWidth: 2, pointRadius: 0 }},
          {{ label: 'TQQQ Buy&Hold', data: points.map(p => p.tqqq_buyhold_equity), borderColor: '#9333ea', borderWidth: 2, pointRadius: 0 }},
          {{ label: 'TQQQ/SQQQ Current Strategy', data: points.map(p => p.tqqq_equity), borderColor: '#b45309', borderWidth: 2, pointRadius: 0 }},
          {{ label: 'China ETF Strategy', data: points.map(p => p.cn_etf_equity), borderColor: '#1d4ed8', borderWidth: 2, pointRadius: 0 }},
        ]
      }},
      options: {{
        responsive: true,
        interaction: {{ mode: 'index', intersect: false }},
        plugins: {{ title: {{ display: true, text: 'Equity Curve Comparison' }} }},
      }}
    }});
    new Chart(document.getElementById('returnChart'), {{
      type: 'line',
      data: {{
        labels,
        datasets: [
          {{ label: 'QQQ Buy&Hold', data: points.map(p => p.qqq_daily_return), borderColor: '#0f766e', borderWidth: 1.2, pointRadius: 0 }},
          {{ label: 'TQQQ Buy&Hold', data: points.map(p => p.tqqq_buyhold_daily_return), borderColor: '#9333ea', borderWidth: 1.2, pointRadius: 0 }},
          {{ label: 'TQQQ/SQQQ Current Strategy', data: points.map(p => p.tqqq_daily_return), borderColor: '#b45309', borderWidth: 1.2, pointRadius: 0 }},
          {{ label: 'China ETF Strategy', data: points.map(p => p.cn_etf_daily_return), borderColor: '#1d4ed8', borderWidth: 1.2, pointRadius: 0 }},
        ]
      }},
      options: {{
        responsive: true,
        interaction: {{ mode: 'index', intersect: false }},
        plugins: {{ title: {{ display: true, text: 'Daily Return Comparison (%)' }} }},
      }}
    }});
    new Chart(document.getElementById('drawdownChart'), {{
      type: 'line',
      data: {{
        labels,
        datasets: [
          {{ label: 'QQQ Buy&Hold', data: points.map(p => p.qqq_drawdown), borderColor: '#0f766e', borderWidth: 1.6, pointRadius: 0 }},
          {{ label: 'TQQQ Buy&Hold', data: points.map(p => p.tqqq_buyhold_drawdown), borderColor: '#9333ea', borderWidth: 1.6, pointRadius: 0 }},
          {{ label: 'TQQQ/SQQQ Current Strategy', data: points.map(p => p.tqqq_drawdown), borderColor: '#b45309', borderWidth: 1.6, pointRadius: 0 }},
          {{ label: 'China ETF Strategy', data: points.map(p => p.cn_etf_drawdown), borderColor: '#1d4ed8', borderWidth: 1.6, pointRadius: 0 }},
        ]
      }},
      options: {{
        responsive: true,
        interaction: {{ mode: 'index', intersect: false }},
        plugins: {{ title: {{ display: true, text: 'Drawdown Comparison (%)' }} }},
      }}
    }});
  </script>
</body>
</html>
"""
    output_html.write_text(html)


def main() -> None:
    report_dir = DEFAULT_REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    qqq_curve = normalize_curve(qqq_buyhold_curve(DEFAULT_PUBLIC_DIR / "QQQ-1d.feather"), "qqq")
    tqqq_buyhold = normalize_curve(buyhold_curve(DEFAULT_PUBLIC_DIR / "TQQQ-1d.feather"), "tqqq_buyhold")
    tqqq_curve = normalize_curve(tqqq_sqqq_curve(), "tqqq")
    cn_curve = normalize_curve(cn_etf_curve(ROOT / "config" / "config.paper.cn-nasdaq100-etf.json"), "cn_etf")

    merged = (
        qqq_curve
        .merge(tqqq_buyhold, on="date", how="inner")
        .merge(tqqq_curve, on="date", how="inner")
        .merge(cn_curve, on="date", how="inner")
    )
    merged = merged.sort_values("date").reset_index(drop=True)
    merged = add_drawdown_columns(merged, ["qqq", "tqqq_buyhold", "tqqq", "cn_etf"])

    merged_path = report_dir / "strategy_equity_return_comparison.csv"
    merged.to_csv(merged_path, index=False)

    annual_table = annual_return_table(
        merged,
        {
            "qqq": "QQQ Buy&Hold",
            "tqqq_buyhold": "TQQQ Buy&Hold",
            "tqqq": "Current TQQQ/SQQQ Strategy",
            "cn_etf": "China ETF Strategy",
        },
    )
    annual_csv = report_dir / "strategy_annual_return_comparison.csv"
    annual_md = report_dir / "strategy_annual_return_comparison.md"
    annual_table.to_csv(annual_csv, index=False)
    annual_md.write_text(annual_table.to_markdown(index=False))

    summary = {
        "range": {
            "start": str(merged["date"].min()),
            "end": str(merged["date"].max()),
            "rows": int(len(merged)),
        },
        "final_equity": {
            "qqq": round(float(merged.iloc[-1]["qqq_equity"]), 2),
            "tqqq_buyhold": round(float(merged.iloc[-1]["tqqq_buyhold_equity"]), 2),
            "tqqq_sqqq_strategy": round(float(merged.iloc[-1]["tqqq_equity"]), 2),
            "cn_etf_strategy": round(float(merged.iloc[-1]["cn_etf_equity"]), 2),
        },
        "max_drawdown_pct": {
            "qqq": round(float(merged["qqq_drawdown"].max()), 2),
            "tqqq_buyhold": round(float(merged["tqqq_buyhold_drawdown"].max()), 2),
            "tqqq_sqqq_strategy": round(float(merged["tqqq_drawdown"].max()), 2),
            "cn_etf_strategy": round(float(merged["cn_etf_drawdown"].max()), 2),
        },
    }
    summary_path = report_dir / "strategy_equity_return_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    html_path = report_dir / "strategy_equity_return_comparison.html"
    write_html_chart(merged, html_path)

    print(merged_path)
    print(annual_csv)
    print(annual_md)
    print(summary_path)
    print(html_path)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
