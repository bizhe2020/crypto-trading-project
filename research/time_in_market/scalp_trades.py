"""第 1 步：用实盘配置无前视重放 BTC scalp，导出逐笔成交（供在场率/组合研究用）。

唯一口径：scripts.live_readiness_report.load_prepared_data → build_precomputed_state_confirmed_4h。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_readiness_report import compact_metrics, load_prepared_data, run_engine, trade_dataframe  # noqa: E402
from scripts.scan_btc_scalp_optimization import load_base_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-15m", required=True)
    parser.add_argument("--data-4h", required=True)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--shadow-gate-backtest",
        action="store_true",
        help="在引擎内启用 shadow risk gate（实盘执行层会跑这道闸门；不开则逐笔流比实盘多出被闸门挡掉的单）",
    )
    args = parser.parse_args()

    payload = load_base_config(Path(args.config))
    if args.shadow_gate_backtest:
        payload["enable_shadow_risk_gate_backtest"] = True
    prepared = load_prepared_data(
        Path(args.data_15m),
        Path(args.data_4h),
        pd.Timestamp(args.start, tz="UTC"),
        payload.get("regime_switcher_thresholds"),
    )
    metrics, engine = run_engine(payload, prepared, args.start)
    trades = trade_dataframe(engine)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in ("entry_time", "exit_time", "direction", "entry_price", "exit_price", "quantity",
                        "pnl", "gross_pnl", "fees", "exit_reason", "capital_after", "leverage") if c in trades.columns]
    trades[keep].to_csv(out, index=False)
    summary = compact_metrics(metrics)
    summary.update({
        "initial_capital": float(metrics.get("initial_capital", 0.0)),
        "final_capital": float(metrics.get("final_capital", 0.0)),
        "data_end": str(prepared.end),
        "columns": list(trades.columns),
    })
    out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("total_return_pct", "max_drawdown_pct", "total_trades", "final_capital", "data_end")}))


if __name__ == "__main__":
    main()
