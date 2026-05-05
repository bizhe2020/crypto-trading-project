#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reproduce_smc_short_only_v1_10x import main as reproduce_main  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_short_only_v3_hf_blayer_10x_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce standalone SMC short-only v3 high-frequency B-layer under 10x.")
    parser.add_argument("--data-15m", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-15m-futures.feather"))
    parser.add_argument("--data-4h", default=str(ROOT / "data" / "okx" / "futures" / "BTC_USDT_USDT-4h-futures.feather"))
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--target-rr", type=float, default=2.0)
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--position-size-pct", type=float, default=1.0)
    parser.add_argument("--maintenance-margin-pct", type=float, default=0.5)
    parser.add_argument("--min-liq-buffer-pct", type=float, default=1.2)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.argv = [
        "reproduce_smc_short_only_v1_10x.py",
        "--data-15m", args.data_15m,
        "--data-4h", args.data_4h,
        "--start-date", args.start_date,
        "--target-rr", str(args.target_rr),
        "--allowed-time-buckets", "other+asia_evening_ny+ny_am_killzone",
        "--swing-n", "2",
        "--min-body-atr", "0.5",
        "--min-range-atr", "0.9",
        "--entry-lookahead-bars", "72",
        "--max-open-positions", "1",
        "--max-mss-lag-bars", "24",
        "--min-displacement-body-atr", "0.3",
        "--leverage", str(args.leverage),
        "--position-size-pct", str(args.position_size_pct),
        "--maintenance-margin-pct", str(args.maintenance_margin_pct),
        "--min-liq-buffer-pct", str(args.min_liq_buffer_pct),
        "--initial-capital", str(args.initial_capital),
        "--output", args.output,
    ]
    reproduce_main()


if __name__ == "__main__":
    main()
