#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 scripts/scan_controlled_scale_in.py \
  --config config/config.live.5x-3pct.json \
  --best-params config/high_leverage_pressure_target_cap_best.params.json \
  --data-15m data/okx/futures/BTC_USDT_USDT-15m-futures.feather \
  --data-4h data/okx/futures/BTC_USDT_USDT-4h-futures.feather \
  --start-date 2022-01-01 \
  --scale-in-trigger-rr-values 1.5,2.0 \
  --scale-in-min-bars-held-values 8 \
  --scale-in-min-interval-bars-values 16 \
  --scale-in-risk-fraction-values 0.1,0.25 \
  --scale-in-total-risk-multiplier-values 1.0 \
  --scale-in-max-total-notional-multiplier-values 1.0 \
  --scale-in-min-target-rr-values 3.0 \
  --scale-in-min-price-move-pct-values 0.5 \
  --scale-in-max-stop-distance-pct-values 1.0,1.5 \
  --scale-in-require-stop-at-breakeven-values true \
  --scale-in-regime-label-sets high_growth \
  --scale-in-trail-style-sets loose \
  --top 10 \
  --output var/high_leverage_expansion/controlled_scale_in_strict_scan.json
