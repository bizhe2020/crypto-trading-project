#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 scripts/scan_pressure_level_trailing.py \
  --config config/config.live.5x-3pct.json \
  --data-15m data/okx/futures/BTC_USDT_USDT-15m-futures.feather \
  --data-4h data/okx/futures/BTC_USDT_USDT-4h-futures.feather \
  --start-date 2022-01-01 \
  --pressure-min-rr-values 2.0 \
  --pressure-lock-rr-values 0.4 \
  --pressure-atr-multiplier-values 3.0 \
  --pressure-proximity-pct-values 0.15 \
  --pressure-rejection-min-rr-values 3.0 \
  --pressure-take-profit-on-rejection-values false \
  --pressure-enable-target-cap-values true \
  --pressure-target-min-rr-values 1.25 \
  --pressure-target-buffer-pct-values 0.03 \
  --pressure-dynamic-target-min-rr-enabled-values false \
  --pressure-regime-label-sets flat \
  --pressure-touch-lock-enabled-values true \
  --pressure-touch-lock-min-rr-values 1.0 \
  --pressure-touch-lock-buffer-pct-values 0.03 \
  --pressure-touch-lock-atr-multiplier-values 0.0 \
  --pressure-touch-lock-requires-touch-values false \
  --top 1 \
  --output var/high_leverage_expansion/pressure_approach_lock_retarget_best_full.json
