#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 scripts/scan_pressure_level_trailing.py \
  --config config/config.live.5x-3pct.json \
  --data-15m data/okx/futures/BTC_USDT_USDT-15m-futures.feather \
  --data-4h data/okx/futures/BTC_USDT_USDT-4h-futures.feather \
  --start-date 2022-01-01 \
  --include-disabled-baseline \
  --pressure-min-rr-values 2.0 \
  --pressure-lock-rr-values 0.4 \
  --pressure-atr-multiplier-values 3.0 \
  --pressure-proximity-pct-values 0.15 \
  --pressure-rejection-min-rr-values 3.0 \
  --pressure-take-profit-on-rejection-values false \
  --pressure-enable-target-cap-values true \
  --pressure-target-min-rr-values 1.5 \
  --pressure-target-buffer-pct-values 0.05 \
  --pressure-regime-label-sets flat \
  --top 5 \
  --output var/high_leverage_expansion/pressure_target_cap_flat_scan_full.json
