#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 scripts/scan_shadow_on_fixed_high_leverage.py \
  --config config/config.live.5x-3pct.json \
  --data-15m data/okx/futures/BTC_USDT_USDT-15m-futures.feather \
  --data-4h data/okx/futures/BTC_USDT_USDT-4h-futures.feather \
  --start-date 2022-01-01 \
  --daily-loss-values 6 \
  --equity-dd-values 15 \
  --equity-cooldown-values 2 \
  --loss-streak-values 0 \
  --top 1 \
  --output var/high_leverage_expansion/shadow_on_fixed_structure_best.json
