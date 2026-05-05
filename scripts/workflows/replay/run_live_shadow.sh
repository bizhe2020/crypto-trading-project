#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

exec python3 scripts/replay_sota_smc_live_shadow.py \
  --start-date 2022-01-01 \
  --confirmed-4h-only \
  --replay-sync-entry-to-signal-price \
  "$@"
