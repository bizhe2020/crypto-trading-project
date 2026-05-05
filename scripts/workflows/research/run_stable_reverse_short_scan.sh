#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

exec python3 scripts/scan_stable_reverse_short_on_score_gated_sota.py \
  --start-date 2022-01-01 \
  --confirmed-4h-only \
  --replay-sync-entry-to-signal-price \
  "$@"
