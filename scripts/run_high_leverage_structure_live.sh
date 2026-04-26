#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-config/config.live.high-leverage-structure.json}"
if [[ ! -f "$CONFIG" ]]; then
  echo "missing config: $CONFIG" >&2
  echo "create it from config/config.live.high-leverage-structure.template.json and fill live credentials" >&2
  exit 2
fi

python3 bot/run_bot.py \
  --config "$CONFIG" \
  --run-loop \
  --poll-interval-seconds "${POLL_INTERVAL_SECONDS:-5}" \
  --close-buffer-seconds "${CLOSE_BUFFER_SECONDS:-5}"
