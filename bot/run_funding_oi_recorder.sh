#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$(command -v python3)"

export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:6244}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:6244}"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"

exec "$PYTHON_BIN" -u "$ROOT_DIR/bot/record_funding_oi.py" \
  --inst-id BTC-USDT-SWAP \
  --inst-type SWAP \
  --output-dir "$ROOT_DIR/var/funding_oi/recorded" \
  --file-prefix btc_funding_oi \
  --poll-interval-seconds 60 \
  --bucket-seconds 60 \
  --include-ticker \
  --mark-price-fallback
