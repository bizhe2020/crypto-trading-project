#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-23.106.133.251}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_DIR="${REMOTE_DIR:-/root/projects/crypto-trading-project}"

if [[ -z "${TOKYO_PASS:-}" ]]; then
  echo "TOKYO_PASS is required"
  exit 1
fi

if ! command -v sshpass >/dev/null 2>&1; then
  echo "sshpass is required"
  exit 1
fi

RSYNC_RSH="sshpass -p $TOKYO_PASS ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password -o PubkeyAuthentication=no"

rsync -az \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'deployment/' \
  --exclude 'data/' \
  --exclude 'state/' \
  --exclude 'var/' \
  -e "$RSYNC_RSH" \
  "$ROOT_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

sshpass -p "$TOKYO_PASS" ssh \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  "$REMOTE_USER@$REMOTE_HOST" \
  "cd '$REMOTE_DIR' && zsh scripts/bootstrap_server.sh && systemctl restart crypto-trading-bot-5x3pct.service"

echo "deploy_complete host=$REMOTE_HOST dir=$REMOTE_DIR"
