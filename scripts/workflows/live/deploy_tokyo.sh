#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

TOKYO_HOST="${TOKYO_HOST:-23.106.133.251}"
TOKYO_USER="${TOKYO_USER:-root}"
TOKYO_SERVICE="${TOKYO_SERVICE:-crypto-strategy-router}"
TOKYO_PROJECT_DIR="${TOKYO_PROJECT_DIR:-}"
TOKYO_ROUTER_CONFIG_PATH="${TOKYO_ROUTER_CONFIG_PATH:-}"
RESTART_SERVICE="${RESTART_SERVICE:-1}"
SYNC_ROUTER_LIVE_CONFIG="${SYNC_ROUTER_LIVE_CONFIG:-1}"

SSH_BASE=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
SCP_BASE=(scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
if [[ -n "${SSHPASS:-}" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "SSHPASS is set but sshpass is not installed." >&2
    exit 1
  fi
  SSH_CMD=(sshpass -e "${SSH_BASE[@]}")
  SCP_CMD=(sshpass -e "${SCP_BASE[@]}")
else
  SSH_CMD=("${SSH_BASE[@]}" -o BatchMode=yes)
  SCP_CMD=("${SCP_BASE[@]}" -o BatchMode=yes)
fi

TARGET="${TOKYO_USER}@${TOKYO_HOST}"
SERVICE_UNIT=""
if [[ -z "$TOKYO_PROJECT_DIR" || -z "$TOKYO_ROUTER_CONFIG_PATH" ]]; then
  SERVICE_UNIT="$("${SSH_CMD[@]}" "$TARGET" "systemctl cat '${TOKYO_SERVICE}' 2>/dev/null || true")"
fi
if [[ -z "$TOKYO_PROJECT_DIR" ]]; then
  DETECTED_PROJECT_DIR="$(printf '%s\n' "$SERVICE_UNIT" | sed -n 's/^WorkingDirectory=//p' | tail -n 1)"
  TOKYO_PROJECT_DIR="${DETECTED_PROJECT_DIR:-/root/projects/crypto-trading-project}"
fi
if [[ -z "$TOKYO_ROUTER_CONFIG_PATH" ]]; then
  DETECTED_ROUTER_CONFIG_PATH="$(printf '%s\n' "$SERVICE_UNIT" | sed -n 's/^ExecStart=.*--config \([^ ]*\).*$/\1/p' | tail -n 1)"
  TOKYO_ROUTER_CONFIG_PATH="${DETECTED_ROUTER_CONFIG_PATH:-${TOKYO_PROJECT_DIR}/config/config.live.strategy-router.json}"
fi

FILES=("$@")
if [[ ${#FILES[@]} -eq 0 ]]; then
  FILES=(
    "bot/strategy_router.py"
    "bot/router_executor.py"
    "bot/qqq_usdt_executor.py"
    "bot/qqq_usdt_signal_adapter.py"
    "config/config.paper.qqq-usdt-aggressive-frozen.json"
    "config/config.live.strategy-router.template.json"
  )
fi

for file in "${FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing local deploy file: $file" >&2
    exit 1
  fi
done
BACKUP_DIR="var/backups/deploy_tokyo_$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_TEMPLATE_PATH="${TOKYO_PROJECT_DIR}/config/config.live.strategy-router.template.json"

echo "Target: ${TARGET}:${TOKYO_PROJECT_DIR}"
echo "Service: ${TOKYO_SERVICE}"
echo "Router config: ${TOKYO_ROUTER_CONFIG_PATH}"
echo "Backup: ${BACKUP_DIR}"
echo "Files:"
printf '  %s\n' "${FILES[@]}"

"${SSH_CMD[@]}" "$TARGET" "mkdir -p '${TOKYO_PROJECT_DIR}/${BACKUP_DIR}'"

for file in "${FILES[@]}"; do
  backup_name="${file//\//__}"
  "${SSH_CMD[@]}" "$TARGET" "cp '${TOKYO_PROJECT_DIR}/${file}' '${TOKYO_PROJECT_DIR}/${BACKUP_DIR}/${backup_name}'"
  "${SCP_CMD[@]}" "$file" "${TARGET}:${TOKYO_PROJECT_DIR}/${file}"
done

"${SSH_CMD[@]}" "$TARGET" "python3 -m json.tool '${TOKYO_PROJECT_DIR}/config/config.paper.qqq-usdt-aggressive-frozen.json' >/dev/null"
"${SSH_CMD[@]}" "$TARGET" "python3 -m json.tool '${TOKYO_PROJECT_DIR}/config/config.live.strategy-router.template.json' >/dev/null"
if [[ "$SYNC_ROUTER_LIVE_CONFIG" == "1" ]]; then
  "${SSH_CMD[@]}" "$TARGET" "if [ -f '${TOKYO_ROUTER_CONFIG_PATH}' ]; then cp '${TOKYO_ROUTER_CONFIG_PATH}' '${TOKYO_PROJECT_DIR}/${BACKUP_DIR}/config__config.live.strategy-router.json'; fi"
  "${SSH_CMD[@]}" "$TARGET" "cp '${REMOTE_TEMPLATE_PATH}' '${TOKYO_ROUTER_CONFIG_PATH}'"
fi
"${SSH_CMD[@]}" "$TARGET" "python3 -m json.tool '${TOKYO_ROUTER_CONFIG_PATH}' >/dev/null"
"${SSH_CMD[@]}" "$TARGET" "cd '${TOKYO_PROJECT_DIR}' && python3 -m py_compile bot/strategy_router.py bot/router_executor.py bot/qqq_usdt_executor.py bot/qqq_usdt_signal_adapter.py"

if [[ "$RESTART_SERVICE" == "1" ]]; then
  "${SSH_CMD[@]}" "$TARGET" "systemctl restart '${TOKYO_SERVICE}'"
  "${SSH_CMD[@]}" "$TARGET" "systemctl is-active '${TOKYO_SERVICE}'"
else
  echo "RESTART_SERVICE=${RESTART_SERVICE}; not restarting ${TOKYO_SERVICE}."
fi
