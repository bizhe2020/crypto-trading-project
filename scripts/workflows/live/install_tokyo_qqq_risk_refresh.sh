#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

TOKYO_HOST="${TOKYO_HOST:-23.106.133.251}"
TOKYO_USER="${TOKYO_USER:-root}"
TOKYO_ROUTER_SERVICE="${TOKYO_ROUTER_SERVICE:-crypto-strategy-router}"
TOKYO_PROJECT_DIR="${TOKYO_PROJECT_DIR:-}"
TOKYO_PYTHON_BIN="${TOKYO_PYTHON_BIN:-}"
TOKYO_RISK_REFRESH_ARGS="${TOKYO_RISK_REFRESH_ARGS:---macro-refresh-mode market_proxy}"
RUN_REFRESH_ONCE="${RUN_REFRESH_ONCE:-1}"
ENABLE_TIMER="${ENABLE_TIMER:-1}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"

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
if [[ -z "$TOKYO_PROJECT_DIR" || -z "$TOKYO_PYTHON_BIN" ]]; then
  SERVICE_UNIT="$("${SSH_CMD[@]}" "$TARGET" "systemctl cat '${TOKYO_ROUTER_SERVICE}' 2>/dev/null || true")"
fi
if [[ -z "$TOKYO_PROJECT_DIR" ]]; then
  DETECTED_PROJECT_DIR="$(printf '%s\n' "$SERVICE_UNIT" | sed -n 's/^WorkingDirectory=//p' | tail -n 1)"
  TOKYO_PROJECT_DIR="${DETECTED_PROJECT_DIR:-/root/projects/crypto-trading-project}"
fi
if [[ -z "$TOKYO_PYTHON_BIN" ]]; then
  DETECTED_PYTHON_BIN="$(printf '%s\n' "$SERVICE_UNIT" | sed -n 's/^ExecStart=\([^ ]*\) .*$/\1/p' | tail -n 1)"
  TOKYO_PYTHON_BIN="${DETECTED_PYTHON_BIN:-${TOKYO_PROJECT_DIR}/.venv/bin/python}"
fi

FILES=("$@")
if [[ ${#FILES[@]} -eq 0 ]]; then
  FILES=(
    "requirements.txt"
    "scripts/fetch_public_etf_history.py"
    "scripts/fetch_qqq_constituent_breadth.py"
    "scripts/fetch_fred_macro_indicators.py"
    "scripts/qqq_drawdown_risk_model.py"
    "scripts/qqq_risk_runtime_generation.py"
    "scripts/generate_qqq_recent_risk_csv.py"
    "scripts/generate_qqq_long_cycle_risk_csv.py"
    "scripts/refresh_qqq_risk_runtime_inputs.py"
    "ops/systemd/qqq-risk-refresh.service"
    "ops/systemd/qqq-risk-refresh.timer"
  )
fi

for file in "${FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing local deploy file: $file" >&2
    exit 1
  fi
done

BACKUP_DIR="var/backups/qqq_risk_refresh_install_$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_SERVICE_PATH="/etc/systemd/system/qqq-risk-refresh.service"
REMOTE_TIMER_PATH="/etc/systemd/system/qqq-risk-refresh.timer"

echo "Target: ${TARGET}:${TOKYO_PROJECT_DIR}"
echo "Router service for autodetect: ${TOKYO_ROUTER_SERVICE}"
echo "Python: ${TOKYO_PYTHON_BIN}"
echo "Risk refresh args: ${TOKYO_RISK_REFRESH_ARGS}"
echo "Backup: ${BACKUP_DIR}"
echo "Files:"
printf '  %s\n' "${FILES[@]}"

"${SSH_CMD[@]}" "$TARGET" "mkdir -p '${TOKYO_PROJECT_DIR}/${BACKUP_DIR}' '${TOKYO_PROJECT_DIR}/var/log'"

for file in "${FILES[@]}"; do
  backup_name="${file//\//__}"
  remote_dir="$(dirname "${TOKYO_PROJECT_DIR}/${file}")"
  "${SSH_CMD[@]}" "$TARGET" "mkdir -p '${remote_dir}'"
  "${SSH_CMD[@]}" "$TARGET" "if [ -f '${TOKYO_PROJECT_DIR}/${file}' ]; then cp '${TOKYO_PROJECT_DIR}/${file}' '${TOKYO_PROJECT_DIR}/${BACKUP_DIR}/${backup_name}'; fi"
  "${SCP_CMD[@]}" "$file" "${TARGET}:${TOKYO_PROJECT_DIR}/${file}"
done

"${SSH_CMD[@]}" "$TARGET" "if [ -f '${REMOTE_SERVICE_PATH}' ]; then cp '${REMOTE_SERVICE_PATH}' '${TOKYO_PROJECT_DIR}/${BACKUP_DIR}/systemd__qqq-risk-refresh.service'; fi"
"${SSH_CMD[@]}" "$TARGET" "if [ -f '${REMOTE_TIMER_PATH}' ]; then cp '${REMOTE_TIMER_PATH}' '${TOKYO_PROJECT_DIR}/${BACKUP_DIR}/systemd__qqq-risk-refresh.timer'; fi"

if [[ "$INSTALL_REQUIREMENTS" == "1" ]]; then
  "${SSH_CMD[@]}" "$TARGET" "'${TOKYO_PYTHON_BIN}' -m pip install -r '${TOKYO_PROJECT_DIR}/requirements.txt'"
fi

"${SSH_CMD[@]}" "$TARGET" "'${TOKYO_PYTHON_BIN}' - <<'PY'
import importlib
for name in ['lightgbm', 'pandas', 'pyarrow', 'requests']:
    importlib.import_module(name)
print('python_deps_ok')
PY"

"${SSH_CMD[@]}" "$TARGET" "cd '${TOKYO_PROJECT_DIR}' && '${TOKYO_PYTHON_BIN}' -m py_compile \
  scripts/fetch_public_etf_history.py \
  scripts/fetch_qqq_constituent_breadth.py \
  scripts/fetch_fred_macro_indicators.py \
  scripts/qqq_drawdown_risk_model.py \
  scripts/qqq_risk_runtime_generation.py \
  scripts/generate_qqq_recent_risk_csv.py \
  scripts/generate_qqq_long_cycle_risk_csv.py \
  scripts/refresh_qqq_risk_runtime_inputs.py"

"${SSH_CMD[@]}" "$TARGET" "sed \
  -e 's|__PROJECT_DIR__|${TOKYO_PROJECT_DIR}|g' \
  -e 's|__PYTHON_BIN__|${TOKYO_PYTHON_BIN}|g' \
  -e 's|__RISK_REFRESH_ARGS__|${TOKYO_RISK_REFRESH_ARGS}|g' \
  '${TOKYO_PROJECT_DIR}/ops/systemd/qqq-risk-refresh.service' > '${REMOTE_SERVICE_PATH}'"
"${SSH_CMD[@]}" "$TARGET" "cp '${TOKYO_PROJECT_DIR}/ops/systemd/qqq-risk-refresh.timer' '${REMOTE_TIMER_PATH}'"
"${SSH_CMD[@]}" "$TARGET" "systemctl daemon-reload"

if [[ "$RUN_REFRESH_ONCE" == "1" ]]; then
  "${SSH_CMD[@]}" "$TARGET" "systemctl start qqq-risk-refresh.service"
  "${SSH_CMD[@]}" "$TARGET" "systemctl show qqq-risk-refresh.service -p Result -p ExecMainStatus --no-pager"
fi

if [[ "$ENABLE_TIMER" == "1" ]]; then
  "${SSH_CMD[@]}" "$TARGET" "systemctl enable --now qqq-risk-refresh.timer"
  "${SSH_CMD[@]}" "$TARGET" "systemctl list-timers qqq-risk-refresh.timer --no-pager"
else
  echo "ENABLE_TIMER=${ENABLE_TIMER}; timer not enabled."
fi

echo "Installed qqq-risk-refresh units on ${TARGET}"
echo "Service: ${REMOTE_SERVICE_PATH}"
echo "Timer:   ${REMOTE_TIMER_PATH}"
