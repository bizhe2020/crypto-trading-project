#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

TOKYO_HOST="${TOKYO_HOST:-23.106.133.251}"
TOKYO_USER="${TOKYO_USER:-root}"
TOKYO_SERVICE="${TOKYO_SERVICE:-crypto-strategy-router}"
TOKYO_PROJECT_DIR="${TOKYO_PROJECT_DIR:-}"
TOKYO_ROUTER_CONFIG_PATH="${TOKYO_ROUTER_CONFIG_PATH:-}"
TOKYO_RISK_TIMER="${TOKYO_RISK_TIMER:-qqq-risk-refresh.timer}"

DEPLOY_REMOTE="${DEPLOY_REMOTE:-origin}"
DEPLOY_REF="${DEPLOY_REF:-$(git rev-parse --abbrev-ref HEAD)}"
PUSH_FIRST="${PUSH_FIRST:-0}"
REQUIRE_CLEAN_LOCAL="${REQUIRE_CLEAN_LOCAL:-1}"
REMOTE_STASH_DIRTY="${REMOTE_STASH_DIRTY:-1}"
SYNC_ROUTER_LIVE_CONFIG="${SYNC_ROUTER_LIVE_CONFIG:-0}"
RESTART_SERVICE="${RESTART_SERVICE:-1}"
RESTART_RISK_TIMER="${RESTART_RISK_TIMER:-0}"

SSH_BASE=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
if [[ -n "${SSHPASS:-}" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "SSHPASS is set but sshpass is not installed." >&2
    exit 1
  fi
  SSH_CMD=(sshpass -e "${SSH_BASE[@]}")
else
  SSH_CMD=("${SSH_BASE[@]}" -o BatchMode=yes)
fi

if [[ "$DEPLOY_REF" == "HEAD" ]]; then
  echo "DEPLOY_REF resolved to detached HEAD; pass a branch name or tag." >&2
  exit 1
fi

if [[ "$REQUIRE_CLEAN_LOCAL" == "1" ]] && [[ -n "$(git status --porcelain)" ]]; then
  echo "Local worktree is dirty. Commit/stash changes before git deploy, or set REQUIRE_CLEAN_LOCAL=0." >&2
  git status --short
  exit 1
fi

if [[ "$PUSH_FIRST" == "1" ]]; then
  git push "$DEPLOY_REMOTE" "$DEPLOY_REF"
fi

TARGET="${TOKYO_USER}@${TOKYO_HOST}"
SERVICE_UNIT=""
if [[ -z "$TOKYO_PROJECT_DIR" || -z "$TOKYO_ROUTER_CONFIG_PATH" ]]; then
  SERVICE_UNIT="$("${SSH_CMD[@]}" "$TARGET" "systemctl cat '${TOKYO_SERVICE}' 2>/dev/null || true")"
fi
if [[ -z "$TOKYO_PROJECT_DIR" ]]; then
  DETECTED_PROJECT_DIR="$(printf '%s\n' "$SERVICE_UNIT" | sed -n 's/^WorkingDirectory=//p' | tail -n 1)"
  TOKYO_PROJECT_DIR="${DETECTED_PROJECT_DIR:-/root/projects/crypto-trading-releases/router-risk-20260603_1a2a61f}"
fi
if [[ -z "$TOKYO_ROUTER_CONFIG_PATH" ]]; then
  DETECTED_ROUTER_CONFIG_PATH="$(printf '%s\n' "$SERVICE_UNIT" | sed -n 's/^ExecStart=.*--config \([^ ]*\).*$/\1/p' | tail -n 1)"
  TOKYO_ROUTER_CONFIG_PATH="${DETECTED_ROUTER_CONFIG_PATH:-${TOKYO_PROJECT_DIR}/config/config.live.strategy-router.json}"
fi

BACKUP_DIR="var/backups/git_deploy_tokyo_$(date -u +%Y%m%dT%H%M%SZ)"

echo "Target: ${TARGET}:${TOKYO_PROJECT_DIR}"
echo "Service: ${TOKYO_SERVICE}"
echo "Router config: ${TOKYO_ROUTER_CONFIG_PATH}"
echo "Deploy ref: ${DEPLOY_REMOTE}/${DEPLOY_REF}"
echo "Remote backup: ${BACKUP_DIR}"
echo "Sync live config: ${SYNC_ROUTER_LIVE_CONFIG}"

"${SSH_CMD[@]}" "$TARGET" "set -e
cd '${TOKYO_PROJECT_DIR}'
mkdir -p '${BACKUP_DIR}'
git status --short --branch > '${BACKUP_DIR}/git_status_before.txt'
git diff --binary > '${BACKUP_DIR}/git_diff_before.patch' || true
if [ -n \"\$(git status --porcelain)\" ]; then
  if [ '${REMOTE_STASH_DIRTY}' = '1' ]; then
    if git status --porcelain -- config/config.paper.qqq-usdt-aggressive-runtime.json | grep -q '^?? '; then
      cp config/config.paper.qqq-usdt-aggressive-runtime.json '${BACKUP_DIR}/config.paper.qqq-usdt-aggressive-runtime.json.untracked'
      rm config/config.paper.qqq-usdt-aggressive-runtime.json
    fi
    git stash push -m 'pre_git_deploy_${BACKUP_DIR##*/}' || true
  else
    echo 'Remote worktree is dirty. Set REMOTE_STASH_DIRTY=1 or clean it first.' >&2
    git status --short
    exit 1
  fi
fi
git fetch '${DEPLOY_REMOTE}'
git checkout '${DEPLOY_REF}'
git pull --ff-only '${DEPLOY_REMOTE}' '${DEPLOY_REF}'
"

"${SSH_CMD[@]}" "$TARGET" "set -e
cd '${TOKYO_PROJECT_DIR}'
python3 -m json.tool config/config.paper.qqq-usdt-aggressive-frozen.json >/dev/null
python3 -m json.tool config/config.paper.qqq-usdt-aggressive-runtime.json >/dev/null
python3 -m json.tool config/config.live.strategy-router.template.json >/dev/null
python3 -m json.tool config/config.paper.googl-high-leverage-runtime.json >/dev/null
python3 -m json.tool config/config.paper.googl-high-leverage-frozen.json >/dev/null
if [ '${SYNC_ROUTER_LIVE_CONFIG}' = '1' ]; then
  cp '${TOKYO_ROUTER_CONFIG_PATH}' '${BACKUP_DIR}/config.live.strategy-router.json.before'
  cp config/config.live.strategy-router.template.json '${TOKYO_ROUTER_CONFIG_PATH}'
fi
python3 -m json.tool '${TOKYO_ROUTER_CONFIG_PATH}' >/dev/null
PYTHON_BIN=\"\$(systemctl cat '${TOKYO_SERVICE}' 2>/dev/null | sed -n 's/^ExecStart=\\([^ ]*\\).*$/\\1/p' | tail -n 1)\"
if [ -z \"\$PYTHON_BIN\" ]; then PYTHON_BIN='.venv/bin/python'; fi
if [ ! -x \"\$PYTHON_BIN\" ]; then
  echo \"Missing systemd python executable: \$PYTHON_BIN\" >&2
  exit 1
fi
\"\$PYTHON_BIN\" -m py_compile \
  bot/strategy_router.py \
  bot/router_executor.py \
  bot/okx_executor.py \
  bot/qqq_shadow_gate.py \
  bot/qqq_usdt_executor.py \
  bot/qqq_usdt_signal_adapter.py \
  bot/googl_usdt_signal_adapter.py \
  scripts/fetch_public_etf_history.py \
  scripts/fetch_googl_daily_prices.py \
  scripts/refresh_qqq_risk_runtime_inputs.py \
  scripts/scan_qqq_usdt_4h_triggers.py \
  scripts/scan_googl_daily_signal.py
"

if [[ "$RESTART_SERVICE" == "1" ]]; then
  "${SSH_CMD[@]}" "$TARGET" "set -e
systemctl restart '${TOKYO_SERVICE}'
for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
  state=\$(systemctl is-active '${TOKYO_SERVICE}' || true)
  if [ \"\$state\" = 'active' ]; then
    echo \"\$state\"
    exit 0
  fi
  sleep 5
done
systemctl is-active '${TOKYO_SERVICE}'
"
else
  echo "RESTART_SERVICE=${RESTART_SERVICE}; not restarting ${TOKYO_SERVICE}."
fi

if [[ "$RESTART_RISK_TIMER" == "1" ]]; then
  "${SSH_CMD[@]}" "$TARGET" "systemctl restart '${TOKYO_RISK_TIMER}' && systemctl is-active '${TOKYO_RISK_TIMER}'"
else
  "${SSH_CMD[@]}" "$TARGET" "systemctl is-active '${TOKYO_RISK_TIMER}' 2>/dev/null || true"
fi

"${SSH_CMD[@]}" "$TARGET" "set -e
cd '${TOKYO_PROJECT_DIR}'
sleep 5
printf '%s\n' '--- router heartbeat ---'
cat state/strategy_router_live.json.heartbeat 2>/dev/null || true
printf '%s\n' '--- latest bootstrap/evaluate ---'
python3 - <<'PY'
import json
from pathlib import Path

items = []
for line in reversed(Path('live_strategy_router.log').read_text(errors='ignore').splitlines()):
    try:
        data = json.loads(line)
    except Exception:
        continue
    if data.get('event') not in {'bootstrap', 'evaluate'}:
        continue
    route = data.get('route') or {}
    qqq = data.get('qqq') or {}
    btc = data.get('btc') or {}
    items.append({
        'event': data.get('event'),
        'status': data.get('status'),
        'decision_reason': route.get('decision_reason'),
        'selected_strategy': route.get('selected_strategy'),
        'btc_bootstrap_error': btc.get('bootstrap_error'),
        'qqq_error': qqq.get('error'),
        'qqq_bootstrap_step': qqq.get('bootstrap_step'),
        'qqq_market_loaded': qqq.get('market_loaded'),
    })
    if len(items) >= 4:
        break
for item in reversed(items):
    print(json.dumps(item, ensure_ascii=False))
PY
"
