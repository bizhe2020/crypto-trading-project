#!/usr/bin/env bash
# 东京实盘部署：git 推送 → 服务器 ff-only 拉取 → 编译检查 → （可选）重启服务。
#
# 用法：
#   bash scripts/ops/deploy_tokyo.sh                      # 只拉代码 + 编译检查，不重启
#   RESTART="trend-basket" bash scripts/ops/deploy_tokyo.sh
#   RESTART="btc-scalp-standalone trend-basket" bash scripts/ops/deploy_tokyo.sh
#
# 重启 btc-scalp-standalone 前先确认空仓（bash scripts/ops/tokyo_status.sh）。
# 每次部署都会在服务器 var/backups/deploy_<时间>/ 备份两个服务的配置与状态库。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TOKYO_HOST="${TOKYO_HOST:-23.106.133.251}"
TOKYO_USER="${TOKYO_USER:-root}"
PROJECT_DIR="${PROJECT_DIR:-/root/projects/crypto-trading-releases/router-risk-20260603_1a2a61f}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
RESTART="${RESTART:-}"
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o BatchMode=yes "${TOKYO_USER}@${TOKYO_HOST}")

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "本地有未提交的改动，先提交再部署。" >&2
  git status --short --untracked-files=no
  exit 1
fi
git push -q origin "$BRANCH"

"${SSH[@]}" bash -s <<REMOTE
set -euo pipefail
cd '${PROJECT_DIR}'
B="var/backups/deploy_\$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "\$B"
cp -p config/config.live.high-leverage-structure.json config/config.live.trend-basket.json "\$B"/ 2>/dev/null || true
cp -p state/runtime_high_leverage_structure_live.db state/trend_basket_live.db "\$B"/ 2>/dev/null || true
git rev-parse HEAD > "\$B/head_before.txt"
git fetch -q origin
git checkout -q '${BRANCH}'
git pull -q --ff-only origin '${BRANCH}'
echo "HEAD: \$(git log --oneline -1)"
.venv/bin/python -m py_compile bot/run_bot.py bot/okx_executor.py bot/trend_basket_executor.py \
  strategy/scalp_robust_v2_core.py strategy/trend_vote.py
for s in ${RESTART}; do
  systemctl restart "\$s"
done
sleep 5
for s in btc-scalp-standalone trend-basket; do
  echo "\$s: \$(systemctl is-active \$s)"
done
echo "backup: \$B"
REMOTE
