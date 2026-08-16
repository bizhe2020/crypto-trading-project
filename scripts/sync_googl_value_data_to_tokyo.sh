#!/usr/bin/env bash
# 把 价值投资project 的 GOOGL 信号输入数据同步到东京服务器。
# 用法: scripts/sync_googl_value_data_to_tokyo.sh [TOKYO_HOST] [TOKYO_USER]
#   TOKYO_HOST 默认 23.106.133.251, TOKYO_USER 默认 root。
#   SSH 认证: 使用 SSHPASS 环境变量（密码）或已配置的 ssh 密钥。
#
# 同步内容:
#   /root/projects/value_data/prices.csv                 (GOOGL 日线, 前复权/raw 口径)
#   /root/projects/value_data/berkshire_13f_holdings.csv (伯克希尔 13F 持仓)
#
# 之后服务器端 crontab 每日重算 GOOGL 日线信号:
#   scripts/scan_googl_daily_signal.py -> var/runtime/googl/googl_daily_signal.csv
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKYO_HOST="${1:-${TOKYO_HOST:-23.106.133.251}}"
TOKYO_USER="${2:-${TOKYO_USER:-root}}"
REMOTE_DIR="/root/projects/value_data"
PRICES_CSV="/Users/laoji/projects/价值投资project/data/prices.csv"
HOLDINGS_CSV="/Users/laoji/projects/价值投资project/data/berkshire_13f_holdings.csv"

if [[ ! -f "$PRICES_CSV" || ! -f "$HOLDINGS_CSV" ]]; then
  echo "缺少本地数据文件:" >&2
  [[ -f "$PRICES_CSV" ]] || echo "  $PRICES_CSV" >&2
  [[ -f "$HOLDINGS_CSV" ]] || echo "  $HOLDINGS_CSV" >&2
  exit 1
fi

SSH_BASE=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
if [[ -n "${SSHPASS:-}" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "SSHPASS 已设置但 sshpass 未安装。" >&2
    exit 1
  fi
  SSH_CMD=(sshpass -e "${SSH_BASE[@]}")
  SCP_CMD=(sshpass -e scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
else
  SSH_CMD=("${SSH_BASE[@]}" -o BatchMode=yes)
  SCP_CMD=(scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
fi

TARGET="${TOKYO_USER}@${TOKYO_HOST}"

echo "创建远端目录: ${TARGET}:${REMOTE_DIR}"
"${SSH_CMD[@]}" "$TARGET" "mkdir -p '${REMOTE_DIR}'"

echo "同步 prices.csv..."
"${SCP_CMD[@]}" "$PRICES_CSV" "${TARGET}:${REMOTE_DIR}/prices.csv"
echo "同步 berkshire_13f_holdings.csv..."
"${SCP_CMD[@]}" "$HOLDINGS_CSV" "${TARGET}:${REMOTE_DIR}/berkshire_13f_holdings.csv"

echo "远端数据目录:"
"${SSH_CMD[@]}" "$TARGET" "ls -la '${REMOTE_DIR}'"
echo "完成。服务器端 crontab 每日重算信号:"
echo "  30 21 * * 1-5 /root/projects/value_data/refresh_googl_signal.sh"
