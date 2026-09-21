#!/usr/bin/env bash
# 只读查询东京服务器上 BTC scalp 腿的当前状态（服务 / 持仓 / 最后处理 K 线）。
#
# 前提：本机公钥已加入服务器 authorized_keys。
#   ssh-copy-id -i ~/.ssh/id_ed25519.pub root@23.106.133.251
#
# 用法：
#   bash scripts/workflows/live/check_tokyo_btc_position.sh
#
# 本脚本**不做任何写操作**：只读 systemd 状态、只读 sqlite、只读日志尾部。
set -euo pipefail

TOKYO_HOST="${TOKYO_HOST:-23.106.133.251}"
TOKYO_USER="${TOKYO_USER:-root}"
BTC_SERVICE="${BTC_SERVICE:-btc-scalp-standalone}"
PROJECT_DIR="${PROJECT_DIR:-/root/projects/crypto-trading-releases/router-risk-20260603_1a2a61f}"
CONFIG_REL="${CONFIG_REL:-config/config.live.high-leverage-structure.json}"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o BatchMode=yes)

echo "== ${TOKYO_USER}@${TOKYO_HOST}  ${BTC_SERVICE} =="

ssh "${SSH_OPTS[@]}" "${TOKYO_USER}@${TOKYO_HOST}" bash -s <<REMOTE
set -u
echo "--- 服务器时间 ---"
date -u

echo
echo "--- 服务状态 ---"
systemctl is-active ${BTC_SERVICE} 2>/dev/null || true
systemctl show -p ActiveEnterTimestamp -p NRestarts ${BTC_SERVICE} 2>/dev/null || true

cd ${PROJECT_DIR} 2>/dev/null || { echo "!! 项目目录不存在: ${PROJECT_DIR}"; exit 0; }

DB=\$(python3 - <<'PY' 2>/dev/null
import json
try:
    print(json.load(open("${CONFIG_REL}")).get("state_db_path",""))
except Exception:
    print("")
PY
)
DB="\${DB:-state/runtime_high_leverage_structure_live.db}"

echo
echo "--- 持仓 / 处理进度（state DB: \$DB）---"
if [ -f "\$DB" ]; then
  sqlite3 "\$DB" <<'SQL' 2>/dev/null || echo "!! 读取失败（sqlite3 缺失或表结构不同）"
.mode line
select value from bot_state where key='last_processed_candle_time';
select value from bot_state where key='strategy_snapshot';
SQL
else
  echo "!! 找不到 state DB: \$DB"
fi

echo
echo "--- 最近 25 条 action_log ---"
if [ -f "\$DB" ]; then
  sqlite3 "\$DB" "select id, action_type, created_at from action_log order by id desc limit 25;" 2>/dev/null || true
fi

echo
echo "--- 日志尾部 ---"
journalctl -u ${BTC_SERVICE} -n 25 --no-pager 2>/dev/null || tail -25 var/log/*.log 2>/dev/null || echo "(无日志)"
REMOTE
