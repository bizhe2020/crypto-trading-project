#!/usr/bin/env bash
# 只读查看东京实盘：两个服务状态、交易所持仓/权益、各自最近日志。不做任何写操作。
#   bash scripts/ops/tokyo_status.sh
set -euo pipefail

TOKYO_HOST="${TOKYO_HOST:-23.106.133.251}"
TOKYO_USER="${TOKYO_USER:-root}"
PROJECT_DIR="${PROJECT_DIR:-/root/projects/crypto-trading-releases/router-risk-20260603_1a2a61f}"

ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o BatchMode=yes "${TOKYO_USER}@${TOKYO_HOST}" bash -s <<REMOTE
set -u
cd '${PROJECT_DIR}'
date -u
echo "HEAD: \$(git log --oneline -1)"
for s in btc-scalp-standalone trend-basket; do
  echo "\$s: \$(systemctl is-active \$s) since \$(systemctl show -p ActiveEnterTimestamp --value \$s)"
done
.venv/bin/python - <<'PY'
import json
from bot.okx_client import OkxClient, OkxCredentials
c = json.load(open("config/config.live.high-leverage-structure.json"))
ex = OkxClient(OkxCredentials(c["api_key"], c["api_secret"], c["api_passphrase"]), trading_mode="live", proxy=c.get("proxy")).exchange
bal = ex.privateGetAccountBalance()["data"][0]
usdt = next(d for d in bal["details"] if d["ccy"] == "USDT")
print(f"权益 {float(bal['totalEq']):.2f} USDT  可用 {float(usdt['availBal']):.2f}")
for p in ex.privateGetAccountPositions()["data"]:
    print(f"  {p['instId']:<16} {p['posSide']:<5} {p['pos']:>8}  均价 {p['avgPx']}  {p['lever']}x {p['mgnMode']}"
          f"  保证金 {float(p['margin'] or 0):.1f}  浮盈 {float(p['upl'] or 0):.2f}  强平 {p['liqPx']}")
PY
echo "--- scalp 最近日志 ---"; tail -n 3 var/log/btc_scalp_standalone.log | cut -c1-200
echo "--- 篮子最近日志 ---"; tail -n 2 var/log/trend_basket.log | cut -c1-300
REMOTE
