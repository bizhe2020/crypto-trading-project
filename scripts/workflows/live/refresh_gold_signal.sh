#!/usr/bin/env bash
# 每日重算黄金日线信号（MA50>MA100 金叉，供 router 影子模式消费）。
# 自闭环：scan_gold_daily_signal.py 内联从 OKX 拉 XAU-USDT-SWAP 日线并算信号，
# 无需本地 scp、无额外数据依赖。
set -euo pipefail
REPO=/root/projects/crypto-trading-releases/router-risk-20260603_1a2a61f
PY="$REPO/.venv/bin/python"
cd "$REPO"
"$PY" scripts/scan_gold_daily_signal.py \
  --out var/runtime/gold/gold_daily_signal.csv
echo "[$(date -u +%FT%TZ)] gold signal refreshed (self-contained)"
