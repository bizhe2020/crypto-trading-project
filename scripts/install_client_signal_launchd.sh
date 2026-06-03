#!/bin/zsh
set -euo pipefail

ROOT="/Users/laoji/projects/crypto-trading-project"
LAUNCH_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$LAUNCH_DIR" "$ROOT/var/log"

for file in \
  com.bizhe.cn-etf-513100-preopen-signal.plist \
  com.bizhe.cn-etf-513100-close-signal.plist \
  com.bizhe.tqqq-sqqq-preopen-signal.plist \
  com.bizhe.tqqq-sqqq-close-signal.plist
do
  cp "$ROOT/ops/launchd/$file" "$LAUNCH_DIR/"
  launchctl unload "$LAUNCH_DIR/$file" >/dev/null 2>&1 || true
  launchctl load "$LAUNCH_DIR/$file"
done

echo "loaded:"
echo "  com.bizhe.cn-etf-513100-preopen-signal"
echo "  com.bizhe.cn-etf-513100-close-signal"
echo "  com.bizhe.tqqq-sqqq-preopen-signal"
echo "  com.bizhe.tqqq-sqqq-close-signal"
