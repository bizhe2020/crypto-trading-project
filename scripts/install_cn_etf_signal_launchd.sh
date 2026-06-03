#!/bin/zsh
set -euo pipefail

ROOT="/Users/laoji/projects/crypto-trading-project"
LAUNCH_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$LAUNCH_DIR"
mkdir -p "$ROOT/var/log"

cp "$ROOT/ops/launchd/com.bizhe.cn-etf-513100-preopen-signal.plist" "$LAUNCH_DIR/"
cp "$ROOT/ops/launchd/com.bizhe.cn-etf-513100-close-signal.plist" "$LAUNCH_DIR/"

launchctl unload "$LAUNCH_DIR/com.bizhe.cn-etf-513100-preopen-signal.plist" >/dev/null 2>&1 || true
launchctl unload "$LAUNCH_DIR/com.bizhe.cn-etf-513100-close-signal.plist" >/dev/null 2>&1 || true

launchctl load "$LAUNCH_DIR/com.bizhe.cn-etf-513100-preopen-signal.plist"
launchctl load "$LAUNCH_DIR/com.bizhe.cn-etf-513100-close-signal.plist"

echo "Installed:"
echo "  com.bizhe.cn-etf-513100-preopen-signal"
echo "  com.bizhe.cn-etf-513100-close-signal"
