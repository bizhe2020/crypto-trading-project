#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$ROOT_DIR/var/log"
DATA_DIR="$ROOT_DIR/var/funding_oi/recorded"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
LABEL="com.bizhe.crypto.funding-oi-recorder"
PLIST_PATH="$LAUNCH_AGENTS_DIR/$LABEL.plist"
LOG_FILE="$LOG_DIR/funding_oi_recorder.log"
USER_ID="$(id -u)"
PYTHON_BIN="$(command -v python3)"

mkdir -p "$LOG_DIR" "$DATA_DIR" "$LAUNCH_AGENTS_DIR"

write_plist() {
  cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$ROOT_DIR/deployment/bot/run_funding_oi_recorder.sh</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$ROOT_DIR</string>
  <key>KeepAlive</key>
  <true/>
  <key>RunAtLoad</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$LOG_FILE</string>
  <key>StandardErrorPath</key>
  <string>$LOG_FILE</string>
</dict>
</plist>
EOF
}

start() {
  write_plist
  launchctl bootout "gui/$USER_ID" "$PLIST_PATH" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/$USER_ID" "$PLIST_PATH"
  launchctl kickstart -k "gui/$USER_ID/$LABEL"
  echo "started label=$LABEL plist=$PLIST_PATH log=$LOG_FILE data_dir=$DATA_DIR"
}

stop() {
  if [[ -f "$PLIST_PATH" ]]; then
    launchctl bootout "gui/$USER_ID" "$PLIST_PATH" >/dev/null 2>&1 || true
  fi
  echo "stopped label=$LABEL"
}

status() {
  if launchctl print "gui/$USER_ID/$LABEL" >/dev/null 2>&1; then
    echo "running label=$LABEL plist=$PLIST_PATH log=$LOG_FILE data_dir=$DATA_DIR"
  else
    echo "not_running label=$LABEL plist=$PLIST_PATH log=$LOG_FILE"
    return 1
  fi
}

tail_log() {
  if [[ -f "$LOG_FILE" ]]; then
    tail -n 20 "$LOG_FILE"
  else
    echo "log_missing path=$LOG_FILE"
  fi
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop || true; start ;;
  status) status ;;
  tail) tail_log ;;
  *)
    echo "usage: $0 {start|stop|restart|status|tail}"
    exit 1
    ;;
esac
