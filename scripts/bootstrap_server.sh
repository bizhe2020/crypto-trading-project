#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
SERVICE_NAME="${SERVICE_NAME:-crypto-trading-bot-5x3pct.service}"

link_if_possible() {
  local target="$1"
  local link_path="$2"
  if [[ -e "$link_path" && ! -L "$link_path" ]]; then
    return
  fi
  ln -sfn "$target" "$link_path"
}

mkdir -p \
  "$ROOT_DIR/state" \
  "$ROOT_DIR/data/okx/futures" \
  "$ROOT_DIR/var/log" \
  "$ROOT_DIR/var/funding_oi/recorded"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$ROOT_DIR/requirements.txt"

mkdir -p "$ROOT_DIR/deployment"
link_if_possible ../bot "$ROOT_DIR/deployment/bot"
link_if_possible ../config "$ROOT_DIR/deployment/config"
link_if_possible ../data "$ROOT_DIR/deployment/data"
link_if_possible ../state "$ROOT_DIR/deployment/state"
link_if_possible ../strategy "$ROOT_DIR/deployment/strategy"
link_if_possible ../systemd "$ROOT_DIR/deployment/systemd"

if [[ ! -f "$ROOT_DIR/deployment/__init__.py" ]]; then
  : > "$ROOT_DIR/deployment/__init__.py"
fi

if command -v systemctl >/dev/null 2>&1; then
  install -m 0644 "$ROOT_DIR/systemd/crypto-trading-bot-5x3pct.service" "/etc/systemd/system/$SERVICE_NAME"
  systemctl daemon-reload
  systemctl enable "$SERVICE_NAME"
fi

echo "bootstrap_complete root=$ROOT_DIR venv=$VENV_DIR service=$SERVICE_NAME"
