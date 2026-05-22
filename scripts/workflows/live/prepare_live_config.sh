#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

echo "The CN ETF branch uses checked-in configs:"
echo "  config/config.paper.cn-nasdaq100-etf.json"
echo "  config/config.live-shadow.cn-nasdaq100-etf.json"
