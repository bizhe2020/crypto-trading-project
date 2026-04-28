#!/usr/bin/env bash
set -euo pipefail

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "high_leverage_10x_research" ]]; then
  echo "research_safety_check=skipped branch=$current_branch"
  exit 0
fi

protected_regex='^(systemd/|scripts/deploy_tokyo\.sh$|scripts/bootstrap_server\.sh$|config/config\.live.*\.json$|config/config\.live.*\.template\.json$)'

changed_files="$(
  {
    git diff --name-only
    git diff --name-only --cached
    git ls-files --others --exclude-standard
  } | sort -u
)"

if [[ -z "$changed_files" ]]; then
  echo "research_safety_check=ok branch=$current_branch changed=0"
  exit 0
fi

violations="$(printf '%s\n' "$changed_files" | grep -E "$protected_regex" || true)"
if [[ -n "$violations" ]]; then
  echo "research_safety_check=failed branch=$current_branch"
  echo "Protected production files changed on research branch:"
  printf '%s\n' "$violations"
  echo
  echo "Move deployment/live-config changes to main, or explicitly bypass this check after review."
  exit 1
fi

echo "research_safety_check=ok branch=$current_branch"
