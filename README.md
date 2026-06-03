# Crypto Trading Project

Current operator docs are intentionally minimal. Start from the frozen docs:

1. `docs/frozen_strategy_router_20260531.md`
2. `docs/qqq_usdt_aggressive_frozen.md`

Canonical command wrappers live under `scripts/workflows/`.

## Repo Layout

- `bot/`: live execution engine and exchange/runtime state handling.
- `strategy/`: strategy engine and signal/trailing logic.
- `scripts/`: shared replay, audit, and research modules.
- `scripts/workflows/`: stable entrypoints grouped by workflow.
- `systemd/`: Tokyo service definitions.
- `docs/`: latest frozen strategy/router docs only.

## Current Rules

- Tokyo deployments are branch-agnostic when an operator explicitly approves the deploy set; verify the active systemd repo path before syncing files.
- Use `scripts/replay_sota_smc_live_shadow.py` as the canonical strategy replay entry.
- Use `scripts/audit_live_replay_trade_convergence.py` and `scripts/replay_tokyo_full_snapshot_anchor.py` for replay/live convergence work.
- Historical research notes are archived outside the tracked docs tree; keep tracked docs focused on current frozen runtime state.

## Quick Start

- Live/Tokyo: `bash scripts/workflows/live/deploy_tokyo.sh`
  The deploy wrapper auto-detects the active `crypto-strategy-router` systemd repo path when possible and syncs the router live JSON from the uploaded template by default.
- QQQ risk refresh/Tokyo: `bash scripts/workflows/live/install_tokyo_qqq_risk_refresh.sh`
  This installs the daily support-data refresh plus recent/long-cycle risk CSV regeneration timer without touching the router live config.
- Replay/Audit: run the relevant `scripts/replay_*` or `scripts/audit_*` entrypoint for the frozen config under review.
- Research/Optimization: use the dedicated `scripts/scan_*` or `scripts/audit_*` module for the candidate line.

Use the frozen docs for current expectations and report references.
