# Crypto Trading Project

This repo now has three canonical workflows. Start from these paths instead of browsing historical research scripts directly:

1. `docs/workflows/live.md`
2. `docs/workflows/replay.md`
3. `docs/workflows/research.md`

Canonical command wrappers live under `scripts/workflows/`.

## Repo Layout

- `bot/`: live execution engine and exchange/runtime state handling.
- `strategy/`: strategy engine and signal/trailing logic.
- `scripts/`: shared replay, audit, and research modules.
- `scripts/workflows/`: stable entrypoints grouped by workflow.
- `systemd/`: Tokyo service definitions.
- `docs/archive/`: historical research notes kept only for reference.

## Current Rules

- Tokyo deployments are branch-agnostic when an operator explicitly approves the deploy set; verify the active systemd repo path before syncing files.
- Use `scripts/replay_sota_smc_live_shadow.py` as the canonical strategy replay entry.
- Use `scripts/audit_live_replay_trade_convergence.py` and `scripts/replay_tokyo_full_snapshot_anchor.py` for replay/live convergence work.
- Treat root-level historical research notes as archived; current operator docs live under `docs/workflows/`.

## Quick Start

- Live/Tokyo: `bash scripts/workflows/live/deploy_tokyo.sh`
  The deploy wrapper auto-detects the active `crypto-strategy-router` systemd repo path when possible and syncs the router live JSON from the uploaded template by default.
- QQQ risk refresh/Tokyo: `bash scripts/workflows/live/install_tokyo_qqq_risk_refresh.sh`
  This installs the daily support-data refresh plus recent/long-cycle risk CSV regeneration timer without touching the router live config.
- Replay/Audit: `bash scripts/workflows/replay/run_live_shadow.sh`
- Research/Optimization: `bash scripts/workflows/research/run_confirmed_score_gate.sh`

Use the workflow docs for the exact command set and current expectations.
