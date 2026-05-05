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

- Deploy Tokyo only from `main`.
- Use `scripts/replay_stable_smc_live_shadow.py` as the canonical strategy replay entry.
- Use `scripts/audit_live_replay_trade_convergence.py` and `scripts/replay_tokyo_full_snapshot_anchor.py` for replay/live convergence work.
- Treat root-level historical research notes as archived; current operator docs live under `docs/workflows/`.

## Quick Start

- Live/Tokyo: `bash scripts/workflows/live/deploy_tokyo.sh`
- Replay/Audit: `bash scripts/workflows/replay/run_live_shadow.sh`
- Research/Optimization: `bash scripts/workflows/research/run_stable_reverse_short_scan.sh`

Use the workflow docs for the exact command set and current expectations.
