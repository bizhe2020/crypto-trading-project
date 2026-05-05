# Replay / Audit

Scope:

- canonical replay for the current strategy stack
- live-shadow chronological arbitration replay
- Tokyo snapshot replay
- live-vs-replay convergence and stop/trailing gap audits

Canonical entrypoints:

- `bash scripts/workflows/replay/run_live_shadow.sh`
- `bash scripts/workflows/replay/run_live_readiness.sh`
- `bash scripts/workflows/replay/run_live_replay_audit.sh`
- `bash scripts/workflows/replay/run_tokyo_snapshot_anchor.sh`

Primary scripts:

- `scripts/replay_sota_smc_live_shadow.py`
- `scripts/live_readiness_report.py`
- `scripts/audit_live_replay_trade_convergence.py`
- `scripts/replay_tokyo_full_snapshot_anchor.py`
- `scripts/audit_tokyo_replay_stop_gap.py`
- `scripts/live_vs_replay_audit.py`

Current replay rule:

- Prefer `confirmed 4h` plus `replay_sync_entry_to_signal_price` when the target is live parity.
- Use score-gated replay variants only when you explicitly want the filtered candidate path.

If replay and live diverge, start from this order:

1. `live_readiness_report.py`
2. `audit_live_replay_trade_convergence.py`
3. `replay_tokyo_full_snapshot_anchor.py`
4. `audit_tokyo_replay_stop_gap.py`
