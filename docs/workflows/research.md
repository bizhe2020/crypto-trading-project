# Research / Parameter Optimization

Scope:

- candidate generation
- score-gated SOTA filtering
- Stable reverse-short parameter search
- trailing mode sensitivity

Canonical entrypoints:

- `bash scripts/workflows/research/run_confirmed_score_gate.sh`
- `bash scripts/workflows/research/run_stable_reverse_short_scan.sh`
- `bash scripts/workflows/research/run_trailing_rr_scan.sh`

Primary scripts:

- `scripts/replay_confirmed_score_gate.py`
- `scripts/scan_confirmed_score_gates.py`
- `scripts/scan_stable_reverse_short_on_score_gated_sota.py`
- `scripts/scan_stable_smc_live_shadow_stable_params.py`
- `scripts/scan_stable_smc_live_shadow_shadow_sensitivity.py`
- `scripts/scan_trailing_rr_modes_live_shadow.py`

Current research rule:

- Research may generate candidates and reports.
- Promotion to `main` must happen only after replay/audit scripts confirm the live-aligned path.
- Historical branch notes in `docs/archive/` are not the current source of truth.
