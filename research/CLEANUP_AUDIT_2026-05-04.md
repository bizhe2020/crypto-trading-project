# Cleanup Audit 2026-05-04

## Scope

This audit classifies the current `new_strategy_research` worktree drift into:

- `keep`: still needed for active research or formal reproduction
- `archive`: worth preserving, but should be moved out of the active top-level workflow
- `drop`: safe to delete after confirming no local-only notes depend on it

This is not a deletion patch. It is a decision document to avoid deleting active research context by accident.

## Current State

- Current branch: `new_strategy_research`
- Modified tracked files:
  - `.gitignore`
  - `scripts/scan_high_leverage_expansion.py`
- Untracked files/dirs: `47`

Important context:

- The current `72W` replay was produced by an untracked local `scripts/replay_stable_smc_live_shadow.py`.
- After aligning SMC fee/slippage with `main`, the same local replay now reproduces the formal `493036.53%` result.
- `main` already has tracked formal versions of:
  - `scripts/replay_stable_smc_live_shadow.py`
  - `scripts/smc_short_event_builder.py`
  - `strategy/sota_overlay_state.py`
  - `tests/test_replay_stable_smc_live_shadow.py`
  - `tests/test_sota_overlay_state.py`

## Keep

These are still part of the active research chain or directly referenced by current research notes.

- `scripts/replay_stable_smc_live_shadow.py`
  - Local working copy is untracked here, but this workflow is still actively used.
  - Action: converge onto the tracked `main` version instead of deleting.
- `scripts/research_stable_reverse_short_plus_smc_short.py`
  - Core local research implementation behind Stable + SMC combination logic.
- `scripts/scan_stable_smc_live_shadow_stable_params.py`
  - Still useful for Stable parameter sensitivity under the live-shadow chronology.
- `scripts/research_reverse_short_from_failed_longs.py`
  - Referenced by `HIGH_LEVERAGE_REVERSE_SHORT_OVERLAY.md`.
- `HIGH_LEVERAGE_REVERSE_SHORT_OVERLAY.md`
  - Active local research note with runnable references to the reverse-short chain.
- `research/pa_ict_liquidity/README.md`
  - Large but clearly still serves as the control-plane note for PA/ICT research.
- `scripts/report_smc_trade_context.py`
- `scripts/scan_smc_context_overlay.py`
- `scripts/scan_smc_context_overlay_formal.py`
- `scripts/scan_pressure_smc_h4_formal.py`
- `scripts/scan_smc_runner_simulation.py`
- `scripts/reproduce_main_baseline_shadow.py`
- `scripts/reproduce_smc_context_overlay.py`
- `scripts/research_smc_standalone_v1.py`

These are all referenced from `research/pa_ict_liquidity/README.md`, so deleting them now would break the current research notebook.

## Archive

These look worth preserving, but they do not need to stay as active top-level scripts forever.

- `research/high_leverage/shadow_pressure_param_recheck.json`
  - Snapshot artifact, useful for provenance, not active logic.
- `scripts/report_htf_pa_ict_context.py`
- `scripts/report_htf_pa_ict_guard_audit.py`
- `scripts/report_pa_ict_liquidity_features.py`
- `scripts/report_pa_ict_trade_alignment.py`
- `scripts/report_smc_htf_liquidity_targets.py`
- `scripts/report_smc_runner_timeout_buckets.py`
- `scripts/report_smc_short_only_loss_attribution.py`
- `scripts/reproduce_htf_pa_ict_guard.py`
- `scripts/research_broader_reverse_short_2026_gate.py`
- `scripts/scan_2026_entry_quality_filters.py`
- `scripts/scan_htf_pa_ict_context_overlay.py`
- `scripts/scan_htf_pa_ict_generator_params.py`
- `scripts/scan_htf_pa_ict_guard_live_feasible.py`
- `scripts/scan_htf_pa_ict_guard_ttl_sensitivity.py`
- `scripts/scan_pa_ict_quality_overlay.py`
- `scripts/scan_pa_ict_shadow_quality_overlay.py`
- `scripts/scan_pullback_windows.py`
- `scripts/scan_smc_short_only_v1_10x_rr.py`
- `scripts/scan_smc_short_only_v3_hf_blayer.py`
- `scripts/scan_smc_short_only_v3_hf_blayer_rr.py`
- `scripts/scan_smc_standalone_combo.py`
- `scripts/scan_smc_standalone_v1_formal.py`
- `scripts/scan_smc_standalone_v2_1_10x_rr_split.py`
- `scripts/scan_smc_standalone_v2_formal.py`

Recommended action:

- Move these into a dedicated archival namespace later, for example:
  - `research/archive/pa_ict_liquidity/`
  - `scripts/archive/high_leverage/`

## Drop

These are the strongest candidates to remove first.

- `scripts/reproduce_reverse_short_overlay_best.sh`
- `scripts/reproduce_reverse_short_overlay_candidates.sh`

Rationale:

- shell wrappers around Python entrypoints
- low information density
- redundant once the Python scripts or a single README command block is kept

## Reclassified From Drop To Archive

The following were initially flagged as drop candidates, but they are still referenced by current local research notes. They should be archived later, not deleted now.

- `scripts/reproduce_reverse_short_overlay_candidates.py`
- `scripts/reproduce_reverse_short_overlay_combo.py`
- `scripts/reproduce_smc_short_only_v1_10x.py`
- `scripts/reproduce_smc_short_only_v3_hf_blayer_10x.py`
- `scripts/reproduce_smc_standalone_v2_1_10x.py`
- `scripts/audit_htf_pa_ict_context_replay.py`

Rationale:

- still referenced by `HIGH_LEVERAGE_REVERSE_SHORT_OVERLAY.md` or `research/pa_ict_liquidity/README.md`
- likely better moved under a research archive namespace later
- deleting them now would break local reproducibility notes

## Formal Redundancy To Clean In Code/Config

These are separate from file cleanup and should be cleaned in the formal strategy/config chain.

- `strategy/scalp_robust_v2_core.py: trailing_config`
  - Present in config/metrics, not used by runtime exit logic.
- Dead ATR override config keys still present in config files:
  - `long_atr_activation_rr`
  - `short_atr_activation_rr`
  - `bear_strong_short_atr_activation_rr`
  - `long_atr_loose_multiplier`
  - `long_atr_normal_multiplier`
  - `long_atr_tight_multiplier`
  - `short_atr_loose_multiplier`
  - `short_atr_normal_multiplier`
  - `short_atr_tight_multiplier`

Recommended action:

- remove these from formal config files only after checking `main` and paper/template/example together
- do not remove them from just one config file in isolation

## Recommended Next Sequence

1. Converge the active local `Stable + SMC` replay chain onto the tracked `main` versions.
2. Keep the research notes and core research scripts.
3. Archive the report/scan backlog into a dedicated research archive location.
4. Delete the shell wrappers and one-off repro helpers that no longer add unique value.
5. In a separate cleanup patch, remove dead formal config fields from the strategy/config chain.
