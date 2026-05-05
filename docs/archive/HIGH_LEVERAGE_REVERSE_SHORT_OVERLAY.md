# High Leverage Reverse Short Overlay Research

This document records the read-only research result for adding a narrow reverse-short overlay after failed high-leverage long events. It does not change the promoted long strategy or live execution.

## Current Finding

The direct symmetric short idea is not promoted. Independent reverse-short streams are weak or negative over the full history. The useful variant is a post-stop overlay on top of the existing SOTA long stream.

The current research keeps three fixed candidates side by side:

- `Stable`: narrow, guard-based source selection. This is the main conservative candidate.
- `Broader Raw`: wider failed-long source selection with stronger headline 2026, but still one-trade dependent.
- `Broader Gate`: wider failed-long source selection tuned to pass the non-single-trade 2026 gate.

Common execution rules:

1. Keep the existing shadow-gated high-leverage SOTA long stream unchanged.
2. Require the source long event to be a real loser: `source return < 0`.
3. Require the source long to exit by `stop_loss`.
4. Enter the reverse short only after the source long exits by `stop_loss`.
5. Give the original SOTA stream priority; add the reverse short only if it does not overlap an existing SOTA event.

## Candidate Comparison

Three fixed candidates are now tracked:

| Candidate | Selector | Full Return | MaxDD | 2026 | 2026 Delta | Trades | Gate | Use |
|---|---|---:|---:|---:|---:|---:|---|---|
| Stable | `guarded_weak_loss` | `138260.83%` | `33.87%` | `39.24%` | `+9.37%` | `290` | pass | main candidate |
| Broader Raw | `bull_high_growth_offense_loss` | `128378.70%` | `33.87%` | `42.14%` | `+12.27%` | `297` | fail | one-trade-dependent observation |
| Broader Gate | `bull_high_growth_offense_loss` | `105015.89%` | `33.87%` | `33.68%` | `+3.81%` | `299` | pass | robust broader fallback |

Baseline shadow SOTA:

```text
full=88481.28% maxDD=33.87% 2026=29.87% trades=282
```

Run the side-by-side candidate reproduction:

```bash
scripts/reproduce_reverse_short_overlay_candidates.sh
```

Expected report path:

```text
var/high_leverage_expansion/reverse_short_overlay_candidate_comparison.json
```

Latest accepted-short attribution:

| Candidate | Selector Hits | Simulated Shorts | Accepted Shorts | Skipped By Base | Accepted Short Win Rate | Accepted Short Compounded | 2026 Accepted Shorts | 2026 Short Compounded |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stable | `15` | `11` | `8` | `3` | `75.00%` | `56.20%` | `2` | `7.22%` |
| Broader Raw | `39` | `20` | `15` | `5` | `66.67%` | `45.04%` | `1` | `9.45%` |
| Broader Gate | `39` | `20` | `17` | `3` | `58.82%` | `18.67%` | `2` | `2.94%` |

Interpretation:

- `Stable` has fewer trades but better accepted-short quality and remains the main candidate.
- `Broader Raw` adds more opportunities without increasing full MaxDD in the fixed reproduction, but the 2026 uplift is concentrated in one accepted short.
- `Broader Gate` proves the broader selector can pass the non-single-trade 2026 gate, but the edge is much smaller than Stable.

Best strict loss-only candidate with unchanged MaxDD after the local stop-loss / take-profit refinement:

| Case | Full Return | MaxDD | 2026 Delta | Last 60d Delta | Trades |
|---|---:|---:|---:|---:|---:|
| Shadow SOTA baseline | `88481.28%` | `33.87%` | - | - | `282` |
| Reverse-short overlay combo, 5x | `138260.83%` | `33.87%` | `+9.37%` | `+0.00%` | `290` |

Recommended Stable params:

```json
{
  "source_stream": "shadow",
  "selector": "guarded_weak_loss",
  "trigger_mode": "stop_loss_reversal",
  "target_rr": 2.75,
  "max_hold_bars": 80,
  "leverage": 5.0,
  "stop_multiplier": 1.1,
  "max_short_stop_pct": 1.75,
  "overlay_allocation": 1.0,
  "combo_mode": "base_priority_single_slot"
}
```

The 5x Stable version is preferred over 6x/8x for the conservative research record because it improves full-cycle compounding without increasing MaxDD. A local refinement around the original Stable parameters improved the full-cycle result further without changing full MaxDD.

Leverage safety scan highlights:

| Effective Short Exposure | Full Return | MaxDD | Delta | DD Delta | Note |
|---:|---:|---:|---:|---:|---|
| `5x` | `128482.39%` | `33.87%` | `+40001.11%` | `+0.00` | preferred research candidate |
| `6x` | `136669.49%` | `34.56%` | `+48188.21%` | `+0.69` | aggressive candidate |
| `8x` | `152893.07%` | `37.30%` | `+64411.79%` | `+3.43` | rejected for promotion |

The result is sample-thin, so it is a research candidate, not a live promotion.

## Reproduction

Run the focused scan:

```bash
scripts/reproduce_reverse_short_overlay_best.sh
```

Expected report path:

```text
var/high_leverage_expansion/reverse_short_overlay_combo_5x_reproduction.json
```

The wrapper expands to:

```bash
python3 scripts/reproduce_reverse_short_overlay_combo.py \
  --config config/config.live.5x-3pct.json \
  --pressure-params config/high_leverage_pressure_target_cap_best.params.json \
  --data-15m data/okx/futures/BTC_USDT_USDT-15m-futures.feather \
  --data-4h data/okx/futures/BTC_USDT_USDT-4h-futures.feather \
  --start-date 2022-01-01 \
  --source-stream shadow \
  --selector guarded_weak_loss \
  --trigger-mode stop_loss_reversal \
  --target-rr 2.75 \
  --max-hold-bars 80 \
  --leverage 5.0 \
  --stop-multiplier 1.1 \
  --max-short-stop-pct 1.75 \
  --overlay-allocation 1.0 \
  --combo-mode base_priority_single_slot \
  --daily-loss-stop-pct 6.0 \
  --equity-drawdown-stop-pct 15.0 \
  --equity-drawdown-cooldown-days 2 \
  --consecutive-loss-stop 0 \
  --sample-trades 20 \
  --output var/high_leverage_expansion/reverse_short_overlay_combo_5x_reproduction.json
```

Use this wider scan command when retesting leverage safety:

```bash
python3 scripts/research_reverse_short_from_failed_longs.py \
  --config config/config.live.5x-3pct.json \
  --pressure-params config/high_leverage_pressure_target_cap_best.params.json \
  --source-streams shadow \
  --selectors guarded_weak_loss \
  --trigger-modes stop_loss_reversal \
  --target-rr-values 1.0,1.25,1.5,2.0,2.5 \
  --max-hold-bars-values 24,32,48,64 \
  --leverage-values 2.0,3.0,4.0,5.0,6.0,8.0 \
  --stop-multiplier-values 0.75,1.0,1.25,1.5 \
  --overlay-allocation-values 0.25,0.5,0.75,1.0 \
  --max-short-stop-pct-values 1.5,2.0,2.5,3.0 \
  --output var/high_leverage_expansion/reverse_short_overlay_leverage_safety_scan.json
```

Use this local scan command when retesting the Stable stop-loss / take-profit neighborhood:

```bash
python3 scripts/research_reverse_short_from_failed_longs.py \
  --config config/config.live.5x-3pct.json \
  --pressure-params config/high_leverage_pressure_target_cap_best.params.json \
  --source-streams shadow \
  --selectors guarded_weak_loss \
  --trigger-modes stop_loss_reversal \
  --target-rr-values 2.0,2.25,2.5,2.75,3.0,3.25 \
  --max-hold-bars-values 48,64,80 \
  --leverage-values 5.0 \
  --stop-multiplier-values 1.0,1.1,1.2,1.25,1.3,1.4,1.5 \
  --overlay-allocation-values 1.0 \
  --max-short-stop-pct-values 1.75,2.0,2.25,2.5 \
  --output var/high_leverage_expansion/reverse_short_overlay_stable_sl_tp_local_scan.json
```

Use this focused scan command when retesting whether Broader can pass the 2026 non-single-trade gate:

```bash
python3 scripts/research_broader_reverse_short_2026_gate.py \
  --config config/config.live.5x-3pct.json \
  --pressure-params config/high_leverage_pressure_target_cap_best.params.json \
  --target-rr-values 0.75,1.0,1.25,1.5,2.0 \
  --max-hold-bars-values 4,6,8,10,12,16,20,24,32 \
  --leverage-values 4.0,5.0,6.0 \
  --stop-multiplier-values 0.75,1.0,1.25 \
  --max-short-stop-pct-values 1.0,1.25,1.5,1.75,2.0 \
  --overlay-allocation-values 0.5,0.75,1.0 \
  --output var/high_leverage_expansion/reverse_short_broader_2026_gate_scan.json
```

## Acceptance Gate

Do not promote either candidate unless a refreshed run keeps all of these true:

- Full return exceeds the SOTA baseline `88481.28%`.
- Stable full MaxDD does not exceed baseline MaxDD.
- Broader Raw full MaxDD does not exceed baseline MaxDD by more than `1%`.
- Broader Gate should pass the non-single-trade 2026 gate and ideally keep full MaxDD unchanged.
- 2026 delta is positive.
- Last 60d delta is not negative.
- Accepted overlay trades remain at least `8`.
- Added overlay losses do not create a new worst drawdown cluster.
- Broader must not rely on a single 2026 short for most of its current-year edge before live promotion.

## Stable + SMC Short Merge

Question tested: can the Stable reverse-short overlay be merged with the separate SMC short-only research line?

Yes, but only as an event-level single-slot merge, not as a separate capital-allocation sleeve. The tested merge keeps the SOTA long stream as base priority, then adds Stable reverse shorts and SMC short-only events only when they do not overlap existing base positions.

Best tested merge:

| Case | Full Return | MaxDD | 2026 | Event Counts | Read |
|---|---:|---:|---:|---|---|
| SOTA baseline | `88481.28%` | `33.87%` | `29.87%` | `282` SOTA | baseline |
| Stable only | `138260.83%` | `33.87%` | `39.24%` | `282` SOTA + `8` Stable shorts | current main candidate |
| Stable + SMC `v2_medium_dispbody05_otherlag4_10x` | `817041.70%` | `32.46%` | `94.48%` | `282` SOTA + `8` Stable shorts + `21` SMC shorts | research base-priority reference |
| Stable + SMC live-shadow high-return | `729707.88%` | `31.97%` | `83.02%` | `279` SOTA + `13` Stable shorts + `21` SMC shorts | current live-feasible main candidate |

Best live-feasible merge params:

```json
{
  "smc_case": "v2_medium_dispbody05_otherlag4_10x",
  "stable_selector": "guarded_weak_loss",
  "stable_target_rr": 2.75,
  "stable_max_hold_bars": 40,
  "stable_leverage": 5.0,
  "stable_stop_multiplier": 1.0,
  "stable_max_short_stop_pct": 1.75,
  "target_rr": 2.0,
  "allowed_time_buckets": "other+asia_evening_ny+ny_am_killzone",
  "swing_n": 2,
  "min_body_atr": 0.7,
  "min_range_atr": 1.1,
  "entry_lookahead_bars": 40,
  "max_open_positions": 1,
  "max_mss_lag_bars": 15,
  "min_displacement_body_atr": 0.5,
  "other_min_mss_lag_bars": 4,
  "leverage": 10.0,
  "position_size_pct": 1.0,
  "smc_allocation": 1.0,
  "combo_mode": "live_shadow_chronological"
}
```

Merge attribution:

| Sleeve | Accepted | Win Rate | Compounded | 2026 Accepted | 2026 Compounded |
|---|---:|---:|---:|---:|---:|
| Stable reverse short | `13` | scan-dependent | scan-dependent | scan-dependent | scan-dependent |
| SMC short | `21` | `71.43%` | `490.59%` | `5` | `39.67%` |

The SMC line is not the main conflict. The live-shadow conflict is between earlier Stable shorts and later SOTA entries. The current high-return live-shadow candidate intentionally allows `3` SOTA events to be blocked by already-open Stable shorts.

Reproduce:

```bash
python3 scripts/research_stable_reverse_short_plus_smc_short.py \
  --config config/config.live.5x-3pct.json \
  --pressure-params config/high_leverage_pressure_target_cap_best.params.json \
  --smc-cases v1_base_other_10x,v1_aggressive_maxlag9_10x,v2_medium_dispbody05_otherlag4_10x,v3_lag4_9_10x \
  --smc-allocation-values 0.25,0.5,0.75,1.0 \
  --output var/high_leverage_expansion/stable_reverse_short_plus_smc_short_combo.json
```

Use this scan when retesting high-return live-shadow Stable parameters:

```bash
python3 scripts/scan_stable_smc_live_shadow_stable_params.py \
  --config config/config.live.5x-3pct.json \
  --pressure-params config/high_leverage_pressure_target_cap_best.params.json \
  --smc-case v2_medium_dispbody05_otherlag4_10x \
  --smc-allocation 1.0 \
  --stable-allocation 1.0 \
  --target-rr-values 2.25,2.5,2.75,3.0 \
  --max-hold-bars-values 24,32,40,48,56,64,72,80 \
  --leverage-values 5.0 \
  --stop-multiplier-values 1.0,1.1,1.2,1.25 \
  --max-short-stop-pct-values 1.5,1.75,2.0 \
  --output var/high_leverage_expansion/stable_smc_live_shadow_stable_param_scan.json
```

Risk note:

- This is a strong research result, but it combines two separately developed short sleeves into one account-level execution stream.
- Before promotion, the merge needs yearly/monthly attribution, recent-window validation, and a live-feasible execution rule that prevents accidental overlap with open SOTA positions.

Live-feasibility audit highlights for the best merge:

| Check | Result | Read |
|---|---:|---|
| Full MaxDD delta vs SOTA | `-1.41` | improves full drawdown in replay |
| Negative months | `16` | needs monthly / streak protection before live |
| All-event min entry gap | `2.0h` | bot must enforce strict single-position lock |
| SMC min entry gap | `74.5h` | SMC sleeve itself is not high-frequency spam |
| SMC top-1 share of positive return | `14.94%` | not single-trade dependent |
| SMC top-3 share of positive return | `35.54%` | moderate concentration, acceptable for research |
| SMC 2026 accepted shorts | `5` | better sample than Broader Raw |
| SMC 2026 compounded | `39.67%` | major contributor to 2026 improvement |

SMC yearly attribution inside the best merge:

| Year | SMC Trades | SMC Win Rate | SMC Compounded |
|---|---:|---:|---:|
| `2022` | `9` | `55.56%` | `61.29%` |
| `2023` | `5` | `100.00%` | `96.05%` |
| `2024` | `1` | `100.00%` | `11.23%` |
| `2025` | `1` | `100.00%` | `20.22%` |
| `2026` | `5` | `60.00%` | `39.67%` |

Practical live gate for the merge:

- SOTA position state must be authoritative. No Stable or SMC short can enter while a SOTA trade is open.
- Stable and SMC shorts must share the same single-position lock; never allow two short sleeves at once.
- SMC events must pass liquidation-buffer diagnostics at order time, not only in research replay.
- Start with paper/live-shadow logging of all rejected overlaps before capital deployment.
- Add a sleeve-level kill switch: disable SMC after `2` consecutive SMC losses or after a monthly SMC drawdown worse than the historical single-month SMC loss bucket.

Live decision order:

1. If any exchange position is open, do nothing except manage that position.
2. If a SOTA long entry is valid, SOTA takes priority.
3. If a confirmed SOTA long stop-loss just closed and Stable eligibility is true, evaluate Stable reverse short.
4. If no SOTA or Stable action is active, evaluate SMC short-only signal.
5. Before sending any short order, re-check account exposure, liquidation buffer, max stop distance, and monthly / streak kill switches.

## Live-Shadow Chronological Replay

The event-level merge above was initially measured with a research `base_priority` filter: if a later SOTA event overlaps an earlier short, the short can be removed in hindsight. That is useful for analysis, but not fully live-feasible because the bot cannot know about future SOTA entries.

A chronological live-shadow replay was added to remove that future-information assumption. It processes candidate events in entry-time order with a strict single-position lock:

```bash
python3 scripts/replay_stable_smc_live_shadow.py \
  --config config/config.live.5x-3pct.json \
  --pressure-params config/high_leverage_pressure_target_cap_best.params.json \
  --smc-case v2_medium_dispbody05_otherlag4_10x \
  --smc-allocation 1.0 \
  --stable-allocation 1.0 \
  --stable-target-rr 2.75 \
  --stable-max-hold-bars 40 \
  --stable-leverage 5.0 \
  --stable-stop-multiplier 1.0 \
  --stable-max-short-stop-pct 1.75 \
  --output var/high_leverage_expansion/stable_smc_live_shadow_replay.json
```

Chronological result:

| Case | Full Return | MaxDD | 2026 | Event Counts |
|---|---:|---:|---:|---|
| Research `base_priority` reference, tuned Stable | `774758.04%` | `32.04%` | `88.69%` | `282` SOTA + `10` Stable + `21` SMC |
| Live-shadow chronological, tuned Stable | `729707.88%` | `31.97%` | `83.02%` | `279` SOTA + `13` Stable + `21` SMC |
| Live-shadow no-SOTA-reject variant | `523049.95%` | `32.04%` | `81.39%` | `282` SOTA + `0` Stable + `21` SMC |

Gap versus research reference:

- Full return gap: `-45050.16%`
- MaxDD gap: `-0.07`
- Decisions: `311` accepted, `9` rejected
- Rejections: all `position_lock_open`
- Rejected by type: `6` SMC shorts, `3` SOTA events

Important interpretation:

- The chronological result is still much stronger than Stable-only, but the correct live-feasible headline is now `729707.88% / 31.97% DD`, not the research-only `817041.70% / 32.46% DD`.
- The tuned live-shadow accepts `13` Stable reverse shorts versus `10` in the tuned hindsight reference.
- Those earlier Stable shorts block `3` later SOTA entries. This is live-executable, but it changes the promoted SOTA priority assumption.
- If Stable is not allowed to block future SOTA opportunities, the best no-SOTA-reject variant drops to `523049.95% / 32.04% DD`.

## Risk Notes

- This is not a symmetric short strategy. It is a post-stop overlay on a specific failed-long bucket.
- The Stable result currently depends on only `8` accepted combo overlay trades, so parameter stability is still weak even after the local refinement.
- The Broader Raw result currently depends on `15` accepted combo overlay trades, but its 2026 edge is one-trade concentrated.
- The Broader Gate result fixes the one-trade issue with `17` accepted combo overlay trades and `2` accepted 2026 shorts, but its full-cycle edge is much smaller than Stable.
- The first broad scan included profitable trailing-stop source events; that was rejected. Broad candidates must require source `stop_loss` and source `return < 0`.
- Live execution would need explicit sequencing: source long close confirmed, no open position, then reverse short eligibility check.
- Virtual invalidation triggers were tested as a way to add earlier short opportunities without moving the real long stop. They underperformed the confirmed `stop_loss_reversal` path and are not promoted.
