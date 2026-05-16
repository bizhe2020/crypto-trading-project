# High-Leverage 2021-Like Fallback Plan

Date: 2026-05-16
Branch: `new_strategy_research`

## Goal

Provide a practical fallback path if the live market starts behaving more like the hostile `2020-2021` environment than the current `2022+` regime.

This is not a full strategy rewrite.
It is a controlled defensive downgrade path.

## Current Main Strategy

Frozen live core:

- `SOTA long`
- `SOTA score gate`
- `feature_bearish_structure=false`
- exact `12x` long bucket sizing
- `SMC short`
- `gap-SMC short`
- `single-position arbitration`
- `shadow gate = 6 / 12 / 2 / 4`

Main replay result:

- `160188.09% / DD 29.98% / 2026 113.55%`

## Why Fallback Exists

Research shows:

- `SMC short` is strongly additive in `2022+`
- but strongly harmful in `2020-2021`
- `gap-SMC` is smaller and less critical
- some conflict long boost buckets also become fragile in hot bull-trend environments

Therefore fallback should not delete the whole system.
It should progressively reduce the parts that fail first.

## Fallback Levels

### Fallback A: Short-Off Defense

Use when:

- short win rate degrades first
- or short losses become clustered
- but long engine still behaves acceptably

Actions:

- disable `SMC short`
- keep `gap-SMC` unchanged
- keep long score gate and structure gate unchanged
- keep bucket sizing unchanged
- keep `shadow gate = 6 / 12 / 2 / 4`

Expected behavior:

- protects against the strongest known early-regime toxin
- preserves most of the long engine

### Fallback B: Short-Off + Conflict-Boost Defense

Use when:

- short edge degrades
- and conflict long boost buckets also start underperforming
- especially in strong positive momentum / bull-overheat environments

Actions:

- disable `SMC short`
- keep `gap-SMC`
- reduce conflict long bucket leverage:
  - `n3_b9_b6_conflict_target12`
  - `bear_total_6_20x_boost`
  - `nbb_6_11_5_conflict_2p5_cap20`
- suggested fallback target:
  - reduce to approximately `2x` effective leverage in hostile regime

Expected behavior:

- keeps the long engine alive
- removes the most fragile amplification layer

### Fallback C: Long-Core Defense

Use when:

- long and short both deteriorate
- stop-loss rate rises broadly
- bucket amplification no longer pays

Actions:

- disable `SMC short`
- disable `gap-SMC`
- disable long score bucket sizing boosts
- keep only the base `SOTA long` with score gate and structure gate
- optionally reduce overall dynamic leverage

Expected behavior:

- alpha drops
- but the system becomes much simpler and more survivable

## Trigger Checklist

Do not switch because of one bad day.
Switch when multiple rolling conditions degrade together.

Recommended monitoring windows:

- rolling `30d`
- rolling `60d`
- last `20` trades

### Watch Items

1. `SMC short` health

- short win rate
- short return sum
- short profit factor
- short loss clustering

2. Long engine health

- `SOTA long` win rate
- `SOTA long` average return
- `high_growth` long return sum
- stop-loss share

3. Boost bucket health

- per-bucket win rate
- per-bucket return sum
- especially conflict buckets

4. Convergence health

- live vs replay drift
- rejected / skipped anomalies

## Suggested Trigger Rules

### Trigger Fallback A

Switch to Fallback A if any of these persist across a rolling window:

- `SMC short` win rate collapses near `0%`
- `SMC short` cumulative return becomes meaningfully negative
- `SMC short` has multiple consecutive losses

### Trigger Fallback B

Switch to Fallback B if:

- Fallback A conditions are true
- and conflict boost buckets also degrade
- especially if conflict buckets show repeated stop-losses in strong positive momentum environments

### Trigger Fallback C

Switch to Fallback C if:

- both long and short health degrade
- stop-loss share rises broadly
- no bucket remains reliably positive

## Practical Recommendation

Default live stance:

- keep the current frozen main strategy
- do not enable a global `SMC short` gate yet

Prepared fallback order:

1. `Fallback A`
2. `Fallback B`
3. `Fallback C`

This is better than emergency full re-optimization.

## Research Status

Current evidence supports:

- keeping `SMC short` in the main `2022+` strategy
- not treating `SMC short` as all-regime universal
- preparing a regime-based fallback rather than deleting short logic
