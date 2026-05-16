# High-Leverage Fallback Switch Runbook

Date: 2026-05-16
Branch: `new_strategy_research`

## Purpose

This runbook explains how to switch from the main frozen strategy to a fallback config and how to switch back safely.

Use this document together with:

- [HIGH_LEVERAGE_2021_FALLBACK_PLAN_20260516.md](/Users/laoji/projects/crypto-trading-project/docs/archive/HIGH_LEVERAGE_2021_FALLBACK_PLAN_20260516.md)
- [fallback_plan_20260516.json](/Users/laoji/projects/crypto-trading-project/var/high_leverage_expansion/fallback_plan_20260516.json)

## Available Fallback Configs

Paper:

- [config.paper.high-leverage-structure.fallback-a.json](/Users/laoji/projects/crypto-trading-project/config/config.paper.high-leverage-structure.fallback-a.json)
- [config.paper.high-leverage-structure.fallback-b.json](/Users/laoji/projects/crypto-trading-project/config/config.paper.high-leverage-structure.fallback-b.json)
- [config.paper.high-leverage-structure.fallback-c.json](/Users/laoji/projects/crypto-trading-project/config/config.paper.high-leverage-structure.fallback-c.json)

Live templates:

- [config.live.high-leverage-structure.fallback-a.template.json](/Users/laoji/projects/crypto-trading-project/config/config.live.high-leverage-structure.fallback-a.template.json)
- [config.live.high-leverage-structure.fallback-b.template.json](/Users/laoji/projects/crypto-trading-project/config/config.live.high-leverage-structure.fallback-b.template.json)
- [config.live.high-leverage-structure.fallback-c.template.json](/Users/laoji/projects/crypto-trading-project/config/config.live.high-leverage-structure.fallback-c.template.json)

## Fallback Meaning

### Fallback A

- disable `SMC short`
- keep `gap-SMC`
- keep long bucket sizing

### Fallback B

- disable `SMC short`
- keep `gap-SMC`
- reduce conflict boost buckets to defensive `2x`

### Fallback C

- disable `SMC short`
- disable `gap-SMC`
- disable long bucket sizing
- keep only the long core

## Trigger Logic

Do not switch on a single bad trade.
Use rolling evidence.

Recommended observation windows:

- rolling `30d`
- rolling `60d`
- last `20` trades

### Trigger Fallback A

Switch if:

- `SMC short` return sum turns clearly negative
- or `SMC short` win rate collapses
- or short losses cluster in a way that is not seen in the current frozen baseline

### Trigger Fallback B

Switch if:

- Fallback A conditions are true
- and conflict boost buckets also degrade

### Trigger Fallback C

Switch if:

- long and short both deteriorate
- stop-loss share rises broadly
- no boosted bucket remains reliable

## Switching Order

Do not go directly from main live to Tokyo deployment unless the situation is urgent.

Recommended order:

1. replay / sanity review
2. paper
3. live-shadow / local live path
4. Tokyo deploy

## Step-by-Step

### 1. Replay sanity review

Confirm which fallback level you want:

- `A`
- `B`
- `C`

Review:

- short metrics
- bucket metrics
- recent live drift

### 2. Paper switch

Use the matching paper config:

- `config/config.paper.high-leverage-structure.fallback-a.json`
- `config/config.paper.high-leverage-structure.fallback-b.json`
- `config/config.paper.high-leverage-structure.fallback-c.json`

Run the normal paper path and observe:

- accepted trade mix
- short trade count
- bucket use
- stop-loss share

Observation window:

- minimum `20` trades
- or minimum `7` days if trade count is low

### 3. Local live-shadow / local live path

If paper looks acceptable, prepare the matching live template into the runtime live config:

- `config/config.live.high-leverage-structure.fallback-a.template.json`
- `config/config.live.high-leverage-structure.fallback-b.template.json`
- `config/config.live.high-leverage-structure.fallback-c.template.json`

Use:

- `bash scripts/workflows/live/prepare_live_config.sh`
- `bash scripts/workflows/live/run_local_bot.sh`

Check:

- config loaded as expected
- Telegram status text reflects the correct short/bucket state
- no obvious startup errors

### 4. Tokyo deployment

When ready:

- point the runtime live config at the desired fallback template
- deploy with `bash scripts/workflows/live/deploy_tokyo.sh`

After deploy:

- verify service health
- verify command/status output
- verify no unexpected strategy mix

## Rollback Rules

Rollback means switching back toward the main frozen core.

### Rollback from C to B or A

Consider rollback only after:

- long core stabilizes
- stop-loss share improves
- recent drift is acceptable

### Rollback from B to A or Main

Consider rollback only after:

- conflict buckets recover
- short health no longer looks toxic

### Rollback from A to Main

Consider rollback only after:

- short win rate recovers
- short return sum recovers
- recent short losses stop clustering

Recommended rollback observation window:

- minimum `20` trades
- or minimum `14` days

Do not roll back after one or two good trades.

## Operating Principle

The priority is:

- preserve the main live edge when conditions are good
- degrade gracefully when regime fit weakens
- avoid panic full re-optimization during live underperformance

Fallbacks are meant to buy time and reduce damage.
They are not replacements for research.
