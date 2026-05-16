# High-Leverage Fallback Trigger Report

Date: 2026-05-16
Branch: `new_strategy_research`

## Purpose

This report script turns recent runtime trade quality into a simple fallback recommendation:

- `stay_main`
- `fallback_a_short_off`
- `fallback_b_short_off_conflict_boost_defense`
- `fallback_c_long_core_only`
- `insufficient_data`

## Script

- [fallback_trigger_report.py](/Users/laoji/projects/crypto-trading-project/scripts/fallback_trigger_report.py)

## Inputs

The script reads:

- strategy config
- runtime sqlite DB
- drift baseline thresholds

Default baseline:

- [live_drift_baseline.high_leverage.json](/Users/laoji/projects/crypto-trading-project/config/live_drift_baseline.high_leverage.json)

## Example

Paper:

```bash
python3 scripts/fallback_trigger_report.py \
  --config config/config.paper.high-leverage-structure.json \
  --state-db state/runtime_high_leverage_structure_paper.db \
  --recent-trades 20 \
  --json
```

Live:

```bash
python3 scripts/fallback_trigger_report.py \
  --config config/config.live.high-leverage-structure.json \
  --state-db state/runtime_high_leverage_structure_live.db \
  --recent-trades 20 \
  --json
```

## Current Behavior

The trigger report is intentionally conservative.

It will recommend:

- `insufficient_data` when recent trade count is too low
- `fallback_a_short_off` when short quality degrades first
- `fallback_b_short_off_conflict_boost_defense` when short degrades and long quality also weakens
- `fallback_c_long_core_only` when both sides broadly degrade and stop-like exits dominate

## Intended Use

Use the trigger report as:

1. a monitoring helper
2. a switch recommendation
3. a consistency layer so fallback decisions are not made ad hoc

Do not treat it as a fully autonomous controller yet.
It is a decision-support tool.
