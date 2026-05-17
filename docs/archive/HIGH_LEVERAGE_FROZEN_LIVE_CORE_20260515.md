# High Leverage Frozen Live Core 2026-05-17

This document freezes the current promoted live core so later research can compare against one stable baseline instead of drifting runtime assumptions.

## Frozen Candidate

- Branch when frozen: `new_strategy_research`
- Commit when frozen: pending current branch commit
- Runtime live config: `config/config.live.high-leverage-structure.json`
- Runtime template: `config/config.live.high-leverage-structure.template.json`
- Paper config: `config/config.paper.high-leverage-structure.json`
- Frozen replay artifact: `var/high_leverage_expansion/frozen_live_core_20260515.json`

## Frozen Stack

- `SOTA long`
- `SOTA score gate`
- `feature_bearish_structure=false` gate for long entries
- `SOTA rejected SMC/FVG recall long`: structure-gate rejected SOTA long can be recalled when `sweep_has_fvg + normal regime`, fixed at 8x
- `long score bucket sizing`
- `SMC short`
- `gap-SMC short`
- `single-position arbitration`
- `shadow gate = 6 / 12 / 2 / 4`

## Replay Scope

- Data start: `2022-01-01 00:00:00+00:00`
- Data end: `2026-05-16 01:15:00+00:00`
- Informative candles: confirmed 4h only
- Entry sync: `replay_sync_entry_to_signal_price=true`
- Fee/slippage: taker fee `0.0005`, slippage `5.0` bps

## Frozen Result

- Full return: `1589905.71%`
- Full max drawdown: `26.76%`
- 2026 return: `202.31%`
- Accepted events: `287`
- Rejected events: `3`
- Rejected reason: all `position_lock_open`

Accepted mix:

- `SOTA long`: `254`
- `SMC short`: `24`
- `gap_smc_short_expansion`: `9`

## Frozen Parameters

- SOTA score gate: `net_min=3`, `bull_min=8`, `bear_max=6`, `conflict_mode=any`
- Long structure gate: `require_non_bearish_structure_for_long=true`
- SOTA rejected SMC/FVG recall: `enabled=true`, `condition=sweep_has_fvg`, `reject_stage=structure_gate`, `regime_label=normal`, `target_effective_leverage=8.0`, recalled `6` of `30` structure rejected candidates
- Exact long bucket:
  - `n3_b9_b6_conflict_target12`
  - `net=3, bull=9, bear=6, conflict=1`
  - `target_effective_leverage=12.0`
- Additional long buckets kept:
  - `fvg_near_bear6_target20` with `recent_fvg_near_entry=true`
  - `bear_total_6_20x_boost` with `regime_label=high_growth` and `conflict_mode=clean`
  - `nbb_6_11_5_conflict_2p5_cap20` with `recent_fvg_near_entry=true`
- Shadow gate:
  - `daily_loss_stop_pct=6`
  - `equity_drawdown_stop_pct=12`
  - `equity_drawdown_cooldown_days=2`
  - `consecutive_loss_stop=4`
- SMC short case: `v2_medium_dispbody05_otherlag4_10x`
- gap-SMC short case: `gap_expansion_21d_other_3x`
- Arbitration priority:
  - `sota_long`
  - `smc_short`
  - `gap_smc_short_expansion`

## Keep / Do Not Touch

Keep frozen unless a new candidate beats this baseline in the same replay/live-shadow/Tokyo audit family:

- `SOTA score gate`
- long structure gate
- SOTA rejected SMC/FVG recall long
- exact 12x bucket
- `SMC short`
- `gap-SMC short`
- `shadow gate 6 / 12 / 2 / 4`

Do not retune shadow gate against a different partial stream and compare it to this baseline. Compare only on the full frozen stack.

## Reproduction Command

```bash
RULES_JSON="$(python3 - <<'PY'
import json
from pathlib import Path
cfg=json.loads(Path('config/config.live.high-leverage-structure.json').read_text())
print(json.dumps(cfg.get('long_score_bucket_sizing_rules', []), ensure_ascii=False))
PY
)"

python3 scripts/replay_sota_smc_live_shadow.py \
  --config config/config.live.high-leverage-structure.json \
  --data-15m data/okx/futures/BTC_USDT_USDT-15m-futures.feather \
  --data-4h data/okx/futures/BTC_USDT_USDT-4h-futures.feather \
  --start-date 2022-01-01 \
  --confirmed-4h-only \
  --replay-sync-entry-to-signal-price \
  --enable-sota-score-gate \
  --require-non-bearish-structure-for-long \
  --enable-sota-rejected-smc-recall-long \
  --enable-long-score-bucket-sizing \
  --long-score-bucket-sizing-rules-json "$RULES_JSON" \
  --enable-gap-smc-short \
  --gap-smc-case gap_expansion_21d_other_3x \
  --gap-smc-min-flat-days 21 \
  --gap-smc-leverage 3 \
  --gap-smc-max-stop-distance-pct 1.5 \
  --smc-min-entry-idx 500 \
  --gap-smc-min-entry-idx 0 \
  --stage-trigger-rr-mode close \
  --time-trailing-rr-mode extreme \
  --atr-activation-rr-mode extreme \
  --daily-loss-stop-pct 6 \
  --equity-drawdown-stop-pct 12 \
  --equity-drawdown-cooldown-days 2 \
  --consecutive-loss-stop 4 \
  --sample-trades 0 \
  --output var/high_leverage_expansion/frozen_live_core_20260515.json
```

Expected key output:

```text
Live-shadow full=1589905.71%/26.76% 2026=202.31%
accepted=287 rejected=3
recall=6 of 30 structure-gate rejected candidates
```

## 2026-05-17 Refreeze

The frozen artifact was replaced with the strict live-config replay after the FVG bear6 20x bucket was promoted and the generic bear6 boost was narrowed.

Refreeze artifact:

- Strict replay source: `var/reports/live_config_bear6_target20_strict_replay_20260517.json`
- Frozen artifact: `var/high_leverage_expansion/frozen_live_core_20260515.json`
- Reference base-priority: `512742.54%`, DD `33.88%`, 2026 `186.88%`
- Live-shadow: `656013.44%`, DD `33.88%`, 2026 `202.31%`
- Accepted events: `287`
- Rejected events: `3`, all `position_lock_open`

## 2026-05-17 Guard Audit

The promoted bucket guard only changes two fragile bucket predicates:

- `bear_total_6_20x_boost` now requires `regime_label=high_growth`.
- `nbb_6_11_5_conflict_2p5_cap20` now requires `recent_fvg_near_entry=true`.

Audit artifacts:

- Formal replay: `var/reports/sota_fragile_guard_live_replay_20260517.json`
- Bucket robustness: `var/reports/sota_fragile_guard_bucket_robustness_audit_20260517.json`
- Top trade robustness: `var/reports/sota_fragile_guard_top_trade_robustness_20260517.json`

Stress checks:

- Remove top 1 winner: `221339.62%`, DD `37.48%`, 2026 `169.64%`
- Remove top 3 winners: `103371.48%`, DD `37.48%`, 2026 `89.40%`
- Remove top 5 winners: `51439.13%`, DD `37.48%`, 2026 `89.40%`

Residual risk:

- `bear_total_6_20x_boost` and `nbb_6_11_5_conflict_2p5_cap20` remain small-sample high-contribution buckets.
- The guard is accepted because it removes three historically harmful leveraged losers, improves full return, and does not increase headline DD or reduce 2026 return in the formal replay.


## 2026-05-17 Recall Refreeze

The frozen artifact was replaced again after promoting the SOTA rejected SMC/FVG recall long path into the live bot and replay chain. This keeps the frozen reference aligned with the deployable live config.

Refreeze artifact:

- Recall-on replay source: `var/reports/live_config_recall_on_replay_20260517.json`
- Full-events audit: `var/reports/live_config_recall_on_full_events_audit_20260517.json`
- Frozen artifact: `var/high_leverage_expansion/frozen_live_core_20260515.json`
- Live-shadow: `1589905.71%`, DD `26.76%`, 2026 `202.31%`
- Recall audit: `6` recalled trades, total recalled unit return `97.89%`; top-trade removal remains above the previous frozen baseline through top-4 removal.
