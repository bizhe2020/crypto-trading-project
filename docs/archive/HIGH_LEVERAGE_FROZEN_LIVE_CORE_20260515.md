# High Leverage Frozen Live Core 2026-05-15

This document freezes the current promoted live core so later research can compare against one stable baseline instead of drifting runtime assumptions.

## Frozen Candidate

- Branch when frozen: `new_strategy_research`
- Commit when frozen: `b7a075c`
- Runtime live config: `config/config.live.high-leverage-structure.json`
- Runtime template: `config/config.live.high-leverage-structure.template.json`
- Paper config: `config/config.paper.high-leverage-structure.json`
- Frozen replay artifact: `var/high_leverage_expansion/frozen_live_core_20260515.json`

## Frozen Stack

- `SOTA long`
- `SOTA score gate`
- `feature_bearish_structure=false` gate for long entries
- `long score bucket sizing`
- `SMC short`
- `gap-SMC short`
- `single-position arbitration`
- `shadow gate = 6 / 12 / 2 / 4`

## Replay Scope

- Data start: `2022-01-01 00:00:00+00:00`
- Data end: `2026-05-15 08:00:00+00:00`
- Informative candles: confirmed 4h only
- Entry sync: `replay_sync_entry_to_signal_price=true`
- Fee/slippage: taker fee `0.0005`, slippage `5.0` bps

## Frozen Result

- Full return: `160188.09%`
- Full max drawdown: `29.98%`
- 2026 return: `113.55%`
- Accepted events: `281`
- Rejected events: `3`
- Rejected reason: all `position_lock_open`

Accepted mix:

- `SOTA long`: `248`
- `SMC short`: `24`
- `gap_smc_short_expansion`: `9`

## Frozen Parameters

- SOTA score gate: `net_min=3`, `bull_min=8`, `bear_max=6`, `conflict_mode=any`
- Long structure gate: `require_non_bearish_structure_for_long=true`
- Exact long bucket:
  - `n3_b9_b6_conflict_target12`
  - `net=3, bull=9, bear=6, conflict=1`
  - `target_effective_leverage=12.0`
- Additional long buckets kept:
  - `bear_total_6_20x_boost`
  - `nbb_6_11_5_conflict_2p5_cap20`
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
  --enable-long-score-bucket-sizing \
  --long-score-bucket-sizing-rules-json "$RULES_JSON" \
  --enable-gap-smc-short \
  --gap-smc-case gap_expansion_21d_other_3x \
  --gap-smc-min-flat-days 21 \
  --gap-smc-leverage 3 \
  --gap-smc-max-stop-distance-pct 1.5 \
  --daily-loss-stop-pct 6 \
  --equity-drawdown-stop-pct 12 \
  --equity-drawdown-cooldown-days 2 \
  --consecutive-loss-stop 4 \
  --sample-trades 0 \
  --output var/high_leverage_expansion/frozen_live_core_20260515.json
```

Expected key output:

```text
Live-shadow full=160188.09%/29.98% 2026=113.55%
accepted=281 rejected=3
```
