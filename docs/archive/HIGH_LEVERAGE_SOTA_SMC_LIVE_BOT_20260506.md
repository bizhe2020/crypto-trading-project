# High Leverage SOTA + SMC Live Bot Candidate 2026-05-06

This archive records the current promoted live-bot candidate after replay/live-shadow convergence work.

## Candidate

- Stack: SOTA long + SMC short + single-position arbitration
- Branch when archived: `new_strategy_research`
- Runtime template: `config/config.live.high-leverage-structure.template.json`
- Paper config: `config/config.paper.high-leverage-structure.json`
- Drift baseline: `config/live_drift_baseline.high_leverage.json`

## Replay Scope

- Data start: `2022-01-01 00:00:00+00:00`
- Data end: `2026-05-04 16:15:00+00:00`
- Entry sync: `replay_sync_entry_to_signal_price=true`
- Informative candles: confirmed 4h only
- Fee/slippage: taker fee `0.0005`, slippage `5.0` bps

## Live-Shadow Result

- Full return: `12911.21%`
- Full max drawdown: `28.36%`
- 2026 return: `30.95%`
- 2026 max drawdown: `17.08%`
- Trades: `299`
- Win rate: `44.82%`
- Profit factor: `1.9301`
- Accepted events: SOTA long `275`, SMC short `24`
- Rejected events: `3`, all due `position_lock_open`

## Promoted Parameters

- SOTA score gate: `net_min=3`, `bull_min=8`, `bear_max=6`, `conflict_mode=any`
- Shadow risk gate: daily loss `6%`, equity DD `12%`, cooldown `2` days, consecutive loss stop `4`
- Dynamic sizing: base `4.0`, high growth `7.5`, tight stop `7.5`, max effective `7.5`
- Failed-breakout guard leverage: `1.5`
- SMC case: `v2_medium_dispbody05_otherlag4_10x`
- Arbitration priority: `["sota_long", "smc_short"]`
- `overlay_skip_dynamic_high_leverage=true`

## Trailing Result

- Keep main trailing: `stage=close`, `time=extreme`, `atr_activation=extreme`
- Keep `atr_activation_rr=2.06`, `atr_normal_multiplier=2.25`
- Keep `enable_auto_time_based_trailing=true`, `T1=10`, `T2=20`, `T_max=144`, `S4_close_rr=0.8`
- `S4_close_rr=1.2` improved 2026 to `31.90%`, but reduced full return to `12642.45%`; it is not promoted.
- Disabling auto time trailing raised full return to `18924.69%`, but increased DD to `35.78%` and reduced 2026 to `25.64%`; it is not promoted.

## Reproduction Command

```bash
python3 scripts/replay_sota_smc_live_shadow.py \
  --config config/config.live.5x-3pct.json \
  --pressure-params config/high_leverage_pressure_target_cap_best.params.json \
  --confirmed-4h-only \
  --replay-sync-entry-to-signal-price \
  --enable-sota-score-gate \
  --sota-score-net-min 3 \
  --sota-score-bull-min 8 \
  --sota-score-bear-max 6 \
  --sota-score-conflict-mode any \
  --stage-trigger-rr-mode close \
  --time-trailing-rr-mode extreme \
  --atr-activation-rr-mode extreme \
  --daily-loss-stop-pct 6 \
  --equity-drawdown-stop-pct 12 \
  --equity-drawdown-cooldown-days 2 \
  --consecutive-loss-stop 4 \
  --output var/high_leverage_expansion/sota_smc_scoregate_net3_atr_extreme_shadow_2026_top_conservative_20260506.json \
  --paper-log-output var/high_leverage_expansion/sota_smc_scoregate_net3_atr_extreme_shadow_2026_top_conservative_20260506.jsonl
```

Expected key output:

```text
Reference base-priority full=12911.21%/28.36% 2026=30.95%
Live-shadow full=12911.21%/28.36% 2026=30.95%
gap=0
```
