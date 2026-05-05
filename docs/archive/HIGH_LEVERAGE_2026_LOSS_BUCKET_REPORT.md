# 2026 Pressure Loss-Bucket Report

This report analyzes the current high-leverage retarget strategy with the active shadow gate:

```bash
python3 scripts/report_2026_pressure_loss_buckets.py \
  --config config/config.live.5x-3pct.json \
  --pressure-params config/high_leverage_pressure_target_cap_best.params.json \
  --data-15m data/okx/futures/BTC_USDT_USDT-15m-futures.feather \
  --data-4h data/okx/futures/BTC_USDT_USDT-4h-futures.feather \
  --start-date 2022-01-01 \
  --year 2026 \
  --daily-loss-stop-pct 6.0 \
  --equity-drawdown-stop-pct 15.0 \
  --equity-drawdown-cooldown-days 2 \
  --consecutive-loss-stop 0 \
  --output var/high_leverage_expansion/pressure_loss_buckets_2026.json
```

## Result

| Bucket | Trades | Return Sum | Average |
|---|---:|---:|---:|
| Accepted 2026 shadow events | `20` | - | - |
| Loss events | `11` | `-32.7095%` | `-2.9736%` |
| Plain stop loss | `7` | `-31.6391%` | `-4.5199%` |
| Late cap, no target after MFE >= 1R | `4` | `-1.0704%` | `-0.2676%` |
| Pressure/integer sourced losses | `0` | `0.0000%` | - |
| Time-stop losses | `0` | `0.0000%` | - |
| Early-cap opportunity events | `0` | `0.0000%` | - |

All 2026 losses in this run are `no_pressure_event`: the pressure/integer cap did not fire before those exits. The large losses are not from early pressure-level take profit; they are ordinary stop-loss trades that never reached a usable pressure/integer/cluster level first.

Worst accepted 2026 loss:

```text
2026-01-16 01:45 UTC -> 2026-01-16 15:30 UTC
direction=BULL, regime=high_growth, risk_mode=offense, effective_leverage=8.0
return=-11.8671%, exit=stop_loss, MFE=0.1055R
```

This trade had almost no favorable excursion, so target-cap logic cannot fix it. It points to entry/risk gating rather than trailing.

## Dynamic Min-RR Test

Implemented but not promoted:

- compression/weak structure: `pressure_dynamic_target_compression_rr`
- normal flat: `pressure_dynamic_target_flat_rr`
- breakout structure: `pressure_dynamic_target_breakout_rr`

The dynamic mode also lowers the pressure activation gate to the minimum dynamic target RR, otherwise a `1.0R` compression target cannot trigger while `pressure_min_rr=2.0`.

2026 comparison:

| Candidate | 2026 Return | MaxDD | Last 60d | Verdict |
|---|---:|---:|---:|---|
| Current static retarget | `47.52%` | `18.15%` | `4.81%` | keep |
| Dynamic, flat only | `26.37%` | `17.96%` | `0.98%` | reject |
| Best dynamic state expansion, all regimes | `28.49%` | `28.45%` | `6.51%` | reject |

Dynamic `pressure_target_min_rr` does not improve the current strategy. It protects a few small MFE losses, but it also cuts profitable expansion paths too early and degrades the 2026 window.

## Next Route

Do not promote dynamic pressure target min-RR to live config yet. The next useful 2026 iteration should focus on failed-breakout leverage gating.

## Failed-Breakout Offense Guard

The follow-up scan uses the same current-best pressure params and active shadow gate, then tests a guard that reduces weak high-growth/offense long leverage:

```bash
python3 scripts/scan_failed_breakout_offense_guard.py \
  --enabled-values false,true \
  --guard-leverage-values 2.0,4.0,7.5 \
  --min-leverage-values 7.5 \
  --min-quality-score-values 2,3,4,5 \
  --min-momentum-pct-values 1.5,4.0,6.0 \
  --min-ema-gap-pct-values 0.35,1.5,2.0 \
  --min-adx-values 22.0,35.0,40.0 \
  --regime-label-sets high_growth \
  --risk-mode-sets offense \
  --direction-sets BULL \
  --top 20 \
  --output var/high_leverage_expansion/failed_breakout_offense_guard_scan.json
```

Best candidate:

| Candidate | Full Return | MaxDD | 2026 | 2026 MaxDD | Last 60d | Last 30d | Guarded |
|---|---:|---:|---:|---:|---:|---:|---:|
| Previous baseline | `48028.76%` | `35.01%` | `20.17%` | `17.86%` | `5.19%` | `9.57%` | `0` |
| Failed-breakout guard | `88481.28%` | `33.87%` | `29.87%` | `11.35%` | `7.85%` | `8.47%` | `18` |

Best guard params:

```json
{
  "failed_breakout_guard_enabled": true,
  "failed_breakout_guard_leverage": 2.0,
  "failed_breakout_guard_min_leverage": 7.5,
  "failed_breakout_guard_min_quality_score": 2,
  "failed_breakout_guard_min_momentum_pct": 6.0,
  "failed_breakout_guard_min_ema_gap_pct": 2.0,
  "failed_breakout_guard_min_adx": 38.0,
  "failed_breakout_guard_regime_labels": ["high_growth"],
  "failed_breakout_guard_risk_modes": ["offense"],
  "failed_breakout_guard_directions": ["BULL"]
}
```

Interpretation:

- The guard does not simply catch the single worst 2026 offense loss; that loss had strong ADX, momentum, EMA gap, and bullish structure.
- The improvement comes from reducing historical high-growth/offense long exposure when the signal is not strong enough across momentum, EMA gap, ADX, and structure.
- This is now implemented as a runtime-configurable dynamic high-leverage guard and is enabled in the paper config and live template.

Next useful steps:

1. Use live-readiness replay after data refresh to detect core strategy drift before running the execution overlay.
2. Try to recover Last 30d above `9.0%`; the refined scan improved full and 2026, but still held Last 30d at `8.47%`.
3. Keep promotion only while Full stays above `88481.28%`, MaxDD stays below `35.5%`, 2026 stays above `29%`, and Last 60d stays above `7.0%`.
