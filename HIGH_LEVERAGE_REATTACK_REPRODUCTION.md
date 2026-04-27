# High Leverage Reattack Strategy Reproduction

This document records how to reproduce the current best dynamic high-leverage expansion result. It replays one fixed parameter set with the short-window reattack state machine.

## Current Best Strategy

The current best reproducible strategy is the pressure-aware target-cap iteration:

1. Base autoTIT config from `config/config.live.5x-3pct.json`.
2. 2026-aware structure high-leverage overlay.
3. Pressure-level target cap only in `flat` regime.
4. Shadow gate retuned on the fixed high-leverage event stream.

Current best result:

| Window | Return | MaxDD | Notes |
|---|---:|---:|---|
| Full, from `2022-01-01` | `38420.70%` | `35.04%` | best current full-window compounding |
| 2026 YTD | `19.48%` | `17.86%` | pressure-aware flat-regime target cap |
| Last 60d | `4.63%` | `10.77%` | positive |
| Last 30d | `8.99%` | `3.37%` | positive |

Use this command to reproduce the current best strategy:

```bash
scripts/reproduce_pressure_target_cap_best.sh
```

The script expands to the fixed one-parameter command:

```bash
python3 scripts/scan_pressure_level_trailing.py --config config/config.live.5x-3pct.json --data-15m data/okx/futures/BTC_USDT_USDT-15m-futures.feather --data-4h data/okx/futures/BTC_USDT_USDT-4h-futures.feather --start-date 2022-01-01 --include-disabled-baseline --pressure-min-rr-values 2.0 --pressure-lock-rr-values 0.4 --pressure-atr-multiplier-values 3.0 --pressure-proximity-pct-values 0.15 --pressure-rejection-min-rr-values 3.0 --pressure-take-profit-on-rejection-values false --pressure-enable-target-cap-values true --pressure-target-min-rr-values 1.5 --pressure-target-buffer-pct-values 0.05 --pressure-regime-label-sets flat --top 5 --output var/high_leverage_expansion/pressure_target_cap_flat_scan_full.json
```

Expected terminal line:

```text
01 score=40217.20 full=38420.70%/35.04% year=19.48% 60d=4.63% params={'enable_pressure_level_trailing': True, ... 'pressure_regime_labels': ['flat'], ...}
```

Previous best without pressure-level target cap:

```text
full=26868.27%/35.56% year=12.73% 60d=3.42%
```

Current best pressure-level parameters:

```json
{
  "enable_pressure_level_trailing": true,
  "pressure_min_rr": 2.0,
  "pressure_lock_rr": 0.4,
  "pressure_atr_multiplier": 3.0,
  "pressure_proximity_pct": 0.15,
  "pressure_rejection_min_rr": 3.0,
  "pressure_take_profit_on_rejection": false,
  "pressure_enable_target_cap": true,
  "pressure_target_min_rr": 1.5,
  "pressure_target_buffer_pct": 0.05,
  "pressure_regime_labels": ["flat"],
  "pressure_round_steps_usdt": [1000.0, 500.0],
  "pressure_cluster_lookback_bars": 192,
  "pressure_cluster_bin_usdt": 250.0,
  "pressure_cluster_min_touches": 4,
  "pressure_cluster_min_volume_ratio": 1.25,
  "pressure_swing_lookback_bars": 96,
  "pressure_rejection_wick_ratio": 0.55,
  "pressure_rejection_close_pct": 0.2,
  "pressure_min_bars_held": 1
}
```

Current best shadow gate parameters:

```json
{
  "daily_loss_stop_pct": 6.0,
  "equity_drawdown_stop_pct": 15.0,
  "equity_drawdown_cooldown_days": 2,
  "consecutive_loss_stop": 0
}
```

The fixed high-leverage overlay parameters used by the command above are embedded in `scripts/scan_shadow_on_fixed_high_leverage.py` as `FIXED_STRUCTURE_PARAMS`:

```json
{
  "base_leverage": 4.0,
  "high_growth_leverage": 7.5,
  "tight_stop_leverage": 8.0,
  "recovery_leverage": 2.0,
  "drawdown_leverage": 2.0,
  "unhealthy_leverage": 2.0,
  "tight_stop_pct": 1.25,
  "max_stop_distance_pct": 1.5,
  "high_growth_max_stop_distance_pct": 2.0,
  "wide_stop_mode": "all_healthy",
  "max_effective_leverage": 8.0,
  "loss_streak_threshold": 3,
  "win_streak_threshold": 2,
  "drawdown_threshold_pct": 20.0,
  "health_lookback_trades": 6,
  "health_min_unit_return_pct": 0.0,
  "health_min_win_rate_pct": 25.0,
  "state_lookback_trades": 8,
  "defense_enter_unit_return_pct": -2.0,
  "defense_enter_win_rate_pct": 20.0,
  "offense_enter_unit_return_pct": -0.5,
  "offense_enter_win_rate_pct": 40.0,
  "reattack_lookback_trades": 2,
  "reattack_unit_return_pct": 0.5,
  "reattack_win_rate_pct": 33.0,
  "reattack_signal_mode": "high_growth_or_tight_or_structure",
  "price_structure_reattack_mode": "none",
  "structure_reattack_min_momentum_pct": 0.0,
  "structure_reattack_min_ema_gap_pct": 0.25,
  "structure_reattack_min_adx": 0.0,
  "defense_leverage": 2.0,
  "defense_max_stop_distance_pct": 1.5,
  "defense_structure_max_stop_distance_pct": 1.9,
  "min_liq_buffer_pct": 1.2,
  "maintenance_margin_pct": 0.5
}
```

## Live / Paper Runtime Files

Runtime implementation files:

- `bot/okx_executor.py`: adds `enable_dynamic_high_leverage_structure`. When enabled, the executor updates a persisted `dynamic_high_leverage_structure_state` after closes and recalculates the next open's target effective leverage before sending the order.
- `config/config.paper.high-leverage-structure.json`: paper runtime config with the current best dynamic parameters and shadow gate params.
- `config/config.live.high-leverage-structure.template.json`: live template with the same parameters. Fill API credentials in a real `config/config.live.high-leverage-structure.json`; do not commit the filled file.
- `scripts/run_high_leverage_structure_live.sh`: live run-loop wrapper.
- `scripts/reproduce_current_best_high_leverage.sh`: one-command reproduction wrapper.

Paper bootstrap:

```bash
python3 bot/run_bot.py --config config/config.paper.high-leverage-structure.json --json
```

Live run command:

```bash
scripts/run_high_leverage_structure_live.sh config/config.live.high-leverage-structure.json
```

Runtime shadow gate parameters:

```json
{
  "enable_shadow_risk_gate": true,
  "shadow_daily_loss_stop_pct": 6.0,
  "shadow_equity_drawdown_stop_pct": 15.0,
  "shadow_equity_drawdown_cooldown_days": 2,
  "shadow_consecutive_loss_stop": 0
}
```

Runtime dynamic high-leverage parameters:

```json
{
  "enable_dynamic_high_leverage_structure": true,
  "leverage": 10,
  "dynamic_base_leverage": 4.0,
  "dynamic_high_growth_leverage": 7.5,
  "dynamic_tight_stop_leverage": 8.0,
  "dynamic_recovery_leverage": 2.0,
  "dynamic_drawdown_leverage": 2.0,
  "dynamic_unhealthy_leverage": 2.0,
  "dynamic_defense_leverage": 2.0,
  "dynamic_tight_stop_pct": 1.25,
  "dynamic_max_stop_distance_pct": 1.5,
  "dynamic_high_growth_max_stop_distance_pct": 2.0,
  "dynamic_defense_max_stop_distance_pct": 1.5,
  "dynamic_defense_structure_max_stop_distance_pct": 1.9,
  "dynamic_max_effective_leverage": 8.0,
  "dynamic_loss_streak_threshold": 3,
  "dynamic_win_streak_threshold": 2,
  "dynamic_drawdown_threshold_pct": 20.0,
  "dynamic_health_lookback_trades": 6,
  "dynamic_health_min_unit_return_pct": 0.0,
  "dynamic_health_min_win_rate_pct": 25.0,
  "dynamic_state_lookback_trades": 8,
  "dynamic_defense_enter_unit_return_pct": -2.0,
  "dynamic_defense_enter_win_rate_pct": 20.0,
  "dynamic_offense_enter_unit_return_pct": -0.5,
  "dynamic_offense_enter_win_rate_pct": 40.0,
  "dynamic_reattack_lookback_trades": 2,
  "dynamic_reattack_unit_return_pct": 0.5,
  "dynamic_reattack_win_rate_pct": 33.0,
  "dynamic_reattack_signal_mode": "high_growth_or_tight_or_structure"
}
```

## Target Result

Data window:

- 15m: `2022-01-01 00:00:00+00:00` to `2026-04-26 11:15:00+00:00`
- 4h: `2022-01-01 00:00:00+00:00` to `2026-04-26 08:00:00+00:00`

Expected top result:

| Window | Return | Sharpe | MaxDD | Notes |
|---|---:|---:|---:|---|
| Full, from `2022-01-01` | `13666.96%` | `3.548` | `34.58%` | `268` accepted, `143` skipped |
| 2026 YTD | `0.83%` | `0.589` | `19.16%` | `16` trades |
| Last 60d | `9.62%` | `5.342` | `4.70%` | `10` trades |
| Last 30d | `8.87%` | `6.682` | `2.45%` | `7` trades |

Reference main shadow baseline on the same current data:

```json
{
  "total_return_pct": 8241.56,
  "max_drawdown_pct": 36.02,
  "sharpe_ratio": 3.021,
  "total_trades": 345,
  "skipped_trades": 66
}
```

## Fixed Reproduction Command

Run from repo root:

```bash
python3 scripts/scan_high_leverage_expansion.py \
  --config config/config.live.5x-3pct.json \
  --data-15m data/okx/futures/BTC_USDT_USDT-15m-futures.feather \
  --data-4h data/okx/futures/BTC_USDT_USDT-4h-futures.feather \
  --start-date 2022-01-01 \
  --base-leverage 4 \
  --high-growth-leverage 6 \
  --tight-stop-leverage 8 \
  --recovery-leverage 2 \
  --drawdown-leverage 1.5 \
  --unhealthy-leverage 1.5 \
  --tight-stop-pct 1.25 \
  --max-stop-distance-pct 1.5 \
  --high-growth-max-stop-distance-pct 2.25 \
  --wide-stop-mode all_healthy \
  --max-effective-leverage 8 \
  --loss-streak-threshold 3 \
  --win-streak-threshold 2 \
  --drawdown-threshold-pct 20 \
  --health-lookback-trades 6 \
  --health-min-unit-return-pct 0 \
  --health-min-win-rate-pct 25 \
  --state-lookback-trades 8 \
  --defense-enter-unit-return-pct=-2 \
  --defense-enter-win-rate-pct 20 \
  --offense-enter-unit-return-pct=-0.5 \
  --offense-enter-win-rate-pct 40 \
  --reattack-lookback-trades 2 \
  --reattack-unit-return-pct 0.5 \
  --reattack-win-rate-pct 33 \
  --reattack-signal-mode high_growth_or_tight \
  --defense-leverage 2 \
  --defense-max-stop-distance-pct 1.5 \
  --min-liq-buffer-pct 1.2 \
  --maintenance-margin-pct 0.5 \
  --max-drawdown-pct 45 \
  --min-2026-return-pct 0 \
  --max-2026-drawdown-pct 30 \
  --min-60d-return-pct 0 \
  --top 1 \
  --output-dir var/high_leverage_expansion
```

Expected output file:

```text
var/high_leverage_expansion/dynamic_expansion_scan_2022-01-01_to_2026-04-26.json
```

The command uses singleton parameter lists, so it does not run the full search grid. It should still emit the same result as the top grid candidate.

## Exact Parameters

```json
{
  "base_leverage": 4.0,
  "high_growth_leverage": 6.0,
  "tight_stop_leverage": 8.0,
  "recovery_leverage": 2.0,
  "drawdown_leverage": 1.5,
  "unhealthy_leverage": 1.5,
  "tight_stop_pct": 1.25,
  "max_stop_distance_pct": 1.5,
  "high_growth_max_stop_distance_pct": 2.25,
  "wide_stop_mode": "all_healthy",
  "max_effective_leverage": 8.0,
  "loss_streak_threshold": 3,
  "win_streak_threshold": 2,
  "drawdown_threshold_pct": 20.0,
  "health_lookback_trades": 6,
  "health_min_unit_return_pct": 0.0,
  "health_min_win_rate_pct": 25.0,
  "state_lookback_trades": 8,
  "defense_enter_unit_return_pct": -2.0,
  "defense_enter_win_rate_pct": 20.0,
  "offense_enter_unit_return_pct": -0.5,
  "offense_enter_win_rate_pct": 40.0,
  "reattack_lookback_trades": 2,
  "reattack_unit_return_pct": 0.5,
  "reattack_win_rate_pct": 33.0,
  "reattack_signal_mode": "high_growth_or_tight",
  "defense_leverage": 2.0,
  "defense_max_stop_distance_pct": 1.5,
  "min_liq_buffer_pct": 1.2,
  "maintenance_margin_pct": 0.5
}
```

## State Machine Meaning

- `offense`: expansion/healthy mode. Allows high-growth leverage, tight-stop leverage, and win-streak expansion.
- `defense`: low-return/chop mode. Caps leverage with `defense_leverage` and caps stop width with `defense_max_stop_distance_pct`.
- `state_lookback_trades = 8`: long state window used to decide the main offense/defense state.
- `defense_enter_unit_return_pct = -2`: enter defense if the long-window unit return falls to `-2%` or worse.
- `defense_enter_win_rate_pct = 20`: enter defense if long-window win rate falls to `20%` or worse.
- `offense_enter_unit_return_pct = -0.5`: leave defense through the normal long-window recovery path once unit return recovers to `-0.5%` or better.
- `offense_enter_win_rate_pct = 40`: long-window recovery also requires at least `40%` win rate.
- `reattack_lookback_trades = 2`: short-window reattack path while in defense.
- `reattack_unit_return_pct = 0.5`: short-window unit return must be at least `0.5%`.
- `reattack_win_rate_pct = 33`: short-window win rate must be at least `33%`.
- `reattack_signal_mode = high_growth_or_tight`: the current signal must be high-growth or tight-stop qualified before defense can re-enter offense early.

## Expected JSON Fields

The top result should contain:

```json
{
  "total_return_pct": 13666.96,
  "max_drawdown_pct": 34.58,
  "sharpe_ratio": 3.548,
  "accepted_trades": 268,
  "skipped_trades": 143,
  "avg_effective_leverage": 3.290299,
  "max_effective_leverage_seen": 8.0,
  "accepted_risk_mode_counts": {
    "offense": 110,
    "defense": 158
  },
  "mode_switches": 39
}
```

Window fields:

```json
{
  "current_year": {
    "total_return_pct": 0.83,
    "max_drawdown_pct": 19.16,
    "trades": 16,
    "avg_effective_leverage": 1.8375,
    "max_effective_leverage": 6.9,
    "risk_mode_counts": {
      "offense": 1,
      "defense": 15
    }
  },
  "last_60d": {
    "total_return_pct": 9.62,
    "max_drawdown_pct": 4.7,
    "trades": 10
  },
  "last_30d": {
    "total_return_pct": 8.87,
    "max_drawdown_pct": 2.45,
    "trades": 7
  }
}
```

## Discovery Grid

The best result above was found with this smaller search grid:

```bash
python3 scripts/scan_high_leverage_expansion.py \
  --config config/config.live.5x-3pct.json \
  --data-15m data/okx/futures/BTC_USDT_USDT-15m-futures.feather \
  --data-4h data/okx/futures/BTC_USDT_USDT-4h-futures.feather \
  --start-date 2022-01-01 \
  --base-leverage 4 \
  --high-growth-leverage 6 \
  --tight-stop-leverage 8 \
  --recovery-leverage 2 \
  --drawdown-leverage 1.5 \
  --unhealthy-leverage 1.5 \
  --tight-stop-pct 1.25 \
  --max-stop-distance-pct 1.5 \
  --high-growth-max-stop-distance-pct 2.25 \
  --wide-stop-mode all_healthy \
  --max-effective-leverage 8 \
  --loss-streak-threshold 3 \
  --win-streak-threshold 2 \
  --drawdown-threshold-pct 20 \
  --health-lookback-trades 6 \
  --health-min-unit-return-pct 0 \
  --health-min-win-rate-pct 25 \
  --state-lookback-trades 6,8 \
  --defense-enter-unit-return-pct=-2,-1,0 \
  --defense-enter-win-rate-pct 20,25,33 \
  --offense-enter-unit-return-pct=-0.5,0,0.5 \
  --offense-enter-win-rate-pct 25,33,40 \
  --reattack-lookback-trades 2,3,4 \
  --reattack-unit-return-pct=-0.5,0,0.5 \
  --reattack-win-rate-pct 33,50 \
  --reattack-signal-mode high_growth_or_tight,high_growth,tight_stop \
  --defense-leverage 1.5,2 \
  --defense-max-stop-distance-pct 1.25,1.5 \
  --min-2026-return-pct 0 \
  --min-60d-return-pct 0 \
  --max-drawdown-pct 45 \
  --top 20 \
  --output-dir /tmp/high_leverage_reattack_grid
```

That grid has `27648` candidates and is slower than the fixed reproduction command.

## Caveats

- This is a research overlay, not live execution code.
- The result is sensitive to the data snapshot. Different OKX downloads or a later data cutoff can change the compounded return.
- The 2026 YTD return remains weak at `0.83%`; the edge in this candidate is mainly full-window expansion with controlled drawdown.
- Compare against the current reproducible main baseline `8241.56% / 36.02%`, not the older README historical record `9240.42% / 36.02%`, unless the original old data snapshot is recovered.

## 2026 Structure Iteration

The first reattack candidate above improves full-window expansion, but its 2026 YTD return stays weak. A second iteration adds 4h price-structure features to each trade and allows structure-qualified defense signals to use a wider stop cap. This better captures 2026 expansion pockets.

Best 2026-aware candidate from `/tmp/high_leverage_structure_2026_grid/dynamic_expansion_scan_2022-01-01_to_2026-04-26.json`:

| Window | Return | Sharpe | MaxDD | Notes |
|---|---:|---:|---:|---|
| Full, from `2022-01-01` | `19050.71%` | `3.351` | `35.56%` | `294` accepted, `117` skipped |
| 2026 YTD | `12.73%` | report output | `12.88%` | better than current main shadow YTD |
| Last 60d | `3.42%` | report output | report output | positive |
| Last 30d | `8.05%` | report output | report output | positive |

Fixed command:

```bash
python3 scripts/scan_high_leverage_expansion.py \
  --config config/config.live.5x-3pct.json \
  --data-15m data/okx/futures/BTC_USDT_USDT-15m-futures.feather \
  --data-4h data/okx/futures/BTC_USDT_USDT-4h-futures.feather \
  --start-date 2022-01-01 \
  --base-leverage 4 \
  --high-growth-leverage 7.5 \
  --tight-stop-leverage 8 \
  --recovery-leverage 2 \
  --drawdown-leverage 2 \
  --unhealthy-leverage 2 \
  --tight-stop-pct 1.25 \
  --max-stop-distance-pct 1.5 \
  --high-growth-max-stop-distance-pct 2.0 \
  --wide-stop-mode all_healthy \
  --max-effective-leverage 8 \
  --loss-streak-threshold 3 \
  --win-streak-threshold 2 \
  --drawdown-threshold-pct 20 \
  --health-lookback-trades 6 \
  --health-min-unit-return-pct 0 \
  --health-min-win-rate-pct 25 \
  --state-lookback-trades 8 \
  --defense-enter-unit-return-pct=-2 \
  --defense-enter-win-rate-pct 20 \
  --offense-enter-unit-return-pct=-0.5 \
  --offense-enter-win-rate-pct 40 \
  --reattack-lookback-trades 2 \
  --reattack-unit-return-pct 0.5 \
  --reattack-win-rate-pct 33 \
  --reattack-signal-mode high_growth_or_tight_or_structure \
  --price-structure-reattack-mode none \
  --structure-reattack-min-momentum-pct 0 \
  --structure-reattack-min-ema-gap-pct 0.25 \
  --structure-reattack-min-adx 0 \
  --defense-leverage 2 \
  --defense-max-stop-distance-pct 1.5 \
  --defense-structure-max-stop-distance-pct 1.9 \
  --min-liq-buffer-pct 1.2 \
  --maintenance-margin-pct 0.5 \
  --max-drawdown-pct 38 \
  --min-2026-return-pct 5 \
  --max-2026-drawdown-pct 25 \
  --min-60d-return-pct 0 \
  --top 1 \
  --output-dir var/high_leverage_expansion
```

Additional structure parameters:

```json
{
  "reattack_signal_mode": "high_growth_or_tight_or_structure",
  "price_structure_reattack_mode": "none",
  "structure_reattack_min_momentum_pct": 0.0,
  "structure_reattack_min_ema_gap_pct": 0.25,
  "structure_reattack_min_adx": 0.0,
  "defense_structure_max_stop_distance_pct": 1.9
}
```

In this candidate, `price_structure_reattack_mode` stays `none`; the improvement comes from two changes:

- short-window reattack can treat current structure as a high-quality signal through `high_growth_or_tight_or_structure`;
- defense mode allows structure-qualified trades to use `1.9%` stop distance instead of the normal `1.5%` defense cap.

## Shadow Gate On Fixed Structure Candidate

After fixing the 2026-aware structure candidate above, scan shadow gate parameters on the fixed high-leverage event stream. This does not change the underlying autoTIT trade sequence or the high-leverage structure parameters.

Best result from the refined local scan `/tmp/shadow_on_fixed_structure_refined_scan.json`:

| Window | Return | Sharpe | MaxDD | Notes |
|---|---:|---:|---:|---|
| Full, from `2022-01-01` | `26868.27%` | report output | `35.56%` | `13` shadow-skipped |
| 2026 YTD | `12.73%` | `2.457` | `12.88%` | unchanged versus fixed structure candidate |
| Last 60d | `3.42%` | `1.134` | `11.03%` | unchanged |
| Last 30d | `8.05%` | `4.077` | `3.37%` | unchanged |

Best shadow gate parameters:

```json
{
  "daily_loss_stop_pct": 6.0,
  "equity_drawdown_stop_pct": 15.0,
  "equity_drawdown_cooldown_days": 2,
  "consecutive_loss_stop": 0
}
```

Reproduction command:

```bash
python3 scripts/scan_shadow_on_fixed_high_leverage.py \
  --config config/config.live.5x-3pct.json \
  --data-15m data/okx/futures/BTC_USDT_USDT-15m-futures.feather \
  --data-4h data/okx/futures/BTC_USDT_USDT-4h-futures.feather \
  --start-date 2022-01-01 \
  --daily-loss-values 6 \
  --equity-dd-values 15 \
  --equity-cooldown-values 2 \
  --loss-streak-values 0 \
  --top 1 \
  --output var/high_leverage_expansion/shadow_on_fixed_structure_best.json
```

Refined local search command:

```bash
python3 scripts/scan_shadow_on_fixed_high_leverage.py \
  --config config/config.live.5x-3pct.json \
  --data-15m data/okx/futures/BTC_USDT_USDT-15m-futures.feather \
  --data-4h data/okx/futures/BTC_USDT_USDT-4h-futures.feather \
  --start-date 2022-01-01 \
  --daily-loss-values 4,5,6,7,8 \
  --equity-dd-values 12,13,14,15,16,17,18,20 \
  --equity-cooldown-values 1,2,3,4,5,6 \
  --loss-streak-values 0,4,5,6,7 \
  --top 20 \
  --output /tmp/shadow_on_fixed_structure_refined_scan.json
```

The fixed high-leverage structure parameters are embedded in `scripts/scan_shadow_on_fixed_high_leverage.py` as `FIXED_STRUCTURE_PARAMS`. They match the 2026-aware structure candidate in the previous section.

Important interpretation:

- Re-tuning shadow gate significantly improves full-window compounding: `19050.71%` to `26868.27%`.
- It does not improve 2026 YTD in this run; 2026 remains `12.73%`.
- The improvement comes from skipping historical compounding-damaging periods while leaving the recent event sequence unchanged.
