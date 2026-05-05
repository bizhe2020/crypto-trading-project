# Shadow Gate Best Result Reproduction

This document records how to reproduce the fixed best `autoTIT + shadow_risk_gate` result. It does not reproduce the parameter search. It replays one fixed parameter set.

## Verified Result

Data cutoff: `2026-04-25 03:45:00 UTC`

| Window | Shadow Gate Return | Sharpe | MaxDD | Notes |
|---|---:|---:|---:|---|
| Full, `2022-01-01` to cutoff | `8264.29%` | `3.027` | `36.02%` | `skip_ratio = 16.10%` |
| 2026 YTD, `2026-01-01` to cutoff | `5.25%` | `1.019` | `21.06%` | same fixed gate params |
| Recent 60d | `-9.88%` | `-1.078` | `15.81%` | start is computed from cutoff |
| Recent 30d | `-5.76%` | `-0.758` | `12.78%` | start is computed from cutoff |

The raw autoTIT baseline for the full window is `7073.64%`, Sharpe `2.590`, MaxDD `45.56%`.

The older README values `9240.42%` full-window return and `17.53%` 2026 YTD return did not reproduce with the current local data snapshot, even when rerun against clean commits `b2c3b43` and `32b7ebc` with the same `2026-04-25 03:45 UTC` cutoff. Treat those older values as historical/non-reproducible unless the original data snapshot is recovered.

## Calling Script

Use:

```bash
python3 scripts/reproduce_shadow_gate_best.py \
  --config config/config.live.5x-3pct.json \
  --data-15m data/okx/futures/BTC_USDT_USDT-15m-futures.feather \
  --data-4h data/okx/futures/BTC_USDT_USDT-4h-futures.feather \
  --start-date 2022-01-01 \
  --end-ts "2026-04-25 03:45:00+00:00" \
  --daily-loss-stop-pct 6 \
  --equity-drawdown-stop-pct 21 \
  --equity-drawdown-cooldown-days 6 \
  --consecutive-loss-stop 4
```

Default output:

```text
var/live_readiness/shadow_gate_best_reproduction.json
```

This script runs only the fixed best result path:

1. Load the configured autoTIT strategy.
2. Truncate input data in memory to `--end-ts`.
3. Run raw autoTIT trades for full, YTD, 60d, and 30d windows.
4. Apply `shadow_risk_gate_overlay` with the fixed best params.
5. Print raw autoTIT metrics, shadow gate metrics, skipped trades, and skip ratio.

It does not scan parameters.

## Required Data Snapshot

The exact target result assumes the input data is available from `2022-01-01` through:

| File | Required cutoff |
|---|---|
| `data/okx/futures/BTC_USDT_USDT-15m-futures.feather` | `2026-04-25 03:45:00+00:00` |
| `data/okx/futures/BTC_USDT_USDT-4h-futures.feather` | last 4h candle at or before the 15m cutoff |

The script accepts newer data files, but it truncates them with `--end-ts` before running. If you omit `--end-ts` or use a later cutoff, the 60d/30d windows and final metrics can change.

## Fixed Shadow Gate Parameters

These are the selected best parameters:

```json
{
  "daily_loss_stop_pct": 6.0,
  "equity_drawdown_stop_pct": 21.0,
  "equity_drawdown_cooldown_days": 6,
  "consecutive_loss_stop": 4
}
```

Interpretation:

- `daily_loss_stop_pct = 6.0`: after accepted trades make the UTC day loss reach 6%, skip new real entries until next UTC day.
- `equity_drawdown_stop_pct = 21.0`: after accepted trades make equity drawdown from the shadow peak reach 21%, skip new real entries.
- `equity_drawdown_cooldown_days = 6`: equity drawdown trigger pauses real mirroring for 6 UTC days and resets the shadow drawdown peak.
- `consecutive_loss_stop = 4`: after 4 consecutive losing accepted trades, skip new real entries until next UTC day.

The strategy engine still continues its paper path. The gate only decides whether real execution mirrors each open signal.

## Core Strategy Parameters

Source config: `config/config.live.5x-3pct.json`

Important execution/backtest assumptions:

```json
{
  "symbol": "BTC/USDT:USDT",
  "timeframe": "15m",
  "informative_timeframe": "4h",
  "leverage": 5,
  "margin_mode": "isolated",
  "max_open_positions": 1,
  "risk_per_trade": 0.035,
  "position_size_pct": 1.0,
  "rr_ratio": 4.0,
  "pullback_window": 40,
  "sl_buffer_pct": 1.25,
  "allow_long": true,
  "allow_short": true,
  "taker_fee_rate": 0.0005,
  "slippage_bps": 5.0,
  "enable_regime_switching": true,
  "enable_directional_regime_switch": true,
  "enable_dual_pending_state": true,
  "enable_regime_layered_exit": true,
  "enable_short_regime_layered_exit": true,
  "enable_target_rr_cap": true,
  "disable_fixed_target_exit": false
}
```

ATR and trailing settings:

```json
{
  "enable_atr_trailing": true,
  "atr_period": 14,
  "atr_activation_rr": 2.06,
  "atr_loose_multiplier": 2.7,
  "atr_normal_multiplier": 2.25,
  "atr_tight_multiplier": 1.8,
  "atr_regime_filter": "tight_style_off",
  "enable_time_based_trailing": false,
  "enable_auto_time_based_trailing": true,
  "T1": 10,
  "T2": 20,
  "T_max": 144,
  "S0_trigger_rr": 0.5,
  "S1_trigger_rr": 0.8,
  "S3_trigger_rr": 3.0,
  "S4_close_rr": 0.8
}
```

autoTIT gate settings:

```json
{
  "auto_tit_mode": "loss_streak",
  "auto_tit_drawdown_pct": 12.0,
  "auto_tit_recent_trades": 6,
  "auto_tit_min_completed_trades": 1,
  "auto_tit_recent_rr_threshold": -1.0,
  "auto_tit_loss_streak": 1,
  "auto_tit_entry_regimes": null,
  "auto_tit_regime_labels": ["high_growth"],
  "auto_tit_trail_styles": ["loose"],
  "auto_tit_directions": null,
  "auto_tit_adx_min": null,
  "auto_tit_adx_max": null,
  "auto_tit_momentum_min": null,
  "auto_tit_momentum_max": null,
  "auto_tit_atr_ratio_min": null,
  "auto_tit_atr_ratio_max": 1.1,
  "auto_tit_ema_gap_min": null,
  "auto_tit_ema_gap_max": null
}
```

Live deployment switch:

```json
{
  "enable_shadow_risk_gate": true,
  "shadow_daily_loss_stop_pct": 6.0,
  "shadow_equity_drawdown_stop_pct": 21.0,
  "shadow_equity_drawdown_cooldown_days": 6,
  "shadow_consecutive_loss_stop": 4
}
```

For reproduction, `enable_shadow_risk_gate` is not what computes the historical overlay. The reproduction script passes the fixed shadow params directly into `shadow_risk_gate_overlay`. The live switch only controls whether the execution layer applies the gate in real trading.

## Expected Console Shape

The output should include lines like:

```text
full         raw= 7073.64%/2.590/45.56% shadow= 8264.29%/3.027/36.02% skipped=66 skip_ratio=16.10%
current_year raw=    8.49%/1.154/30.49% shadow=    5.25%/1.019/21.06% skipped=6 skip_ratio=20.00%
recent_60d   raw=    2.54%/0.747/18.18% shadow=   -9.88%/-1.078/15.81% skipped=2 skip_ratio=8.70%
recent_30d   raw=   10.34%/2.322/ 8.34% shadow=   -5.76%/-0.758/12.78% skipped=1 skip_ratio=9.09%
```

Small formatting differences are acceptable. Metric differences are not expected if the code, config, and cutoff are the same.

## Reproduction Caveats

- Do not use the parameter scanner for this reproduction. The scanner was used to discover the params; this document fixes them.
- Do not compare against a run using data beyond `2026-04-25 03:45 UTC`; recent windows will shift.
- Do not move the risk cooldown into the strategy core. The result is an execution-layer overlay where the strategy paper path continues uninterrupted.
- Do not include exchange fills or live slippage in this historical result. This is a deterministic backtest overlay on completed strategy trades.
