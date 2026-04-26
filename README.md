# AutoTIT Shadow Gate Strategy

This branch keeps one active research path: `autoTIT` entries managed by an execution-layer `shadow_risk_gate`.

The strategy engine always runs the continuous paper path. The execution gate decides whether the real account mirrors each open signal. This is intentional: it preserves `pending_pullback`, `waiting_for_pullback`, and autoTIT state while allowing live execution to skip trades during risk cooldowns.

## Active Config

Live config: `config/config.live.5x-3pct.json`

Core autoTIT settings:

- `enable_time_based_trailing = false`
- `enable_auto_time_based_trailing = true`
- `T1 = 10`
- `T2 = 20`
- `T_max = 144`
- `S0_trigger_rr = 0.5`
- `S1_trigger_rr = 0.8`
- `S3_trigger_rr = 3.0`
- `S4_close_rr = 0.8`
- `auto_tit_mode = "loss_streak"`
- `auto_tit_loss_streak = 1`
- `auto_tit_regime_labels = ["high_growth"]`
- `auto_tit_trail_styles = ["loose"]`
- `auto_tit_atr_ratio_max = 1.1`

Shadow gate settings:

- `enable_shadow_risk_gate = false` by default
- `shadow_daily_loss_stop_pct = 6.0`
- `shadow_equity_drawdown_stop_pct = 21.0`
- `shadow_equity_drawdown_cooldown_days = 6`
- `shadow_consecutive_loss_stop = 4`

Enable `enable_shadow_risk_gate` only after confirming exchange state and using small live size first.

## Latest Backtest

Data window ends at `2026-04-25 03:45 UTC`.

| Window | Raw autoTIT | Shadow Gate | Sharpe | MaxDD | Skipped |
|---|---:|---:|---:|---:|---:|
| `2022-01-01` to data end | `6742.00%` | `9240.42%` | `3.096` | `36.02%` | `65` |
| `2026-01-01` to data end | `3.48%` | `17.53%` | `2.319` | `21.06%` | `5` |

These results are for the shadow-gate execution model, not a strategy-internal hard stop. Strategy-internal hard guards were removed because they interrupt the signal path and do not reproduce the selected research result.

## Commands

Run live-readiness:

```bash
python3 scripts/live_readiness_report.py --config config/config.live.5x-3pct.json
```

Scan shadow gate parameters:

```bash
python3 scripts/scan_shadow_risk_gate_params.py --config config/config.live.5x-3pct.json
```

Run syntax checks:

```bash
python3 -m py_compile strategy/scalp_robust_v2_core.py bot/okx_executor.py scripts/live_readiness_report.py scripts/scan_shadow_risk_gate_params.py
```

## Architecture Rule

Long-term iteration should keep this split:

- `Strategy Engine`: produces the uninterrupted paper path.
- `Execution Gate`: decides whether real capital mirrors each open signal.
- `State Reconciliation`: checks that shadow state and exchange state do not conflict.

Do not add risk cooldowns back into the strategy core if the goal is to preserve the shadow-gate research result.
