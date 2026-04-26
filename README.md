# AutoTIT Research Baseline

This branch keeps the reproducible `0bd614e` TIT baseline and adds `autoTIT`, a gated time-indexed trailing mode for the scalp strategy.

## Current Live Rule

`config/config.live.5x-3pct.json` keeps fixed TIT disabled and enables autoTIT only for a narrow set of entries:

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

The position decides whether TIT is active at entry time and stores that decision on the position state. This avoids changing exit rules mid-trade.

## Backtest Results

Data window: `2022-01-01` to `2026-04-18`.

| Case | Return | Sharpe | MaxDD | Trades |
|---|---:|---:|---:|---:|
| Baseline without TIT | `4616.04%` | `2.445` | `45.56%` | `397` |
| Fixed best TIT | `733.80%` | `1.470` | `54.17%` | `505` |
| Optimized autoTIT | `6924.62%` | `2.598` | `45.56%` | `407` |

Data window: `2026-01-01` to `2026-04-18`.

| Case | Return | Sharpe | MaxDD | Trades |
|---|---:|---:|---:|---:|
| Baseline without TIT | `-2.73%` | `0.193` | `36.36%` | `27` |
| Optimized autoTIT | `6.24%` | `1.053` | `30.49%` | `27` |

## Reproduction

Run the current live config:

```bash
python3 scripts/backtest_config_report.py --config config/config.live.5x-3pct.json --start-date 2022-01-01 --end-date 2026-04-18
python3 scripts/backtest_config_report.py --config config/config.live.5x-3pct.json --start-date 2026-01-01 --end-date 2026-04-18
```

Run syntax checks:

```bash
python3 -m py_compile strategy/scalp_robust_v2_core.py bot/okx_executor.py scripts/backtest_config_report.py
```
