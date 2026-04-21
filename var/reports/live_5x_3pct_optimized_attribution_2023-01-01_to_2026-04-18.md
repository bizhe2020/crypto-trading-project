# Live 5x 3pct Optimized Strategy Attribution Report

## Scope
- Data: 15m=`BTC_USDT_USDT-15m-futures.feather`, 4h=`BTC_USDT_USDT-4h-futures.feather`
- Range: `2023-01-01` to `2026-04-18` inclusive, UTC filter
- Cases:
  - `baseline_3247_rerun`: current config plus ATR and fixed-target overrides, but bear_strong_short optimization reset
  - `optimized_full`: current config plus ATR/fixed-target overrides and full bear_strong_short optimized parameters
  - `raw_head_config`: `git HEAD` version of `config.live.5x-3pct.json`, kept only as extra reference
- Note: the historically referenced `3247%` baseline reruns to `1116.07%` under the current workspace code and the `2023-01-01` to `2026-04-18` data window.

## Applied Optimized Overrides
```json
{
  "atr_regime_filter": "tight_style_off",
  "atr_activation_rr": 2.0,
  "atr_loose_multiplier": 2.7,
  "atr_normal_multiplier": 2.25,
  "atr_tight_multiplier": 1.8,
  "disable_fixed_target_exit": false,
  "enable_atr_trailing": true,
  "allow_bear_strong_short": true,
  "bear_strong_short_pullback_window": 10,
  "bear_strong_short_sl_buffer_pct": 0.5,
  "bear_strong_short_retrace_min_ob_fill_pct": 0.8,
  "bear_strong_short_entry_min_ob_fill_pct": 0.55,
  "bear_strong_short_rr_ratio_override": 2.5,
  "bear_strong_short_trail_style_override": "tight",
  "bear_strong_short_max_hold_bars": 96,
  "bear_strong_short_atr_activation_rr": 1.5,
  "bear_strong_short_atr_loose_multiplier": 1.5
}
```

## 1. Overall
| Case | Return | Sharpe | MaxDD | Trades | WinRate | PF |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_3247_rerun | 1116.07% | 1.478 | 62.36% | 598 | 38.80% | 1.106 |
| optimized_full | 1013.61% | 1.524 | 62.36% | 541 | 39.74% | 1.111 |

## 2. Annual Returns
2026 is YTD through 2026-04-18; `Annualized` is only annualized for partial-year rows.
| Year | Period | PnL | Return | Annualized | StartCap | EndCap |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | Full | 1896.42 | 189.64% | 189.64% | 1000.00 | 2896.42 |
| 2024 | Full | 12211.54 | 421.61% | 421.61% | 2896.42 | 15107.96 |
| 2025 | Full | -3860.74 | -25.55% | -25.55% | 15107.96 | 11247.22 |
| 2026 | YTD | -111.09 | -0.99% | -3.30% | 11247.22 | 11136.13 |

## 3. Long vs Short Breakdown
| Direction | Trades | PnL | Return | WinRate | PF |
| --- | --- | --- | --- | --- | --- |
| LONG | 294 | 12071.09 | 1207.11% | 38.44% | 1.171 |
| SHORT | 247 | -1934.96 | -193.50% | 41.30% | 0.905 |

## 4. Regime Breakdown by LONG / SHORT
| Regime | Direction | Trades | PnL | Return | WinRate | PF | AvgRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bull_strong | LONG | 207 | 13214.05 | 1321.41% | 41.06% | 1.217 | 0.298 |
| bull_strong | SHORT | 1 | 1.45 | 0.15% | 100.00% | 0.000 | 0.139 |
| bull_weak | LONG | 87 | -1142.96 | -114.30% | 32.18% | 0.883 | 0.103 |
| bull_weak | SHORT | 56 | 1430.87 | 143.09% | 30.36% | 1.343 | 0.005 |
| bear_strong | SHORT | 113 | -3105.20 | -310.52% | 36.28% | 0.732 | -0.090 |
| bear_weak | SHORT | 77 | -262.08 | -26.21% | 55.84% | 0.944 | -0.036 |

## 5. Bear_strong Short Before vs After
| Case | Trades | PnL | Return | WinRate | PF | StopLoss | AvgHoldH |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_3247_rerun | 171 | -1659.88 | -165.99% | 34.50% | 0.932 | 97.66% | 24.85 |
| optimized_full | 113 | -3105.20 | -310.52% | 36.28% | 0.732 | 68.14% | 4.54 |

Bear_strong short delta vs `baseline_3247_rerun`:
| Bucket | ReturnDelta | SharpeDelta | MaxDDDelta | TradesDelta | WinRateDelta | PFDelta |
| --- | --- | --- | --- | --- | --- | --- |
| overall | -102.46pp | +0.046 | +0.00pp | -57 | +0.94pp | +0.005 |
| bear_strong_short | -144.53pp | - | - | -58 | +1.78pp | -0.200 |

- Bear_strong short pnl delta: `-1445.32`
- Bear_strong short stop-loss delta: `-29.52pp`

## 6. Trail Style Distribution (B / M / S)
| Style | Trades | Share | PnL | Return | WinRate | PF |
| --- | --- | --- | --- | --- | --- | --- |
| B | 207 | 38.26% | 13214.05 | 1321.41% | 41.06% | 1.217 |
| M | 137 | 25.32% | 151.66 | 15.17% | 29.20% | 1.011 |
| S | 197 | 36.41% | -3229.58 | -322.96% | 45.69% | 0.805 |

## 7. Exit Reason Distribution
| ExitReason | Trades | PnL | Return | WinRate | AvgHoldH |
| --- | --- | --- | --- | --- | --- |
| stop_loss | 475 | -11411.63 | -1141.16% | 32.21% | 24.36 |
| target_4r | 62 | 21794.91 | 2179.49% | 100.00% | 21.06 |
| time_exit | 4 | -247.16 | -24.72% | 0.00% | 24.00 |

## 8. Monthly PnL Top 5
| Month | Trades | PnL | Return | StartCap | EndCap |
| --- | --- | --- | --- | --- | --- |
| 2024-11 | 16 | 6085.68 | 71.45% | 8517.11 | 14602.79 |
| 2024-02 | 12 | 4207.30 | 175.67% | 2395.03 | 6602.32 |
| 2024-10 | 12 | 3110.08 | 57.52% | 5407.03 | 8517.11 |
| 2025-08 | 11 | 2157.02 | 24.94% | 8648.11 | 10805.13 |
| 2026-02 | 10 | 1607.84 | 12.79% | 12575.78 | 14183.62 |

## 9. Monthly PnL Bottom 5
| Month | Trades | PnL | Return | StartCap | EndCap |
| --- | --- | --- | --- | --- | --- |
| 2026-03 | 21 | -3386.47 | -23.88% | 14183.62 | 10797.16 |
| 2025-04 | 16 | -2077.27 | -18.77% | 11065.52 | 8988.25 |
| 2025-06 | 15 | -1651.87 | -15.69% | 10530.98 | 8879.11 |
| 2025-03 | 11 | -1538.35 | -12.21% | 12603.87 | 11065.52 |
| 2024-03 | 27 | -1511.02 | -22.89% | 6602.32 | 5091.31 |

## 10. Comparison vs Original 3247 Configuration
| Comparison | ReturnDelta | SharpeDelta | MaxDDDelta | TradesDelta | WinRateDelta | PFDelta |
| --- | --- | --- | --- | --- | --- | --- |
| optimized_full vs baseline_3247_rerun | -102.46pp | +0.046 | +0.00pp | -57 | +0.94pp | +0.005 |

## Appendix. Extra Reference vs Raw Head Config
| Comparison | ReturnDelta | SharpeDelta | MaxDDDelta | TradesDelta | WinRateDelta | PFDelta |
| --- | --- | --- | --- | --- | --- | --- |
| optimized_full vs raw_head_config | +407.58pp | +0.139 | -1.50pp | +75 | +10.98pp | -0.138 |
