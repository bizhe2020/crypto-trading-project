# Proxy Strategy Router Replay

This is a proxy history replay, not a real long-history QQQ/USDT contract backtest.

## Scope

- Period: `2026-03-04 -> 2026-05-29`
- Days: `87`
- BTC leg: research frozen live-shadow artifact `/private/tmp/crypto-trading-new-strategy-research-work/var/high_leverage_expansion/frozen_live_core_20260515.json`
- QQQ leg: QQQ/USDT leveraged aggressive frozen
- Router cadence: daily

## Results

- BTC-only: `89.79% / DD 15.25%`
- QQQ/USDT-only: `124.82% / DD 26.02%`
- Router: `202.01% / DD 26.02%`

## Selection

- BTC days: `27`
- QQQ proxy days: `30`
- Cash days: `30`
- Switches: `18`

## QQQ Execution Guards

- Stop reentry guard: `True`
- Reclaim buffer: `0.25%`
- Min closed 4h bars: `3`
- Allow new daily signal reset: `True`
- Same-signal stop locks: `0`
- Reentry guard blocks: `0`

## Interpretation

This corrected replay uses the research frozen BTC strategy and the leveraged QQQ/USDT frozen candidate with prior-closed-4h signal execution.
On the short OKX QQQ/USDT overlap window, the leveraged QQQ leg beats BTC frozen.
Daily routing beats BTC frozen on return and has higher drawdown.
The QQQ/USDT sample is still too short to prove long-cycle robustness.
