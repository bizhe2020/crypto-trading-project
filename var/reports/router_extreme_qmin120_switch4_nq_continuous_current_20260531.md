# Proxy Strategy Router Replay

This is a proxy history replay, not a real long-history QQQ/USDT contract backtest.

## Scope

- Period: `2022-01-03 -> 2026-05-29`
- Days: `1108`
- BTC leg: research frozen live-shadow artifact `/Users/laoji/projects/crypto-trading-project/var/high_leverage_expansion/frozen_live_core_20260515.json`
- QQQ leg: QQQ/USDT leveraged aggressive frozen
- Router cadence: daily
- Router params: `btc_min=35.0`, `qqq_min=120.0`, `switch=4.0`, `btc_takeover=4.0`, `qqq_takeover=4.0`

## Results

- BTC-only: `668050.25% / DD 28.49%`
- QQQ/USDT-only: `4835.78% / DD 53.93%`
- Router: `7806515.29% / DD 38.41%`

## Selection

- BTC days: `303`
- QQQ proxy days: `151`
- Cash days: `654`
- Switches: `247`

## Interpretation

This corrected replay uses the research frozen BTC strategy and the leveraged QQQ/USDT frozen candidate.
On the short OKX QQQ/USDT overlap window, the leveraged QQQ leg beats BTC frozen, and daily routing beats both standalone legs.
The result is promising for near-term routing, but the QQQ/USDT sample is still too short to prove long-cycle robustness.
