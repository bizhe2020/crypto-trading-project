# Proxy Strategy Router Replay

This is a proxy history replay, not a real long-history QQQ/USDT contract backtest.

## Scope

- Period: `2026-03-04 -> 2026-05-29`
- Days: `87`
- BTC leg: research frozen live-shadow artifact `/Users/laoji/projects/crypto-trading-project/var/high_leverage_expansion/frozen_live_core_20260515.json`
- QQQ leg: QQQ/USDT leveraged aggressive frozen
- Router cadence: daily
- Router params: `btc_min=35.0`, `qqq_min=120.0`, `switch=4.0`, `btc_takeover=4.0`, `qqq_takeover=4.0`

## Results

- BTC-only: `104.16% / DD 15.25%`
- QQQ/USDT-only: `156.86% / DD 22.13%`
- Router: `204.39% / DD 22.13%`

## Selection

- BTC days: `30`
- QQQ proxy days: `22`
- Cash days: `35`
- Switches: `19`

## Interpretation

This corrected replay uses the research frozen BTC strategy and the leveraged QQQ/USDT frozen candidate.
On the short OKX QQQ/USDT overlap window, the leveraged QQQ leg beats BTC frozen, and daily routing beats both standalone legs.
The result is promising for near-term routing, but the QQQ/USDT sample is still too short to prove long-cycle robustness.
