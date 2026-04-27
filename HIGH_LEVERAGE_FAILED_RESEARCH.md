# High Leverage Failed Research Archive

This file records high-leverage research branches that were tested and rejected. It is an archive only; the executable scan chain for these rejected ideas has been removed from the active branch.

## Controlled Scale-In, Max 2 Slots

Goal: add a controlled second slot to the current best pressure target-cap strategy without increasing total open risk.

Design tested:

- Single aggregate position, not independent concurrent positions.
- At most 2 slots.
- Shared total risk cap.
- Only allowed during `high_growth` regime.
- Required the stop to be at breakeven before adding size.
- Required enough remaining target RR after the add.
- Required max projected stop distance cap after averaging entry.

Result:

| Case | Full Return | MaxDD | 2026 | Last 60d | Add Events | Decision |
|---|---:|---:|---:|---:|---:|---|
| Current best baseline | `38420.70%` | `35.04%` | `19.48%` | `4.63%` | `0` | keep |
| Best actual add-size candidate | `37657.53%` | `35.04%` | `19.48%` | `4.63%` | `1` | reject |
| More permissive sample | `29649.80%` | `35.04%` | `17.31%` | `2.73%` | `8` | reject |

Reason rejected:

- The only candidate that actually added size reduced full-window compounding versus the no-add baseline.
- More permissive settings increased add events but damaged both full-cycle and recent-window returns.
- The best current live/reproducible strategy remains the pressure-level target cap version with no controlled add-size behavior.

Historical strict grid tested:

```json
{
  "max_slots": 2,
  "trigger_rr": [1.5, 2.0],
  "min_bars_held": 8,
  "min_interval_bars": 16,
  "risk_fraction": [0.1, 0.25],
  "total_risk_multiplier": 1.0,
  "max_total_notional_multiplier": 1.0,
  "min_target_rr": 3.0,
  "min_price_move_pct": 0.5,
  "max_stop_distance_pct": [1.0, 1.5],
  "require_stop_at_breakeven": true,
  "regime_labels": ["high_growth"],
  "trail_styles": ["loose"]
}
```
