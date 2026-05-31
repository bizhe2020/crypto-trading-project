# Frozen Strategy Router 2026-05-31

## Status

Final paper/shadow router baseline:

- Main config: `config/config.paper.strategy-router.json`
- BTC leg: `config/config.paper.high-leverage-structure.json`
- QQQ leg: `config/config.paper.qqq-usdt-aggressive-frozen.json`
- Research comparison config retained: `config/config.paper.strategy-router.extreme-qqq-research.json`

The router is frozen around fair BTC vs QQQ competition with trend continuity. BTC is not a core allocation and QQQ is not a permanent satellite. Both can hold the single capital pool after passing their own empirical validity gates.

## Frozen Rules

Parameters:

```json
{
  "btc_min_route_score": 35.0,
  "qqq_min_route_score": 96.0,
  "switch_advantage": 6.0,
  "btc_takeover_advantage": 6.0,
  "qqq_takeover_advantage": 6.0
}
```

Selection rules:

- Flat state: candidates must be active and pass their own min score; highest route score wins. `switch_advantage` is not used while flat.
- Holding BTC: QQQ can take over only if QQQ is active, `qqq_score >= 96`, and `qqq_score >= btc_score + 6`.
- Holding QQQ: BTC can take over only if BTC is active, `btc_score >= 35`, and `btc_score >= qqq_score + 6`.
- Entry min score is not an exit rule. A current active leg can remain held below its entry min until it becomes inactive or a qualified challenger wins.
- Runtime hysteresis is based on execution state `current_executed_strategy`, not stale previous router selection.
- If switching into QQQ is blocked by QQQ risk-on window or shadow gate, BTC is not flattened first.
- Router-triggered BTC flatten uses `OkxExecutionEngine.close_for_router_switch()` so BTC shadow gate and dynamic leverage close state stay synchronized.

## Evidence

All rows use the corrected replay summary: initial capital is the denominator, so first-day switch cost is included. BTC-only in the table is calendar-aligned with the router replay. `btc_equity_raw` in CSV files is a separate full BTC artifact event-compounded line and must not be mixed with the aligned summary.

| Candidate | Path | Return | MaxDD | 2026 | BTC days | QQQ days | Cash days | Switches |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline `qmin60/switch8` | Long NQ proxy | `4247969.68%` | `52.69%` | `336.56%` | `235` | `455` | `418` | `205` |
| Baseline `qmin60/switch8` | Real OKX overlap | `269.50%` | `22.13%` | `269.50%` | `27` | `30` | `30` | `18` |
| Frozen fair-trend `qmin96/switch6` | Long NQ proxy | `5770959.23%` | `52.69%` | `430.18%` | `255` | `397` | `456` | `218` |
| Frozen fair-trend `qmin96/switch6` | Real OKX overlap | `269.50%` | `22.13%` | `269.50%` | `27` | `30` | `30` | `18` |
| Extreme comparison `qmin120/switch4` | Long NQ proxy | `7806515.29%` | `38.41%` | `388.46%` | `303` | `151` | `654` | `247` |
| Extreme comparison `qmin120/switch4` | Real OKX overlap | `204.39%` | `22.13%` | `204.39%` | `30` | `22` | `35` | `19` |

Final artifacts:

- `var/reports/router_fair_trend_qmin96_switch6_nq_continuous_20260531.json`
- `var/reports/router_fair_trend_qmin96_switch6_nq_continuous_20260531.csv`
- `var/reports/router_fair_trend_qmin96_switch6_real_overlap_20260531.json`
- `var/reports/router_fair_trend_qmin96_switch6_real_overlap_20260531.csv`

Comparison artifacts retained:

- `var/reports/router_baseline_qmin60_switch8_nq_continuous_current_20260531.*`
- `var/reports/router_baseline_qmin60_switch8_real_overlap_current_20260531.*`
- `var/reports/router_extreme_qmin120_switch4_nq_continuous_current_20260531.*`
- `var/reports/router_extreme_qmin120_switch4_real_overlap_current_20260531.*`

## Verification

Commands run:

```bash
python3 -m py_compile bot/strategy_router.py bot/router_executor.py bot/okx_executor.py scripts/replay_proxy_strategy_router.py scripts/scan_router_calibrated_utility.py
pytest -q
```

Result:

- `py_compile` passed.
- `pytest -q`: `48 passed`, with one external urllib3 LibreSSL warning.
- Replayed final fair-trend long NQ proxy and real OKX overlap.
- Replayed baseline and extreme comparisons after fixing summary denominator.

## Remaining Risks

- BTC and QQQ route scores are still not calibrated expected-return scores. The frozen router handles this by using separate empirical entry gates plus symmetric takeover margins.
- Long NQ proxy is synthetic. Real QQQ/USDT overlap is short, so overlap remains a sanity check, not a full-cycle proof.
- Router replay cadence is daily. Runtime QQQ execution is 4h with risk overlay, stop, fees, slippage, funding, and shadow gate behavior.
- Exact score ties still follow `strategy_priority`, currently BTC before QQQ. This should only matter in rare equal-score flat-state cases.
