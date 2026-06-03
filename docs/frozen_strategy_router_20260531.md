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
- `var/reports/qqq_shadow_gate_v2_combined_audit_20220101_20260529.json`

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

## 2026-06-02 Runtime Shadow Gate V2

The QQQ leg was updated from the 2026-05-30 balanced shadow profile to the 2026-06-02 low-DD profile:

```json
{
  "stop_loss_pct": 4.0,
  "reentry_rule": "clear",
  "reentry_clear_bars": 2,
  "loss_streak_stop": 0,
  "loss_streak_cooldown_bars": 0,
  "equity_dd_stop_pct": 15.0,
  "equity_dd_cooldown_bars": 20
}
```

The combined audit now runs risk overlay and shadow gate in the same replay path, using this frozen router config (`btc_min=35`, `qqq_min=96`, `switch=6`, takeover margins `6/6`).

| Candidate | Full NQ Return | Full NQ MaxDD | Closed-only Return | Closed-only MaxDD | Real OKX overlap |
|---|---:|---:|---:|---:|---:|
| Current balanced + risk | `11005839.37%` | `48.49%` | `10732464.72%` | `48.49%` | `269.87% / DD 22.13%` |
| Shadow V2 low-DD + risk | `18505422.00%` | `44.64%` | `18045766.50%` | `44.64%` | `269.87% / DD 22.13%` |

Rolling window improvement of V2 versus current balanced:

- `126d`: DD improved in `27.66%` of windows, CVaR in `51.06%`, Calmar/return in `36.17%`.
- `252d`: DD improved in `43.90%` of windows, CVaR in `75.61%`, Calmar/return in `63.41%`.

Runtime config alignment:

- `config/config.paper.qqq-usdt-aggressive-frozen.json` now carries the V2 shadow profile.
- `config/config.live.strategy-router.template.json` is aligned to the frozen router thresholds: `qqq_min_route_score=96`, `switch_advantage=6`, takeover margins `6/6`.
- Legacy router-side `qqq_stop_reentry_*` settings are removed from the live template; V2 reentry and cooldown are owned only by `shadow_gate_replay_profile`.

## 2026-06-03 QQQ Dollar Cap Overlay

The QQQ frozen leg now also carries a signal-day-aligned macro proxy overlay in the shared runtime/replay/audit path:

- Rule: when `macro_broad_dollar_index_z_252d >= 1.5`, cap QQQ target exposure notional to `50%`
- Data: `data/public/macro/fred_macro-1d.feather`
- Order: `risk overlay -> macro dollar cap -> shadow gate`
- Alignment: macro z-score is evaluated on `daily_signal_timestamp/session_day`, not on repeated natural `4h` bar dates

Shared-path verification against the same config with macro overlay disabled:

| Path | Candidate | Baseline no-macro |
|---|---:|---:|
| Combined audit full NQ | `26115047.82% / DD 41.74%` | `18505422.00% / DD 44.64%` |
| Combined audit closed-only | `25466378.13% / DD 41.74%` | `18045766.50% / DD 44.64%` |
| Real OKX overlap | `269.87% / DD 22.13%` | `269.87% / DD 22.13%` |
| Router replay real overlap | `269.50% / DD 22.13%` | `269.50% / DD 22.13%` |

Current overlap remains unchanged because the short OKX window has `0` macro-trigger bars. In the long NQ proxy sample, the shared audit path reproduces the prior sidecar study topline while using the stricter signal-day alignment.

Artifacts:

- `var/reports/qqq_shadow_gate_v2_combined_audit_dollar_cap50_z1_5_20260603.json`
- `var/reports/qqq_shadow_gate_v2_combined_audit_shadow_v2_nomacro_20260603.json`
- `var/reports/router_replay_qqq_usdt_dollar_cap50_z1_5_20260603.json`
- `var/reports/router_replay_qqq_usdt_shadow_v2_nomacro_20260603.json`

## Remaining Risks

- BTC and QQQ route scores are still not calibrated expected-return scores. The frozen router handles this by using separate empirical entry gates plus symmetric takeover margins.
- Long NQ proxy is synthetic. Real QQQ/USDT overlap is short, so overlap remains a sanity check, not a full-cycle proof.
- Router replay cadence is daily. Runtime QQQ execution is 4h with risk overlay, macro dollar cap, stop, fees, slippage, funding, and shadow gate behavior.
- Exact score ties still follow `strategy_priority`, currently BTC before QQQ. This should only matter in rare equal-score flat-state cases.
