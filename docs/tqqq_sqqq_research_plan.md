# TQQQ / SQQQ Research Plan

This document restores the earlier `TQQQ/SQQQ` trend-research branch as a side research track.

It is not the current production baseline. The current production-facing path remains the CN Nasdaq-100 ETF workflow in `docs/cn_nasdaq100_research_plan.md`.

## Scope

- Research target: `TQQQ/CASH` trend-following first
- `SQQQ` is a secondary overlay / comparison path, not the default mainline
- Data source: public daily ETF history
- Execution assumption: end-of-day / next-day daily rotation research, not intraday live execution

## Restored Research Stack

- Data health audit:
  - `scripts/audit_tqqq_sqqq_data_health.py`
- Simple baseline replay:
  - `scripts/replay_tqqq_sqqq_trend_baseline.py`
- Trend parameter scan:
  - `scripts/scan_tqqq_cash_trend_params.py`
- Exit profile scan:
  - `scripts/scan_tqqq_cash_exit_profiles.py`
- Regime context audit:
  - `scripts/audit_tqqq_cash_regime_context.py`
- Regime filter scan:
  - `scripts/scan_tqqq_cash_regime_filters.py`
- Regime-aware exit scan:
  - `scripts/scan_tqqq_cash_regime_exit_profiles.py`
- Selective `SQQQ` overlay scan:
  - `scripts/scan_tqqq_sqqq_overlay.py`
- Robustness audit:
  - `scripts/audit_tqqq_cash_regime_exit_robustness.py`
- Walk-forward audit:
  - `scripts/audit_tqqq_cash_walk_forward.py`

## Historical Mainline

The historical research direction was:

- `QQQ` moving-average trend as the driver
- Trade `TQQQ/CASH`
- Add `IXIC trend` and `VIX` as regime filters
- Optimize exits before adding more entry complexity

The most important conclusion from that line was:

- `TQQQ/CASH` was much stronger than mechanically switching `TQQQ/SQQQ`
- Regime awareness mattered more than adding extra entry signals
- The useful filters were mainly `IXIC up` and `VIX low/normal`

## Current Candidate Stack

Current useful candidates split into three tiers:

- `Stable`
  - `QQQ 25/200`
  - long mask: `vix_ixic`
  - exit: `90d + 30d/12%`
  - result: `2799.69% / DD 15.23% / 2026 39.62%`
  - actual trades: `4`
  - average hold: about `79.25` trading days
- `Middle`
  - `QQQ 25/200`
  - long mask: `vix_ixic`
  - exit: `90d + 10d/10%`
  - result: `2383.95% / DD 15.23% / 2026 39.62%`
  - actual trades: `10`
  - average hold: about `36.38` trading days
- `Broader`
  - `QQQ 25/200`
  - long mask: `ixic_filter`
  - exit: `90d + 10d/8%`
  - result: `1560.92% / DD 16.46% / 2026 39.62%`
  - actual trades: `15`
  - average hold: about `28.0` trading days

The practical takeaway is:

- tightening exit from `30d/12%` to `10d/8%-10%` is the main way to increase frequency
- relaxing the regime filter from `vix_ixic` to `ixic_only` is the main way to broaden coverage without fully collapsing quality
- widening all the way to `vix_only` or no regime filter still looks weak

## Selective SQQQ Overlay

The recent overlay experiment tested `TQQQ / SQQQ / CASH` on the same daily framework instead of mechanical `TQQQ/SQQQ` switching.

Main conclusion:

- broad short masks are still bad
- the only positive overlay so far is an extremely narrow `SQQQ` trigger
- that trigger is effectively `rel_weak_vix_high`, and in practice it only adds `1` short trade

Concrete results:

- `Stable + SQQQ(rel_weak_vix_high)`
  - `2950.41% / DD 15.23% / 2026 39.62%`
  - compared with `Stable` long-only `2799.69% / DD 15.23%`
  - net effect: positive, but frequency increase is only from one extra short trade
- `Middle + SQQQ(rel_weak_vix_high)`
  - `2513.06% / DD 15.23% / 2026 39.62%`
  - compared with `Middle` long-only `2383.95% / DD 15.23%`
- `Broader + SQQQ(rel_weak_vix_high)`
  - `1647.25% / DD 16.46% / 2026 39.62%`
  - compared with `Broader` long-only `1560.92% / DD 16.46%`

What failed:

- wider short masks such as `ixic_down_vix_high`
  - increase trade count a lot
  - but push drawdown from about `15%-16%` to about `42%`
  - so they should not be treated as baseline candidates

Current recommendation:

- keep the mainline as `TQQQ/CASH`
- treat `SQQQ` only as a rare overlay, not as a symmetric short system
- if the goal is both higher return and higher frequency, continue from `Middle` and `Broader`
- if the goal is highest return with acceptable complexity, `Stable + narrow SQQQ overlay` is currently the best candidate

## BTC-Style Context / Bucket Migration

The first useful migration from the BTC research line was not parameter copying. It was the framework:

- use context scores instead of one binary filter
- allow different exit behavior for different long-quality buckets
- expand `SQQQ` from one narrow trigger into a small bearish context bucket

First audited result:

- base comparison
  - `Stable + short=off`: `2799.69% / DD 15.23%`
  - `Stable + narrow_base`: `2950.41% / DD 15.23%`
  - `Stable + bearish_score5`: `5090.00% / DD 15.23%`

What `bearish_score5` means in practice:

- `IXIC down`
- `VIX high/extreme`
- weak or failing `QQQ` context
- negative momentum / distance / breakdown context combined into a short score

Observed behavior:

- it expands `SQQQ` from `1` trade to `6` trades
- most of the gain still comes from the long side, but the short bucket adds real incremental return
- drawdown did not increase in this first audit

Important caveat:

- the short bucket is still concentrated
- the single best short trade was `2025-04-03 -> 2025-04-08`, about `44.6%`
- removing that one short still leaves the candidate above the old `Stable` baseline, but the contribution is meaningful enough that it must be treated as a research candidate, not production baseline yet

What did not work:

- long-side bucketized exits did not beat `stable_base` by themselves
- the strongest improvement so far came from the expanded bearish context bucket, not from more complicated long-side exit splitting

Current research direction:

- keep `Stable` as the long backbone
- continue hard-auditing the `bearish_score5` short bucket
- only after that, consider a second layer such as bucket-specific long exits or a stricter breakout/pullback long context profile

## Practical Use

Typical order to rerun this line:

1. `python3 scripts/fetch_public_etf_history.py --symbol QQQ --symbol TQQQ --symbol SQQQ --symbol SPY --symbol ^IXIC --symbol ^VIX`
2. `python3 scripts/audit_tqqq_sqqq_data_health.py`
3. `python3 scripts/replay_tqqq_sqqq_trend_baseline.py`
4. `python3 scripts/scan_tqqq_cash_trend_params.py`
5. `python3 scripts/scan_tqqq_cash_exit_profiles.py`
6. `python3 scripts/audit_tqqq_cash_regime_context.py`
7. `python3 scripts/scan_tqqq_cash_regime_filters.py`
8. `python3 scripts/scan_tqqq_cash_regime_exit_profiles.py`
9. `python3 scripts/audit_tqqq_cash_regime_exit_robustness.py`
10. `python3 scripts/scan_tqqq_sqqq_overlay.py`
11. `python3 scripts/scan_tqqq_context_bucket_overlays.py`
12. `python3 scripts/audit_tqqq_context_bucket_overlay.py`
13. `python3 scripts/audit_tqqq_cash_walk_forward.py`

## Position In Repo

- `CN ETF` remains the active mainline
- `TQQQ/SQQQ` is restored as reusable side research
- Do not mix the two result series into one baseline
