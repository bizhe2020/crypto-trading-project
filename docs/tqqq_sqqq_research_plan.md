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
10. `python3 scripts/audit_tqqq_cash_walk_forward.py`

## Position In Repo

- `CN ETF` remains the active mainline
- `TQQQ/SQQQ` is restored as reusable side research
- Do not mix the two result series into one baseline
