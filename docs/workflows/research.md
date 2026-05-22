# Research / Parameter Optimization

Scope:

- CN Nasdaq-100 ETF parameter scans
- next-open baseline optimization
- conditional leverage and tiered sizing research
- Kelly bucket audit

Canonical entrypoints:

- `python3 scripts/scan_cn_nasdaq100_params.py`
- `python3 scripts/scan_cn_nasdaq100_next_open_baseline.py`
- `python3 scripts/scan_cn_nasdaq100_next_open_exit_only.py`
- `python3 scripts/scan_cn_nasdaq100_next_open_tiered_only.py`

Primary scripts:

- `scripts/scan_cn_nasdaq100_params.py`
- `scripts/scan_cn_nasdaq100_conditional_leverage.py`
- `scripts/scan_cn_nasdaq100_conditional_leverage_refine.py`
- `scripts/scan_cn_nasdaq100_tiered_sizing.py`
- `scripts/scan_cn_nasdaq100_secondary_reentry.py`

Current research rule:

- Formal baseline lives in `docs/cn_nasdaq100_research_plan.md`.
- New candidates should be evaluated under the same `next_open` execution rule before promotion.
