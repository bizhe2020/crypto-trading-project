# Replay / Audit

Scope:

- CN Nasdaq-100 ETF formal replay
- execution timing/slippage audit
- baseline robustness audit
- conditional leverage and tiered sizing scans

Canonical entrypoints:

- `python3 scripts/run_cn_nasdaq_etf_live_shadow.py`
- `python3 scripts/audit_cn_nasdaq100_execution_slippage.py`
- `python3 scripts/audit_cn_nasdaq100_baseline_robustness.py`

Primary scripts:

- `scripts/nasdaq100_cn_strategy_utils.py`
- `scripts/audit_cn_nasdaq100_execution_slippage.py`
- `scripts/audit_cn_nasdaq100_baseline_robustness.py`
- `scripts/audit_cn_nasdaq100_bucket_kelly.py`

Current replay rule:

- Use `execution_price_mode=next_open` as the formal baseline.
- Compare candidate upgrades against the current baseline in `docs/cn_nasdaq100_research_plan.md`.
