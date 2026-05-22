# Live / Shadow

Scope:

- CN Nasdaq-100 ETF paper execution
- CN Nasdaq-100 ETF live-shadow logging

Canonical entrypoints:

- `python3 scripts/run_cn_nasdaq_etf_paper_plan.py`
- `python3 scripts/run_cn_nasdaq_etf_live_shadow.py`

Primary production surfaces:

- `config/config.paper.cn-nasdaq100-etf.json`
- `config/config.live-shadow.cn-nasdaq100-etf.json`
- `scripts/nasdaq100_cn_strategy_utils.py`
- `scripts/run_cn_nasdaq_etf_paper_plan.py`
- `scripts/run_cn_nasdaq_etf_live_shadow.py`

Current baseline:

- `QQQ 25/200`
- `IXIC up + VIX low/normal`
- `max_hold_days=90`
- `trailing_lookback_days=5`
- `trailing_drawdown_pct=8`
- `execution_price_mode=next_open`
- `vix_low = 2.0x`
- `vix_normal + qqq_strong = 1.75x`

Reference document:

- `docs/cn_nasdaq100_research_plan.md`
