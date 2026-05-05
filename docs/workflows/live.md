# Live / Tokyo

Scope:

- local live config preparation
- local bot startup
- Tokyo server deployment
- production health and drift monitoring

Canonical entrypoints:

- `bash scripts/workflows/live/prepare_live_config.sh`
- `bash scripts/workflows/live/run_local_bot.sh`
- `bash scripts/workflows/live/deploy_tokyo.sh`

Primary production surfaces:

- `bot/run_bot.py`
- `bot/okx_executor.py`
- `scripts/prepare_high_leverage_live_config.py`
- `scripts/run_high_leverage_structure_live.sh`
- `scripts/deploy_tokyo.sh`
- `systemd/crypto-trading-bot-5x3pct.service`

Recommended flow:

1. Prepare `config/config.live.high-leverage-structure.json` from the template and current live secrets.
2. Run the local live bot path if you need a dry local boot.
3. Deploy to Tokyo from `main`.
4. Use `scripts/live_drift_monitor.py` after startup if you need runtime drift diagnostics.

Current live candidate stack:

- `SOTA long + SMC short + single-position arbitration`
- `SOTA score gate` is part of the live candidate filter path when enabled in config
- `SMC short` remains an overlay candidate; `overlay_skip_dynamic_high_leverage` should stay enabled unless explicitly re-audited

Do not use research scan scripts as a deployment entry.
