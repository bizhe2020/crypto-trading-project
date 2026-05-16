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

- `SOTA long + SMC short + gap-SMC short + single-position arbitration`
- `SOTA score gate` is part of the live candidate filter path when enabled in config
- `feature_bearish_structure=false` is part of the promoted long entry filter path
- `SMC short` and `gap-SMC short` remain overlay candidates; `overlay_skip_dynamic_high_leverage` should stay enabled unless explicitly re-audited
- Current promoted gate: `sota_score_net_min=3`, `sota_score_bull_min=8`, `sota_score_bear_max=6`, `sota_score_conflict_mode=any`
- Current promoted exact long bucket: `n3_b9_b6_conflict_target12`
- Current promoted shadow risk gate: daily loss `6%`, equity DD `12%`, cooldown `2` days, consecutive loss stop `4`
- Current promoted trailing: `stage=close`, `time=extreme`, `atr_activation=extreme`, `atr_activation_rr=2.06`, `T_max=144`, `S4_close_rr=0.8`
- Current promoted dynamic sizing: base `4.0`, high-growth/tight-stop/max-effective `7.5`, failed-breakout guard `1.5`
- Frozen baseline reference: `docs/archive/HIGH_LEVERAGE_FROZEN_LIVE_CORE_20260515.md`
- Fallback plan reference: `docs/archive/HIGH_LEVERAGE_2021_FALLBACK_PLAN_20260516.md`
- Fallback switch runbook: `docs/archive/HIGH_LEVERAGE_FALLBACK_SWITCH_RUNBOOK_20260516.md`

Do not use research scan scripts as a deployment entry.
