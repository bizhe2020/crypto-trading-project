# High Leverage Research Workspace

Use this directory for high-leverage experiments that should not be confused with the live Tokyo bot.

## What Goes Here

- New scan plans.
- Temporary parameter grids.
- Result summaries.
- Loss-bucket notes.
- Candidate promotion notes.

## Current Promoted Baseline

The latest promoted research result is still documented in:

- `HIGH_LEVERAGE_REATTACK_REPRODUCTION.md`
- `config/high_leverage_pressure_target_cap_best.params.json`
- `scripts/reproduce_pressure_target_cap_best.sh`

Target reference:

- Full: `88481.28%`
- MaxDD: `33.87%`
- 2026: `29.87%`
- 2026 MaxDD: `11.35%`
- Last 60d: `7.85%`
- Last 30d: `8.47%`

## Workflow

1. Keep exploratory outputs in this directory or under ignored `var/`.
2. Do not deploy from this branch.
3. Run `bash scripts/check_research_branch_safety.sh` before committing.
4. When a candidate beats the baseline, update the reproduction command and parameter notes first.

## Live Drift Monitor

The promoted baseline now has a drift monitor config:

- `config/live_drift_baseline.high_leverage.json`
- `scripts/live_drift_monitor.py`

Local check:

```bash
python3 scripts/live_drift_monitor.py \
  --config config/config.live.high-leverage-structure.template.json \
  --window-days 30 \
  --recent-trades 20
```

Telegram push requires the selected config to contain `telegram_token` and `telegram_chat_id`:

```bash
python3 scripts/live_drift_monitor.py \
  --config config/config.live.high-leverage-structure.template.json \
  --window-days 30 \
  --recent-trades 20 \
  --send-telegram
```

The live bot also answers Telegram commands when `telegram_enabled` is true:

- `/drift`, `/health`, `/体检`: drift health report.
- `/ob`, `/status`, `/状态`: OB status and missing open conditions.

During `run_loop`, the bot pushes OB status hourly and drift status every 24 hours by default.
