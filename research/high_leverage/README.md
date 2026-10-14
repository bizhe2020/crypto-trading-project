# High Leverage Research Workspace

Use this directory for high-leverage experiments that should not be confused with the live Tokyo bot.

## What Goes Here

- New scan plans.
- Temporary parameter grids.
- Result summaries.
- Loss-bucket notes.
- Candidate promotion notes.

## Current Promoted Live-Bot Candidate

The current promoted live-bot candidate is:

- `docs/archive/HIGH_LEVERAGE_SOTA_SMC_LIVE_BOT_20260506.md`
- `config/config.live.high-leverage-structure.template.json`
- `config/config.paper.high-leverage-structure.json`
- `config/live_drift_baseline.high_leverage.json`

Target reference:

- Full: `12911.21%`
- MaxDD: `28.36%`
- 2026: `30.95%`
- 2026 MaxDD: `17.08%`
- Accepted events: SOTA long `275`, SMC short `24`
- Rejected events: `3`, all due single-position lock

## Previous Promoted Baseline

The previous pressure-shadow research result is documented in:

- `docs/archive/HIGH_LEVERAGE_REATTACK_REPRODUCTION.md`
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

The Telegram integration (drift push, `/drift` `/ob` commands, hourly OB status, open/close notifications) was removed from the BTC bot on 2026-10-13; run the monitor locally or on the server when a drift check is needed.
