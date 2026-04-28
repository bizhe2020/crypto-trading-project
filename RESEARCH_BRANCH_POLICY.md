# Research Branch Policy

This branch is for high-leverage strategy research only.

## Branch Roles

- `main`: deployable production code for the Tokyo bot.
- `high_leverage_10x_research`: research code, scans, reports, and parameter experiments.

Tokyo deployment should always be done from `main`, not from this branch.

## Allowed On This Branch

- Strategy experiments in `strategy/` when needed for research.
- Research scripts in `scripts/`.
- Research notes and reproduction docs.
- Files under `research/high_leverage/`.
- Tests that validate research or strategy behavior.

## Avoid On This Branch

Do not change production deployment surfaces here:

- `systemd/`
- `scripts/deploy_tokyo.sh`
- `scripts/bootstrap_server.sh`
- `config/config.live*.json`
- `config/config.live*.template.json`
- Telegram command/bot UX code unless the explicit task is to research bot UX.

If a research result should go live, promote it deliberately:

1. Record the exact result and parameters in the reproduction document.
2. Cherry-pick or merge the minimal strategy/config changes into `main`.
3. Verify the production config reproduces the selected result.
4. Deploy Tokyo only from `main`.

## Safety Check

Run this before committing research work:

```bash
bash scripts/check_research_branch_safety.sh
```

The check fails if production deployment files are modified on this branch.
