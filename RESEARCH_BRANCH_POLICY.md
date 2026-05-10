# Research Branch Policy

Current branch roles:

- `new_strategy_research`: canonical branch for research, replay/live convergence, and Tokyo production deployment.
- `main`: compatibility mirror only. Keep it aligned to `new_strategy_research` when external tooling expects `main`; do not develop separate production logic there.
- `high_leverage_10x_research`: historical long-running research branch; not the canonical promotion branch anymore.

Tokyo deployment should be done from `new_strategy_research`.

## Canonical Workflow Docs

- `docs/workflows/live.md`
- `docs/workflows/replay.md`
- `docs/workflows/research.md`

## Allowed On Research Branches

- Strategy experiments in `strategy/` when needed for replay/live convergence.
- Replay, audit, and research scripts under `scripts/`.
- Research notes and reports under `research/` or `docs/archive/`.
- Tests that validate replay/live behavior.

## Protected Production Surfaces

Treat these as production surfaces and review them as promotion changes, not casual research edits:

- `systemd/`
- `scripts/deploy_tokyo.sh`
- `scripts/bootstrap_server.sh`
- `config/config.live*.json`
- `config/config.live*.template.json`
- live Telegram command/bot UX code

## Deployment Rule

If a research result should go live:

1. Reproduce it through the replay/audit workflow.
2. Reduce it to the minimal production diff on `new_strategy_research`.
3. Push `new_strategy_research`.
4. Optionally mirror `main` to the same commit for compatibility.
5. Deploy Tokyo from `new_strategy_research`.

## Safety Check

Run:

```bash
bash scripts/check_research_branch_safety.sh
```

The check is enforced on active research branches and fails if protected production files are changed without deliberate promotion.
