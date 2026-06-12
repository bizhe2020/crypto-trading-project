# Tokyo Git Deployment

This is the canonical deployment path for the Tokyo live router. Do not deploy by copying ad hoc files with `scp`; commit the deployable state, push it, then let the remote release pull from git.

## Active Services

- Router service: `crypto-strategy-router`
- Risk refresh timer: `qqq-risk-refresh.timer`
- Active project directory: detected from `systemctl cat crypto-strategy-router`
- Live config: detected from the router `ExecStart --config` argument

The live config file is intentionally ignored by git because it may contain live-only credentials or notification settings. The git deploy script validates it but does not overwrite it by default.

## Deploy Command

Use a clean local worktree:

```bash
git status --short
python3 -m pytest -q
git push origin hygiene/router-risk-20260603
SSHPASS=... PUSH_FIRST=0 bash scripts/workflows/live/deploy_tokyo.sh
```

Useful flags:

- `DEPLOY_REF=hygiene/router-risk-20260603`: branch to deploy; defaults to current branch.
- `PUSH_FIRST=1`: push the branch before remote pull.
- `SYNC_ROUTER_LIVE_CONFIG=1`: copy `config/config.live.strategy-router.template.json` over the live config. Default is `0`; only use this after confirming live-only fields are safe to replace.
- `RESTART_RISK_TIMER=1`: restart the risk refresh timer after deploy. Default is `0`; the timer normally picks up new code on its next run.

The script backs up remote dirty git status and diff under `var/backups/git_deploy_tokyo_*`, stashes remote uncommitted changes, runs `git fetch`, `git checkout`, `git pull --ff-only`, validates JSON configs, compiles the router/runtime modules, restarts the router, and prints heartbeat plus recent bootstrap/evaluate summaries.

## Frozen And Runtime Configs

The frozen replay config and live runtime config are deliberately separate:

- Frozen/replay QQQ config: `config/config.paper.qqq-usdt-aggressive-frozen.json`
- Frozen label: `qqq_usdt_aggressive_fixed10_risk_overlay_stop4_shadow_v2_low_dd_dollar_cap50_z1_5_20260603`
- Frozen risk CSV inputs: `var/reports/qqq_drawdown_lgb_shadow_predictions_macro_subfactor_core.csv` and `var/reports/qqq_long_cycle_correction20d10_qqqonly_lgb_predictions.csv`
- Live/runtime QQQ config: `config/config.paper.qqq-usdt-aggressive-runtime.json`
- Runtime label: `qqq_usdt_aggressive_fixed10_no_risk_overlay_shadow_v2_low_dd_dollar_cap50_z1_5_20260609_runtime`
- Runtime risk CSV path namespace: `var/runtime/qqq_risk/`

Current Tokyo runtime uses fixed `10x` QQQ exchange leverage, `4.0%` trailing stop, macro dollar cap overlay enabled, and shadow gate V2 with `clock=signal_session`. The runtime config keeps rolling risk CSV paths for rollback/diagnostics, but `risk_overlay_enabled=false` in the current no-risk runtime variant.

The live router template points QQQ at the runtime config:

```json
{
  "qqq_strategy_config": "config/config.paper.qqq-usdt-aggressive-runtime.json",
  "qqq_min_route_score": 96.0,
  "switch_advantage": 6.0,
  "okx_markets_cache_path": "var/okx/markets_cache.json"
}
```

## Post-Deploy Checks

Run these if the wrapper output is insufficient:

```bash
systemctl is-active crypto-strategy-router
cat state/strategy_router_live.json.heartbeat
tail -n 120 live_strategy_router.log | grep -E 'bootstrap|evaluate|Traceback|TypeError|错误|异常'
```

Expected healthy bootstrap:

```json
{
  "status": "ok",
  "qqq": {
    "market_loaded": true,
    "bootstrap_step": "completed",
    "error": null
  }
}
```

