# Crypto Trading Project

## ⚠️ 硬规则：回测前视口径

**回测、复现、参数扫描一律不得使用「当时尚未收盘」的 4h / 日线信息。**
唯一合法入口是 `strategy.scalp_robust_v2_core.build_precomputed_state_confirmed_4h`。

> **动手改任何多周期代码前，先读 `docs/backtest_lookahead_policy.md`（唯一权威说明）。**
> 违反此规则的实测代价：2022 至今回测总收益从 +396% 虚增到 **+8525%**，回撤被低估 11pp。
> 修改后必须重跑 `python3 scripts/audit_backtest_lookahead.py --klines-15m <15m数据>`。

## 文档

关于 BTC scalp 策略的研究结论与外部证据见 `docs/btc_scalp_research.md`。
其余为既有的 frozen 策略文档：

1. `docs/frozen_strategy_router_20260531.md`
2. `docs/qqq_usdt_aggressive_frozen.md`
3. `docs/tokyo_git_deploy.md`

Canonical command wrappers live under `scripts/workflows/`.

## Repo Layout

- `bot/`: live execution engine and exchange/runtime state handling.
- `strategy/`: strategy engine and signal/trailing logic.
- `scripts/`: shared replay, audit, and research modules.
- `scripts/workflows/`: stable entrypoints grouped by workflow.
- `systemd/`: Tokyo service definitions.
- `docs/`: frozen strategy/router docs + 前视口径政策 + BTC scalp 研究。

## Current Rules

- **回测口径**：唯一入口 `build_precomputed_state_confirmed_4h`，不得新增口径开关（见 `docs/backtest_lookahead_policy.md`）。
- Tokyo deployments go through git: commit, push, remote `fetch + pull --ff-only`, then restart and verify.
- Use `scripts/replay_sota_smc_live_shadow.py` as the canonical strategy replay entry.
- Use `scripts/audit_live_replay_convergence.py` for replay/live convergence work.
- Historical research notes are archived outside the tracked docs tree; keep tracked docs focused on current frozen runtime state.

## Quick Start

- Live/Tokyo: `bash scripts/workflows/live/deploy_tokyo.sh`
  The deploy wrapper auto-detects the active `crypto-strategy-router` systemd repo path, deploys via git, and does not overwrite the live router JSON by default.
- QQQ risk refresh/Tokyo: `bash scripts/workflows/live/install_tokyo_qqq_risk_refresh.sh`
  This installs the daily support-data refresh plus recent/long-cycle risk CSV regeneration timer without touching the router live config.
- Replay/Audit: run the relevant `scripts/replay_*` or `scripts/audit_*` entrypoint for the frozen config under review.
- Research/Optimization: use the dedicated `scripts/scan_*` or `scripts/audit_*` module for the candidate line.

Use the frozen docs for current expectations and report references.
