# Frozen Strategy Router 2026-05-29

## Scope

当前 frozen 是 `BTC SOTA` 与 `QQQ/USDT aggressive` 的单仓路由版本。

入口：

- `bot/run_strategy_router.py`
- `config/config.paper.strategy-router.json`
- `config/config.live.strategy-router.template.json`

路由规则：

- BTC 最低分：`35`
- QQQ 最低分：`60`
- 切换滞回：`8`
- 单仓：切换时先 flatten 旧策略，再执行新策略
- 无信号：允许 flatten

## BTC Leg

配置：

- `config/config.paper.high-leverage-structure.json`
- live template：`config/config.live.high-leverage-structure.template.json`

核心：

- `SOTA long + SMC short + gap-SMC short`
- `single-position arbitration`
- `SOTA score gate`
- `structure gate`
- `long score bucket sizing`
- `shadow risk gate`
- `confirmed_4h_only=true`

BTC route score 已统一到：

- `bot/btc_route_scoring.py`

评分输入：

- 信号类型：`sota_long / smc_short / gap_smc_short_expansion / smc_long`
- 杠杆强度：`execution_effective_leverage / requested_effective_leverage / source_effective_leverage / leverage`
- SOTA 质量：`net_score / bull_total / bear_total / conflict`
- 上下文：`risk_regime / regime_label / recent_fvg_near_entry / recent_sweep_status`
- 风险扣分：`bearish_structure`

## QQQ Leg

配置：

- `config/config.paper.qqq-usdt-aggressive-frozen.json`

核心参数：

- 标的：`QQQ/USDT:USDT`
- 信号源：`config/config.paper.tqqq-only-strict-recovery-frozen.json`
- 执行周期：`4h`
- 杠杆：`base10/offense10/defense1`
- 止损：`3.5%`
- 止盈：`none`
- data refresh：启用
- daily signal refresh：启用
- stale guard：启用，最大 `5` 个自然日

QQQ route score：

- 基础分 `40`
- 杠杆 `*4`
- `high_growth +12`
- `defense_state -8`
- `qqq_strong +8`
- `base entry +4`
- `recovery_reentry +10`
- `breakout_12 +4`
- `vix_low +4`

## Replay Snapshot

正式交叉窗口：

- 区间：`2026-03-04 -> 2026-05-29`
- BTC source：`var/high_leverage_expansion/frozen_live_core_20260515.json`
- QQQ source：`QQQ/USDT leveraged base10/off10/def1`

结果：

- Router：`454.32% / DD 15.42%`
- QQQ-only：`301.04% / DD 6.78%`
- BTC-only：`89.79% / DD 15.25%`
- 选择：BTC `29` 天，QQQ `28` 天，Cash `30` 天
- 切换：`18` 次

输出：

- `var/reports/proxy_strategy_router_replay_research_frozen_btc_qqq_usdt_leveraged_def1_btc_layered_score_20260529.json`
- `var/reports/proxy_strategy_router_replay_research_frozen_btc_qqq_usdt_leveraged_def1_btc_layered_score_20260529.md`

## Current Paper Preview

2026-05-29 paper preview：

- 选中：`QQQ/USDT`
- route score：`96.0`
- BTC：`no_live_candidate`

解释：

- 当前不是 BTC 分数低输给 QQQ，而是 BTC 没有可执行候选。

## Audit Log

Router 执行层已启用 append-only JSONL 审计日志：

- paper：`var/log/strategy_router_paper_audit.jsonl`
- live：`var/log/strategy_router_live_audit.jsonl`

每轮 `evaluate_latest` 追加一行，核心字段包括：

- `route`：完整 candidates、selected strategy、route score、decision reason
- `execution_results`：本轮实际开仓、平仓、调仓、stop 更新结果
- `local_state.router_execution`：router 当前执行策略和 last status
- `local_state.btc_snapshot`：BTC executor 本地 snapshot
- `local_state.qqq_state`：QQQ/USDT 本地 position / stop / leverage state
- `exchange_state.btc`：BTC long/short 交易所仓位快照
- `exchange_state.qqq`：QQQ/USDT 交易所仓位快照

审计日志只追加，不参与交易决策；采集失败会写入 `audit_errors`，不会阻断下单。

## Verification

已通过：

```bash
python3 -m py_compile bot/btc_route_scoring.py bot/btc_signal_adapter.py scripts/replay_proxy_strategy_router.py tests/test_strategy_router.py
python3 -m pytest tests/test_strategy_router.py -q
```

结果：

- `15 passed`

完整 `pytest -q` 在临时 `new_strategy_research` worktree 中有 4 个既有 Tokyo 审计测试失败，原因是本地未提交的 `var/tokyo_audit/config.live.high-leverage-structure.json` 缺失，不是本次 router frozen 改动导致。

`bot/run_strategy_router.py --config config/config.paper.strategy-router.json --evaluate-once` 在临时 worktree 中会因为缺少本地 `data/okx/futures/QQQ_USDT_USDT-4h-futures.feather` 失败。该数据文件属于本地行情数据，不随 git 提交。

## Deployment Boundary

本次 frozen 只提交 template 和 paper 配置，不提交真实 live 配置、交易所密钥、Telegram token、本地 state、行情数据或邮箱配置。
