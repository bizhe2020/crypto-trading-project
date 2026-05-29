# Strategy Router Plan

当前分支已新增一层最小可运行的跨策略路由：

- `BTC/USDT:USDT`：
  - 配置来源：`config/config.paper.high-leverage-structure.json`
  - 执行底座：`new_strategy_research` / `high_leverage_10x_research` 的 BTC live executor 迁入当前分支
- `QQQ/USDT:USDT`：
  - 配置来源：`config/config.paper.qqq-usdt-aggressive-frozen.json`
  - 信号来源：`TQQQ-only strict recovery frozen`
  - 合约执行语义：已接入最小 long-only OKX 执行器，由 router 统一开平仓

当前统一入口：

- `bot/run_strategy_router.py`
- `config/config.paper.strategy-router.json`
- `config/config.live.strategy-router.template.json`
- `systemd/crypto-strategy-router.service`

## 当前路由规则

- BTC 候选：
  - 优先读取 executor state snapshot
  - 若当前无持仓、且最近没有 live candidate，则视为 inactive
  - route score 已改为分层质量评分：
    - 基础信号类型：`sota_long / smc_short / gap_smc_short_expansion / smc_long`
    - 仓位强度：`execution_effective_leverage / requested_effective_leverage / source_effective_leverage / leverage`
    - SOTA 质量桶：`net_score / bull_total / bear_total / conflict`
    - 上下文加分：`risk_regime / regime_label / recent_fvg_near_entry / recent_sweep_status`
    - 风险扣分：`bearish_structure` 对多头扣分，空头方向轻微扣分
  - live adapter、普通 engine trade replay、frozen live-shadow event replay 共用 `bot/btc_route_scoring.py`
- QQQ 候选：
  - 读取 frozen 信号桥接后的 `QQQ/USDT 4h` 最新状态
  - 按 `high_growth / defense / base / rel_strength / recovery` 计算 route score

路由器会输出：

- 当前候选列表
- 每个候选的 `route_score`
- 最终选中的 `selected_strategy`
- 切换原因和滞回判断

## 执行编排

`--execute` 模式会执行以下状态机：

- `selected_strategy changed`：先 flatten 旧策略，再执行新策略
- `btc_sota`：调用既有 BTC SOTA live executor
- `qqq_usdt_aggressive`：调用 `QqqUsdtExecutionEngine` 开 QQQ/USDT 多单
- `no_signal`：默认 flatten 当前执行策略
- QQQ/USDT 使用 `var/okx/markets_cache.json` 缓存 OKX 合约规格，缓存从日本服务器导出

QQQ/USDT 风控口径：

- live 模板默认打开初始交易所止损：`qqq_enable_exchange_stop=true`
- 本地仍跟踪 trailing stop，触发后市价平仓
- 同一根 QQQ 执行 bar stop-hit 后不会立刻按同一条 active 信号重开

## 当前限制

这版已从 `strategy selection layer` 推进到最小 `execution orchestration layer`，但还有三点边界：

- QQQ/USDT 仍是 long-only，没有 SQQQ/short 腿
- QQQ/USDT 的 trailing stop 上移当前由本地状态跟踪，交易所侧只保证初始止损
- BTC flatten 使用 reduce-only 市价单清掉交易所仓位，BTC 本地策略状态依赖既有 manual sync 在后续轮询收敛

## 下一步建议

如果继续往真实部署推进，优先级应是：

1. 用 `config/config.paper.strategy-router.json --execute --evaluate-once` 做本地 paper smoke test
2. 在日本服务器生成 `config/config.live.strategy-router.json`
3. 先停旧 BTC 单策略服务，再启动 `crypto-strategy-router.service`
4. 观察首轮 router 日志确认只持有一个 symbol

## 2026-05-29 审计结论

最新交叉区间审计见：

- `docs/strategy_router_overlap_audit_20260529.md`

当前明确结论：

- 今日路由应选 `QQQ/USDT`
- 当前交叉区间 `2026-04-09 -> 2026-05-29` 中：
  - `BTC-only` 同期仅 `+1.29% / DD 20.15%`
  - `QQQ/USDT` 当前 frozen 主候选为 `base10/off10/def1`

需要注意的口径边界：

- `104.59% / DD 29.92%` 是错误桥接口径，不代表当前 frozen
- 当前 frozen 的正确口径是：
  - `base10_off10_def1`
  - `stop_loss_pct = 3.5%`
  - `no fixed take profit`
  - 主 replay：`+291.51% / DD 9.99%`
  - 单边 `0.02%` 调仓成本口径：`+252.96% / DD 10.94%`

当前还不能证明的是：

- `router` 是否比固定做 `QQQ/USDT frozen` 更强

当前能证明的是：

- `router` 明显优于留在弱得多的 `BTC`

## BTC 分层评分回放

2026-05-29 已把 BTC 分层评分接到 live adapter 和 proxy router replay：

- 交叉窗口：`2026-03-04 -> 2026-05-29`
- BTC source：`var/high_leverage_expansion/frozen_live_core_20260515.json`
- QQQ source：`QQQ/USDT leveraged base10/off10/def1`
- Router：`+454.32% / DD 15.42%`
- QQQ-only：`+301.04% / DD 6.78%`
- BTC-only：`+89.79% / DD 15.25%`
- 选择分布：BTC `29` 天，QQQ `28` 天，Cash `30` 天，切换 `18` 次

对应输出：

- `var/reports/proxy_strategy_router_replay_research_frozen_btc_qqq_usdt_leveraged_def1_btc_layered_score_20260529.json`
- `var/reports/proxy_strategy_router_replay_research_frozen_btc_qqq_usdt_leveraged_def1_btc_layered_score_20260529.md`

当前 paper 预览：

- 选中：`QQQ/USDT`
- route score：`96.0`
- BTC 状态：`no_live_candidate`
- 含义：今天不是 BTC 评分输给 QQQ，而是 BTC 侧当前没有可执行候选。

## OKX 元数据

本地缓存来自日本服务器：

- `BTC/USDT:USDT`：`contractSize=0.01`，最大杠杆 `100x`
- `QQQ/USDT:USDT`：`contractSize=1.0`，最大杠杆 `20x`

刷新命令：

```bash
python3 scripts/export_okx_markets_cache.py --output var/okx/markets_cache.json
```
