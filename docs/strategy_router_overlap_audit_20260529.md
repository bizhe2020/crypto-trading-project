# Strategy Router Overlap Audit 2026-05-29

## 目的

回答两个问题：

- 今天路由策略应该走 `BTC` 还是 `QQQ/USDT`
- 在 `BTC` 与 `QQQ/USDT` 可交叉回放的区间内，路由后收益是否优于单做某一边

## 今日路由结果

基于当前 paper 路由配置：

- 配置：`config/config.paper.strategy-router.json`
- 结果状态：`state/strategy_router_paper.json`

当天结果：

- `selected_strategy = qqq_usdt_aggressive`
- `selected_symbol = QQQ/USDT:USDT`
- `selected_route_score = 98.0`

候选状态：

- `BTC`：
  - `active = false`
  - `route_score = 0.0`
  - `reason = no_live_candidate`
- `QQQ/USDT`：
  - `active = true`
  - `route_score = 98.0`

结论：

- `2026-05-29` 这一天，当前路由应该走 `QQQ/USDT`，不是 `BTC`

## 交叉区间

公平对比区间使用 `QQQ/USDT 4h` 可用窗口：

- `2026-04-09 -> 2026-05-29`

## 三个口径必须分开

### 1. BTC 部署策略同期回测

文件：

- `var/reports/backtest_config.paper.high-leverage-structure_2026-04-09_to_2026-05-29.json`

结果：

- 收益：`+1.29%`
- 最大回撤：`20.15%`

### 2. QQQ/USDT 冻结主候选口径

文件：

- `docs/qqq_usdt_aggressive_frozen.md`
- `var/reports/qqq_usdt_leverage_state_scan_offcap10.json`
- `var/reports/qqq_usdt_defense_execution_policy_scan_fee002pct.json`

冻结参数：

- 日线信号：`TQQQ-only strict recovery frozen`
- 执行：`QQQ/USDT 4h`
- 杠杆：`base10_off10_def1`
- 止损：`3.5%`
- 固定止盈：`none`

结果：

- 主 replay：`+291.51% / DD 9.99%`
- 单边 `0.02%` 调仓成本口径：`+252.96% / DD 10.94%`

### 3. 错误桥接口径

文件：

- `var/reports/qqq_usdt_overlap_tmp.json`

这不是当前 frozen 激进口径，而是：

- 旧 frozen 日线信号
- 固定 `10x`
- `3.5% stop`

结果：

- 收益：`+104.59%`
- 最大回撤：`29.92%`

这个口径不能再拿来代表当前 `QQQ/USDT` frozen。

## 最终结论

### 今天该走哪边

- 该走 `QQQ/USDT`

### 路由后收益是否提升

如果比较：

- `BTC-only`：`+1.29%`
- `QQQ/USDT frozen-only`：`+291.51%`

那么当前交叉周期里，路由相对 `BTC-only` 的提升非常明显。

### 路由本身是否有独立 alpha

当前还不能证明。

原因：

- 这段区间内 `BTC` 候选长期不活跃
- 路由器基本会一直选 `QQQ/USDT`

所以目前能确认的是：

- 路由避免了留在明显更弱的 `BTC`
- 但还没有证明“动态切换”本身比“固定做 `QQQ/USDT frozen`”更强

## 后续建议

如果要正式证明路由 alpha，需要补一版历史路由回放：

- 每个时点同时重算 `BTC route_score`
- 每个时点同时重算 `QQQ route_score`
- 应用 `switch_advantage`
- 形成完整 `BTC-only / QQQ-only / router` 三路净值对比
