# QQQ/USDT Aggressive Frozen

当前 `QQQ/USDT` 合约研究的第一版激进 frozen 候选如下：

- 信号来源：`QQQ` 日线 frozen 主线  
  `config/config.paper.tqqq-only-strict-recovery-frozen.json`
- 执行标的：`QQQ/USDT:USDT`
- 执行周期：`4h`
- `1h` 入场优化：已审，当前样本无额外增益，不纳入 frozen
- 杠杆结构：`fixed10`
- 止损：`3.5%`
- 止盈：`none`
- 配置成本口径：
  - `taker_fee_rate = 0.0005`
  - `slippage_bps = 5.0`
  - `funding`: 使用 `QQQ_USDT_USDT-8h-funding_rate.feather`
- 实盘费率审计口径：
  - 单边交易费率 `0.02%`
  - 未额外计入滑点

当前 live-like replay 结果：

- QQQ/USDT-only：`124.82% / DD 29.92%`
- BTC frozen + QQQ/USDT router：`202.01% / DD 26.02%`
- funding 成本估计：`17.31%`
- 固定持仓杠杆：`10.00x`
- 部分调仓成本：`0.00%`

真实 notional 滚仓专项：

- 不滚仓基线：`96.59% / DD 19.68%`
- 收益最高滚仓：`139.08% / DD 33.78%`
- 推荐上线滚仓：`125.86% / DD 25.01%`
- 推荐规则：实际杠杆低于 `9.5x`、持仓盈利、单笔最多滚仓 `4` 次、冷却 `1` 根 4h、`defense` 状态不滚仓

重要边界：

- `QQQ/USDT` 可用历史非常短
  - `1d`: `2026-03-04 -> 2026-05-29`
  - `4h`: `2026-04-09 -> 2026-05-29`
  - `1h`: `2026-05-16 -> 2026-05-29`
- 因此该 frozen 只能视为短样本激进候选，不能等同于长期稳定结论。

本轮研究结论：

- 可迁移的 BTC frozen 核心逻辑是趋势信号和路由仲裁，不再使用 QQQ 侧动态降杠杆作为主线。
- 当前主候选固定为 `fixed10`
- `defense` 分段、确认后降杠杆、offense 0-1 连续化均已扫过；在 live-like 口径下，`def1` 因频繁调仓和错过风险上调窗口明显弱于固定 `10x`
- 盈利滚仓只作为补仓到目标 notional 的 live 执行层 overlay，不改变初始 `3.5%` stop，也不允许 stop 下移
- 当前样本里不值得加入：
  - `failed breakout guard`
  - `touch lock`
  - `fixed take profit`
  - `funding-aware leverage gate`
  - `1h` 额外入场触发
