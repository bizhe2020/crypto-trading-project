# QQQ/USDT Aggressive Frozen

当前 `QQQ/USDT` 合约研究的第一版激进 frozen 候选如下：

- 信号来源：`QQQ` 日线 frozen 主线  
  `config/config.paper.tqqq-only-strict-recovery-frozen.json`
- 执行标的：`QQQ/USDT:USDT`
- 执行周期：`4h`
- `1h` 入场优化：已审，当前样本无额外增益，不纳入 frozen
- 杠杆结构：`base10_off10_def2`
- 止损：`3.5%`
- 止盈：`none`
- 成本口径：
  - `taker_fee_rate = 0.0005`
  - `slippage_bps = 5.0`
  - `funding`: 使用 `QQQ_USDT_USDT-8h-funding_rate.feather`

当前激进样本结果：

- `180.10% / DD 9.27%`
- funding 成本估计：`4.66%`
- 平均实际杠杆：`6.5x`

重要边界：

- `QQQ/USDT` 可用历史非常短
  - `1d`: `2026-03-04 -> 2026-05-29`
  - `4h`: `2026-04-09 -> 2026-05-29`
  - `1h`: `2026-05-16 -> 2026-05-29`
- 因此该 frozen 只能视为短样本激进候选，不能等同于长期稳定结论。

本轮研究结论：

- 可迁移的 BTC frozen 核心逻辑是 `dynamic leverage`
- 当前样本里不值得加入：
  - `failed breakout guard`
  - `touch lock`
  - `fixed take profit`
  - `funding-aware leverage gate`
  - `1h` 额外入场触发
