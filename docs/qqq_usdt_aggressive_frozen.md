# QQQ/USDT Aggressive Frozen

当前 `QQQ/USDT` 合约 frozen 主入口按东京服务器实盘版本对齐：`fixed10` 基线 + 两层风险 overlay。

- 信号来源：`QQQ` 日线 frozen 主线  
  `config/config.paper.tqqq-only-strict-recovery-frozen.json`
- 执行标的：`QQQ/USDT:USDT`
- 执行周期：`4h`
- `1h` 入场优化：已审，当前样本无额外增益，不纳入 frozen
- 杠杆结构：`fixed10`
  - 常规：`10x`
  - 高增长：`10x`
  - 防御：`10x`
  - OKX 仓位杠杆参数：固定 `10x`
  - 风控 cap：只改变目标曝光 notional，不把 OKX 仓位杠杆参数改成 `7.5x/5x/2.5x`
  - 卖出执行：cash/trailing 完全平仓与 cap 降曝光都走 `reduceOnly`，默认按每笔最多 `10` 张分批
- 止损：`4.0%`
- 止盈：`none`
- 两层风控 overlay：已接入 `QqqUsdtSignalAdapter.preview()`
  - 近期战术层：`var/reports/qqq_drawdown_lgb_shadow_predictions_macro_subfactor_core.csv`
  - 近期战术规则：上一条可用 `raw_prob_10d >= 0.50` 时 QQQ/USDT 候选直接转现金
  - 长周期层：`var/reports/qqq_long_cycle_correction20d10_qqqonly_lgb_predictions.csv`
  - 长周期规则：`raw_prob_10d >= 0.35/0.50/0.65` 时分别把杠杆乘以 `0.75/0.50/0.25`
  - 风控文件使用上一条已完成日线信号，避免同日收盘信号前视
  - `risk_overlay_fail_open=false`，风控文件缺失、字段缺失或过期时 QQQ/USDT 主入口硬失败，不静默忽略
- 配置成本口径：
  - `taker_fee_rate = 0.0005`
  - `slippage_bps = 5.0`
  - `funding`: 使用 `QQQ_USDT_USDT-8h-funding_rate.feather`
- 实盘费率审计口径：
  - 单边交易费率 `0.02%`
  - 未额外计入滑点

2026-05-30 trailing/shadow gate 审计冻结与 runtime 接入：

- Runtime 生效参数：`stop_loss_pct = 4.0`
- Runtime shadow gate profile：
  - `reentry_rule = clear`
  - `reentry_clear_bars = 2`
  - `loss_streak_stop = 2`
  - `loss_streak_cooldown_bars = 20`
  - `equity_dd_stop_pct = 25.0`
  - `equity_dd_cooldown_bars = 10`
- Runtime 接入点：
  - QQQ executor 开仓前检查 shadow gate，阻断时不下单
  - QQQ executor 平仓后更新 loss streak / equity drawdown / clear-bars 状态
  - Router 切到 QQQ 前先检查 shadow gate，阻断时保持原执行策略，避免先 flatten BTC
  - 轨迹写入 `StateStore.action_log`，`action_type = QQQ_SHADOW_GATE`
- NQ proxy closed-only 第二层结果：`960368.95% / DD 49.05% / CVaR5 -14.7966%`
- Full 结果：`984833.54% / DD 49.05% / CVaR5 -14.7966%`
- Real OKX overlap closed-only：`243.95% / DD 23.68%`，该窗口 `0` 次 stop hit，不能区分 `4.0` 与 `4.125`
- 报告：
  - `var/reports/qqq_usdt_shadow_second_layer_stop_clear_equity_scan_20220101_20260529.json`
  - `var/reports/qqq_usdt_shadow_second_layer_stop_clear_equity_scan_closed_20220101_20260528.json`
  - `var/reports/qqq_usdt_shadow_second_layer_stop_clear_equity_scan_real_overlap_closed_20260304_20260528.json`

配置级 router replay 验证已迁移到最终 router frozen 文档：

- `docs/frozen_strategy_router_20260531.md`
- 旧的 `proxy_strategy_router_replay_config_frozen_stop4_*` 产物已删除，避免与修正后的 router summary 口径混用。

运行边界：

- 当前 runtime QQQ 执行实际读取 `stop_loss_pct`、两层 risk overlay 字段与 `shadow_gate_replay_profile.runtime_enabled=true` 的 shadow gate 参数。
- Shadow gate 状态持久化在 QQQ state DB 的 `qqq_shadow_gate_state` key；动作轨迹写入同一 DB 的 `action_log`。
- Shadow gate 只阻断新的 QQQ 开仓/切换，不会阻止已有 QQQ 仓位因 signal off、router switch 或 stop hit 被平仓。

长周期 QQQ 日线 proxy 结果：

- 样本：`2010-09-15 -> 2026-05-29`
- 成本：单边 `0.02%`，未计 funding
- 主候选：`13892.83% / DD 68.56% / 2026 52.45%`
- 去最大盈利单后：`4426.97% / DD 68.56%`
- 平均实际杠杆：`4.469x`

重要边界：

- `QQQ/USDT` 可用历史非常短
  - `1d`: `2026-03-04 -> 2026-05-29`
  - `4h`: `2026-04-09 -> 2026-05-29`
  - `1h`: `2026-05-16 -> 2026-05-29`
- 因此该 frozen 只能视为短样本激进候选，不能等同于长期稳定结论。
- 两个风险 CSV 目前是离线生成产物；如果没有纳入每日生成流程，5 个自然日 stale guard 会阻断 QQQ/USDT 候选。

本轮研究结论：

- 本主链路只保留东京服务器已部署的 `fixed10` 基线；context-sizing 与 longproxy 旁路不再作为 frozen 主入口
- `VIX` 不直接接入止损；当前风险控制由近期战术现金门和长周期杠杆 cap 承担
- trailing stop 从 `3.5%` 提升到 `4.0%`，来自 2026-05-30 `3.75 -> 4.25` 细扫与第二层联合审计
- 重要风险：长周期 proxy 未计 QQQ/USDT 永续 funding，实盘前必须继续做 funding guard
