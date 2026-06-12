# QQQ/USDT Aggressive Frozen

当前 `QQQ/USDT` 合约 frozen 主入口按东京服务器实盘版本对齐：`fixed10` 基线 + 两层风险 overlay + 宏观 dollar cap overlay + shadow gate V2。

- 信号来源：`QQQ` 日线 frozen 主线  
  `config/config.paper.tqqq-only-strict-recovery-frozen.json`
- Frozen/replay 配置：`config/config.paper.qqq-usdt-aggressive-frozen.json`
- Live/runtime 配置：`config/config.paper.qqq-usdt-aggressive-runtime.json`
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
  - 实盘 runtime 近期战术层：`var/runtime/qqq_risk/qqq_recent_risk_runtime_predictions.csv`
  - Frozen/replay 近期战术层固定输入：`var/reports/qqq_drawdown_lgb_shadow_predictions_macro_subfactor_core.csv`
  - 近期战术规则：上一条可用 `raw_prob_10d >= 0.50` 时 QQQ/USDT 候选直接转现金
  - 实盘 runtime 长周期层：`var/runtime/qqq_risk/qqq_long_cycle_risk_runtime_predictions.csv`
  - Frozen/replay 长周期层固定输入：`var/reports/qqq_long_cycle_correction20d10_qqqonly_lgb_predictions.csv`
  - 长周期规则：`raw_prob_10d >= 0.35/0.50/0.65` 时分别把杠杆乘以 `0.75/0.50/0.25`
  - 风控文件使用上一条已完成日线信号，避免同日收盘信号前视
  - `risk_overlay_fail_open=false`，风控文件缺失、字段缺失或过期时 QQQ/USDT 主入口硬失败，不静默忽略
- 宏观 proxy overlay：已接入 `QqqUsdtSignalAdapter.preview()`、`scripts/replay_proxy_strategy_router.py`、`scripts/audit_qqq_shadow_gate_v2_combined.py`
  - 数据源：`data/public/macro/fred_macro-1d.feather`
  - 模式：`dollar_zscore_cap`
  - 规则：当 `macro_broad_dollar_index_z_252d >= 1.5` 时，把目标曝光 notional cap 到 `50%`
  - 对齐口径：先按 `daily_signal_timestamp/session_day` 生成日级上下文，再 backward merge 到 `4h` bar；不按自然日重复 `4h` bar 滚动 z-score
  - `macro_proxy_overlay_use_previous_signal=false`，沿用日级 available-as-of 宏观表的当日信号
  - `macro_proxy_overlay_fail_open=false`；文件缺失、字段缺失或 stale 时主入口硬失败，前期 z-score warmup 的 `NaN` 只是不触发 cap
- 配置成本口径：
  - `taker_fee_rate = 0.0005`
  - `slippage_bps = 5.0`
  - `funding`: 使用 `QQQ_USDT_USDT-8h-funding_rate.feather`
- 实盘费率审计口径：
  - 单边交易费率 `0.02%`
  - 未额外计入滑点

2026-06-02 shadow gate V2 审计冻结与 runtime 接入：

- Runtime 生效参数：`stop_loss_pct = 4.0`
- Runtime shadow gate profile：
  - `clock = signal_session`
  - `reentry_rule = clear`
  - `reentry_clear_bars = 2`
  - `loss_streak_stop = 0`
  - `loss_streak_cooldown_bars = 0`
  - `equity_dd_stop_pct = 15.0`
  - `equity_dd_cooldown_bars = 20`
- Runtime 接入点：
  - QQQ executor 开仓前检查 shadow gate，阻断时不下单
  - QQQ executor 平仓后更新 loss streak / equity drawdown / clear-bars 状态
  - Router 切到 QQQ 前先检查 shadow gate，阻断时保持原执行策略，避免先 flatten BTC
  - 轨迹写入 `StateStore.action_log`，`action_type = QQQ_SHADOW_GATE`
- Combined risk-overlay + shadow-gate audit 使用当前 frozen router 口径：`btc_min=35 / qqq_min=96 / switch=6 / takeover=6`
- Full NQ proxy：V2 `18505422.00% / DD 44.64% / CVaR5 -13.4575%`，当前 balanced `11005839.37% / DD 48.49% / CVaR5 -14.0675%`
- NQ closed-only：V2 `18045766.50% / DD 44.64% / CVaR5 -13.4575%`，当前 balanced `10732464.72% / DD 48.49% / CVaR5 -14.0675%`
- Real OKX overlap：V2 与当前 balanced 同为 `269.87% / DD 22.13% / CVaR5 -10.7878%`；该窗口 `0` 次 stop hit，只能做 live data sanity check，不能证明 shadow gate 的完整周期稳定性
- Rolling vs current balanced：
  - `126d`: DD 改善 `27.66%`，CVaR 改善 `51.06%`，Calmar/return 改善 `36.17%`
  - `252d`: DD 改善 `43.90%`，CVaR 改善 `75.61%`，Calmar/return 改善 `63.41%`
- 报告：
  - `var/reports/qqq_shadow_gate_v2_combined_audit_20220101_20260529.json`
  - `scripts/audit_qqq_shadow_gate_v2_combined.py`

2026-06-03 dollar `cap50 z1.5` 冻结与 shared live-chain audit/replay 接入：

- Frozen config：
  - `config/config.paper.qqq-usdt-aggressive-frozen.json`
  - live/runtime variant: `config/config.paper.qqq-usdt-aggressive-runtime.json`
  - `frozen_label = qqq_usdt_aggressive_fixed10_risk_overlay_stop4_shadow_v2_low_dd_dollar_cap50_z1_5_20260603`
- Combined audit 对比口径：candidate 为当前 frozen；baseline 为同配置临时关闭 macro overlay
- Full NQ router：
  - candidate：`26115047.82% / DD 41.74% / CVaR5 -12.2179%`
  - baseline：`18505422.00% / DD 44.64% / CVaR5 -13.4575%`
- NQ closed-only router：
  - candidate：`25466378.13% / DD 41.74% / CVaR5 -12.2179%`
  - baseline：`18045766.50% / DD 44.64% / CVaR5 -13.4575%`
- Real OKX overlap：
  - candidate 与 baseline 同为 `269.87% / DD 22.13% / CVaR5 -10.7878%`
  - shared router replay 同样不变：`269.50% / DD 22.13%`
  - 原因是当前 overlap 窗口 `macro_trigger_bars = 0 / macro_cap_bars = 0`
- Shared combined audit 的 QQQ 4h path 触发 footprint：
  - `macro_trigger_bars = 85`
  - `macro_cap_bars = 85`
  - `macro_cash_bars = 0`
  - `macro_exit_events = 0`
- 与此前 sidecar 研究对比：
  - topline router 指标已对齐到同一量级：`26115047.82% / DD 41.74% / CVaR5 -12.2179%`
  - signal-day 对齐后的 shared 路径 cap bar 计数为 `85`，比旧 sidecar 统计少 `1` 个 bar，但未造成 topline 漂移
- 报告：
  - `var/reports/qqq_shadow_gate_v2_combined_audit_dollar_cap50_z1_5_20260603.json`
  - `var/reports/qqq_shadow_gate_v2_combined_audit_shadow_v2_nomacro_20260603.json`
  - `var/reports/router_replay_qqq_usdt_dollar_cap50_z1_5_20260603.json`
  - `var/reports/router_replay_qqq_usdt_shadow_v2_nomacro_20260603.json`

配置级 router replay 验证已迁移到最终 router frozen 文档：

- `docs/frozen_strategy_router_20260531.md`
- 旧的 `proxy_strategy_router_replay_config_frozen_stop4_*` 产物已删除，避免与修正后的 router summary 口径混用。

运行边界：

- 当前 runtime QQQ 执行实际读取 `stop_loss_pct`、两层 risk overlay 字段、macro proxy overlay 字段与 `shadow_gate_replay_profile.runtime_enabled=true` 的 shadow gate V2 参数。
- router live 模板不再携带旧 `qqq_stop_reentry_*` 字段；V2 的 reentry / cooldown 只由 QQQ frozen 配置内的 `shadow_gate_replay_profile` 驱动。
- Shadow gate 状态持久化在 QQQ state DB 的 `qqq_shadow_gate_state` key；动作轨迹写入同一 DB 的 `action_log`。
- Shadow gate 只阻断新的 QQQ 开仓/切换，不会阻止已有 QQQ 仓位因 signal off、router switch 或 stop hit 被平仓。
- Shadow gate 的 live runtime clock 与 frozen NQ dailyproxy 扫参对齐为 `signal_session`：`equity_dd_cooldown_bars = 20` 表示约 20 个 QQQ/NQ 信号交易日 observation，不是 20 个自然日，也不是 20 根 4h execution bar。由于 frozen 扫参使用 `gate_until_idx = idx + bars` / `idx < gate_until_idx`，实际释放发生在第 20 个新 observation，通常表现为后续 19 个 session rows 被 gate active 覆盖。

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
- 风控 CSV 已按用途隔离：实盘刷新脚本默认只写 `var/runtime/qqq_risk/`；`var/reports/` 下两份白名单 CSV 是 frozen/replay 固定输入，不能被 runtime refresh 覆盖。
- 如果 runtime 风控 CSV 没有纳入每日生成流程，5 个自然日 stale guard 会阻断 QQQ/USDT 候选。

本轮研究结论：

- 本主链路只保留东京服务器已部署的 `fixed10` 基线；context-sizing 与 longproxy 旁路不再作为 frozen 主入口
- `VIX` 不直接接入止损；当前风险控制由近期战术现金门和长周期杠杆 cap 承担
- trailing stop 从 `3.5%` 提升到 `4.0%`，来自 2026-05-30 `3.75 -> 4.25` 细扫与第二层联合审计
- shadow gate 从 2026-05-30 balanced profile 切到 2026-06-02 V2 low-DD profile：禁用 loss-streak gate，使用 `15%` equity-DD 与 `20` bar cooldown
- 2026-06-03 frozen 再叠加 broad dollar `z >= 1.5 -> cap 50%` 的 signal-day 对齐 macro overlay；长样本明显改善，当前 real overlap 仍是 no-op
- 重要风险：长周期 proxy 未计 QQQ/USDT 永续 funding，实盘前必须继续做 funding guard
