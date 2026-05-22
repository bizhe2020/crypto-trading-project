# 国泰纳斯达克100ETF 研究计划

## 主线

用 `QQQ` 的美股夜盘信号，驱动国内 `513100.SS` 的次日持仓决策。

当前冻结口径：

- 入场：`QQQ 25/200 MA`
- regime：`IXIC up + VIX low/normal`
- 退出：`max_hold_days=90`、`trailing_lookback_days=5`、`trailing_drawdown_pct=8`
- hold mode：`hard_exit`
- 仓位：`vix_low = 2.0x`，`vix_normal + qqq_strong = 1.75x`
- 执行：`next_open`
- 成本：`10bps`

## 已验证

- `QQQ`、`SPY`、`^IXIC`、`^VIX`、`513100.SS` 日线数据已可用
- `paper` 与 `live-shadow` 已跑通
- 最新结果：
  - 总收益 `639.33%`
  - 最大回撤 `13.21%`
  - 2026 YTD `16.88%`

## 下一步

1. 做 `513100.SS` 自身的成交/滑点审计。
2. 做 `QQQ -> 513100.SS` 时区和交易日错位审计。
3. 研究条件式 `2x`，优先看 `vix_low` 阶段上杠杆。
4. 接 Telegram 日报与状态输出。
5. 再决定是否接真实账户。

## 入口

- Paper config: `config/config.paper.cn-nasdaq100-etf.json`
- Live-shadow config: `config/config.live-shadow.cn-nasdaq100-etf.json`
- Paper command: `python3 scripts/run_cn_nasdaq_etf_paper_plan.py`
- Live-shadow command: `python3 scripts/run_cn_nasdaq_etf_live_shadow.py`

## 旧候选对比

- 全程 `2x` 不是优先方向
- 当前最值钱的是 `vix_low` 阶段上 `2x`
- 初步结果：
  - `base_1x`: `226.68% / DD 6.28%`
  - `vix_low_2x`: `610.44% / DD 10.94%`
- 这条线值得继续做更细的持仓/风控扫描

更细分层结果：

- 旧 baseline：`vix_low 2.0x`
- 更均衡候选：`vix_low 1.5x`
  - `384.00% / DD 8.60%`
- 更保守候选：`qqq_strong 1.5x`
  - `298.48% / DD 6.89%`

## 二次入场审计

- 已测“退出后等更强趋势确认再入场”的 breakout re-entry
- 最优候选约 `458.44% / DD 12.07%`
- 明显弱于当前 baseline `vix_low 2.0x`
- 结论：不作为主线扩展，先不接入默认配置

## 分层仓位审计

- 已测“分层仓位细化”，不再增加 entry 复杂度
- 方向：保留 `vix_low 2.0x`，并给 `vix_normal + qqq_strong` 一档中等杠杆
- 当前最优候选：
  - `vix_low = 2.0x`
  - `vix_normal + qqq_strong = 1.75x`
  - `same_close` 研究结果：`702.50% / DD 10.94% / 2026 15.57%`
  - `next_open` 优化后 baseline：`639.33% / DD 13.21% / 2026 16.88%`
- 对比当前 baseline：
  - 旧 baseline：`610.44% / DD 10.94% / 2026 10.01%`
  - 增益主要来自把强趋势但非低波的那部分也放大，而不是放松退出
- 结论：这条线已晋升为新 baseline

## 成交/滑点审计

- 已对新 baseline 做日线执行敏感性审计
- 参考口径：`same_close_reference`
  - `702.50% / DD 10.94% / 2026 15.57%`
- 更接近真实的次日执行：
  - `next_open`: `571.66% / DD 15.29% / 2026 13.88%`
  - `next_mid`: `529.11% / DD 14.98% / 2026 10.71%`
  - `next_close`: `473.96% / DD 17.17% / 2026 8.01%`
- 结论：
  - 流动性本身不是主要问题，成交日名义成交额中位数约 `6.17e8`
  - 真正的 gap 来自执行时点，不是交易量不足
  - baseline 已正式迁到 `next_open`

## Next-Open 口径优化

- 已在 `next_open` 正式口径上重新扫 exit 与 tiered sizing
- 最值钱的两处调整：
  - trailing 从 `5%` 放宽到 `8%`
  - `vix_normal + qqq_strong` 从 `1.5x` 提高到 `1.75x`
- 正式结果提升到：
  - `639.33% / DD 13.21% / 2026 16.88%`

## 稳健性审计

- 已对当前 `next_open` baseline 做去单笔、年度分布、成本压力测试
- 交易统计：
  - `34` 笔完整交易
  - 胜率 `73.53%`
  - 中位单笔 `1.82%`
  - 最佳单笔 `52.49%`
  - 最差单笔 `-2.70%`
- 去 top trade 后：
  - 去掉最大单笔后仍有 `401.20%`
  - 去掉前两大单后仍有 `440.66%`
  - 说明有集中度，但不是单笔撑起全策略
- 年度分布：
  - `2023`: `106.96%`
  - `2024`: `133.92%`
  - `2025`: `33.29%`
  - `2026 YTD`: `16.88%`
  - 主要利润集中在 `2023-2024`，但 `2025-2026` 仍保持正收益
- 成本压力：
  - 额外 `+2.5bps`：`627.13%`
  - 额外 `+5bps`：`615.13%`
  - 额外 `+10bps`：`591.72%`
  - DD 基本不变，说明换手不高、对额外执行成本不算脆弱

## Kelly 启发

- 凯利只适合作为分层仓位上限，不适合全局满配
- 当前 bucket 结果：
  - `vix_low_other`: Kelly `0.7669`
  - `vix_normal_strong`: Kelly `0.5893`
  - overall Kelly `0.7029`
- 实盘建议：
  - 使用 `0.25~0.5 Kelly` 作为 cap
  - `vix_low_other` 可给更高上限
  - `vix_normal + qqq_strong` 保持保守一些
