# BTC Scalp 深度研究报告：历史交易诊断 + 外部策略证据

**日期**：2026-09-20  
**对象**：东京服务器 `btc-scalp-standalone.service`（`config.live.high-leverage-structure.json`，BTC/USDT:USDT 15m，SMC/ICT 体系）  
**样本**：2026-04-29 → 2026-09-19，机器人记录 21 笔已平仓交易，全部与 OKX 权威持仓 1:1 对齐  
**配套文件**：外部证据调研要点见本文档附录

---

## 0. 核心结论

**这套策略的入场信号在统计上与随机入场无法区分（经验 p = 0.327）；止盈目标在几何上不可达（要求 6.57% 价格移动，实测典型有利波动只有 1.33%）；而把账户打穿的直接原因是 `bull_strong_long_risk_per_trade = 0.12`（每笔 12% 风险）在最差的一段行情里被放大到最大。此外回测默认口径存在 4h 前视偏差（约 11% 的 K 线方向判断与实盘不一致）。**

五条互相独立的证据链：

| # | 证据 | 来源 | 结论 |
|---|---|---|---|
| 1 | 真实入场 vs 随机对照组，**p = 0.327** | 本项目实测（21 笔 × 300 轮随机对照） | 入场无优势 |
| 2 | SMC 形态中位 MFE 仅 **+0.9R~1.1R**，期望为负 | 外部（Botsfolio / Northmark 995,550 笔预注册检验） | 形态本身无 edge |
| 3 | 实测 MFE 均值 **1.22R**，配置目标 **5.5R**，**0/21 触及** | 本项目实测 | 目标在分布之外 |
| 4 | 止损中位 **4.30×ATR**（80% 超过 3.5×），目标 **21.6×ATR** | 本项目实测（第二轮） | 宽止损把目标推远 5 倍 |
| 5 | 回测默认 `confirmed_4h_only=False`，4h bias 分歧率 **11.33%** | 本项目实测（第二轮） | 回测系统性乐观 |

### 已闭合的因果链

```
止损 = 订单块下沿 − 缓冲（结构决定，非 ATR 决定）
   → 止损中位 1.26% 价格 = 4.30×ATR（宽）
   → 目标 = 5.5R × 止损 = 6.57% 价格 = 21.6×ATR（极远）
   → BTC 15m 典型有利波动仅 1.33%（1.06R）
   → 目标要求是典型波动的 5.0 倍
   → 0/21 触及止盈，全部由交易所止损单收场
```

**关键洞察**：止损宽度与目标 R 倍数是**相乘**关系。用结构止损（宽）配 5.5R 目标，等于要求一次 21.6×ATR 的趋势 —— 这在 15m 上不存在。若把止损收窄到外部实证推荐的 2.0–3.5×ATR 高原，同样的 5.5R 目标只需约 3.6% 移动，2R 目标则约 1.3%（正好等于典型波动）。

---

# 第一部分：历史交易深度研究

## 1. 数据与方法（全部只读）

| 数据源 | 内容 |
|---|---|
| `state/runtime_high_leverage_structure_live.db` → `action_log` | 31,004 条事件，含每笔开/平仓完整元数据（止损、目标、风险金额、regime、ADX、FVG 特征…） |
| OKX `/account/positions-history` | 100 笔已平仓持仓权威记录（含手续费、资金费） |
| OKX `/trade/fills-history` | 512 笔成交明细 |
| BTC-USDT-SWAP 15m K 线 | 13,693 根（2026-05-01 → 2026-09-20），用于逐笔 MFE/MAE |
| 当前账户权益 | 8,338.38 USD |

**对账口径**：机器人记录的平仓时间是「信号 K 线时间」，OKX `uTime` 是实际成交时间，实测最多差约 10 分钟（一根 15m K），故采用 20 分钟容差 + 1:1 贪心匹配。**21/21 全部对齐，无重复计入。**

分析脚本（可复现）：

- `scripts/audit_btc_scalp_live_history.py` —— 交易重建、成本拆解、MFE/MAE、滑点、执行健康度
- `scripts/audit_btc_scalp_exit_ablation.py` —— 固定入场、替换出场规则与风险预算的反事实消融
- `scripts/audit_btc_scalp_entry_edge.py` —— 真实入场 vs 随机对照组

## 2. 止盈从未被触及：0 / 21

| 出场原因 | 笔数 |
|---|---:|
| `external_stop_loss`（交易所侧止损单成交） | 20 |
| `external_flat_sync`（外部平仓同步） | 1 |
| **打到止盈目标** | **0** |

配置的目标盈亏比中位 **5.5R**，而实测历史**最佳** MFE 仅 **3.50R**。

### MFE 分布

| 曾达到 | 笔数 | 占比 |
|---|---:|---:|
| ≥ 1.0 R | 11 | 55% |
| ≥ 1.5 R | 6 | 30% |
| ≥ 2.0 R | 4 | 20% |
| ≥ 3.0 R | 2 | 10% |
| **≥ 4.0 R** | **0** | **0%** |
| ≥ 5.5 R | 0 | 0% |

- 平均 MFE **1.22R**，中位 **1.06R**
- 平均实际实现 **−0.02R**
- **MFE 回吐合计 24.71R**

> 这与外部数据高度一致：BTC 4H 的 Order Block / FVG / Liquidity Sweep 中位 MFE 分别为 **+0.91R / +0.94R / +1.01R**。**「形态能给出 4R+ 空间」这个前提本身不成立**，所以问题不是「止损太紧」，而是**目标设在分布之外**。

## 3. 成本结构：手续费吃掉 42.5% 的亏损

| 项目 | 金额 (USDT) | 占总亏损比 |
|---|---:|---:|
| 毛盈亏（不含费用） | −2,161.51 | 55.1% |
| 手续费 | −1,668.95 | 42.5% |
| 资金费 | −94.41 | 2.4% |
| **净已实现盈亏** | **−3,924.87** | 100.0% |

折算成 R：**手续费拖累平均 0.116R / 笔（21 笔合计 2.44R）**。这是判断任何改进「够不够大」的标尺 —— **毛收益改善小于 0.1R/笔的方案，扣费后等于零。**

机器人子集（21 笔）净已实现 **−4,443.95 USDT**，比账户整体更差（账户里其他手动交易反而是赚的）。

## 4. 风险预算失控：最大单一亏损来源

| risk_per_trade | 笔数 | 胜率 | 平均实现 R | 净盈亏 (USDT) |
|---|---:|---:|---:|---:|
| 1.5% | 1 | 100% | 0.30 | +28.44 |
| 2.5% | 4 | 25% | −0.45 | −1,799.91 |
| 5% | 1 | 0% | 0.10 | −31.57 |
| 6% | 7 | 57% | 0.76 | **+4,167.14** |
| 12% | 8 | 12% | −0.44 | **−6,582.18** |

（胜率按「净盈亏 > 0」统计）

`bull_strong_long_risk_per_trade = 0.12` 于 2026-08-22 生效，随后 8 笔里 7 笔亏，每笔约 **−12% 净值**。`risk_per_trade` 与实现 R 的 Spearman 相关 **−0.333** —— 风险开得越大结果越差。

**风险预算在策略最差的一段行情里被放大到最大。**

### 风控闸门为什么没拦住

配置里 `enable_shadow_risk_gate = true`、`shadow_consecutive_loss_stop = 4`、`shadow_equity_drawdown_stop_pct = 12`，但实际连亏 8 笔仍在交易。原因是**暂停语义太弱**（代码位置 `strategy/scalp_robust_v2_core.py:1342 _shadow_gate_after_close`）：

- 连亏 4 笔 → 只暂停到**下一个 UTC 日**，而交易间隔本来就有数天，等于没停
- 单日亏 6% → 同样只暂停到下一 UTC 日
- 权益回撤 12% → 只冷却 **2 天**，且触发后 `drawdown_peak` 被重置，于是形成「亏 12% → 停 2 天 → 再亏 12%」的阶梯式放血，**恰好解释了 8/23、8/25、9/1、9/4、9/7、9/9 每 2~3 天一笔亏损的节奏**

## 5. 其他量化发现

| 维度 | 发现 |
|---|---|
| 入场 ADX > 30 | 8 笔，**+2.51R**，胜率 50% |
| 入场 ADX < 20 | 6 笔，**−2.04R**，胜率 33% |
| regime = high_growth | 8 笔，+2.09R |
| regime = normal | 10 笔，−0.51R |
| regime = flat | 2 笔，−1.97R，胜率 0% |
| trail_style = normal | 4 笔，−2.01R（三档最差） |
| MAE < 0.6R | 9 笔，**+9.91R**，胜率 89% |
| MAE > 1.0R | 10 笔，−10.17R，胜率 0% |
| MFE < 0.5R | 6 笔，−6.20R，胜率 0% |
| 持仓 < 24 bar | 5 笔，−3.92R，胜率 20% |

**滑点不是问题**：平均入场滑点 −1.5bp（实际成交略优于假设价），中位 0.0bp。

---

# 第二部分：能否提升？

## 6. 出场规则反事实消融（固定 21 个真实入场）

（保守路径假设：同根 K 线同时触及止损与止盈时判定止损先成交）

| 出场规则 | 总R | 扣费后总R | 胜率 | 等风险复利终值* | 最大回撤 |
|---|---:|---:|---:|---:|---:|
| **实际历史（基准）** | −0.39 | **−2.44** | 38% | 9,010 | **23.8%** |
| 固定 1R 止盈 | +1.11 | −0.97 | 57% | 9,560 | 18.2% |
| 固定 2R 止盈 | −2.23 | −4.31 | 38% | 8,465 | 27.2% |
| **固定 5.5R 止盈（当前配置）** | **−4.39** | **−6.48** | 33% | **7,831** | **29.5%** |
| 1R 启动 0.5R 追踪 | +2.81 | +0.73 | 57% | 10,098 | 18.6% |
| 停滞离场：12bar 未达 0.5R 则走 | −0.10 | −2.18 | 38% | 9,160 | 18.7% |
| **停滞离场 + 1R 启动 0.5R 追踪** | +1.70 | **−0.38** | 43% | **9,766** | **9.9%** |

\* 起始 10,000，每笔固定 3.5% 风险复利，**已扣手续费**

**读法**：

1. **当前配置（5.5R 固定止盈）是所有方案里最差的之一** —— 目标不可达，还占用了出场逻辑
2. 最佳组合能把等风险最大回撤从 **23.8% 压到 9.9%**，扣费后总 R 从 **−2.44 提到 −0.38**
3. **但没有任何一条出场规则能让策略转正** —— 因为入场毛边际≈0，而成本是 0.116R/笔

**出场优化是「减亏」工具，不是「盈利」工具。**

## 7. 入场边际检验：与随机对照组无法区分

方法：对每个真实入场，在同一时间区间随机抽取「假入场」（随机时间 + 随机方向 + 从真实分布重采样止损宽度），用**完全相同的出场规则**结算，重复 300 轮。

| 指标 | 真实入场 | 随机对照（均值） | 随机对照（5%~95%） |
|---|---:|---:|---:|
| 平均 R | **0.134** | 0.029 | −0.340 ~ 0.454 |
| 胜率 | **57.1%** | 49.9% | 33.3% ~ 66.7% |

**经验 p 值 = 0.327**

真实入场的表现**完全落在随机分布中部**。结合 0.116R/笔的手续费：`+0.134R（毛） − 0.116R（费） ≈ +0.02R` —— 约等于零。

> 样本量仅 21 笔，检验功效有限。p = 0.327 应读作「**没有证据表明入场有优势**」，而非「证明没有优势」。但方向与外部大样本证据完全一致。

## 8. 外部证据（要点见本文档附录）

| 证据 | 规模 | 结论 |
|---|---|---|
| Northmark / FXBrokerCompass (2026) | **995,550 笔**，328 配置，预注册 + 随机对照 | **0/328 通过**；总 −2,119,412 点，平均 −2.13 点/笔；**随机对照组胜率与 SMC 形态基本相同甚至更高**（FVG 28.8% vs 随机 31.9%） |
| Botsfolio 多品种回测 | BTC 1H/4H/1D，N=137~927 | OB / FVG / Sweep 期望 **−0.15R ~ −0.25R**；中位 MFE **+0.91R ~ +1.01R** |
| Kitron & Wengrowicz (2026, arXiv:2608.21888) | 183 个 Binance 交易对 | 15 分钟尺度 **90% 存在显著均值回归**（即 BOS+回踩的延续逻辑站在统计规律反面）；但毛 edge 仅 ~1.3bp vs 往返成本 ~5bp |
| Zeng et al. (2026, arXiv:2608.25348) | Binance BTCUSDT 永续 | **23 个策略全部正 IC 但净 Sharpe 为负**；460 个「手续费×滑点」组合**无一为正** |
| Mesfin (2026, arXiv:2605.04004) | MNQ 5 分钟，947 交易日，14 信号族 | **0/14 通过**预注册标准（含阳性对照证明方法有检出力） |

**关键补充**：该领域**同行评审实证文献实际是空白的** —— 找不到任何一篇同行评审论文给出纯规则化 ICT OB/FVG 入场策略扣成本后的胜率/PF/Sharpe/p 值。

**「正 IC 但净 Sharpe 为负」** 是已复现的普遍现象：信号方向对，但幅度不足以覆盖成本 —— 这正是本项目实测到的形态（毛 +0.134R vs 成本 0.116R）。

---

# 第三部分：建议

## 9. 改进清单（按性价比排序）

### 优先级 1：风险预算归一化 —— 唯一「不需要 edge 就能改善结果」的改动

`bull_strong_long_risk_per_trade` 从 **0.12 降到 ≤0.035**，并移除 regime 对风险的方向性放大。

- 证据：12% 桶 8 笔亏 6,582 USDT；6% 桶 7 笔赚 4,167 USDT
- 用真实 R 序列复利推演：固定 3.5% 风险下终值 9,010（−10%），而实际约 −37%
- 同时修 `shadow_consecutive_loss_stop` 的暂停语义（从「暂停到下一 UTC 日」改为「暂停 N 笔 / N 天」），否则风控仍然形同虚设

### 优先级 2：止盈目标从 5.5R 降到 1.5R~2R

- 证据：MFE ≥2R 只 4/21，≥4R 为 0；5.5R 在消融里是最差档
- **但单靠降目标不够**（固定 2R 反而更差，因为止损被打掉比例上升），必须配合优先级 3

### 优先级 3：加入「停滞离场」+ 降低追踪启动阈值

规则：持仓 12 根 15m K 后若浮盈 < 0.5R，市价离场；追踪止损启动从 2.06R 降到 1R（距离 0.5R）。

- 证据：MFE < 0.5R 的 6 笔全亏（−6.20R）；持仓 < 24bar 的 5 笔 −3.92R
- 效果：等风险最大回撤 **23.8% → 9.9%**，扣费后总 R **−2.44 → −0.38**
- 实现位置：`strategy/scalp_robust_v2_core.py` 出场判定链

### 优先级 4：把出场层从 4 层砍到 2 层

当前是「固定目标 + ATR 追踪 + 压力位/整数关口 + 时间止损」互相覆盖。建议保留 **1 个主出场（追踪）+ 1 个风控出场（硬止损/停滞）**。

- 理由：外部调研未找到任何证据支持「整数关口/成交量簇/前高前低/长影线」作为止盈位在 BTC 永续上有效（明确的「未找到可靠证据」清单第 6 条）
- 复杂度本身是负债：当前系统有 ~180 个配置参数、21 笔样本，**验证能力与复杂度严重不匹配**

### 优先级 5：入场过滤（先影子验证）

候选：`ADX > 30` 且 `regime_label != flat`；以及高波动 regime 过滤（ATR14 > 2× ATR50 均值时形态可靠性显著恶化）。

- 证据：过滤后 8 笔 +2.51R（胜率 50%）；被过滤 12 笔 −2.90R（胜率 33%）
- 样本仅 21 笔，**统计功效很弱**，建议先以 shadow 模式记录 20~30 笔再启用

### 战略级选项：升周期

15m 尺度上加密市场以均值回归为主且毛 edge (~1.3bp) 不足覆盖成本 (~5bp)。若继续做方向性押注，**1H/4H 的证据基础明显更好**。但这是战略级决定，建议先做完优先级 1–3。

## 10. 不建议做的事

- ❌ 继续叠加出场层（压力位 + ATR + 时间 + 固定目标已互相覆盖）
- ❌ 提高杠杆（**当前 live 配置是 20x**，模板里写 10x，实际执行以 20x 为准）
- ❌ 在入场边际未证实为正之前加大风险预算
- ❌ 继续在 21 笔样本上调参 —— 这是「用复杂度买过拟合」

## 11. 下一步验证

1. **影子模式跑过滤规则**：`ADX>30 且非 flat` 作为唯一改动，记录 20~30 笔虚拟交易
2. **手续费敏感性**：测试 maker 挂单，看 0.116R/笔能否压到 0.05R 以下
3. **样本扩充**：纳入 2026-04-11 起账户全部 100 笔已平仓持仓，扩大统计功效
4. **形态级研究**：当前证据指向「入场边际≈0」，需要回答的是**哪一类 SMC 形态（sweep+MSS+FVG 组合）在 BTC 15m 上有正期望** —— 这需要大样本形态级回测，而非继续调出场参数
5. **止损位决策**：如果优先级 1–3 做完仍是负期望，**正确的决定是停用该策略族，而不是继续调参**

## 12. 被排除的假设（诚实记录）

| 假设 | 检验结果 |
|---|---|
| 「影子风控闸门拦掉实盘平仓，导致只能被交易所止损打掉」 | **不成立**。155 次 `shadow_gate_skipped_close` 与 8,496 条 `Live state mismatch` 错误全部集中在 2026-05-06→05-15 与 07-21→07-31 的空仓/纸面期，**没有一次落在实盘持仓区间内** |
| 「入场滑点/成交质量差导致亏损」 | **不成立**。平均滑点 −1.5bp（有利方向），中位 0.0bp |
| 「止损太紧」 | **不是主因**。最佳 MFE 只 3.5R，即便止损完美，5.5R 目标也永不触发 |

另发现两处低频状态不一致（真实但非主因）：

- `close_without_open` id=14213（2026-05-07 幽灵平仓）
- `open_without_close` id=24638（2026-08-20 开仓无对应平仓记录）

---

# 第四部分：第二轮补充审计

## 13. 回测前视偏差审计

外部最强证据（Northmark，995,550 笔预注册检验）指出 SMC/ICT 回测最常见的工具级缺陷是**在成交当根 K 线上判定止盈**，修正后其 6 个标的全部由盈转亏。据此逐项审计本项目。

### 已排除的风险 ✅

| 检查项 | 结论 | 证据 |
|---|---|---|
| 开仓当根 K 线是否判定出场 | **否** | `evaluate_range` 先调 `manage_position(i)`，再走开仓逻辑；开仓后循环进入 `i+1`（`strategy/scalp_robust_v2_core.py:2562-2610`） |
| 出场内部止损/止盈判定顺序 | **止损先判**（保守） | BULL 分支 `curr.l <= sl_price` 在 `curr.h >= target_price` 之前（`:1610` vs `:1638`） |
| 成交价假设 | 信号 bar **收盘价** `entry = curr.c` + `slippage_bps` | `_open_action_from_pending:3000` |
| 结构/订单块查找是否回看 | **只回看** | `find_ob` 用 `range(start, idx)`，不含 `idx`（`:674`） |
| SMC 事件路径的 4h 对齐 | **已自行修正** | `completed_4h_idx_for_entry = mapping[entry_idx] - 1`（`report_smc_trade_context.py:104`） |
| 实盘 4h 信息口径 | **无前视** | `bot/okx_executor.py:2395` 使用 `build_precomputed_state_confirmed_4h` |

### 发现的风险 ⚠️（历史记录 · 已封堵）

> **下方描述的是清理前的状态。相关开关与函数已全部删除，见 §22 与
> `docs/backtest_lookahead_policy.md`。保留本节是为了记录问题的成因与发现过程。**

**当时 `scripts/live_readiness_report.py` 默认使用有前视的 4h 信息口径。**

`align_timeframes()` 的语义是「取 ts ≤ 当前 15m 时间的 4h」→ 指向**当时仍在形成中的那根 4h**。

```python
# strategy/scalp_robust_v2_core.py:455
while c4h_idx + 1 < len(c4h) and c4h[c4h_idx + 1].ts <= candle.ts:
    c4h_idx += 1        # ← 4h bar 的 ts 是「开盘时间」
```

而 `bias_4h[i]` 用的是**该 4h 的收盘价**（`precompute_4h_bias:491  cp = c4h[c4h_idx].c`），`regime_1d_*` 同样把 `candle.c` 并入 EMA（`precompute_1d_regime:526`）。

`build_precomputed_state_confirmed_4h()` 用 `confirmed_idx = mapped_idx - 1` 修正，并把结果写入 `*_for_15m` 数组；`_bias_for_idx` 优先读它。**但只有它被调用时才生效。**

当时的调用链（**现已不存在**）：

```python
# scripts/research_smc_standalone_v1.py:359 等调用点
load_prepared_data(data_15m_path=..., data_4h_path=..., start=..., threshold_payload=None)
#                  ↑ 未传 confirmed_4h_only / informative_asof_from_15m，两者默认 False
# → else 分支: precomputed = build_precomputed_state(c4h, c15m)   # 裸口径
```

### 量化影响

| 指标 | 分歧率（raw 有前视 vs confirmed 无前视） |
|---|---:|
| **4h bias（方向）** | **11.33%** |
| 1d regime bull EMA100 | 3.86% |
| 1d regime bull EMA200 | 2.69% |
| 1d regime bear EMA100 | 3.97% |
| 4h bull trend score | 11.92% |
| 4h bear trend score | 11.33% |

（13,693 根 15m；只在 `mapping[i] >= 1` 的 13,677 根上统计，相邻两根 4h 的 bias 方向不同比例为 **11.3%**）

**这意味着回测在约 11% 的 K 线上「多看了一根尚未收盘的 4h」，方向判断与实盘不一致。** 影响面明确，但**对盈亏的具体影响尚未量化** —— 需要把 `confirmed_4h_only=True` 重跑一遍基线才能给出数字。

> 复现：`python3 scripts/audit_backtest_lookahead.py --klines-15m <15m.csv>`

## 14. 止损宽度校准：止损不是太紧，是太宽

我此前的假设「止损太紧导致被噪音打掉」**被数据否定**。用项目自身的 Wilder ATR(14)（15m）计算每笔 `止损距离 / ATR`：

| 统计量 | 值 |
|---|---:|
| 最小 | 1.56× |
| 25 分位 | 3.62× |
| **中位数** | **4.30×** |
| 75 分位 | 5.72× |
| 最大 | 8.84× |

| 区间 | 笔数 | 占比 | 外部参考（噪音触发率） |
|---|---:|---:|---|
| < 1.0×（噪音区） | **0** | **0%** | >65% 被噪音打掉 |
| 1.0 – 1.5× | 0 | 0% | ~50% |
| 1.5 – 2.0× | 1 | 5% | ~38% |
| 2.0 – 3.5×（外部推荐高原） | 3 | 15% | ~21%，期望值最高区 |
| **> 3.5×** | **16** | **80%** | 更宽，盈亏比下降 |

**止损中位 4.30×ATR，80% 的交易宽于外部实证推荐的高原（2.0–3.5×），没有一笔落在噪音区。**

### 目标距离：0/21 的几何解释

| 项目 | 中位数 |
|---|---:|
| 止损距离（%价格） | 1.26% |
| 目标 R 倍数 | 5.50R |
| **目标距离（%价格）** | **6.57%** |
| **目标距离（×ATR）** | **21.6×** |

- 实测**典型有利波动** ≈ **1.33%** 价格移动（中位 MFE 1.06R × 中位止损 1.26%）
- 配置**目标要求**中位 **6.57%** → 是典型波动的 **5.0 倍**

**止损宽度与目标 R 是相乘关系。** 结构止损（订单块下沿）天生偏宽，再乘以 5.5R，就得到一个 21.6×ATR 的目标 —— 这在 BTC 15m 上不存在。

### 诚实的限制

- **高原区只有 3 笔**，因此「收窄止损更好」这个读数**没有统计支撑**（n=3）
- R 归一化本身有机械性：止损越窄，同样的价格移动折算的 R 越大。跨桶比较 MFE_R 部分是同义反复
- 真正的问题不是「哪个桶的 R 更高」，而是「**宽止损 × 高 R 目标**这个组合在几何上不可达」—— 这一点不依赖桶间比较

> 复现：`python3 scripts/audit_btc_scalp_stop_atr.py --trades <trades.csv> --klines <15m.csv>`

## 15. 一处事实修正

live 配置里 **`enable_smc_short_live = False`、`enable_gap_smc_short_live = False`** —— 两条 SMC 做空路径**已关闭**。实盘当前只有 **SOTA 多头**一条路径在跑（`config.live.high-leverage-structure.json`）。

因此第 13 节的前视偏差只影响 SOTA 多头路径；SMC 事件路径（`scan_events` + `trade_rows_for_events`）已用 `mapping - 1` 自行修正，不受影响。

## 16. 修正后的改进优先级

在第 9 节的基础上，按第二轮结果重排：

| 新排序 | 改进 | 变化 |
|---|---|---|
| **1** | 风险预算归一化（0.12 → ≤0.035）+ 修风控暂停语义 | 不变 |
| **2** | **在回测里开启 `confirmed_4h_only=True`，重跑基线** | 🆕 新增，零成本，先确认历史参数是否建立在乐观口径上 |
| **3** | **止损收窄到 2.0–3.5×ATR 高原，或把目标降到 ~2R** | ⬆️ 从第 2 位提升，机制已闭合 |
| **4** | 停滞离场 + 追踪启动阈值降到 1R | ⬇️ 降级（外部时间止损证据为负） |
| **5** | 入场过滤（ADX>30 且非 flat）先影子验证 | ⬇️ 降级（PRUVIQ 反证 + 小样本） |

**优先级 2 与 3 的关系**：2 是「先确认真实基线」，3 是「针对已闭合的机制动手术」。如果 2 的重跑显示历史参数确实建立在乐观口径上，那么 3 的参数需要重新拟合而不是直接沿用。

---

# 第五部分：前视偏差的实测量化代价

## 17. A/B 回测：2022-01-01 → 2026-09-20

用**同一份配置、同一段行情**（165,470 根 15m K），只切换 4h 信息口径：

```bash
# A: 回测默认口径（有前视）
python3 scripts/live_readiness_report.py --start-date 2022-01-01 ...
# B: 实盘口径（无前视）
python3 scripts/live_readiness_report.py --start-date 2022-01-01 --confirmed-4h-only ...
```

| 指标 | A raw（有前视） | B confirmed（无前视） | 差异 |
|---|---:|---:|---:|
| **总收益** | **+8,525.45%** | **+396.26%** | **−8,129.19** |
| Sharpe | 2.620 | 1.616 | −1.004 |
| **最大回撤** | 80.61% | **92.00%** | **+11.39** |
| 盈亏因子 | 1.257 | 1.082 | −0.175 |
| 胜率 | 40.26% | 35.79% | −4.47 |
| 交易数 | 313 | 299 | −14 |
| 止盈率 | 8.31% | 6.69% | −1.62 |
| 手续费 | 52,055.76 | 6,723.23 | −45,332.53 |
| 滑点成本 | 26,066.19 | 3,365.38 | −22,700.81 |
| 最长连亏 | 9 | 11 | +2 |

**结论：回测默认口径把 2022 年至今的总收益夸大了约 21 倍（8,525% → 396%），同时把最大回撤低估了 11 个百分点（80.6% → 92.0%）。**

即「回测里那个漂亮曲线」里，**绝大部分是那 11% 的 K 线上多看一眼未收盘 4h 换来的**。

注意手续费差异（52,056 vs 6,723）：这不是费率变了，而是 A 的权益被虚高放大后，同等风险比例下的仓位与费用被复利放大。

## 18. 实盘 / 回测收敛性核对

README 把 `scripts/audit_live_replay_trade_convergence.py` 列为 replay/live 收敛入口，但**该文件此前并不存在**。本轮实现它：用同一份配置跑回测，把每一笔回测交易与实盘真实成交按开仓时间对齐。

| 口径 | 回测笔数 | 匹配上实盘 | 回测多出 | 实盘漏掉 |
|---|---:|---:|---:|---:|
| raw（有前视） | 48 | **7** | 41 | 14 |
| **confirmed（无前视）** | 47 | **11** | 36 | 10 |

| 指标（匹配子集） | 实盘 | raw 回测 | confirmed 回测 |
|---|---:|---:|---:|
| 平均 R | 0.31 | 0.16 | **0.25** |
| R 合计 | 3.38 | 1.13 | **2.70** |
| 开仓时间差（中位） | – | 0.0 分钟 | **0.0 分钟** |
| 开仓价差（中位） | – | 0.0 bp | **0.0 bp** |

**两个结论：**

1. **无前视口径的收敛性明显更好** —— 匹配笔数 7 → 11，平均 R 更接近实盘。这独立印证了第 13 节的判断。
2. 凡是匹配上的交易，**开仓时间差与开仓价差中位数都是 0.0** —— 说明回测的入场逻辑与实盘完全一致，差异只在「哪些信号被执行」。

### 亏损的分布很集中

| 实盘 21 笔 | 笔数 | 净盈亏 (USDT) |
|---|---:|---:|
| 回测能复现的 | 11 | **+3,751** |
| 回测复现不了的 | 10 | **−7,970** |

**全部亏损集中在回测复现不出来的那 10 笔上。**

## 19. ⚠️ 一个必须排除的混淆因素：配置漂移

上面的收敛结论**有一个尚未排除的混淆**：我是用**当前配置**（mtime 2026-09-09）回测 4 月至今，而配置在期间被反复修改——服务器上仅 8/20–9/10 就有 **10+ 个备份**：

```
bak_20260820T085919Z          bak_15x_80pct_20260821T110744Z
bak_notional_3000_20260821T035546Z   bak_pct30_20260821T040341Z
bak_lev15_20260821T161346Z    bak_nodynamic_20260822T144223Z
bak_risk15x_20260822T145809Z  bak_bs12_20260822T154937Z   ← 把 bull_strong 风险设为 12%
bak_1R_bw5_20260823T033743Z   pre_alpha_gate_20260910T025529Z
```

`bak_bs12_20260822T154937Z` 正是第 4 节里那次把 `bull_strong_long_risk_per_trade` 提到 **0.12** 的改动，时间点在最大连亏之前。

**因此「实盘多做了 10 笔亏损交易」这个读法目前不成立** —— 更可能的原因是那些交易发生在不同配置下。要坐实必须做**逐段 point-in-time 回测**：按配置备份的时间戳把区间切片，每段用当时生效的配置回测再拼接。

这一步尚未执行。

## 20. 更新后的结论与下一步

**已经站得住的结论（不受配置漂移影响）：**

1. 回测默认口径有真实的前视偏差，代价是 **+8,129 个百分点**、回撤低估 11pp —— 这一条是同一配置的 A/B，干净
2. 无前视口径与实盘的收敛性显著更好（11/21 vs 7/21，误差中位 0.0）
3. 即便用无前视口径，2022 至今也只有 **+396% / 最大回撤 92%** —— 风险调整后并不优秀，且 92% 回撤在实盘意味着已被清算

**下一步（按顺序）：**

| # | 事项 | 目的 |
|---|---|---|
| 1 | **逐段 point-in-time 回测** | 排除配置漂移，弄清实盘亏损到底是配置问题还是执行问题 |
| 2 | 把 `confirmed_4h_only=True` 设为默认 | 防止后续所有参数决策继续建立在乐观口径上 |
| 3 | 用无前视口径重跑参数稳健性审计 | `audit_sota_bucket_robustness.py` 等结论可能同样被高估 |
| 4 | 风险预算归一化（0.12 → ≤0.035） | 不变，仍是唯一「不需要 edge 就能改善」的改动 |
| 5 | 止损收窄到 2.0–3.5×ATR / 目标降到 ~2R | 机制已闭合 |

> **重要提醒**：第 3 条意味着**过去所有基于旧口径回测做出的参数选择都需要重新审视** —— 包括那 180 个参数是怎么选出来的。

---

# 第六部分：三条路径的前视偏差逐项对照

## 21. 日本服务器部署版本：无前视 ✅

**结论：线上运行的代码是三条路径里唯一完全因果的，没有前视偏差。**

验证方式：服务器上 `strategy/scalp_robust_v2_core.py` 与本地 **md5 完全一致**（`8fc0d1d59ef7a3a8fcdf8ea3d2f48c9f`），因此对本地代码的审计结论可直接适用于部署版本。

### 逐项对照

| 检查项 | 实盘（服务器部署） | 回测默认 raw | 回测 confirmed |
|---|---|---|---|
| 开仓当根 K 线是否判出场 | **无** ✅ | 无 ✅ | 无 ✅ |
| 止损/止盈判定顺序 | 止损先判（保守）✅ | 同 | 同 |
| 成交价假设 | 信号 bar 收盘价 + 滑点 ✅ | 同 | 同 |
| `find_ob` 是否回看 | 只回看 ✅ | 同 | 同 |
| **4h bias 方向** | `build_precomputed_state_confirmed_4h` → `mapping − 1` ✅ | `mapping` ❌ | `mapping − 1` ✅ |
| **1d regime flags** | 同上 ✅ | ❌ | ✅ |
| **4h bull/bear trend score** | 同上 ✅ | ❌ | ✅ |
| **4h regime label / features** | 缓存为空 → `c4h[:mapping]` ✅ | 因果 ✅ | 因果 ✅ |

### 实盘为什么是干净的

线上 `bot/okx_executor.py:2395` 构建引擎时用的是：

```python
engine = ScalpRobustEngine(
    informative_candles,
    primary_candles,
    align_timeframes(informative_candles, primary_candles),
    build_precomputed_state_confirmed_4h(informative_candles, primary_candles),  # ← 无前视
    self.config.to_scalp_strategy_config(),
)
```

三条回退链在实盘里的终点都是因果的：

1. `bias` / `regime flags` / `trend score` —— 由 `confirmed_4h` 直接写入 `*_for_15m` 数组（`mapped_idx − 1`）
2. `regime label` —— 线上**从不注入** `_regime_switch_cache`（该注入只存在于 `scripts/live_readiness_report.py:295`），故回退到 `_regime_label_for_idx` → `_effective_regime_history(idx)` → **`self.c4h[:c4h_idx]`（切片不含当前那根未收盘 4h）**
3. `regime features` —— 同理，`_effective_regime_history` 后交给 `compute_regime_features`

### 因此产生了一个关键的策略性推论

**实盘和 canonical 回放工具跑的都是无前视口径；只有 `live_readiness_report.py` 这一条路径在修改前是有前视的。**

| 工具 / 路径 | 修改前口径 | 是否受影响 |
|---|---|---|
| 实盘 `bot/okx_executor.py` | 无前视 ✅ | 否 |
| **`scripts/replay_sota_smc_live_shadow.py`**（canonical 参数回放，bucket 审计的输入来源） | **无前视** ✅ | **否** |
| `research_smc_standalone_v1.py` / `reproduce_smc_*` / `scan_*` | 不消费 4h 数组（直接引用 0 次），且用 `completed_4h_idx_for_entry` ✅ | 否 |
| **`scripts/live_readiness_report.py`** | **有前视** ❌ | **是** |
| `ScalpRobustEngine.from_candles` | 有前视（潜在陷阱）❌ | 是（已修） |

关键证据：2026-05-17 的提交 `5d3497e`（"Promote chained FVG bucket live replay"）把

```json
"confirmed_4h_only": true,
```

**写进了 `config/config.live.high-leverage-structure.template.json` 与 `config/config.paper.high-leverage-structure.json`**，同时给 `replay_sota_smc_live_shadow.py` 加了 `_apply_config_defaults()`，让它**从配置读取**该开关。

所以 canonical 回放自 2026-05-17 起就是无前视的；服务器上 8/20–9/10 的 **9 个 live 配置备份全部是 `"confirmed_4h_only": true`**。

**修正后的结论**：第 17 节那 +8,129 个百分点的虚增，**只影响 `live_readiness_report.py` 的输出**（那是一个 readiness 报告工具），**不影响经 canonical replay 得出的参数与 bucket 结论**。此前「所有历史参数决策都要推倒重验」的说法**范围过宽，予以更正**。

`live_readiness_report.py` 的修复仍然必要 —— 它让这个工具与 canonical 回放口径对齐，避免以后用它做决策时再次被误导。

> 复现：`python3 scripts/audit_backtest_lookahead.py --klines-15m <15m.csv>` 会逐项打印上表的检查结果。

---

# 第七部分：修复与验证

## 22. 已落地的代码修复

> 本节记录**最终状态**。清理过程中曾经短暂存在过的 `--raw-4h-state` 逃生通道与
> `--confirmed-4h-only` flag **已全部删除**，见 `docs/backtest_lookahead_policy.md`。

### 修复 1：回测口径收敛为唯一路径

`scripts/live_readiness_report.py` 的 `load_prepared_data()` 删除三路分支，只剩：

```python
precomputed = build_precomputed_state_confirmed_4h(c4h, c15m)
```

`confirmed_4h_only` / `informative_asof_from_15m` 两个参数与对应的三个 CLI flag 全部移除。
所有调用点（`research_smc_standalone_v1.py` 等）自动使用无前视口径。

### 修复 2：彻底封死歧义入口

| 已删除 | 类型 |
|---|---|
| `--raw-4h-state` | CLI flag（曾有前视逃生通道） |
| `--confirmed-4h-only` | CLI flag |
| `--informative-asof-from-15m` | CLI flag |
| `load_prepared_data(confirmed_4h_only=...)` | 函数参数 |
| `build_precomputed_state_asof_15m()` | 函数（无人使用的第二因果实现） |
| `build_precomputed_state()` | **改名 `_build_precomputed_state_lookahead()` 并私有化** |

**验证**：清理前 `--raw-4h-state` 的输出与旧默认逐项完全一致（175.41% / Sharpe 8.625 /
DD 28.17% / 12 笔），确认那条通道就是有前视口径；删除后回测在同一窗口输出 204.31%。

### 修复 3：`from_candles` 的潜在前视陷阱

`strategy/scalp_robust_v2_core.py`

```python
@classmethod
def from_candles(cls, c4h, c15m, config=None):
-    precomputed = build_precomputed_state(c4h, c15m)
+    precomputed = build_precomputed_state_confirmed_4h(c4h, c15m)
```

**为什么这条重要**：`bot/okx_executor.py` 构建线上引擎时有两条分支——

```python
if bool(self.config.enable_sota_score_gate_live):
    engine = ScalpRobustEngine(..., build_precomputed_state_confirmed_4h(...))  # 无前视 ✅
else:
    engine = ScalpRobustEngine.from_candles(...)   # 修复前：裸 builder ❌
```

当前 live 配置 `enable_sota_score_gate_live = True`，所以实盘走的是安全分支。**但只要有人把它翻成 `False`，实盘就会静默地变成有前视**。这条修复消除了这个陷阱。

`from_candles` 还被两条回测路径使用，修复后它们也一并转为无前视：

- `scripts/backtest_config_report.py`
- `scripts/replay_proxy_strategy_router.py`

### 修复 4：恢复 README 引用但缺失的文件

| 文件 | 状态 |
|---|---|
| `scripts/replay_sota_smc_live_shadow.py` | README 称其为 canonical 回放入口，实为 `ba669df` 删除 → **已从 git 恢复**（1165 行） |
| `config/high_leverage_pressure_target_cap_best.params.json` | 同一 commit 删除，回放脚本依赖它 → **已从 git 恢复**（36 个参数） |
| `scripts/audit_live_replay_trade_convergence.py` | README 引用但从未存在 → **本轮新写** |

### 回归验证

```
python3 -m pytest -q   →   173 passed
```

**173 个测试全部通过**，修改共享默认值未破坏任何现有路径。

## 23. 尚未完成

| 事项 | 状态 |
|---|---|
| 用无前视口径重跑 bucket 稳健性审计 | **不需要** —— canonical replay 本来就是无前视（见第 21 节修正） |
| 逐段 point-in-time 回测（排除配置漂移） | 未开始 |

## 24. ⚠️ 一处需要更正的中间结论

本轮调查过程中我曾给出「所有基于旧口径回测的历史参数决策都需要推倒重验」的判断。**该判断范围过宽，予以更正。**

原因是我最初只检查了 `load_prepared_data` 的函数默认值（`False`），据此推断所有调用方都走有前视口径。后来发现 **canonical 回放脚本有自己的配置驱动逻辑**（`_apply_config_defaults`），而配置里写的是 `true`。教训：**函数默认值 ≠ 实际生效口径**，必须查调用方。

更正后的准确表述：

- ✅ `live_readiness_report.py` 在修改前确实有前视，其输出被高估约 21 倍
- ✅ `ScalpRobustEngine.from_candles` 确实存在前视陷阱
- ❌ ~~canonical 回放 / bucket 审计 / 参数选择受影响~~ —— **不受影响**

---

## 附：产出文件

**文档（全仓仅两份）**

| 文件 | 说明 |
|---|---|
| `docs/backtest_lookahead_policy.md` | **前视口径政策** —— 唯一规则、禁止事项、验证方法 |
| `docs/btc_scalp_research.md` | 本文件（研究结论 + 附录 A/B 外部证据） |

**分析脚本（全部无前视口径，可复现）**

| 文件 | 说明 |
|---|---|
| `scripts/audit_backtest_lookahead.py` | 前视审计：口径检查 + 分歧率量化 |
| `scripts/audit_btc_scalp_live_history.py` | 历史交易深度审计（21 笔） |
| `scripts/audit_btc_scalp_exit_ablation.py` | 出场规则 / 风险预算反事实消融（19 条规则） |
| `scripts/audit_btc_scalp_entry_edge.py` | 入场边际 vs 随机对照组（p = 0.327） |
| `scripts/audit_btc_scalp_stop_atr.py` | 止损宽度 ATR 校准（中位 4.30×ATR） |
| `scripts/audit_live_replay_convergence.py` | 实盘 / 回测收敛性核对（README 曾引用但文件缺失，已补） |
| `scripts/replay_sota_smc_live_shadow.py` | canonical 回放（`ba669df` 删除，已从 git 恢复） |

**数据**

| 文件 | 说明 |
|---|---|
| `var/tokyo_audit/btc_scalp_20260920/BTC_USDT_USDT-15m-futures.feather` | 服务器权威 15m（2020-01-01 → 2026-09-20，235,646 行） |
| `var/tokyo_audit/btc_scalp_20260920/BTC_USDT_USDT-4h-futures.feather` | 服务器权威 4h（14,728 行） |
| `var/tokyo_audit/btc_scalp_20260920/trades_audit.csv` | 21 笔交易结构化明细 |
| `var/tokyo_audit/btc_scalp_20260920/okx_history.json` | OKX 权威持仓历史（原始） |
| `var/live_readiness/clean_smoke/` | 清理后的回测输出示例 |

> 各 `scripts/audit_*.py` 的 `--out-md` 会按需重新生成明细报告；
> 这些是**可再生的中间产物**，不作为文档保留，以免与本文档结论产生版本漂移。

---

# 附录 A：外部证据调研 — 明确的「未找到可靠证据」清单

> 以下问题经 150+ 组检索**未能找到**可靠证据，不做任何编造。
> 这些空白意味着相关结论只能自建回测得出，不能靠引用文献。

为避免误导，以下问题我**未能找到**可靠证据，不做任何编造：

1. **未找到**任何主流同行评审期刊上对 order block / FVG / liquidity sweep 做过严格统计检验、并给出扣成本后胜率 / PF / Sharpe / p 值 / t 值的论文。**该领域的同行评审实证文献实际是空的。**
2. **未找到**「固定 R 目标 vs 追踪止损」在加密永续上的直接实证对比。
3. **未找到任何 R-multiple 百分比分布表，也没有任何 4R 目标实际达成率的数据**（60+ 组检索，零命中）。连「趋势跟踪胜率 30–40%」也无法追溯到一手带数据来源。
4. **未找到加密 MAE/MFE 分布的可靠数据**（完全空白）。
5. **未找到** BTC 15m 周期上「最优 ATR 止损倍数」的学术数值（只有 6h 周期的 α∈[2.0,3.5] 高原）。
6. **未找到**任何同行评审判的加密时间止损研究；唯一一份干净的实证检验（Alvarez，均值回归）结论是**不改善收益**。
7. **未找到**「分批出场降低期望值」这一流行论证的原始出处（Van Tharp 原文、Curtis Faith 段落、Adam Grimes 一手分析均无法定位；Price Action Lab 经站内搜索确认不存在此类文章）。**没有任何同行评审论文带数字讨论 partial exits / staged exits**，也**没有任何已验证数字表明 scale-out 改善 Sharpe 或回撤**。
8. **未找到**任何证据支持「整数关口 / 成交量簇 / 前高前低 / 长影线」作为止盈位在 BTC 永续上的有效性。
9. **未找到**加密领域「Kelly vs 固定分数 vs 波动率目标」的严格三方对比实证。
10. **未找到**「20x 杠杆 + 1.25% 止损」这一特定组合的爆仓/滑点实证研究。
11. **未找到**可靠证据支持「资金费率极值可作反向择时信号」。
12. **未找到 FVG 填补率（fill rate）本身**作为独立统计量的研究 —— Northmark 测的是「回撤进入缺口的入场」，不是「缺口被填补的频率」。
13. **未找到**任何 SMC/ICT 的 meta 分析或系统性文献综述；**Quantpedia / Robot Wealth / QuantInsti / Hudson & Thames 上没有任何 SMC/ICT 专题研究**。
14. **未能取得全文**（故未引用具体数值）：Dai et al. (2021) 追踪止损论文数值表、Baur & Cahill (2019) 日内效应论文、Springer 清算生存分析论文、Harvey et al. (2018) 波动率目标论文、Glynn & Iglehart (1995)、Zarattini et al. (2025) 的部分子数字。
15. **未能读取的 PDF-only 高价值线索**（若后续有 PDF 通道应优先攻）：
    - `insis.vse.cz/zp/88190` —— 片段：「Comparing **34,442 trades in low/medium volatility** against **9,435 trades in high volatility**, the test yielded a significant result...」
    - `acfr.aut.ac.nz` 加密动量论文 —— 片段：「**Every long-short portfolio with a holding period of less than a week yields a negative mean return**」（直接回答持有期问题）
    - Griffith University 仓库 —— 含字符串「**Ho: Sharpe_trailing|HIGH > 0.00**」（高波动下追踪止损 Sharpe 的正式假设检验）
    - vtad.de Oliver Reiss (2025) —— 含「Time stop horizon vs Profit Probability」表格
    - Scilit《Enhancing Smart Money Concept Win-Rates using a Naïve Bayes Volume Filter》—— 唯一发现的「用成交量过滤提升 SMC 胜率」疑似学术标题（403）
    - Quantica Capital《Quarterly Insights》—— 含真实 CTA 逐笔盈亏分布图

### 方法学陷阱提示

- **「Fair Value Gap」是歧义术语**：主流金融/技术分析 = **估值缺口**（例如 Subedi et al. 2024, Sudurpaschim Spectrum 2(2):81-95 讨论的尼泊尔银行股就是此类），ICT = **三根 K 线 imbalance**。**引用时必须核对定义**，否则会得到完全错误的结论。
- **本次检索中每一个声称 SMC/ICT 高胜率（70–80%）的来源，要么是营销内容，要么存在可识别的统计缺陷**（成交 K 线前视偏差、用样本外交易训练 ML 过滤器、无成本模型、无样本量）。
- **引文偏差警告 1**：arXiv:2602.11708 把 Kaminski & Lo 转述为「adaptive exits significantly improves downside protection」，这**窄于且偏离了**原文结论（原文：随机游走下止损**总是降低**期望收益；期刊版更把增值限定在长采样频率）。**引用二手转述需谨慎。**
- **引文偏差警告 2（本领域被误引最多的论文）**：**Glynn & Iglehart (1995), *Management Science* 41(6):1096–1106 是纯理论/数学论文，不是实证回测。** 它考虑的是「离散随机游走」与「连续布朗运动，两者均为正漂移」，推导追踪止损下收益与持仓时长的分布。**它没有标的、没有样本期、没有 Sharpe、没有胜率、没有回撤 —— 摘要中零实证数字。任何「Glynn & Iglehart 发现追踪止损收益 X%」的说法都是误引。** 它的结构性含义是：正漂移下追踪止损按构造**截断右尾**。
- **「Sharpe 口径 vs 期望值口径」必须分清**：arXiv:2604.27150 支持「75% 分批止盈」是按 **Sharpe** 排序的，**不是按期望值**。该文**不能**用来反驳「分批降低期望值」。这是本次调研中最容易误用的一处证据。
- **「参数组合数」不等于「独立交易数」**：BreakOrb 声称的「2,870 万」是**参数组合数**而非独立交易笔数（对比 Northmark 明确报告的 995,550 笔交易）。两者不可同日而语。

---

---

# 附录 B：外部证据调研 — 引用来源汇总

### 同行评审期刊

| 文献 | 期刊 | 本报告用途 | 标签 |
|---|---|---|---|
| Kaminski & Lo (2014) | *Journal of Financial Markets* 18:234–254 | 止损在随机游走下必然降期望；期刊版限定于长采样频率 | [同行评审] |
| Lo & Remorov (2017) | *Journal of Financial Markets* 34:1–32 | 紧止损因交易成本跑输买入持有 | [同行评审] |
| Libertini | *Journal of Investment Strategies* | 止损在均值回归/反趋势策略中无效 | [同行评审] |
| Sadaqat & Butt (2023) | *J. Behavioral and Experimental Finance* 39:100833 | 加密 147 币：止损动量策略 Sharpe/alpha 显著更高 | [同行评审] |
| Białkowski (2020) | *Economics Letters* 191:108834 | **加密多头存活率 ≤35%**；止损使波动率与收益近乎减半 | [同行评审] |
| Dai, Marshall, Nguyen & Visaltanachoti (2021) | *International Review of Finance* 21(4):1334–1352 | 追踪止损降风险但不增收益；下跌市最有效；紧止损被成本摧毁 | [同行评审] |
| Glynn & Iglehart (1995) | *Management Science* 41(6):1096–1106 | **纯理论论文**（无实证数字）；追踪止损按构造截断右尾 | [同行评审] |
| Fonseca (2026) | *Mathematics* 14(12):2182 | ATR 倍数是「结构性近冗余参数」，CAGR 平坦高原过 Bonferroni | [同行评审] |
| Duarte (2022) | *EJBMR* 7(3) | 追踪止损在极高波动/危机场景减少亏损 | [同行评审] |
| Moreira & Muir (2017) | *Journal of Finance* 72(4) | 波动率管理组合改善 Sharpe | [同行评审] |
| Harvey et al. (2018) | *Journal of Portfolio Management* | 波动率目标化改善风险调整收益、降回撤 | [同行评审] |
| Wu & Pinsky (2026) | *JRFM* 19(9):692 | 日内时段策略样本外不显著，SPA 检验无法拒绝原假设 | [同行评审] |
| Baur & Cahill (2019) | *Finance Research Letters* | BTC 日内/周内/月内效应（**具体数值未取得**） | [同行评审] |
| Brauneis et al. (2025) | *Review of Quantitative Finance and Accounting* 64(1) | 加密日内周期性强且各度量一致 | [同行评审] |
| Oukhouya, Noureddine & Imad (2026) | *Informatica* 50(13) | 唯一同行评审的 SMC/ICT 策略论文；纯规则基线 Sharpe 0.86（**无 p 值/样本量**） | [同行评审] |
| Friday, Pati, Mishra & Mishra (2026) | *IEEE Access* 14:26581–26603 | 提供 FVG 的确定性三根 K 线形式化定义（**非盈利证据**） | [同行评审] |

### 预印本

| 文献 | 编号 | 本报告用途 |
|---|---|---|
| Kitron & Wengrowicz (2026) | arXiv:2608.21888 | 15m 加密均值回归显著；毛 edge 1.3bp vs 成本 5bp |
| Zeng, Yang, Han & He (2026) | arXiv:2608.25348 | BTC 永续负结果；23/23 正 IC 但负净 Sharpe；460 成本格全负 |
| Bui & Nguyen (2026) | arXiv:2602.11708 | 加密永续 ATR 止损 α∈[2.0,3.5] 高原；消融 Sharpe 2.41→1.68 |
| Li, Laryea & Ihlamur (2026) | arXiv:2604.27150 | 加密分批止盈 75% 档；48h stale close（**Sharpe 口径，样本内**） |
| Mesfin (2026) | arXiv:2605.04004 | MNQ 14 个信号族 0/14 通过预注册标准；阳性对照 T=3.11/4.30 |
| Zarattini, Pagani & Barbon (2025) | SFI RP 25-80 / SSRN 5209907 | 加密趋势跟踪基准：净费 Sharpe >1.5，年化 alpha 10.8% |
| Kumar & Jenefer (2026) | Zenodo 19671502 | 波动率缩放 TSMOM：Sharpe 0.82→1.22，t 检验 p=0.5835 不显著 |
| Low & Herremans | *Expert Systems with Applications* | BTC 波动率缩放仓位提升累计收益与 Sharpe |
| Lempérière et al. (2014) | arXiv:1404.3274 | 趋势跟踪分年代 Sharpe |
| Sepp & Lucic (2026) | arXiv:2607.19497 | 趋势跟踪正偏度是结构性的 |

### 从业者研究（有数据披露）

- **Northmark / FXBrokerCompass** — [OB/FVG/Sweep 995,550 笔验证](https://www.fxbroker-compass.com/en/blog/fx-order-block-fvg-liquidity-sweep-verification) ｜ [Fibonacci+Elliott+MACD+RSI 198,000 次回测](https://www.fxbroker-compass.com/en/blog/fx-fibonacci-elliott-macd-rsi-verification) — **B−**（大样本+随机对照+预注册，但含联盟链接，原始数据需开户解锁）
- **Botsfolio SMC 回测库** — [Order Block](https://botsfolio.com/learn/order-block) ｜ [Fair Value Gap](https://botsfolio.com/learn/fair-value-gap) ｜ [Liquidity Sweep](https://botsfolio.com/learn/liquidity-sweep) — 期望值与 MFE/MAE 分布；**表格与正文自相矛盾，数字水平不可全信**
- **StratProof 独立复现** — [横截面动量失效](https://stratproof.com/blog/crypto-cross-sectional-momentum-does-not-work-anymore) ｜ [3,925 个噪声策略](https://stratproof.com/blog/my-graduation-gate-let-3925-fluke-strategies-through) ｜ [基差交易真实收益](https://stratproof.com/blog/basis-trade-works-but-not-as-much-as-they-say)
- **OKX 官方费率与资金费机制** — [X-Perps fees overview](https://www.marketzones.io/en-eu/help/okx-x-perps-eea-fees-overview)（Maker 0.02% / Taker 0.05%，资金费每 8h）
- **Volatility Box** — [波动率调整止损](https://volatilitybox.com/research/volatility-adjusted-stop-losses/)（止损宽度 vs 胜率/期望值/噪音触发率）
- **Alvarez Quant Trading** — [N 日出场（均值回归）](https://alvarezquanttrading.com/blog/n-day-exits-with-mean-reversion/)（时间止损不改善收益）
- **Bailey & López de Prado** — [Deflated Sharpe Ratio](https://pypi.org/project/deflated-sharpe/)

### 明确未采信的低质量来源

- **dev.to ICT OB+FVG 回测** — 作者为内容站运营者；代码含前视/重绘问题（`ta.atr` 与 `close` 同 bar、`process_orders_on_close=true`）；未披露滑点与资金费
- **Zenodo MNOG 预印本** — 「60–65% 回撤概率」明确是**转述他人从业者回测**，作者自述不可单独使用；无样本量/无统计量
- **TanvirCCC/algo-trading** — 自述「Random Forest classifier is trained on **622 OOS trades (post-WFV)**」，**用样本外交易训练过滤器，直接污染 OOS 声明**
- **starckyang/smc_quant** — 无成本、无交易笔数、无检验；训练期 23% vs 测试期 50% 胜率断裂说明不稳定
- **AsiaForexMentor / BrokerChampion / pinescriptforge / OPKYEI / lunefi** — 营销或 SEO 内容，无方法论
- **TradingView algolego 系列**（含「Placebo Control」设计）— **全部子域拒绝抓取，具体数字无法核实，标注为未找到可靠证据**
- **子术语陷阱**：Subedi et al. (2024), *Sudurpaschim Spectrum* 2(2):81-95 的「fair value gap」指用 SMA/MACD/布林/RSI/斐波那契识别的**估值缺口**，与 ICT 的三根 K 线 imbalance **不是一回事**，不可互引
- Zenodo MNOG / puffbird SMC 论文（评审强度与可复现性无法确认）
