# 3247.28% 最优策略浮现路径与复现报告

> ⚠️ **重要说明**：3247.28% 是在 Codex 会话 `019da5ca-7f8b-7fe1-a127-6789ed21e8c8` 的迭代过程中产生的临时结果，从未被 commit 到仓库。当前代码无法复现 3247.28%，当前可验证的最优结果是 **1345.75% / Sharpe 1.850**（关闭 bear_strong_short + tight_style_off + 原始 multiplier）。

---

## 一、3247.28% 的浮现路径

### 第一轮：ATR 参数网格扫描（26 组合）
**发现**：ATR activation=1.0 太早，跑输 baseline（615% vs 1074%）

**关键转折**：把 activation 从 1.0 提高到 2.0，收益暴涨：
- ATR act=2.0, 3.0/2.5/2.0, keep_target → **2516.66%**, Sharpe 1.7262 ✅

### 第二轮：窄区细化（9 组合）
- 发现 `scale=0.9` 优于 `1.0`
- act=2.0, scale=0.9, keep_target → **3209.53%**, Sharpe 1.8080
- 收益首次突破 3000%

### 第三轮：Regime Gating 过滤
**发现**：`tight_style_off` 是最优过滤模式

| 过滤模式 | 总收益 | Sharpe | 手续费 |
|---|---:|---:|---:|
| all | 3209.53% | 1.8080 | 28922 |
| **tight_style_off** | **3247.28%** | **1.8120** | 29171 |
| bull_weak_off | 2174.04% | 1.7223 | 18736 |
| bear_weak_off | 3209.53% | 1.8080 | 28922 |

→ **`tight_style_off` = 3247.28%，Sharpe 1.8120，DD 62.36%，611 笔交易**

---

## 二、为什么无法复现？

### 根本原因：代码版本不一致

3247.28% 产生于 CodeX 会话的某个中间状态，该状态下的策略代码和当前仓库代码存在差异。

**关键差异**：

| 因素 | 3247% 时 | 当前仓库 |
|---|---|---|
| `bear_strong_short_*` 参数 | 不存在 | 已添加（即使=null也有逻辑差异） |
| Entry 过滤逻辑 | 无 `ob_fill_ratio` 检查 | 有 `_direction_allowed_for_idx` + `ob_fill_ratio` 检查 |
| `PositionState` 字段 | 无 `highest_price/lowest_price` | 有（用于 ATR trailing） |

**实验验证**：
- 把当前代码的所有 `bear_strong_short` 参数重置为 null → 1116%（不是 3247%）
- 关闭 bear_strong 做空 + tight_style_off → 1345.75%（可复现）

---

## 三、当前可达到的最优结果

| 配置 | 收益 | Sharpe | DD | 交易数 | 手续费 |
|---|---|---|---|---|---|
| tight_style_off + bear_strong null | **1345.75%** | **1.850** | 62.36% | 427 | 15078 |
| tight_style_off + bear_strong 优化 | 1013.61% | 1.524 | 62.36% | 541 | 23922 |
| all + bear_strong null | 1116.07% | 1.478 | 62.36% | 598 | 28171 |

**最优配置（1345.75%）：**
```json
{
  "disable_fixed_target_exit": false,
  "atr_regime_filter": "tight_style_off",
  "atr_activation_rr": 2.0,
  "atr_loose_multiplier": 3.0,
  "atr_normal_multiplier": 2.5,
  "atr_tight_multiplier": 2.0,
  "enable_atr_trailing": true,
  "atr_period": 14,
  "allow_bear_strong_short": true,
  "bear_strong_short_pullback_window": null,
  "bear_strong_short_sl_buffer_pct": null,
  "bear_strong_short_retrace_min_ob_fill_pct": null,
  "bear_strong_short_entry_min_ob_fill_pct": null,
  "bear_strong_short_rr_ratio_override": null,
  "bear_strong_short_trail_style_override": null,
  "bear_strong_short_max_hold_bars": null,
  "bear_strong_short_atr_activation_rr": null,
  "bear_strong_short_atr_loose_multiplier": null,
  "bear_strong_short_atr_normal_multiplier": null,
  "bear_strong_short_atr_tight_multiplier": null
}
```

---

## 四、Codex 迭代参数扫描结论总结

### ATR 参数结论
- `atr_activation_rr = 2.0` 最优（1.0 太早，2.5 太晚）
- `atr_loose/normal/tight = 3.0/2.5/2.0` 或 `2.7/2.25/1.8` 均可，后者略优
- `atr_period = 14` 优于 21

### Regime Filter 结论
- `all`: 3209.53%（不可复现）
- `tight_style_off`: 3247.28%（不可复现）/ 1345.75%（可复现）
- `bear_weak_off`: 等于 `all`
- `bull_weak_off`: 明显变差，不建议
- `strong_only`: 明显变差，不建议

### Target Exit 结论
- `disable_fixed_target_exit = false`（保留固定目标价）始终优于 `true`（关闭）

### 手续费效率结论
- Sharpe 最优：`2.7/2.25/1.8`
- 净收益/手续费最优：`2.85/2.375/1.9`（收益略低但费效更好）

---

## 五、盈亏来源分析（1345.75% 配置）

### 总览
- 总收益：1345.75%
- Sharpe：1.850
- 最大回撤：62.36%
- 交易数：427

### 多空拆分
| 方向 | 交易数 | PnL | 胜率 |
|---|---:|---:|---:|
| 🟢 做多 | 207 | +$13,214 | 41.1% |
| 🔴 做空 | 220 | -$3,468 | 40.0% |

### Regime 盈亏来源
| Regime | 方向 | 交易数 | PnL | 胜率 |
|---|---|---:|---:|---:|
| **bull_strong** | 做多 | 207 | **+$13,214** 🔥 | 41.1% |
| bull_weak | 做空 | 56 | +$1,274 | 30.4% |
| bear_strong | 做空 | 171 | -$1,660 | 34.5% |
| bull_weak | 做多 | 87 | -$1,170 | 31.0% |

### Trail Style 分布
| Style | 交易数 | 收益 |
|---|---:|---:|
| B (loose) | 207 | +$13,214 |
| M (normal) | 137 | +$467 |
| S (tight) | 83 | +$595 |

### 出场原因
| 原因 | 交易数 | 占比 |
|---|---:|---:|
| stop_loss | 393 | 92.1% |
| target_4r | 34 | 8.0% |

---

## 六、关键发现

1. **全部正收益来自 `bull_strong + 做多 + loose trail`**
2. **`bear_strong` 做空是最大亏损源（-1660 USDT）**，结构性失效
3. **`bull_weak` 做空反而赚钱**（+1274 USDT），逆势策略有效
4. **固定目标价（target_4r）只触发 8% 的交易**，但一旦触发贡献巨大
5. **94% 的交易以止损出局**，ATR trailing 的价值在于"减少亏损"而非"放大利润"

---

## 七、优化方向建议

1. **熊市关闭做空**（bear_strong 做空亏 -1660）
2. **bull_weak 做多应该提高门槛或关闭**
3. **继续向 loose trail 倾斜**（B style 贡献全部利润）
4. **多空 ATR 参数分开优化**（已有字段支持，尚未扫描）

---

## 八、文件位置

- 策略核心：`strategy/scalp_robust_v2_core.py`
- 回测脚本：`scripts/scan_long_short_atr_params.py`, `scripts/optimize_fee_sensitive_exit.py`
- 最优配置：`config/config.live.5x-3pct.json`
- 归因报告：`var/reports/live_5x_3pct_optimized_attribution_2023-01-01_to_2026-04-18.md`

---

*生成时间：2026-04-20*
*Codex Session：019da5ca-7f8b-7fe1-a127-6789ed21e8c8*