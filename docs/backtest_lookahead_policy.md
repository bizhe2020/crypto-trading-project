# 回测前视口径政策

> **本文件是全仓关于「前视偏差 / 多周期口径」的唯一权威说明。**
> 任何回测、复现、参数扫描、稳健性审计都必须遵守本文件；与本文件冲突的代码或注释以本文件为准。

---

## 1. 唯一规则

**每根 15m K 线只能使用「在它收盘那一刻已经完成」的 4h / 日线信息。**

工程上只有**一个**合法入口：

```python
from strategy.scalp_robust_v2_core import build_precomputed_state_confirmed_4h

precomputed = build_precomputed_state_confirmed_4h(c4h, c15m)
```

它会对每根 15m 的映射索引做 `confirmed_idx = mapped_idx - 1`，把结果写进 `*_for_15m` 数组，供引擎优先读取。

---

## 2. 为什么要有这条规则

`align_timeframes()` 的语义是「取 `ts <= 当前 15m 时间` 的 4h」：

```python
# strategy/scalp_robust_v2_core.py
while c4h_idx + 1 < len(c4h) and c4h[c4h_idx + 1].ts <= candle.ts:
    c4h_idx += 1        # ← 4h bar 的 ts 是「开盘时间」
```

4h bar 的 `ts` 是**开盘时间**，所以 `mapping[idx]` 指向的是**当时仍在形成中的那根 4h**。
而 `bias_4h[i]` / `regime_*[i]` / `trend_score_*[i]` 都使用该 bar 的**收盘价**计算
（例如 `precompute_4h_bias` 里 `cp = c4h[c4h_idx].c`）。

**后果**：回测会读到信号发出后才发生的 4h 收盘信息。

**实测代价**（BTC/USDT:USDT，2022-01-01 → 2026-09-20，同一份配置只切换口径）：

| 指标 | 有前视 | 无前视 | 差异 |
|---|---:|---:|---:|
| 总收益 | +8,525% | **+396%** | **−8,129 pp** |
| Sharpe | 2.62 | 1.62 | −1.00 |
| 最大回撤 | 80.6% | 92.0% | +11.4 pp |
| 止盈率 | 8.31% | 6.69% | −1.62 pp |

**回测默认口径曾把 4.7 年收益夸大 21 倍，同时低估回撤 11 个百分点。**

分歧率（同一段行情，两种口径下 4h 方向判断不一致的比例）：

| 指标 | 分歧率 |
|---|---:|
| 4h bias（方向） | **11.33%** |
| 4h bull/bear trend score | 11.3% / 11.9% |
| 1d regime flags | 2.7% – 4.0% |

---

## 3. 当前状态（三条路径全部无前视）

| 路径 | 口径 | 说明 |
|---|---|---|
| 实盘 `bot/okx_executor.py` | ✅ 无前视 | 硬编码 `build_precomputed_state_confirmed_4h` |
| 回放 `scripts/replay_sota_smc_live_shadow.py` | ✅ 无前视 | 口径已硬编码，无开关 |
| 报告 `scripts/live_readiness_report.py` | ✅ 无前视 | `load_prepared_data` 只剩一条路径 |
| `ScalpRobustEngine.from_candles` | ✅ 无前视 | 已改用 confirmed 版本 |

**收敛性佐证**：无前视口径下，回测与实盘的开仓时间差中位 **0.0 分钟**、开仓价差中位 **0.0 bp**。

---

## 4. 已删除的歧义来源（不得恢复）

| 已删除 | 类型 | 删除原因 |
|---|---|---|
| `--raw-4h-state` | CLI flag | 显式的有前视逃生通道 |
| `--confirmed-4h-only` | CLI flag | 默认已唯一，保留只会制造第二种口径 |
| `--informative-asof-from-15m` | CLI flag | 第三种口径，无实际使用者 |
| `load_prepared_data(confirmed_4h_only=...)` | 函数参数 | 三路分支本身就是歧义源 |
| `build_precomputed_state_asof_15m()` | 函数 | 无人使用的第二因果实现 |
| `build_precomputed_state()` | 函数 | **改名为 `_build_precomputed_state_lookahead()` 并私有化**，只能被 confirmed 版本内部调用 |
| `_asof_4h_series()` / `_asof_bias_from_fvgs()` / `_partial_regime_state()` / `_partial_trend_scores()` / `_partial_4h_candle()` | 函数 | 上一条删除后遗留的死代码（仅定义、无引用）。`_partial_4h_candle` 尤其危险 —— 名字听起来就是「未收盘 4h」 |

---

## 5. 禁止事项

1. ❌ **不要**直接调用 `_build_precomputed_state_lookahead()`（名字已自带警告）
2. ❌ **不要**把 `mapping[idx]` 与任何「按 4h 索引且含当前 bar 收盘价」的数组配合使用
3. ❌ **不要**为口径增加任何开关、flag、环境变量或配置键 —— 唯一口径是设计目标
4. ❌ **不要**在报告里引用未标注数据截止日的历史回测数字（基线随行情推进持续漂移）
5. ⚠️ 新增多周期特征时，**必须**同时提供 `*_for_15m` 因果数组，或证明其计算只依赖已收盘 bar

---

## 6. 如何验证

```bash
python3 scripts/audit_backtest_lookahead.py \
    --klines-15m var/tokyo_audit/btc_scalp_20260920/BTC_USDT_USDT-15m-2026.csv
```

输出包含：开仓当根是否判出场、止损/止盈判定顺序、成交价假设、三条路径的口径、
以及两种口径的分歧率。**修改任何多周期代码后都应重跑此脚本。**

回归测试：

```bash
python3 -m pytest -q     # 173 passed
```

---

## 7. 已知的合规实现（无需修改）

以下位置虽然**使用 `mapping[idx]` 直接索引**，但**不是前视**，因为它们的数据本身就是「已收盘日」口径：

| 位置 | 原因 |
|---|---|
| `precompute_daily_dev_pct()` | `completed = day_idx` 是严格早于当日的已收盘日数，SMA 与 close 都只取已完成日 |
| `_effective_regime_history()` | 返回 `self.c4h[:c4h_idx]`，切片不含当前那根未成收盘 4h |
| `report_smc_trade_context.completed_4h_idx_for_entry()` | 显式 `mapping[entry_idx] - 1` |
| `live_readiness_report.precompute_regime_state()` | `end_idx = c4h_idx - 1` |

**判据**：看计算是否排除了「当前 bar 自身的收盘价」，而不是看索引写法。

---

## 8. 数据权威来源

| 数据 | 路径 | 覆盖 |
|---|---|---|
| BTC 15m | 东京服务器 `data/okx/futures/BTC_USDT_USDT-15m-futures.feather` | 2020-01-01 → 至今（每日更新） |
| BTC 4h | 东京服务器 `data/okx/futures/BTC_USDT_USDT-4h-futures.feather` | 同上 |

本地副本见 `var/tokyo_audit/btc_scalp_20260920/`。**不要用本地 `data/okx/` 的旧副本回测到近期**（它停在 2026-05-29）。

---

*配套研究结论见 `docs/btc_scalp_research.md`。*
