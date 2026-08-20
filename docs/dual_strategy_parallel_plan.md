# 双策略并行架构实施方案（BTC 剥头皮 30% + 趋势 70%）

> 状态：设计定稿，未部署。所有回测数字见正文，均含 funding/fee/滑点。
> 日期：2026-08。

## 1. 目标架构

```
账户权益（OKX，共享）
├── 30% → 策略A：BTC 剥头皮（独立进程，SMC/ICT，动态杠杆，固定 0.30 仓位）
└── 70% → 策略B：趋势 router（GOOGL + QQQ + 黄金，内部赢家通吃）
```

两个进程**各自独立结算、独立止损、互不抢仓位**——这是多策略组合（fund 架构），替代现在的"单 router 赢家通吃"。

## 2. 资金分配依据（30/70）

BTC 剥头皮（w） vs 趋势（1-w）分配扫（2024-05→2026-05，日频再平衡）：

| BTC 占比 | 组合收益 | maxDD | 夏普 |
|---|---|---|---|
| 0% | +8488% | 65.5% | 1.76 |
| 25% | +5744% | 52.7% | 1.86 |
| **30%（选定）** | **+5119%** | **50.0%** | **1.89** |
| 40% | +3917% | 44.4% | 1.92 |
| 50%（夏普最优） | +2848% | 39.0% | 1.93 |

**选定 30/70 的理由**：收益 50 倍级（+5119%）、maxDD 50%（可控）、夏普 1.89（接近峰值）。
25/75 偏收益（+5744%/52.7% DD），50/50 偏稳健（+2848%/39% DD）。BTC 的价值是**降回撤分散化**，不是赚收益。

## 3. 三条腿的最终形态

| 腿 | 策略 | 杠杆 | 打分 | 归属 |
|---|---|---|---|---|
| BTC | SMC/ICT 剥头皮（现有，不改） | 10x 动态 | 不参与打分 | 策略A（独立 30%） |
| GOOGL | 单均线破位趋势（现有，不改） | 11.2x offense / 7.5x base | 107.8 | 策略B（趋势 70%） |
| QQQ | TQQQ 趋势（现有，不改） | 10x | ≥98 | 策略B |
| 黄金 | **MA50>MA100 金叉 + 5% trailing**（新增） | **4x** | **50** | 策略B |

黄金参数依据：GC=F 2010-2026 回测，MA 交叉远优于单均线（+1837% vs +288%，4x）；XAU-USDT-SWAP 499 天核验 +273%/48.7%（50/100 4x）。打分 50 < GOOGL 107.8 → 只填空仓、不抢 GOOGL alpha。

## 4. 具体改动清单

### 4.1 资金分配（config 键值）

> ⚠️ **代码 review 修正**：BTC 的 live 下单 sizing（`okx_executor.py:4500-4515`）是**风险预算驱动**（`risk_per_trade + stop_distance` 倒推仓位），`position_size_pct` 只是上限、不是实际分配。所以 BTC 的 30% 分配**不能用 position_size_pct**，要用 `fixed_notional_usdt` 钉死名义。

**BTC 独立 bot**（`config/config.live.high-leverage-structure.json`）：
- `fixed_notional_usdt`：设为「当前权益 × 0.30 × 10」（例：$13,000 权益 → $39,000），**每月手动更新一次**（固定名义不随权益自动扩）。
- 不动 `position_size_pct` / `risk_per_trade`（它们决定风险预算，改了会破坏 BTC 策略盈亏结构）。

**趋势 router**（`config/config.live.strategy-router.json`，gitignored）：
- `qqq_position_size_pct`：`1.0` → `0.70`
- `googl_position_size_pct`：`1.0` → `0.70`
- `gold_position_size_pct`：新增 `0.70`
- `enable_btc`：保持 `false`（BTC 不再走 router 切换）
- `strategy_priority`：移除 `btc_sota`，加入 `gold_*`

**为什么这样不互相挤占**：BTC 用固定名义（不碰 available 余额），趋势用 `total_equity`（含已锁保证金）——两个进程的 sizing 互不依赖对方的锁仓状态，没有竞态。

### 4.2 BTC 独立化

- 入口：`bot/run_bot.py`（已存在），指向 `high-leverage-structure` config，独立 systemd 服务。
- BTC 的 `okx_executor.py` / `btc_signal_adapter.py` / 动态杠杆 / shadow gate 全部复用，无需改策略逻辑。
- 唯一变化：从「router 切换选中才开仓」→「独立进程、固定 0.30 仓位、自己开平」。

### 4.3 黄金腿（新增，工作量最大）

- 新增 `bot/gold_usdt_executor.py`（复用 GOOGL 的爬坡/止损/资金费框架）。
- 新增 `config/config.paper.gold-trend-runtime.json`：入场 MA50>MA100、止损 5%、杠杆 4x。
- 新增黄金信号生成（日线 MA 交叉，输出 gold_daily_signal.csv，格式对齐 GOOGL）。
- `strategy_router.py` 加 `gold_*` 字段；`router_executor.py` 加 GOLD 分支 + pre-switch 守卫（对齐 GOOGL 的 risk window / shadow gate）。

### 4.4 数据

- 黄金日线：GC=F（长历史，已本地 `data/public/macro` 区）+ XAU-USDT-SWAP（OKX 永续，实盘执行用，历史仅 499 天）。
- 黄金 4h：XAU-USDT-SWAP 4h K 线（执行层回测用，需服务器拉取，同 GOOGL 4h 路径）。

## 5. 分阶段部署顺序（先 shadow 后实盘）

| 阶段 | 内容 | 风险 | 验证标准 |
|---|---|---|---|
| 0 | 全部改动本地 + 单元测试 | 零 | pytest 全绿 |
| 1 | **BTC 独立化 + 30/70 分配**（趋势内部不动） | 低 | BTC 独立开平仓正常、QQQ/GOOGL 仓位变 70% |
| 2 | **黄金腿 shadow**（只观察不开仓） | 零 | shadow 候选正常、打分 50、MA 交叉信号正确 |
| 3 | 黄金腿翻实盘（打分 50 填空仓） | 中 | 用 XAU-USDT-SWAP 实盘数据核验参数 |
| 4 | 观察组合回撤，微调 30/70 | 低 | 组合 maxDD 是否落在 39~53% 预期区间 |

## 6. 风险清单

1. **组合回撤仍会 >50%**（30/70 档 maxDD 50%）：趋势含黄金 4x 的 65.5% DD，BTC 分散只摊到 50%。
2. **BTC 曾有过切换失败事故**（已修 Fix A/B，107 测试绿），独立化后要重新验证 BTC 独立开平仓。
3. **黄金 4x 是高杠杆腿**（16 年 maxDD 82%），不是"低风险分散"，分散的是相关性不是风险。
4. **XAU-USDT-SWAP 历史短（499 天）**：黄金参数用 GC=F 定档，实盘前必须等 XAU 数据复核。
5. **押注黄金牛市继续**：黄金腿 +1269%（2.6 年）的前提是黄金延续 2024-2026 的 +110% 行情。
6. **两进程共享同一 OKX 账户**：需确认保证金隔离（isolated）与资金分配不互相挤占。

## 7. 一句话总结

**拆成「BTC 剥头皮 30%（独立 bot）+ GOOGL/QQQ/黄金趋势 70%（router）」两个并行进程。** 黄金腿 = MA50>MA100 + 5% stop + 4x + 打分 50。组合预期 +5119% / 50% DD / 夏普 1.89。先 BTC 独立化（低风险），再黄金 shadow → 实盘。
