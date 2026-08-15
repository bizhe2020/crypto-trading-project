# GOOGL 高倍合约策略 — 研究与设计

> 分支：`feature/googl-high-leverage-strategy`
> 目标：在 OKX GOOGL-USDT-SWAP 上做 10x+ 高倍做多策略，复用现有 QQQ/BTC 高倍合约框架（信号生成 → 风控叠加 → 执行 → 路由），并引入价值投资项目（伯克希尔 13F）的长期信念过滤。

## 1. 动机与前提

- 现有 `qqq_usdt_aggressive` 是 10x 杠杆的 QQQ-USDT-SWAP 做多策略，已稳定运行。
- 用户持有价值投资数据（伯克希尔 13F 持仓、全历史日线价格），想把它和现有高倍合约执行框架结合，做一个 GOOGL 合约策略。
- 伯克希尔自 **2025Q3 起建仓 Alphabet**，到 2026Q2 累计约 1.06 亿股（市值约 378 亿美元，含 GOOG/GOOGL 两个份额）——这是很强的长期价值信念信号。

## 2. 数据清点

| 数据 | 来源 | 覆盖 | 用途 |
|---|---|---|---|
| GOOGL 日线（前复权） | `价值投资project/data/prices.csv` | 2007-03 → 2026-08 | 日线信号 |
| SPY 日线（前复权） | 同上 | 同范围 | 市场 regime 过滤 |
| 伯克希尔 13F | `价值投资project/data/berkshire_13f_holdings.csv` | 2012Q4 → 2026Q2 | 长期信念过滤 |
| GOOGL-USDT-SWAP 4h K线 | OKX（`marketHistoryCandles`） | 2026-03-04 → 至今 | 4h 执行/止损 |
| FRED 宏观（美元指数） | 现有 `data/public/macro/fred_macro-1d.feather` | 现有 | macro 覆盖 |

### 2.1 关键约束：OKX GOOGL 合约仅 5 个月历史

`GOOGL-USDT-SWAP` 2026-03-04 上线，OKX 侧最多只有 ~5 个月 4h 数据。这决定了：

- **日线信号可以用 2007-2026 完整历史回测**（价值项目日线）。
- **4h 执行层只能回测 2026-03 之后**，或以后台验证为主。
- 策略结构必须保持"日线决策 + 4h 执行"解耦，便于分别验证。

### 2.2 OKX 合约规格（实测）

- 最大杠杆：**20x**（逐仓）；当前账户设置 3x，需在开仓时设置目标杠杆。
- ctVal=1（1 张 = 1 GOOGL 股），tickSz=0.01，lotSz=0.01。
- 24h 成交 ~758 万 USDT，价差 ~0.003%，全本金 10x（≈12.5 万 USDT）市价滑点约 0.065% —— 流动性足够。

## 3. 现有框架复用映射

| 现有组件 | 文件 | GOOGL 策略复用方式 |
|---|---|---|
| 日线信号 → 持仓序列 | `scan_qqq_usdt_4h_triggers.py::load_signal_path / attach_daily_state` | 新写 `scan_googl_daily_signal.py`，产出 `date,position` 序列 |
| 信号适配器 | `bot/qqq_usdt_signal_adapter.py` | 新写 `bot/googl_usdt_signal_adapter.py`，产出 `RoutedSignalCandidate` |
| 执行器（10x、trailing stop、rebalance） | `bot/qqq_usdt_executor.py` | 参数化复用（symbol/杠杆配置换成 GOOGL） |
| 高倍杠杆保护 | `bot/okx_executor.py::_high_leverage_guard*`、`_dynamic_high_leverage*` | 直接复用：清算缓冲 ≥1.2%、止损距离上限、动态杠杆 |
| 风控叠加 | `qqq_shadow_gate`、`qqq_macro_proxy_overlay` | 直接复用（shadow gate 与标的无关，macro 用 FRED 数据） |
| 路由 | `bot/strategy_router.py`、`bot/router_executor.py` | 后续加入 googl 候选，与 btc/qqq 三选 |

## 4. 策略设计

### 4.1 分层结构

```
价值投资信念（季度/长期）
   └── 伯克希尔 13F 持有 GOOGL？ → 信念开关（conviction_on）
                │
GOOGL 日线趋势（日频）
   ├── GOOGL 收盘 > ma(60) → entry_signal（pre 段慢 MA）
   ├── GOOGL 收盘 > ma(20) → 快速重入场（conviction 段）
   ├── 无 SPY regime 过滤（慢 MA 自含回撤管理）
   └── GOOGL 回撤 trailing(10%) → hard_exit（跌破 10 日峰值*(1-0.10)）
                │
        每日 position 序列（GOOGL / FLAT）
                │
OKX GOOGL 4h 执行（4h）
   ├── enrich_bars 入场节奏（breakout / pullback）
   ├── 杠杆档位：conviction+regime → offense(15-20x) / base(10x) / defense(5x)
   ├── ATR/峰值 trailing stop
   └── 高倍保护：清算缓冲 ≥1.2%、止损距离 ≤2%、动态高杠杆
```

### 4.2 信号层（日频，已实现的两段式）

**段 1 pre-conviction（2007-06 → 2025-11-13，熊市保护模块）**（v0.2 优化）
- `entry_signal` = `GOOGL 收盘 > ma(60)`（慢 MA 入场，非 20/60 交叉）
- regime：**无 SPY 过滤**（慢 MA 自含回撤管理，回测更优）
- trailing hard exit 10%（收盘跌破 10 日峰值 × 0.90）→ FLAT
- `max_hold_days=0`（无时间限制）

**段 2 conviction（2025-11-14 起，信念做多模块）**
- `entry_signal` = `GOOGL 收盘 > fast_ma(20)`（快入场，信念段快速重入场）
- regime **放宽**：conviction 替代 SPY 过滤（信念 = 长期看多，SPY 弱时仍允许做多）
- trailing / max_hold **关闭**（信念穿透回调）
- 资本从段 1 连续滚入段 2（`initial_capital = 段1最终资本`）

**伯克希尔信念（conviction_on）**
- `conviction_on[t]` = 最新已披露 13F（`filing_date ≤ t`）持有 ALPHABET（shares > 0）
- 用 filing_date（披露日）而非 report_date，避免前视偏差
- 实际数据：2025-11-14 首次披露 → 至今
- 信念开启时：杠杆档位升 offense（15x）、放宽 regime 过滤

**杠杆档（执行层参考，信号层输出 leverage_tier）**
| tier | 触发 | 目标杠杆 |
|---|---|---|
| offense | conviction 且在市 | 15x |
| base | 非信念在市 | 10x |
| defense | 预留（conviction 深跌降档，v2） | 5x |
| flat | FLAT | 0 |

**关键研究发现（2026-08 实测，详见第 5 节）**
1. GOOGL 是持续复利型资产：buy&hold 14.6% CAGR / maxDD 65.3%。**慢 MA 入场（close>ma60）反而同时提升收益并降低回撤**——慢 MA 的滞后不对称性（下跌时 MA 挂在更高位置 → 提前退出；反弹时 MA 挂在更低位置 → 提前进场）。2008 只 -8.9%（buy&hold -55%）、2022 只 -22.1%（buy&hold -39%）。
2. **fast(20)>slow(60) 交叉入场是最大收益拖累**：v0.2 改为 close>ma60 + 10% trailing + 去 SPY regime + 去 max_hold，信号层 342.6% → 1951.0%，maxDD 33.4% → 36.9%（+3.5pp），conviction 段行为不变。
3. 对高倍杠杆，maxDD 管理就是一切：恒定 15x 无执行止损直接清零。
4. `close>ma20` 入场在 conviction 段表现最佳（+23.3% vs buy&hold +25.4%，maxDD 14.3% 更低），慢交叉太慢导致 2026-06 回调后错过反弹。

### 4.3 杠杆结构（高倍）

| 档位 | 触发 | 目标杠杆 |
|---|---|---|
| offense | conviction_on + googl 趋势上 + SPY up | **15x**（上限 20x 受清算缓冲约束） |
| base | googl 趋势上 | **10x** |
| defense | 回撤/信念弱 | **5x** |
| flat | 信号翻空 | 0 |

高倍保护（复用现有逻辑）：
- 清算缓冲 ≥ 1.2%（`high_leverage_min_liquidation_buffer_pct`）
- 止损距离 ≤ 2%（`high_leverage_max_stop_distance_pct`）
- 动态有效杠杆 ≤ 20（`dynamic_max_effective_leverage`）
- shadow gate：连续亏损/回撤自动降杠杆或暂停

### 4.4 执行层（4h，复用 QQQ executor）

- 4h K线来自 OKX `GOOGL-USDT-SWAP`
- 入场节奏：`enrich_bars` 的 breakout（突破 12 根高点）/ pullback（回踩 20 均线企稳）——与 QQQ 相同模式，在 GOOGL 4h 上计算
- Trailing stop：峰值 × 0.96（4%）随 4h 收盘 ratchet，ATR 加速止损；交易所条件单兜底
- 只做多（与 QQQ aggressive 一致），先不做空（GOOGL 单票做空逻辑不同）

### 4.5 路由接入（后续）

`strategy_router` 增加 `googl_usdt_aggressive` 候选：
- `googl_min_route_score` 阈值（如 96）
- 与 btc_sota / qqq_usdt_aggressive 三选一
- 切换前复用 Fix A（`pre_switch_open_confirm`）与 Fix B（回滚查交易所）防护

## 5. 回测结果（v0.2 优化，2026-08-15）

运行：`python scripts/backtest_googl_high_leverage.py --prices-csv ... --holdings-csv ...`
完整报告：`var/reports/googl_high_leverage_backtest_2007_2026.json`

| 指标 | 信号层（两段式，v0.2） | v0.1（fast20>slow60+SPY200） | buy&hold GOOGL |
|---|---|---|---|
| 累计收益 | **+1951.0%** | +342.6% | +1270.5% |
| CAGR | **17.1%** | 8.1% | 14.6% |
| maxDD | 36.9% | 33.4% | 65.3% |
| 交易次数 | 127 | 61 | — |

**conviction 段（2025-11-14 → 2026-08-14）**
- GOOGL buy&hold +25.4%（maxDD 21.1%）
- 策略在市 103/187 天（55.1%），捕获 +23.3%，maxDD 14.3%
- conviction 段行为与 v0.1 完全一致（优化只在 pre 段）

**逐年收益（信号层）**
2008 -3.6%｜2009 +80.8%｜2013 +42.8%｜2017 +25.4%｜2018 +1.2%｜2020 +17.1%｜2021 +32.9%｜**2022 -22.1%**｜2023 +20.0%｜2024 +30.7%｜2025 +76.4%｜2026 +16.5%

**参数优化前沿（2026-08-15 全历史扫描，conviction 段固定 close>ma20）**
| pre 段入场 | pre regime | pre trailing | 收益 | maxDD |
|---|---|---|---|---|
| close>ma60 | 无 | 10% | **+1951%** | **36.9%** |
| close>ma60 | 无 | 0% | +1735% | 37.4% |
| close>ma60 | SPY>50 | 0% | +1258% | 32.7% |
| close>ma50 | 无 | 0% | +1067% | 48.1% |
| close>ma60 | SPY>200 | 0% | +892% | 31.7% |
| close>ma20 | SPY>50 | 0% | +802% | 21.4% |
| fast20>slow60+SPY200 | 15% | (v0.1) | +343% | 33.4% |

→ 慢 MA（60 日）比 50 日更优：下跌时 MA 挂在更高位提前退出、反弹时挂在更低位提前进场（滞后不对称性）。加 SPY regime 把 maxDD 压到 ~32% 但砍掉 40-70% 收益——对"最高收益率"目标，去掉 regime 更优。trailing 10% 在 close>ma60 上又提升 ~220pp（早止损避开深跌且重入场点不变）。

**杠杆敏感性（worst-case 恒定满杠杆，无执行层止损）**
- 恒定 15/10/5 倍无止损：-100%（清零，单日爆仓 0 天，序列亏损耗尽）
- stop-protected 4%（单日 clamp）：+8627%（但 maxDD 99.7%，是"次日满仓重暴露"的粗糙模拟）
- **结论：日线信号层不能直接恒定满杠杆持有。真实系统必须在 4h 执行层用 trailing stop + 交易所条件单 + shadow gate 保护（单笔亏损 ~4%、止损后保持 flat 等重入场）——这就是现有 QQQ 系统的执行层，GOOGL 直接复用。杠杆列是理论下界，不是预期表现。**

### 5.1 杠杆可行性前沿（2026-08-15）

**日频逐笔模型（无日内保护）**：1x CAGR 18.4%（maxDD 33.8%）→ 2x 34.5%/58% → 3x 46.7%/75% → 5x 53.8%/92.5% → **10x -100%（清零）**。
**结论：原配置 10x/15x 在日频信号上不可行，3-5x 是收益/回撤甜点。**

**conviction 段单独看（2025-11-14 起，1x maxDD 仅 10.5%）**：
| 杠杆 | 9个月收益 | 年化 | maxDD |
|---|---|---|---|
| 3x | +116% | 180% | 30.5% |
| 5x | +198% | 332% | 48.6% |
| 10x | +253% | 441% | 81.9% |
| 15x | +49% | 71% | 96.4%（接近清零） |

→ 信念让 conviction 段平滑（maxDD 10.5% vs 全历史 33.8%），支撑 3-5x 甚至 10x；但 15x 仍不可行。**杠杆动机在 conviction 段成立，前提是把 offense 从 15x 降到 5x 左右。**

**4h 执行层回测**：`scripts/replay_googl_usdt_4h.py`（日线信号 + 4h trailing stop + fee/slippage×杠杆 + funding + shadow gate）让 10x 在 conviction 段不再清零（合成数据 +941%/maxDD 71%，真实数据待用户捞取）。日频模型看不到的日内止损正是 10x 存活的钥匙——但最终杠杆档位需真实 4h 数据验证。

**组件归因（2007-2026 全历史，v0.2 各组件贡献）**
| 变体 | 收益 | maxDD |
|---|---|---|
| buy&hold（全开） | 1270% | 65.3% |
| 纯 close>ma60 | 1735% | 37.4% |
| close>ma60 + 10% trailing | 1951% | 36.9% |

→ close>ma60 慢 MA 入场是收益与回撤的联合来源（既高于 buy&hold 又砍半 maxDD）；10% trailing 边际再贡献 ~220pp。SPY regime 对"最高收益"目标是净负贡献。

## 6. 分支落地清单

- [x] 新分支 `feature/googl-high-leverage-strategy`
- [x] `config/config.paper.googl-high-leverage-frozen.json`（信号源配置）
- [x] `config/config.paper.googl-high-leverage-runtime.json`（运行时配置，镜像 QQQ runtime）
- [x] `scripts/scan_googl_daily_signal.py`（日线信号生成，两段式）
- [x] `bot/googl_usdt_signal_adapter.py`（信号适配器）
- [x] `scripts/backtest_googl_high_leverage.py`（回测 + 参数扫描）
- [x] `tests/test_googl_daily_signal.py`（单元测试，8 个）
- [x] v0.2 参数优化：pre 段 close>ma60 + 10% trailing + 无 regime + 无 max_hold（342.6% → 1951.0%，maxDD 33.4% → 36.9%）
- [x] `scripts/replay_googl_usdt_4h.py`（4h 执行层回测：日线信号 + 4h trailing stop + fee/slippage×杠杆 + funding + shadow gate，`--sweep` 杠杆扫描）
- [x] `tests/test_googl_4h_execution.py`（4h 执行层单元测试，7 个）
- [ ] 路由接入（第二阶段，可选）
- [ ] 4h 执行层回测真实数据验证（GOOGL-USDT-SWAP 更长历史，用户捞取中）
- [ ] 实时信号定时刷新（cron 集成）

## 7. 风险与未决问题

- **OKX 合约历史短**：GOOGL-USDT-SWAP 2026-03-04 上线，4h 执行层回测样本有限，先以日线信号验证为主。
- **单票高倍风险**：GOOGL 财报跳空（2025-10 曾单日 -5% 级别），高倍杠杆下需靠 liquidation buffer + shadow gate 防御，考虑财报日降档。
- **趋势过滤器 vs buy&hold 的张力（v0.2 已解决）**：慢 MA 入场（close>ma60 + 10% trailing）同时提升收益并降低回撤（1951% vs 1270%，maxDD 36.9% vs 65.3%）——强复利资产上慢 MA 的滞后不对称性既有择时收益又有回撤保护。v0.1 的 fast20>slow60 交叉 + SPY regime 才是收益拖累，已弃用。
- **13F 滞后**：13F 每季度披露且滞后约 45 天，信念信号是慢变量，只作长期开关，不作择时。
- **杠杆模拟是下界**：恒定满杠杆回测清零，真实行为依赖 4h 执行层（stop-and-stay-flat），4h 执行层回测是第二阶段必做项。
- **做空未纳入**：GOOGL 单票做空与做多逻辑不同，首版只做多。
- **20x 上限 vs 可行性**：合约支持 20x，但回测（§5.1）表明日频信号上 10x 清零、15x 接近清零；conviction 段可行上限 ~10x、建议 offense 5x。**当前 config 的 15x/10x 档待 4h 真实数据验证后下调**，在 4h 执行层回测确认前不应按 15x 实盘。
