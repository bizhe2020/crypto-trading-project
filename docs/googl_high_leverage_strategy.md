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
   ├── fast_ma(20) > slow_ma(60) → entry_signal
   ├── SPY 日线趋势 → 市场 regime（spy_up / spy_down）
   └── GOOGL 回撤 trailing → hard_exit（跌破峰值*(1-trail)）
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

**段 1 pre-conviction（2007-06 → 2025-11-13，熊市保护模块）**
- `entry_signal` = `fast_ma(20) > slow_ma(60)`（慢交叉，低换手）
- regime：`SPY 收盘 > SPY 200日均线`
- trailing hard exit 15%（收盘跌破 10 日峰值 × 0.85）→ FLAT
- `max_hold_days=90`（最长持仓限制）

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
1. GOOGL 是持续复利型资产：buy&hold 14.6% CAGR / maxDD 65.3%。趋势过滤器大幅降低收益（342% vs 1270%），但把 maxDD 砍半（33.4% vs 65.3%）。
2. 对高倍杠杆，maxDD 管理就是一切：恒定 15x 无执行止损直接清零。
3. `close>ma20` 入场在 conviction 段表现最佳（+23.3% vs buy&hold +25.4%，maxDD 14.3% 更低），慢交叉太慢导致 2026-06 回调后错过反弹。

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

## 5. 回测结果（v0.1，2026-08-15）

运行：`python scripts/backtest_googl_high_leverage.py --prices-csv ... --holdings-csv ...`
完整报告：`var/reports/googl_high_leverage_backtest_2007_2026.json`

| 指标 | 信号层（两段式） | buy&hold GOOGL |
|---|---|---|
| 累计收益 | +342.6% | +1270.5% |
| CAGR | 8.1% | 14.6% |
| maxDD | 33.4% | 65.3% |
| 交易次数 | 61 | — |

**conviction 段（2025-11-14 → 2026-08-14）**
- GOOGL buy&hold +25.4%（maxDD 21.1%）
- 策略在市 103/187 天（55.1%），捕获 +23.3%，maxDD 14.3%

**杠杆敏感性（worst-case 恒定满杠杆，无执行层止损）**
- 恒定 15/10/5 倍无止损：-100%（清零，单日爆仓 0 天，序列亏损耗尽）
- stop-protected 4%（单日 clamp）：-80.8%
- **结论：日线信号层不能直接恒定满杠杆持有。真实系统必须在 4h 执行层用 trailing stop + 交易所条件单 + shadow gate 保护（单笔亏损 ~4%、止损后保持 flat 等重入场）——这就是现有 QQQ 系统的执行层，GOOGL 直接复用。杠杆列是理论下界，不是预期表现。**

**组件归因（2015-2026 全历史，max_hold 关闭）**
| 变体 | 收益 | maxDD | 在市率 |
|---|---|---|---|
| buy&hold（全开） | 1263% | 65.3% | 100% |
| 纯 SPY regime | 681% | 58.4% | 75.7% |
| 纯 close>ma20 | 340% | 59.4% | 63.5% |
| entry+SPY+trail15 | 168% | 59.1% | 53.4% |

→ GOOGL 自身 MA 交叉 filter 是最大收益拖累（趋势复利资产）；SPY regime 的 maxDD 管理价值大于收益价值；trailing exit 影响有限。

## 6. 分支落地清单

- [x] 新分支 `feature/googl-high-leverage-strategy`
- [x] `config/config.paper.googl-high-leverage-frozen.json`（信号源配置）
- [x] `config/config.paper.googl-high-leverage-runtime.json`（运行时配置，镜像 QQQ runtime）
- [x] `scripts/scan_googl_daily_signal.py`（日线信号生成，两段式）
- [x] `bot/googl_usdt_signal_adapter.py`（信号适配器）
- [x] `scripts/backtest_googl_high_leverage.py`（回测 + 参数扫描）
- [x] `tests/test_googl_daily_signal.py`（单元测试，8 个）
- [ ] 路由接入（第二阶段，可选）
- [ ] 4h 执行层回测（GOOGL-USDT-SWAP 2026-03 起）
- [ ] 实时信号定时刷新（cron 集成）

## 7. 风险与未决问题

- **OKX 合约历史短**：GOOGL-USDT-SWAP 2026-03-04 上线，4h 执行层回测样本有限，先以日线信号验证为主。
- **单票高倍风险**：GOOGL 财报跳空（2025-10 曾单日 -5% 级别），高倍杠杆下需靠 liquidation buffer + shadow gate 防御，考虑财报日降档。
- **趋势过滤器 vs buy&hold 的张力**：GOOGL 是持续复利资产，趋势过滤器把收益从 1270% 降到 343%（maxDD 从 65% 降到 33%）。对高倍杠杆这是必要代价，但收益代价大，需在 v2 用执行层保护 + defense 档优化。
- **13F 滞后**：13F 每季度披露且滞后约 45 天，信念信号是慢变量，只作长期开关，不作择时。
- **杠杆模拟是下界**：恒定满杠杆回测清零，真实行为依赖 4h 执行层（stop-and-stay-flat），4h 执行层回测是第二阶段必做项。
- **做空未纳入**：GOOGL 单票做空与做多逻辑不同，首版只做多。
- **20x 上限**：合约支持 20x，但设计默认 offense 15x、base 10x，具体以回测清算风险为准。
