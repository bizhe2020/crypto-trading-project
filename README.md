# Crypto Trading Project

这是一个基于 `freqtrade 2026.3` 的本地量化交易工作目录，当前重点放在 `1h` 主周期、`4h` 趋势过滤的中频现货策略迭代，也包含 OKX 合约模式的做空回测配置。

## 当前结构

- `freqtrade_data/config.json`: 默认配置，已显式绑定 `user_data/data` 数据目录。
- `freqtrade_data/config-futures.json`: OKX 合约做空回测配置，默认策略为 `V8V70_Short`。
- `user_data/strategies/base_trend_strategy.py`: 三套趋势策略共享的指标与信号基类。
- `user_data/strategies/adx_di_confirm_strategy.py`: 在基础版本上增加 DI 趋势确认。
- `user_data/strategies/pattern_confirm_strategy.py`: 使用 K 线形态确认回踩。
- `user_data/strategies/pattern_di_hybrid_strategy.py`: 形态 + 轻量 DI 的混合版本。
- `user_data/strategies/hybrid_trend_guard_strategy.py`: 当前更稳健的混合优化版，增加高周期趋势保护和更快的失败退出。
- `user_data/download_binance_klines.py`: 按需导出 Binance JSON 数据的辅助脚本。

## 沉淀索引

### 策略库

| 策略 | 状态 | 核心思路 | 回测摘要 | 文档 |
| --- | --- | --- | --- | --- |
| `V8V70_BTCOnlyReclaim` | 已沉淀 | BTC 仅保留 reclaim，ETH/SOL 保留 dual-template | 39 笔 / 14.02% / DD 2.07% / 胜率 64.1% | `strategies_library/V8V70_BTCOnlyReclaim.md` |

- 策略总索引：`strategies_library/README.md`

### 因子库

- 因子总索引：`factor_library/README.md`
- 当前沉淀文档：`factor_library/pullback_reclaim_confirmation.md`

## 常用命令

列出策略：

```bash
./.venv312/bin/freqtrade list-strategies
```

回测单个策略：

```bash
./.venv312/bin/freqtrade backtesting \
  -c freqtrade_data/config.json \
  --strategy PatternDiHybridStrategy
```

回测当前优化版本：

```bash
./.venv312/bin/freqtrade backtesting \
  -c freqtrade_data/config.json \
  --strategy HybridTrendGuardStrategy
```

查看最近一次回测：

```bash
./.venv312/bin/freqtrade backtesting-show \
  -c freqtrade_data/config.json
```

合约模式回测做空策略：

```bash
./.venv312/bin/freqtrade backtesting \
  -c freqtrade_data/config-futures.json \
  --strategy V8V70_Short \
  --timerange 20230101-20260404
```

## 数据说明

- 当前项目的 `1h` 和 `4h` 回测数据在 `user_data/data` 下，格式为 `feather`。
- `5m` 的 OKX 数据在 `user_data/data/okx` 下，可用于后续降周期实验。
- OKX 合约回测数据应位于 `user_data/data/okx/futures`；当前已将 `user_data/data/futures` 下的 ETH/SOL 合约数据链接到该目录，供 `config-futures.json` 直接使用。
- 当前 `funding_rate` 数据从 `2026-01-01` 开始；回测更早区间时会有缺失告警，但不影响策略和做空模式验证。
- 如果需要额外拉取 Binance JSON 数据，可按需生成到 `user_data/data/binance`：

```bash
./.venv312/bin/python user_data/download_binance_klines.py --symbol BTCUSDT --pair BTC/USDT --interval 5m --days 30
```

## 清理约定

- `user_data/backtest_results`、`user_data/hyperopt_results`、`user_data/logs`、`user_data/plot` 都视为运行产物，不保留在仓库中。
- `__pycache__`、`.DS_Store`、本地虚拟环境和临时 notebook 默认忽略，避免仓库继续变脏。

## 迭代建议

- 先固定交易对和时间范围，对三套策略做同区间回测比较。
- 再进入 `hyperopt` 或参数网格搜索，避免先在结构混乱时调参。
- 回测后优先看 `enter_tag/exit_tag` 分布，确认信号质量，再决定是否继续加过滤条件。
