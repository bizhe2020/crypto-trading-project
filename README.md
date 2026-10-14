# Crypto Trading Project

本仓库唯一的文档。实盘只有两条腿，跑在同一个 OKX 账户（单币种 USDT 保证金，`acctLv=2`，双向持仓）上：

| 服务（systemd） | 代码入口 | 交易品种 | 运行配置（服务器，含密钥，不进 git） | 模板 |
|---|---|---|---|---|
| `btc-scalp-standalone` | `bot/run_bot.py` → `bot/okx_executor.py` | BTC-USDT-SWAP，逐仓 20x，多空 | `config/config.live.high-leverage-structure.json` | `config/config.live.high-leverage-structure.template.json` |
| `trend-basket` | `bot/trend_basket_executor.py` | XAU-USDT-SWAP、QQQ-USDT-SWAP，逐仓 10x，只做多 | `config/config.live.trend-basket.json`（不含密钥，复用上一份的 API key） | `config/config.live.trend-basket.template.json` |

两者互不碰对方合约：scalp 只按 `symbol` 查自己的仓位，篮子在配置校验里禁止任何 BTC 合约。
旧的 router（BTC/QQQ/GOOGL/黄金切换）已停用、开机自启已关，代码已从主分支删除（见文末「归档」）。

## 1. 目录

```
bot/        实盘执行：okx_executor（scalp）、trend_basket_executor（篮子）、okx_client、state_store、market_data
strategy/   scalp_robust_v2_core（scalp 引擎）、trend_vote（篮子 5 规则投票，研究与实盘共用）
scripts/    scalp 回测/重放/审计模块（被 okx_executor 和研究脚本 import）+ fetch_public_etf_history
scripts/ops deploy_tokyo.sh（部署）、tokyo_status.sh（只读巡检）
research/time_in_market/  在场率 / 篮子研究脚本（第 1~5 步，见 §5）
systemd/    两个服务的 unit 文件（与服务器 /etc/systemd/system 一致）
tests/      pytest（python3 -m pytest -q）
```

## 2. 部署与运维（东京 `root@23.106.133.251`）

- 服务器项目目录：`/root/projects/crypto-trading-releases/router-risk-20260603_1a2a61f`（目录名是历史遗留，两个服务都跑在这里）。
- 日志：`var/log/btc_scalp_standalone.log`、`var/log/trend_basket.log`；状态库：`state/runtime_high_leverage_structure_live.db`、`state/trend_basket_live.db`。
- 巡检（只读）：`bash scripts/ops/tokyo_status.sh` —— 服务状态、账户权益、全部持仓（含强平价）、两边最近日志。
- 部署：提交后 `bash scripts/ops/deploy_tokyo.sh`。默认只拉代码 + 编译检查不重启；
  `RESTART="trend-basket"` 或 `RESTART="btc-scalp-standalone trend-basket"` 指定重启。
  每次都会在服务器 `var/backups/deploy_<时间>/` 备份两份配置和状态库。**重启 scalp 前先确认空仓。**
- 改篮子参数：编辑服务器 `config/config.live.trend-basket.json` →
  `systemctl stop trend-basket` → `.venv/bin/python bot/trend_basket_executor.py --config config/config.live.trend-basket.json --evaluate-once --force`（立即按新参数再平衡）→ `systemctl start trend-basket`。
- 停篮子但保留仓位：`systemctl stop trend-basket`；平仓需手动或把 `enabled`/投票清零后 `--force`。
- 服务器上已关闭：`crypto-strategy-router`（disabled）、`qqq-risk-refresh.timer`（disabled）。

## 3. 当前实盘参数与依据

### BTC scalp
- 引擎 `scalp_robust_v2`，日线乖离 alpha 门 **cap12**（`enable_daily_alpha_gate`、`daily_alpha_max_dev_pct=12`：日收盘高于 SMA200 超过 12% 禁新多）。
- **shadow 风控闸门**（执行层）：单日亏 6% / 权益回撤 12%（冷却 2 天）/ 连亏 4 笔 → 暂停开仓。
- 实盘口径重放（cap12 + 闸门，2022-01 → 2026-10）：+487%，最大回撤 68.7%；2025 起 +56%，近 12 月 +13%，近 6 月 +69%，近 3 月 +79%。
  重放时必须加 `--shadow-gate-backtest`，否则逐笔流会多出实盘被闸门挡掉的单（曾因此误判为 +327% / 回撤 92%）。
- K 线处理延迟（2026-10 修复）：唤醒时间与「最新收盘」用同一时钟、向上取整；`dataframe_to_candles` 改按列转换（23.8 万根从 44s → 约 1.7s）。实测收盘后约 12 秒处理完。

### 趋势投票篮子（用 scalp 的闲置保证金）
- 信号：GLD / QQQ ETF **已收盘**日线（Yahoo，纽约 16:15 后才认当日 bar），每腿 5 规则投票
  `sma50/100/200（收盘>均线）+ tsmom30/90（动量>0）` 的平均 = 0~1 目标比例（`strategy/trend_vote.py`）。
- 执行：每个新交易日把各腿多头名义调到 `k × 账户权益 × 0.5 × 投票`；偏离 < 10% 满仓名义不调；先减后加；按信号日去重，失败 15 分钟后重试。
- 参数：`basket_multiplier k=2`、交易所杠杆 `10x` 逐仓、篮子保证金上限 `0.25 × 权益`（满票时约占 20%）。
- 依据（同账户逐日模拟，含逐仓强平与 scalp 拒单，2022 起）：scalp 单跑 +496% / 回撤 67%；+ 篮子 k=1 +900%；**k=2 +1481%，回撤 65.9%**，日盈亏比 1.184（单跑 1.197）。
- 代价：篮子自身历史回撤约 41%（2006 起，主要是 2008）；10x 下 2006 起被强平 3 次（2008-10 黄金、2011-10 QQQ），2022 起 0 次；强平不自动补保证金。
- 与 scalp 的保证金关系：scalp 实盘最高约 12x 有效杠杆（重放最高 16x，需 80% 保证金）；篮子满票约占 20%，极端同时满载会挤占 scalp。
- 资金费实测（2026-07~10）：XAU 7.9%/年，QQQ 3.7%/年（研究里按 6%/年，压力测试 2 倍）。

## 4. 回测口径（硬规则）

**每根 15m K 线只能使用在它收盘那一刻已经完成的 4h / 日线信息。** 唯一合法入口：

```python
from strategy.scalp_robust_v2_core import build_precomputed_state_confirmed_4h
precomputed = build_precomputed_state_confirmed_4h(c4h, c15m)   # confirmed_idx = mapped_idx - 1
```

`align_timeframes()` 映射到的是**仍在形成中**的 4h（4h 的 ts 是开盘时间），直接用会读到未来收盘价。
实测代价：2022→2026-09 同一配置总收益 +396%（无前视）被虚增到 +8525%，回撤低估 11pp，4h 方向分歧率 11.3%。
`_build_precomputed_state_lookahead` 是内部私有构造器，禁止直接调用；不得新增口径开关。
改任何多周期代码后跑：`python3 scripts/audit_backtest_lookahead.py --klines-15m <15m 数据>`。

篮子研究同理：只用已收盘日线，t 收盘信号 → t+1 开盘执行；ETF 当日盘中 bar 必须丢弃。

## 5. 研究结论与复现

数据放在 `var/tmp/time_in_market/`（gitignore）：BTC 15m/4h feather 从服务器 `data/okx/futures/` 拷；
ETF 日线本机访问 Yahoo 会 403，需在东京跑 `scripts/fetch_public_etf_history.py --symbol GLD --symbol QQQ --timeframe 1d --start 2004-11-18 --output-dir /tmp/etf` 再 scp 回来（去掉当天未收盘 bar）。

| 步骤 | 脚本 | 结论 |
|---|---|---|
| 1 | `scalp_trades.py --config <实盘配置> --data-15m … --data-4h … --start 2022-01-01 --shadow-gate-backtest --output …csv` | 实盘口径逐笔 |
| 2 | `explore.py --btc-15m … --gld-1d … --qqq-1d … --scalp-trades … --output …json` | 篮子与 scalp 日收益相关 ≈0（GLD 0.01 / QQQ 0.03）；BTC 趋势腿与 scalp 同合约冲突且最弱，弃用 |
| 3 | `search_basket.py`（27 候选 + 走步） | V5 样本内 Sharpe 只排 13/27，但「每年挑最优」样本外 Sharpe 0.42 < 固定 V5 0.52 → 不调参；20 年里择时的价值是降回撤（21% vs 买入持有 35%）而非收益 |
| 4 | `search_basket_k_cap.py` | SMA200 乖离 cap：低阈值（8~15%）降收益，高阈值（20~30%）的增益只来自 4~10 段行情 → 不加 cap |
| 5 | `sim_basket_liquidation.py` | k × 杠杆同账户模拟（见 §3）；杠杆 ≤10x 强平影响可忽略，20x 以上强平频繁、收益下降；k 越大收益越高、盈亏比越低 |

scalp 引擎此前的参数扫描（`scripts/scan_btc_scalp_optimization.py`）结论：vol-target 仓位、ATR 止损钳制、乖离下限门等 4 个候选均被否决，无新 edge；`config/config.candidate.btc-scalp-optimized-20260921.json` 是回撤更低（42%）但收益更低的备选，未上线。

## 6. 分支与归档

只有两个分支：
- `dev`：开发分支，日常提交、研究都在这里。
- `main`：部署分支，东京服务器只跟踪它；只允许 `git switch main && git merge --ff-only dev` 前进，然后 `bash scripts/ops/deploy_tokyo.sh`。
  `main` 上每个提交都上过实盘，回滚即服务器 `git checkout <main 上一提交>` 后重启对应服务。

2026-10-14 之前的所有分支都已转成 `archive/<原分支名>-20261014` 标签（共 24 个，已推送）；
恢复某个旧分支：`git switch -c <名字> archive/<名字>-20261014`。常用的几个：
- 美股/A 股 ETF（QQQ/TQQQ/GOOGL/CN ETF）未提交研究：`archive/us-equity-wip-20261014`；router 与各腿 executor 完整代码：`c4664aa` 及更早、`archive/美股专项-20261014`。
- BTC headroom review 会话未提交工作：`archive/iter/btc-scalp-headroom-review-20261013-20261014`。
- 旧 `main`（5 月的延迟/拒单修复线）：`archive/main-origin-20261014`、`archive/main-local-20261014`。
- 与项目无关的文档、18MB 期权报价缓存：`~/Documents/crypto-project-archive/`。
