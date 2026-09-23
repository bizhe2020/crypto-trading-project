# BTC vs QQQ vs 趋势路由 — 近 3 个月可复现回测（2026-09-23）

> 状态：研究结论，只读回测，未改任何实盘配置。
> 目的：验证"纯趋势 router / BTC+QQQ 路由优于单腿"在**近 3 个月**是否成立。

## 窗口与口径

- 窗口：`2026-06-23 → 2026-09-20`（90 天，近 3 个月）
- BTC 腿：当前 live 配置 `var/tokyo_audit/btc_scalp_20260920/config.live.high-leverage-structure.json`，引擎重放（`--recompute-btc-engine`，confirmed_4h 口径）
- QQQ 腿：真实 QQQ/USDT 永续 4h + 资金费，`config/config.paper.qqq-usdt-aggressive-runtime.json`（10x，stop 4%，risk_overlay=false = 线上当前语义）
- 路由：`btc_min=35 / qqq_min=98 / switch=6 / takeover 6-6`，切换成本 10bp，日频
- 初始资金 1000 USDT

## 结果

| 腿 | 收益 | maxDD | Calmar(收益/DD) |
|---|---:|---:|---:|
| BTC-only（剥头皮） | **+81.30%** | **42.69%** | **1.90** |
| QQQ-only（10x 永续） | **−48.52%** | 54.52% | −0.89 |
| 路由（BTC+QQQ 赢家通吃） | +99.94% | 54.53% | 1.83 |

路由选腿：BTC 16 天 / QQQ 41 天 / 现金 33 天，切换 20 次。

QQQ 腿细节：5 笔交易、胜率 20%、单笔均收益 −12.18%、资金费成本 4.1%、手续费 0.9%、持仓时平均杠杆 10x。

## 结论

1. **QQQ 趋势腿在近 3 个月是明显亏损腿（−48.52%）**，与 8/24 那次实测（−61.78%，窗口 5/26–8/21）方向一致——杠杆在震荡市里被反复磨死，不是指数跌。
2. **路由并不能救 QQQ**：路由收益 +99.94% 仅比 BTC-only +81.30% 高 18.6pp，但 maxDD 54.53% 比 BTC-only 42.69% 更差，Calmar 1.83 < 1.90。**风险调整后，路由不优于单独跑 BTC。**
3. **近 3 个月的最优选择是 BTC-only**，不是"纯趋势 router"，也不是"BTC+QQQ 路由"。
4. 这再次印证 8/24 把趋势腿关掉、只留 BTC 剥头皮的决定是对的。

## 可复现命令

数据（在东京服务器上拉取后 scp 回本地，`scripts/fetch_qqq_usdt_4h.py` 为本任务新增的可提交脚本）：

```bash
# 服务器（能连 OKX / Yahoo）：
.venv/bin/python scripts/fetch_qqq_usdt_4h.py --out data/okx/futures/QQQ_USDT_USDT-4h-futures.feather
.venv/bin/python scripts/fetch_okx_funding_history.py --symbol QQQ/USDT:USDT --output data/okx/futures/QQQ_USDT_USDT-8h-funding_rate.feather --limit 400
.venv/bin/python scripts/fetch_public_etf_history.py --symbol TQQQ --timeframe 1d --start 2026-08-10T00:00:00Z

# 本地重放（唯一入口，遵守 backtest_lookahead_policy）：
python3 scripts/replay_proxy_strategy_router.py \
  --btc-config var/tokyo_audit/btc_scalp_20260920/config.live.high-leverage-structure.json \
  --btc-15m var/tokyo_audit/btc_scalp_20260920/BTC_USDT_USDT-15m-futures.feather \
  --btc-4h var/tokyo_audit/btc_scalp_20260920/BTC_USDT_USDT-4h-futures.feather \
  --recompute-btc-engine \
  --qqq-source usdt_leveraged \
  --qqq-usdt-config config/config.paper.qqq-usdt-aggressive-runtime.json \
  --qqq-usdt-data-4h data/okx/futures/QQQ_USDT_USDT-4h-futures.feather \
  --qqq-usdt-funding data/okx/futures/QQQ_USDT_USDT-8h-funding_rate.feather \
  --start-date 2026-06-23 --end-date 2026-09-20 \
  --btc-min-score 35 --qqq-min-score 98 --switch-advantage 6 \
  --btc-takeover-advantage 6 --qqq-takeover-advantage 6 \
  --output-json var/reports/router_3mo_btc_qqq_20260923.json \
  --output-md var/reports/router_3mo_btc_qqq_20260923.md \
  --output-csv var/reports/router_3mo_btc_qqq_20260923.csv
```

## 边界与注意事项

- 窗口只有 90 天、QQQ 腿只有 5 笔交易，统计意义有限；但方向与 8/24 独立实测一致。
- QQQ 腿用 runtime 配置（risk_overlay=false）。风险 overlay 的输入 CSV 停在 6/7，即便打开在窗口内也无信号，结果不变。
- 自动生成的 `router_3mo_btc_qqq_20260923.md` 中「BTC leg」一行与 Interpretation 段落是脚本里写死的旧文案，与本窗口不符；以本文档为准。
