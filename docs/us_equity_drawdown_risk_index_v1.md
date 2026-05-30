# US Equity Drawdown Risk Index v1

目标：构建一个只用于风控的美股回撤风险指数，预测 QQQ/SPY 未来 5/10/20 个交易日出现较大回撤的概率倾向。它不产生独立多空 alpha，只作为 QQQ 补充腿的减仓、禁开仓、降杠杆闸门。

## 标签

- `label_dd_5d`: 未来 5 个交易日 QQQ close-to-close 最大回撤 <= -3%
- `label_dd_10d`: 未来 10 个交易日 QQQ close-to-close 最大回撤 <= -5%
- `label_dd_20d`: 未来 20 个交易日 QQQ close-to-close 最大回撤 <= -8%

## 因子分组

- `trend`: QQQ/SPY/^IXIC 均线破坏、20d 弱势、60d 高点回撤。
- `volatility`: VIX 水位、VIX 相对 MA20、VIX 5d 变化、QQQ 20d 实现波动、日内 range。
- `breadth_proxy`: QQEW/QQQ、RSP/SPY 的相对弱势，用等权 ETF 代理市场宽度。
- `credit`: HYG/IEF 信用风险偏好、TLT/SPY risk-off bid。
- `cross_asset`: BTC 弱势与 BTC 实现波动。

所有 raw 因子统一转为“越高风险越高”的方向，再做 rolling 252d z-score。

## 风险指数

类别权重：

```text
trend          25%
breadth_proxy 25%
volatility    20%
credit        20%
cross_asset   10%
```

最终 `risk_score` 是 composite risk z-score 的 rolling 756d percentile，范围 0-100。早期样本不足时使用 clipped z-score fallback。

## 风控分层

```text
0-35    QQQ 正常 risk-on
35-55   保守，禁止加杠杆
55-70   QQQ 目标 exposure 50%
70-85   QQQ 目标 exposure 25%
85+     QQQ 目标 exposure 0%
```

## 运行

```bash
python3 scripts/research_us_equity_drawdown_risk.py --refresh
```

输出：

- `var/reports/us_equity_drawdown_risk_v1.json`
- `var/reports/us_equity_drawdown_risk_v1_daily.csv`

## 审计原则

- 所有因子只使用当日及之前数据。
- overlay 回测使用 `risk_score.shift(1)` 控制下一交易日 exposure，避免 close-to-close 同日偷看。
- 该指数只先作为研究层，不直接接入 live；接入前必须同步 live/replay 口径。

## 2026-05-30 v1 初跑结论

样本：`2016-01-01 -> 2026-05-29`

最新状态：

```text
risk_score: 0.13
suggested_exposure: 1.00
latest trading day: 2026-05-29
```

10 日回撤标签表现：

```text
base event rate: 13.70%
risk >= 55 event rate: 16.21%
risk >= 70 event rate: 17.05%
risk >= 85 event rate: 19.07%
```

overlay 粗测：

```text
QQQ buy-hold total return: +574.26%
QQQ buy-hold max DD: -35.62%
risk-scaled total return: +169.79%
risk-scaled max DD: -33.34%
avg exposure: 60.45%
```

解读：v1 更像“压力状态识别器”，能在风险高分位看到回撤概率上升，但直接按 55/70/85 分层减仓会过度牺牲收益，DD 改善不够。下一步应加入更强的 forward-looking fragility 因子和概率校准，再决定是否接入 QQQ 风控闸门。

## v2：fragility + walk-forward logistic

新增 forward-looking fragility 因子：

- QQQ 相对 MA50 的过度扩张。
- QQQ 相对 60d low 的扩张。
- VIX/realized vol 压缩。
- QQQ 上涨但 QQEW/QQQ 走弱的宽度背离。
- QQQ 上涨但 HYG/IEF 走弱的信用背离。

模型：

- `LogisticRegression(C=0.5)`
- walk-forward 训练，默认训练窗起点约 756 个交易日。
- 预测目标：`label_dd_10d`
- 训练时对最后 `horizon` 天做 embargo，避免未来标签泄露。
- 不使用 `class_weight=balanced`，因为它会破坏概率校准。

v2 结果：

```text
10d base event rate: 15.22%
model prob >= 0.16 event rate: 22.17%
model prob >= 0.22 event rate: 24.13%
model prob >= 0.35 event rate: 26.79%
top decile event rate: 27.17%
ROC AUC: 0.6004
Average precision: 0.2188
Brier: 0.1384
```

模型 overlay 粗测：

```text
QQQ buy-hold total return: +574.26%
QQQ buy-hold max DD: -35.62%
model risk-scaled total return: +414.79%
model risk-scaled max DD: -29.13%
avg exposure: 89.05%
zero exposure days: 176
```

当前最新交易日 `2026-05-29`：

```text
stress percentile risk_score: 3.84
model_probability_10d: 0.5432
model_suggested_exposure: 0.00
```

解读：v2 的排序能力和风控收益/回撤权衡明显优于 v1，但最新概率较高主要来自 fragility 因子（QQQ 扩张、信用/宽度背离、VIX 压缩），这类信号可能提前较久，不能直接硬清仓。接入 live 前建议先做影子模式，只记录 `model_probability_10d` 和建议 exposure，不实际改仓。
