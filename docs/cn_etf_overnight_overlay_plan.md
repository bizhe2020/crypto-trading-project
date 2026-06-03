# 中国ETF 夜盘 Overlay 计划

当前主线：

- 主策略：`21/200 + ixic_filter + hold=120 + trailing 4d/4%`
- frozen 杠杆：
  - `vix_normal + qqq_strong = 2.0x`
  - `vix_normal + qqq_neutral = 1.5x`

目标：

- 不改主方向信号
- 只在中国开盘前，用美国前夜的额外信息做：
  - `de-risk`
  - `skip`
  - `leverage adjust`

## 当前结论

1. `QQQ after-hours` 理论上可用

- 美国东部 `16:00-20:00` 已结束时，中国仍未开盘
- 所以这部分不属于未来信息

2. 现阶段更合适做 overlay，而不是替代主信号

- 主策略已经由 walk-forward 和 bucket 审计支持
- 夜盘更适合作为修正层，而不是重写趋势底座

3. 当前数据抓取有硬阻塞

- 这台机器当前无法稳定从 Yahoo 拉取 `QQQ 5m`
- 直连和 `127.0.0.1:6244` 代理都失败
- 所以现阶段还不能正式回测 `after-hours / overnight` overlay

## 先做的底座

1. 清洗美国日线数据

- 当前 `QQQ-1d` 等文件存在同一美国交易日的重复尾行
- 需要先按美国本地交易日去重，避免 overlay 将来叠在脏底座上

2. 固定 overlay 输入定义

未来一旦拿到分钟级数据，优先抽这三类特征：

- `after_hours_return_t1`
  - 前一美国交易日 `16:00 -> 20:00`
- `after_hours_gap_vs_close`
  - 盘后最终价相对正式收盘价偏离
- `overnight_to_cn_open_proxy`
  - 若后续接入 `NQ`，优先用纳指期货连续夜盘替代 `QQQ after-hours`

## 研究优先级

1. `de-risk overlay`

- 当前已经持有 `2.0x / 1.5x`
- 若夜盘显著转弱，则降到 `1.0x`

2. `skip overlay`

- 只对第二天新开仓生效
- 夜盘极差时允许跳过

3. `boost overlay`

- 只有在已有 frozen 高质量 bucket 上，且夜盘继续强化时，才考虑额外放大

## 暂不做

- 不直接把夜盘特征做成新的主 entry
- 不在没有分钟级数据时伪造 after-hours 研究结果
- 不把重复日线尾行直接当夜盘代理
