# Crypto Trading Project

当前仓库以策略研究、回测和数据集成为主。

## Main Layout

- `strategy`: 策略核心与回测逻辑。
- `data_tools`: 数据采集、健康检查和 OHLCV 读取工具。
- `data`: 本地数据缓存目录，使用 `feather` 存储 OHLCV。
- `var`: 研究过程中的录制数据、日志和中间产物。

## Notes

- `feather` 只是研究和回测共用的行情缓存格式，不代表 freqtrade 策略体系。
- 实盘部署代码、live 配置和运行态数据库不作为当前分支主代码保留。

## Common Commands

录制 Funding/OI：

```bash
python3 data_tools/record_funding_oi.py --run-once
```

检查 Funding/OI 数据健康：

```bash
python3 data_tools/check_funding_oi_health.py
```
