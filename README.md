# OBI Trailing Research

当前仓库只保留策略研究、回测、订单簿录制和 OBI 退出层验证相关代码。

## Main Layout

- `strategy`: 策略核心、trailing overlay 与回测逻辑。
- `bot`: 研究工具脚本，包括行情加载、订单簿录制、OBI 回放、收益重算。
- `config`: 回测样例配置。
- `data`: OHLCV 数据缓存目录。
- `var/research`: 导出的标准交易清单与研究结果。

## Notes

- 已删除实盘执行、systemd、状态库和交易所下单层，只保留研究所需的数据集成能力。
- `feather` 是本地回测共用的行情缓存格式。
- 运行态和数据文件不会进入 git：
  - `data/`
  - `var/orderbook/`
  - `var/replay/`

## Common Commands

导出 OBI 替换回测基线交易：

```bash
python3 bot/export_obi_replay_trades.py \
  --config config/config.example.json \
  --data-root data/okx/futures \
  --start-date 2023-01-01 \
  --output-json var/research/obi_replay_input.json \
  --output-csv var/research/obi_replay_input.csv
```

录制 OKX `books5`：

```bash
python3 bot/record_orderbook.py \
  --inst-id BTC-USDT-SWAP \
  --channel books5 \
  --output-dir var/orderbook \
  --rotate-utc daily
```
