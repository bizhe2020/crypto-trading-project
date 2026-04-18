# OBI Forward Experiment

## Objective

在没有近三年历史订单簿的前提下，用未来录制的 `books5` 数据验证 OBI 退出层是否优于原始退出逻辑。

## Baseline

统一使用 `2023-01-01` 起跑、`target_rr_cap` 已接入后的标准交易清单。

目标口径：

- `total_return_pct ~= 1032.46%`
- `profit_factor ~= 1.2457`

## Workflow

1. 导出标准交易输入

```bash
python3 deployment/bot/export_obi_replay_trades.py \
  --config deployment/config/config.live.5x-3pct.json \
  --data-root deployment/data/okx/futures \
  --start-date 2023-01-01 \
  --output-json var/research/obi_replay_input_20230101_cap.json \
  --output-csv var/research/obi_replay_input_20230101_cap.csv
```

2. 录盘口

```bash
python3 deployment/bot/record_orderbook.py \
  --inst-id BTC-USDT-SWAP \
  --channel books5 \
  --output var/orderbook/btc_books5_$(date +%Y%m%d_%H%M).jsonl \
  --duration-seconds 14400
```

3. 批量回放

```bash
python3 deployment/bot/batch_replay_obi_overlay.py \
  --trades-json var/research/obi_replay_input_20230101_cap.json \
  --orderbook-input var/orderbook/<books5>.jsonl \
  --output-json var/replay/obi_forward_batch.json \
  --output-csv var/replay/obi_forward_batch.csv
```

## Evaluation

重点看：

- `covered_trades`
- `trades_with_tighten`
- `trades_with_exit`
- 触发交易的 `delta_rr`
- 组合层 `alpha / PF / MDD`
