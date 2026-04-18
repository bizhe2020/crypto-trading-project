# Crypto Trading Project

当前仓库以东京服务器正在运行的非-freqtrade OKX bot 为主。

## Main Layout

- `deployment/bot`: 实盘执行层，包含 `run_bot.py`、OKX client、执行引擎、状态存储。
- `deployment/strategy`: 当前 bot 使用的策略与回测核心。
- `deployment/config`: 可提交的样例配置与模板。真实 live 配置不入库。
- `deployment/systemd`: 服务器上的 systemd service 文件。
- `deployment/data`: 本地/服务器共享的数据缓存目录，使用 `feather` 存储 OHLCV。

## Notes

- `feather` 只是当前 bot 和回测共用的行情缓存格式，不代表 freqtrade 策略体系。
- 旧的 freqtrade 目录与 `V8V70_BTCOnlyReclaim` 相关结构不再作为主代码保留。
- 运行态文件不会进入 git：
  - `deployment/config/config.live*.json`
  - `deployment/state/`
  - `deployment/data/`
  - `live_bot*.log`

## Common Commands

启动一次 bootstrap：

```bash
python3 deployment/bot/run_bot.py --config deployment/config/config.example.json
```

运行循环：

```bash
python3 deployment/bot/run_bot.py --config deployment/config/config.live.5x-3pct.json --run-loop
```
