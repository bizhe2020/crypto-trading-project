# Crypto Trading Project

当前仓库是东京服务器 OKX bot 的主部署仓库，只保留实盘 bot、策略和部署脚本。

## Main Layout

- `bot`: 实盘执行层，包含 `run_bot.py`、OKX client、执行引擎、状态存储。
- `strategy`: 当前 bot 使用的策略与回测核心。
- `config`: 当前 live 配置和模板。
- `systemd`: 服务器上的 systemd service 文件。
- `data`: 本地/服务器共享的数据缓存目录，使用 `feather` 存储 OHLCV。
- `scripts`: 服务器 bootstrap 和东京服务器部署脚本。
- `state`: SQLite 运行态目录，占位保留，实际数据库不提交。
- `var`: Funding/OI 录制日志与产物目录，占位保留。

## Notes

- `feather` 只是当前 bot 和回测共用的行情缓存格式，不代表 freqtrade 策略体系。
- 东京服务器当前仍有旧的 `deployment/` 路径习惯；`scripts/bootstrap_server.sh` 会自动补兼容软链并安装新的 flat service。
- 运行态文件不会进入 git：
  - `state/*.db`
  - `data/*`
  - `var/funding_oi/recorded/*`
  - `var/log/*`
  - `live_bot*.log`

## Common Commands

启动一次 bootstrap：

```bash
python3 bot/run_bot.py --config config/config.example.json
```

运行循环：

```bash
python3 bot/run_bot.py --config config/config.live.5x-3pct.json --run-loop
```

服务器一键初始化：

```bash
zsh scripts/bootstrap_server.sh
```

从本机一键部署到东京服务器：

```bash
TOKYO_PASS='your-password' zsh scripts/deploy_tokyo.sh
```
