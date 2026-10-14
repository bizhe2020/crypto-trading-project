# 运行健康检查（round 18）

> 本轮本意是给双腿 paper 跟踪器加自动化，过程中发现**你的线上信号服务已经停了、且数据源已失效**。
> 以下都是可复核的现场证据（日志、plist、launchctl 输出）。

---

## 1. 线上 TQQQ/SQQQ 信号服务：已停止运行

| 项 | 现场值 |
|---|---|
| launchd 服务 | `com.bizhe.tqqq-sqqq-close-signal` / `-preopen-signal` |
| `launchctl list \| grep tqqq` | **无输出**（服务未加载） |
| 日志最后写入 | `var/log/tqqq_sqqq_close_signal.log` → **2026-07-27 10:00** |
| 日志里最后一封成功邮件 | `[TQQQ][做多TQQQ] ... 2026-07-24  sent_to r10046lcn@163.com` |
| 结论 | **约 7 周没有发出信号**（plist 未加载 + 数据刷新失败） |

## 2. 数据刷新已失效（这是它停摆的直接原因）

`config/tqqq_sqqq_signal_email.json` 里：

```json
"refresh_before_send": true,
"refresh_fail_open": true,          // ← 刷新失败仍继续（用旧数据）
"refresh_proxy": "http://127.0.0.1:7892",
"refresh_symbols": ["QQQ","TQQQ","SQQQ","SPY","^IXIC","^VIX"]
```

日志显示每次运行都失败：

```
refresh_proxy_failed_retry_direct QQQ 1d: curl: (7) Failed to connect to 127.0.0.1 port 7892
refresh_failed_fallback QQQ 1d: 403 Client Error: Forbidden
   ...（TQQQ / SQQQ / SPY / ^IXIC / ^VIX 全部 403）
```

* 本地代理（7892）**没在跑**；
* 直连 Yahoo Chart API 返回 **403**（我本轮也独立验证过：Stooq 有 JS 挑战、Yahoo 403、AlphaVantage demo 无权限）；
* 由于 `refresh_fail_open: true`，服务**不会报错退出**，而是用本地已有数据继续出信号——这正是
  `data/public/etf/^VIX-1d.feather` 停在 2026-08-14、BTC 停在 2026-08-13 的原因。

## 3. 我原本担心的"前视"问题：线上路径是安全的（重要澄清）

round 15 审计的那条 194% 曲线来自 `scripts/plot_strategy_comparison.py::tqqq_sqqq_curve()`（**诊断图脚本**）。
你**线上邮件信号**走的是另一条路径 `scripts/tqqq_cash_strict_utils.py::run_strict_candidate()`，我核对了它的时序：

```python
base_desired_today = bool(int(frame.iloc[idx-1]["entry_signal"]) > 0
                          and bool(allow_mask.iloc[idx-1]))     # ← 用 t-1 的闸门
...
capital *= float(row["tqqq_open"]) / previous_close             # ← 用 t 日开盘成交
```

**它用的是 t-1 的 allow_mask（已滞后）、并在 t 日开盘成交**——符合
`docs/backtest_lookahead_policy.md` 的口径。所以：

* ✅ **线上信号路径：无前视**；
* ❌ **诊断图脚本：有前视**（round 15 已量化：CAGR 201% → 滞后版 2.7%）——**不要用那张图做决策**。

## 4. 建议（按优先级）

1. **先决定要不要恢复服务**：如果是有意停掉的，忽略即可；如果是意外停的，
   `launchctl load ~/Library/LaunchAgents/com.bizhe.tqqq-sqqq-*-signal.plist` 可恢复（但数据源要先修好）。
2. **修数据源**（否则恢复也没意义）：
   * 启动本地代理（配置里写的是 `127.0.0.1:7892`），或
   * 换可用的数据源（Yahoo 403 已持续至少到 7 月底），或
   * 至少把 `refresh_fail_open` 改成 `false`——**让刷新失败硬失败**，避免"静默用旧数据出信号"。
3. **加新鲜度闸门**：我在 `research/qqq_spot_leverage/daily_runner.py` 里实现了这个模式
   （逐输入报告"落后几个交易日"，`--strict` 时过期直接退出码 2）——可以照搬到线上信号脚本。
4. `anchor_fill_price: 77.84`、`principal: 5000` 这些状态在你决定恢复时再核对一次。

## 4b. 追加（round 19）：代理确认已死 + 新鲜度闸门已实现

**代理确认**：
- `nc -z 127.0.0.1 7892` → **CLOSED**（配置里指定的刷新代理确实没在跑）；
- 直连路径仍 403（我本轮独立验证：Yahoo 403、Stooq JS 挑战、AlphaVantage demo 无权限）；
- 仅 `config/tqqq_sqqq_signal_email.json` 与对应测试引用该代理（7892 没有其它配置引用）。
- 结论：**在代理重启或换源之前，任何依赖 Yahoo 的刷新都不会成功**。

**新鲜度闸门**：`check_freshness.py`（只读、不联网、不改任何东西）
以**今天**为基准日检查 `data/public/etf/` 下的信号输入（用"文件日期之间互比"是看不出系统性停摆的，
这正是本次服务静默 7 周没被发现的盲区）。实测：

```
基准日 2026-09-22；阈值 3 个交易日
  QQQ   最后 2026-08-14  落后 27 交易日  [STALE]
  TQQQ  最后 2026-08-14  落后 27 交易日  [STALE]
  SQQQ  最后 2026-08-14  落后 27 交易日  [STALE]
  SPY   最后 2026-08-14  落后 27 交易日  [STALE]
  ^IXIC 最后 2026-08-14  落后 27 交易日  [STALE]
  ^VIX  最后 2026-08-14  落后 28 交易日  [STALE]
--strict → 退出码 2
```

也就是说：**信号服务的全部输入都停在 2026-08-14，落后约 27 个交易日**。

接入方式（3 行，不改你原有逻辑）：

```python
from research.qqq_spot_leverage.check_freshness import check_freshness
ok, detail = check_freshness(Path("data/public/etf"), max_lag_trading_days=3)
if not ok and not config.get("refresh_fail_open"):
    raise SystemExit(f"数据过期，拒绝出信号: {detail}")
```

## 4c. 追加（round 23）：离线数据导入通道（网络不可用时的替代方案）

**代理排查结论（更彻底）**：
- 扫了 7890 / 7891 / 7892 / 7897 / 1080 / 1087 / 8888 / 6152 / 6244 / 8118 / 10808 / 10809 —— **全部无监听**；
- 环境变量里也没有 `http_proxy` / `https_proxy`；
- 全仓库只有 `config/tqqq_sqqq_signal_email.json` 引用 7892（外加对应测试），说明这是一个**过期的配置残留**；
- 结论：**本机当前没有任何可用代理**，直连又被 403；在换源或恢复代理之前，自动刷新不可能成功。

**替代通道**：`import_market_data.py` —— 从任意来源导出的 CSV/TSV/feather 更新缓存。

```bash
# 校验但不写入
python3 research/qqq_spot_leverage/import_market_data.py     --symbol SPY --input ~/Downloads/SPY.csv --data-root data/public/etf --dry-run

# 正式导入（自动备份、按日期合并去重、保留更长历史）
python3 research/qqq_spot_leverage/import_market_data.py     --symbol SPY --input ~/Downloads/SPY.csv --data-root data/public/etf
```

安全设计（都已实测）：
| 机制 | 实测 |
|---|---|
| 列名/顺序不敏感、自动识别日期列 | ✅（用大小写混排的 CSV 通过） |
| **拼接缝连续性校验** | ✅ 把 776.34 错接成 643.50（−17.1%）→ **拒绝写入，退出码 2** |
| 阈值自适应（clip(8×MAD 日波动, 6%, 12%)） | ✅ 正确数据通过；只检查导入日期附近的 5~11 个位置，不误伤 2025-04-09 这类真实极端日 |
| 默认拒绝缩短历史 | ✅ `--allow-shrink` 才允许 |
| 写入前自动备份 | ✅ `SPY-1d.feather.bak-<时间戳>` |
| `--dry-run` | ✅ 只校验不落盘 |

**标准离线流程**：外部导出 → `import_market_data.py` 导入 → `check_freshness.py` 确认 → `daily_runner.py` 出信号。

## 5. 我这轮实际交付的东西

* `daily_runner.py`：**数据新鲜度检查 + 双腿 paper 信号 + 每日状态落盘**，支持 `--strict` 阻断；
  当前实测输出：

  ```
  QQQ   最后 2026-09-02  落后  0 交易日  [ok]
  VIX   最后 2026-08-14  落后 13 交易日  [STALE]
  VIX3M 最后 2026-08-14  落后 13 交易日  [STALE]
  SPY   最后 2026-08-14  落后 13 交易日  [STALE]
  BTC   最后 2026-08-13  落后 14 交易日  [STALE]
  ```
  `--strict --no-paper` → 退出码 2（实测）。

* 每日状态追加到 `state/two_leg_daily_status.jsonl`，便于事后对账。
