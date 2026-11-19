#!/usr/bin/env python3
"""不依赖 Telegram 的实盘巡检告警（跑在东京服务器，cron 每 5 分钟一次，推送飞书自定义机器人）。

起因：2026-11-13 两个服务被停掉后没有任何告警，QQQ 多头无人管理 6 天。

检查项（告警 key → 含义）：
  down:<service>     期望运行的服务不是 active
  orphan:<instId>    交易所里有仓位，但负责管它的服务没在运行
  stale_log          scalp 服务 active 但日志超过 N 分钟没更新（卡死 / 网络断）
  dd:<pct>           权益相对已观测峰值回撤超过阈值（15 / 25 / 35）
  liq:<instId>       标记价距强平价 < N%
  api                OKX 查询失败（无法确认仓位，本身就是告警）

行为：同一个 key 持续存在时，每 renotify_hours 小时才重发一次；key 消失发「恢复」；
每天 summary_hour_utc 点发一条心跳摘要（收不到心跳 = 巡检本身挂了）。
只读：只调用 OKX 查询接口和 systemctl is-active，不下单、不改服务。

用法（服务器项目目录）：
  .venv/bin/python scripts/ops/health_check.py                # 正常运行（有 webhook 才推送）
  .venv/bin/python scripts/ops/health_check.py --print-only   # 只打印，不推送、不写状态
  .venv/bin/python scripts/ops/health_check.py --test-webhook # 发一条测试消息
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "healthcheck.json"
DEFAULT_STATE = ROOT / "state" / "healthcheck_state.json"
OKX_CONFIG = ROOT / "config" / "config.live.high-leverage-structure.json"
SCALP_LOG = ROOT / "var" / "log" / "btc_scalp_standalone.log"

# 仓位 → 负责管它的 systemd 服务（前缀匹配 instId）
POSITION_OWNER = {
    "BTC-": "btc-scalp-standalone",
    "QQQ-": "trend-basket",
    "XAU-": "trend-basket",
}
ALL_SERVICES = ("btc-scalp-standalone", "trend-basket")

DEFAULT_CFG: dict[str, Any] = {
    "webhook_url": "",
    "keyword": "东京实盘",          # 飞书机器人若设了关键词校验，消息必须包含它
    "expected_active": [],         # 期望一直运行的服务；当前两个都故意停着，所以为空
    "scalp_log_max_age_min": 45,
    "drawdown_alert_pct": [15, 25, 35],
    "liq_distance_alert_pct": 8.0,
    "renotify_hours": 6,
    "summary_hour_utc": 1,
}


# --------------------------------------------------------------------------- 纯函数（便于测试）


def owner_of(inst_id: str) -> str | None:
    for prefix, service in POSITION_OWNER.items():
        if inst_id.startswith(prefix):
            return service
    return None


def liq_distance_pct(mark: float, liq: float, pos_side: str) -> float | None:
    """多头：(mark-liq)/mark；空头：(liq-mark)/mark。缺数据返回 None。"""
    if mark <= 0 or liq <= 0:
        return None
    dist = (mark - liq) / mark if pos_side == "long" else (liq - mark) / mark
    return dist * 100.0


def evaluate(snapshot: dict[str, Any], cfg: dict[str, Any], peak_equity: float) -> dict[str, str]:
    """返回当前成立的告警 {key: 文案}。snapshot 见 collect_snapshot。"""
    alerts: dict[str, str] = {}
    services: dict[str, bool] = snapshot["services"]

    for name in cfg["expected_active"]:
        if not services.get(name, False):
            alerts[f"down:{name}"] = f"服务 {name} 未运行（期望常驻）"

    if snapshot.get("api_error"):
        alerts["api"] = f"OKX 查询失败：{snapshot['api_error']}"

    for pos in snapshot.get("positions", []):
        inst = pos["instId"]
        owner = owner_of(inst)
        if owner and not services.get(owner, False):
            alerts[f"orphan:{inst}"] = (
                f"{inst} {pos['posSide']} {pos['pos']} 张，浮盈 {pos['upl']:+.2f}，"
                f"但 {owner} 未运行——仓位无人管理"
            )
        dist = liq_distance_pct(pos["markPx"], pos["liqPx"], pos["posSide"])
        if dist is not None and dist < float(cfg["liq_distance_alert_pct"]):
            alerts[f"liq:{inst}"] = f"{inst} 距强平仅 {dist:.1f}%（标记 {pos['markPx']} / 强平 {pos['liqPx']}）"

    age = snapshot.get("scalp_log_age_min")
    if services.get("btc-scalp-standalone", False) and age is not None and age > float(cfg["scalp_log_max_age_min"]):
        alerts["stale_log"] = f"scalp 服务 active 但日志已 {age:.0f} 分钟没更新"

    equity = snapshot.get("equity")
    if equity is not None and peak_equity > 0:
        dd = (1.0 - equity / peak_equity) * 100.0
        crossed = [p for p in cfg["drawdown_alert_pct"] if dd >= float(p)]
        if crossed:
            top = max(crossed)
            alerts[f"dd:{int(top)}"] = f"权益 {equity:.0f}，较峰值 {peak_equity:.0f} 回撤 {dd:.1f}%（阈值 {top}%）"
    return alerts


def decide_notifications(
    alerts: dict[str, str], state: dict[str, Any], now_ts: float, renotify_hours: float
) -> tuple[list[str], list[str]]:
    """返回 (需要推送的告警 key, 已恢复的 key)，并就地更新 state['active']。"""
    active: dict[str, float] = state.setdefault("active", {})
    fire: list[str] = []
    for key in alerts:
        last = active.get(key)
        if last is None or now_ts - last >= renotify_hours * 3600:
            fire.append(key)
            active[key] = now_ts
    recovered = [k for k in list(active) if k not in alerts]
    for k in recovered:
        del active[k]
    return fire, recovered


def should_send_summary(state: dict[str, Any], now: datetime, summary_hour_utc: int) -> bool:
    today = now.date().isoformat()
    return now.hour >= summary_hour_utc and state.get("last_summary_date") != today


def format_summary(snapshot: dict[str, Any], peak_equity: float, alerts: dict[str, str]) -> str:
    lines = []
    eq = snapshot.get("equity")
    if eq is not None:
        dd = (1.0 - eq / peak_equity) * 100.0 if peak_equity > 0 else 0.0
        lines.append(f"权益 {eq:.2f} USDT（峰值 {peak_equity:.0f}，回撤 {dd:.1f}%）")
    svc = "、".join(f"{n}={'运行' if ok else '停止'}" for n, ok in snapshot["services"].items())
    lines.append(f"服务：{svc}")
    positions = snapshot.get("positions", [])
    if positions:
        for p in positions:
            dist = liq_distance_pct(p["markPx"], p["liqPx"], p["posSide"])
            dist_txt = f"，距强平 {dist:.1f}%" if dist is not None else ""
            lines.append(f"  {p['instId']} {p['posSide']} {p['pos']} 张 浮盈 {p['upl']:+.2f}{dist_txt}")
    else:
        lines.append("无持仓")
    lines.append(f"当前告警：{len(alerts)} 条" + ("" if not alerts else "（" + "；".join(alerts) + "）"))
    return "\n".join(lines)


# --------------------------------------------------------------------------- IO


def load_cfg(path: Path) -> dict[str, Any]:
    cfg = dict(DEFAULT_CFG)
    if path.exists():
        cfg.update(json.loads(path.read_text(encoding="utf-8")))
    return cfg


def service_active(name: str) -> bool:
    res = subprocess.run(["systemctl", "is-active", name], capture_output=True, text=True, timeout=10)
    return res.stdout.strip() == "active"


def collect_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {"services": {n: service_active(n) for n in ALL_SERVICES}, "positions": []}
    snap["scalp_log_age_min"] = (time.time() - SCALP_LOG.stat().st_mtime) / 60.0 if SCALP_LOG.exists() else None
    try:
        sys.path.insert(0, str(ROOT))
        from bot.okx_client import OkxClient, OkxCredentials

        c = json.loads(OKX_CONFIG.read_text(encoding="utf-8"))
        ex = OkxClient(
            OkxCredentials(c["api_key"], c["api_secret"], c["api_passphrase"]),
            trading_mode="live",
            proxy=c.get("proxy"),
        ).exchange
        snap["equity"] = float(ex.privateGetAccountBalance()["data"][0]["totalEq"])
        for p in ex.privateGetAccountPositions()["data"]:
            if float(p.get("pos") or 0) == 0:
                continue
            snap["positions"].append(
                {
                    "instId": p["instId"],
                    "posSide": p["posSide"],
                    "pos": float(p["pos"]),
                    "upl": float(p.get("upl") or 0),
                    "markPx": float(p.get("markPx") or 0),
                    "liqPx": float(p.get("liqPx") or 0),
                }
            )
    except Exception as exc:  # noqa: BLE001 - 查询失败本身就是要告警的事件
        snap["api_error"] = f"{type(exc).__name__}: {str(exc)[:150]}"
    return snap


def send_feishu(cfg: dict[str, Any], text: str) -> None:
    body = json.dumps(
        {"msg_type": "text", "content": {"text": f"【{cfg['keyword']}】{text}"}}, ensure_ascii=False
    ).encode("utf-8")
    req = urllib.request.Request(
        cfg["webhook_url"], data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8") or "{}")
    if payload.get("code", 0) != 0:
        raise RuntimeError(f"飞书返回错误：{payload}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--print-only", action="store_true", help="只打印，不推送、不写状态")
    parser.add_argument("--test-webhook", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(Path(args.config))
    if args.test_webhook:
        if not cfg["webhook_url"]:
            print("webhook_url 为空")
            return 2
        send_feishu(cfg, "巡检告警通道测试：收到这条说明推送正常。")
        print("sent")
        return 0

    state_path = Path(args.state)
    state: dict[str, Any] = json.loads(state_path.read_text()) if state_path.exists() else {}
    now = datetime.now(timezone.utc)
    snapshot = collect_snapshot()

    equity = snapshot.get("equity")
    peak = float(state.get("peak_equity") or 0.0)
    if equity is not None:
        peak = max(peak, equity)
    alerts = evaluate(snapshot, cfg, peak)
    fire, recovered = decide_notifications(alerts, state, now.timestamp(), float(cfg["renotify_hours"]))

    messages = [f"🔴 {alerts[k]}" for k in fire] + [f"🟢 已恢复：{k}" for k in recovered]
    if should_send_summary(state, now, int(cfg["summary_hour_utc"])):
        messages.append("📋 每日摘要\n" + format_summary(snapshot, peak, alerts))
        state["last_summary_date"] = now.date().isoformat()

    if args.print_only:
        print(format_summary(snapshot, peak, alerts))
        print("将推送：", messages or "无")
        return 0

    state["peak_equity"] = peak
    if messages and cfg["webhook_url"]:
        try:
            send_feishu(cfg, "\n".join(messages))
        except Exception as exc:  # noqa: BLE001
            print(f"推送失败：{exc}", file=sys.stderr)
            # 推送失败不能吞掉告警：清掉去重记录，下个周期重试
            for k in fire:
                state["active"].pop(k, None)
            if any("每日摘要" in m for m in messages):
                state.pop("last_summary_date", None)
    elif messages:
        print("（未配置 webhook_url）\n" + "\n".join(messages))

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
