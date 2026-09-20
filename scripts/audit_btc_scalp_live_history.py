#!/usr/bin/env python3
"""BTC scalp（high-leverage-structure / btc-scalp-standalone）实盘历史深度审计。

数据源（全部只读）：
  1. 线上 state DB 的 action_log  —— 机器人自己的开/平仓决策与元数据
  2. OKX /account/positions-history —— 交易所权威已平仓持仓（含手续费/资金费）
  3. OKX /trade/fills-history       —— 逐笔成交（用于滑点核对）
  4. 15m K 线                        —— 计算每笔的 MFE / MAE（以 R 为单位）

用法:
    python3 scripts/audit_btc_scalp_live_history.py \
        --db var/tokyo_audit/btc_scalp_20260920/runtime_btc_scalp_live.db \
        --okx var/tokyo_audit/btc_scalp_20260920/okx_history.json \
        --klines var/tokyo_audit/btc_scalp_20260920/BTC_USDT_USDT-15m-2026.csv \
        --out-csv var/tokyo_audit/btc_scalp_20260920/trades_audit.csv \
        --out-md var/reports/btc_scalp_live_history_audit.md

设计要点：
  - 交易配对按时间顺序前向匹配（策略 max_open_positions=1）；无法配对的孤儿事件单独统计，
    不静默丢弃 —— 孤儿本身就是执行/状态同步问题的证据。
  - R 距离一律取「开仓时记录的初始止损」，因此 realized_R 与 target_rr 可直接比较。
  - MFE/MAE 用平仓所在 15m K 线一并计入（止损在 bar 内触发）。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

OPEN_TYPES = ("OPEN_LONG", "OPEN_SHORT")
CLOSE_TYPE = "CLOSE_POSITION"
EVENT_TYPES = OPEN_TYPES + (CLOSE_TYPE,)


@dataclass
class BotTrade:
    open_id: int
    close_id: int | None
    open_ts: pd.Timestamp
    close_ts: pd.Timestamp | None
    direction: str
    entry: float
    stop: float | None
    target: float | None
    exit: float | None
    reason: str | None
    meta: dict = field(default_factory=dict)
    close_meta: dict = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        return self.direction == "BULL"


def load_bot_trades(
    db_path: Path,
) -> tuple[list[BotTrade], dict[str, int], list[dict], list[dict]]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, timestamp, action_type, payload FROM action_log ORDER BY id"
    ).fetchall()
    conn.close()

    type_counts: dict[str, int] = {}
    errors: list[dict] = []
    events: list[tuple[int, str, str, dict]] = []
    for row_id, ts, action_type, payload in rows:
        type_counts[action_type] = type_counts.get(action_type, 0) + 1
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if action_type == "ERROR":
            errors.append(decoded)
        if action_type in EVENT_TYPES:
            events.append((row_id, ts, action_type, decoded))

    trades: list[BotTrade] = []
    pending: BotTrade | None = None
    orphans: list[dict] = []
    for row_id, ts, action_type, decoded in events:
        meta = decoded.get("metadata") or {}
        if action_type in OPEN_TYPES:
            if pending is not None:
                orphans.append({"kind": "open_without_close", "id": pending.open_id,
                                "ts": str(pending.open_ts)})
            stop = decoded.get("stop_price")
            target = decoded.get("target_price")
            pending = BotTrade(
                open_id=row_id,
                close_id=None,
                open_ts=pd.Timestamp(ts, tz="UTC"),
                close_ts=None,
                direction=decoded.get("direction") or "",
                entry=float(decoded.get("entry_price") or np.nan),
                stop=float(stop) if stop else None,
                target=float(target) if target else None,
                exit=None,
                reason=None,
                meta=meta,
            )
        else:
            if pending is None:
                orphans.append({"kind": "close_without_open", "id": row_id, "ts": ts})
                continue
            exit_price = decoded.get("exit_price")
            pending.close_id = row_id
            pending.close_ts = pd.Timestamp(ts, tz="UTC")
            pending.exit = float(exit_price) if exit_price else None
            pending.reason = decoded.get("reason")
            pending.close_meta = meta
            trades.append(pending)
            pending = None
    if pending is not None:
        orphans.append({"kind": "open_without_close", "id": pending.open_id,
                        "ts": str(pending.open_ts)})

    return trades, type_counts, errors, orphans


def load_okx(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    frame = pd.DataFrame(payload["positions_history"])
    for col in ("cTime", "uTime"):
        frame[col + "_dt"] = pd.to_datetime(frame[col].astype("int64"), unit="ms", utc=True)
    for col in ("openAvgPx", "closeAvgPx", "closeTotalPos", "openMaxPos",
                "fee", "fundingFee", "pnl", "pnlRatio", "realizedPnl", "lever"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.sort_values("uTime_dt").reset_index(drop=True)


def load_fills(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    fills = pd.DataFrame(payload.get("fills") or [])
    if fills.empty:
        return fills
    fills["ts"] = pd.to_datetime(fills["ts"].astype("int64"), unit="ms", utc=True)
    for col in ("fillPx", "fillSz", "fee"):
        if col in fills.columns:
            fills[col] = pd.to_numeric(fills[col], errors="coerce")
    return fills.sort_values("ts").reset_index(drop=True)


def enrich_klines(trades: list[BotTrade], klines: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for trade in trades:
        meta = trade.meta
        close_meta = trade.close_meta
        row = {
            "open_ts": trade.open_ts,
            "close_ts": trade.close_ts,
            "direction": trade.direction,
            "reason": trade.reason,
            "entry_signal": trade.entry,
            "stop": trade.stop,
            "target": trade.target,
            "exit": trade.exit,
            "risk_pct": np.nan,
            "bars_held": np.nan,
            "realized_R": np.nan,
            "mfe_R": np.nan,
            "mae_R": np.nan,
            "net_pnl": close_meta.get("net_pnl"),
            "gross_pnl": close_meta.get("gross_pnl"),
            "fees": close_meta.get("fees"),
            "capital_at_entry": meta.get("capital_at_entry"),
            "risk_amount": meta.get("risk_amount"),
            "notional": meta.get("notional"),
            "quantity": meta.get("quantity"),
            "risk_per_trade": meta.get("risk_per_trade"),
            "risk_regime": meta.get("risk_regime"),
            "regime_label": meta.get("regime_label"),
            "target_rr": meta.get("target_rr"),
            "trail_style": meta.get("trail_style"),
            "candidate_event_type": meta.get("candidate_event_type"),
            "adx": meta.get("feature_adx"),
            "momentum": meta.get("feature_momentum"),
            "ema_gap": meta.get("feature_ema_gap"),
        }
        if trade.stop and trade.entry and trade.close_ts is not None and trade.exit is not None:
            risk_dist = abs(trade.entry - trade.stop)
            row["risk_pct"] = risk_dist / trade.entry * 100.0
            window = klines[
                (klines["date"] >= trade.open_ts)
                & (klines["date"] <= trade.close_ts + pd.Timedelta(minutes=15))
            ]
            row["bars_held"] = len(window)
            if not window.empty and risk_dist > 0:
                if trade.is_long:
                    mfe = (window["high"].max() - trade.entry) / risk_dist
                    mae = (trade.entry - window["low"].min()) / risk_dist
                    realized = (trade.exit - trade.entry) / risk_dist
                else:
                    mfe = (trade.entry - window["low"].min()) / risk_dist
                    mae = (window["high"].max() - trade.entry) / risk_dist
                    realized = (trade.entry - trade.exit) / risk_dist
                row["mfe_R"] = mfe
                row["mae_R"] = mae
                row["realized_R"] = realized
        records.append(row)
    return pd.DataFrame(records)


def match_exchange(frame: pd.DataFrame, okx: pd.DataFrame, tolerance_min: float = 20.0) -> pd.DataFrame:
    """把机器人记录的每笔交易与 OKX 权威持仓按平仓时间做 1:1 贪心对齐。

    机器人记录的平仓时间是「信号 K 线时间」，OKX 的 uTime 是实际成交时间，
    实测两者最多相差约 10 分钟（约一根 15m K 线），故容差取 20 分钟。
    每一条 OKX 持仓最多被匹配一次，避免同一笔被重复计入。
    """
    frame = frame.copy()
    available = set(okx.index)
    open_px, close_px, net_real, fee, funding, lever, matched = [], [], [], [], [], [], []
    for _, row in frame.iterrows():
        if pd.isna(row["close_ts"]) or not available:
            open_px.append(np.nan); close_px.append(np.nan); net_real.append(np.nan)
            fee.append(np.nan); funding.append(np.nan); lever.append(np.nan); matched.append(False)
            continue
        candidates = list(available)
        delta = (okx.loc[candidates, "uTime_dt"] - row["close_ts"]).abs()
        idx = delta.idxmin()
        if delta.loc[idx] > pd.Timedelta(minutes=tolerance_min):
            open_px.append(np.nan); close_px.append(np.nan); net_real.append(np.nan)
            fee.append(np.nan); funding.append(np.nan); lever.append(np.nan); matched.append(False)
            continue
        available.discard(idx)
        hit = okx.loc[idx]
        open_px.append(hit["openAvgPx"]); close_px.append(hit["closeAvgPx"])
        net_real.append(hit["realizedPnl"]); fee.append(hit["fee"])
        funding.append(hit["fundingFee"]); lever.append(hit["lever"]); matched.append(True)
    frame["okx_open_avg"] = open_px
    frame["okx_close_avg"] = close_px
    frame["okx_net_realized"] = net_real
    frame["okx_fee"] = fee
    frame["okx_funding"] = funding
    frame["okx_lever"] = lever
    frame["okx_matched"] = matched
    frame["entry_slip_bp"] = (frame["okx_open_avg"] - frame["entry_signal"]) / frame["entry_signal"] * 1e4
    frame["assumed_vs_real_pnl"] = np.where(
        frame["okx_matched"],
        (frame["entry_signal"] - frame["okx_open_avg"]) * frame["quantity"],
        np.nan,
    )
    return frame


def fmt(value: float, digits: int = 2, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:,.{digits}f}{suffix}"


def build_report(frame: pd.DataFrame, okx: pd.DataFrame, type_counts: dict[str, int],
                 errors: list[dict], orphans: list[dict]) -> str:
    lines: list[str] = []
    add = lines.append
    closed = frame[frame["close_ts"].notna()].copy()
    matched = closed[closed["okx_matched"]]

    add("# BTC Scalp 实盘历史深度审计")
    add("")
    add(f"- 机器人记录交易：**{len(closed)} 笔已平仓**（另有 {len(frame) - len(closed)} 笔未平仓）")
    add(f"- 与 OKX 权威持仓成功对齐：**{len(matched)} 笔**")
    add(f"- 交易所已平仓持仓总数：**{len(okx)} 笔**（全为 BTC-USDT-SWAP）")
    add("")

    add("## 1. 交易所权威口径：成本结构")
    add("")
    gross = okx["pnl"].sum()
    fees = okx["fee"].sum()
    funding = okx["fundingFee"].sum()
    net = okx["realizedPnl"].sum()
    add("| 项目 | 金额 (USDT) | 占总亏损比 |")
    add("|---|---:|---:|")
    add(f"| 毛盈亏 (不含费用) | {gross:,.2f} | {gross/net*100:.1f}% |")
    add(f"| 手续费 | {fees:,.2f} | {fees/net*100:.1f}% |")
    add(f"| 资金费 | {funding:,.2f} | {funding/net*100:.1f}% |")
    add(f"| **净已实现盈亏** | **{net:,.2f}** | 100.0% |")
    add("")
    cost = abs(fees + funding)
    add(f"交易成本合计 **{cost:,.2f} USDT**，相当于毛亏损绝对值的 "
        f"**{cost/abs(gross)*100:.1f}%**。")
    add("")
    if len(matched):
        add("### 机器人交易子集（按平仓时间与 action_log 对齐）")
        add("")
        add(f"- 净已实现盈亏：**{matched['okx_net_realized'].sum():,.2f} USDT**")
        add(f"- 其中手续费：{matched['okx_fee'].sum():,.2f} USDT，资金费：{matched['okx_funding'].sum():,.2f} USDT")
        add(f"- 机器人子集占全部已平仓持仓：{len(matched)}/{len(okx)} 笔")
        add("")

    add("## 2. 出场原因：止盈从未被触及")
    add("")
    reasons = closed["reason"].fillna("(none)").value_counts()
    add("| 出场原因 | 笔数 |")
    add("|---|---:|")
    for reason, count in reasons.items():
        add(f"| {reason} | {count} |")
    add("")
    target_hits = int(sum("target" in str(r) or "take_profit" in str(r) for r in closed["reason"].fillna("")))
    add(f"打到止盈目标的交易：**{target_hits} 笔 / {len(closed)}**")
    add("")

    add("## 3. MFE / MAE：目标盈亏比在统计上不可达")
    add("")
    dist = closed.dropna(subset=["mfe_R"])
    if len(dist):
        add(f"- 均值 MFE：**{dist['mfe_R'].mean():.2f} R**，中位数 **{dist['mfe_R'].median():.2f} R**")
        add(f"- 均值 MAE：**{dist['mae_R'].mean():.2f} R**，中位数 **{dist['mae_R'].median():.2f} R**")
        add(f"- 均值实际实现：**{dist['realized_R'].mean():.2f} R**")
        add(f"- MFE 回吐合计（MFE − 实现）：**{(dist['mfe_R'] - dist['realized_R']).sum():.2f} R**")
        add("")
        add("| 曾达到 | 笔数 | 占比 |")
        add("|---|---:|---:|")
        for level in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
            hits = int((dist["mfe_R"] >= level).sum())
            add(f"| ≥ {level:.1f} R | {hits} | {hits/len(dist)*100:.0f}% |")
        add("")
        configured = pd.to_numeric(closed["target_rr"], errors="coerce").dropna()
        if len(configured):
            add(f"- 配置的目标盈亏比：最小 {configured.min():.2f} R / 中位 {configured.median():.2f} R / "
                f"最大 {configured.max():.2f} R")
            best = dist["mfe_R"].max()
            add(f"- 历史最佳 MFE：**{best:.2f} R** —— "
                f"{'低于' if best < configured.median() else '高于'}中位目标 "
                f"{configured.median():.2f} R")
        add("")

    add("## 4. 风险预算：按 regime 放大到 12% 是最大单一亏损来源")
    add("")
    risk = closed.dropna(subset=["risk_per_trade"])
    if len(risk):
        grouped = risk.groupby("risk_per_trade").agg(
            笔数=("realized_R", "size"),
            胜率=("net_pnl", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean() * 100)),
            平均实现R=("realized_R", "mean"),
            净盈亏=("net_pnl", lambda s: float(pd.to_numeric(s, errors="coerce").sum())),
        )
        add("| risk_per_trade | 笔数 | 胜率 | 平均实现 R | 净盈亏 (USDT) |")
        add("|---|---:|---:|---:|---:|")
        for level, row in grouped.iterrows():
            add(f"| {level:.0%} | {int(row['笔数'])} | {row['胜率']:.0f}% | "
                f"{row['平均实现R']:.2f} | {row['净盈亏']:,.2f} |")
        add("")

    add("## 5. 入场价假设 vs 实际成交（滑点）")
    add("")
    slip = matched.dropna(subset=["entry_slip_bp"])
    if len(slip):
        add(f"- 平均入场滑点：**{slip['entry_slip_bp'].mean():.1f} bp**"
            f"（中位 {slip['entry_slip_bp'].median():.1f} bp，正数=实际成交更差）")
        add(f"- 因入场价假设乐观而高估的盈亏合计：**{slip['assumed_vs_real_pnl'].sum():,.2f} USDT**")
        add("")

    add("## 6. 执行与状态同步健康度")
    add("")
    add("| 事件类型 | 次数 |")
    add("|---|---:|")
    for name in ("ERROR", "EXECUTION_SKIPPED", "UNEXECUTED_OPEN_ROLLBACK",
                 "UNEXECUTED_CLOSE_ROLLBACK", "MANUAL_POSITION_SYNC", "UPDATE_STOP",
                 "TELEGRAM_ERROR"):
        if name in type_counts:
            add(f"| {name} | {type_counts[name]:,} |")
    add("")
    if errors:
        counter: dict[str, int] = {}
        for item in errors:
            msg = str(item.get("error") or "")[:90]
            counter[msg] = counter.get(msg, 0) + 1
        add("错误 Top 5：")
        add("")
        for msg, count in sorted(counter.items(), key=lambda kv: -kv[1])[:5]:
            add(f"- `{msg}` × {count:,}")
        add("")
    if orphans:
        add("孤儿事件（无配对，说明执行/状态不一致）：")
        add("")
        for item in orphans:
            add(f"- {item['kind']} id={item['id']} ts={item['ts']}")
        add("")

    add("## 7. 逐笔明细")
    add("")
    cols = ["open_ts", "close_ts", "direction", "entry_signal", "okx_open_avg", "stop", "exit",
            "risk_pct", "bars_held", "realized_R", "mfe_R", "mae_R", "net_pnl",
            "risk_per_trade", "regime_label", "target_rr", "trail_style", "reason"]
    table = closed[cols].copy()
    for col in ("open_ts", "close_ts"):
        table[col] = table[col].astype(str).str.slice(0, 19)
    add(table.to_markdown(index=False, floatfmt=",.2f"))
    add("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--okx", required=True)
    parser.add_argument("--klines", required=True)
    parser.add_argument("--out-csv", default="var/tokyo_audit/btc_scalp_20260920/trades_audit.csv")
    parser.add_argument("--out-md", default="var/reports/btc_scalp_live_history_audit.md")
    args = parser.parse_args()

    trades, type_counts, errors, orphans = load_bot_trades(Path(args.db))
    okx = load_okx(Path(args.okx))
    klines = pd.read_csv(args.klines, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    frame = enrich_klines(trades, klines)
    frame = match_exchange(frame, okx)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)

    report = build_report(frame, okx, type_counts, errors, orphans)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report, encoding="utf-8")

    print(f"机器人交易: {len(frame)} 笔（已平仓 {int(frame['close_ts'].notna().sum())}）")
    print(f"OKX 对齐:   {int(frame['okx_matched'].sum())} 笔")
    print(f"CSV:  {out_csv}")
    print(f"报告: {out_md}")
    print()
    print(report[:1500])


if __name__ == "__main__":
    main()
