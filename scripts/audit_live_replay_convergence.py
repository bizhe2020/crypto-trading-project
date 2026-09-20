#!/usr/bin/env python3
"""实盘 / 回测收敛性核对。

README 把 `scripts/audit_live_replay_trade_convergence.py` 列为 replay/live 收敛工作的
入口，但该文件并不存在。本脚本实现它：

    用同一份配置与同一段行情跑回测，把回测产生的每一笔交易与**实盘真实成交**
    （来自 OKX 持仓历史 + action_log）按开仓时间对齐，逐笔比较：
      开仓时间 / 开仓价 / 出场原因 / R 倍数
    并统计回测「多出来」和「漏掉」的交易。

这直接回答「回测赚钱、实盘亏钱」到底差在哪一步。

用法:
    python3 scripts/audit_live_replay_convergence.py \
        --config var/tokyo_audit/btc_scalp_20260920/config.live.high-leverage-structure.json \
        --data-15m var/tokyo_audit/btc_scalp_20260920/BTC_USDT_USDT-15m-futures.feather \
        --data-4h  var/tokyo_audit/btc_scalp_20260920/BTC_USDT_USDT-4h-futures.feather \
        --live-trades var/tokyo_audit/btc_scalp_20260920/trades_audit.csv \
        --start-date 2026-04-01 \
        --out-md var/reports/btc_scalp_live_replay_convergence.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_readiness_report import (  # noqa: E402
    load_config_payload,
    load_prepared_data,
    run_engine,
    trade_dataframe,
)


def replay_trades(config: Path, d15: Path, d4: Path, start_date: str) -> pd.DataFrame:
    """回测口径唯一（无前视，与实盘一致）；口径开关已全部移除。"""
    prepared = load_prepared_data(
        data_15m_path=d15,
        data_4h_path=d4,
        start=pd.Timestamp(start_date, tz="UTC"),
        threshold_payload=None,
    )
    payload = load_config_payload(config)
    _, engine = run_engine(payload, prepared, start_date)
    return trade_dataframe(engine)


def match(live: pd.DataFrame, replay: pd.DataFrame, tolerance_min: float = 30.0) -> pd.DataFrame:
    """按开仓时间做 1:1 贪心匹配。"""
    available = set(replay.index)
    rows = []
    for _, l in live.iterrows():
        if not available:
            rows.append({"live_entry": l["open_ts"], "matched": False})
            continue
        cand = list(available)
        delta = (replay.loc[cand, "entry_time"] - l["open_ts"]).abs()
        idx = delta.idxmin()
        if delta.loc[idx] > pd.Timedelta(minutes=tolerance_min):
            rows.append({"live_entry": l["open_ts"], "matched": False})
            continue
        available.discard(idx)
        r = replay.loc[idx]
        rows.append({
            "live_entry": l["open_ts"],
            "replay_entry": r["entry_time"],
            "matched": True,
            "live_dir": l["direction"],
            "replay_dir": r.get("direction"),
            "live_entry_px": l["entry_signal"],
            "replay_entry_px": r.get("entry_price"),
            "live_exit_reason": l["reason"],
            "replay_exit_reason": r.get("exit_reason"),
            "live_R": l["realized_R"],
            "replay_R": r.get("rr_ratio"),
        })
    out = pd.DataFrame(rows)
    out.attrs["unmatched_replay"] = sorted(available)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-15m", required=True)
    parser.add_argument("--data-4h", required=True)
    parser.add_argument("--live-trades", required=True)
    parser.add_argument("--start-date", default="2026-04-01")
    parser.add_argument("--tolerance-min", type=float, default=30.0)
    parser.add_argument("--out-md", default="var/reports/btc_scalp_live_replay_convergence.md")
    parser.add_argument("--out-csv", default="var/tokyo_audit/btc_scalp_20260920/convergence.csv")
    args = parser.parse_args()

    live = pd.read_csv(args.live_trades)
    live["open_ts"] = pd.to_datetime(live["open_ts"], utc=True)
    live = live[live["stop"].notna()].reset_index(drop=True)

    results = {}
    label = "confirmed（唯一口径，无前视 = 实盘口径）"
    rep = replay_trades(Path(args.config), Path(args.data_15m), Path(args.data_4h),
                        args.start_date)
    results[label] = (rep, match(live, rep, args.tolerance_min))

    lines: list[str] = []
    add = lines.append
    add("# 实盘 / 回测收敛性核对")
    add("")
    add(f"- 实盘交易：**{len(live)} 笔**（{live['open_ts'].min()} → {live['open_ts'].max()}）")
    add(f"- 回测窗口起点：{args.start_date}")
    add(f"- 匹配容差：±{args.tolerance_min:.0f} 分钟")
    add("")
    add("## 总览")
    add("")
    add("| 口径 | 回测笔数 | 匹配上实盘 | 回测多出 | 实盘漏掉 |")
    add("|---|---:|---:|---:|---:|")
    for label, (rep, m) in results.items():
        matched = int(m["matched"].sum())
        extra = len(rep) - matched
        add(f"| {label} | {len(rep)} | {matched} | {extra} | {len(live) - matched} |")
    add("")

    for label, (rep, m) in results.items():
        add(f"## {label}")
        add("")
        if rep.empty:
            add("回测未产生任何交易。")
            add("")
            continue
        add(f"- 回测总 R（rr_ratio 合计）：**{pd.to_numeric(rep.get('rr_ratio'), errors='coerce').sum():.2f}**")
        add(f"- 回测出场原因分布：{rep['exit_reason'].value_counts().to_dict()}")
        add("")
        mm = m[m["matched"]].copy()
        if mm.empty:
            add("**没有任何一笔回测交易与实盘对齐。** 说明回测与实盘的入场逻辑差异极大。")
            add("")
            continue
        mm["live_R"] = pd.to_numeric(mm["live_R"], errors="coerce")
        mm["replay_R"] = pd.to_numeric(mm["replay_R"], errors="coerce")
        mm["entry_delta_min"] = (
            pd.to_datetime(mm["replay_entry"], utc=True) - pd.to_datetime(mm["live_entry"], utc=True)
        ).dt.total_seconds() / 60.0
        mm["entry_px_delta_bp"] = (
            pd.to_numeric(mm["replay_entry_px"], errors="coerce")
            - pd.to_numeric(mm["live_entry_px"], errors="coerce")
        ) / pd.to_numeric(mm["live_entry_px"], errors="coerce") * 1e4
        add("| 指标 | 实盘 | 回测 |")
        add("|---|---:|---:|")
        add(f"| 匹配笔数 | {len(mm)} | {len(mm)} |")
        add(f"| 平均 R | {mm['live_R'].mean():.2f} | {mm['replay_R'].mean():.2f} |")
        add(f"| R 合计 | {mm['live_R'].sum():.2f} | {mm['replay_R'].sum():.2f} |")
        add(f"| 开仓时间差（分钟，中位） | – | {mm['entry_delta_min'].median():.1f} |")
        add(f"| 开仓价差（bp，中位） | – | {mm['entry_px_delta_bp'].median():.1f} |")
        add("")
        add("逐笔对照：")
        add("")
        cols = ["live_entry", "replay_entry", "live_dir", "replay_dir", "live_entry_px",
                "replay_entry_px", "live_exit_reason", "replay_exit_reason", "live_R", "replay_R"]
        table = mm[cols].copy()
        for c in ("live_entry", "replay_entry"):
            table[c] = pd.to_datetime(table[c], utc=True).astype(str).str.slice(0, 16)
        add(table.to_markdown(index=False, floatfmt=",.2f"))
        add("")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [m.assign(口径=label) for label, (_, m) in results.items()],
        ignore_index=True,
    ).to_csv(out_csv, index=False)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
