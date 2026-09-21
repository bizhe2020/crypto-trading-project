#!/usr/bin/env python3
"""canonical replay（live-shadow 事件流）vs 实盘真实成交 的逐笔收敛核对。

与 `scripts/audit_live_replay_convergence.py` 的区别：那个脚本比的是**引擎**口径，
而实盘真正跑的是一条更窄的路径（SOTA score gate + structure gate + bucket sizing +
shadow 风控重放）。本脚本用 `replay_sota_smc_live_shadow.py` 产出的
**live-shadow 事件流**（paper log jsonl）去对实盘 21 笔成交，回答：

    回测说 +111%、实盘却是 −37%，差在哪一步？

分解成三块：
  1. **matched**：两边都做了的交易 → 差异 = 执行/配置差
  2. **replay_only**：回测做了、实盘没做 → 差异 = 「错过」的信号
  3. **live_only**：实盘做了、回测没做 → 差异 = 「多出」的交易

用法::

    python3 scripts/audit_btc_scalp_replay_vs_live_events.py \
        --replay-events var/tmp/btc_opt/replay_base_2023.jsonl \
        --live-trades var/tokyo_audit/btc_scalp_20260920/trades_audit.csv \
        --start 2026-04-01 --end 2026-09-20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_REPLAY = ROOT / "var" / "tmp" / "btc_opt" / "replay_base_2023.jsonl"
DEFAULT_LIVE = ROOT / "var" / "tokyo_audit" / "btc_scalp_20260920" / "trades_audit.csv"
DEFAULT_OUTPUT = ROOT / "var" / "tmp" / "btc_opt" / "replay_vs_live.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="canonical replay vs 实盘逐笔收敛核对")
    parser.add_argument("--replay-events", default=str(DEFAULT_REPLAY))
    parser.add_argument("--live-trades", default=str(DEFAULT_LIVE))
    parser.add_argument("--start", default="2026-04-01")
    parser.add_argument("--end", default="2026-09-20")
    parser.add_argument("--tolerance-min", type=float, default=20.0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def load_replay_events(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    frame = frame[frame["decision"] == "accepted"].copy()
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
    frame = frame[(frame["entry_time"] >= start) & (frame["entry_time"] <= end)]
    frame = frame.sort_values("entry_time").reset_index(drop=True)
    frame["replay_return_pct"] = frame["return_pct"].astype(float)
    return frame


def load_live_trades(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_csv(path).rename(columns={"open_ts": "entry_time", "close_ts": "exit_time"})
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
    frame = frame[(frame["entry_time"] >= start) & (frame["entry_time"] <= end)].copy()
    frame["live_return_pct"] = frame["net_pnl"] / frame["capital_at_entry"] * 100.0
    frame["net_r"] = frame["net_pnl"] / frame["risk_amount"]
    return frame.sort_values("entry_time").reset_index(drop=True)


def compound(returns: list[float]) -> float:
    value = 1.0
    for item in returns:
        value *= 1.0 + item / 100.0
    return (value - 1.0) * 100.0


def match_events(replay: pd.DataFrame, live: pd.DataFrame, tolerance_min: float) -> tuple[list, list, list]:
    available = set(live.index)
    matched: list[dict[str, Any]] = []
    replay_only: list[dict[str, Any]] = []
    for _, event in replay.iterrows():
        best_idx = None
        best_gap = None
        for idx in available:
            gap = abs((live.loc[idx, "entry_time"] - event["entry_time"]).total_seconds()) / 60.0
            if gap <= tolerance_min and (best_gap is None or gap < best_gap):
                best_idx, best_gap = idx, gap
        if best_idx is None:
            replay_only.append(
                {
                    "entry_time": str(event["entry_time"]),
                    "event_type": event.get("event_type"),
                    "replay_return_pct": float(event["replay_return_pct"]),
                }
            )
        else:
            available.discard(best_idx)
            matched.append(
                {
                    "entry_time": str(event["entry_time"]),
                    "gap_min": round(best_gap, 1),
                    "event_type": event.get("event_type"),
                    "replay_return_pct": float(event["replay_return_pct"]),
                    "live_return_pct": float(live.loc[best_idx, "live_return_pct"]),
                    "live_net_r": round(float(live.loc[best_idx, "net_r"]), 3),
                    "live_exit_reason": live.loc[best_idx, "reason"],
                }
            )
    live_only = [
        {
            "entry_time": str(live.loc[idx, "entry_time"]),
            "live_return_pct": float(live.loc[idx, "live_return_pct"]),
            "live_net_r": round(float(live.loc[idx, "net_r"]), 3),
            "risk_per_trade": float(live.loc[idx, "risk_per_trade"]),
            "rank": live.loc[idx, "regime_label"] if "regime_label" in live.columns else None,
        }
        for idx in sorted(available)
    ]
    return matched, replay_only, live_only


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    replay = load_replay_events(Path(args.replay_events), start, end)
    live = load_live_trades(Path(args.live_trades), start, end)
    matched, replay_only, live_only = match_events(replay, live, args.tolerance_min)

    replay_all_return = compound(replay["replay_return_pct"].tolist())
    live_all_return = compound(live["live_return_pct"].tolist())
    matched_replay_return = compound([m["replay_return_pct"] for m in matched])
    matched_live_return = compound([m["live_return_pct"] for m in matched])

    report = {
        "window": {"start": args.start, "end": args.end},
        "counts": {
            "replay_events": int(len(replay)),
            "live_trades": int(len(live)),
            "matched": len(matched),
            "replay_only": len(replay_only),
            "live_only": len(live_only),
        },
        "returns_pct": {
            "replay_all": round(replay_all_return, 2),
            "live_all": round(live_all_return, 2),
            "matched_replay": round(matched_replay_return, 2),
            "matched_live": round(matched_live_return, 2),
            "replay_only_contribution": round(compound([e["replay_return_pct"] for e in replay_only]), 2)
            if replay_only
            else 0.0,
            "live_only_contribution": round(compound([e["live_return_pct"] for e in live_only]), 2)
            if live_only
            else 0.0,
        },
        "net_r": {
            "live_all_sum": round(float(live["net_r"].sum()), 2),
            "live_matched_sum": round(sum(m["live_net_r"] for m in matched), 2),
            "live_only_sum": round(sum(e["live_net_r"] for e in live_only), 2),
        },
        "matched": sorted(matched, key=lambda m: m["entry_time"]),
        "replay_only": sorted(replay_only, key=lambda m: m["entry_time"]),
        "live_only": sorted(live_only, key=lambda m: m["entry_time"]),
    }

    print(json.dumps({k: v for k, v in report.items() if k not in ("matched", "replay_only", "live_only")},
                     ensure_ascii=False, indent=2))
    print("\n-- matched（两边都做）--")
    for item in report["matched"]:
        print(
            f"  {item['entry_time'][:16]}  gap={item['gap_min']:>5.1f}m  "
            f"replay={item['replay_return_pct']:>8.2f}%  live={item['live_return_pct']:>8.2f}%  "
            f"({item['live_exit_reason']})"
        )
    print("\n-- replay only（回测做了、实盘没做）--")
    for item in report["replay_only"]:
        print(f"  {item['entry_time'][:16]}  {item['event_type']:<12} replay={item['replay_return_pct']:>8.2f}%")
    print("\n-- live only（实盘做了、回测没做）--")
    for item in report["live_only"]:
        print(
            f"  {item['entry_time'][:16]}  live={item['live_return_pct']:>8.2f}%  R={item['live_net_r']:>6.3f}  "
            f"risk={item['risk_per_trade']}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"\n输出: {output_path}")


if __name__ == "__main__":
    main()
