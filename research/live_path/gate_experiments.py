"""拆解「执行层 shadow 闸门 vs 引擎内 shadow 闸门」差异的对照实验（只用于研究，不改实盘代码）。

在 replay_live_path 的完整实盘链路上，用猴子补丁逐一替换语义：
  exec            实盘现状：引擎先开仓 → 执行层发现暂停 → 回滚（挂起的回踩入场条件已被消耗）
  engine          引擎内闸门：暂停时 open_position 直接返回 None（挂起条件保留，暂停结束后还能入场）；执行层闸门关
  keep_pending    执行层闸门的状态与触发规则不变，但改为在 open_position 处拦截（= 只换「拦截位置」）
  +cooldown_exit  回撤冷却改成「平仓时刻 + N 天」（引擎口径），实盘是「平仓当日 UTC 0 点 + N 天」
同时记录每次暂停的触发原因与暂停时长。
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.live_path.replay_live_path import build, run  # noqa: E402
from scripts.live_readiness_report import trade_dataframe  # noqa: E402
from scripts.scan_btc_scalp_optimization import load_base_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data-15m", required=True)
    p.add_argument("--data-4h", required=True)
    p.add_argument("--end", default=None)
    p.add_argument("--mode", choices=["exec", "engine", "keep_pending", "shadow_hold"], required=True)
    p.add_argument("--cooldown-exit", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    payload = load_base_config(Path(args.config))
    for item in args.set:
        key, raw = item.split("=", 1)
        payload[key] = json.loads(raw)
    if args.mode == "engine":
        payload["enable_shadow_risk_gate_backtest"] = True
    pauses: list[dict] = []
    blocked_at_open = Counter()

    with tempfile.TemporaryDirectory() as tmp:
        if args.mode == "engine":
            ex, engine = build(payload, Path(args.data_15m), Path(args.data_4h), "2022-01-01", Path(tmp) / "s.db",
                               keep_engine_gate=True)
            assert engine.config.enable_shadow_risk_gate, "engine gate not enabled"
            ex._shadow_gate_enabled = lambda: False  # type: ignore[method-assign]
            orig_after = engine._shadow_gate_after_close

            def after(pnl, exit_ts):
                before = float(engine.shadow_state.get("pause_until_ts", 0.0) or 0.0)
                orig_after(pnl, exit_ts)
                now = float(engine.shadow_state.get("pause_until_ts", 0.0) or 0.0)
                if now > before:
                    pauses.append({"t": exit_ts, "until": now, "triggers": ["engine"]})

            engine._shadow_gate_after_close = after  # type: ignore[method-assign]
        else:
            ex, engine = build(payload, Path(args.data_15m), Path(args.data_4h), "2022-01-01", Path(tmp) / "s.db")
            orig_after = ex._shadow_gate_after_close

            def after(action, eng):
                before = float(ex._load_shadow_gate_state(eng).get("pause_until_ts", 0.0) or 0.0)
                orig_after(action, eng)
                st = ex._load_shadow_gate_state(eng)
                now = float(st.get("pause_until_ts", 0.0) or 0.0)
                ev = (st.get("events") or [{}])[-1]
                if now > before:
                    pauses.append({"t": ex._action_timestamp(action).timestamp(), "until": now,
                                   "triggers": [str(x).split(":")[0] for x in ev.get("triggers") or []]})

            ex._shadow_gate_after_close = after  # type: ignore[method-assign]
            if args.cooldown_exit:
                ex._shadow_cooldown_until_ts = lambda dt, days: dt.timestamp() + max(1, int(days)) * 86400.0  # type: ignore[method-assign]
            if args.mode == "shadow_hold":
                # d934c59（2026-07-13）之前的语义：闸门跳过的开仓保留为本地影子仓（不下真单），
                # 影子仓走完自己的止损/止盈，期间引擎不会开别的单；影子仓的平仓被 skip_close 回滚。
                orig_rb = ex._rollback_unexecuted_open

                def rb(action, eng, *, reason):
                    if str(reason).startswith("shadow_gate"):
                        blocked_at_open["shadow_held"] += 1
                        return None
                    return orig_rb(action, eng, reason=reason)

                ex._rollback_unexecuted_open = rb  # type: ignore[method-assign]
            if args.mode == "keep_pending":
                orig_open = engine.open_position

                def gated_open(idx, *a, **k):
                    st = ex._load_shadow_gate_state(engine)
                    if engine.c15m[idx].ts < float(st.get("pause_until_ts", 0.0) or 0.0):
                        blocked_at_open["blocked"] += 1
                        return None
                    return orig_open(idx, *a, **k)

                engine.open_position = gated_open  # type: ignore[method-assign]
        stats = run(ex, engine, None, pd.Timestamp(args.end) if args.end else None, None)

    trades = trade_dataframe(engine)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    trades[["entry_time", "exit_time", "direction", "entry_price", "exit_price", "quantity", "pnl", "gross_pnl", "fees",
            "exit_reason", "capital_at_entry", "notional"]].to_csv(out, index=False)
    paused_days = 0.0
    last_end = 0.0
    for pz in sorted(pauses, key=lambda x: x["t"]):  # 合并重叠区间
        start = max(pz["t"], last_end)
        if pz["until"] > start:
            paused_days += (pz["until"] - start) / 86400.0
            last_end = max(last_end, pz["until"])
    summary = {
        "mode": args.mode, "cooldown_exit": args.cooldown_exit, "trades": int(len(trades)),
        "final_capital": round(float(engine.capital), 2), "pause_events": len(pauses),
        "pause_triggers": dict(Counter(t for pz in pauses for t in pz["triggers"])),
        "paused_days": round(paused_days, 1), "blocked_at_open": blocked_at_open["blocked"],
        "shadow_held": blocked_at_open["shadow_held"], "set": args.set, **stats,
    }
    out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
