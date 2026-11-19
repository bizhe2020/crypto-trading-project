"""BTC scalp 完整实盘链路的历史重放：引擎 + 执行层候选仲裁 + 执行层 shadow 闸门 + 高杠杆守卫。

做法：不重写任何门逻辑，而是离线逐根驱动实盘的 `OkxExecutionEngine`（mode=paper）：
  每根 15m 收盘 i：
    actions = engine.evaluate_range(i, i+1)                      # 与实盘 evaluate_latest 相同
    actions, arb = ex._apply_live_candidate_arbitration(...)     # 打分门 / 结构门 / FVG 召回 / 仲裁
    ex.execute_action(a, engine)                                 # shadow 闸门、sizing、高杠杆守卫、paper 记账
  被拒/跳过的开仓会像实盘一样清掉本地仓位（不占位）；被跳过的平仓回滚 PnL。

与实盘仍有的差异（已知、未建模）：
  * 出场按引擎在 K 线高低点上的触发价成交（实盘是交易所挂单盘中触发，可能有滑点）。
  * 资金按引擎复利（实盘每根用 OKX 总权益同步）。
  * 实盘每根从快照重建引擎；这里引擎常驻内存。
  * 动态高杠杆在实盘配置里是关的 → 分数分桶 sizing / 召回 8x 在实盘与这里都不生效。

输出：
  --output-trades  逐笔 CSV（列与 research/time_in_market/scalp_trades.py 兼容，可直接喂蒙特卡洛）
  --output-bars    逐根决策 JSONL（只写 --bars-from 之后的 K 线，格式对齐 var/tmp/live_path/live_bars.jsonl）
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.okx_executor import ExecutorConfig, OkxExecutionEngine  # noqa: E402
from scripts.confirmed_multiframe_score_utils import (  # noqa: E402
    align_confirmed_mapping,
    resample_confirmed_1h,
    score_snapshot,
)
from scripts.live_readiness_report import load_prepared_data, trade_dataframe  # noqa: E402
from scripts.scan_btc_scalp_optimization import load_base_config  # noqa: E402
from strategy.scalp_robust_v2_core import ActionType, ScalpRobustEngine  # noqa: E402

OPEN_TYPES = {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}


def _summ(items: list[dict] | None) -> list[dict]:
    out = []
    for x in items or []:
        c = x.get("candidate") or {}
        out.append({"ts": c.get("timestamp"), "dir": c.get("direction"), "px": c.get("entry_price"), "reason": x.get("reason")})
    return out


def build(payload: dict, data_15m: Path, data_4h: Path, start: str, db_path: Path,
          keep_engine_gate: bool = False) -> tuple[OkxExecutionEngine, ScalpRobustEngine]:
    payload = dict(payload)
    payload.update(
        mode="paper",
        telegram_enabled=False,
        state_db_path=str(db_path),
        markets_cache_path=str(ROOT / "var/okx/markets_cache.json"),
    )
    if not keep_engine_gate:
        payload.pop("enable_shadow_risk_gate_backtest", None)  # 引擎内闸门关；闸门只在执行层（实盘口径）
    cfg = ExecutorConfig.from_dict(payload)
    ex = OkxExecutionEngine(cfg)
    ex.check_safety()

    prepared = load_prepared_data(data_15m, data_4h, pd.Timestamp(start, tz="UTC"), payload.get("regime_switcher_thresholds"))
    engine = ScalpRobustEngine(prepared.c4h, prepared.c15m, prepared.mapping, prepared.precomputed, cfg.to_scalp_strategy_config())
    engine._regime_switch_cache = {k: (lab, engine._config_for_regime(lab)) for k, lab in prepared.regime_labels.items()}
    engine._regime_feature_cache = dict(prepared.regime_features)

    # 打分快照：实盘每个候选都对整段 15m 重采样 1h（慢）；结果只依赖 idx 之前已收盘的 1h，缓存等价。
    c1h = resample_confirmed_1h(engine.c15m)
    mapping_1h = align_confirmed_mapping(c1h, engine.c15m)
    ex._sota_score_snapshot_for_idx = lambda eng, idx: asdict(score_snapshot(eng, c1h, mapping_1h, int(idx)))  # type: ignore[method-assign]

    # paper 模式下 CLOSE 的 sizing 仍要向交易所查持仓；离线用刚平掉的那笔（engine.trades[-1]）充当交易所持仓。
    contract_size = float(ex._market().get("contractSize") or 1.0)

    def fake_position_state(pos_side: str, reference_price: float | None = None) -> dict:
        tr = engine.trades[-1] if engine.trades else None
        qty = float(tr.quantity) if tr is not None else 0.0
        px = float(reference_price or (tr.exit_price if tr is not None else 0.0))
        return {"contracts": qty / contract_size, "base_amount_btc": qty, "notional_usdt": qty * px}

    ex._fetch_position_state = fake_position_state  # type: ignore[method-assign]
    return ex, engine


def run(ex: OkxExecutionEngine, engine: ScalpRobustEngine, bars_from: pd.Timestamp | None, end_ts: pd.Timestamp | None,
        bars_out) -> dict:
    n = len(engine.c15m)
    start_idx = 100  # 与 run_backtest(start_date=数据起点) 的 start_idx + 100 一致
    stats = {"bars": 0, "raw_opens": 0, "score_rej": 0, "struct_rej": 0, "recalled": 0, "skipped": {}}
    bars_from_s = bars_from.strftime("%Y-%m-%d %H:%M") if bars_from is not None else None
    end_s = end_ts.strftime("%Y-%m-%d %H:%M") if end_ts is not None else None
    t0 = time.time()
    for i in range(start_idx, n - 1):
        t = engine._timestamp_for_idx(i)
        if end_s and t > end_s:
            break
        actions = engine.evaluate_range(i, i + 1)
        raw_opens = [a for a in actions if a.type in OPEN_TYPES]
        arb: dict = {}
        if ex._live_candidate_arbitration_enabled():  # 与 evaluate_latest 相同的开关判断
            actions, arb = ex._apply_live_candidate_arbitration(engine, actions, i)
        execs = []
        for a in actions:
            res = ex.execute_action(a, engine)
            st = str(res.get("status"))
            execs.append(st)
            if st not in {"paper_recorded", "recorded_only"}:
                stats["skipped"][st] = stats["skipped"].get(st, 0) + 1
        stats["bars"] += 1
        stats["raw_opens"] += len(raw_opens)
        stats["score_rej"] += len(arb.get("score_gate_rejected") or [])
        stats["struct_rej"] += len(arb.get("structure_gate_rejected") or [])
        stats["recalled"] += len(arb.get("structure_gate_recalled") or [])
        if bars_out is not None and bars_from_s and t >= bars_from_s:
            bars_out.write(json.dumps({
                "kind": "bar", "t": t,
                "raw_opens": [{"dir": a.direction, "entry": a.entry_price, "stop": a.stop_price} for a in raw_opens],
                "actions": [{"type": a.type.value, "dir": a.direction, "entry": a.entry_price, "exit": a.exit_price,
                             "stop": a.stop_price, "reason": a.reason} for a in actions],
                "exec": execs,
                "decision": arb.get("decision"),
                "selected": (arb.get("selected") or {}).get("timestamp") if arb.get("selected") else None,
                "score_rej": _summ(arb.get("score_gate_rejected")),
                "struct_rej": _summ(arb.get("structure_gate_rejected")),
                "recalled": _summ(arb.get("structure_gate_recalled")),
                "position_open": engine.position is not None,
                "capital": engine.capital,
            }, ensure_ascii=False) + "\n")
        if i % 20000 == 0:
            print(f"  {t}  trades={len(engine.trades)}  cap={engine.capital:.1f}  {time.time() - t0:.0f}s", flush=True)
    stats["open_position_at_end"] = engine.position is not None
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data-15m", required=True)
    p.add_argument("--data-4h", required=True)
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=None, help="最后一根要处理的 15m（UTC，含）")
    p.add_argument("--bars-from", default=None, help="从这根起写逐根决策 JSONL")
    p.add_argument("--output-trades", required=True)
    p.add_argument("--output-bars", default=None)
    p.add_argument("--no-arbitration", action="store_true", help="对照：关掉执行层仲裁（打分/结构/召回）")
    p.add_argument("--no-exec-shadow-gate", action="store_true", help="对照：关掉执行层 shadow 闸门")
    p.add_argument("--no-high-leverage-guard", action="store_true", help="对照：关掉高杠杆守卫")
    p.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="覆盖配置项（值按 JSON 解析），可多次")
    args = p.parse_args()

    payload = load_base_config(Path(args.config))
    for item in args.set:
        key, raw = item.split("=", 1)
        payload[key] = json.loads(raw)
    if args.no_arbitration:
        payload["enable_live_candidate_arbitration"] = False
    if args.no_exec_shadow_gate:
        payload["enable_shadow_risk_gate"] = False
    if args.no_high_leverage_guard:
        payload["enable_high_leverage_guard"] = False

    out_trades = Path(args.output_trades)
    out_trades.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        ex, engine = build(payload, Path(args.data_15m), Path(args.data_4h), args.start, Path(tmp) / "state.db")
        bars_out = open(args.output_bars, "w") if args.output_bars else None
        try:
            stats = run(ex, engine, pd.Timestamp(args.bars_from) if args.bars_from else None,
                        pd.Timestamp(args.end) if args.end else None, bars_out)
        finally:
            if bars_out is not None:
                bars_out.close()

    trades = trade_dataframe(engine)
    keep = [c for c in ("entry_time", "exit_time", "direction", "entry_price", "exit_price", "quantity", "pnl", "gross_pnl",
                        "fees", "exit_reason", "capital_at_entry", "notional", "candidate_event_type",
                        "initial_stop_price", "risk_regime", "regime_label", "trail_style") if c in trades.columns]
    trades[keep].to_csv(out_trades, index=False)
    final = float(engine.capital)
    summary = {
        "trades": int(len(trades)),
        "initial_capital": float(engine.config.initial_capital) if hasattr(engine.config, "initial_capital") else None,
        "final_capital": round(final, 2),
        "win_rate": round(float((trades["pnl"] > 0).mean()), 4) if len(trades) else None,
        "flags": {"no_arbitration": args.no_arbitration, "no_exec_shadow_gate": args.no_exec_shadow_gate,
                  "no_high_leverage_guard": args.no_high_leverage_guard, "set": args.set},
        **stats,
    }
    out_trades.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
