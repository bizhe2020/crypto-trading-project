#!/usr/bin/env python3
"""canonical replay 事件流的季度切片 / 窗口 / 两半对比（BTC scalp）。

输入是 `scripts/replay_sota_smc_live_shadow.py --paper-log-output` 写出的 JSONL
（每行一个 decision；只统计 decision == "accepted" 的事件）。第一个 --events 是 base，
其余是候选。复利口径与 replay 自身一致：capital *= (1 + return_pct / 100)，按入场顺序。

用法:
    python3 scripts/audit_btc_scalp_replay_quarters.py \
        --events var/tmp/btc_headroom/replay_base_2023.jsonl --label base \
        --events var/tmp/btc_headroom/replay_cand_2023.jsonl --label cand \
        --windows 2026-01-01,2026-07-15,2026-09-10 --split 2025-01-01 \
        --output var/tmp/btc_headroom/replay_quarters.json

季度按 **平仓时间** 归属（与 scripts/scan_btc_scalp_quarters.py 的引擎口径一致）；
窗口按 **入场时间** 过滤（与 replay 自身的 current_year / last_60d 窗口一致）。
某季没有任何事件 = 该季收益 0%（不是缺失），这样放宽类候选新增的季度也会被计入。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def quarter_label(ts: Any) -> str:
    stamp = pd.Timestamp(ts)
    return f"{stamp.year}Q{(stamp.month - 1) // 3 + 1}"


def load_accepted_events(path: Path, event_types: set[str] | None = None) -> list[dict[str, Any]]:
    """读取 accepted 事件；event_types 非空时只保留这些路径（如只看 sota_long）。"""
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("decision") != "accepted":
                continue
            if event_types and str(row.get("event_type") or "") not in event_types:
                continue
            events.append(row)
    events.sort(key=lambda e: (int(e.get("entry_idx", 0) or 0), int(e.get("exit_idx", 0) or 0)))
    return events


def compound_stats(returns_pct: Iterable[float]) -> dict[str, float]:
    """按顺序复利：返回总收益 %、最大回撤 %（峰值从初始资金算起）、笔数。"""
    capital = 1.0
    peak = 1.0
    max_dd = 0.0
    count = 0
    for value in returns_pct:
        capital = max(0.0, capital * (1.0 + float(value) / 100.0))
        peak = max(peak, capital)
        if peak > 0:
            max_dd = max(max_dd, (peak - capital) / peak * 100.0)
        count += 1
    return {"return_pct": (capital - 1.0) * 100.0, "max_dd_pct": max_dd, "trades": float(count)}


def compound_by_quarter(
    events: list[dict[str, Any]],
    time_key: str = "exit_time",
    return_key: str = "return_pct",
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = {}
    for event in events:
        buckets.setdefault(quarter_label(event[time_key]), []).append(float(event[return_key]))
    return {quarter: compound_stats(values) for quarter, values in sorted(buckets.items())}


def window_stats(
    events: list[dict[str, Any]],
    start: str | None = None,
    end: str | None = None,
    time_key: str = "entry_time",
    return_key: str = "return_pct",
) -> dict[str, float]:
    lo = pd.Timestamp(start, tz="UTC") if start else None
    hi = pd.Timestamp(end, tz="UTC") if end else None
    picked: list[float] = []
    for event in events:
        stamp = pd.Timestamp(event[time_key])
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        if lo is not None and stamp < lo:
            continue
        if hi is not None and stamp >= hi:
            continue
        picked.append(float(event[return_key]))
    return compound_stats(picked)


def quarter_concentration(candidate: dict[str, float], base: dict[str, float]) -> dict[str, Any]:
    """候选 vs base 的季度对比（输入都是 {季度: 季度收益%}）。

    - 季度取并集；某一方没有该季 = 0%（该季无交易）。
    - beat = 候选季度收益严格高于 base 的季度数（两边都为 0 的季度不计入分母）。
    - share = 最大单季正超额 / 全部正超额之和；越接近 1 越说明改进集中在单一季度。
    """
    quarters = sorted(set(candidate) | set(base))
    diffs: dict[str, float] = {}
    for quarter in quarters:
        cand_value = float(candidate.get(quarter, 0.0))
        base_value = float(base.get(quarter, 0.0))
        if quarter not in candidate and quarter not in base:
            continue
        diffs[quarter] = cand_value - base_value
    active = {q: d for q, d in diffs.items() if not (candidate.get(q, 0.0) == 0.0 and base.get(q, 0.0) == 0.0)}
    positive = {q: d for q, d in active.items() if d > 1e-9}
    total_positive = sum(positive.values())
    best_quarter = max(positive, key=lambda q: positive[q]) if positive else None
    return {
        "beat_quarters": len(positive),
        "total_quarters": len(active),
        "best_quarter": best_quarter,
        "best_quarter_excess_pct": round(positive[best_quarter], 2) if best_quarter else 0.0,
        "positive_excess_sum_pct": round(total_positive, 2),
        "best_share_of_positive": round(positive[best_quarter] / total_positive, 3) if best_quarter else None,
        "diffs": {q: round(d, 2) for q, d in active.items()},
    }


def r_space_decomposition(base: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    """按 (entry_idx, direction) 配对两条事件流，拆分 R 空间差异（需要事件里有 net_r / gross_r / fee_r）。

    - matched：两边都做了的交易，比较 R 是否改变（出场变化）；
    - removed：只在 base 里（被候选挤掉 / 拦掉）；
    - added：只在候选里（新放行 / 顶替）。
    净 R 差 = matched 的 R 差 + added - removed。改进若只来自换了一批交易，而 matched 不变，
    就不是「选得更准」，而是路径效应。
    """

    def key(event: dict[str, Any]) -> tuple[int, str]:
        return int(event.get("entry_idx", 0) or 0), str(event.get("direction") or "")

    def sums(events: Iterable[dict[str, Any]]) -> dict[str, float]:
        items = [e for e in events if e.get("net_r") is not None]
        return {
            "n": float(len(items)),
            "net_r": round(sum(float(e["net_r"]) for e in items), 3),
            "gross_r": round(sum(float(e["gross_r"]) for e in items), 3),
            "fee_r": round(sum(float(e["fee_r"]) for e in items), 3),
        }

    base_map = {key(e): e for e in base}
    cand_map = {key(e): e for e in candidate}
    matched_keys = sorted(set(base_map) & set(cand_map))
    removed = [base_map[k] for k in sorted(set(base_map) - set(cand_map))]
    added = [cand_map[k] for k in sorted(set(cand_map) - set(base_map))]
    changed = [k for k in matched_keys
               if abs(float(base_map[k].get("net_r") or 0.0) - float(cand_map[k].get("net_r") or 0.0)) > 1e-9]
    matched_delta = sum(float(cand_map[k].get("net_r") or 0.0) - float(base_map[k].get("net_r") or 0.0)
                        for k in matched_keys)
    base_total, cand_total = sums(base), sums(candidate)
    return {
        "base": base_total,
        "candidate": cand_total,
        "net_r_delta": round(cand_total["net_r"] - base_total["net_r"], 3),
        "gross_r_delta": round(cand_total["gross_r"] - base_total["gross_r"], 3),
        "fee_r_delta": round(cand_total["fee_r"] - base_total["fee_r"], 3),
        "matched": {"n": len(matched_keys), "exit_changed": len(changed), "net_r_delta": round(matched_delta, 3)},
        "removed": sums(removed),
        "added": sums(added),
        "added_by_direction": {d: sums([e for e in added if e.get("direction") == d]) for d in ("BULL", "BEAR")},
    }


def summarize_stream(events: list[dict[str, Any]], windows: list[str], split: str | None) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    for event in events:
        key = str(event.get("event_type") or "unknown")
        by_type[key] = by_type.get(key, 0) + 1
    out: dict[str, Any] = {
        "full": compound_stats(float(e["return_pct"]) for e in events),
        "accepted_by_type": by_type,
        "quarters": compound_by_quarter(events),
        "windows": {w: window_stats(events, start=w) for w in windows},
    }
    if split:
        out["halves"] = {
            f"< {split}": window_stats(events, end=split),
            f">= {split}": window_stats(events, start=split),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", action="append", required=True, help="replay paper-log JSONL；第一个是 base")
    parser.add_argument("--label", action="append", default=[], help="与 --events 一一对应的名字")
    parser.add_argument("--windows", default="2026-01-01", help="逗号分隔的窗口起点（按入场时间）")
    parser.add_argument("--split", default="2025-01-01", help="两半切分点；空字符串=不切")
    parser.add_argument("--only-types", default="",
                        help="逗号分隔，只统计这些 event_type（如 sota_long：实盘关闭了 smc_short 时用）")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    labels = list(args.label) + [Path(p).stem for p in args.events[len(args.label):]]
    windows = [w.strip() for w in str(args.windows).split(",") if w.strip()]
    split = str(args.split).strip() or None
    only_types = {t.strip() for t in str(args.only_types).split(",") if t.strip()} or None

    streams = {label: load_accepted_events(Path(path), only_types) for label, path in zip(labels, args.events)}
    summaries = {label: summarize_stream(events, windows, split) for label, events in streams.items()}
    base_label = labels[0]
    base_q = {q: v["return_pct"] for q, v in summaries[base_label]["quarters"].items()}
    def has_r(events: list[dict[str, Any]]) -> bool:
        return bool(events) and all(e.get("net_r") is not None for e in events)

    for label in labels[1:]:
        cand_q = {q: v["return_pct"] for q, v in summaries[label]["quarters"].items()}
        summaries[label]["vs_base"] = quarter_concentration(cand_q, base_q)
        if has_r(streams[base_label]) and has_r(streams[label]):
            summaries[label]["r_decomposition"] = r_space_decomposition(streams[base_label], streams[label])

    quarters = sorted({q for s in summaries.values() for q in s["quarters"]})
    print(f"{'stream':<26}" + "".join(f"{q:>9}" for q in quarters) + f"{'total%':>11}{'dd%':>7}{'n':>5}{'beat':>7}{'share':>7}")
    for label in labels:
        s = summaries[label]
        row = f"{label:<26}"
        for quarter in quarters:
            value = s["quarters"].get(quarter, {}).get("return_pct")
            row += f"{value:>9.1f}" if value is not None else f"{'0':>9}"
        vs = s.get("vs_base")
        beat = f"{vs['beat_quarters']}/{vs['total_quarters']}" if vs else "-"
        share = f"{vs['best_share_of_positive']:.2f}" if vs and vs["best_share_of_positive"] is not None else "-"
        row += f"{s['full']['return_pct']:>11.1f}{s['full']['max_dd_pct']:>7.1f}{int(s['full']['trades']):>5d}{beat:>7}{share:>7}"
        print(row)
    print("\n窗口（按入场时间起算）与两半：")
    for label in labels:
        s = summaries[label]
        parts = [f"{w}: {v['return_pct']:+.2f}% dd{v['max_dd_pct']:.1f} n{int(v['trades'])}" for w, v in s["windows"].items()]
        if split:
            parts += [f"{k}: {v['return_pct']:+.2f}% n{int(v['trades'])}" for k, v in s["halves"].items()]
        print(f"  {label}: " + " | ".join(parts) + f" | by_type={s['accepted_by_type']}")

    decomposed = [label for label in labels[1:] if "r_decomposition" in summaries[label]]
    if decomposed:
        print("\nR 空间拆分（按 entry_idx+direction 配对；R 含手续费、与仓位无关）：")
        for label in decomposed:
            d = summaries[label]["r_decomposition"]
            print(f"  {label}: 净R {d['base']['net_r']:+.2f} -> {d['candidate']['net_r']:+.2f} (Δ{d['net_r_delta']:+.2f}, "
                  f"毛R Δ{d['gross_r_delta']:+.2f}, 费R Δ{d['fee_r_delta']:+.2f}) | matched {d['matched']['n']} "
                  f"(出场变 {d['matched']['exit_changed']}, ΔR {d['matched']['net_r_delta']:+.2f}) | "
                  f"removed {int(d['removed']['n'])} ({d['removed']['net_r']:+.2f}R) | added {int(d['added']['n'])} "
                  f"({d['added']['net_r']:+.2f}R; 多 {d['added_by_direction']['BULL']['net_r']:+.2f} / "
                  f"空 {d['added_by_direction']['BEAR']['net_r']:+.2f})")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")
        print(f"\n输出: {out}")


if __name__ == "__main__":
    main()
