"""shadow 闸门有没有「系统性」价值：在打乱后的逐笔收益序列上比较 有闸门 vs 无闸门。

历史只有一条路径，闸门挡掉哪几笔大赚单几乎决定了结果（参数网格非单调）。这里：
  * 取无闸门的完整链路逐笔（仲裁 + 高杠杆守卫），保留每笔的开/平仓时间（交易节奏不变），
  * 把收益在时间槽之间重新分配（iid 打乱 / 按块打乱保留连亏聚集），
  * 在每条路径上按实盘执行层语义跑闸门：单日亏损 ≥6% 停到次日 UTC 0 点、连亏 4 笔停到次日、
    回撤 ≥12%（触发后峰值重置）停到「平仓当日 UTC 0 点 + 冷却天数」；暂停期间开仓的笔被跳过。
近似：被跳过的单不会腾出仓位让引擎开别的单（真实重放里会多出约 10 笔）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86400.0


def gate_path(entry_ts, exit_ts, r, *, daily=6.0, streak=4, dd=12.0, cooldown=2, peak_reset=True):
    cap, peak, pause, ls = 1.0, 1.0, 0.0, 0
    day_start: dict[int, float] = {}
    day_pnl: dict[int, float] = {}
    eq = [1.0]
    taken = 0
    for e, x, ret in zip(entry_ts, exit_ts, r):
        if e < pause:
            continue
        taken += 1
        pnl = cap * ret
        before = cap
        cap += pnl
        eq.append(cap)
        if cap <= 0:
            break
        peak = max(peak, cap)
        d = int(x // DAY)
        if d not in day_start:
            day_start[d], day_pnl[d] = before, 0.0
        day_pnl[d] += pnl
        next_day = (d + 1) * DAY
        ls = 0 if pnl > 0 else ls + 1
        if daily > 0 and -day_pnl[d] / day_start[d] * 100 >= daily:
            pause = max(pause, next_day)
        if streak > 0 and ls >= streak:
            pause = max(pause, next_day)
            ls = 0
        if dd > 0 and (peak - cap) / peak * 100 >= dd:
            pause = max(pause, d * DAY + max(1, cooldown) * DAY)
            if peak_reset:
                peak = cap
            ls = 0
    eq = np.array(eq)
    return eq[-1], float((1 - eq / np.maximum.accumulate(eq)).max()), taken


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trades", required=True, help="无闸门的完整链路逐笔 CSV")
    p.add_argument("--paths", type=int, default=3000)
    p.add_argument("--block", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    t = pd.read_csv(args.trades, parse_dates=["entry_time", "exit_time"]).sort_values("entry_time").reset_index(drop=True)
    e = t["entry_time"].astype("int64").to_numpy() / 1e9
    x = t["exit_time"].astype("int64").to_numpy() / 1e9
    r = (t["pnl"] / t["capital_at_entry"]).to_numpy()
    variants = {
        "no_gate": dict(daily=0, streak=0, dd=0),
        "live": dict(),
        "daily+streak": dict(dd=0),
        "dd_only": dict(daily=0, streak=0),
        "dd12_cd1": dict(cooldown=1),
        "dd20": dict(dd=20),
    }
    rng = np.random.default_rng(args.seed)
    n = len(r)
    out: dict = {"historical_order": {k: gate_path(e, x, r, **v)[:2] for k, v in variants.items()}}
    for scheme in ("iid", "block"):
        res = {k: [] for k in variants}
        for _ in range(args.paths):
            if scheme == "iid":
                rr = rng.permutation(r)
            else:
                nb = int(np.ceil(n / args.block))
                blocks = [r[i * args.block:(i + 1) * args.block] for i in range(nb)]
                rr = np.concatenate([blocks[j] for j in rng.permutation(nb)])[:n]
            for k, v in variants.items():
                res[k].append(gate_path(e, x, rr, **v)[:2])
        summ = {}
        base = np.array(res["no_gate"])
        for k, vals in res.items():
            a = np.array(vals)
            lf, mdd = np.log(np.clip(a[:, 0], 1e-9, None)), a[:, 1]
            summ[k] = {
                "final_median": round(float(np.median(a[:, 0])), 2),
                "final_p10": round(float(np.percentile(a[:, 0], 10)), 2),
                "mdd_median_pct": round(float(np.median(mdd)) * 100, 1),
                "mdd_p90_pct": round(float(np.percentile(mdd, 90)) * 100, 1),
                "p_beats_no_gate_final": round(float((a[:, 0] > base[:, 0]).mean()), 3),
                "p_lower_mdd_than_no_gate": round(float((mdd < base[:, 1]).mean()), 3),
                "median_log_diff_vs_no_gate": round(float(np.median(lf - np.log(np.clip(base[:, 0], 1e-9, None)))), 3),
            }
        out[scheme] = summ
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
