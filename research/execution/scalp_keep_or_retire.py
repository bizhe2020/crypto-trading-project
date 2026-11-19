"""scalp 留还是停：把所有独立样本放在一起看真实 edge（2026-11-20）。

样本（R = 净盈亏 / 计划风险金额）：
  * IS   ：2022-01 → 2026-10 完整链路重放（不带闸门），参数就是在这段上调出来的 → 有向上偏差。
  * OOS  ：2020-08 → 2021-12 同一链路重放，参数从未见过。
  * LIVE ：实盘 action_log 的 22 笔平仓（2026-04 → 11）；8/22 起的 10 笔（8/23 参数定型后）是真正的样本外。
重放两段都换成「真实执行」：信号收盘后第一根 1m 开盘 ±1bp、taker（maker_entry_study 的 market 口径），
以及 maker 挂单 5 分钟未成交追市价（maker5_chase）。
输出：各段 mean R / t、合并后的块自助置信区间，以及当前 0.25 倍风险预算下的年化贡献区间。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.execution.maker_entry_study import TAKER, entry_variants, fetch_window  # noqa: E402

# 实盘逐笔 (信号时间, 净盈亏, 计划风险)，来自服务器 state/runtime_high_leverage_structure_live.db 的 OPEN/CLOSE 配对
LIVE = [
    ("2026-04-29 12:00", -327.52, 731.66), ("2026-05-02 06:45", 352.56, 711.28),
    ("2026-05-04 09:15", -735.22, 731.75), ("2026-05-04 21:15", 604.33, 686.19),
    ("2026-05-09 21:00", -1127.06, 693.01), ("2026-05-19 10:45", 28.44, 270.13),
    ("2026-05-25 16:45", -1509.78, 575.34), ("2026-07-10 07:45", -42.92, 234.63),
    ("2026-07-14 19:00", -300.46, 233.38), ("2026-07-17 21:45", 53.25, 226.07),
    ("2026-08-21 04:00", 3350.34, 549.89), ("2026-08-21 14:30", 2049.71, 780.57),
    ("2026-08-22 18:15", -1858.35, 1784.19), ("2026-08-25 05:15", -1770.77, 1524.11),
    ("2026-08-27 14:30", -107.78, 1305.07), ("2026-08-31 21:30", -1354.25, 1288.66),
    ("2026-09-03 19:15", -1169.42, 1153.44), ("2026-09-06 03:00", -1121.98, 996.15),
    ("2026-09-09 12:30", -1170.94, 865.76), ("2026-09-14 13:45", -31.57, 307.73),
    ("2026-09-18 10:45", 1971.31, 734.29), ("2026-11-05 00:30", -103.39, 91.53),
]
MEAN_RISK_FRAC_1X = 0.0923  # sizing_nogate.json：现行倾斜 1.0 倍时每笔平均风险占权益
SCALE_NOW = 0.25


def replay_R(path: str, variant: str, cache: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    df["risk"] = (df.entry_price - df.initial_stop_price).abs() * df.quantity
    out = []
    for _, tr in df.iterrows():
        close_ms = int(pd.Timestamp(tr.entry_time).timestamp() * 1000) + 15 * 60_000
        ev = entry_variants(tr, fetch_window(close_ms, cache), [5], 5.0)
        price, fee, _ = ev[variant]
        s = 1.0 if tr.direction == "BULL" else -1.0
        pnl = tr.pnl + s * tr.quantity * (tr.entry_price - price) + tr.entry_price * tr.quantity * TAKER - tr.quantity * price * fee
        out.append(pnl / tr.risk)
    return pd.Series(out, index=pd.to_datetime(df.entry_time, utc=True))


def describe(name: str, r: pd.Series) -> None:
    t = r.mean() / r.std(ddof=1) * np.sqrt(len(r)) if len(r) > 1 else float("nan")
    print(f"  {name:34s} n={len(r):3d}  mean R {r.mean():+.3f}  t {t:+.2f}  win {(r > 0).mean():.0%}")


def boot_ci(r: np.ndarray, n_boot: int = 20000, block: int = 5, seed: int = 7) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(r)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = []
        while len(idx) < n:
            st = rng.integers(0, n)
            idx.extend(range(st, min(st + block, n)))
        means[b] = r[np.array(idx[:n])].mean()
    return float(np.percentile(means, 5)), float(np.percentile(means, 95)), float((means <= 0).mean())


def main() -> None:
    cache = ROOT / "var/tmp/execution/1m"
    live = pd.Series([p / k for _, p, k in LIVE], index=pd.to_datetime([t for t, _, _ in LIVE], utc=True))
    for variant in ("market", "maker5_chase"):
        is_r = replay_R(str(ROOT / "var/tmp/struct/nogate.csv"), variant, cache)
        oos_r = replay_R(str(ROOT / "var/tmp/struct/oos_nogate.csv"), variant, cache)
        print(f"\n=== 执行口径：{variant} ===")
        describe("IS 2022-01→2026-10（调参样本）", is_r)
        describe("IS 只看 2025+", is_r[is_r.index >= "2025-01-01"])
        describe("OOS 2020-08→2021-12", oos_r)
        describe("LIVE 全部 22 笔（实盘，taker）", live)
        describe("LIVE 8/22 起 10 笔（定参后）", live[live.index >= "2026-08-22"])
        for y in (2022, 2023, 2024, 2025, 2026):
            describe(f"IS {y}", is_r[is_r.index.year == y])
        honest = pd.concat([oos_r, is_r[~is_r.index.year.isin([2023, 2024])]]).sort_index()
        pooled = pd.concat([oos_r, is_r]).sort_index()
        for name, r in (("OOS + IS 全部", pooled), ("OOS + IS 去掉 2023–24", honest)):
            lo, hi, p0 = boot_ci(r.to_numpy())
            describe(name, r)
            print(f"      块自助 90% 区间 [{lo:+.3f}, {hi:+.3f}]，P(mean R ≤ 0) = {p0:.0%}")
        n_year = len(is_r) / ((is_r.index[-1] - is_r.index[0]).days / 365.25)
        risk = MEAN_RISK_FRAC_1X * SCALE_NOW
        sd = pooled.std()
        print(f"  当前 0.25 倍：每笔约 {risk*100:.1f}% 权益风险 × 每年 {n_year:.0f} 笔；波动拖累约 {0.5*n_year*(risk*sd)**2*100:.1f}%/年")
        for mR in (-0.10, 0.0, 0.05, 0.10, 0.17):
            print(f"    若真实 mean R = {mR:+.2f} → 年化贡献约 {(n_year*risk*mR - 0.5*n_year*(risk*sd)**2)*100:+.1f}%")


if __name__ == "__main__":
    main()
