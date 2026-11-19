"""BTC scalp 冻结参数的前向测试（每月跑一次，不需要实盘运行）。

背景（research/execution/scalp_keep_or_retire.py，2026-11-20）：真实执行口径下 edge 几乎全部来自 2023–24，
其余样本（2020-21 OOS、2022、2025、2026、实盘）≈0 或为负 → scalp 停止实盘，改为前向测试。
参数冻结在 FREEZE_DATE 当天的实盘模板（config/config.live.high-leverage-structure.template.json）；
只有冻结日之后开仓的单才算真正的样本外证据。

流程：
  1. 从 OKX 公共接口把 BTC 15m / 4h 补到最新已收盘 K 线（存 var/tmp/forward/，首次从 var/tmp/live_path/ 复制种子）。
  2. 用 research/live_path/replay_live_path.py 跑完整实盘链路（仲裁 + 高杠杆守卫，不带执行层闸门 → 交易集合与仓位无关），
     从 --replay-start 起（默认 2025-01-01，留足日线 SMA200 预热）。
  3. 冻结日之后的逐笔换成真实执行口径（信号收盘后第一根 1m 开盘 ±1bp、taker），算 R、t。
  4. 对照重启门槛：样本外 ≥ MIN_TRADES 笔且 mean R ≥ MIN_MEAN_R。另外报告与冻结日前重放的重合度（口径自检）。

用法：python3 research/execution/scalp_forward_test.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.execution.maker_entry_study import TAKER, entry_variants, fetch_window  # noqa: E402

FREEZE_DATE = "2026-11-20"
MIN_TRADES = 50
MIN_MEAN_R = 0.10
CONFIG = ROOT / "config/config.live.high-leverage-structure.template.json"
WORK = ROOT / "var/tmp/forward"
SEED = ROOT / "var/tmp/live_path"
REFERENCE = ROOT / "var/tmp/struct/nogate.csv"  # 2022 起完整重放，用于口径自检
BARS = {"15m": ("15m", 15 * 60_000), "4h": ("4H", 4 * 3_600_000)}
URL = "https://www.okx.com/api/v5/market/history-candles?instId=BTC-USDT-SWAP&bar={bar}&limit=100&after={after}"


def _get(url: str) -> list:
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read())
            if payload.get("code") == "0":
                return payload["data"]
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0 + attempt)
    raise RuntimeError(f"OKX 请求失败：{url}")


def update_candles(tf: str) -> Path:
    """补到最新已收盘 K 线；最后 3 根用接口数据覆盖（种子文件末根可能是未收盘 bar）。"""
    bar, step = BARS[tf]
    path = WORK / f"BTC_USDT_USDT-{tf}-futures.feather"
    if not path.exists():
        shutil.copy(SEED / path.name, path)
    df = pd.read_feather(path)
    keep_until = df.date.iloc[-1] - pd.Timedelta(milliseconds=3 * step)
    df = df[df.date <= keep_until]
    stop_ms = int(keep_until.timestamp() * 1000)
    rows, after = [], int(time.time() * 1000) + step
    while True:
        data = _get(URL.format(bar=bar, after=after))
        if not data:
            break
        rows += [r for r in data if r[8] == "1"]  # 只要已收盘
        after = int(data[-1][0])
        if after <= stop_ms:
            break
        time.sleep(0.12)
    new = pd.DataFrame(
        [(pd.Timestamp(int(r[0]), unit="ms", tz="UTC"), *map(float, (r[1], r[2], r[3], r[4], r[6]))) for r in rows],
        columns=["date", "open", "high", "low", "close", "volume"],
    )
    new = new[new.date > keep_until]
    out = pd.concat([df, new]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    tail = out[out.date >= keep_until]
    if (tail.date.diff().dropna() != pd.Timedelta(milliseconds=step)).any():
        raise RuntimeError(f"{tf} 补数据后有缺口")
    out.to_feather(path)
    print(f"  {tf}: {out.date.iloc[0]} → {out.date.iloc[-1]}（新增 {len(new)} 根）", flush=True)
    return path


def realistic_R(trades: pd.DataFrame, cache: Path) -> pd.Series:
    out = []
    for _, tr in trades.iterrows():
        risk = abs(tr.entry_price - tr.initial_stop_price) * tr.quantity
        close_ms = int(pd.Timestamp(tr.entry_time).timestamp() * 1000) + 15 * 60_000
        ev = entry_variants(tr, fetch_window(close_ms, cache), [5], 5.0)
        if not ev:  # 1m 窗口不完整（信号刚发生）：保留原口径
            out.append(tr.pnl / risk)
            continue
        price, fee, _ = ev["market"]
        s = 1.0 if tr.direction == "BULL" else -1.0
        pnl = tr.pnl + s * tr.quantity * (tr.entry_price - price) + tr.entry_price * tr.quantity * TAKER - tr.quantity * price * fee
        out.append(pnl / risk)
    return pd.Series(out, index=trades.index, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-start", default="2025-01-01")
    ap.add_argument("--skip-fetch", action="store_true")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    print("1) 补 K 线", flush=True)
    paths = {tf: (WORK / f"BTC_USDT_USDT-{tf}-futures.feather") if args.skip_fetch else update_candles(tf) for tf in BARS}

    print("2) 冻结参数重放（不带执行层闸门）", flush=True)
    trades_csv = WORK / f"replay_{stamp}.csv"
    subprocess.run(
        [sys.executable, str(ROOT / "research/live_path/replay_live_path.py"),
         "--config", str(CONFIG), "--data-15m", str(paths["15m"]), "--data-4h", str(paths["4h"]),
         "--start", args.replay_start, "--no-exec-shadow-gate", "--output-trades", str(trades_csv)],
        check=True, cwd=ROOT,
    )
    tr = pd.read_csv(trades_csv, parse_dates=["entry_time", "exit_time"])
    for col in ("entry_time", "exit_time"):
        if tr[col].dt.tz is None:
            tr[col] = tr[col].dt.tz_localize("UTC")

    print("3) 结果", flush=True)
    report: dict = {"run_date": stamp, "freeze_date": FREEZE_DATE, "config": str(CONFIG.relative_to(ROOT)),
                    "data_end": str(pd.read_feather(paths["15m"]).date.iloc[-1])}
    if REFERENCE.exists():  # 口径自检：同一段时间与 2022 起重放的入场时间重合度
        ref = pd.read_csv(REFERENCE, parse_dates=["entry_time"])
        lo, hi = pd.Timestamp("2025-03-01", tz="UTC"), ref.entry_time.max()
        a = set(tr.entry_time[(tr.entry_time >= lo) & (tr.entry_time <= hi)].astype(str))
        b = set(ref.entry_time[(ref.entry_time >= lo) & (ref.entry_time <= hi)].astype(str))
        report["selfcheck_overlap"] = round(len(a & b) / max(len(a | b), 1), 3)
        print(f"  自检：2025-03 起与 2022 起重放的入场重合度 {report['selfcheck_overlap']:.0%}（{len(a & b)}/{len(a | b)}）")
    fwd = tr[tr.entry_time >= pd.Timestamp(FREEZE_DATE, tz="UTC")].reset_index(drop=True)
    R = realistic_R(fwd, ROOT / "var/tmp/execution/1m") if len(fwd) else pd.Series(dtype=float)
    n = len(R)
    mean = float(R.mean()) if n else None
    t = float(R.mean() / R.std(ddof=1) * np.sqrt(n)) if n > 1 else None
    report.update(n=n, mean_R=mean, t=t, sum_R=float(R.sum()) if n else 0.0,
                  win_rate=float((R > 0).mean()) if n else None,
                  trades=[{"entry": str(e), "dir": d, "R": round(float(r), 3)} for e, d, r in zip(fwd.entry_time, fwd.direction, R)])
    if n < MIN_TRADES:
        verdict = f"继续观察：样本外 {n}/{MIN_TRADES} 笔"
    elif mean >= MIN_MEAN_R:
        verdict = f"达到重启门槛（mean R {mean:+.3f} ≥ {MIN_MEAN_R}）→ 可考虑带 maker 入场重新上线"
    else:
        verdict = f"未达门槛（mean R {mean:+.3f} < {MIN_MEAN_R}）→ 维持停用"
    report["verdict"] = verdict
    mean_txt = f"{mean:+.3f}" if mean is not None else "n/a"
    t_txt = f"{t:+.2f}" if t is not None else "n/a"
    print(f"  冻结日 {FREEZE_DATE} 之后：{n} 笔，mean R {mean_txt}，t {t_txt}")
    print(f"  结论：{verdict}")
    out = WORK / f"report_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"  报告：{out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
