"""BTC scalp 入场执行研究：市价单（现状）vs post-only 挂单（maker），用真实 1m K 线判断能否成交。

背景：实盘 482 笔 BTC 成交全部是 taker（5bp），账户 maker 费率 2bp。不带闸门的完整链路 236 笔里，
手续费占毛利约 1/3（每笔 0.088R / 毛 0.268R）。出场 219/236 是止损触发（必须 taker），止盈只有 16 笔，
所以唯一可省的是入场腿。

口径（只改入场价与入场费，止损/出场价与数量不变 → 交易集合不变）：
  * 信号价 P = 信号 15m K 线收盘（= 该 15m 内最后一根 1m 的收盘），C = 收盘时刻。
  * backtest  ：回测假设，P ×(1 ± slip_bps)，taker。
  * market    ：实盘近似，C 后第一根 1m 开盘价 ×(1 ± 1bp)，taker（实盘成交在收盘后 40~100 秒）。
  * maker_W   ：C 时挂 post-only 限价 = P。只有窗口 [C, C+W) 内某根 1m 「穿价」（多：low < P；空：high > P）
                才算成交（保守：不假设排队靠前），成交价 P，maker 费。
                未成交 → chase：窗口末根 1m 收盘 ×(1 ± 1bp) 市价追，taker；或 skip：放弃这笔。
  * 逐笔收益 ret = pnl / capital_at_entry（不带闸门的逐笔与仓位无关，可离线精确复利）。

用法：
  python3 research/execution/maker_entry_study.py \
      --trades var/tmp/struct/nogate.csv --trades var/tmp/struct/oos_nogate.csv \
      --cache var/tmp/execution/1m --output var/tmp/execution/maker_entry.json
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

OKX = "https://www.okx.com/api/v5/market/history-candles?instId=BTC-USDT-SWAP&bar=1m&limit=100&after={after}"
TAKER, MAKER = 0.0005, 0.0002


def fetch_window(close_ms: int, cache: Path) -> pd.DataFrame:
    """[C-1m, C+30m] 的 1m K 线（ts = 开盘时间，ms）。"""
    path = cache / f"{close_ms}.json"
    if path.exists():
        rows = json.loads(path.read_text())
    else:
        url = OKX.format(after=close_ms + 31 * 60_000)
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    payload = json.loads(resp.read())
                if payload.get("code") == "0":
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0 + attempt)
        else:
            raise RuntimeError(f"fetch failed {close_ms}")
        rows = payload["data"]
        path.write_text(json.dumps(rows))
        time.sleep(0.12)  # 公共接口限速 20 次/2 秒
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v", "vc", "vq", "confirm"]).astype(float)
    df["ts"] = df["ts"].astype("int64")
    return df[(df.ts >= close_ms - 60_000) & (df.ts <= close_ms + 30 * 60_000)].sort_values("ts").reset_index(drop=True)


def entry_variants(tr: pd.Series, bars: pd.DataFrame, windows: list[int], slip_bps: float) -> dict:
    s = 1.0 if tr.direction == "BULL" else -1.0
    close_ms = int(pd.Timestamp(tr.entry_time).timestamp() * 1000) + 15 * 60_000
    last = bars[bars.ts == close_ms - 60_000]
    after = bars[bars.ts >= close_ms]
    if last.empty or len(after) < max(windows):
        return {}
    P = float(last.c.iloc[0])
    out = {"P": P, "P_vs_csv_bp": (tr.entry_price / (P * (1 + s * slip_bps / 1e4)) - 1) * 1e4}
    out["backtest"] = (P * (1 + s * slip_bps / 1e4), TAKER, True)
    out["market"] = (float(after.o.iloc[0]) * (1 + s * 1e-4), TAKER, True)
    for w in windows:
        win = after.iloc[:w]
        filled = bool((win.l < P).any()) if s > 0 else bool((win.h > P).any())
        if filled:
            out[f"maker{w}_chase"] = (P, MAKER, True)
            out[f"maker{w}_skip"] = (P, MAKER, True)
        else:
            out[f"maker{w}_chase"] = (float(win.c.iloc[-1]) * (1 + s * 1e-4), TAKER, False)
            out[f"maker{w}_skip"] = (None, None, False)
    return out


def summarize(df: pd.DataFrame, pnl: pd.Series, keep: pd.Series) -> dict:
    d = df[keep]
    p = pnl[keep]
    R = p / d.risk
    ret = p / d.capital_at_entry
    eq = np.cumprod(1 + ret.values)
    mdd = float((1 - eq / np.maximum.accumulate(eq)).max()) if len(eq) else 0.0
    yrs = max((d.exit_time.max() - d.entry_time.min()).days / 365.25, 1e-9)
    f25 = d.exit_time >= pd.Timestamp("2025-01-01", tz="UTC")
    return {
        "n": int(len(d)),
        "mean_R": round(float(R.mean()), 4),
        "t": round(float(R.mean() / R.std() * np.sqrt(len(R))), 2) if len(R) > 1 else None,
        "mean_R_2025+": round(float(R[f25].mean()), 4) if f25.any() else None,
        "final_x": round(float(eq[-1]), 3) if len(eq) else None,
        "cagr_pct": round((float(eq[-1]) ** (1 / yrs) - 1) * 100, 1) if len(eq) else None,
        "mdd_pct": round(mdd * 100, 1),
    }


def study(df: pd.DataFrame, cache: Path, windows: list[int], slip_bps: float) -> dict:
    df = df.copy()
    df["risk"] = (df.entry_price - df.initial_stop_price).abs() * df.quantity
    variants: dict[str, list] = {}
    meta = []
    for i, tr in df.iterrows():
        close_ms = int(pd.Timestamp(tr.entry_time).timestamp() * 1000) + 15 * 60_000
        ev = entry_variants(tr, fetch_window(close_ms, cache), windows, slip_bps)
        meta.append(ev.get("P_vs_csv_bp"))
        for name, val in ev.items():
            if isinstance(val, tuple):
                variants.setdefault(name, [None] * len(df))[i] = val
    ok = pd.Series([m is not None for m in meta], index=df.index)
    s = np.where(df.direction == "BULL", 1.0, -1.0)
    orig_entry_cost = df.entry_price * df.quantity * TAKER  # 回测里的入场费
    res: dict = {"n_trades": int(len(df)), "n_with_1m": int(ok.sum()),
                 "P_vs_csv_bp_median": round(float(np.nanmedian([m for m in meta if m is not None])), 3)}
    res["original_csv"] = summarize(df, df.pnl, ok)
    for name, vals in variants.items():
        price = pd.Series([v[0] if v else np.nan for v in vals], index=df.index, dtype=float)
        fee = pd.Series([v[1] if v and v[1] is not None else np.nan for v in vals], index=df.index, dtype=float)
        filled = pd.Series([bool(v[2]) if v else False for v in vals], index=df.index)
        keep = ok & price.notna()
        # Δpnl：入场价差 + 入场费差（数量不变，出场腿不变）
        new_pnl = df.pnl + s * df.quantity * (df.entry_price - price) + orig_entry_cost - df.quantity * price * fee
        row = summarize(df, new_pnl, keep)
        if name.startswith("maker"):
            row["fill_rate"] = round(float(filled[ok].mean()), 3)
            R0 = df.pnl / df.risk
            row["orig_R_filled"] = round(float(R0[ok & filled].mean()), 3)
            row["orig_R_unfilled"] = round(float(R0[ok & ~filled].mean()), 3) if (ok & ~filled).any() else None
        res[name] = row
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", action="append", required=True)
    ap.add_argument("--cache", default="var/tmp/execution/1m")
    ap.add_argument("--windows", default="1,3,5,15")
    ap.add_argument("--slippage-bps", type=float, default=5.0, help="回测入场滑点假设（实盘配置 slippage_bps）")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    windows = [int(x) for x in args.windows.split(",")]
    out = {}
    for path in args.trades:
        df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
        for col in ("entry_time", "exit_time"):
            if df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize("UTC")
        out[path] = study(df, cache, windows, args.slippage_bps)
        print(path)
        for k, v in out[path].items():
            print(f"  {k:16s} {v}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
