#!/usr/bin/env python3
"""黄金日线信号生成器：MA50 > MA100 金叉做多。

产出 gold_daily_signal.csv（列 date,position,leverage_tier,target_leverage），
对齐 GOOGL 信号 CSV 结构，供 GoldUsdtSignalAdapter 转成 router 候选。

信号规则（回测定档，docs/dual_strategy_parallel_plan.md）：
    MA50 > MA100 → position=GOLD, leverage_tier=base, target_leverage=4.0
    否则         → position=FLAT, leverage_tier=flat, target_leverage=0.0

数据源：XAU-USDT-SWAP 日线（OKX），由服务器直接拉取。本地测试可 --input 指定已有 CSV。

用法:
    python scripts/scan_gold_daily_signal.py --out var/runtime/gold/gold_daily_signal.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MA_FAST = 50
MA_SLOW = 100
LEVERAGE = 4.0
OKX_CANDLES = "https://www.okx.com/api/v5/market/candles"


def fetch_okx_daily(inst_id: str = "XAU-USDT-SWAP") -> pd.DataFrame:
    """从 OKX 拉 XAU-USDT-SWAP 日线（分页，最多 1000 根）。"""
    rows: list[list[str]] = []
    after: str | None = None
    for _ in range(5):
        params = {"instId": inst_id, "bar": "1D", "limit": "300"}
        if after:
            params["after"] = after
        r = requests.get(OKX_CANDLES, params=params, timeout=25)
        data = r.json().get("data", [])
        if not data:
            break
        rows.extend(data)
        after = str(int(data[-1][0]) - 1)
        if len(data) < 300:
            break
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "vol", "volCcy", "volCcyQuote", "confirm"])
    df = df.drop_duplicates("ts")
    df["date"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True).dt.normalize()
    df["close"] = df["c"].astype(float)
    return df.sort_values("date")[["date", "close"]].reset_index(drop=True)


def load_daily(input_path: Path | None) -> pd.DataFrame:
    if input_path and input_path.exists():
        df = pd.read_csv(input_path)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        return df[["date", "close"]].sort_values("date").reset_index(drop=True)
    return fetch_okx_daily()


def generate_signal(daily: pd.DataFrame) -> pd.DataFrame:
    c = daily["close"]
    ma_fast = c.rolling(MA_FAST).mean()
    ma_slow = c.rolling(MA_SLOW).mean()
    long = ma_fast > ma_slow
    sig = pd.DataFrame({"date": daily["date"]})
    sig["position"] = long.map({True: "GOLD", False: "FLAT"}).fillna("FLAT")
    sig["leverage_tier"] = long.map({True: "base", False: "flat"}).fillna("flat")
    sig["target_leverage"] = long.map({True: LEVERAGE, False: 0.0}).fillna(0.0)
    return sig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GOLD daily MA-cross signal.")
    parser.add_argument("--out", default="var/runtime/gold/gold_daily_signal.csv")
    parser.add_argument("--input", default=None, help="本地已有日线 CSV（date,close），跳过 OKX 拉取")
    args = parser.parse_args()

    daily = load_daily(Path(args.input) if args.input else None)
    sig = generate_signal(daily)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sig.to_csv(out, index=False)
    print(f"gold signal -> {out}  ({len(sig)} 行, 最新 {sig.iloc[-1].date.date()} {sig.iloc[-1].position})")


if __name__ == "__main__":
    main()
