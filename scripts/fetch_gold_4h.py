#!/usr/bin/env python3
"""黄金 XAU-USDT-SWAP 4h K 线拉取 — 东京服务器自闭环数据源。

供 gold_usdt_executor._load_bars() 消费（data_4h config 键）。镜像
scan_gold_daily_signal.py 的 OKX 分页拉取方式，产出与 GOOGL 4h feather
同构的 date/open/high/low/close（+volume）序列。

用法:
    python scripts/fetch_gold_4h.py --out data/okx/futures/XAU_USDT_USDT-4h-futures.feather

行为:
    - 从 OKX market/candles 分页拉 XAU-USDT-SWAP 4H，最多 ~1400 根（约 233 天）。
    - 覆盖写出 feather（date/open/high/low/close/volume）。
    - 拉取失败时退出码非 0，不覆盖已有文件。
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

OKX_CANDLES = "https://www.okx.com/api/v5/market/candles"
INST_ID = "XAU-USDT-SWAP"
BAR = "4H"
LIMIT = 300
MAX_PAGES = 6  # ~1800 bars ≈ 300 天


def fetch_okx_4h(inst_id: str = INST_ID) -> pd.DataFrame:
    rows: list[list[str]] = []
    after: str | None = None
    for _ in range(MAX_PAGES):
        params = {"instId": inst_id, "bar": BAR, "limit": str(LIMIT)}
        if after:
            params["after"] = after
        r = requests.get(OKX_CANDLES, params=params, timeout=25)
        data = r.json().get("data", [])
        if not data:
            break
        rows.extend(data)
        after = str(int(data[-1][0]) - 1)
        if len(data) < LIMIT:
            break
    if not rows:
        raise RuntimeError(f"{inst_id} {BAR} fetch returned no rows")
    df = pd.DataFrame(
        rows,
        columns=["ts", "o", "h", "l", "c", "vol", "volCcy", "volCcyQuote", "confirm"],
    )
    df = df.drop_duplicates("ts")
    df["date"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    for col, src in [("open", "o"), ("high", "h"), ("low", "l"), ("close", "c")]:
        df[col] = df[src].astype(float)
    df["volume"] = df["vol"].astype(float)
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch XAU-USDT-SWAP 4H candles to feather.")
    parser.add_argument("--out", default="data/okx/futures/XAU_USDT_USDT-4h-futures.feather")
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (ROOT / args.out).resolve()

    frame = fetch_okx_4h()
    if frame.empty:
        raise RuntimeError("XAU 4h fetch returned empty frame")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    frame.to_feather(tmp)
    tmp.replace(out_path)

    print(
        f"XAU-USDT-SWAP 4h: wrote {len(frame)} rows "
        f"({frame['date'].min()} -> {frame['date'].max()}) to {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
