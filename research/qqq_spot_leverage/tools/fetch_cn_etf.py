#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓取 A 股场内纳指 QDII ETF 的日线 + 基金净值（NAV），用于溢价/折价建模。

数据源（东财公开接口）：
  * 日线: push2his.eastmoney.com/api/qt/stock/kline/get  (secid: 0.=深 1.=沪)
  * 净值: api.fund.eastmoney.com/f10/lsjz                (需 Referer: fundf10.eastmoney.com)

输出（写入 data/public/etf/）：
  * <code>.SS/.SZ-1d.feather   —— 行情（open/high/low/close/volume，**前复权 fqt=1**）
  * <code>.SS/.SZ-nav.feather  —— 基金净值（nav），用于计算溢价 premium = close/nav - 1

用法：
    python3 research/qqq_spot_leverage/tools/fetch_cn_etf.py --codes 159501 513100 159941
    python3 research/qqq_spot_leverage/tools/fetch_cn_etf.py --all --dry-run
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ETF_DIR = REPO / "data" / "public" / "etf"

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# 主流 A 股场内纳斯达克100 ETF（同指数，可互为执行替代）
UNIVERSE = {
    "159501": ("SZ", "纳指ETF嘉实", "嘉实纳斯达克100ETF(QDII)"),
    "513100": ("SH", "纳指ETF国泰", "国泰纳斯达克100ETF(QDII)"),
    "159941": ("SZ", "纳指ETF广发", "广发纳斯达克100ETF(QDII)"),
    "513300": ("SH", "纳斯达克ETF华夏", "华夏纳斯达克100ETF(QDII)"),
    "159632": ("SZ", "纳斯达克ETF华安", "华安纳斯达克100ETF(QDII)"),
    "513390": ("SH", "纳指100ETF博时", "博时纳斯达克100ETF(QDII)"),
    "513110": ("SH", "纳指ETF招商", "招商纳斯达克100ETF(QDII)"),
    "159696": ("SZ", "纳指ETF易方达", "易方达纳斯达克100ETF(QDII)"),
    "513870": ("SH", "纳指ETF富国", "富国纳斯达克100ETF(QDII)"),
    "159934": ("SZ", "黄金ETF易方达", None),  # 占位对照，通常不用
}
UNIVERSE = {k: v for k, v in UNIVERSE.items() if v[2] is not None}


def secid(code: str, mkt: str) -> str:
    return f"{'1' if mkt == 'SH' else '0'}.{code}"


def fetch_json(url: str, referer: str | None = None, tries: int = 3) -> dict:
    for i in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": UA, **({"Referer": referer} if referer else {})})
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                raw = r.read().decode("utf-8", "replace")
            return json.loads(raw)
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))
    return {}


def fetch_kline(code: str, mkt: str, beg: str = "20000101") -> pd.DataFrame:
    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get?"
           + urllib.parse.urlencode({
               "secid": secid(code, mkt),
               "fields1": "f1,f2,f3,f4,f5",
               "fields2": "f51,f52,f53,f54,f55,f56,f57",
               "klt": "101", "fqt": "1", "beg": beg, "end": "20500101", "lmt": "100000",  # fqt=1 前复权，消除拆分跳变
           }))
    js = fetch_json(url)
    data = js.get("data") or {}
    kl = data.get("klines") or []
    if not kl:
        return pd.DataFrame()
    rows = []
    for line in kl:
        p = line.split(",")
        rows.append((p[0], float(p[1]), float(p[3]), float(p[4]), float(p[2]), float(p[5]), float(p[6] or 0)))
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    df["name"] = data.get("name")
    return df


def fetch_nav(code: str, max_pages: int = 200, quiet: bool = False) -> pd.DataFrame:
    """基金历史净值（单位净值 DWJZ）。

    注意：东财该接口**忽略 pageSize，硬上限 20 行/页**，必须翻页到 TotalCount。
    """
    out: list[tuple[str, float]] = []
    total = None
    page = 1
    while page <= max_pages:
        url = ("https://api.fund.eastmoney.com/f10/lsjz?"
               + urllib.parse.urlencode({"fundCode": code, "pageIndex": str(page), "pageSize": "200"}))
        try:
            js = fetch_json(url, referer="https://fundf10.eastmoney.com/")
        except Exception:  # noqa: BLE001
            break
        if total is None:
            total = int(js.get("TotalCount") or 0)
        lst = ((js.get("Data") or {}).get("LSJZList")) or []
        if not lst:
            break
        for it in lst:
            d, v = it.get("FSRQ"), it.get("DWJZ")
            if d and v not in (None, ""):
                out.append((d, float(v)))
        if total and len(out) >= total:
            break
        page += 1
        time.sleep(0.12)
    if not quiet and total:
        print(f"    [nav] {code}: 翻 {page} 页取回 {len(out)}/{total} 行")
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out, columns=["date", "nav"])
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def save_feather(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_feather(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", nargs="*", default=None, help="ETF 代码，如 159501 513100")
    ap.add_argument("--all", action="store_true", help="抓取内置全部纳指 ETF")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    codes = list(UNIVERSE) if a.all else (a.codes or [])
    if not codes:
        ap.error("请用 --codes 或 --all 指定代码")

    for code in codes:
        mkt, short, _full = UNIVERSE.get(code, ("SZ" if code.startswith("1") else "SH", code, code))
        suffix = f"{code}.{mkt}"
        try:
            k = fetch_kline(code, mkt)
        except Exception as e:  # noqa: BLE001
            print(f"[kline] {code} 失败: {e}")
            continue
        if k.empty:
            print(f"[kline] {code} 无数据（接口未返回）")
            continue
        try:
            n = fetch_nav(code)
        except Exception as e:  # noqa: BLE001
            print(f"[nav]   {code} 失败: {e}")
            n = pd.DataFrame()
        print(f"{code} {short:12s} 行情 {len(k):5d} 行 {k['date'].min():%Y-%m-%d}~{k['date'].max():%Y-%m-%d}"
              f" | 净值 {len(n):5d} 行" + (f" {n['date'].min():%Y-%m-%d}~{n['date'].max():%Y-%m-%d}" if len(n) else ""))
        if a.dry_run:
            continue
        save_feather(k.drop(columns=["name"]), ETF_DIR / f"{suffix}-1d.feather")
        if len(n):
            save_feather(n, ETF_DIR / f"{suffix}-nav.feather")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
