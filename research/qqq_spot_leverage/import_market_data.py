#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线数据导入器（round 23）：在没有可用代理/外网时更新行情缓存。

背景（round 19 取证）
--------------------
* `config/tqqq_sqqq_signal_email.json` 的 `refresh_proxy` 指向 127.0.0.1:7892，但本机
  **7890/7891/7892/1080/… 全部端口扫过，没有任何代理在听**，环境变量里也没有 proxy；
* 直连 Yahoo Chart API 返回 403（Stooq 有 JS 挑战、AlphaVantage demo 无权限）；
* 因此 `refresh_before_send` 永远失败，配合 `refresh_fail_open: true` 造成"静默用旧数据"。

在网络恢复之前，用本脚本从**任意来源导出的 CSV/feather** 更新缓存：

    python3 research/qqq_spot_leverage/import_market_data.py \
        --symbol QQQ --input ~/Downloads/QQQ.csv --data-root data/public/etf

支持：
  * CSV/TSV/feather 输入；自动识别日期列（date/timestamp/datetime）
  * 列名大小写与顺序不敏感，至少需要 close，其余缺失列自动补 NaN
  * 与现有 feather **按日期合并去重（新数据覆盖旧的同一天）**，然后校验并覆盖写回
  * `--dry-run` 只校验不写入；`--allow-shrink` 才允许新数据比旧数据更短（默认拒绝，防手滑覆盖）
  * 写回前自动备份到 `<file>.bak-<timestamp>`
"""
from __future__ import annotations

import argparse
import shutil
import sys

import numpy as np
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO / "data" / "public" / "etf"
COLS = ["date", "open", "high", "low", "close", "volume"]


def _abs_returns(closes: "np.ndarray") -> "np.ndarray":
    r = np.abs(np.diff(closes) / closes[:-1])
    r[~np.isfinite(r)] = np.nan
    return r


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".feather":
        df = pd.read_feather(path)
    else:
        sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
        df = pd.read_csv(path, sep=sep)
    df.columns = [str(c).strip().lower() for c in df.columns]
    date_col = next((c for c in ("date", "timestamp", "datetime", "open_time") if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"{path}: 找不到日期列（date/timestamp/datetime）")
    df["date"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    for c in COLS[1:]:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[COLS].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if df["close"].isna().all():
        raise ValueError(f"{path}: close 列全为空")
    return df


def validate(df: pd.DataFrame, symbol: str) -> list[str]:
    issues: list[str] = []
    if df.empty:
        issues.append("数据为空")
    if df["close"].isna().any():
        issues.append(f"close 有 {int(df['close'].isna().sum())} 个空值")
    if (df["close"] <= 0).any():
        issues.append("存在非正收盘价")
    dup = df["date"].duplicated().sum()
    if dup:
        issues.append(f"日期重复 {dup} 行")
    r = df["close"].pct_change().abs()
    if (r > 0.5).any():
        issues.append(f"存在单日 |涨跌| >50% 的异常（{int((r > 0.5).sum())} 天），请确认复权口径")
    return issues


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline market-data importer")
    ap.add_argument("--symbol", required=True, help="如 QQQ / TQQQ / SPY / ^VIX（文件名 <SYM>-1d.feather 的 SYM）")
    ap.add_argument("--input", required=True, help="外部导出的 CSV/TSV/feather")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-shrink", action="store_true", help="允许新数据短于旧数据（默认拒绝）")
    ap.add_argument("--force", action="store_true", help="跳过拼接缝异动检查（仅在确认复权口径无误时使用）")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    target = data_root / f"{args.symbol.upper()}-1d.feather"
    incoming = read_any(Path(args.input).expanduser())
    issues = validate(incoming, args.symbol)

    existing = pd.read_feather(target) if target.exists() else pd.DataFrame(columns=COLS)
    existing["date"] = pd.to_datetime(existing["date"], utc=True, errors="coerce")
    old_last = existing["date"].max() if len(existing) else None
    new_last = incoming["date"].max() if len(incoming) else None

    print(f"目标文件 : {target}（{'存在' if target.exists() else '不存在'}）")
    print(f"现有数据 : {len(existing)} 行" + (f"，最后 {old_last.date()}" if old_last is not None else ""))
    print(f"导入数据 : {len(incoming)} 行，{incoming['date'].min().date()} ~ {new_last.date()}")
    if issues:
        print("校验问题 :")
        for i in issues:
            print(f"  - {i}")
    else:
        print("校验问题 : 无")

    if old_last is not None and new_last is not None and new_last <= old_last:
        print(f"⚠️ 导入数据不新于现有数据（{new_last.date()} <= {old_last.date()}），无增量。")

    merged = (
        pd.concat([existing, incoming], ignore_index=True)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    print(f"合并结果 : {len(merged)} 行，{merged['date'].min().date()} ~ {merged['date'].max().date()}")

    # 只在"本次导入涉及日期及其后一日"检查连续性。
    # ⚠️ 不能全历史检查：2025-04-09 SPY +10.5% 这类真实极端日会被误伤。
    merged_dates = pd.DatetimeIndex(merged["date"])
    dset = set(merged_dates)
    pos = sorted({int(merged_dates.searchsorted(d)) for d in pd.DatetimeIndex(incoming["date"]) if d in dset})
    check_pos = sorted({q for base in pos for q in (base, base + 1) if 1 <= q < len(merged)})
    closes = merged["close"].to_numpy(dtype=float)
    abs_ret = pd.Series(_abs_returns(closes)).tail(252).dropna()
    hist_vol = float(abs_ret.median() * 1.4826) if len(abs_ret) else 0.01
    jump_threshold = min(max(8.0 * hist_vol, 0.06), 0.12)
    bad = []
    for q in check_pos:
        prev, cur = closes[q - 1], closes[q]
        if prev > 0 and cur > 0 and abs(cur / prev - 1.0) > jump_threshold:
            bad.append((merged["date"].iloc[q - 1], prev, merged["date"].iloc[q], cur, abs(cur / prev - 1.0)))
    span = f"{merged_dates[pos[0]].date()} ~ {merged_dates[pos[-1]].date()}" if pos else "（无重合日期）"
    print(f"连续性阈值 : {jump_threshold*100:.1f}%（= clip(8×{hist_vol*100:.2f}%, 6%, 12%)），"
          f"检查 {len(check_pos)} 个位置（{span}）")
    if bad and not args.force:
        print(f"❌ 导入数据附近存在超过 {jump_threshold*100:.1f}% 的单日异动，拒绝写入：")
        for d0, p0, d1, p1, rr in bad[:5]:
            print(f"   {d0.date()} {p0:.2f} → {d1.date()} {p1:.2f}（{rr*100:.1f}%）")
        print("   如确认是复权/拆股导致，请先统一复权口径，或（不推荐）加 --force")
        raise SystemExit(2)

    if len(merged) < len(existing) and not args.allow_shrink:
        print("❌ 合并后行数少于现有（可能覆盖了更长历史）；如确认无误请加 --allow-shrink")
        raise SystemExit(2)
    if args.dry_run:
        print("（--dry-run，不写入）")
        return
    data_root.mkdir(parents=True, exist_ok=True)
    if target.exists():
        backup = target.with_suffix(f".feather.bak-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        shutil.copy2(target, backup)
        print(f"已备份   : {backup}")
    merged.to_feather(target)
    print(f"✅ 已写入 : {target}")


if __name__ == "__main__":
    main()
