#!/usr/bin/env python3
"""把 GOOGL 真实分钟数据重采样为 4h/15m UTC bars，写入执行层回测路径。

真实数据来源：价值投资project/data/googl_minute.csv
（Polygon 风格分钟聚合，含盘前/盘中/盘后，不规则间隔 = 仅记录有成交的分钟）。

重采样规则（镜像 OKX candle 语义）：
    - UTC 桶（4h: [00:00,04:00) [04:00,08:00) ...；15m: 对齐 00:00 起 15 分钟）
    - open = 桶内首根分钟 open；close = 桶内末根分钟 close
    - high = max(high)；low = min(low)；volume = sum(volume)
    - 空桶（无成交，如夜间/周末/节假日）不生成 bar —— 回测按"无 bar = 无交易
      = 不检查止损"处理，与实际股票交易时段一致。

用法:
    python scripts/build_googl_4h_from_minute.py \\
        --minute /Users/laoji/projects/价值投资project/data/googl_minute.csv \\
        --out data/okx/futures/GOOGL_USDT_USDT-4h-futures.feather
    # 15m（SMC 事件扫描研究用）
    python scripts/build_googl_4h_from_minute.py --tf 15m \
        --out data/okx/futures/GOOGL_USDT_USDT-15m-futures.feather
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MINUTE = Path("/Users/laoji/projects/价值投资project/data/googl_minute.csv")
DEFAULT_OUT = ROOT / "data" / "okx" / "futures" / "GOOGL_USDT_USDT-4h-futures.feather"


def resample_minute_to_tf(minute_csv: Path, timeframe: str = "4h") -> pd.DataFrame:
    """读分钟 CSV → UTC OHLC bars（feather 会附带 date 列）。

    timeframe 支持 "4h" / "15m"（SMC 研究需要 15m 事件粒度）。
    注意 pandas 频率 "15m" 会被解析为 15 个月（'m'=月），分钟须用 "15min"。
    """
    freq = {"15m": "15min", "15min": "15min"}.get(timeframe, timeframe)
    df = pd.read_csv(minute_csv, low_memory=False)
    df["date"] = pd.to_datetime(df["ts_utc"], unit="s", utc=True)
    df = df.sort_values("date")
    df = df.set_index("date")

    ohlcv = df.resample(freq, origin="epoch").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    ohlcv = ohlcv.dropna(subset=["open", "high", "low", "close"])
    ohlcv = ohlcv[ohlcv["open"] > 0]
    out = ohlcv.reset_index()
    out["date"] = out["date"].dt.tz_convert("UTC")
    return out[["date", "open", "high", "low", "close", "volume"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="GOOGL 分钟数据 → UTC bars")
    parser.add_argument("--minute", default=str(DEFAULT_MINUTE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--tf", default="4h", help="重采样周期: 4h（默认）/ 15m")
    args = parser.parse_args()

    src = Path(args.minute)
    if not src.exists():
        raise FileNotFoundError(f"分钟数据不存在: {src}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    bars = resample_minute_to_tf(src, timeframe=args.tf)
    bars.to_feather(out)
    # 同时写 CSV 便于人工检查
    csv_out = out.with_suffix(".csv")
    bars.to_csv(csv_out, index=False)

    # 汇总
    bars_per_day = bars.set_index("date").index.normalize().value_counts()
    print(f"写入 {out}（+ {csv_out}）")
    print(f"{args.tf} bars 总数: {len(bars)}")
    print(f"覆盖: {bars['date'].min()} → {bars['date'].max()}")
    print(f"交易日数: {bars_per_day.nunique()} 日（含空桶日）")
    print(f"平均每日 bars: {len(bars)/max(bars_per_day.nunique(),1):.1f}，日 bars 分布:")
    print(bars_per_day.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
