"""趋势投票信号（无前视）：research/time_in_market/explore.py 与实盘篮子腿共用的唯一实现。

每条腿 5 条预先声明的规则，各给 0/1 票，平均成 0~1 的目标仓位比例：
  sma50 / sma100 / sma200：收盘 > 简单均线
  tsmom30 / tsmom90：收盘 / n 日前收盘 - 1 > 0
预热不足的规则记 0 票。信号只用已收盘日线；t 收盘算出的目标在 t+1 执行。
"""
from __future__ import annotations

import pandas as pd

VOTE_RULES: tuple[str, ...] = ("sma50_L", "sma100_L", "sma200_L", "tsmom30_L", "tsmom90_L")


def vote_components(close: pd.Series) -> dict[str, pd.Series]:
    """返回每条规则的 0/1 票（与 explore.signal_family 中同名规则逐位一致）。"""
    out: dict[str, pd.Series] = {}
    for n in (50, 100, 200):
        sma = close.rolling(n).mean()
        out[f"sma{n}_L"] = (close > sma).astype(float).where(sma.notna(), 0.0)
    for n in (30, 90):
        mom = close / close.shift(n) - 1
        out[f"tsmom{n}_L"] = (mom > 0).astype(float).where(mom.notna(), 0.0)
    return out


def vote_target(close: pd.Series) -> pd.Series:
    """0~1 的目标仓位比例序列（步长 0.2）。"""
    parts = vote_components(close)
    return sum(parts[name] for name in VOTE_RULES) / len(VOTE_RULES)
