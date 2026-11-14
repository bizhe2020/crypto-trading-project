"""趋势投票信号（无前视）：research/time_in_market/explore.py 与实盘篮子腿共用的唯一实现。

每条腿 5 条预先声明的规则，各给 0/1 票，平均成 0~1 的目标仓位比例：
  sma50 / sma100 / sma200：收盘 > 简单均线
  tsmom30 / tsmom90：收盘 / n 日前收盘 - 1 > 0
预热不足的规则记 0 票。信号只用已收盘日线；t 收盘算出的目标在 t+1 执行。
"""
from __future__ import annotations

import numpy as np
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


def sma200_target(close: pd.Series) -> pd.Series:
    """Gayed & Bilello (2016)「Leverage for the Long Run」：收盘 > SMA200 满仓，否则空仓（0/1）。"""
    return vote_components(close)["sma200_L"]


def sma200_buffer_target(close: pd.Series, buffer: float = 0.01) -> pd.Series:
    """带对称缓冲的 SMA200 状态机（0/1）。

    buffer=0 时与 sma200_target 等价；buffer=0.01 表示：
      * 空仓 → 收盘 > SMA200 × 1.01 才进场
      * 满仓 → 收盘 < SMA200 × 0.99 才出场
    只用已收盘日线；t 收盘的状态在 t+1 执行。
    """
    if buffer < 0:
        raise ValueError("buffer must be non-negative")
    sma = close.rolling(200).mean()
    c = close.to_numpy(dtype=float)
    s = sma.to_numpy(dtype=float)
    state = 0
    out = np.zeros(len(c), dtype=float)
    for i in range(len(c)):
        if np.isnan(s[i]):
            out[i] = 0.0
            continue
        if state == 0:
            if c[i] > s[i] * (1.0 + buffer):
                state = 1
        elif c[i] < s[i] * (1.0 - buffer):
            state = 0
        out[i] = float(state)
    return pd.Series(out, index=close.index, name="sma200_buffer")


def sma200_buf1_target(close: pd.Series) -> pd.Series:
    """研究候选规则：SMA200 对称 1% 缓冲（走步验证未确认，默认不启用）。"""
    return sma200_buffer_target(close, buffer=0.01)


# 每条腿可选的信号规则（实盘配置 legs[].signal_rule）
SIGNAL_RULES = {
    "v5": (vote_target, VOTE_RULES),
    "sma200": (sma200_target, ("sma200_L",)),
    "sma200_buf1": (sma200_buf1_target, ("sma200_L",)),
}


def signal_target(close: pd.Series, rule: str = "v5") -> pd.Series:
    if rule not in SIGNAL_RULES:
        raise ValueError(f"unknown signal rule: {rule}")
    return SIGNAL_RULES[rule][0](close)
