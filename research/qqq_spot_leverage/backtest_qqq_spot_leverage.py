#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QQQ 现货杠杆策略（<=1.5x）回测 2020-01 起。

设计目标
--------
把 crypto-trading-project 里"纳指趋势/加密合约"那套「趋势方向 + 波动率 regime + 分级杠杆」
翻译成 QQQ **现货**可执行版本：只做多、现金/杠杆互换、杠杆硬上限 1.5x。

策略结构（三层 + 修复再入场）
-----------------------------
1. 趋势方向层：QQQ 收盘 vs MA200 决定是否在场；MA50>MA200 决定是否给满杠杆。
2. Regime 风控层：VIX / VIX3M 期限结构做"恐惧闸门"（带迟滞），高波动时降杠杆或空仓。
3. 修复再入场：收盘在 MA200 下方但"站上 20 日线 + 20 日动量转正 + 未深跌 + MA200 斜率未加速下行"时给 0.75x。
4. 执行层：D 日收盘定信号，D+1 开盘成交（按股数记账），融资只对实际借款余额按日计息。

无前视口径（与 docs/backtest_lookahead_policy.md 同一标准）
---------------------------------------------------------
* 第 t 日收盘后才算出 target[t]，第 t+1 日开盘才成交；
* 所有 rolling/MA/分位数只用 <= t 的收盘数据；漂移再平衡判据用"上日收盘口径"。
* 成交与估值统一用 **前复权** 价格空间（adj_open / adj_close），
  避免"未复权开盘买、前复权收盘估值"造成的假杠杆（2020 年前复权因子≈0.58）。

数据源
------
* ~/projects/期权量化/data/qqq_daily.csv      前复权日线（2001-01-02 ~ 2026-09-02；2005-2011 有 6.4 年缺口，2001-2004 有坏点）
* ~/projects/期权量化/data/qqq_daily_raw.csv  未复权日线（同区间，行权价/名义价口径）
* <repo>/data/public/etf_full/^VIX-1d.feather    VIX 收盘（1999 ~ 2026-05）
* <repo>/data/public/etf/^VIX-1d.feather         覆盖 2022-01 ~ 2026-08 的补充
* <repo>/data/public/etf_full/^VIX3M-1d.feather VIX3M 收盘
* <repo>/data/public/etf_full/SPY-1d.feather     SPY 收盘（相对强弱用）

用法
----
    python3 research/qqq_spot_leverage/backtest_qqq_spot_leverage.py
    python3 research/qqq_spot_leverage/backtest_qqq_spot_leverage.py --end 2026-08-14
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "out"
SRC_DATA_DIR = Path("/Users/laoji/projects/期权量化/data")

START_DEFAULT = "2020-01-02"
# VIX/VIX3M 覆盖到 2026-08-14，之后没有 regime 输入 -> 主回测统一截到这里
END_DEFAULT = "2026-08-14"
QQQ_LAST_KNOWN = "2026-09-02"

# 融资利率：只对 (L-1) 的借款部分计息（IBKR Reg-T 近似，分段）
FIN_RATES = {
    2020: 0.0150,
    2021: 0.0100,
    2022: 0.0250,
    2023: 0.0450,
    2024: 0.0550,
    2025: 0.0550,
    2026: 0.0500,
}
COST_BPS_PER_TURN = 1.0  # 单边 1bp 调仓成本（含点差/滑点，ETF 流动性极好）


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def _load_feather(path: Path, name: str) -> pd.Series:
    df = pd.read_feather(path)
    # 统一到"美东自然日"的 naive Timestamp，和 QQQ CSV 的日期口径对齐
    df["session_day"] = (
        pd.to_datetime(df["date"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    )
    df = df.sort_values("session_day").drop_duplicates("session_day", keep="last")
    return df.set_index("session_day")["close"].rename(name)


def load_market_frame(start: str, end: str, price_source: str = "csv") -> pd.DataFrame:
    """返回以 QQQ 交易日为索引、含价格与 regime 输入的日线表。

    price_source:
      * "csv"     : 期权量化/data 的两份 CSV（前复权 close + 未复权 open）。
                    2020 起精确；但 **缺 2005-2011**，且 2001-2004 复权因子损坏。
      * "feather" : data/public/etf_full/QQQ-1d.feather（1999 ~ 2026-05，连续）。
                    该序列不做股息复权（≈价格指数），总收益会低估股息率，
                    但历史连续，适合 2001-2026 的长样本压力测试。
    """
    qqq_path = SRC_DATA_DIR / "qqq_daily.csv"
    raw_path = SRC_DATA_DIR / "qqq_daily_raw.csv"
    if not qqq_path.exists():
        raise FileNotFoundError(f"missing {qqq_path}")

    qqq = pd.read_csv(qqq_path, parse_dates=["date"]).sort_values("date")
    qqq = qqq.drop_duplicates("date", keep="last").set_index("date")
    raw = pd.read_csv(raw_path, parse_dates=["date"]).sort_values("date")
    raw = raw.drop_duplicates("date", keep="last").set_index("date")

    if price_source == "feather":
        fe = pd.read_feather(REPO / "data" / "public" / "etf_full" / "QQQ-1d.feather")
        fe["session_day"] = (
            pd.to_datetime(fe["date"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        )
        fe = fe.sort_values("session_day").drop_duplicates("session_day", keep="last").set_index("session_day")
        frame = pd.DataFrame(index=fe.index)
        frame["adj_close"] = fe["close"].astype(float)
        frame["adj_open"] = fe["open"].astype(float)
        frame["raw_close"] = fe["close"].astype(float)
        frame["raw_open"] = fe["open"].astype(float)
        frame["adj_high"] = fe["high"].astype(float)
        frame["adj_low"] = fe["low"].astype(float)
        frame["volume"] = fe["volume"].astype(float)
    else:
        frame = pd.DataFrame(index=qqq.index)
        frame["adj_close"] = qqq["close"].astype(float)
        frame["raw_close"] = raw["close"].reindex(frame.index).astype(float)
        frame["raw_open"] = raw["open"].reindex(frame.index).astype(float)
        frame["adj_high"] = qqq["high"].reindex(frame.index).astype(float)
        frame["adj_low"] = qqq["low"].reindex(frame.index).astype(float)
        frame["volume"] = raw["volume"].reindex(frame.index).astype(float)
        # 统一复权口径：调整后的开盘价（= 未复权开盘 × 当日复权因子）
        # 必须用同一个价格空间做"成交+估值"，否则 2020 年前复权因子≠1 时会出现
        # "用 raw_open 买、用 adj_close 估值"的虚假 2 倍杠杆与假暴跌。
        factor = frame["adj_close"] / frame["raw_close"]
        frame["adj_open"] = frame["raw_open"] * factor

    def _merge(name: str, *paths: Path) -> None:
        series = pd.Series(dtype=float, name=name)
        for p in paths:
            if p.exists():
                s = _load_feather(p, name)
                series = s if series.empty else series.combine_first(s)
        series = series[~series.index.duplicated(keep="last")]
        frame[name] = series.reindex(frame.index)

    etf = REPO / "data" / "public" / "etf"
    etf_full = REPO / "data" / "public" / "etf_full"
    _merge("vix", etf_full / "^VIX-1d.feather", etf / "^VIX-1d.feather")
    _merge("vix3m", etf_full / "^VIX3M-1d.feather", etf / "^VIX3M-1d.feather")
    _merge("spy", etf_full / "SPY-1d.feather", etf / "SPY-1d.feather")

    frame = frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]
    frame = frame[~frame["adj_close"].isna()]
    return frame


def add_indicators(frame: pd.DataFrame, fast_window: int, slow_window: int) -> pd.DataFrame:
    df = frame.copy()
    df["ret"] = df["adj_close"].pct_change()
    df["ma_fast"] = df["adj_close"].rolling(fast_window).mean()
    df["ma_slow"] = df["adj_close"].rolling(slow_window).mean()
    df["above_slow"] = df["adj_close"] > df["ma_slow"]
    df["fast_above_slow"] = df["ma_fast"] > df["ma_slow"]
    df["ret_20"] = df["adj_close"].pct_change(20)

    # 20 日已实现波动（年化）——只用 t 及之前的收益
    df["rv20"] = df["ret"].rolling(20).std() * math.sqrt(252)
    df["rv60"] = df["ret"].rolling(60).std() * math.sqrt(252)

    df["vix_ratio"] = df["vix"] / df["vix"].rolling(60).mean()
    df["vix_term"] = df["vix"] / df["vix3m"]  # >1 = 近月恐慌（backwardation）
    df["spy_ret_20"] = df["spy"].pct_change(20)
    # 相对强弱只在 SPY 与 QQQ 同日都有数据时有效（避免一边 stale 一边新鲜产生伪信号）
    spy_valid = df["spy"].notna() & df["spy"].shift(1).notna()
    df["rs20"] = (df["ret_20"] - df["spy_ret_20"]).where(spy_valid)
    df["rs60"] = (df["adj_close"].pct_change(60) - df["spy"].pct_change(60)).where(spy_valid)
    # QQQ 距 252 日高点（趋势位置）
    df["dd_from_high"] = df["adj_close"] / df["adj_close"].rolling(252, min_periods=60).max() - 1.0

    # VIX 档位迟滞所需的状态量（全部只用 <=t 的数据；在 apply_vix_timing 之后重算）
    df["vix_stress_run"] = _run_length(df["vix"] >= 28.0)
    df["vix_extreme_run"] = _run_length(df["vix"] >= 35.0)
    df["vix_term_stress_run"] = _run_length(df["vix_term"] >= 1.0)
    # 修复信号（用于 MA200 下方的再入场）：价格重新站上 20 日线
    df["ma_recovery"] = df["adj_close"].rolling(20).mean()
    df["above_recovery_ma"] = df["adj_close"] > df["ma_recovery"]
    # MA200 斜率（用于判断"持续熊市"，抑制下跌中继的假修复信号）
    df["ma_slow_slope"] = df["ma_slow"] / df["ma_slow"].shift(20) - 1.0
    df["fast_slope"] = df["ma_fast"] / df["ma_fast"].shift(20) - 1.0

    # 备用趋势规则（趋势模型对比实验用），全部只用 <=t 数据
    df["donchian_hi"] = df["adj_close"].rolling(slow_window).max()
    df["ma_slow_prev"] = df["ma_slow"].shift(1)
    df["ema_fast"] = df["adj_close"].ewm(span=fast_window, adjust=False).mean()
    df["ema_slow"] = df["adj_close"].ewm(span=slow_window, adjust=False).mean()
    return df


def apply_vix_timing(df: pd.DataFrame, vix_timing: str, vix_shift_days: int) -> pd.DataFrame:
    """VIX 使用口径 + 重算依赖 VIX 的派生列：

    * same_close : 用 D 日 VIX 收盘（15:00 ET 结算，早于股票 16:00 收盘，信息合法）
    * prior_close: 保守口径，只用 D-1 日 VIX 收盘（防"结算时间差"争议）
    """
    if vix_timing == "same_close":
        return df
    out = df.copy()
    for col in ["vix", "vix3m"]:
        if col in out.columns:
            out[col] = out[col].shift(vix_shift_days)
    out["vix_ratio"] = out["vix"] / out["vix"].rolling(60).mean()
    out["vix_term"] = out["vix"] / out["vix3m"]
    return out


def prepare_indicators(frame: pd.DataFrame, p: Params) -> pd.DataFrame:
    """完整的无前视指标管线：基础指标 -> VIX 口径 -> VIX 迟滞状态量。"""
    df = add_indicators(frame, p.fast_window, p.slow_window)
    df = apply_vix_timing(df, p.vix_timing, p.vix_shift_days)
    return recompute_vix_state(df, p)


def _run_length(condition: pd.Series) -> pd.Series:
    """连续为真的天数（截至当日），只用 <=t 的信息。"""
    groups = (~condition.fillna(False)).cumsum()
    return condition.fillna(False).astype(int).groupby(groups).cumsum()


def recompute_vix_state(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """按当前 VIX 口径（阈值+迟滞）重算连续超阈值天数。"""
    out = df.copy()
    out["vix_stress_run"] = _run_length(out["vix"] >= (p.vix_stress + p.vix_hysteresis))
    out["vix_extreme_run"] = _run_length(out["vix"] >= p.vix_extreme)
    out["vix_term_stress_run"] = _run_length(out["vix_term"] >= p.vix_term_stress)
    return out


# --------------------------------------------------------------------------- #
# 策略参数
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    fast_window: int = 50
    slow_window: int = 200
    # 杠杆阶梯（0 <= lev <= max_leverage）
    lev_full: float = 1.50      # 趋势确认 + regime 允许
    lev_half: float = 1.00      # 在 MA200 上方但趋势未确认（或 VIX 偏中性）
    lev_defense: float = 0.25   # VIX 高压 / 期限结构倒挂 -> 留底仓
    max_leverage: float = 1.50
    # VIX regime 闸门
    vix_neutral: float = 22.0   # >= 此值 -> 降到 lev_half
    vix_stress: float = 28.0    # >= 此值或期限倒挂 -> 降到 lev_defense
    vix_extreme: float = 35.0   # >= 此值 -> 空仓
    vix_term_stress: float = 1.0  # vix/vix3m >= 此值 -> 视为恐慌
    use_vix_gate: bool = False   # round 7 修正口径后：VIX 闸门为负贡献，默认关闭
    # VIX 闸门迟滞（避免 VIX 在阈值附近抖动造成 0x<->1.5x 反复切换）
    vix_hysteresis: float = 2.0        # 降档需 VIX >= 阈值 + 该缓冲（防抖动）
    vix_confirm_days: int = 0          # VIX 超阈值需连续 N 日才降档（0=只看当日）
    vix_term_bypass_above: float = 0.0 # VIX >= 此值时，期限倒挂不再单独触发降档（0=不生效）
    # 再入场（recovery）：收盘在 MA200 下方但出现修复信号时，允许小仓位参与
    # —— 2026-09 迭代新增：修复 2020 V 型反转错过 + 用 MA200 斜率护栏挡掉熊市接刀
    use_recovery: bool = False   # round 7 修正口径后：修复仓为负贡献，默认关闭
    recovery_lev: float = 0.75
    recovery_ret20_min: float = 0.02   # 20 日动量下限
    recovery_require_above_fast_ma: bool = True
    recovery_floor_pct: float = 0.10   # 收盘不得低于 MA200*(1-此值)，避免下跌中途接刀
    recovery_ma_slope_days: int = 20   # MA200 斜率回看天数
    recovery_ma_slope_min: float = -0.02  # MA200 20日跌幅超过此值 -> 禁止再入场（持续熊市）
    # 相对强弱闸门（QQQ vs SPY，实验用；rs_mode="" 表示关闭）
    rs_mode: str = ""                 # round 7 修正口径后：RS 闸门为负贡献，默认关闭
    rs_min: float = 0.0               # RS 低于该值 -> 杠杆最多 1.0x（v5）
    # 趋势规则选择（模型对比实验用）
    trend_mode: str = "dual_ma"        # dual_ma | dual_ma_flat | donchian | ema_dual
    trend_flat_slope_min: float = -0.02  # dual_ma_flat: MA200 20日斜率下限
    # VIX 使用口径（防"VIX 结算时间差"争议的保守检验）
    vix_timing: str = "same_close"     # same_close | prior_close
    vix_shift_days: int = 1            # prior_close 时平移天数
    # 组合层（权益路径）风控叠加——默认关闭，经实验检验后再决定是否启用
    dd_derisk_start: float = 0.0       # 权益自峰值回撤 >= 此值 -> 杠杆打折（0=关闭）
    dd_derisk_factor: float = 0.5      # 打折系数
    dd_rearm_level: float = 0.05       # 回撤修复到该水平以内 -> 恢复
    rr_derisk_60d: float = 0.0         # 60 日收益 >= 此值 -> 杠杆打折（0=关闭）
    rr_derisk_factor: float = 0.5
    vix_gate_confirm_decline: bool = True   # 降档还需"当日 QQQ 下跌"确认（v3 默认开启，见报告第 6 节）
    # 波动率目标（soft scaling，叠加在阶梯之上；默认关闭——见报告：它降波动但也降收益）
    # 单腿 1.5x 下的最佳风控（round 24 聚焦评测）：22% 波动率目标在 4 个窗口全部改善 Sharpe/Calmar
    vol_target: float = 0.22    # 0 = 关闭；>0 时按 clip(vol_target/rv20, floor, 1) 缩放
    vol_scale_floor: float = 0.5
    vol_override: bool = False  # True = 波动率缩放覆盖 VIX 档位（只留 VIX>=extreme 清仓）
    # 风控
    # 单腿 1.5x 聚焦（round 24）：快速硬止损让 MA200 的慢退出提前，四个窗口 Sharpe/回撤全部改善
    trail_lookback: int = 25        # 价格滚动高点回看（交易日）
    trail_drawdown: float = 0.08    # 自滚动高点回撤 8% -> 清仓进入冷却
    cooldown_days: int = 5          # 冷却天数
    reentry_ret20_min: float = 0.0  # 冷却后重入所需的 20 日动量下限
    stop_warmup_days: int = 30      # 策略启动初期不触发（避免空仓期基线误判）
    # 成本
    fin_spread: float = 0.0     # 在分段基准利率之上加的利差
    cost_bps: float = COST_BPS_PER_TURN   # 固定单边成本（佣金+正常点差）
    slip_vix_coef: float = 0.0            # 波动率相关滑点：slip_bps = coef * VIX（0=关闭）
    slip_max_bps: float = 60.0            # 滑点上限（防极端日）
    slip_move_frac: float = 0.0           # 成交日"当日振幅"的额外滑点比例（0=关闭）
    slip_move_max_frac: float = 0.02      # 单次额外滑点上限（占总价比例）
    min_trade_delta: float = 0.0  # 调仓死区：目标档位变化 < 该值则不交易（省成本）
    drift_band: float = 0.25      # 目标不变时的漂移再平衡带：|实际-目标| > 该值才调
    confirm_days: int = 1         # 信号确认天数：目标档位需连续 N 日一致才执行（降换手）


def data_quality(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "start": str(frame.index[0].date()),
        "end": str(frame.index[-1].date()),
        "missing": {c: int(frame[c].isna().sum()) for c in frame.columns},
        "vix_last_valid": str(frame["vix"].dropna().index[-1].date()) if frame["vix"].notna().any() else None,
        "vix3m_last_valid": str(frame["vix3m"].dropna().index[-1].date()) if frame["vix3m"].notna().any() else None,
        "spy_last_valid": str(frame["spy"].dropna().index[-1].date()) if frame["spy"].notna().any() else None,
    }


def trend_state(row: pd.Series, p: Params) -> tuple[bool, bool]:
    """返回 (是否在场, 是否趋势确认=可满杠杆)。

    支持多种趋势规则（trend_mode），全部只用 <=t 的收盘数据：
      dual_ma      : 收盘>MA_slow 且 MA_fast>MA_slow（当前主策略）
      dual_ma_flat : 收盘>MA_slow 且 MA_slow 未显著下行（不要求 MA_fast>MA_slow）
      donchian     : 收盘 >= 过去 slow 日最高收盘（新高突破）
      ema_dual     : EMA_fast > EMA_slow
    """
    close = float(row["adj_close"])
    if p.trend_mode == "donchian":
        hi = row.get("donchian_hi")
        if pd.isna(hi):
            return False, False
        in_market = close >= float(hi) * (1.0 - 1e-9)
        return bool(in_market), bool(in_market)

    above_slow = bool(row["above_slow"]) and pd.notna(row["ma_slow"])
    if p.trend_mode == "ema_dual":
        if pd.isna(row.get("ema_slow")):
            return False, False
        in_market = float(row["ema_fast"]) > float(row["ema_slow"])
        return bool(in_market), bool(in_market)

    if p.trend_mode == "dual_ma_flat":
        slope = float(row["ma_slow_slope"]) if pd.notna(row.get("ma_slow_slope")) else 0.0
        in_market = above_slow and slope >= p.trend_flat_slope_min
        return bool(in_market), bool(in_market)

    # dual_ma（默认）
    return bool(above_slow), bool(row["fast_above_slow"])


def base_target(row: pd.Series, p: Params) -> float:
    """收盘信号 -> 目标杠杆（不含路径依赖的止损层）。"""
    in_market, confirmed = trend_state(row, p)

    if not in_market:
        # 修复再入场：MA200 下方，但出现"站上快线 + 动量转正 + 未深跌"的修复信号
        if p.use_recovery and pd.notna(row["ma_slow"]):
            close = float(row["adj_close"])
            near = close >= float(row["ma_slow"]) * (1.0 - p.recovery_floor_pct)
            mom_ok = pd.notna(row["ret_20"]) and float(row["ret_20"]) >= p.recovery_ret20_min
            fast_ok = (not p.recovery_require_above_fast_ma) or bool(row["above_recovery_ma"])
            slope = float(row["ma_slow_slope"]) if pd.notna(row.get("ma_slow_slope")) else 0.0
            bear_ok = slope >= p.recovery_ma_slope_min
            if near and mom_ok and fast_ok and bear_ok:
                lev = min(p.recovery_lev, p.max_leverage)
                return _apply_vix_gate(row, p, lev)
        return 0.0

    lev = p.lev_full if confirmed else p.lev_half
    return _apply_vix_gate(row, p, lev)


def _apply_vix_gate(row: pd.Series, p: Params, lev: float) -> float:
    """VIX / 期限结构闸门（支持迟滞与连续确认），返回调整后的杠杆。"""
    if p.use_vix_gate and pd.notna(row.get("vix")):
        vix = float(row["vix"])
        term = float(row["vix_term"]) if pd.notna(row.get("vix_term")) else 1.0
        vix_stress_run = int(row.get("vix_stress_run", 0) or 0)
        vix_extreme_run = int(row.get("vix_extreme_run", 0) or 0)
        term_run = int(row.get("vix_term_stress_run", 0) or 0)

        if vix >= p.vix_extreme and vix_extreme_run >= max(1, p.vix_confirm_days):
            return 0.0

        if not (p.vol_target > 0 and p.vol_override):
            stress_threshold = p.vix_stress + p.vix_hysteresis
            term_is_stress = term >= p.vix_term_stress and not (
                p.vix_term_bypass_above > 0 and vix >= p.vix_term_bypass_above
            )
            stress_hit = vix >= stress_threshold or term_is_stress
            if p.vix_gate_confirm_decline:
                stress_hit = stress_hit and bool(row["ret"] < 0)
            if p.vix_confirm_days > 0:
                stress_hit = (vix >= stress_threshold) or (vix >= p.vix_stress and (vix_stress_run >= p.vix_confirm_days or term_run >= p.vix_confirm_days))
            if stress_hit:
                lev = min(lev, p.lev_defense)
            elif vix >= p.vix_neutral:
                lev = min(lev, max(p.lev_half, p.lev_defense))

    if p.rs_mode and pd.notna(row.get(p.rs_mode)):
        if float(row[p.rs_mode]) < p.rs_min:
            lev = min(lev, max(p.lev_half, p.lev_defense))
    if p.vol_target > 0 and pd.notna(row.get("rv20")) and row["rv20"] > 0:
        ratio = p.vol_target / float(row["rv20"])
        scale = min(1.0, max(p.vol_scale_floor, ratio))
        lev *= scale

    return float(min(lev, p.max_leverage))


def build_target_series(df: pd.DataFrame, p: Params) -> pd.Series:
    raw = df.apply(lambda r: base_target(r, p), axis=1)
    if p.confirm_days > 1:
        confirmed = raw.copy()
        for i in range(len(raw)):
            window = raw.iloc[i - p.confirm_days + 1 : i + 1]
            if len(window) < p.confirm_days or window.nunique() > 1:
                confirmed.iloc[i] = raw.iloc[max(0, i - 1)] if i > 0 else 0.0
        raw = confirmed
    return raw.rename("target_lev")


def fin_rate(year: int, spread: float) -> float:
    base = FIN_RATES.get(year, FIN_RATES[max(FIN_RATES)])
    return base + spread


# --------------------------------------------------------------------------- #
# 回测引擎
# --------------------------------------------------------------------------- #
def run_engine(df: pd.DataFrame, target: pd.Series, p: Params, label: str) -> dict:
    """精确 next_open 执行引擎（按股数记账，无前视、不虚增建仓当日收益）。

    时间线（D = 第 t 日）:
      * D 日收盘算出 target[t]（只用 <=D 的数据）；
      * D+1 开盘按 open[D+1] 成交，把 notional 调到 target[t] × 当时净值；
      * 持仓按当日收盘价 mark-to-market，隔夜融资按上日收盘后的借款逐日计提。

    这样建仓/平仓那一根 bar 只赚"从开盘价起"的部分，不会把开盘前（信号尚未生效）的
    涨幅算进来。日志里 weight[t] = D 日收盘后的持仓市值 / 净值（即 open[D+1] 成交后的杠杆）。
    """
    dates = df.index
    n = len(dates)
    open_px = df["adj_open"].to_numpy(dtype=float)
    close_px = df["adj_close"].to_numpy(dtype=float)
    target_lev = target.to_numpy(dtype=float)

    equity = np.ones(n)
    shares = np.zeros(n)
    cash = np.ones(n)
    weights = np.zeros(n)
    net_ret = np.zeros(n)
    gross_ret = np.zeros(n)
    costs = np.zeros(n)
    borrow_ratio = np.zeros(n)
    stop_state = np.zeros(n, dtype=bool)

    cooldown_days_left = 0
    last_target = np.nan
    derisk_state = False
    pnl_cum = np.zeros(n)
    ret20 = df["ret_20"].to_numpy(dtype=float)
    above_slow = df["above_slow"].to_numpy(dtype=bool)
    fast_above = df["fast_above_slow"].to_numpy(dtype=bool)
    vix_series = df["vix"].to_numpy(dtype=float) if "vix" in df.columns else np.full(n, np.nan)

    for t in range(1, n):
        sh_prev = shares[t - 1]
        cash_prev = cash[t - 1]
        o_cur = open_px[t]
        c_cur = close_px[t]
        c_prev = close_px[t - 1]
        tradeable = np.isfinite(o_cur) and o_cur > 0
        px_exec = c_cur if np.isfinite(c_cur) else (o_cur if tradeable else np.nan)
        vix_now = vix_series[t] if t < len(vix_series) else np.nan

        # 1) 隔夜融资：对"上日收盘后"的真实现金余额（负数=借款）计提今日利息
        rate = fin_rate(dates[t].year, p.fin_spread)
        debt = max(-cash_prev, 0.0)
        fin_cost = debt * rate / 252.0
        cash_t = cash_prev - fin_cost
        borrow_ratio[t] = debt / equity[t - 1] if equity[t - 1] > 0 else 0.0

        # 2) 目标（含组合层叠加）；调仓发生在 **今日收盘**（MOC），
        #    今日 close[t-1]->close[t] 的收益仍由"昨日收盘建立的仓位"承担。
        raw_new = 0.0 if stop_state[t - 1] else target_lev[t]
        eff_target = round(raw_new * 4) / 4 if raw_new > 0 else 0.0

        if p.dd_derisk_start > 0 and equity[t - 1] > 0:
            peak_prev = equity[:t].max()
            dd_prev = equity[t - 1] / peak_prev - 1.0 if peak_prev > 0 else 0.0
            if dd_prev <= -p.dd_derisk_start:
                derisk_state = True
            elif dd_prev >= -p.dd_rearm_level:
                derisk_state = False
            if derisk_state and eff_target > 0:
                eff_target = round(max(eff_target * p.dd_derisk_factor, 0.25) * 4) / 4
        if p.rr_derisk_60d > 0 and equity[t - 1] > 0 and t >= 61:
            rr60 = equity[t - 1] / equity[t - 61] - 1.0
            if rr60 >= p.rr_derisk_60d and eff_target > 0:
                eff_target = round(max(eff_target * p.dd_derisk_factor, 0.25) * 4) / 4

        equity_prev = equity[t - 1]
        current_lev_prev_close = (sh_prev * c_prev / equity_prev) if equity_prev > 0 else 0.0
        # 只在"目标档位变化"或"杠杆漂移超过带宽"时调仓；
        # 恒定目标（如买入持有）下漂移带很宽 -> 自然变成"不再平衡"，净值精确等于资产收益
        target_changed = (not np.isfinite(last_target)) or abs(eff_target - last_target) >= max(p.min_trade_delta, 1e-9)
        drift_exceeded = abs(current_lev_prev_close - eff_target) > p.drift_band
        if target_changed or drift_exceeded:
            w_new = eff_target
            last_target = eff_target
        else:
            w_new = current_lev_prev_close

        if np.isfinite(px_exec) and px_exec > 0:
            nav_at_close = sh_prev * c_cur + cash_t
            want_shares = w_new * nav_at_close / px_exec
            trade = want_shares - sh_prev
            slip_bps = p.cost_bps
            if p.slip_vix_coef > 0 and np.isfinite(vix_now) and vix_now > 0:
                slip_bps = min(p.slip_max_bps, p.cost_bps + p.slip_vix_coef * float(vix_now))
            trade_cost = abs(trade) * px_exec * slip_bps / 1e4
            if p.slip_move_frac > 0 and np.isfinite(c_cur) and np.isfinite(o_cur) and o_cur > 0:
                move_frac = abs(c_cur / o_cur - 1.0)
                trade_cost += abs(trade) * px_exec * min(move_frac, p.slip_move_max_frac) * p.slip_move_frac
            sh_t = want_shares
            cash_t = cash_t - trade * px_exec - trade_cost
        else:
            trade_cost = 0.0
            sh_t = sh_prev

        shares[t] = sh_t
        cash[t] = cash_t
        equity[t] = sh_t * c_cur + cash_t
        weights[t] = ((sh_t * c_cur) / equity[t]) if equity[t] > 0 else 0.0
        net_ret[t] = equity[t] / equity[t - 1] - 1.0 if equity[t - 1] > 0 else 0.0
        gross_ret[t] = net_ret[t] + (fin_cost + trade_cost) / max(equity[t - 1], 1e-9)
        costs[t] = fin_cost + trade_cost
        pnl_cum[t] = np.log(max(equity[t], 1e-12))

        # 3) 路径依赖止损层：用截至今日收盘的信息，作用于明天
        win_start = max(0, t - p.trail_lookback + 1)
        window_peak = close_px[win_start : t + 1].max()
        window_price_dd = close_px[t] / window_peak - 1.0 if window_peak > 0 else 0.0

        stop_state[t] = stop_state[t - 1]
        if t <= p.stop_warmup_days:
            stop_state[t] = False
        elif cooldown_days_left > 0:
            cooldown_days_left -= 1
            stop_state[t] = True
            if cooldown_days_left == 0:
                mom_ok = np.isfinite(ret20[t]) and ret20[t] >= p.reentry_ret20_min
                if mom_ok and above_slow[t] and fast_above[t] and target_lev[t] > 0:
                    stop_state[t] = False
        elif stop_state[t - 1]:
            if above_slow[t] and fast_above[t] and target_lev[t] > 0:
                stop_state[t] = False
        elif (p.trail_drawdown > 0) and (window_price_dd <= -p.trail_drawdown) and target_lev[t] > 0:
            cooldown_days_left = p.cooldown_days
            stop_state[t] = True

    out = pd.DataFrame(
        {
            "date": dates,
            "equity": equity,
            "net_ret": net_ret,
            "gross_ret": gross_ret,
            "pnl_cum": pnl_cum,
            "target_lev": target_lev,
            "weight": weights,
            "cost": costs,
            "borrow": borrow_ratio,
            "shares": shares,
            "cash": cash,
            "stopped": stop_state,
        }
    ).set_index("date")
    return {"label": label, "path": out, "params": p}


def metrics(path: pd.DataFrame, target: pd.Series | None = None) -> dict:
    eq = path["equity"]
    rets = path["net_ret"]
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    vol = float(rets.std() * math.sqrt(252))
    mean = float(rets.mean() * 252)
    sharpe = mean / vol if vol > 0 else 0.0
    downside = float(rets[rets < 0].std() * math.sqrt(252))
    sortino = mean / downside if downside > 0 else 0.0
    dd = eq / eq.cummax() - 1.0
    mdd = float(dd.min())
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    turnover = float(path["weight"].diff().abs().sum())
    rebalances = int((path["weight"].diff().abs() > 0.02).sum())
    lev_series = path["weight"]
    invested = lev_series > 0.10          # 真正持有仓位的天数
    exposure = float(invested.mean())
    borrow_series = path.get("borrow")
    avg_borrow = float(borrow_series.mean()) if borrow_series is not None else 0.0
    cost = path["cost"]
    avg_equity = float(eq.mean()) if float(eq.mean()) > 0 else 1.0
    return {
        "total_return_pct": round(total * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "vol_pct": round(vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(mdd * 100, 2),
        "calmar": round(calmar, 3),
        "avg_leverage": round(float(lev_series.mean()), 3),
        "avg_leverage_when_invested": round(float(lev_series[invested].mean()) if invested.any() else 0.0, 3),
        "exposure_pct": round(exposure * 100, 1),
        "avg_borrow_ratio": round(avg_borrow, 4),
        "turnover_x": round(turnover, 1),
        "rebalances": rebalances,
        # 成本拖累：相对"账户平均规模"的年化比例（长期复利后，占初始本金的累计比例会失真）
        "cost_drag_pct": round(float(cost.sum()) / avg_equity * 100, 2),
        "cost_drag_annual_pct": round(float(cost.sum()) / avg_equity / max(years, 1e-9) * 100, 2),
        "cost_drag_on_initial_pct": round(float(cost.sum()) * 100, 2),
        "years": round(years, 2),
        "final_equity": round(float(eq.iloc[-1]), 4),
    }


def buy_hold(df: pd.DataFrame, leverage: float, p: Params, label: str) -> dict:
    target = pd.Series(leverage, index=df.index, name="target_lev")
    saved = replace(p, trail_drawdown=0.0, vol_target=0.0, use_vix_gate=False)
    return run_engine(df, target, saved, label)


# --------------------------------------------------------------------------- #
# 报表
# --------------------------------------------------------------------------- #
def yearly_table(path: pd.DataFrame, bench: pd.DataFrame | None = None) -> list[dict]:
    eq = path["equity"]
    rows = []
    for year, group in eq.groupby(eq.index.year):
        ret = group.iloc[-1] / group.iloc[0] - 1.0
        dd = (group / group.cummax() - 1.0).min()
        w = path.loc[group.index, "weight"]
        row = {
            "year": int(year),
            "return_pct": round(float(ret) * 100, 2),
            "max_dd_pct": round(float(dd) * 100, 2),
            "avg_lev": round(float(w.mean()), 2),
        }
        if bench is not None:
            bg = bench["equity"].loc[group.index]
            row["bench_return_pct"] = round(float(bg.iloc[-1] / bg.iloc[0] - 1.0) * 100, 2)
            row["excess_pct"] = round(row["return_pct"] - row["bench_return_pct"], 2)
        rows.append(row)
    return rows


def monthly_returns(path: pd.DataFrame) -> pd.DataFrame:
    eq = path["equity"]
    m = eq.resample("ME").last()
    first = eq.resample("ME").first()
    ret = (m / first - 1.0) * 100.0
    table = pd.DataFrame({"year": ret.index.year, "month": ret.index.month, "ret": ret.values})
    return table.pivot(index="year", columns="month", values="ret").round(2)


def drawdown_episodes(path: pd.DataFrame, top: int = 5) -> list[dict]:
    eq = path["equity"]
    dd = eq / eq.cummax() - 1.0
    episodes = []
    in_dd = False
    start = None
    trough = None
    for date, value in dd.items():
        if value < -1e-9 and not in_dd:
            in_dd = True
            start = date
            trough = date
        elif in_dd:
            if value < dd.loc[trough]:
                trough = date
            if value >= -1e-9:
                episodes.append(
                    {
                        "start": str(start.date()),
                        "trough": str(trough.date()),
                        "recovery": str(date.date()),
                        "depth_pct": round(float(dd.loc[trough]) * 100, 2),
                        "days_to_trough": int((trough - start).days),
                        "days_to_recover": int((date - trough).days),
                    }
                )
                in_dd = False
    if in_dd and start is not None:
        episodes.append(
            {
                "start": str(start.date()),
                "trough": str(trough.date()),
                "recovery": None,
                "depth_pct": round(float(dd.loc[trough]) * 100, 2),
                "days_to_trough": int((trough - start).days),
                "days_to_recover": None,
            }
        )
    return sorted(episodes, key=lambda e: e["depth_pct"])[:top]


def trade_log(path: pd.DataFrame) -> pd.DataFrame:
    change = path["weight"].diff()
    events = path.loc[change.abs() > 0.02].copy()
    events["prev_weight"] = events["weight"] - change[change.abs() > 0.02]
    return events[["prev_weight", "weight", "target_lev", "stopped"]]


def rolling_12m(path: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    def series(p: pd.DataFrame) -> pd.Series:
        eq = p["equity"]
        return (eq / eq.shift(252) - 1.0) * 100.0

    out = pd.DataFrame({"strategy": series(path), "bench": series(bench)}).dropna()
    return out


# --------------------------------------------------------------------------- #
# 参数扫描
# --------------------------------------------------------------------------- #
def sweep(df: pd.DataFrame, start: str) -> pd.DataFrame:
    """参数稳健性网格：fast / slow / VIX 闸门 / 波动率目标。

    df 必须是含起始日之前历史的 warmup 框，内部按 start 切片，保证 MA 已 warmup。
    """
    results = []
    start_ts = pd.Timestamp(start)
    for fast in [20, 30, 40, 50, 60, 80, 100]:
        for slow in [120, 150, 180, 200, 220, 250]:
            if fast >= slow:
                continue
            base = Params(fast_window=fast, slow_window=slow)
            ind = prepare_indicators(df, replace(base, fast_window=fast, slow_window=slow)).loc[lambda d: d.index >= start_ts]
            for gate in [False, True]:
                for vol_target in [0.0, 0.15, 0.18, 0.20]:
                    p = replace(base, use_vix_gate=gate, vol_target=vol_target)
                    target = build_target_series(ind, p)
                    run = run_engine(ind, target, p, f"f{fast}-s{slow}-gate{int(gate)}-vt{vol_target}")
                    m = metrics(run["path"])
                    results.append({"fast": fast, "slow": slow, "vix_gate": gate, "vol_target": vol_target, **m})
    return pd.DataFrame(results)


def walk_forward(frame: pd.DataFrame, p: Params) -> list[dict]:
    """逐年独立重启：只看该年内策略表现（数据含 2019 起 warmup，各年之间不共享持仓）。"""
    rows = []
    ind = add_indicators(frame, p.fast_window, p.slow_window)
    for year in sorted(set(ind.index.year)):
        if year < 2020:
            continue
        seg = ind.loc[(ind.index >= pd.Timestamp(f"{year}-01-01")) & (ind.index <= pd.Timestamp(f"{year}-12-31"))]
        if len(seg) < 40:
            continue
        run = run_engine(seg, build_target_series(seg, p), p, f"wf{year}")
        m = metrics(run["path"])
        rows.append({"year": year, **m})
    return rows


def cost_sensitivity(frame: pd.DataFrame, p: Params, start: str) -> list[dict]:
    rows = []
    start_ts = pd.Timestamp(start)
    for cost in [1.0, 3.0, 5.0, 10.0]:
        for spread in [0.0, 0.005, 0.01]:
            params = replace(p, cost_bps=cost, fin_spread=spread)
            ind = add_indicators(frame, params.fast_window, params.slow_window).loc[lambda d: d.index >= start_ts]
            run = run_engine(ind, build_target_series(ind, params), params, f"c{cost}-s{spread}")
            m = metrics(run["path"])
            rows.append({"cost_bps": cost, "fin_spread": spread, **m})
    return rows


def plot_chart(nav: pd.DataFrame, main_name: str, main_weight: pd.Series, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[chart] matplotlib unavailable: {exc}")
        return

    plt.rcParams["font.size"] = 9
    for font_path in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
        if Path(font_path).exists():
            try:
                from matplotlib import font_manager

                font_manager.fontManager.addfont(font_path)
                plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
                plt.rcParams["axes.unicode_minus"] = False
                break
            except Exception:
                continue
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2, 1.0]})
    colors = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#7f8c8d", "#16a085", "#2c3e50", "#95a5a6", "#d35400"]
    for i, col in enumerate(nav.columns):
        axes[0].plot(nav.index, nav[col], label=col, linewidth=1.4 if col == main_name else 0.9,
                     color=colors[i % len(colors)], alpha=1.0 if col == main_name else 0.7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("NAV (log)")
    axes[0].set_title("QQQ 现货杠杆 <=1.5x — 2020-01 ~ 2026-08（次日开盘成交，含融资与交易成本）")
    axes[0].legend(loc="upper left", fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)

    for i, col in enumerate(nav.columns):
        if col == main_name or col.startswith("B0"):
            eq = nav[col]
            dd = (eq / eq.cummax() - 1.0) * 100
            axes[1].plot(dd.index, dd.values, label=col, linewidth=1.3, color=colors[i % len(colors)])
    axes[1].set_ylabel("drawdown %")
    axes[1].legend(loc="lower left", fontsize=7)
    axes[1].grid(alpha=0.3)

    axes[2].plot(main_weight.index, main_weight.values, color="#2c3e50")
    axes[2].set_ylabel("target leverage")
    axes[2].set_ylim(-0.05, 1.6)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[chart] {out_path}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="QQQ spot leverage (<=1.5x) backtest since 2020")
    ap.add_argument("--start", default=START_DEFAULT)
    ap.add_argument("--end", default=END_DEFAULT)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 关键口径：指标（MA200/VIX 均值等）必须用起始日之前的历史数据 warmup，
    # 否则 2020 年头 200 个交易日的 MA 全是 NaN，会人为把策略按在空仓。
    warm_frame = load_market_frame("2001-01-02", args.end)
    start_ts = pd.Timestamp(args.start)
    frame = warm_frame.loc[warm_frame.index >= start_ts]
    if frame.empty:
        raise SystemExit("no data loaded")
    p = Params()
    p = replace(p, stop_warmup_days=0)
    ind = prepare_indicators(warm_frame, p).loc[lambda d: d.index >= start_ts]
    frame = ind.copy()

    runs: dict[str, dict] = {}
    target = build_target_series(ind, p)
    runs["S 主策略 双均线阶梯+VIX闸门 (1.5x上限)"] = run_engine(ind, target, p, "main")

    variants = {
        "S 主策略 双均线阶梯+VIX闸门 (1.5x上限)": p,
        "S1 双均线阶梯 (无VIX闸门)": replace(p, use_vix_gate=False, use_recovery=False),
        "S2 200MA on/off 1.5x (无VIX闸门)": replace(p, lev_half=0.0, use_vix_gate=False, use_recovery=False),
        "S3 主策略 + 波动率目标18%": replace(p, vol_target=0.18),
        "S4 双均线阶梯 + 波动率目标18% (无VIX闸门)": replace(p, use_vix_gate=False, vol_target=0.18),
        "S5 主策略 + 波动率目标18%且覆盖VIX档": replace(p, vol_target=0.18, vol_override=True),
        "S6 主策略(无硬止损)": replace(p, trail_drawdown=0.0),
        "S7 保守 双均线1.25x上限(无VIX闸门)": replace(
            p, lev_full=1.25, lev_half=1.0, max_leverage=1.25, use_vix_gate=False, use_recovery=False
        ),
        "S8 (冗余) 主策略 + 0.25 死区": replace(p, min_trade_delta=0.25),
        "S9 (冗余) 主策略 + 0.5 死区": replace(p, min_trade_delta=0.5),
    }
    for name, params in variants.items():
        vdf = ind if params.fast_window == p.fast_window and params.slow_window == p.slow_window else prepare_indicators(warm_frame, params).loc[lambda d: d.index >= start_ts]
        runs[name] = run_engine(vdf, build_target_series(vdf, params), params, name)

    runs["B0 QQQ 买入持有 1x"] = buy_hold(ind, 1.0, p, "bh1")
    runs["B1 QQQ 恒定 1.5x"] = buy_hold(ind, 1.5, p, "bh15")

    summary = {name: metrics(run["path"]) for name, run in runs.items()}
    bench = runs["B0 QQQ 买入持有 1x"]

    # 导出
    nav = pd.DataFrame({name: run["path"]["equity"] for name, run in runs.items()})
    nav.to_csv(out_dir / "qqq_spot_leverage_nav.csv")
    pd.DataFrame(summary).T.to_csv(out_dir / "qqq_spot_leverage_summary.csv")
    yearly = yearly_table(runs["S 主策略 双均线阶梯+VIX闸门 (1.5x上限)"]["path"], bench["path"])
    pd.DataFrame(yearly).to_csv(out_dir / "qqq_spot_leverage_yearly.csv", index=False)
    all_year_returns = {}
    for name, run in runs.items():
        all_year_returns[name] = {str(row["year"]): row["return_pct"] for row in yearly_table(run["path"])}
        all_year_returns[name]["MDD"] = metrics(run["path"])["max_drawdown_pct"]
    pd.DataFrame(all_year_returns).T.to_csv(out_dir / "qqq_spot_leverage_yearly_all_variants.csv")
    monthly_returns(runs["S 主策略 双均线阶梯+VIX闸门 (1.5x上限)"]["path"]).to_csv(out_dir / "qqq_spot_leverage_monthly.csv")
    tl = trade_log(runs["S 主策略 双均线阶梯+VIX闸门 (1.5x上限)"]["path"])
    tl.to_csv(out_dir / "qqq_spot_leverage_trades.csv")

    sweep_df = sweep(warm_frame, args.start)
    sweep_df.to_csv(out_dir / "qqq_spot_leverage_sweep.csv", index=False)

    # 逐年独立重启 & 成本敏感性（walk-forward 视角）
    wf = walk_forward(warm_frame, p)
    pd.DataFrame(wf).to_csv(out_dir / "qqq_spot_leverage_walk_forward.csv", index=False)
    costs = cost_sensitivity(warm_frame, p, args.start)
    pd.DataFrame(costs).to_csv(out_dir / "qqq_spot_leverage_cost_sensitivity.csv", index=False)

    # 杠杆上限敏感性：同一信号，把上限截断到不同 L
    lev_sens = []
    for cap in [1.0, 1.25, 1.5, 1.75, 2.0]:
        params = replace(p, lev_full=cap, max_leverage=cap)
        cap_df = prepare_indicators(warm_frame, params).loc[lambda d: d.index >= start_ts]
        run = run_engine(cap_df, build_target_series(cap_df, params), params, f"cap{cap}")
        lev_sens.append({"leverage_cap": cap, **metrics(run["path"])})
    pd.DataFrame(lev_sens).to_csv(out_dir / "qqq_spot_leverage_cap_sensitivity.csv", index=False)

    # 起始年敏感性（与主回测同口径：先用 2001 起数据 warmup 出信号，再截取区间独立跑）
    start_sens = []
    warm = prepare_indicators(warm_frame, p)
    for start in ["2020-01-02", "2021-01-04", "2022-01-03", "2023-01-03"]:
        params = replace(p)
        sub = warm.loc[warm.index >= pd.Timestamp(start)]
        run = run_engine(sub, build_target_series(sub, params), params, f"start{start}")
        start_sens.append({"start": start, **metrics(run["path"])})
    pd.DataFrame(start_sens).to_csv(out_dir / "qqq_spot_leverage_start_sensitivity.csv", index=False)

    # 长样本分段（QQQ CSV 在 2005-01 ~ 2011-04 有 6.4 年缺口，不能连起来复利）
    gap_days = warm_frame.index.to_series().diff().dt.days
    gaps = [
        {"from": str(warm_frame.index[i - 1].date()), "to": str(warm_frame.index[i].date()), "days": int(gap_days.iloc[i])}
        for i in range(1, len(warm_frame))
        if gap_days.iloc[i] is not None and gap_days.iloc[i] > 30
    ]
    long_sample = {}
    for label, seg_start, seg_end in [
        ("2001_bear_2001_2004", "2001-01-02", "2004-12-31"),
        ("modern_2011_2026", "2011-04-26", args.end),
    ]:
        seg = prepare_indicators(warm_frame, p)
        seg = seg.loc[(seg.index >= pd.Timestamp(seg_start)) & (seg.index <= pd.Timestamp(seg_end))]
        run = run_engine(seg, build_target_series(seg, p), p, label)
        bh = buy_hold(seg, 1.0, p, label + "-bh")
        long_sample[label] = {
            "start": seg_start,
            "end": str(seg.index[-1].date()),
            "strategy": metrics(run["path"]),
            "benchmark_1x": metrics(bh["path"]),
            "yearly": yearly_table(run["path"]),
            "benchmark_yearly": yearly_table(bh["path"]),
        }
        pd.DataFrame(yearly_table(run["path"])).to_csv(out_dir / f"qqq_spot_leverage_long_{label}_yearly.csv", index=False)
    pd.DataFrame(long_sample["modern_2011_2026"]["yearly"]).to_csv(
        out_dir / "qqq_spot_leverage_long_sample_yearly.csv", index=False
    )

    # 最新状态 + 前瞻信号（QQQ 数据到 2026-09-02，但 VIX 只到 2026-08-14，
    # 所以长端信号用"最后有效 VIX"延续并显式标注）
    last = ind.iloc[-1]
    latest_target = float(build_target_series(ind, p).iloc[-1])
    ext = load_market_frame("2001-01-02", QQQ_LAST_KNOWN)
    for col in ["vix", "vix3m"]:
        ext[col] = ext[col].ffill()
    ext_ind = prepare_indicators(ext, p)
    ext_target = build_target_series(ext_ind, p)
    forward = []
    for ts_start in [d for d in ext_ind.index if d > ind.index[-1]]:
        row = ext_ind.loc[ts_start]
        forward.append(
            {
                "date": str(ts_start.date()),
                "qqq_adj_close": round(float(row["adj_close"]), 2),
                "ma50": round(float(row["ma_fast"]), 2) if pd.notna(row["ma_fast"]) else None,
                "ma200": round(float(row["ma_slow"]), 2) if pd.notna(row["ma_slow"]) else None,
                "target_lev_with_stale_vix": round(float(ext_target.loc[ts_start]), 2),
            }
        )
    latest = {
        "date": str(ind.index[-1].date()),
        "qqq_adj_close": round(float(last["adj_close"]), 2),
        "qqq_raw_close": round(float(last["raw_close"]), 2),
        "ma_fast": round(float(last["ma_fast"]), 2),
        "ma_slow": round(float(last["ma_slow"]), 2),
        "vix": round(float(last["vix"]), 2) if pd.notna(last["vix"]) else None,
        "vix3m": round(float(last["vix3m"]), 2) if pd.notna(last["vix3m"]) else None,
        "vix_term": round(float(last["vix_term"]), 3) if pd.notna(last["vix_term"]) else None,
        "rv20_pct": round(float(last["rv20"]) * 100, 1) if pd.notna(last["rv20"]) else None,
        "target_lev_backtest_end": round(latest_target, 2),
        "qqq_last_data_date": QQQ_LAST_KNOWN,
        "vix_last_valid_date": str(ind["vix"].dropna().index[-1].date()),
        "forward_signals": forward,
    }

    payload = {
        "period": {"start": str(ind.index[0].date()), "end": str(ind.index[-1].date()), "days": int(len(ind))},
        "data_quality": data_quality(ind),
        "params": p.__dict__,
        "fin_rates": FIN_RATES,
        "summary": summary,
        "yearly": yearly,
        "dd_episodes": drawdown_episodes(runs["S 主策略 双均线阶梯+VIX闸门 (1.5x上限)"]["path"]),
        "cap_sensitivity": lev_sens,
        "start_sensitivity": start_sens,
        "walk_forward": wf,
        "cost_sensitivity": costs,
        "long_sample": long_sample,
        "data_gaps": gaps,
        "latest": latest,
    }
    (out_dir / "qqq_spot_leverage_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    plot_chart(nav, "S 主策略 双均线阶梯+VIX闸门 (1.5x上限)",
               runs["S 主策略 双均线阶梯+VIX闸门 (1.5x上限)"]["path"]["weight"],
               out_dir / "qqq_spot_leverage_nav.png")

    print(f"回测区间 {ind.index[0].date()} ~ {ind.index[-1].date()}  ({len(ind)} 个交易日)")
    print(f"{'策略':<34}{'累计':>10}{'CAGR':>8}{'MDD':>8}{'Sharpe':>8}{'Calmar':>8}{'均杠杆':>8}")
    for name, m in summary.items():
        print(
            f"{name:<34}{m['total_return_pct']:>+9.1f}%{m['cagr_pct']:>7.1f}%"
            f"{m['max_drawdown_pct']:>7.1f}%{m['sharpe']:>8.2f}{m['calmar']:>8.2f}{m['avg_leverage']:>8.2f}"
        )
    print(f"\n最新 {latest['date']}: QQQ {latest['qqq_adj_close']} / MA{ p.fast_window } {latest['ma_fast']} / "
          f"MA{ p.slow_window } {latest['ma_slow']} / VIX {latest['vix']} -> 目标 {latest['target_lev_backtest_end']}x")
    if forward:
        print(f"前瞻信号（VIX 停留在 {latest['vix_last_valid_date']}）: "
              f"{forward[-1]['date']} -> {forward[-1]['target_lev_with_stale_vix']}x")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
