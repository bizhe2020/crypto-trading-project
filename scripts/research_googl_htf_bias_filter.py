#!/usr/bin/env python3
"""GOOGL 4h 执行层回测 × SMC 高阶结构 bias 过滤器研究。

在"比特币聪明钱（SMC）能否提升 GOOGL 收益率"研究中，15m 粒度 SMC 独立策略
被证实在 GOOGL 上无边缘（各配置 win rate 均低于盈亏平衡 9pp）。本脚本测试
SMC 中最可能迁移的一个概念 —— 高阶（4h）结构 bias 对齐（"只顺着结构做"）：

    allow_long_biased = allow_long AND (4h 结构 bias 满足过滤规则)

验证规则:
    - baseline: 原策略（无过滤）
    - bull_only: 仅 4h 结构为 BULL 时允许（跨 bar 持有时若翻 BEAR 则开盘平仓）
    - bull_or_none: BULL 或 NONE 允许（仅排除 BEAR）

用法:
    python scripts/research_googl_htf_bias_filter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_googl_usdt_4h import attach_googl_daily_state, load_okx_4h, run_googl_4h_replay  # noqa: E402
from scripts.research_smc_standalone_v1 import htf_structure_bias  # noqa: E402
from strategy.scalp_robust_v2_core import Candle, dataframe_to_candles, precompute_swings  # noqa: E402

DEFAULT_SIGNAL = ROOT / "var" / "runtime" / "googl" / "googl_daily_signal.csv"
DEFAULT_OKX_4H = ROOT / "data" / "okx" / "futures" / "GOOGL_USDT_USDT-4h-futures.feather"
DEFAULT_CONFIG = ROOT / "config" / "config.paper.googl-high-leverage-runtime.json"


def compute_4h_bias(bars: pd.DataFrame, swing_n: int = 2, lookback: int = 80, asof_lag: int = 2) -> list[str]:
    """逐 bar 计算 4h 结构 bias（SMC 高阶结构），与 research_smc 相同定义。

    asof_lag 消除前视泄漏：swing 点 i 需 candle i+1 收盘确认；在本 bar open
    做入场决策时，可用信息到上一根 close（candle idx-1），因此最新可确认的
    swing 在 idx-2。asof_lag=2 即用 htf_structure_bias(..., idx-2)。
    """
    c4h = dataframe_to_candles(bars[["date", "open", "high", "low", "close"]])
    highs, lows = precompute_swings(c4h, n=swing_n, lookback=lookback)
    biases: list[str] = []
    for idx in range(len(c4h)):
        biases.append(htf_structure_bias(c4h, highs, lows, max(0, idx - asof_lag)))
    return biases


def main() -> None:
    import json

    bars = load_okx_4h(DEFAULT_OKX_4H)
    signal = pd.read_csv(DEFAULT_SIGNAL)
    config = json.loads(DEFAULT_CONFIG.read_text())

    merged = attach_googl_daily_state(bars, signal)
    merged["bias"] = compute_4h_bias(merged)

    tiers = {
        "offense": float(config["offense_leverage"]),
        "base": float(config["base_leverage"]),
        "defense": float(config["defense_leverage"]),
        "flat": 0.0,
    }
    kwargs = {
        "leverage_tiers": tiers,
        "stop_loss_pct": float(config["stop_loss_pct"]),
        "taker_fee_rate": float(config["taker_fee_rate"]),
        "slippage_bps": float(config["slippage_bps"]),
        "initial_capital": float(config["initial_capital"]),
        "equity_dd_stop_pct": float(config["shadow_gate_replay_profile"]["equity_dd_stop_pct"]),
        "equity_dd_cooldown_bars": int(config["shadow_gate_replay_profile"]["equity_dd_cooldown_bars"]),
        "reentry_rule": config["shadow_gate_replay_profile"]["reentry_rule"],
        "reentry_clear_bars": int(config["shadow_gate_replay_profile"]["reentry_clear_bars"]),
    }

    def run_variant(name: str, bias_ok: set[str] | None) -> None:
        df = merged.copy()
        if bias_ok is not None:
            df["allow_long"] = df["allow_long"] & df["bias"].isin(bias_ok)
        result = run_googl_4h_replay(df, None, **kwargs)
        s = result["summary"]
        print(
            f"{name:<18} ret={s['total_return_pct']:>9.2f}%  maxDD={s['max_drawdown_pct']:>6.2f}%  "
            f"trades={s['trades']:>3d}  win={s['win_rate_pct']:>6.2f}%  "
            f"invested={s['invested_bars']:>5d}/{s['bars']}  gate={s['equity_dd_gate_events']}"
        )

    print(f"bars={len(merged)}  bias 分布: {merged['bias'].value_counts().to_dict()}")
    print(f"杠杆档: {tiers}  stop={kwargs['stop_loss_pct']}%")
    print()
    run_variant("baseline", None)
    run_variant("bull_only", {"BULL"})
    run_variant("bull_or_none", {"BULL", "NONE"})


if __name__ == "__main__":
    main()
