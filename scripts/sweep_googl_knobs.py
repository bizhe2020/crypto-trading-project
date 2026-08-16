#!/usr/bin/env python3
"""GOOGL 策略旋钮单维扫描（shadow gate / stop / ramp）。

在 compare_qqq_googl_router 同一窗口（2026-03-04 → 08-07）上，逐旋钮一维扫描
`run_googl_4h_replay` 的参数，确认单 GOOGL 策略还有多少可提升空间。

窗口注意：replay 仍跑完整 2024-01→2026-08 历史（equity_dd 门有运行历史，更真实），
但只取窗口内的日频 return 评估（与 router 比较同口径，从 1000 起复合）。

过拟合警戒：窗口仅 ~157 天，单维扫描的"最优"点必须在邻域内单调/稳健才采纳，
否则维持基线。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_qqq_googl_router import (  # noqa: E402
    build_googl_daily,
    max_drawdown_pct,
    START,
    END,
    GOOGL_4H,
    GOOGL_FUNDING,
    GOOGL_SIGNAL_CSV,
    GOOGL_TIERS,
    GOOGL_RUNTIME,
)
from scripts.replay_googl_usdt_4h import attach_googl_daily_state, run_googl_4h_replay, load_funding  # noqa: E402


def load_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    gcfg = json.loads(GOOGL_RUNTIME.read_text())
    googl_4h = pd.read_feather(GOOGL_4H)
    googl_4h["date"] = pd.to_datetime(googl_4h["date"], utc=True)
    googl_4h = googl_4h.sort_values("date").reset_index(drop=True)
    funding = load_funding(GOOGL_FUNDING)
    gsig_raw = pd.read_csv(GOOGL_SIGNAL_CSV)
    gsig_raw["date"] = pd.to_datetime(gsig_raw["date"], utc=True)
    merged = attach_googl_daily_state(googl_4h, gsig_raw)
    return merged, funding, gcfg


def eval_window(daily: pd.DataFrame) -> tuple[float, float, int]:
    """窗口内评估：return / maxDD / 持仓天数（与 router 比较同口径）。"""
    win = daily[(daily["date"] >= START.normalize()) & (daily["date"] <= END.normalize())].reset_index(drop=True)
    eq = (1.0 + win["googl_return"]).cumprod() * 1000.0
    total = (eq.iloc[-1] / 1000.0 - 1.0) * 100.0
    active_days = int(win["googl_active"].sum())
    return total, max_drawdown_pct(eq), active_days


def run_one(merged, funding, gcfg, **overrides) -> pd.DataFrame:
    params = dict(
        stop_loss_pct=float(gcfg.get("stop_loss_pct", 4.0)),
        equity_dd_stop_pct=15.0,
        equity_dd_cooldown_bars=20,
        reentry_rule="clear",
        reentry_clear_bars=2,
        ramp_confirm_pct=float(gcfg.get("ramp_confirm_pct", 0.5)),
        ramp_pre_stop_pct=float(gcfg.get("ramp_pre_stop_pct", 2.0)),
    )
    params.update(overrides)
    result = run_googl_4h_replay(
        merged,
        funding,
        leverage_tiers=GOOGL_TIERS,
        taker_fee_rate=float(gcfg.get("taker_fee_rate", 0.0005)),
        slippage_bps=float(gcfg.get("slippage_bps", 5.0)),
        initial_capital=float(gcfg.get("initial_capital", 1000.0)),
        include_funding=True,
        capture_open_gaps=True,
        entry_price_col=None,
        ramp_stop_pct=0.0,
        be_lock_pct=0.0,
        **params,
    )
    path = result["path"].copy()
    path["day"] = path["date"].dt.floor("D")
    daily = pd.DataFrame(
        {
            "date": path.groupby("day")["date"].last().dt.floor("D"),
            "googl_return": path.groupby("day")["daily_return"].apply(lambda s: float((1.0 + s).prod() - 1.0)),
            "googl_active": (
                path.groupby("day")["holding"].any()
                | path.groupby("day")["entered_today"].any()
                | path.groupby("day")["exited_today"].any()
            ),
        }
    ).reset_index(drop=True)
    return daily


def main() -> None:
    merged, funding, gcfg = load_base()
    base_daily = run_one(merged, funding, gcfg)
    bt, bd, ba = eval_window(base_daily)
    print(f"基线 GOOGL: total={bt:+.2f}%  maxDD={bd:.2f}%  active_days={ba}")
    print("=" * 84)

    sweeps = {
        "equity_dd_stop_pct (门 DD)": [8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 100.0],
        "equity_dd_cooldown_bars": [5, 10, 20, 40, 80],
        "stop_loss_pct": [3.0, 3.5, 4.0, 5.0, 6.0, 8.0],
        "ramp_pre_stop_pct": [0.0, 0.5, 1.0, 2.0, 3.0],
        "ramp_confirm_pct": [0.2, 0.3, 0.5, 0.8, 1.0, 2.0],
    }
    for label, key, values in [
        ("equity_dd_stop_pct (门 DD)", "equity_dd_stop_pct", sweeps["equity_dd_stop_pct (门 DD)"]),
        ("equity_dd_cooldown_bars", "equity_dd_cooldown_bars", sweeps["equity_dd_cooldown_bars"]),
        ("stop_loss_pct", "stop_loss_pct", sweeps["stop_loss_pct"]),
        ("ramp_pre_stop_pct", "ramp_pre_stop_pct", sweeps["ramp_pre_stop_pct"]),
        ("ramp_confirm_pct", "ramp_confirm_pct", sweeps["ramp_confirm_pct"]),
    ]:
        print(f"\n--- {label} (其余=基线) ---")
        for v in values:
            d = run_one(merged, funding, gcfg, **{key: v})
            t, dd, ad = eval_window(d)
            delta = t - bt
            print(f"  {key}={v:<8}  total={t:+9.2f}% (Δ{delta:+9.2f})  maxDD={dd:6.2f}%  active_days={ad:>3}")


if __name__ == "__main__":
    main()
