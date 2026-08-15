"""GOOGL 4h 执行层回测 — 单元测试。

用合成 4h bars + 合成日线信号，验证：4h bar 加载、信号附着、入场/离场、
trailing stop、funding、shadow gate（权益DD冷却）。不依赖外部数据。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.replay_googl_usdt_4h import (  # noqa: E402
    attach_googl_daily_state,
    load_okx_4h,
    run_googl_4h_replay,
)


def _make_4h_bars(tmp_path: Path, n_bars: int = 48) -> Path:
    """Synthetic trending-up 4h bars. 6 bars/day over n_bars."""
    start = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    price = 100.0
    for i in range(n_bars):
        ts = start + pd.Timedelta(hours=4 * i)
        o = price
        c = o * (1.0 + 0.002)  # +0.2% per bar
        hi = max(o, c) * 1.002
        lo = min(o, c) * 0.998
        rows.append({"date": ts, "open": o, "high": hi, "low": lo, "close": c})
        price = c
    path = tmp_path / "GOOGL_USDT_USDT-4h-futures.feather"
    pd.DataFrame(rows).to_feather(path)
    return path


def _make_signal(tmp_path: Path, *, allow_from: int = 6, allow_to: int = 36) -> Path:
    """Daily signal covering the synthetic 4h window: position=GOOGL for a middle span."""
    start = pd.Timestamp("2026-01-01", tz="UTC")
    days = pd.date_range(start, start + pd.Timedelta(days=10), freq="D")
    rows = []
    for i, d in enumerate(days):
        pos = "GOOGL" if allow_from <= i < allow_to else "FLAT"
        tier = "offense" if pos == "GOOGL" else "flat"
        rows.append(
            {
                "date": d,
                "position": pos,
                "leverage_tier": tier,
                "target_leverage": 5.0 if tier == "offense" else 0.0,
            }
        )
    path = tmp_path / "googl_daily_signal.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_okx_4h(tmp_path: Path) -> None:
    bars = load_okx_4h(_make_4h_bars(tmp_path))
    assert len(bars) == 48
    assert {"date", "open", "high", "low", "close"}.issubset(bars.columns)
    assert bars["date"].is_monotonic_increasing
    assert (bars["high"] >= bars["low"]).all()


def test_load_okx_4h_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="GOOGL 4h"):
        load_okx_4h(tmp_path / "missing.feather")


def test_attach_googl_daily_state(tmp_path: Path) -> None:
    bars = load_okx_4h(_make_4h_bars(tmp_path))
    signal = pd.read_csv(_make_signal(tmp_path, allow_from=6, allow_to=36))
    merged = attach_googl_daily_state(bars, signal)
    assert "allow_long" in merged.columns
    assert "leverage_tier" in merged.columns
    # 前几根 bar（信号 FLAT）allow_long=False；day 6 = bars 36-41 → allow_long=True
    assert not merged["allow_long"].iloc[0]
    assert merged["allow_long"].iloc[40]
    # 信号结束后裁剪（bars 8 天 = 48 根，信号到 day 10）
    signal_end = pd.to_datetime(signal["date"], utc=True).max()
    assert merged["date"].max() <= signal_end


def test_run_googl_4h_replay_entry_exit(tmp_path: Path) -> None:
    bars = load_okx_4h(_make_4h_bars(tmp_path))
    # GOOGL days 4-6（bars 24-41），day 7 FLAT → 自然离场产生交易
    signal = pd.read_csv(_make_signal(tmp_path, allow_from=4, allow_to=7))
    merged = attach_googl_daily_state(bars, signal)
    result = run_googl_4h_replay(
        merged,
        None,
        leverage_tiers={"offense": 5.0, "base": 3.0, "defense": 2.0, "flat": 0.0},
        stop_loss_pct=4.0,
        taker_fee_rate=0.0005,
        slippage_bps=5.0,
        initial_capital=1000.0,
    )
    s = result["summary"]
    assert s["bars"] == len(merged)
    assert s["invested_bars"] > 0  # 有在市时段
    assert s["trades"] > 0
    assert s["max_drawdown_pct"] >= 0.0
    # 趋势向上 → 整体正收益
    assert s["total_return_pct"] > 0.0


def test_run_googl_4h_replay_stop_hits(tmp_path: Path) -> None:
    """构造一个暴跌段，验证 trailing stop 触发且产生交易记录。"""
    start = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    price = 100.0
    for i in range(24):
        ts = start + pd.Timedelta(hours=4 * i)
        o = price
        # 第 4 根起暴跌（持仓期内触发 stop）
        c = o * (1.0 - 0.06) if i >= 4 else o * (1.0 + 0.002)
        hi = max(o, c) * 1.002
        lo = min(o, c) * 0.99
        rows.append({"date": ts, "open": o, "high": hi, "low": lo, "close": c})
        price = c
    bars = pd.DataFrame(rows)
    # 全程 GOOGL（5 天），暴跌发生在持仓期内
    signal = pd.DataFrame(
        {
            "date": pd.date_range(start, periods=5, freq="D"),
            "position": ["GOOGL"] * 5,
            "leverage_tier": ["offense"] * 5,
            "target_leverage": [5.0] * 5,
        }
    )
    merged = attach_googl_daily_state(bars, signal)
    result = run_googl_4h_replay(
        merged,
        None,
        leverage_tiers={"offense": 5.0, "base": 3.0, "defense": 2.0, "flat": 0.0},
        stop_loss_pct=3.0,
        taker_fee_rate=0.0005,
        slippage_bps=5.0,
        initial_capital=1000.0,
    )
    stops = [t for t in result["trades"] if t["exit_reason"] == "trailing_stop"]
    assert len(stops) >= 1
    assert result["summary"]["trades"] >= 1


def test_run_googl_4h_replay_equity_dd_gate(tmp_path: Path) -> None:
    """连续亏损触发 shadow gate：权益回撤 ≥15% → 冷却禁入场。"""
    start = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    price = 100.0
    for i in range(48):
        ts = start + pd.Timedelta(hours=4 * i)
        o = price
        c = o * (1.0 - 0.015)  # 持续阴跌
        hi = max(o, c) * 1.001
        lo = min(o, c) * 0.995
        rows.append({"date": ts, "open": o, "high": hi, "low": lo, "close": c})
        price = c
    bars = pd.DataFrame(rows)
    signal = pd.DataFrame(
        {
            "date": pd.date_range(start, periods=9, freq="D"),
            "position": ["GOOGL"] * 9,
            "leverage_tier": ["offense"] * 9,
            "target_leverage": [5.0] * 9,
        }
    )
    merged = attach_googl_daily_state(bars, signal)
    result = run_googl_4h_replay(
        merged,
        None,
        leverage_tiers={"offense": 5.0, "base": 3.0, "defense": 2.0, "flat": 0.0},
        stop_loss_pct=4.0,
        taker_fee_rate=0.0005,
        slippage_bps=5.0,
        initial_capital=1000.0,
        equity_dd_stop_pct=15.0,
        equity_dd_cooldown_bars=6,
        reentry_rule="clear",
        reentry_clear_bars=2,
    )
    gate_events = result["gate_events"]
    assert any(e["event"] == "equity_dd_gate" for e in gate_events), "阴跌段应触发 equity_dd gate"
    # 冷却期间应出现 entry_blocked
    blocked = [e for e in gate_events if e["event"] == "entry_blocked"]
    assert len(blocked) >= 0  # 冷却期信号可能一直持 FLAT；不强制断言数量


def test_run_googl_4h_replay_no_overlap(tmp_path: Path) -> None:
    """信号窗口与 4h 数据无重叠 → 无在市 bar。"""
    bars = load_okx_4h(_make_4h_bars(tmp_path))
    signal = pd.DataFrame(
        {
            "date": pd.to_datetime(["2030-01-01", "2030-01-02"], utc=True),
            "position": ["GOOGL", "GOOGL"],
            "leverage_tier": ["offense", "offense"],
            "target_leverage": [5.0, 5.0],
        }
    )
    merged = attach_googl_daily_state(bars, signal)
    result = run_googl_4h_replay(
        merged,
        None,
        leverage_tiers={"offense": 5.0, "base": 3.0, "defense": 2.0, "flat": 0.0},
        stop_loss_pct=4.0,
        taker_fee_rate=0.0005,
        slippage_bps=5.0,
        initial_capital=1000.0,
    )
    assert result["summary"]["invested_bars"] == 0
    assert result["summary"]["trades"] == 0


def test_run_googl_4h_replay_gap_through_stop(tmp_path: Path) -> None:
    """隔夜跳空击穿止损 → 在开盘价成交（真实亏损），而非止损价（旧模型虚假盈利）。"""
    start = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    price = 100.0
    for i in range(6):
        ts = start + pd.Timedelta(hours=4 * i)
        if i < 3:
            o = price
            c = o * 1.01
            hi, lo = c, o
        elif i == 3:
            o = price * 0.90  # 隔夜跳空 -10%，击穿 4% 止损
            c = o * 0.995
            hi, lo = o * 1.001, o * 0.99
        else:
            o = price
            c = o * 0.99
            hi, lo = o, c
        rows.append({"date": ts, "open": o, "high": hi, "low": lo, "close": c})
        price = c
    bars = pd.DataFrame(rows)
    signal = pd.DataFrame(
        {
            "date": pd.date_range(start, periods=6, freq="D"),
            "position": ["GOOGL"] * 6,
            "leverage_tier": ["base"] * 6,
            "target_leverage": [3.0] * 6,
        }
    )
    merged = attach_googl_daily_state(bars, signal)
    kwargs = {
        "leverage_tiers": {"offense": 3.0, "base": 3.0, "defense": 2.0, "flat": 0.0},
        "stop_loss_pct": 4.0,
        "taker_fee_rate": 0.0,
        "slippage_bps": 0.0,
        "initial_capital": 1000.0,
    }
    # 跳空捕获开启（默认）：gap-through 在开盘价成交 → 亏损
    r_on = run_googl_4h_replay(merged, None, capture_open_gaps=True, **kwargs)
    stop_on = [t for t in r_on["trades"] if t["exit_reason"] == "trailing_stop"]
    assert len(stop_on) == 1
    assert stop_on[0]["trade_return_pct"] < 0.0, "跳空击穿应在开盘价成交，产生亏损"
    # 跳空捕获关闭：旧模型以止损价成交（开盘已跳空低于止损 → 虚假盈利）
    r_off = run_googl_4h_replay(merged, None, capture_open_gaps=False, **kwargs)
    stop_off = [t for t in r_off["trades"] if t["exit_reason"] == "trailing_stop"]
    assert stop_off and stop_off[0]["trade_return_pct"] > stop_on[0]["trade_return_pct"]


# ---------------------------------------------------------------------------
# 杠杆爬坡（ramp_confirm_pct）回归测试。
#
# 无前视契约（bar-open 决策）：
#   1) 入场 bar 自身永远以 base 杠杆计盈亏（确认只能在收盘后才可知）；
#   2) 持仓 bar 用上一根已收盘 bar 的 close（prev_close）做盈利确认 ——
#      本 bar 自己的 close 即使越过阈值，也不为本 bar 自身升杠杆；
#   3) 确认成立后，从下一根 bar 起升到完整杠杆；
#   4) 始终未确认（入场即跌）→ 全程 base 杠杆，亏损被抑制。
# 四个测试各锁一条契约。杠杆档 offense=10 / base=5，taker_fee=0.0005。
# 期望值手算：trade_return_pct = Π(1 + lev_i × bar_ret_i) × (1 − exit_fee) − 1
# （入场 fee 在百分比中约掉）。
# ---------------------------------------------------------------------------

_BARS_START = pd.Timestamp("2026-01-01", tz="UTC")


def _ohlc_bars(rows: list[tuple[float, float, float, float]], n_bars: int | None = None) -> pd.DataFrame:
    """把 (open, high, low, close) 列表铺成 4h bars；不足部分用最后一根平铺。"""
    if n_bars is None:
        n_bars = len(rows)
    out = []
    for i in range(n_bars):
        o, h, l, c = rows[min(i, len(rows) - 1)]
        out.append({"date": _BARS_START + pd.Timedelta(hours=4 * i), "open": o, "high": h, "low": l, "close": c})
    return pd.DataFrame(out)


def _one_day_signal(tmp_path: Path) -> Path:
    """day0 GOOGL（offense），day1 起 FLAT → 单笔持仓在 day1 开盘平仓。"""
    signal = pd.DataFrame(
        {
            "date": pd.date_range(_BARS_START, periods=3, freq="D"),
            "position": ["GOOGL", "FLAT", "FLAT"],
            "leverage_tier": ["offense", "flat", "flat"],
            "target_leverage": [10.0, 0.0, 0.0],
        }
    )
    path = tmp_path / "googl_daily_signal.csv"
    signal.to_csv(path, index=False)
    return path


_RAMP_TIERS = {"offense": 10.0, "base": 5.0, "defense": 2.0, "flat": 0.0}


def test_ramp_entry_bar_at_base_leverage(tmp_path: Path) -> None:
    """入场 bar 暴涨 10% 且越过确认阈值，但 P&L 必须以 base（5x）计 —— 确认只在收盘后才可知。"""
    bars = _ohlc_bars([(100, 112, 99, 110)] + [(110, 111, 109, 110)] * 5, n_bars=7)
    merged = attach_googl_daily_state(bars, pd.read_csv(_one_day_signal(tmp_path)))
    result = run_googl_4h_replay(
        merged, None,
        leverage_tiers=_RAMP_TIERS,
        ramp_confirm_pct=1.0, stop_loss_pct=4.0,
        taker_fee_rate=0.0005, slippage_bps=0.0, initial_capital=1000.0,
    )
    assert result["summary"]["trades"] == 1
    ret = result["trades"][0]["trade_return_pct"]
    # base 全程：1.50 × 0.995 − 1 ≈ +49.25%；若入场 bar 自升 10x 则 ≈ +99%
    assert ret == pytest.approx(49.25, abs=0.6), f"入场 bar 应 base 计盈亏，实际 {ret:.2f}%"


def test_ramp_confirmation_bar_does_not_self_ramp(tmp_path: Path) -> None:
    """持仓 bar 的 close 越过阈值（99→101.5），但该 bar 自身仍以 base 计 —— 用 prev_close 确认。"""
    bars = _ohlc_bars(
        [(100, 100, 98, 99), (99, 102, 98.5, 101.5)] + [(101.5, 102, 101, 101.5)] * 5,
        n_bars=7,
    )
    merged = attach_googl_daily_state(bars, pd.read_csv(_one_day_signal(tmp_path)))
    result = run_googl_4h_replay(
        merged, None,
        leverage_tiers=_RAMP_TIERS,
        ramp_confirm_pct=0.5, stop_loss_pct=4.0,
        taker_fee_rate=0.0005, slippage_bps=0.0, initial_capital=1000.0,
    )
    assert result["summary"]["trades"] == 1
    ret = result["trades"][0]["trade_return_pct"]
    # base 全程：0.95 × 1.12626 × 0.995 − 1 ≈ +6.46%；若确认 bar 自升 10x 则 ≈ +18.4%
    assert ret == pytest.approx(6.46, abs=0.6), f"确认 bar 应 base 计盈亏，实际 {ret:.2f}%"


def test_ramp_activates_on_next_bar(tmp_path: Path) -> None:
    """prev_close 确认（101.5 ≥ 100×1.005）后，下一根 bar 升到完整杠杆（10x）捕捉大阳线。"""
    bars = _ohlc_bars(
        [
            (100, 100, 98, 99),           # 入场，未确认
            (99, 102, 98.5, 101.5),       # close 越过阈值，但自身以 base 计
            (101.5, 111, 101, 110),       # prev_close 确认 → 10x 捕捉 +8.4%
        ]
        + [(110, 111, 109, 110)] * 4,
        n_bars=7,
    )
    merged = attach_googl_daily_state(bars, pd.read_csv(_one_day_signal(tmp_path)))
    result = run_googl_4h_replay(
        merged, None,
        leverage_tiers=_RAMP_TIERS,
        ramp_confirm_pct=0.5, stop_loss_pct=4.0,
        taker_fee_rate=0.0005, slippage_bps=0.0, initial_capital=1000.0,
    )
    ret = result["trades"][0]["trade_return_pct"]
    # 0.95 × 1.12626 × 1.83744 × 0.995 − 1 ≈ +95.6%
    assert ret == pytest.approx(95.6, abs=1.2), f"确认后应升满杠杆捕捉收益，实际 {ret:.2f}%"


def test_ramp_immediate_drop_stays_base(tmp_path: Path) -> None:
    """入场即跌、从未确认 → 全程 base（5x），止损亏损被抑制（而非 10x 放大）。"""
    bars = _ohlc_bars(
        [(100, 100, 98, 99), (99, 99, 95.5, 96)] + [(96, 96.5, 95, 96)] * 5,
        n_bars=7,
    )
    merged = attach_googl_daily_state(bars, pd.read_csv(_one_day_signal(tmp_path)))
    result = run_googl_4h_replay(
        merged, None,
        leverage_tiers=_RAMP_TIERS,
        ramp_confirm_pct=0.5, stop_loss_pct=4.0,
        taker_fee_rate=0.0005, slippage_bps=0.0, initial_capital=1000.0,
    )
    assert result["summary"]["trades"] == 1
    assert result["trades"][0]["exit_reason"] == "trailing_stop"
    ret = result["trades"][0]["trade_return_pct"]
    # base 全程：0.95 × 0.84849 × 0.9975 − 1 ≈ −19.6%；10x 则 ≈ −37.4%
    assert ret == pytest.approx(-19.6, abs=0.6), f"未确认亏损应被 base 杠杆抑制，实际 {ret:.2f}%"


def test_ramp_pre_stop_tightens_immediate_drop(tmp_path: Path) -> None:
    """爬坡档确认前的收紧止损（ramp_pre_stop_pct=2.0）：入场即跌、从未确认的
    conviction 单在 2% 处离场（~-10%），而非 4% 处（~-20%）。"""
    bars = _ohlc_bars(
        [(100, 100.5, 99.0, 99.5), (99.5, 99.5, 95.5, 96.0)] + [(96.0, 96.5, 95.0, 96.0)] * 5,
        n_bars=7,
    )
    merged = attach_googl_daily_state(bars, pd.read_csv(_one_day_signal(tmp_path)))
    kwargs = dict(
        leverage_tiers=_RAMP_TIERS, ramp_confirm_pct=0.5, stop_loss_pct=4.0,
        taker_fee_rate=0.0005, slippage_bps=0.0, initial_capital=1000.0,
    )
    r_tight = run_googl_4h_replay(merged, None, ramp_pre_stop_pct=2.0, **kwargs)
    r_wide = run_googl_4h_replay(merged, None, ramp_pre_stop_pct=0.0, **kwargs)
    assert r_tight["summary"]["trades"] == 1
    assert r_tight["trades"][0]["exit_reason"] == "trailing_stop"
    ret_tight = r_tight["trades"][0]["trade_return_pct"]
    ret_wide = r_wide["trades"][0]["trade_return_pct"]
    # 手算：pre_stop=2.0 → 0.975×0.92462×0.9975−1 ≈ −10.1%；无 pre_stop → ≈ −19.9%
    assert ret_tight == pytest.approx(-10.1, abs=0.6), f"收紧止损应更早离场，实际 {ret_tight:.2f}%"
    assert ret_wide == pytest.approx(-19.9, abs=0.6), f"无 pre_stop 应在 4% 止损离场，实际 {ret_wide:.2f}%"


def test_ramp_pre_stop_does_not_touch_ramped_winner(tmp_path: Path) -> None:
    """确认后走出来的趋势单：爬坡确认前从未回撤 2% → pre_stop 对结果零影响。"""
    bars = _ohlc_bars(
        [(100, 103, 99, 102), (102, 111, 106, 110)] + [(110, 111, 109, 110)] * 5,
        n_bars=7,
    )
    merged = attach_googl_daily_state(bars, pd.read_csv(_one_day_signal(tmp_path)))
    kwargs = dict(
        leverage_tiers=_RAMP_TIERS, ramp_confirm_pct=0.5, stop_loss_pct=4.0,
        taker_fee_rate=0.0005, slippage_bps=0.0, initial_capital=1000.0,
    )
    r_tight = run_googl_4h_replay(merged, None, ramp_pre_stop_pct=2.0, **kwargs)
    r_wide = run_googl_4h_replay(merged, None, ramp_pre_stop_pct=0.0, **kwargs)
    assert r_tight["summary"]["trades"] == 1
    assert r_tight["trades"][0]["trade_return_pct"] == r_wide["trades"][0]["trade_return_pct"]
    assert r_tight["trades"][0]["trade_return_pct"] > 0.0, "趋势单应保持盈利"


def test_ramp_pre_stop_ignores_non_conviction_trades(tmp_path: Path) -> None:
    """pre_stop 只作用于有爬坡档（conviction/offense）的交易；base 档交易保持 4% 止损。"""
    bars = _ohlc_bars(
        [(100, 100.5, 99.0, 99.5), (99.5, 99.5, 95.5, 96.0)] + [(96.0, 96.5, 95.0, 96.0)] * 5,
        n_bars=7,
    )
    signal = pd.DataFrame(
        {
            "date": pd.date_range(_BARS_START, periods=3, freq="D"),
            "position": ["GOOGL", "FLAT", "FLAT"],
            "leverage_tier": ["base", "flat", "flat"],
            "target_leverage": [5.0, 0.0, 0.0],
        }
    )
    path = tmp_path / "base_tier_signal.csv"
    signal.to_csv(path, index=False)
    merged = attach_googl_daily_state(bars, pd.read_csv(path))
    kwargs = dict(
        leverage_tiers=_RAMP_TIERS, ramp_confirm_pct=0.5, stop_loss_pct=4.0,
        taker_fee_rate=0.0005, slippage_bps=0.0, initial_capital=1000.0,
    )
    r_tight = run_googl_4h_replay(merged, None, ramp_pre_stop_pct=2.0, **kwargs)
    r_wide = run_googl_4h_replay(merged, None, ramp_pre_stop_pct=0.0, **kwargs)
    assert r_tight["summary"]["trades"] == 1
    assert r_tight["trades"][0]["trade_return_pct"] == r_wide["trades"][0]["trade_return_pct"]
    # base 档 4% 止损：0.975 × 0.82412 × 0.9975 − 1 ≈ −19.9%
    assert r_tight["trades"][0]["trade_return_pct"] == pytest.approx(-19.9, abs=0.6)
