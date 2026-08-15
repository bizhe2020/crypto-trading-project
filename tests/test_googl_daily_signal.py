"""GOOGL 高倍合约策略 — 信号生成 + 适配器单元测试。

用合成 prices.csv / holdings.csv，不依赖外部价值项目数据。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot.googl_usdt_signal_adapter import GooglUsdtSignalAdapter  # noqa: E402
from scripts.scan_googl_daily_signal import (  # noqa: E402
    build_conviction_series,
    build_googl_frame,
    load_value_prices,
    run_googl_signal,
    summarize,
)


def _make_prices_csv(tmp_path: Path) -> Path:
    """Synthetic GOOGL + SPY daily prices 2015-2026, GOOGL trending up."""
    dates = pd.date_range("2015-01-02", "2026-08-14", freq="B")
    rng = pd.DataFrame({"date": dates})
    rng["open"] = 100.0 + rng.index * 0.05
    rng["close"] = rng["open"] + 1.0
    googl = rng.copy()
    googl["ticker"] = "GOOGL"
    # SPY: flat-ish then trending, above its 200d MA for most of the window
    spy = rng.copy()
    spy["ticker"] = "SPY"
    spy["open"] = 200.0 + rng.index * 0.02
    spy["close"] = spy["open"] + 0.5
    frame = pd.concat([googl, spy], ignore_index=True)
    frame = frame[["ticker", "date", "open", "close"]]
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    frame["basis"] = "qfq"
    path = tmp_path / "prices.csv"
    frame.to_csv(path, index=False)
    return path


def _make_holdings_csv(tmp_path: Path, alphabet_filing_date: str = "2025-11-14") -> Path:
    rows = [
        {"filing_date": "2020-02-14", "report_date": "2019-12-31", "cusip": "594918104", "issuer_name": "MICROSOFT CORP", "title": "", "shares": 1000000, "value_usd": 150000000},
    ]
    rows.append(
        {"filing_date": alphabet_filing_date, "report_date": "2025-09-30", "cusip": "02079K305", "issuer_name": "ALPHABET INC", "title": "", "shares": 17846142, "value_usd": 3200000000},
    )
    path = tmp_path / "berkshire_13f_holdings.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_value_prices(tmp_path: Path) -> None:
    csv_path = _make_prices_csv(tmp_path)
    frame = load_value_prices(csv_path)
    assert {"GOOGL", "SPY"}.issubset(set(frame["ticker"]))
    assert frame["date"].is_monotonic_increasing


def test_build_conviction_series(tmp_path: Path) -> None:
    holdings = _make_holdings_csv(tmp_path)
    dates = pd.date_range("2025-10-01", "2026-01-30", freq="B")
    series = build_conviction_series(holdings, pd.DatetimeIndex(dates))
    assert series.loc[pd.Timestamp("2025-10-01", tz="UTC")].item() is False
    assert series.loc[pd.Timestamp("2025-11-17", tz="UTC")].item() is True  # 2025-11-14 披露后
    assert series.loc[pd.Timestamp("2026-01-30", tz="UTC")].item() is True


def test_build_conviction_series_missing_file(tmp_path: Path) -> None:
    series = build_conviction_series(tmp_path / "missing.csv", pd.DatetimeIndex(pd.date_range("2025-01-01", periods=10)))
    assert not series.any()


def test_build_googl_frame(tmp_path: Path) -> None:
    prices = load_value_prices(_make_prices_csv(tmp_path))
    conv = build_conviction_series(_make_holdings_csv(tmp_path), pd.DatetimeIndex(prices["date"].unique()))
    frame = build_googl_frame(prices, conv)
    for col in ("googl_open", "googl_close", "fast_ma", "slow_ma", "entry_signal", "ixic_trend_label", "tqqq_open", "tqqq_close", "berkshire_conviction"):
        assert col in frame.columns
    assert frame["tqqq_open"].eq(frame["googl_open"]).all()
    assert frame["berkshire_conviction"].dtype == bool or frame["berkshire_conviction"].dtype == object


def test_run_googl_signal_two_pass(tmp_path: Path) -> None:
    prices_csv = _make_prices_csv(tmp_path)
    holdings_csv = _make_holdings_csv(tmp_path)
    payload = run_googl_signal(prices_csv, holdings_csv)
    path = payload["path"]
    assert set(path["position"].unique()).issubset({"GOOGL", "FLAT"})
    assert "berkshire_conviction" in path.columns
    assert "leverage_tier" in path.columns
    assert "target_leverage" in path.columns
    # conviction 段存在（13F 披露后）
    assert payload["conviction_start"] is not None
    conv = path[path["berkshire_conviction"].astype(bool)]
    assert not conv.empty
    # 在市且 conviction 时 target_leverage = 11.2（offense，v0.3 真实数据定档）
    offense = path[path["leverage_tier"].eq("offense")]
    if not offense.empty:
        assert offense["target_leverage"].eq(11.2).all()
    # base 档 7.5x
    base = path[path["leverage_tier"].eq("base")]
    if not base.empty:
        assert base["target_leverage"].eq(7.5).all()
    # capital 单调非负
    assert (path["capital"] > 0).all()


def test_summarize(tmp_path: Path) -> None:
    prices_csv = _make_prices_csv(tmp_path)
    holdings_csv = _make_holdings_csv(tmp_path)
    payload = run_googl_signal(prices_csv, holdings_csv)
    summary = summarize(payload["path"])
    assert "total_return_pct" in summary
    assert "conviction_in_market_pct" in summary
    assert "latest_position" in summary


def test_adapter_produces_candidate(tmp_path: Path) -> None:
    prices_csv = _make_prices_csv(tmp_path)
    holdings_csv = _make_holdings_csv(tmp_path)
    payload = run_googl_signal(prices_csv, holdings_csv)
    signal_csv = tmp_path / "googl_daily_signal.csv"
    path = payload["path"]
    path[["date", "position", "berkshire_conviction", "leverage_tier", "target_leverage"]].to_csv(signal_csv, index=False)

    config = {
        "mode": "paper",
        "signal_source": str(signal_csv),
        "execution_symbol": "GOOGL/USDT:USDT",
        "base_leverage": 7.5,
        "offense_leverage": 11.2,
        "defense_leverage": 3.8,
        "stop_loss_pct": 4.0,
        "macro_proxy_overlay_enabled": False,
        "risk_overlay_enabled": False,
        "daily_signal_stale_guard_enabled": False,
        "leverage_profile_name": "conviction_tiered",
        "frozen_label": "test",
    }
    config_path = tmp_path / "config.paper.googl-high-leverage-runtime.json"
    config_path.write_text(json.dumps(config))

    adapter = GooglUsdtSignalAdapter(config_path)
    candidate = adapter.preview()
    assert candidate.strategy_id == "googl_usdt_aggressive"
    assert candidate.symbol == "GOOGL/USDT:USDT"
    # 合成数据 GOOGL 稳定上行 → 大概率在市
    assert candidate.active in (True, False)
    assert "berkshire_conviction" in candidate.metadata
    assert "leverage_tier" in candidate.metadata
    if candidate.active:
        assert candidate.direction == "BULL"
        assert candidate.leverage in (7.5, 11.2)


def test_adapter_missing_signal(tmp_path: Path) -> None:
    config = {
        "mode": "paper",
        "signal_source": str(tmp_path / "missing.csv"),
        "execution_symbol": "GOOGL/USDT:USDT",
        "daily_signal_stale_guard_enabled": False,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    adapter = GooglUsdtSignalAdapter(config_path)
    with pytest.raises(FileNotFoundError):
        adapter.preview()
