from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from bot.trend_basket_executor import (
    BasketLeg,
    TrendBasketConfig,
    TrendBasketExecutor,
    completed_daily_bars,
    plan_leg_orders,
)
from research.time_in_market.explore import signal_family
from strategy.trend_vote import VOTE_RULES, vote_target

LEGS = [
    BasketLeg("gold", "GLD", "XAU/USDT:USDT", 0.5),
    BasketLeg("qqq", "QQQ", "QQQ/USDT:USDT", 0.5),
]


def _daily(n: int = 300, drift: float = 0.001, seed: int = 0, end: str = "2026-10-09") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(drift + 0.01 * rng.standard_normal(n)))
    dates = pd.bdate_range(end=end, periods=n, tz="UTC") + pd.Timedelta(hours=13, minutes=30)
    return pd.DataFrame({"date": dates, "open": close, "high": close, "low": close, "close": close, "volume": 1.0})


def test_vote_target_matches_research_signal_family():
    d = _daily(400, drift=0.0).set_index("date")
    fam = signal_family(d)
    expected = sum(fam[r] for r in VOTE_RULES) / len(VOTE_RULES)
    pd.testing.assert_series_equal(vote_target(d["close"]), expected, check_names=False)


def test_completed_daily_bars_drops_live_session_bar():
    df = _daily(5, end="2026-10-13")
    # 2026-10-13 15:00 纽约时间（盘中）→ 当日 bar 未收盘
    intraday = datetime(2026, 10, 13, 19, 0, tzinfo=timezone.utc)
    assert completed_daily_bars(df, intraday)["date"].iloc[-1].date().isoformat() == "2026-10-12"
    # 16:20 纽约时间（收盘 + 15 分钟缓冲之后）→ 当日 bar 已收盘
    after_close = datetime(2026, 10, 13, 20, 20, tzinfo=timezone.utc)
    assert completed_daily_bars(df, after_close)["date"].iloc[-1].date().isoformat() == "2026-10-13"


def test_latest_expected_signal_day():
    from bot.trend_basket_executor import latest_expected_signal_day

    # 周一 08:00 纽约 → 上周五
    assert latest_expected_signal_day(datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)) == "2026-10-09"
    # 周二 16:20 纽约 → 当天
    assert latest_expected_signal_day(datetime(2026, 10, 13, 20, 20, tzinfo=timezone.utc)) == "2026-10-13"
    # 周日 → 周五
    assert latest_expected_signal_day(datetime(2026, 10, 11, 23, 0, tzinfo=timezone.utc)) == "2026-10-09"


def test_hold_skips_data_fetch_when_caught_up(tmp_path):
    now = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
    ex, _ = _executor(tmp_path, dry_run=True, now=now)
    ex.evaluate_once()
    calls = []
    ex._fetch_history = lambda leg, days: calls.append(leg) or (_ for _ in ()).throw(AssertionError("fetched"))
    assert ex.evaluate_once()["event"] == "hold"
    assert calls == []


def _plan(**overrides):
    kwargs = dict(
        equity=10_000.0,
        legs=LEGS,
        votes={"gold": 1.0, "qqq": 0.6},
        positions={},
        prices={"gold": 4000.0, "qqq": 600.0},
        contract_sizes={"gold": 0.001, "qqq": 1.0},
        min_contracts={"gold": 1.0, "qqq": 0.01},
        contract_steps={"gold": 1.0, "qqq": 0.01},
        multiplier=1.0,
        leverage=6,
        max_margin_frac=0.2,
        band_frac=0.1,
        min_order_notional=20.0,
    )
    kwargs.update(overrides)
    return plan_leg_orders(**kwargs)


def test_plan_targets_notional_from_votes():
    plan = _plan()
    assert plan["legs"]["gold"]["target_notional"] == pytest.approx(5000.0)
    assert plan["legs"]["gold"]["delta_contracts"] == 1250  # 5000 / (4000 × 0.001)
    assert plan["legs"]["qqq"]["target_notional"] == pytest.approx(3000.0)
    assert plan["legs"]["qqq"]["delta_contracts"] == pytest.approx(5.0)


def test_plan_respects_band_and_full_close():
    plan = _plan(positions={"gold": {"contracts": 1200.0}, "qqq": {"contracts": 5.0}}, votes={"gold": 1.0, "qqq": 0.0})
    assert plan["legs"]["gold"]["reason"] == "within_band"  # 4800 vs 5000，偏差 < 10% × 5000
    assert plan["legs"]["gold"]["delta_contracts"] == 0
    assert plan["legs"]["qqq"]["delta_contracts"] == -5.0  # 投票清零全平


def test_plan_caps_basket_margin():
    plan = _plan(multiplier=2.0, votes={"gold": 1.0, "qqq": 1.0}, max_margin_frac=0.2, leverage=6)
    total = sum(leg["target_notional"] for leg in plan["legs"].values())
    assert total == pytest.approx(0.2 * 10_000 * 6)  # 20000 → 被压到 12000
    assert plan["margin_scale"] == pytest.approx(0.6)


def test_config_rejects_btc_contracts():
    with pytest.raises(ValueError, match="BTC"):
        TrendBasketConfig.from_dict(
            {"legs": [{"name": "btc", "signal_symbol": "BTC-USD", "execution_symbol": "BTC/USDT:USDT", "weight": 1.0}]}
        )


class FakeExchange:
    def __init__(self):
        self.orders: list[tuple] = []

    def fetch_ticker(self, symbol):
        return {"last": {"XAU/USDT:USDT": 4000.0, "QQQ/USDT:USDT": 600.0}[symbol]}

    def amount_to_precision(self, symbol, amount):
        return str(amount)


class FakeClient:
    def __init__(self):
        self.exchange = FakeExchange()
        self.positions: list[dict] = []

    def load_markets(self):
        return {
            "XAU/USDT:USDT": {"contractSize": 0.001, "limits": {"amount": {"min": 1}}, "precision": {"amount": 1}},
            "QQQ/USDT:USDT": {"contractSize": 1.0, "limits": {"amount": {"min": 0.01}}, "precision": {"amount": 0.01}},
        }

    def fetch_balance(self):
        return {"info": {"data": [{"totalEq": "10000"}]}, "USDT": {"total": 10000.0, "free": 10000.0}}

    def fetch_positions(self, symbols):
        return self.positions

    def set_leverage(self, *args, **kwargs):
        return {}

    def create_order(self, symbol, order_type, side, amount, price=None, *, params=None):
        self.exchange.orders.append((symbol, side, amount, params))
        return {"id": str(len(self.exchange.orders))}


def _executor(tmp_path, dry_run: bool, now: datetime):
    cfg = TrendBasketConfig(
        dry_run=dry_run,
        state_db_path=str(tmp_path / "basket.db"),
        signal_data_dir=str(tmp_path / "etf"),
        legs=LEGS,
    )
    histories = {"GLD": _daily(300, drift=0.002, seed=1), "QQQ": _daily(300, drift=-0.002, seed=2)}
    client = FakeClient()
    ex = TrendBasketExecutor(cfg, client=client, fetch_history=lambda leg, days: histories[leg.signal_symbol], now_fn=lambda: now)
    return ex, client


def test_executor_dry_run_places_no_orders_then_live_rebalances_once(tmp_path):
    now = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
    ex, client = _executor(tmp_path, dry_run=True, now=now)
    ex.bootstrap()
    out = ex.evaluate_once()
    assert out["status"] == "ok" and client.exchange.orders == []

    # 切到实盘：同一信号日也要重新执行（dry_run 的去重键不能挡住实盘首单）
    ex.config.dry_run = False
    out = ex.evaluate_once()
    assert out["status"] == "ok"
    sides = {sym: (side, params) for sym, side, _, params in client.exchange.orders}
    assert all(params["tdMode"] == "isolated" and params["posSide"] == "long" for _, params in sides.values())
    assert all(side == "buy" for side, _ in sides.values())
    n_orders = len(client.exchange.orders)
    assert n_orders >= 1
    # 同一信号日再次评估：不重复下单
    assert ex.evaluate_once()["event"] == "hold"
    assert len(client.exchange.orders) == n_orders


def test_executor_reduces_with_reduce_only(tmp_path):
    now = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
    ex, client = _executor(tmp_path, dry_run=False, now=now)
    # 下跌趋势的 QQQ 投票低；持有过多 QQQ 应减仓
    client.positions = [{"symbol": "QQQ/USDT:USDT", "contracts": 20.0, "side": "long", "initialMargin": 2000.0}]
    out = ex.evaluate_once()
    qqq_orders = [o for o in client.exchange.orders if o[0] == "QQQ/USDT:USDT"]
    assert qqq_orders and qqq_orders[0][1] == "sell" and qqq_orders[0][3]["reduceOnly"] is True
    assert out["plan"]["legs"]["qqq"]["target_notional"] < 12_000


def test_sma200_rule_matches_research_and_drives_leg_signal():
    from bot.trend_basket_executor import leg_signal
    from strategy.trend_vote import signal_target

    d = _daily(400, drift=0.0, seed=3)
    close = d.set_index("date")["close"]
    sma = close.rolling(200).mean()
    expected = (close > sma).astype(float).where(sma.notna(), 0.0)
    pd.testing.assert_series_equal(signal_target(close, "sma200"), expected, check_names=False)
    sig = leg_signal(d, "sma200")
    assert sig["rule"] == "sma200" and set(sig["rules"]) == {"sma200_L"}
    assert sig["vote"] == float(expected.iloc[-1])


def test_config_rejects_unknown_signal_rule():
    with pytest.raises(ValueError, match="signal_rule"):
        TrendBasketConfig.from_dict(
            {"legs": [{"name": "qqq", "signal_symbol": "QQQ", "execution_symbol": "QQQ/USDT:USDT",
                       "weight": 1.0, "signal_rule": "magic"}]}
        )
