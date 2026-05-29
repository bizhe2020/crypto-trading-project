from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.btc_route_scoring import btc_effective_leverage, btc_route_score
from bot.qqq_usdt_executor import QqqOrderContext, QqqUsdtExecutionEngine
from bot.qqq_usdt_signal_adapter import QqqUsdtSignalAdapter
from bot.router_executor import StrategyRouterExecutionEngine
from bot.strategy_router import RoutedSignalCandidate, StrategyRouter, StrategyRouterConfig


def build_router() -> StrategyRouter:
    return StrategyRouter(
        StrategyRouterConfig(
            mode="paper",
            state_path="state/test_strategy_router.json",
            btc_strategy_config="config/config.paper.high-leverage-structure.json",
            qqq_strategy_config="config/config.paper.qqq-usdt-aggressive-frozen.json",
            btc_min_route_score=35.0,
            qqq_min_route_score=60.0,
            switch_advantage=8.0,
            persist_state=False,
        )
    )


def test_router_prefers_higher_score_when_margin_large() -> None:
    router = build_router()
    btc = RoutedSignalCandidate("btc_sota", "BTC/USDT:USDT", True, 42.0)
    qqq = RoutedSignalCandidate("qqq_usdt_aggressive", "QQQ/USDT:USDT", True, 72.0)
    selected, reason = router._choose_candidate([btc, qqq], current_strategy="btc_sota")
    assert selected is not None
    assert selected.strategy_id == "qqq_usdt_aggressive"
    assert reason == "best_route_score"


def test_router_holds_current_when_advantage_small() -> None:
    router = build_router()
    btc = RoutedSignalCandidate("btc_sota", "BTC/USDT:USDT", True, 58.0)
    qqq = RoutedSignalCandidate("qqq_usdt_aggressive", "QQQ/USDT:USDT", True, 63.0)
    selected, reason = router._choose_candidate([btc, qqq], current_strategy="btc_sota")
    assert selected is not None
    assert selected.strategy_id == "btc_sota"
    assert reason == "hold_current_hysteresis"


def test_router_returns_none_when_no_candidate_clears_threshold() -> None:
    router = build_router()
    btc = RoutedSignalCandidate("btc_sota", "BTC/USDT:USDT", True, 20.0)
    qqq = RoutedSignalCandidate("qqq_usdt_aggressive", "QQQ/USDT:USDT", False, 0.0)
    selected, reason = router._choose_candidate([btc, qqq], current_strategy=None)
    assert selected is None
    assert reason == "no_eligible_candidates"


def test_router_keeps_current_active_candidate_below_new_entry_threshold() -> None:
    router = build_router()
    btc = RoutedSignalCandidate("btc_sota", "BTC/USDT:USDT", False, 0.0)
    qqq = RoutedSignalCandidate("qqq_usdt_aggressive", "QQQ/USDT:USDT", True, 52.0, leverage=1.0)
    selected, reason = router._choose_candidate([btc, qqq], current_strategy="qqq_usdt_aggressive")
    assert selected is qqq
    assert reason == "hold_current_active_below_threshold"


def test_router_evaluate_uses_execution_strategy_override(tmp_path: Path) -> None:
    router = StrategyRouter(
        StrategyRouterConfig(
            mode="paper",
            state_path=str(tmp_path / "router.json"),
            btc_strategy_config="config/config.paper.high-leverage-structure.json",
            qqq_strategy_config="config/config.paper.qqq-usdt-aggressive-frozen.json",
            btc_min_route_score=35.0,
            qqq_min_route_score=60.0,
            persist_state=True,
        )
    )
    qqq = RoutedSignalCandidate("qqq_usdt_aggressive", "QQQ/USDT:USDT", True, 52.0, leverage=1.0)
    router._collect_candidates = lambda: [RoutedSignalCandidate("btc_sota", "BTC/USDT:USDT", False, 0.0), qqq]  # type: ignore[method-assign]
    payload = router.evaluate_latest(current_strategy_override="qqq_usdt_aggressive")
    assert payload["selected_strategy"] == "qqq_usdt_aggressive"
    assert payload["decision_reason"] == "hold_current_active_below_threshold"
    assert payload["previous_selected_strategy"] == "qqq_usdt_aggressive"


def test_btc_route_score_uses_quality_layers() -> None:
    weak = {
        "event_type": "sota_long",
        "direction": "BULL",
        "source_effective_leverage": 1.5,
        "net_score": 2,
        "bull_total": 4,
        "bear_total": 7,
        "conflict": True,
    }
    strong = {
        "event_type": "sota_long",
        "direction": "BULL",
        "source_effective_leverage": 10,
        "net_score": 12,
        "bull_total": 14,
        "bear_total": 2,
        "regime_label": "high_growth",
        "feature_recent_fvg_near_entry": True,
        "feature_recent_sweep_status": "mss_with_fvg",
    }
    assert btc_route_score(strong) > btc_route_score(weak) + 55.0


def test_btc_route_score_reads_nested_sota_score_gate() -> None:
    nested = {
        "event_type": "sota_long",
        "direction": "BULL",
        "leverage": 5,
        "net_score": None,
        "sota_score_gate": {
            "score": {
                "net_score": 10,
                "bull_total": 12,
                "bear_total": 2,
            }
        },
    }
    flat = {
        "event_type": "sota_long",
        "direction": "BULL",
        "leverage": 5,
        "net_score": 0,
        "bull_total": 0,
        "bear_total": 2,
    }
    assert btc_effective_leverage(nested) == 5.0
    assert btc_route_score(nested) > btc_route_score(flat) + 20.0


def test_qqq_same_signal_stop_lock_blocks_reopen(tmp_path: Path) -> None:
    config_path = tmp_path / "qqq.json"
    config_path.write_text(
        """
{
  "execution_symbol": "QQQ/USDT:USDT",
  "base_leverage": 10.0,
  "offense_leverage": 10.0,
  "stop_loss_pct": 3.5
}
""".strip()
    )
    engine = QqqUsdtExecutionEngine(
        StrategyRouterConfig(
            mode="paper",
            state_path=str(tmp_path / "router.json"),
            btc_strategy_config="config/config.paper.high-leverage-structure.json",
            qqq_strategy_config=str(config_path),
            qqq_state_db_path=str(tmp_path / "qqq_state.db"),
        ),
        config_path,
    )
    engine.save_state({"last_stop_hit": {"candidate_timestamp": "2026-05-22 12:00:00+00:00"}})

    context = QqqOrderContext(
        symbol="QQQ/USDT:USDT",
        margin_mode="isolated",
        leverage=10.0,
        stop_loss_pct=3.5,
        reference_price=720.0,
        latest_low=710.0,
        stop_price=695.0,
        stop_hit=False,
        route_score=98.0,
        candidate={"timestamp": "2026-05-22 12:00:00+00:00"},
    )
    assert engine._same_signal_stop_locked(context) is True


def test_qqq_order_precision_uses_cached_okx_step(tmp_path: Path) -> None:
    config_path = tmp_path / "qqq.json"
    config_path.write_text(
        """
{
  "execution_symbol": "QQQ/USDT:USDT",
  "base_leverage": 10.0,
  "offense_leverage": 10.0,
  "stop_loss_pct": 3.5
}
""".strip()
    )
    engine = QqqUsdtExecutionEngine(
        StrategyRouterConfig(
            mode="paper",
            state_path=str(tmp_path / "router.json"),
            btc_strategy_config="config/config.paper.high-leverage-structure.json",
            qqq_strategy_config=str(config_path),
            qqq_state_db_path=str(tmp_path / "qqq_state.db"),
        ),
        config_path,
    )
    market = {
        "contract": True,
        "contractSize": 1.0,
        "precision": {"amount": 0.01, "price": 0.01},
        "limits": {"amount": {"min": 0.01}},
    }
    assert engine._amount_to_precision(market, 13.899) == "13.89"
    assert engine._price_to_precision(market, 694.6263) == "694.62"


def test_qqq_live_trailing_stop_amends_exchange_stop(tmp_path: Path) -> None:
    config_path = tmp_path / "qqq.json"
    config_path.write_text(
        """
{
  "execution_symbol": "QQQ/USDT:USDT",
  "base_leverage": 10.0,
  "offense_leverage": 10.0,
  "stop_loss_pct": 3.5
}
""".strip()
    )
    engine = QqqUsdtExecutionEngine(
        StrategyRouterConfig(
            mode="live",
            state_path=str(tmp_path / "router.json"),
            btc_strategy_config="config/config.paper.high-leverage-structure.json",
            qqq_strategy_config=str(config_path),
            qqq_state_db_path=str(tmp_path / "qqq_state.db"),
            qqq_enable_exchange_stop=True,
        ),
        config_path,
    )
    engine._markets_cache = {"QQQ/USDT:USDT": {"id": "QQQ-USDT-SWAP", "precision": {"price": 0.01}}}
    engine.save_state(
        {
            "position": {
                "symbol": "QQQ/USDT:USDT",
                "stop_price": 690.0,
                "exchange_attach_algo_id": "algo-1",
                "exchange_attach_algo_client_id": "client-1",
            }
        }
    )

    calls = []

    class FakeClient:
        def amend_algo_order(self, request):
            calls.append(request)
            return {"code": "0", "data": [{"algoId": "algo-1"}]}

    engine.client = FakeClient()
    context = QqqOrderContext(
        symbol="QQQ/USDT:USDT",
        margin_mode="isolated",
        leverage=10.0,
        stop_loss_pct=3.5,
        reference_price=730.0,
        latest_low=720.0,
        stop_price=704.45,
        stop_hit=False,
        route_score=98.0,
        candidate={"timestamp": "2026-05-29 12:00:00+00:00"},
    )

    result = engine.update_trailing_stop(
        context,
        {"contracts": 1.0, "notional_usdt": 1000.0, "close_order_algos": [], "raw": None},
    )

    assert result is not None
    assert result["status"] == "tracked"
    assert result["exchange_stop"]["status"] == "amended"
    assert calls[0]["algoId"] == "algo-1"
    assert calls[0]["newSlTriggerPx"] == "704.45"


def test_qqq_live_trailing_stop_amend_failure_does_not_advance_local_stop(tmp_path: Path) -> None:
    config_path = tmp_path / "qqq.json"
    config_path.write_text(
        """
{
  "execution_symbol": "QQQ/USDT:USDT",
  "base_leverage": 10.0,
  "offense_leverage": 10.0,
  "stop_loss_pct": 3.5
}
""".strip()
    )
    engine = QqqUsdtExecutionEngine(
        StrategyRouterConfig(
            mode="live",
            state_path=str(tmp_path / "router.json"),
            btc_strategy_config="config/config.paper.high-leverage-structure.json",
            qqq_strategy_config=str(config_path),
            qqq_state_db_path=str(tmp_path / "qqq_state.db"),
            qqq_enable_exchange_stop=True,
        ),
        config_path,
    )
    engine._markets_cache = {"QQQ/USDT:USDT": {"id": "QQQ-USDT-SWAP", "precision": {"price": 0.01}}}
    engine.save_state(
        {
            "position": {
                "symbol": "QQQ/USDT:USDT",
                "stop_price": 690.0,
                "peak_price": 715.0,
                "exchange_attach_algo_id": "algo-1",
            }
        }
    )

    class FakeClient:
        def amend_algo_order(self, request):
            raise RuntimeError("boom")

    engine.client = FakeClient()
    context = QqqOrderContext(
        symbol="QQQ/USDT:USDT",
        margin_mode="isolated",
        leverage=10.0,
        stop_loss_pct=3.5,
        reference_price=730.0,
        latest_low=720.0,
        stop_price=704.45,
        stop_hit=False,
        route_score=98.0,
        candidate={"timestamp": "2026-05-29 12:00:00+00:00"},
    )

    result = engine.update_trailing_stop(
        context,
        {"contracts": 1.0, "notional_usdt": 1000.0, "close_order_algos": [], "raw": None},
    )

    assert result is not None
    assert result["status"] == "error"
    assert result["reason"] == "exchange_stop_amend_failed"
    assert engine.load_state()["position"]["stop_price"] == 690.0


def test_qqq_live_rebalance_reduces_position_instead_of_reopen(tmp_path: Path) -> None:
    config_path = tmp_path / "qqq.json"
    config_path.write_text(
        """
{
  "execution_symbol": "QQQ/USDT:USDT",
  "base_leverage": 10.0,
  "offense_leverage": 10.0,
  "stop_loss_pct": 3.5
}
""".strip()
    )
    engine = QqqUsdtExecutionEngine(
        StrategyRouterConfig(
            mode="live",
            state_path=str(tmp_path / "router.json"),
            btc_strategy_config="config/config.paper.high-leverage-structure.json",
            qqq_strategy_config=str(config_path),
            qqq_state_db_path=str(tmp_path / "qqq_state.db"),
            qqq_enable_exchange_stop=False,
        ),
        config_path,
    )
    engine._markets_cache = {
        "QQQ/USDT:USDT": {
            "id": "QQQ-USDT-SWAP",
            "contract": True,
            "contractSize": 1.0,
            "precision": {"amount": 0.01, "price": 0.01},
            "limits": {"amount": {"min": 0.01}},
        }
    }
    calls = []

    class FakeClient:
        def set_leverage(self, leverage, symbol, margin_mode="isolated", pos_side=None):
            calls.append(("set_leverage", leverage, symbol, margin_mode, pos_side))
            return {"code": "0"}

        def create_order(self, symbol, order_type, side, amount, price=None, *, params=None):
            calls.append(("create_order", symbol, order_type, side, amount, params))
            return {"id": "reduce-1"}

    engine.client = FakeClient()
    context = QqqOrderContext(
        symbol="QQQ/USDT:USDT",
        margin_mode="isolated",
        leverage=2.0,
        stop_loss_pct=3.5,
        reference_price=700.0,
        latest_low=690.0,
        stop_price=675.5,
        stop_hit=False,
        route_score=70.0,
        candidate={"timestamp": "2026-05-29 12:00:00+00:00"},
    )

    result = engine.rebalance_position(
        context,
        {"contracts": 10.0, "notional_usdt": 7000.0, "raw": None},
        {"leverage": 10.0},
        target_notional=1400.0,
    )

    assert result["status"] == "submitted"
    assert result["side"] == "sell"
    assert result["amount"] == 8.0
    assert calls[0][0] == "create_order"
    assert calls[0][3] == "sell"
    assert calls[0][5]["reduceOnly"] is True
    assert calls[1] == ("set_leverage", 2, "QQQ/USDT:USDT", "isolated", "long")


def test_qqq_market_orders_split_by_okx_max_market_size(tmp_path: Path) -> None:
    config_path = tmp_path / "qqq.json"
    config_path.write_text(
        """
{
  "execution_symbol": "QQQ/USDT:USDT",
  "base_leverage": 10.0,
  "offense_leverage": 10.0,
  "stop_loss_pct": 3.5
}
""".strip()
    )
    engine = QqqUsdtExecutionEngine(
        StrategyRouterConfig(
            mode="live",
            state_path=str(tmp_path / "router.json"),
            btc_strategy_config="config/config.paper.high-leverage-structure.json",
            qqq_strategy_config=str(config_path),
            qqq_state_db_path=str(tmp_path / "qqq_state.db"),
        ),
        config_path,
    )
    engine._markets_cache = {
        "QQQ/USDT:USDT": {
            "id": "QQQ-USDT-SWAP",
            "contract": True,
            "contractSize": 1.0,
            "precision": {"amount": 0.01, "price": 0.01},
            "limits": {"amount": {"min": 0.01}},
            "info": {"maxMktSz": "210"},
        }
    }
    calls = []

    class FakeClient:
        def create_order(self, symbol, order_type, side, amount, price=None, *, params=None):
            calls.append((symbol, order_type, side, amount, params))
            return {"id": f"order-{len(calls)}", "amount": amount}

    engine.client = FakeClient()
    orders = engine._submit_market_orders(
        "QQQ/USDT:USDT",
        "sell",
        302.14,
        params={"reduceOnly": True, "tdMode": "isolated", "posSide": "long"},
    )

    assert [item["amount"] for item in orders] == [210.0, 92.14]
    assert [call[3] for call in calls] == [210.0, 92.14]


def test_qqq_live_rebalance_adds_only_delta_position(tmp_path: Path) -> None:
    config_path = tmp_path / "qqq.json"
    config_path.write_text(
        """
{
  "execution_symbol": "QQQ/USDT:USDT",
  "base_leverage": 10.0,
  "offense_leverage": 10.0,
  "stop_loss_pct": 3.5
}
""".strip()
    )
    engine = QqqUsdtExecutionEngine(
        StrategyRouterConfig(
            mode="live",
            state_path=str(tmp_path / "router.json"),
            btc_strategy_config="config/config.paper.high-leverage-structure.json",
            qqq_strategy_config=str(config_path),
            qqq_state_db_path=str(tmp_path / "qqq_state.db"),
            qqq_enable_exchange_stop=False,
        ),
        config_path,
    )
    engine._markets_cache = {
        "QQQ/USDT:USDT": {
            "id": "QQQ-USDT-SWAP",
            "contract": True,
            "contractSize": 1.0,
            "precision": {"amount": 0.01, "price": 0.01},
            "limits": {"amount": {"min": 0.01}},
        }
    }
    calls = []

    class FakeClient:
        def set_leverage(self, leverage, symbol, margin_mode="isolated", pos_side=None):
            calls.append(("set_leverage", leverage, symbol, margin_mode, pos_side))
            return {"code": "0"}

        def create_order(self, symbol, order_type, side, amount, price=None, *, params=None):
            calls.append(("create_order", symbol, order_type, side, amount, params))
            return {"id": "add-1"}

    engine.client = FakeClient()
    context = QqqOrderContext(
        symbol="QQQ/USDT:USDT",
        margin_mode="isolated",
        leverage=10.0,
        stop_loss_pct=3.5,
        reference_price=700.0,
        latest_low=690.0,
        stop_price=675.5,
        stop_hit=False,
        route_score=100.0,
        candidate={"timestamp": "2026-05-29 12:00:00+00:00"},
    )

    result = engine.rebalance_position(
        context,
        {"contracts": 2.0, "notional_usdt": 1400.0, "raw": None},
        {"leverage": 2.0},
        target_notional=7000.0,
    )

    assert result["status"] == "submitted"
    assert result["side"] == "buy"
    assert result["amount"] == 8.0
    assert calls[0] == ("set_leverage", 10, "QQQ/USDT:USDT", "isolated", "long")
    assert calls[1][0] == "create_order"
    assert calls[1][3] == "buy"
    assert calls[1][4] == 8.0


def test_qqq_live_evaluate_syncs_existing_exchange_position_without_rebalance(tmp_path: Path) -> None:
    config_path = tmp_path / "qqq.json"
    config_path.write_text(
        """
{
  "execution_symbol": "QQQ/USDT:USDT",
  "base_leverage": 10.0,
  "offense_leverage": 10.0,
  "stop_loss_pct": 3.5
}
""".strip()
    )
    engine = QqqUsdtExecutionEngine(
        StrategyRouterConfig(
            mode="live",
            state_path=str(tmp_path / "router.json"),
            btc_strategy_config="config/config.paper.high-leverage-structure.json",
            qqq_strategy_config=str(config_path),
            qqq_state_db_path=str(tmp_path / "qqq_state.db"),
            qqq_enable_exchange_stop=True,
        ),
        config_path,
    )
    context = QqqOrderContext(
        symbol="QQQ/USDT:USDT",
        margin_mode="isolated",
        leverage=10.0,
        stop_loss_pct=3.5,
        reference_price=730.0,
        latest_low=725.0,
        stop_price=704.45,
        stop_hit=False,
        route_score=100.0,
        candidate={"timestamp": "2026-05-29 12:00:00+00:00"},
    )
    engine._build_context = lambda candidate: context  # type: ignore[method-assign]
    engine.fetch_position_state = lambda: {  # type: ignore[method-assign]
        "contracts": 302.14,
        "notional_usdt": 222907.19,
        "close_order_algos": [
            {
                "attachAlgoId": "algo-1",
                "attachAlgoClOrdId": "client-1",
                "slTriggerPx": "710.00",
            }
        ],
        "raw": {"entryPrice": "734.6"},
    }

    def fail_rebalance(*args, **kwargs):
        raise AssertionError("existing exchange position must be synced before any rebalance")

    engine.rebalance_position = fail_rebalance  # type: ignore[method-assign]
    engine.open_position = fail_rebalance  # type: ignore[method-assign]

    result = engine.evaluate_latest(
        RoutedSignalCandidate(
            strategy_id="qqq_usdt_aggressive",
            symbol="QQQ/USDT:USDT",
            active=True,
            route_score=100.0,
            timestamp="2026-05-29 12:00:00+00:00",
            direction="BULL",
            event_type="qqq_usdt_long",
            leverage=10.0,
        )
    )

    assert result["status"] == "ok"
    assert result["position_open"] is True
    assert result["actions"][0]["status"] == "synced"
    state = engine.load_state()["position"]
    assert state["exchange_contracts"] == 302.14
    assert state["exchange_notional_usdt"] == 222907.19
    assert state["exchange_entry_price"] == 734.6
    assert state["exchange_attach_algo_id"] == "algo-1"
    assert state["exchange_attach_algo_client_id"] == "client-1"
    assert state["stop_price"] == 710.0


def test_qqq_daily_signal_stale_guard_blocks_old_signal() -> None:
    frame = pd.DataFrame({"date": [pd.Timestamp("2026-05-22T13:30:00Z")]})
    status = QqqUsdtSignalAdapter._daily_signal_stale_status(
        {"daily_signal_stale_guard_enabled": True, "daily_signal_max_stale_calendar_days": 5},
        frame,
        now=pd.Timestamp("2026-05-29T08:00:00Z"),
    )
    assert status["stale"] is True


def test_qqq_daily_signal_stale_guard_allows_recent_signal() -> None:
    frame = pd.DataFrame({"date": [pd.Timestamp("2026-05-27T13:30:00Z")]})
    status = QqqUsdtSignalAdapter._daily_signal_stale_status(
        {"daily_signal_stale_guard_enabled": True, "daily_signal_max_stale_calendar_days": 5},
        frame,
        now=pd.Timestamp("2026-05-29T08:00:00Z"),
    )
    assert status["stale"] is False


def test_router_telegram_route_message_uses_router_context() -> None:
    payload = {
        "previous_executed_strategy": "btc_sota",
        "current_executed_strategy": "qqq_usdt_aggressive",
    }
    route = {
        "decision_reason": "best_route_score",
        "selected_route_score": 94.0,
        "selected_candidate": {
            "timestamp": "2026-05-28 12:00:00+00:00",
            "leverage": 10.0,
            "metadata": {
                "daily_signal_stale": {
                    "latest": "2026-05-28 13:30:00+00:00",
                    "lag_days": 1,
                }
            },
        },
    }
    message = StrategyRouterExecutionEngine._format_route_message(payload, route)
    assert "Router 选路更新" in message
    assert "QQQ/USDT" in message
    assert "BTC SOTA" in message
    assert "10.0x" in message
    assert "lag 1" in message


def test_router_data_warning_detects_stale_qqq_signal() -> None:
    route = {
        "candidates": [
            {
                "strategy_id": "qqq_usdt_aggressive",
                "metadata": {
                    "daily_signal_stale": {
                        "stale": True,
                        "latest": "2026-05-22 13:30:00+00:00",
                        "lag_days": 7,
                    }
                },
            }
        ]
    }
    warning = StrategyRouterExecutionEngine._data_warning(route)
    assert warning is not None
    assert "QQQ 日线信号过期" in warning["message"]


def test_router_audit_log_appends_jsonl(tmp_path: Path) -> None:
    config = StrategyRouterConfig(
        mode="paper",
        state_path=str(tmp_path / "router.json"),
        btc_strategy_config="config/config.paper.high-leverage-structure.json",
        qqq_strategy_config="config/config.paper.qqq-usdt-aggressive-frozen.json",
        qqq_state_db_path=str(tmp_path / "qqq_state.db"),
        router_audit_log_path=str(tmp_path / "router_audit.jsonl"),
        router_audit_log_enabled=True,
    )
    engine = StrategyRouterExecutionEngine.__new__(StrategyRouterExecutionEngine)
    engine.config = config
    engine.audit_log_path = Path(config.router_audit_log_path)
    engine.execution_state_path = tmp_path / "router.json.execution"
    engine.router = type("RouterStub", (), {"config_path": tmp_path / "router_config.json", "state_path": tmp_path / "router.json"})()

    class FakeBtc:
        def _load_snapshot_payload(self):
            return {"position": {"direction": "BULL"}, "trade_count": 1}

    class FakeQqq:
        symbol = "QQQ/USDT:USDT"

        def load_state(self):
            return {"position": {"leverage": 10.0, "stop_price": 700.0}}

        def fetch_position_state(self):
            return {"contracts": 1.0, "notional_usdt": 1000.0, "raw": {"id": "pos"}}

    engine.btc_executor = FakeBtc()
    engine.qqq_executor = FakeQqq()

    payload = {
        "route": {"selected_strategy": "qqq_usdt_aggressive", "selected_route_score": 96.0},
        "previous_executed_strategy": "btc_sota",
        "current_executed_strategy": "qqq_usdt_aggressive",
        "execution_results": [{"strategy": "qqq_usdt_aggressive", "result": {"status": "paper_opened"}}],
        "updated_at": 1780065600,
    }
    engine._append_audit_log(payload)

    lines = engine.audit_log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = __import__("json").loads(lines[0])
    assert record["event"] == "strategy_router_evaluate"
    assert record["route"]["selected_strategy"] == "qqq_usdt_aggressive"
    assert record["local_state"]["btc_snapshot"]["trade_count"] == 1
    assert record["local_state"]["qqq_state"]["position"]["leverage"] == 10.0
    assert record["exchange_state"]["qqq"]["position"]["notional_usdt"] == 1000.0
