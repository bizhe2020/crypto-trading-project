from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.btc_route_scoring import btc_effective_leverage, btc_route_score
from bot.qqq_macro_proxy_overlay import apply_macro_proxy_overlay
from bot.qqq_usdt_executor import QqqOrderContext, QqqUsdtExecutionEngine
from bot.qqq_runtime_policy import filter_closed_bars, market_time_window_status, trading_calendar_status
from bot.qqq_usdt_signal_adapter import QqqUsdtSignalAdapter
from bot.router_executor import StrategyRouterExecutionEngine
from bot.strategy_router import RoutedSignalCandidate, StrategyRouter, StrategyRouterConfig
import scripts.scan_qqq_usdt_4h_triggers as qqq_trigger_scan
from scripts.replay_qqq_usdt_10x import run_10x_replay


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


def test_strategy_router_config_defaults_notional_gap_rebalance_for_total_equity_basis() -> None:
    config = StrategyRouterConfig.from_dict(
        {
            "mode": "live",
            "state_path": "state/test_strategy_router.json",
            "btc_strategy_config": "config/config.paper.high-leverage-structure.json",
            "qqq_strategy_config": "config/config.paper.qqq-usdt-aggressive-frozen.json",
            "qqq_sizing_basis": "total_equity",
        }
    )

    assert config.qqq_rebalance_on_notional_gap is True


def test_strategy_router_config_respects_explicit_notional_gap_disable() -> None:
    config = StrategyRouterConfig.from_dict(
        {
            "mode": "live",
            "state_path": "state/test_strategy_router.json",
            "btc_strategy_config": "config/config.paper.high-leverage-structure.json",
            "qqq_strategy_config": "config/config.paper.qqq-usdt-aggressive-frozen.json",
            "qqq_sizing_basis": "total_equity",
            "qqq_rebalance_on_notional_gap": False,
        }
    )

    assert config.qqq_rebalance_on_notional_gap is False


def test_strategy_router_config_ignores_legacy_stop_reentry_fields() -> None:
    config = StrategyRouterConfig.from_dict(
        {
            "mode": "live",
            "state_path": "state/test_strategy_router.json",
            "btc_strategy_config": "config/config.paper.high-leverage-structure.json",
            "qqq_strategy_config": "config/config.paper.qqq-usdt-aggressive-frozen.json",
            "qqq_stop_reentry_guard_enabled": True,
            "qqq_stop_reentry_min_closed_bars": 3,
            "qqq_stop_reentry_price_buffer_pct": 0.25,
            "qqq_stop_reentry_allow_new_daily_signal": True,
        }
    )

    assert config.qqq_strategy_config == "config/config.paper.qqq-usdt-aggressive-frozen.json"
    assert not hasattr(config, "qqq_stop_reentry_guard_enabled")


def test_live_router_template_uses_shadow_gate_v2_profile_only() -> None:
    template = json.loads((ROOT / "config" / "config.live.strategy-router.template.json").read_text())

    assert "qqq_stop_reentry_guard_enabled" not in template
    assert "qqq_stop_reentry_min_closed_bars" not in template
    assert "qqq_stop_reentry_price_buffer_pct" not in template
    assert "qqq_stop_reentry_allow_new_daily_signal" not in template

    qqq_config = json.loads((ROOT / str(template["qqq_strategy_config"])).read_text())
    profile = qqq_config["shadow_gate_replay_profile"]
    assert profile["runtime_enabled"] is True
    assert profile["reentry_rule"] == "clear"
    assert profile["reentry_clear_bars"] == 2
    assert profile["loss_streak_stop"] == 0
    assert profile["equity_dd_stop_pct"] == 15.0
    assert profile["equity_dd_cooldown_bars"] == 20


def test_router_holds_current_when_advantage_small() -> None:
    router = build_router()
    btc = RoutedSignalCandidate("btc_sota", "BTC/USDT:USDT", True, 58.0)
    qqq = RoutedSignalCandidate("qqq_usdt_aggressive", "QQQ/USDT:USDT", True, 63.0)
    selected, reason = router._choose_candidate([btc, qqq], current_strategy="btc_sota")
    assert selected is not None
    assert selected.strategy_id == "btc_sota"
    assert reason == "hold_current_hysteresis"


def test_router_holds_current_even_when_current_below_entry_threshold() -> None:
    router = build_router()
    btc = RoutedSignalCandidate("btc_sota", "BTC/USDT:USDT", True, 64.0)
    qqq = RoutedSignalCandidate("qqq_usdt_aggressive", "QQQ/USDT:USDT", True, 58.0)
    selected, reason = router._choose_candidate([btc, qqq], current_strategy="qqq_usdt_aggressive")
    assert selected is not None
    assert selected.strategy_id == "qqq_usdt_aggressive"
    assert reason == "hold_current_hysteresis"


def test_router_keeps_current_below_entry_threshold_when_no_challenger() -> None:
    router = build_router()
    btc = RoutedSignalCandidate("btc_sota", "BTC/USDT:USDT", False, 0.0)
    qqq = RoutedSignalCandidate("qqq_usdt_aggressive", "QQQ/USDT:USDT", True, 58.0)
    selected, reason = router._choose_candidate([btc, qqq], current_strategy="qqq_usdt_aggressive")
    assert selected is not None
    assert selected.strategy_id == "qqq_usdt_aggressive"
    assert reason == "hold_current_no_challenger"


def test_router_supports_asymmetric_takeover_advantage() -> None:
    router = StrategyRouter(
        StrategyRouterConfig(
            mode="paper",
            state_path="state/test_strategy_router.json",
            btc_strategy_config="config/config.paper.high-leverage-structure.json",
            qqq_strategy_config="config/config.paper.qqq-usdt-aggressive-frozen.json",
            btc_min_route_score=35.0,
            qqq_min_route_score=60.0,
            switch_advantage=8.0,
            btc_takeover_advantage=4.0,
            qqq_takeover_advantage=12.0,
            persist_state=False,
        )
    )

    btc = RoutedSignalCandidate("btc_sota", "BTC/USDT:USDT", True, 105.0)
    qqq = RoutedSignalCandidate("qqq_usdt_aggressive", "QQQ/USDT:USDT", True, 100.0)
    selected, reason = router._choose_candidate([btc, qqq], current_strategy="qqq_usdt_aggressive")
    assert selected is not None
    assert selected.strategy_id == "btc_sota"
    assert reason == "best_route_score"

    btc = RoutedSignalCandidate("btc_sota", "BTC/USDT:USDT", True, 100.0)
    qqq = RoutedSignalCandidate("qqq_usdt_aggressive", "QQQ/USDT:USDT", True, 110.0)
    selected, reason = router._choose_candidate([btc, qqq], current_strategy="btc_sota")
    assert selected is not None
    assert selected.strategy_id == "btc_sota"
    assert reason == "hold_current_hysteresis"


def test_router_execution_current_strategy_overrides_stale_selected_state(tmp_path: Path) -> None:
    router = StrategyRouter(
        StrategyRouterConfig(
            mode="paper",
            state_path=str(tmp_path / "router.json"),
            btc_strategy_config="config/config.paper.high-leverage-structure.json",
            qqq_strategy_config="config/config.paper.qqq-usdt-aggressive-frozen.json",
            btc_min_route_score=35.0,
            qqq_min_route_score=60.0,
            switch_advantage=8.0,
            persist_state=False,
        )
    )
    (tmp_path / "router.json").write_text('{"selected_strategy": "qqq_usdt_aggressive"}')

    class FakeAdapter:
        def __init__(self, candidate: RoutedSignalCandidate):
            self.candidate = candidate

        def preview(self) -> RoutedSignalCandidate:
            return self.candidate

    router.btc_adapter = FakeAdapter(RoutedSignalCandidate("btc_sota", "BTC/USDT:USDT", True, 96.0))
    router.qqq_adapter = FakeAdapter(RoutedSignalCandidate("qqq_usdt_aggressive", "QQQ/USDT:USDT", True, 100.0))

    payload = router.evaluate_latest(current_strategy="btc_sota")

    assert payload["previous_selected_strategy"] == "qqq_usdt_aggressive"
    assert payload["route_current_strategy"] == "btc_sota"
    assert payload["selected_strategy"] == "btc_sota"
    assert payload["decision_reason"] == "hold_current_hysteresis"


def test_router_execution_accepts_legacy_current_strategy_override(tmp_path: Path) -> None:
    router = StrategyRouter(
        StrategyRouterConfig(
            mode="paper",
            state_path=str(tmp_path / "router.json"),
            btc_strategy_config="config/config.paper.high-leverage-structure.json",
            qqq_strategy_config="config/config.paper.qqq-usdt-aggressive-frozen.json",
            btc_min_route_score=35.0,
            qqq_min_route_score=60.0,
            switch_advantage=8.0,
            persist_state=False,
        )
    )
    (tmp_path / "router.json").write_text('{"selected_strategy": "qqq_usdt_aggressive"}')

    class FakeAdapter:
        def __init__(self, candidate: RoutedSignalCandidate):
            self.candidate = candidate

        def preview(self) -> RoutedSignalCandidate:
            return self.candidate

    router.btc_adapter = FakeAdapter(RoutedSignalCandidate("btc_sota", "BTC/USDT:USDT", True, 96.0))
    router.qqq_adapter = FakeAdapter(RoutedSignalCandidate("qqq_usdt_aggressive", "QQQ/USDT:USDT", True, 100.0))

    payload = router.evaluate_latest(current_strategy_override="btc_sota")

    assert payload["previous_selected_strategy"] == "qqq_usdt_aggressive"
    assert payload["route_current_strategy"] == "btc_sota"
    assert payload["selected_strategy"] == "btc_sota"
    assert payload["decision_reason"] == "hold_current_hysteresis"


def test_router_returns_none_when_no_candidate_clears_threshold() -> None:
    router = build_router()
    btc = RoutedSignalCandidate("btc_sota", "BTC/USDT:USDT", True, 20.0)
    qqq = RoutedSignalCandidate("qqq_usdt_aggressive", "QQQ/USDT:USDT", False, 0.0)
    selected, reason = router._choose_candidate([btc, qqq], current_strategy=None)
    assert selected is None
    assert reason == "no_eligible_candidates"


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
  "exchange_leverage": 10.0,
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


def test_qqq_fixed10_profile_keeps_defense_at_ten_x() -> None:
    profile = QqqUsdtSignalAdapter._leverage_profile(
        {
            "leverage_profile_name": "fixed10",
            "base_leverage": 10.0,
            "offense_leverage": 10.0,
            "defense_leverage": 10.0,
        }
    )

    assert QqqUsdtSignalAdapter._current_leverage(profile, pd.Series({})) == (10.0, "base")
    assert QqqUsdtSignalAdapter._current_leverage(profile, pd.Series({"high_growth": True})) == (10.0, "offense")
    assert QqqUsdtSignalAdapter._current_leverage(profile, pd.Series({"defense_state": True})) == (10.0, "defense")


def test_qqq_daily_refresh_passes_configured_timeout(monkeypatch: Any, tmp_path: Path) -> None:
    adapter = QqqUsdtSignalAdapter(tmp_path / "qqq.json")
    seen: dict[str, Any] = {}

    def fake_fetch_timeframe(**kwargs: Any) -> pd.DataFrame:
        seen["timeout_seconds"] = kwargs["timeout_seconds"]
        return pd.DataFrame({"date": [pd.Timestamp("2026-06-02", tz="UTC")]})

    monkeypatch.setattr("bot.qqq_usdt_signal_adapter.fetch_timeframe", fake_fetch_timeframe)

    status = adapter._refresh_daily_signal_source(
        {
            "daily_signal_refresh_symbols": ["QQQ"],
            "daily_signal_refresh_timeout_seconds": 7.5,
            "daily_signal_refresh_total_timeout_seconds": 60.0,
        },
        {"data_root": str(tmp_path)},
    )

    assert status["status"] == "ok"
    assert seen["timeout_seconds"] == 7.5


def test_qqq_daily_refresh_reports_total_timeout(monkeypatch: Any, tmp_path: Path) -> None:
    adapter = QqqUsdtSignalAdapter(tmp_path / "qqq.json")

    def slow_fetch_timeframe(**_: Any) -> pd.DataFrame:
        time.sleep(0.02)
        raise RuntimeError("slow upstream")

    monkeypatch.setattr("bot.qqq_usdt_signal_adapter.fetch_timeframe", slow_fetch_timeframe)

    status = adapter._refresh_daily_signal_source(
        {
            "daily_signal_refresh_symbols": ["QQQ", "TQQQ"],
            "daily_signal_refresh_fail_open": True,
            "daily_signal_refresh_max_attempts": 2,
            "daily_signal_refresh_retry_sleep_seconds": 0.0,
            "daily_signal_refresh_timeout_seconds": 5.0,
            "daily_signal_refresh_total_timeout_seconds": 0.01,
        },
        {"data_root": str(tmp_path)},
    )

    assert status["status"] == "timeout"
    assert status["timed_out"] is True
    assert "QQQ" in status["errors"]


def test_router_evaluation_timeout_defaults_when_config_null() -> None:
    engine = StrategyRouterExecutionEngine.__new__(StrategyRouterExecutionEngine)
    engine.config = StrategyRouterConfig(
        mode="live",
        state_path="state/test_strategy_router.json",
        btc_strategy_config="config/config.paper.high-leverage-structure.json",
        qqq_strategy_config="config/config.paper.qqq-usdt-aggressive-frozen.json",
        router_evaluation_timeout_seconds=None,
    )

    assert engine._evaluation_timeout_seconds() == 240.0

    engine.config.router_evaluation_timeout_seconds = 0.0
    assert engine._evaluation_timeout_seconds() == 0.0


def test_qqq_live_signal_uses_strict_max_hold_days_without_override(monkeypatch: Any) -> None:
    config_path = ROOT / "config" / "config.paper.qqq-usdt-aggressive-frozen.json"
    adapter_config = json.loads(config_path.read_text())
    signal_path = ROOT / str(adapter_config["signal_source"])
    strict_config = json.loads(signal_path.read_text())

    monkeypatch.setattr(qqq_trigger_scan, "load_strict_config", lambda path: dict(strict_config))
    monkeypatch.setattr(qqq_trigger_scan, "load_strict_frame_with_overlay_context", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(qqq_trigger_scan, "run_strict_candidate", lambda frame, **kwargs: {"path": pd.DataFrame(), "kwargs": kwargs})

    signal_config, _ = qqq_trigger_scan.load_signal_path(signal_path, overrides=QqqUsdtSignalAdapter._signal_overrides(adapter_config))

    assert strict_config["max_hold_days"] == 90
    assert signal_config["max_hold_days"] == 90


def test_qqq_replay_funding_settles_once_per_8h_and_credits_negative_rate() -> None:
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-01T00:00:00Z",
                    "2026-05-01T04:00:00Z",
                    "2026-05-01T08:00:00Z",
                    "2026-05-01T12:00:00Z",
                ]
            ),
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "allow_long": [True, True, True, True],
        }
    )
    funding = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-01T00:00:00Z", "2026-05-01T08:00:00Z"]),
            "funding_event_time": pd.to_datetime(["2026-05-01T00:00:00Z", "2026-05-01T08:00:00Z"]),
            "funding_rate_value": [0.001, -0.002],
        }
    )

    result = run_10x_replay(
        bars,
        funding,
        leverage=10.0,
        taker_fee_rate=0.0,
        slippage_bps=0.0,
        stop_loss_pct=10.0,
        initial_capital=1000.0,
    )

    summary = result["summary"]
    assert summary["funding_settlement_events"] == 2
    assert summary["positive_funding_cost_pct_est"] == 1.0
    assert summary["negative_funding_credit_pct_est"] == 2.0
    assert summary["total_funding_cost_pct_est"] == -1.0
    assert summary["total_return_pct"] == 0.98


def _mock_qqq_preview(
    monkeypatch: Any,
    tmp_path: Path,
    config_overrides: dict[str, Any],
    *,
    allow_long: bool = True,
) -> RoutedSignalCandidate:
    config = {
        "signal_source": str(tmp_path / "signal.json"),
        "data_4h": str(tmp_path / "bars.feather"),
        "execution_symbol": "QQQ/USDT:USDT",
        "execution_timeframe": "4h",
        "use_closed_execution_bars": False,
        "data_refresh_enabled": False,
        "daily_signal_refresh_enabled": False,
        "daily_signal_stale_guard_enabled": False,
        "base_leverage": 10.0,
        "offense_leverage": 10.0,
        "defense_leverage": 10.0,
        "stop_loss_pct": 4.0,
        "risk_overlay_enabled": False,
        "macro_proxy_overlay_enabled": False,
    }
    config.update(config_overrides)
    config_path = tmp_path / "qqq.json"
    config_path.write_text(json.dumps(config))
    signal_path = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-29T00:00:00Z"]),
            "position": ["TQQQ" if allow_long else "CASH"],
            "entry_type": ["base"],
            "overlay_mode": [False],
            "overlay_allocation": [0.0],
            "vix_label": ["vix_normal"],
            "ixic_trend_label": ["trend_up"],
            "rel_strength_label": ["qqq_strong"],
        }
    )
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-29T00:00:00Z"]),
            "open": [700.0],
            "high": [710.0],
            "low": [695.0],
            "close": [705.0],
            "allow_long": [allow_long],
            "high_growth": [False],
            "defense_state": [False],
            "breakout_12": [False],
        }
    )

    monkeypatch.setattr("bot.qqq_usdt_signal_adapter.load_strict_config", lambda path: {"data_root": str(tmp_path)})
    monkeypatch.setattr("bot.qqq_usdt_signal_adapter.load_signal_path", lambda path, overrides=None: ({}, signal_path.copy()))
    monkeypatch.setattr("bot.qqq_usdt_signal_adapter.load_okx_4h", lambda path: bars.copy())
    monkeypatch.setattr("bot.qqq_usdt_signal_adapter.attach_daily_state", lambda okx, signal: okx.copy())
    monkeypatch.setattr("bot.qqq_usdt_signal_adapter.enrich_bars", lambda frame: frame.copy())

    return QqqUsdtSignalAdapter(config_path).preview()


def test_qqq_preview_recent_risk_cash_gate_overrides_base_long(monkeypatch: Any, tmp_path: Path) -> None:
    recent = tmp_path / "recent.csv"
    recent.write_text("date,raw_prob_10d\n2026-05-28 00:00:00+00:00,0.61\n")

    candidate = _mock_qqq_preview(
        monkeypatch,
        tmp_path,
        {
            "risk_overlay_enabled": True,
            "risk_overlay_fail_open": False,
            "risk_overlay_stale_guard_enabled": False,
            "recent_risk_predictions_csv": str(recent),
            "recent_risk_score_column": "raw_prob_10d",
            "recent_risk_cash_threshold": 0.5,
        },
    )

    assert candidate.active is False
    assert candidate.strength_label == "flat"
    assert candidate.metadata["pre_risk_allow_long"] is True
    assert candidate.metadata["risk_adjusted_allow_long"] is False
    assert candidate.metadata["risk_overlay"]["cash_gate"] is True
    assert candidate.metadata["risk_overlay"]["cash_gate_layer"] == "recent"


def test_qqq_preview_long_cycle_risk_caps_without_flipping_direction(monkeypatch: Any, tmp_path: Path) -> None:
    recent = tmp_path / "recent.csv"
    recent.write_text("date,raw_prob_10d\n2026-05-28 00:00:00+00:00,0.10\n")
    long_cycle = tmp_path / "long_cycle.csv"
    long_cycle.write_text("date,raw_prob_10d\n2026-05-28 00:00:00+00:00,0.52\n")

    candidate = _mock_qqq_preview(
        monkeypatch,
        tmp_path,
        {
            "risk_overlay_enabled": True,
            "risk_overlay_fail_open": False,
            "risk_overlay_stale_guard_enabled": False,
            "recent_risk_predictions_csv": str(recent),
            "recent_risk_score_column": "raw_prob_10d",
            "recent_risk_cash_threshold": 0.5,
            "long_cycle_risk_predictions_csv": str(long_cycle),
            "long_cycle_risk_score_column": "raw_prob_10d",
            "long_cycle_risk_cap_rules": [
                {"threshold": 0.65, "leverage_multiplier": 0.25},
                {"threshold": 0.5, "leverage_multiplier": 0.5},
                {"threshold": 0.35, "leverage_multiplier": 0.75},
            ],
        },
    )

    assert candidate.active is True
    assert candidate.direction == "BULL"
    assert candidate.leverage == 5.0
    assert candidate.metadata["risk_adjusted_allow_long"] is True
    assert candidate.metadata["risk_overlay"]["cap_layer"] == "long_cycle"
    assert candidate.metadata["risk_overlay"]["leverage_multiplier"] == 0.5


def test_qqq_preview_stale_risk_signal_fails_closed(monkeypatch: Any, tmp_path: Path) -> None:
    recent = tmp_path / "recent.csv"
    recent.write_text("date,raw_prob_10d\n2000-01-03 00:00:00+00:00,0.10\n")

    candidate = _mock_qqq_preview(
        monkeypatch,
        tmp_path,
        {
            "risk_overlay_enabled": True,
            "risk_overlay_fail_open": False,
            "risk_overlay_stale_guard_enabled": True,
            "risk_overlay_max_stale_calendar_days": 5,
            "recent_risk_predictions_csv": str(recent),
            "recent_risk_score_column": "raw_prob_10d",
        },
    )

    assert candidate.active is False
    assert candidate.metadata["reason"] == "risk_overlay_error"
    assert "stale" in candidate.metadata["error"]
    assert candidate.metadata["pre_risk_allow_long"] is True


def test_qqq_preview_malformed_risk_csv_fails_closed(monkeypatch: Any, tmp_path: Path) -> None:
    recent = tmp_path / "recent.csv"
    recent.write_text("date,not_raw_prob\n2026-05-28 00:00:00+00:00,0.10\n")

    candidate = _mock_qqq_preview(
        monkeypatch,
        tmp_path,
        {
            "risk_overlay_enabled": True,
            "risk_overlay_fail_open": False,
            "risk_overlay_stale_guard_enabled": False,
            "recent_risk_predictions_csv": str(recent),
            "recent_risk_score_column": "raw_prob_10d",
        },
    )

    assert candidate.active is False
    assert candidate.metadata["reason"] == "risk_overlay_error"
    assert "missing score column" in candidate.metadata["error"]


def test_qqq_recent_risk_overlay_cash_gates_on_raw_prob(tmp_path: Path) -> None:
    recent = tmp_path / "recent.csv"
    recent.write_text("date,raw_prob_10d\n2026-05-28 00:00:00+00:00,0.51\n")
    adapter = QqqUsdtSignalAdapter(tmp_path / "qqq.json")

    decision = adapter._risk_overlay_decision(
        {
            "risk_overlay_enabled": True,
            "risk_overlay_fail_open": False,
            "risk_overlay_stale_guard_enabled": False,
            "recent_risk_predictions_csv": str(recent),
            "recent_risk_score_column": "raw_prob_10d",
            "recent_risk_cash_threshold": 0.5,
        },
        pd.Series({"date": pd.Timestamp("2026-05-29T00:00:00Z")}),
    )

    assert decision["cash_gate"] is True
    assert decision["cash_gate_layer"] == "recent"
    assert decision["layers"]["recent"]["score"] == 0.51


def test_qqq_long_cycle_risk_overlay_caps_leverage_after_recent_layer(tmp_path: Path) -> None:
    recent = tmp_path / "recent.csv"
    recent.write_text("date,raw_prob_10d\n2026-05-28 00:00:00+00:00,0.10\n")
    long_cycle = tmp_path / "long_cycle.csv"
    long_cycle.write_text("date,raw_prob_10d\n2026-05-28 00:00:00+00:00,0.52\n")
    adapter = QqqUsdtSignalAdapter(tmp_path / "qqq.json")

    decision = adapter._risk_overlay_decision(
        {
            "risk_overlay_enabled": True,
            "risk_overlay_fail_open": False,
            "risk_overlay_stale_guard_enabled": False,
            "recent_risk_predictions_csv": str(recent),
            "recent_risk_score_column": "raw_prob_10d",
            "recent_risk_cash_threshold": 0.5,
            "long_cycle_risk_predictions_csv": str(long_cycle),
            "long_cycle_risk_score_column": "raw_prob_10d",
            "long_cycle_risk_cap_rules": [
                {"threshold": 0.65, "leverage_multiplier": 0.25},
                {"threshold": 0.5, "leverage_multiplier": 0.5},
                {"threshold": 0.35, "leverage_multiplier": 0.75},
            ],
        },
        pd.Series({"date": pd.Timestamp("2026-05-29T00:00:00Z")}),
    )

    assert decision["cash_gate"] is False
    assert decision["leverage_multiplier"] == 0.5
    assert decision["cap_layer"] == "long_cycle"
    assert decision["layers"]["recent"]["score"] == 0.1
    assert decision["layers"]["long_cycle"]["score"] == 0.52


def test_qqq_macro_proxy_overlay_disabled_keeps_leverage_identity(tmp_path: Path) -> None:
    adapter = QqqUsdtSignalAdapter(tmp_path / "qqq.json")
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-29T20:00:00Z"]),
            "daily_signal_timestamp": pd.to_datetime(["2026-05-29T00:00:00Z"]),
        }
    )

    decision = adapter._macro_proxy_overlay_decision({"macro_proxy_overlay_enabled": False}, bars, bars.iloc[-1])
    adjusted = apply_macro_proxy_overlay(allow_long=True, leverage_target=10.0, overlay=decision)

    assert decision["enabled"] is False
    assert adjusted["allow_long"] is True
    assert adjusted["leverage_target"] == 10.0
    assert adjusted["capped"] is False


def test_qqq_macro_proxy_overlay_caps_target_exposure_at_50pct(tmp_path: Path) -> None:
    macro = tmp_path / "macro.feather"
    pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-25T00:00:00Z",
                    "2026-05-26T00:00:00Z",
                    "2026-05-27T00:00:00Z",
                    "2026-05-28T00:00:00Z",
                    "2026-05-29T00:00:00Z",
                ]
            ),
            "macro_broad_dollar_index": [100.0, 100.0, 100.0, 100.0, 120.0],
        }
    ).to_feather(macro)
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-25T20:00:00Z",
                    "2026-05-26T20:00:00Z",
                    "2026-05-27T20:00:00Z",
                    "2026-05-28T20:00:00Z",
                    "2026-05-29T20:00:00Z",
                ]
            ),
            "daily_signal_timestamp": pd.to_datetime(
                [
                    "2026-05-25T00:00:00Z",
                    "2026-05-26T00:00:00Z",
                    "2026-05-27T00:00:00Z",
                    "2026-05-28T00:00:00Z",
                    "2026-05-29T00:00:00Z",
                ]
            ),
        }
    )
    adapter = QqqUsdtSignalAdapter(tmp_path / "qqq.json")

    decision = adapter._macro_proxy_overlay_decision(
        {
            "macro_proxy_overlay_enabled": True,
            "macro_proxy_overlay_fail_open": False,
            "macro_proxy_overlay_stale_guard_enabled": False,
            "macro_proxy_overlay_use_previous_signal": False,
            "macro_proxy_overlay_path": str(macro),
            "macro_proxy_overlay_mode": "dollar_zscore_cap",
            "macro_proxy_overlay_value_column": "macro_broad_dollar_index",
            "macro_proxy_overlay_dollar_z_window": 5,
            "macro_proxy_overlay_dollar_z_min_periods": 5,
            "macro_proxy_overlay_dollar_z_threshold": 1.5,
            "macro_proxy_overlay_leverage_multiplier": 0.5,
        },
        bars,
        bars.iloc[-1],
    )
    adjusted = apply_macro_proxy_overlay(allow_long=True, leverage_target=10.0, overlay=decision)

    assert decision["triggered"] is True
    assert decision["reason"] == "dollar_zscore_cap"
    assert decision["score"] is not None and float(decision["score"]) > 1.5
    assert adjusted["allow_long"] is True
    assert adjusted["leverage_target"] == 5.0
    assert adjusted["capped"] is True


def test_qqq_macro_proxy_overlay_uses_signal_day_alignment_not_natural_bar_days(tmp_path: Path) -> None:
    macro = tmp_path / "macro.feather"
    pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-02T00:00:00Z",
                    "2026-01-05T00:00:00Z",
                ]
            ),
            "macro_broad_dollar_index": [100.0, 100.0, 120.0],
        }
    ).to_feather(macro)
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01T20:00:00Z",
                    "2026-01-02T20:00:00Z",
                    "2026-01-03T00:00:00Z",
                    "2026-01-04T00:00:00Z",
                    "2026-01-05T20:00:00Z",
                ]
            ),
            "daily_signal_timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-02T00:00:00Z",
                    "2026-01-02T00:00:00Z",
                    "2026-01-02T00:00:00Z",
                    "2026-01-05T00:00:00Z",
                ]
            ),
        }
    )
    adapter = QqqUsdtSignalAdapter(tmp_path / "qqq.json")

    decision = adapter._macro_proxy_overlay_decision(
        {
            "macro_proxy_overlay_enabled": True,
            "macro_proxy_overlay_fail_open": False,
            "macro_proxy_overlay_stale_guard_enabled": False,
            "macro_proxy_overlay_use_previous_signal": False,
            "macro_proxy_overlay_path": str(macro),
            "macro_proxy_overlay_mode": "dollar_zscore_cap",
            "macro_proxy_overlay_value_column": "macro_broad_dollar_index",
            "macro_proxy_overlay_dollar_z_window": 5,
            "macro_proxy_overlay_dollar_z_min_periods": 5,
            "macro_proxy_overlay_dollar_z_threshold": 1.0,
            "macro_proxy_overlay_leverage_multiplier": 0.5,
        },
        bars,
        bars.iloc[-1],
    )
    adjusted = apply_macro_proxy_overlay(allow_long=True, leverage_target=10.0, overlay=decision)

    assert decision["available"] is False
    assert decision["reason"] == "score_nan"
    assert decision["triggered"] is False
    assert adjusted["allow_long"] is True
    assert adjusted["leverage_target"] == 10.0


def test_qqq_macro_proxy_overlay_missing_current_signal_day_noops(tmp_path: Path) -> None:
    macro = tmp_path / "macro.feather"
    pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-25T00:00:00Z",
                    "2026-05-26T00:00:00Z",
                    "2026-05-27T00:00:00Z",
                    "2026-05-28T00:00:00Z",
                    "2026-05-29T00:00:00Z",
                ]
            ),
            "macro_broad_dollar_index": [100.0, 100.0, 100.0, 100.0, 120.0],
        }
    ).to_feather(macro)
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-25T20:00:00Z",
                    "2026-05-26T20:00:00Z",
                    "2026-05-27T20:00:00Z",
                    "2026-05-28T20:00:00Z",
                    "2026-05-29T20:00:00Z",
                    "2026-06-01T20:00:00Z",
                ]
            ),
            "daily_signal_timestamp": pd.to_datetime(
                [
                    "2026-05-25T00:00:00Z",
                    "2026-05-26T00:00:00Z",
                    "2026-05-27T00:00:00Z",
                    "2026-05-28T00:00:00Z",
                    "2026-05-29T00:00:00Z",
                    "2026-06-01T00:00:00Z",
                ]
            ),
        }
    )
    adapter = QqqUsdtSignalAdapter(tmp_path / "qqq.json")

    decision = adapter._macro_proxy_overlay_decision(
        {
            "macro_proxy_overlay_enabled": True,
            "macro_proxy_overlay_fail_open": False,
            "macro_proxy_overlay_stale_guard_enabled": True,
            "macro_proxy_overlay_max_stale_calendar_days": 5,
            "macro_proxy_overlay_use_previous_signal": False,
            "macro_proxy_overlay_path": str(macro),
            "macro_proxy_overlay_mode": "dollar_zscore_cap",
            "macro_proxy_overlay_value_column": "macro_broad_dollar_index",
            "macro_proxy_overlay_dollar_z_window": 5,
            "macro_proxy_overlay_dollar_z_min_periods": 5,
            "macro_proxy_overlay_dollar_z_threshold": 1.5,
            "macro_proxy_overlay_leverage_multiplier": 0.5,
        },
        bars,
        bars.iloc[-1],
    )
    adjusted = apply_macro_proxy_overlay(allow_long=True, leverage_target=10.0, overlay=decision)

    assert decision["available"] is False
    assert decision["ignored"] is True
    assert decision["reason"] == "macro_signal_not_current"
    assert decision["macro_signal_date"] == "2026-05-29 00:00:00+00:00"
    assert decision["source_lag_days"] == 3
    assert adjusted["allow_long"] is True
    assert adjusted["leverage_target"] == 10.0


def test_qqq_risk_overlay_missing_file_fails_closed_when_configured(tmp_path: Path) -> None:
    adapter = QqqUsdtSignalAdapter(tmp_path / "qqq.json")

    try:
        adapter._risk_overlay_decision(
            {
                "risk_overlay_enabled": True,
                "risk_overlay_fail_open": False,
                "recent_risk_predictions_csv": str(tmp_path / "missing.csv"),
                "recent_risk_score_column": "raw_prob_10d",
            },
            pd.Series({"date": pd.Timestamp("2026-05-29T00:00:00Z")}),
        )
    except FileNotFoundError:
        return

    raise AssertionError("missing risk prediction CSV should fail closed")


def test_qqq_risk_overlay_stale_signal_fails_closed_when_configured(tmp_path: Path) -> None:
    recent = tmp_path / "recent.csv"
    recent.write_text("date,raw_prob_10d\n2000-01-03 00:00:00+00:00,0.10\n")
    adapter = QqqUsdtSignalAdapter(tmp_path / "qqq.json")

    try:
        adapter._risk_overlay_decision(
            {
                "risk_overlay_enabled": True,
                "risk_overlay_fail_open": False,
                "risk_overlay_stale_guard_enabled": True,
                "risk_overlay_max_stale_calendar_days": 5,
                "risk_overlay_use_previous_signal": True,
                "recent_risk_predictions_csv": str(recent),
                "recent_risk_score_column": "raw_prob_10d",
            },
            pd.Series({"date": pd.Timestamp("2026-05-29T00:00:00Z")}),
        )
    except RuntimeError as exc:
        assert "stale" in str(exc)
        return

    raise AssertionError("stale risk prediction CSV should fail closed")


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


def test_qqq_extract_exchange_stop_fields_reads_pending_algos(tmp_path: Path) -> None:
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

    class FakeClient:
        def fetch_pending_algo_orders(self, params):
            if params["ordType"] != "conditional":
                return {"data": []}
            return {
                "data": [
                    {
                        "instId": "QQQ-USDT-SWAP",
                        "ordType": "conditional",
                        "state": "live",
                        "posSide": "long",
                        "side": "sell",
                        "reduceOnly": "true",
                        "algoId": "algo-1",
                        "algoClOrdId": "client-1",
                        "slTriggerPx": "716.55",
                        "sz": "258.1",
                    },
                    {
                        "instId": "QQQ-USDT-SWAP",
                        "ordType": "conditional",
                        "state": "live",
                        "posSide": "long",
                        "side": "sell",
                        "reduceOnly": "true",
                        "algoId": "algo-2",
                        "algoClOrdId": "client-2",
                        "slTriggerPx": "709.22",
                        "sz": "270",
                    },
                ]
            }

    engine.client = FakeClient()
    fields = engine._extract_exchange_stop_fields(
        {"contracts": 401.84, "notional_usdt": 299525.73, "close_order_algos": [], "raw": {"info": {"closeOrderAlgo": []}}},
        {"exchange_attach_algo_client_id": "client-1"},
    )

    assert fields["status"] == "diverged"
    assert fields["algo_client_id"] == "client-1"
    assert fields["stop_price"] == 716.55
    assert fields["order_count"] == 2
    assert fields["stop_price_min"] == 709.22
    assert fields["stop_price_max"] == 716.55


def test_qqq_live_trailing_stop_syncs_all_pending_exchange_stops_without_local_advance(tmp_path: Path) -> None:
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
                "stop_price": 716.55,
                "exchange_attach_algo_client_id": "client-1",
            }
        }
    )
    amend_calls = []

    class FakeClient:
        def fetch_pending_algo_orders(self, params):
            if params["ordType"] != "conditional":
                return {"data": []}
            return {
                "data": [
                    {
                        "instId": "QQQ-USDT-SWAP",
                        "ordType": "conditional",
                        "state": "live",
                        "posSide": "long",
                        "side": "sell",
                        "reduceOnly": "true",
                        "algoId": "algo-1",
                        "algoClOrdId": "client-1",
                        "slTriggerPx": "716.55",
                        "sz": "258.1",
                    },
                    {
                        "instId": "QQQ-USDT-SWAP",
                        "ordType": "conditional",
                        "state": "live",
                        "posSide": "long",
                        "side": "sell",
                        "reduceOnly": "true",
                        "algoId": "algo-2",
                        "algoClOrdId": "client-2",
                        "slTriggerPx": "709.22",
                        "sz": "270",
                    },
                ]
            }

        def amend_algo_order(self, request):
            amend_calls.append(request)
            return {"code": "0", "data": [{"algoId": request.get("algoId")}]}

    engine.client = FakeClient()
    context = QqqOrderContext(
        symbol="QQQ/USDT:USDT",
        margin_mode="isolated",
        leverage=10.0,
        stop_loss_pct=3.5,
        reference_price=742.8,
        latest_low=736.59,
        stop_price=716.55,
        stop_hit=False,
        route_score=96.0,
        candidate={"timestamp": "2026-06-02 12:00:00+00:00"},
    )

    result = engine.update_trailing_stop(
        context,
        {"contracts": 401.84, "notional_usdt": 299525.73, "close_order_algos": [], "raw": {"info": {"closeOrderAlgo": []}}},
    )

    assert result is not None
    assert result["status"] == "synced"
    assert result["exchange_stop"]["status"] == "amended"
    assert result["exchange_stop"]["amended_count"] == 2
    assert [request["algoId"] for request in amend_calls] == ["algo-1", "algo-2"]
    assert all(request["newSlTriggerPx"] == "716.55" for request in amend_calls)


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
            qqq_max_market_order_contracts=3.0,
            qqq_market_order_chunk_delay_seconds=0.0,
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
        leverage=2.5,
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
        target_notional=1750.0,
    )

    assert result["status"] == "submitted"
    assert result["side"] == "sell"
    assert result["amount"] == 7.5
    assert [call[4] for call in calls if call[0] == "create_order"] == [3.0, 3.0, 1.5]
    assert all(call[3] == "sell" for call in calls if call[0] == "create_order")
    assert all(call[5]["reduceOnly"] is True for call in calls if call[0] == "create_order")
    assert result["target_exposure_leverage"] == 2.5
    assert result["exchange_leverage"] == 10.0
    assert [order["amount"] for order in result["orders"]] == [3.0, 3.0, 1.5]
    assert calls[-1] == ("set_leverage", 10, "QQQ/USDT:USDT", "isolated", "long")


def test_qqq_live_rebalance_adds_only_delta_position(tmp_path: Path) -> None:
    config_path = tmp_path / "qqq.json"
    config_path.write_text(
        """
{
  "execution_symbol": "QQQ/USDT:USDT",
  "exchange_leverage": 10.0,
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


def test_qqq_live_rebalance_add_chunks_by_market_max(tmp_path: Path) -> None:
    config_path = tmp_path / "qqq.json"
    config_path.write_text(
        """
{
  "execution_symbol": "QQQ/USDT:USDT",
  "exchange_leverage": 10.0,
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
            qqq_market_order_chunk_delay_seconds=0.0,
        ),
        config_path,
    )
    engine._markets_cache = {
        "QQQ/USDT:USDT": {
            "id": "QQQ-USDT-SWAP",
            "contract": True,
            "contractSize": 1.0,
            "precision": {"amount": 0.01, "price": 0.01},
            "limits": {"amount": {"min": 0.01, "max": None}},
            "info": {"maxMktSz": "3"},
        }
    }
    calls = []

    class FakeClient:
        def set_leverage(self, leverage, symbol, margin_mode="isolated", pos_side=None):
            calls.append(("set_leverage", leverage, symbol, margin_mode, pos_side))
            return {"code": "0"}

        def create_order(self, symbol, order_type, side, amount, price=None, *, params=None):
            calls.append(("create_order", symbol, order_type, side, amount, params))
            return {"id": f"add-{len([call for call in calls if call[0] == 'create_order'])}"}

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
    assert [call[4] for call in calls if call[0] == "create_order"] == [3.0, 3.0, 2.0]
    assert [order["amount"] for order in result["orders"]] == [3.0, 3.0, 2.0]


def test_qqq_live_open_chunks_by_market_max(tmp_path: Path) -> None:
    config_path = tmp_path / "qqq.json"
    config_path.write_text(
        """
{
  "execution_symbol": "QQQ/USDT:USDT",
  "exchange_leverage": 10.0,
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
            qqq_market_order_chunk_delay_seconds=0.0,
        ),
        config_path,
    )
    engine._markets_cache = {
        "QQQ/USDT:USDT": {
            "id": "QQQ-USDT-SWAP",
            "contract": True,
            "contractSize": 1.0,
            "precision": {"amount": 0.01, "price": 0.01},
            "limits": {"amount": {"min": 0.01, "max": None}},
            "info": {"maxMktSz": "3"},
        }
    }
    calls = []

    class FakeClient:
        def fetch_balance(self):
            return {"USDT": {"free": 560.0}}

        def set_leverage(self, leverage, symbol, margin_mode="isolated", pos_side=None):
            calls.append(("set_leverage", leverage, symbol, margin_mode, pos_side))
            return {"code": "0"}

        def create_order(self, symbol, order_type, side, amount, price=None, *, params=None):
            calls.append(("create_order", symbol, order_type, side, amount, params))
            return {"id": f"open-{len([call for call in calls if call[0] == 'create_order'])}"}

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

    result = engine.open_position(context)

    assert result["status"] == "submitted"
    assert result["amount"] == 8.0
    assert [call[4] for call in calls if call[0] == "create_order"] == [3.0, 3.0, 2.0]
    assert [order["amount"] for order in result["orders"]] == [3.0, 3.0, 2.0]


def test_qqq_target_notional_uses_total_equity_basis_and_buffer(tmp_path: Path) -> None:
    config_path = tmp_path / "qqq.json"
    config_path.write_text(
        """
{
  "execution_symbol": "QQQ/USDT:USDT",
  "exchange_leverage": 10.0,
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
            qqq_sizing_basis="total_equity",
            qqq_sizing_cash_buffer_usdt=100.0,
        ),
        config_path,
    )

    class FakeClient:
        def fetch_balance(self):
            return {
                "USDT": {"free": 1000.0},
                "info": {"data": [{"details": [{"ccy": "USDT", "eq": "3000"}]}]},
            }

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

    assert engine._target_notional(context) == 29000.0


def test_qqq_rebalance_on_notional_gap_triggers_without_leverage_change(tmp_path: Path) -> None:
    config_path = tmp_path / "qqq.json"
    config_path.write_text(
        """
{
  "execution_symbol": "QQQ/USDT:USDT",
  "exchange_leverage": 10.0,
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
            qqq_sizing_basis="total_equity",
            qqq_sizing_cash_buffer_usdt=100.0,
            qqq_rebalance_on_notional_gap=True,
            qqq_min_rebalance_notional_usdt=500.0,
        ),
        config_path,
    )
    engine.save_state({"position": {"leverage": 10.0, "stop_price": 675.5}})

    class FakeClient:
        def fetch_balance(self):
            return {
                "USDT": {"free": 1000.0},
                "info": {"data": [{"details": [{"ccy": "USDT", "eq": "3000"}]}]},
            }

    engine.client = FakeClient()
    engine.fetch_position_state = lambda: {"contracts": 1.0, "notional_usdt": 10000.0, "raw": None}  # type: ignore[method-assign]
    engine._risk_on_window_status = lambda: {"enabled": True, "open": True}  # type: ignore[method-assign]
    captured: dict[str, Any] = {}

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
    engine._build_context = lambda candidate: context  # type: ignore[method-assign]

    def fake_rebalance_position(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured["target_notional"] = kwargs["target_notional"]
        return {"status": "submitted", "action": "rebalance_qqq_position", "side": "buy"}

    engine.rebalance_position = fake_rebalance_position  # type: ignore[method-assign]

    result = engine.evaluate_latest(RoutedSignalCandidate("qqq_usdt_aggressive", "QQQ/USDT:USDT", True, 100.0))

    assert result["actions"][0]["action"] == "rebalance_qqq_position"
    assert captured["target_notional"] == 29000.0


def test_qqq_inactive_target_closes_existing_position_once(tmp_path: Path) -> None:
    config_path = tmp_path / "qqq.json"
    config_path.write_text(
        """
{
  "execution_symbol": "QQQ/USDT:USDT",
  "exchange_leverage": 10.0,
  "base_leverage": 10.0,
  "offense_leverage": 10.0,
  "stop_loss_pct": 4.0
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
    engine.save_state(
        {
            "position": {
                "symbol": "QQQ/USDT:USDT",
                "leverage": 10.0,
                "stop_loss_pct": 4.0,
                "peak_price": 700.0,
                "latest_low": 690.0,
                "stop_price": 672.0,
                "route_score": 92.0,
            }
        }
    )
    inactive = RoutedSignalCandidate("qqq_usdt_aggressive", "QQQ/USDT:USDT", False, 0.0)

    first = engine.evaluate_latest(inactive)
    second = engine.evaluate_latest(inactive)

    assert len(first["actions"]) == 1
    assert first["actions"][0]["status"] == "paper_closed"
    assert first["actions"][0]["reason"] == "router_no_qqq_signal"
    assert first["position_open"] is False
    assert second["actions"] == []
    assert second["position_open"] is False


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


def test_qqq_daily_signal_refresh_retries_transient_errors(monkeypatch, tmp_path: Path) -> None:
    calls = {"QQQ": 0}

    def fake_fetch_timeframe(**kwargs: Any) -> pd.DataFrame:
        calls[str(kwargs["symbol"])] += 1
        if calls[str(kwargs["symbol"])] < 3:
            raise RuntimeError("transient yahoo 400")
        return pd.DataFrame({"date": [pd.Timestamp("2026-05-29T20:00:00Z")]})

    monkeypatch.setattr("bot.qqq_usdt_signal_adapter.fetch_timeframe", fake_fetch_timeframe)
    monkeypatch.setattr("bot.qqq_usdt_signal_adapter.time.sleep", lambda _: None)

    adapter = QqqUsdtSignalAdapter(tmp_path / "qqq.json")
    status = adapter._refresh_daily_signal_source(
        {
            "daily_signal_refresh_enabled": True,
            "daily_signal_refresh_symbols": ["QQQ"],
            "daily_signal_refresh_max_attempts": 4,
            "daily_signal_refresh_retry_sleep_seconds": 0,
            "daily_signal_refresh_fail_open": True,
        },
        {"data_root": str(tmp_path)},
    )

    assert status["status"] == "ok"
    assert status["errors"] == {}
    assert status["attempts"] == {"QQQ": 3}
    assert calls["QQQ"] == 3


def test_qqq_daily_signal_refresh_raises_after_retry_exhaustion(monkeypatch, tmp_path: Path) -> None:
    calls = {"QQQ": 0}

    def fake_fetch_timeframe(**kwargs: Any) -> pd.DataFrame:
        calls[str(kwargs["symbol"])] += 1
        raise RuntimeError("persistent yahoo 400")

    monkeypatch.setattr("bot.qqq_usdt_signal_adapter.fetch_timeframe", fake_fetch_timeframe)
    monkeypatch.setattr("bot.qqq_usdt_signal_adapter.time.sleep", lambda _: None)

    adapter = QqqUsdtSignalAdapter(tmp_path / "qqq.json")
    try:
        adapter._refresh_daily_signal_source(
            {
                "daily_signal_refresh_enabled": True,
                "daily_signal_refresh_symbols": ["QQQ"],
                "daily_signal_refresh_max_attempts": 3,
                "daily_signal_refresh_retry_sleep_seconds": 0,
                "daily_signal_refresh_fail_open": False,
            },
            {"data_root": str(tmp_path)},
        )
    except RuntimeError as exc:
        assert "persistent yahoo 400" in str(exc)
    else:
        raise AssertionError("persistent refresh failure should raise when fail_open is disabled")
    assert calls["QQQ"] == 3


def test_qqq_closed_bar_filter_drops_incomplete_4h_bar() -> None:
    frame = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2026-05-29T00:00:00Z"),
                pd.Timestamp("2026-05-29T04:00:00Z"),
                pd.Timestamp("2026-05-29T08:00:00Z"),
            ],
            "close": [1.0, 2.0, 3.0],
        }
    )

    closed = filter_closed_bars(frame, timeframe="4h", now=pd.Timestamp("2026-05-29T11:59:00Z"))

    assert list(closed["date"]) == [
        pd.Timestamp("2026-05-29T00:00:00Z"),
        pd.Timestamp("2026-05-29T04:00:00Z"),
    ]


def test_qqq_market_hours_window_uses_us_regular_session() -> None:
    open_status = market_time_window_status(
        enabled=True,
        timezone_name="America/New_York",
        start_time="09:30",
        end_time="16:00",
        now=pd.Timestamp("2026-05-29T14:00:00Z"),
    )
    closed_status = market_time_window_status(
        enabled=True,
        timezone_name="America/New_York",
        start_time="09:30",
        end_time="16:00",
        now=pd.Timestamp("2026-05-29T21:00:00Z"),
    )

    assert open_status["open"] is True
    assert closed_status["open"] is False


def test_qqq_market_calendar_blocks_nyse_holiday() -> None:
    status = market_time_window_status(
        enabled=True,
        timezone_name="America/New_York",
        start_time="09:30",
        end_time="16:00",
        now=pd.Timestamp("2026-07-03T14:00:00Z"),
    )

    assert status["open"] is False
    assert status["reason"] == "non_trading_day"
    assert status["trading_calendar"]["holiday"] is True


def test_qqq_market_calendar_handles_nyse_half_day() -> None:
    early_open = market_time_window_status(
        enabled=True,
        timezone_name="America/New_York",
        start_time="09:30",
        end_time="16:00",
        now=pd.Timestamp("2026-11-27T17:30:00Z"),
    )
    after_early_close = market_time_window_status(
        enabled=True,
        timezone_name="America/New_York",
        start_time="09:30",
        end_time="16:00",
        now=pd.Timestamp("2026-11-27T19:00:00Z"),
    )

    assert early_open["open"] is True
    assert early_open["end"] == "13:00"
    assert early_open["trading_calendar"]["half_day"] is True
    assert after_early_close["open"] is False
    assert after_early_close["reason"] == "after_window"


def test_qqq_trading_calendar_status_supports_juneteenth() -> None:
    status = trading_calendar_status("NYSE", pd.Timestamp("2026-06-19").date())

    assert status["trading_day"] is False
    assert status["holiday"] is True


def test_qqq_risk_on_window_blocks_add_but_not_reduce(tmp_path: Path) -> None:
    config_path = tmp_path / "qqq.json"
    config_path.write_text(
        """
{
  "execution_symbol": "QQQ/USDT:USDT",
  "base_leverage": 10.0,
  "offense_leverage": 10.0,
  "defense_leverage": 1.0,
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
            qqq_rebalance_risk_on_market_hours_only=True,
        ),
        config_path,
    )
    engine._risk_on_window_status = lambda: {"enabled": True, "open": False}  # type: ignore[method-assign]
    add_context = QqqOrderContext(
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
    reduce_context = QqqOrderContext(
        symbol="QQQ/USDT:USDT",
        margin_mode="isolated",
        leverage=1.0,
        stop_loss_pct=3.5,
        reference_price=700.0,
        latest_low=690.0,
        stop_price=675.5,
        stop_hit=False,
        route_score=60.0,
        candidate={"timestamp": "2026-05-29 12:00:00+00:00"},
    )

    assert engine._risk_on_window_status()["open"] is False

    # Opening a new risk-on position is blocked outside the configured market window.
    engine.fetch_position_state = lambda: {"contracts": 0.0, "notional_usdt": 0.0, "raw": None}  # type: ignore[method-assign]
    engine._build_context = lambda candidate: add_context  # type: ignore[method-assign]
    result = engine.evaluate_latest(RoutedSignalCandidate("qqq_usdt_aggressive", "QQQ/USDT:USDT", True, 100.0, leverage=10.0))
    assert result["actions"][0]["reason"] == "qqq_risk_on_window_closed"

    # Reducing exposure is still allowed even outside the window.
    engine.save_state({"position": {"leverage": 10.0}})
    engine.fetch_position_state = lambda: {"contracts": 10.0, "notional_usdt": 7000.0, "raw": None}  # type: ignore[method-assign]
    engine._build_context = lambda candidate: reduce_context  # type: ignore[method-assign]
    engine.rebalance_position = lambda *args, **kwargs: {"status": "paper_rebalanced", "side": "sell"}  # type: ignore[method-assign]
    result = engine.evaluate_latest(RoutedSignalCandidate("qqq_usdt_aggressive", "QQQ/USDT:USDT", True, 60.0, leverage=1.0))
    assert result["actions"][0]["side"] == "sell"


def test_router_does_not_flatten_btc_before_blocked_qqq_risk_on_switch(tmp_path: Path) -> None:
    router = StrategyRouter(
        StrategyRouterConfig(
            mode="paper",
            state_path=str(tmp_path / "router.json"),
            btc_strategy_config="config/config.paper.high-leverage-structure.json",
            qqq_strategy_config="config/config.paper.qqq-usdt-aggressive-frozen.json",
            btc_min_route_score=35.0,
            qqq_min_route_score=60.0,
            persist_state=False,
            qqq_rebalance_risk_on_market_hours_only=True,
        )
    )
    engine = StrategyRouterExecutionEngine.__new__(StrategyRouterExecutionEngine)
    engine.router = router
    engine.config = router.config
    engine.execution_state_path = tmp_path / "router.execution.json"

    flatten_calls = []
    engine._current_executed_strategy = lambda: "btc_sota"  # type: ignore[method-assign]
    engine._set_current_executed_strategy = lambda strategy: None  # type: ignore[method-assign]
    engine._maybe_send_telegram_notifications = lambda payload: None  # type: ignore[method-assign]
    engine._load_execution_state = lambda: {"current_executed_strategy": "btc_sota"}  # type: ignore[method-assign]
    engine._save_execution_state = lambda payload: None  # type: ignore[method-assign]
    engine._flatten_strategy = lambda strategy, *, reason: flatten_calls.append((strategy, reason)) or []  # type: ignore[method-assign]
    route_current_strategies = []

    def evaluate_route(current_strategy=None):
        route_current_strategies.append(current_strategy)
        return {
            "selected_strategy": "qqq_usdt_aggressive",
            "selected_candidate": {
                "strategy_id": "qqq_usdt_aggressive",
                "symbol": "QQQ/USDT:USDT",
                "active": True,
                "route_score": 100.0,
                "leverage": 10.0,
                "metadata": {},
            },
        }

    engine.router.evaluate_latest = evaluate_route  # type: ignore[method-assign]

    class FakeQqqExecutor:
        def risk_on_window_status(self):
            return {"enabled": True, "open": False}

    class FakeBtcExecutor:
        def evaluate_latest(self):
            return {"status": "ok"}

    engine.qqq_executor = FakeQqqExecutor()
    engine.btc_executor = FakeBtcExecutor()

    result = engine.evaluate_latest()

    assert flatten_calls == []
    assert route_current_strategies == ["btc_sota"]
    assert result["current_executed_strategy"] == "btc_sota"
    assert result["execution_results"][0]["result"]["reason"] == "qqq_risk_on_window_closed_before_switch"


def test_router_does_not_flatten_btc_before_blocked_qqq_shadow_gate_switch(tmp_path: Path) -> None:
    router = StrategyRouter(
        StrategyRouterConfig(
            mode="paper",
            state_path=str(tmp_path / "router.json"),
            btc_strategy_config="config/config.paper.high-leverage-structure.json",
            qqq_strategy_config="config/config.paper.qqq-usdt-aggressive-frozen.json",
            btc_min_route_score=35.0,
            qqq_min_route_score=60.0,
            persist_state=False,
        )
    )
    engine = StrategyRouterExecutionEngine.__new__(StrategyRouterExecutionEngine)
    engine.router = router
    engine.config = router.config
    engine.execution_state_path = tmp_path / "router.execution.json"

    flatten_calls = []
    engine._current_executed_strategy = lambda: "btc_sota"  # type: ignore[method-assign]
    engine._set_current_executed_strategy = lambda strategy: None  # type: ignore[method-assign]
    engine._maybe_send_telegram_notifications = lambda payload: None  # type: ignore[method-assign]
    engine._load_execution_state = lambda: {"current_executed_strategy": "btc_sota"}  # type: ignore[method-assign]
    engine._save_execution_state = lambda payload: None  # type: ignore[method-assign]
    engine._flatten_strategy = lambda strategy, *, reason: flatten_calls.append((strategy, reason)) or []  # type: ignore[method-assign]
    engine.router.evaluate_latest = lambda current_strategy=None: {  # type: ignore[method-assign]
        "selected_strategy": "qqq_usdt_aggressive",
        "selected_candidate": {
            "strategy_id": "qqq_usdt_aggressive",
            "symbol": "QQQ/USDT:USDT",
            "active": True,
            "route_score": 100.0,
            "leverage": 10.0,
            "metadata": {},
        },
    }

    class FakeQqqExecutor:
        def risk_on_window_status(self):
            return {"enabled": True, "open": True}

        def shadow_gate_pre_switch_status(self, candidate):
            return {"enabled": True, "allow": False, "reason": "gate_cooldown", "gate_remaining_bars": 7}

    class FakeBtcExecutor:
        def evaluate_latest(self):
            return {"status": "ok"}

    engine.qqq_executor = FakeQqqExecutor()
    engine.btc_executor = FakeBtcExecutor()

    result = engine.evaluate_latest()

    assert flatten_calls == []
    assert result["current_executed_strategy"] == "btc_sota"
    assert result["execution_results"][0]["result"]["reason"] == "qqq_shadow_gate_blocked_before_switch"


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


def test_qqq_close_position_chunks_and_confirms_flat(tmp_path: Path) -> None:
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
            qqq_max_close_order_contracts=3.0,
            qqq_close_confirm_timeout_seconds=0.0,
            qqq_close_confirm_poll_seconds=0.1,
            qqq_close_chunk_delay_seconds=0.0,
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
    position_contracts = {"value": 7.0}

    class FakeClient:
        def create_order(self, symbol, order_type, side, amount, price=None, *, params=None):
            calls.append((symbol, order_type, side, amount, params))
            position_contracts["value"] = max(0.0, position_contracts["value"] - float(amount))
            return {"id": f"close-{len(calls)}"}

    engine.client = FakeClient()
    engine.fetch_position_state = lambda: {  # type: ignore[method-assign]
        "contracts": position_contracts["value"],
        "notional_usdt": position_contracts["value"] * 700.0,
        "raw": None,
    }

    result = engine.close_position(reason="router_switch_to_btc_sota")

    assert result["status"] == "closed_confirmed"
    assert result["amount"] == 7.0
    assert result["remaining_contracts"] == 0.0
    assert [call[3] for call in calls] == [3.0, 3.0, 1.0]
    assert all(call[4]["reduceOnly"] is True for call in calls)


def test_router_blocks_btc_open_when_qqq_flatten_unconfirmed(tmp_path: Path) -> None:
    router = StrategyRouter(
        StrategyRouterConfig(
            mode="live",
            state_path=str(tmp_path / "router_state.json"),
            btc_strategy_config="config/config.paper.high-leverage-structure.json",
            qqq_strategy_config="config/config.paper.qqq-usdt-aggressive-frozen.json",
            persist_state=False,
            execution_enabled=True,
            flatten_before_switch=True,
        )
    )
    engine = StrategyRouterExecutionEngine.__new__(StrategyRouterExecutionEngine)
    engine.router = router
    engine.config = router.config
    engine.execution_state_path = tmp_path / "router_state.execution"
    btc_calls = []
    execution_state = {"current_executed_strategy": "qqq_usdt_aggressive"}

    engine._load_execution_state = lambda: dict(execution_state)  # type: ignore[method-assign]

    def save_execution_state(payload):
        execution_state.clear()
        execution_state.update(payload)

    engine._save_execution_state = save_execution_state  # type: ignore[method-assign]
    engine._maybe_send_telegram_notifications = lambda payload: None  # type: ignore[method-assign]
    engine._flatten_strategy = lambda strategy, *, reason: [  # type: ignore[method-assign]
        {
            "strategy": strategy,
            "result": {
                "status": "submitted_but_unconfirmed",
                "action": "close_qqq_usdt_long",
                "remaining_contracts": 1.0,
                "reason": reason,
            },
        }
    ]
    engine.router.evaluate_latest = lambda current_strategy=None: {  # type: ignore[method-assign]
        "selected_strategy": "btc_sota",
        "selected_candidate": {
            "strategy_id": "btc_sota",
            "symbol": "BTC/USDT:USDT",
            "active": True,
            "route_score": 96.0,
        },
    }

    class FakeBtcExecutor:
        def evaluate_latest(self):
            btc_calls.append("btc.evaluate_latest")
            return {"status": "mock_btc_opened"}

    engine.btc_executor = FakeBtcExecutor()
    engine.qqq_executor = object()

    result = engine.evaluate_latest()

    assert result["status"] == "blocked"
    assert result["blocked_reason"] == "flatten_not_confirmed"
    assert result["current_executed_strategy"] == "qqq_usdt_aggressive"
    assert btc_calls == []


def test_router_flattens_btc_through_executor_router_switch_close() -> None:
    engine = StrategyRouterExecutionEngine.__new__(StrategyRouterExecutionEngine)
    engine.config = StrategyRouterConfig(
        mode="live",
        state_path="state/test_strategy_router.json",
        btc_strategy_config="config/config.paper.high-leverage-structure.json",
        qqq_strategy_config="config/config.paper.qqq-usdt-aggressive-frozen.json",
    )
    calls = []

    class FakeBtcExecutor:
        def close_for_router_switch(self, *, reason: str) -> dict[str, str]:
            calls.append(reason)
            return {"status": "submitted", "reason": reason}

    engine.btc_executor = FakeBtcExecutor()

    result = engine._flatten_strategy("btc_sota", reason="router_switch_to_qqq_usdt_aggressive")

    assert calls == ["router_switch_to_qqq_usdt_aggressive"]
    assert result == [
        {
            "strategy": "btc_sota",
            "result": {"status": "submitted", "reason": "router_switch_to_qqq_usdt_aggressive"},
        }
    ]
