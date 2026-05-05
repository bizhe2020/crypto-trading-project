from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from bot.okx_executor import ExecutorConfig, OkxExecutionEngine, OverlayRuntimePosition
from strategy.scalp_robust_v2_core import ActionType, Direction, PendingPullback, StrategyAction, StrategySnapshot


class LiveOverlayRuntimeTest(unittest.TestCase):
    def build_executor(self) -> OkxExecutionEngine:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return OkxExecutionEngine(
            ExecutorConfig(
                mode="paper",
                symbol="BTC/USDT:USDT",
                timeframe="15m",
                informative_timeframe="4h",
                leverage=5,
                margin_mode="isolated",
                max_open_positions=1,
                risk_per_trade=0.035,
                state_db_path=str(Path(tmp.name) / "state.db"),
                telegram_enabled=False,
                enable_live_overlay_strategy=True,
            )
        )

    def test_overlay_runtime_position_closes_on_target(self) -> None:
        executor = self.build_executor()
        engine = SimpleNamespace(
            capital=1000.0,
            trades=[],
            exit_reasons={},
            c15m=[SimpleNamespace(c=100.0, h=101.0, l=99.0), SimpleNamespace(c=97.0, h=100.0, l=94.0)],
            _timestamp_for_idx=lambda idx: f"t{idx}",
        )
        position = OverlayRuntimePosition(
            event_type="stable_reverse_short",
            direction=Direction.BEAR,
            entry_idx=0,
            entry_time="t0",
            exit_idx=None,
            target_rr=2.875,
            max_hold_bars=40,
            allocation=1.0,
            leverage=5.0,
            capital_at_entry=1000.0,
            signal_entry_price=100.0,
            entry_price=100.0,
            sl_price=101.0,
            initial_sl_price=101.0,
            target_price=95.0,
            risk_points=1.0,
            quantity=50.0,
            notional=5000.0,
            entry_fee=2.5,
            entry_slippage_cost=0.0,
            stop_reason="stop_loss",
            target_reason="target_rr",
            metadata={},
        )
        executor._save_overlay_runtime_position(position)

        actions = executor._overlay_manage_runtime_position(engine, 1)

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].type, ActionType.CLOSE_POSITION)
        self.assertEqual(actions[0].reason, "target_rr")

        executor._overlay_post_execute_runtime_update(engine, actions[0], 1)

        self.assertIsNone(executor._load_overlay_runtime_position())
        self.assertEqual(len(engine.trades), 1)
        self.assertEqual(engine.trades[0].exit_reason, "target_rr")

    def test_live_overlay_prefers_stable_overlapping_sota_open(self) -> None:
        executor = self.build_executor()
        executor._sota_overlay_account_lock_pre_open = lambda action, engine, candidate=None: (  # type: ignore[method-assign]
            None if str((action.metadata or {}).get("overlay_event_type") or "sota_long") == "stable_reverse_short" else {"status": "rejected"}
        )
        base_action = StrategyAction(
            type=ActionType.OPEN_LONG,
            timestamp="t5",
            direction=Direction.BULL,
            entry_price=100.0,
            stop_price=99.0,
            target_price=103.0,
            metadata={"index": 5, "overlay_event_type": "sota_long"},
        )
        engine = SimpleNamespace(
            capital=1000.0,
            position=None,
            trades=[],
            exit_reasons={},
            c15m=[SimpleNamespace(c=100.0, h=101.0, l=99.0) for _ in range(8)],
            _timestamp_for_idx=lambda idx: f"t{idx}",
            evaluate_range=lambda start, end: [base_action] if start == 5 else [],
        )
        executor._overlay_manage_runtime_position = lambda loaded, idx: []  # type: ignore[method-assign]
        executor._overlay_maybe_build_stable_candidate = lambda loaded, idx: StrategyAction(  # type: ignore[method-assign]
            type=ActionType.OPEN_SHORT,
            timestamp="t5",
            direction=Direction.BEAR,
            entry_price=100.0,
            stop_price=101.0,
            target_price=95.0,
            metadata={"index": 5, "entry_idx": 5, "overlay_event_type": "stable_reverse_short", "target_rr": 2.875, "max_hold_bars": 40},
        )
        executor._overlay_maybe_build_smc_candidate = lambda loaded, idx: None  # type: ignore[method-assign]

        actions = executor._evaluate_latest_with_live_overlay(engine, 5, 5)

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].type, ActionType.OPEN_SHORT)
        self.assertEqual((actions[0].metadata or {}).get("overlay_event_type"), "stable_reverse_short")

    def test_runtime_position_blocks_new_candidates_same_bar(self) -> None:
        executor = self.build_executor()
        position = OverlayRuntimePosition(
            event_type="stable_reverse_short",
            direction=Direction.BEAR,
            entry_idx=0,
            entry_time="t0",
            exit_idx=None,
            target_rr=2.875,
            max_hold_bars=40,
            allocation=1.0,
            leverage=5.0,
            capital_at_entry=1000.0,
            signal_entry_price=100.0,
            entry_price=100.0,
            sl_price=101.0,
            initial_sl_price=101.0,
            target_price=95.0,
            risk_points=1.0,
            quantity=50.0,
            notional=5000.0,
            entry_fee=2.5,
            entry_slippage_cost=0.0,
            stop_reason="stop_loss",
            target_reason="target_rr",
            metadata={},
        )
        executor._save_overlay_runtime_position(position)
        engine = SimpleNamespace(
            capital=1000.0,
            position=None,
            trades=[],
            exit_reasons={},
            c15m=[SimpleNamespace(c=100.0, h=100.5, l=99.5) for _ in range(4)],
            _timestamp_for_idx=lambda idx: f"t{idx}",
            evaluate_range=Mock(side_effect=AssertionError("should not evaluate new opens while overlay position is active")),
        )

        actions = executor._evaluate_latest_with_live_overlay(engine, 1, 1)

        self.assertEqual(actions, [])
        engine.evaluate_range.assert_not_called()

    def test_live_sync_accepts_overlay_runtime_position(self) -> None:
        executor = self.build_executor()
        executor.config.mode = "live"
        executor._fetch_position_state = lambda pos_side, reference_price=None: {  # type: ignore[method-assign]
            "contracts": 1.0 if pos_side == "short" else 0.0,
            "base_amount_btc": 1.0 if pos_side == "short" else 0.0,
            "notional_usdt": 100.0 if pos_side == "short" else 0.0,
            "close_order_algos": [],
            "raw": {},
        }
        executor._shadow_gate_enabled = lambda: False  # type: ignore[method-assign]
        position = OverlayRuntimePosition(
            event_type="stable_reverse_short",
            direction=Direction.BEAR,
            entry_idx=0,
            entry_time="t0",
            exit_idx=None,
            target_rr=2.875,
            max_hold_bars=40,
            allocation=1.0,
            leverage=5.0,
            capital_at_entry=1000.0,
            signal_entry_price=100.0,
            entry_price=100.0,
            sl_price=101.0,
            initial_sl_price=101.0,
            target_price=95.0,
            risk_points=1.0,
            quantity=1.0,
            notional=100.0,
            entry_fee=0.1,
            entry_slippage_cost=0.0,
            stop_reason="stop_loss",
            target_reason="target_rr",
            metadata={},
        )
        executor._save_overlay_runtime_position(position)
        engine = SimpleNamespace(position=None)

        executor._assert_live_state_synced(engine, context="unit_test")

    def test_live_overlay_runtime_does_not_locally_close_when_exchange_brackets_are_enabled(self) -> None:
        executor = self.build_executor()
        executor.config.mode = "live"
        executor.config.enable_exchange_brackets = True
        position = OverlayRuntimePosition(
            event_type="stable_reverse_short",
            direction=Direction.BEAR,
            entry_idx=0,
            entry_time="t0",
            exit_idx=None,
            target_rr=2.875,
            max_hold_bars=40,
            allocation=1.0,
            leverage=5.0,
            capital_at_entry=1000.0,
            signal_entry_price=100.0,
            entry_price=100.0,
            sl_price=101.0,
            initial_sl_price=101.0,
            target_price=95.0,
            risk_points=1.0,
            quantity=50.0,
            notional=5000.0,
            entry_fee=2.5,
            entry_slippage_cost=0.0,
            stop_reason="stop_loss",
            target_reason="target_rr",
            metadata={},
        )
        executor._save_overlay_runtime_position(position)
        engine = SimpleNamespace(
            c15m=[SimpleNamespace(c=100.0, h=102.0, l=94.0)],
            _timestamp_for_idx=lambda idx: f"t{idx}",
        )

        self.assertEqual(executor._overlay_manage_runtime_position(engine, 0), [])

    def test_stable_preempt_keeps_base_paper_position_for_formal_replay(self) -> None:
        executor = self.build_executor()
        executor._sota_overlay_account_lock_pre_open = lambda action, engine, candidate=None: (  # type: ignore[method-assign]
            None if str((action.metadata or {}).get("overlay_event_type") or "sota_long") == "stable_reverse_short" else {"status": "rejected"}
        )
        base_action = StrategyAction(
            type=ActionType.OPEN_LONG,
            timestamp="t5",
            direction=Direction.BULL,
            entry_price=100.0,
            stop_price=99.0,
            target_price=103.0,
            metadata={"index": 5, "overlay_event_type": "sota_long"},
        )
        engine = SimpleNamespace(
            capital=1000.0,
            position=None,
            trades=[],
            exit_reasons={},
            config=SimpleNamespace(use_hfvf_filter=False, enable_dual_pending_state=False),
            waiting_for_pullback=True,
            ob_zone={"bottom": 99.0, "top": 101.0},
            waiting_direction=Direction.BULL,
            waiting_pullback_window=40,
            bos_idx=1,
            precomputed=SimpleNamespace(highs_set=set(), lows_set=set(), bias_4h=[Direction.BULL] * 8),
            mapping=[0] * 8,
            c15m=[SimpleNamespace(c=100.0, h=101.0, l=99.0) for _ in range(8)],
            _timestamp_for_idx=lambda idx: f"t{idx}",
            _apply_regime_switch_for_idx=lambda idx: "static",
            _pending_expired=lambda idx, pending: False,
            _build_pending_pullback=lambda idx, bias: None,
            manage_position=lambda idx: [],
        )

        def _open_action_from_pending(idx: int, pending: PendingPullback) -> StrategyAction:
            engine.position = SimpleNamespace(entry_time="t5", direction=Direction.BULL, entry_idx=5)
            return base_action

        engine._open_action_from_pending = _open_action_from_pending
        executor._overlay_maybe_build_stable_candidate = lambda loaded, idx: StrategyAction(  # type: ignore[method-assign]
            type=ActionType.OPEN_SHORT,
            timestamp="t5",
            direction=Direction.BEAR,
            entry_price=100.0,
            stop_price=101.0,
            target_price=95.0,
            metadata={"index": 5, "entry_idx": 5, "overlay_event_type": "stable_reverse_short", "target_rr": 2.875, "max_hold_bars": 40},
        )
        executor._overlay_maybe_build_smc_candidate = lambda loaded, idx: None  # type: ignore[method-assign]

        actions = executor._evaluate_latest_with_live_overlay(engine, 5, 5)

        self.assertEqual(len(actions), 1)
        self.assertEqual((actions[0].metadata or {}).get("overlay_event_type"), "stable_reverse_short")
        self.assertFalse(engine.waiting_for_pullback)
        self.assertIsNone(engine.ob_zone)
        self.assertIsNone(engine.waiting_direction)
        self.assertIsNotNone(engine.position)

    @patch("bot.okx_executor.scan_events")
    @patch("bot.okx_executor.build_event_scan_args")
    def test_live_smc_candidate_allows_incomplete_tail_scan(self, build_event_scan_args_mock: Mock, scan_events_mock: Mock) -> None:
        executor = self.build_executor()
        idx = 200
        base_ts = 1640995200
        c15m = [
            SimpleNamespace(
                ts=base_ts + bar_idx * 900,
                o=100.0 + bar_idx * 0.1,
                h=101.0 + bar_idx * 0.1,
                l=99.0 + bar_idx * 0.1,
                c=100.0 + bar_idx * 0.1,
                v=1.0,
            )
            for bar_idx in range(idx + 1)
        ]
        build_event_scan_args_mock.return_value = SimpleNamespace(allow_incomplete_tail=False)
        matching_event = SimpleNamespace(
            direction="BEAR",
            retest=SimpleNamespace(
                idx=idx,
                confirmed=True,
                fvg_touched=False,
                ote_touched=True,
                close=118.0,
                timestamp="t200",
            ),
            mss_idx=190,
            sweep_idx=185,
            displacement_body_atr=0.6,
            displacement_range_atr=1.2,
            sweep_distance_pct=0.05,
            sweep_extreme=120.0,
        )

        def _scan(candles: list[SimpleNamespace], args: SimpleNamespace) -> list[SimpleNamespace]:
            self.assertEqual(len(candles), idx + 1)
            self.assertTrue(args.allow_incomplete_tail)
            return [matching_event]

        scan_events_mock.side_effect = _scan

        candidate = executor._overlay_live_smc_candidate(
            c15m=c15m,
            idx=idx,
            c4h=[],
            daily=[],
            h4_highs=[],
            h4_lows=[],
            d1_highs=[],
            d1_lows=[],
            case_args=SimpleNamespace(target_rr=2.0),
            smc_args=SimpleNamespace(
                atr_period=14,
                require_confirmed_retest=True,
                require_fvg_touch=False,
                allow_ote_only=True,
                require_ote_touch=True,
                allowed_time_buckets="all",
                allowed_directions="BEAR",
                max_mss_lag_bars=15,
                min_displacement_body_atr=0.5,
                min_displacement_range_atr=0.0,
                bear_min_sweep_distance_pct=0.03,
                require_h4_bias_align=False,
                require_d1_bias_align=False,
                require_htf_bias_align=False,
                stop_buffer_atr=0.05,
                target_rr=2.0,
            ),
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["entry_idx"], idx)
        self.assertEqual(candidate["entry_time"], "t200")
        self.assertAlmostEqual(candidate["entry_price"], 118.0)
        self.assertGreater(candidate["stop_price"], candidate["entry_price"])
        self.assertLess(candidate["target_price"], candidate["entry_price"])
        self.assertTrue(build_event_scan_args_mock.return_value.allow_incomplete_tail)

    @patch("bot.okx_executor.scan_events")
    @patch("bot.okx_executor.build_event_scan_args")
    def test_live_smc_candidate_keeps_formal_order_for_duplicate_entry_idx(self, build_event_scan_args_mock: Mock, scan_events_mock: Mock) -> None:
        executor = self.build_executor()
        idx = 200
        base_ts = 1640995200
        c15m = [
            SimpleNamespace(
                ts=base_ts + bar_idx * 900,
                o=100.0,
                h=101.0,
                l=99.0,
                c=100.0,
                v=1.0,
            )
            for bar_idx in range(idx + 1)
        ]
        build_event_scan_args_mock.return_value = SimpleNamespace(allow_incomplete_tail=False)
        first_event = SimpleNamespace(
            direction="BEAR",
            retest=SimpleNamespace(
                idx=idx,
                confirmed=True,
                fvg_touched=True,
                ote_touched=True,
                close=100.0,
                timestamp="t200",
            ),
            mss_idx=190,
            sweep_idx=180,
            displacement_body_atr=0.7,
            displacement_range_atr=1.2,
            sweep_distance_pct=0.2,
            sweep_extreme=110.0,
        )
        later_event = SimpleNamespace(
            direction="BEAR",
            retest=SimpleNamespace(
                idx=idx,
                confirmed=True,
                fvg_touched=True,
                ote_touched=True,
                close=100.0,
                timestamp="t200",
            ),
            mss_idx=190,
            sweep_idx=185,
            displacement_body_atr=0.7,
            displacement_range_atr=1.2,
            sweep_distance_pct=0.2,
            sweep_extreme=105.0,
        )
        scan_events_mock.return_value = [first_event, later_event]

        candidate = executor._overlay_live_smc_candidate(
            c15m=c15m,
            idx=idx,
            c4h=[],
            daily=[],
            h4_highs=[],
            h4_lows=[],
            d1_highs=[],
            d1_lows=[],
            case_args=SimpleNamespace(target_rr=2.0),
            smc_args=SimpleNamespace(
                atr_period=14,
                require_confirmed_retest=True,
                require_fvg_touch=False,
                allow_ote_only=True,
                require_ote_touch=True,
                allowed_time_buckets="all",
                allowed_directions="BEAR",
                max_mss_lag_bars=15,
                min_displacement_body_atr=0.5,
                min_displacement_range_atr=0.0,
                bear_min_sweep_distance_pct=0.03,
                require_h4_bias_align=False,
                require_d1_bias_align=False,
                require_htf_bias_align=False,
                stop_buffer_atr=0.0,
                target_rr=2.0,
            ),
        )

        self.assertIsNotNone(candidate)
        self.assertAlmostEqual(candidate["stop_price"], 110.0)
        self.assertAlmostEqual(candidate["target_price"], 80.0)

    def test_formal_shadow_event_can_spawn_stable_candidate(self) -> None:
        executor = self.build_executor()
        engine = SimpleNamespace(
            capital=1000.0,
            c15m=[SimpleNamespace(c=100.0, h=101.0, l=99.0) for _ in range(10)],
            _timestamp_for_idx=lambda idx: f"t{idx}",
        )
        state = executor._load_overlay_formal_state(engine)
        state["last_shadow_event"] = {
            "direction": Direction.BULL,
            "regime_label": "high_growth",
            "risk_mode": "offense",
            "exit_reason": "stop_loss",
            "return": -0.02,
            "failed_breakout_guard_applied": True,
            "feature_adx": 10.0,
            "feature_momentum": 0.0,
            "feature_ema_gap": 0.0,
            "feature_bullish_structure": False,
            "feature_bearish_structure": False,
            "entry_idx": 3,
            "exit_idx": 5,
            "exit_price": 100.0,
            "stop_distance_pct": 1.0,
            "effective_leverage": 8.0,
        }
        executor._save_overlay_formal_state(state)

        action = executor._overlay_maybe_build_stable_candidate(engine, 5)

        self.assertIsNotNone(action)
        self.assertEqual(action.type, ActionType.OPEN_SHORT)
        self.assertEqual((action.metadata or {}).get("overlay_event_type"), "stable_reverse_short")

    def test_formal_sota_active_lock_blocks_later_open_but_allows_same_open_execution(self) -> None:
        executor = self.build_executor()
        engine = SimpleNamespace(capital=1000.0)
        state = executor._load_overlay_formal_state(engine)
        state["active_sota_entry_idx"] = 5
        state["active_sota_entry_time"] = "t5"
        executor._save_overlay_formal_state(state)
        same_open = StrategyAction(
            type=ActionType.OPEN_LONG,
            timestamp="t5",
            direction=Direction.BULL,
            entry_price=100.0,
            stop_price=99.0,
            target_price=103.0,
            metadata={"index": 5, "overlay_event_type": "sota_long", "overlay_formal_fixed": True},
        )
        later_open = StrategyAction(
            type=ActionType.OPEN_SHORT,
            timestamp="t6",
            direction=Direction.BEAR,
            entry_price=100.0,
            stop_price=101.0,
            target_price=95.0,
            metadata={"index": 6, "entry_idx": 6, "overlay_event_type": "smc_short"},
        )

        self.assertFalse(executor._local_position_blocks_new_open(same_open, engine))
        self.assertTrue(executor._local_position_blocks_new_open(later_open, engine))

    def test_executor_config_accepts_live_candidate_aliases(self) -> None:
        config = ExecutorConfig.from_dict(
            {
                "mode": "paper",
                "symbol": "BTC/USDT:USDT",
                "timeframe": "15m",
                "informative_timeframe": "4h",
                "leverage": 10,
                "margin_mode": "isolated",
                "max_open_positions": 1,
                "risk_per_trade": 0.035,
                "state_db_path": "/tmp/runtime.db",
                "enable_live_candidate_arbitration": True,
                "overlay_skip_dynamic_high_leverage": True,
                "stable_reverse_short_live_params": {
                    "enabled": True,
                    "allocation": 0.8,
                    "target_rr": 2.5,
                    "max_hold_bars": 24,
                    "leverage": 4.0,
                    "stop_multiplier": 1.1,
                    "max_short_stop_pct": 1.5,
                    "use_formal_fixed_shadow": False,
                },
                "smc_short_live_params": {
                    "enabled": True,
                    "case": "v2_medium_dispbody05_otherlag4_10x",
                    "allocation": 0.6,
                },
            }
        )

        self.assertTrue(config.enable_live_overlay_strategy)
        self.assertTrue(config.enable_live_candidate_arbitration)
        self.assertTrue(config.overlay_skip_dynamic_high_leverage)
        self.assertAlmostEqual(config.live_overlay_stable_allocation, 0.8)
        self.assertAlmostEqual(config.live_overlay_stable_target_rr, 2.5)
        self.assertEqual(config.live_overlay_stable_max_hold_bars, 24)
        self.assertAlmostEqual(config.live_overlay_stable_leverage, 4.0)
        self.assertAlmostEqual(config.live_overlay_stable_stop_multiplier, 1.1)
        self.assertAlmostEqual(config.live_overlay_stable_max_short_stop_pct, 1.5)
        self.assertFalse(config.live_overlay_use_formal_fixed_shadow)
        self.assertEqual(config.live_overlay_smc_case, "v2_medium_dispbody05_otherlag4_10x")
        self.assertAlmostEqual(config.live_overlay_smc_allocation, 0.6)

    def test_overlay_open_action_keeps_candidate_leverage_metadata(self) -> None:
        executor = self.build_executor()
        engine = SimpleNamespace(
            capital=1000.0,
            c15m=[SimpleNamespace(c=100.0, h=101.0, l=99.0)],
            _timestamp_for_idx=lambda idx: f"t{idx}",
        )

        action, runtime_position = executor._overlay_build_open_short_action(
            engine=engine,
            idx=0,
            event_type="stable_reverse_short",
            signal_entry_price=100.0,
            stop_price=101.0,
            target_price=95.0,
            target_rr=2.875,
            max_hold_bars=40,
            allocation=1.0,
            leverage=5.0,
        )

        self.assertAlmostEqual(runtime_position.leverage, 5.0)
        self.assertAlmostEqual((action.metadata or {})["overlay_leverage"], 5.0)
        self.assertAlmostEqual((action.metadata or {})["leverage"], 5.0)
        self.assertAlmostEqual((action.metadata or {})["margin_usdt"], runtime_position.notional / 5.0)
        restored = executor._overlay_runtime_position_from_action(action)
        self.assertIsNotNone(restored)
        self.assertAlmostEqual(restored.leverage, 5.0)

    def test_dynamic_high_leverage_skips_overlay_short_when_configured(self) -> None:
        executor = self.build_executor()
        executor.config.enable_dynamic_high_leverage_structure = True
        executor.config.overlay_skip_dynamic_high_leverage = True
        action = StrategyAction(
            type=ActionType.OPEN_SHORT,
            timestamp="t5",
            direction=Direction.BEAR,
            entry_price=100.0,
            stop_price=101.0,
            target_price=95.0,
            metadata={"overlay_event_type": "stable_reverse_short", "overlay_leverage": 5.0},
        )

        sizing, decision = executor._dynamic_high_leverage_pre_open(action, {"status": "ok"}, SimpleNamespace())

        self.assertEqual(sizing, {"status": "ok"})
        self.assertIsNone(decision)

    def test_high_leverage_guard_uses_overlay_candidate_leverage(self) -> None:
        executor = self.build_executor()
        executor.config.enable_high_leverage_guard = True
        executor.config.high_leverage_guard_min_leverage = 8.0
        action = StrategyAction(
            type=ActionType.OPEN_SHORT,
            timestamp="t5",
            direction=Direction.BEAR,
            entry_price=100.0,
            stop_price=101.0,
            target_price=95.0,
            metadata={"overlay_event_type": "stable_reverse_short", "overlay_leverage": 5.0},
        )
        sizing = {
            "expected_notional_usdt": 5000.0,
            "available_usdt": 1000.0,
        }

        self.assertIsNone(executor._high_leverage_guard_pre_open(action, sizing))

    def test_live_sync_clears_overlay_runtime_when_exchange_is_flat(self) -> None:
        executor = self.build_executor()
        executor.config.mode = "live"
        executor._shadow_gate_enabled = lambda: False  # type: ignore[method-assign]
        executor._fetch_position_state = lambda pos_side, reference_price=None: {  # type: ignore[method-assign]
            "contracts": 0.0,
            "base_amount_btc": 0.0,
            "notional_usdt": 0.0,
            "close_order_algos": [],
            "raw": {},
        }
        executor._current_live_total_usdt = lambda fallback: float(fallback) - 50.0  # type: ignore[method-assign]
        position = OverlayRuntimePosition(
            event_type="stable_reverse_short",
            direction=Direction.BEAR,
            entry_idx=5,
            entry_time="t5",
            exit_idx=None,
            target_rr=2.875,
            max_hold_bars=40,
            allocation=1.0,
            leverage=5.0,
            capital_at_entry=1000.0,
            signal_entry_price=100.0,
            entry_price=100.0,
            sl_price=101.0,
            initial_sl_price=101.0,
            target_price=95.0,
            risk_points=1.0,
            quantity=50.0,
            notional=5000.0,
            entry_fee=2.5,
            entry_slippage_cost=0.0,
            stop_reason="stop_loss",
            target_reason="target_rr",
            metadata={},
        )
        executor._save_overlay_runtime_position(position)
        engine = SimpleNamespace(
            position=None,
            capital=950.0,
            trades=[],
            exit_reasons={},
        )
        engine.snapshot = lambda: StrategySnapshot(
            capital=engine.capital,
            position=None,
            exit_reasons=engine.exit_reasons,
            trade_count=len(engine.trades),
        )

        executor._assert_live_state_synced(engine, context="unit_test", timestamp="t6", exit_idx=6)

        self.assertIsNone(executor._load_overlay_runtime_position())
        self.assertEqual(len(engine.trades), 1)
        self.assertEqual(engine.trades[0].exit_idx, 6)
        self.assertEqual(engine.trades[0].regime_label, "stable_reverse_short")

    def test_base_actions_do_not_open_new_trade_while_base_position_is_held(self) -> None:
        executor = self.build_executor()
        engine = SimpleNamespace(
            position=SimpleNamespace(entry_idx=5),
            manage_position=Mock(return_value=[]),
            _apply_regime_switch_for_idx=Mock(return_value="static"),
        )
        executor._overlay_maybe_build_base_open_candidate = Mock(  # type: ignore[method-assign]
            side_effect=AssertionError("should not build a new base open while a base position is held")
        )

        base_open, non_open = executor._overlay_base_actions_for_idx(engine, 6)

        self.assertIsNone(base_open)
        self.assertEqual(non_open, [])
        engine.manage_position.assert_called_once_with(6)

    def test_formal_sota_rejects_raw_base_short(self) -> None:
        executor = self.build_executor()
        engine = SimpleNamespace(capital=1000.0)
        raw_short = StrategyAction(
            type=ActionType.OPEN_SHORT,
            timestamp="t5",
            direction=Direction.BEAR,
            entry_price=100.0,
            stop_price=101.0,
            target_price=95.0,
            metadata={"index": 5},
        )

        self.assertIsNone(executor._overlay_formal_sota_action(engine, raw_short))

    def test_formal_state_initializes_without_historical_warmup_by_default(self) -> None:
        executor = self.build_executor()
        engine = SimpleNamespace(
            capital=1000.0,
            c15m=[SimpleNamespace(c=100.0, h=101.0, l=99.0) for _ in range(120)],
            _timestamp_for_idx=lambda idx: f"t{idx}",
        )
        executor._overlay_rebuild_formal_state_from_history = Mock(  # type: ignore[method-assign]
            side_effect=AssertionError("historical formal warmup should be opt-in only")
        )

        executor._overlay_ensure_formal_state_warmup(engine, 100)
        state = executor._load_overlay_formal_state(engine)

        self.assertTrue(state.get("initialized_without_history"))
        self.assertEqual(state.get("initialized_at_idx"), 100)
        self.assertEqual(state.get("initialized_at_time"), "t100")

    @patch("bot.okx_executor.precompute_regime_state")
    def test_load_engine_preloads_regime_caches_when_switching_enabled(self, precompute_mock: Mock) -> None:
        executor = self.build_executor()
        executor.config.enable_regime_switching = True
        executor.config.regime_switcher_thresholds = {"atr_period": 14}
        dates_15m = pd.date_range("2026-01-01", periods=160, freq="15min", tz="UTC")
        dates_4h = pd.date_range("2025-12-01", periods=80, freq="4h", tz="UTC")
        primary = pd.DataFrame(
            {
                "date": dates_15m,
                "open": [100.0 + idx * 0.1 for idx in range(len(dates_15m))],
                "high": [101.0 + idx * 0.1 for idx in range(len(dates_15m))],
                "low": [99.0 + idx * 0.1 for idx in range(len(dates_15m))],
                "close": [100.5 + idx * 0.1 for idx in range(len(dates_15m))],
                "volume": [1.0] * len(dates_15m),
            }
        )
        informative = pd.DataFrame(
            {
                "date": dates_4h,
                "open": [100.0 + idx for idx in range(len(dates_4h))],
                "high": [101.0 + idx for idx in range(len(dates_4h))],
                "low": [99.0 + idx for idx in range(len(dates_4h))],
                "close": [100.5 + idx for idx in range(len(dates_4h))],
                "volume": [1.0] * len(dates_4h),
            }
        )
        executor.market_data.load_pair = Mock(
            return_value=SimpleNamespace(
                primary_candles=primary,
                informative_candles=informative,
            )
        )
        precompute_mock.return_value = (
            {0: "flat", 1: "high_growth"},
            {0: {"adx": 10.0}, 1: {"adx": 35.0}},
        )

        engine, _start_idx = executor.load_engine()

        self.assertEqual(precompute_mock.call_count, 1)
        self.assertEqual(
            precompute_mock.call_args.args[2],
            executor.config.regime_switcher_thresholds,
        )
        self.assertEqual(engine._regime_feature_cache, {0: {"adx": 10.0}, 1: {"adx": 35.0}})
        self.assertEqual(engine._regime_switch_cache[0][0], "flat")
        self.assertEqual(engine._regime_switch_cache[1][0], "high_growth")


if __name__ == "__main__":
    unittest.main()
