from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime as real_datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bot.okx_executor import ExecutorConfig, OkxExecutionEngine
from strategy.scalp_robust_v2_core import Direction


class TelegramCommandTests(unittest.TestCase):
    def _engine(self) -> OkxExecutionEngine:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return OkxExecutionEngine(
            ExecutorConfig(
                mode="paper",
                symbol="BTC/USDT:USDT",
                timeframe="15m",
                informative_timeframe="4h",
                leverage=10,
                margin_mode="isolated",
                max_open_positions=1,
                risk_per_trade=0.035,
                state_db_path=str(Path(tmp.name) / "state.db"),
                telegram_enabled=True,
                telegram_token="test-token",
                telegram_chat_id="123",
            )
        )

    def test_stop_and_start_toggle_open_pause(self) -> None:
        engine = self._engine()

        stop_reply = engine._telegram_command_reply("/stop")
        self.assertIn("已暂停新开仓", stop_reply)
        self.assertTrue(engine._telegram_open_paused())

        start_reply = engine._telegram_command_reply("/start")
        self.assertIn("已恢复开仓", start_reply)
        self.assertFalse(engine._telegram_open_paused())

    def test_help_and_status_reply(self) -> None:
        engine = self._engine()

        help_text = engine._telegram_command_reply("/help")
        self.assertIn("/balance", help_text)
        self.assertIn("/drift", help_text)
        self.assertIn("/strategy", help_text)
        self.assertIn("/ob", help_text)
        status = engine._telegram_command_reply("/status")
        self.assertIn("📡 状态雷达", status)
        self.assertIn("BTC/USDT:USDT", status)

    def test_status_includes_exchange_bracket_prices(self) -> None:
        engine = self._engine()
        engine.config.mode = "live"
        engine._fetch_position_state = lambda pos_side: (  # type: ignore[method-assign]
            {"contracts": 1.0, "notional_usdt": 1000.0}
            if pos_side == "long"
            else {"contracts": 0.0, "notional_usdt": 0.0}
        )
        engine._select_pending_algo_order = lambda pos_side: {  # type: ignore[method-assign]
            "algoId": "algo-123",
            "slTriggerPx": "90000",
            "tpTriggerPx": "110000",
        }

        status = engine._telegram_command_reply("/status")

        self.assertIn("🏛️ 交易所仓位：🟢 long", status)
        self.assertIn("🛡️ 交易所止损：90000.0", status)
        self.assertIn("🎯 交易所止盈：110000.0", status)
        self.assertIn("🔐 保护单ID：algo-123", status)

        table = engine._telegram_command_reply("/status table")
        self.assertIn("🧾 状态面板", table)
        self.assertIn("📦 仓位\n🏛️ 交易所：🟢 long", table)
        self.assertIn("🛡️ 止损：90000.0", table)
        self.assertNotIn("|", table)

    def test_status_and_performance_include_execution_overlay(self) -> None:
        engine = self._engine()
        engine.store.set_value(
            "strategy_snapshot",
            json.dumps(
                {
                    "capital": 9744.64,
                    "position": {
                        "direction": "BULL",
                        "entry_time": "2026-04-29 12:00",
                        "entry_price": 77553.0,
                        "sl_price": 76555.2,
                        "target_price": 83292.7,
                        "capital_at_entry": 12182.16,
                        "notional": 24371.04,
                        "quantity": 0.3143,
                        "execution_effective_leverage": 2.0,
                        "execution_risk_mode": "offense",
                        "execution_leverage_reasons": ["base", "high_growth", "failed_breakout_guard:0/2"],
                        "execution_requested_notional": 54769.97,
                        "execution_target_notional": 24388.64,
                        "execution_guard_diagnostics": {
                            "feature_adx": 13.48,
                            "feature_momentum": -0.0083,
                            "feature_ema_gap": 0.0017,
                        },
                    },
                    "exit_reasons": {},
                    "trade_count": 0,
                },
                ensure_ascii=False,
            ),
        )

        status = engine._telegram_command_reply("/status")
        self.assertIn("⚡ 账户有效杠杆：2.00x", status)
        self.assertIn("🎚️ 执行杠杆：2.00x / offense", status)
        self.assertIn("🧯 压仓原因：基础 + 扩张期 + 防假突破保护 0/2", status)
        self.assertIn("📊 理论/实际仓位：54770U -> 24371U", status)

        performance = engine._telegram_command_reply("/performance")
        self.assertIn("📐 当前执行", performance)
        self.assertIn("⚡ 有效杠杆：2.00x / 执行 2.00x", performance)
        self.assertIn("🧪 质量：ADX 13.5 / 动量 -0.83% / EMA差 0.17%", performance)

    def test_drift_aliases_reply_with_drift_report(self) -> None:
        engine = self._engine()
        engine._build_drift_report_message = lambda: "DRIFT_REPORT"  # type: ignore[method-assign]

        self.assertEqual(engine._telegram_command_reply("/drift@mybot"), "DRIFT_REPORT")
        self.assertEqual(engine._telegram_command_reply("/health"), "DRIFT_REPORT")
        self.assertEqual(engine._telegram_command_reply("/体检"), "DRIFT_REPORT")

    def test_ob_aliases_reply_with_ob_report(self) -> None:
        engine = self._engine()
        engine._build_strategy_status_message = lambda: "STRATEGY_REPORT"  # type: ignore[method-assign]
        engine._build_ob_status_message = lambda: "OB_REPORT"  # type: ignore[method-assign]

        self.assertEqual(engine._telegram_command_reply("/ob"), "STRATEGY_REPORT")
        self.assertEqual(engine._telegram_command_reply("/状态"), "STRATEGY_REPORT")
        self.assertEqual(engine._telegram_command_reply("/ob full"), "OB_REPORT")

    def test_strategy_aliases_reply_with_strategy_report(self) -> None:
        engine = self._engine()
        engine._build_strategy_status_message = lambda: "STRATEGY_REPORT"  # type: ignore[method-assign]

        self.assertEqual(engine._telegram_command_reply("/strategy"), "STRATEGY_REPORT")
        self.assertEqual(engine._telegram_command_reply("/策略"), "STRATEGY_REPORT")
        self.assertEqual(engine._telegram_command_reply("/链路"), "STRATEGY_REPORT")

    def test_strategy_status_message_reflects_current_live_chain(self) -> None:
        engine = self._engine()
        engine.config.enable_live_candidate_arbitration = True
        engine.config.enable_sota_score_gate_live = True
        engine.config.enable_long_score_bucket_sizing_live = True
        engine.config.long_score_bucket_sizing_rules = [
            {
                "name": "nbb_6_11_5_conflict_2p5_cap20",
                "net_eq": 6,
                "bull_eq": 11,
                "bear_eq": 5,
                "conflict_mode": "conflict",
                "leverage_multiplier": 2.5,
                "max_effective_leverage": 20.0,
            }
        ]
        engine.config.enable_smc_short_live = True
        engine.config.smc_case = "v2_medium_dispbody05_otherlag4_10x"
        engine.config.enable_sota_soft_stop_recovery_overlay_live = True
        engine.config.enable_pressure_level_trailing = True
        engine.store.set_value(
            "strategy_snapshot",
            json.dumps(
                {
                    "capital": 10000.0,
                    "position": {
                        "direction": "BULL",
                        "entry_time": "2026-05-07 10:15",
                        "entry_price": 95500.0,
                        "sl_price": 94800.0,
                        "target_price": 98000.0,
                        "notional": 20000.0,
                        "capital_at_entry": 10000.0,
                        "execution_effective_leverage": 2.0,
                        "execution_risk_mode": "offense",
                        "execution_leverage_reasons": ["base", "score_bucket:nbb_6_11_5_conflict_2p5_cap20"],
                        "execution_requested_notional": 15000.0,
                        "execution_target_notional": 20000.0,
                        "candidate_event_type": "sota_long",
                    },
                    "trade_count": 12,
                    "exit_reasons": {},
                },
                ensure_ascii=False,
            ),
        )
        engine.store.append_action(
            "2026-05-07 10:15",
            "LIVE_CANDIDATE_ARBITRATION",
            {
                "decision": "accepted",
                "selected": {
                    "event_type": "sota_long",
                    "timestamp": "2026-05-07 10:15",
                    "direction": "BULL",
                    "entry_price": 95500.0,
                },
                "rejected": [{"event_type": "smc_short"}],
                "score_gate_rejected": [],
            },
        )

        report = engine._build_strategy_status_message()

        self.assertIn("🧭 策略控制台", report)
        self.assertIn("主链: SOTA Long > SMC Short", report)
        self.assertIn("SOTA gate: ON net>=3 / bull>=8 / bear<=6 / any", report)
        self.assertIn("Long bucket: ON 6/11/5冲突 2.5x cap20", report)
        self.assertIn("SOTA soft-stop: AUDIT net>=15 / bear<=0 / lev<=2.0x / buf 1.00R / 4 bars", report)
        self.assertIn("SMC short: ON v2_medium_dispbody05_otherlag4_10x / 10.0x / RR 2.00", report)
        self.assertIn("当前策略仓位: 🟢 多头 / SOTA Long", report)
        self.assertIn("压仓: 基础 + Score桶 6/11/5冲突 2.5x cap20", report)
        self.assertIn("Selected: SOTA Long", report)
        self.assertIn("Rejected: smc_short x1", report)

    def test_daily_profit_excludes_skipped_live_shadow_close(self) -> None:
        engine = self._engine()
        engine.config.mode = "live"
        engine.config.enable_shadow_risk_gate = True
        engine.config.shadow_daily_loss_stop_pct = 1.0
        today = "2026-05-07"
        engine.store.append_action(
            f"{today} 10:15",
            "EXECUTION_SKIPPED",
            {
                "action": {
                    "type": "CLOSE_POSITION",
                    "timestamp": f"{today} 10:15",
                    "direction": "BULL",
                    "reason": "stop_loss",
                    "metadata": {"net_pnl": -805.9},
                },
                "decision": {"status": "shadow_gate_skipped_close"},
            },
        )
        engine.store.append_action(
            f"{today} 12:00",
            "CLOSE_POSITION",
            {
                "type": "CLOSE_POSITION",
                "timestamp": f"{today} 12:00",
                "direction": "BULL",
                "reason": "external_stop_loss",
                "metadata": {"source": "external_flat_sync", "net_pnl": 100.0},
            },
        )

        with patch("bot.okx_executor.datetime") as mocked_datetime:
            mocked_datetime.now.return_value = real_datetime(2026, 5, 7, 15, 0, 0)
            events = engine._realized_pnl_events(daily=True)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["pnl"], 100.0)

    def test_count_excludes_unrealized_shadow_close(self) -> None:
        engine = self._engine()
        engine.config.mode = "live"
        engine.config.enable_shadow_risk_gate = True
        engine.config.shadow_daily_loss_stop_pct = 1.0
        engine.store.append_action(
            "2026-05-07 10:15",
            "CLOSE_POSITION",
            {
                "type": "CLOSE_POSITION",
                "timestamp": "2026-05-07 10:15",
                "direction": "BULL",
                "reason": "stop_loss",
                "metadata": {"net_pnl": -805.9, "ignored_for_realized_pnl": True},
            },
        )
        engine.store.append_action(
            "2026-05-07 12:00",
            "CLOSE_POSITION",
            {
                "type": "CLOSE_POSITION",
                "timestamp": "2026-05-07 12:00",
                "direction": "BULL",
                "reason": "external_stop_loss",
                "metadata": {"source": "external_flat_sync", "net_pnl": 100.0},
            },
        )

        reply = engine._telegram_count_text()

        self.assertIn("🔒 最近记录平仓：1", reply)

    def test_ob_regime_display_labels_compression_bucket_plainly(self) -> None:
        engine = object.__new__(OkxExecutionEngine)

        lines = engine._regime_display_lines(
            "high_growth",
            {
                "adx": 13.5049,
                "momentum": -0.015076,
                "ema_gap": 0.005431,
                "atr_ratio": 0.865,
                "strong_growth_score": 0,
                "compression_growth_score": 4,
            },
        )

        self.assertIn("市场状态: 🟡 压缩蓄势", lines)
        self.assertIn("策略桶: high_growth", lines)
        self.assertIn("动量 -1.51%", "\n".join(lines))

    def test_ob_regime_display_keeps_strong_growth_distinct(self) -> None:
        engine = object.__new__(OkxExecutionEngine)

        label = engine._regime_display_label(
            "high_growth",
            {
                "adx": 36.0,
                "momentum": 0.05,
                "strong_growth_score": 3,
                "compression_growth_score": 1,
            },
        )

        self.assertEqual(label, "🟢 强趋势扩张")

    def test_ob_stronger_bear_break_must_be_lower_than_primary(self) -> None:
        engine = object.__new__(OkxExecutionEngine)
        strategy_engine = SimpleNamespace(
            c15m=[
                SimpleNamespace(c=120.0, h=121.0, l=119.0),
                SimpleNamespace(c=118.0, h=120.0, l=115.0),
                SimpleNamespace(c=94.0, h=96.0, l=90.0),
                SimpleNamespace(c=106.0, h=108.0, l=105.0),
                SimpleNamespace(c=102.0, h=110.0, l=100.0),
                SimpleNamespace(c=104.0, h=109.0, l=101.0),
            ],
            precomputed=SimpleNamespace(highs_15m=[], lows_15m=[1, 2, 3, 4]),
            _timestamp_for_idx=lambda idx: f"t{idx}",
        )

        reference = engine._structure_reference(strategy_engine, 5, Direction.BEAR)

        self.assertEqual(reference["primary"]["break_price"], 100.0)
        self.assertEqual(reference["primary"]["strong_break_price"], 90.0)

    def test_ob_stronger_break_omitted_when_not_more_extreme(self) -> None:
        engine = object.__new__(OkxExecutionEngine)
        strategy_engine = SimpleNamespace(
            c15m=[
                SimpleNamespace(c=120.0, h=121.0, l=119.0),
                SimpleNamespace(c=118.0, h=120.0, l=115.0),
                SimpleNamespace(c=111.0, h=113.0, l=110.0),
                SimpleNamespace(c=106.0, h=108.0, l=105.0),
                SimpleNamespace(c=102.0, h=110.0, l=100.0),
                SimpleNamespace(c=104.0, h=109.0, l=101.0),
            ],
            precomputed=SimpleNamespace(highs_15m=[], lows_15m=[1, 2, 3, 4]),
            _timestamp_for_idx=lambda idx: f"t{idx}",
        )

        reference = engine._structure_reference(strategy_engine, 5, Direction.BEAR)

        self.assertEqual(reference["primary"]["break_price"], 100.0)
        self.assertNotIn("strong_break_price", reference["primary"])

    def test_ob_stronger_bull_break_must_be_higher_than_primary(self) -> None:
        engine = object.__new__(OkxExecutionEngine)
        strategy_engine = SimpleNamespace(
            c15m=[
                SimpleNamespace(c=80.0, h=81.0, l=79.0),
                SimpleNamespace(c=92.0, h=95.0, l=90.0),
                SimpleNamespace(c=108.0, h=110.0, l=106.0),
                SimpleNamespace(c=96.0, h=98.0, l=94.0),
                SimpleNamespace(c=102.0, h=100.0, l=90.0),
                SimpleNamespace(c=99.0, h=101.0, l=97.0),
            ],
            precomputed=SimpleNamespace(highs_15m=[1, 2, 3, 4], lows_15m=[]),
            _timestamp_for_idx=lambda idx: f"t{idx}",
        )

        reference = engine._structure_reference(strategy_engine, 5, Direction.BULL)

        self.assertEqual(reference["primary"]["break_price"], 100.0)
        self.assertEqual(reference["primary"]["strong_break_price"], 110.0)

    def test_ob_stronger_bull_break_omitted_when_not_more_extreme(self) -> None:
        engine = object.__new__(OkxExecutionEngine)
        strategy_engine = SimpleNamespace(
            c15m=[
                SimpleNamespace(c=80.0, h=81.0, l=79.0),
                SimpleNamespace(c=92.0, h=95.0, l=90.0),
                SimpleNamespace(c=96.0, h=98.0, l=94.0),
                SimpleNamespace(c=99.0, h=99.0, l=95.0),
                SimpleNamespace(c=102.0, h=100.0, l=90.0),
                SimpleNamespace(c=99.0, h=101.0, l=97.0),
            ],
            precomputed=SimpleNamespace(highs_15m=[1, 2, 3, 4], lows_15m=[]),
            _timestamp_for_idx=lambda idx: f"t{idx}",
        )

        reference = engine._structure_reference(strategy_engine, 5, Direction.BULL)

        self.assertEqual(reference["primary"]["break_price"], 100.0)
        self.assertNotIn("strong_break_price", reference["primary"])


if __name__ == "__main__":
    unittest.main()
