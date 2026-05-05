from __future__ import annotations

import argparse
import bisect
import json
import random
import time
import uuid
import requests
import pandas as pd
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from types import SimpleNamespace

from bot.market_data import OhlcvRepository
from bot.okx_client import OkxClient, OkxCredentials
from bot.state_store import StateStore
from scripts.smc_short_event_builder import (
    allowed_bucket,
    allowed_direction,
    FORMAL_SMC_CASE_NAMES,
    SMC_CASES,
    atr_series,
    build_event_scan_args,
    daily_candles_from_4h,
    htf_structure_bias,
    scan_events,
    smc_case_namespace,
    smc_strategy_args,
    time_bucket,
)
from strategy.scalp_robust_v2_core import (
    ActionType,
    Direction,
    PendingPullback,
    PositionState,
    ScalpRobustEngine,
    StrategyAction,
    StrategyConfig,
    Trade,
    dataframe_to_candles,
)
from strategy.scalp_robust_v2_core import precompute_swings
from strategy.live_overlay_shared import (
    FIXED_STRUCTURE_PARAMS,
    FixedStructureState,
    fixed_structure_entry_decision,
    fixed_structure_step,
    high_leverage_trade_diagnostics,
    precompute_regime_state,
    quality_snapshot,
    selected_by,
)
from strategy.sota_overlay_state import OverlayCandidate, account_lock_decision, candidate_from_action, leveraged_net_return


@dataclass
class ExecutorConfig:
    mode: str
    symbol: str
    timeframe: str
    informative_timeframe: str
    leverage: int
    margin_mode: str
    max_open_positions: int
    risk_per_trade: float
    state_db_path: str
    strategy_type: str = "scalp_robust_v2"
    position_size_pct: float = 0.35
    fixed_notional_usdt: float | None = None
    pos_side: str = "long"
    data_root: str = "data/okx/futures"
    rr_ratio: float = 4.0
    pullback_window: int = 30
    sl_buffer_pct: float = 1.0
    allow_long: bool = True
    allow_short: bool = True
    regime_filter_1d_ema_period: int | None = None
    enable_directional_regime_switch: bool = False
    long_regime_filter_1d_ema_period: int | None = None
    short_regime_filter_1d_ema_period: int | None = None
    enable_dual_pending_state: bool = False
    enable_regime_layered_exit: bool = False
    enable_short_regime_layered_exit: bool = False
    short_pullback_window: int | None = None
    short_sl_buffer_pct: float | None = None
    short_strong_rr_ratio: float = 5.0
    short_mid_rr_ratio: float = 4.0
    short_weak_rr_ratio: float = 3.0
    enable_target_rr_cap: bool = False
    loose_target_rr_cap: float | None = None
    normal_target_rr_cap: float | None = None
    tight_target_rr_cap: float | None = None
    enable_regime_directional_risk: bool = False
    bull_strong_long_risk_per_trade: float | None = None
    bull_strong_short_risk_per_trade: float | None = None
    bull_weak_long_risk_per_trade: float | None = None
    bull_weak_short_risk_per_trade: float | None = None
    bear_weak_long_risk_per_trade: float | None = None
    bear_weak_short_risk_per_trade: float | None = None
    bear_strong_long_risk_per_trade: float | None = None
    bear_strong_short_risk_per_trade: float | None = None
    bull_weak_long_rr_ratio_override: float | None = None
    bull_weak_long_trail_style_override: str | None = None
    bear_weak_short_rr_ratio_override: float | None = None
    bear_weak_short_trail_style_override: str | None = None
    enable_stage_trailing: bool = True
    enable_atr_trailing: bool = False
    atr_period: int = 14
    atr_activation_rr: float = 2.0
    atr_activation_rr_mode: str = "close"
    atr_loose_multiplier: float = 2.7
    atr_normal_multiplier: float = 2.25
    atr_tight_multiplier: float = 1.8
    enable_time_based_trailing: bool = False
    time_trailing_rr_mode: str = "close"
    T1: int = 15
    T2: int = 40
    T_max: int = 96
    S0_trigger_rr: float = 0.5
    S1_trigger_rr: float = 1.0
    S3_trigger_rr: float = 3.0
    S4_close_rr: float = 0.5
    stage_trigger_rr_mode: str = "close"
    enable_auto_time_based_trailing: bool = False
    auto_tit_mode: str = "health"
    auto_tit_drawdown_pct: float = 12.0
    auto_tit_recent_trades: int = 6
    auto_tit_min_completed_trades: int = 3
    auto_tit_recent_rr_threshold: float = -1.0
    auto_tit_loss_streak: int = 3
    auto_tit_entry_regimes: list[str] | None = None
    auto_tit_regime_labels: list[str] | None = None
    auto_tit_trail_styles: list[str] | None = None
    auto_tit_directions: list[str] | None = None
    auto_tit_adx_min: float | None = None
    auto_tit_adx_max: float | None = None
    auto_tit_momentum_min: float | None = None
    auto_tit_momentum_max: float | None = None
    auto_tit_atr_ratio_min: float | None = None
    auto_tit_atr_ratio_max: float | None = None
    auto_tit_ema_gap_min: float | None = None
    auto_tit_ema_gap_max: float | None = None
    atr_regime_filter: str = "all"
    disable_fixed_target_exit: bool = False
    enable_pressure_level_trailing: bool = False
    pressure_min_rr: float = 1.0
    pressure_rejection_min_rr: float = 1.25
    pressure_lock_rr: float = 0.8
    pressure_atr_multiplier: float = 1.2
    pressure_proximity_pct: float = 0.35
    pressure_round_steps_usdt: list[float] | None = None
    pressure_cluster_lookback_bars: int = 192
    pressure_cluster_bin_usdt: float = 250.0
    pressure_cluster_min_touches: int = 4
    pressure_cluster_min_volume_ratio: float = 1.25
    pressure_swing_lookback_bars: int = 96
    pressure_rejection_wick_ratio: float = 0.45
    pressure_rejection_close_pct: float = 0.12
    pressure_min_bars_held: int = 1
    pressure_take_profit_on_rejection: bool = True
    pressure_enable_target_cap: bool = False
    pressure_target_min_rr: float = 1.5
    pressure_target_buffer_pct: float = 0.05
    pressure_dynamic_target_min_rr_enabled: bool = False
    pressure_dynamic_target_compression_rr: float = 1.0
    pressure_dynamic_target_flat_rr: float = 1.25
    pressure_dynamic_target_breakout_rr: float = 1.5
    pressure_dynamic_target_compression_adx_max: float = 18.0
    pressure_dynamic_target_compression_momentum_abs_pct: float = 1.0
    pressure_dynamic_target_compression_ema_gap_abs_pct: float = 0.25
    pressure_dynamic_target_breakout_adx_min: float = 22.0
    pressure_dynamic_target_breakout_momentum_pct: float = 1.5
    pressure_dynamic_target_breakout_ema_gap_pct: float = 0.35
    pressure_touch_lock_enabled: bool = False
    pressure_touch_lock_min_rr: float = 1.5
    pressure_touch_lock_buffer_pct: float = 0.08
    pressure_touch_lock_atr_multiplier: float = 1.0
    pressure_touch_lock_requires_touch: bool = True
    pressure_regime_labels: list[str] | None = None
    pressure_trail_styles: list[str] | None = None
    enable_shadow_risk_gate: bool = False
    shadow_daily_loss_stop_pct: float = 0.0
    shadow_equity_drawdown_stop_pct: float = 0.0
    shadow_equity_drawdown_cooldown_days: int = 0
    shadow_consecutive_loss_stop: int = 0
    enable_high_leverage_guard: bool = False
    high_leverage_guard_min_leverage: float = 10.0
    high_leverage_min_liquidation_buffer_pct: float = 1.2
    high_leverage_max_stop_distance_pct: float = 2.0
    high_leverage_max_account_effective_leverage: float = 5.0
    high_leverage_maintenance_margin_pct: float = 0.5
    enable_dynamic_high_leverage_structure: bool = False
    dynamic_base_leverage: float = 4.0
    dynamic_high_growth_leverage: float = 7.5
    dynamic_tight_stop_leverage: float = 8.0
    dynamic_recovery_leverage: float = 2.0
    dynamic_drawdown_leverage: float = 2.0
    dynamic_unhealthy_leverage: float = 2.0
    dynamic_defense_leverage: float = 2.0
    dynamic_tight_stop_pct: float = 1.25
    dynamic_max_stop_distance_pct: float = 1.5
    dynamic_high_growth_max_stop_distance_pct: float = 2.0
    dynamic_defense_max_stop_distance_pct: float = 1.5
    dynamic_defense_structure_max_stop_distance_pct: float = 1.9
    dynamic_max_effective_leverage: float = 8.0
    dynamic_loss_streak_threshold: int = 3
    dynamic_win_streak_threshold: int = 2
    dynamic_drawdown_threshold_pct: float = 20.0
    dynamic_health_lookback_trades: int = 6
    dynamic_health_min_unit_return_pct: float = 0.0
    dynamic_health_min_win_rate_pct: float = 25.0
    dynamic_state_lookback_trades: int = 8
    dynamic_defense_enter_unit_return_pct: float = -2.0
    dynamic_defense_enter_win_rate_pct: float = 20.0
    dynamic_offense_enter_unit_return_pct: float = -0.5
    dynamic_offense_enter_win_rate_pct: float = 40.0
    dynamic_reattack_lookback_trades: int = 2
    dynamic_reattack_unit_return_pct: float = 0.5
    dynamic_reattack_win_rate_pct: float = 33.0
    dynamic_reattack_signal_mode: str = "high_growth_or_tight_or_structure"
    dynamic_min_liquidation_buffer_pct: float = 1.2
    dynamic_failed_breakout_guard_enabled: bool = False
    dynamic_failed_breakout_guard_leverage: float = 2.0
    dynamic_failed_breakout_guard_min_leverage: float = 7.5
    dynamic_failed_breakout_guard_min_quality_score: int = 2
    dynamic_failed_breakout_guard_min_momentum_pct: float = 6.0
    dynamic_failed_breakout_guard_min_ema_gap_pct: float = 2.0
    dynamic_failed_breakout_guard_min_adx: float = 35.0
    dynamic_failed_breakout_guard_regime_labels: list[str] | None = None
    dynamic_failed_breakout_guard_risk_modes: list[str] | None = None
    dynamic_failed_breakout_guard_directions: list[str] | None = None
    enable_regime_switching: bool = False
    regime_switcher_thresholds: dict[str, Any] | None = None
    regime_switcher_hg_overrides: dict[str, Any] | None = None
    regime_switcher_normal_overrides: dict[str, Any] | None = None
    regime_switcher_flat_overrides: dict[str, Any] | None = None
    taker_fee_rate: float = 0.0005
    slippage_bps: float = 2.0
    replay_sync_entry_to_signal_price: bool = False
    enable_exchange_brackets: bool = False
    exchange_trigger_price_type: str = "mark"
    enable_manual_position_sync: bool = True
    manual_position_sync_size_tolerance_ratio: float = 0.02
    manual_position_sync_entry_price_tolerance_bps: float = 10.0
    telegram_enabled: bool = False
    telegram_token: str | None = None
    telegram_chat_id: str | None = None
    telegram_command_enabled: bool = True
    telegram_poll_interval_seconds: int = 30
    telegram_ob_status_enabled: bool = True
    telegram_ob_status_interval_minutes: int = 60
    telegram_drift_report_enabled: bool = True
    telegram_drift_report_interval_hours: int = 24
    telegram_drift_window_days: int = 30
    telegram_drift_recent_trades: int = 20
    telegram_drift_baseline_path: str = "config/live_drift_baseline.high_leverage.json"
    enable_live_candidate_arbitration: bool = False
    enable_live_overlay_strategy: bool = False
    enable_stable_reverse_short_live: bool = True
    enable_smc_short_live: bool = True
    overlay_skip_dynamic_high_leverage: bool = False
    live_candidate_priority: list[str] | None = None
    stable_selector: str = "guarded_weak_loss"
    stable_max_quality_score: int = 1
    stable_target_rr: float | None = None
    stable_max_hold_bars: int | None = None
    stable_leverage: float | None = None
    stable_position_size_pct: float = 1.0
    stable_stop_multiplier: float | None = None
    stable_max_short_stop_pct: float | None = None
    stable_trail_style: str = "tight"
    smc_case: str | None = None
    smc_target_rr: float = 2.0
    smc_max_hold_bars: int = 40
    smc_trail_style: str = "tight"
    smc_leverage: float = 10.0
    smc_position_size_pct: float = 1.0
    smc_min_liq_buffer_pct: float = 1.2
    smc_maintenance_margin_pct: float = 0.5
    stable_reverse_short_live_params: dict[str, Any] | None = None
    smc_short_live_params: dict[str, Any] | None = None
    live_overlay_smc_case: str = "v2_medium_dispbody05_otherlag4_10x"
    live_overlay_smc_allocation: float = 1.0
    live_overlay_stable_allocation: float = 1.0
    live_overlay_stable_target_rr: float = 2.875
    live_overlay_stable_max_hold_bars: int = 40
    live_overlay_stable_leverage: float = 5.0
    live_overlay_stable_stop_multiplier: float = 1.0
    live_overlay_stable_max_short_stop_pct: float = 1.75
    live_overlay_use_formal_fixed_shadow: bool = True
    live_overlay_rebuild_formal_state_from_history: bool = False
    proxy: str | None = None
    api_key: str | None = None
    api_secret: str | None = None
    api_passphrase: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExecutorConfig":
        normalized_payload = dict(payload)
        if normalized_payload.get("enable_live_candidate_arbitration") is not None:
            normalized_payload["enable_live_overlay_strategy"] = bool(
                normalized_payload.get("enable_live_overlay_strategy", False)
                or normalized_payload["enable_live_candidate_arbitration"]
            )

        stable_params = normalized_payload.get("stable_reverse_short_live_params")
        if isinstance(stable_params, dict):
            normalized_payload["enable_stable_reverse_short_live"] = bool(
                stable_params.get(
                    "enabled",
                    normalized_payload.get("enable_stable_reverse_short_live", True),
                )
            )
            for source_key, target_key in (
                ("allocation", "live_overlay_stable_allocation"),
                ("target_rr", "live_overlay_stable_target_rr"),
                ("max_hold_bars", "live_overlay_stable_max_hold_bars"),
                ("leverage", "live_overlay_stable_leverage"),
                ("stop_multiplier", "live_overlay_stable_stop_multiplier"),
                ("max_short_stop_pct", "live_overlay_stable_max_short_stop_pct"),
                ("use_formal_fixed_shadow", "live_overlay_use_formal_fixed_shadow"),
                ("rebuild_formal_state_from_history", "live_overlay_rebuild_formal_state_from_history"),
            ):
                if source_key in stable_params and target_key not in normalized_payload:
                    normalized_payload[target_key] = stable_params[source_key]
        else:
            legacy_stable_map = (
                ("stable_target_rr", "live_overlay_stable_target_rr"),
                ("stable_max_hold_bars", "live_overlay_stable_max_hold_bars"),
                ("stable_leverage", "live_overlay_stable_leverage"),
                ("stable_stop_multiplier", "live_overlay_stable_stop_multiplier"),
                ("stable_max_short_stop_pct", "live_overlay_stable_max_short_stop_pct"),
            )
            for source_key, target_key in legacy_stable_map:
                if source_key in normalized_payload and target_key not in normalized_payload:
                    normalized_payload[target_key] = normalized_payload[source_key]
            if "stable_position_size_pct" in normalized_payload and "live_overlay_stable_allocation" not in normalized_payload:
                normalized_payload["live_overlay_stable_allocation"] = normalized_payload["stable_position_size_pct"]

        smc_params = normalized_payload.get("smc_short_live_params")
        if isinstance(smc_params, dict):
            normalized_payload["enable_smc_short_live"] = bool(
                smc_params.get(
                    "enabled",
                    normalized_payload.get("enable_smc_short_live", True),
                )
            )
            for source_key, target_key in (
                ("case", "live_overlay_smc_case"),
                ("allocation", "live_overlay_smc_allocation"),
            ):
                if source_key in smc_params and target_key not in normalized_payload:
                    normalized_payload[target_key] = smc_params[source_key]
        else:
            if "smc_case" in normalized_payload and "live_overlay_smc_case" not in normalized_payload:
                normalized_payload["live_overlay_smc_case"] = normalized_payload["smc_case"]

        filtered_payload = {
            key: value
            for key, value in normalized_payload.items()
            if key in cls.__dataclass_fields__
        }
        return cls(**filtered_payload)

    def to_scalp_strategy_config(self) -> StrategyConfig:
        return StrategyConfig(
            leverage=float(self.leverage),
            risk_per_trade=self.risk_per_trade,
            position_size_pct=self.position_size_pct,
            fixed_notional_usdt=self.fixed_notional_usdt,
            rr_ratio=self.rr_ratio,
            pullback_window=self.pullback_window,
            sl_buffer_pct=self.sl_buffer_pct,
            allow_long=self.allow_long,
            allow_short=self.allow_short,
            regime_filter_1d_ema_period=self.regime_filter_1d_ema_period,
            enable_directional_regime_switch=self.enable_directional_regime_switch,
            long_regime_filter_1d_ema_period=self.long_regime_filter_1d_ema_period,
            short_regime_filter_1d_ema_period=self.short_regime_filter_1d_ema_period,
            enable_dual_pending_state=self.enable_dual_pending_state,
            enable_regime_layered_exit=self.enable_regime_layered_exit,
            enable_short_regime_layered_exit=self.enable_short_regime_layered_exit,
            short_pullback_window=self.short_pullback_window,
            short_sl_buffer_pct=self.short_sl_buffer_pct,
            short_strong_rr_ratio=self.short_strong_rr_ratio,
            short_mid_rr_ratio=self.short_mid_rr_ratio,
            short_weak_rr_ratio=self.short_weak_rr_ratio,
            enable_target_rr_cap=self.enable_target_rr_cap,
            loose_target_rr_cap=self.loose_target_rr_cap,
            normal_target_rr_cap=self.normal_target_rr_cap,
            tight_target_rr_cap=self.tight_target_rr_cap,
            enable_regime_directional_risk=self.enable_regime_directional_risk,
            bull_strong_long_risk_per_trade=self.bull_strong_long_risk_per_trade,
            bull_strong_short_risk_per_trade=self.bull_strong_short_risk_per_trade,
            bull_weak_long_risk_per_trade=self.bull_weak_long_risk_per_trade,
            bull_weak_short_risk_per_trade=self.bull_weak_short_risk_per_trade,
            bear_weak_long_risk_per_trade=self.bear_weak_long_risk_per_trade,
            bear_weak_short_risk_per_trade=self.bear_weak_short_risk_per_trade,
            bear_strong_long_risk_per_trade=self.bear_strong_long_risk_per_trade,
            bear_strong_short_risk_per_trade=self.bear_strong_short_risk_per_trade,
            bull_weak_long_rr_ratio_override=self.bull_weak_long_rr_ratio_override,
            bull_weak_long_trail_style_override=self.bull_weak_long_trail_style_override,
            bear_weak_short_rr_ratio_override=self.bear_weak_short_rr_ratio_override,
            bear_weak_short_trail_style_override=self.bear_weak_short_trail_style_override,
            enable_stage_trailing=self.enable_stage_trailing,
            enable_atr_trailing=self.enable_atr_trailing,
            atr_period=self.atr_period,
            atr_activation_rr=self.atr_activation_rr,
            atr_activation_rr_mode=self.atr_activation_rr_mode,
            atr_loose_multiplier=self.atr_loose_multiplier,
            atr_normal_multiplier=self.atr_normal_multiplier,
            atr_tight_multiplier=self.atr_tight_multiplier,
            enable_time_based_trailing=self.enable_time_based_trailing,
            time_trailing_rr_mode=self.time_trailing_rr_mode,
            T1=self.T1,
            T2=self.T2,
            T_max=self.T_max,
            S0_trigger_rr=self.S0_trigger_rr,
            S1_trigger_rr=self.S1_trigger_rr,
            S3_trigger_rr=self.S3_trigger_rr,
            S4_close_rr=self.S4_close_rr,
            stage_trigger_rr_mode=self.stage_trigger_rr_mode,
            enable_auto_time_based_trailing=self.enable_auto_time_based_trailing,
            auto_tit_mode=self.auto_tit_mode,
            auto_tit_drawdown_pct=self.auto_tit_drawdown_pct,
            auto_tit_recent_trades=self.auto_tit_recent_trades,
            auto_tit_min_completed_trades=self.auto_tit_min_completed_trades,
            auto_tit_recent_rr_threshold=self.auto_tit_recent_rr_threshold,
            auto_tit_loss_streak=self.auto_tit_loss_streak,
            auto_tit_entry_regimes=self.auto_tit_entry_regimes,
            auto_tit_regime_labels=self.auto_tit_regime_labels,
            auto_tit_trail_styles=self.auto_tit_trail_styles,
            auto_tit_directions=self.auto_tit_directions,
            auto_tit_adx_min=self.auto_tit_adx_min,
            auto_tit_adx_max=self.auto_tit_adx_max,
            auto_tit_momentum_min=self.auto_tit_momentum_min,
            auto_tit_momentum_max=self.auto_tit_momentum_max,
            auto_tit_atr_ratio_min=self.auto_tit_atr_ratio_min,
            auto_tit_atr_ratio_max=self.auto_tit_atr_ratio_max,
            auto_tit_ema_gap_min=self.auto_tit_ema_gap_min,
            auto_tit_ema_gap_max=self.auto_tit_ema_gap_max,
            atr_regime_filter=self.atr_regime_filter,
            disable_fixed_target_exit=self.disable_fixed_target_exit,
            enable_pressure_level_trailing=self.enable_pressure_level_trailing,
            pressure_min_rr=self.pressure_min_rr,
            pressure_rejection_min_rr=self.pressure_rejection_min_rr,
            pressure_lock_rr=self.pressure_lock_rr,
            pressure_atr_multiplier=self.pressure_atr_multiplier,
            pressure_proximity_pct=self.pressure_proximity_pct,
            pressure_round_steps_usdt=self.pressure_round_steps_usdt,
            pressure_cluster_lookback_bars=self.pressure_cluster_lookback_bars,
            pressure_cluster_bin_usdt=self.pressure_cluster_bin_usdt,
            pressure_cluster_min_touches=self.pressure_cluster_min_touches,
            pressure_cluster_min_volume_ratio=self.pressure_cluster_min_volume_ratio,
            pressure_swing_lookback_bars=self.pressure_swing_lookback_bars,
            pressure_rejection_wick_ratio=self.pressure_rejection_wick_ratio,
            pressure_rejection_close_pct=self.pressure_rejection_close_pct,
            pressure_min_bars_held=self.pressure_min_bars_held,
            pressure_take_profit_on_rejection=self.pressure_take_profit_on_rejection,
            pressure_enable_target_cap=self.pressure_enable_target_cap,
            pressure_target_min_rr=self.pressure_target_min_rr,
            pressure_target_buffer_pct=self.pressure_target_buffer_pct,
            pressure_dynamic_target_min_rr_enabled=self.pressure_dynamic_target_min_rr_enabled,
            pressure_dynamic_target_compression_rr=self.pressure_dynamic_target_compression_rr,
            pressure_dynamic_target_flat_rr=self.pressure_dynamic_target_flat_rr,
            pressure_dynamic_target_breakout_rr=self.pressure_dynamic_target_breakout_rr,
            pressure_dynamic_target_compression_adx_max=self.pressure_dynamic_target_compression_adx_max,
            pressure_dynamic_target_compression_momentum_abs_pct=self.pressure_dynamic_target_compression_momentum_abs_pct,
            pressure_dynamic_target_compression_ema_gap_abs_pct=self.pressure_dynamic_target_compression_ema_gap_abs_pct,
            pressure_dynamic_target_breakout_adx_min=self.pressure_dynamic_target_breakout_adx_min,
            pressure_dynamic_target_breakout_momentum_pct=self.pressure_dynamic_target_breakout_momentum_pct,
            pressure_dynamic_target_breakout_ema_gap_pct=self.pressure_dynamic_target_breakout_ema_gap_pct,
            pressure_touch_lock_enabled=self.pressure_touch_lock_enabled,
            pressure_touch_lock_min_rr=self.pressure_touch_lock_min_rr,
            pressure_touch_lock_buffer_pct=self.pressure_touch_lock_buffer_pct,
            pressure_touch_lock_atr_multiplier=self.pressure_touch_lock_atr_multiplier,
            pressure_touch_lock_requires_touch=self.pressure_touch_lock_requires_touch,
            pressure_regime_labels=self.pressure_regime_labels,
            pressure_trail_styles=self.pressure_trail_styles,
            enable_regime_switching=self.enable_regime_switching,
            regime_switcher_thresholds=self.regime_switcher_thresholds,
            regime_switcher_hg_overrides=self.regime_switcher_hg_overrides,
            regime_switcher_normal_overrides=self.regime_switcher_normal_overrides,
            regime_switcher_flat_overrides=self.regime_switcher_flat_overrides,
            taker_fee_rate=self.taker_fee_rate,
            slippage_bps=self.slippage_bps,
            replay_sync_entry_to_signal_price=self.replay_sync_entry_to_signal_price,
        )


@dataclass
class OverlayRuntimePosition:
    event_type: str
    direction: str
    entry_idx: int
    entry_time: str
    exit_idx: int | None
    target_rr: float | None
    max_hold_bars: int | None
    allocation: float
    leverage: float
    capital_at_entry: float
    signal_entry_price: float
    entry_price: float
    sl_price: float
    initial_sl_price: float
    target_price: float
    risk_points: float
    quantity: float
    notional: float
    entry_fee: float
    entry_slippage_cost: float
    stop_reason: str | None = None
    target_reason: str | None = None
    max_open_positions: int = 1
    stop_buffer_atr: float | None = None
    smc_case: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class OverlayRuntimeDecision:
    base_action: StrategyAction | None
    candidate_actions: list[StrategyAction]
    execution_actions: list[StrategyAction]
    overlay_decisions: list[dict[str, Any]]

class OkxExecutionEngine:
    def __init__(self, config: ExecutorConfig, config_path: str | Path | None = None):
        self.config = config
        self.config_path = Path(config_path).resolve() if config_path else None
        credentials = None
        if config.api_key and config.api_secret and config.api_passphrase:
            credentials = OkxCredentials(
                api_key=config.api_key,
                api_secret=config.api_secret,
                api_passphrase=config.api_passphrase,
            )
        self.client = OkxClient(credentials, trading_mode=config.mode, proxy=config.proxy)
        self.store = StateStore(config.state_db_path)
        self.market_data = OhlcvRepository(config.data_root)
        self._markets_cache: dict[str, Any] | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "OkxExecutionEngine":
        config_path = Path(path).resolve()
        payload = json.loads(config_path.read_text())
        return cls(ExecutorConfig.from_dict(payload), config_path=config_path)

    def check_safety(self) -> None:
        if self.config.symbol != "BTC/USDT:USDT":
            raise ValueError("First version only allows BTC/USDT:USDT")
        if self.config.max_open_positions != 1:
            raise ValueError("First version only supports exactly one open position")
        if self.config.mode not in {"paper", "live"}:
            raise ValueError("mode must be paper or live")
        if self.config.mode == "live":
            missing = [
                name
                for name, value in {
                    "api_key": self.config.api_key,
                    "api_secret": self.config.api_secret,
                    "api_passphrase": self.config.api_passphrase,
                }.items()
                if not value
            ]
            if missing:
                raise ValueError(f"live mode missing credentials: {', '.join(missing)}")

    def _telegram_proxies(self) -> dict[str, str] | None:
        if not self.config.proxy:
            return None
        return {"http": self.config.proxy, "https": self.config.proxy}

    def _send_telegram(self, message: str, chat_id: str | None = None) -> bool:
        if not self.config.telegram_enabled:
            return False
        target_chat_id = chat_id or self.config.telegram_chat_id
        if not self.config.telegram_token or not target_chat_id:
            return False
        url = f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage"
        try:
            response = requests.post(
                url,
                json={"chat_id": target_chat_id, "text": message},
                timeout=10,
                proxies=self._telegram_proxies(),
            )
            response.raise_for_status()
            return True
        except Exception:
            return False

    def _send_telegram_reply(self, message: str, chat_id: str | int | None = None) -> None:
        if not self.config.telegram_enabled:
            return
        if not self.config.telegram_token:
            return
        target_chat_id = str(chat_id or self.config.telegram_chat_id or "")
        if not target_chat_id:
            return
        url = f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage"
        try:
            requests.post(
                url,
                json={
                    "chat_id": target_chat_id,
                    "text": message,
                    "reply_markup": self._telegram_reply_markup(),
                },
                timeout=10,
                proxies=self._telegram_proxies(),
            )
        except Exception:
            pass

    def _telegram_reply_markup(self) -> dict[str, Any]:
        return {
            "keyboard": [
                ["/daily", "/profit", "/balance"],
                ["/status", "/ob", "/drift"],
                ["/status table", "/performance"],
                ["/count", "/start", "/stop", "/help"],
            ],
            "resize_keyboard": True,
            "one_time_keyboard": False,
        }

    def _configure_telegram_commands(self) -> None:
        if not self.config.telegram_enabled or not self.config.telegram_token:
            return
        url = f"https://api.telegram.org/bot{self.config.telegram_token}/setMyCommands"
        commands = [
            {"command": "daily", "description": "今日收益"},
            {"command": "profit", "description": "累计收益"},
            {"command": "balance", "description": "账户余额"},
            {"command": "status", "description": "机器人状态"},
            {"command": "ob", "description": "OB开仓条件"},
            {"command": "drift", "description": "实盘体检"},
            {"command": "performance", "description": "策略表现"},
            {"command": "count", "description": "交易次数"},
            {"command": "start", "description": "恢复开仓"},
            {"command": "stop", "description": "暂停开仓"},
            {"command": "help", "description": "命令帮助"},
        ]
        try:
            requests.post(url, json={"commands": commands}, timeout=10, proxies=self._telegram_proxies())
        except Exception:
            pass

    def _telegram_get_updates(self) -> list[dict[str, Any]]:
        if not self.config.telegram_enabled or not self.config.telegram_command_enabled or not self.config.telegram_token:
            return []
        offset_raw = self.store.get_value("telegram_update_offset")
        try:
            offset = int(offset_raw) if offset_raw else None
        except ValueError:
            offset = None
        url = f"https://api.telegram.org/bot{self.config.telegram_token}/getUpdates"
        params: dict[str, Any] = {"timeout": 0, "limit": 20, "allowed_updates": json.dumps(["message"])}
        if offset is not None:
            params["offset"] = offset
        response = requests.get(url, params=params, timeout=10, proxies=self._telegram_proxies())
        payload = response.json()
        if not payload.get("ok"):
            return []
        updates = payload.get("result")
        if not isinstance(updates, list):
            return []
        if updates:
            max_update_id = max(int(update.get("update_id", 0)) for update in updates if isinstance(update, dict))
            self.store.set_value("telegram_update_offset", str(max_update_id + 1))
        return [update for update in updates if isinstance(update, dict)]

    def _handle_telegram_commands(self) -> None:
        if not self.config.telegram_enabled or not self.config.telegram_token:
            return
        try:
            updates = self._telegram_get_updates()
        except Exception as exc:
            self.store.append_action(
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "TELEGRAM_ERROR",
                {"error": str(exc)},
            )
            return
        for update in updates:
            message = update.get("message")
            if not isinstance(message, dict):
                continue
            chat = message.get("chat")
            chat_id = chat.get("id") if isinstance(chat, dict) else None
            if str(chat_id or "") != str(self.config.telegram_chat_id or ""):
                continue
            text = str(message.get("text") or "").strip()
            if not text:
                continue
            reply = self._telegram_command_reply(text)
            self.store.append_action(
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "TELEGRAM_COMMAND",
                {"text": text, "chat_id": chat_id, "reply": reply},
            )
            self._send_telegram_reply(reply, chat_id)

    def _telegram_command_reply(self, text: str) -> str:
        raw = text.strip()
        command = raw.split("@", 1)[0].strip().lower()
        command_name = raw.split(maxsplit=1)[0].split("@", 1)[0].strip().lower() if raw else ""
        if command_name in {"/drift", "/health", "/体检"}:
            return self._build_drift_report_message()
        if command_name in {"/ob", "/状态"}:
            return self._build_ob_status_message()
        if command == "/help":
            return self._telegram_help_text()
        if command == "/start":
            self.store.set_value("telegram_open_paused", "false")
            return "\n".join([self._telegram_title("🟢", "Bot 控制台"), "🚀 状态：已恢复开仓", self._telegram_time_line()])
        if command == "/stop":
            self.store.set_value("telegram_open_paused", "true")
            return "\n".join(
                [
                    self._telegram_title("🛑", "Bot 控制台"),
                    "🚧 状态：已暂停新开仓",
                    "🛡️ 说明：不会强平已有仓位",
                    self._telegram_time_line(),
                ]
            )
        if command == "/status" or command == "/status table":
            return self._telegram_status_text(table=command == "/status table")
        if command == "/balance":
            return self._telegram_balance_text()
        if command == "/daily":
            return self._telegram_profit_text(daily=True)
        if command == "/profit":
            return self._telegram_profit_text(daily=False)
        if command == "/performance":
            return self._telegram_performance_text()
        if command == "/count":
            return self._telegram_count_text()
        return self._telegram_help_text()

    def _telegram_help_text(self) -> str:
        return "\n".join(
            [
                self._telegram_title("🧭", "指令面板"),
                "💰 /daily 今日已实现收益",
                "📈 /profit 累计已实现收益",
                "🏦 /balance 账户余额",
                "📡 /status 运行和持仓状态",
                "🧭 /ob OB开仓雷达",
                "🩺 /drift 实盘体检",
                "🧾 /status table 面板版状态",
                "🚀 /performance 策略表现",
                "🔢 /count 交易次数",
                "🟢 /start 恢复开仓",
                "🛑 /stop 暂停新开仓",
            ]
        )

    def _local_time_text(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _telegram_mood(self, value: float) -> str:
        if value > 0:
            return "profit"
        if value < 0:
            return "loss"
        return "neutral"

    def _telegram_random_icon(self, mood: str) -> str:
        pools = {
            "profit": ["🚀", "💰", "🔥", "🤑", "📈", "⚡", "🏆", "💎"],
            "loss": ["🛡️", "⚠️", "🥶", "📉", "🧯", "🔻", "🚧", "🛑"],
            "neutral": ["👀", "🤖", "🧭", "⚪", "🕯️", "📡", "🧊", "🔎"],
        }
        return random.choice(pools.get(mood, pools["neutral"]))

    def _telegram_command_mood(self, command: str) -> str:
        if command == "/daily":
            total = sum(float(event["pnl"]) for event in self._realized_pnl_events(daily=True))
            return self._telegram_mood(total)
        if command == "/profit":
            total = sum(float(event["pnl"]) for event in self._realized_pnl_events(daily=False))
            return self._telegram_mood(total)
        if command == "/performance":
            dyn = self._load_dynamic_high_leverage_state() if self._dynamic_high_leverage_enabled() else {}
            unit_returns = dyn.get("unit_returns") if isinstance(dyn.get("unit_returns"), list) else []
            if not unit_returns:
                return "neutral"
            recent = self._dynamic_recent_stats(unit_returns, min(len(unit_returns), int(self.config.dynamic_state_lookback_trades)))
            return self._telegram_mood(float(recent.get("unit_return_pct", 0.0) or 0.0))
        return "neutral"

    def _telegram_title(self, icon: str, title: str) -> str:
        return f"{icon} {title}\n━━━━━━━━━━━━"

    def _telegram_time_line(self) -> str:
        return f"⏱ 时间：{self._local_time_text()}"

    def _open_status_text(self, paused: bool) -> str:
        return "🔴 暂停" if paused else "🟢 允许"

    def _side_status_text(self, side: str) -> str:
        if side == "long":
            return "🟢 long"
        if side == "short":
            return "🔴 short"
        if side == "flat":
            return "⚪ flat"
        return str(side or "-")

    def _overlay_event_label(self, event_type: Any) -> str:
        event = str(event_type or "").lower()
        labels = {
            "sota": "SOTA做多",
            "sota_long": "SOTA做多",
            "main_sota": "SOTA做多",
            "scalp_robust_v2": "SOTA做多",
            "sota_short": "SOTA做空",
            "stable": "Stable反手空",
            "stable_reverse_short": "Stable反手空",
            "smc": "SMC短空",
            "smc_short": "SMC短空",
        }
        return labels.get(event, str(event_type or "-"))

    def _overlay_reason_label(self, reason: Any) -> str:
        labels = {
            "priority_available": "可执行",
            "position_lock_open": "单仓锁",
            "local_position_open": "本地已有仓位",
            "account_position_open": "交易所已有仓位",
            "account_state_unavailable": "账户状态不可用",
        }
        return labels.get(str(reason or ""), str(reason or "-"))

    def _overlay_candidate_text(self, candidate: Any | None) -> str:
        if candidate is None:
            return "空闲"
        label = self._overlay_event_label(getattr(candidate, "event_type", None))
        direction = self._direction_label(getattr(candidate, "direction", None))
        exit_time = getattr(candidate, "exit_time", None)
        if exit_time:
            return f"🔒 {label} / {direction} / 至 {exit_time}"
        return f"🔒 {label} / {direction}"

    def _latest_overlay_decision(self) -> dict[str, Any] | None:
        for item in self.store.recent_actions(200):
            if item.get("action_type") != "SOTA_OVERLAY_LOCK":
                continue
            payload = item.get("payload")
            if isinstance(payload, dict):
                return payload
        return None

    def _overlay_decision_text(self, decision: dict[str, Any] | None) -> str:
        if not decision:
            return "暂无"
        event_label = self._overlay_event_label(decision.get("event_type"))
        status = str(decision.get("decision") or "")
        reason = self._overlay_reason_label(decision.get("reason"))
        paper_tag = str(decision.get("paper_tag") or "")
        blocking = decision.get("blocking_event_type")
        blocking_label = self._overlay_event_label(blocking) if blocking else None
        if paper_tag == "stable_preempted_sota":
            return f"Stable抢占SOTA：{blocking_label or 'Stable'} 挡住 {event_label}"
        if status == "accepted":
            return f"✅ 接受 {event_label}"
        if status == "rejected":
            if blocking_label:
                return f"⛔ {event_label} 被 {blocking_label} 拦截：{reason}"
            return f"⛔ {event_label}：{reason}"
        return f"{event_label}：{reason}"

    def _overlay_formal_state_text(self) -> str:
        if not self._live_overlay_enabled():
            return "关闭"
        if not self._overlay_formal_fixed_shadow_enabled():
            return "legacy"
        raw = self.store.get_value("live_overlay_formal_state")
        if not raw:
            return "未初始化"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return "状态异常"
        if not isinstance(payload, dict):
            return "状态异常"
        if bool(payload.get("initialized_without_history")):
            text = "冷启动(无历史)"
        elif payload.get("warmed_until_time"):
            text = f"历史重建至 {payload.get('warmed_until_time')}"
        else:
            text = "持久化"
        active_entry = payload.get("active_sota_entry_idx")
        if active_entry is not None:
            text = f"{text} / SOTA#{active_entry}"
        return text

    def _overlay_status_rows(self) -> list[tuple[str, str]]:
        return [
            ("Overlay锁仓", self._overlay_candidate_text(self._load_sota_overlay_open_candidate())),
            ("Formal状态", self._overlay_formal_state_text()),
            ("最近Overlay", self._overlay_decision_text(self._latest_overlay_decision())),
        ]

    def _overlay_compact_line(self) -> str:
        rows = dict(self._overlay_status_rows())
        return f"Overlay: {rows['Overlay锁仓']} | Formal: {rows['Formal状态']} | 最近: {rows['最近Overlay']}"

    def _load_snapshot_payload(self) -> dict[str, Any]:
        snapshot = self.store.load_snapshot()
        return snapshot if isinstance(snapshot, dict) else {}

    def _position_summary(self) -> dict[str, Any]:
        snapshot = self._load_snapshot_payload()
        local_position = snapshot.get("position") if isinstance(snapshot.get("position"), dict) else None
        long_state = {"contracts": 0.0, "notional_usdt": 0.0}
        short_state = {"contracts": 0.0, "notional_usdt": 0.0}
        pending_bracket = {
            "algo_id": None,
            "algo_client_id": None,
            "stop_price": None,
            "target_price": None,
        }
        if self.config.mode == "live":
            try:
                long_state = self._fetch_position_state("long")
                short_state = self._fetch_position_state("short")
                if float(long_state.get("contracts", 0.0) or 0.0) > 0:
                    pending_bracket = self._extract_pending_algo_metadata(self._select_pending_algo_order("long"))
                elif float(short_state.get("contracts", 0.0) or 0.0) > 0:
                    pending_bracket = self._extract_pending_algo_metadata(self._select_pending_algo_order("short"))
            except Exception:
                pass
        return {
            "local_position": local_position,
            "long": long_state,
            "short": short_state,
            "pending_bracket": pending_bracket,
        }

    def _format_optional_price(self, value: Any) -> str:
        numeric = self._safe_float(value)
        return f"{numeric:.1f}" if numeric is not None else "-"

    def _format_optional_usdt(self, value: Any, *, digits: int = 2) -> str:
        numeric = self._safe_float(value)
        return f"{numeric:.{digits}f}U" if numeric is not None else "-"

    def _format_optional_leverage(self, value: Any) -> str:
        numeric = self._safe_float(value)
        return f"{numeric:.2f}x" if numeric is not None else "-"

    def _latest_open_action_metadata(self, entry_time: str | None) -> dict[str, Any]:
        if not entry_time:
            return {}
        for item in self.store.recent_actions(50):
            if item.get("timestamp") != entry_time:
                continue
            if item.get("action_type") not in {ActionType.OPEN_LONG.value, ActionType.OPEN_SHORT.value}:
                continue
            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue
            metadata = payload.get("metadata")
            return metadata if isinstance(metadata, dict) else {}
        return {}

    def _current_position_execution_context(
        self,
        snapshot: dict[str, Any],
        dyn: dict[str, Any],
    ) -> dict[str, Any]:
        position = snapshot.get("position") if isinstance(snapshot.get("position"), dict) else None
        if not position:
            return {}

        entry_time = str(position.get("entry_time") or "")
        action_metadata = self._latest_open_action_metadata(entry_time)
        last_decision = dyn.get("last_decision") if isinstance(dyn.get("last_decision"), dict) else {}
        decision_matches_position = (
            bool(last_decision)
            and str(dyn.get("last_update_time") or "") == entry_time
        )

        actual_notional = self._safe_float(position.get("notional"))
        capital_at_entry = self._safe_float(position.get("capital_at_entry")) or self._safe_float(snapshot.get("capital"))
        actual_effective_leverage = (
            actual_notional / capital_at_entry
            if actual_notional is not None and capital_at_entry and capital_at_entry > 0
            else None
        )
        selected_effective_leverage = self._safe_float(position.get("execution_effective_leverage"))
        risk_mode = position.get("execution_risk_mode")
        leverage_reasons = position.get("execution_leverage_reasons")
        diagnostics = position.get("execution_guard_diagnostics")
        if selected_effective_leverage is None and decision_matches_position:
            selected_effective_leverage = self._safe_float(last_decision.get("effective_leverage"))
        if not risk_mode and decision_matches_position:
            risk_mode = last_decision.get("risk_mode")
        if not isinstance(leverage_reasons, list) and decision_matches_position:
            leverage_reasons = last_decision.get("leverage_reasons")
        if not isinstance(diagnostics, dict) and decision_matches_position:
            diagnostics = last_decision.get("diagnostics")

        theoretical_notional = self._safe_float(position.get("execution_requested_notional"))
        if theoretical_notional is None:
            theoretical_notional = self._safe_float(action_metadata.get("risk_based_notional"))
        if theoretical_notional is None:
            theoretical_notional = self._safe_float(action_metadata.get("notional"))
        target_notional = self._safe_float(position.get("execution_target_notional")) or actual_notional

        return {
            "actual_notional": actual_notional,
            "actual_effective_leverage": actual_effective_leverage,
            "selected_effective_leverage": selected_effective_leverage,
            "risk_mode": str(risk_mode or "-"),
            "leverage_reasons": leverage_reasons if isinstance(leverage_reasons, list) else [],
            "reason_text": self._dynamic_leverage_reason_text(leverage_reasons if isinstance(leverage_reasons, list) else []),
            "theoretical_notional": theoretical_notional,
            "target_notional": target_notional,
            "diagnostics": diagnostics if isinstance(diagnostics, dict) else {},
        }

    def _dynamic_leverage_reason_text(self, reasons: list[Any]) -> str:
        if not reasons:
            return "-"
        labels = {
            "base": "基础",
            "high_growth": "扩张期",
            "tight_stop": "窄止损",
            "win_streak_expand": "连胜扩张",
            "market_unhealthy_reduce": "健康度降杠杆",
            "drawdown_reduce": "回撤降杠杆",
            "state_defense_reduce": "防守档降杠杆",
        }
        parts: list[str] = []
        for raw in reasons:
            reason = str(raw)
            if reason.startswith("failed_breakout_guard:"):
                score = reason.split(":", 1)[1]
                parts.append(f"防假突破保护 {score}")
            else:
                parts.append(labels.get(reason, reason))
        return " + ".join(parts)

    def _position_execution_rows(self, snapshot: dict[str, Any], dyn: dict[str, Any]) -> list[tuple[str, str]]:
        context = self._current_position_execution_context(snapshot, dyn)
        if not context:
            return []
        theoretical = context.get("theoretical_notional")
        actual = context.get("actual_notional")
        target = context.get("target_notional")
        rows = [
            ("账户有效杠杆", self._format_optional_leverage(context.get("actual_effective_leverage"))),
            (
                "执行杠杆",
                f"{self._format_optional_leverage(context.get('selected_effective_leverage'))} / {context.get('risk_mode') or '-'}",
            ),
            ("压仓原因", str(context.get("reason_text") or "-")),
        ]
        if theoretical is not None or actual is not None:
            rows.append(
                (
                    "理论/实际仓位",
                    f"{self._format_optional_usdt(theoretical, digits=0)} -> {self._format_optional_usdt(actual, digits=0)}",
                )
            )
        elif target is not None:
            rows.append(("目标仓位", self._format_optional_usdt(target, digits=0)))
        return rows

    def _telegram_status_text(self, *, table: bool = False) -> str:
        snapshot = self._load_snapshot_payload()
        position = self._position_summary()
        local_position = position["local_position"]
        dyn = self._load_dynamic_high_leverage_state() if self._dynamic_high_leverage_enabled() else {}
        shadow = self._load_shadow_gate_state() if self._shadow_gate_enabled() else {}
        paused = self._telegram_open_paused()
        long_contracts = float(position["long"].get("contracts", 0.0) or 0.0)
        short_contracts = float(position["short"].get("contracts", 0.0) or 0.0)
        bracket = position.get("pending_bracket") if isinstance(position.get("pending_bracket"), dict) else {}
        exchange_side = "long" if long_contracts > 0 else "short" if short_contracts > 0 else "flat"
        local_side = "-"
        if local_position:
            local_side = "long" if local_position.get("direction") == "BULL" else "short"
        bracket_id = bracket.get("algo_id") or bracket.get("algo_client_id") or "-"
        execution_rows = self._position_execution_rows(snapshot, dyn)
        overlay_rows = self._overlay_status_rows()
        lines = [self._telegram_title("📡", "状态雷达") if not table else self._telegram_title("🧾", "状态面板")]
        rows = [
            ("标的", self.config.symbol),
            ("模式", self.config.mode),
            ("开仓", self._open_status_text(paused)),
            ("交易所仓位", self._side_status_text(exchange_side)),
            ("本地仓位", self._side_status_text(local_side) if local_side in {"long", "short", "flat"} else local_side),
            ("交易所止损", self._format_optional_price(bracket.get("stop_price"))),
            ("交易所止盈", self._format_optional_price(bracket.get("target_price"))),
            ("保护单ID", str(bracket_id)),
            *execution_rows,
            ("策略资金", f"{float(snapshot.get('capital', 0.0) or 0.0):.2f}U"),
            ("交易次数", str(int(snapshot.get("trade_count", 0) or 0))),
            ("最近K线", self.store.get_value("last_processed_candle_time") or "-"),
            ("动态档位", str(dyn.get("mode") or "-")),
            ("Shadow暂停到", self._shadow_format_ts(float(shadow.get("pause_until_ts", 0.0) or 0.0)) or "-"),
            *overlay_rows,
            ("时间", self._local_time_text()),
        ]
        if table:
            row_map = dict(rows)
            lines = [
                self._telegram_title("🧾", "状态面板"),
                "🧭 运行",
                f"🎯 标的：{row_map['标的']}",
                f"⚙️ 模式：{row_map['模式']}",
                f"🚦 开仓：{row_map['开仓']}",
                "",
                "📦 仓位",
                f"🏛️ 交易所：{row_map['交易所仓位']}",
                f"🧠 本地：{row_map['本地仓位']}",
                f"🛡️ 止损：{row_map['交易所止损']}",
                f"🎯 止盈：{row_map['交易所止盈']}",
                f"🔐 保护单：{row_map['保护单ID']}",
            ]
            if execution_rows:
                lines.extend(
                    [
                        "",
                        "📐 执行",
                        f"⚡ 账户有效：{row_map['账户有效杠杆']}",
                        f"🎚️ 执行杠杆：{row_map['执行杠杆']}",
                        f"🧯 压仓：{row_map['压仓原因']}",
                    ]
                )
                if "理论/实际仓位" in row_map:
                    lines.append(f"📊 仓位：{row_map['理论/实际仓位']}")
            lines.extend(
                [
                    "",
                    "🚀 策略",
                    f"💎 资金：{row_map['策略资金']}",
                    f"🔢 交易：{row_map['交易次数']}",
                    f"⚡ 档位：{row_map['动态档位']}",
                    f"👤 Shadow：{row_map['Shadow暂停到']}",
                    f"🧠 Formal：{row_map['Formal状态']}",
                    f"🧩 Overlay：{row_map['Overlay锁仓']}",
                    f"📝 最近决策：{row_map['最近Overlay']}",
                    "",
                    "⏱ 时间",
                    f"🕯️ K线：{row_map['最近K线']}",
                    f"📅 {row_map['时间']}",
                ]
            )
        else:
            labels = {
                "标的": "🎯 标的",
                "模式": "⚙️ 模式",
                "开仓": "🚦 开仓",
                "交易所仓位": "🏛️ 交易所仓位",
                "本地仓位": "🧠 本地仓位",
                "交易所止损": "🛡️ 交易所止损",
                "交易所止盈": "🎯 交易所止盈",
                "保护单ID": "🔐 保护单ID",
                "账户有效杠杆": "⚡ 账户有效杠杆",
                "执行杠杆": "🎚️ 执行杠杆",
                "压仓原因": "🧯 压仓原因",
                "理论/实际仓位": "📊 理论/实际仓位",
                "目标仓位": "📊 目标仓位",
                "策略资金": "💎 策略资金",
                "交易次数": "🔢 交易次数",
                "最近K线": "🕯️ 最近K线",
                "动态档位": "⚡ 动态档位",
                "Shadow暂停到": "👤 Shadow暂停到",
                "Formal状态": "🧠 Formal状态",
                "Overlay锁仓": "🧩 Overlay锁仓",
                "最近Overlay": "📝 最近Overlay",
                "时间": "📅 时间",
            }
            lines.extend(f"{labels.get(name, name)}：{value}" for name, value in rows)
        return "\n".join(lines)

    def _telegram_balance_text(self) -> str:
        lines = [self._telegram_title("🏦", "账户余额")]
        try:
            balance = self.client.fetch_balance()
            available, available_source = self._extract_available_usdt(balance)
            total = self._extract_total_usdt(balance)
            lines.append(f"💵 可用：{available:.2f} USDT")
            lines.append(f"💎 权益：{total:.2f} USDT")
            lines.append(f"🔎 来源：{available_source}")
        except Exception as exc:
            lines.append("🔴 状态：查询失败")
            lines.append(f"⚠️ 错误：{exc}")
        lines.append(self._telegram_time_line())
        return "\n".join(lines)

    def _realized_pnl_events(self, *, daily: bool) -> list[dict[str, Any]]:
        actions = self.store.recent_actions(1000)
        today = datetime.now().strftime("%Y-%m-%d")
        events = []
        for item in actions:
            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue
            if item.get("action_type") != ActionType.CLOSE_POSITION.value:
                continue
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                continue
            pnl = self._safe_float(metadata.get("net_pnl"))
            if pnl is None:
                continue
            timestamp = str(item.get("timestamp") or "")
            if daily and not timestamp.startswith(today):
                continue
            events.append({"timestamp": timestamp, "pnl": pnl, "reason": payload.get("reason")})
        return events

    def _telegram_profit_text(self, *, daily: bool) -> str:
        events = self._realized_pnl_events(daily=daily)
        total = sum(float(event["pnl"]) for event in events)
        wins = sum(1 for event in events if float(event["pnl"]) > 0)
        mood = self._telegram_mood(total)
        title = self._telegram_title(self._telegram_random_icon(mood), "今日收益") if daily else self._telegram_title(self._telegram_random_icon(mood), "累计收益")
        pnl_icon = self._telegram_random_icon(mood)
        return "\n".join(
            [
                title,
                f"{pnl_icon} 已实现 PnL：{total:.2f} USDT",
                f"🔒 平仓笔数：{len(events)}",
                f"🏆 胜率：{(wins / len(events) * 100.0):.1f}%" if events else "🏆 胜率：-",
                self._telegram_time_line(),
            ]
        )

    def _telegram_performance_text(self) -> str:
        snapshot = self._load_snapshot_payload()
        dyn = self._load_dynamic_high_leverage_state() if self._dynamic_high_leverage_enabled() else {}
        shadow = self._load_shadow_gate_state() if self._shadow_gate_enabled() else {}
        unit_returns = dyn.get("unit_returns") if isinstance(dyn.get("unit_returns"), list) else []
        recent = self._dynamic_recent_stats(unit_returns, min(len(unit_returns), int(self.config.dynamic_state_lookback_trades))) if unit_returns else {}
        capital = float(snapshot.get("capital", 0.0) or 0.0)
        trade_count = int(snapshot.get("trade_count", 0) or 0)
        exits = snapshot.get("exit_reasons") if isinstance(snapshot.get("exit_reasons"), dict) else {}
        recent_return = float(recent.get("unit_return_pct", 0.0) or 0.0)
        mood = self._telegram_mood(recent_return)
        execution = self._current_position_execution_context(snapshot, dyn)
        lines = [
            self._telegram_title(self._telegram_random_icon(mood), "策略表现"),
            f"💎 策略资金：{capital:.2f}U",
            f"🔢 交易次数：{trade_count}",
            f"⚡ 动态档位：{dyn.get('mode') or '-'}",
            f"{self._telegram_random_icon(mood)} 近期单位收益：{recent_return:.2f}%",
            f"🏆 近期胜率：{float(recent.get('win_rate_pct', 0.0) or 0.0):.1f}%",
            f"👤 Shadow资金：{float(shadow.get('capital', 0.0) or 0.0):.2f}U",
            f"🧩 {self._overlay_compact_line()}",
        ]
        if execution:
            lines.extend(
                [
                    "",
                    "📐 当前执行",
                    (
                        "⚡ 有效杠杆："
                        f"{self._format_optional_leverage(execution.get('actual_effective_leverage'))} "
                        f"/ 执行 {self._format_optional_leverage(execution.get('selected_effective_leverage'))}"
                    ),
                    (
                        "📊 仓位：理论 "
                        f"{self._format_optional_usdt(execution.get('theoretical_notional'), digits=0)} "
                        f"-> 实际 {self._format_optional_usdt(execution.get('actual_notional'), digits=0)}"
                    ),
                    f"🧯 风控：{execution.get('reason_text') or '-'}",
                ]
            )
            diagnostics = execution.get("diagnostics") if isinstance(execution.get("diagnostics"), dict) else {}
            if diagnostics:
                adx = self._safe_float(diagnostics.get("feature_adx"))
                momentum = self._safe_float(diagnostics.get("feature_momentum"))
                ema_gap = self._safe_float(diagnostics.get("feature_ema_gap"))
                quality_parts = []
                if adx is not None:
                    quality_parts.append(f"ADX {adx:.1f}")
                if momentum is not None:
                    quality_parts.append(f"动量 {momentum * 100.0:.2f}%")
                if ema_gap is not None:
                    quality_parts.append(f"EMA差 {ema_gap * 100.0:.2f}%")
                if quality_parts:
                    lines.append("🧪 质量：" + " / ".join(quality_parts))
        if exits:
            lines.append("🚪 退出原因：" + ", ".join(f"{k}:{v}" for k, v in sorted(exits.items())))
        lines.append(self._telegram_time_line())
        return "\n".join(lines)

    def _telegram_count_text(self) -> str:
        snapshot = self._load_snapshot_payload()
        actions = self.store.recent_actions(1000)
        open_count = sum(1 for item in actions if item.get("action_type") in {ActionType.OPEN_LONG.value, ActionType.OPEN_SHORT.value})
        close_count = sum(1 for item in actions if item.get("action_type") == ActionType.CLOSE_POSITION.value)
        return "\n".join(
            [
                self._telegram_title("🔢", "交易计数"),
                f"🧠 策略交易数：{int(snapshot.get('trade_count', 0) or 0)}",
                f"🟢 最近记录开仓：{open_count}",
                f"🔒 最近记录平仓：{close_count}",
                self._telegram_time_line(),
            ]
        )

    def _telegram_open_paused(self) -> bool:
        return str(self.store.get_value("telegram_open_paused") or "false").lower() in {"1", "true", "yes", "on"}

    def _sleep_with_telegram(self, seconds: float, poll_interval_seconds: int) -> None:
        deadline = time.time() + max(float(seconds), 0.0)
        interval = max(min(int(poll_interval_seconds), 30), 1)
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return
            time.sleep(min(remaining, interval))
            self._run_telegram_background_tasks()

    def _send_startup_telegram(self, bootstrap_status: dict[str, Any]) -> None:
        status_text = "成功" if not bootstrap_status.get("bootstrap_error") else "异常"
        snapshot_loaded = "是" if bootstrap_status.get("snapshot_loaded") else "否"
        market_loaded = "是" if bootstrap_status.get("market_loaded") else "否"
        lines = [
            "[Bot启动]",
            f"状态: {status_text}",
            f"标的: {self.config.symbol}",
            f"模式: {self.config.mode}",
            f"市场加载: {market_loaded}",
            f"快照加载: {snapshot_loaded}",
            f"Formal状态: {bootstrap_status.get('formal_state_status') or '-'}",
            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if bootstrap_status.get("bootstrap_error"):
            lines.append(f"错误: {bootstrap_status['bootstrap_error']}")
        self._send_telegram("\n".join(lines))

    def _resolve_runtime_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        if self.config_path is not None:
            return self.config_path.parents[0].parents[0] / path
        return Path.cwd() / path

    def _build_drift_report_message(self) -> str:
        try:
            from scripts.live_drift_monitor import (
                build_live_trades,
                build_report,
                format_report,
                load_action_log,
                load_json,
            )

            baseline_path = self._resolve_runtime_path(self.config.telegram_drift_baseline_path)
            baseline = load_json(baseline_path)
            state_db = Path(self.store.db_path).resolve()
            actions = load_action_log(state_db)
            trades, diagnostics = build_live_trades(actions)
            report = build_report(
                config_path=self.config_path or Path("runtime_config"),
                state_db=state_db,
                baseline=baseline,
                actions=actions,
                trades=trades,
                diagnostics=diagnostics,
                window_days=int(self.config.telegram_drift_window_days),
                recent_trades=int(self.config.telegram_drift_recent_trades),
            )
            return format_report(report)
        except Exception as exc:
            return "\n".join(
                [
                    "🩺 <Drift 体检>",
                    "状态: ⚠️ 生成失败",
                    f"原因: {exc}",
                    f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ]
            )

    def _format_price(self, value: float | None) -> str:
        if value is None:
            return "-"
        return f"{value:,.1f}"

    def _format_distance(self, current_price: float, target_price: float | None) -> str:
        if target_price is None or current_price <= 0:
            return "-"
        diff = target_price - current_price
        pct = diff / current_price * 100.0
        direction = "向上" if diff > 0 else "向下" if diff < 0 else "当前价"
        return f"{direction} {abs(diff):,.1f}U / {abs(pct):.2f}%"

    def _format_level_condition(self, current_price: float, target_price: float | None, *, expect: str) -> str:
        if target_price is None or current_price <= 0:
            return "-"
        diff = target_price - current_price
        pct = abs(diff) / current_price * 100.0
        amount = abs(diff)
        if expect == "above":
            if current_price >= target_price:
                return f"已在上方 {amount:,.1f}U / {pct:.2f}%"
            return f"向上还差 {amount:,.1f}U / {pct:.2f}%"
        if expect == "below":
            if current_price <= target_price:
                return f"已在下方 {amount:,.1f}U / {pct:.2f}%"
            return f"向下还差 {amount:,.1f}U / {pct:.2f}%"
        return self._format_distance(current_price, target_price)

    def _direction_label(self, direction: str | None) -> str:
        if direction == Direction.BULL:
            return "🟢 多头"
        if direction == Direction.BEAR:
            return "🔴 空头"
        return "⚪ 无"

    def _latest_shadow_status_lines(self) -> list[str]:
        if not self._shadow_gate_enabled():
            return ["🛡 Shadow: 关闭"]
        state = self._load_shadow_gate_state()
        pause_until_ts = float(state.get("pause_until_ts", 0.0) or 0.0)
        now_ts = datetime.now(timezone.utc).timestamp()
        if pause_until_ts > now_ts:
            return [f"🛡 Shadow: 防守冷却中，至 {self._shadow_format_ts(pause_until_ts)} UTC"]
        return ["🛡 Shadow: 可进攻"]

    def _pending_entry_conditions(self, engine: Any, idx: int, pending: Any) -> dict[str, Any]:
        curr = engine.c15m[idx]
        direction = pending.direction
        top = float(pending.ob_zone["top"])
        bottom = float(pending.ob_zone["bottom"])
        in_ob = bottom <= curr.l <= top or bottom <= curr.h <= top
        direction_allowed = (
            direction == Direction.BULL
            and engine.config.allow_long
            and engine._regime_ok_for_direction_idx(idx, Direction.BULL)
        ) or (
            direction == Direction.BEAR
            and engine.config.allow_short
            and engine._regime_ok_for_direction_idx(idx, Direction.BEAR)
        )
        candle_confirmed = (direction == Direction.BULL and curr.c > curr.o) or (
            direction == Direction.BEAR and curr.c < curr.o
        )
        expired = engine._pending_expired(idx, pending)
        missing: list[str] = []
        if expired:
            missing.append("OB 等待窗口已过期")
        if not direction_allowed:
            missing.append("方向被 regime / allow_long / allow_short 过滤")
        if not in_ob:
            if curr.c > top:
                distance_pct = (curr.c - top) / curr.c * 100.0 if curr.c > 0 else 0.0
                missing.append(f"价格还没回踩 OB，上沿还差约 {distance_pct:.2f}%")
            elif curr.c < bottom:
                distance_pct = (bottom - curr.c) / curr.c * 100.0 if curr.c > 0 else 0.0
                missing.append(f"价格已在 OB 下方，需重新收回区间约 {distance_pct:.2f}%")
            else:
                missing.append("影线尚未触及 OB 区间")
        if not candle_confirmed:
            missing.append("缺少确认 K：多头需收阳，空头需收阴")
        expires_in = pending.bos_idx + pending.pullback_window - idx
        return {
            "ready": not missing,
            "missing": missing,
            "in_ob": in_ob,
            "expires_in_bars": expires_in,
            "current_close": float(curr.c),
            "current_low": float(curr.l),
            "current_high": float(curr.h),
            "top": top,
            "bottom": bottom,
        }

    def _active_ob_candidates(self, engine: Any, latest_idx: int) -> list[tuple[Any, dict[str, Any]]]:
        max_window = max(int(self.config.pullback_window or 0), int(self.config.short_pullback_window or 0), 40)
        scan_start = max(100, latest_idx - max(300, max_window + 120))
        pending_by_direction: dict[str, Any | None] = {Direction.BULL: None, Direction.BEAR: None}

        for idx in range(scan_start, latest_idx + 1):
            engine._apply_regime_switch_for_idx(idx)
            for direction in (Direction.BULL, Direction.BEAR):
                pending = pending_by_direction[direction]
                if pending is None:
                    continue
                detail = self._pending_entry_conditions(engine, idx, pending)
                if detail["ready"] and idx < latest_idx:
                    pending_by_direction[direction] = None
                    continue
                if engine._pending_expired(idx, pending):
                    pending_by_direction[direction] = None

            if idx not in (engine.precomputed.highs_set | engine.precomputed.lows_set):
                continue
            bias = engine.precomputed.bias_4h[engine.mapping[idx]]
            active_pending = any(pending_by_direction.values())
            if engine.config.use_hfvf_filter and bias == Direction.NONE and not active_pending:
                continue
            pending = engine._build_pending_pullback(idx, bias)
            if pending and pending_by_direction[pending.direction] is None:
                pending_by_direction[pending.direction] = pending

        out: list[tuple[Any, dict[str, Any]]] = []
        engine._apply_regime_switch_for_idx(latest_idx)
        for pending in pending_by_direction.values():
            if pending is not None:
                out.append((pending, self._pending_entry_conditions(engine, latest_idx, pending)))
        return out

    def _structure_reference(self, engine: Any, latest_idx: int, bias: str) -> dict[str, Any]:
        latest = engine.c15m[latest_idx]
        highs = [idx for idx in engine.precomputed.highs_15m if idx < latest_idx]
        lows = [idx for idx in engine.precomputed.lows_15m if idx < latest_idx]
        reference: dict[str, Any] = {
            "current_price": float(latest.c),
            "bias": bias,
            "primary": None,
            "secondary": None,
            "opposite": None,
        }
        if lows:
            low_idx = lows[-1]
            low = engine.c15m[low_idx]
            break_price = float(low.l)
            reference["bear"] = {
                "break_price": break_price,
                "reclaim_price": float(low.h),
                "time": engine._timestamp_for_idx(low_idx),
            }
            stronger_low_indices = [
                idx for idx in lows[-6:-1] if float(engine.c15m[idx].l) < break_price
            ]
            if stronger_low_indices:
                strong_low_idx = min(
                    stronger_low_indices,
                    key=lambda idx: float(engine.c15m[idx].l),
                )
                strong_low = engine.c15m[strong_low_idx]
                reference["bear"]["strong_break_price"] = float(strong_low.l)
                reference["bear"]["strong_time"] = engine._timestamp_for_idx(strong_low_idx)
        if highs:
            high_idx = highs[-1]
            high = engine.c15m[high_idx]
            break_price = float(high.h)
            reference["bull"] = {
                "break_price": break_price,
                "reclaim_price": float(high.l),
                "time": engine._timestamp_for_idx(high_idx),
            }
            stronger_high_indices = [
                idx for idx in highs[-6:-1] if float(engine.c15m[idx].h) > break_price
            ]
            if stronger_high_indices:
                strong_high_idx = max(
                    stronger_high_indices,
                    key=lambda idx: float(engine.c15m[idx].h),
                )
                strong_high = engine.c15m[strong_high_idx]
                reference["bull"]["strong_break_price"] = float(strong_high.h)
                reference["bull"]["strong_time"] = engine._timestamp_for_idx(strong_high_idx)

        if bias == Direction.BEAR:
            reference["primary"] = reference.get("bear")
            reference["opposite"] = reference.get("bull")
        elif bias == Direction.BULL:
            reference["primary"] = reference.get("bull")
            reference["opposite"] = reference.get("bear")
        return reference

    def _structure_reference_lines(self, reference: dict[str, Any]) -> list[str]:
        current_price = float(reference.get("current_price", 0.0) or 0.0)
        primary = reference.get("primary") if isinstance(reference.get("primary"), dict) else None
        bias = reference.get("bias")
        if not primary:
            return ["📍 结构参考: 暂无足够关键高低点"]

        is_bear = bias == Direction.BEAR
        break_label = "先跌破" if is_bear else "先突破"
        reclaim_label = "再收回到" if is_bear else "再回踩守住"
        break_expect = "below" if is_bear else "above"
        lines = [
            "",
            "📍 结构参考价",
            f"方向: {self._direction_label(bias)}",
            f"{break_label}: {self._format_price(primary.get('break_price'))} "
            f"({self._format_level_condition(current_price, primary.get('break_price'), expect=break_expect)})",
        ]
        if primary.get("strong_break_price") is not None:
            strong_label = "更强跌破" if is_bear else "更强突破"
            lines.append(
                f"{strong_label}: {self._format_price(primary.get('strong_break_price'))} "
                f"({self._format_level_condition(current_price, primary.get('strong_break_price'), expect=break_expect)})"
            )
        lines.append(
            f"{reclaim_label}: {self._format_price(primary.get('reclaim_price'))} "
            f"({self._format_level_condition(current_price, primary.get('reclaim_price'), expect='above')})"
        )
        if is_bear:
            lines.append("白话: 先向下打穿关键低点，再拉回确认价上方，才进入找 OB 的下一步。")
        else:
            lines.append("白话: 先向上打穿关键高点，再回踩不破确认价，才进入找 OB 的下一步。")
        return lines

    def _regime_display_label(self, regime: str, features: dict[str, Any]) -> str:
        if regime == "high_growth":
            compression_score = int(features.get("compression_growth_score", 0) or 0)
            strong_score = int(features.get("strong_growth_score", 0) or 0)
            adx = float(features.get("adx", 0.0) or 0.0)
            momentum = float(features.get("momentum", 0.0) or 0.0)
            if compression_score >= 4 and strong_score < 3:
                return "🟡 压缩蓄势"
            if adx >= 35.0 and momentum >= 0.04:
                return "🟢 强趋势扩张"
            return "🟢 趋势扩张"
        if regime == "flat":
            return "⚪ 震荡防守"
        if regime == "normal":
            return "🔵 常规过滤"
        if regime == "static":
            return "⚙️ 静态参数"
        return regime

    def _regime_display_lines(self, regime: str, features: dict[str, Any]) -> list[str]:
        lines = [f"市场状态: {self._regime_display_label(regime, features)}"]
        lines.append(f"策略桶: {regime}")
        if features:
            adx = float(features.get("adx", 0.0) or 0.0)
            momentum_pct = float(features.get("momentum", 0.0) or 0.0) * 100.0
            ema_gap_pct = float(features.get("ema_gap", 0.0) or 0.0) * 100.0
            atr_ratio = float(features.get("atr_ratio", 0.0) or 0.0)
            lines.append(
                f"因子: ADX {adx:.1f} | 动量 {momentum_pct:+.2f}% | "
                f"EMA差 {ema_gap_pct:+.2f}% | ATR {atr_ratio:.2f}x"
            )
        return lines

    def _build_ob_status_message(self) -> str:
        try:
            engine, _ = self.load_engine()
            latest_idx = self._latest_closed_index(engine)
            if latest_idx is None:
                return "🧭 <OB 简报>\n状态: 等待最新收盘 K 线"

            engine._apply_regime_switch_for_idx(latest_idx)
            latest = engine.c15m[latest_idx]
            bias = engine.precomputed.bias_4h[engine.mapping[latest_idx]]
            regime = engine._regime_label_for_idx(latest_idx)
            regime_features = engine._regime_features_for_idx(latest_idx)
            timestamp = engine._timestamp_for_idx(latest_idx)
            structure_reference = self._structure_reference(engine, latest_idx, bias)
            shadow_lines = self._latest_shadow_status_lines()
            overlay_line = self._overlay_compact_line()
            lines = [
                "🧭 <OB 简报>",
                f"标的: {self.config.symbol}",
                f"价格: {self._format_price(float(latest.c))}",
                f"方向: {self._direction_label(bias)}",
                f"市场: {self._regime_display_label(regime, regime_features)}",
                f"保护: {shadow_lines[0].replace('🛡 ', '')}",
                f"{overlay_line}",
            ]

            if engine.position is not None:
                state = self._load_shadow_gate_state(engine) if self._shadow_gate_enabled() else {}
                real_open = bool(state.get("real_position_open", True))
                position = engine.position
                label = "持仓中" if real_open else "Shadow paper position"
                lines.extend(
                    [
                        "",
                        f"状态: {label}，不开新仓",
                        f"方向: {self._direction_label(getattr(position, 'direction', None))}",
                        f"入场: {self._format_price(getattr(position, 'entry_price', None))}",
                        f"止损: {self._format_price(getattr(position, 'sl_price', None))}",
                        f"止盈: {self._format_price(getattr(position, 'target_price', None))}",
                        "下一步: 等当前仓位结束",
                        f"时间: {timestamp} UTC",
                    ]
                )
                return "\n".join(lines)

            candidates = self._active_ob_candidates(engine, latest_idx)
            if not candidates:
                next_step = "等待方向性 4H bias"
                if bias == Direction.NONE:
                    next_step = "等待方向性 4H bias"
                else:
                    primary = structure_reference.get("primary") if isinstance(structure_reference.get("primary"), dict) else {}
                    break_price = primary.get("break_price")
                    reclaim_price = primary.get("reclaim_price")
                    if bias == Direction.BEAR and break_price is not None and reclaim_price is not None:
                        next_step = f"跌破 {self._format_price(break_price)} 后收回 {self._format_price(reclaim_price)}"
                    elif bias == Direction.BULL and break_price is not None and reclaim_price is not None:
                        next_step = f"突破 {self._format_price(break_price)} 后回踩守住 {self._format_price(reclaim_price)}"
                    else:
                        next_step = "等待关键价破位和收回确认"
                lines.extend(
                    [
                        "",
                        "状态: 暂无 OB 候选",
                        f"下一步: {next_step}",
                        f"时间: {timestamp} UTC",
                    ]
                )
                return "\n".join(lines)

            ready_candidates = [(pending, detail) for pending, detail in candidates if detail["ready"]]
            if ready_candidates:
                lines.append("")
                lines.append("状态: 有 OB 候选满足，等策略风控确认")
            else:
                lines.append("")
                lines.append("状态: 有 OB 候选，但未触发")

            for idx, (pending, detail) in enumerate(candidates[:2], start=1):
                missing = detail["missing"]
                next_step = "等待策略评估/风控确认" if detail["ready"] else (str(missing[0]) if missing else "等待确认 K")
                lines.extend(
                    [
                        "",
                        f"候选{idx}: {self._direction_label(pending.direction)} | {self._format_price(detail['bottom'])}-{self._format_price(detail['top'])}",
                        f"窗口: {detail['expires_in_bars']} 根15m",
                        f"下一步: {next_step}",
                    ]
                )
            lines.append(f"时间: {timestamp} UTC")
            return "\n".join(lines)
        except Exception as exc:
            return "\n".join(
                [
                    "🧭 <OB 简报>",
                    "状态: 生成失败",
                    f"原因: {exc}",
                    f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ]
            )

    def _interval_due(self, key: str, interval_seconds: int) -> bool:
        raw = self.store.get_value(key)
        if not raw:
            return True
        try:
            last_ts = float(raw)
        except ValueError:
            return True
        return datetime.now(timezone.utc).timestamp() - last_ts >= interval_seconds

    def _mark_interval_sent(self, key: str) -> None:
        self.store.set_value(key, str(datetime.now(timezone.utc).timestamp()))

    def _run_telegram_background_tasks(self) -> None:
        if not self.config.telegram_enabled:
            return
        try:
            self._handle_telegram_commands()
        except Exception as exc:
            self.store.append_action("runtime", "TELEGRAM_COMMAND_ERROR", {"error": str(exc)})

        if self.config.telegram_ob_status_enabled:
            key = "telegram_last_ob_status_ts"
            interval = max(1, int(self.config.telegram_ob_status_interval_minutes)) * 60
            if self._interval_due(key, interval):
                if self._send_telegram(self._build_ob_status_message()):
                    self._mark_interval_sent(key)

        if self.config.telegram_drift_report_enabled:
            key = "telegram_last_drift_report_ts"
            interval = max(1, int(self.config.telegram_drift_report_interval_hours)) * 3600
            if self._interval_due(key, interval):
                if self._send_telegram(self._build_drift_report_message()):
                    self._mark_interval_sent(key)

    def bootstrap(self) -> dict[str, Any]:
        self.check_safety()
        markets = None
        market_loaded = False
        bootstrap_error = None
        try:
            markets = self._load_markets()
            market_loaded = self.config.symbol in markets
            self.client.set_leverage(
                self.config.leverage,
                self.config.symbol,
                margin_mode=self.config.margin_mode,
                pos_side="long",
            )
            self.client.set_leverage(
                self.config.leverage,
                self.config.symbol,
                margin_mode=self.config.margin_mode,
                pos_side="short",
            )
        except Exception as exc:
            bootstrap_error = str(exc)
        snapshot = self.store.load_snapshot()
        status = {
            "mode": self.config.mode,
            "symbol": self.config.symbol,
            "market_loaded": market_loaded,
            "snapshot_loaded": snapshot is not None,
            "bootstrap_error": bootstrap_error,
            "formal_state_status": self._overlay_formal_state_text() if self._live_overlay_enabled() else "关闭",
        }
        self.store.append_action("bootstrap", "BOOTSTRAP", status)
        return status

    def load_engine(self) -> tuple[Any, int]:
        bundle = self.market_data.load_pair(
            self.config.symbol,
            client=self.client,
            timeframe=self.config.timeframe,
            informative_timeframe=self.config.informative_timeframe,
        )
        primary_candles = dataframe_to_candles(bundle.primary_candles)
        informative_candles = dataframe_to_candles(bundle.informative_candles)
        if not primary_candles:
            raise ValueError("No market data loaded for executor")

        if self.config.strategy_type != "scalp_robust_v2":
            raise ValueError(f"Unsupported strategy_type: {self.config.strategy_type}")

        if not informative_candles:
            raise ValueError("No informative market data loaded for scalp executor")
        engine = ScalpRobustEngine.from_candles(
            informative_candles,
            primary_candles,
            self.config.to_scalp_strategy_config(),
        )
        if self.config.enable_regime_switching:
            regime_labels, regime_features = precompute_regime_state(
                informative_candles,
                sorted(set(engine.mapping)),
                self.config.regime_switcher_thresholds,
            )
            engine._regime_switch_cache = {
                c4h_idx: (label, engine._config_for_regime(label))
                for c4h_idx, label in regime_labels.items()
            }
            engine._regime_feature_cache = dict(regime_features)
        engine.restore_snapshot(self.store.load_snapshot())
        start_idx = max(100, self._find_resume_index(primary_candles))
        return engine, start_idx

    def evaluate_latest(self) -> dict[str, Any]:
        engine, start_idx = self.load_engine()
        live_capital = self._sync_live_capital(engine)
        latest_closed_idx = self._latest_closed_index(engine)
        if latest_closed_idx is None:
            return {
                "status": "waiting_for_closed_candle",
                "symbol": self.config.symbol,
                "actions": [],
                "trade_count": 0,
                "position_open": False,
                "live_capital": live_capital,
            }
        if not self.store.get_value("last_processed_candle_time"):
            return self._initialize_without_replay(engine, latest_closed_idx)
        last_closed_timestamp = engine._timestamp_for_idx(latest_closed_idx)
        self._assert_live_state_synced(
            engine,
            context="before_evaluate",
            timestamp=last_closed_timestamp,
            exit_idx=latest_closed_idx,
        )
        if latest_closed_idx < start_idx:
            snapshot = engine.snapshot()
            self.store.save_snapshot(snapshot)
            return {
                "status": "insufficient_data",
                "symbol": self.config.symbol,
                "processed_candle_time": engine._timestamp_for_idx(latest_closed_idx),
                "actions": [],
                "trade_count": snapshot.trade_count,
                "position_open": snapshot.position is not None,
                "live_capital": engine.capital,
            }

        if self._live_overlay_enabled():
            self._overlay_ensure_formal_state_warmup(engine, start_idx)

        if self._live_overlay_enabled():
            actions = self._evaluate_latest_with_live_overlay(engine, start_idx, latest_closed_idx)
        else:
            # evaluate_range uses a right-open end index. Include latest_closed_idx;
            # otherwise live can mark a candle processed without evaluating it.
            actions = engine.evaluate_range(start_idx, latest_closed_idx + 1)
        execution_results = []
        for action in actions:
            result = self.execute_action(action, engine)
            execution_results.append({"action": asdict(action), "result": result})
        self._assert_live_state_synced(
            engine,
            context="after_execute",
            timestamp=last_closed_timestamp,
            exit_idx=latest_closed_idx,
        )

        last_timestamp = last_closed_timestamp
        snapshot = engine.snapshot()
        self.store.set_value("last_processed_candle_time", last_timestamp)
        self.store.save_snapshot(snapshot)

        status = {
            "status": "ok",
            "symbol": self.config.symbol,
            "processed_candle_time": last_timestamp,
            "actions": [asdict(action) for action in actions],
            "execution_results": execution_results,
            "trade_count": snapshot.trade_count,
            "position_open": engine.position is not None,
            "snapshot": asdict(snapshot),
            "live_capital": engine.capital,
        }
        self.store.append_action(last_timestamp, "EVALUATE", status)
        return status

    def run_loop(self, poll_interval_seconds: int = 5, close_buffer_seconds: int = 5) -> None:
        bootstrap_status = self.bootstrap()
        print(json.dumps({"event": "bootstrap", **bootstrap_status}, ensure_ascii=False))
        self._configure_telegram_commands()
        self._send_startup_telegram(bootstrap_status)
        self._run_telegram_background_tasks()
        while True:
            try:
                self._run_telegram_background_tasks()
                wait_seconds = self.seconds_until_next_close(close_buffer_seconds)
                latest_closed_time = self.latest_closed_candle_time(close_buffer_seconds)
                last_processed = self.store.get_value("last_processed_candle_time")
                if last_processed == latest_closed_time:
                    sleep_seconds = max(wait_seconds, poll_interval_seconds)
                    payload = {
                        "event": "waiting",
                        "symbol": self.config.symbol,
                        "last_processed_candle_time": last_processed,
                        "next_closed_candle_time": self.next_closed_candle_time(close_buffer_seconds),
                        "sleep_seconds": sleep_seconds,
                    }
                    self.store.append_action(latest_closed_time, "WAIT", payload)
                    print(json.dumps(payload, ensure_ascii=False))
                    self._sleep_with_telegram(max(wait_seconds, poll_interval_seconds), poll_interval_seconds)
                    continue

                status = self.evaluate_latest()
                print(json.dumps({"event": "evaluate", **status}, ensure_ascii=False))
                self._run_telegram_background_tasks()
                self._sleep_with_telegram(poll_interval_seconds, poll_interval_seconds)
            except KeyboardInterrupt:
                stop_payload = {"event": "stopped", "symbol": self.config.symbol}
                self.store.append_action("runtime", "STOP", stop_payload)
                print(json.dumps(stop_payload, ensure_ascii=False))
                raise
            except Exception as exc:
                error_payload = {
                    "event": "error",
                    "symbol": self.config.symbol,
                    "error": str(exc),
                    "retry_in_seconds": poll_interval_seconds,
                }
                self.store.append_action("runtime", "ERROR", error_payload)
                print(json.dumps(error_payload, ensure_ascii=False))
                self._sleep_with_telegram(poll_interval_seconds, poll_interval_seconds)

    def _evaluate_latest_with_live_overlay(self, engine: Any, start_idx: int, latest_closed_idx: int) -> list[StrategyAction]:
        actions: list[StrategyAction] = []
        for idx in range(start_idx, latest_closed_idx + 1):
            runtime_position_before = self._load_overlay_runtime_position()
            managed_actions = self._overlay_manage_runtime_position(engine, idx)
            actions.extend(managed_actions)
            for action in managed_actions:
                if action.type == ActionType.CLOSE_POSITION:
                    self._overlay_post_execute_runtime_update(engine, action, idx)
            if runtime_position_before is not None and getattr(engine, "position", None) is None:
                continue

            base_open_action, non_open_actions = self._overlay_base_actions_for_idx(engine, idx)
            non_open_actions = [
                self._overlay_mark_formal_close_action(engine, action)
                if action.type == ActionType.CLOSE_POSITION
                else action
                for action in non_open_actions
            ]
            actions.extend(non_open_actions)
            for action in non_open_actions:
                if action.type == ActionType.CLOSE_POSITION:
                    self._overlay_update_formal_state_after_base_close(engine, action)
            candidate_actions: list[StrategyAction] = []
            if base_open_action is not None:
                formal_base_action = self._overlay_formal_sota_action(engine, base_open_action)
                if formal_base_action is not None:
                    candidate_actions.append(formal_base_action)
            stable_action = self._overlay_maybe_build_stable_candidate(engine, idx)
            if stable_action is not None:
                candidate_actions.append(stable_action)
            smc_action = self._overlay_maybe_build_smc_candidate(engine, idx)
            if smc_action is not None:
                candidate_actions.append(smc_action)

            if not candidate_actions:
                continue

            for action in sorted(
                candidate_actions,
                key=lambda item: (
                    int(((item.metadata or {}).get("entry_idx", (item.metadata or {}).get("index", idx)) or idx)),
                    {"sota_long": 0, "stable_reverse_short": 1, "smc_short": 2}.get(
                        str((item.metadata or {}).get("overlay_event_type") or candidate_from_action(item).event_type),
                        9,
                    ),
                ),
            ):
                decision = self._sota_overlay_account_lock_pre_open(action, engine, candidate_from_action(action))
                if decision is not None:
                    continue
                overlay_runtime_position = self._overlay_runtime_position_from_action(action)
                if overlay_runtime_position is not None:
                    if base_open_action is not None and hasattr(engine, "_open_action_from_pending"):
                        self._overlay_commit_base_open_action(engine, base_open_action)
                    self._overlay_bind_runtime_position(overlay_runtime_position)
                elif base_open_action is not None:
                    self._overlay_commit_base_open_action(engine, action)
                actions.append(action)
                break
        return actions

    def record_action(self, action: StrategyAction) -> None:
        self.store.append_action(action.timestamp, action.type.value, asdict(action))

    def execute_action(self, action: StrategyAction, engine: Any) -> dict[str, Any]:
        self.record_action(action)
        if action.type == ActionType.HOLD:
            return {"status": "ignored", "reason": "hold"}
        if action.type in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT} and self._telegram_open_paused():
            if self._shadow_gate_enabled():
                state = self._load_shadow_gate_state(engine)
                state["real_position_open"] = False
                state["real_position_direction"] = None
                state["paper_entry_time"] = action.timestamp
                self._shadow_append_event(
                    state,
                    {
                        "time": action.timestamp,
                        "event": "skip_open",
                        "reason": "telegram_open_paused",
                        "direction": action.direction,
                    },
                )
                self._save_shadow_gate_state(state)
            return {
                "status": "telegram_paused_skipped_open",
                "action": action.type.value,
                "direction": action.direction,
                "reason": "telegram_open_paused",
            }
        shadow_decision = self._shadow_gate_pre_execute(action, engine)
        if shadow_decision is not None:
            if action.type == ActionType.CLOSE_POSITION:
                self._clear_sota_overlay_open_candidate()
            return shadow_decision
        if action.type == ActionType.UPDATE_STOP:
            if self.config.mode != "live" or not self.config.enable_exchange_brackets:
                return {"status": "recorded_only", "action": action.type.value, "stop_price": action.stop_price}
            return self._amend_exchange_brackets(action, engine)

        overlay_candidate = candidate_from_action(action) if action.type in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT} else None
        account_lock = self._sota_overlay_account_lock_pre_open(action, engine, overlay_candidate)
        if account_lock is not None:
            return account_lock

        sizing = self._resolve_order_sizing(action, engine)
        if sizing.get("status") != "ok":
            return sizing
        overlay_skipped_dynamic = self._overlay_should_skip_dynamic_high_leverage(action)
        sizing, dynamic_decision = self._dynamic_high_leverage_pre_open(action, sizing, engine)
        if dynamic_decision is not None:
            return dynamic_decision
        high_leverage_decision = self._high_leverage_guard_pre_open(action, sizing)
        if high_leverage_decision is not None:
            return high_leverage_decision

        if self.config.mode == "paper":
            if action.type in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}:
                if overlay_skipped_dynamic:
                    position = getattr(engine, "position", None)
                    if position is not None:
                        effective_leverage = self._action_configured_leverage(action)
                        requested_notional = (
                            self._safe_float(sizing.get("risk_based_notional_usdt"))
                            or self._safe_float((action.metadata or {}).get("risk_based_notional"))
                            or self._safe_float(sizing.get("notional_usdt"))
                            or self._safe_float((action.metadata or {}).get("notional"))
                        )
                        setattr(position, "execution_effective_leverage", round(effective_leverage, 6))
                        setattr(position, "execution_risk_mode", "overlay_fixed")
                        setattr(
                            position,
                            "execution_leverage_reasons",
                            [f"overlay_fixed:{str((action.metadata or {}).get('candidate_event_type') or (action.metadata or {}).get('overlay_event_type') or 'overlay')}"],
                        )
                        setattr(position, "execution_requested_notional", requested_notional)
                        setattr(position, "execution_target_notional", requested_notional)
                if not (self._overlay_formal_fixed_shadow_enabled() and bool((action.metadata or {}).get("overlay_formal_fixed"))):
                    self._shadow_gate_mark_real_position(True, action, "paper_open_accepted")
                self._save_sota_overlay_open_candidate(overlay_candidate)
                overlay_runtime_position = self._overlay_runtime_position_from_action(action)
                if overlay_runtime_position is not None:
                    self._overlay_bind_runtime_position(overlay_runtime_position)
            if action.type == ActionType.CLOSE_POSITION:
                if not (self._overlay_formal_fixed_shadow_enabled() and bool((action.metadata or {}).get("overlay_formal_fixed"))):
                    self._shadow_gate_after_close(action, engine)
                if not self._overlay_should_skip_dynamic_high_leverage(action):
                    self._dynamic_high_leverage_after_close(action, engine)
                self._clear_sota_overlay_open_candidate()
                self._overlay_post_execute_runtime_update(
                    engine,
                    action,
                    int((action.metadata or {}).get("index", -1) or -1),
                )
            return {
                "status": "paper_recorded",
                "action": action.type.value,
                "amount": sizing.get("amount"),
                "order_unit": sizing.get("order_unit"),
                "notional_usdt": sizing.get("notional_usdt"),
                "expected_notional_usdt": sizing.get("expected_notional_usdt"),
                "balance_source": sizing.get("balance_source"),
                "position_size_pct": self.config.position_size_pct,
                "overlay_skipped_dynamic_high_leverage": overlay_skipped_dynamic,
                "dynamic_high_leverage": sizing.get("dynamic_high_leverage"),
            }

        if action.type in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}:
            side = "buy" if action.type == ActionType.OPEN_LONG else "sell"
            pos_side = "long" if action.type == ActionType.OPEN_LONG else "short"
            order_params = {"tdMode": self.config.margin_mode, "posSide": pos_side}
            attach_algo_client_id = None
            if self.config.enable_exchange_brackets:
                attach_algo_client_id = self._generate_attach_algo_client_id()
                order_params.update(self._build_exchange_bracket_params(action, attach_algo_client_id))
            order = self.client.create_order(
                self.config.symbol,
                "market",
                side,
                sizing["amount"],
                params=order_params,
            )
            observed = self._wait_for_position_state(pos_side, expect_open=True, reference_price=action.entry_price)
            direction = "做多" if action.type == ActionType.OPEN_LONG else "做空"
            self._apply_open_execution_metadata(engine, order, observed, attach_algo_client_id)
            if observed["contracts"] <= 0:
                self._shadow_gate_mark_real_position(False, action, "open_unconfirmed")
                self._send_telegram(
                    "\n".join(
                        [
                            "[开仓异常]",
                            f"方向: {direction}",
                            f"标的: {self.config.symbol}",
                            "订单已提交，但未确认到持仓",
                            f"计划下单: {sizing['amount']:.4f} {sizing['order_unit']} (~{sizing['expected_notional_usdt']:.2f}U)",
                            f"订单: {order.get('id')}",
                            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        ]
                    )
                )
                return {"status": "submitted_but_unconfirmed", "order": order, "observed_position": observed, **sizing}
            self._shadow_gate_mark_real_position(True, action, "open_confirmed")
            self._save_sota_overlay_open_candidate(overlay_candidate)
            overlay_runtime_position = self._overlay_runtime_position_from_action(action)
            if overlay_runtime_position is not None:
                self._overlay_bind_runtime_position(overlay_runtime_position)
            dynamic_info = sizing.get("dynamic_high_leverage") if isinstance(sizing.get("dynamic_high_leverage"), dict) else {}
            signal_label = self._overlay_event_label(overlay_candidate.event_type) if overlay_candidate is not None else "-"
            open_lines = [
                "[开仓已确认]",
                f"信号: {signal_label}",
                f"方向: {direction}",
                f"标的: {self.config.symbol}",
                f"成交: {observed['contracts']:.4f} 张 (~{observed['notional_usdt']:.2f}U)",
                f"目标仓位: {sizing['amount']:.4f} {sizing['order_unit']} (~{sizing['expected_notional_usdt']:.2f}U)",
                f"杠杆: {self.config.leverage}x",
            ]
            if dynamic_info:
                open_lines.extend(
                    [
                        f"有效杠杆: {self._format_optional_leverage(dynamic_info.get('effective_leverage'))}",
                        f"动态档位: {dynamic_info.get('risk_mode') or '-'}",
                        f"压仓原因: {self._dynamic_leverage_reason_text(dynamic_info.get('leverage_reasons') if isinstance(dynamic_info.get('leverage_reasons'), list) else [])}",
                    ]
                )
                if sizing.get("risk_based_notional_usdt") is not None:
                    open_lines.append(
                        "理论/实际仓位: "
                        f"{self._format_optional_usdt(sizing.get('risk_based_notional_usdt'), digits=0)} "
                        f"-> {self._format_optional_usdt(observed.get('notional_usdt'), digits=0)}"
                    )
            open_lines.extend(
                [
                    f"入场: {action.entry_price:.1f}" if action.entry_price is not None else "入场: -",
                    f"止损: {action.stop_price:.1f}" if action.stop_price is not None else "止损: -",
                    f"止盈: {action.target_price:.1f}" if action.target_price is not None else "止盈: -",
                    f"订单: {order.get('id')}",
                    f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ]
            )
            self._send_telegram(
                "\n".join(open_lines)
            )
            return {"status": "submitted", "order": order, "observed_position": observed, **sizing}

        if action.type == ActionType.CLOSE_POSITION:
            side = "sell" if action.direction == "BULL" else "buy"
            pos_side = "long" if action.direction == "BULL" else "short"
            order = self.client.create_order(
                self.config.symbol,
                "market",
                side,
                sizing["amount"],
                params={"reduceOnly": True, "tdMode": self.config.margin_mode, "posSide": pos_side},
            )
            direction = "多仓" if action.direction == "BULL" else "空仓"
            observed = self._wait_for_position_state(pos_side, expect_open=False, reference_price=action.exit_price)
            if observed["contracts"] > 0:
                self._send_telegram(
                    "\n".join(
                        [
                            "[平仓异常]",
                            f"方向: 平{direction}",
                            f"标的: {self.config.symbol}",
                            "订单已提交，但仓位仍存在",
                            f"剩余: {observed['contracts']:.4f} 张 (~{observed['notional_usdt']:.2f}U)",
                            f"订单: {order.get('id')}",
                            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        ]
                    )
                )
                return {"status": "submitted_but_unconfirmed", "order": order, "observed_position": observed, **sizing}
            if not (self._overlay_formal_fixed_shadow_enabled() and bool((action.metadata or {}).get("overlay_formal_fixed"))):
                self._shadow_gate_after_close(action, engine)
            if not self._overlay_should_skip_dynamic_high_leverage(action):
                self._dynamic_high_leverage_after_close(action, engine)
            self._clear_sota_overlay_open_candidate()
            self._overlay_post_execute_runtime_update(
                engine,
                action,
                int((action.metadata or {}).get("index", -1) or -1),
            )
            self._send_telegram(
                "\n".join(
                    [
                        "[平仓已确认]",
                        f"方向: 平{direction}",
                        f"标的: {self.config.symbol}",
                        f"平仓: {sizing['amount']:.4f} {sizing['order_unit']}",
                        f"退出价: {action.exit_price:.1f}" if action.exit_price is not None else "退出价: -",
                        f"原因: {action.reason or '-'}",
                        f"订单: {order.get('id')}",
                        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    ]
                )
            )
            return {"status": "submitted", "order": order, "observed_position": observed, **sizing}

        return {"status": "recorded_only", "action": action.type.value}

    def _sota_overlay_account_lock_pre_open(
        self,
        action: StrategyAction,
        engine: Any,
        candidate: Any | None = None,
    ) -> dict[str, Any] | None:
        if action.type not in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}:
            return None

        if candidate is None:
            candidate = candidate_from_action(action)
        exchange_long_contracts = 0.0
        exchange_short_contracts = 0.0
        if self.config.mode == "live":
            try:
                long_state = self._fetch_position_state("long", reference_price=action.entry_price)
                short_state = self._fetch_position_state("short", reference_price=action.entry_price)
            except Exception as exc:
                return {
                    "status": "sota_overlay_account_lock_error",
                    "action": action.type.value,
                    "direction": action.direction,
                    "reason": "account_state_unavailable",
                    "error": str(exc),
                }
            exchange_long_contracts = float(long_state.get("contracts", 0.0) or 0.0)
            exchange_short_contracts = float(short_state.get("contracts", 0.0) or 0.0)

        decision = account_lock_decision(
            candidate,
            local_position_open=self._local_position_blocks_new_open(action, engine),
            exchange_long_contracts=exchange_long_contracts,
            exchange_short_contracts=exchange_short_contracts,
            blocking_candidate=self._load_sota_overlay_open_candidate(),
        )
        self.store.append_action(action.timestamp, "SOTA_OVERLAY_LOCK", decision)
        if decision["decision"] == "accepted":
            return None
        if action.type in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}:
            self._send_telegram(
                "\n".join(
                    [
                        "[Overlay拦截开仓]",
                        f"信号: {self._overlay_event_label(candidate.event_type)}",
                        f"方向: {self._direction_label(candidate.direction)}",
                        f"原因: {self._overlay_decision_text(decision)}",
                        f"锁仓: {self._overlay_candidate_text(self._load_sota_overlay_open_candidate())}",
                        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    ]
                )
            )
        return {
            "status": "sota_overlay_skipped_open",
            "action": action.type.value,
            "direction": action.direction,
            "reason": decision["reason"],
            "decision": decision,
        }

    def _local_position_blocks_new_open(self, action: StrategyAction, engine: Any) -> bool:
        if self._overlay_formal_fixed_shadow_enabled():
            overlay_position = self._load_overlay_runtime_position()
            if overlay_position is not None:
                return not (
                    getattr(overlay_position, "entry_time", None) == action.timestamp
                    and getattr(overlay_position, "direction", None) == action.direction
                )
            state = self._load_overlay_formal_state(engine)
            active_entry_idx = state.get("active_sota_entry_idx")
            if active_entry_idx is not None:
                metadata = action.metadata or {}
                action_entry_idx = metadata.get("entry_idx", metadata.get("index"))
                try:
                    same_entry = int(active_entry_idx) == int(action_entry_idx)
                except (TypeError, ValueError):
                    same_entry = False
                same_direction = action.direction == Direction.BULL
                return not (same_entry and same_direction)
            shadow_state = self._load_shadow_gate_state(engine)
            return bool(shadow_state.get("real_position_open"))
        position = getattr(engine, "position", None)
        if position is None:
            position = self._managed_local_position(engine)
        if position is None:
            return False
        if action.type not in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}:
            return True
        return not (
            getattr(position, "entry_time", None) == action.timestamp
            and getattr(position, "direction", None) == action.direction
        )

    def _save_sota_overlay_open_candidate(self, candidate: Any) -> None:
        if candidate is None:
            return
        self.store.set_value(
            "sota_overlay_open_candidate",
            json.dumps(
                {
                    "event_type": candidate.event_type,
                    "direction": candidate.direction,
                    "entry_idx": candidate.entry_idx,
                    "exit_idx": candidate.exit_idx,
                    "entry_time": candidate.entry_time,
                    "exit_time": candidate.exit_time,
                    "return_rate": candidate.return_rate,
                    "metadata": candidate.metadata,
                },
                ensure_ascii=False,
            ),
        )

    def _load_sota_overlay_open_candidate(self) -> Any | None:
        from strategy.sota_overlay_state import OverlayCandidate

        raw = self.store.get_value("sota_overlay_open_candidate")
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        if not payload.get("event_type"):
            return None
        return OverlayCandidate(
            event_type=str(payload.get("event_type") or "unknown"),
            direction=payload.get("direction"),
            entry_idx=payload.get("entry_idx"),
            exit_idx=payload.get("exit_idx"),
            entry_time=payload.get("entry_time"),
            exit_time=payload.get("exit_time"),
            return_rate=payload.get("return_rate"),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        )

    def _clear_sota_overlay_open_candidate(self) -> None:
        self.store.set_value("sota_overlay_open_candidate", "{}")

    def _live_overlay_enabled(self) -> bool:
        return bool(
            getattr(self.config, "enable_live_overlay_strategy", False)
            or getattr(self.config, "enable_live_candidate_arbitration", False)
        )

    def _stable_live_enabled(self) -> bool:
        return self._live_overlay_enabled() and bool(getattr(self.config, "enable_stable_reverse_short_live", True))

    def _smc_live_enabled(self) -> bool:
        return self._live_overlay_enabled() and bool(getattr(self.config, "enable_smc_short_live", True))

    def _overlay_formal_fixed_shadow_enabled(self) -> bool:
        return self._live_overlay_enabled() and bool(getattr(self.config, "live_overlay_use_formal_fixed_shadow", True))

    def _overlay_formal_state_default(self, engine: Any | None = None) -> dict[str, Any]:
        capital = float(getattr(engine, "capital", 0.0) or 0.0) if engine is not None else 0.0
        fixed_state = FixedStructureState(
            capital=capital,
            peak=capital,
            signal_health_returns=[],
        ).to_dict()
        return {
            "fixed": fixed_state,
            "shadow": {
                "capital": capital,
                "drawdown_peak": capital,
                "loss_streak": 0,
                "pause_until_ts": 0.0,
                "day_start_capital": {},
                "day_pnl": {},
                "events": [],
            },
            "last_formal_event": None,
            "last_shadow_event": None,
            "last_trade_key": None,
            "active_sota_entry_idx": None,
            "active_sota_entry_time": None,
            "warmed_until_idx": None,
            "warmed_until_time": None,
        }

    def _load_overlay_formal_state(self, engine: Any | None = None) -> dict[str, Any]:
        raw = self.store.get_value("live_overlay_formal_state")
        if not raw:
            return self._overlay_formal_state_default(engine)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return self._overlay_formal_state_default(engine)
        default = self._overlay_formal_state_default(engine)
        if isinstance(payload, dict):
            default.update(payload)
        shadow = default.get("shadow")
        if not isinstance(shadow, dict):
            shadow = self._overlay_formal_state_default(engine)["shadow"]
            default["shadow"] = shadow
        shadow.setdefault("events", [])
        shadow.setdefault("day_start_capital", {})
        shadow.setdefault("day_pnl", {})
        fixed = default.get("fixed")
        if not isinstance(fixed, dict):
            default["fixed"] = self._overlay_formal_state_default(engine)["fixed"]
        return default

    def _save_overlay_formal_state(self, state: dict[str, Any]) -> None:
        self.store.set_value("live_overlay_formal_state", json.dumps(state, ensure_ascii=False))

    def _overlay_rebuild_formal_state_from_history(self, engine: Any, end_idx: int) -> dict[str, Any]:
        state = self._overlay_formal_state_default(engine)
        sim_engine = ScalpRobustEngine(
            list(engine.c4h),
            list(engine.c15m),
            list(engine.mapping),
            engine.precomputed,
            self.config.to_scalp_strategy_config(),
        )
        if hasattr(engine, "_regime_switch_cache"):
            sim_engine._regime_switch_cache = deepcopy(getattr(engine, "_regime_switch_cache", {}))
        if hasattr(engine, "_regime_feature_cache"):
            sim_engine._regime_feature_cache = deepcopy(getattr(engine, "_regime_feature_cache", {}))
        sim_engine.evaluate_range(100, max(101, int(end_idx) + 1))
        for trade in getattr(sim_engine, "trades", []):
            if int(getattr(trade, "exit_idx", -1) or -1) > int(end_idx):
                continue
            if str(getattr(trade, "regime_label", "") or "") in {"stable_reverse_short", "smc_short"}:
                continue
            fixed_state = FixedStructureState.from_dict(state.get("fixed"), float(state.get("fixed", {}).get("capital", 0.0) or 0.0))
            formal_trade = self._overlay_enriched_trade_for_formal_step(sim_engine, trade)
            next_fixed_state, formal_event, _decision = fixed_structure_step(formal_trade, fixed_state, FIXED_STRUCTURE_PARAMS)
            state["fixed"] = next_fixed_state.to_dict()
            state["last_trade_key"] = self._overlay_trade_key(trade)
            state["last_formal_event"] = formal_event
            if formal_event is not None:
                state["last_shadow_event"] = self._overlay_shadow_accept_event(state, formal_event)
            else:
                state["last_shadow_event"] = None
        state["active_sota_entry_idx"] = None
        state["active_sota_entry_time"] = None
        state["warmed_until_idx"] = int(end_idx)
        state["warmed_until_time"] = self._overlay_action_timestamp(engine, int(end_idx)) if 0 <= int(end_idx) < len(engine.c15m) else None
        return state

    def _overlay_ensure_formal_state_warmup(self, engine: Any, start_idx: int) -> None:
        if not self._overlay_formal_fixed_shadow_enabled():
            return
        if self.store.get_value("live_overlay_formal_state"):
            return
        if not bool(getattr(self.config, "live_overlay_rebuild_formal_state_from_history", False)):
            state = self._overlay_formal_state_default(engine)
            state["initialized_without_history"] = True
            state["initialized_at_idx"] = int(start_idx)
            state["initialized_at_time"] = self._overlay_action_timestamp(engine, int(start_idx)) if 0 <= int(start_idx) < len(engine.c15m) else None
            self._save_overlay_formal_state(state)
            return
        warm_end_idx = max(100, int(start_idx) - 1)
        if warm_end_idx <= 100:
            self._save_overlay_formal_state(self._overlay_formal_state_default(engine))
            return
        state = self._overlay_rebuild_formal_state_from_history(engine, warm_end_idx)
        self._save_overlay_formal_state(state)

    def _overlay_trade_key(self, trade: Any) -> str:
        return (
            f"{getattr(trade, 'entry_time', '')}|"
            f"{getattr(trade, 'exit_time', '')}|"
            f"{getattr(trade, 'entry_idx', '')}|"
            f"{getattr(trade, 'exit_idx', '')}|"
            f"{getattr(trade, 'direction', '')}"
        )

    def _overlay_utc_timestamp(self, value: Any) -> pd.Timestamp:
        timestamp = pd.Timestamp(str(value))
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    def _overlay_shadow_accept_event(self, state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any] | None:
        shadow = state.get("shadow")
        if not isinstance(shadow, dict):
            return None
        entry_time = self._overlay_utc_timestamp(event["entry_time"])
        exit_time = self._overlay_utc_timestamp(event["exit_time"])
        pause_until_ts = float(shadow.get("pause_until_ts", 0.0) or 0.0)
        if entry_time.timestamp() < pause_until_ts:
            return None

        capital_before = float(shadow.get("capital", 0.0) or 0.0)
        trade_return = float(event["return"])
        pnl = capital_before * trade_return
        capital = capital_before + pnl
        shadow["capital"] = capital
        shadow["drawdown_peak"] = max(float(shadow.get("drawdown_peak", capital) or capital), capital)
        accepted = dict(event)
        accepted["shadow_capital"] = capital
        events = shadow.get("events")
        if not isinstance(events, list):
            events = []
        events.append(accepted)
        shadow["events"] = events[-500:]

        exit_day = exit_time.normalize().strftime("%Y-%m-%d")
        day_start_capital = shadow.get("day_start_capital")
        day_pnl = shadow.get("day_pnl")
        if not isinstance(day_start_capital, dict):
            day_start_capital = {}
        if not isinstance(day_pnl, dict):
            day_pnl = {}
        if exit_day not in day_start_capital:
            day_start_capital[exit_day] = capital_before
            day_pnl[exit_day] = 0.0
        day_pnl[exit_day] = float(day_pnl.get(exit_day, 0.0) or 0.0) + pnl
        shadow["day_start_capital"] = day_start_capital
        shadow["day_pnl"] = day_pnl

        if pnl > 0:
            shadow["loss_streak"] = 0
        else:
            shadow["loss_streak"] = int(shadow.get("loss_streak", 0) or 0) + 1

        daily_stop = float(self.config.shadow_daily_loss_stop_pct or 0.0)
        start_capital = float(day_start_capital.get(exit_day, 0.0) or 0.0)
        if daily_stop > 0 and start_capital > 0:
            daily_loss_pct = -float(day_pnl[exit_day]) / start_capital * 100.0
            if daily_loss_pct >= daily_stop:
                shadow["pause_until_ts"] = max(
                    float(shadow.get("pause_until_ts", 0.0) or 0.0),
                    self._shadow_next_utc_day_ts(exit_time.to_pydatetime()),
                )
        streak_stop = int(self.config.shadow_consecutive_loss_stop or 0)
        if streak_stop > 0 and int(shadow.get("loss_streak", 0) or 0) >= streak_stop:
            shadow["pause_until_ts"] = max(
                float(shadow.get("pause_until_ts", 0.0) or 0.0),
                self._shadow_next_utc_day_ts(exit_time.to_pydatetime()),
            )
            shadow["loss_streak"] = 0
        dd_stop = float(self.config.shadow_equity_drawdown_stop_pct or 0.0)
        peak = float(shadow.get("drawdown_peak", capital) or capital)
        if dd_stop > 0 and peak > 0:
            drawdown_pct = (peak - capital) / peak * 100.0
            if drawdown_pct >= dd_stop:
                shadow["pause_until_ts"] = max(
                    float(shadow.get("pause_until_ts", 0.0) or 0.0),
                    self._shadow_cooldown_until_ts(exit_time.to_pydatetime(), int(self.config.shadow_equity_drawdown_cooldown_days or 0)),
                )
                shadow["drawdown_peak"] = capital
                shadow["loss_streak"] = 0
        state["shadow"] = shadow
        state["last_shadow_event"] = accepted
        return accepted

    def _overlay_update_formal_state_after_base_close(self, engine: Any, action: StrategyAction) -> None:
        if not self._overlay_formal_fixed_shadow_enabled():
            return
        latest_trade = engine.trades[-1] if getattr(engine, "trades", None) else None
        if latest_trade is None:
            return
        if str(getattr(latest_trade, "regime_label", "") or "") in {"stable_reverse_short", "smc_short"}:
            return
        trade_key = self._overlay_trade_key(latest_trade)
        state = self._load_overlay_formal_state(engine)
        active_entry_idx = state.get("active_sota_entry_idx")
        if active_entry_idx is None or int(active_entry_idx) != int(getattr(latest_trade, "entry_idx", -1) or -1):
            return
        if trade_key == str(state.get("last_trade_key") or ""):
            return
        fixed_state = FixedStructureState.from_dict(state.get("fixed"), float(getattr(engine, "capital", 0.0) or 0.0))
        formal_trade = self._overlay_enriched_trade_for_formal_step(engine, latest_trade)
        next_fixed_state, formal_event, _decision = fixed_structure_step(formal_trade, fixed_state, FIXED_STRUCTURE_PARAMS)
        state["fixed"] = next_fixed_state.to_dict()
        state["last_trade_key"] = trade_key
        state["active_sota_entry_idx"] = None
        state["active_sota_entry_time"] = None
        state["last_formal_event"] = formal_event
        if formal_event is not None:
            shadow_event = self._overlay_shadow_accept_event(state, formal_event)
            state["last_shadow_event"] = shadow_event
        else:
            state["last_shadow_event"] = None
        self._save_overlay_formal_state(state)

    def _overlay_mark_formal_close_action(self, engine: Any, action: StrategyAction) -> StrategyAction:
        if not self._overlay_formal_fixed_shadow_enabled() or action.type != ActionType.CLOSE_POSITION:
            return action
        latest_trade = engine.trades[-1] if getattr(engine, "trades", None) else None
        if latest_trade is None:
            return action
        state = self._load_overlay_formal_state(engine)
        active_entry_idx = state.get("active_sota_entry_idx")
        if active_entry_idx is None or int(active_entry_idx) != int(getattr(latest_trade, "entry_idx", -1) or -1):
            return action
        metadata = dict(action.metadata or {})
        metadata["overlay_formal_fixed"] = True
        metadata["overlay_event_type"] = "sota_long"
        return StrategyAction(
            type=action.type,
            timestamp=action.timestamp,
            direction=action.direction,
            entry_price=action.entry_price,
            exit_price=action.exit_price,
            stop_price=action.stop_price,
            target_price=action.target_price,
            reason=action.reason,
            metadata=metadata,
        )

    def _overlay_enriched_trade_for_formal_step(self, engine: Any, trade: Any) -> pd.Series:
        entry_idx = int(getattr(trade, "entry_idx", -1) or -1)
        features: dict[str, Any] = {}
        if entry_idx >= 0 and hasattr(engine, "_regime_features_for_idx"):
            try:
                features = engine._regime_features_for_idx(entry_idx) or {}
            except Exception:
                features = {}
        return pd.Series(
            {
                "entry_time": getattr(trade, "entry_time", ""),
                "exit_time": getattr(trade, "exit_time", ""),
                "entry_idx": getattr(trade, "entry_idx", None),
                "exit_idx": getattr(trade, "exit_idx", None),
                "exit_reason": getattr(trade, "exit_reason", ""),
                "rr_ratio": getattr(trade, "rr_ratio", 0.0),
                "pnl": getattr(trade, "pnl", 0.0),
                "notional": getattr(trade, "notional", 0.0),
                "quantity": getattr(trade, "quantity", 0.0),
                "direction": getattr(trade, "direction", ""),
                "entry_price": getattr(trade, "entry_price", 0.0),
                "exit_price": getattr(trade, "exit_price", 0.0),
                "initial_stop_price": getattr(trade, "initial_stop_price", 0.0),
                "regime_label": getattr(trade, "regime_label", "") or "",
                "trail_style": getattr(trade, "trail_style", "") or "",
                "pressure_target_applied": getattr(trade, "pressure_target_applied", False),
                "pressure_target_source": getattr(trade, "pressure_target_source", None),
                "pressure_target_level": getattr(trade, "pressure_target_level", None),
                "pressure_target_rr": getattr(trade, "pressure_target_rr", None),
                "pressure_target_min_rr": getattr(trade, "pressure_target_min_rr", None),
                "pressure_target_dynamic_reason": getattr(trade, "pressure_target_dynamic_reason", None),
                "pressure_target_update_idx": getattr(trade, "pressure_target_update_idx", None),
                "pressure_touch_lock_applied": getattr(trade, "pressure_touch_lock_applied", False),
                "pressure_touch_lock_source": getattr(trade, "pressure_touch_lock_source", None),
                "pressure_touch_lock_level": getattr(trade, "pressure_touch_lock_level", None),
                "pressure_touch_lock_rr": getattr(trade, "pressure_touch_lock_rr", None),
                "pressure_touch_lock_update_idx": getattr(trade, "pressure_touch_lock_update_idx", None),
                "feature_adx": float(features.get("adx", 0.0) or 0.0),
                "feature_momentum": float(features.get("momentum", 0.0) or 0.0),
                "feature_ema_gap": float(features.get("ema_gap", 0.0) or 0.0),
                "feature_bullish_structure": bool(features.get("bullish_structure", False)),
                "feature_bearish_structure": bool(features.get("bearish_structure", False)),
            }
        )

    def _overlay_runtime_state_default(self) -> dict[str, Any]:
        return {
            "position": None,
            "last_managed_idx": None,
        }

    def _load_overlay_runtime_state(self) -> dict[str, Any]:
        raw = self.store.get_value("live_overlay_runtime_state")
        if not raw:
            return self._overlay_runtime_state_default()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return self._overlay_runtime_state_default()
        default = self._overlay_runtime_state_default()
        if isinstance(payload, dict):
            default.update(payload)
        return default

    def _save_overlay_runtime_state(self, state: dict[str, Any]) -> None:
        self.store.set_value("live_overlay_runtime_state", json.dumps(state, ensure_ascii=False))

    def _load_overlay_runtime_position(self) -> OverlayRuntimePosition | None:
        state = self._load_overlay_runtime_state()
        payload = state.get("position")
        if not isinstance(payload, dict):
            return None
        try:
            return OverlayRuntimePosition(**payload)
        except TypeError:
            return None

    def _save_overlay_runtime_position(self, position: OverlayRuntimePosition | None, *, last_managed_idx: int | None = None) -> None:
        state = self._load_overlay_runtime_state()
        state["position"] = asdict(position) if position is not None else None
        if last_managed_idx is not None:
            state["last_managed_idx"] = int(last_managed_idx)
        self._save_overlay_runtime_state(state)

    def _clear_overlay_runtime_position(self, *, last_managed_idx: int | None = None) -> None:
        self._save_overlay_runtime_position(None, last_managed_idx=last_managed_idx)

    def _managed_local_position(self, engine: Any) -> Any | None:
        local_position = getattr(engine, "position", None)
        if local_position is not None:
            return local_position
        overlay_position = self._load_overlay_runtime_position()
        if overlay_position is None:
            return None
        return SimpleNamespace(
            direction=overlay_position.direction,
            entry_time=overlay_position.entry_time,
            quantity=overlay_position.quantity,
            sl_price=overlay_position.sl_price,
            target_price=overlay_position.target_price,
            entry_price=overlay_position.entry_price,
            initial_sl_price=overlay_position.initial_sl_price,
            event_type=overlay_position.event_type,
            entry_idx=overlay_position.entry_idx,
            risk_regime="overlay",
            regime_label=overlay_position.event_type,
            time_based_trailing_enabled=False,
            auto_tit_reason=None,
        )

    def _action_configured_leverage(self, action: StrategyAction | None = None) -> float:
        metadata = (action.metadata or {}) if action is not None else {}
        return float(
            (
                metadata.get("candidate_leverage")
                or metadata.get("overlay_leverage")
                or metadata.get("leverage")
                or self.config.leverage
            )
            or self.config.leverage
        )

    def _action_maintenance_margin_pct(self, action: StrategyAction | None = None) -> float:
        metadata = (action.metadata or {}) if action is not None else {}
        return float(
            (
                metadata.get("overlay_maintenance_margin_pct")
                or metadata.get("maintenance_margin_pct")
                or self.config.high_leverage_maintenance_margin_pct
            )
            or self.config.high_leverage_maintenance_margin_pct
        )

    def _overlay_should_skip_dynamic_high_leverage(self, action: StrategyAction | None = None) -> bool:
        if not bool(getattr(self.config, "overlay_skip_dynamic_high_leverage", False)):
            return False
        if action is None:
            return False
        metadata = action.metadata or {}
        event_type = str(
            metadata.get("overlay_event_type")
            or metadata.get("candidate_event_type")
            or metadata.get("event_type")
            or ""
        ).lower()
        return event_type in {"stable", "stable_reverse_short", "smc", "smc_short"}

    def _overlay_action_timestamp(self, engine: Any, idx: int) -> str:
        return engine._timestamp_for_idx(int(idx))

    def _overlay_capital(self, engine: Any) -> float:
        return float(getattr(engine, "capital", 0.0) or 0.0)

    def _overlay_rr_observation_price(self, candle: Any, direction: str, mode: str) -> float:
        normalized_mode = str(mode or "close").strip().lower()
        if normalized_mode == "close":
            return float(candle.c)
        if normalized_mode == "extreme":
            return float(candle.h) if direction == Direction.BULL else float(candle.l)
        raise ValueError(f"Unsupported overlay RR observation mode: {mode}")

    def _overlay_realized_action_return(
        self,
        *,
        signal_return_pct: float,
        leverage: float,
        allocation: float,
    ) -> dict[str, float]:
        return leveraged_net_return(
            signal_return_pct=float(signal_return_pct),
            leverage=float(leverage),
            position_size_pct=1.0,
            allocation=float(allocation),
            taker_fee_rate=float(self.config.taker_fee_rate),
            slippage_bps=float(self.config.slippage_bps),
        )

    def _overlay_build_open_short_action(
        self,
        *,
        engine: Any,
        idx: int,
        event_type: str,
        signal_entry_price: float | None = None,
        stop_price: float,
        target_price: float,
        target_rr: float | None,
        max_hold_bars: int | None,
        allocation: float,
        leverage: float,
        stop_reason: str | None = None,
        target_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[StrategyAction, OverlayRuntimePosition]:
        candle = engine.c15m[idx]
        entry_price = float(signal_entry_price if signal_entry_price is not None else candle.c)
        capital = self._overlay_capital(engine)
        notional = capital * float(leverage) * float(allocation)
        quantity = notional / entry_price if entry_price > 0 else 0.0
        entry_fee = notional * float(self.config.taker_fee_rate)
        slippage_rate = float(self.config.slippage_bps) / 10_000.0
        filled_entry_price = entry_price * (1.0 - slippage_rate)
        entry_slippage_cost = quantity * abs(filled_entry_price - entry_price)
        risk_points = float(stop_price) - float(entry_price)
        runtime_position = OverlayRuntimePosition(
            event_type=str(event_type),
            direction=Direction.BEAR,
            entry_idx=int(idx),
            entry_time=self._overlay_action_timestamp(engine, idx),
            exit_idx=None,
            target_rr=float(target_rr) if target_rr is not None else None,
            max_hold_bars=int(max_hold_bars) if max_hold_bars is not None else None,
            allocation=float(allocation),
            leverage=float(leverage),
            capital_at_entry=capital,
            signal_entry_price=entry_price,
            entry_price=filled_entry_price,
            sl_price=float(stop_price),
            initial_sl_price=float(stop_price),
            target_price=float(target_price),
            risk_points=float(risk_points),
            quantity=float(quantity),
            notional=float(notional),
            entry_fee=float(entry_fee),
            entry_slippage_cost=float(entry_slippage_cost),
            stop_reason=stop_reason,
            target_reason=target_reason,
            metadata=dict(metadata or {}),
        )
        action = StrategyAction(
            type=ActionType.OPEN_SHORT,
            timestamp=runtime_position.entry_time,
            direction=Direction.BEAR,
            entry_price=runtime_position.entry_price,
            stop_price=runtime_position.sl_price,
            target_price=runtime_position.target_price,
            metadata={
                "index": int(idx),
                "entry_idx": int(idx),
                "overlay_event_type": runtime_position.event_type,
                "exit_idx": runtime_position.exit_idx,
                "position_size_pct": float(allocation),
                "capital_at_entry": capital,
                "notional": runtime_position.notional,
                "max_notional": runtime_position.notional,
                "risk_based_notional": runtime_position.notional,
                "margin_usdt": runtime_position.notional / max(float(leverage), 1.0),
                "quantity": runtime_position.quantity,
                "signal_entry_price": runtime_position.signal_entry_price,
                "target_rr": runtime_position.target_rr,
                "max_hold_bars": runtime_position.max_hold_bars,
                "trail_style": "overlay",
                "risk_regime": "overlay",
                "regime_label": runtime_position.event_type,
                "leverage": float(leverage),
                "overlay_leverage": float(leverage),
                "candidate_leverage": float(leverage),
                "entry_fee": runtime_position.entry_fee,
                "entry_slippage_cost": runtime_position.entry_slippage_cost,
                **(dict(metadata or {})),
            },
        )
        return action, runtime_position

    def _overlay_runtime_position_from_action(self, action: StrategyAction) -> OverlayRuntimePosition | None:
        metadata = action.metadata or {}
        event_type = str(metadata.get("overlay_event_type") or "")
        if event_type not in {"stable_reverse_short", "smc_short"}:
            return None
        try:
            return OverlayRuntimePosition(
                event_type=event_type,
                direction=str(action.direction or Direction.BEAR),
                entry_idx=int(metadata.get("entry_idx", metadata.get("index", 0)) or 0),
                entry_time=str(action.timestamp),
                exit_idx=metadata.get("exit_idx"),
                target_rr=float(metadata["target_rr"]) if metadata.get("target_rr") is not None else None,
                max_hold_bars=int(metadata["max_hold_bars"]) if metadata.get("max_hold_bars") is not None else None,
                allocation=float(metadata.get("position_size_pct", 1.0) or 1.0),
                leverage=float((metadata.get("leverage") or metadata.get("overlay_leverage") or self.config.leverage) or self.config.leverage),
                capital_at_entry=float(metadata.get("capital_at_entry", 0.0) or 0.0),
                signal_entry_price=float(metadata.get("signal_entry_price", action.entry_price or 0.0) or 0.0),
                entry_price=float(action.entry_price or 0.0),
                sl_price=float(action.stop_price or 0.0),
                initial_sl_price=float(action.stop_price or 0.0),
                target_price=float(action.target_price or 0.0),
                risk_points=max(0.0, float(action.stop_price or 0.0) - float(metadata.get("signal_entry_price", action.entry_price or 0.0) or 0.0)),
                quantity=float(metadata.get("quantity", 0.0) or 0.0),
                notional=float(metadata.get("notional", 0.0) or 0.0),
                entry_fee=float(metadata.get("entry_fee", 0.0) or 0.0),
                entry_slippage_cost=float(metadata.get("entry_slippage_cost", 0.0) or 0.0),
                stop_reason="stop_loss",
                target_reason="target_rr" if event_type == "stable_reverse_short" else str(metadata.get("smc_target_reason") or "target_2.0r"),
                smc_case=str(metadata.get("smc_case") or "") or None,
                metadata=dict(metadata),
            )
        except (TypeError, ValueError):
            return None

    def _overlay_build_close_action(
        self,
        *,
        engine: Any,
        position: OverlayRuntimePosition,
        idx: int,
        exit_price: float,
        reason: str,
    ) -> StrategyAction:
        filled_exit_price = float(exit_price) * (1.0 + float(self.config.slippage_bps) / 10_000.0)
        gross_pnl = position.quantity * (position.entry_price - filled_exit_price)
        exit_fee = position.quantity * filled_exit_price * float(self.config.taker_fee_rate)
        fees = position.entry_fee + exit_fee
        slippage_cost = position.entry_slippage_cost + position.quantity * abs(filled_exit_price - float(exit_price))
        pnl = gross_pnl - fees
        if position.capital_at_entry > 0:
            pnl_pct = pnl / position.capital_at_entry
        else:
            pnl_pct = 0.0
        rr_ratio = pnl / (position.risk_points * position.quantity) if position.risk_points > 0 and position.quantity > 0 else 0.0
        return StrategyAction(
            type=ActionType.CLOSE_POSITION,
            timestamp=self._overlay_action_timestamp(engine, idx),
            direction=Direction.BEAR,
            exit_price=filled_exit_price,
            reason=reason,
            metadata={
                "index": int(idx),
                "overlay_event_type": position.event_type,
                "entry_idx": position.entry_idx,
                "gross_pnl": gross_pnl,
                "fees": fees,
                "slippage_cost": slippage_cost,
                "net_pnl": pnl,
                "signal_exit_price": float(exit_price),
                "capital_at_entry": position.capital_at_entry,
                "rr_ratio": rr_ratio,
                "pnl_pct": pnl_pct,
            },
        )

    def _overlay_append_trade(self, engine: Any, position: OverlayRuntimePosition, action: StrategyAction) -> None:
        metadata = action.metadata or {}
        trade = Trade(
            entry_time=position.entry_time,
            exit_time=action.timestamp,
            direction=Direction.BEAR,
            signal_entry_price=position.signal_entry_price,
            entry_price=position.entry_price,
            signal_exit_price=float(metadata.get("signal_exit_price", action.exit_price or 0.0) or 0.0),
            exit_price=float(action.exit_price or 0.0),
            gross_pnl=float(metadata.get("gross_pnl", 0.0) or 0.0),
            fees=float(metadata.get("fees", 0.0) or 0.0),
            slippage_cost=float(metadata.get("slippage_cost", 0.0) or 0.0),
            pnl=float(metadata.get("net_pnl", 0.0) or 0.0),
            pnl_pct=float(metadata.get("pnl_pct", 0.0) or 0.0),
            rr_ratio=float(metadata.get("rr_ratio", 0.0) or 0.0),
            exit_reason=str(action.reason or "overlay_exit"),
            capital_at_entry=position.capital_at_entry,
            notional=position.notional,
            quantity=position.quantity,
            entry_idx=position.entry_idx,
            initial_stop_price=position.initial_sl_price,
            trail_style="overlay",
            risk_regime="overlay",
            regime_label=position.event_type,
            time_based_trailing_enabled=False,
            auto_tit_reason=None,
            exit_idx=int(metadata.get("index", position.entry_idx) or position.entry_idx),
        )
        engine.trades.append(trade)
        engine.exit_reasons[str(action.reason or "overlay_exit")] = int(engine.exit_reasons.get(str(action.reason or "overlay_exit"), 0) or 0) + 1
        engine.capital = max(0.0, float(getattr(engine, "capital", 0.0) or 0.0) + float(metadata.get("net_pnl", 0.0) or 0.0))

    def _overlay_formal_trade_stub_from_action(self, engine: Any, action: StrategyAction) -> pd.Series:
        metadata = action.metadata or {}
        entry_price = float(action.entry_price or 0.0)
        quantity = float(metadata.get("quantity", 0.0) or 0.0)
        notional = float(metadata.get("notional", 0.0) or 0.0)
        return pd.Series(
            {
                "entry_time": action.timestamp,
                "direction": str(action.direction or ""),
                "entry_price": entry_price,
                "initial_stop_price": float(action.stop_price or 0.0),
                "notional": notional,
                "quantity": quantity,
                "regime_label": str(metadata.get("regime_label") or ""),
                "trail_style": str(metadata.get("trail_style") or ""),
                "feature_adx": float(metadata.get("feature_adx", 0.0) or 0.0),
                "feature_momentum": float(metadata.get("feature_momentum", 0.0) or 0.0),
                "feature_ema_gap": float(metadata.get("feature_ema_gap", 0.0) or 0.0),
                "feature_bullish_structure": bool(metadata.get("feature_bullish_structure", False)),
                "feature_bearish_structure": bool(metadata.get("feature_bearish_structure", False)),
            }
        )

    def _overlay_formal_sota_action(self, engine: Any, action: StrategyAction) -> StrategyAction | None:
        if not self._overlay_formal_fixed_shadow_enabled():
            return action
        if action.type != ActionType.OPEN_LONG:
            return None
        shadow = self._load_overlay_formal_state(engine).get("shadow")
        pause_until_ts = float(shadow.get("pause_until_ts", 0.0) or 0.0) if isinstance(shadow, dict) else 0.0
        if pause_until_ts > 0:
            try:
                if self._action_timestamp(action).timestamp() < pause_until_ts:
                    return None
            except ValueError:
                pass
        trade_stub = self._overlay_formal_trade_stub_from_action(engine, action)
        state = self._load_overlay_formal_state(engine)
        fixed_state = FixedStructureState.from_dict(state.get("fixed"), float(getattr(engine, "capital", 0.0) or 0.0))
        decision = fixed_structure_entry_decision(trade_stub, fixed_state, FIXED_STRUCTURE_PARAMS)
        if not bool(decision.get("accepted")):
            return None
        metadata = dict(action.metadata or {})
        metadata["overlay_event_type"] = "sota_long"
        metadata["overlay_formal_fixed"] = True
        metadata["overlay_formal_effective_leverage"] = float(decision.get("effective_leverage", 0.0) or 0.0)
        metadata["overlay_formal_risk_mode"] = decision.get("risk_mode")
        metadata["overlay_formal_leverage_reasons"] = list(decision.get("leverage_reasons") or [])
        metadata["overlay_formal_guard_diagnostics"] = dict(decision.get("failed_breakout_guard_diagnostics") or {})
        effective_leverage = float(decision.get("effective_leverage", 0.0) or 0.0)
        if effective_leverage > 0 and action.entry_price:
            capital_at_entry = float(metadata.get("capital_at_entry", getattr(engine, "capital", 0.0)) or 0.0)
            notional = capital_at_entry * effective_leverage
            quantity = notional / float(action.entry_price)
            metadata["notional"] = notional
            metadata["max_notional"] = notional
            metadata["risk_based_notional"] = notional
            metadata["quantity"] = quantity
            metadata["margin_usdt"] = notional / max(float(self.config.leverage), 1.0)
        return StrategyAction(
            type=action.type,
            timestamp=action.timestamp,
            direction=action.direction,
            entry_price=action.entry_price,
            exit_price=action.exit_price,
            stop_price=action.stop_price,
            target_price=action.target_price,
            reason=action.reason,
            metadata=metadata,
        )

    def _quality_snapshot_from_features(self, direction: str, features: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "direction": direction,
            "feature_adx": features.get("feature_adx", 0.0),
            "feature_momentum": features.get("feature_momentum", 0.0),
            "feature_ema_gap": features.get("feature_ema_gap", 0.0),
            "feature_bullish_structure": features.get("feature_bullish_structure", False),
            "feature_bearish_structure": features.get("feature_bearish_structure", False),
        }
        return quality_snapshot(payload)

    def _stable_guard_would_apply(self, source: dict[str, Any]) -> bool:
        leverage = float(source.get("effective_leverage", 0.0) or 0.0)
        if leverage < float(self.config.dynamic_failed_breakout_guard_min_leverage):
            return False
        if str(source.get("regime_label") or "") != "high_growth":
            return False
        if str(source.get("risk_mode") or "") != "offense":
            return False
        quality = self._quality_snapshot_from_features(str(source.get("direction") or ""), source)
        return int(quality["quality_score"]) < int(self.config.dynamic_failed_breakout_guard_min_quality_score)

    def _stable_source_event_from_trade(self, trade: Trade, close_action: StrategyAction | None = None) -> dict[str, Any]:
        diagnostics = getattr(trade, "execution_guard_diagnostics", None) or {}
        close_metadata = (close_action.metadata or {}) if close_action is not None else {}
        return {
            "entry_time": getattr(trade, "entry_time", None),
            "exit_time": getattr(trade, "exit_time", None),
            "entry_idx": getattr(trade, "entry_idx", None),
            "exit_idx": getattr(trade, "exit_idx", None),
            "direction": getattr(trade, "direction", None),
            "entry_price": float(getattr(trade, "entry_price", 0.0) or 0.0),
            "exit_price": float(getattr(trade, "exit_price", 0.0) or 0.0),
            "initial_stop_price": float(getattr(trade, "initial_stop_price", 0.0) or 0.0),
            "exit_reason": getattr(trade, "exit_reason", None),
            "return": float(getattr(trade, "pnl_pct", 0.0) or 0.0),
            "regime_label": getattr(trade, "regime_label", None),
            "risk_mode": getattr(trade, "execution_risk_mode", None),
            "effective_leverage": getattr(trade, "execution_effective_leverage", None),
            "failed_breakout_guard_applied": any(
                str(reason).startswith("failed_breakout_guard")
                for reason in (getattr(trade, "execution_leverage_reasons", None) or [])
            ),
            "feature_adx": diagnostics.get("feature_adx", 0.0),
            "feature_momentum": diagnostics.get("feature_momentum", 0.0),
            "feature_ema_gap": diagnostics.get("feature_ema_gap", 0.0),
            "feature_bullish_structure": diagnostics.get("feature_bullish_structure", False),
            "feature_bearish_structure": diagnostics.get("feature_bearish_structure", False),
            "last_stop_update_reason": close_metadata.get("last_stop_update_reason"),
            "pressure_target_applied": bool(getattr(trade, "pressure_target_applied", False) or close_metadata.get("pressure_target_applied")),
            "pressure_touch_lock_applied": bool(getattr(trade, "pressure_touch_lock_applied", False) or close_metadata.get("pressure_touch_lock_applied")),
            "time_based_trailing_enabled": bool(getattr(trade, "time_based_trailing_enabled", False)),
        }

    def _stable_selector_allows(self, source: dict[str, Any]) -> bool:
        selector = str(getattr(self.config, "stable_selector", "guarded_weak_loss") or "guarded_weak_loss")
        direction = str(source.get("direction") or "")
        exit_reason = str(source.get("exit_reason") or "")
        return_value = float(source.get("return", 0.0) or 0.0)
        stop_update_reason = str(source.get("last_stop_update_reason") or "")
        pressure_target_applied = bool(source.get("pressure_target_applied"))
        pressure_touch_lock_applied = bool(source.get("pressure_touch_lock_applied"))
        time_based_trailing_enabled = bool(source.get("time_based_trailing_enabled"))

        if selector.endswith("_profit_reverse"):
            base = (
                direction == Direction.BULL
                and exit_reason == "stop_loss"
                and return_value > 0.0
                and str(source.get("regime_label") or "") == "high_growth"
                and str(source.get("risk_mode") or "") == "offense"
            )
            if not base:
                return False
            if selector == "trailing_stop_profit_reverse":
                return True
            if selector == "trailing_stage_profit_reverse":
                return stop_update_reason.startswith("trail_stage_")
            if selector == "trailing_atr_profit_reverse":
                return stop_update_reason == "atr_trail"
            if selector == "trailing_pressure_profit_reverse":
                return stop_update_reason == "pressure_level_trail" or pressure_target_applied or pressure_touch_lock_applied
            if selector == "trailing_pressure_touch_lock_profit_reverse":
                return pressure_touch_lock_applied
            if selector == "trailing_time_enabled_profit_reverse":
                return time_based_trailing_enabled
            if selector == "plain_stop_profit_reverse":
                return not stop_update_reason
        if selector == "bull_high_growth_offense_loss":
            return (
                direction == Direction.BULL
                and str(source.get("regime_label") or "") == "high_growth"
                and str(source.get("risk_mode") or "") == "offense"
                and exit_reason == "stop_loss"
                and return_value < 0.0
            )
        if not (
            direction == Direction.BULL
            and str(source.get("regime_label") or "") == "high_growth"
            and str(source.get("risk_mode") or "") == "offense"
        ):
            return False
        quality = self._quality_snapshot_from_features(direction, source)
        guarded = bool(source.get("failed_breakout_guard_applied")) or self._stable_guard_would_apply(source)
        weak_quality = int(quality["quality_score"]) <= int(getattr(self.config, "stable_max_quality_score", 1))
        if selector == "guarded_weak":
            return guarded
        if selector == "guarded_weak_loss":
            return guarded and return_value < 0.0
        if selector == "weak_quality":
            return weak_quality
        if selector == "weak_quality_loss":
            return weak_quality and return_value < 0.0
        if selector == "weak_or_guarded":
            return weak_quality or guarded
        if selector == "weak_or_guarded_loss":
            return (weak_quality or guarded) and return_value < 0.0
        if selector == "all_high_growth_offense":
            return True
        if selector == "actual_loss_oracle":
            return return_value < 0.0
        return False

    def _stable_reverse_short_candidate(
        self,
        engine: Any,
        close_action: StrategyAction | None,
        latest_closed_idx: int,
    ) -> dict[str, Any] | None:
        if not bool(getattr(self.config, "enable_stable_reverse_short_live", True)):
            return None
        if not getattr(engine, "trades", None):
            return None
        trade = engine.trades[-1]
        if trade.direction != Direction.BULL or str(trade.exit_reason or "") != "stop_loss":
            return None
        source = self._stable_source_event_from_trade(trade, close_action)
        if not self._stable_selector_allows(source):
            return None
        entry_price = float(source.get("exit_price", 0.0) or 0.0)
        source_entry = float(source.get("entry_price", 0.0) or 0.0)
        source_stop = float(source.get("initial_stop_price", 0.0) or 0.0)
        if entry_price <= 0 or source_entry <= 0 or source_stop <= 0:
            return None
        source_stop_pct = abs(source_entry - source_stop) / source_entry
        stop_pct = source_stop_pct * float(self.config.live_overlay_stable_stop_multiplier)
        if stop_pct <= 0 or stop_pct * 100.0 > float(self.config.live_overlay_stable_max_short_stop_pct):
            return None
        stop_price = entry_price * (1.0 + stop_pct)
        target_price = entry_price * (1.0 - stop_pct * float(self.config.live_overlay_stable_target_rr))
        if target_price <= 0 or stop_price <= entry_price:
            return None
        leverage = float(self.config.live_overlay_stable_leverage)
        position_size_pct = float(getattr(self.config, "stable_position_size_pct", self.config.live_overlay_stable_allocation) or 0.0)
        notional = max(float(getattr(engine, "capital", 0.0) or 0.0), 0.0) * leverage * position_size_pct
        return {
            "event_type": "stable_reverse_short",
            "entry_idx": latest_closed_idx,
            "direction": Direction.BEAR,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "target_rr": float(self.config.live_overlay_stable_target_rr),
            "max_hold_bars": int(self.config.live_overlay_stable_max_hold_bars),
            "trail_style": str(getattr(self.config, "stable_trail_style", "tight") or "tight"),
            "leverage": leverage,
            "position_size_pct": position_size_pct,
            "maintenance_margin_pct": float(self.config.high_leverage_maintenance_margin_pct),
            "requested_notional": notional,
            "source": source,
        }

    def _overlay_maybe_build_formal_stable_candidate(self, engine: Any, idx: int) -> StrategyAction | None:
        state = self._load_overlay_formal_state(engine)
        event = state.get("last_shadow_event")
        if not isinstance(event, dict):
            return None
        if int(event.get("exit_idx", -1) or -1) != idx:
            return None
        if not self._stable_selector_allows(event):
            return None
        entry_price = float(event.get("exit_price", 0.0) or 0.0)
        if entry_price <= 0:
            return None
        stop_pct = float(event.get("stop_distance_pct", 0.0) or 0.0) / 100.0
        stop_pct *= float(self.config.live_overlay_stable_stop_multiplier)
        if stop_pct <= 0 or stop_pct * 100.0 > float(self.config.live_overlay_stable_max_short_stop_pct):
            return None
        stop_price = entry_price * (1.0 + stop_pct)
        target_price = entry_price * (1.0 - stop_pct * float(self.config.live_overlay_stable_target_rr))
        action, _ = self._overlay_build_open_short_action(
            engine=engine,
            idx=idx,
            event_type="stable_reverse_short",
            signal_entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            target_rr=float(self.config.live_overlay_stable_target_rr),
            max_hold_bars=int(self.config.live_overlay_stable_max_hold_bars),
            allocation=float(self.config.live_overlay_stable_allocation),
            leverage=float(self.config.live_overlay_stable_leverage),
            stop_reason="stop_loss",
            target_reason="target_rr",
            metadata={
                "source_trade_exit_reason": event.get("exit_reason"),
                "source_trade_entry_idx": event.get("entry_idx"),
                "source_trade_exit_idx": event.get("exit_idx"),
                "source_quality_score": quality_snapshot(event)["quality_score"],
                "source_effective_leverage": event.get("effective_leverage"),
                "source_failed_breakout_guard_applied": bool(event.get("failed_breakout_guard_applied")),
                "overlay_formal_fixed": True,
            },
        )
        return action

    def _overlay_should_open_stable_from_trade(self, trade: Trade) -> bool:
        return self._stable_selector_allows(self._stable_source_event_from_trade(trade))

    def _overlay_maybe_build_stable_candidate(self, engine: Any, idx: int) -> StrategyAction | None:
        if not self._stable_live_enabled():
            return None
        if self._overlay_formal_fixed_shadow_enabled():
            return self._overlay_maybe_build_formal_stable_candidate(engine, idx)
        if not getattr(engine, "trades", None):
            return None
        latest_trade = engine.trades[-1]
        if latest_trade.exit_idx != idx:
            return None
        if not self._overlay_should_open_stable_from_trade(latest_trade):
            return None
        entry_price = float(latest_trade.exit_price or 0.0)
        if entry_price <= 0:
            return None
        source_stop_pct = abs(float(latest_trade.initial_stop_price or 0.0) - float(latest_trade.signal_entry_price or 0.0))
        signal_entry = float(latest_trade.signal_entry_price or 0.0)
        if signal_entry <= 0:
            return None
        stop_pct = (source_stop_pct / signal_entry) * float(self.config.live_overlay_stable_stop_multiplier)
        if stop_pct <= 0 or stop_pct * 100.0 > float(self.config.live_overlay_stable_max_short_stop_pct):
            return None
        stop_price = entry_price * (1.0 + stop_pct)
        target_price = entry_price * (1.0 - stop_pct * float(self.config.live_overlay_stable_target_rr))
        action, _ = self._overlay_build_open_short_action(
            engine=engine,
            idx=idx,
            event_type="stable_reverse_short",
            signal_entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            target_rr=float(self.config.live_overlay_stable_target_rr),
            max_hold_bars=int(self.config.live_overlay_stable_max_hold_bars),
            allocation=float(self.config.live_overlay_stable_allocation),
            leverage=float(self.config.live_overlay_stable_leverage),
            stop_reason="stop_loss",
            target_reason="target_rr",
            metadata={
                "source_trade_exit_reason": latest_trade.exit_reason,
                "source_trade_entry_idx": latest_trade.entry_idx,
                "source_trade_exit_idx": latest_trade.exit_idx,
                "source_quality_score": quality_snapshot(
                    {
                        "direction": latest_trade.direction,
                        "feature_adx": (getattr(latest_trade, "execution_guard_diagnostics", None) or {}).get("feature_adx", 0.0),
                        "feature_momentum": (getattr(latest_trade, "execution_guard_diagnostics", None) or {}).get("feature_momentum", 0.0),
                        "feature_ema_gap": (getattr(latest_trade, "execution_guard_diagnostics", None) or {}).get("feature_ema_gap", 0.0),
                        "feature_bullish_structure": (getattr(latest_trade, "execution_guard_diagnostics", None) or {}).get("feature_bullish_structure", False),
                        "feature_bearish_structure": (getattr(latest_trade, "execution_guard_diagnostics", None) or {}).get("feature_bearish_structure", False),
                    }
                )["quality_score"],
            },
        )
        return action

    def _overlay_smc_case_args(self) -> Any:
        case_name = str(self.config.live_overlay_smc_case or FORMAL_SMC_CASE_NAMES[0])
        case_params = SMC_CASES[case_name]
        defaults = {
            "swing_n": 3,
            "target_rr": 2.0,
            "allowed_time_buckets": "other",
            "min_body_atr": 0.7,
            "min_range_atr": 1.1,
            "entry_lookahead_bars": 40,
            "min_displacement_body_atr": 0.0,
            "min_displacement_range_atr": 0.0,
            "max_mss_lag_bars": 15,
            "max_open_positions": 1,
            "initial_capital": 1000.0,
        }
        merged_case = defaults | case_params
        base_args = smc_strategy_args(
            SimpleNamespace(
                data_15m="",
                data_4h="",
                start_date="",
                swing_n=merged_case["swing_n"],
                entry_lookahead_bars=merged_case["entry_lookahead_bars"],
                min_body_atr=merged_case["min_body_atr"],
                min_range_atr=merged_case["min_range_atr"],
                target_rr=merged_case["target_rr"],
                allowed_time_buckets=merged_case["allowed_time_buckets"],
                min_displacement_body_atr=merged_case["min_displacement_body_atr"],
                min_displacement_range_atr=merged_case["min_displacement_range_atr"],
                max_mss_lag_bars=merged_case["max_mss_lag_bars"],
                max_open_positions=merged_case["max_open_positions"],
                initial_capital=merged_case["initial_capital"],
            )
        )
        config_ns = SimpleNamespace(
            data_15m="",
            data_4h="",
            start_date="",
        )
        merged = vars(base_args) | vars(smc_case_namespace(config_ns, case_params))
        return SimpleNamespace(**merged)

    def _overlay_maybe_build_smc_candidate(self, engine: Any, idx: int) -> StrategyAction | None:
        if not self._smc_live_enabled():
            return None
        case_name = str(self.config.live_overlay_smc_case or "")
        if case_name not in SMC_CASES:
            return None
        case_args = self._overlay_smc_case_args()
        smc_args = case_args
        c15m = list(engine.c15m[: idx + 1])
        if len(c15m) < max(int(getattr(smc_args, "swing_lookback", 80)), int(getattr(smc_args, "liquidity_lookback_bars", 192))) + 5:
            return None
        c4h_idx = int(engine.mapping[idx]) if idx < len(engine.mapping) else -1
        if c4h_idx <= 2:
            return None
        c4h = list(engine.c4h[: c4h_idx + 1])
        daily = daily_candles_from_4h(c4h)
        h4_highs, h4_lows = precompute_swings(c4h, n=2, lookback=80)
        d1_highs, d1_lows = precompute_swings(daily, n=2, lookback=20)
        candidate = self._overlay_live_smc_candidate(
            c15m=c15m,
            idx=idx,
            c4h=c4h,
            daily=daily,
            h4_highs=h4_highs,
            h4_lows=h4_lows,
            d1_highs=d1_highs,
            d1_lows=d1_lows,
            case_args=case_args,
            smc_args=smc_args,
        )
        if candidate is None:
            return None
        entry_price = float(candidate["entry_price"])
        stop_price = float(candidate["stop_price"])
        target_price = float(candidate["target_price"])
        risk_points = stop_price - entry_price
        if risk_points <= 0:
            return None
        target_rr = float(getattr(case_args, "target_rr", 2.0) or 2.0)
        trade_stub = pd.Series(
            {
                "entry_time": candidate["entry_time"],
                "direction": "BEAR",
                "entry_price": entry_price,
                "initial_stop_price": stop_price,
                "notional": self._overlay_capital(engine) * float(getattr(case_args, "leverage", 10.0)) * float(getattr(case_args, "position_size_pct", 1.0)),
            }
        )
        diagnostics = high_leverage_trade_diagnostics(
            trade_stub,
            capital=self._overlay_capital(engine),
            leverage=float(getattr(case_args, "leverage", 10.0)),
            maintenance_margin_pct=float(getattr(case_args, "maintenance_margin_pct", 0.5)),
        )
        if float(diagnostics["liquidation_buffer_pct"]) < float(getattr(case_args, "min_liq_buffer_pct", 1.2)):
            return None
        action, _ = self._overlay_build_open_short_action(
            engine=engine,
            idx=idx,
            event_type="smc_short",
            signal_entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            target_rr=target_rr,
            max_hold_bars=int(getattr(smc_args, "outcome_lookahead_bars", 96)),
            allocation=float(self.config.live_overlay_smc_allocation),
            leverage=float(getattr(case_args, "leverage", 10.0)),
            stop_reason="stop_loss",
            target_reason=f"target_{target_rr:.1f}r",
            metadata={
                "smc_case": case_name,
                "smc_time_bucket": candidate.get("time_bucket"),
                "smc_ny_time": candidate.get("ny_time"),
                "smc_mss_lag_bars": candidate.get("mss_lag_bars"),
                "smc_h4_bias": candidate.get("h4_bias"),
                "smc_d1_bias": candidate.get("d1_bias"),
                "smc_fvg_touched": bool(candidate.get("fvg_touched", False)),
                "smc_ote_touched": bool(candidate.get("ote_touched", False)),
                "smc_stop_buffer_atr": float(getattr(smc_args, "stop_buffer_atr", 0.05)),
                "smc_live_safe": True,
            },
        )
        return action

    def _overlay_live_smc_candidate(
        self,
        *,
        c15m: list[Any],
        idx: int,
        c4h: list[Any],
        daily: list[Any],
        h4_highs: list[int],
        h4_lows: list[int],
        d1_highs: list[int],
        d1_lows: list[int],
        case_args: Any,
        smc_args: Any,
    ) -> dict[str, Any] | None:
        scan_args = build_event_scan_args(smc_args)
        scan_args.allow_incomplete_tail = True
        curr = c15m[idx]
        atr_values = atr_series(c15m, int(getattr(smc_args, "atr_period", 14)))
        recent_window = max(
            int(getattr(smc_args, "liquidity_lookback_bars", 192))
            + int(getattr(smc_args, "mss_lookahead_bars", 24))
            + int(getattr(smc_args, "entry_lookahead_bars", 40))
            + int(getattr(smc_args, "fvg_lookback_bars", 8))
            + 16,
            int(getattr(smc_args, "swing_lookback", 80)) + 64,
        )
        window_start = max(0, len(c15m) - recent_window)
        scan_c15m = c15m[window_start:]
        events = scan_events(scan_c15m, scan_args)
        matching_events = [
            event
            for event in events
            if event.direction == "BEAR"
            and event.retest is not None
            and int(event.retest.idx) + window_start == idx
        ]
        if not matching_events:
            return None
        bucket, ny_time = time_bucket(curr.ts)
        h4_idx = max(0, len(c4h) - 2)
        h4_bias = htf_structure_bias(c4h, h4_highs, h4_lows, h4_idx) if len(c4h) >= 3 else "NONE"
        daily_ts = [candle.ts for candle in daily]
        d1_idx = bisect.bisect_left(daily_ts, pd.Timestamp(curr.ts, unit="s", tz="UTC").normalize().timestamp()) - 1
        d1_bias = htf_structure_bias(daily, d1_highs, d1_lows, d1_idx) if d1_idx >= 0 else "NONE"

        matching_event = None
        matching_mss_lag_bars = None
        for event in matching_events:
            if bool(getattr(smc_args, "require_confirmed_retest", False)) and not bool(event.retest.confirmed):
                continue
            if bool(getattr(smc_args, "require_fvg_touch", False)) and not bool(event.retest.fvg_touched):
                continue
            if not bool(getattr(smc_args, "allow_ote_only", True)) and not bool(event.retest.fvg_touched):
                continue
            if bool(getattr(smc_args, "require_ote_touch", False)) and not bool(event.retest.ote_touched):
                continue
            if not allowed_bucket(bucket, str(getattr(smc_args, "allowed_time_buckets", "all"))):
                continue
            if not allowed_direction("BEAR", str(getattr(smc_args, "allowed_directions", "all"))):
                continue
            if bool(getattr(case_args, "drop_asia_session", False)) and bucket == "asia_evening_ny":
                continue
            mss_lag_bars = (int(event.mss_idx) - int(event.sweep_idx)) if event.mss_idx is not None else None
            if int(getattr(smc_args, "max_mss_lag_bars", 0)) > 0 and mss_lag_bars is not None and mss_lag_bars > int(smc_args.max_mss_lag_bars):
                continue
            if int(getattr(case_args, "global_min_mss_lag_bars", 0)) > 0 and mss_lag_bars is not None and mss_lag_bars < int(case_args.global_min_mss_lag_bars):
                continue
            if int(getattr(case_args, "global_max_mss_lag_bars", 0)) > 0 and mss_lag_bars is not None and mss_lag_bars > int(case_args.global_max_mss_lag_bars):
                continue
            if bucket == "ny_am_killzone" and int(getattr(case_args, "ny_max_mss_lag_bars", 0)) > 0 and mss_lag_bars is not None and mss_lag_bars > int(case_args.ny_max_mss_lag_bars):
                continue
            if bucket == "other" and int(getattr(case_args, "other_min_mss_lag_bars", 0)) > 0 and mss_lag_bars is not None and mss_lag_bars < int(case_args.other_min_mss_lag_bars):
                continue
            if float(event.displacement_body_atr or 0.0) < float(getattr(smc_args, "min_displacement_body_atr", 0.0) or 0.0):
                continue
            if float(event.displacement_range_atr or 0.0) < float(getattr(smc_args, "min_displacement_range_atr", 0.0) or 0.0):
                continue
            if float(getattr(smc_args, "bear_min_sweep_distance_pct", 0.0) or 0.0) > 0.0 and float(event.sweep_distance_pct or 0.0) < float(getattr(smc_args, "bear_min_sweep_distance_pct", 0.0) or 0.0):
                continue
            if bool(getattr(smc_args, "require_h4_bias_align", False)) and bool(getattr(smc_args, "require_htf_bias_align", False)) and h4_bias != "BEAR":
                continue
            if bool(getattr(smc_args, "require_h4_bias_align", False)) and not bool(getattr(smc_args, "require_htf_bias_align", False)) and h4_bias not in {"BEAR", "NONE"}:
                continue
            if bool(getattr(smc_args, "require_d1_bias_align", False)) and bool(getattr(smc_args, "require_htf_bias_align", False)) and d1_bias != "BEAR":
                continue
            if bool(getattr(smc_args, "require_d1_bias_align", False)) and not bool(getattr(smc_args, "require_htf_bias_align", False)) and d1_bias not in {"BEAR", "NONE"}:
                continue
            matching_event = event
            matching_mss_lag_bars = mss_lag_bars
            break
        if matching_event is None:
            return None

        stop_buffer = atr_values[idx] * float(getattr(smc_args, "stop_buffer_atr", 0.05)) if idx < len(atr_values) else 0.0
        stop_price = float(matching_event.sweep_extreme) + stop_buffer
        entry_price = float(matching_event.retest.close)
        risk_points = stop_price - entry_price
        if risk_points <= 0:
            return None
        target_price = entry_price - risk_points * float(getattr(case_args, "target_rr", 2.0) or 2.0)
        return {
            "entry_idx": idx,
            "entry_time": matching_event.retest.timestamp,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "time_bucket": bucket,
            "ny_time": ny_time,
            "mss_lag_bars": matching_mss_lag_bars,
            "h4_bias": h4_bias,
            "d1_bias": d1_bias,
            "fvg_touched": bool(matching_event.retest.fvg_touched),
            "ote_touched": bool(matching_event.retest.ote_touched),
        }

    def _overlay_manage_runtime_position(self, engine: Any, idx: int) -> list[StrategyAction]:
        position = self._load_overlay_runtime_position()
        if position is None:
            return []
        if self.config.mode == "live" and self.config.enable_exchange_brackets:
            return []
        candle = engine.c15m[idx]
        actions: list[StrategyAction] = []
        if position.max_hold_bars is not None and idx - position.entry_idx >= position.max_hold_bars:
            close_action = self._overlay_build_close_action(
                engine=engine,
                position=position,
                idx=idx,
                exit_price=float(candle.c),
                reason="time_exit",
            )
            actions.append(close_action)
            return actions
        if float(candle.h) >= float(position.sl_price):
            close_action = self._overlay_build_close_action(
                engine=engine,
                position=position,
                idx=idx,
                exit_price=float(position.sl_price),
                reason=str(position.stop_reason or "stop_loss"),
            )
            actions.append(close_action)
            return actions
        if float(candle.l) <= float(position.target_price):
            close_action = self._overlay_build_close_action(
                engine=engine,
                position=position,
                idx=idx,
                exit_price=float(position.target_price),
                reason=str(position.target_reason or "target_rr"),
            )
            actions.append(close_action)
            return actions
        return actions

    def _overlay_post_execute_runtime_update(self, engine: Any, action: StrategyAction, idx: int) -> None:
        if not self._live_overlay_enabled():
            return
        metadata = action.metadata or {}
        event_type = str(metadata.get("overlay_event_type") or "")
        if action.type == ActionType.OPEN_SHORT and event_type in {"stable_reverse_short", "smc_short"}:
            position = self._load_overlay_runtime_position()
            if position is not None:
                self._save_overlay_runtime_position(position, last_managed_idx=idx)
            return
        if action.type == ActionType.CLOSE_POSITION:
            position = self._load_overlay_runtime_position()
            if position is None:
                return
            if event_type and event_type != position.event_type:
                return
            self._overlay_append_trade(engine, position, action)
            self._clear_overlay_runtime_position(last_managed_idx=idx)

    def _overlay_bind_runtime_position(self, runtime_position: OverlayRuntimePosition) -> None:
        self._save_overlay_runtime_position(runtime_position)

    def _overlay_base_actions_for_idx(self, engine: Any, idx: int) -> tuple[StrategyAction | None, list[StrategyAction]]:
        if not hasattr(engine, "_apply_regime_switch_for_idx"):
            candle_actions = engine.evaluate_range(idx, idx + 1)
            base_open_action: StrategyAction | None = None
            non_open_actions: list[StrategyAction] = []
            for action in candle_actions:
                if action.type in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}:
                    base_open_action = action
                else:
                    non_open_actions.append(action)
            return base_open_action, non_open_actions
        engine._apply_regime_switch_for_idx(idx)
        non_open_actions: list[StrategyAction] = []
        if getattr(engine, "position", None):
            position_actions = engine.manage_position(idx)
            for action in position_actions:
                if action.type in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}:
                    raise ValueError("Unexpected open action during base position management")
                non_open_actions.append(action)
            return None, non_open_actions

        base_open_action = self._overlay_maybe_build_base_open_candidate(engine, idx)
        return base_open_action, non_open_actions

    def _overlay_commit_base_open_action(self, engine: Any, action: StrategyAction) -> None:
        if not hasattr(engine, "_open_action_from_pending"):
            return
        idx = int((action.metadata or {}).get("index", 0) or 0)
        committed = self._overlay_maybe_commit_base_open_candidate(engine, idx, action.direction)
        if committed is None:
            raise ValueError("Base overlay open candidate could not be replayed deterministically for live commit")
        position = getattr(engine, "position", None)
        if position is None:
            raise ValueError("Base overlay open candidate replay did not create local engine position")
        metadata = action.metadata or {}
        if bool(metadata.get("overlay_formal_fixed")):
            effective_leverage = float(metadata.get("overlay_formal_effective_leverage", 0.0) or 0.0)
            capital_at_entry = float(getattr(position, "capital_at_entry", 0.0) or 0.0)
            entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
            if effective_leverage > 0 and capital_at_entry > 0 and entry_price > 0:
                notional = capital_at_entry * effective_leverage
                quantity = notional / entry_price
                entry_fee = notional * float(self.config.taker_fee_rate)
                signal_entry_price = float(getattr(position, "signal_entry_price", entry_price) or entry_price)
                entry_slippage_cost = quantity * abs(entry_price - signal_entry_price)
                setattr(position, "notional", notional)
                setattr(position, "quantity", quantity)
                setattr(position, "entry_fee", entry_fee)
                setattr(position, "entry_slippage_cost", entry_slippage_cost)
            setattr(position, "execution_effective_leverage", effective_leverage)
            setattr(position, "execution_risk_mode", metadata.get("overlay_formal_risk_mode"))
            setattr(position, "execution_leverage_reasons", list(metadata.get("overlay_formal_leverage_reasons") or []))
            setattr(position, "execution_guard_diagnostics", dict(metadata.get("overlay_formal_guard_diagnostics") or {}))
            setattr(position, "execution_requested_notional", float(getattr(position, "notional", 0.0) or 0.0))
            setattr(position, "execution_target_notional", float(getattr(position, "notional", 0.0) or 0.0))
            state = self._load_overlay_formal_state(engine)
            state["active_sota_entry_idx"] = int(getattr(position, "entry_idx", idx) or idx)
            state["active_sota_entry_time"] = str(getattr(position, "entry_time", action.timestamp) or action.timestamp)
            self._save_overlay_formal_state(state)

    def _overlay_capture_base_open_from_pending(self, engine: Any, idx: int, pending: Any) -> StrategyAction | None:
        original_position = getattr(engine, "position", None)
        try:
            return engine._open_action_from_pending(idx, pending)
        finally:
            engine.position = original_position

    def _overlay_maybe_build_base_open_candidate(self, engine: Any, idx: int) -> StrategyAction | None:
        bias = engine.precomputed.bias_4h[engine.mapping[idx]]
        active_pending = bool(engine.waiting_for_pullback or any(engine.pending_by_direction.values()))
        if engine.config.use_hfvf_filter and bias == Direction.NONE and not active_pending:
            return None

        if engine.config.enable_dual_pending_state:
            for direction in (Direction.BULL, Direction.BEAR):
                pending = engine.pending_by_direction[direction]
                if pending is None:
                    continue
                action = self._overlay_capture_base_open_from_pending(engine, idx, pending)
                if action is not None:
                    return action
                if engine._pending_expired(idx, pending):
                    engine.pending_by_direction[direction] = None
            if idx in (engine.precomputed.highs_set | engine.precomputed.lows_set):
                pending = engine._build_pending_pullback(idx, bias)
                if pending and engine.pending_by_direction[pending.direction] is None:
                    engine.pending_by_direction[pending.direction] = pending
            return None

        if engine.waiting_for_pullback and engine.ob_zone and engine.waiting_direction:
            pending = PendingPullback(
                direction=engine.waiting_direction,
                bos_idx=engine.bos_idx,
                ob_zone=engine.ob_zone,
                pullback_window=engine.waiting_pullback_window,
            )
            action = self._overlay_capture_base_open_from_pending(engine, idx, pending)
            if action is not None:
                return action
            if engine._pending_expired(idx, pending):
                engine.waiting_for_pullback = False
                engine.ob_zone = None
                engine.waiting_direction = None
                return None

        if not getattr(engine, "position", None) and not engine.waiting_for_pullback and idx in (engine.precomputed.highs_set | engine.precomputed.lows_set):
            pending = engine._build_pending_pullback(idx, bias)
            if pending:
                engine.waiting_for_pullback = True
                engine.bos_idx = pending.bos_idx
                engine.ob_zone = pending.ob_zone
                engine.waiting_direction = pending.direction
                engine.waiting_pullback_window = pending.pullback_window
        if not getattr(engine, "position", None) and not engine.waiting_for_pullback:
            for pending, detail in self._active_ob_candidates(engine, idx):
                if not detail.get("ready"):
                    continue
                action = self._overlay_capture_base_open_from_pending(engine, idx, pending)
                if action is not None:
                    return action
        return None

    def _overlay_maybe_commit_base_open_candidate(self, engine: Any, idx: int, direction: Any) -> StrategyAction | None:
        if engine.config.enable_dual_pending_state:
            pending = engine.pending_by_direction.get(str(direction))
            if pending is None:
                return None
            action = engine._open_action_from_pending(idx, pending)
            if action is None:
                return None
            engine.pending_by_direction[Direction.BULL] = None
            engine.pending_by_direction[Direction.BEAR] = None
            return action

        if not (engine.waiting_for_pullback and engine.ob_zone and engine.waiting_direction == direction):
            return None
        pending = PendingPullback(
            direction=engine.waiting_direction,
            bos_idx=engine.bos_idx,
            ob_zone=engine.ob_zone,
            pullback_window=engine.waiting_pullback_window,
        )
        action = engine._open_action_from_pending(idx, pending)
        if action is None:
            return None
        engine.waiting_for_pullback = False
        engine.ob_zone = None
        engine.waiting_direction = None
        return action

    def _overlay_discard_base_open_candidate(self, engine: Any, action: StrategyAction) -> None:
        if getattr(getattr(engine, "config", None), "enable_dual_pending_state", False):
            engine.pending_by_direction[Direction.BULL] = None
            engine.pending_by_direction[Direction.BEAR] = None
            return
        if not getattr(engine, "waiting_for_pullback", False):
            return
        engine.waiting_for_pullback = False
        engine.ob_zone = None
        engine.waiting_direction = None

    def _shadow_gate_enabled(self) -> bool:
        return (
            bool(self.config.enable_shadow_risk_gate)
            and (
                self.config.shadow_daily_loss_stop_pct > 0
                or self.config.shadow_equity_drawdown_stop_pct > 0
                or self.config.shadow_consecutive_loss_stop > 0
            )
        )

    def _shadow_gate_default_state(self, engine: Any | None = None) -> dict[str, Any]:
        capital = float(getattr(engine, "capital", 0.0) or 0.0) if engine is not None else 0.0
        return {
            "mode": "shadow_risk_gate",
            "capital": capital,
            "drawdown_peak": capital,
            "pause_until_ts": 0.0,
            "real_position_open": False,
            "real_position_direction": None,
            "paper_entry_time": None,
            "day_start_capital": {},
            "day_pnl": {},
            "loss_streak": 0,
            "events": [],
        }

    def _load_shadow_gate_state(self, engine: Any | None = None) -> dict[str, Any]:
        raw = self.store.get_value("shadow_risk_gate_state")
        if not raw:
            return self._shadow_gate_default_state(engine)
        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            return self._shadow_gate_default_state(engine)
        default = self._shadow_gate_default_state(engine)
        default.update(state if isinstance(state, dict) else {})
        if default["capital"] <= 0 and engine is not None:
            default["capital"] = float(getattr(engine, "capital", 0.0) or 0.0)
        if default["drawdown_peak"] <= 0:
            default["drawdown_peak"] = default["capital"]
        return default

    def _save_shadow_gate_state(self, state: dict[str, Any]) -> None:
        self.store.set_value("shadow_risk_gate_state", json.dumps(state, ensure_ascii=False))

    def _action_timestamp(self, action: StrategyAction) -> datetime:
        value = action.timestamp
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _shadow_next_utc_day_ts(self, dt: datetime) -> float:
        day_start = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
        return (day_start + timedelta(days=1)).timestamp()

    def _shadow_cooldown_until_ts(self, dt: datetime, days: int) -> float:
        day_start = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
        return (day_start + timedelta(days=max(1, int(days)))).timestamp()

    def _shadow_format_ts(self, ts: float) -> str:
        if ts <= 0:
            return ""
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    def _shadow_append_event(self, state: dict[str, Any], event: dict[str, Any]) -> None:
        events = state.get("events")
        if not isinstance(events, list):
            events = []
        events.append(event)
        state["events"] = events[-500:]

    def _shadow_gate_mark_real_position(self, mirrored: bool, action: StrategyAction, reason: str) -> None:
        if not self._shadow_gate_enabled():
            return
        state = self._load_shadow_gate_state()
        state["real_position_open"] = bool(mirrored)
        state["real_position_direction"] = action.direction if mirrored else None
        state["paper_entry_time"] = action.timestamp
        self._shadow_append_event(
            state,
            {
                "time": action.timestamp,
                "event": "mirror_open" if mirrored else "mirror_open_failed",
                "reason": reason,
                "direction": action.direction,
            },
        )
        self._save_shadow_gate_state(state)

    def _shadow_gate_pre_execute(self, action: StrategyAction, engine: Any) -> dict[str, Any] | None:
        if not self._shadow_gate_enabled():
            return None
        if self._overlay_formal_fixed_shadow_enabled() and bool((action.metadata or {}).get("overlay_formal_fixed")):
            return None

        state = self._load_shadow_gate_state(engine)
        if action.type in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}:
            action_ts = self._action_timestamp(action).timestamp()
            pause_until_ts = float(state.get("pause_until_ts", 0.0) or 0.0)
            if action_ts < pause_until_ts:
                state["real_position_open"] = False
                state["real_position_direction"] = None
                state["paper_entry_time"] = action.timestamp
                self._shadow_append_event(
                    state,
                    {
                        "time": action.timestamp,
                        "event": "skip_open",
                        "direction": action.direction,
                        "pause_until": self._shadow_format_ts(pause_until_ts),
                    },
                )
                self._save_shadow_gate_state(state)
                return {
                    "status": "shadow_gate_skipped_open",
                    "action": action.type.value,
                    "direction": action.direction,
                    "pause_until": self._shadow_format_ts(pause_until_ts),
                }
            return None

        if action.type == ActionType.UPDATE_STOP and not bool(state.get("real_position_open")):
            return {"status": "shadow_gate_skipped_update_stop", "action": action.type.value}

        if action.type == ActionType.CLOSE_POSITION and not bool(state.get("real_position_open")):
            self._shadow_append_event(
                state,
                {
                    "time": action.timestamp,
                    "event": "skip_close",
                    "direction": action.direction,
                    "reason": action.reason,
                },
            )
            state["real_position_direction"] = None
            state["paper_entry_time"] = None
            self._save_shadow_gate_state(state)
            return {
                "status": "shadow_gate_skipped_close",
                "action": action.type.value,
                "direction": action.direction,
                "reason": "paper_position_not_mirrored",
            }

        return None

    def _high_leverage_guard_enabled(self, action: StrategyAction | None = None) -> bool:
        return (
            bool(self.config.enable_high_leverage_guard)
            and self._action_configured_leverage(action) >= float(self.config.high_leverage_guard_min_leverage)
        )

    def _high_leverage_guard_pre_open(self, action: StrategyAction, sizing: dict[str, Any]) -> dict[str, Any] | None:
        if not self._high_leverage_guard_enabled(action):
            return None
        if action.type not in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}:
            return None
        diagnostics = self._high_leverage_open_diagnostics(action, sizing)
        failures = self._high_leverage_guard_failures(diagnostics)
        if not failures:
            return None

        reason = "high_leverage_guard_" + failures[0]
        if self._shadow_gate_enabled():
            state = self._load_shadow_gate_state()
            state["real_position_open"] = False
            state["real_position_direction"] = None
            state["paper_entry_time"] = action.timestamp
            self._shadow_append_event(
                state,
                {
                    "time": action.timestamp,
                    "event": "skip_open",
                    "reason": reason,
                    "direction": action.direction,
                    "diagnostics": diagnostics,
                    "failures": failures,
                },
            )
            self._save_shadow_gate_state(state)
            return {
                "status": "high_leverage_guard_skipped_open",
                "action": action.type.value,
                "direction": action.direction,
                "reason": reason,
                "failures": failures,
                "diagnostics": diagnostics,
            }

        return {
            "status": "error",
            "reason": "high_leverage_guard_requires_shadow_risk_gate",
            "failures": failures,
            "diagnostics": diagnostics,
        }

    def _high_leverage_open_diagnostics(self, action: StrategyAction, sizing: dict[str, Any]) -> dict[str, Any]:
        entry_price = float(action.entry_price or 0.0)
        stop_price = float(action.stop_price or 0.0)
        leverage = self._action_configured_leverage(action)
        maintenance_margin_pct = max(self._action_maintenance_margin_pct(action), 0.0) / 100.0
        stop_distance_pct = (
            abs(entry_price - stop_price) / entry_price * 100.0
            if entry_price > 0 and stop_price > 0
            else 0.0
        )
        liquidation_price = 0.0
        liquidation_buffer_pct = 0.0
        if entry_price > 0 and leverage > 0:
            if action.type == ActionType.OPEN_LONG:
                liquidation_price = entry_price * (1.0 - (1.0 / leverage) + maintenance_margin_pct)
                liquidation_buffer_pct = (stop_price - liquidation_price) / entry_price * 100.0
            else:
                liquidation_price = entry_price * (1.0 + (1.0 / leverage) - maintenance_margin_pct)
                liquidation_buffer_pct = (liquidation_price - stop_price) / entry_price * 100.0
        metadata = action.metadata or {}
        available_usdt = float(
            sizing.get("available_usdt", 0.0)
            or metadata.get("available_usdt", 0.0)
            or metadata.get("capital_at_entry", 0.0)
            or 0.0
        )
        expected_notional_usdt = float(sizing.get("expected_notional_usdt", 0.0) or 0.0)
        account_effective_leverage = (
            expected_notional_usdt / available_usdt
            if available_usdt > 0
            else 0.0
        )
        return {
            "configured_leverage": round(leverage, 6),
            "entry_price": round(entry_price, 6),
            "stop_price": round(stop_price, 6),
            "estimated_liquidation_price": round(liquidation_price, 6),
            "stop_distance_pct": round(stop_distance_pct, 6),
            "liquidation_buffer_pct": round(liquidation_buffer_pct, 6),
            "account_effective_leverage": round(account_effective_leverage, 6),
            "expected_notional_usdt": round(expected_notional_usdt, 6),
            "available_usdt": round(available_usdt, 6),
            "min_liquidation_buffer_pct": round(float(self.config.high_leverage_min_liquidation_buffer_pct), 6),
            "max_stop_distance_pct": round(float(self.config.high_leverage_max_stop_distance_pct), 6),
            "max_account_effective_leverage": round(float(self.config.high_leverage_max_account_effective_leverage), 6),
            "maintenance_margin_pct": round(self._action_maintenance_margin_pct(action), 6),
        }

    def _high_leverage_guard_failures(self, diagnostics: dict[str, Any]) -> list[str]:
        failures: list[str] = []
        if diagnostics["entry_price"] <= 0 or diagnostics["stop_price"] <= 0:
            failures.append("missing_entry_or_stop")
        min_buffer = float(self.config.high_leverage_min_liquidation_buffer_pct)
        if min_buffer > 0 and diagnostics["liquidation_buffer_pct"] < min_buffer:
            failures.append("liquidation_buffer_too_small")
        max_stop_distance = float(self.config.high_leverage_max_stop_distance_pct)
        if max_stop_distance > 0 and diagnostics["stop_distance_pct"] > max_stop_distance:
            failures.append("stop_distance_too_wide")
        max_account_leverage = float(self.config.high_leverage_max_account_effective_leverage)
        if max_account_leverage > 0 and diagnostics["account_effective_leverage"] > max_account_leverage:
            failures.append("account_effective_leverage_too_high")
        return failures

    def _shadow_gate_after_close(self, action: StrategyAction, engine: Any) -> None:
        if not self._shadow_gate_enabled():
            return
        state = self._load_shadow_gate_state(engine)
        metadata = action.metadata or {}
        pnl = float(metadata.get("net_pnl", 0.0) or 0.0)
        post_close_capital = float(getattr(engine, "capital", 0.0) or 0.0)
        if post_close_capital > 0:
            capital = post_close_capital
            capital_before = post_close_capital - pnl
        else:
            capital_before = float(state.get("capital", 0.0) or 0.0)
            capital = capital_before + pnl
        state["capital"] = capital
        state["drawdown_peak"] = max(float(state.get("drawdown_peak", capital) or capital), capital)
        action_dt = self._action_timestamp(action)
        day_key = action_dt.strftime("%Y-%m-%d")
        day_start_capital = state.get("day_start_capital")
        day_pnl = state.get("day_pnl")
        if not isinstance(day_start_capital, dict):
            day_start_capital = {}
        if not isinstance(day_pnl, dict):
            day_pnl = {}
        if day_key not in day_start_capital:
            day_start_capital[day_key] = capital_before
            day_pnl[day_key] = 0.0
        day_pnl[day_key] = float(day_pnl.get(day_key, 0.0) or 0.0) + pnl
        state["day_start_capital"] = day_start_capital
        state["day_pnl"] = day_pnl

        if pnl > 0:
            state["loss_streak"] = 0
        else:
            state["loss_streak"] = int(state.get("loss_streak", 0) or 0) + 1

        triggered: list[str] = []
        daily_stop = float(self.config.shadow_daily_loss_stop_pct or 0.0)
        start_capital = float(day_start_capital[day_key])
        if daily_stop > 0 and start_capital > 0:
            daily_loss_pct = -float(day_pnl[day_key]) / start_capital * 100.0
            if daily_loss_pct >= daily_stop:
                triggered.append(f"daily_loss:{daily_loss_pct:.2f}")
                state["pause_until_ts"] = max(
                    float(state.get("pause_until_ts", 0.0) or 0.0),
                    self._shadow_next_utc_day_ts(action_dt),
                )

        streak_stop = int(self.config.shadow_consecutive_loss_stop or 0)
        if streak_stop > 0 and int(state.get("loss_streak", 0) or 0) >= streak_stop:
            triggered.append(f"consecutive_loss:{state['loss_streak']}")
            state["pause_until_ts"] = max(
                float(state.get("pause_until_ts", 0.0) or 0.0),
                self._shadow_next_utc_day_ts(action_dt),
            )
            state["loss_streak"] = 0

        dd_stop = float(self.config.shadow_equity_drawdown_stop_pct or 0.0)
        peak = float(state.get("drawdown_peak", capital) or capital)
        if dd_stop > 0 and peak > 0:
            drawdown_pct = (peak - capital) / peak * 100.0
            if drawdown_pct >= dd_stop:
                triggered.append(f"equity_drawdown:{drawdown_pct:.2f}")
                state["pause_until_ts"] = max(
                    float(state.get("pause_until_ts", 0.0) or 0.0),
                    self._shadow_cooldown_until_ts(action_dt, int(self.config.shadow_equity_drawdown_cooldown_days or 0)),
                )
                state["drawdown_peak"] = capital
                state["loss_streak"] = 0

        state["real_position_open"] = False
        state["real_position_direction"] = None
        state["paper_entry_time"] = None
        self._shadow_append_event(
            state,
            {
                "time": action.timestamp,
                "event": "mirror_close",
                "direction": action.direction,
                "pnl": pnl,
                "capital": capital,
                "triggers": triggered,
                "pause_until": self._shadow_format_ts(float(state.get("pause_until_ts", 0.0) or 0.0)),
            },
        )
        self._save_shadow_gate_state(state)

    def _shadow_gate_allows_unmirrored_local_position(self, local_position: Any) -> bool:
        raw = self.store.get_value("shadow_risk_gate_state")
        if not raw:
            return False
        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            return False
        if not isinstance(state, dict):
            return False
        if bool(state.get("real_position_open")):
            return False
        if state.get("real_position_direction"):
            return False

        local_entry_time = getattr(local_position, "entry_time", None)
        paper_entry_time = state.get("paper_entry_time")
        if not local_entry_time or not paper_entry_time or paper_entry_time != local_entry_time:
            return False

        local_direction = getattr(local_position, "direction", None)
        events = state.get("events")
        if not isinstance(events, list):
            return False
        for event in reversed(events[-50:]):
            if not isinstance(event, dict):
                continue
            if event.get("time") != paper_entry_time:
                continue
            if event.get("direction") != local_direction:
                continue
            if event.get("event") in {"skip_open", "mirror_open_failed"}:
                return True
        return False

    def _resolve_order_sizing(self, action: StrategyAction, engine: Any) -> dict[str, Any]:
        candles = self._engine_candles(engine)
        reference_price = action.entry_price or action.exit_price or (candles[-1].c if candles else 0.0)
        if reference_price <= 0:
            return {"status": "error", "reason": "invalid_reference_price"}
        if self.config.mode == "paper":
            return self._resolve_paper_order_sizing(action, engine, float(reference_price))

        if action.type in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}:
            metadata = action.metadata or {}
            requested_notional = metadata.get("notional")
            if requested_notional is not None:
                notional = float(requested_notional)
                max_notional = float(metadata.get("max_notional", notional))
                risk_based_notional = float(metadata.get("risk_based_notional", notional))
                available_usdt = float(
                    metadata.get("available_usdt", 0.0)
                    or metadata.get("capital_at_entry", 0.0)
                    or getattr(engine, "capital", 0.0)
                    or 0.0
                )
                balance_source = str(metadata.get("balance_source", "action_metadata"))
                leverage = self._action_configured_leverage(action)
                margin_usdt = float(
                    metadata.get(
                        "margin_usdt",
                        notional / leverage if leverage > 0 else notional,
                    )
                )
            else:
                try:
                    balance = self.client.fetch_balance()
                    available_usdt, balance_source = self._extract_available_usdt(balance)
                except Exception as exc:
                    return {"status": "error", "reason": "balance_unavailable", "error": str(exc)}
                max_notional = (
                    self.config.fixed_notional_usdt
                    if self.config.fixed_notional_usdt is not None
                    else available_usdt * self.config.position_size_pct * self.config.leverage
                )
                stop_price = action.stop_price
                risk_amount = available_usdt * self.config.risk_per_trade
                stop_distance = abs(reference_price - stop_price) if stop_price is not None else 0.0
                risk_based_notional = (
                    (risk_amount / stop_distance) * reference_price
                    if stop_distance > 0
                    else max_notional
                )
                notional = min(max_notional, risk_based_notional)
                margin_usdt = available_usdt * self.config.position_size_pct
            amount = round(notional / reference_price, 6)
            sizing = self._build_order_sizing(amount, notional, reference_price)
            if sizing["amount"] <= 0:
                return {"status": "error", "reason": "non_positive_amount", "notional_usdt": notional}
            return {
                "status": "ok",
                **sizing,
                "notional_usdt": round(notional, 6),
                "max_notional_usdt": round(max_notional, 6),
                "risk_based_notional_usdt": round(risk_based_notional, 6),
                "margin_usdt": round(margin_usdt, 6),
                "available_usdt": round(available_usdt, 6),
                "balance_source": balance_source,
            }

        if action.type == ActionType.CLOSE_POSITION:
            try:
                pos_side = "long" if action.direction == "BULL" else "short"
                position_state = self._fetch_position_state(pos_side, reference_price=reference_price)
            except Exception as exc:
                return {"status": "error", "reason": "position_unavailable", "error": str(exc)}
            amount = position_state["contracts"] if self._market().get("contract") else position_state["base_amount_btc"]
            if amount <= 0:
                return {"status": "error", "reason": "no_open_position_size"}
            return {
                "status": "ok",
                "amount": amount,
                "order_unit": "contracts" if self._market().get("contract") else "BTC",
                "close_source": "exchange_position",
                "expected_notional_usdt": position_state["notional_usdt"],
                "base_amount_btc": position_state["base_amount_btc"],
                "contracts": position_state["contracts"],
            }

        return {"status": "ok", "amount": 0.0}

    def _resolve_paper_order_sizing(self, action: StrategyAction, engine: Any, reference_price: float) -> dict[str, Any]:
        metadata = action.metadata or {}
        if action.type in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}:
            quantity = float(metadata.get("quantity", 0.0) or 0.0)
            notional = float(
                metadata.get("notional", 0.0)
                or metadata.get("expected_notional_usdt", 0.0)
                or (quantity * reference_price)
                or 0.0
            )
            max_notional = float(metadata.get("max_notional", notional) or notional)
            risk_based_notional = float(metadata.get("risk_based_notional", notional) or notional)
            capital_at_entry = float(metadata.get("capital_at_entry", getattr(engine, "capital", 0.0)) or 0.0)
            leverage = float(
                metadata.get("leverage", 0.0)
                or metadata.get("overlay_leverage", 0.0)
                or getattr(self.config, "leverage", 0.0)
                or 0.0
            )
            margin_usdt = float(
                metadata.get("margin_usdt", 0.0)
                or (notional / leverage if leverage > 0 else notional)
                or 0.0
            )
            return {
                "status": "ok",
                "amount": quantity,
                "order_unit": "BTC",
                "requested_base_amount_btc": round(quantity, 8),
                "base_amount_btc": round(quantity, 8),
                "contract_size": 0.0,
                "expected_notional_usdt": round(notional, 6),
                "requested_notional_usdt": round(notional, 6),
                "notional_usdt": round(notional, 6),
                "max_notional_usdt": round(max_notional, 6),
                "risk_based_notional_usdt": round(risk_based_notional, 6),
                "margin_usdt": round(margin_usdt, 6),
                "available_usdt": round(capital_at_entry, 6),
                "balance_source": "paper_action_metadata",
            }

        if action.type == ActionType.CLOSE_POSITION:
            position = self._managed_local_position(engine)
            quantity = float(getattr(position, "quantity", 0.0) or 0.0) if position is not None else 0.0
            notional = quantity * reference_price if quantity > 0 and reference_price > 0 else 0.0
            return {
                "status": "ok",
                "amount": quantity,
                "order_unit": "BTC",
                "close_source": "paper_local_position",
                "expected_notional_usdt": round(notional, 6),
                "base_amount_btc": round(quantity, 8),
                "contracts": 0.0,
            }

        return {"status": "ok", "amount": 0.0}

    def _dynamic_high_leverage_enabled(self) -> bool:
        return bool(self.config.enable_dynamic_high_leverage_structure)

    def _dynamic_high_leverage_default_state(self, engine: Any | None = None) -> dict[str, Any]:
        capital = float(getattr(engine, "capital", 0.0) or 0.0) if engine is not None else 0.0
        return {
            "mode": "offense",
            "capital": capital,
            "drawdown_peak": capital,
            "unit_returns": [],
            "loss_streak": 0,
            "win_streak": 0,
            "last_update_time": None,
            "last_decision": None,
        }

    def _load_dynamic_high_leverage_state(self, engine: Any | None = None) -> dict[str, Any]:
        raw = self.store.get_value("dynamic_high_leverage_structure_state")
        if not raw:
            return self._dynamic_high_leverage_default_state(engine)
        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            return self._dynamic_high_leverage_default_state(engine)
        default = self._dynamic_high_leverage_default_state(engine)
        default.update(state if isinstance(state, dict) else {})
        if not isinstance(default.get("unit_returns"), list):
            default["unit_returns"] = []
        return default

    def _save_dynamic_high_leverage_state(self, state: dict[str, Any]) -> None:
        self.store.set_value("dynamic_high_leverage_structure_state", json.dumps(state, ensure_ascii=False))

    def _dynamic_recent_stats(self, unit_returns: list[Any], lookback: int) -> dict[str, float]:
        values = [float(item) for item in unit_returns[-max(lookback, 0):] if item is not None]
        if not values:
            return {"unit_return_pct": 0.0, "win_rate_pct": 0.0, "count": 0.0}
        wins = sum(1 for item in values if item > 0)
        return {
            "unit_return_pct": sum(values) * 100.0,
            "win_rate_pct": wins / len(values) * 100.0,
            "count": float(len(values)),
        }

    def _dynamic_action_diagnostics(self, action: StrategyAction, sizing: dict[str, Any], engine: Any) -> dict[str, Any]:
        entry_price = float(action.entry_price or 0.0)
        stop_price = float(action.stop_price or 0.0)
        stop_distance_pct = (
            abs(entry_price - stop_price) / entry_price * 100.0
            if entry_price > 0 and stop_price > 0
            else 0.0
        )
        metadata = action.metadata or {}
        regime_label = str(metadata.get("regime_label") or "")
        trail_style = str(metadata.get("trail_style") or "")
        is_high_growth = regime_label == "high_growth"
        is_tight_stop = 0.0 < stop_distance_pct <= float(self.config.dynamic_tight_stop_pct)
        return {
            "entry_price": entry_price,
            "stop_price": stop_price,
            "stop_distance_pct": stop_distance_pct,
            "regime_label": regime_label,
            "trail_style": trail_style,
            "direction": action.direction,
            "feature_adx": float(metadata.get("feature_adx", 0.0) or 0.0),
            "feature_momentum": float(metadata.get("feature_momentum", 0.0) or 0.0),
            "feature_ema_gap": float(metadata.get("feature_ema_gap", 0.0) or 0.0),
            "feature_bullish_structure": bool(metadata.get("feature_bullish_structure", False)),
            "feature_bearish_structure": bool(metadata.get("feature_bearish_structure", False)),
            "is_high_growth": is_high_growth,
            "is_tight_stop": is_tight_stop,
            "available_usdt": float(
                sizing.get("available_usdt", 0.0)
                or metadata.get("capital_at_entry", 0.0)
                or getattr(engine, "capital", 0.0)
                or 0.0
            ),
        }

    def _dynamic_configured_set(self, value: Any) -> set[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        try:
            items = [str(item) for item in value if str(item)]
        except TypeError:
            return None
        return set(items) if items else None

    def _dynamic_failed_breakout_guard(
        self,
        leverage: float,
        risk_mode: str,
        diagnostics: dict[str, Any],
    ) -> tuple[float, list[str]]:
        if not bool(self.config.dynamic_failed_breakout_guard_enabled):
            return leverage, []
        if leverage < float(self.config.dynamic_failed_breakout_guard_min_leverage):
            return leverage, []

        direction = str(diagnostics.get("direction") or "")
        regime_label = str(diagnostics.get("regime_label") or "")
        allowed_directions = self._dynamic_configured_set(self.config.dynamic_failed_breakout_guard_directions)
        allowed_regimes = self._dynamic_configured_set(self.config.dynamic_failed_breakout_guard_regime_labels)
        allowed_modes = self._dynamic_configured_set(self.config.dynamic_failed_breakout_guard_risk_modes)
        if allowed_directions is not None and direction not in allowed_directions:
            return leverage, []
        if allowed_regimes is not None and regime_label not in allowed_regimes:
            return leverage, []
        if allowed_modes is not None and risk_mode not in allowed_modes:
            return leverage, []

        sign = 1.0 if direction == "BULL" else -1.0
        momentum_pct = float(diagnostics.get("feature_momentum", 0.0) or 0.0) * 100.0 * sign
        ema_gap_pct = float(diagnostics.get("feature_ema_gap", 0.0) or 0.0) * 100.0 * sign
        adx = float(diagnostics.get("feature_adx", 0.0) or 0.0)
        directional_structure = (
            bool(diagnostics.get("feature_bullish_structure", False))
            if direction == "BULL"
            else bool(diagnostics.get("feature_bearish_structure", False))
        )
        checks = {
            "momentum": momentum_pct >= float(self.config.dynamic_failed_breakout_guard_min_momentum_pct),
            "ema_gap": ema_gap_pct >= float(self.config.dynamic_failed_breakout_guard_min_ema_gap_pct),
            "adx": adx >= float(self.config.dynamic_failed_breakout_guard_min_adx),
            "structure": directional_structure,
        }
        quality_score = sum(1 for passed in checks.values() if passed)
        min_score = int(self.config.dynamic_failed_breakout_guard_min_quality_score)
        if quality_score >= min_score:
            return leverage, []
        guarded_leverage = min(leverage, float(self.config.dynamic_failed_breakout_guard_leverage))
        if guarded_leverage >= leverage:
            return leverage, []
        return guarded_leverage, [f"failed_breakout_guard:{quality_score}/{min_score}"]

    def _dynamic_signal_allows_reattack(self, diagnostics: dict[str, Any]) -> bool:
        mode = str(self.config.dynamic_reattack_signal_mode or "high_growth_or_tight")
        if mode == "any":
            return True
        if mode == "high_growth":
            return bool(diagnostics["is_high_growth"])
        if mode == "tight":
            return bool(diagnostics["is_tight_stop"])
        if mode in {"high_growth_or_tight", "high_growth_or_tight_or_structure"}:
            return bool(diagnostics["is_high_growth"] or diagnostics["is_tight_stop"])
        return bool(diagnostics["is_high_growth"] or diagnostics["is_tight_stop"])

    def _dynamic_next_mode(
        self,
        state: dict[str, Any],
        diagnostics: dict[str, Any],
    ) -> tuple[str, list[str], dict[str, Any]]:
        unit_returns = state.get("unit_returns") if isinstance(state.get("unit_returns"), list) else []
        recent = self._dynamic_recent_stats(unit_returns, int(self.config.dynamic_state_lookback_trades))
        short = self._dynamic_recent_stats(unit_returns, int(self.config.dynamic_reattack_lookback_trades))
        mode = str(state.get("mode") or "offense")
        capital = float(state.get("capital", 0.0) or 0.0)
        peak = float(state.get("drawdown_peak", capital) or capital)
        drawdown_pct = (peak - capital) / peak * 100.0 if peak > 0 else 0.0
        reasons: list[str] = []

        if mode != "defense":
            if recent["count"] > 0 and recent["unit_return_pct"] <= float(self.config.dynamic_defense_enter_unit_return_pct):
                reasons.append("low_recent_unit_return")
            if recent["count"] > 0 and recent["win_rate_pct"] <= float(self.config.dynamic_defense_enter_win_rate_pct):
                reasons.append("low_recent_win_rate")
            if int(state.get("loss_streak", 0) or 0) >= int(self.config.dynamic_loss_streak_threshold):
                reasons.append("loss_streak")
            if drawdown_pct >= float(self.config.dynamic_drawdown_threshold_pct):
                reasons.append("drawdown")
            return ("defense" if reasons else "offense"), reasons, {"recent": recent, "short": short, "drawdown_pct": drawdown_pct}

        recovered = (
            recent["count"] > 0
            and recent["unit_return_pct"] >= float(self.config.dynamic_offense_enter_unit_return_pct)
            and recent["win_rate_pct"] >= float(self.config.dynamic_offense_enter_win_rate_pct)
        )
        if recovered:
            return "offense", ["recovered_recent_signal"], {"recent": recent, "short": short, "drawdown_pct": drawdown_pct}

        reattack = (
            short["count"] > 0
            and short["unit_return_pct"] >= float(self.config.dynamic_reattack_unit_return_pct)
            and short["win_rate_pct"] >= float(self.config.dynamic_reattack_win_rate_pct)
            and self._dynamic_signal_allows_reattack(diagnostics)
        )
        if reattack:
            return "offense", ["short_window_reattack"], {"recent": recent, "short": short, "drawdown_pct": drawdown_pct}
        return "defense", reasons, {"recent": recent, "short": short, "drawdown_pct": drawdown_pct}

    def _dynamic_select_effective_leverage(
        self,
        state: dict[str, Any],
        risk_mode: str,
        diagnostics: dict[str, Any],
        mode_stats: dict[str, Any],
    ) -> tuple[float, list[str]]:
        max_leverage = float(self.config.dynamic_max_effective_leverage)
        if risk_mode == "defense":
            return min(float(self.config.dynamic_defense_leverage), max_leverage), ["state_defense_reduce"]

        leverage = float(self.config.dynamic_base_leverage)
        reasons = ["base"]
        if diagnostics["is_high_growth"]:
            leverage = max(leverage, float(self.config.dynamic_high_growth_leverage))
            reasons.append("high_growth")
        if diagnostics["is_tight_stop"]:
            leverage = max(leverage, float(self.config.dynamic_tight_stop_leverage))
            reasons.append("tight_stop")
        if int(state.get("win_streak", 0) or 0) >= int(self.config.dynamic_win_streak_threshold):
            leverage = min(max_leverage, leverage * 1.15)
            reasons.append("win_streak_expand")

        health = self._dynamic_recent_stats(
            state.get("unit_returns") if isinstance(state.get("unit_returns"), list) else [],
            int(self.config.dynamic_health_lookback_trades),
        )
        if (
            health["count"] > 0
            and (
                health["unit_return_pct"] < float(self.config.dynamic_health_min_unit_return_pct)
                or health["win_rate_pct"] < float(self.config.dynamic_health_min_win_rate_pct)
            )
        ):
            leverage = min(leverage, float(self.config.dynamic_unhealthy_leverage))
            reasons.append("market_unhealthy_reduce")

        if float(mode_stats.get("drawdown_pct", 0.0) or 0.0) >= float(self.config.dynamic_drawdown_threshold_pct):
            leverage = min(leverage, float(self.config.dynamic_drawdown_leverage))
            reasons.append("drawdown_reduce")

        guarded_leverage, guard_reasons = self._dynamic_failed_breakout_guard(leverage, risk_mode, diagnostics)
        if guard_reasons:
            leverage = guarded_leverage
            reasons.extend(guard_reasons)

        return max(0.0, min(leverage, max_leverage)), reasons

    def _dynamic_high_leverage_pre_open(
        self,
        action: StrategyAction,
        sizing: dict[str, Any],
        engine: Any,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if not self._dynamic_high_leverage_enabled() or action.type not in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}:
            return sizing, None
        if bool((action.metadata or {}).get("overlay_formal_fixed")):
            return sizing, None
        if self._overlay_should_skip_dynamic_high_leverage(action):
            return sizing, None

        state = self._load_dynamic_high_leverage_state(engine)
        diagnostics = self._dynamic_action_diagnostics(action, sizing, engine)
        risk_mode, mode_reasons, mode_stats = self._dynamic_next_mode(state, diagnostics)
        effective_leverage, leverage_reasons = self._dynamic_select_effective_leverage(
            state,
            risk_mode,
            diagnostics,
            mode_stats,
        )
        max_stop_distance = (
            float(self.config.dynamic_defense_max_stop_distance_pct)
            if risk_mode == "defense"
            else (
                float(self.config.dynamic_high_growth_max_stop_distance_pct)
                if diagnostics["is_high_growth"]
                else float(self.config.dynamic_max_stop_distance_pct)
            )
        )
        decision = {
            "risk_mode": risk_mode,
            "mode_reasons": mode_reasons,
            "mode_stats": mode_stats,
            "effective_leverage": round(effective_leverage, 6),
            "leverage_reasons": leverage_reasons,
            "diagnostics": diagnostics,
            "max_stop_distance_pct": max_stop_distance,
        }
        state["mode"] = risk_mode
        state["last_decision"] = decision
        state["last_update_time"] = action.timestamp
        self._save_dynamic_high_leverage_state(state)

        if diagnostics["stop_distance_pct"] > max_stop_distance:
            if self._shadow_gate_enabled():
                shadow_state = self._load_shadow_gate_state(engine)
                shadow_state["real_position_open"] = False
                shadow_state["real_position_direction"] = None
                shadow_state["paper_entry_time"] = action.timestamp
                self._shadow_append_event(
                    shadow_state,
                    {
                        "time": action.timestamp,
                        "event": "skip_open",
                        "reason": "dynamic_high_leverage_stop_distance_too_wide",
                        "direction": action.direction,
                        "decision": decision,
                    },
                )
                self._save_shadow_gate_state(shadow_state)
            return sizing, {
                "status": "dynamic_high_leverage_skipped_open",
                "action": action.type.value,
                "direction": action.direction,
                "reason": "stop_distance_too_wide",
                "decision": decision,
            }

        available_usdt = float(diagnostics["available_usdt"])
        if available_usdt <= 0 or effective_leverage <= 0:
            return sizing, {
                "status": "dynamic_high_leverage_skipped_open",
                "action": action.type.value,
                "direction": action.direction,
                "reason": "invalid_available_usdt_or_leverage",
                "decision": decision,
            }

        target_notional = available_usdt * effective_leverage
        reference_price = float(action.entry_price or 0.0)
        if reference_price <= 0:
            return sizing, {
                "status": "dynamic_high_leverage_skipped_open",
                "action": action.type.value,
                "direction": action.direction,
                "reason": "invalid_reference_price",
                "decision": decision,
            }
        metadata = action.metadata or {}
        requested_notional = (
            self._safe_float(sizing.get("risk_based_notional_usdt"))
            or self._safe_float(metadata.get("risk_based_notional"))
            or self._safe_float(sizing.get("notional_usdt"))
            or self._safe_float(metadata.get("notional"))
        )
        position = getattr(engine, "position", None)
        if position is not None:
            setattr(position, "execution_effective_leverage", round(effective_leverage, 6))
            setattr(position, "execution_risk_mode", risk_mode)
            setattr(position, "execution_leverage_reasons", list(leverage_reasons))
            setattr(position, "execution_requested_notional", requested_notional)
            setattr(position, "execution_target_notional", round(target_notional, 6))
            setattr(position, "execution_guard_diagnostics", diagnostics)
        if self.config.mode == "paper":
            quantity = target_notional / reference_price
            adjusted = {
                "amount": round(quantity, 8),
                "order_unit": "BTC",
                "requested_base_amount_btc": round(quantity, 8),
                "base_amount_btc": round(quantity, 8),
                "contract_size": 0.0,
                "expected_notional_usdt": round(target_notional, 6),
                "requested_notional_usdt": round(target_notional, 6),
            }
        else:
            adjusted = self._build_order_sizing(target_notional / reference_price, target_notional, reference_price)
        adjusted.update(
            {
                "status": "ok",
                "notional_usdt": round(target_notional, 6),
                "max_notional_usdt": round(target_notional, 6),
                "risk_based_notional_usdt": round(float(sizing.get("risk_based_notional_usdt", target_notional) or target_notional), 6),
                "margin_usdt": round(target_notional / self.config.leverage if self.config.leverage > 0 else target_notional, 6),
                "available_usdt": round(available_usdt, 6),
                "balance_source": sizing.get("balance_source", "dynamic_high_leverage"),
                "dynamic_high_leverage": decision,
            }
        )
        return adjusted, None

    def _dynamic_high_leverage_after_close(self, action: StrategyAction, engine: Any) -> None:
        if not self._dynamic_high_leverage_enabled() or action.type != ActionType.CLOSE_POSITION:
            return
        state = self._load_dynamic_high_leverage_state(engine)
        latest_trade = engine.trades[-1] if getattr(engine, "trades", None) else None
        pnl = float((action.metadata or {}).get("net_pnl", 0.0) or 0.0)
        notional = float(getattr(latest_trade, "notional", 0.0) or 0.0) if latest_trade is not None else 0.0
        unit_return = pnl / notional if notional > 0 else 0.0
        unit_returns = state.get("unit_returns") if isinstance(state.get("unit_returns"), list) else []
        unit_returns.append(unit_return)
        state["unit_returns"] = unit_returns[-100:]
        if pnl > 0:
            state["win_streak"] = int(state.get("win_streak", 0) or 0) + 1
            state["loss_streak"] = 0
        else:
            state["loss_streak"] = int(state.get("loss_streak", 0) or 0) + 1
            state["win_streak"] = 0
        capital = float(getattr(engine, "capital", 0.0) or state.get("capital", 0.0) or 0.0)
        state["capital"] = capital
        state["drawdown_peak"] = max(float(state.get("drawdown_peak", capital) or capital), capital)
        state["last_update_time"] = action.timestamp
        state["last_close"] = {
            "time": action.timestamp,
            "pnl": pnl,
            "notional": notional,
            "unit_return": unit_return,
            "capital": capital,
        }
        self._save_dynamic_high_leverage_state(state)

    def _load_markets(self) -> dict[str, Any]:
        if self._markets_cache is None:
            self._markets_cache = self.client.load_markets()
        return self._markets_cache

    def _market(self) -> dict[str, Any]:
        markets = self._load_markets()
        market = markets.get(self.config.symbol)
        if market is None:
            raise ValueError(f"Market metadata missing for {self.config.symbol}")
        return market

    def _build_order_sizing(self, base_amount_btc: float, requested_notional_usdt: float, reference_price: float) -> dict[str, Any]:
        market = self._market()
        if market.get("contract"):
            contract_size = float(market.get("contractSize") or 1.0)
            contracts = base_amount_btc / contract_size if contract_size > 0 else 0.0
            amount = float(self.client.exchange.amount_to_precision(self.config.symbol, contracts))
            actual_base_amount = amount * contract_size
            order_unit = "contracts"
        else:
            amount = float(self.client.exchange.amount_to_precision(self.config.symbol, base_amount_btc))
            actual_base_amount = amount
            contract_size = 0.0
            order_unit = "BTC"
        expected_notional_usdt = actual_base_amount * reference_price
        return {
            "amount": amount,
            "order_unit": order_unit,
            "requested_base_amount_btc": round(base_amount_btc, 8),
            "base_amount_btc": round(actual_base_amount, 8),
            "contract_size": contract_size,
            "expected_notional_usdt": round(expected_notional_usdt, 6),
            "requested_notional_usdt": round(requested_notional_usdt, 6),
        }

    def _sync_live_capital(self, engine: Any) -> float:
        live_capital = float(getattr(engine, "capital", 0.0) or 0.0)
        try:
            balance = self.client.fetch_balance()
            available_usdt, _ = self._extract_available_usdt(balance)
        except Exception:
            return live_capital
        engine.capital = available_usdt
        return float(engine.capital)

    def _build_exchange_bracket_params(self, action: StrategyAction, attach_algo_client_id: str | None = None) -> dict[str, Any]:
        if action.stop_price is None or action.target_price is None:
            return {}
        trigger_price_type = self.config.exchange_trigger_price_type
        return {
            "attachAlgoOrds": [
                {
                    "slTriggerPx": self.client.exchange.price_to_precision(self.config.symbol, action.stop_price),
                    "slOrdPx": "-1",
                    "slTriggerPxType": trigger_price_type,
                    "tpTriggerPx": self.client.exchange.price_to_precision(self.config.symbol, action.target_price),
                    "tpOrdPx": "-1",
                    "tpTriggerPxType": trigger_price_type,
                    **({"attachAlgoClOrdId": attach_algo_client_id} if attach_algo_client_id else {}),
                }
            ]
        }

    def _extract_available_usdt(self, balance: dict[str, Any]) -> tuple[float, str]:
        candidates = []
        usdt_entry = balance.get("USDT") if isinstance(balance, dict) else None
        if isinstance(usdt_entry, dict):
            for key in ("free", "available", "availableBalance", "cash", "total"):
                value = usdt_entry.get(key)
                if value is not None:
                    candidates.append((value, f"USDT.{key}"))

        info = balance.get("info") if isinstance(balance, dict) else None
        if isinstance(info, dict):
            details = info.get("data")
            if isinstance(details, list):
                for row in details:
                    if not isinstance(row, dict):
                        continue
                    details_list = row.get("details")
                    if not isinstance(details_list, list):
                        continue
                    for detail in details_list:
                        if not isinstance(detail, dict):
                            continue
                        if detail.get("ccy") != "USDT":
                            continue
                        for key in ("availBal", "cashBal", "eq", "availEq"):
                            value = detail.get(key)
                            if value not in (None, ""):
                                candidates.append((value, f"info.details.{key}"))

        for value, source in candidates:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric > 0:
                return numeric, source
        raise ValueError("Unable to extract positive USDT balance from exchange response")

    def _extract_position_amount(self, positions: list[dict[str, Any]], pos_side: str) -> float:
        return self._extract_position_state(positions, pos_side)["contracts"]

    def _extract_position_state(
        self,
        positions: list[dict[str, Any]],
        pos_side: str,
        reference_price: float | None = None,
    ) -> dict[str, Any]:
        market = self._market()
        default_contract_size = float(market.get("contractSize") or 1.0) if market.get("contract") else 0.0
        for position in positions:
            if not isinstance(position, dict):
                continue
            symbol = position.get("symbol") or position.get("instId")
            if symbol and symbol != self.config.symbol:
                continue
            # Check posSide matches
            position_pos_side = position.get("posSide") or position.get("side")
            info = position.get("info")
            if isinstance(info, dict):
                position_pos_side = position_pos_side or info.get("posSide") or info.get("side")
            if position_pos_side and position_pos_side.lower() != pos_side.lower():
                continue
            contracts = 0.0
            for key in ("contracts", "positionAmt", "pos", "size"):
                value = position.get(key)
                if value in (None, ""):
                    continue
                try:
                    contracts = abs(float(value))
                except (TypeError, ValueError):
                    continue
                if contracts > 0:
                    break
            if isinstance(info, dict):
                for key in ("pos", "availPos"):
                    value = info.get(key)
                    if value in (None, ""):
                        continue
                    try:
                        contracts = abs(float(value))
                    except (TypeError, ValueError):
                        continue
                    if contracts > 0:
                        break
            if contracts <= 0:
                continue
            contract_size = default_contract_size
            for key in ("contractSize",):
                value = position.get(key)
                if value not in (None, ""):
                    try:
                        contract_size = float(value)
                    except (TypeError, ValueError):
                        pass
            if isinstance(info, dict):
                for key in ("contractSize", "ctVal"):
                    value = info.get(key)
                    if value not in (None, ""):
                        try:
                            contract_size = float(value)
                        except (TypeError, ValueError):
                            pass
            base_amount_btc = contracts * contract_size if market.get("contract") else contracts
            notional_usdt = None
            for key in ("notional", "notionalUsd", "positionValue"):
                value = position.get(key)
                if value not in (None, ""):
                    try:
                        notional_usdt = abs(float(value))
                    except (TypeError, ValueError):
                        continue
                    if notional_usdt > 0:
                        break
            if (notional_usdt is None or notional_usdt <= 0) and isinstance(info, dict):
                for key in ("notionalUsd", "notional", "posValue"):
                    value = info.get(key)
                    if value not in (None, ""):
                        try:
                            notional_usdt = abs(float(value))
                        except (TypeError, ValueError):
                            continue
                        if notional_usdt > 0:
                            break
            if (notional_usdt is None or notional_usdt <= 0) and reference_price and reference_price > 0:
                notional_usdt = base_amount_btc * reference_price
            close_order_algos = info.get("closeOrderAlgo") if isinstance(info, dict) and isinstance(info.get("closeOrderAlgo"), list) else []
            return {
                "contracts": contracts,
                "contract_size": contract_size,
                "base_amount_btc": base_amount_btc,
                "notional_usdt": float(notional_usdt or 0.0),
                "close_order_algos": close_order_algos,
                "raw": position,
            }
        return {
            "contracts": 0.0,
            "contract_size": default_contract_size,
            "base_amount_btc": 0.0,
            "notional_usdt": 0.0,
            "close_order_algos": [],
            "raw": None,
        }

    def _fetch_position_state(self, pos_side: str, reference_price: float | None = None) -> dict[str, Any]:
        positions = self.client.fetch_positions([self.config.symbol])
        return self._extract_position_state(positions, pos_side, reference_price=reference_price)

    def _wait_for_position_state(
        self,
        pos_side: str,
        *,
        expect_open: bool,
        reference_price: float | None = None,
        retries: int = 5,
        delay_seconds: float = 1.0,
    ) -> dict[str, Any]:
        last_state = {
            "contracts": 0.0,
            "contract_size": float(self._market().get("contractSize") or 0.0),
            "base_amount_btc": 0.0,
            "notional_usdt": 0.0,
            "raw": None,
        }
        for attempt in range(retries):
            last_state = self._fetch_position_state(pos_side, reference_price=reference_price)
            if expect_open and last_state["contracts"] > 0:
                return last_state
            if not expect_open and last_state["contracts"] <= 0:
                return last_state
            if attempt + 1 < retries:
                time.sleep(delay_seconds)
        return last_state

    def _generate_attach_algo_client_id(self) -> str:
        return f"tpsl{uuid.uuid4().hex[:28]}"

    def _extract_attached_algo_identity(self, position_state: dict[str, Any]) -> dict[str, str | None]:
        close_order_algos = position_state.get("close_order_algos") or []
        for algo in close_order_algos:
            if not isinstance(algo, dict):
                continue
            algo_id = algo.get("attachAlgoId") or algo.get("algoId")
            algo_client_id = (
                algo.get("attachAlgoClOrdId")
                or algo.get("algoClOrdId")
                or algo.get("slAttachAlgoClOrdId")
                or algo.get("tpAttachAlgoClOrdId")
            )
            if algo_id or algo_client_id:
                return {
                    "attach_algo_id": str(algo_id) if algo_id else None,
                    "attach_algo_client_id": str(algo_client_id) if algo_client_id else None,
                }
        return {"attach_algo_id": None, "attach_algo_client_id": None}

    def _safe_float(self, value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_total_usdt(self, balance: dict[str, Any]) -> float:
        candidates = []
        usdt_entry = balance.get("USDT") if isinstance(balance, dict) else None
        if isinstance(usdt_entry, dict):
            for key in ("total", "cash", "equity", "free"):
                value = usdt_entry.get(key)
                if value is not None:
                    candidates.append(value)

        info = balance.get("info") if isinstance(balance, dict) else None
        if isinstance(info, dict):
            details = info.get("data")
            if isinstance(details, list):
                for row in details:
                    if not isinstance(row, dict):
                        continue
                    details_list = row.get("details")
                    if not isinstance(details_list, list):
                        continue
                    for detail in details_list:
                        if not isinstance(detail, dict) or detail.get("ccy") != "USDT":
                            continue
                        for key in ("eq", "cashBal", "availEq", "availBal"):
                            value = detail.get(key)
                            if value not in (None, ""):
                                candidates.append(value)

        for value in candidates:
            numeric = self._safe_float(value)
            if numeric is not None and numeric > 0:
                return numeric
        raise ValueError("Unable to extract positive total USDT balance from exchange response")

    def _fetch_pending_algo_orders(self, ord_type: str = "oco") -> list[dict[str, Any]]:
        response = self.client.fetch_pending_algo_orders({"ordType": ord_type})
        data = response.get("data")
        return data if isinstance(data, list) else []

    def _select_pending_algo_order(self, pos_side: str, local_position: Any | None = None) -> dict[str, Any] | None:
        try:
            pending_orders = self._fetch_pending_algo_orders("oco")
        except Exception:
            return None
        market_id = self._market()["id"]
        candidates = []
        for order in pending_orders:
            if not isinstance(order, dict):
                continue
            if order.get("instId") != market_id:
                continue
            if order.get("ordType") != "oco":
                continue
            if order.get("state") not in {"live", "effective"}:
                continue
            if order.get("posSide") != pos_side:
                continue
            candidates.append(order)

        if not candidates:
            return None

        local_algo_id = str(getattr(local_position, "exchange_attach_algo_id", "") or "")
        local_algo_client_id = str(getattr(local_position, "exchange_attach_algo_client_id", "") or "")
        for order in candidates:
            algo_id = str(order.get("algoId") or "")
            algo_client_id = str(order.get("algoClOrdId") or "")
            if local_algo_id and algo_id == local_algo_id:
                return order
            if local_algo_client_id and algo_client_id == local_algo_client_id:
                return order
        return candidates[0]

    def _extract_pending_algo_metadata(self, algo_order: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(algo_order, dict):
            return {
                "algo_id": None,
                "algo_client_id": None,
                "stop_price": None,
                "target_price": None,
            }
        return {
            "algo_id": str(algo_order.get("algoId")) if algo_order.get("algoId") else None,
            "algo_client_id": str(algo_order.get("algoClOrdId")) if algo_order.get("algoClOrdId") else None,
            "stop_price": self._safe_float(algo_order.get("slTriggerPx")),
            "target_price": self._safe_float(algo_order.get("tpTriggerPx")),
        }

    def _extract_exchange_entry_price(self, exchange_state: dict[str, Any], fallback: float | None = None) -> float | None:
        raw = exchange_state.get("raw")
        if isinstance(raw, dict):
            for key in ("entryPrice", "avgPx"):
                value = self._safe_float(raw.get(key))
                if value is not None and value > 0:
                    return value
            info = raw.get("info")
            if isinstance(info, dict):
                for key in ("avgPx", "entryPrice"):
                    value = self._safe_float(info.get(key))
                    if value is not None and value > 0:
                        return value
        return fallback

    def _extract_position_fee(self, exchange_state: dict[str, Any], fallback: float | None = None) -> float | None:
        raw = exchange_state.get("raw")
        if isinstance(raw, dict):
            info = raw.get("info")
            if isinstance(info, dict):
                fee_value = self._safe_float(info.get("fee"))
                if fee_value is not None and fee_value != 0:
                    return abs(fee_value)
        return fallback

    def _scale_value_by_quantity(self, value: float | None, old_quantity: float, new_quantity: float) -> float:
        if value is None:
            return 0.0
        if old_quantity <= 0:
            return float(value)
        return float(value) * (new_quantity / old_quantity)

    def _save_engine_snapshot(self, engine: Any) -> dict[str, Any]:
        snapshot = engine.snapshot()
        self.store.save_snapshot(snapshot)
        return asdict(snapshot)

    def _current_live_total_usdt(self, fallback: float) -> float:
        try:
            balance = self.client.fetch_balance()
            total = self._extract_total_usdt(balance)
            if total > 0:
                return total
            available, _ = self._extract_available_usdt(balance)
            if available > 0:
                return available
        except Exception:
            pass
        return float(fallback)

    def _estimate_external_exit_price(self, position: Any, net_pnl: float) -> tuple[float, float, float]:
        quantity = abs(float(getattr(position, "quantity", 0.0) or 0.0))
        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        entry_fee = float(getattr(position, "entry_fee", 0.0) or 0.0)
        fee_rate = max(float(self.config.taker_fee_rate or 0.0), 0.0)
        if quantity <= 0 or entry_price <= 0:
            return entry_price, net_pnl, 0.0
        if getattr(position, "direction", None) == Direction.BULL:
            denominator = quantity * max(1.0 - fee_rate, 1e-9)
            exit_price = (float(net_pnl) + quantity * entry_price + entry_fee) / denominator
            gross_pnl = quantity * (exit_price - entry_price)
        else:
            denominator = quantity * (1.0 + fee_rate)
            exit_price = (quantity * entry_price - float(net_pnl) - entry_fee) / denominator
            gross_pnl = quantity * (entry_price - exit_price)
        exit_fee = quantity * max(exit_price, 0.0) * fee_rate
        return exit_price, gross_pnl, entry_fee + exit_fee

    def _external_flat_exit_reason(self, position: Any, exit_price: float) -> str:
        stop_price = self._safe_float(getattr(position, "sl_price", None))
        target_price = self._safe_float(getattr(position, "target_price", None))
        direction = getattr(position, "direction", None)
        if exit_price > 0 and stop_price and stop_price > 0:
            stop_diff = abs(exit_price - stop_price) / stop_price
            if stop_diff <= 0.003:
                return "external_stop_loss"
            if direction == Direction.BULL and exit_price <= stop_price:
                return "external_stop_loss"
            if direction == Direction.BEAR and exit_price >= stop_price:
                return "external_stop_loss"
        if exit_price > 0 and target_price and target_price > 0:
            target_diff = abs(exit_price - target_price) / target_price
            if target_diff <= 0.003:
                return "external_target_rr"
            if direction == Direction.BULL and exit_price >= target_price:
                return "external_target_rr"
            if direction == Direction.BEAR and exit_price <= target_price:
                return "external_target_rr"
        return "external_flat_sync"

    def _record_external_flat_close(
        self,
        engine: Any,
        position: Any,
        *,
        context: str,
        timestamp: str,
        exit_idx: int | None = None,
    ) -> StrategyAction | None:
        quantity = abs(float(getattr(position, "quantity", 0.0) or 0.0))
        capital_at_entry = float(getattr(position, "capital_at_entry", 0.0) or 0.0)
        if quantity <= 0 or capital_at_entry <= 0:
            return None
        live_total = self._current_live_total_usdt(float(getattr(engine, "capital", 0.0) or capital_at_entry))
        net_pnl = live_total - capital_at_entry
        exit_price, gross_pnl, fees = self._estimate_external_exit_price(position, net_pnl)
        slippage_cost = float(getattr(position, "entry_slippage_cost", 0.0) or 0.0)
        risk_amount = float(getattr(position, "risk_amount", 0.0) or 0.0)
        rr_ratio = net_pnl / risk_amount if risk_amount > 0 else 0.0
        pnl_pct = net_pnl / capital_at_entry if capital_at_entry > 0 else 0.0
        reason = self._external_flat_exit_reason(position, exit_price)
        trade = Trade(
            entry_time=str(getattr(position, "entry_time", "")),
            exit_time=timestamp,
            direction=str(getattr(position, "direction", "")),
            signal_entry_price=float(getattr(position, "signal_entry_price", 0.0) or 0.0),
            entry_price=float(getattr(position, "entry_price", 0.0) or 0.0),
            signal_exit_price=exit_price,
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            fees=fees,
            slippage_cost=slippage_cost,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            rr_ratio=rr_ratio,
            exit_reason=reason,
            capital_at_entry=capital_at_entry,
            notional=self._safe_float(getattr(position, "notional", None)),
            quantity=quantity,
            entry_idx=getattr(position, "entry_idx", None),
            initial_stop_price=self._safe_float(getattr(position, "initial_sl_price", None)),
            trail_style=getattr(position, "trail_style", None),
            risk_regime=getattr(position, "risk_regime", None),
            regime_label=getattr(position, "regime_label", None),
            time_based_trailing_enabled=bool(getattr(position, "time_based_trailing_enabled", False)),
            auto_tit_reason=getattr(position, "auto_tit_reason", None),
            exit_idx=exit_idx,
            pressure_target_applied=bool(getattr(position, "pressure_target_applied", False)),
            pressure_target_source=getattr(position, "pressure_target_source", None),
            pressure_target_level=self._safe_float(getattr(position, "pressure_target_level", None)),
            pressure_target_rr=self._safe_float(getattr(position, "pressure_target_rr", None)),
            pressure_target_min_rr=self._safe_float(getattr(position, "pressure_target_min_rr", None)),
            pressure_target_dynamic_reason=getattr(position, "pressure_target_dynamic_reason", None),
            pressure_target_update_idx=getattr(position, "pressure_target_update_idx", None),
            pressure_touch_lock_applied=bool(getattr(position, "pressure_touch_lock_applied", False)),
            pressure_touch_lock_source=getattr(position, "pressure_touch_lock_source", None),
            pressure_touch_lock_level=self._safe_float(getattr(position, "pressure_touch_lock_level", None)),
            pressure_touch_lock_rr=self._safe_float(getattr(position, "pressure_touch_lock_rr", None)),
            pressure_touch_lock_update_idx=getattr(position, "pressure_touch_lock_update_idx", None),
        )
        engine.trades.append(trade)
        engine.exit_reasons[reason] = int(engine.exit_reasons.get(reason, 0) or 0) + 1
        engine.capital = live_total
        action = StrategyAction(
            type=ActionType.CLOSE_POSITION,
            timestamp=timestamp,
            direction=getattr(position, "direction", None),
            exit_price=exit_price,
            reason=reason,
            metadata={
                "synthetic": True,
                "source": "external_flat_sync",
                "context": context,
                "gross_pnl": gross_pnl,
                "fees": fees,
                "slippage_cost": slippage_cost,
                "net_pnl": net_pnl,
                "signal_exit_price": exit_price,
                "capital_at_entry": capital_at_entry,
                "live_total_usdt": live_total,
                "rr_ratio": rr_ratio,
                "pnl_pct": pnl_pct,
                "index": exit_idx,
                "exit_idx": exit_idx,
                "candidate_event_type": getattr(position, "candidate_event_type", None),
                "execution_effective_leverage": getattr(position, "execution_effective_leverage", None),
                "execution_risk_mode": getattr(position, "execution_risk_mode", None),
                "execution_leverage_reasons": getattr(position, "execution_leverage_reasons", None),
                "execution_requested_notional": getattr(position, "execution_requested_notional", None),
                "execution_target_notional": getattr(position, "execution_target_notional", None),
                "execution_guard_diagnostics": getattr(position, "execution_guard_diagnostics", None),
            },
        )
        self.record_action(action)
        self._shadow_gate_after_close(action, engine)
        self._dynamic_high_leverage_after_close(action, engine)
        return action

    def _sync_manual_flat_position(self, engine: Any, *, context: str, timestamp: str | None = None, exit_idx: int | None = None) -> None:
        position = getattr(engine, "position", None)
        if position is None:
            return
        direction = getattr(position, "direction", None)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        synthetic_action = self._record_external_flat_close(
            engine,
            position,
            context=context,
            timestamp=timestamp,
            exit_idx=exit_idx,
        )
        payload = {
            "context": context,
            "direction": direction,
            "previous_quantity": float(getattr(position, "quantity", 0.0) or 0.0),
            "message": "Exchange position no longer exists; cleared local snapshot.",
        }
        engine.position = None
        snapshot = self._save_engine_snapshot(engine)
        payload["snapshot"] = snapshot
        if synthetic_action is not None:
            payload["synthetic_close_action"] = asdict(synthetic_action)
        elif self._shadow_gate_enabled():
            gate_state = self._load_shadow_gate_state(engine)
            gate_state["real_position_open"] = False
            gate_state["real_position_direction"] = None
            events = gate_state.setdefault("events", [])
            if isinstance(events, list):
                events.append(
                    {
                        "time": timestamp,
                        "event": "manual_flat_sync",
                        "reason": context,
                        "direction": direction,
                    }
                )
            self._save_shadow_gate_state(gate_state)
            payload["shadow_gate_state"] = gate_state
        self._clear_sota_overlay_open_candidate()
        self._clear_overlay_runtime_position(last_managed_idx=exit_idx)
        self.store.append_action(timestamp, "MANUAL_POSITION_SYNC", payload)
        direction_label = "做多" if direction == "BULL" else "做空" if direction == "BEAR" else "-"
        pnl_line = ""
        if synthetic_action is not None and isinstance(synthetic_action.metadata, dict):
            pnl = self._safe_float(synthetic_action.metadata.get("net_pnl"))
            reason = synthetic_action.reason or "-"
            pnl_line = f"估算PnL: {pnl:.2f}U / {reason}" if pnl is not None else ""
        self._send_telegram(
            "\n".join(
                [
                    "[手动平仓已同步]",
                    f"标的: {self.config.symbol}",
                    f"方向: {direction_label}",
                    f"来源: {context}",
                    "检测到交易所仓位已被手动平掉，本地状态已清空",
                    *([pnl_line] if pnl_line else []),
                    f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ]
            )
        )

    def _record_external_overlay_flat_close(
        self,
        engine: Any,
        position: OverlayRuntimePosition,
        *,
        context: str,
        timestamp: str,
        exit_idx: int | None = None,
    ) -> StrategyAction | None:
        quantity = abs(float(position.quantity or 0.0))
        capital_at_entry = float(position.capital_at_entry or 0.0)
        if quantity <= 0 or capital_at_entry <= 0:
            return None
        live_total = self._current_live_total_usdt(float(getattr(engine, "capital", 0.0) or capital_at_entry))
        net_pnl = live_total - capital_at_entry
        exit_price, gross_pnl, fees = self._estimate_external_exit_price(position, net_pnl)
        slippage_cost = float(position.entry_slippage_cost or 0.0)
        risk_amount = quantity * abs(float(position.risk_points or 0.0))
        rr_ratio = net_pnl / risk_amount if risk_amount > 0 else 0.0
        pnl_pct = net_pnl / capital_at_entry if capital_at_entry > 0 else 0.0
        reason = self._external_flat_exit_reason(position, exit_price)
        trade = Trade(
            entry_time=str(position.entry_time),
            exit_time=timestamp,
            direction=str(position.direction),
            signal_entry_price=float(position.signal_entry_price or 0.0),
            entry_price=float(position.entry_price or 0.0),
            signal_exit_price=exit_price,
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            fees=fees,
            slippage_cost=slippage_cost,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            rr_ratio=rr_ratio,
            exit_reason=reason,
            capital_at_entry=capital_at_entry,
            notional=float(position.notional or 0.0),
            quantity=quantity,
            entry_idx=int(position.entry_idx),
            initial_stop_price=float(position.initial_sl_price or 0.0),
            trail_style="overlay",
            risk_regime="overlay",
            regime_label=str(position.event_type),
            time_based_trailing_enabled=False,
            auto_tit_reason=None,
            exit_idx=exit_idx,
        )
        engine.trades.append(trade)
        engine.exit_reasons[reason] = int(engine.exit_reasons.get(reason, 0) or 0) + 1
        engine.capital = live_total
        action = StrategyAction(
            type=ActionType.CLOSE_POSITION,
            timestamp=timestamp,
            direction=str(position.direction),
            exit_price=exit_price,
            reason=reason,
            metadata={
                "synthetic": True,
                "source": "external_flat_sync",
                "context": context,
                "overlay_event_type": position.event_type,
                "entry_idx": int(position.entry_idx),
                "index": exit_idx,
                "exit_idx": exit_idx,
                "gross_pnl": gross_pnl,
                "fees": fees,
                "slippage_cost": slippage_cost,
                "net_pnl": net_pnl,
                "signal_exit_price": exit_price,
                "capital_at_entry": capital_at_entry,
                "live_total_usdt": live_total,
                "rr_ratio": rr_ratio,
                "pnl_pct": pnl_pct,
            },
        )
        self.record_action(action)
        self._shadow_gate_after_close(action, engine)
        if not self._overlay_should_skip_dynamic_high_leverage(action):
            self._dynamic_high_leverage_after_close(action, engine)
        return action

    def _sync_overlay_flat_position(
        self,
        engine: Any,
        overlay_position: OverlayRuntimePosition,
        *,
        context: str,
        timestamp: str | None = None,
        exit_idx: int | None = None,
    ) -> None:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        synthetic_action = self._record_external_overlay_flat_close(
            engine,
            overlay_position,
            context=context,
            timestamp=timestamp,
            exit_idx=exit_idx,
        )
        payload = {
            "context": context,
            "direction": overlay_position.direction,
            "previous_quantity": float(overlay_position.quantity or 0.0),
            "overlay_event_type": overlay_position.event_type,
            "message": "Exchange overlay position no longer exists; cleared local runtime state.",
        }
        self._clear_overlay_runtime_position(last_managed_idx=exit_idx)
        self._clear_sota_overlay_open_candidate()
        snapshot = self._save_engine_snapshot(engine)
        payload["snapshot"] = snapshot
        if synthetic_action is not None:
            payload["synthetic_close_action"] = asdict(synthetic_action)
        self.store.append_action(timestamp, "MANUAL_POSITION_SYNC", payload)
        direction_label = "做多" if overlay_position.direction == "BULL" else "做空" if overlay_position.direction == "BEAR" else "-"
        pnl_line = ""
        if synthetic_action is not None and isinstance(synthetic_action.metadata, dict):
            pnl = self._safe_float(synthetic_action.metadata.get("net_pnl"))
            reason = synthetic_action.reason or "-"
            pnl_line = f"估算PnL: {pnl:.2f}U / {reason}" if pnl is not None else ""
        self._send_telegram(
            "\n".join(
                [
                    "[Overlay平仓已同步]",
                    f"标的: {self.config.symbol}",
                    f"方向: {direction_label}",
                    f"来源: {context}",
                    "检测到交易所 overlay 仓位已被平掉，本地 runtime 状态已清空",
                    *([pnl_line] if pnl_line else []),
                    f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ]
            )
        )

    def _position_requires_manual_sync(
        self,
        local_position: Any,
        exchange_state: dict[str, Any],
        pending_algo: dict[str, Any] | None,
    ) -> bool:
        local_quantity = abs(float(getattr(local_position, "quantity", 0.0) or 0.0))
        exchange_quantity = abs(float(exchange_state.get("base_amount_btc", 0.0) or 0.0))
        if exchange_quantity <= 0:
            return False
        tolerance_ratio = max(float(self.config.manual_position_sync_size_tolerance_ratio), 0.0)
        quantity_base = max(local_quantity, exchange_quantity, 1e-9)
        quantity_diff_ratio = abs(exchange_quantity - local_quantity) / quantity_base
        if quantity_diff_ratio > tolerance_ratio:
            return True

        exchange_entry_price = self._extract_exchange_entry_price(exchange_state, getattr(local_position, "entry_price", None))
        local_entry_price = self._safe_float(getattr(local_position, "entry_price", None))
        if exchange_entry_price and local_entry_price and local_entry_price > 0:
            entry_diff_bps = abs(exchange_entry_price - local_entry_price) / local_entry_price * 10000
            if entry_diff_bps > max(float(self.config.manual_position_sync_entry_price_tolerance_bps), 0.0):
                return True

        pending = self._extract_pending_algo_metadata(pending_algo)
        if pending["algo_id"] and pending["algo_id"] != getattr(local_position, "exchange_attach_algo_id", None):
            return True
        if pending["algo_client_id"] and pending["algo_client_id"] != getattr(local_position, "exchange_attach_algo_client_id", None):
            return True
        for local_field, pending_key in (("sl_price", "stop_price"), ("target_price", "target_price")):
            pending_price = pending[pending_key]
            local_price = self._safe_float(getattr(local_position, local_field, None))
            if pending_price is None or local_price is None or local_price <= 0:
                continue
            if abs(pending_price - local_price) / local_price > 0.00001:
                return True
        return False

    def _reconcile_manual_position(
        self,
        engine: Any,
        *,
        exchange_state: dict[str, Any],
        pos_side: str,
        context: str,
        pending_algo: dict[str, Any] | None,
    ) -> None:
        position = getattr(engine, "position", None)
        if position is None:
            return

        quantity = abs(float(exchange_state.get("base_amount_btc", 0.0) or 0.0))
        if quantity <= 0:
            return

        pending = self._extract_pending_algo_metadata(pending_algo)
        old_quantity = abs(float(getattr(position, "quantity", 0.0) or 0.0))
        old_entry_price = self._safe_float(getattr(position, "entry_price", None)) or 0.0
        old_stop_price = self._safe_float(getattr(position, "sl_price", None))
        old_target_price = self._safe_float(getattr(position, "target_price", None))

        entry_price = self._extract_exchange_entry_price(exchange_state, getattr(position, "entry_price", None))
        if entry_price is None or entry_price <= 0:
            raise ValueError(f"Unable to reconcile live position ({context}): missing exchange entry price")

        stop_price = pending["stop_price"] or self._safe_float(getattr(position, "sl_price", None))
        initial_sl_price = self._safe_float(getattr(position, "initial_sl_price", None))
        if initial_sl_price is None or initial_sl_price <= 0:
            initial_sl_price = stop_price
        stage = getattr(position, "stage", -1)
        if stage is None:
            stage = -1
        if stage < 0 and stop_price is not None:
            initial_sl_price = stop_price
        if initial_sl_price is None or initial_sl_price <= 0:
            initial_sl_price = entry_price

        target_rr = float(getattr(position, "target_rr", self.config.rr_ratio) or self.config.rr_ratio)
        target_price = pending["target_price"] or self._safe_float(getattr(position, "target_price", None))
        if target_price is None and initial_sl_price > 0:
            risk_price = abs(entry_price - initial_sl_price)
            target_price = entry_price + risk_price * target_rr if pos_side == "long" else entry_price - risk_price * target_rr

        notional = self._safe_float(exchange_state.get("notional_usdt"))
        if notional is None or notional <= 0:
            notional = quantity * entry_price

        risk_price = abs(entry_price - initial_sl_price)
        if risk_price <= 0 and stop_price is not None:
            risk_price = abs(entry_price - stop_price)
        risk_amount = quantity * risk_price

        entry_fee = self._extract_position_fee(
            exchange_state,
            fallback=self._scale_value_by_quantity(getattr(position, "entry_fee", 0.0), old_quantity, quantity),
        )
        entry_slippage_cost = self._scale_value_by_quantity(
            getattr(position, "entry_slippage_cost", 0.0),
            old_quantity,
            quantity,
        )

        try:
            balance = self.client.fetch_balance()
            available_usdt, _ = self._extract_available_usdt(balance)
            total_usdt = self._extract_total_usdt(balance)
            engine.capital = available_usdt
            capital_at_entry = total_usdt
        except Exception:
            capital_at_entry = float(getattr(position, "capital_at_entry", engine.capital) or engine.capital)

        setattr(position, "entry_price", entry_price)
        setattr(position, "sl_price", stop_price if stop_price is not None else getattr(position, "sl_price", None))
        setattr(position, "initial_sl_price", initial_sl_price)
        setattr(position, "target_price", target_price if target_price is not None else getattr(position, "target_price", None))
        setattr(position, "capital_at_entry", capital_at_entry)
        setattr(position, "risk_amount", risk_amount)
        setattr(position, "notional", notional)
        setattr(position, "quantity", quantity)
        setattr(position, "entry_fee", float(entry_fee or 0.0))
        setattr(position, "entry_slippage_cost", float(entry_slippage_cost))
        if pending["algo_id"]:
            setattr(position, "exchange_attach_algo_id", pending["algo_id"])
        if pending["algo_client_id"]:
            setattr(position, "exchange_attach_algo_client_id", pending["algo_client_id"])

        snapshot = self._save_engine_snapshot(engine)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "context": context,
            "direction": getattr(position, "direction", None),
            "pos_side": pos_side,
            "old_quantity": old_quantity,
            "new_quantity": quantity,
            "old_entry_price": old_entry_price,
            "new_entry_price": entry_price,
            "old_stop_price": old_stop_price,
            "new_stop_price": getattr(position, "sl_price", None),
            "old_target_price": old_target_price,
            "new_target_price": getattr(position, "target_price", None),
            "exchange_attach_algo_id": getattr(position, "exchange_attach_algo_id", None),
            "exchange_attach_algo_client_id": getattr(position, "exchange_attach_algo_client_id", None),
            "snapshot": snapshot,
        }
        self.store.append_action(timestamp, "MANUAL_POSITION_SYNC", payload)
        self._send_telegram(
            "\n".join(
                [
                    "[手动仓位已对齐]",
                    f"方向: {'做多' if pos_side == 'long' else '做空'}",
                    f"标的: {self.config.symbol}",
                    f"数量: {old_quantity:.6f} BTC -> {quantity:.6f} BTC",
                    f"均价: {old_entry_price:.1f} -> {entry_price:.1f}",
                    (
                        f"止损/止盈: {getattr(position, 'sl_price', 0.0):.1f} / {getattr(position, 'target_price', 0.0):.1f}"
                        if getattr(position, "sl_price", None) is not None and getattr(position, "target_price", None) is not None
                        else "止损/止盈: -"
                    ),
                    f"来源: {context}",
                    f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ]
            )
        )

    def _apply_open_execution_metadata(
        self,
        engine: Any,
        order: dict[str, Any],
        observed_position: dict[str, Any],
        attach_algo_client_id: str | None,
    ) -> None:
        position = getattr(engine, "position", None)
        if position is None:
            return
        setattr(position, "exchange_order_id", order.get("id"))
        if attach_algo_client_id:
            setattr(position, "exchange_attach_algo_client_id", attach_algo_client_id)
        identity = self._extract_attached_algo_identity(observed_position)
        if identity["attach_algo_id"]:
            setattr(position, "exchange_attach_algo_id", identity["attach_algo_id"])
        if identity["attach_algo_client_id"]:
            setattr(position, "exchange_attach_algo_client_id", identity["attach_algo_client_id"])
        pos_side = "long" if getattr(position, "direction", None) == "BULL" else "short"
        self._reconcile_manual_position(
            engine,
            exchange_state=observed_position,
            pos_side=pos_side,
            context="open_execution_metadata",
            pending_algo=None,
        )

    def _build_attached_algo_amend_order_request(
        self,
        *,
        parent_order_id: str,
        attach_algo_id: str | None,
        attach_algo_client_id: str | None,
        stop_price: float,
        target_price: float,
    ) -> dict[str, Any]:
        trigger_price_type = self.config.exchange_trigger_price_type
        attach_algo: dict[str, Any] = {
            "newSlTriggerPx": self.client.exchange.price_to_precision(self.config.symbol, stop_price),
            "newSlOrdPx": "-1",
            "slTriggerPxType": trigger_price_type,
            "newTpTriggerPx": self.client.exchange.price_to_precision(self.config.symbol, target_price),
            "newTpOrdPx": "-1",
            "tpTriggerPxType": trigger_price_type,
        }
        if attach_algo_id:
            attach_algo["attachAlgoId"] = attach_algo_id
        elif attach_algo_client_id:
            attach_algo["attachAlgoClOrdId"] = attach_algo_client_id
        else:
            raise ValueError("Missing attached algo identifier for amend-order request")
        return {
            "instId": self._market()["id"],
            "ordId": parent_order_id,
            "attachAlgoOrds": [attach_algo],
        }

    def _build_algo_amend_request(
        self,
        *,
        attach_algo_id: str | None,
        attach_algo_client_id: str | None,
        stop_price: float,
        target_price: float,
    ) -> dict[str, Any]:
        trigger_price_type = self.config.exchange_trigger_price_type
        request = {
            "instId": self._market()["id"],
            "newSlTriggerPx": self.client.exchange.price_to_precision(self.config.symbol, stop_price),
            "newSlOrdPx": "-1",
            "newSlTriggerPxType": trigger_price_type,
            "newTpTriggerPx": self.client.exchange.price_to_precision(self.config.symbol, target_price),
            "newTpOrdPx": "-1",
            "newTpTriggerPxType": trigger_price_type,
        }
        if attach_algo_id:
            request["algoId"] = attach_algo_id
        elif attach_algo_client_id:
            request["algoClOrdId"] = attach_algo_client_id
        else:
            raise ValueError("Missing attached algo identifier for amend-algo request")
        return request

    def _amend_exchange_brackets(self, action: StrategyAction, engine: Any) -> dict[str, Any]:
        position = getattr(engine, "position", None)
        if position is None:
            return {"status": "error", "reason": "no_local_position_for_update_stop"}
        if action.stop_price is None:
            return {"status": "error", "reason": "missing_stop_price"}
        target_price = getattr(position, "target_price", None)
        if target_price is None:
            return {"status": "error", "reason": "missing_target_price"}
        pos_side = "long" if getattr(position, "direction", None) == "BULL" else "short"
        observed_position = self._fetch_position_state(pos_side, reference_price=action.stop_price)
        identity = self._extract_attached_algo_identity(observed_position)
        if identity["attach_algo_id"]:
            setattr(position, "exchange_attach_algo_id", identity["attach_algo_id"])
        if identity["attach_algo_client_id"]:
            setattr(position, "exchange_attach_algo_client_id", identity["attach_algo_client_id"])

        primary_error = None
        response = None
        if getattr(position, "exchange_order_id", None) and (
            getattr(position, "exchange_attach_algo_id", None) or getattr(position, "exchange_attach_algo_client_id", None)
        ):
            try:
                response = self.client.amend_order(
                    self._build_attached_algo_amend_order_request(
                        parent_order_id=str(position.exchange_order_id),
                        attach_algo_id=getattr(position, "exchange_attach_algo_id", None),
                        attach_algo_client_id=getattr(position, "exchange_attach_algo_client_id", None),
                        stop_price=action.stop_price,
                        target_price=target_price,
                    )
                )
            except Exception as exc:
                primary_error = str(exc)

        if response is None:
            try:
                response = self.client.amend_algo_order(
                    self._build_algo_amend_request(
                        attach_algo_id=getattr(position, "exchange_attach_algo_id", None),
                        attach_algo_client_id=getattr(position, "exchange_attach_algo_client_id", None),
                        stop_price=action.stop_price,
                        target_price=target_price,
                    )
                )
            except Exception as exc:
                return {
                    "status": "error",
                    "reason": "exchange_bracket_amend_failed",
                    "error": str(exc),
                    "primary_error": primary_error,
                    "stop_price": action.stop_price,
                    "target_price": target_price,
                }

        refreshed_position = self._fetch_position_state(pos_side, reference_price=action.stop_price)
        refreshed_identity = self._extract_attached_algo_identity(refreshed_position)
        if refreshed_identity["attach_algo_id"]:
            setattr(position, "exchange_attach_algo_id", refreshed_identity["attach_algo_id"])
        if refreshed_identity["attach_algo_client_id"]:
            setattr(position, "exchange_attach_algo_client_id", refreshed_identity["attach_algo_client_id"])
        previous_stop_price = getattr(position, "sl_price", None)
        self._send_telegram(
            "\n".join(
                [
                    "[移动止损]",
                    f"方向: {'做多' if pos_side == 'long' else '做空'}",
                    f"标的: {self.config.symbol}",
                    (
                        f"止损更新: {previous_stop_price:.1f} -> {action.stop_price:.1f}"
                        if previous_stop_price is not None
                        else f"止损更新: -> {action.stop_price:.1f}"
                    ),
                    f"止盈保持: {target_price:.1f}",
                    f"阶段: {action.reason or '-'}",
                    f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ]
            )
        )
        return {
            "status": "submitted",
            "action": action.type.value,
            "stop_price": action.stop_price,
            "target_price": target_price,
            "response": response,
            "position_side": pos_side,
            "primary_error": primary_error,
        }

    def _exchange_has_open_position(self) -> bool:
        if self.config.mode != "live":
            return False
        positions = self.client.fetch_positions([self.config.symbol])
        return self._extract_position_amount(positions, "long") > 0 or self._extract_position_amount(positions, "short") > 0

    def _initialize_without_replay(self, engine: Any, latest_closed_idx: int) -> dict[str, Any]:
        if self._exchange_has_open_position():
            raise ValueError(
                "Refusing to initialize live bot with empty local state while exchange still has an open position. "
                "Flatten or manually reconcile the exchange position first."
            )

        last_timestamp = engine._timestamp_for_idx(latest_closed_idx)
        snapshot = engine.snapshot()
        self.store.set_value("last_processed_candle_time", last_timestamp)
        self.store.save_snapshot(snapshot)
        status = {
            "status": "initialized_without_replay",
            "symbol": self.config.symbol,
            "processed_candle_time": last_timestamp,
            "actions": [],
            "trade_count": snapshot.trade_count,
            "position_open": snapshot.position is not None,
            "snapshot": asdict(snapshot),
            "live_capital": engine.capital,
        }
        self.store.append_action(last_timestamp, "INITIALIZE", status)
        return status

    def _assert_live_state_synced(
        self,
        engine: Any,
        *,
        context: str,
        timestamp: str | None = None,
        exit_idx: int | None = None,
    ) -> None:
        if self.config.mode != "live":
            return
        base_position = getattr(engine, "position", None)
        overlay_position = self._load_overlay_runtime_position() if self._live_overlay_enabled() else None
        local_position = base_position if base_position is not None else overlay_position
        local_has_position = local_position is not None
        long_state = self._fetch_position_state("long")
        short_state = self._fetch_position_state("short")
        exchange_has_position = long_state["contracts"] > 0 or short_state["contracts"] > 0
        if self._shadow_gate_enabled():
            gate_state = self._load_shadow_gate_state(engine)
            mirrored = bool(gate_state.get("real_position_open"))
            if local_has_position and not mirrored and not exchange_has_position:
                if self._shadow_gate_allows_unmirrored_local_position(local_position):
                    return
                raise ValueError(
                    f"Live state mismatch ({context}): local shadow position is not mirrored, "
                    "but shadow gate state has no matching skip/open-failed record"
                )
            if local_has_position and not mirrored and exchange_has_position:
                raise ValueError(
                    f"Live state mismatch ({context}): shadow position is not mirrored, "
                    "but exchange still has an open position"
                )
        if local_has_position != exchange_has_position:
            if self.config.enable_manual_position_sync and local_has_position and not exchange_has_position:
                if overlay_position is not None and base_position is None:
                    self._sync_overlay_flat_position(
                        engine,
                        overlay_position,
                        context=context,
                        timestamp=timestamp,
                        exit_idx=exit_idx,
                    )
                else:
                    self._sync_manual_flat_position(engine, context=context, timestamp=timestamp, exit_idx=exit_idx)
                return
            raise ValueError(
                f"Live state mismatch ({context}): local_position={local_has_position}, "
                f"exchange_position={exchange_has_position}"
            )
        if not local_has_position:
            return
        expected_pos_side = "long" if getattr(local_position, "direction", None) == "BULL" else "short"
        exchange_state = long_state if long_state["contracts"] > 0 else short_state
        actual_pos_side = "long" if long_state["contracts"] > 0 else "short"
        if actual_pos_side != expected_pos_side:
            raise ValueError(
                f"Live direction mismatch ({context}): local={expected_pos_side}, exchange={actual_pos_side}"
            )
        if overlay_position is None:
            pending_algo = self._select_pending_algo_order(expected_pos_side, local_position)
            if self.config.enable_manual_position_sync and self._position_requires_manual_sync(
                local_position,
                exchange_state,
                pending_algo,
            ):
                self._reconcile_manual_position(
                    engine,
                    exchange_state=exchange_state,
                    pos_side=expected_pos_side,
                    context=context,
                    pending_algo=pending_algo,
                )
                return
        local_base_amount = abs(float(getattr(local_position, "quantity", 0.0) or 0.0))
        exchange_base_amount = abs(float(exchange_state["base_amount_btc"] or 0.0))
        if local_base_amount > 0:
            tolerance_ratio = max(float(self.config.manual_position_sync_size_tolerance_ratio), 0.0)
            quantity_base = max(local_base_amount, exchange_base_amount, 1e-9)
            quantity_diff_ratio = abs(exchange_base_amount - local_base_amount) / quantity_base
            if quantity_diff_ratio > tolerance_ratio:
                raise ValueError(
                    f"Live size mismatch ({context}): local_base_amount={local_base_amount:.8f} BTC, "
                    f"exchange_base_amount={exchange_base_amount:.8f} BTC"
                )

    def _find_resume_index(self, candles: list[Any]) -> int:
        last_processed = self.store.get_value("last_processed_candle_time")
        min_start = self._minimum_start_index()
        if not last_processed:
            return min_start
        for idx, candle in enumerate(candles):
            candle_time = self._timestamp_from_ts(candle.ts)
            if candle_time > last_processed:
                return max(min_start, idx)
        return max(min_start, len(candles) - 1)

    def _latest_closed_index(self, engine: Any, close_buffer_seconds: int = 5) -> int | None:
        candles = self._engine_candles(engine)
        latest_closed_time = self.latest_closed_candle_time(close_buffer_seconds)
        for idx in range(len(candles) - 1, -1, -1):
            if self._timestamp_from_ts(candles[idx].ts) <= latest_closed_time:
                return idx
        return None

    def latest_closed_candle_time(self, close_buffer_seconds: int = 5) -> str:
        now = datetime.now(timezone.utc) - timedelta(seconds=close_buffer_seconds)
        # OHLCV timestamps are candle open times. At a boundary plus buffer,
        # the candle that just closed is the previous timeframe bucket.
        closed = self._floor_to_timeframe(now) - timedelta(seconds=self._timeframe_seconds())
        return closed.strftime("%Y-%m-%d %H:%M")

    def next_closed_candle_time(self, close_buffer_seconds: int = 5) -> str:
        now = datetime.now(timezone.utc)
        current_boundary = self._floor_to_timeframe(now)
        next_boundary = current_boundary + timedelta(seconds=self._timeframe_seconds() + close_buffer_seconds)
        return next_boundary.strftime("%Y-%m-%d %H:%M:%S")

    def seconds_until_next_close(self, close_buffer_seconds: int = 5) -> int:
        now = datetime.now(timezone.utc)
        current_boundary = self._floor_to_timeframe(now)
        next_boundary = current_boundary + timedelta(seconds=self._timeframe_seconds() + close_buffer_seconds)
        return max(int((next_boundary - now).total_seconds()), 1)

    def _engine_candles(self, engine: Any) -> list[Any]:
        candles = getattr(engine, "candles", None)
        if candles is not None:
            return candles
        candles = getattr(engine, "c15m", None)
        if candles is not None:
            return candles
        raise ValueError(f"Unsupported engine type for candle access: {type(engine).__name__}")

    def _minimum_start_index(self) -> int:
        return 100

    def _timeframe_seconds(self) -> int:
        timeframe = self.config.timeframe.strip().lower()
        unit = timeframe[-1]
        try:
            value = int(timeframe[:-1])
        except ValueError as exc:
            raise ValueError(f"Unsupported timeframe format: {self.config.timeframe}") from exc
        multipliers = {"m": 60, "h": 3600, "d": 86400}
        if unit not in multipliers:
            raise ValueError(f"Unsupported timeframe unit: {self.config.timeframe}")
        return value * multipliers[unit]

    def _floor_to_timeframe(self, dt: datetime) -> datetime:
        timeframe_seconds = self._timeframe_seconds()
        timestamp = int(dt.timestamp())
        floored_timestamp = timestamp - (timestamp % timeframe_seconds)
        return datetime.fromtimestamp(floored_timestamp, tz=timezone.utc)

    def _timestamp_from_ts(self, ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
