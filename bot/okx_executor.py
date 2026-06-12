from __future__ import annotations

import argparse
import json
import random
import time
import uuid
import requests
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from bot.market_data import OhlcvRepository
from bot.okx_client import OkxClient, OkxCredentials
from bot.state_store import StateStore
from strategy.scalp_robust_v2_core import (
    ActionType,
    Direction,
    ScalpRobustEngine,
    StrategyAction,
    StrategyConfig,
    Trade,
    align_timeframes,
    build_precomputed_state_confirmed_4h,
    dataframe_to_candles,
)
from scripts.sota_long_filters import apply_sota_structure_gate
from scripts.sota_liquidity_context import flatten_context_features, liquidity_context_for_entry


FVG_BEAR6_LOOSE_EXIT_PROFILE_NAME = "fvg_bear6_loose_runner"
FVG_BEAR6_LOOSE_EXIT_PROFILE_BUCKET = "fvg_near_bear6_target20"
FVG_BEAR6_LOOSE_EXIT_PROFILE_OVERRIDES = {
    "stage_trigger_rr_mode": "close",
    "time_trailing_rr_mode": "extreme",
    "atr_activation_rr_mode": "extreme",
    "atr_activation_rr": 2.6,
    "atr_loose_multiplier": 3.3,
    "atr_normal_multiplier": 2.8,
    "atr_tight_multiplier": 2.25,
    "S0_trigger_rr": 0.7,
    "S1_trigger_rr": 1.0,
    "S3_trigger_rr": 3.5,
    "S4_close_rr": 1.25,
    "pressure_target_min_rr": 1.75,
    "pressure_touch_lock_enabled": False,
}


@dataclass
class ExternalFlatFillClose:
    exit_price: float
    gross_pnl: float
    fees: float
    net_pnl: float
    live_total: float
    source: str
    synthetic: bool
    entry_fee: float
    exit_fee: float
    close_order_id: str | None = None
    close_fill_count: int = 0
    entry_fill_count: int = 0
    exit_time: str | None = None


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
    markets_cache_path: str | None = "var/okx/markets_cache.json"
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
    live_candidate_priority: list[str] | None = None
    enable_sota_score_gate_live: bool = False
    sota_score_net_min: int = 3
    sota_score_bull_min: int = 8
    sota_score_bear_max: int = 6
    sota_score_conflict_mode: str = "any"
    require_non_bearish_structure_for_long_live: bool = False
    enable_sota_rejected_smc_recall_long_live: bool = False
    sota_rejected_smc_recall_long_condition: str = "sweep_has_fvg"
    sota_rejected_smc_recall_long_reject_stage: str = "structure_gate"
    sota_rejected_smc_recall_long_regime_label: str = "normal"
    sota_rejected_smc_recall_long_target_leverage: float = 8.0
    enable_long_score_bucket_sizing_live: bool = False
    long_score_bucket_sizing_rules: list[dict[str, Any]] | None = None
    enable_fvg_bear6_loose_exit_profile_live: bool = False
    fvg_bear6_loose_exit_profile_bucket: str = FVG_BEAR6_LOOSE_EXIT_PROFILE_BUCKET
    fvg_bear6_loose_exit_profile_name: str = FVG_BEAR6_LOOSE_EXIT_PROFILE_NAME
    fvg_bear6_loose_exit_profile_overrides: dict[str, Any] | None = None
    enable_sota_soft_stop_recovery_overlay_live: bool = False
    sota_soft_stop_live_mode: str = "audit"
    sota_soft_stop_net_min: int = 15
    sota_soft_stop_bear_max: int = 0
    sota_soft_stop_max_leverage: float = 2.0
    sota_soft_stop_buffer_r: float = 1.0
    sota_soft_stop_target_rr: float = 0.0
    sota_soft_stop_max_extension_bars: int = 4
    sota_soft_stop_exclude_score_buckets: list[str] | None = None
    enable_smc_short_live: bool = False
    smc_case: str = "v2_medium_dispbody05_otherlag4_10x"
    smc_target_rr: float = 2.0
    smc_max_hold_bars: int = 40
    smc_trail_style: str = "tight"
    smc_leverage: float = 10.0
    smc_position_size_pct: float = 1.0
    smc_min_liq_buffer_pct: float = 1.2
    smc_maintenance_margin_pct: float = 0.5
    smc_min_entry_idx: int = 0
    enable_gap_smc_short_live: bool = False
    gap_smc_short_case: str = "gap_expansion_21d_other_3x"
    gap_smc_short_min_flat_days: float = 21.0
    gap_smc_short_target_rr: float = 2.0
    gap_smc_short_max_hold_bars: int = 40
    gap_smc_short_trail_style: str = "tight"
    gap_smc_short_leverage: float = 3.0
    gap_smc_short_position_size_pct: float = 1.0
    gap_smc_short_min_liq_buffer_pct: float = 1.2
    gap_smc_short_maintenance_margin_pct: float = 0.5
    gap_smc_short_min_entry_idx: int = 0
    enable_smc_long_live: bool = False
    smc_long_case: str = "fvg_hist_total_rr15_10x_half"
    smc_long_target_rr: float = 1.5
    smc_long_max_hold_bars: int = 40
    smc_long_trail_style: str = "tight"
    smc_long_leverage: float = 10.0
    smc_long_position_size_pct: float = 0.5
    smc_long_min_liq_buffer_pct: float = 1.2
    smc_long_maintenance_margin_pct: float = 0.5
    overlay_skip_dynamic_high_leverage: bool = True
    proxy: str | None = None
    api_key: str | None = None
    api_secret: str | None = None
    api_passphrase: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExecutorConfig":
        filtered_payload = {
            key: value
            for key, value in payload.items()
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
            require_non_bearish_structure_for_long=self.require_non_bearish_structure_for_long_live,
        )

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
                ["/strategy", "/ob", "/drift"],
                ["/status", "/status table", "/performance"],
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
            {"command": "strategy", "description": "策略链路/最近仲裁"},
            {"command": "ob", "description": "候选雷达精简版"},
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
        first_token = raw.split(maxsplit=1)[0] if raw else ""
        command_name = first_token.split("@", 1)[0].strip().lower()
        tail = raw[len(first_token) :].strip().lower() if first_token else ""
        command = f"{command_name} {tail}".strip()
        if command_name in {"/drift", "/health", "/体检"}:
            return self._build_drift_report_message()
        if command in {"/ob full", "/ob_full", "/obfull"}:
            return self._build_ob_status_message()
        if command_name in {"/strategy", "/策略", "/链路"}:
            return self._build_strategy_status_message()
        if command_name in {"/ob", "/状态"}:
            return self._build_strategy_status_message()
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
                "🧭 /strategy 策略链路/最近仲裁",
                "🔎 /ob 候选雷达精简版",
                "🧱 /ob full 旧OB细节诊断",
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

    def _load_sota_soft_stop_state(self) -> dict[str, Any]:
        raw = self.store.get_value("sota_soft_stop_live_state")
        if not raw:
            return {"mode": "sota_soft_stop_live", "active": None, "history": []}
        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            return {"mode": "sota_soft_stop_live", "active": None, "history": []}
        if not isinstance(state, dict):
            return {"mode": "sota_soft_stop_live", "active": None, "history": []}
        if not isinstance(state.get("history"), list):
            state["history"] = []
        state.setdefault("mode", "sota_soft_stop_live")
        state.setdefault("active", None)
        return state

    def _save_sota_soft_stop_state(self, state: dict[str, Any]) -> None:
        history = state.get("history")
        if isinstance(history, list):
            state["history"] = history[-200:]
        self.store.set_value("sota_soft_stop_live_state", json.dumps(state, ensure_ascii=False))

    def _sota_soft_stop_excluded_buckets(self) -> set[str]:
        raw = self.config.sota_soft_stop_exclude_score_buckets
        if raw is None:
            return set()
        if isinstance(raw, str):
            return {item.strip() for item in raw.split(",") if item.strip()}
        try:
            return {str(item).strip() for item in raw if str(item).strip()}
        except TypeError:
            return set()

    def _score_bucket_name_from_action(self, action: StrategyAction) -> str | None:
        metadata = action.metadata or {}
        dynamic = metadata.get("dynamic_high_leverage") if isinstance(metadata.get("dynamic_high_leverage"), dict) else {}
        bucket = dynamic.get("score_bucket_sizing") if isinstance(dynamic.get("score_bucket_sizing"), dict) else {}
        if not bucket:
            bucket = metadata.get("score_bucket_sizing") if isinstance(metadata.get("score_bucket_sizing"), dict) else {}
        rule = bucket.get("rule") if isinstance(bucket.get("rule"), dict) else {}
        name = rule.get("name")
        return str(name) if name else None

    def _score_bucket_names_from_action(self, action: StrategyAction) -> set[str]:
        metadata = action.metadata or {}
        dynamic = metadata.get("dynamic_high_leverage") if isinstance(metadata.get("dynamic_high_leverage"), dict) else {}
        bucket = dynamic.get("score_bucket_sizing") if isinstance(dynamic.get("score_bucket_sizing"), dict) else {}
        if not bucket:
            bucket = metadata.get("score_bucket_sizing") if isinstance(metadata.get("score_bucket_sizing"), dict) else {}
        if not bool(bucket.get("applied")):
            return set()
        names: set[str] = set()
        applied_rules = bucket.get("applied_rules")
        if isinstance(applied_rules, list):
            for item in applied_rules:
                if not isinstance(item, dict):
                    continue
                if not bool(item.get("applied")):
                    continue
                item_rule = item.get("rule") if isinstance(item.get("rule"), dict) else {}
                if item_rule.get("name"):
                    names.add(str(item_rule["name"]))
        if not names:
            rule = bucket.get("rule") if isinstance(bucket.get("rule"), dict) else {}
            if rule.get("name"):
                names.add(str(rule["name"]))
        return names

    def _configured_fvg_bear6_exit_profile_overrides(self) -> dict[str, Any]:
        configured = self.config.fvg_bear6_loose_exit_profile_overrides
        if isinstance(configured, dict) and configured:
            return dict(configured)
        return dict(FVG_BEAR6_LOOSE_EXIT_PROFILE_OVERRIDES)

    def _fvg_bear6_loose_exit_profile_decision(self, action: StrategyAction) -> dict[str, Any]:
        bucket_name = str(
            self.config.fvg_bear6_loose_exit_profile_bucket
            or FVG_BEAR6_LOOSE_EXIT_PROFILE_BUCKET
        )
        profile_name = str(
            self.config.fvg_bear6_loose_exit_profile_name
            or FVG_BEAR6_LOOSE_EXIT_PROFILE_NAME
        )
        decision: dict[str, Any] = {
            "enabled": bool(self.config.enable_fvg_bear6_loose_exit_profile_live),
            "applied": False,
            "profile": profile_name,
            "bucket": bucket_name,
        }
        if not decision["enabled"]:
            decision["reason"] = "disabled"
            return decision
        if self._open_action_event_type(action) != "sota_long" or action.type != ActionType.OPEN_LONG:
            decision["reason"] = "not_sota_long"
            return decision
        bucket_names = self._score_bucket_names_from_action(action)
        decision["matched_buckets"] = sorted(bucket_names)
        if bucket_name not in bucket_names:
            decision["reason"] = "bucket_not_matched"
            return decision
        overrides = self._configured_fvg_bear6_exit_profile_overrides()
        decision.update(
            {
                "applied": True,
                "reason": f"score_bucket:{bucket_name}",
                "overrides": overrides,
            }
        )
        return decision

    def _apply_open_exit_profile_metadata(self, engine: Any, action: StrategyAction) -> dict[str, Any] | None:
        if action.type not in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}:
            return None
        decision = self._fvg_bear6_loose_exit_profile_decision(action)
        if not bool(decision.get("applied")):
            return decision if bool(decision.get("enabled")) else None
        metadata = dict(action.metadata or {})
        metadata["exit_profile"] = decision["profile"]
        metadata["exit_profile_reason"] = decision["reason"]
        metadata["exit_profile_overrides"] = dict(decision["overrides"])
        metadata["exit_profile_decision"] = decision
        action.metadata = metadata
        position = getattr(engine, "position", None)
        if position is not None:
            setattr(position, "exit_profile", decision["profile"])
            setattr(position, "exit_profile_reason", decision["reason"])
            setattr(position, "exit_profile_overrides", dict(decision["overrides"]))
        return decision

    def _sota_soft_stop_score_payload(self, action: StrategyAction) -> dict[str, Any] | None:
        score_gate = (action.metadata or {}).get("sota_score_gate")
        if not isinstance(score_gate, dict):
            return None
        score = score_gate.get("score")
        return score if isinstance(score, dict) else None

    def _sota_soft_stop_gate_decision(self, action: StrategyAction, sizing: dict[str, Any] | None = None) -> dict[str, Any]:
        reasons: list[str] = []
        if not bool(self.config.enable_sota_soft_stop_recovery_overlay_live):
            reasons.append("disabled")
        if str(self.config.sota_soft_stop_live_mode or "audit") != "audit":
            reasons.append("unsupported_live_mode")
        if action.type != ActionType.OPEN_LONG:
            reasons.append("not_open_long")
        if self._open_action_event_type(action) != "sota_long":
            reasons.append("not_sota_long")
        if action.direction != Direction.BULL:
            reasons.append("not_bull")

        score = self._sota_soft_stop_score_payload(action)
        if not isinstance(score, dict):
            reasons.append("missing_score")
        else:
            if int(score.get("net_score", 0) or 0) < int(self.config.sota_soft_stop_net_min):
                reasons.append("net_too_low")
            if int(score.get("bear_total", 0) or 0) > int(self.config.sota_soft_stop_bear_max):
                reasons.append("bear_too_high")

        effective_leverage = None
        metadata = action.metadata or {}
        dynamic = metadata.get("dynamic_high_leverage") if isinstance(metadata.get("dynamic_high_leverage"), dict) else {}
        if isinstance(dynamic, dict):
            effective_leverage = self._safe_float(dynamic.get("effective_leverage"))
        if effective_leverage is None and sizing is not None and isinstance(sizing.get("dynamic_high_leverage"), dict):
            effective_leverage = self._safe_float(sizing["dynamic_high_leverage"].get("effective_leverage"))
        if effective_leverage is None:
            effective_leverage = self._safe_float(metadata.get("candidate_leverage")) or float(self.config.leverage)
        if effective_leverage > float(self.config.sota_soft_stop_max_leverage):
            reasons.append("leverage_too_high")

        bucket_name = self._score_bucket_name_from_action(action)
        if bucket_name and bucket_name in self._sota_soft_stop_excluded_buckets():
            reasons.append("excluded_score_bucket")

        entry_price = self._safe_float(action.entry_price)
        stop_price = self._safe_float(action.stop_price)
        risk_price = abs(entry_price - stop_price) if entry_price and stop_price else 0.0
        if not entry_price or not stop_price or risk_price <= 0.0:
            reasons.append("invalid_entry_or_stop")
        soft_stop_price = (
            entry_price - (1.0 + float(self.config.sota_soft_stop_buffer_r)) * risk_price
            if entry_price and risk_price > 0.0
            else None
        )
        target_price = (
            entry_price + float(self.config.sota_soft_stop_target_rr) * risk_price
            if entry_price and risk_price > 0.0
            else None
        )
        return {
            "enabled": bool(self.config.enable_sota_soft_stop_recovery_overlay_live),
            "mode": str(self.config.sota_soft_stop_live_mode or "audit"),
            "eligible": len(reasons) == 0,
            "reasons": reasons,
            "score": score,
            "effective_leverage": effective_leverage,
            "score_bucket": bucket_name,
            "entry_price": entry_price,
            "initial_stop_price": stop_price,
            "risk_price": risk_price,
            "soft_stop_price": soft_stop_price,
            "target_price": target_price,
            "max_extension_bars": int(self.config.sota_soft_stop_max_extension_bars),
            "buffer_r": float(self.config.sota_soft_stop_buffer_r),
            "target_rr": float(self.config.sota_soft_stop_target_rr),
        }

    def _sota_soft_stop_prepare_open(self, action: StrategyAction, sizing: dict[str, Any]) -> dict[str, Any]:
        decision = self._sota_soft_stop_gate_decision(action, sizing)
        metadata = dict(action.metadata or {})
        metadata["sota_soft_stop_live"] = decision
        action.metadata = metadata
        return decision

    def _sota_soft_stop_after_open(
        self,
        action: StrategyAction,
        sizing: dict[str, Any],
        observed: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        decision = self._sota_soft_stop_gate_decision(action, sizing)
        metadata = dict(action.metadata or {})
        metadata["sota_soft_stop_live"] = decision
        action.metadata = metadata
        state = self._load_sota_soft_stop_state()
        event = {
            "time": action.timestamp,
            "event": "open_audit",
            "eligible": bool(decision.get("eligible")),
            "reasons": decision.get("reasons", []),
            "direction": action.direction,
            "entry_price": action.entry_price,
            "initial_stop_price": action.stop_price,
            "soft_stop_price": decision.get("soft_stop_price"),
            "target_price": decision.get("target_price"),
            "effective_leverage": decision.get("effective_leverage"),
            "score": decision.get("score"),
            "observed_contracts": observed.get("contracts") if isinstance(observed, dict) else None,
            "observed_notional_usdt": observed.get("notional_usdt") if isinstance(observed, dict) else None,
        }
        if bool(decision.get("eligible")):
            state["active"] = {
                **event,
                "status": "audit_only",
                "max_extension_bars": decision.get("max_extension_bars"),
                "buffer_r": decision.get("buffer_r"),
                "target_rr": decision.get("target_rr"),
            }
        else:
            state["active"] = None
        history = state.get("history") if isinstance(state.get("history"), list) else []
        history.append(event)
        state["history"] = history
        self._save_sota_soft_stop_state(state)
        self.store.append_action(action.timestamp, "SOTA_SOFT_STOP_AUDIT", event)
        return decision

    def _sota_soft_stop_after_close(self, action: StrategyAction) -> None:
        state = self._load_sota_soft_stop_state()
        active = state.get("active") if isinstance(state.get("active"), dict) else None
        if active is None:
            return
        event = {
            "time": action.timestamp,
            "event": "close_audit",
            "entry_time": active.get("time"),
            "close_reason": action.reason,
            "exit_price": action.exit_price,
            "soft_stop_price": active.get("soft_stop_price"),
            "metadata": action.metadata if isinstance(action.metadata, dict) else {},
        }
        history = state.get("history") if isinstance(state.get("history"), list) else []
        history.append(event)
        state["history"] = history
        state["active"] = None
        self._save_sota_soft_stop_state(state)
        self.store.append_action(action.timestamp, "SOTA_SOFT_STOP_AUDIT", event)

    def _sota_soft_stop_open_line(self, decision: dict[str, Any] | None) -> str | None:
        if not isinstance(decision, dict):
            return None
        if bool(decision.get("eligible")):
            return f"Soft-stop: AUDIT eligible / disaster {self._format_optional_price(decision.get('soft_stop_price'))}"
        reasons = decision.get("reasons")
        if isinstance(reasons, list) and reasons:
            return "Soft-stop: AUDIT skip " + ",".join(str(item) for item in reasons[:3])
        return "Soft-stop: AUDIT skip"

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
            elif reason.startswith("score_bucket:"):
                name = reason.split(":", 1)[1]
                parts.append(f"Score桶 {self._score_bucket_reason_label(name)}")
            elif reason.startswith("overlay_fixed:"):
                event_type = reason.split(":", 1)[1]
                parts.append(f"固定候选 {self._strategy_event_label(event_type)}")
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
            if not self._close_action_counts_as_realized(payload):
                continue
            timestamp = str(item.get("timestamp") or "")
            if daily and not timestamp.startswith(today):
                continue
            events.append({"timestamp": timestamp, "pnl": pnl, "reason": payload.get("reason")})
        return events

    def _close_action_counts_as_realized(self, payload: dict[str, Any]) -> bool:
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return False
        if bool(metadata.get("ignored_for_realized_pnl")):
            return False
        if self.config.mode != "live" or not self._shadow_gate_enabled():
            return True
        return str(metadata.get("source") or "") in {"external_flat_sync", "exchange_fill_sync", "live_order_execution"}

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
        open_count = sum(
            1
            for item in actions
            if item.get("action_type") in {ActionType.OPEN_LONG.value, ActionType.OPEN_SHORT.value}
        )
        close_count = sum(
            1
            for item in actions
            if item.get("action_type") == ActionType.CLOSE_POSITION.value
            and isinstance(item.get("payload"), dict)
            and self._close_action_counts_as_realized(item["payload"])
        )
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
            self._telegram_title("🤖", "Bot启动"),
            f"状态: {status_text}",
            f"标的: {self.config.symbol}",
            f"模式: {self.config.mode}",
            f"主链: {self._strategy_priority_text()}",
            f"SOTA gate: {self._score_gate_status_text()}",
            f"Bucket sizing: {self._long_score_bucket_status_text()}",
            f"SOTA soft-stop: {self._sota_soft_stop_status_text()}",
            f"SMC short: {self._smc_status_text('short')}",
            f"市场加载: {market_loaded}",
            f"快照加载: {snapshot_loaded}",
            self._telegram_time_line(),
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
            from scripts.fallback_trigger_report import build_report_from_paths, format_summary_lines

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
            message = format_report(report)
            try:
                fallback_report = build_report_from_paths(
                    config_path=(self.config_path or Path("runtime_config")).resolve(),
                    state_db=state_db,
                    baseline_path=baseline_path,
                    recent_trades=int(self.config.telegram_drift_recent_trades),
                )
                message = "\n".join([message, "", *format_summary_lines(fallback_report)])
            except Exception:
                pass
            return message
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

    def _strategy_event_label(self, event_type: Any) -> str:
        labels = {
            "sota_long": "SOTA Long",
            "smc_short": "SMC Short",
            "gap_smc_short_expansion": "Gap SMC Short",
            "smc_long": "SMC Long",
        }
        return labels.get(str(event_type or ""), str(event_type or "-"))

    def _score_bucket_reason_label(self, name: str) -> str:
        labels = {
            "bear_total_6_20x_boost": "bear=6 20x",
            "nbb_6_11_5_conflict_2p5_cap20": "6/11/5冲突 2.5x cap20",
            "bear_total_6_light_boost": "bear=6轻放大",
            "fvg_near_bear6_target20": "fvg近场 bear=6 20x",
            "fvg_near_bear5_target8": "fvg近场 bear=5 8x",
            "fvg_near_bear3_target5": "fvg近场 bear=3 5x",
            "fvg_hg_net8_target4": "fvg高增长 net=8 4x",
        }
        return labels.get(str(name or ""), str(name or "-"))

    def _strategy_priority_text(self) -> str:
        priority = " > ".join(self._strategy_event_label(item) for item in self._live_candidate_priority())
        if not priority:
            priority = "SOTA Long"
        suffix = "" if self._live_candidate_arbitration_enabled() else " (仲裁OFF)"
        return f"{priority}{suffix}"

    def _score_gate_status_text(self) -> str:
        if not bool(self.config.enable_sota_score_gate_live):
            return "OFF"
        rule = self._sota_score_gate_rule()
        return (
            "ON "
            f"net>={rule.get('net_min')} / bull>={rule.get('bull_min')} / "
            f"bear<={rule.get('bear_max')} / {rule.get('conflict_mode')}"
        )

    def _score_payload_text(self, score: Any) -> str:
        if not isinstance(score, dict):
            return "-"
        net = score.get("net_score")
        bull = score.get("bull_total")
        bear = score.get("bear_total")
        conflict = "conflict" if bool(score.get("conflict")) else "clean"
        return f"net {net} / bull {bull} / bear {bear} / {conflict}"

    def _score_bucket_rule_text(self, rule: dict[str, Any]) -> str:
        name = self._score_bucket_reason_label(str(rule.get("name") or "unnamed"))
        criteria: list[str] = []
        for prefix, label in (("net", "net"), ("bull", "bull"), ("bear", "bear")):
            if rule.get(f"{prefix}_eq") is not None:
                criteria.append(f"{label}={rule[f'{prefix}_eq']}")
            if rule.get(f"{prefix}_min") is not None:
                criteria.append(f"{label}>={rule[f'{prefix}_min']}")
            if rule.get(f"{prefix}_max") is not None:
                criteria.append(f"{label}<={rule[f'{prefix}_max']}")
        conflict_mode = str(rule.get("conflict_mode") or "any")
        if conflict_mode != "any":
            criteria.append(conflict_mode)
        required_true = rule.get("required_true_features")
        if isinstance(required_true, str):
            required_true = [required_true]
        if required_true:
            try:
                criteria.extend(str(item) for item in required_true if str(item))
            except TypeError:
                pass
        feature_equals = rule.get("feature_equals")
        if isinstance(feature_equals, dict):
            criteria.extend(f"{key}={value}" for key, value in feature_equals.items())
        regime_labels = rule.get("regime_labels")
        if isinstance(regime_labels, str):
            regime_labels = [regime_labels]
        if regime_labels:
            try:
                criteria.extend(f"regime={item}" for item in regime_labels if str(item))
            except TypeError:
                pass
        if bool(rule.get("continue")):
            criteria.append("continue")
        leverage = (
            f"target {float(rule['target_effective_leverage']):.2f}x"
            if rule.get("target_effective_leverage") is not None
            else f"x{float(rule.get('leverage_multiplier', rule.get('multiplier', 1.0)) or 1.0):.2f}"
        )
        if rule.get("max_effective_leverage") is not None:
            leverage += f" cap {float(rule['max_effective_leverage']):.1f}x"
        body = ", ".join(criteria + [leverage])
        return f"{name}({body})" if body else name

    def _long_score_bucket_status_text(self) -> str:
        if not bool(self.config.enable_long_score_bucket_sizing_live):
            return "OFF"
        rules = self.config.long_score_bucket_sizing_rules
        if not isinstance(rules, list) or not rules:
            return "ON default"
        rendered = [self._score_bucket_rule_text(rule) for rule in rules[:3] if isinstance(rule, dict)]
        extra = len(rules) - len(rendered)
        suffix = f" +{extra}" if extra > 0 else ""
        return "ON " + "; ".join(rendered) + suffix

    def _sota_soft_stop_status_text(self) -> str:
        if not bool(self.config.enable_sota_soft_stop_recovery_overlay_live):
            return "OFF"
        excluded = self.config.sota_soft_stop_exclude_score_buckets or []
        excluded_text = ",".join(str(item) for item in excluded) if excluded else "-"
        return (
            f"{str(self.config.sota_soft_stop_live_mode or 'audit').upper()} "
            f"net>={int(self.config.sota_soft_stop_net_min)} / "
            f"bear<={int(self.config.sota_soft_stop_bear_max)} / "
            f"lev<={float(self.config.sota_soft_stop_max_leverage):.1f}x / "
            f"buf {float(self.config.sota_soft_stop_buffer_r):.2f}R / "
            f"{int(self.config.sota_soft_stop_max_extension_bars)} bars / "
            f"exclude {excluded_text}"
        )

    def _smc_status_text(self, side: str) -> str:
        if side == "short":
            if not bool(self.config.enable_smc_short_live):
                return "OFF"
            return f"ON {self.config.smc_case} / {float(self.config.smc_leverage):.1f}x / RR {float(self.config.smc_target_rr):.2f}"
        if not bool(self.config.enable_smc_long_live):
            return "OFF"
        return (
            f"ON {self.config.smc_long_case} / "
            f"{float(self.config.smc_long_leverage):.1f}x / RR {float(self.config.smc_long_target_rr):.2f}"
        )

    def _trailing_status_text(self) -> str:
        flags = [
            f"stage {'ON' if self.config.enable_stage_trailing else 'OFF'}",
            f"ATR {'ON' if self.config.enable_atr_trailing else 'OFF'}",
            f"time {'ON' if self.config.enable_time_based_trailing else 'OFF'}",
            f"pressure {'ON' if self.config.enable_pressure_level_trailing else 'OFF'}",
        ]
        return " / ".join(flags)

    def _latest_action_payload(self, action_type: str, *, limit: int = 200) -> dict[str, Any] | None:
        for item in self.store.recent_actions(limit):
            if item.get("action_type") != action_type:
                continue
            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue
            return {
                **payload,
                "_timestamp": item.get("timestamp"),
                "_created_at": item.get("created_at"),
            }
        return None

    def _format_arbitration_candidate(self, candidate: Any) -> str:
        if not isinstance(candidate, dict):
            return "-"
        event_type = self._strategy_event_label(candidate.get("event_type"))
        direction = self._direction_label(candidate.get("direction"))
        entry = self._format_optional_price(candidate.get("entry_price"))
        timestamp = candidate.get("timestamp") or candidate.get("entry_time") or "-"
        return f"{event_type} {direction} @ {entry} / {timestamp}"

    def _arbitration_status_lines(self) -> list[str]:
        payload = self._latest_action_payload("LIVE_CANDIDATE_ARBITRATION")
        if not payload:
            return ["最近仲裁: 暂无"]
        decision = str(payload.get("decision") or "-")
        lines = [
            "最近仲裁",
            f"结果: {decision} / {payload.get('_timestamp') or '-'}",
        ]
        selected = payload.get("selected")
        if isinstance(selected, dict):
            lines.append(f"Selected: {self._format_arbitration_candidate(selected)}")
        rejected = payload.get("rejected") if isinstance(payload.get("rejected"), list) else []
        if rejected:
            counts: dict[str, int] = {}
            for item in rejected:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("reason") or item.get("event_type") or "unknown")
                counts[key] = counts.get(key, 0) + 1
            if counts:
                lines.append("Rejected: " + ", ".join(f"{key} x{value}" for key, value in sorted(counts.items())))
        score_gate_rejected = payload.get("score_gate_rejected")
        if isinstance(score_gate_rejected, list) and score_gate_rejected:
            lines.append(f"Score gate拒绝: {len(score_gate_rejected)}")
        structure_gate_rejected = payload.get("structure_gate_rejected")
        if isinstance(structure_gate_rejected, list) and structure_gate_rejected:
            lines.append(f"Structure gate拒绝: {len(structure_gate_rejected)}")
        return lines

    def _current_strategy_position_lines(self, snapshot: dict[str, Any], dyn: dict[str, Any]) -> list[str]:
        position = snapshot.get("position") if isinstance(snapshot.get("position"), dict) else None
        if not position:
            return ["当前策略仓位: flat"]
        entry_time = str(position.get("entry_time") or "")
        action_metadata = self._latest_open_action_metadata(entry_time)
        event_type = position.get("candidate_event_type") or action_metadata.get("candidate_event_type") or "sota_long"
        context = self._current_position_execution_context(snapshot, dyn)
        lines = [
            f"当前策略仓位: {self._direction_label(position.get('direction'))} / {self._strategy_event_label(event_type)}",
            (
                "入场/止损/止盈: "
                f"{self._format_optional_price(position.get('entry_price'))} / "
                f"{self._format_optional_price(position.get('sl_price'))} / "
                f"{self._format_optional_price(position.get('target_price'))}"
            ),
        ]
        if context:
            lines.append(
                "杠杆: "
                f"账户 {self._format_optional_leverage(context.get('actual_effective_leverage'))} / "
                f"执行 {self._format_optional_leverage(context.get('selected_effective_leverage'))} / "
                f"{context.get('risk_mode') or '-'}"
            )
            lines.append(
                "仓位: "
                f"{self._format_optional_usdt(context.get('theoretical_notional'), digits=0)} -> "
                f"{self._format_optional_usdt(context.get('actual_notional'), digits=0)}"
            )
            lines.append(f"压仓: {context.get('reason_text') or '-'}")
        if position.get("exit_profile"):
            lines.append(f"Exit profile: {position.get('exit_profile')}")
        return lines

    def _open_strategy_context_lines(self, action: StrategyAction, dynamic_info: dict[str, Any]) -> list[str]:
        metadata = action.metadata or {}
        event_type = self._open_action_event_type(action)
        lines = [f"候选: {self._strategy_event_label(event_type)}"]
        score_gate = metadata.get("sota_score_gate")
        if event_type == "sota_long" and isinstance(score_gate, dict):
            lines.append(f"Score: {self._score_payload_text(score_gate.get('score'))}")
            accepted = "通过" if bool(score_gate.get("accepted", True)) else "拒绝"
            lines.append(f"Gate: {accepted}")
        if event_type in {"smc_short", "gap_smc_short_expansion", "smc_long"}:
            source = metadata.get("candidate_source") if isinstance(metadata.get("candidate_source"), dict) else {}
            if event_type == "smc_long":
                case = source.get("smc_case") or self.config.smc_long_case
            elif event_type == "gap_smc_short_expansion":
                case = source.get("smc_case") or self.config.gap_smc_short_case
            else:
                case = source.get("smc_case") or self.config.smc_case
            time_bucket = source.get("time_bucket")
            lag = source.get("mss_lag_bars")
            parts = [str(case)]
            if time_bucket is not None:
                parts.append(str(time_bucket))
            if lag is not None:
                parts.append(f"lag {lag}")
            if event_type == "gap_smc_short_expansion" and source.get("flat_days") is not None:
                parts.append(f"flat {float(source.get('flat_days') or 0.0):.1f}d")
            lines.append("SMC: " + " / ".join(parts))
        bucket = dynamic_info.get("score_bucket_sizing") if isinstance(dynamic_info.get("score_bucket_sizing"), dict) else None
        if isinstance(bucket, dict) and bool(bucket.get("applied")):
            rule = bucket.get("rule") if isinstance(bucket.get("rule"), dict) else {}
            name = self._score_bucket_reason_label(str(rule.get("name") or "unnamed"))
            lines.append(f"Bucket: {name} -> {self._format_optional_leverage(bucket.get('target_effective_leverage'))}")
        return lines

    def _build_strategy_status_message(self) -> str:
        try:
            snapshot = self._load_snapshot_payload()
            dyn = self._load_dynamic_high_leverage_state() if self._dynamic_high_leverage_enabled() else {}
            position = self._position_summary()
            local_position = position["local_position"]
            long_contracts = float(position["long"].get("contracts", 0.0) or 0.0)
            short_contracts = float(position["short"].get("contracts", 0.0) or 0.0)
            exchange_side = "long" if long_contracts > 0 else "short" if short_contracts > 0 else "flat"
            local_side = "flat"
            if local_position:
                local_side = "long" if local_position.get("direction") == "BULL" else "short"
            bracket = position.get("pending_bracket") if isinstance(position.get("pending_bracket"), dict) else {}
            lines = [
                self._telegram_title("🧭", "策略控制台"),
                f"标的: {self.config.symbol}",
                f"模式: {self.config.mode}",
                f"开仓: {self._open_status_text(self._telegram_open_paused())}",
                f"主链: {self._strategy_priority_text()}",
                f"单仓仲裁: {'ON' if self._live_candidate_arbitration_enabled() else 'OFF'} / max_pos={self.config.max_open_positions}",
                f"SOTA gate: {self._score_gate_status_text()}",
                f"Long bucket: {self._long_score_bucket_status_text()}",
                f"SOTA soft-stop: {self._sota_soft_stop_status_text()}",
                f"SMC short: {self._smc_status_text('short')}",
            ]
            if bool(self.config.enable_smc_long_live):
                lines.append(f"SMC long: {self._smc_status_text('long')}")
            lines.extend(
                [
                    f"Trailing: {self._trailing_status_text()}",
                    "",
                    "仓位",
                    f"交易所: {self._side_status_text(exchange_side)}",
                    f"本地: {self._side_status_text(local_side)}",
                    (
                        "保护: "
                        f"SL {self._format_optional_price(bracket.get('stop_price'))} / "
                        f"TP {self._format_optional_price(bracket.get('target_price'))}"
                    ),
                ]
            )
            lines.extend(self._current_strategy_position_lines(snapshot, dyn))
            lines.append("")
            lines.extend(self._arbitration_status_lines())
            lines.append("")
            lines.extend(self._latest_shadow_status_lines())
            lines.append(self._telegram_time_line())
            return "\n".join(lines)
        except Exception as exc:
            return "\n".join(
                [
                    self._telegram_title("🧭", "策略控制台"),
                    "状态: 生成失败",
                    f"原因: {exc}",
                    self._telegram_time_line(),
                ]
            )

    def _build_ob_status_message(self) -> str:
        try:
            engine, _ = self.load_engine()
            latest_idx = self._latest_closed_index(engine)
            if latest_idx is None:
                return "🧭 <OB 雷达>\n状态: 🕒 等待最新收盘 K 线"

            engine._apply_regime_switch_for_idx(latest_idx)
            latest = engine.c15m[latest_idx]
            bias = engine.precomputed.bias_4h[engine.mapping[latest_idx]]
            regime = engine._regime_label_for_idx(latest_idx)
            regime_features = engine._regime_features_for_idx(latest_idx)
            timestamp = engine._timestamp_for_idx(latest_idx)
            structure_reference = self._structure_reference(engine, latest_idx, bias)
            lines = [
                "🧭 <OB 开仓雷达>",
                f"标的: {self.config.symbol}",
                f"时间: {timestamp} UTC",
                f"价格: {self._format_price(float(latest.c))}",
                f"4H Bias: {self._direction_label(bias)}",
            ]
            lines.extend(self._regime_display_lines(regime, regime_features))
            lines.extend(self._structure_reference_lines(structure_reference))

            if engine.position is not None:
                state = self._load_shadow_gate_state(engine) if self._shadow_gate_enabled() else {}
                real_open = bool(state.get("real_position_open", True))
                position = engine.position
                label = "📌 持仓中" if real_open else "🧪 Shadow paper position"
                lines.extend(
                    [
                        "",
                        f"状态: {label}",
                        f"方向: {self._direction_label(getattr(position, 'direction', None))}",
                        f"入场: {self._format_price(getattr(position, 'entry_price', None))}",
                        f"止损: {self._format_price(getattr(position, 'sl_price', None))}",
                        f"止盈: {self._format_price(getattr(position, 'target_price', None))}",
                        "开仓条件: 等当前策略仓位结束后再寻找下一组 OB",
                    ]
                )
                lines.extend(self._latest_shadow_status_lines())
                return "\n".join(lines)

            candidates = self._active_ob_candidates(engine, latest_idx)
            lines.append("")
            lines.extend(self._latest_shadow_status_lines())
            if not candidates:
                missing = []
                if bias == Direction.NONE:
                    missing.append("形成方向性 4H bias")
                else:
                    primary = structure_reference.get("primary") if isinstance(structure_reference.get("primary"), dict) else {}
                    break_price = primary.get("break_price")
                    reclaim_price = primary.get("reclaim_price")
                    if bias == Direction.BEAR and break_price is not None and reclaim_price is not None:
                        missing.append(
                            f"价格路径需出现: 跌破 {self._format_price(break_price)} -> 收回 {self._format_price(reclaim_price)} 上方"
                        )
                    elif bias == Direction.BULL and break_price is not None and reclaim_price is not None:
                        missing.append(
                            f"价格路径需出现: 突破 {self._format_price(break_price)} -> 回踩守住 {self._format_price(reclaim_price)}"
                        )
                    else:
                        missing.append("先形成新的关键价破位和收回确认")
                missing.extend(
                    [
                        "找到合格 OB 实体区",
                        "价格回踩 OB 区间",
                        "收出确认 K：多头收阳 / 空头收阴",
                    ]
                )
                lines.extend(
                    [
                        "",
                        "状态: 🕵️ 暂无有效 OB 等待区",
                        "还差:",
                    ]
                )
                lines.extend(f"{pos}. {item}" for pos, item in enumerate(missing, start=1))
                return "\n".join(lines)

            for idx, (pending, detail) in enumerate(candidates, start=1):
                lines.extend(
                    [
                        "",
                        f"候选 {idx}: {self._direction_label(pending.direction)}",
                        f"OB 区间: {self._format_price(detail['bottom'])} - {self._format_price(detail['top'])}",
                        f"剩余窗口: {detail['expires_in_bars']} 根 15m K",
                        f"当前高低: {self._format_price(detail['current_low'])} / {self._format_price(detail['current_high'])}",
                    ]
                )
                if detail["ready"]:
                    lines.append("状态: 🚀 OB 条件已满足，等待下一次策略评估/风控确认")
                else:
                    lines.append("还差:")
                    lines.extend(f"{pos}. {item}" for pos, item in enumerate(detail["missing"], start=1))
            return "\n".join(lines)
        except Exception as exc:
            return "\n".join(
                [
                    "🧭 <OB 开仓雷达>",
                    "状态: ⚠️ 生成失败",
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
                if self._send_telegram(self._build_strategy_status_message()):
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
        if bool(self.config.enable_sota_score_gate_live):
            engine = ScalpRobustEngine(
                informative_candles,
                primary_candles,
                align_timeframes(informative_candles, primary_candles),
                build_precomputed_state_confirmed_4h(informative_candles, primary_candles),
                self.config.to_scalp_strategy_config(),
            )
        else:
            engine = ScalpRobustEngine.from_candles(
                informative_candles,
                primary_candles,
                self.config.to_scalp_strategy_config(),
            )
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
        self._assert_live_state_synced(
            engine,
            context="before_evaluate",
            timestamp=engine._timestamp_for_idx(latest_closed_idx),
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

        # evaluate_range uses a right-open end index. Include latest_closed_idx;
        # otherwise live can mark a candle processed without evaluating it.
        actions = engine.evaluate_range(start_idx, latest_closed_idx + 1)
        arbitration = None
        if self._live_candidate_arbitration_enabled():
            actions, arbitration = self._apply_live_candidate_arbitration(engine, actions, latest_closed_idx)
        execution_results = []
        for action in actions:
            result = self.execute_action(action, engine)
            execution_results.append({"action": asdict(action), "result": result})
        self._assert_live_state_synced(
            engine,
            context="after_execute",
            timestamp=engine._timestamp_for_idx(latest_closed_idx),
            exit_idx=latest_closed_idx,
        )

        last_timestamp = engine._timestamp_for_idx(latest_closed_idx)
        snapshot = engine.snapshot()
        self.store.set_value("last_processed_candle_time", last_timestamp)
        self.store.save_snapshot(snapshot)

        status = {
            "status": "ok",
            "symbol": self.config.symbol,
            "processed_candle_time": last_timestamp,
            "actions": [asdict(action) for action in actions],
            "execution_results": execution_results,
            "live_candidate_arbitration": arbitration,
            "trade_count": snapshot.trade_count,
            "position_open": engine.position is not None,
            "snapshot": asdict(snapshot),
            "live_capital": engine.capital,
        }
        self.store.append_action(last_timestamp, "EVALUATE", status)
        return status

    def _live_candidate_arbitration_enabled(self) -> bool:
        return bool(self.config.enable_live_candidate_arbitration)

    def _live_candidate_priority(self) -> list[str]:
        configured = self.config.live_candidate_priority
        if configured is None:
            return ["sota_long", "smc_short", "gap_smc_short_expansion"]
        if isinstance(configured, str):
            return [item.strip() for item in configured.split(",") if item.strip()]
        return [str(item) for item in configured if str(item)]

    def _live_candidate_priority_value(self, event_type: str) -> int:
        priority = {name: idx for idx, name in enumerate(self._live_candidate_priority())}
        return priority.get(str(event_type), 99)

    def _candidate_seen_state(self) -> dict[str, Any]:
        raw = self.store.get_value("live_candidate_seen")
        if not raw:
            return {"keys": []}
        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            return {"keys": []}
        keys = state.get("keys") if isinstance(state, dict) else []
        return {"keys": [str(item) for item in keys[-500:]] if isinstance(keys, list) else []}

    def _candidate_seen(self, key: str) -> bool:
        return key in set(self._candidate_seen_state()["keys"])

    def _mark_candidate_seen(self, key: str) -> None:
        state = self._candidate_seen_state()
        keys = [item for item in state["keys"] if item != key]
        keys.append(key)
        self.store.set_value("live_candidate_seen", json.dumps({"keys": keys[-500:]}, ensure_ascii=False))

    def _open_action_event_type(self, action: StrategyAction) -> str:
        metadata = action.metadata or {}
        return str(metadata.get("candidate_event_type") or "sota_long")

    def _is_overlay_open_action(self, action: StrategyAction) -> bool:
        return self._open_action_event_type(action) in {"smc_short", "gap_smc_short_expansion", "smc_long"}

    def _is_stop_loss_reason(self, reason: str | None) -> bool:
        return str(reason or "") in {"stop_loss", "external_stop_loss"}

    def _action_candidate_summary(self, action: StrategyAction, event_type: str | None = None) -> dict[str, Any]:
        metadata = action.metadata or {}
        return {
            "event_type": event_type or self._open_action_event_type(action),
            "timestamp": action.timestamp,
            "direction": action.direction,
            "entry_price": action.entry_price,
            "stop_price": action.stop_price,
            "target_price": action.target_price,
            "source": metadata.get("source"),
            "source_key": metadata.get("source_key"),
            "entry_idx": metadata.get("index"),
            "sota_liquidity_context": metadata.get("sota_liquidity_context"),
            "feature_recent_fvg_near_entry": metadata.get("feature_recent_fvg_near_entry"),
            "feature_recent_sweep_status": metadata.get("feature_recent_sweep_status"),
        }

    def _clear_local_position_for_rejected_open_actions(self, engine: Any, rejected: list[dict[str, Any]]) -> bool:
        position = getattr(engine, "position", None)
        if position is None:
            return False
        rejected_times: set[str] = set()
        for item in rejected:
            candidate = item.get("candidate") if isinstance(item, dict) else None
            if isinstance(candidate, dict) and candidate.get("timestamp"):
                rejected_times.add(str(candidate["timestamp"]))
        if getattr(position, "entry_time", None) in rejected_times:
            engine.position = None
            return True
        return False

    def _sota_score_gate_rule(self) -> dict[str, Any]:
        return {
            "net_min": int(self.config.sota_score_net_min),
            "bull_min": int(self.config.sota_score_bull_min),
            "bear_max": int(self.config.sota_score_bear_max),
            "conflict_mode": str(self.config.sota_score_conflict_mode or "any"),
        }

    def _sota_score_snapshot_for_idx(self, engine: Any, entry_idx: int) -> dict[str, Any]:
        from scripts.confirmed_multiframe_score_utils import (
            align_confirmed_mapping,
            resample_confirmed_1h,
            score_snapshot,
        )

        c1h = resample_confirmed_1h(engine.c15m)
        mapping_1h = align_confirmed_mapping(c1h, engine.c15m)
        snapshot = score_snapshot(engine, c1h, mapping_1h, int(entry_idx))
        return asdict(snapshot)

    def _sota_score_gate_decision(self, engine: Any, action: StrategyAction, entry_idx: int) -> dict[str, Any]:
        rule = self._sota_score_gate_rule()
        snapshot = self._sota_score_snapshot_for_idx(engine, entry_idx)
        event = {**snapshot, "event_type": "sota_long"}
        from scripts.confirmed_multiframe_score_utils import passes_score_gate

        accepted = passes_score_gate(event, **rule)
        return {
            "enabled": True,
            "accepted": bool(accepted),
            "rule": rule,
            "score": snapshot,
            "candidate": self._action_candidate_summary(action, "sota_long"),
        }

    def _sota_liquidity_context_for_idx(self, engine: Any, entry_idx: int, direction: str) -> dict[str, Any]:
        lookback = 360
        start_idx = max(0, int(entry_idx) - lookback)
        candles = engine.c15m[start_idx : int(entry_idx) + 1]
        return liquidity_context_for_entry(
            candles,
            len(candles) - 1,
            str(direction),
        )

    def _apply_sota_score_gate_to_open_actions(
        self,
        engine: Any,
        open_actions: list[StrategyAction],
        latest_closed_idx: int,
    ) -> tuple[list[StrategyAction], list[dict[str, Any]]]:
        if not bool(self.config.enable_sota_score_gate_live):
            return open_actions, []
        accepted: list[StrategyAction] = []
        rejected: list[dict[str, Any]] = []
        for action in open_actions:
            event_type = self._open_action_event_type(action)
            if event_type != "sota_long":
                accepted.append(action)
                continue
            metadata = dict(action.metadata or {})
            entry_idx = int(metadata.get("index", latest_closed_idx) or latest_closed_idx)
            liquidity_context = self._sota_liquidity_context_for_idx(engine, entry_idx, action.direction)
            metadata["sota_liquidity_context"] = liquidity_context
            metadata.update(flatten_context_features(liquidity_context))
            decision = self._sota_score_gate_decision(engine, action, entry_idx)
            metadata["sota_score_gate"] = {
                "accepted": bool(decision["accepted"]),
                "rule": decision["rule"],
                "score": decision["score"],
            }
            action.metadata = metadata
            if bool(decision["accepted"]):
                accepted.append(action)
            else:
                decision["decision"] = "rejected"
                decision["reason"] = "sota_score_gate"
                rejected.append(decision)
        return accepted, rejected

    def _apply_sota_structure_gate_to_open_actions(
        self,
        open_actions: list[StrategyAction],
    ) -> tuple[list[StrategyAction], list[dict[str, Any]], list[dict[str, Any]]]:
        if not bool(self.config.require_non_bearish_structure_for_long_live):
            return open_actions, [], []
        accepted: list[StrategyAction] = []
        rejected: list[dict[str, Any]] = []
        recalled: list[dict[str, Any]] = []
        for action in open_actions:
            event_type = self._open_action_event_type(action)
            if event_type != "sota_long" or action.type != ActionType.OPEN_LONG:
                accepted.append(action)
                continue
            metadata = dict(action.metadata or {})
            payload = {
                "feature_bearish_structure": bool(metadata.get("feature_bearish_structure", False)),
            }
            _filtered, decision = apply_sota_structure_gate([payload], enabled=True)
            metadata["sota_structure_gate"] = {
                "accepted": decision["filtered_candidates"] > 0,
                "rule": decision["rule"],
                "feature_bearish_structure": payload["feature_bearish_structure"],
            }
            action.metadata = metadata
            if decision["filtered_candidates"] > 0:
                accepted.append(action)
            else:
                recall_decision = self._sota_rejected_smc_recall_long_decision(
                    action,
                    metadata,
                    reject_stage="structure_gate",
                )
                if bool(recall_decision.get("accepted")):
                    metadata["sota_rejected_smc_recall_long"] = recall_decision
                    metadata.setdefault("candidate_event_type", "sota_long")
                    action.metadata = metadata
                    accepted.append(action)
                    recalled.append(recall_decision)
                    continue
                rejected.append(
                    {
                        "enabled": True,
                        "accepted": False,
                        "rule": decision["rule"],
                        "reason": "sota_structure_gate",
                        "feature_bearish_structure": payload["feature_bearish_structure"],
                        "candidate": self._action_candidate_summary(action, "sota_long"),
                        "decision": "rejected",
                    }
                )
        return accepted, rejected, recalled

    def _sota_rejected_smc_recall_long_decision(
        self,
        action: StrategyAction,
        metadata: dict[str, Any],
        reject_stage: str,
    ) -> dict[str, Any]:
        enabled = bool(self.config.enable_sota_rejected_smc_recall_long_live)
        condition = str(self.config.sota_rejected_smc_recall_long_condition or "")
        expected_stage = str(self.config.sota_rejected_smc_recall_long_reject_stage or "")
        expected_regime = str(self.config.sota_rejected_smc_recall_long_regime_label or "")
        target_leverage = float(self.config.sota_rejected_smc_recall_long_target_leverage or 0.0)
        recent_sweep_has_fvg = bool(metadata.get("feature_recent_sweep_has_fvg", False))
        recent_fvg_near_entry = bool(metadata.get("feature_recent_fvg_near_entry", False))
        recent_sweep_status = str(metadata.get("feature_recent_sweep_status") or "")
        regime_label = str(metadata.get("regime_label") or "")
        condition_matched = False
        if condition == "sweep_has_fvg":
            condition_matched = recent_sweep_has_fvg
        elif condition == "recent_fvg_near_entry":
            condition_matched = recent_fvg_near_entry
        elif condition == "mss_with_fvg":
            condition_matched = recent_sweep_status == "mss_with_fvg"
        elif condition == "fvg_near_or_mss_with_fvg":
            condition_matched = recent_fvg_near_entry or recent_sweep_status == "mss_with_fvg"
        else:
            condition_matched = False
        reasons: list[str] = []
        if not enabled:
            reasons.append("disabled")
        if expected_stage and reject_stage != expected_stage:
            reasons.append("reject_stage_mismatch")
        if expected_regime and regime_label != expected_regime:
            reasons.append("regime_mismatch")
        if not condition_matched:
            reasons.append("condition_mismatch")
        if target_leverage <= 0:
            reasons.append("invalid_target_leverage")
        accepted = enabled and not reasons
        return {
            "enabled": enabled,
            "accepted": accepted,
            "reason": "accepted" if accepted else ",".join(reasons),
            "condition": condition,
            "reject_stage": reject_stage,
            "expected_reject_stage": expected_stage,
            "regime_label": regime_label,
            "expected_regime_label": expected_regime,
            "target_effective_leverage": target_leverage,
            "features": {
                "feature_recent_sweep_has_fvg": recent_sweep_has_fvg,
                "feature_recent_fvg_near_entry": recent_fvg_near_entry,
                "feature_recent_sweep_status": recent_sweep_status,
                "feature_bearish_structure": bool(metadata.get("feature_bearish_structure", False)),
            },
            "candidate": self._action_candidate_summary(action, "sota_long"),
        }

    def _apply_long_score_bucket_sizing(
        self,
        action: StrategyAction,
        effective_leverage: float,
        risk_mode: str,
    ) -> tuple[float, dict[str, Any] | None]:
        if not bool(self.config.enable_long_score_bucket_sizing_live):
            return effective_leverage, None
        if self._open_action_event_type(action) != "sota_long" or action.type != ActionType.OPEN_LONG:
            return effective_leverage, None

        metadata = action.metadata or {}
        recall_decision = metadata.get("sota_rejected_smc_recall_long")
        if isinstance(recall_decision, dict) and bool(recall_decision.get("accepted")):
            target_leverage = float(
                recall_decision.get(
                    "target_effective_leverage",
                    self.config.sota_rejected_smc_recall_long_target_leverage,
                )
                or 0.0
            )
            if target_leverage > 0:
                return target_leverage, {
                    "enabled": True,
                    "applied": abs(float(effective_leverage) - target_leverage) > 1e-9,
                    "reason": "sota_rejected_smc_recall_long",
                    "source_effective_leverage": float(effective_leverage),
                    "target_effective_leverage": target_leverage,
                    "recall": recall_decision,
                }
        score = metadata.get("sota_score_gate")
        score_payload = score.get("score") if isinstance(score, dict) else None
        if not isinstance(score_payload, dict):
            return effective_leverage, {
                "enabled": True,
                "applied": False,
                "reason": "missing_score",
            }

        from scripts.score_bucket_sizing_utils import apply_score_bucket_leverage

        adjusted, decision = apply_score_bucket_leverage(
            effective_leverage=float(effective_leverage),
            score={
                **score_payload,
                "risk_mode": risk_mode,
                "regime_label": metadata.get("regime_label"),
                **{
                    key: value
                    for key, value in metadata.items()
                    if str(key).startswith("feature_")
                },
            },
            enabled=True,
            rules=self.config.long_score_bucket_sizing_rules,
        )
        return adjusted, decision

    def _smc_case_params(self) -> dict[str, Any]:
        try:
            from scripts.smc_live_utils import SMC_CASES

            if str(self.config.smc_case) in SMC_CASES:
                return dict(SMC_CASES[str(self.config.smc_case)])
        except Exception:
            pass
        return {
            "target_rr": float(self.config.smc_target_rr),
            "allowed_time_buckets": "other+asia_evening_ny+ny_am_killzone",
            "swing_n": 2,
            "min_body_atr": 0.7,
            "min_range_atr": 1.1,
            "entry_lookahead_bars": 40,
            "max_open_positions": 1,
            "max_mss_lag_bars": 15,
            "min_displacement_body_atr": 0.5,
            "other_min_mss_lag_bars": 4,
        }

    def _gap_smc_short_case_params(self) -> dict[str, Any]:
        try:
            from scripts.smc_live_utils import SMC_CASES

            if str(self.config.gap_smc_short_case) in SMC_CASES:
                return dict(SMC_CASES[str(self.config.gap_smc_short_case)])
        except Exception:
            pass
        return {
            "target_rr": float(self.config.gap_smc_short_target_rr),
            "allowed_time_buckets": "other",
            "swing_n": 2,
            "min_body_atr": 0.5,
            "min_range_atr": 0.9,
            "entry_lookahead_bars": 40,
            "max_open_positions": 1,
            "max_mss_lag_bars": 15,
            "min_displacement_body_atr": 0.5,
            "require_confirmed_retest": False,
            "require_fvg_touch": True,
            "allow_ote_only": False,
            "require_htf_bias_align": False,
            "require_h4_bias_align": False,
            "require_d1_bias_align": False,
            "allowed_directions": "BEAR",
        }

    def _smc_long_case_params(self) -> dict[str, Any]:
        try:
            from scripts.smc_live_utils import SMC_LONG_CASES

            if str(self.config.smc_long_case) in SMC_LONG_CASES:
                return dict(SMC_LONG_CASES[str(self.config.smc_long_case)])
        except Exception:
            pass
        return {
            "target_rr": float(self.config.smc_long_target_rr),
            "allowed_time_buckets": "other+asia_evening_ny+ny_am_killzone",
            "swing_n": 2,
            "min_body_atr": 0.7,
            "min_range_atr": 1.1,
            "entry_lookahead_bars": 40,
            "max_open_positions": 1,
            "max_mss_lag_bars": 15,
            "min_displacement_body_atr": 0.9,
            "min_displacement_range_atr": 1.3,
            "require_fvg_touch": True,
            "allow_ote_only": False,
            "require_ote_touch": False,
        }

    def _smc_strategy_args(self) -> argparse.Namespace:
        defaults = {
            "data_15m": "",
            "data_4h": "",
            "start_date": "1970-01-01",
            "target_rr": float(self.config.smc_target_rr),
            "allowed_time_buckets": "other",
            "swing_n": 3,
            "min_body_atr": 0.7,
            "min_range_atr": 1.1,
            "entry_lookahead_bars": 40,
            "max_open_positions": 1,
            "min_displacement_body_atr": 0.0,
            "min_displacement_range_atr": 0.0,
            "max_mss_lag_bars": 15,
        }
        merged = defaults | self._smc_case_params()
        merged["target_rr"] = float(self.config.smc_target_rr)
        try:
            from scripts.reproduce_smc_short_only_v1_10x import strategy_args as smc_short_v1_strategy_args

            return smc_short_v1_strategy_args(argparse.Namespace(**merged))
        except Exception:
            merged.update(
                {
                    "swing_lookback": 80,
                    "liquidity_lookback_bars": 192,
                    "mss_lookahead_bars": 24,
                    "fvg_lookback_bars": 8,
                    "outcome_lookahead_bars": 0,
                    "atr_period": 14,
                    "stop_buffer_atr": 0.05,
                    "require_confirmed_retest": True,
                    "require_fvg_touch": False,
                    "allow_ote_only": True,
                    "require_htf_bias_align": True,
                    "require_h4_bias_align": True,
                    "require_d1_bias_align": False,
                    "allowed_directions": "BEAR",
                    "require_ote_touch": False,
                    "bull_min_displacement_body_atr": 0.0,
                    "bull_max_displacement_body_atr": 0.0,
                    "bull_min_displacement_range_atr": 0.0,
                    "bull_max_displacement_range_atr": 0.0,
                    "min_fvg_size_pct": 0.0,
                    "max_fvg_fill_pct": 0.0,
                    "bear_min_sweep_distance_pct": 0.0,
                    "bear_require_fvg_touch": False,
                    "bear_min_fvg_size_pct": 0.0,
                }
            )
            return argparse.Namespace(**merged)

    def _gap_smc_short_strategy_args(self) -> argparse.Namespace:
        defaults = {
            "data_15m": "",
            "data_4h": "",
            "start_date": "1970-01-01",
            "target_rr": float(self.config.gap_smc_short_target_rr),
            "allowed_time_buckets": "other",
            "swing_n": 2,
            "min_body_atr": 0.5,
            "min_range_atr": 0.9,
            "entry_lookahead_bars": 40,
            "max_open_positions": 1,
            "min_displacement_body_atr": 0.5,
            "min_displacement_range_atr": 0.0,
            "max_mss_lag_bars": 15,
        }
        merged = defaults | self._gap_smc_short_case_params()
        merged["target_rr"] = float(self.config.gap_smc_short_target_rr)
        try:
            from scripts.reproduce_smc_short_only_v1_10x import strategy_args as smc_short_v1_strategy_args

            return smc_short_v1_strategy_args(argparse.Namespace(**merged))
        except Exception:
            merged.update(
                {
                    "swing_lookback": 80,
                    "liquidity_lookback_bars": 192,
                    "mss_lookahead_bars": 24,
                    "fvg_lookback_bars": 8,
                    "outcome_lookahead_bars": 0,
                    "atr_period": 14,
                    "stop_buffer_atr": 0.05,
                    "require_confirmed_retest": False,
                    "require_fvg_touch": True,
                    "allow_ote_only": False,
                    "require_htf_bias_align": False,
                    "require_h4_bias_align": False,
                    "require_d1_bias_align": False,
                    "allowed_directions": "BEAR",
                    "require_ote_touch": False,
                    "bull_min_displacement_body_atr": 0.0,
                    "bull_max_displacement_body_atr": 0.0,
                    "bull_min_displacement_range_atr": 0.0,
                    "bull_max_displacement_range_atr": 0.0,
                    "min_fvg_size_pct": 0.0,
                    "max_fvg_fill_pct": 0.0,
                    "bear_min_sweep_distance_pct": 0.0,
                    "bear_require_fvg_touch": False,
                    "bear_min_fvg_size_pct": 0.0,
                }
            )
            return argparse.Namespace(**merged)

    def _smc_long_strategy_args(self) -> argparse.Namespace:
        defaults = {
            "data_15m": "",
            "data_4h": "",
            "start_date": "1970-01-01",
            "target_rr": float(self.config.smc_long_target_rr),
            "allowed_time_buckets": "other+asia_evening_ny+ny_am_killzone",
            "swing_n": 2,
            "min_body_atr": 0.7,
            "min_range_atr": 1.1,
            "entry_lookahead_bars": 40,
            "max_open_positions": 1,
            "min_displacement_body_atr": 0.9,
            "min_displacement_range_atr": 1.3,
            "max_mss_lag_bars": 15,
        }
        merged = defaults | self._smc_long_case_params()
        merged["target_rr"] = float(self.config.smc_long_target_rr)
        merged.update(
            {
                "swing_lookback": 80,
                "liquidity_lookback_bars": 192,
                "mss_lookahead_bars": 24,
                "fvg_lookback_bars": 8,
                "outcome_lookahead_bars": 0,
                "atr_period": 14,
                "stop_buffer_atr": 0.05,
                "require_confirmed_retest": True,
                "require_fvg_touch": bool(merged.get("require_fvg_touch", True)),
                "allow_ote_only": bool(merged.get("allow_ote_only", False)),
                "require_htf_bias_align": True,
                "require_h4_bias_align": True,
                "require_d1_bias_align": False,
                "allowed_directions": "BULL",
                "require_ote_touch": bool(merged.get("require_ote_touch", False)),
                "bull_min_displacement_body_atr": 0.0,
                "bull_max_displacement_body_atr": 0.0,
                "bull_min_displacement_range_atr": 0.0,
                "bull_max_displacement_range_atr": 0.0,
                "min_fvg_size_pct": 0.0,
                "max_fvg_fill_pct": 0.0,
                "bear_min_sweep_distance_pct": 0.0,
                "bear_require_fvg_touch": False,
                "bear_min_fvg_size_pct": 0.0,
            }
        )
        return argparse.Namespace(**merged)

    def _smc_event_allowed(
        self,
        engine: Any,
        event: Any,
        latest_closed_idx: int,
        smc_args: argparse.Namespace,
        direction: str,
        case_params: dict[str, Any],
    ) -> bool:
        retest = getattr(event, "retest", None)
        if retest is None or int(getattr(retest, "idx", -1)) != int(latest_closed_idx):
            return False
        if str(getattr(event, "direction", "")) != direction:
            return False
        if bool(getattr(smc_args, "require_confirmed_retest", True)) and not bool(getattr(retest, "confirmed", False)):
            return False
        if bool(getattr(smc_args, "require_fvg_touch", False)) and not bool(getattr(retest, "fvg_touched", False)):
            return False
        if not bool(getattr(smc_args, "allow_ote_only", True)) and not bool(getattr(retest, "fvg_touched", False)):
            return False
        try:
            from scripts.research_smc_standalone_v1 import allowed_bucket, allowed_direction, htf_structure_bias
            from scripts.report_pa_ict_liquidity_features import time_bucket
            from scripts.report_smc_trade_context import completed_4h_idx_for_entry, completed_d1_idx_for_entry, daily_candles_from_4h
            from strategy.scalp_robust_v2_core import precompute_swings
        except Exception:
            return False

        bucket, _ = time_bucket(engine.c15m[latest_closed_idx].ts)
        if not allowed_bucket(bucket, str(getattr(smc_args, "allowed_time_buckets", "all"))):
            return False
        if not allowed_direction(direction, str(getattr(smc_args, "allowed_directions", "all"))):
            return False
        daily = daily_candles_from_4h(engine.c4h)
        daily_ts = [candle.ts for candle in daily]
        h4_highs, h4_lows = precompute_swings(engine.c4h, n=2, lookback=80)
        d1_highs, d1_lows = precompute_swings(daily, n=2, lookback=20)
        h4_idx = completed_4h_idx_for_entry(engine.mapping, latest_closed_idx)
        d1_idx = completed_d1_idx_for_entry(daily_ts, engine.c15m[latest_closed_idx].ts)
        h4_bias = htf_structure_bias(engine.c4h, h4_highs, h4_lows, h4_idx) if h4_idx >= 0 else Direction.NONE
        d1_bias = htf_structure_bias(daily, d1_highs, d1_lows, d1_idx) if d1_idx >= 0 else Direction.NONE
        if bool(getattr(smc_args, "require_h4_bias_align", True)) and bool(getattr(smc_args, "require_htf_bias_align", True)) and h4_bias != direction:
            return False
        if bool(getattr(smc_args, "require_d1_bias_align", False)) and bool(getattr(smc_args, "require_htf_bias_align", True)) and d1_bias != direction:
            return False
        if bool(getattr(smc_args, "require_ote_touch", False)) and not bool(getattr(retest, "ote_touched", False)):
            return False
        if float(getattr(event, "displacement_body_atr", 0.0) or 0.0) < float(getattr(smc_args, "min_displacement_body_atr", 0.0)):
            return False
        if float(getattr(event, "displacement_range_atr", 0.0) or 0.0) < float(getattr(smc_args, "min_displacement_range_atr", 0.0)):
            return False
        lag = (int(event.mss_idx) - int(event.sweep_idx)) if getattr(event, "mss_idx", None) is not None else None
        max_lag = int(getattr(smc_args, "max_mss_lag_bars", 0) or 0)
        if max_lag > 0 and lag is not None and lag > max_lag:
            return False
        other_floor = int(case_params.get("other_min_mss_lag_bars", 0) or 0)
        if other_floor > 0 and bucket == "other" and lag is not None and lag < other_floor:
            return False
        global_floor = int(case_params.get("global_min_mss_lag_bars", 0) or 0)
        if global_floor > 0 and lag is not None and lag < global_floor:
            return False
        global_ceiling = int(case_params.get("global_max_mss_lag_bars", 0) or 0)
        if global_ceiling > 0 and lag is not None and lag > global_ceiling:
            return False
        ny_limit = int(case_params.get("ny_max_mss_lag_bars", 0) or 0)
        if ny_limit > 0 and bucket == "ny_am_killzone" and lag is not None and lag > ny_limit:
            return False
        if bool(case_params.get("drop_asia_session", False)) and bucket == "asia_evening_ny":
            return False
        if direction == Direction.BEAR and float(getattr(smc_args, "bear_min_sweep_distance_pct", 0.0) or 0.0) > 0.0:
            if float(getattr(event, "sweep_distance_pct", 0.0) or 0.0) < float(getattr(smc_args, "bear_min_sweep_distance_pct", 0.0)):
                return False
        return True

    def _smc_short_candidate(self, engine: Any, latest_closed_idx: int) -> dict[str, Any] | None:
        if not bool(self.config.enable_smc_short_live):
            return None
        if int(latest_closed_idx) < int(self.config.smc_min_entry_idx):
            return None
        try:
            from scripts.report_pa_ict_liquidity_features import atr_series, scan_events
            from scripts.research_smc_standalone_v1 import build_event_scan_args
        except Exception:
            return None
        smc_args = self._smc_strategy_args()
        case_params = self._smc_case_params()
        scan_args = build_event_scan_args(smc_args)
        scan_args.allow_incomplete_tail = True
        scan_args.outcome_lookahead_bars = 0
        events = scan_events(engine.c15m[: latest_closed_idx + 1], scan_args)
        matches = [
            event
            for event in events
            if self._smc_event_allowed(engine, event, latest_closed_idx, smc_args, Direction.BEAR, case_params)
        ]
        if not matches:
            return None
        event = sorted(matches, key=lambda item: int(item.sweep_idx), reverse=True)[0]
        retest = event.retest
        if retest is None:
            return None
        source_key = f"smc|{self.config.smc_case}|{retest.timestamp}|{event.sweep_idx}|{event.mss_idx}"
        if self._candidate_seen(source_key):
            return None
        atr = atr_series(engine.c15m[: latest_closed_idx + 1], int(getattr(smc_args, "atr_period", 14)))
        entry_price = float(retest.close)
        stop_buffer = atr[latest_closed_idx] * float(getattr(smc_args, "stop_buffer_atr", 0.05)) if latest_closed_idx < len(atr) else 0.0
        stop_price = float(event.sweep_extreme) + stop_buffer
        risk = stop_price - entry_price
        if entry_price <= 0 or risk <= 0:
            self._mark_candidate_seen(source_key)
            return None
        target_rr = float(self.config.smc_target_rr)
        target_price = entry_price - risk * target_rr
        if target_price <= 0:
            self._mark_candidate_seen(source_key)
            return None
        leverage = float(self.config.smc_leverage)
        position_size_pct = float(self.config.smc_position_size_pct)
        notional = max(float(getattr(engine, "capital", 0.0) or 0.0), 0.0) * leverage * position_size_pct
        maintenance = max(float(self.config.smc_maintenance_margin_pct), 0.0) / 100.0
        liquidation_price = entry_price * (1.0 + (1.0 / max(leverage, 1e-9)) - maintenance)
        liquidation_buffer_pct = (liquidation_price - stop_price) / entry_price * 100.0
        if liquidation_buffer_pct < float(self.config.smc_min_liq_buffer_pct):
            self._mark_candidate_seen(source_key)
            return None
        return {
            "event_type": "smc_short",
            "source_key": source_key,
            "entry_idx": latest_closed_idx,
            "direction": Direction.BEAR,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "target_rr": target_rr,
            "max_hold_bars": int(self.config.smc_max_hold_bars),
            "trail_style": str(self.config.smc_trail_style or "tight"),
            "leverage": leverage,
            "position_size_pct": position_size_pct,
            "maintenance_margin_pct": float(self.config.smc_maintenance_margin_pct),
            "requested_notional": notional,
            "source": {
                "smc_case": self.config.smc_case,
                "sweep_idx": event.sweep_idx,
                "sweep_time": event.sweep_time,
                "mss_idx": event.mss_idx,
                "mss_time": event.mss_time,
                "time_bucket": event.time_bucket,
                "mss_lag_bars": (int(event.mss_idx) - int(event.sweep_idx)) if event.mss_idx is not None else None,
                "liquidation_buffer_pct": liquidation_buffer_pct,
            },
        }

    def _last_flat_start_idx(self, engine: Any, latest_closed_idx: int) -> int:
        snapshot = self.store.load_snapshot()
        if isinstance(snapshot, dict):
            position = snapshot.get("position")
            if isinstance(position, dict):
                try:
                    entry_idx = int(position.get("entry_idx", latest_closed_idx) or latest_closed_idx)
                except (TypeError, ValueError):
                    entry_idx = latest_closed_idx
                return max(0, entry_idx)
        last_exit_idx: int | None = None
        for item in self.store.recent_actions(1000):
            if item.get("action_type") != ActionType.CLOSE_POSITION.value:
                continue
            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                continue
            try:
                idx = int(metadata.get("index"))
            except (TypeError, ValueError):
                continue
            if idx <= int(latest_closed_idx):
                last_exit_idx = max(last_exit_idx or idx, idx)
        if last_exit_idx is not None:
            return int(last_exit_idx)
        return max(0, min(100, int(latest_closed_idx)))

    def _flat_days_since_last_position(self, engine: Any, latest_closed_idx: int) -> float:
        start_idx = self._last_flat_start_idx(engine, latest_closed_idx)
        start_ts = datetime.fromtimestamp(engine.c15m[start_idx].ts, tz=timezone.utc)
        current_ts = datetime.fromtimestamp(engine.c15m[latest_closed_idx].ts, tz=timezone.utc)
        return max((current_ts - start_ts).total_seconds() / 86400.0, 0.0)

    def _gap_smc_short_candidate(self, engine: Any, latest_closed_idx: int) -> dict[str, Any] | None:
        if not bool(self.config.enable_gap_smc_short_live):
            return None
        if int(latest_closed_idx) < int(self.config.gap_smc_short_min_entry_idx):
            return None
        if getattr(engine, "position", None) is not None:
            return None
        flat_days = self._flat_days_since_last_position(engine, latest_closed_idx)
        if flat_days < float(self.config.gap_smc_short_min_flat_days):
            return None
        try:
            from scripts.report_pa_ict_liquidity_features import atr_series, scan_events
            from scripts.research_smc_standalone_v1 import build_event_scan_args
        except Exception:
            return None
        smc_args = self._gap_smc_short_strategy_args()
        case_params = self._gap_smc_short_case_params()
        scan_args = build_event_scan_args(smc_args)
        scan_args.allow_incomplete_tail = True
        scan_args.outcome_lookahead_bars = 0
        events = scan_events(engine.c15m[: latest_closed_idx + 1], scan_args)
        matches = [
            event
            for event in events
            if self._smc_event_allowed(engine, event, latest_closed_idx, smc_args, Direction.BEAR, case_params)
        ]
        if not matches:
            return None
        event = sorted(matches, key=lambda item: int(item.sweep_idx), reverse=True)[0]
        retest = event.retest
        if retest is None:
            return None
        source_key = f"gap_smc|{self.config.gap_smc_short_case}|{retest.timestamp}|{event.sweep_idx}|{event.mss_idx}"
        if self._candidate_seen(source_key):
            return None
        atr = atr_series(engine.c15m[: latest_closed_idx + 1], int(getattr(smc_args, "atr_period", 14)))
        entry_price = float(retest.close)
        stop_buffer = atr[latest_closed_idx] * float(getattr(smc_args, "stop_buffer_atr", 0.05)) if latest_closed_idx < len(atr) else 0.0
        stop_price = float(event.sweep_extreme) + stop_buffer
        risk = stop_price - entry_price
        if entry_price <= 0 or risk <= 0:
            self._mark_candidate_seen(source_key)
            return None
        target_rr = float(self.config.gap_smc_short_target_rr)
        target_price = entry_price - risk * target_rr
        if target_price <= 0:
            self._mark_candidate_seen(source_key)
            return None
        leverage = float(self.config.gap_smc_short_leverage)
        position_size_pct = float(self.config.gap_smc_short_position_size_pct)
        notional = max(float(getattr(engine, "capital", 0.0) or 0.0), 0.0) * leverage * position_size_pct
        maintenance = max(float(self.config.gap_smc_short_maintenance_margin_pct), 0.0) / 100.0
        liquidation_price = entry_price * (1.0 + (1.0 / max(leverage, 1e-9)) - maintenance)
        liquidation_buffer_pct = (liquidation_price - stop_price) / entry_price * 100.0
        if liquidation_buffer_pct < float(self.config.gap_smc_short_min_liq_buffer_pct):
            self._mark_candidate_seen(source_key)
            return None
        return {
            "event_type": "gap_smc_short_expansion",
            "source_key": source_key,
            "entry_idx": latest_closed_idx,
            "direction": Direction.BEAR,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "target_rr": target_rr,
            "max_hold_bars": int(self.config.gap_smc_short_max_hold_bars),
            "trail_style": str(self.config.gap_smc_short_trail_style or "tight"),
            "leverage": leverage,
            "position_size_pct": position_size_pct,
            "maintenance_margin_pct": float(self.config.gap_smc_short_maintenance_margin_pct),
            "requested_notional": notional,
            "source": {
                "smc_case": self.config.gap_smc_short_case,
                "flat_days": round(flat_days, 4),
                "min_flat_days": float(self.config.gap_smc_short_min_flat_days),
                "sweep_idx": event.sweep_idx,
                "sweep_time": event.sweep_time,
                "mss_idx": event.mss_idx,
                "mss_time": event.mss_time,
                "time_bucket": event.time_bucket,
                "mss_lag_bars": (int(event.mss_idx) - int(event.sweep_idx)) if event.mss_idx is not None else None,
                "liquidation_buffer_pct": liquidation_buffer_pct,
            },
        }

    def _smc_long_candidate(self, engine: Any, latest_closed_idx: int) -> dict[str, Any] | None:
        if not bool(self.config.enable_smc_long_live):
            return None
        try:
            from scripts.report_pa_ict_liquidity_features import atr_series, scan_events
            from scripts.research_smc_standalone_v1 import build_event_scan_args
        except Exception:
            return None
        smc_args = self._smc_long_strategy_args()
        case_params = self._smc_long_case_params()
        scan_args = build_event_scan_args(smc_args)
        scan_args.allow_incomplete_tail = True
        scan_args.outcome_lookahead_bars = 0
        events = scan_events(engine.c15m[: latest_closed_idx + 1], scan_args)
        matches = [
            event
            for event in events
            if self._smc_event_allowed(engine, event, latest_closed_idx, smc_args, Direction.BULL, case_params)
        ]
        if not matches:
            return None
        event = sorted(matches, key=lambda item: int(item.sweep_idx), reverse=True)[0]
        retest = event.retest
        if retest is None:
            return None
        source_key = f"smc_long|{self.config.smc_long_case}|{retest.timestamp}|{event.sweep_idx}|{event.mss_idx}"
        if self._candidate_seen(source_key):
            return None
        atr = atr_series(engine.c15m[: latest_closed_idx + 1], int(getattr(smc_args, "atr_period", 14)))
        entry_price = float(retest.close)
        stop_buffer = atr[latest_closed_idx] * float(getattr(smc_args, "stop_buffer_atr", 0.05)) if latest_closed_idx < len(atr) else 0.0
        stop_price = float(event.sweep_extreme) - stop_buffer
        risk = entry_price - stop_price
        if entry_price <= 0 or risk <= 0:
            self._mark_candidate_seen(source_key)
            return None
        target_rr = float(self.config.smc_long_target_rr)
        target_price = entry_price + risk * target_rr
        leverage = float(self.config.smc_long_leverage)
        position_size_pct = float(self.config.smc_long_position_size_pct)
        notional = max(float(getattr(engine, "capital", 0.0) or 0.0), 0.0) * leverage * position_size_pct
        maintenance = max(float(self.config.smc_long_maintenance_margin_pct), 0.0) / 100.0
        liquidation_price = entry_price * (1.0 - (1.0 / max(leverage, 1e-9)) + maintenance)
        liquidation_buffer_pct = (stop_price - liquidation_price) / entry_price * 100.0
        if liquidation_buffer_pct < float(self.config.smc_long_min_liq_buffer_pct):
            self._mark_candidate_seen(source_key)
            return None
        return {
            "event_type": "smc_long",
            "source_key": source_key,
            "entry_idx": latest_closed_idx,
            "direction": Direction.BULL,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "target_rr": target_rr,
            "max_hold_bars": int(self.config.smc_long_max_hold_bars),
            "trail_style": str(self.config.smc_long_trail_style or "tight"),
            "leverage": leverage,
            "position_size_pct": position_size_pct,
            "maintenance_margin_pct": float(self.config.smc_long_maintenance_margin_pct),
            "requested_notional": notional,
            "source": {
                "smc_case": self.config.smc_long_case,
                "sweep_idx": event.sweep_idx,
                "sweep_time": event.sweep_time,
                "mss_idx": event.mss_idx,
                "mss_time": event.mss_time,
                "time_bucket": event.time_bucket,
                "mss_lag_bars": (int(event.mss_idx) - int(event.sweep_idx)) if event.mss_idx is not None else None,
                "liquidation_buffer_pct": liquidation_buffer_pct,
            },
        }

    def _open_overlay_candidate(self, engine: Any, candidate: dict[str, Any]) -> StrategyAction:
        action = engine.open_position(
            int(candidate["entry_idx"]),
            str(candidate["direction"]),
            float(candidate["entry_price"]),
            float(candidate["stop_price"]),
            float(candidate["target_price"]),
            target_rr_override=float(candidate["target_rr"]),
            max_hold_bars_override=int(candidate["max_hold_bars"]),
            trail_style_override=str(candidate["trail_style"]),
            candidate_event_type=str(candidate["event_type"]),
            requested_notional_override=float(candidate.get("requested_notional", 0.0) or 0.0),
        )
        metadata = dict(action.metadata or {})
        metadata.update(
            {
                "candidate_event_type": candidate["event_type"],
                "source": "live_candidate_arbitration",
                "source_key": candidate.get("source_key"),
                "candidate_source": candidate.get("source"),
                "candidate_leverage": candidate.get("leverage"),
                "candidate_position_size_pct": candidate.get("position_size_pct"),
                "candidate_maintenance_margin_pct": candidate.get("maintenance_margin_pct"),
            }
        )
        action.metadata = metadata
        return action

    def _apply_live_candidate_arbitration(
        self,
        engine: Any,
        actions: list[StrategyAction],
        latest_closed_idx: int,
    ) -> tuple[list[StrategyAction], dict[str, Any]]:
        raw_open_actions = [action for action in actions if action.type in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}]
        open_actions, score_gate_rejected = self._apply_sota_score_gate_to_open_actions(
            engine,
            raw_open_actions,
            latest_closed_idx,
        )
        open_actions, structure_gate_rejected, structure_gate_recalled = (
            self._apply_sota_structure_gate_to_open_actions(open_actions)
        )
        close_actions = [action for action in actions if action.type == ActionType.CLOSE_POSITION]
        candidates: list[dict[str, Any]] = []
        for action in open_actions:
            event_type = self._open_action_event_type(action)
            metadata = dict(action.metadata or {})
            metadata.setdefault("candidate_event_type", event_type)
            action.metadata = metadata
            candidates.append(
                {
                    "event_type": event_type,
                    "source_key": f"{event_type}|{action.timestamp}|{action.direction}|{action.entry_price}",
                    "entry_idx": int(metadata.get("index", latest_closed_idx) or latest_closed_idx),
                    "action": action,
                }
            )

        smc_short_candidate = self._smc_short_candidate(engine, latest_closed_idx)
        if smc_short_candidate is not None:
            candidates.append(smc_short_candidate)
        gap_smc_short_candidate = self._gap_smc_short_candidate(engine, latest_closed_idx)
        if gap_smc_short_candidate is not None:
            candidates.append(gap_smc_short_candidate)
        smc_long_candidate = self._smc_long_candidate(engine, latest_closed_idx)
        if smc_long_candidate is not None:
            candidates.append(smc_long_candidate)

        if not candidates:
            cleared_rejected_position = self._clear_local_position_for_rejected_open_actions(
                engine,
                score_gate_rejected + structure_gate_rejected,
            )
            output_actions = [action for action in actions if action.type not in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}]
            decision = {
                "enabled": True,
                "decision": "no_candidates",
                "candidates": [],
                "score_gate_rejected": score_gate_rejected,
                "structure_gate_rejected": structure_gate_rejected,
                "structure_gate_recalled": structure_gate_recalled,
                "cleared_rejected_position": cleared_rejected_position,
            }
            if score_gate_rejected or structure_gate_rejected:
                self.store.append_action(engine._timestamp_for_idx(latest_closed_idx), "LIVE_CANDIDATE_ARBITRATION", decision)
            return output_actions, decision

        candidates.sort(
            key=lambda item: (
                int(item.get("entry_idx", latest_closed_idx) or latest_closed_idx),
                self._live_candidate_priority_value(str(item["event_type"])),
            )
        )
        selected = candidates[0]
        selected_event_type = str(selected["event_type"])
        rejected = [item for item in candidates[1:]]
        for item in candidates:
            source_key = item.get("source_key")
            if source_key:
                self._mark_candidate_seen(str(source_key))

        output_actions = [action for action in actions if action.type not in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}]
        if "action" in selected:
            selected_action = selected["action"]
        else:
            if getattr(engine, "position", None) is not None and open_actions:
                local_entry_times = {action.timestamp for action in open_actions}
                if getattr(engine.position, "entry_time", None) in local_entry_times:
                    engine.position = None
            selected_action = self._open_overlay_candidate(engine, selected)
        output_actions.append(selected_action)

        decision = {
            "enabled": True,
            "decision": "accepted",
            "selected": self._action_candidate_summary(selected_action, selected_event_type),
            "rejected": [
                self._action_candidate_summary(item["action"], str(item["event_type"]))
                if "action" in item
                else {
                    "event_type": item.get("event_type"),
                    "source_key": item.get("source_key"),
                    "entry_idx": item.get("entry_idx"),
                    "direction": item.get("direction"),
                    "entry_price": item.get("entry_price"),
                    "stop_price": item.get("stop_price"),
                    "target_price": item.get("target_price"),
                }
                for item in rejected
            ] + score_gate_rejected,
            "priority": self._live_candidate_priority(),
            "score_gate_rejected": score_gate_rejected,
            "structure_gate_rejected": structure_gate_rejected,
            "structure_gate_recalled": structure_gate_recalled,
        }
        self.store.append_action(selected_action.timestamp, "LIVE_CANDIDATE_ARBITRATION", decision)
        return output_actions, decision

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

    def record_action(self, action: StrategyAction) -> None:
        self.store.append_action(action.timestamp, action.type.value, asdict(action))

    def execute_action(self, action: StrategyAction, engine: Any) -> dict[str, Any]:
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
                self._rollback_unexecuted_close(action, engine, reason=str(shadow_decision.get("status") or "skipped_close"))
                self.store.append_action(
                    action.timestamp,
                    "EXECUTION_SKIPPED",
                    {
                        "action": asdict(action),
                        "decision": shadow_decision,
                    },
                )
            return shadow_decision
        if action.type == ActionType.UPDATE_STOP:
            if self.config.mode != "live" or not self.config.enable_exchange_brackets:
                self.record_action(action)
                return {"status": "recorded_only", "action": action.type.value, "stop_price": action.stop_price}
            self.record_action(action)
            return self._amend_exchange_brackets(action, engine)

        sizing = self._resolve_order_sizing(action, engine)
        if sizing.get("status") != "ok":
            if action.type in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}:
                self._rollback_unexecuted_open(
                    action,
                    engine,
                    reason=str(sizing.get("reason") or sizing.get("status") or "sizing_failed"),
                )
                self.store.append_action(
                    action.timestamp,
                    "EXECUTION_SKIPPED",
                    {
                        "action": asdict(action),
                        "decision": sizing,
                    },
                )
            return sizing
        overlay_skipped_dynamic = False
        if not (self._is_overlay_open_action(action) and bool(self.config.overlay_skip_dynamic_high_leverage)):
            sizing, dynamic_decision = self._dynamic_high_leverage_pre_open(action, sizing, engine)
            if dynamic_decision is not None:
                self._rollback_unexecuted_open(
                    action,
                    engine,
                    reason=str(dynamic_decision.get("status") or "skipped_open"),
                )
                self.store.append_action(
                    action.timestamp,
                    "EXECUTION_SKIPPED",
                    {
                        "action": asdict(action),
                        "decision": dynamic_decision,
                    },
                )
                return dynamic_decision
        else:
            overlay_skipped_dynamic = True
            self._apply_overlay_execution_metadata(engine, action, sizing)
        exit_profile_decision = self._apply_open_exit_profile_metadata(engine, action)
        high_leverage_decision = self._high_leverage_guard_pre_open(action, sizing)
        if high_leverage_decision is not None:
            self._rollback_unexecuted_open(
                action,
                engine,
                reason=str(high_leverage_decision.get("status") or "skipped_open"),
            )
            self.store.append_action(
                action.timestamp,
                "EXECUTION_SKIPPED",
                {
                    "action": asdict(action),
                    "decision": high_leverage_decision,
                },
            )
            return high_leverage_decision
        soft_stop_decision = None
        if action.type in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT} and bool(self.config.enable_sota_soft_stop_recovery_overlay_live):
            soft_stop_decision = self._sota_soft_stop_prepare_open(action, sizing)

        if self.config.mode == "paper":
            self.record_action(action)
            if action.type in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}:
                self._shadow_gate_mark_real_position(True, action, "paper_open_accepted")
                if bool(self.config.enable_sota_soft_stop_recovery_overlay_live):
                    soft_stop_decision = self._sota_soft_stop_after_open(action, sizing)
            if action.type == ActionType.CLOSE_POSITION:
                self._shadow_gate_after_close(action, engine)
                self._dynamic_high_leverage_after_close(action, engine)
                self._sota_soft_stop_after_close(action)
            return {
                "status": "paper_recorded",
                "action": action.type.value,
                "amount": sizing.get("amount"),
                "order_unit": sizing.get("order_unit"),
                "notional_usdt": sizing.get("notional_usdt"),
                "expected_notional_usdt": sizing.get("expected_notional_usdt"),
                "balance_source": sizing.get("balance_source"),
                "position_size_pct": self.config.position_size_pct,
                "dynamic_high_leverage": sizing.get("dynamic_high_leverage"),
                "overlay_skipped_dynamic_high_leverage": overlay_skipped_dynamic,
                "sota_soft_stop_live": soft_stop_decision,
                "exit_profile": exit_profile_decision,
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
            if bool(self.config.enable_sota_soft_stop_recovery_overlay_live):
                soft_stop_decision = self._sota_soft_stop_after_open(action, sizing, observed)
            self.record_action(action)
            dynamic_info = sizing.get("dynamic_high_leverage") if isinstance(sizing.get("dynamic_high_leverage"), dict) else {}
            open_lines = [
                "[开仓已确认]",
                f"方向: {direction}",
                *self._open_strategy_context_lines(action, dynamic_info),
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
            soft_stop_line = self._sota_soft_stop_open_line(soft_stop_decision)
            if soft_stop_line:
                open_lines.append(soft_stop_line)
            if isinstance(exit_profile_decision, dict) and bool(exit_profile_decision.get("applied")):
                open_lines.append(f"Exit profile: {exit_profile_decision.get('profile')}")
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
            metadata = dict(action.metadata or {})
            metadata["source"] = metadata.get("source") or "live_order_execution"
            metadata["order_id"] = order.get("id")
            metadata["observed_contracts"] = observed.get("contracts")
            metadata["observed_notional_usdt"] = observed.get("notional_usdt")
            action.metadata = metadata
            self.record_action(action)
            self._shadow_gate_after_close(action, engine)
            self._dynamic_high_leverage_after_close(action, engine)
            self._sota_soft_stop_after_close(action)
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

        self.record_action(action)
        return {"status": "recorded_only", "action": action.type.value}

    def close_for_router_switch(self, *, reason: str) -> dict[str, Any]:
        if self.config.mode == "paper":
            return {"status": "paper_flatten_skipped", "reason": reason}

        try:
            engine, _start_idx = self.load_engine()
        except Exception as exc:
            engine = None
            load_error = str(exc)
        else:
            load_error = None

        results: list[dict[str, Any]] = []
        for pos_side, side in [("long", "sell"), ("short", "buy")]:
            state = self._fetch_position_state(pos_side)
            amount = float(state.get("contracts", 0.0) or 0.0)
            if amount <= 0:
                continue
            order = self.client.create_order(
                self.config.symbol,
                "market",
                side,
                amount,
                params={"reduceOnly": True, "tdMode": self.config.margin_mode, "posSide": pos_side},
            )
            observed = self._wait_for_position_state(pos_side, expect_open=False)
            results.append(
                {
                    "pos_side": pos_side,
                    "amount": amount,
                    "order": order,
                    "observed_position": observed,
                    "confirmed_flat": float(observed.get("contracts", 0.0) or 0.0) <= 0,
                }
            )

        confirmed_flat = bool(results) and all(item["confirmed_flat"] for item in results)
        should_sync_flat = not results or confirmed_flat
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if should_sync_flat and engine is not None:
            had_local_position = getattr(engine, "position", None) is not None
            self._sync_manual_flat_position(engine, context=reason, timestamp=timestamp)
            should_mark_shadow_flat = not had_local_position
        else:
            should_mark_shadow_flat = should_sync_flat

        if should_mark_shadow_flat and self._shadow_gate_enabled():
            state = self._load_shadow_gate_state()
            state["real_position_open"] = False
            state["real_position_direction"] = None
            state["paper_entry_time"] = None
            self._shadow_append_event(
                state,
                {
                    "time": timestamp,
                    "event": "router_flatten_btc",
                    "reason": reason,
                    "load_engine_error": load_error,
                },
            )
            self._save_shadow_gate_state(state)

        if not results:
            return {"status": "skipped", "reason": reason, "orders": [], "load_engine_error": load_error}

        status = "submitted" if confirmed_flat else "submitted_but_unconfirmed"
        return {"status": status, "reason": reason, "orders": results, "load_engine_error": load_error}

    def _rollback_unexecuted_open(self, action: StrategyAction, engine: Any, *, reason: str) -> None:
        rolled_back = False
        position = getattr(engine, "position", None)
        if position is not None:
            entry_time = str(getattr(position, "entry_time", "") or "")
            direction = str(getattr(position, "direction", "") or "")
            direction_matches = not action.direction or direction == str(action.direction)
            if entry_time == action.timestamp and direction_matches:
                engine.position = None
                rolled_back = True

        self.store.append_action(
            action.timestamp,
            "UNEXECUTED_OPEN_ROLLBACK",
            {
                "reason": reason,
                "rolled_back": rolled_back,
                "direction": action.direction,
                "entry_price": action.entry_price,
                "stop_price": action.stop_price,
                "target_price": action.target_price,
            },
        )

    def _rollback_unexecuted_close(self, action: StrategyAction, engine: Any, *, reason: str) -> None:
        metadata = action.metadata or {}
        pnl = self._safe_float(metadata.get("net_pnl"))
        if pnl is not None:
            engine.capital = float(getattr(engine, "capital", 0.0) or 0.0) - pnl

        if action.reason and isinstance(getattr(engine, "exit_reasons", None), dict):
            current = int(engine.exit_reasons.get(action.reason, 0) or 0)
            if current <= 1:
                engine.exit_reasons.pop(action.reason, None)
            else:
                engine.exit_reasons[action.reason] = current - 1

        trades = getattr(engine, "trades", None)
        if isinstance(trades, list) and trades:
            latest = trades[-1]
            latest_exit_time = str(getattr(latest, "exit_time", "") or "")
            latest_reason = str(getattr(latest, "exit_reason", "") or "")
            latest_pnl = self._safe_float(getattr(latest, "pnl", None))
            pnl_matches = pnl is None or latest_pnl is None or abs(float(latest_pnl) - float(pnl)) < 1e-6
            if latest_exit_time == action.timestamp and latest_reason == str(action.reason or "") and pnl_matches:
                trades.pop()

        self.store.append_action(
            action.timestamp,
            "UNEXECUTED_CLOSE_ROLLBACK",
            {
                "reason": reason,
                "direction": action.direction,
                "exit_price": action.exit_price,
                "close_reason": action.reason,
                "net_pnl": pnl,
            },
        )

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

    def _high_leverage_guard_enabled(self) -> bool:
        return (
            bool(self.config.enable_high_leverage_guard)
            and float(self.config.leverage) >= float(self.config.high_leverage_guard_min_leverage)
        )

    def _high_leverage_guard_pre_open(self, action: StrategyAction, sizing: dict[str, Any]) -> dict[str, Any] | None:
        if not self._high_leverage_guard_enabled():
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
            state["paper_entry_time"] = None
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
        metadata = action.metadata or {}
        leverage = self._safe_float(metadata.get("candidate_leverage")) or float(self.config.leverage)
        maintenance_margin_pct_raw = self._safe_float(metadata.get("candidate_maintenance_margin_pct"))
        if maintenance_margin_pct_raw is None:
            maintenance_margin_pct_raw = float(self.config.high_leverage_maintenance_margin_pct)
        maintenance_margin_pct = max(maintenance_margin_pct_raw, 0.0) / 100.0
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
            "maintenance_margin_pct": round(maintenance_margin_pct_raw, 6),
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
                margin_usdt = float(
                    metadata.get(
                        "margin_usdt",
                        notional / self.config.leverage if self.config.leverage > 0 else notional,
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

        state = self._load_dynamic_high_leverage_state(engine)
        diagnostics = self._dynamic_action_diagnostics(action, sizing, engine)
        risk_mode, mode_reasons, mode_stats = self._dynamic_next_mode(state, diagnostics)
        effective_leverage, leverage_reasons = self._dynamic_select_effective_leverage(
            state,
            risk_mode,
            diagnostics,
            mode_stats,
        )
        effective_leverage, score_bucket_decision = self._apply_long_score_bucket_sizing(
            action,
            effective_leverage,
            risk_mode,
        )
        if isinstance(score_bucket_decision, dict) and bool(score_bucket_decision.get("applied")):
            leverage_reasons.append(str(score_bucket_decision.get("reason") or "score_bucket_sizing"))
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
            "score_bucket_sizing": score_bucket_decision,
        }
        state["mode"] = risk_mode
        state["last_decision"] = decision
        state["last_update_time"] = action.timestamp
        self._save_dynamic_high_leverage_state(state)
        metadata = dict(action.metadata or {})
        metadata["dynamic_high_leverage"] = decision
        if score_bucket_decision is not None:
            metadata["score_bucket_sizing"] = score_bucket_decision
        action.metadata = metadata

        if diagnostics["stop_distance_pct"] > max_stop_distance:
            if self._shadow_gate_enabled():
                shadow_state = self._load_shadow_gate_state(engine)
                shadow_state["real_position_open"] = False
                shadow_state["real_position_direction"] = None
                shadow_state["paper_entry_time"] = None
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

    def _apply_overlay_execution_metadata(
        self,
        engine: Any,
        action: StrategyAction,
        sizing: dict[str, Any],
    ) -> None:
        position = getattr(engine, "position", None)
        if position is None:
            return
        metadata = action.metadata or {}
        requested_notional = (
            self._safe_float(sizing.get("notional_usdt"))
            or self._safe_float(metadata.get("notional"))
            or self._safe_float(metadata.get("requested_notional_override"))
        )
        leverage = self._safe_float(metadata.get("candidate_leverage"))
        if leverage is None:
            capital_at_entry = self._safe_float(metadata.get("capital_at_entry")) or self._safe_float(getattr(position, "capital_at_entry", None))
            leverage = (
                requested_notional / capital_at_entry
                if requested_notional is not None and capital_at_entry and capital_at_entry > 0
                else None
            )
        diagnostics = self._high_leverage_open_diagnostics(action, sizing)
        setattr(position, "execution_effective_leverage", round(leverage, 6) if leverage is not None else None)
        setattr(position, "execution_risk_mode", "overlay_fixed")
        setattr(position, "execution_leverage_reasons", [f"overlay_fixed:{self._open_action_event_type(action)}"])
        setattr(position, "execution_requested_notional", requested_notional)
        setattr(position, "execution_target_notional", requested_notional)
        setattr(position, "execution_guard_diagnostics", diagnostics)

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
            cache = self._load_markets_cache()
            if cache:
                self._markets_cache = cache
                self._hydrate_exchange_markets(cache)
            else:
                self._markets_cache = self.client.load_markets()
        return self._markets_cache

    def _load_markets_cache(self) -> dict[str, Any]:
        configured = self.config.markets_cache_path
        if not configured:
            return {}
        path = self._resolve_runtime_path(configured)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        symbol = str(self.config.symbol)
        market = payload.get(symbol)
        if isinstance(market, dict):
            return {symbol: market}
        return {key: value for key, value in payload.items() if isinstance(key, str) and isinstance(value, dict)}

    def _hydrate_exchange_markets(self, markets: dict[str, Any]) -> None:
        try:
            self.client.exchange.set_markets(list(markets.values()))
        except Exception:
            return

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

    def _entry_time_to_ms(self, value: Any) -> int | None:
        text = str(value or "").strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        parsed: datetime | None = None
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    parsed = None
        if parsed is None:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return int(parsed.timestamp() * 1000)

    def _trade_fill_timestamp_ms(self, trade: dict[str, Any]) -> int | None:
        candidates: list[Any] = [
            trade.get("timestamp"),
            trade.get("datetime"),
        ]
        info = trade.get("info")
        if isinstance(info, dict):
            candidates.extend([info.get("fillTime"), info.get("uTime"), info.get("cTime"), info.get("ts")])
        for candidate in candidates:
            if candidate in (None, ""):
                continue
            if isinstance(candidate, (int, float)):
                value = float(candidate)
                if value > 1e12:
                    return int(value)
                if value > 1e9:
                    return int(value * 1000)
                continue
            text = str(candidate).strip()
            if not text:
                continue
            if text.isdigit():
                value = int(text)
                if value > 1_000_000_000_000:
                    return value
                if value > 1_000_000_000:
                    return value * 1000
                continue
            normalized = text.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            return int(parsed.timestamp() * 1000)
        return None

    def _trade_fill_fee_abs(self, trade: dict[str, Any]) -> float:
        info = trade.get("info")
        candidates: list[Any] = [trade.get("fee")]
        if isinstance(info, dict):
            candidates.extend([info.get("fee"), info.get("fillFee")])
        for candidate in candidates:
            if isinstance(candidate, dict):
                numeric = self._safe_float(candidate.get("cost"))
                if numeric is not None:
                    return abs(numeric)
                continue
            numeric = self._safe_float(candidate)
            if numeric is not None:
                return abs(numeric)
        return 0.0

    def _trade_fill_realized_pnl(self, trade: dict[str, Any]) -> float:
        info = trade.get("info")
        candidates: list[Any] = []
        if isinstance(info, dict):
            candidates.extend([info.get("fillPnl"), info.get("pnl")])
        candidates.append(trade.get("profit"))
        for candidate in candidates:
            numeric = self._safe_float(candidate)
            if numeric is not None:
                return float(numeric)
        return 0.0

    def _trade_fill_order_id(self, trade: dict[str, Any]) -> str | None:
        candidates: list[Any] = [trade.get("order"), trade.get("orderId"), trade.get("id")]
        info = trade.get("info")
        if isinstance(info, dict):
            candidates.extend([info.get("ordId"), info.get("orderId"), info.get("tradeId")])
        for candidate in candidates:
            text = str(candidate or "").strip()
            if text:
                return text
        return None

    def _trade_fill_pos_side(self, trade: dict[str, Any]) -> str:
        info = trade.get("info")
        candidates: list[Any] = [trade.get("posSide")]
        if isinstance(info, dict):
            candidates.append(info.get("posSide"))
        candidates.extend([trade.get("side")])
        if isinstance(info, dict):
            candidates.append(info.get("side"))
        for candidate in candidates:
            text = str(candidate or "").strip().lower()
            if text:
                return text
        return ""

    def _trade_fill_side(self, trade: dict[str, Any]) -> str:
        side = str(trade.get("side") or "").strip().lower()
        if side:
            return side
        info = trade.get("info")
        if isinstance(info, dict):
            side = str(info.get("side") or "").strip().lower()
        return side

    def _market_inst_id(self) -> str:
        market = self._market()
        inst_id = market.get("id")
        if inst_id:
            return str(inst_id)
        return str(self.config.symbol)

    def _fetch_external_flat_fill_close(self, position: Any) -> ExternalFlatFillClose | None:
        entry_time_ms = self._entry_time_to_ms(getattr(position, "entry_time", None))
        if entry_time_ms is None:
            return None
        order_id = str(getattr(position, "exchange_order_id", "") or "").strip()
        direction = getattr(position, "direction", None)
        expected_pos_side = "long" if direction == Direction.BULL else "short"
        expected_close_side = "sell" if direction == Direction.BULL else "buy"
        window_since = max(0, entry_time_ms - 6 * 60 * 60 * 1000)
        entry_fills: list[dict[str, Any]] = []
        if order_id:
            try:
                entry_fills = self.client.fetch_order_fills(inst_id=self._market_inst_id(), order_id=order_id)
            except Exception:
                entry_fills = []
        try:
            trades = self.client.fetch_my_trades(self.config.symbol, since=window_since, limit=200)
        except Exception:
            return None
        if not isinstance(trades, list) or not trades:
            return None

        close_fills: list[dict[str, Any]] = []
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            fill_ts = self._trade_fill_timestamp_ms(trade)
            if fill_ts is not None and fill_ts + 60_000 < entry_time_ms:
                continue
            pos_side = self._trade_fill_pos_side(trade)
            if pos_side and pos_side != expected_pos_side:
                continue
            side = self._trade_fill_side(trade)
            if not side:
                continue
            trade_order_id = self._trade_fill_order_id(trade)
            if order_id and trade_order_id == order_id and not entry_fills:
                entry_fills.append(trade)
                continue
            if side == expected_close_side:
                close_fills.append(trade)

        if not close_fills:
            return None

        close_fills.sort(key=lambda item: self._trade_fill_timestamp_ms(item) or 0)
        if not entry_fills and order_id:
            for trade in trades:
                if not isinstance(trade, dict):
                    continue
                if self._trade_fill_order_id(trade) == order_id:
                    entry_fills.append(trade)

        exit_notional = 0.0
        exit_quantity = 0.0
        exit_fee = 0.0
        gross_pnl = 0.0
        close_order_ids: list[str] = []
        close_exit_time_ms: int | None = None
        for trade in close_fills:
            price = self._safe_float(trade.get("price"))
            amount = self._safe_float(trade.get("amount"))
            if price is None or amount is None or price <= 0 or amount <= 0:
                info = trade.get("info")
                if isinstance(info, dict):
                    price = self._safe_float(info.get("fillPx"))
                    amount = self._safe_float(info.get("fillSz"))
            if price is None or amount is None or price <= 0 or amount <= 0:
                continue
            exit_notional += float(price) * float(amount)
            exit_quantity += float(amount)
            exit_fee += self._trade_fill_fee_abs(trade)
            gross_pnl += self._trade_fill_realized_pnl(trade)
            order_ref = self._trade_fill_order_id(trade)
            if order_ref:
                close_order_ids.append(order_ref)
            fill_ts = self._trade_fill_timestamp_ms(trade)
            if fill_ts is not None:
                close_exit_time_ms = max(close_exit_time_ms or fill_ts, fill_ts)

        if exit_quantity <= 0:
            return None

        entry_fee = sum(self._trade_fill_fee_abs(trade) for trade in entry_fills)
        net_pnl = gross_pnl - entry_fee - exit_fee
        capital_at_entry = float(getattr(position, "capital_at_entry", 0.0) or 0.0)
        live_total = capital_at_entry + net_pnl if capital_at_entry > 0 else net_pnl
        exit_price = exit_notional / exit_quantity
        exit_time = (
            datetime.fromtimestamp(close_exit_time_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if close_exit_time_ms is not None
            else None
        )
        close_order_id = close_order_ids[-1] if close_order_ids else None
        return ExternalFlatFillClose(
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            fees=entry_fee + exit_fee,
            net_pnl=net_pnl,
            live_total=live_total,
            source="exchange_fill_sync",
            synthetic=False,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            close_order_id=close_order_id,
            close_fill_count=len(close_fills),
            entry_fill_count=len(entry_fills),
            exit_time=exit_time,
        )

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

    def _estimate_external_flat_close(self, engine: Any, position: Any) -> ExternalFlatFillClose | None:
        fill_close = self._fetch_external_flat_fill_close(position)
        if fill_close is not None:
            return fill_close
        quantity = abs(float(getattr(position, "quantity", 0.0) or 0.0))
        capital_at_entry = float(getattr(position, "capital_at_entry", 0.0) or 0.0)
        if quantity <= 0 or capital_at_entry <= 0:
            return None
        live_total = self._current_live_total_usdt(float(getattr(engine, "capital", 0.0) or capital_at_entry))
        net_pnl = live_total - capital_at_entry
        exit_price, gross_pnl, fees = self._estimate_external_exit_price(position, net_pnl)
        entry_fee = float(getattr(position, "entry_fee", 0.0) or 0.0)
        exit_fee = max(0.0, fees - entry_fee)
        return ExternalFlatFillClose(
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            fees=fees,
            net_pnl=net_pnl,
            live_total=live_total,
            source="external_flat_sync",
            synthetic=True,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
        )

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
        close = self._estimate_external_flat_close(engine, position)
        if close is None:
            return None
        exit_price = close.exit_price
        gross_pnl = close.gross_pnl
        fees = close.fees
        net_pnl = close.net_pnl
        live_total = close.live_total
        slippage_cost = float(getattr(position, "entry_slippage_cost", 0.0) or 0.0)
        risk_amount = float(getattr(position, "risk_amount", 0.0) or 0.0)
        rr_ratio = net_pnl / risk_amount if risk_amount > 0 else 0.0
        pnl_pct = net_pnl / capital_at_entry if capital_at_entry > 0 else 0.0
        reason = self._external_flat_exit_reason(position, exit_price)
        exit_timestamp = close.exit_time or timestamp
        trade = Trade(
            entry_time=str(getattr(position, "entry_time", "")),
            exit_time=exit_timestamp,
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
            candidate_event_type=getattr(position, "candidate_event_type", None),
            execution_effective_leverage=self._safe_float(getattr(position, "execution_effective_leverage", None)),
            execution_risk_mode=getattr(position, "execution_risk_mode", None),
            execution_leverage_reasons=getattr(position, "execution_leverage_reasons", None),
            execution_requested_notional=self._safe_float(getattr(position, "execution_requested_notional", None)),
            execution_target_notional=self._safe_float(getattr(position, "execution_target_notional", None)),
            execution_guard_diagnostics=getattr(position, "execution_guard_diagnostics", None),
            time_based_trailing_enabled=bool(getattr(position, "time_based_trailing_enabled", False)),
            auto_tit_reason=getattr(position, "auto_tit_reason", None),
            exit_profile=getattr(position, "exit_profile", None),
            exit_profile_reason=getattr(position, "exit_profile_reason", None),
            exit_profile_overrides=getattr(position, "exit_profile_overrides", None),
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
            last_stop_update_reason=getattr(position, "last_stop_update_reason", None),
            last_stop_update_idx=getattr(position, "last_stop_update_idx", None),
            final_stop_price=self._safe_float(getattr(position, "sl_price", None)),
        )
        engine.trades.append(trade)
        engine.exit_reasons[reason] = int(engine.exit_reasons.get(reason, 0) or 0) + 1
        engine.capital = live_total
        action = StrategyAction(
            type=ActionType.CLOSE_POSITION,
            timestamp=exit_timestamp,
            direction=getattr(position, "direction", None),
            exit_price=exit_price,
            reason=reason,
            metadata={
                "synthetic": close.synthetic,
                "source": close.source,
                "context": context,
                "gross_pnl": gross_pnl,
                "fees": fees,
                "entry_fee": close.entry_fee,
                "exit_fee": close.exit_fee,
                "slippage_cost": slippage_cost,
                "net_pnl": net_pnl,
                "signal_exit_price": exit_price,
                "capital_at_entry": capital_at_entry,
                "live_total_usdt": live_total,
                "rr_ratio": rr_ratio,
                "pnl_pct": pnl_pct,
                "index": exit_idx,
                "candidate_event_type": getattr(position, "candidate_event_type", None),
                "exit_profile": getattr(position, "exit_profile", None),
                "exit_profile_reason": getattr(position, "exit_profile_reason", None),
                "exit_profile_overrides": getattr(position, "exit_profile_overrides", None),
                "execution_effective_leverage": self._safe_float(getattr(position, "execution_effective_leverage", None)),
                "execution_risk_mode": getattr(position, "execution_risk_mode", None),
                "execution_leverage_reasons": getattr(position, "execution_leverage_reasons", None),
                "exchange_order_id": getattr(position, "exchange_order_id", None),
                "exchange_close_order_id": close.close_order_id,
                "exchange_close_fill_count": close.close_fill_count,
                "exchange_entry_fill_count": close.entry_fill_count,
            },
        )
        self.record_action(action)
        self._shadow_gate_after_close(action, engine)
        self._dynamic_high_leverage_after_close(action, engine)
        return action

    def _sync_manual_flat_position(
        self,
        engine: Any,
        *,
        context: str,
        timestamp: str | None = None,
        exit_idx: int | None = None,
    ) -> None:
        position = getattr(engine, "position", None)
        if position is None:
            return
        direction = getattr(position, "direction", None)
        timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
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
        local_position = getattr(engine, "position", None)
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
                self._sync_manual_flat_position(
                    engine,
                    context=context,
                    timestamp=timestamp,
                    exit_idx=exit_idx,
                )
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
