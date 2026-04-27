from __future__ import annotations

import json
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
    ScalpRobustEngine,
    StrategyAction,
    StrategyConfig,
    dataframe_to_candles,
)


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
    enable_atr_trailing: bool = False
    atr_period: int = 14
    atr_activation_rr: float = 2.0
    atr_loose_multiplier: float = 2.7
    atr_normal_multiplier: float = 2.25
    atr_tight_multiplier: float = 1.8
    enable_time_based_trailing: bool = False
    T1: int = 15
    T2: int = 40
    T_max: int = 96
    S0_trigger_rr: float = 0.5
    S1_trigger_rr: float = 1.0
    S3_trigger_rr: float = 3.0
    S4_close_rr: float = 0.5
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
    enable_exchange_brackets: bool = False
    exchange_trigger_price_type: str = "mark"
    enable_manual_position_sync: bool = True
    manual_position_sync_size_tolerance_ratio: float = 0.02
    manual_position_sync_entry_price_tolerance_bps: float = 10.0
    telegram_enabled: bool = False
    telegram_token: str | None = None
    telegram_chat_id: str | None = None
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
            enable_atr_trailing=self.enable_atr_trailing,
            atr_period=self.atr_period,
            atr_activation_rr=self.atr_activation_rr,
            atr_loose_multiplier=self.atr_loose_multiplier,
            atr_normal_multiplier=self.atr_normal_multiplier,
            atr_tight_multiplier=self.atr_tight_multiplier,
            enable_time_based_trailing=self.enable_time_based_trailing,
            T1=self.T1,
            T2=self.T2,
            T_max=self.T_max,
            S0_trigger_rr=self.S0_trigger_rr,
            S1_trigger_rr=self.S1_trigger_rr,
            S3_trigger_rr=self.S3_trigger_rr,
            S4_close_rr=self.S4_close_rr,
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
        )

class OkxExecutionEngine:
    def __init__(self, config: ExecutorConfig):
        self.config = config
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
        payload = json.loads(Path(path).read_text())
        return cls(ExecutorConfig.from_dict(payload))

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

    def _send_telegram(self, message: str) -> None:
        if not self.config.telegram_enabled:
            return
        if not self.config.telegram_token or not self.config.telegram_chat_id:
            return
        url = f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage"
        try:
            requests.post(
                url,
                json={"chat_id": self.config.telegram_chat_id, "text": message},
                timeout=10,
            )
        except Exception:
            pass

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
            )
        except Exception:
            pass

    def _telegram_reply_markup(self) -> dict[str, Any]:
        return {
            "keyboard": [
                ["/daily", "/profit", "/balance"],
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
            {"command": "performance", "description": "策略表现"},
            {"command": "count", "description": "交易次数"},
            {"command": "start", "description": "恢复开仓"},
            {"command": "stop", "description": "暂停开仓"},
            {"command": "help", "description": "命令帮助"},
        ]
        try:
            requests.post(url, json={"commands": commands}, timeout=10)
        except Exception:
            pass

    def _telegram_get_updates(self) -> list[dict[str, Any]]:
        if not self.config.telegram_enabled or not self.config.telegram_token:
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
        response = requests.get(url, params=params, timeout=10)
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
        command = text.split("@", 1)[0].strip().lower()
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
                "🧾 /status table 面板版状态",
                "🚀 /performance 策略表现",
                "🔢 /count 交易次数",
                "🟢 /start 恢复开仓",
                "🛑 /stop 暂停新开仓",
            ]
        )

    def _local_time_text(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
            timestamp = str(item.get("timestamp") or "")
            if daily and not timestamp.startswith(today):
                continue
            events.append({"timestamp": timestamp, "pnl": pnl, "reason": payload.get("reason")})
        return events

    def _telegram_profit_text(self, *, daily: bool) -> str:
        events = self._realized_pnl_events(daily=daily)
        total = sum(float(event["pnl"]) for event in events)
        wins = sum(1 for event in events if float(event["pnl"]) > 0)
        title = self._telegram_title("💰", "今日收益") if daily else self._telegram_title("📈", "累计收益")
        return "\n".join(
            [
                title,
                f"💵 已实现 PnL：{total:.2f} USDT",
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
        lines = [
            self._telegram_title("🚀", "策略表现"),
            f"💎 策略资金：{capital:.2f}U",
            f"🔢 交易次数：{trade_count}",
            f"⚡ 动态档位：{dyn.get('mode') or '-'}",
            f"📊 近期单位收益：{float(recent.get('unit_return_pct', 0.0) or 0.0):.2f}%",
            f"🏆 近期胜率：{float(recent.get('win_rate_pct', 0.0) or 0.0):.1f}%",
            f"👤 Shadow资金：{float(shadow.get('capital', 0.0) or 0.0):.2f}U",
        ]
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
            self._handle_telegram_commands()

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
            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if bootstrap_status.get("bootstrap_error"):
            lines.append(f"错误: {bootstrap_status['bootstrap_error']}")
        self._send_telegram("\n".join(lines))

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
        self._assert_live_state_synced(engine, context="before_evaluate")
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
        execution_results = []
        for action in actions:
            result = self.execute_action(action, engine)
            execution_results.append({"action": asdict(action), "result": result})
        self._assert_live_state_synced(engine, context="after_execute")

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
        self._handle_telegram_commands()
        while True:
            try:
                self._handle_telegram_commands()
                wait_seconds = self.seconds_until_next_close(close_buffer_seconds)
                latest_closed_time = self.latest_closed_candle_time(close_buffer_seconds)
                last_processed = self.store.get_value("last_processed_candle_time")
                if last_processed == latest_closed_time:
                    payload = {
                        "event": "waiting",
                        "symbol": self.config.symbol,
                        "last_processed_candle_time": last_processed,
                        "next_closed_candle_time": self.next_closed_candle_time(close_buffer_seconds),
                        "sleep_seconds": wait_seconds,
                    }
                    self.store.append_action(latest_closed_time, "WAIT", payload)
                    print(json.dumps(payload, ensure_ascii=False))
                    self._sleep_with_telegram(max(wait_seconds, poll_interval_seconds), poll_interval_seconds)
                    continue

                status = self.evaluate_latest()
                print(json.dumps({"event": "evaluate", **status}, ensure_ascii=False))
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
            return shadow_decision
        if action.type == ActionType.UPDATE_STOP:
            if self.config.mode != "live" or not self.config.enable_exchange_brackets:
                return {"status": "recorded_only", "action": action.type.value, "stop_price": action.stop_price}
            return self._amend_exchange_brackets(action, engine)

        sizing = self._resolve_order_sizing(action, engine)
        if sizing.get("status") != "ok":
            return sizing
        sizing, dynamic_decision = self._dynamic_high_leverage_pre_open(action, sizing, engine)
        if dynamic_decision is not None:
            return dynamic_decision
        high_leverage_decision = self._high_leverage_guard_pre_open(action, sizing)
        if high_leverage_decision is not None:
            return high_leverage_decision

        if self.config.mode == "paper":
            if action.type in {ActionType.OPEN_LONG, ActionType.OPEN_SHORT}:
                self._shadow_gate_mark_real_position(True, action, "paper_open_accepted")
            if action.type == ActionType.CLOSE_POSITION:
                self._shadow_gate_after_close(action, engine)
                self._dynamic_high_leverage_after_close(action, engine)
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
            self._send_telegram(
                "\n".join(
                    [
                        "[开仓已确认]",
                        f"方向: {direction}",
                        f"标的: {self.config.symbol}",
                        f"成交: {observed['contracts']:.4f} 张 (~{observed['notional_usdt']:.2f}U)",
                        f"目标仓位: {sizing['amount']:.4f} {sizing['order_unit']} (~{sizing['expected_notional_usdt']:.2f}U)",
                        f"杠杆: {self.config.leverage}x",
                        f"入场: {action.entry_price:.1f}" if action.entry_price is not None else "入场: -",
                        f"止损: {action.stop_price:.1f}" if action.stop_price is not None else "止损: -",
                        f"止盈: {action.target_price:.1f}" if action.target_price is not None else "止盈: -",
                        f"订单: {order.get('id')}",
                        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    ]
                )
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
            self._shadow_gate_after_close(action, engine)
            self._dynamic_high_leverage_after_close(action, engine)
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
        leverage = float(self.config.leverage)
        maintenance_margin_pct = max(float(self.config.high_leverage_maintenance_margin_pct), 0.0) / 100.0
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
            "maintenance_margin_pct": round(float(self.config.high_leverage_maintenance_margin_pct), 6),
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
        capital_before = float(state.get("capital", 0.0) or 0.0)
        if capital_before <= 0:
            capital_before = float(getattr(engine, "capital", 0.0) or 0.0) - pnl
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

    def _sync_manual_flat_position(self, engine: Any, *, context: str) -> None:
        position = getattr(engine, "position", None)
        if position is None:
            return
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "context": context,
            "direction": getattr(position, "direction", None),
            "previous_quantity": float(getattr(position, "quantity", 0.0) or 0.0),
            "message": "Exchange position no longer exists; cleared local snapshot.",
        }
        engine.position = None
        snapshot = self._save_engine_snapshot(engine)
        payload["snapshot"] = snapshot
        self.store.append_action(timestamp, "MANUAL_POSITION_SYNC", payload)
        direction = "做多" if payload["direction"] == "BULL" else "做空" if payload["direction"] == "BEAR" else "-"
        self._send_telegram(
            "\n".join(
                [
                    "[手动平仓已同步]",
                    f"标的: {self.config.symbol}",
                    f"方向: {direction}",
                    f"来源: {context}",
                    "检测到交易所仓位已被手动平掉，本地状态已清空",
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

    def _assert_live_state_synced(self, engine: Any, *, context: str) -> None:
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
                self._sync_manual_flat_position(engine, context=context)
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
                return max(min_start, idx - 1)
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
        closed = self._floor_to_timeframe(now)
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
