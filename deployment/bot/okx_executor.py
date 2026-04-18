from __future__ import annotations

import json
import time
import uuid
import requests
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from deployment.bot.market_data import OhlcvRepository
from deployment.bot.okx_client import OkxClient, OkxCredentials
from deployment.bot.state_store import StateStore
from deployment.strategy.ema_cross_volume_core import EmaCrossVolumeConfig, EmaCrossVolumeEngine
from deployment.strategy.scalp_robust_v2_core import (
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
    data_root: str = "deployment/data/okx/futures"
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
    ema_fast_period: int = 9
    ema_slow_period: int = 21
    volume_ma_period: int = 20
    volume_multiplier: float = 1.2
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
        return cls(**payload)

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
            taker_fee_rate=self.taker_fee_rate,
            slippage_bps=self.slippage_bps,
        )

    def to_ema_strategy_config(self) -> EmaCrossVolumeConfig:
        return EmaCrossVolumeConfig(
            leverage=float(self.leverage),
            risk_per_trade=self.risk_per_trade,
            position_size_pct=self.position_size_pct,
            fixed_notional_usdt=self.fixed_notional_usdt,
            allow_long=self.allow_long,
            allow_short=self.allow_short,
            ema_fast_period=getattr(self, "ema_fast_period", 9),
            ema_slow_period=getattr(self, "ema_slow_period", 21),
            volume_ma_period=getattr(self, "volume_ma_period", 20),
            volume_multiplier=getattr(self, "volume_multiplier", 1.2),
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

        if self.config.strategy_type == "ema_cross_volume":
            engine = EmaCrossVolumeEngine.from_candles(
                primary_candles,
                informative_candles,
                self.config.to_ema_strategy_config(),
            )
            engine.restore_snapshot(self.store.load_snapshot())
            start_idx = max(self.config.ema_slow_period, self.config.volume_ma_period) + 1
            start_idx = max(start_idx, self._find_resume_index(primary_candles))
            return engine, start_idx

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
        if latest_closed_idx <= start_idx:
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

        actions = engine.evaluate_range(start_idx, latest_closed_idx)
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
        while True:
            try:
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
                    time.sleep(max(wait_seconds, poll_interval_seconds))
                    continue

                status = self.evaluate_latest()
                print(json.dumps({"event": "evaluate", **status}, ensure_ascii=False))
                time.sleep(poll_interval_seconds)
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
                time.sleep(poll_interval_seconds)

    def record_action(self, action: StrategyAction) -> None:
        self.store.append_action(action.timestamp, action.type.value, asdict(action))

    def execute_action(self, action: StrategyAction, engine: Any) -> dict[str, Any]:
        self.record_action(action)
        if action.type == ActionType.HOLD:
            return {"status": "ignored", "reason": "hold"}
        if action.type == ActionType.UPDATE_STOP:
            if self.config.mode != "live" or not self.config.enable_exchange_brackets:
                return {"status": "recorded_only", "action": action.type.value, "stop_price": action.stop_price}
            return self._amend_exchange_brackets(action, engine)

        sizing = self._resolve_order_sizing(action, engine)
        if sizing.get("status") != "ok":
            return sizing

        if self.config.mode == "paper":
            return {
                "status": "paper_recorded",
                "action": action.type.value,
                "amount": sizing.get("amount"),
                "order_unit": sizing.get("order_unit"),
                "notional_usdt": sizing.get("notional_usdt"),
                "expected_notional_usdt": sizing.get("expected_notional_usdt"),
                "balance_source": sizing.get("balance_source"),
                "position_size_pct": self.config.position_size_pct,
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
                available_usdt = float(metadata.get("available_usdt", 0.0))
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
        if self.config.strategy_type == "ema_cross_volume":
            return max(self.config.ema_slow_period, self.config.volume_ma_period) + 1
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
