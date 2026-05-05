from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scripts.regime_detector import _adx_series, _atr_series, _ema, _to_thresholds


FIXED_STRUCTURE_PARAMS: dict[str, Any] = {
    "base_leverage": 4.0,
    "high_growth_leverage": 7.5,
    "tight_stop_leverage": 8.0,
    "recovery_leverage": 2.0,
    "drawdown_leverage": 2.0,
    "unhealthy_leverage": 2.0,
    "tight_stop_pct": 1.25,
    "max_stop_distance_pct": 1.5,
    "high_growth_max_stop_distance_pct": 2.0,
    "wide_stop_mode": "all_healthy",
    "max_effective_leverage": 8.0,
    "loss_streak_threshold": 3,
    "win_streak_threshold": 2,
    "drawdown_threshold_pct": 20.0,
    "health_lookback_trades": 6,
    "health_min_unit_return_pct": 0.0,
    "health_min_win_rate_pct": 25.0,
    "state_lookback_trades": 8,
    "defense_enter_unit_return_pct": -2.0,
    "defense_enter_win_rate_pct": 20.0,
    "offense_enter_unit_return_pct": -0.5,
    "offense_enter_win_rate_pct": 40.0,
    "reattack_lookback_trades": 2,
    "reattack_unit_return_pct": 0.5,
    "reattack_win_rate_pct": 33.0,
    "reattack_signal_mode": "high_growth_or_tight_or_structure",
    "price_structure_reattack_mode": "none",
    "structure_reattack_min_momentum_pct": 0.0,
    "structure_reattack_min_ema_gap_pct": 0.25,
    "structure_reattack_min_adx": 0.0,
    "defense_leverage": 2.0,
    "defense_max_stop_distance_pct": 1.5,
    "defense_structure_max_stop_distance_pct": 1.9,
    "failed_breakout_guard_enabled": True,
    "failed_breakout_guard_leverage": 2.0,
    "failed_breakout_guard_min_leverage": 7.5,
    "failed_breakout_guard_min_quality_score": 2,
    "failed_breakout_guard_min_momentum_pct": 6.0,
    "failed_breakout_guard_min_ema_gap_pct": 2.0,
    "failed_breakout_guard_min_adx": 38.0,
    "failed_breakout_guard_regime_labels": ["high_growth"],
    "failed_breakout_guard_risk_modes": ["offense"],
    "failed_breakout_guard_directions": ["BULL"],
    "min_liq_buffer_pct": 1.2,
    "maintenance_margin_pct": 0.5,
}


def regime_label_from_features(features: dict[str, Any], thresholds: Any) -> str:
    if (
        features["adx"] >= thresholds.strong_high_growth_adx_min
        and features["momentum"] >= thresholds.strong_high_growth_momentum_min
    ):
        return "high_growth"
    if features["compression_growth_score"] >= thresholds.compression_growth_score_min:
        return "high_growth"
    if (
        features["adx"] >= thresholds.high_growth_adx_min
        and features["momentum"] >= thresholds.high_growth_momentum_min
        and features["ema_gap"] >= thresholds.high_growth_ema_gap_min
        and not features["bearish_structure"]
    ):
        return "high_growth"
    if (
        features["flat_score"] >= thresholds.flat_score_min
        and features["momentum"] >= thresholds.flat_momentum_min
        and features["ema_gap"] >= thresholds.flat_ema_gap_min
    ):
        return "flat"
    if features["normal_score"] >= thresholds.normal_score_min:
        return "normal"
    return "normal"


def structure_flags_for_idx(highs: list[float], lows: list[float], end_idx: int, window: int) -> dict[str, bool]:
    window = max(int(window), 2)
    if end_idx + 1 < window * 2:
        return {"higher_high": False, "higher_low": False, "lower_high": False, "lower_low": False}
    recent_start = end_idx - window + 1
    prev_start = end_idx - window * 2 + 1
    recent_high = max(highs[recent_start : end_idx + 1])
    prev_high = max(highs[prev_start:recent_start])
    recent_low = min(lows[recent_start : end_idx + 1])
    prev_low = min(lows[prev_start:recent_start])
    return {
        "higher_high": recent_high > prev_high,
        "higher_low": recent_low > prev_low,
        "lower_high": recent_high < prev_high,
        "lower_low": recent_low < prev_low,
    }


def precompute_regime_state(
    c4h: list[Any],
    c4h_indices: list[int],
    threshold_payload: dict[str, Any] | None,
) -> tuple[dict[int, str], dict[int, dict[str, Any]]]:
    thresholds = _to_thresholds(threshold_payload)
    highs = [float(_get_value(candle, "h", 0.0) or 0.0) for candle in c4h]
    lows = [float(_get_value(candle, "l", 0.0) or 0.0) for candle in c4h]
    closes = [float(_get_value(candle, "c", 0.0) or 0.0) for candle in c4h]
    atr_values = _atr_series(highs, lows, closes, thresholds.atr_period)
    adx_values = _adx_series(highs, lows, closes, thresholds.adx_period)
    ema_fast = _ema(closes, thresholds.ema_fast_period)
    ema_slow = _ema(closes, thresholds.ema_slow_period)
    min_history = max(
        thresholds.atr_baseline_window,
        thresholds.momentum_window,
        thresholds.ema_slow_period,
        thresholds.structure_window * 2,
    )
    atr_prefix = [0.0]
    for value in atr_values:
        atr_prefix.append(atr_prefix[-1] + value)

    labels: dict[int, str] = {}
    features_by_idx: dict[int, dict[str, Any]] = {}
    for c4h_idx in c4h_indices:
        history_len = int(c4h_idx)
        if history_len < min_history:
            labels[c4h_idx] = "flat"
            features_by_idx[c4h_idx] = {}
            continue

        end_idx = history_len - 1
        atr_start = end_idx - thresholds.atr_baseline_window + 1
        atr_baseline = (atr_prefix[end_idx + 1] - atr_prefix[atr_start]) / thresholds.atr_baseline_window
        atr_now = atr_values[end_idx]
        momentum = closes[end_idx] / closes[end_idx - thresholds.momentum_window] - 1.0
        ema_gap = ema_fast[end_idx] / ema_slow[end_idx] - 1.0 if ema_slow[end_idx] != 0 else 0.0
        structure = structure_flags_for_idx(highs, lows, end_idx, thresholds.structure_window)
        trend_conflict = (momentum > 0 and ema_gap < 0) or (momentum < 0 and ema_gap > 0)
        bearish_structure = structure["lower_high"] and structure["lower_low"]
        bullish_structure = structure["higher_high"] and structure["higher_low"]
        atr_ratio = atr_now / atr_baseline if atr_baseline > 0 else 1.0
        adx = adx_values[end_idx]
        strong_growth_score = sum(
            [
                adx >= thresholds.high_growth_adx_min,
                momentum >= thresholds.high_growth_momentum_min,
                ema_gap >= thresholds.high_growth_ema_gap_min,
                bullish_structure,
            ]
        )
        compression_growth_score = sum(
            [
                adx <= thresholds.compression_growth_adx_max,
                thresholds.compression_growth_atr_ratio_min <= atr_ratio <= thresholds.compression_growth_atr_ratio_max,
                momentum >= thresholds.compression_growth_momentum_min,
                ema_gap >= thresholds.compression_growth_ema_gap_min,
            ]
        )
        flat_score = sum(
            [
                adx <= thresholds.flat_adx_max,
                atr_ratio <= thresholds.flat_atr_ratio_max,
                abs(momentum) <= thresholds.flat_momentum_abs_max,
                ema_gap >= thresholds.flat_ema_gap_min,
            ]
        )
        normal_score = sum(
            [
                momentum < thresholds.normal_momentum_max,
                adx >= thresholds.normal_adx_min,
                bearish_structure or trend_conflict,
            ]
        )
        features = {
            "atr": atr_now,
            "atr_ratio": atr_ratio,
            "adx": adx,
            "momentum": momentum,
            "ema_gap": ema_gap,
            "structure": structure,
            "bullish_structure": bullish_structure,
            "bearish_structure": bearish_structure,
            "trend_conflict": trend_conflict,
            "strong_growth_score": strong_growth_score,
            "compression_growth_score": compression_growth_score,
            "high_growth_score": max(strong_growth_score, compression_growth_score),
            "flat_score": flat_score,
            "normal_score": normal_score,
        }
        labels[c4h_idx] = regime_label_from_features(features, thresholds)
        features_by_idx[c4h_idx] = features

    return labels, features_by_idx


@dataclass
class FixedStructureState:
    capital: float
    peak: float
    loss_streak: int = 0
    win_streak: int = 0
    risk_mode: str = "offense"
    signal_health_returns: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "capital": float(self.capital),
            "peak": float(self.peak),
            "loss_streak": int(self.loss_streak),
            "win_streak": int(self.win_streak),
            "risk_mode": str(self.risk_mode or "offense"),
            "signal_health_returns": [float(value) for value in (self.signal_health_returns or [])],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None, initial_capital: float) -> "FixedStructureState":
        if not isinstance(payload, dict):
            return cls(capital=float(initial_capital), peak=float(initial_capital), signal_health_returns=[])
        capital = float(payload.get("capital", initial_capital) or initial_capital)
        peak = float(payload.get("peak", capital) or capital)
        raw_returns = payload.get("signal_health_returns")
        signal_returns = [float(value) for value in raw_returns] if isinstance(raw_returns, list) else []
        return cls(
            capital=capital,
            peak=peak,
            loss_streak=int(payload.get("loss_streak", 0) or 0),
            win_streak=int(payload.get("win_streak", 0) or 0),
            risk_mode=str(payload.get("risk_mode") or "offense"),
            signal_health_returns=signal_returns,
        )


def _get_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    getter = getattr(item, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:
            pass
    return getattr(item, key, default)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except Exception:
        return False


def configured_set(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or normalized == "all":
            return None
        return {item.strip() for item in normalized.split("+") if item.strip()}
    try:
        items = {str(item).strip() for item in value if str(item).strip()}
    except TypeError:
        return {str(value).strip()}
    return None if "all" in items else items


def high_leverage_trade_diagnostics(
    trade: Any,
    capital: float,
    leverage: float,
    maintenance_margin_pct: float,
) -> dict[str, Any]:
    entry_price = float(_get_value(trade, "entry_price", 0.0) or 0.0)
    stop_price = float(_get_value(trade, "initial_stop_price", 0.0) or 0.0)
    direction = str(_get_value(trade, "direction", "") or "")
    notional = abs(float(_get_value(trade, "notional", 0.0) or 0.0))
    if notional <= 0:
        notional = abs(float(_get_value(trade, "quantity", 0.0) or 0.0) * entry_price)
    stop_distance_pct = (
        abs(entry_price - stop_price) / entry_price * 100.0
        if entry_price > 0 and stop_price > 0
        else 0.0
    )
    maintenance = max(float(maintenance_margin_pct), 0.0) / 100.0
    liquidation_price = 0.0
    liquidation_buffer_pct = 0.0
    if entry_price > 0 and leverage > 0:
        if direction == "BULL":
            liquidation_price = entry_price * (1.0 - (1.0 / leverage) + maintenance)
            liquidation_buffer_pct = (stop_price - liquidation_price) / entry_price * 100.0
        elif direction == "BEAR":
            liquidation_price = entry_price * (1.0 + (1.0 / leverage) - maintenance)
            liquidation_buffer_pct = (liquidation_price - stop_price) / entry_price * 100.0
    account_effective_leverage = notional / float(capital) if capital > 0 else 0.0
    return {
        "entry_time": str(_get_value(trade, "entry_time", "")),
        "direction": direction,
        "entry_price": round(entry_price, 6),
        "initial_stop_price": round(stop_price, 6),
        "estimated_liquidation_price": round(liquidation_price, 6),
        "stop_distance_pct": round(stop_distance_pct, 6),
        "liquidation_buffer_pct": round(liquidation_buffer_pct, 6),
        "account_effective_leverage": round(account_effective_leverage, 6),
        "notional": round(notional, 6),
        "capital": round(float(capital), 6),
    }


def high_leverage_failures(
    diagnostics: dict[str, Any],
    min_liquidation_buffer_pct: float,
    max_stop_distance_pct: float,
    max_account_effective_leverage: float,
) -> list[str]:
    failures: list[str] = []
    if diagnostics["entry_price"] <= 0 or diagnostics["initial_stop_price"] <= 0:
        failures.append("missing_entry_or_stop")
    if min_liquidation_buffer_pct > 0 and diagnostics["liquidation_buffer_pct"] < min_liquidation_buffer_pct:
        failures.append("liquidation_buffer_too_small")
    if max_stop_distance_pct > 0 and diagnostics["stop_distance_pct"] > max_stop_distance_pct:
        failures.append("stop_distance_too_wide")
    if max_account_effective_leverage > 0 and diagnostics["account_effective_leverage"] > max_account_effective_leverage:
        failures.append("account_effective_leverage_too_high")
    return failures


def unit_trade_return(trade: Any) -> float:
    notional = abs(float(_get_value(trade, "notional", 0.0) or 0.0))
    if notional <= 0:
        entry_price = float(_get_value(trade, "entry_price", 0.0) or 0.0)
        notional = abs(float(_get_value(trade, "quantity", 0.0) or 0.0) * entry_price)
    if notional <= 0:
        return 0.0
    return float(_get_value(trade, "pnl", 0.0) or 0.0) / notional


def recent_signal_stats(signal_returns: list[float], lookback: int) -> dict[str, Any]:
    if lookback <= 0 or len(signal_returns) < lookback:
        return {
            "ready": False,
            "lookback": lookback,
            "recent_unit_return_pct": 0.0,
            "recent_win_rate_pct": 0.0,
        }
    recent = signal_returns[-lookback:]
    return {
        "ready": True,
        "lookback": lookback,
        "recent_unit_return_pct": round(sum(recent) * 100.0, 6),
        "recent_win_rate_pct": round(sum(1 for value in recent if value > 0) / len(recent) * 100.0, 6),
    }


def price_structure_qualified(trade: Any, params: dict[str, Any]) -> bool:
    direction = str(_get_value(trade, "direction", "") or "")
    momentum = float(_get_value(trade, "feature_momentum", 0.0) or 0.0)
    ema_gap = float(_get_value(trade, "feature_ema_gap", 0.0) or 0.0)
    adx = float(_get_value(trade, "feature_adx", 0.0) or 0.0)
    min_momentum = float(params.get("structure_reattack_min_momentum_pct", 0.0) or 0.0) / 100.0
    min_ema_gap = float(params.get("structure_reattack_min_ema_gap_pct", 0.0) or 0.0) / 100.0
    min_adx = float(params.get("structure_reattack_min_adx", 0.0) or 0.0)
    bullish = bool(_get_value(trade, "feature_bullish_structure", False))
    bearish = bool(_get_value(trade, "feature_bearish_structure", False))
    if adx < min_adx:
        return False
    if direction == "BULL":
        return bullish and momentum >= min_momentum and ema_gap >= min_ema_gap
    if direction == "BEAR":
        return bearish and momentum <= -min_momentum and ema_gap <= -min_ema_gap
    return False


def failed_breakout_guard(
    trade: Any,
    leverage: float,
    params: dict[str, Any],
    risk_mode: str,
) -> tuple[float, list[str], dict[str, Any]]:
    if not bool(params.get("failed_breakout_guard_enabled", False)):
        return leverage, [], {}
    if leverage < float(params.get("failed_breakout_guard_min_leverage", 7.5) or 7.5):
        return leverage, [], {}

    direction = str(_get_value(trade, "direction", "") or "")
    regime_label = str(_get_value(trade, "regime_label", "") or "")
    allowed_directions = configured_set(params.get("failed_breakout_guard_directions", ["BULL"]))
    allowed_regimes = configured_set(params.get("failed_breakout_guard_regime_labels", ["high_growth"]))
    allowed_modes = configured_set(params.get("failed_breakout_guard_risk_modes", ["offense"]))
    if allowed_directions is not None and direction not in allowed_directions:
        return leverage, [], {}
    if allowed_regimes is not None and regime_label not in allowed_regimes:
        return leverage, [], {}
    if allowed_modes is not None and risk_mode not in allowed_modes:
        return leverage, [], {}

    snapshot = quality_snapshot(event_from_trade(trade), params=params)
    quality_score = int(snapshot["quality_score"])
    min_score = int(params.get("failed_breakout_guard_min_quality_score", 2) or 0)
    diagnostics = {
        "quality_score": quality_score,
        "min_quality_score": min_score,
        "momentum_pct": snapshot["directional_momentum_pct"],
        "ema_gap_pct": snapshot["directional_ema_gap_pct"],
        "adx": snapshot["adx"],
        "checks": snapshot["checks"],
    }
    if quality_score >= min_score:
        return leverage, [], diagnostics

    guarded_leverage = min(leverage, float(params.get("failed_breakout_guard_leverage", 2.0) or 2.0))
    return guarded_leverage, [f"failed_breakout_guard:{quality_score}/{min_score}"], diagnostics


def signal_allows_reattack(trade: Any, diagnostics: dict[str, Any], params: dict[str, Any]) -> bool:
    mode = str(params.get("reattack_signal_mode", "high_growth_or_tight"))
    if mode == "any":
        return True
    regime_label = str(_get_value(trade, "regime_label", "") or "")
    trail_style = str(_get_value(trade, "trail_style", "") or "")
    tight_signal = float(diagnostics["stop_distance_pct"]) <= float(params["tight_stop_pct"]) or trail_style == "tight"
    high_growth = regime_label == "high_growth"
    if mode == "high_growth":
        return high_growth
    if mode == "tight_stop":
        return tight_signal
    structure = price_structure_qualified(trade, params)
    if mode == "structure":
        return structure
    if mode == "high_growth_or_structure":
        return high_growth or structure
    if mode == "high_growth_or_tight_or_structure":
        return high_growth or tight_signal or structure
    return high_growth or tight_signal


def signal_allows_price_structure_reattack(trade: Any, diagnostics: dict[str, Any], params: dict[str, Any]) -> bool:
    mode = str(params.get("price_structure_reattack_mode", "none"))
    if mode == "none":
        return False
    structure = price_structure_qualified(trade, params)
    if mode == "structure":
        return structure
    regime_label = str(_get_value(trade, "regime_label", "") or "")
    high_growth = regime_label == "high_growth"
    if mode == "high_growth_or_structure":
        return high_growth or structure
    trail_style = str(_get_value(trade, "trail_style", "") or "")
    tight_signal = float(diagnostics["stop_distance_pct"]) <= float(params["tight_stop_pct"]) or trail_style == "tight"
    return high_growth or tight_signal or structure


def next_risk_mode(
    trade: Any,
    diagnostics: dict[str, Any],
    current_mode: str,
    signal_returns: list[float],
    loss_streak: int,
    drawdown_pct: float,
    params: dict[str, Any],
) -> tuple[str, list[str], dict[str, Any]]:
    lookback = int(params.get("state_lookback_trades", 0) or 0)
    stats = recent_signal_stats(signal_returns, lookback)
    short_stats = recent_signal_stats(signal_returns, int(params.get("reattack_lookback_trades", 0) or 0))
    stats["reattack"] = short_stats
    reasons: list[str] = []
    if not stats["ready"]:
        return current_mode, reasons, stats

    recent_return = float(stats["recent_unit_return_pct"])
    recent_win_rate = float(stats["recent_win_rate_pct"])
    if current_mode == "offense":
        if recent_return <= float(params["defense_enter_unit_return_pct"]):
            reasons.append("low_recent_unit_return")
        if recent_win_rate <= float(params["defense_enter_win_rate_pct"]):
            reasons.append("low_recent_win_rate")
        if loss_streak >= int(params["loss_streak_threshold"]):
            reasons.append("loss_streak")
        if drawdown_pct >= float(params["drawdown_threshold_pct"]):
            reasons.append("drawdown")
        if reasons:
            return "defense", reasons, stats
        return "offense", reasons, stats

    if (
        recent_return >= float(params["offense_enter_unit_return_pct"])
        and recent_win_rate >= float(params["offense_enter_win_rate_pct"])
        and loss_streak < int(params["loss_streak_threshold"])
        and drawdown_pct < float(params["drawdown_threshold_pct"])
    ):
        reasons.append("recovered_recent_signal")
        return "offense", reasons, stats
    if short_stats["ready"]:
        short_return = float(short_stats["recent_unit_return_pct"])
        short_win_rate = float(short_stats["recent_win_rate_pct"])
        if (
            short_return >= float(params["reattack_unit_return_pct"])
            and short_win_rate >= float(params["reattack_win_rate_pct"])
            and loss_streak < int(params["loss_streak_threshold"])
            and drawdown_pct < float(params["drawdown_threshold_pct"])
            and signal_allows_reattack(trade, diagnostics, params)
        ):
            reasons.append("short_window_reattack")
            return "offense", reasons, stats
    if (
        loss_streak < int(params["loss_streak_threshold"])
        and drawdown_pct < float(params["drawdown_threshold_pct"])
        and signal_allows_price_structure_reattack(trade, diagnostics, params)
    ):
        reasons.append("price_structure_reattack")
        return "offense", reasons, stats
    return "defense", reasons, stats


def dynamic_stop_distance_cap(
    trade: Any,
    drawdown_pct: float,
    loss_streak: int,
    win_streak: int,
    market_healthy: bool,
    params: dict[str, Any],
) -> float:
    base_cap = float(params["max_stop_distance_pct"])
    high_growth_cap = float(params.get("high_growth_max_stop_distance_pct", base_cap))
    regime_label = str(_get_value(trade, "regime_label", "") or "")
    mode = str(params.get("wide_stop_mode", "high_growth"))
    healthy = (
        market_healthy
        and drawdown_pct < float(params["drawdown_threshold_pct"])
        and loss_streak < int(params["loss_streak_threshold"])
    )
    if mode == "all_healthy" and healthy:
        return max(base_cap, high_growth_cap)
    if mode == "healthy" and healthy and (
        regime_label in {"high_growth", "normal"} or win_streak >= int(params["win_streak_threshold"])
    ):
        return max(base_cap, high_growth_cap)
    if mode == "high_growth" and healthy and regime_label == "high_growth":
        return max(base_cap, high_growth_cap)
    return base_cap


def is_market_healthy(signal_returns: list[float], params: dict[str, Any]) -> bool:
    lookback = int(params.get("health_lookback_trades", 0) or 0)
    if lookback <= 0 or len(signal_returns) < lookback:
        return True
    recent = signal_returns[-lookback:]
    min_return = float(params.get("health_min_unit_return_pct", 0.0) or 0.0) / 100.0
    min_win_rate = float(params.get("health_min_win_rate_pct", 0.0) or 0.0) / 100.0
    recent_return = sum(recent)
    recent_win_rate = sum(1 for value in recent if value > 0) / len(recent)
    return recent_return >= min_return and recent_win_rate >= min_win_rate


def select_effective_leverage(
    trade: Any,
    diagnostics: dict[str, Any],
    params: dict[str, Any],
    loss_streak: int,
    win_streak: int,
    drawdown_pct: float,
    market_healthy: bool,
    risk_mode: str,
) -> tuple[float, list[str]]:
    leverage = float(params["base_leverage"])
    reasons = ["base"]
    regime_label = str(_get_value(trade, "regime_label", "") or "")
    trail_style = str(_get_value(trade, "trail_style", "") or "")
    stop_distance_pct = float(diagnostics["stop_distance_pct"])

    if regime_label == "high_growth":
        leverage = max(leverage, float(params["high_growth_leverage"]))
        reasons.append("high_growth")
    if stop_distance_pct <= float(params["tight_stop_pct"]):
        leverage = max(leverage, float(params["tight_stop_leverage"]))
        reasons.append("tight_stop")
    if trail_style == "tight":
        leverage = max(leverage, float(params["tight_stop_leverage"]))
        reasons.append("tight_trail")
    if win_streak >= int(params["win_streak_threshold"]):
        leverage = min(float(params["max_effective_leverage"]), leverage * 1.15)
        reasons.append("win_streak_expand")
    if loss_streak >= int(params["loss_streak_threshold"]):
        leverage = min(leverage, float(params["recovery_leverage"]))
        reasons.append("loss_streak_reduce")
    if drawdown_pct >= float(params["drawdown_threshold_pct"]):
        leverage = min(leverage, float(params["drawdown_leverage"]))
        reasons.append("drawdown_reduce")
    if not market_healthy:
        leverage = min(leverage, float(params["unhealthy_leverage"]))
        reasons.append("market_unhealthy_reduce")
    if risk_mode == "defense":
        leverage = min(leverage, float(params["defense_leverage"]))
        reasons.append("state_defense_reduce")
    leverage, guard_reasons, _guard_diagnostics = failed_breakout_guard(
        trade=trade,
        leverage=leverage,
        params=params,
        risk_mode=risk_mode,
    )
    reasons.extend(guard_reasons)
    leverage = max(0.0, min(leverage, float(params["max_effective_leverage"])))
    return leverage, reasons


def fixed_structure_step(
    trade: Any,
    state: FixedStructureState,
    params: dict[str, Any] | None = None,
) -> tuple[FixedStructureState, dict[str, Any] | None, dict[str, Any]]:
    active_params = dict(params or FIXED_STRUCTURE_PARAMS)
    signal_returns = [float(value) for value in (state.signal_health_returns or [])]
    diagnostics = high_leverage_trade_diagnostics(
        trade,
        capital=float(state.capital),
        leverage=10.0,
        maintenance_margin_pct=float(active_params["maintenance_margin_pct"]),
    )
    drawdown_pct = (state.peak - state.capital) / state.peak * 100.0 if state.peak > 0 else 0.0
    market_healthy = is_market_healthy(signal_returns, active_params)
    risk_mode, mode_reasons, mode_stats = next_risk_mode(
        trade,
        diagnostics,
        str(state.risk_mode or "offense"),
        signal_returns,
        loss_streak=int(state.loss_streak),
        drawdown_pct=drawdown_pct,
        params=active_params,
    )
    max_stop_distance_pct = dynamic_stop_distance_cap(
        trade=trade,
        drawdown_pct=drawdown_pct,
        loss_streak=int(state.loss_streak),
        win_streak=int(state.win_streak),
        market_healthy=market_healthy,
        params=active_params,
    )
    if risk_mode == "defense":
        max_stop_distance_pct = min(max_stop_distance_pct, float(active_params["defense_max_stop_distance_pct"]))
        if price_structure_qualified(trade, active_params):
            max_stop_distance_pct = max(
                max_stop_distance_pct,
                float(active_params["defense_structure_max_stop_distance_pct"]),
            )
    signal_unit_return = unit_trade_return(trade)
    signal_returns.append(signal_unit_return)
    failures = high_leverage_failures(
        diagnostics,
        min_liquidation_buffer_pct=float(active_params["min_liq_buffer_pct"]),
        max_stop_distance_pct=max_stop_distance_pct,
        max_account_effective_leverage=0.0,
    )
    decision = {
        "diagnostics": diagnostics,
        "drawdown_pct": round(drawdown_pct, 6),
        "market_healthy": market_healthy,
        "risk_mode": risk_mode,
        "mode_reasons": mode_reasons,
        "mode_stats": mode_stats,
        "stop_distance_cap_pct": max_stop_distance_pct,
        "signal_return": signal_unit_return,
        "failures": failures,
    }
    if failures:
        next_state = FixedStructureState(
            capital=float(state.capital),
            peak=float(state.peak),
            loss_streak=int(state.loss_streak),
            win_streak=int(state.win_streak),
            risk_mode=risk_mode,
            signal_health_returns=signal_returns[-100:],
        )
        return next_state, None, decision

    effective_leverage, reasons = select_effective_leverage(
        trade,
        diagnostics,
        active_params,
        loss_streak=int(state.loss_streak),
        win_streak=int(state.win_streak),
        drawdown_pct=drawdown_pct,
        market_healthy=market_healthy,
        risk_mode=risk_mode,
    )
    _unused_guard_leverage, _unused_guard_reasons, guard_diagnostics = failed_breakout_guard(
        trade=trade,
        leverage=float(active_params.get("failed_breakout_guard_min_leverage", 7.5) or 7.5),
        params=active_params,
        risk_mode=risk_mode,
    )
    trade_return = signal_unit_return * effective_leverage
    capital_before = float(state.capital)
    capital = max(0.0, capital_before * (1.0 + trade_return))
    peak = max(float(state.peak), capital)
    if capital > capital_before:
        win_streak = int(state.win_streak) + 1
        loss_streak = 0
    else:
        loss_streak = int(state.loss_streak) + 1
        win_streak = 0

    event = {
        "entry_time": str(_get_value(trade, "entry_time", "")),
        "exit_time": str(_get_value(trade, "exit_time", "")),
        "entry_idx": None if _is_missing(_get_value(trade, "entry_idx")) else int(_get_value(trade, "entry_idx")),
        "exit_idx": None if _is_missing(_get_value(trade, "exit_idx")) else int(_get_value(trade, "exit_idx")),
        "exit_reason": str(_get_value(trade, "exit_reason", "") or ""),
        "rr_ratio": float(_get_value(trade, "rr_ratio", 0.0) or 0.0),
        "signal_return": signal_unit_return,
        "return": trade_return,
        "capital": capital,
        "effective_leverage": effective_leverage,
        "regime_label": str(_get_value(trade, "regime_label", "") or ""),
        "trail_style": str(_get_value(trade, "trail_style", "") or ""),
        "direction": str(_get_value(trade, "direction", "") or ""),
        "entry_price": float(_get_value(trade, "entry_price", 0.0) or 0.0),
        "exit_price": float(_get_value(trade, "exit_price", 0.0) or 0.0),
        "initial_stop_price": float(_get_value(trade, "initial_stop_price", 0.0) or 0.0),
        "pressure_target_applied": bool(_get_value(trade, "pressure_target_applied", False)),
        "pressure_target_source": str(_get_value(trade, "pressure_target_source", "") or ""),
        "pressure_target_level": None if _is_missing(_get_value(trade, "pressure_target_level")) else float(_get_value(trade, "pressure_target_level")),
        "pressure_target_rr": None if _is_missing(_get_value(trade, "pressure_target_rr")) else float(_get_value(trade, "pressure_target_rr")),
        "pressure_target_min_rr": None if _is_missing(_get_value(trade, "pressure_target_min_rr")) else float(_get_value(trade, "pressure_target_min_rr")),
        "pressure_target_dynamic_reason": str(_get_value(trade, "pressure_target_dynamic_reason", "") or ""),
        "pressure_target_update_idx": None if _is_missing(_get_value(trade, "pressure_target_update_idx")) else int(_get_value(trade, "pressure_target_update_idx")),
        "pressure_touch_lock_applied": bool(_get_value(trade, "pressure_touch_lock_applied", False)),
        "pressure_touch_lock_source": str(_get_value(trade, "pressure_touch_lock_source", "") or ""),
        "pressure_touch_lock_level": None if _is_missing(_get_value(trade, "pressure_touch_lock_level")) else float(_get_value(trade, "pressure_touch_lock_level")),
        "pressure_touch_lock_rr": None if _is_missing(_get_value(trade, "pressure_touch_lock_rr")) else float(_get_value(trade, "pressure_touch_lock_rr")),
        "pressure_touch_lock_update_idx": None if _is_missing(_get_value(trade, "pressure_touch_lock_update_idx")) else int(_get_value(trade, "pressure_touch_lock_update_idx")),
        "feature_adx": float(_get_value(trade, "feature_adx", 0.0) or 0.0),
        "feature_momentum": float(_get_value(trade, "feature_momentum", 0.0) or 0.0),
        "feature_ema_gap": float(_get_value(trade, "feature_ema_gap", 0.0) or 0.0),
        "feature_bullish_structure": bool(_get_value(trade, "feature_bullish_structure", False)),
        "feature_bearish_structure": bool(_get_value(trade, "feature_bearish_structure", False)),
        "stop_distance_pct": float(diagnostics["stop_distance_pct"]),
        "stop_distance_cap_pct": max_stop_distance_pct,
        "reasons": reasons,
        "failed_breakout_guard_applied": any(str(reason).startswith("failed_breakout_guard") for reason in reasons),
        "failed_breakout_guard_diagnostics": guard_diagnostics,
        "market_healthy": market_healthy,
        "risk_mode": risk_mode,
        "risk_mode_stats": mode_stats,
    }
    next_state = FixedStructureState(
        capital=capital,
        peak=peak,
        loss_streak=loss_streak,
        win_streak=win_streak,
        risk_mode=risk_mode,
        signal_health_returns=signal_returns[-100:],
    )
    decision.update(
        {
            "effective_leverage": effective_leverage,
            "leverage_reasons": reasons,
            "accepted_return": trade_return,
        }
    )
    return next_state, event, decision


def fixed_structure_entry_decision(
    trade: Any,
    state: FixedStructureState,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_params = dict(params or FIXED_STRUCTURE_PARAMS)
    signal_returns = [float(value) for value in (state.signal_health_returns or [])]
    diagnostics = high_leverage_trade_diagnostics(
        trade,
        capital=float(state.capital),
        leverage=10.0,
        maintenance_margin_pct=float(active_params["maintenance_margin_pct"]),
    )
    drawdown_pct = (state.peak - state.capital) / state.peak * 100.0 if state.peak > 0 else 0.0
    market_healthy = is_market_healthy(signal_returns, active_params)
    risk_mode, mode_reasons, mode_stats = next_risk_mode(
        trade,
        diagnostics,
        str(state.risk_mode or "offense"),
        signal_returns,
        loss_streak=int(state.loss_streak),
        drawdown_pct=drawdown_pct,
        params=active_params,
    )
    max_stop_distance_pct = dynamic_stop_distance_cap(
        trade=trade,
        drawdown_pct=drawdown_pct,
        loss_streak=int(state.loss_streak),
        win_streak=int(state.win_streak),
        market_healthy=market_healthy,
        params=active_params,
    )
    if risk_mode == "defense":
        max_stop_distance_pct = min(max_stop_distance_pct, float(active_params["defense_max_stop_distance_pct"]))
        if price_structure_qualified(trade, active_params):
            max_stop_distance_pct = max(
                max_stop_distance_pct,
                float(active_params["defense_structure_max_stop_distance_pct"]),
            )
    failures = high_leverage_failures(
        diagnostics,
        min_liquidation_buffer_pct=float(active_params["min_liq_buffer_pct"]),
        max_stop_distance_pct=max_stop_distance_pct,
        max_account_effective_leverage=0.0,
    )
    effective_leverage = 0.0
    leverage_reasons: list[str] = []
    guard_diagnostics: dict[str, Any] = {}
    if not failures:
        effective_leverage, leverage_reasons = select_effective_leverage(
            trade,
            diagnostics,
            active_params,
            loss_streak=int(state.loss_streak),
            win_streak=int(state.win_streak),
            drawdown_pct=drawdown_pct,
            market_healthy=market_healthy,
            risk_mode=risk_mode,
        )
        _unused_guard_leverage, _unused_guard_reasons, guard_diagnostics = failed_breakout_guard(
            trade=trade,
            leverage=float(active_params.get("failed_breakout_guard_min_leverage", 7.5) or 7.5),
            params=active_params,
            risk_mode=risk_mode,
        )
    return {
        "accepted": not failures,
        "failures": failures,
        "diagnostics": diagnostics,
        "drawdown_pct": round(drawdown_pct, 6),
        "market_healthy": market_healthy,
        "risk_mode": risk_mode,
        "mode_reasons": mode_reasons,
        "mode_stats": mode_stats,
        "stop_distance_cap_pct": max_stop_distance_pct,
        "effective_leverage": effective_leverage,
        "leverage_reasons": leverage_reasons,
        "failed_breakout_guard_applied": any(str(reason).startswith("failed_breakout_guard") for reason in leverage_reasons),
        "failed_breakout_guard_diagnostics": guard_diagnostics,
    }


def event_from_trade(trade: Any) -> dict[str, Any]:
    return {
        "direction": str(_get_value(trade, "direction", "") or ""),
        "feature_adx": float(_get_value(trade, "feature_adx", 0.0) or 0.0),
        "feature_momentum": float(_get_value(trade, "feature_momentum", 0.0) or 0.0),
        "feature_ema_gap": float(_get_value(trade, "feature_ema_gap", 0.0) or 0.0),
        "feature_bullish_structure": bool(_get_value(trade, "feature_bullish_structure", False)),
        "feature_bearish_structure": bool(_get_value(trade, "feature_bearish_structure", False)),
    }


def quality_snapshot(event: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
    active_params = params or FIXED_STRUCTURE_PARAMS
    direction = str(event.get("direction") or "")
    sign = 1.0 if direction == "BULL" else -1.0
    momentum_pct = float(event.get("feature_momentum", 0.0) or 0.0) * 100.0 * sign
    ema_gap_pct = float(event.get("feature_ema_gap", 0.0) or 0.0) * 100.0 * sign
    adx = float(event.get("feature_adx", 0.0) or 0.0)
    structure_ok = (
        bool(event.get("feature_bullish_structure"))
        if direction == "BULL"
        else bool(event.get("feature_bearish_structure"))
    )
    checks = {
        "momentum": momentum_pct >= float(active_params["failed_breakout_guard_min_momentum_pct"]),
        "ema_gap": ema_gap_pct >= float(active_params["failed_breakout_guard_min_ema_gap_pct"]),
        "adx": adx >= float(active_params["failed_breakout_guard_min_adx"]),
        "structure": structure_ok,
    }
    return {
        "quality_score": sum(1 for passed in checks.values() if passed),
        "directional_momentum_pct": round(momentum_pct, 6),
        "directional_ema_gap_pct": round(ema_gap_pct, 6),
        "adx": round(adx, 6),
        "checks": checks,
    }


def selected_by(event: dict[str, Any], selector: str, max_quality_score: int) -> bool:
    direction = str(event.get("direction") or "")
    exit_reason = str(event.get("exit_reason") or "")
    if selector != "guarded_weak_loss":
        raise ValueError(f"Unsupported selector: {selector}")
    return (
        direction == "BULL"
        and str(event.get("regime_label") or "") == "high_growth"
        and str(event.get("risk_mode") or "") == "offense"
        and exit_reason == "stop_loss"
        and float(event.get("return", 0.0) or 0.0) < 0.0
        and bool(event.get("failed_breakout_guard_applied"))
        and int(quality_snapshot(event)["quality_score"]) <= max_quality_score
    )
