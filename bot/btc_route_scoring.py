from __future__ import annotations

from typing import Any


def _float_value(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _nested_score(payload: dict[str, Any]) -> dict[str, Any]:
    score_gate = payload.get("sota_score_gate")
    if isinstance(score_gate, dict):
        nested = score_gate.get("score")
        if isinstance(nested, dict):
            return nested
    return {}


def _payload_value(payload: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in payload and payload.get(key) not in (None, ""):
        return payload.get(key)
    nested = _nested_score(payload)
    return nested.get(key, default)


def btc_effective_leverage(payload: dict[str, Any]) -> float:
    for key in [
        "execution_effective_leverage",
        "requested_effective_leverage",
        "source_effective_leverage",
        "leverage",
    ]:
        value = payload.get(key)
        if value not in (None, ""):
            return _float_value(value)
    sizing = payload.get("long_score_bucket_sizing")
    if isinstance(sizing, dict):
        value = sizing.get("source_effective_leverage")
        if value not in (None, ""):
            return _float_value(value)
    return 0.0


def btc_route_score(payload: dict[str, Any], *, base_score: float = 55.0) -> float:
    event_type = str(payload.get("event_type") or payload.get("candidate_event_type") or "")
    direction = str(payload.get("direction") or "")
    risk_regime = str(payload.get("risk_regime") or "")
    regime_label = str(payload.get("regime_label") or "")
    recent_sweep_status = str(payload.get("feature_recent_sweep_status") or "")

    score = float(base_score)
    score += min(max(btc_effective_leverage(payload), 0.0) * 3.5, 35.0)

    if event_type == "sota_long":
        score += 10.0
    elif event_type == "gap_smc_short_expansion":
        score += 9.0
    elif event_type == "smc_short":
        score += 7.0
    elif event_type == "smc_long":
        score += 6.0
    elif event_type:
        score += 4.0

    net_score = _float_value(_payload_value(payload, "net_score"))
    bull_total = _float_value(_payload_value(payload, "bull_total"))
    bear_total = _float_value(_payload_value(payload, "bear_total"))
    score += min(max(net_score, 0.0) * 1.6, 26.0)
    score += min(max(bull_total, 0.0) * 0.7, 14.0)
    score -= min(max(bear_total, 0.0) * 0.8, 14.0)

    if _bool_value(payload.get("conflict")):
        score -= 5.0
    if direction == "BEAR":
        score -= 2.0

    if "strong" in risk_regime:
        score += 8.0
    elif "weak" in risk_regime:
        score += 4.0
    if regime_label == "high_growth":
        score += 8.0
    elif regime_label == "normal":
        score += 4.0

    if _bool_value(payload.get("feature_recent_fvg_near_entry")):
        score += 6.0
    if recent_sweep_status == "mss_with_fvg":
        score += 7.0
    elif recent_sweep_status == "confirmed_retest":
        score += 5.0
    elif recent_sweep_status == "sweep_only":
        score += 1.0

    if _bool_value(payload.get("feature_bearish_structure")) and direction == "BULL":
        score -= 6.0
    if _bool_value(payload.get("feature_bullish_structure")) and direction == "BULL":
        score += 3.0

    return round(max(score, 0.0), 2)


def btc_strength_label(payload: dict[str, Any]) -> str:
    event_type = str(payload.get("event_type") or payload.get("candidate_event_type") or "btc")
    regime_label = str(payload.get("regime_label") or "")
    net_score = _float_value(_payload_value(payload, "net_score"))
    leverage = btc_effective_leverage(payload)
    if net_score >= 12 and leverage >= 8:
        tier = "elite"
    elif net_score >= 8 or leverage >= 6:
        tier = "strong"
    elif net_score >= 4:
        tier = "medium"
    else:
        tier = "base"
    return f"{event_type}:{regime_label or tier}:{tier}"
