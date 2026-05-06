from __future__ import annotations

from typing import Any


DEFAULT_LONG_SCORE_BUCKET_SIZING_RULE = {
    "name": "bear_total_6",
    "bear_eq": 6,
    "leverage_multiplier": 1.35,
    "max_effective_leverage": 8.0,
}


def normalize_score_bucket_rules(rules: Any) -> list[dict[str, Any]]:
    if rules is None:
        return [dict(DEFAULT_LONG_SCORE_BUCKET_SIZING_RULE)]
    if isinstance(rules, dict):
        return [dict(rules)]
    if not isinstance(rules, list):
        return [dict(DEFAULT_LONG_SCORE_BUCKET_SIZING_RULE)]
    normalized = [dict(rule) for rule in rules if isinstance(rule, dict)]
    return normalized or [dict(DEFAULT_LONG_SCORE_BUCKET_SIZING_RULE)]


def _score_int(score: dict[str, Any], key: str) -> int:
    return int(score.get(key, 0) or 0)


def _matches_numeric(score: dict[str, Any], field: str, prefix: str, rule: dict[str, Any]) -> bool:
    value = _score_int(score, field)
    exact = rule.get(f"{prefix}_eq")
    min_value = rule.get(f"{prefix}_min")
    max_value = rule.get(f"{prefix}_max")
    if exact is not None and value != int(exact):
        return False
    if min_value is not None and value < int(min_value):
        return False
    if max_value is not None and value > int(max_value):
        return False
    return True


def score_bucket_rule_matches(score: dict[str, Any], rule: dict[str, Any]) -> bool:
    if not _matches_numeric(score, "net_score", "net", rule):
        return False
    if not _matches_numeric(score, "bull_total", "bull", rule):
        return False
    if not _matches_numeric(score, "bear_total", "bear", rule):
        return False

    conflict_mode = str(rule.get("conflict_mode") or "any")
    conflict = bool(score.get("conflict"))
    if conflict_mode == "conflict" and not conflict:
        return False
    if conflict_mode == "clean" and conflict:
        return False

    allowed_risk_modes = rule.get("risk_modes")
    if allowed_risk_modes:
        allowed = {str(item) for item in allowed_risk_modes if str(item)}
        if str(score.get("risk_mode") or "") not in allowed:
            return False

    return True


def apply_score_bucket_leverage(
    *,
    effective_leverage: float,
    score: dict[str, Any] | None,
    enabled: bool,
    rules: Any = None,
) -> tuple[float, dict[str, Any]]:
    base_leverage = float(effective_leverage or 0.0)
    decision: dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "source_effective_leverage": round(base_leverage, 6),
    }
    if not enabled:
        return base_leverage, decision
    if score is None:
        decision["reason"] = "missing_score"
        return base_leverage, decision
    if base_leverage <= 0.0:
        decision["reason"] = "invalid_source_leverage"
        return base_leverage, decision

    for rule in normalize_score_bucket_rules(rules):
        if not score_bucket_rule_matches(score, rule):
            continue

        target = rule.get("target_effective_leverage")
        multiplier = float(rule.get("leverage_multiplier", rule.get("multiplier", 1.0)) or 1.0)
        if target is not None:
            selected = float(target)
        else:
            selected = base_leverage * multiplier

        max_effective = rule.get("max_effective_leverage")
        if max_effective is not None:
            selected = min(selected, float(max_effective))
        selected = max(0.0, selected)

        decision.update(
            {
                "matched": True,
                "rule": rule,
                "target_effective_leverage": round(selected, 6),
                "leverage_multiplier": round(selected / base_leverage, 6) if base_leverage > 0 else 0.0,
                "score": {
                    "net_score": _score_int(score, "net_score"),
                    "bull_total": _score_int(score, "bull_total"),
                    "bear_total": _score_int(score, "bear_total"),
                    "conflict": bool(score.get("conflict")),
                    "risk_mode": score.get("risk_mode"),
                },
            }
        )
        if selected <= base_leverage + 1e-9:
            decision["reason"] = "matched_without_increase"
            return base_leverage, decision

        decision["applied"] = True
        decision["reason"] = f"score_bucket:{rule.get('name') or 'unnamed'}"
        return selected, decision

    decision["matched"] = False
    decision["reason"] = "no_matching_rule"
    return base_leverage, decision


def apply_score_bucket_sizing_to_events(
    events: list[dict[str, Any]],
    *,
    enabled: bool,
    rules: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    applied = 0
    matched = 0
    for event in events:
        updated = dict(event)
        if str(updated.get("event_type") or "") != "sota_long" or str(updated.get("direction") or "") != "BULL":
            adjusted.append(updated)
            continue

        source_leverage = float(updated.get("source_effective_leverage", 0.0) or 0.0)
        score = {
            "net_score": updated.get("net_score"),
            "bull_total": updated.get("bull_total"),
            "bear_total": updated.get("bear_total"),
            "conflict": updated.get("conflict"),
            "risk_mode": updated.get("risk_mode"),
        }
        target_leverage, decision = apply_score_bucket_leverage(
            effective_leverage=source_leverage,
            score=score,
            enabled=enabled,
            rules=rules,
        )
        if bool(decision.get("matched")):
            matched += 1
        if not bool(decision.get("applied")):
            if enabled:
                updated["long_score_bucket_sizing"] = decision
            adjusted.append(updated)
            continue

        scale = target_leverage / source_leverage if source_leverage > 0 else 1.0
        updated["return"] = float(updated.get("return", 0.0) or 0.0) * scale
        updated["return_pct"] = round(float(updated["return"]) * 100.0, 4)
        updated["source_effective_leverage"] = round(target_leverage, 6)
        updated["pre_bucket_source_effective_leverage"] = round(source_leverage, 6)
        updated["long_score_bucket_sizing"] = decision
        applied += 1
        adjusted.append(updated)

    return adjusted, {
        "enabled": bool(enabled),
        "rules": normalize_score_bucket_rules(rules) if enabled else [],
        "matched_trades": matched,
        "applied_trades": applied,
    }
