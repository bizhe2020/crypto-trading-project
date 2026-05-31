from __future__ import annotations

from typing import Any


def apply_sota_structure_gate(
    sota_events: list[dict[str, Any]],
    *,
    enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not enabled:
        return sota_events, {
            "enabled": False,
            "rule": {
                "require_non_bearish_structure_for_long": False,
            },
            "original_candidates": len(sota_events),
            "filtered_candidates": len(sota_events),
            "removed_candidates": 0,
        }

    filtered: list[dict[str, Any]] = []
    removed = 0
    for event in sota_events:
        if bool(event.get("feature_bearish_structure", False)):
            removed += 1
            continue
        filtered.append(event)
    return filtered, {
        "enabled": True,
        "rule": {
            "require_non_bearish_structure_for_long": True,
        },
        "original_candidates": len(sota_events),
        "filtered_candidates": len(filtered),
        "removed_candidates": removed,
    }
