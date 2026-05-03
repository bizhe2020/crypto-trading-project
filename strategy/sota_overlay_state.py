from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SOTA_EVENT_TYPES = {"sota", "sota_long", "sota_short", "main_sota", "scalp_robust_v2"}
STABLE_EVENT_TYPES = {"stable", "stable_reverse_short"}
SMC_EVENT_TYPES = {"smc", "smc_short"}

EVENT_PRIORITY = {
    **{event_type: 0 for event_type in SOTA_EVENT_TYPES},
    **{event_type: 1 for event_type in STABLE_EVENT_TYPES},
    **{event_type: 2 for event_type in SMC_EVENT_TYPES},
}


@dataclass(frozen=True)
class OverlayCandidate:
    event_type: str
    direction: str | None = None
    entry_idx: int | None = None
    exit_idx: int | None = None
    entry_time: str | None = None
    exit_time: str | None = None
    return_rate: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        left = "" if self.entry_idx is None else str(self.entry_idx)
        right = "" if self.exit_idx is None else str(self.exit_idx)
        return f"{self.event_type}|{left}|{right}"


def roundtrip_cost_rate(taker_fee_rate: float, slippage_bps: float) -> float:
    return 2.0 * float(taker_fee_rate) + 2.0 * float(slippage_bps) / 10_000.0


def leveraged_net_return(
    *,
    signal_return_pct: float,
    leverage: float,
    position_size_pct: float,
    allocation: float = 1.0,
    taker_fee_rate: float,
    slippage_bps: float,
) -> dict[str, float]:
    gross_unit_return = float(signal_return_pct) / 100.0
    cost = roundtrip_cost_rate(taker_fee_rate, slippage_bps)
    net_unit_return = gross_unit_return - cost
    account_return = net_unit_return * float(leverage) * float(position_size_pct) * float(allocation)
    return {
        "gross_unit_return": gross_unit_return,
        "roundtrip_cost": cost,
        "net_unit_return": net_unit_return,
        "account_return": account_return,
        "gross_unit_return_pct": gross_unit_return * 100.0,
        "roundtrip_cost_pct": cost * 100.0,
        "net_unit_return_pct": net_unit_return * 100.0,
        "account_return_pct": account_return * 100.0,
    }


def event_priority(event_type: str | None) -> int:
    return EVENT_PRIORITY.get(str(event_type or "").lower(), 9)


def is_sota_event(event_type: str | None) -> bool:
    return str(event_type or "").lower() in SOTA_EVENT_TYPES


def is_stable_event(event_type: str | None) -> bool:
    return str(event_type or "").lower() in STABLE_EVENT_TYPES


def candidate_from_action(action: Any, *, default_event_type: str | None = None) -> OverlayCandidate:
    metadata = getattr(action, "metadata", None) or {}
    event_type = (
        metadata.get("overlay_event_type")
        or metadata.get("event_type")
        or default_event_type
        or _default_event_type_for_action(action)
    )
    return OverlayCandidate(
        event_type=str(event_type),
        direction=getattr(action, "direction", None),
        entry_idx=_optional_int(metadata.get("index") or metadata.get("entry_idx")),
        exit_idx=_optional_int(metadata.get("exit_idx")),
        entry_time=getattr(action, "timestamp", None),
        exit_time=metadata.get("exit_time"),
        metadata=dict(metadata),
    )


def account_lock_decision(
    candidate: OverlayCandidate,
    *,
    local_position_open: bool,
    exchange_long_contracts: float = 0.0,
    exchange_short_contracts: float = 0.0,
    blocking_candidate: OverlayCandidate | None = None,
) -> dict[str, Any]:
    decision = _base_decision(candidate)
    long_contracts = abs(float(exchange_long_contracts or 0.0))
    short_contracts = abs(float(exchange_short_contracts or 0.0))
    if long_contracts > 0.0 or short_contracts > 0.0:
        return _with_blocking_candidate(
            decision | {
                "decision": "rejected",
                "reason": "account_position_open",
                "exchange_long_contracts": long_contracts,
                "exchange_short_contracts": short_contracts,
                "paper_tag": "account_lock_rejected",
            },
            candidate,
            blocking_candidate,
            default_tag="account_lock_rejected",
        )
    if local_position_open:
        return _with_blocking_candidate(
            decision | {
                "decision": "rejected",
                "reason": "local_position_open",
                "paper_tag": "single_position_lock_rejected",
            },
            candidate,
            blocking_candidate,
            default_tag="single_position_lock_rejected",
        )
    return decision | {
        "decision": "accepted",
        "reason": "priority_available",
        "paper_tag": f"accepted_{candidate.event_type}",
    }


def replay_single_position_events(candidates: list[OverlayCandidate]) -> tuple[list[OverlayCandidate], list[dict[str, Any]]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            int(item.entry_idx or 0),
            event_priority(item.event_type),
            int(item.exit_idx or item.entry_idx or 0),
        ),
    )
    accepted: list[OverlayCandidate] = []
    decisions: list[dict[str, Any]] = []
    active_until_idx = -1
    active_event: OverlayCandidate | None = None

    for candidate in ordered:
        entry_idx = int(candidate.entry_idx or 0)
        if active_event is not None and entry_idx < active_until_idx:
            decision = _base_decision(candidate) | {
                "decision": "rejected",
                "reason": "position_lock_open",
                "blocking_event_key": active_event.key(),
                "blocking_event_type": active_event.event_type,
                "blocking_exit_idx": int(active_event.exit_idx or active_until_idx),
                "blocking_exit_time": active_event.exit_time,
                "paper_tag": "position_lock_rejected",
            }
            if is_sota_event(candidate.event_type) and is_stable_event(active_event.event_type):
                decision["paper_tag"] = "stable_preempted_sota"
            decisions.append(decision)
            continue

        accepted.append(candidate)
        active_event = candidate
        active_until_idx = max(entry_idx, int(candidate.exit_idx or entry_idx))
        decisions.append(
            _base_decision(candidate)
            | {
                "decision": "accepted",
                "reason": "priority_available",
                "paper_tag": f"accepted_{candidate.event_type}",
            }
        )

    return accepted, decisions


def _base_decision(candidate: OverlayCandidate) -> dict[str, Any]:
    return {
        "event_key": candidate.key(),
        "event_type": candidate.event_type,
        "entry_idx": candidate.entry_idx,
        "exit_idx": candidate.exit_idx,
        "entry_time": candidate.entry_time,
        "exit_time": candidate.exit_time,
        "direction": candidate.direction,
        "return_pct": round(float(candidate.return_rate or 0.0) * 100.0, 4),
        "priority": event_priority(candidate.event_type),
    }


def _with_blocking_candidate(
    decision: dict[str, Any],
    candidate: OverlayCandidate,
    blocking_candidate: OverlayCandidate | None,
    *,
    default_tag: str,
) -> dict[str, Any]:
    if blocking_candidate is None:
        return decision
    decision |= {
        "blocking_event_key": blocking_candidate.key(),
        "blocking_event_type": blocking_candidate.event_type,
        "blocking_entry_idx": blocking_candidate.entry_idx,
        "blocking_exit_idx": blocking_candidate.exit_idx,
        "blocking_entry_time": blocking_candidate.entry_time,
        "blocking_exit_time": blocking_candidate.exit_time,
    }
    if is_sota_event(candidate.event_type) and is_stable_event(blocking_candidate.event_type):
        decision["paper_tag"] = "stable_preempted_sota"
    else:
        decision["paper_tag"] = default_tag
    return decision


def _default_event_type_for_action(action: Any) -> str:
    action_type = str(getattr(action, "type", "") or "")
    if action_type.endswith("OPEN_SHORT"):
        return "sota_short"
    return "sota_long"


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
