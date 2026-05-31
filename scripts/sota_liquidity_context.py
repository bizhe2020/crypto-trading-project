from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def build_default_scan_args() -> SimpleNamespace:
    return SimpleNamespace(
        swing_n=3,
        swing_lookback=80,
        liquidity_lookback_bars=192,
        mss_lookahead_bars=24,
        fvg_lookback_bars=8,
        entry_lookahead_bars=40,
        outcome_lookahead_bars=96,
        atr_period=14,
        min_body_atr=0.7,
        min_range_atr=1.1,
        stop_buffer_atr=0.05,
        target_rr=2.0,
        allow_incomplete_tail=True,
    )


def _empty_context() -> dict[str, Any]:
    return {
        "recent_sweep": False,
        "recent_sweep_mss": False,
        "recent_sweep_status": None,
        "recent_sweep_has_fvg": False,
        "recent_sweep_retest_confirmed": False,
        "recent_sweep_lag_bars": None,
        "recent_sweep_distance_pct": None,
        "recent_fvg_near_entry": False,
        "recent_fvg_size_pct": None,
    }


def _live_safe_sweep_context(event: Any, entry_idx: int) -> dict[str, Any]:
    lag = int(entry_idx) - int(event.sweep_idx)
    mss_available = event.mss_idx is not None and int(event.mss_idx) <= int(entry_idx)
    fvg_available = event.fvg is not None and int(event.fvg.idx) <= int(entry_idx)
    retest_available = event.retest is not None and int(event.retest.idx) <= int(entry_idx)

    status = "sweep_only"
    if mss_available:
        status = "mss_with_fvg" if fvg_available else "mss_no_fvg"
    if retest_available:
        status = "confirmed_retest" if bool(event.retest.confirmed) else "unconfirmed_retest"

    return {
        "recent_sweep": True,
        "recent_sweep_mss": bool(mss_available),
        "recent_sweep_status": status,
        "recent_sweep_has_fvg": bool(fvg_available),
        "recent_sweep_retest_confirmed": bool(retest_available and event.retest and event.retest.confirmed),
        "recent_sweep_lag_bars": int(lag),
        "recent_sweep_distance_pct": round(float(event.sweep_distance_pct), 4),
    }


def liquidity_context_for_entry(
    candles: list[Any],
    entry_idx: int,
    direction: str,
    *,
    liquidity_events: list[Any] | None = None,
    recent_sweep_lookback_bars: int = 96,
    recent_fvg_lookback_bars: int = 32,
) -> dict[str, Any]:
    from scripts.report_pa_ict_liquidity_features import recent_fvg, scan_events, zone_touched

    context = _empty_context()
    if entry_idx < 0 or entry_idx >= len(candles):
        return context

    prefix = candles[: int(entry_idx) + 1]
    events = liquidity_events if liquidity_events is not None else scan_events(prefix, build_default_scan_args())
    candidates: list[tuple[int, Any]] = []
    for event in events:
        if str(getattr(event, "direction", "")) != str(direction):
            continue
        sweep_idx = int(getattr(event, "sweep_idx", -1))
        if sweep_idx > int(entry_idx):
            continue
        lag = int(entry_idx) - sweep_idx
        if 0 <= lag <= int(recent_sweep_lookback_bars):
            candidates.append((lag, event))
    if candidates:
        _lag, event = sorted(candidates, key=lambda item: (item[0], -int(item[1].sweep_idx)))[0]
        context.update(_live_safe_sweep_context(event, int(entry_idx)))

    fvg = recent_fvg(candles, str(direction), max(2, int(entry_idx) - int(recent_fvg_lookback_bars)), int(entry_idx))
    if fvg is not None:
        entry_candle = candles[int(entry_idx)]
        context["recent_fvg_near_entry"] = bool(zone_touched(entry_candle, fvg.bottom, fvg.top))
        context["recent_fvg_size_pct"] = round(float(fvg.size_pct), 4)
    return context


def flatten_context_features(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_recent_fvg_near_entry": bool(context.get("recent_fvg_near_entry", False)),
        "feature_recent_sweep": bool(context.get("recent_sweep", False)),
        "feature_recent_sweep_mss": bool(context.get("recent_sweep_mss", False)),
        "feature_recent_sweep_status": context.get("recent_sweep_status"),
        "feature_recent_sweep_has_fvg": bool(context.get("recent_sweep_has_fvg", False)),
        "feature_recent_sweep_lag_bars": context.get("recent_sweep_lag_bars"),
    }


def annotate_sota_events_with_liquidity_context(
    candles: list[Any],
    events: list[dict[str, Any]],
    *,
    recent_sweep_lookback_bars: int = 96,
    recent_fvg_lookback_bars: int = 32,
) -> list[dict[str, Any]]:
    from scripts.report_pa_ict_liquidity_features import scan_events

    liquidity_events = scan_events(candles, build_default_scan_args())
    annotated: list[dict[str, Any]] = []
    for event in events:
        if str(event.get("event_type") or "") != "sota_long":
            annotated.append(event)
            continue
        context = liquidity_context_for_entry(
            candles,
            int(event.get("entry_idx", 0) or 0),
            str(event.get("direction") or "BULL"),
            liquidity_events=liquidity_events,
            recent_sweep_lookback_bars=recent_sweep_lookback_bars,
            recent_fvg_lookback_bars=recent_fvg_lookback_bars,
        )
        enriched = dict(event)
        enriched["sota_liquidity_context"] = context
        enriched.update(flatten_context_features(context))
        annotated.append(enriched)
    return annotated
