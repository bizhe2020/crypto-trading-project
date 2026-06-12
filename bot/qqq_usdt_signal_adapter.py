from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from bot.market_data import OhlcvRepository
from bot.okx_client import OkxClient
from bot.qqq_macro_proxy_overlay import apply_macro_proxy_overlay, build_macro_proxy_context, macro_proxy_overlay_for_bar
from bot.qqq_runtime_policy import filter_closed_bars
from bot.strategy_router import RoutedSignalCandidate
from scripts.fetch_public_etf_history import fetch_timeframe, output_path_for
from scripts.scan_qqq_usdt_4h_triggers import attach_daily_state, load_okx_4h, load_signal_path
from scripts.scan_qqq_usdt_btc_logic_overlays import enrich_bars
from scripts.tqqq_cash_strict_utils import load_strict_config


ROOT = Path(__file__).resolve().parents[1]


class QqqUsdtSignalAdapter:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()

    def preview(self) -> RoutedSignalCandidate:
        config = json.loads(self.config_path.read_text())
        signal_source = self._resolve_path(config["signal_source"])
        data_4h = self._resolve_path(config["data_4h"])
        signal_overrides = self._signal_overrides(config)
        signal_refresh_status: dict[str, Any] = {"enabled": False}
        refresh_status = self._refresh_okx_4h(config)
        try:
            signal_config = load_strict_config(signal_source)
            signal_refresh_status = self._refresh_daily_signal_source(config, signal_config)
            _, signal_path = load_signal_path(signal_source, overrides=signal_overrides)
        except Exception as exc:
            return self._inactive_candidate(
                config,
                reason="daily_signal_source_error",
                metadata={
                    "data_refresh": refresh_status,
                    "daily_signal_refresh": signal_refresh_status,
                    "signal_overrides": signal_overrides,
                    "error": str(exc),
                },
            )
        stale_status = self._daily_signal_stale_status(config, signal_path)
        if bool(stale_status.get("stale")):
            return self._inactive_candidate(
                config,
                reason="daily_signal_stale",
                metadata={
                    "data_refresh": refresh_status,
                    "daily_signal_refresh": signal_refresh_status,
                    "daily_signal_stale": stale_status,
                    "signal_overrides": signal_overrides,
                },
            )
        bars = enrich_bars(attach_daily_state(load_okx_4h(data_4h), signal_path, trim_to_signal_end=False))
        if bool(config.get("use_closed_execution_bars", True)):
            bars = filter_closed_bars(
                bars,
                timeframe=str(config.get("execution_timeframe", "4h")),
                grace_seconds=int(config.get("closed_bar_grace_seconds", 30) or 0),
            )
        bars = self._attach_daily_columns(bars, signal_path, trim_to_signal_end=False)
        if bars.empty:
            return self._inactive_candidate(
                config,
                reason="no_closed_bars" if bool(config.get("use_closed_execution_bars", True)) else "no_bars",
                metadata={
                    "data_refresh": refresh_status,
                    "daily_signal_refresh": signal_refresh_status,
                    "daily_signal_stale": stale_status,
                    "signal_overrides": signal_overrides,
                },
            )
        latest = bars.iloc[-1]
        allow_long = bool(latest.get("allow_long", False))
        lev_profile = self._leverage_profile(config)
        leverage_now, strength_label = self._current_leverage(lev_profile, latest)
        pre_risk_allow_long = allow_long
        pre_risk_leverage = float(leverage_now)
        pre_risk_strength_label = strength_label
        try:
            risk_overlay = self._risk_overlay_decision(config, latest)
        except Exception as exc:
            return self._inactive_candidate(
                config,
                reason="risk_overlay_error",
                metadata={
                    "data_refresh": refresh_status,
                    "daily_signal_refresh": signal_refresh_status,
                    "daily_signal_stale": stale_status,
                    "signal_overrides": signal_overrides,
                    "daily_signal_timestamp": str(pd.Timestamp(latest.get("daily_signal_timestamp", latest.get("date")))),
                    "pre_risk_allow_long": bool(pre_risk_allow_long),
                    "pre_risk_leverage": pre_risk_leverage,
                    "pre_risk_strength_label": pre_risk_strength_label,
                    "error": str(exc),
                },
            )
        if bool(risk_overlay.get("cash_gate", False)):
            allow_long = False
            leverage_now = 0.0
            strength_label = str(risk_overlay.get("strength_label") or "risk_cash_gate")
        elif allow_long:
            multiplier = float(risk_overlay.get("leverage_multiplier", 1.0) or 1.0)
            if multiplier < 1.0:
                leverage_now = round(float(leverage_now) * max(0.0, multiplier), 4)
                strength_label = f"{strength_label}_risk_cap"
        risk_adjusted_allow_long = bool(allow_long)
        risk_adjusted_leverage = float(leverage_now) if allow_long else 0.0
        try:
            macro_overlay = self._macro_proxy_overlay_decision(config, bars, latest)
        except Exception as exc:
            return self._inactive_candidate(
                config,
                reason="macro_proxy_overlay_error",
                metadata={
                    "data_refresh": refresh_status,
                    "daily_signal_refresh": signal_refresh_status,
                    "daily_signal_stale": stale_status,
                    "signal_overrides": signal_overrides,
                    "daily_signal_timestamp": str(pd.Timestamp(latest.get("daily_signal_timestamp", latest.get("date")))),
                    "pre_risk_allow_long": bool(pre_risk_allow_long),
                    "pre_risk_leverage": pre_risk_leverage,
                    "pre_risk_strength_label": pre_risk_strength_label,
                    "risk_adjusted_allow_long": bool(risk_adjusted_allow_long),
                    "risk_adjusted_leverage": float(risk_adjusted_leverage),
                    "risk_overlay": risk_overlay,
                    "error": str(exc),
                },
            )
        macro_adjusted = apply_macro_proxy_overlay(
            allow_long=risk_adjusted_allow_long,
            leverage_target=risk_adjusted_leverage,
            overlay=macro_overlay,
        )
        allow_long = bool(macro_adjusted["allow_long"])
        leverage_now = round(float(macro_adjusted["leverage_target"]), 4) if allow_long else 0.0
        if bool(macro_adjusted.get("cash_gate", False)):
            strength_label = "macro_cash_gate"
        elif bool(macro_adjusted.get("capped", False)):
            strength_label = f"{strength_label}_macro_cap"
        route_score = self._route_score(latest, leverage_now)
        return RoutedSignalCandidate(
            strategy_id="qqq_usdt_aggressive",
            symbol=str(config["execution_symbol"]),
            active=allow_long,
            route_score=route_score if allow_long else 0.0,
            timestamp=str(pd.Timestamp(latest["date"])),
            direction="BULL" if allow_long else None,
            event_type="qqq_usdt_long" if allow_long else None,
            leverage=leverage_now if allow_long else None,
            strength_label=strength_label if allow_long else "flat",
            source_config=str(self.config_path),
            metadata={
                "data_refresh": refresh_status,
                "daily_signal_refresh": signal_refresh_status,
                "daily_signal_stale": stale_status,
                "signal_overrides": signal_overrides,
                "daily_signal_timestamp": str(pd.Timestamp(latest.get("daily_signal_timestamp", latest.get("date")))),
                "entry_type": latest.get("entry_type"),
                "overlay_mode": bool(latest.get("overlay_mode", False)),
                "overlay_allocation": float(latest.get("overlay_allocation", 0.0) or 0.0),
                "rel_strength_label": latest.get("rel_strength_label"),
                "vix_label": latest.get("vix_label"),
                "ixic_trend_label": latest.get("ixic_trend_label"),
                "high_growth": bool(latest.get("high_growth", False)),
                "defense_state": bool(latest.get("defense_state", False)),
                "breakout_12": bool(latest.get("breakout_12", False)),
                "stop_loss_pct": float(config.get("stop_loss_pct", 0.0) or 0.0),
                "frozen_label": config.get("frozen_label"),
                "leverage_profile_name": config.get("leverage_profile_name"),
                "pre_risk_allow_long": bool(pre_risk_allow_long),
                "pre_risk_leverage": pre_risk_leverage,
                "pre_risk_strength_label": pre_risk_strength_label,
                "risk_adjusted_allow_long": bool(risk_adjusted_allow_long),
                "risk_adjusted_leverage": float(risk_adjusted_leverage),
                "risk_overlay": risk_overlay,
                "macro_adjusted_allow_long": bool(allow_long),
                "macro_adjusted_leverage": float(leverage_now) if allow_long else 0.0,
                "macro_proxy_overlay": macro_overlay,
            },
        )

    def _inactive_candidate(self, config: dict[str, Any], *, reason: str, metadata: dict[str, Any] | None = None) -> RoutedSignalCandidate:
        return RoutedSignalCandidate(
            strategy_id="qqq_usdt_aggressive",
            symbol=str(config["execution_symbol"]),
            active=False,
            route_score=0.0,
            direction=None,
            event_type=None,
            source_config=str(self.config_path),
            metadata={"reason": reason, **(metadata or {})},
        )

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (ROOT / value).resolve()

    @staticmethod
    def _signal_overrides(config: dict[str, Any]) -> dict[str, Any]:
        overrides = config.get("signal_overrides", {})
        if overrides is None:
            return {}
        if not isinstance(overrides, dict):
            raise TypeError("signal_overrides must be an object")
        return dict(overrides)

    def _risk_overlay_decision(self, config: dict[str, Any], latest: pd.Series) -> dict[str, Any]:
        enabled = bool(config.get("risk_overlay_enabled", False))
        payload: dict[str, Any] = {
            "enabled": enabled,
            "cash_gate": False,
            "leverage_multiplier": 1.0,
            "layers": {},
        }
        if not enabled:
            return payload

        recent = self._latest_risk_score(
            config,
            layer_name="recent",
            path_key="recent_risk_predictions_csv",
            score_column_key="recent_risk_score_column",
            default_score_column="raw_prob_10d",
            latest=latest,
        )
        payload["layers"]["recent"] = recent
        recent_threshold = float(config.get("recent_risk_cash_threshold", 0.50) or 0.50)
        recent_score = recent.get("score")
        if recent.get("available") and recent_score is not None and float(recent_score) >= recent_threshold:
            payload.update(
                {
                    "cash_gate": True,
                    "cash_gate_layer": "recent",
                    "cash_gate_reason": "recent_raw_risk_threshold",
                    "cash_gate_threshold": recent_threshold,
                    "strength_label": "recent_risk_cash_gate",
                }
            )
            return payload

        long_cycle = self._latest_risk_score(
            config,
            layer_name="long_cycle",
            path_key="long_cycle_risk_predictions_csv",
            score_column_key="long_cycle_risk_score_column",
            default_score_column="raw_prob_10d",
            latest=latest,
        )
        payload["layers"]["long_cycle"] = long_cycle
        long_score = long_cycle.get("score")
        if long_cycle.get("available") and long_score is not None:
            multiplier = self._long_cycle_risk_multiplier(config, float(long_score))
            payload["leverage_multiplier"] = multiplier
            if multiplier < 1.0:
                payload["cap_layer"] = "long_cycle"
                payload["cap_reason"] = "long_cycle_raw_risk_cap"
        return payload

    def _latest_risk_score(
        self,
        config: dict[str, Any],
        *,
        layer_name: str,
        path_key: str,
        score_column_key: str,
        default_score_column: str,
        latest: pd.Series,
    ) -> dict[str, Any]:
        configured_path = config.get(path_key)
        fail_open = bool(config.get("risk_overlay_fail_open", True))
        if not configured_path:
            if not fail_open:
                raise RuntimeError(f"{layer_name} risk prediction path is not configured: {path_key}")
            return {"enabled": False, "available": False, "reason": "missing_path"}
        score_column = str(config.get(score_column_key, default_score_column) or default_score_column)
        try:
            frame = pd.read_csv(self._resolve_path(str(configured_path)))
            if "date" not in frame.columns:
                raise ValueError("risk prediction CSV missing date column")
            if score_column not in frame.columns:
                raise ValueError(f"risk prediction CSV missing score column: {score_column}")
            frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
            frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
            frame = frame.dropna(subset=["date", score_column]).sort_values("date").reset_index(drop=True)
            if frame.empty:
                raise ValueError("risk prediction CSV has no usable rows")
            latest_date = pd.Timestamp(latest["date"]).tz_convert("UTC")
            use_previous = bool(config.get("risk_overlay_use_previous_signal", True))
            eligible = frame[frame["date"] < latest_date] if use_previous else frame[frame["date"] <= latest_date]
            if eligible.empty:
                if not fail_open:
                    raise RuntimeError(f"{layer_name} risk signal has no eligible row before latest bar: {latest_date}")
                return {
                    "enabled": True,
                    "available": False,
                    "reason": "no_signal_before_latest_bar",
                    "path": str(configured_path),
                    "score_column": score_column,
                }
            row = eligible.iloc[-1]
            stale_status = self._risk_stale_status(config, pd.Timestamp(row["date"]))
            stale = bool(stale_status.get("stale", False))
            if stale and not fail_open:
                raise RuntimeError(f"{layer_name} risk signal is stale: {stale_status}")
            return {
                "enabled": True,
                "available": not stale,
                "ignored": bool(stale and fail_open),
                "reason": "stale_fail_open" if stale and fail_open else "stale" if stale else "ok",
                "path": str(configured_path),
                "score_column": score_column,
                "score": round(float(row[score_column]), 6),
                "signal_date": str(pd.Timestamp(row["date"])),
                "latest_bar_date": str(latest_date),
                "stale": stale_status,
                "layer": layer_name,
            }
        except Exception as exc:
            if bool(config.get("risk_overlay_fail_open", True)):
                return {
                    "enabled": True,
                    "available": False,
                    "ignored": True,
                    "reason": "error_fail_open",
                    "error": str(exc),
                    "path": str(configured_path),
                    "score_column": score_column,
                }
            raise

    def _macro_proxy_overlay_decision(self, config: dict[str, Any], bars: pd.DataFrame, latest: pd.Series) -> dict[str, Any]:
        context = build_macro_proxy_context(config, bars)
        return macro_proxy_overlay_for_bar(
            config,
            context,
            pd.Timestamp(latest["date"]),
            signal_timestamp=pd.Timestamp(latest["daily_signal_timestamp"]) if pd.notna(latest.get("daily_signal_timestamp")) else None,
        )

    @staticmethod
    def _risk_stale_status(config: dict[str, Any], signal_date: pd.Timestamp) -> dict[str, Any]:
        enabled = bool(config.get("risk_overlay_stale_guard_enabled", True))
        current = pd.Timestamp.now(tz="UTC")
        signal = pd.Timestamp(signal_date).tz_convert("UTC")
        lag_days = int((current.normalize() - signal.normalize()).days)
        max_lag = int(config.get("risk_overlay_max_stale_calendar_days", 7) or 7)
        stale = enabled and lag_days > max_lag
        return {
            "enabled": enabled,
            "stale": bool(stale),
            "signal_date": str(signal),
            "now": str(current),
            "lag_days": lag_days,
            "max_stale_calendar_days": max_lag,
        }

    @staticmethod
    def _long_cycle_risk_multiplier(config: dict[str, Any], score: float) -> float:
        rules = config.get("long_cycle_risk_cap_rules") or [
            {"threshold": 0.65, "leverage_multiplier": 0.25},
            {"threshold": 0.50, "leverage_multiplier": 0.50},
            {"threshold": 0.35, "leverage_multiplier": 0.75},
        ]
        multiplier = 1.0
        for rule in sorted(rules, key=lambda item: float(item.get("threshold", 0.0)), reverse=True):
            if score >= float(rule.get("threshold", 0.0)):
                multiplier = float(rule.get("leverage_multiplier", multiplier))
                break
        return max(0.0, min(1.0, multiplier))

    def _refresh_okx_4h(self, config: dict[str, Any]) -> dict[str, Any]:
        if not bool(config.get("data_refresh_enabled", True)):
            return {"enabled": False}
        try:
            data_4h = self._resolve_path(config["data_4h"])
            repo = OhlcvRepository(data_4h.parent)
            client = OkxClient(None, trading_mode="live", proxy=config.get("proxy"))
            frame = repo.load_pair(
                str(config["execution_symbol"]),
                client=client,
                timeframe=str(config.get("execution_timeframe", "4h")),
                informative_timeframe=str(config.get("execution_timeframe", "4h")),
            ).primary_candles
            return {
                "enabled": True,
                "status": "ok",
                "rows": int(len(frame)),
                "latest": str(frame["date"].max()) if not frame.empty else None,
            }
        except Exception as exc:
            return {"enabled": True, "status": "error", "error": str(exc)}

    def _refresh_daily_signal_source(self, config: dict[str, Any], signal_config: dict[str, Any]) -> dict[str, Any]:
        if not bool(config.get("daily_signal_refresh_enabled", True)):
            return {"enabled": False}
        data_root = self._resolve_path(str(signal_config.get("data_root", "data/public/etf")))
        symbols = config.get("daily_signal_refresh_symbols") or ["QQQ", "TQQQ", "SPY", "^IXIC", "^VIX", "SQQQ"]
        timeframe = str(config.get("daily_signal_refresh_timeframe", "1d"))
        proxy = config.get("daily_signal_refresh_proxy", config.get("proxy"))
        start = str(config.get("daily_signal_refresh_start") or "2022-01-01T00:00:00Z")
        sleep_seconds = float(config.get("daily_signal_refresh_sleep_seconds", 0.25) or 0.25)
        max_attempts = max(1, int(config.get("daily_signal_refresh_max_attempts", 4) or 4))
        retry_sleep_seconds = max(0.0, float(config.get("daily_signal_refresh_retry_sleep_seconds", 1.0) or 1.0))
        fetch_timeout_seconds = max(1.0, float(config.get("daily_signal_refresh_timeout_seconds", 60.0) or 60.0))
        total_timeout_seconds = max(
            0.0,
            float(config.get("daily_signal_refresh_total_timeout_seconds", 180.0) or 180.0),
        )
        deadline = time.monotonic() + total_timeout_seconds if total_timeout_seconds > 0 else None
        fail_open = bool(config.get("daily_signal_refresh_fail_open", False))

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 qqq-usdt-router-refresh"})
        if proxy:
            session.proxies.update({"http": str(proxy), "https": str(proxy)})

        latest_by_symbol: dict[str, str | None] = {}
        attempts_by_symbol: dict[str, int] = {}
        errors: dict[str, str] = {}
        timed_out = False
        for symbol in [str(item) for item in symbols]:
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                errors[symbol] = "daily_signal_refresh_total_timeout"
                if not fail_open:
                    raise TimeoutError(errors[symbol])
                break
            last_error: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                attempts_by_symbol[symbol] = attempt
                try:
                    request_timeout = fetch_timeout_seconds
                    if deadline is not None:
                        request_timeout = max(1.0, min(fetch_timeout_seconds, deadline - time.monotonic()))
                    frame = fetch_timeframe(
                        session=session,
                        symbol=symbol,
                        timeframe=timeframe,
                        start=start,
                        end=config.get("daily_signal_refresh_end"),
                        output_path=output_path_for(data_root, symbol, timeframe),
                        sleep_seconds=sleep_seconds,
                        proxy=str(proxy) if proxy else None,
                        timeout_seconds=request_timeout,
                    )
                    latest_by_symbol[symbol] = str(frame["date"].max()) if not frame.empty else None
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < max_attempts:
                        sleep_for = retry_sleep_seconds * attempt
                        if deadline is not None:
                            sleep_for = min(sleep_for, max(0.0, deadline - time.monotonic()))
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                    if deadline is not None and time.monotonic() >= deadline:
                        timed_out = True
                        break
            if last_error is not None:
                errors[symbol] = str(last_error)
                if not fail_open:
                    raise last_error
            if timed_out:
                break
        return {
            "enabled": True,
            "status": "timeout" if timed_out else "error" if errors else "ok",
            "proxy": str(proxy) if proxy else None,
            "symbols": latest_by_symbol,
            "attempts": attempts_by_symbol,
            "errors": errors,
            "timeout_seconds": fetch_timeout_seconds,
            "total_timeout_seconds": total_timeout_seconds,
            "timed_out": timed_out,
        }

    @staticmethod
    def _daily_signal_stale_status(
        config: dict[str, Any],
        signal_path: pd.DataFrame,
        *,
        now: pd.Timestamp | None = None,
    ) -> dict[str, Any]:
        enabled = bool(config.get("daily_signal_stale_guard_enabled", True))
        if signal_path.empty or "date" not in signal_path.columns:
            return {"enabled": enabled, "stale": enabled, "reason": "empty_signal_path"}
        latest = pd.to_datetime(signal_path["date"], utc=True).max()
        current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
        lag_days = int((current.normalize() - latest.normalize()).days)
        max_lag = int(config.get("daily_signal_max_stale_calendar_days", 5) or 5)
        stale = enabled and lag_days > max_lag
        return {
            "enabled": enabled,
            "stale": bool(stale),
            "latest": str(latest),
            "now": str(current),
            "lag_days": lag_days,
            "max_stale_calendar_days": max_lag,
        }

    @staticmethod
    def _attach_daily_columns(
        bars: pd.DataFrame,
        signal_path: pd.DataFrame,
        *,
        trim_to_signal_end: bool = True,
    ) -> pd.DataFrame:
        daily = signal_path[
            [
                "date",
                "entry_type",
                "overlay_mode",
                "overlay_allocation",
                "vix_label",
                "ixic_trend_label",
                "rel_strength_label",
            ]
        ].copy()
        daily["daily_signal_timestamp"] = daily["date"]
        left = bars.drop(columns=["daily_signal_timestamp"], errors="ignore")
        merged = pd.merge_asof(
            left.sort_values("date"),
            daily.sort_values("date"),
            on="date",
            direction="backward",
            allow_exact_matches=True,
        )
        if trim_to_signal_end:
            merged = merged[merged["date"] <= daily["date"].max()].copy()
        return merged.reset_index(drop=True)

    @staticmethod
    def _leverage_profile(config: dict[str, Any]) -> dict[str, float]:
        return {
            "base": float(config.get("base_leverage", 8.0) or 8.0),
            "offense": float(config.get("offense_leverage", 10.0) or 10.0),
            "defense": float(config.get("defense_leverage", 2.0) or 2.0),
        }

    @staticmethod
    def _current_leverage(profile: dict[str, float], latest: pd.Series) -> tuple[float, str]:
        if bool(latest.get("high_growth", False)):
            return float(profile["offense"]), "offense"
        if bool(latest.get("defense_state", False)):
            return float(profile["defense"]), "defense"
        return float(profile["base"]), "base"

    @staticmethod
    def _route_score(latest: pd.Series, leverage_now: float) -> float:
        score = 40.0 + float(leverage_now) * 4.0
        if bool(latest.get("high_growth", False)):
            score += 12.0
        if bool(latest.get("defense_state", False)):
            score -= 8.0
        rel_label = str(latest.get("rel_strength_label", "") or "")
        if rel_label == "qqq_strong":
            score += 8.0
        elif rel_label == "qqq_neutral":
            score += 4.0
        if bool(latest.get("overlay_mode", False)):
            score += 6.0
        entry_type = str(latest.get("entry_type", "") or "")
        if entry_type == "recovery_reentry":
            score += 10.0
        elif entry_type == "base":
            score += 4.0
        if bool(latest.get("breakout_12", False)):
            score += 4.0
        vix_label = str(latest.get("vix_label", "") or "")
        if vix_label == "vix_low":
            score += 4.0
        elif vix_label == "vix_normal":
            score += 2.0
        return round(score, 2)
