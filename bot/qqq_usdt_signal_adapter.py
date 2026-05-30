from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from bot.market_data import OhlcvRepository
from bot.okx_client import OkxClient
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
        signal_refresh_status: dict[str, Any] = {"enabled": False}
        refresh_status = self._refresh_okx_4h(config)
        try:
            signal_config = load_strict_config(signal_source)
            signal_refresh_status = self._refresh_daily_signal_source(config, signal_config)
            _, signal_path = load_signal_path(signal_source)
        except Exception as exc:
            return self._inactive_candidate(
                config,
                reason="daily_signal_source_error",
                metadata={
                    "data_refresh": refresh_status,
                    "daily_signal_refresh": signal_refresh_status,
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
                },
            )
        bars = enrich_bars(attach_daily_state(load_okx_4h(data_4h), signal_path))
        if bool(config.get("use_closed_execution_bars", True)):
            bars = filter_closed_bars(
                bars,
                timeframe=str(config.get("execution_timeframe", "4h")),
                grace_seconds=int(config.get("closed_bar_grace_seconds", 30) or 0),
            )
        bars = self._attach_daily_columns(bars, signal_path)
        if bars.empty:
            return self._inactive_candidate(
                config,
                reason="no_closed_bars" if bool(config.get("use_closed_execution_bars", True)) else "no_bars",
                metadata={
                    "data_refresh": refresh_status,
                    "daily_signal_refresh": signal_refresh_status,
                    "daily_signal_stale": stale_status,
                },
            )

        latest = bars.iloc[-1]
        allow_long = bool(latest.get("allow_long", False))
        lev_profile = self._leverage_profile(config)
        leverage_now, strength_label = self._current_leverage(lev_profile, latest)
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
        fail_open = bool(config.get("daily_signal_refresh_fail_open", False))

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 qqq-usdt-router-refresh"})
        if proxy:
            session.proxies.update({"http": str(proxy), "https": str(proxy)})

        latest_by_symbol: dict[str, str | None] = {}
        errors: dict[str, str] = {}
        for symbol in [str(item) for item in symbols]:
            try:
                frame = fetch_timeframe(
                    session=session,
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start,
                    end=config.get("daily_signal_refresh_end"),
                    output_path=output_path_for(data_root, symbol, timeframe),
                    sleep_seconds=sleep_seconds,
                    proxy=str(proxy) if proxy else None,
                )
                latest_by_symbol[symbol] = str(frame["date"].max()) if not frame.empty else None
            except Exception as exc:
                errors[symbol] = str(exc)
                if not fail_open:
                    raise
        return {
            "enabled": True,
            "status": "error" if errors else "ok",
            "proxy": str(proxy) if proxy else None,
            "symbols": latest_by_symbol,
            "errors": errors,
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
    def _attach_daily_columns(bars: pd.DataFrame, signal_path: pd.DataFrame) -> pd.DataFrame:
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
