"""GOOGL 高倍合约策略 — 信号适配器。

把 scripts/scan_googl_daily_signal.py 产出的日线信号
（var/runtime/googl/googl_daily_signal.csv: date,position,berkshire_conviction,
leverage_tier,target_leverage）转成 strategy_router 的 RoutedSignalCandidate。

与 QQQ 适配器不同：
- 信号源是本地价值数据两段式信号 CSV，不是 TQQQ strict 机制。
- 无 GOOGL 专属 LightGBM 风险模型（risk_overlay 关闭）。
- GOOGL-USDT-SWAP 2026-03 上线、4h 数据文件可缺省：缺省时退回日线分辨率 bars
  （信号层已验证，4h 执行层为第二阶段）。
- 杠杆档位直接来自日线信号 leverage_tier（offense/base/flat）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bot.qqq_macro_proxy_overlay import apply_macro_proxy_overlay, build_macro_proxy_context, macro_proxy_overlay_for_bar
from bot.strategy_router import RoutedSignalCandidate


ROOT = Path(__file__).resolve().parents[1]

SIGNAL_COLUMNS = ["date", "position", "berkshire_conviction", "leverage_tier", "target_leverage"]


class GooglUsdtSignalAdapter:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()

    def preview(self) -> RoutedSignalCandidate:
        config = json.loads(self.config_path.read_text())
        signal_path = self._load_signal_path(config)
        stale_status = self._stale_status(config, signal_path)
        if bool(stale_status.get("stale")):
            return self._inactive_candidate(
                config,
                reason="daily_signal_stale",
                metadata={"daily_signal_stale": stale_status},
            )
        latest = signal_path.iloc[-1]
        allow_long = bool(latest.get("position") == "GOOGL")
        leverage_tier = str(latest.get("leverage_tier") or ("flat" if not allow_long else "base"))
        leverage_now = float(latest.get("target_leverage") or 0.0) if allow_long else 0.0
        if allow_long and leverage_now <= 0.0:
            # 旧信号缺 leverage 列时按 tier 兜底
            profile = self._leverage_profile(config)
            leverage_now = float(profile.get(leverage_tier, profile["base"]))
        pre_macro_allow_long = allow_long
        pre_macro_leverage = float(leverage_now)
        pre_macro_strength_label = leverage_tier if allow_long else "flat"

        try:
            macro_overlay = self._macro_proxy_overlay_decision(config, signal_path, latest)
        except Exception as exc:
            return self._inactive_candidate(
                config,
                reason="macro_proxy_overlay_error",
                metadata={
                    "daily_signal_timestamp": str(latest.get("date")),
                    "pre_macro_allow_long": bool(pre_macro_allow_long),
                    "pre_macro_leverage": pre_macro_leverage,
                    "error": str(exc),
                },
            )
        macro_adjusted = apply_macro_proxy_overlay(
            allow_long=pre_macro_allow_long,
            leverage_target=pre_macro_leverage,
            overlay=macro_overlay,
        )
        allow_long = bool(macro_adjusted["allow_long"])
        leverage_now = round(float(macro_adjusted["leverage_target"]), 4) if allow_long else 0.0
        strength_label = leverage_tier if allow_long else "flat"
        if bool(macro_adjusted.get("cash_gate", False)):
            strength_label = "macro_cash_gate"
        elif bool(macro_adjusted.get("capped", False)):
            strength_label = f"{strength_label}_macro_cap"
        route_score = self._route_score(latest, leverage_tier, leverage_now)

        return RoutedSignalCandidate(
            strategy_id="googl_usdt_aggressive",
            symbol=str(config["execution_symbol"]),
            active=allow_long,
            route_score=route_score if allow_long else 0.0,
            timestamp=str(pd.Timestamp(latest["date"])),
            direction="BULL" if allow_long else None,
            event_type="googl_usdt_long" if allow_long else None,
            leverage=leverage_now if allow_long else None,
            strength_label=strength_label if allow_long else "flat",
            source_config=str(self.config_path),
            metadata={
                "daily_signal_timestamp": str(pd.Timestamp(latest["date"])),
                "berkshire_conviction": bool(latest.get("berkshire_conviction", False)),
                "leverage_tier": leverage_tier,
                "daily_signal_stale": stale_status,
                "pre_macro_allow_long": bool(pre_macro_allow_long),
                "pre_macro_leverage": pre_macro_leverage,
                "macro_adjusted_allow_long": bool(allow_long),
                "macro_adjusted_leverage": float(leverage_now) if allow_long else 0.0,
                "macro_proxy_overlay": macro_overlay,
                "stop_loss_pct": float(config.get("stop_loss_pct", 0.0) or 0.0),
                "frozen_label": config.get("frozen_label"),
                "leverage_profile_name": config.get("leverage_profile_name"),
            },
        )

    def _inactive_candidate(self, config: dict[str, Any], *, reason: str, metadata: dict[str, Any] | None = None) -> RoutedSignalCandidate:
        return RoutedSignalCandidate(
            strategy_id="googl_usdt_aggressive",
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

    def _load_signal_path(self, config: dict[str, Any]) -> pd.DataFrame:
        path = self._resolve_path(str(config["signal_source"]))
        if not path.exists():
            raise FileNotFoundError(f"GOOGL daily signal CSV not found: {path}")
        frame = pd.read_csv(path)
        missing = [c for c in ("date", "position") if c not in frame.columns]
        if missing:
            raise ValueError(f"GOOGL daily signal CSV missing columns: {missing}")
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if frame.empty:
            raise ValueError("GOOGL daily signal CSV has no rows")
        return frame

    @staticmethod
    def _stale_status(
        config: dict[str, Any],
        signal_path: pd.DataFrame,
        *,
        now: pd.Timestamp | None = None,
    ) -> dict[str, Any]:
        enabled = bool(config.get("daily_signal_stale_guard_enabled", True))
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

    def _macro_proxy_overlay_decision(self, config: dict[str, Any], signal_path: pd.DataFrame, latest: pd.Series) -> dict[str, Any]:
        if not bool(config.get("macro_proxy_overlay_enabled", False)):
            return {"enabled": False}
        context = build_macro_proxy_context(config, signal_path["date"])
        return macro_proxy_overlay_for_bar(
            config,
            context,
            pd.Timestamp(latest["date"]),
            signal_timestamp=pd.Timestamp(latest["date"]),
        )

    @staticmethod
    def _leverage_profile(config: dict[str, Any]) -> dict[str, float]:
        return {
            "base": float(config.get("base_leverage", 10.0) or 10.0),
            "offense": float(config.get("offense_leverage", 15.0) or 15.0),
            "defense": float(config.get("defense_leverage", 5.0) or 5.0),
        }

    @staticmethod
    def _route_score(latest: pd.Series, leverage_tier: str, leverage_now: float) -> float:
        score = 40.0 + float(leverage_now) * 4.0
        if leverage_tier == "offense":
            score += 15.0  # 信念在市，最高优先级
        elif leverage_tier == "base":
            score += 5.0
        if bool(latest.get("berkshire_conviction", False)):
            score += 8.0
        return round(score, 2)


if __name__ == "__main__":
    import sys

    config_arg = sys.argv[1] if len(sys.argv) > 1 else "config/config.paper.googl-high-leverage-runtime.json"
    adapter = GooglUsdtSignalAdapter(Path(config_arg))
    print(json.dumps(adapter.preview().to_dict(), ensure_ascii=False, indent=2))
