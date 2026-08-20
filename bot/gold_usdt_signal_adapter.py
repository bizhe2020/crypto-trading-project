"""黄金信号适配器：把 scan_gold_daily_signal.py 产出的日线信号转成 router 候选。

信号 CSV（var/runtime/gold/gold_daily_signal.csv）：
    date,position(GOLD/FLAT),leverage_tier(base/flat),target_leverage(4.0/0.0)

黄金腿（shadow 阶段只出候选不开仓）：
    - 无爬坡、无 pre_stop、无伯克希尔信念、无宏观 overlay。
    - route_score = 50（低于 GOOGL 107.8 → 只填空仓、不抢 alpha）。
    - 杠杆固定 4x。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from bot.strategy_router import RoutedSignalCandidate  # noqa: E402

STRATEGY_ID = "gold_usdt_trend"
SCORE = 50.0


class GoldUsdtSignalAdapter:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()

    def preview(self) -> RoutedSignalCandidate:
        config = json.loads(self.config_path.read_text())
        signal_path = self._load_signal_path(config)
        latest = signal_path.iloc[-1]
        allow_long = bool(latest.get("position") == "GOLD")
        leverage = float(latest.get("target_leverage") or 0.0) if allow_long else 0.0
        tier = str(latest.get("leverage_tier") or ("flat" if not allow_long else "base"))

        return RoutedSignalCandidate(
            strategy_id=STRATEGY_ID,
            symbol=str(config.get("execution_symbol", "XAU-USDT-SWAP")),
            active=allow_long,
            route_score=SCORE if allow_long else 0.0,
            timestamp=str(pd.Timestamp(latest["date"])),
            direction="BULL" if allow_long else None,
            event_type="gold_usdt_long" if allow_long else None,
            leverage=leverage if allow_long else None,
            strength_label=tier if allow_long else "flat",
            source_config=str(self.config_path),
            metadata={
                "daily_signal_timestamp": str(pd.Timestamp(latest["date"])),
                "leverage_tier": tier,
                "stop_loss_pct": float(config.get("stop_loss_pct", 5.0) or 0.0),
                "frozen_label": config.get("frozen_label"),
                "leverage_profile_name": config.get("leverage_profile_name"),
            },
        )

    def _inactive_candidate(self, config: dict[str, Any], *, reason: str) -> RoutedSignalCandidate:
        return RoutedSignalCandidate(
            strategy_id=STRATEGY_ID,
            symbol=str(config.get("execution_symbol", "XAU-USDT-SWAP")),
            active=False,
            route_score=0.0,
            direction=None,
            event_type=None,
            source_config=str(self.config_path),
            metadata={"reason": reason},
        )

    def _load_signal_path(self, config: dict[str, Any]) -> pd.DataFrame:
        path = Path(config["signal_source"])
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"gold daily signal CSV not found: {path}")
        frame = pd.read_csv(path)
        missing = [c for c in ("date", "position") if c not in frame.columns]
        if missing:
            raise ValueError(f"gold signal CSV missing columns: {missing}")
        if frame.empty:
            raise ValueError("gold signal CSV has no rows")
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        return frame.sort_values("date").reset_index(drop=True)
