from __future__ import annotations

from typing import Any

import ccxt


class OkxClient:
    def __init__(
        self,
        *,
        proxy: str | None = None,
    ):
        self.exchange = self._build_exchange(proxy=proxy)

    def _build_exchange(self, proxy: str | None = None):
        config: dict[str, Any] = {
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
        if proxy:
            config["aiohttp_proxy"] = proxy
        exchange = ccxt.okx(config)
        return exchange

    def load_markets(self) -> dict[str, Any]:
        return self.exchange.load_markets()

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> list[list[float]]:
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
