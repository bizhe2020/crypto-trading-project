from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ccxt


@dataclass
class OkxCredentials:
    api_key: str
    api_secret: str
    api_passphrase: str


class OkxClient:
    def __init__(
        self,
        credentials: OkxCredentials | None,
        *,
        trading_mode: str = "paper",
        proxy: str | None = None,
    ):
        self.trading_mode = trading_mode
        self.credentials = credentials
        self.exchange = self._build_exchange(proxy=proxy)

    def _build_exchange(self, proxy: str | None = None):
        config: dict[str, Any] = {
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
        if proxy:
            config["aiohttp_proxy"] = proxy
        if self.credentials:
            config.update(
                {
                    "apiKey": self.credentials.api_key,
                    "secret": self.credentials.api_secret,
                    "password": self.credentials.api_passphrase,
                }
            )
        if self.trading_mode == "paper":
            config["options"]["sandboxMode"] = True
        exchange = ccxt.okx(config)
        if self.trading_mode == "paper":
            exchange.set_sandbox_mode(True)
        return exchange

    def load_markets(self) -> dict[str, Any]:
        return self.exchange.load_markets()

    def fetch_balance(self) -> dict[str, Any]:
        return self.exchange.fetch_balance()

    def fetch_positions(self, symbols: list[str] | None = None) -> list[dict[str, Any]]:
        return self.exchange.fetch_positions(symbols=symbols)

    def fetch_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return self.exchange.fetch_open_orders(symbol)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> list[list[float]]:
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def set_leverage(
        self,
        leverage: int,
        symbol: str,
        margin_mode: str = "isolated",
        pos_side: str | None = None,
    ) -> Any:
        params = {"marginMode": margin_mode}
        if pos_side:
            params["posSide"] = pos_side
        return self.exchange.set_leverage(leverage, symbol, params=params)

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None = None,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.exchange.create_order(symbol, order_type, side, amount, price, params or {})

    def amend_order(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.exchange.privatePostTradeAmendOrder(request)

    def amend_algo_order(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.exchange.privatePostTradeAmendAlgos(request)

    def fetch_pending_algo_orders(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.exchange.privateGetTradeOrdersAlgoPending(params or {})

    def cancel_order(self, order_id: str, symbol: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.exchange.cancel_order(order_id, symbol, params or {})
