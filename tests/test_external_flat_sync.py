from __future__ import annotations

import unittest
from types import SimpleNamespace

from bot.okx_executor import Direction, ExternalFlatFillClose, OkxExecutionEngine


class StubClient:
    def __init__(self, *, trades: list[dict], order_fills: list[dict]) -> None:
        self._trades = trades
        self._order_fills = order_fills

    def fetch_order_fills(self, *, inst_id: str, order_id: str, inst_type: str = "SWAP") -> list[dict]:
        return list(self._order_fills)

    def fetch_my_trades(self, symbol: str, *, since: int | None = None, limit: int | None = None, params: dict | None = None) -> list[dict]:
        return list(self._trades)

    def fetch_balance(self) -> dict:
        return {"USDT": {"total": 0.0, "free": 0.0}}


class ExternalFlatSyncTest(unittest.TestCase):
    def build_engine(self, client: StubClient) -> OkxExecutionEngine:
        engine = object.__new__(OkxExecutionEngine)
        engine.client = client
        engine.config = SimpleNamespace(symbol="BTC/USDT:USDT", taker_fee_rate=0.0005)
        engine._markets_cache = {"BTC/USDT:USDT": {"id": "BTC-USDT-SWAP"}}
        return engine

    def build_position(self) -> SimpleNamespace:
        return SimpleNamespace(
            entry_time="2026-05-19 11:01:17",
            exchange_order_id="3579790720498950144",
            direction=Direction.BEAR,
            quantity=0.9287,
            entry_price=77026.96357273609,
            capital_at_entry=1000.0,
            entry_fee=35.767470535,
        )

    def test_fetch_external_flat_fill_close_prefers_exchange_fills(self) -> None:
        client = StubClient(
            order_fills=[
                {"fee": "-20.0", "info": {"fee": "-20.0"}},
                {"fee": "-15.767470535", "info": {"fee": "-15.767470535"}},
            ],
            trades=[
                {
                    "side": "buy",
                    "price": 76957.6,
                    "amount": 0.9287,
                    "fee": {"cost": -35.73526156},
                    "info": {
                        "posSide": "short",
                        "fillPnl": "64.41795",
                        "ordId": "3580019906396315648",
                        "fillTime": "1779195307000",
                    },
                    "timestamp": 1779195307,
                }
            ],
        )
        engine = self.build_engine(client)
        close = engine._fetch_external_flat_fill_close(self.build_position())

        self.assertIsInstance(close, ExternalFlatFillClose)
        assert close is not None
        self.assertEqual(close.source, "exchange_fill_sync")
        self.assertFalse(close.synthetic)
        self.assertAlmostEqual(close.entry_fee, 35.767470535)
        self.assertAlmostEqual(close.exit_fee, 35.73526156)
        self.assertAlmostEqual(close.gross_pnl, 64.41795)
        self.assertAlmostEqual(close.net_pnl, -7.084782095)
        self.assertAlmostEqual(close.exit_price, 76957.6)
        self.assertEqual(close.close_order_id, "3580019906396315648")

    def test_estimate_external_flat_close_falls_back_to_synthetic(self) -> None:
        client = StubClient(order_fills=[], trades=[])
        engine = self.build_engine(client)
        engine._current_live_total_usdt = lambda fallback: 1028.4379084399916
        close = engine._estimate_external_flat_close(SimpleNamespace(capital=1000.0), self.build_position())

        self.assertIsInstance(close, ExternalFlatFillClose)
        assert close is not None
        self.assertEqual(close.source, "external_flat_sync")
        self.assertTrue(close.synthetic)
        self.assertAlmostEqual(close.net_pnl, 28.43790843999159)

    def test_okx_fee_formula_matches_contract_notional(self) -> None:
        contracts = 92.87
        contract_size = 0.01
        multiplier = 1.0
        price = 76957.6
        fee_rate = 0.0005
        expected = contracts * contract_size * multiplier * price * fee_rate
        quantity_btc = contracts * contract_size * multiplier
        replay_formula = quantity_btc * price * fee_rate
        self.assertAlmostEqual(expected, 35.73526156)
        self.assertAlmostEqual(replay_formula, expected)


if __name__ == "__main__":
    unittest.main()
