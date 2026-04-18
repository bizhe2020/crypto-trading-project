from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import websocket

from deployment.strategy.obi_trailing import OrderBookLevel, OrderBookSnapshot


class OkxOrderBookFeed:
    def __init__(self, inst_id: str = "BTC-USDT-SWAP", channel: str = "books5"):
        self.inst_id = inst_id
        self.channel = channel
        self.url = "wss://ws.okx.com:8443/ws/v5/public"
        self._latest_snapshot: OrderBookSnapshot | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ws: websocket.WebSocketApp | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)

    def latest_snapshot(self) -> OrderBookSnapshot | None:
        with self._lock:
            return self._latest_snapshot

    def _run(self) -> None:
        while not self._stop.is_set():
            self._ws = websocket.WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            try:
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                pass
            if not self._stop.is_set():
                time.sleep(1)

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        ws.send(
            json.dumps(
                {
                    "op": "subscribe",
                    "args": [{"channel": self.channel, "instId": self.inst_id}],
                }
            )
        )

    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        payload = json.loads(message)
        data = payload.get("data") or []
        if not data:
            return
        snapshot = data[0]
        ts = int(snapshot.get("ts") or 0)
        bids = tuple(OrderBookLevel(price=float(item[0]), size=float(item[1])) for item in snapshot.get("bids", []))
        asks = tuple(OrderBookLevel(price=float(item[0]), size=float(item[1])) for item in snapshot.get("asks", []))
        with self._lock:
            self._latest_snapshot = OrderBookSnapshot(
                inst_id=self.inst_id,
                timestamp_ms=ts,
                bids=bids,
                asks=asks,
            )

    def _on_error(self, ws: websocket.WebSocketApp, error: Any) -> None:
        return

    def _on_close(self, ws: websocket.WebSocketApp, status_code: Any, message: Any) -> None:
        return
