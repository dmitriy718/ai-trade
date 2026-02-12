"""
Coinbase Advanced Trade WebSocket Client.

Normalizes Coinbase WS market data into the internal message format
used by the engine:
{
  "channel": "ticker"|"ohlc"|"book",
  "data": [ { "symbol": "BTC/USD", ... } ]
}
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from src.core.logger import get_logger

logger = get_logger("coinbase_ws")


class CoinbaseWebSocketClient:
    DEFAULT_WS_URL = "wss://advanced-trade-ws.coinbase.com"

    def __init__(
        self,
        url: Optional[str] = None,
        max_reconnect_attempts: int = 50,
        heartbeat_interval: int = 30,
    ):
        self.url = url or self.DEFAULT_WS_URL
        self.max_reconnect_attempts = max_reconnect_attempts
        self.heartbeat_interval = heartbeat_interval

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._running = False
        self._reconnect_count = 0
        self._last_heartbeat = 0.0
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Order book state per product
        self._books: Dict[str, Dict[str, Dict[float, float]]] = {}

    # ------------------------------------------------------------------
    # Connection Management
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._running = True
        self._reconnect_count = 0

        while self._running and self._reconnect_count < self.max_reconnect_attempts:
            try:
                logger.info("Connecting to Coinbase WebSocket", url=self.url)
                self._ws = await websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=2 ** 20,
                )
                self._connected = True
                self._reconnect_count = 0
                self._last_heartbeat = time.time()
                logger.info("Coinbase WebSocket connected")
                await self._resubscribe()
                await self._message_loop()
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                self._connected = False
                self._reconnect_count += 1
                if not self._running:
                    break
                delay = min(2 ** self._reconnect_count, 60)
                logger.warning(
                    "Coinbase WS disconnected, reconnecting",
                    error=str(e),
                    attempt=self._reconnect_count,
                    delay=delay,
                )
                await asyncio.sleep(delay)
            except Exception as e:
                self._connected = False
                logger.error("Coinbase WS unexpected error", error=str(e))
                if self._running:
                    await asyncio.sleep(5)

        if self._reconnect_count >= self.max_reconnect_attempts:
            logger.critical("Coinbase WS max reconnection attempts reached")

    async def disconnect(self) -> None:
        self._running = False
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("Coinbase WS disconnected")

    async def _message_loop(self) -> None:
        if not self._ws:
            return
        async for raw in self._ws:
            try:
                self._last_heartbeat = time.time()
                msg = json.loads(raw)
                channel = msg.get("channel") or msg.get("type") or ""
                if channel in ("heartbeat", "heartbeats"):
                    continue
                await self._handle_message(msg)
            except json.JSONDecodeError:
                logger.warning("Coinbase WS invalid JSON", raw=str(raw)[:200])
            except Exception as e:
                logger.error("Coinbase WS message error", error=str(e))

    # ------------------------------------------------------------------
    # Subscription Management
    # ------------------------------------------------------------------

    async def subscribe_ticker(self, pairs: List[str]) -> None:
        await self._subscribe("ticker", pairs)

    async def subscribe_ohlc(self, pairs: List[str], interval: int = 1) -> None:
        # Coinbase candles channel is 5m; avoid if not compatible.
        if interval not in (5,):
            logger.info("Coinbase WS candles disabled (interval not supported)", interval=interval)
            return
        await self._subscribe("candles", pairs)

    async def subscribe_book(self, pairs: List[str], depth: int = 25) -> None:
        await self._subscribe("level2", pairs, {"depth": depth})

    async def subscribe_trade(self, pairs: List[str]) -> None:
        await self._subscribe("market_trades", pairs)

    async def _subscribe(
        self,
        channel: str,
        pairs: List[str],
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        product_ids = [self._pair_to_product_id(p) for p in pairs]
        self._subscriptions[channel] = {"product_ids": product_ids, "params": params or {}}
        if self._ws and self._connected:
            payload = {
                "type": "subscribe",
                "channel": channel,
                "product_ids": product_ids,
            }
            await self._ws.send(json.dumps(payload))

    async def _resubscribe(self) -> None:
        if not self._ws:
            return
        for channel, sub in self._subscriptions.items():
            payload = {
                "type": "subscribe",
                "channel": channel,
                "product_ids": sub.get("product_ids", []),
            }
            await self._ws.send(json.dumps(payload))

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_ticker(self, cb: Callable) -> None:
        self._callbacks["ticker"].append(cb)

    def on_ohlc(self, cb: Callable) -> None:
        self._callbacks["ohlc"].append(cb)

    def on_book(self, cb: Callable) -> None:
        self._callbacks["book"].append(cb)

    def on_trade(self, cb: Callable) -> None:
        self._callbacks["trade"].append(cb)

    async def _route(self, channel: str, data: Dict[str, Any]) -> None:
        for cb in self._callbacks.get(channel, []):
            await cb(data)

    # ------------------------------------------------------------------
    # Message normalization
    # ------------------------------------------------------------------

    async def _handle_message(self, msg: Dict[str, Any]) -> None:
        channel = msg.get("channel", "")
        events = msg.get("events") or []
        if isinstance(events, dict):
            events = [events]
        if channel == "ticker":
            await self._handle_ticker(events)
        elif channel == "candles":
            await self._handle_candles(events)
        elif channel in ("level2", "l2_data"):
            await self._handle_book(events)
        elif channel == "market_trades":
            await self._handle_trades(events)

    async def _handle_ticker(self, events: List[Dict[str, Any]]) -> None:
        data_list: List[Dict[str, Any]] = []
        for ev in events:
            tickers = ev.get("tickers") or []
            for t in tickers:
                product_id = t.get("product_id")
                if not product_id:
                    continue
                pair = self._product_id_to_pair(product_id)
                last = t.get("price")
                best_bid = t.get("best_bid")
                best_ask = t.get("best_ask")
                data_list.append({
                    "symbol": pair,
                    "last": last,
                    "b": [best_bid, "0", "0"],
                    "a": [best_ask, "0", "0"],
                })
        if data_list:
            await self._route("ticker", {"channel": "ticker", "data": data_list})

    async def _handle_candles(self, events: List[Dict[str, Any]]) -> None:
        data_list: List[Dict[str, Any]] = []
        for ev in events:
            candles = ev.get("candles") or []
            for c in candles:
                product_id = c.get("product_id")
                if not product_id:
                    continue
                pair = self._product_id_to_pair(product_id)
                data_list.append({
                    "symbol": pair,
                    "interval_begin": c.get("start"),
                    "open": c.get("open"),
                    "high": c.get("high"),
                    "low": c.get("low"),
                    "close": c.get("close"),
                    "volume": c.get("volume"),
                })
        if data_list:
            await self._route("ohlc", {"channel": "ohlc", "data": data_list})

    async def _handle_book(self, events: List[Dict[str, Any]]) -> None:
        data_list: List[Dict[str, Any]] = []
        for ev in events:
            product_id = ev.get("product_id")
            if not product_id:
                continue
            pair = self._product_id_to_pair(product_id)
            if product_id not in self._books:
                self._books[product_id] = {"bids": {}, "asks": {}}
            book = self._books[product_id]
            event_type = str(ev.get("event_type") or ev.get("type") or "").lower()
            if event_type == "snapshot":
                book["bids"].clear()
                book["asks"].clear()
            # Snapshot support (if provided)
            if ev.get("bids") or ev.get("asks"):
                book["bids"].clear()
                book["asks"].clear()
                for b in ev.get("bids", []):
                    try:
                        book["bids"][float(b.get("price") or b[0])] = float(b.get("size") or b[1])
                    except Exception:
                        continue
                for a in ev.get("asks", []):
                    try:
                        book["asks"][float(a.get("price") or a[0])] = float(a.get("size") or a[1])
                    except Exception:
                        continue
            updates = ev.get("updates") or []
            for upd in updates:
                side = upd.get("side")
                price = upd.get("price_level")
                size = upd.get("new_quantity")
                if price is None or size is None:
                    continue
                try:
                    p = float(price)
                    s = float(size)
                except (TypeError, ValueError):
                    continue
                side_norm = str(side).lower()
                target = book["bids"] if side_norm in ("bid", "buy") else book["asks"]
                if s <= 0:
                    target.pop(p, None)
                else:
                    target[p] = s
            bids = [[p, sz] for p, sz in sorted(book["bids"].items(), key=lambda x: x[0], reverse=True)[:50]]
            asks = [[p, sz] for p, sz in sorted(book["asks"].items(), key=lambda x: x[0])[:50]]
            data_list.append({"symbol": pair, "bids": bids, "asks": asks})

        if data_list:
            await self._route("book", {"channel": "book", "data": data_list})

    async def _handle_trades(self, events: List[Dict[str, Any]]) -> None:
        data_list: List[Dict[str, Any]] = []
        for ev in events:
            trades = ev.get("trades") or []
            for t in trades:
                product_id = t.get("product_id")
                if not product_id:
                    continue
                pair = self._product_id_to_pair(product_id)
                data_list.append({
                    "symbol": pair,
                    "price": t.get("price"),
                    "size": t.get("size"),
                    "side": t.get("side"),
                    "timestamp": t.get("time"),
                })
        if data_list:
            await self._route("trade", {"channel": "trade", "data": data_list})

    # ------------------------------------------------------------------
    # Pair helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pair_to_product_id(pair: str) -> str:
        if "/" in pair:
            return pair.replace("/", "-")
        return pair

    @staticmethod
    def _product_id_to_pair(product_id: str) -> str:
        if "-" in product_id:
            return product_id.replace("-", "/")
        return product_id
