"""
Coinbase Advanced Trade REST API Client.

Implements the core exchange interface expected by the engine/executor:
- Market data (ticker, candles, book)
- Order management (place/cancel/query)
- Account data (balances)

Uses JWT (ES256) authentication with the Coinbase CDP API keys.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import jwt
from cryptography.hazmat.primitives import serialization

from src.core.logger import get_logger

logger = get_logger("coinbase_rest")


@dataclass
class CoinbaseAuthConfig:
    key_name: str
    private_key_pem: str


class CoinbaseRESTClient:
    """
    Coinbase Advanced Trade REST API client.

    Notes:
    - Uses JWT (ES256) authentication.
    - Only signs requests when authenticated endpoints are called.
    """

    DEFAULT_REST_URL = "https://api.coinbase.com"
    DEFAULT_SANDBOX_URL = "https://api-sandbox.coinbase.com"

    def __init__(
        self,
        rest_url: Optional[str] = None,
        market_data_url: Optional[str] = None,
        rate_limit: int = 10,
        max_retries: int = 5,
        timeout: int = 30,
        sandbox: bool = False,
        auth_config: Optional[CoinbaseAuthConfig] = None,
    ):
        self.rest_url = rest_url or (self.DEFAULT_SANDBOX_URL if sandbox else self.DEFAULT_REST_URL)
        self.market_data_url = market_data_url or self.rest_url
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.timeout = timeout
        self.sandbox = sandbox
        self._auth_config = auth_config
        self._client: Optional[httpx.AsyncClient] = None
        self._market_client: Optional[httpx.AsyncClient] = None
        self._rate_semaphore = asyncio.Semaphore(max(1, int(rate_limit)))
        self._host = urlparse(self.rest_url).netloc
        self._market_host = urlparse(self.market_data_url).netloc
        self._private_key = None

    async def initialize(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.rest_url,
            timeout=float(self.timeout),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            headers={"User-Agent": "AITradingBot/2.0"},
        )
        if self.market_data_url != self.rest_url:
            self._market_client = httpx.AsyncClient(
                base_url=self.market_data_url,
                timeout=float(self.timeout),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                headers={"User-Agent": "AITradingBot/2.0"},
            )
        if self._auth_config:
            self._private_key = self._load_private_key(self._auth_config.private_key_pem)
        logger.info("Coinbase REST client initialized", base_url=self.rest_url)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._market_client:
            await self._market_client.aclose()
            self._market_client = None

    # ------------------------------------------------------------------
    # Auth / JWT helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_private_key(pem: str):
        return serialization.load_pem_private_key(
            pem.encode("utf-8"),
            password=None,
        )

    def _build_jwt(self, method: str, path: str, host_override: Optional[str] = None) -> str:
        if not self._auth_config or not self._private_key:
            raise RuntimeError("Coinbase auth config not set")
        now = int(time.time())
        payload = {
            "sub": self._auth_config.key_name,
            "iss": "cdp",
            "nbf": now,
            "exp": now + 120,
            "uri": f"{method.upper()} {(host_override or self._host)}{path}",
        }
        headers = {
            "kid": self._auth_config.key_name,
            "nonce": secrets.token_hex(16),
        }
        token = jwt.encode(payload, self._private_key, algorithm="ES256", headers=headers)
        return token if isinstance(token, str) else token.decode("utf-8")

    # ------------------------------------------------------------------
    # Core request
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        authenticated: bool = False,
        use_market_client: bool = False,
    ) -> Dict[str, Any]:
        client = self._market_client if use_market_client and self._market_client else self._client
        if not client:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            headers: Dict[str, str] = {}
            if authenticated:
                host_override = self._market_host if use_market_client else None
                token = self._build_jwt(method, path, host_override=host_override)
                headers["Authorization"] = f"Bearer {token}"

            async with self._rate_semaphore:
                try:
                    if method.upper() == "GET":
                        resp = await client.get(path, params=params, headers=headers)
                    else:
                        resp = await client.request(
                            method.upper(),
                            path,
                            params=params,
                            json=json_body,
                            headers=headers,
                        )
                    resp.raise_for_status()
                    return resp.json() if resp.content else {}
                except httpx.HTTPStatusError as e:
                    last_error = e
                    if e.response.status_code < 500:
                        raise
                except (httpx.ConnectError, httpx.ReadTimeout) as e:
                    last_error = e

            if last_error and attempt < self.max_retries:
                delay = (2 ** attempt) * 0.5 + 0.5
                logger.warning(
                    "Coinbase request failed, retrying",
                    error=str(last_error),
                    delay=delay,
                    attempt=attempt + 1,
                )
                await asyncio.sleep(delay)
                last_error = None
            elif last_error:
                raise last_error

        raise RuntimeError(f"Request failed after {self.max_retries} retries")

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

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    async def get_ticker(self, pair: str) -> Dict[str, Any]:
        product_id = self._pair_to_product_id(pair)
        path = f"/api/v3/brokerage/products/{product_id}/ticker"
        data = await self._request(
            "GET",
            path,
            authenticated=bool(self._auth_config),
            use_market_client=True,
        )
        best_bid = data.get("best_bid") or data.get("bid")
        best_ask = data.get("best_ask") or data.get("ask")
        last = data.get("price")
        return {
            "symbol": self._product_id_to_pair(product_id),
            "last": last,
            "b": [best_bid, "0", "0"],
            "a": [best_ask, "0", "0"],
        }

    async def get_ohlc(
        self,
        pair: str,
        interval: int = 1,
        limit: int = 350,
        since: Optional[int] = None,
    ) -> List[List[Any]]:
        product_id = self._pair_to_product_id(pair)
        granularity = self._interval_to_granularity(interval)
        end_ts = int(time.time())
        if since:
            start_ts = int(since)
        else:
            start_ts = end_ts - int(interval * 60 * max(1, limit))
        path = f"/api/v3/brokerage/products/{product_id}/candles"
        params = {
            "start": str(start_ts),
            "end": str(end_ts),
            "granularity": granularity,
        }
        data = await self._request(
            "GET",
            path,
            params=params,
            authenticated=bool(self._auth_config),
            use_market_client=True,
        )
        candles = data.get("candles", []) if isinstance(data, dict) else []
        ohlc: List[List[Any]] = []
        for c in candles:
            # c may be dict-like
            ts = float(c.get("start", 0))
            o = c.get("open")
            h = c.get("high")
            l = c.get("low")
            cl = c.get("close")
            v = c.get("volume", 0)
            ohlc.append([ts, o, h, l, cl, 0, v, 0])
        ohlc.sort(key=lambda x: float(x[0]) if x and x[0] is not None else 0)
        return ohlc

    async def get_order_book(self, pair: str, depth: int = 25) -> Dict[str, Any]:
        product_id = self._pair_to_product_id(pair)
        path = "/api/v3/brokerage/product_book"
        params = {"product_id": product_id, "limit": str(depth)}
        data = await self._request(
            "GET",
            path,
            params=params,
            authenticated=bool(self._auth_config),
            use_market_client=True,
        )
        book = data.get("pricebook") if isinstance(data, dict) else {}
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        return {
            "symbol": self._product_id_to_pair(product_id),
            "bids": [[b.get("price"), b.get("size")] for b in bids],
            "asks": [[a.get("price"), a.get("size")] for a in asks],
        }

    async def get_trade_history_public(self, pair: str) -> Dict[str, Any]:
        product_id = self._pair_to_product_id(pair)
        path = f"/api/v3/brokerage/products/{product_id}/ticker"
        return await self._request("GET", path, authenticated=False)

    async def get_asset_pairs(self) -> Dict[str, Any]:
        path = "/api/v3/brokerage/products"
        return await self._request("GET", path, authenticated=False)

    # ------------------------------------------------------------------
    # Private endpoints
    # ------------------------------------------------------------------

    async def get_balance(self) -> Dict[str, float]:
        path = "/api/v3/brokerage/accounts"
        data = await self._request("GET", path, authenticated=True)
        balances: Dict[str, float] = {}
        accounts = data.get("accounts", []) if isinstance(data, dict) else []
        for acct in accounts:
            currency = acct.get("currency")
            available = acct.get("available_balance", {}).get("value")
            if currency and available is not None:
                try:
                    balances[currency] = float(available)
                except (TypeError, ValueError):
                    continue
        return balances

    async def get_trade_balance(self, asset: str = "USD") -> Dict[str, Any]:
        balances = await self.get_balance()
        return {"asset": asset, "balance": balances.get(asset, 0.0)}

    async def get_open_orders(self) -> Dict[str, Any]:
        orders = await self._list_orders(statuses=["OPEN"])
        return {"open": {o["order_id"]: self._normalize_order(o) for o in orders}}

    async def get_closed_orders(self) -> Dict[str, Any]:
        orders = await self._list_orders(statuses=["FILLED", "CANCELED", "EXPIRED"])
        return {"closed": {o["order_id"]: self._normalize_order(o) for o in orders}}

    async def get_order_info(self, txid: str) -> Dict[str, Any]:
        path = f"/api/v3/brokerage/orders/historical/{txid}"
        data = await self._request("GET", path, authenticated=True)
        order = data.get("order", {}) if isinstance(data, dict) else {}
        if not order:
            return {}
        return {txid: self._normalize_order(order)}

    async def place_order(
        self,
        pair: str,
        side: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        client_order_id: str = "",
        post_only: bool = False,
        validate_only: bool = False,
    ) -> Dict[str, Any]:
        product_id = self._pair_to_product_id(pair)
        path = "/api/v3/brokerage/orders"
        side_val = "BUY" if side.lower() == "buy" else "SELL"
        order_conf: Dict[str, Any] = {}
        if order_type == "market":
            order_conf["market_market_ioc"] = {
                "base_size": str(volume),
            }
        else:
            order_conf["limit_limit_gtc"] = {
                "base_size": str(volume),
                "limit_price": str(price if price is not None else 0),
                "post_only": bool(post_only),
            }
        body: Dict[str, Any] = {
            "client_order_id": client_order_id or secrets.token_hex(12),
            "product_id": product_id,
            "side": side_val,
            "order_configuration": order_conf,
        }
        if validate_only:
            body["preview"] = True
        data = await self._request("POST", path, json_body=body, authenticated=True)
        # Normalize to Kraken-like response shape
        order_id = None
        if isinstance(data, dict):
            if "order_id" in data:
                order_id = data.get("order_id")
            elif "success_response" in data:
                order_id = data.get("success_response", {}).get("order_id")
        if order_id:
            return {"txid": order_id}
        return data

    async def cancel_order(self, txid: str) -> Dict[str, Any]:
        path = "/api/v3/brokerage/orders/batch_cancel"
        body = {"order_ids": [txid]}
        return await self._request("POST", path, json_body=body, authenticated=True)

    async def cancel_all_orders(self) -> Dict[str, Any]:
        try:
            open_orders = await self.get_open_orders()
            order_ids = list(open_orders.get("open", {}).keys())
            if not order_ids:
                return {"cancelled": 0}
            path = "/api/v3/brokerage/orders/batch_cancel"
            body = {"order_ids": order_ids}
            return await self._request("POST", path, json_body=body, authenticated=True)
        except Exception as e:
            logger.warning("Cancel all orders failed", error=str(e))
            return {"cancelled": 0}

    async def get_trades_history(self, start: int, end: int) -> Dict[str, Any]:
        path = "/api/v3/brokerage/orders/historical/fills"
        params = {
            "start_sequence_timestamp": str(start),
            "end_sequence_timestamp": str(end),
        }
        data = await self._request("GET", path, params=params, authenticated=True)
        fills = data.get("fills", []) if isinstance(data, dict) else []
        trades: Dict[str, Dict[str, Any]] = {}
        for f in fills:
            trade_id = f.get("trade_id") or f.get("fill_id") or secrets.token_hex(8)
            trades[trade_id] = {
                "ordertxid": f.get("order_id"),
                "vol": f.get("size"),
                "price": f.get("price"),
                "fee": f.get("commission"),
            }
        return {"trades": trades}

    async def get_min_order_size(self, pair: str) -> float:
        product = await self._get_product(pair)
        try:
            return float(product.get("base_min_size", 0))
        except (TypeError, ValueError):
            return 0.0

    async def get_pair_decimals(self, pair: str) -> Tuple[int, int]:
        product = await self._get_product(pair)
        price_inc = product.get("quote_increment", "0.01")
        size_inc = product.get("base_increment", "0.00000001")
        return self._increment_to_decimals(price_inc), self._increment_to_decimals(size_inc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_product(self, pair: str) -> Dict[str, Any]:
        product_id = self._pair_to_product_id(pair)
        path = f"/api/v3/brokerage/products/{product_id}"
        data = await self._request("GET", path, authenticated=False)
        return data.get("product", data) if isinstance(data, dict) else {}

    async def _list_orders(self, statuses: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        path = "/api/v3/brokerage/orders/historical/batch"
        params: Dict[str, Any] = {"limit": "100"}
        if statuses:
            params["order_status"] = ",".join(statuses)
        data = await self._request("GET", path, params=params, authenticated=True)
        return data.get("orders", []) if isinstance(data, dict) else []

    def _normalize_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        order_id = order.get("order_id") or order.get("id") or ""
        filled_size = order.get("filled_size") or order.get("filled_quantity") or 0
        average_filled_price = order.get("average_filled_price") or order.get("avg_filled_price") or 0
        total_fees = order.get("total_fees") or order.get("fees") or order.get("commission") or 0
        total_size = order.get("size") or 0
        if not total_size:
            total_size = self._extract_order_size(order.get("order_configuration", {}))
        limit_price = self._extract_limit_price(order.get("order_configuration", {}))
        price = average_filled_price or limit_price or 0
        cost = order.get("filled_value") or (float(price) * float(filled_size) if price and filled_size else 0)
        return {
            "id": order_id,
            "vol": total_size,
            "vol_exec": filled_size,
            "price": price,
            "avg_price": average_filled_price,
            "cost": cost,
            "fee": total_fees,
            "descr": {"price": limit_price},
        }

    @staticmethod
    def _extract_order_size(order_conf: Dict[str, Any]) -> float:
        if "limit_limit_gtc" in order_conf:
            return float(order_conf["limit_limit_gtc"].get("base_size", 0) or 0)
        if "market_market_ioc" in order_conf:
            return float(order_conf["market_market_ioc"].get("base_size", 0) or 0)
        return 0.0

    @staticmethod
    def _extract_limit_price(order_conf: Dict[str, Any]) -> float:
        if "limit_limit_gtc" in order_conf:
            return float(order_conf["limit_limit_gtc"].get("limit_price", 0) or 0)
        return 0.0

    @staticmethod
    def _interval_to_granularity(interval: int) -> str:
        mapping = {
            1: "ONE_MINUTE",
            5: "FIVE_MINUTE",
            15: "FIFTEEN_MINUTE",
            30: "THIRTY_MINUTE",
            60: "ONE_HOUR",
            240: "FOUR_HOUR",
            1440: "ONE_DAY",
        }
        return mapping.get(int(interval), "ONE_MINUTE")

    @staticmethod
    def _increment_to_decimals(value: Any) -> int:
        try:
            s = str(value)
            if "." not in s:
                return 0
            return len(s.rstrip("0").split(".")[1])
        except Exception:
            return 8
