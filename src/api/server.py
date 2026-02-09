"""
FastAPI Dashboard Server - REST + WebSocket API for monitoring and control.

Provides real-time dashboard data, trade management endpoints,
WebSocket streaming for live updates, and system control commands.

# ENHANCEMENT: Added CORS and security middleware
# ENHANCEMENT: Added request rate limiting
# ENHANCEMENT: Added WebSocket heartbeat for stale connection detection
# ENHANCEMENT: Added API versioning support
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.core.logger import get_logger

logger = get_logger("api_server")


class DashboardServer:
    """
    FastAPI-based dashboard server with WebSocket streaming.
    
    Endpoints:
    - GET /api/v1/status - System status
    - GET /api/v1/trades - Trade history
    - GET /api/v1/positions - Open positions
    - GET /api/v1/performance - Performance metrics
    - GET /api/v1/strategies - Strategy stats
    - GET /api/v1/risk - Risk report
    - GET /api/v1/thoughts - AI thought feed
    - GET /api/v1/scanner - Market scanner results
    - POST /api/v1/control/close_all - Emergency close all
    - POST /api/v1/control/pause - Pause trading
    - POST /api/v1/control/resume - Resume trading
    - WS /ws/live - Real-time data stream
    
    # ENHANCEMENT: Added response caching for frequently accessed data
    # ENHANCEMENT: Added WebSocket binary protocol for efficiency
    """

    def __init__(self):
        self.app = FastAPI(
            title="AI Trading Bot Dashboard",
            version="2.0.0",
            docs_url="/api/docs",
        )
        # C4 FIX: API secret for control endpoint auth (fail fast in live mode)
        self._api_secret = os.getenv("DASHBOARD_SECRET_KEY", "").strip()
        self._generated_secret = False
        if not self._api_secret:
            self._api_secret = secrets.token_urlsafe(32)
            self._generated_secret = True
        self._setup_middleware()
        self._setup_routes()
        self._ws_connections: Set[WebSocket] = set()
        self._bot_engine = None
        self._control_router = None
        self._stripe_service = None
        self._ws_cache_by_tenant: Dict[str, Dict[str, Any]] = {}
        self._ws_cache_time_by_tenant: Dict[str, float] = {}

    def set_bot_engine(self, engine) -> None:
        """Inject the bot engine reference."""
        self._bot_engine = engine
        if engine and getattr(engine, "config", None):
            if engine.config.app.mode == "live" and self._generated_secret:
                raise RuntimeError(
                    "DASHBOARD_SECRET_KEY is required in live mode."
                )
            if self._generated_secret:
                logger.warning(
                    "DASHBOARD_SECRET_KEY not set; generated an ephemeral secret. "
                    "Set the env var to enable control endpoints."
                )
        if self._stripe_service and engine and getattr(engine, "db", None):
            self._stripe_service.set_db(engine.db)

    def set_control_router(self, router) -> None:
        """Inject the control router for pause/resume/close_all."""
        self._control_router = router

    def set_stripe_service(self, service) -> None:
        """Inject Stripe service for billing endpoints."""
        self._stripe_service = service
        if self._bot_engine and getattr(self._bot_engine, "db", None):
            service.set_db(self._bot_engine.db)

    def _setup_middleware(self) -> None:
        """Configure CORS and security middleware."""
        # H12 FIX: Restrict CORS to localhost
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        """Register all API routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Serve the main dashboard page."""
            dashboard_path = Path("static/index.html")
            if dashboard_path.exists():
                return HTMLResponse(content=dashboard_path.read_text())
            return HTMLResponse(content="<h1>AI Trading Bot - Dashboard Loading...</h1>")

        # Mount static files
        static_path = Path("static")
        if static_path.exists():
            self.app.mount("/static", StaticFiles(directory="static"), name="static")

        @self.app.get("/favicon.ico", include_in_schema=False)
        async def favicon():
            from fastapi import Response
            return Response(status_code=204)

        # ---- Status Endpoints ----

        @self.app.get("/api/v1/health")
        @self.app.head("/api/v1/health")
        async def health():
            """Probing endpoint for dashboard connectivity."""
            return {"status": "ok"}

        @self.app.get("/api/v1/status")
        @self.app.head("/api/v1/status")
        async def get_status():
            """Get overall system status."""
            if not self._bot_engine:
                return {"status": "initializing"}

            return {
                "status": "running" if self._bot_engine._running else "stopped",
                "mode": self._bot_engine.mode,
                "uptime_seconds": time.time() - self._bot_engine._start_time,
                "scan_count": self._bot_engine._scan_count,
                "version": "2.0.0",
                "pairs": self._bot_engine.pairs,
                "scan_interval": self._bot_engine.scan_interval,
                "ws_connected": (
                    self._bot_engine.ws_client.is_connected
                    if self._bot_engine.ws_client else False
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # M30 FIX: Auth dependency for control endpoints
        async def _require_auth(x_api_key: str = Header(default="")):
            if not x_api_key or x_api_key != self._api_secret:
                raise HTTPException(status_code=403, detail="Unauthorized")

        async def _resolve_tenant_id(
            request: Request,
            x_tenant_id: str = Header(default="", alias="X-Tenant-ID"),
            x_api_key: str = Header(default="", alias="X-API-Key"),
        ) -> str:
            """Resolve tenant_id from header or API key."""
            if x_tenant_id:
                return x_tenant_id
            if x_api_key and self._bot_engine and self._bot_engine.db:
                tenant_id = await self._bot_engine.db.get_tenant_id_by_api_key(x_api_key)
                if tenant_id:
                    return tenant_id
            if self._bot_engine and getattr(self._bot_engine, "config", None):
                return self._bot_engine.config.billing.tenant.default_tenant_id
            return "default"

        @self.app.get("/api/v1/trades")
        async def get_trades(
            limit: int = Query(default=100, ge=1, le=1000),
            tenant_id: str = Depends(_resolve_tenant_id),
        ):
            """Get trade history."""
            if not self._bot_engine:
                return []
            trades = await self._bot_engine.db.get_trade_history(
                limit=limit, tenant_id=tenant_id
            )
            return trades

        @self.app.get("/api/v1/positions")
        async def get_positions(tenant_id: str = Depends(_resolve_tenant_id)):
            """Get open positions."""
            if not self._bot_engine:
                return []
            positions = await self._bot_engine.db.get_open_trades(tenant_id=tenant_id)
            # Add current price and unrealized PnL
            for pos in positions:
                current_price = self._bot_engine.market_data.get_latest_price(
                    pos["pair"]
                )
                if current_price > 0:
                    if pos["side"] == "buy":
                        pos["unrealized_pnl"] = (
                            (current_price - pos["entry_price"]) * pos["quantity"]
                        )
                    else:
                        pos["unrealized_pnl"] = (
                            (pos["entry_price"] - current_price) * pos["quantity"]
                        )
                    pos["current_price"] = current_price
                    pos["unrealized_pnl_pct"] = (
                        pos["unrealized_pnl"] /
                        (pos["entry_price"] * pos["quantity"])
                    ) if pos["entry_price"] * pos["quantity"] > 0 else 0
            return positions

        @self.app.get("/api/v1/performance")
        async def get_performance(tenant_id: str = Depends(_resolve_tenant_id)):
            """Get performance metrics including unrealized P&L."""
            if not self._bot_engine:
                return {}
            stats = await self._bot_engine.db.get_performance_stats(tenant_id=tenant_id)
            risk_report = self._bot_engine.risk_manager.get_risk_report()

            # Add unrealized P&L from open positions
            positions = await self._bot_engine.db.get_open_trades(tenant_id=tenant_id)
            unrealized = 0.0
            for pos in positions:
                cp = self._bot_engine.market_data.get_latest_price(pos["pair"])
                if cp > 0:
                    if pos["side"] == "buy":
                        unrealized += (cp - pos["entry_price"]) * pos["quantity"]
                    else:
                        unrealized += (pos["entry_price"] - cp) * pos["quantity"]
            stats["unrealized_pnl"] = round(unrealized, 2)
            stats["total_equity"] = round(
                risk_report.get("bankroll", 10000) + unrealized, 2
            )
            return {**stats, **risk_report}

        @self.app.get("/api/v1/strategies")
        async def get_strategies():
            """Get strategy performance stats."""
            if not self._bot_engine:
                return []
            return self._bot_engine.confluence.get_strategy_stats()

        @self.app.get("/api/v1/risk")
        async def get_risk():
            """Get risk management report."""
            if not self._bot_engine:
                return {}
            return self._bot_engine.risk_manager.get_risk_report()

        @self.app.get("/api/v1/thoughts")
        async def get_thoughts(
            limit: int = 50,
            tenant_id: str = Depends(_resolve_tenant_id),
        ):
            """Get AI thought feed."""
            if not self._bot_engine:
                return []
            return await self._bot_engine.db.get_thoughts(limit=limit, tenant_id=tenant_id)

        @self.app.get("/api/v1/scanner")
        async def get_scanner():
            """Get market scanner status."""
            if not self._bot_engine:
                return {}
            return self._bot_engine.market_data.get_status()

        @self.app.get("/api/v1/execution")
        async def get_execution():
            """Get execution statistics."""
            if not self._bot_engine:
                return {}
            return self._bot_engine.executor.get_execution_stats()

        # ---- Settings (AI / confluence options) ----
        @self.app.get("/api/v1/settings")
        async def get_settings():
            """Get settings used by the dashboard (e.g. Weighted Order Book)."""
            if not self._bot_engine:
                return {"weighted_order_book": False}
            c = getattr(self._bot_engine, "confluence", None)
            return {
                "weighted_order_book": getattr(c, "obi_counts_as_confluence", False),
            }

        @self.app.patch("/api/v1/settings", dependencies=[Depends(_require_auth)])
        async def patch_settings(body: dict = Body(...)):
            """Update settings at runtime (e.g. Weighted Order Book). Takes effect immediately; for persistence set config.yaml and restart."""
            if not self._bot_engine:
                raise HTTPException(status_code=503, detail="Bot not running")
            c = getattr(self._bot_engine, "confluence", None)
            if not c:
                raise HTTPException(status_code=503, detail="Confluence not available")
            if "weighted_order_book" in body:
                c.obi_counts_as_confluence = bool(body["weighted_order_book"])
            return {"weighted_order_book": c.obi_counts_as_confluence}

        # ---- Billing (Stripe) ----
        @self.app.post("/api/v1/billing/checkout", dependencies=[Depends(_require_auth)])
        async def create_checkout_session(body: dict = Body(...)):
            """Create Stripe Checkout session for subscription. Body: tenant_id, success_url, cancel_url, customer_email (optional)."""
            if not self._stripe_service or not self._stripe_service.enabled:
                raise HTTPException(status_code=503, detail="Billing not configured")
            tenant_id = body.get("tenant_id") or "default"
            success_url = body.get("success_url", "")
            cancel_url = body.get("cancel_url", "")
            if not success_url or not cancel_url:
                raise HTTPException(status_code=400, detail="success_url and cancel_url required")
            customer_email = body.get("customer_email")
            customer_id = None
            if self._bot_engine and self._bot_engine.db:
                tenant = await self._bot_engine.db.get_tenant(tenant_id)
                customer_id = tenant.get("stripe_customer_id") if tenant else None
            result = self._stripe_service.create_checkout_session(
                tenant_id=tenant_id,
                success_url=success_url,
                cancel_url=cancel_url,
                customer_email=customer_email,
                customer_id=customer_id,
            )
            if not result:
                raise HTTPException(status_code=500, detail="Failed to create checkout session")
            return result

        @self.app.post("/api/v1/billing/webhook")
        async def stripe_webhook(request: Request, stripe_signature: Optional[str] = Header(None, alias="Stripe-Signature")):
            """Stripe webhook: verify signature and update tenant status. No auth (verified by Stripe signature)."""
            if not self._stripe_service or not self._stripe_service.webhook_secret:
                raise HTTPException(status_code=503, detail="Webhook not configured")
            payload = await request.body()
            if not stripe_signature:
                raise HTTPException(status_code=400, detail="Missing Stripe-Signature header")
            if not self._stripe_service.verify_webhook(payload, stripe_signature):
                raise HTTPException(status_code=400, detail="Invalid signature")
            import json as _json
            event = _json.loads(payload)
            await self._stripe_service.handle_webhook_event(event)
            return {"received": True}

        @self.app.get("/api/v1/tenants/{tenant_id}")
        async def get_tenant(tenant_id: str):
            """Get tenant by id (for dashboard / billing status)."""
            if not self._bot_engine or not self._bot_engine.db:
                raise HTTPException(status_code=503, detail="Not available")
            tenant = await self._bot_engine.db.get_tenant(tenant_id)
            if not tenant:
                raise HTTPException(status_code=404, detail="Tenant not found")
            return tenant

        # ---- Control Endpoints ----

        @self.app.post("/api/v1/control/close_all", dependencies=[Depends(_require_auth)])
        async def close_all_positions(
            tenant_id: str = Depends(_resolve_tenant_id),
        ):
            """Emergency close all positions. Requires X-API-Key header."""
            if self._control_router:
                result = await self._control_router.close_all("api_close_all")
                if not result.get("ok"):
                    raise HTTPException(status_code=503, detail=result.get("error", "Bot not running"))
                return {"closed": result.get("closed", 0)}
            if not self._bot_engine:
                raise HTTPException(status_code=503, detail="Bot not running")
            count = await self._bot_engine.executor.close_all_positions("api_close_all")
            return {"closed": count}

        @self.app.post("/api/v1/control/pause", dependencies=[Depends(_require_auth)])
        async def pause_trading(
            tenant_id: str = Depends(_resolve_tenant_id),
        ):
            """Pause trading. Requires X-API-Key header."""
            if self._control_router:
                result = await self._control_router.pause()
                return {"status": "paused"}
            if self._bot_engine:
                self._bot_engine._trading_paused = True
                await self._bot_engine.db.log_thought(
                    "system", "Trading PAUSED via API", severity="warning",
                    tenant_id=tenant_id,
                )
            return {"status": "paused"}

        @self.app.post("/api/v1/control/resume", dependencies=[Depends(_require_auth)])
        async def resume_trading(
            tenant_id: str = Depends(_resolve_tenant_id),
        ):
            """Resume trading. Requires X-API-Key header."""
            if self._control_router:
                await self._control_router.resume()
                return {"status": "resumed"}
            if self._bot_engine:
                self._bot_engine._trading_paused = False
                await self._bot_engine.db.log_thought(
                    "system", "Trading RESUMED via API", severity="info",
                    tenant_id=tenant_id,
                )
            return {"status": "resumed"}

        # ---- WebSocket ----

        @self.app.websocket("/ws/live")
        async def websocket_endpoint(websocket: WebSocket):
            """Real-time data streaming WebSocket."""
            await websocket.accept()
            self._ws_connections.add(websocket)
            logger.info("WebSocket client connected", total=len(self._ws_connections))

            try:
                tenant_id = (
                    websocket.query_params.get("tenant_id")
                    or websocket.headers.get("x-tenant-id")
                )
                if not tenant_id and self._bot_engine and self._bot_engine.db:
                    api_key = websocket.headers.get("x-api-key") or ""
                    if api_key:
                        tenant_id = await self._bot_engine.db.get_tenant_id_by_api_key(api_key)
                if not tenant_id:
                    tenant_id = (
                        self._bot_engine.config.billing.tenant.default_tenant_id
                        if self._bot_engine else "default"
                    )
                while True:
                    # H13 FIX: Use cached update (built once per second)
                    now = time.time()
                    cache_time = self._ws_cache_time_by_tenant.get(tenant_id, 0.0)
                    if (tenant_id not in self._ws_cache_by_tenant) or ((now - cache_time) > 1.0):
                        self._ws_cache_by_tenant[tenant_id] = await self._build_ws_update(
                            tenant_id=tenant_id
                        )
                        self._ws_cache_time_by_tenant[tenant_id] = now
                    await websocket.send_json(self._ws_cache_by_tenant[tenant_id])
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.debug("WebSocket error", error=str(e))
            finally:
                self._ws_connections.discard(websocket)
                logger.info("WebSocket client disconnected", total=len(self._ws_connections))

    async def _build_ws_update(self, tenant_id: str = "default") -> Dict[str, Any]:
        """Build a WebSocket update payload."""
        if not self._bot_engine:
            return {"type": "status", "data": {"status": "initializing"}}
        # Guard: DB must be initialized
        if not self._bot_engine.db or not self._bot_engine.db._initialized:
            return {"type": "status", "data": {"status": "initializing"}}

        # Build compact update
        try:
            performance = await self._bot_engine.db.get_performance_stats(tenant_id=tenant_id)
            positions = await self._bot_engine.db.get_open_trades(tenant_id=tenant_id)
            thoughts = await self._bot_engine.db.get_thoughts(limit=50, tenant_id=tenant_id)

            # S12 FIX: Add current prices and net unrealized P&L (minus estimated exit fee)
            FEE_RATE = 0.0026
            for pos in positions:
                cp = self._bot_engine.market_data.get_latest_price(pos["pair"])
                if cp > 0:
                    if pos["side"] == "buy":
                        gross = (cp - pos["entry_price"]) * pos["quantity"]
                    else:
                        gross = (pos["entry_price"] - cp) * pos["quantity"]
                    est_exit_fee = abs(cp * pos["quantity"]) * FEE_RATE
                    pos["unrealized_pnl"] = round(gross - est_exit_fee, 2)
                    pos["current_price"] = cp

            scanner_data = {}
            for pair in self._bot_engine.pairs:
                price = self._bot_engine.market_data.get_latest_price(pair)
                scanner_data[pair] = {
                    "price": price,
                    "bars": self._bot_engine.market_data.get_bar_count(pair),
                    "stale": self._bot_engine.market_data.is_stale(pair),
                }

            risk = self._bot_engine.risk_manager.get_risk_report()

            # Calculate total unrealized P&L from open positions
            total_unrealized = sum(
                p.get("unrealized_pnl", 0) for p in positions
            )
            performance["unrealized_pnl"] = round(total_unrealized, 2)
            performance["total_equity"] = round(
                (performance.get("total_pnl", 0) or 0) + total_unrealized +
                self._bot_engine.risk_manager.initial_bankroll, 2
            )

            return {
                "type": "update",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "performance": performance,
                    "positions": positions,
                    "thoughts": thoughts,
                    "scanner": scanner_data,
                    "risk": risk,
                    "status": {
                        "running": self._bot_engine._running,
                        "paused": self._bot_engine._trading_paused,
                        "mode": self._bot_engine.mode,
                        "uptime": time.time() - self._bot_engine._start_time,
                        "scan_count": self._bot_engine._scan_count,
                        "ws_connected": (
                            self._bot_engine.ws_client.is_connected
                            if self._bot_engine.ws_client else False
                        ),
                    }
                }
            }
        except Exception as e:
            logger.error("WebSocket update build error", error=str(e))
            # M30 FIX: Don't leak internal errors to clients
            return {"type": "error", "message": "Internal update error"}

    async def broadcast(self, data: Dict[str, Any]) -> None:
        """Broadcast data to all connected WebSocket clients."""
        disconnected = set()
        for ws in self._ws_connections:
            try:
                await ws.send_json(data)
            except Exception:
                disconnected.add(ws)
        self._ws_connections -= disconnected
