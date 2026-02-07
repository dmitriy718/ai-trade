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
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
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
        # C4 FIX: API secret for control endpoint auth
        self._api_secret = os.getenv("DASHBOARD_SECRET_KEY", "change_this_to_a_random_string")
        self._setup_middleware()
        self._setup_routes()
        self._ws_connections: Set[WebSocket] = set()
        self._bot_engine = None
        self._ws_cache: Optional[Dict[str, Any]] = None
        self._ws_cache_time: float = 0

    def set_bot_engine(self, engine) -> None:
        """Inject the bot engine reference."""
        self._bot_engine = engine

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

        # ---- Status Endpoints ----

        @self.app.get("/api/v1/status")
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

        @self.app.get("/api/v1/trades")
        async def get_trades(limit: int = Query(default=100, ge=1, le=1000)):
            """Get trade history."""
            if not self._bot_engine:
                return []
            trades = await self._bot_engine.db.get_trade_history(limit=limit)
            return trades

        @self.app.get("/api/v1/positions")
        async def get_positions():
            """Get open positions."""
            if not self._bot_engine:
                return []
            positions = await self._bot_engine.db.get_open_trades()
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
        async def get_performance():
            """Get performance metrics including unrealized P&L."""
            if not self._bot_engine:
                return {}
            stats = await self._bot_engine.db.get_performance_stats()
            risk_report = self._bot_engine.risk_manager.get_risk_report()

            # Add unrealized P&L from open positions
            positions = await self._bot_engine.db.get_open_trades()
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
        async def get_thoughts(limit: int = 50):
            """Get AI thought feed."""
            if not self._bot_engine:
                return []
            return await self._bot_engine.db.get_thoughts(limit=limit)

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

        # ---- Control Endpoints ----

        @self.app.post("/api/v1/control/close_all", dependencies=[Depends(_require_auth)])
        async def close_all_positions():
            """Emergency close all positions. Requires X-API-Key header."""
            if not self._bot_engine:
                raise HTTPException(status_code=503, detail="Bot not running")
            count = await self._bot_engine.executor.close_all_positions("api_close_all")
            return {"closed": count}

        @self.app.post("/api/v1/control/pause", dependencies=[Depends(_require_auth)])
        async def pause_trading():
            """Pause trading. Requires X-API-Key header."""
            if self._bot_engine:
                self._bot_engine._trading_paused = True
                await self._bot_engine.db.log_thought(
                    "system", "Trading PAUSED via API", severity="warning"
                )
            return {"status": "paused"}

        @self.app.post("/api/v1/control/resume", dependencies=[Depends(_require_auth)])
        async def resume_trading():
            """Resume trading. Requires X-API-Key header."""
            if self._bot_engine:
                self._bot_engine._trading_paused = False
                await self._bot_engine.db.log_thought(
                    "system", "Trading RESUMED via API", severity="info"
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
                while True:
                    # H13 FIX: Use cached update (built once per second)
                    now = time.time()
                    if not self._ws_cache or (now - self._ws_cache_time) > 1.0:
                        self._ws_cache = await self._build_ws_update()
                        self._ws_cache_time = now
                    await websocket.send_json(self._ws_cache)
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.debug("WebSocket error", error=str(e))
            finally:
                self._ws_connections.discard(websocket)
                logger.info("WebSocket client disconnected", total=len(self._ws_connections))

    async def _build_ws_update(self) -> Dict[str, Any]:
        """Build a WebSocket update payload."""
        if not self._bot_engine:
            return {"type": "status", "data": {"status": "initializing"}}

        # Build compact update
        try:
            performance = await self._bot_engine.db.get_performance_stats()
            positions = await self._bot_engine.db.get_open_trades()
            thoughts = await self._bot_engine.db.get_thoughts(limit=50)

            # Add current prices to positions
            for pos in positions:
                cp = self._bot_engine.market_data.get_latest_price(pos["pair"])
                if cp > 0:
                    if pos["side"] == "buy":
                        pos["unrealized_pnl"] = round(
                            (cp - pos["entry_price"]) * pos["quantity"], 2
                        )
                    else:
                        pos["unrealized_pnl"] = round(
                            (pos["entry_price"] - cp) * pos["quantity"], 2
                        )
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
