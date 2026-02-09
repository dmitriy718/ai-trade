"""
Control Router - Single point for pause, resume, close_all, kill, status.

All control channels (Web API, Telegram, Discord, Slack) call this router
so auth and routing stay in one place. The router delegates to the engine/executor.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.core.logger import get_logger

logger = get_logger("control_router")


class ControlRouter:
    """
    Central control interface for the trading bot.
    
    Exposes pause(), resume(), close_all(reason), kill(), get_status(),
    get_pnl(), get_positions(), get_risk() so Web, Telegram, Discord, Slack
    all use the same logic.
    """

    def __init__(self, engine) -> None:
        self._engine = engine

    async def pause(self) -> Dict[str, Any]:
        """Pause trading. Returns status dict."""
        if not self._engine:
            return {"ok": False, "error": "engine not set"}
        self._engine._trading_paused = True
        if self._engine.db:
            await self._engine.db.log_thought(
                "system", "Trading PAUSED via control", severity="warning",
                tenant_id=getattr(self._engine, "tenant_id", "default"),
            )
        logger.info("Trading paused via control router")
        return {"ok": True, "status": "paused"}

    async def resume(self) -> Dict[str, Any]:
        """Resume trading. Returns status dict."""
        if not self._engine:
            return {"ok": False, "error": "engine not set"}
        self._engine._trading_paused = False
        if self._engine.db:
            await self._engine.db.log_thought(
                "system", "Trading RESUMED via control", severity="info",
                tenant_id=getattr(self._engine, "tenant_id", "default"),
            )
        logger.info("Trading resumed via control router")
        return {"ok": True, "status": "resumed"}

    async def close_all(self, reason: str = "control") -> Dict[str, Any]:
        """Close all open positions. Returns closed count."""
        if not self._engine:
            return {"ok": False, "error": "engine not set", "closed": 0}
        count = await self._engine.executor.close_all_positions(reason)
        logger.info("Close all via control router", reason=reason, closed=count)
        return {"ok": True, "closed": count}

    async def kill(self) -> Dict[str, Any]:
        """Emergency: close all positions and stop the engine."""
        if not self._engine:
            return {"ok": False, "error": "engine not set"}
        count = await self._engine.executor.close_all_positions("emergency_kill")
        await self._engine.stop()
        logger.warning("Kill via control router", closed=count)
        return {"ok": True, "closed": count}

    def get_status(self) -> Dict[str, Any]:
        """Get system status (sync)."""
        if not self._engine:
            return {"status": "no_engine"}
        return {
            "status": "running" if self._engine._running else "stopped",
            "mode": getattr(self._engine, "mode", "unknown"),
            "paused": getattr(self._engine, "_trading_paused", False),
            "uptime_seconds": (
                time.time() - self._engine._start_time
                if getattr(self._engine, "_start_time", None)
                else 0
            ),
            "scan_count": getattr(self._engine, "_scan_count", 0),
            "pairs": getattr(self._engine, "pairs", []),
            "scan_interval": getattr(self._engine, "scan_interval", 0),
            "ws_connected": (
                self._engine.ws_client.is_connected
                if getattr(self._engine, "ws_client", None) else False
            ),
        }

    async def get_pnl(self) -> Dict[str, Any]:
        """Get performance stats and risk (P&L view)."""
        if not self._engine or not self._engine.db:
            return {}
        stats = await self._engine.db.get_performance_stats(
            tenant_id=getattr(self._engine, "tenant_id", "default")
        )
        risk = self._engine.risk_manager.get_risk_report()
        return {**stats, **risk}

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get open positions with current prices."""
        if not self._engine or not self._engine.db:
            return []
        positions = await self._engine.db.get_open_trades(
            tenant_id=getattr(self._engine, "tenant_id", "default")
        )
        for pos in positions:
            cp = self._engine.market_data.get_latest_price(pos.get("pair", ""))
            pos["current_price"] = cp
        return positions

    def get_risk(self) -> Dict[str, Any]:
        """Get risk report."""
        if not self._engine:
            return {}
        return self._engine.risk_manager.get_risk_report()
