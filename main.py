#!/usr/bin/env python3
"""
AI Trading Bot - Main Entry Point

H5 FIX: Single clean lifecycle - main.py owns init/run/shutdown.
L4 FIX: preflight_checks returns False on critical failures.
L5 FIX: traceback imported at module level.
L2 FIX: Uses get_running_loop() instead of deprecated get_event_loop().
"""

from __future__ import annotations

import asyncio
import signal as sig
import sys
import time
import traceback
from pathlib import Path

import uvicorn

from src.core.config import ConfigManager
from src.core.engine import BotEngine
from src.core.logger import get_logger, setup_logging


def preflight_checks() -> bool:
    """Run pre-flight system checks before startup."""
    ok = True

    # Create required directories
    for directory in ["data", "logs", "models", "config"]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Warn about missing config
    if not Path("config/config.yaml").exists():
        print("[WARN] config/config.yaml not found, using defaults")

    # Check .env exists
    if not Path(".env").exists():
        if Path(".env.example").exists():
            print("[WARN] No .env file. Copy .env.example -> .env and configure.")
            ok = False
        else:
            print("[WARN] No .env or .env.example found")
            ok = False

    return ok


async def run_bot():
    """Initialize and run the bot engine with dashboard server."""
    engine = BotEngine()

    # Phase 1: Initialize all subsystems
    await engine.initialize()
    await engine.warmup()

    # Phase 2: Start uvicorn dashboard (without its own signal handlers)
    uvi_config = uvicorn.Config(
        app=engine.dashboard.app,
        host=engine.config.dashboard.host,
        port=engine.config.dashboard.port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(uvi_config)
    server.install_signal_handlers = lambda: None

    # Phase 3: Setup state and signal handling
    engine._running = True
    engine._start_time = time.time()

    shutdown_event = asyncio.Event()

    def _request_shutdown():
        engine._running = False
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for s in (sig.SIGINT, sig.SIGTERM):
        loop.add_signal_handler(s, _request_shutdown)

    await engine.db.log_thought(
        "system",
        "Bot engine STARTED - All systems operational",
        severity="info",
        tenant_id=engine.tenant_id,
    )

    async def _run_with_logging(coro, name):
        try:
            await coro
        except Exception as e:
            logger.error(f"Background task {name} failed", error=str(e), traceback=traceback.format_exc())

    # Phase 4: Start background tasks
    engine._tasks = [
        asyncio.create_task(_run_with_logging(engine._main_scan_loop(), "scan_loop")),
        asyncio.create_task(_run_with_logging(engine._position_management_loop(), "position_loop")),
        asyncio.create_task(_run_with_logging(engine._ws_data_loop(), "ws_loop")),
        asyncio.create_task(_run_with_logging(engine._health_monitor(), "health_monitor")),
        asyncio.create_task(_run_with_logging(engine._cleanup_loop(), "cleanup_loop")),
        asyncio.create_task(_run_with_logging(engine.retrainer.run(), "auto_retrainer")),
    ]
    server_task = asyncio.create_task(server.serve())

    # Phase 5: Wait for shutdown signal
    await shutdown_event.wait()

    # Phase 6: Graceful shutdown
    logger = get_logger("main")
    logger.info("Shutdown signal received, cleaning up...")
    await engine.stop()
    server.should_exit = True

    try:
        await asyncio.wait_for(server_task, timeout=5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass


def main():
    """Main entry point."""
    preflight_checks()  # Warn but don't block for paper mode

    # Setup logging before any engine imports use loggers
    config = ConfigManager()
    setup_logging(
        log_level=config.config.app.log_level,
        log_dir="logs",
    )

    logger = get_logger("main")
    logger.info(
        "Starting AI Trading Bot",
        version="2.0.0",
        python=sys.version,
        mode=config.config.app.mode,
    )

    print("""
    ╔══════════════════════════════════════════════╗
    ║        AI CRYPTO TRADING BOT v2.0.0          ║
    ║   Multi-Strategy • AI-Powered • Self-Learning ║
    ╚══════════════════════════════════════════════╝
    """)

    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("Shutdown requested via keyboard interrupt")
    except Exception as e:
        traceback.print_exc()
        logger.critical("Fatal error", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
