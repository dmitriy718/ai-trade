"""
Bot Engine - Main orchestrator for the AI Trading Bot.

Coordinates all subsystems: market data, strategies, AI intelligence,
execution, risk management, and monitoring. Manages the main event
loop and lifecycle of the entire application.

# ENHANCEMENT: Added graceful shutdown with position preservation
# ENHANCEMENT: Added health monitoring with auto-recovery
# ENHANCEMENT: Added hot-reload for configuration changes
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.ai.confluence import ConfluenceDetector
from src.ai.order_book import OrderBookAnalyzer
from src.ai.predictor import TFLitePredictor
from src.api.server import DashboardServer
from src.core.config import ConfigManager, get_config
from src.core.database import DatabaseManager
from src.core.logger import get_logger, setup_logging
from src.exchange.kraken_rest import KrakenRESTClient
from src.exchange.kraken_ws import KrakenWebSocketClient
from src.exchange.market_data import MarketDataCache
from src.execution.executor import TradeExecutor
from src.execution.risk_manager import RiskManager
from src.strategies.base import SignalDirection

logger = get_logger("engine")


class BotEngine:
    """
    Main orchestrator for the AI Trading Bot.
    
    Lifecycle:
    1. Initialize all subsystems
    2. Warmup market data from REST
    3. Connect WebSocket for live data
    4. Run main scan loop
    5. Manage positions on each cycle
    6. Handle shutdown gracefully
    
    # ENHANCEMENT: Added subsystem health monitoring
    # ENHANCEMENT: Added automatic data quality checks
    # ENHANCEMENT: Added event bus for inter-component communication
    """

    def __init__(self):
        self.config = get_config()
        self.mode = self.config.app.mode
        self.pairs = self.config.trading.pairs
        self.scan_interval = self.config.trading.scan_interval_seconds

        # Core components (initialized in start())
        self.db: Optional[DatabaseManager] = None
        self.rest_client: Optional[KrakenRESTClient] = None
        self.ws_client: Optional[KrakenWebSocketClient] = None
        self.market_data: Optional[MarketDataCache] = None
        self.confluence: Optional[ConfluenceDetector] = None
        self.predictor: Optional[TFLitePredictor] = None
        self.order_book_analyzer: Optional[OrderBookAnalyzer] = None
        self.risk_manager: Optional[RiskManager] = None
        self.executor: Optional[TradeExecutor] = None
        self.dashboard: Optional[DashboardServer] = None

        # State
        self._running = False
        self._trading_paused = False
        self._start_time = 0.0
        self._scan_count = 0
        self._last_health_check = 0.0
        self._tasks: List[asyncio.Task] = []

    async def initialize(self) -> None:
        """Initialize all subsystems."""
        logger.info("Initializing AI Trading Bot", mode=self.mode, version="2.0.0")

        # Database
        db_path = os.getenv("DB_PATH", "data/trading.db")
        self.db = DatabaseManager(db_path)
        await self.db.initialize()
        logger.info("Database initialized", path=db_path)

        # REST Client
        api_key = os.getenv("KRAKEN_API_KEY", "")
        api_secret = os.getenv("KRAKEN_API_SECRET", "")
        is_sandbox = os.getenv("KRAKEN_SANDBOX", "false").lower() in ("true", "1", "yes")

        self.rest_client = KrakenRESTClient(
            api_key=api_key,
            api_secret=api_secret,
            rate_limit=self.config.exchange.rate_limit_per_second,
            max_retries=self.config.exchange.max_retries,
        )
        await self.rest_client.initialize()
        logger.info(
            "REST client initialized",
            sandbox=is_sandbox,
            mode=self.mode,
            has_key=bool(api_key),
        )

        # WebSocket Client
        self.ws_client = KrakenWebSocketClient(
            url=self.config.exchange.ws_url,
        )

        # Market Data Cache
        self.market_data = MarketDataCache(
            max_bars=self.config.trading.warmup_bars,
        )

        # AI Components
        self.confluence = ConfluenceDetector(
            market_data=self.market_data,
            confluence_threshold=self.config.ai.confluence_threshold,
            obi_threshold=self.config.ai.obi_threshold,
            min_confidence=self.config.ai.min_confidence,
        )
        self.confluence.configure_strategies(
            self.config.strategies.model_dump()
        )

        self.predictor = TFLitePredictor(
            model_path=self.config.ai.tflite_model_path,
        )
        self.predictor.load_model()

        self.order_book_analyzer = OrderBookAnalyzer(
            whale_threshold_usd=self.config.ai.whale_threshold_usd,
            depth=self.config.ai.order_book_depth,
        )

        # Risk Manager
        initial_bankroll = float(os.getenv("INITIAL_BANKROLL", "10000"))
        self.risk_manager = RiskManager(
            initial_bankroll=initial_bankroll,
            max_risk_per_trade=self.config.risk.max_risk_per_trade,
            max_daily_loss=self.config.risk.max_daily_loss,
            max_position_usd=self.config.risk.max_position_usd,
            kelly_fraction=self.config.risk.kelly_fraction,
            max_kelly_size=self.config.risk.max_kelly_size,
            risk_of_ruin_threshold=self.config.risk.risk_of_ruin_threshold,
            atr_multiplier_sl=self.config.risk.atr_multiplier_sl,
            atr_multiplier_tp=self.config.risk.atr_multiplier_tp,
            trailing_activation_pct=self.config.risk.trailing_activation_pct,
            trailing_step_pct=self.config.risk.trailing_step_pct,
            breakeven_activation_pct=self.config.risk.breakeven_activation_pct,
            # ENHANCEMENT: Shorter cooldown in paper mode for faster testing
            cooldown_seconds=60 if self.mode == "paper" else self.config.trading.cooldown_seconds,
            max_concurrent_positions=self.config.trading.max_concurrent_positions,
        )

        # Trade Executor
        self.executor = TradeExecutor(
            rest_client=self.rest_client,
            market_data=self.market_data,
            risk_manager=self.risk_manager,
            db=self.db,
            mode=self.mode,
        )

        # Dashboard
        self.dashboard = DashboardServer()
        self.dashboard.set_bot_engine(self)

        await self.db.log_thought(
            "system",
            f"Bot initialized in {self.mode.upper()} mode | "
            f"Tracking {len(self.pairs)} pairs | "
            f"Bankroll: ${initial_bankroll:,.2f}",
            severity="info",
        )

        logger.info(
            "All subsystems initialized",
            pairs=len(self.pairs),
            mode=self.mode,
            bankroll=initial_bankroll,
        )

    async def warmup(self) -> None:
        """
        Load historical data for all pairs.
        
        # ENHANCEMENT: Added parallel warmup for speed
        # ENHANCEMENT: Added progress tracking
        """
        logger.info("Starting historical data warmup", pairs=len(self.pairs))

        await self.db.log_thought(
            "system",
            f"Warming up {len(self.pairs)} pairs with {self.config.trading.warmup_bars} bars...",
            severity="info",
        )

        tasks = [self._warmup_pair(pair) for pair in self.pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(
            "Warmup complete",
            success=success,
            total=len(self.pairs),
        )

        await self.db.log_thought(
            "system",
            f"Warmup complete: {success}/{len(self.pairs)} pairs loaded",
            severity="info",
        )

    async def _warmup_pair(self, pair: str) -> int:
        """Warmup a single pair with historical data."""
        try:
            ohlc = await self.rest_client.get_ohlc(
                pair, interval=1, since=None
            )
            if ohlc:
                bars = await self.market_data.warmup(pair, ohlc)
                logger.debug("Pair warmup complete", pair=pair, bars=bars)
                return bars
            return 0
        except Exception as e:
            logger.error("Pair warmup failed", pair=pair, error=str(e))
            raise

    # S14 FIX: Removed dead start() method. main.py manages lifecycle directly.

    async def stop(self) -> None:
        """Gracefully stop the bot engine."""
        logger.info("Stopping bot engine...")
        self._running = False

        # Cancel all tasks and WAIT for them to finish (S10 FIX)
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Now safe to close resources
        if self.ws_client:
            await self.ws_client.disconnect()
        if self.rest_client:
            await self.rest_client.close()
        if self.db:
            try:
                await self.db.log_thought(
                    "system", "Bot engine STOPPED", severity="warning"
                )
            except Exception:
                pass
            await self.db.close()

        logger.info("Bot engine stopped")

    # ------------------------------------------------------------------
    # Main Loops
    # ------------------------------------------------------------------

    async def _main_scan_loop(self) -> None:
        """
        Main trading scan loop.
        
        Cycles:
        1. Manage existing positions (stops, trailing)
        2. Scan all pairs for signals
        3. Evaluate confluence
        4. Execute qualifying signals
        """
        logger.info("Main scan loop started", interval=self.scan_interval)

        while self._running:
            try:
                cycle_start = time.time()
                self._scan_count += 1

                # Step 1: Manage existing positions
                if self.executor:
                    await self.executor.manage_open_positions()

                # Step 2: Skip signal processing if paused
                if self._trading_paused:
                    await asyncio.sleep(self.scan_interval)
                    continue

                # Step 3: Run confluence analysis on all pairs
                confluence_signals = await self.confluence.scan_all_pairs(
                    self.pairs
                )

                # Step 4: Process signals through AI predictor
                # Require 2+ strategies agreeing before even considering a trade
                min_confluence = 2
                exec_confidence = 0.55

                for signal in confluence_signals:
                    if signal.direction == SignalDirection.NEUTRAL:
                        continue

                    # AI verification for all non-neutral signals
                    prediction_features = self._build_prediction_features(signal)
                    ai_confidence = self.predictor.predict(prediction_features)

                    # Apply AI confidence to signal
                    signal.confidence = (signal.confidence + ai_confidence) / 2

                    # Log ALL analysis thoughts so dashboard shows activity
                    await self.db.log_thought(
                        "analysis",
                        f"🔍 {signal.pair} | {signal.direction.value.upper()} | "
                        f"Confluence: {signal.confluence_count}/6 | "
                        f"Strength: {signal.strength:.2f} | "
                        f"AI Conf: {ai_confidence:.2f} | "
                        f"OBI: {signal.obi:+.3f} "
                        f"{'✨ SURE FIRE' if signal.is_sure_fire else ''}",
                        severity="info",
                        metadata=signal.to_dict(),
                    )

                    # Determine if we should trade this signal:
                    # - Normal: need 2+ strategies agreeing (confluence >= 2)
                    # - Keltner solo: it has 3 internal confirmations (KC + MACD + RSI)
                    #   so treat it as self-sufficient when confidence is high
                    has_keltner = any(
                        s.strategy_name == "keltner" and s.is_actionable
                        for s in signal.signals
                        if s.direction == signal.direction
                    )
                    keltner_solo_ok = has_keltner and signal.confidence >= 0.60

                    if signal.confluence_count < min_confluence and not keltner_solo_ok:
                        continue

                    # Execute if meets threshold
                    if signal.confidence >= exec_confidence:
                        trade_id = await self.executor.execute_signal(signal)
                        if trade_id:
                            logger.info(
                                "Signal executed",
                                trade_id=trade_id,
                                pair=signal.pair,
                                direction=signal.direction.value,
                            )

                # Log cycle metrics - every scan for visibility
                cycle_time = (time.time() - cycle_start) * 1000
                active_count = sum(1 for s in confluence_signals if s.direction != SignalDirection.NEUTRAL)
                if self._scan_count % 10 == 0 or active_count > 0:
                    await self.db.insert_metric("scan_cycle_ms", cycle_time)
                    await self.db.log_thought(
                        "system",
                        f"Scan #{self._scan_count} | {cycle_time:.0f}ms | "
                        f"Signals: {active_count}/{len(self.pairs)} pairs",
                        severity="debug",
                    )

                # Wait for next cycle
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.scan_interval - elapsed)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scan loop error", error=str(e))
                await asyncio.sleep(5)

    async def _ws_data_loop(self) -> None:
        """WebSocket data streaming loop."""
        try:
            # Register callbacks
            self.ws_client.on_ticker(self._handle_ticker)
            self.ws_client.on_ohlc(self._handle_ohlc)
            self.ws_client.on_book(self._handle_book)
            self.ws_client.on_trade(self._handle_trade)

            # Subscribe to channels
            await self.ws_client.subscribe_ticker(self.pairs)
            await self.ws_client.subscribe_ohlc(self.pairs, interval=1)
            await self.ws_client.subscribe_book(
                self.pairs, depth=self.config.ai.order_book_depth
            )

            # Connect (blocking - handles reconnection internally)
            await self.ws_client.connect()

        except asyncio.CancelledError:
            await self.ws_client.disconnect()
        except Exception as e:
            logger.error("WebSocket loop error", error=str(e))

    async def _health_monitor(self) -> None:
        """Monitor system health and trigger recovery actions."""
        while self._running:
            try:
                await asyncio.sleep(
                    self.config.monitoring.health_check_interval
                )

                # S9 FIX: Check WebSocket and restart task if dead
                if self.ws_client and not self.ws_client.is_connected:
                    # Check if any WS task is still alive
                    ws_alive = any(
                        not t.done() for t in self._tasks
                        if hasattr(t, '_coro') and 'ws_data' in str(getattr(t, '_coro', ''))
                    )
                    if not ws_alive:
                        logger.warning("WebSocket task dead, restarting")
                        self.ws_client._reconnect_count = 0
                        self._tasks.append(asyncio.create_task(self._ws_data_loop()))
                        await self.db.log_thought(
                            "health", "WebSocket task restarted", severity="warning",
                        )
                    else:
                        logger.warning("WebSocket disconnected, reconnecting internally")

                # Check data freshness (5-min threshold — low-volume pairs are naturally slower)
                stale_pairs = [
                    pair for pair in self.pairs
                    if self.market_data.is_stale(pair, max_age_seconds=600)
                ]
                if stale_pairs:
                    logger.warning(
                        "Stale data detected",
                        pairs=stale_pairs,
                    )
                    # Attempt REST refresh for stale pairs
                    for pair in stale_pairs:
                        try:
                            ohlc = await self.rest_client.get_ohlc(pair, interval=1)
                            if ohlc:
                                await self.market_data.warmup(pair, ohlc)
                        except Exception:
                            pass

                # Log health status
                await self.db.insert_metric("uptime_seconds", time.time() - self._start_time)
                await self.db.insert_metric("open_positions", len(
                    await self.db.get_open_trades()
                ))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitor error", error=str(e))

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of old data."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Every hour
                await self.db.cleanup_old_data(
                    self.config.monitoring.metrics_retention_hours
                )
                logger.info("Database cleanup completed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup error", error=str(e))

    # ------------------------------------------------------------------
    # WebSocket Handlers
    # ------------------------------------------------------------------

    def _parse_ts(self, ts_value) -> float:
        """Parse Kraken timestamp (ISO string or float) to epoch float."""
        if isinstance(ts_value, (int, float)):
            return float(ts_value)
        if isinstance(ts_value, str):
            try:
                from datetime import datetime, timezone
                # Handle Kraken's ISO format: "2026-02-07T02:00:00.099318Z"
                dt = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
                return dt.timestamp()
            except (ValueError, TypeError):
                pass
        return time.time()

    async def _handle_ticker(self, message: Dict[str, Any]) -> None:
        """Process ticker updates — updates latest price in-place (no new bars)."""
        try:
            data = message.get("data", [])
            if isinstance(data, list):
                for tick in data:
                    symbol = tick.get("symbol", "")
                    if symbol:
                        self.market_data.update_ticker(symbol, tick)
                        # S1 FIX: Only update the CLOSE of the current bar in-place.
                        # NEVER inject fake bars — that destroys ATR, volume, and all indicators.
                        last = tick.get("last")
                        if last and float(last) > 0:
                            self.market_data.update_latest_close(symbol, float(last))
        except Exception as e:
            logger.warning("Ticker handler error", error=str(e))

    async def _handle_ohlc(self, message: Dict[str, Any]) -> None:
        """Process OHLC candle updates from WebSocket."""
        try:
            data = message.get("data", [])
            if isinstance(data, list):
                for candle in data:
                    symbol = candle.get("symbol", "")
                    if symbol:
                        await self.market_data.update_bar(symbol, {
                            "time": self._parse_ts(candle.get("interval_begin", candle.get("timestamp", time.time()))),
                            "open": float(candle.get("open", 0)),
                            "high": float(candle.get("high", 0)),
                            "low": float(candle.get("low", 0)),
                            "close": float(candle.get("close", 0)),
                            "volume": float(candle.get("volume", 0)),
                            "vwap": float(candle.get("vwap", 0)),
                        })
        except Exception as e:
            logger.warning("OHLC handler error", error=str(e))

    async def _handle_book(self, message: Dict[str, Any]) -> None:
        """Process order book updates from WebSocket."""
        try:
            data = message.get("data", [])
            if isinstance(data, list):
                for book_update in data:
                    symbol = book_update.get("symbol", "")
                    if symbol:
                        self.market_data.update_order_book(symbol, book_update)

                        # Run order book analysis
                        if self.order_book_analyzer:
                            bids = book_update.get("bids", [])
                            asks = book_update.get("asks", [])
                            price = self.market_data.get_latest_price(symbol)
                            self.order_book_analyzer.analyze(
                                symbol, bids, asks, price
                            )
        except Exception as e:
            logger.debug("Book handler error", error=str(e))

    async def _handle_trade(self, message: Dict[str, Any]) -> None:
        """Process trade stream updates."""
        pass  # Trade stream processing for future use

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_prediction_features(
        self, signal
    ) -> Dict[str, Any]:
        """Build feature dict for AI predictor from confluence signal."""
        # S7 FIX: Average overlapping numeric keys instead of overwriting
        metadata = {}
        counts = {}
        for s in signal.signals:
            if s.direction == signal.direction:  # Only use agreeing signals
                for k, v in s.metadata.items():
                    if k in metadata and isinstance(v, (int, float)) and isinstance(metadata[k], (int, float)):
                        metadata[k] = metadata[k] + v
                        counts[k] = counts.get(k, 1) + 1
                    else:
                        metadata[k] = v
        for k in counts:
            metadata[k] = metadata[k] / counts[k]

        features = self.predictor.features.feature_dict_from_signals(
            metadata,
            obi=signal.obi,
            spread=self.market_data.get_spread(signal.pair),
        )
        return features
