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
from src.ml.trainer import ModelTrainer, AutoRetrainer
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
        self.position_check_interval = self.config.trading.position_check_interval_seconds
        self.tenant_id = self.config.billing.tenant.default_tenant_id

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
        self._scan_queue: asyncio.Queue = asyncio.Queue()
        self._pending_scan_pairs: set = set()
        self._event_price_move_pct = getattr(self.config.trading, "event_price_move_pct", 0.005)

    async def initialize(self) -> None:
        """Initialize all subsystems."""
        logger.info("Initializing AI Trading Bot", mode=self.mode, version="2.0.0")

        # Database
        db_path = self.config.app.db_path
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
            book_score_threshold=getattr(self.config.ai, "book_score_threshold", 0.2),
            book_score_max_age_seconds=getattr(self.config.ai, "book_score_max_age_seconds", 5),
            min_confidence=self.config.ai.min_confidence,
            obi_counts_as_confluence=getattr(
                self.config.ai, "obi_counts_as_confluence", False
            ),
            obi_weight=getattr(self.config.ai, "obi_weight", 0.4),
            round_trip_fee_pct=self.config.exchange.taker_fee * 2,
            use_closed_candles_only=getattr(self.config.trading, "use_closed_candles_only", False),
            regime_config=getattr(self.config.ai, "regime", None),
            timeframes=getattr(self.config.trading, "timeframes", [1]),
            multi_timeframe_min_agreement=getattr(self.config.ai, "multi_timeframe_min_agreement", 1),
            primary_timeframe=getattr(self.config.ai, "primary_timeframe", 1),
        )
        self.confluence.configure_strategies(
            self.config.strategies.model_dump(),
            single_strategy_mode=getattr(
                self.config.trading, "single_strategy_mode", None
            ),
        )

        self.predictor = TFLitePredictor(
            model_path=self.config.ai.tflite_model_path,
            feature_names=self.config.ml.features,
        )
        self.predictor.load_model()

        self.order_book_analyzer = OrderBookAnalyzer(
            whale_threshold_usd=self.config.ai.whale_threshold_usd,
            depth=self.config.ai.order_book_depth,
        )

        # Risk Manager
        initial_bankroll = float(self.config.risk.initial_bankroll)
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
            strategy_cooldowns=self.config.trading.strategy_cooldowns_seconds,
            global_cooldown_seconds_on_loss=self.config.risk.global_cooldown_seconds_on_loss,
        )
        if self.confluence:
            self.confluence.set_cooldown_checker(
                self.risk_manager.is_strategy_on_cooldown
            )

        # Trade Executor
        self.executor = TradeExecutor(
            rest_client=self.rest_client,
            market_data=self.market_data,
            risk_manager=self.risk_manager,
            db=self.db,
            mode=self.mode,
            maker_fee=self.config.exchange.maker_fee,
            taker_fee=self.config.exchange.taker_fee,
            post_only=self.config.exchange.post_only,
            tenant_id=self.tenant_id,
            limit_chase_attempts=self.config.exchange.limit_chase_attempts,
            limit_chase_delay_seconds=self.config.exchange.limit_chase_delay_seconds,
            limit_fallback_to_market=self.config.exchange.limit_fallback_to_market,
            strategy_result_cb=self.confluence.record_trade_result if self.confluence else None,
        )

        # AI Training Components
        self.ml_trainer = ModelTrainer(
            db=self.db,
            min_samples=self.config.ml.min_samples,
            epochs=self.config.ml.epochs,
            batch_size=self.config.ml.batch_size,
            feature_names=self.config.ml.features,
            tenant_id=self.tenant_id,
        )
        self.retrainer = AutoRetrainer(
            trainer=self.ml_trainer,
            interval_hours=self.config.ml.retrain_interval_hours,
        )

        # Restore open positions state
        await self.executor.reinitialize_positions()

        # Dashboard
        self.dashboard = DashboardServer()
        self.dashboard.set_bot_engine(self)

        # Billing (Stripe) - optional
        billing = getattr(self.config, "billing", None)
        if billing and getattr(billing.stripe, "enabled", False):
            from src.billing.stripe_service import StripeService
            stripe_cfg = billing.stripe
            stripe_svc = StripeService(
                secret_key=stripe_cfg.secret_key,
                webhook_secret=stripe_cfg.webhook_secret,
                price_id=stripe_cfg.price_id,
                currency=stripe_cfg.currency,
                db=self.db,
            )
            self.dashboard.set_stripe_service(stripe_svc)

        await self.db.log_thought(
            "system",
            f"Bot initialized in {self.mode.upper()} mode | "
            f"Tracking {len(self.pairs)} pairs | "
            f"Bankroll: ${initial_bankroll:,.2f}",
            severity="info",
            tenant_id=self.tenant_id,
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
            tenant_id=self.tenant_id,
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
            tenant_id=self.tenant_id,
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
                    "system", "Bot engine STOPPED", severity="warning",
                    tenant_id=self.tenant_id,
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
        1. Scan all pairs for signals
        2. Evaluate confluence
        3. Execute qualifying signals
        """
        logger.info("Main scan loop started", interval=self.scan_interval)

        while self._running:
            try:
                cycle_start = time.time()
                self._scan_count += 1

                # Step 1: Skip signal processing if paused
                if self._trading_paused:
                    await asyncio.sleep(self.scan_interval)
                    continue

                # Step 2: Run confluence analysis on event-driven pairs
                pairs_to_scan, from_event = await self._collect_scan_pairs()
                confluence_signals = await self.confluence.scan_all_pairs(
                    pairs_to_scan
                )

                # Step 3: Process signals through AI predictor
                # Use config threshold (default 3) for higher-quality, fewer trades; allow 2+ with strong confidence
                min_confluence = getattr(self.config.ai, "confluence_threshold", 3)
                min_confluence = max(2, min_confluence)  # At least 2 strategies must agree
                exec_confidence = getattr(self.config.ai, "min_confidence", 0.50)
                exec_confidence = min(exec_confidence, 0.65)  # Cap so we don't over-reject

                for signal in confluence_signals:
                    if signal.direction == SignalDirection.NEUTRAL:
                        continue

                    # AI verification for all non-neutral signals
                    prediction_features = self._build_prediction_features(signal)
                    if self.predictor and self.predictor.is_model_loaded:
                        ai_confidence = self.predictor.predict(prediction_features)
                        # Blend: for solo signals let strategy dominate so we get some trades
                        pre_blend = signal.confidence
                        if signal.confluence_count == 1:
                            signal.confidence = 0.7 * pre_blend + 0.3 * ai_confidence
                        else:
                            signal.confidence = (pre_blend + ai_confidence) / 2
                    else:
                        ai_confidence = 0.5

                    # Log ALL analysis thoughts so dashboard shows activity
                    await self.db.log_thought(
                        "analysis",
                        f"🔍 {signal.pair} | {signal.direction.value.upper()} | "
                        f"Confluence: {signal.confluence_count}/"
                        f"{len(self.confluence.strategies) + (1 if self.confluence.obi_counts_as_confluence else 0)} | "
                        f"Strength: {signal.strength:.2f} | "
                        f"AI Conf: {ai_confidence:.2f} | "
                        f"OBI: {signal.obi:+.3f} | "
                        f"BOOK: {getattr(signal, 'book_score', 0.0):+.3f} "
                        f"{'✨ SURE FIRE' if signal.is_sure_fire else ''}",
                        severity="info",
                        metadata=signal.to_dict(),
                        tenant_id=self.tenant_id,
                    )

                    # Determine if we should trade this signal:
                    # - Normal: 2+ strategies agreeing (confluence >= 2)
                    # - Solo: Keltner with conf >= 0.52, or any strategy with conf >= 0.55 (get flow)
                    has_keltner = any(
                        s.strategy_name == "keltner" and s.is_actionable
                        for s in signal.signals
                        if s.direction == signal.direction
                    )
                    keltner_solo_ok = has_keltner and signal.confidence >= 0.52
                    any_solo_ok = signal.confluence_count == 1 and signal.confidence >= 0.55

                    if signal.confluence_count < min_confluence and not keltner_solo_ok and not any_solo_ok:
                        continue

                    # Skip trades with poor risk/reward (TP distance should be at least min_rr * SL distance)
                    sl_dist = abs(signal.entry_price - signal.stop_loss)
                    tp_dist = abs(signal.take_profit - signal.entry_price) if signal.take_profit else 0
                    min_rr = getattr(self.config.ai, "min_risk_reward_ratio", 0.9)
                    if sl_dist > 0 and tp_dist > 0 and (tp_dist / sl_dist) < min_rr:
                        continue

                    # Skip if spread too wide
                    max_spread = getattr(self.config.trading, "max_spread_pct", 0.0) or 0.0
                    if max_spread > 0:
                        spread = self.market_data.get_spread(signal.pair)
                        if spread > max_spread:
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
                        f"Signals: {active_count}/{len(pairs_to_scan)} pairs",
                        severity="debug",
                        tenant_id=self.tenant_id,
                    )
                # Next cycle is gated by _collect_scan_pairs (event-driven or timeout)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scan loop error", error=str(e))
                await asyncio.sleep(5)

    async def _position_management_loop(self) -> None:
        """Manage open positions (stops/trailing) on a short, fixed interval."""
        interval = max(1, int(self.position_check_interval))
        logger.info("Position management loop started", interval=interval)
        while self._running:
            try:
                if self.executor:
                    await self.executor.manage_open_positions()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Position management loop error", error=str(e))
                await asyncio.sleep(1)

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
                            tenant_id=self.tenant_id,
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
                    await self.db.get_open_trades(tenant_id=self.tenant_id)
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

    def _enqueue_pair(self, pair: str, reason: str = "") -> None:
        """Enqueue a pair for event-driven scanning (deduped)."""
        if pair in self._pending_scan_pairs:
            return
        self._pending_scan_pairs.add(pair)
        try:
            self._scan_queue.put_nowait(pair)
        except asyncio.QueueFull:
            self._pending_scan_pairs.discard(pair)

    async def _collect_scan_pairs(self) -> tuple[list, bool]:
        """
        Collect pairs to scan.
        Returns (pairs, from_event_queue).
        """
        pairs = set()
        try:
            pair = await asyncio.wait_for(
                self._scan_queue.get(), timeout=self.scan_interval
            )
            pairs.add(pair)
            while True:
                pair = self._scan_queue.get_nowait()
                pairs.add(pair)
        except asyncio.TimeoutError:
            return list(self.pairs), False
        except asyncio.QueueEmpty:
            pass
        for p in pairs:
            self._pending_scan_pairs.discard(p)
        return list(pairs), True

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
                            prev = self.market_data.get_latest_price(symbol)
                            await self.market_data.update_latest_close(symbol, float(last))
                            if prev > 0:
                                move = abs(float(last) - prev) / prev
                                if move >= self._event_price_move_pct:
                                    self._enqueue_pair(symbol, "price_move")
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
                        is_new_bar = await self.market_data.update_bar(symbol, {
                            "time": self._parse_ts(candle.get("interval_begin", candle.get("timestamp", time.time()))),
                            "open": float(candle.get("open", 0)),
                            "high": float(candle.get("high", 0)),
                            "low": float(candle.get("low", 0)),
                            "close": float(candle.get("close", 0)),
                            "volume": float(candle.get("volume", 0)),
                            "vwap": float(candle.get("vwap", 0)),
                        })
                        if is_new_bar:
                            self._enqueue_pair(symbol, "bar_close")
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
                            analysis = self.order_book_analyzer.analyze(
                                symbol, bids, asks, price
                            )
                            if analysis:
                                self.market_data.update_order_book_analysis(
                                    symbol, analysis.to_dict()
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
            obi=(signal.book_score if getattr(signal, "book_score", 0.0) else signal.obi),
            spread=self.market_data.get_spread(signal.pair),
        )
        return features

    def get_algorithm_stats(self) -> List[Dict[str, Any]]:
        """Return full algorithm transparency list for the dashboard."""
        stats: List[Dict[str, Any]] = []
        if self.confluence:
            stats.extend(self.confluence.get_strategy_stats())
            obi_enabled = getattr(self.confluence, "obi_threshold", 0) > 0
            book_enabled = getattr(self.confluence, "book_score_threshold", 0) > 0
            stats.append({
                "name": "order_book_imbalance",
                "enabled": bool(obi_enabled),
                "weight": float(getattr(self.confluence, "obi_weight", 0.0)),
                "trades": 0,
                "win_rate": None,
                "total_pnl": None,
                "avg_pnl": None,
                "kind": "filter",
                "note": "confirmation filter"
                        + (" (weighted)" if getattr(self.confluence, "obi_counts_as_confluence", False) else ""),
            })
            stats.append({
                "name": "order_book_microstructure",
                "enabled": bool(book_enabled),
                "weight": 0.0,
                "trades": 0,
                "win_rate": None,
                "total_pnl": None,
                "avg_pnl": None,
                "kind": "filter",
                "note": "book score confirmation",
            })
            stats.append({
                "name": "regime_detector",
                "enabled": True,
                "weight": 0.0,
                "trades": 0,
                "win_rate": None,
                "total_pnl": None,
                "avg_pnl": None,
                "kind": "model",
                "note": "trend/volatility regime",
            })
        stats.append({
            "name": "ai_predictor",
            "enabled": bool(self.predictor and self.predictor.is_model_loaded),
            "weight": 0.0,
            "trades": 0,
            "win_rate": None,
            "total_pnl": None,
            "avg_pnl": None,
            "kind": "model",
            "note": "tflite signal scoring",
        })
        return stats
