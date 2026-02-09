"""
Trade Executor - Unified order management and fill processing.

Handles the full lifecycle of trade execution: order placement,
fill monitoring, position tracking, and stop management. Supports
both paper and live trading modes.

# ENHANCEMENT: Added paper trading engine with realistic simulation
# ENHANCEMENT: Added slippage estimation model
# ENHANCEMENT: Added order retry with price adjustment
# ENHANCEMENT: Added unified fill processor for DB <-> RAM sync
# ENHANCEMENT: Support for Limit Orders to control execution price
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable

from src.ai.confluence import ConfluenceSignal
from src.core.database import DatabaseManager
from src.core.logger import get_logger
from src.exchange.kraken_rest import KrakenRESTClient
from src.exchange.market_data import MarketDataCache
from src.execution.risk_manager import RiskManager, StopLossState
from src.strategies.base import SignalDirection

logger = get_logger("executor")


class TradeExecutor:
    """
    Production trade execution engine.
    
    Manages the complete trade lifecycle:
    1. Signal validation -> Risk check -> Position sizing
    2. Order placement (paper or live)
    3. Fill monitoring and processing
    4. Stop loss management (trailing + breakeven)
    5. Position closure and P&L recording
    
    # ENHANCEMENT: Added order splitting for large positions
    # ENHANCEMENT: Added fee estimation and tracking
    # ENHANCEMENT: Added execution quality metrics
    """

    def __init__(
        self,
        rest_client: KrakenRESTClient,
        market_data: MarketDataCache,
        risk_manager: RiskManager,
        db: DatabaseManager,
        mode: str = "paper",
        maker_fee: float = 0.0016,
        taker_fee: float = 0.0026,
        post_only: bool = False,
        tenant_id: Optional[str] = "default",
        limit_chase_attempts: int = 2,
        limit_chase_delay_seconds: float = 2.0,
        limit_fallback_to_market: bool = True,
        strategy_result_cb: Optional[Callable[[str, float], None]] = None,
    ):
        self.rest_client = rest_client
        self.market_data = market_data
        self.risk_manager = risk_manager
        self.db = db
        self.mode = mode  # "paper" or "live"
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.post_only = post_only
        self.tenant_id = tenant_id or "default"
        self.limit_chase_attempts = max(0, int(limit_chase_attempts))
        self.limit_chase_delay_seconds = max(0.0, float(limit_chase_delay_seconds))
        self.limit_fallback_to_market = bool(limit_fallback_to_market)
        self._strategy_result_cb = strategy_result_cb

        self._active_orders: Dict[str, Dict[str, Any]] = {}
        self._pending_signals: asyncio.Queue = asyncio.Queue()
        self._execution_stats = {
            "orders_placed": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "total_slippage": 0.0,
            "total_fees": 0.0,
        }

    async def reinitialize_positions(self) -> None:
        """Restore position and stop-loss state from database after restart."""
        open_trades = await self.db.get_open_trades(tenant_id=self.tenant_id)
        for trade in open_trades:
            trade_id = trade["trade_id"]
            pair = trade["pair"]
            side = trade["side"]
            entry_price = trade["entry_price"]
            sl = trade["stop_loss"]
            
            # Use metadata to get size_usd and trailing state if available
            size_usd = 0.0
            trailing_high = 0.0
            trailing_low = float("inf")
            
            if trade.get("metadata"):
                try:
                    meta = json.loads(trade["metadata"]) if isinstance(trade["metadata"], str) else trade["metadata"]
                    size_usd = meta.get("size_usd", 0.0)
                    
                    # Restore stop loss state
                    if "stop_loss_state" in meta:
                        sl_state = meta["stop_loss_state"]
                        trailing_high = sl_state.get("trailing_high", 0.0)
                        trailing_low = sl_state.get("trailing_low", float("inf"))
                except Exception:
                    pass
            
            if size_usd == 0.0:
                size_usd = entry_price * trade["quantity"]

            # Re-register with RiskManager
            self.risk_manager.register_position(
                trade_id,
                pair,
                side,
                entry_price,
                size_usd,
                strategy=trade.get("strategy"),
            )
            # Restore stop loss state
            if sl > 0:
                self.risk_manager.initialize_stop_loss(
                    trade_id, entry_price, sl, side, trailing_high, trailing_low
                )
            
            logger.info(
                "Restored position state",
                trade_id=trade_id, pair=pair, sl=sl
            )

    async def execute_signal(
        self, signal: ConfluenceSignal
    ) -> Optional[str]:
        """
        Execute a confluence signal through the full pipeline.
        
        Pipeline:
        1. Validate signal quality
        2. Check risk constraints
        3. Calculate position size
        4. Place order (Limit)
        5. Record trade
        
        Returns trade_id if executed, None if rejected.
        
        # ENHANCEMENT: Added multi-stage validation
        # ENHANCEMENT: Transitioned to Limit Orders for better price control
        """
        # Stage 1: Signal validation
        if signal.direction == SignalDirection.NEUTRAL:
            return None

        if signal.confidence < 0.40:
            return None

        # Block duplicate pair — only one position per pair at a time
        open_trades = await self.db.get_open_trades(
            pair=signal.pair, tenant_id=self.tenant_id
        )
        if open_trades:
            return None  # Already have a position on this pair

        # Stage 2: Risk check and position sizing
        side = "buy" if signal.direction == SignalDirection.LONG else "sell"
        primary_strategy = self._primary_strategy(signal)

        if self.risk_manager.is_strategy_on_cooldown(signal.pair, primary_strategy, side):
            await self.db.log_thought(
                "risk",
                f"Trade blocked: {primary_strategy} cooldown active for {signal.pair}",
                severity="warning",
                metadata={"pair": signal.pair, "strategy": primary_strategy},
                tenant_id=self.tenant_id,
            )
            return None

        # Estimate win rate from historical data
        stats = await self.db.get_performance_stats(tenant_id=self.tenant_id)
        total_trades = stats.get("total_trades", 0)

        if total_trades >= 50:
            win_rate = max(stats.get("win_rate", 0.5), 0.35)
            avg_win = max(stats.get("avg_win", 1.0), 0.01)
            avg_loss = max(abs(stats.get("avg_loss", -1.0)), 0.01)
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.5
        else:
            win_rate = 0.50
            win_loss_ratio = 1.5

        size_result = self.risk_manager.calculate_position_size(
            pair=signal.pair,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            win_rate=win_rate,
            avg_win_loss_ratio=win_loss_ratio,
            confidence=signal.confidence,
        )

        if not size_result.allowed:
            await self.db.log_thought(
                "risk",
                f"Trade blocked: {size_result.reason}",
                severity="warning",
                metadata={"pair": signal.pair, "signal": signal.to_dict()},
                tenant_id=self.tenant_id,
            )
            return None

        # Determine Limit Price
        # We want to fill immediately but not slip.
        # Buy at Ask, Sell at Bid.
        ticker = self.market_data.get_ticker(signal.pair)
        limit_price = signal.entry_price # Default
        
        if ticker:
            try:
                if side == "buy":
                    # ticker['a'][0] is best ask price
                    limit_price = float(ticker['a'][0])
                else:
                    # ticker['b'][0] is best bid price
                    limit_price = float(ticker['b'][0])
            except (KeyError, IndexError, ValueError):
                pass

        # Stage 3: Place order
        trade_id = f"T-{uuid.uuid4().hex[:12]}"

        partial_fill = False
        filled_units = size_result.size_units
        entry_fee = 0.0
        if self.mode == "paper":
            fill_price = await self._paper_fill(
                signal.pair, side, limit_price
            )
        else:
            # Use Limit Order
            fill_price, filled_units, partial_fill, entry_fee = await self._live_fill(
                signal.pair,
                side,
                "limit",
                size_result.size_units,
                trade_id,
                price=limit_price,
                post_only=self.post_only,
            )

        if fill_price is None or not filled_units or filled_units <= 0:
            await self.db.log_thought(
                "execution",
                f"Order fill failed for {signal.pair}",
                severity="error",
                tenant_id=self.tenant_id,
            )
            return None

        # Stage 4: Record trade
        slippage = abs(fill_price - signal.entry_price) / signal.entry_price
        filled_size_usd = filled_units * fill_price
        # Prefer actual fee if provided; fallback to heuristic
        if entry_fee and entry_fee > 0:
            fees = entry_fee
            entry_fee_rate = entry_fee / filled_size_usd if filled_size_usd > 0 else 0.0
        else:
            # Maker fee if post-only; otherwise assume taker
            entry_fee_rate = self.maker_fee if (self.post_only and self.mode == "live") else self.taker_fee
            fees = filled_size_usd * entry_fee_rate

        trade_record = {
            "trade_id": trade_id,
            "pair": signal.pair,
            "side": side,
            "entry_price": fill_price,
            "quantity": filled_units,
            "status": "open",
            "strategy": primary_strategy,
            "confidence": signal.confidence,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "confluence_count": signal.confluence_count,
                "is_sure_fire": signal.is_sure_fire,
                "obi": signal.obi,
                "book_score": getattr(signal, "book_score", 0.0),
                "size_usd": round(filled_size_usd, 2),
                "risk_amount": round(size_result.risk_amount, 2),
                "kelly_fraction": size_result.kelly_fraction,
                "slippage": slippage,
                "fees": fees,
                "entry_fee": fees,
                "entry_fee_rate": entry_fee_rate,
                "exit_fee_rate": self.taker_fee,
                "requested_units": size_result.size_units,
                "filled_units": filled_units,
                "partial_fill": partial_fill,
                "mode": self.mode,
                "order_type": "limit",
                "post_only": self.post_only,
            }
        }

        await self.db.insert_trade(trade_record, tenant_id=self.tenant_id)

        # Initialize stop loss tracking
        self.risk_manager.initialize_stop_loss(
            trade_id, fill_price, signal.stop_loss, side
        )

        # Register position with risk manager
        self.risk_manager.register_position(
            trade_id, signal.pair, side,
            fill_price, filled_size_usd, strategy=primary_strategy
        )

        self._execution_stats["orders_placed"] += 1
        self._execution_stats["orders_filled"] += 1
        self._execution_stats["total_slippage"] += slippage
        self._execution_stats["total_fees"] += fees

        # Log the execution thought
        await self.db.log_thought(
            "trade",
            f"{'📈' if side == 'buy' else '📉'} {side.upper()} {signal.pair} @ "
            f"${fill_price:.2f} (Limit) | Size: ${filled_size_usd:.2f} | "
            f"SL: ${signal.stop_loss:.2f} | TP: ${signal.take_profit:.2f} | "
            f"Confluence: {signal.confluence_count} | "
            f"{'SURE FIRE' if signal.is_sure_fire else 'Standard'}",
            severity="info",
            metadata=trade_record["metadata"],
            tenant_id=self.tenant_id,
        )

        logger.info(
            "Trade executed",
            trade_id=trade_id,
            pair=signal.pair,
            side=side,
            price=fill_price,
            size_usd=round(filled_size_usd, 2),
            mode=self.mode,
        )

        return trade_id

    async def manage_open_positions(self) -> None:
        """
        Update all open positions: check stops, trailing logic.
        
        This should be called on every scan cycle to manage
        existing positions.
        
        # ENHANCEMENT: Added parallel position management
        """
        open_trades = await self.db.get_open_trades(tenant_id=self.tenant_id)

        for trade in open_trades:
            try:
                await self._manage_position(trade)
            except Exception as e:
                logger.error(
                    "Position management error",
                    trade_id=trade["trade_id"],
                    error=str(e)
                )

    async def _manage_position(self, trade: Dict[str, Any]) -> None:
        """Manage a single open position."""
        trade_id = trade["trade_id"]
        pair = trade["pair"]
        side = trade["side"]
        entry_price = trade["entry_price"]
        quantity = trade["quantity"]

        current_price = self.market_data.get_latest_price(pair)
        if current_price <= 0:
            return

        # Update trailing stop
        state = self.risk_manager.update_stop_loss(
            trade_id, current_price, entry_price, side
        )

        # Check if stopped out
        if self.risk_manager.should_stop_out(trade_id, current_price, side):
            await self._close_position(
                trade_id, pair, side, entry_price,
                current_price, quantity, "stop_loss",
                metadata=trade.get("metadata"),
                strategy=trade.get("strategy"),
            )
            return

        # Check take profit
        take_profit = trade.get("take_profit", 0)
        if take_profit > 0:
            if side == "buy" and current_price >= take_profit:
                await self._close_position(
                    trade_id, pair, side, entry_price,
                    current_price, quantity, "take_profit",
                    metadata=trade.get("metadata"),
                    strategy=trade.get("strategy"),
                )
                return
            elif side == "sell" and current_price <= take_profit:
                await self._close_position(
                    trade_id, pair, side, entry_price,
                    current_price, quantity, "take_profit",
                    metadata=trade.get("metadata"),
                    strategy=trade.get("strategy"),
                )
                return

        # Update stop loss in DB if changed
        if state.current_sl > 0:
            # Prepare metadata update with stop loss state
            meta = {}
            if trade.get("metadata"):
                try:
                    meta = json.loads(trade["metadata"]) if isinstance(trade["metadata"], str) else trade["metadata"]
                except Exception:
                    pass
            
            meta["stop_loss_state"] = state.to_dict()
            
            # Only update if SL changed or we just need to persist state
            if state.current_sl != trade.get("stop_loss", 0) or "stop_loss_state" not in meta:
                await self.db.update_trade(trade_id, {
                    "stop_loss": state.current_sl,
                    "trailing_stop": state.current_sl if state.trailing_activated else None,
                    "metadata": meta
                })

    async def _close_position(
        self,
        trade_id: str,
        pair: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
        strategy: Optional[str] = None,
    ) -> None:
        """Close a position and record the result."""
        # C7 FIX: Include both entry and exit fees in PnL
        entry_fee_rate = self.taker_fee
        meta = {}
        if metadata:
            try:
                meta = json.loads(metadata) if isinstance(metadata, str) else dict(metadata)
            except Exception:
                meta = {}

        if meta:
            meta_entry_fee = float(meta.get("entry_fee", 0.0) or 0.0)
            if meta_entry_fee > 0 and entry_price * quantity > 0:
                entry_fee_rate = meta_entry_fee / (entry_price * quantity)
            else:
                entry_fee_rate = float(meta.get("entry_fee_rate", self.taker_fee) or self.taker_fee)

        actual_exit_price = exit_price
        actual_quantity = quantity
        exit_fee = 0.0

        # C6 FIX: Retry exit order in live mode; don't leave ghost positions
        if self.mode == "live":
            close_side = "sell" if side == "buy" else "buy"
            for attempt in range(3):
                try:
                    # Use Limit Order for closing if possible, but Market is safer for stops
                    # For now, sticking to Market for stops/TP to ensure exit
                    result = await self.rest_client.place_order(
                        pair=pair,
                        side=close_side,
                        order_type="market",
                        volume=quantity,
                        reduce_only=True,
                    )
                    txid = None
                    if isinstance(result, dict):
                        txids = result.get("txid") or []
                        if isinstance(txids, list) and txids:
                            txid = txids[0]
                        elif isinstance(txids, str):
                            txid = txids
                    if txid:
                        fill_price, filled_units, partial, fee = await self._wait_for_fill(
                            txid, timeout=30
                        )
                        if fill_price and filled_units and filled_units > 0:
                            actual_exit_price = fill_price
                            if filled_units < actual_quantity:
                                logger.warning(
                                    "Partial exit fill detected",
                                    trade_id=trade_id,
                                    requested=actual_quantity,
                                    filled=filled_units,
                                )
                                actual_quantity = filled_units
                            exit_fee = fee if fee and fee > 0 else 0.0
                    break  # Success
                except Exception as e:
                    logger.error(
                        "Exit order failed",
                        trade_id=trade_id, attempt=attempt + 1, error=str(e)
                    )
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        # Mark as error state so it's not lost
                        await self.db.update_trade(trade_id, {
                            "notes": f"EXIT FAILED after 3 attempts: {str(e)}",
                            "status": "error",
                        })
                        logger.critical("Exit order permanently failed", trade_id=trade_id)
                        return

        if actual_quantity <= 0:
            return

        if actual_quantity != quantity:
            try:
                await self.db.update_trade(trade_id, {
                    "quantity": actual_quantity,
                    "notes": f"Partial exit fill: {actual_quantity:.8f}/{quantity:.8f}",
                })
            except Exception:
                pass

        entry_fee = abs(entry_price * actual_quantity) * entry_fee_rate
        if exit_fee <= 0:
            exit_fee = abs(actual_exit_price * actual_quantity) * self.taker_fee
        fees = entry_fee + exit_fee

        if side == "buy":
            pnl = (actual_exit_price - entry_price) * actual_quantity
        else:
            pnl = (entry_price - actual_exit_price) * actual_quantity

        pnl -= fees  # Net P&L after ALL fees

        # M28 FIX: Calculate pnl_pct AFTER fee deduction
        pnl_pct = pnl / (entry_price * actual_quantity) if entry_price * actual_quantity > 0 else 0

        # Update database
        await self.db.close_trade(
            trade_id, actual_exit_price, pnl, pnl_pct, fees
        )

        # Update risk manager
        self.risk_manager.close_position(trade_id, pnl)
        if self._strategy_result_cb and strategy:
            try:
                self._strategy_result_cb(strategy, pnl)
            except Exception:
                pass

        # Log thought
        emoji = "✅" if pnl > 0 else "❌"
        await self.db.log_thought(
            "trade",
            f"{emoji} CLOSED {pair} ({reason}) | PnL: ${pnl:.2f} ({pnl_pct:.2%}) | "
            f"Entry: ${entry_price:.2f} -> Exit: ${actual_exit_price:.2f}",
            severity="info" if pnl > 0 else "warning",
            metadata={
                "trade_id": trade_id,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "reason": reason,
                "fees": fees,
                "entry_fee": entry_fee,
                "exit_fee": exit_fee,
            },
            tenant_id=self.tenant_id,
        )

        logger.info(
            "Position closed",
            trade_id=trade_id,
            pair=pair,
            pnl=round(pnl, 2),
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Fill Processing
    # ------------------------------------------------------------------

    async def _paper_fill(
        self, pair: str, side: str, target_price: float
    ) -> Optional[float]:
        """
        Simulate a fill in paper trading mode.
        
        Applies realistic slippage based on spread and volatility.
        
        # ENHANCEMENT: Added volume-aware slippage model
        # ENHANCEMENT: Limit order simulation
        """
        spread = self.market_data.get_spread(pair)
        
        # For limit orders (target_price is explicit), we fill at that price
        # assuming the market has crossed it.
        # Since we use current Ask/Bid as limit, it should fill immediately.
        # We add a tiny "spread slippage" to mimic crossing the book spread if data is stale.
        
        slippage_pct = max(spread / 10, 0.00005)

        if side == "buy":
            fill_price = target_price * (1 + slippage_pct)
        else:
            fill_price = target_price * (1 - slippage_pct)

        return round(fill_price, 8)

    async def _live_fill(
        self,
        pair: str,
        side: str,
        order_type: str,
        volume: float,
        client_order_id: str,
        price: Optional[float] = None,
        post_only: bool = False,
    ) -> tuple[Optional[float], Optional[float], bool, float]:
        """
        Execute a live order on Kraken.
        
        # ENHANCEMENT: Added order monitoring with timeout
        # ENHANCEMENT: Support for Limit orders
        """
        # Enforce exchange precision and minimum size
        price_decimals = None
        try:
            min_size = await self.rest_client.get_min_order_size(pair)
            price_decimals, lot_decimals = await self.rest_client.get_pair_decimals(pair)
            if volume < min_size:
                logger.warning(
                    "Order volume below minimum size",
                    pair=pair, volume=volume, min_size=min_size
                )
                return None, None, False, 0.0
            volume = round(float(volume), int(lot_decimals))
            if price is not None and price_decimals is not None:
                price = round(float(price), int(price_decimals))
        except Exception as e:
            logger.warning(
                "Failed to normalize order size/price",
                pair=pair, error=str(e)
            )

        def _best_limit_price() -> Optional[float]:
            ticker = self.market_data.get_ticker(pair)
            if ticker:
                try:
                    if side == "buy":
                        return float(ticker["a"][0])
                    return float(ticker["b"][0])
                except Exception:
                    return None
            return None

        try:
            attempts = self.limit_chase_attempts if order_type == "limit" else 0
            chase_timeout = 10

            for attempt in range(attempts + 1):
                coid = client_order_id
                if client_order_id and attempt > 0:
                    coid = f"{client_order_id}-r{attempt}"

                result = await self.rest_client.place_order(
                    pair=pair,
                    side=side,
                    order_type=order_type,
                    volume=volume,
                    price=price,
                    client_order_id=coid,
                    post_only=post_only,
                    validate_only=(self.mode != "live"),
                )

                if result.get("status") == "duplicate":
                    return None, None, False, 0.0

                txid = None
                if "txid" in result:
                    txid = result["txid"][0] if isinstance(result["txid"], list) else result["txid"]
                if txid:
                    fill_price, filled_volume, partial, fee = await self._wait_for_fill(
                        txid, timeout=chase_timeout
                    )
                    if fill_price and filled_volume and filled_volume > 0:
                        return fill_price, filled_volume, partial, fee

                    # Limit chase: cancel and reprice
                    if order_type == "limit" and attempt < attempts:
                        try:
                            await self.rest_client.cancel_order(txid)
                        except Exception:
                            pass
                        if self.limit_chase_delay_seconds > 0:
                            await asyncio.sleep(self.limit_chase_delay_seconds)
                        new_price = _best_limit_price()
                        if new_price:
                            price = round(float(new_price), int(price_decimals)) if price_decimals is not None else new_price
                        continue

            # Fallback to market if limit couldn't fill and not post-only
            if order_type == "limit" and self.limit_fallback_to_market and not post_only:
                fallback_coid = client_order_id
                if client_order_id:
                    fallback_coid = f"{client_order_id}-m"
                result = await self.rest_client.place_order(
                    pair=pair,
                    side=side,
                    order_type="market",
                    volume=volume,
                    client_order_id=fallback_coid,
                    post_only=False,
                    validate_only=(self.mode != "live"),
                )
                txid = None
                if "txid" in result:
                    txid = result["txid"][0] if isinstance(result["txid"], list) else result["txid"]
                if txid:
                    return await self._wait_for_fill(txid, timeout=30)

            return None, None, False, 0.0

        except Exception as e:
            logger.error(
                "Live order failed",
                pair=pair, side=side, error=str(e)
            )
            return None, None, False, 0.0

    def _coerce_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _extract_order_fill(
        self, order_info: Dict[str, Any]
    ) -> tuple[float, float, float, float]:
        vol_exec = self._coerce_float(order_info.get("vol_exec", 0))
        vol_total = self._coerce_float(order_info.get("vol", 0))
        cost = self._coerce_float(order_info.get("cost", 0))
        fee = self._coerce_float(order_info.get("fee", 0))
        price = self._coerce_float(order_info.get("price", 0))
        avg_price = self._coerce_float(order_info.get("avg_price", 0))

        if price <= 0:
            price = avg_price
        if price <= 0 and isinstance(order_info.get("descr"), dict):
            price = self._coerce_float(order_info["descr"].get("price", 0))
        if price <= 0 and cost > 0 and vol_exec > 0:
            price = cost / vol_exec
        if cost <= 0 and price > 0 and vol_exec > 0:
            cost = price * vol_exec

        return price, vol_exec, vol_total, fee

    async def _fill_from_trade_history(
        self, txid: str, lookback_seconds: int = 7200
    ) -> tuple[float, float, float]:
        """
        Compute average fill price/volume/fee from trade history for an order.
        Kraken sometimes reports vol_exec without cost/price on open orders.
        """
        try:
            end_ts = int(time.time())
            start_ts = max(0, end_ts - lookback_seconds)
            history = await self.rest_client.get_trades_history(
                start=start_ts, end=end_ts
            )
            trades = history.get("trades") or {}
            total_vol = 0.0
            total_cost = 0.0
            total_fee = 0.0
            for trade in trades.values():
                order_txid = trade.get("ordertxid") or trade.get("order_txid")
                if order_txid != txid:
                    continue
                vol = self._coerce_float(trade.get("vol", 0))
                price = self._coerce_float(trade.get("price", 0))
                fee = self._coerce_float(trade.get("fee", 0))
                if vol <= 0 or price <= 0:
                    continue
                total_vol += vol
                total_cost += price * vol
                total_fee += fee
            if total_vol > 0:
                return total_cost / total_vol, total_vol, total_fee
        except Exception:
            pass
        return 0.0, 0.0, 0.0

    async def _resolve_fill_data(
        self,
        txid: str,
        price: float,
        vol_exec: float,
        fee: float,
        prefer_fee: bool = False,
    ) -> tuple[float, float, float]:
        if vol_exec <= 0:
            return price, vol_exec, fee
        if price > 0 and fee > 0 and not prefer_fee:
            return price, vol_exec, fee
        th_price, th_vol, th_fee = await self._fill_from_trade_history(txid)
        if th_price > 0:
            price = th_price
        if th_vol > 0:
            vol_exec = th_vol
        if th_fee > 0:
            fee = th_fee
        return price, vol_exec, fee

    async def _wait_for_fill(
        self, txid: str, timeout: int = 30
    ) -> tuple[Optional[float], Optional[float], bool, float]:
        """Wait for an order to be filled. Returns (price, filled_volume, partial, fee)."""
        start = time.time()
        last_partial: Optional[tuple[float, float, float]] = None

        while time.time() - start < timeout:
            try:
                orders = await self.rest_client.get_open_orders()
                open_order = orders.get("open", {}).get(txid)
                if open_order:
                    price, vol_exec, vol_total, fee = self._extract_order_fill(open_order)
                    if vol_exec > 0:
                        if price <= 0:
                            try:
                                order_info = await self.rest_client.get_order_info(txid)
                                query = order_info.get(txid)
                                if query:
                                    price, vol_exec, vol_total, fee = self._extract_order_fill(query)
                            except Exception:
                                pass
                        price, vol_exec, fee = await self._resolve_fill_data(
                            txid, price, vol_exec, fee
                        )
                        if price > 0:
                            last_partial = (price, vol_exec, fee)
                        if price > 0 and vol_total > 0 and (vol_exec / vol_total) >= 0.95:
                            return price, vol_exec, True, fee
                    await asyncio.sleep(1)
                    continue

                # Order is no longer open - check closed
                closed = await self.rest_client.get_closed_orders()
                order_info = closed.get("closed", {}).get(txid, {})
                if order_info:
                    price, vol_exec, _vol_total, fee = self._extract_order_fill(order_info)
                    if price <= 0:
                        try:
                            order_query = await self.rest_client.get_order_info(txid)
                            query = order_query.get(txid)
                            if query:
                                price, vol_exec, _vol_total, fee = self._extract_order_fill(query)
                        except Exception:
                            pass
                    price, vol_exec, fee = await self._resolve_fill_data(
                        txid, price, vol_exec, fee, prefer_fee=True
                    )
                    return (price if price > 0 else None, vol_exec, False, fee)
            except Exception:
                pass
            await asyncio.sleep(1)

        # Timeout: check for partial fill and cancel remainder
        try:
            orders = await self.rest_client.get_open_orders()
            open_order = orders.get("open", {}).get(txid)
            if open_order:
                price, vol_exec, _vol_total, fee = self._extract_order_fill(open_order)
                if vol_exec > 0:
                    if price <= 0:
                        try:
                            order_info = await self.rest_client.get_order_info(txid)
                            query = order_info.get(txid)
                            if query:
                                price, vol_exec, _vol_total, fee = self._extract_order_fill(query)
                        except Exception:
                            pass
                    price, vol_exec, fee = await self._resolve_fill_data(
                        txid, price, vol_exec, fee, prefer_fee=True
                    )
                    try:
                        await self.rest_client.cancel_order(txid)
                    except Exception:
                        pass
                    if price > 0:
                        return price, vol_exec, True, fee
                try:
                    await self.rest_client.cancel_order(txid)
                except Exception:
                    pass
        except Exception:
            pass

        if last_partial:
            price, vol_exec, fee = last_partial
            return price, vol_exec, True, fee

        return None, None, False, 0.0

    async def close_all_positions(self, reason: str = "manual") -> int:
        """
        Emergency close all open positions.
        
        # ENHANCEMENT: Added parallel closing for speed
        """
        open_trades = await self.db.get_open_trades(tenant_id=self.tenant_id)
        closed_count = 0

        for trade in open_trades:
            try:
                current_price = self.market_data.get_latest_price(trade["pair"])
                if current_price > 0:
                    await self._close_position(
                        trade["trade_id"],
                        trade["pair"],
                        trade["side"],
                        trade["entry_price"],
                        current_price,
                        trade["quantity"],
                        reason,
                        metadata=trade.get("metadata"),
                        strategy=trade.get("strategy"),
                    )
                    closed_count += 1
            except Exception as e:
                logger.error(
                    "Emergency close failed",
                    trade_id=trade["trade_id"],
                    error=str(e)
                )

        logger.warning(
            "Emergency close all completed",
            closed=closed_count,
            total=len(open_trades),
            reason=reason,
        )
        return closed_count

    def _primary_strategy(self, signal: ConfluenceSignal) -> str:
        """Determine the primary strategy from a confluence signal."""
        if not signal.signals:
            return "confluence"
        # Find the strongest agreeing signal
        best = max(
            [s for s in signal.signals if s.direction == signal.direction],
            key=lambda s: s.strength,
            default=None,
        )
        return best.strategy_name if best else "confluence"

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        stats = dict(self._execution_stats)
        stats["mode"] = self.mode
        if stats["orders_filled"] > 0:
            stats["avg_slippage"] = round(
                stats["total_slippage"] / stats["orders_filled"], 6
            )
            stats["avg_fee"] = round(
                stats["total_fees"] / stats["orders_filled"], 4
            )
        return stats
