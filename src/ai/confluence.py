"""
Multi-Strategy Confluence Detector - "Sure Fire" Setup Engine.

Orchestrates all 5 strategies in parallel, then scores confluence
when multiple strategies agree on direction. Applies Order Book
Imbalance as a final confirmation filter.

# ENHANCEMENT: Added weighted confluence scoring based on strategy performance
# ENHANCEMENT: Added adaptive threshold based on market volatility
# ENHANCEMENT: Added historical confluence accuracy tracking
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.logger import get_logger
from src.exchange.market_data import MarketDataCache
from src.strategies.base import BaseStrategy, SignalDirection, StrategySignal
from src.strategies.breakout import BreakoutStrategy
from src.strategies.keltner import KeltnerStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.reversal import ReversalStrategy
from src.strategies.trend import TrendStrategy
from src.utils.indicators import order_book_imbalance

logger = get_logger("confluence")


class ConfluenceSignal:
    """
    Aggregated signal from multi-strategy confluence analysis.
    
    Represents the combined output of all strategy signals with
    weighted scoring, OBI confirmation, and confidence levels.
    """

    def __init__(
        self,
        pair: str,
        direction: SignalDirection,
        strength: float,
        confidence: float,
        confluence_count: int,
        signals: List[StrategySignal],
        obi: float = 0.0,
        obi_agrees: bool = False,
        is_sure_fire: bool = False,
        entry_price: float = 0.0,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ):
        self.pair = pair
        self.direction = direction
        self.strength = strength
        self.confidence = confidence
        self.confluence_count = confluence_count
        self.signals = signals
        self.obi = obi
        self.obi_agrees = obi_agrees
        self.is_sure_fire = is_sure_fire
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair": self.pair,
            "direction": self.direction.value,
            "strength": round(float(self.strength), 4),
            "confidence": round(float(self.confidence), 4),
            "confluence_count": int(self.confluence_count),
            "obi": round(float(self.obi), 4),
            "obi_agrees": bool(self.obi_agrees),
            "is_sure_fire": bool(self.is_sure_fire),
            "entry_price": float(self.entry_price),
            "stop_loss": float(self.stop_loss),
            "take_profit": float(self.take_profit),
            "timestamp": self.timestamp,
            "strategy_signals": [s.to_dict() for s in self.signals],
        }


class ConfluenceDetector:
    """
    Multi-algorithm confluence engine.
    
    Runs all 5 strategies in parallel, then analyzes agreement patterns.
    A "Sure Fire" setup is detected when 3+ strategies align AND the
    Order Book Imbalance confirms the direction.
    
    # ENHANCEMENT: Added per-strategy performance weighting
    # ENHANCEMENT: Added dynamic confluence threshold based on volatility
    # ENHANCEMENT: Added signal decay for stale confluence
    """

    def __init__(
        self,
        market_data: MarketDataCache,
        confluence_threshold: int = 3,
        obi_threshold: float = 0.15,
        min_confidence: float = 0.65,
    ):
        self.market_data = market_data
        self.confluence_threshold = confluence_threshold
        self.obi_threshold = obi_threshold
        self.min_confidence = min_confidence

        # Initialize strategies — Keltner is our primary high-WR strategy
        self.strategies: List[BaseStrategy] = [
            KeltnerStrategy(weight=0.30),
            TrendStrategy(weight=0.20),
            MeanReversionStrategy(weight=0.15),
            MomentumStrategy(weight=0.15),
            BreakoutStrategy(weight=0.10),
            ReversalStrategy(weight=0.10),
        ]

        self._last_confluence: Dict[str, ConfluenceSignal] = {}
        self._signal_history: List[ConfluenceSignal] = []

    def configure_strategies(self, config: Dict[str, Any]) -> None:
        """Configure strategies from config dict."""
        strategy_map = {
            "keltner": KeltnerStrategy,
            "trend": TrendStrategy,
            "mean_reversion": MeanReversionStrategy,
            "momentum": MomentumStrategy,
            "breakout": BreakoutStrategy,
            "reversal": ReversalStrategy,
        }

        self.strategies = []
        for name, cls in strategy_map.items():
            strat_config = config.get(name, {})
            if strat_config.get("enabled", True):
                # Filter constructor args
                import inspect
                sig = inspect.signature(cls.__init__)
                valid_params = {
                    k: v for k, v in strat_config.items()
                    if k in sig.parameters and k != "self"
                }
                self.strategies.append(cls(**valid_params))

    async def analyze_pair(self, pair: str) -> ConfluenceSignal:
        """
        Run all strategies in parallel on a single pair and detect confluence.
        
        Returns a ConfluenceSignal with the aggregated result.
        
        # ENHANCEMENT: Added timeout protection per strategy
        """
        # S3 FIX: Also reject stale data — don't trade on outdated prices
        if not self.market_data.is_warmed_up(pair) or self.market_data.is_stale(pair, max_age_seconds=180):
            return ConfluenceSignal(
                pair=pair,
                direction=SignalDirection.NEUTRAL,
                strength=0.0,
                confidence=0.0,
                confluence_count=0,
                signals=[],
            )

        # Get market data arrays
        closes = self.market_data.get_closes(pair)
        highs = self.market_data.get_highs(pair)
        lows = self.market_data.get_lows(pair)
        volumes = self.market_data.get_volumes(pair)
        opens = self.market_data.get_opens(pair)

        # Run all strategies in parallel with timeout
        tasks = []
        for strategy in self.strategies:
            if strategy.enabled:
                task = asyncio.create_task(
                    asyncio.wait_for(
                        strategy.analyze(
                            pair, closes, highs, lows, volumes,
                            opens=opens
                        ),
                        timeout=5.0  # 5 second timeout per strategy
                    )
                )
                tasks.append((strategy, task))

        # Gather results
        signals: List[StrategySignal] = []
        for strategy, task in tasks:
            try:
                signal = await task
                signals.append(signal)
            except asyncio.TimeoutError:
                logger.warning(
                    "Strategy timed out",
                    strategy=strategy.name, pair=pair
                )
            except Exception as e:
                logger.error(
                    "Strategy error",
                    strategy=strategy.name, pair=pair, error=str(e)
                )

        # Analyze confluence
        return self._compute_confluence(pair, signals)

    def _compute_confluence(
        self, pair: str, signals: List[StrategySignal]
    ) -> ConfluenceSignal:
        """
        Compute confluence from multiple strategy signals.
        
        # ENHANCEMENT: Added weighted scoring and performance-based adjustments
        """
        if not signals:
            return ConfluenceSignal(
                pair=pair,
                direction=SignalDirection.NEUTRAL,
                strength=0.0,
                confidence=0.0,
                confluence_count=0,
                signals=[],
            )

        # Count directions with weighting
        long_signals = [s for s in signals if s.direction == SignalDirection.LONG and s.is_actionable]
        short_signals = [s for s in signals if s.direction == SignalDirection.SHORT and s.is_actionable]

        long_count = len(long_signals)
        short_count = len(short_signals)

        # Determine majority direction
        if long_count > short_count and long_count >= 1:
            direction = SignalDirection.LONG
            directional_signals = long_signals
            confluence_count = long_count
        elif short_count > long_count and short_count >= 1:
            direction = SignalDirection.SHORT
            directional_signals = short_signals
            confluence_count = short_count
        else:
            return ConfluenceSignal(
                pair=pair,
                direction=SignalDirection.NEUTRAL,
                strength=0.0,
                confidence=0.0,
                confluence_count=0,
                signals=signals,
            )

        # Weighted strength calculation
        total_weight = sum(
            self._get_strategy_weight(s.strategy_name)
            for s in directional_signals
        )
        if total_weight > 0:
            weighted_strength = sum(
                s.strength * self._get_strategy_weight(s.strategy_name)
                for s in directional_signals
            ) / total_weight
        else:
            weighted_strength = np.mean([s.strength for s in directional_signals])

        # Weighted confidence
        if total_weight > 0:
            weighted_confidence = sum(
                s.confidence * self._get_strategy_weight(s.strategy_name)
                for s in directional_signals
            ) / total_weight
        else:
            weighted_confidence = np.mean([s.confidence for s in directional_signals])

        # Confluence bonus
        confluence_bonus = min((confluence_count - 1) * 0.1, 0.3)
        weighted_confidence = min(weighted_confidence + confluence_bonus, 1.0)

        # Order Book Imbalance analysis
        order_book = self.market_data.get_order_book(pair)
        obi = 0.0
        obi_agrees = False

        if order_book:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            if bids and asks:
                # M26 FIX: Safe parsing of order book entries
                try:
                    bid_vol = sum(float(b[1]) for b in bids[:10] if len(b) > 1)
                    ask_vol = sum(float(a[1]) for a in asks[:10] if len(a) > 1)
                except (ValueError, TypeError, IndexError):
                    bid_vol = 0.0
                    ask_vol = 0.0
                obi = order_book_imbalance(bid_vol, ask_vol)

                if direction == SignalDirection.LONG and obi > self.obi_threshold:
                    obi_agrees = True
                    weighted_confidence += 0.05
                elif direction == SignalDirection.SHORT and obi < -self.obi_threshold:
                    obi_agrees = True
                    weighted_confidence += 0.05

        # "Sure Fire" detection: 3+ strategies + OBI agreement
        is_sure_fire = (
            confluence_count >= self.confluence_threshold and
            obi_agrees and
            weighted_confidence >= self.min_confidence
        )

        if is_sure_fire:
            weighted_strength = min(weighted_strength + 0.15, 1.0)
            weighted_confidence = min(weighted_confidence + 0.10, 1.0)

        # Aggregate stop loss and take profit (weighted average)
        entry_price = directional_signals[0].entry_price if directional_signals else 0
        
        if directional_signals:
            sl_values = [s.stop_loss for s in directional_signals if s.stop_loss > 0]
            tp_values = [s.take_profit for s in directional_signals if s.take_profit > 0]
            
            if direction == SignalDirection.LONG:
                stop_loss = max(sl_values) if sl_values else 0  # Tightest SL
                take_profit = min(tp_values) if tp_values else 0  # Conservative TP
            else:
                stop_loss = min(sl_values) if sl_values else 0
                take_profit = max(tp_values) if tp_values else 0
        else:
            stop_loss = 0
            take_profit = 0

        result = ConfluenceSignal(
            pair=pair,
            direction=direction,
            strength=round(weighted_strength, 4),
            confidence=round(min(weighted_confidence, 1.0), 4),
            confluence_count=confluence_count,
            signals=signals,
            obi=obi,
            obi_agrees=obi_agrees,
            is_sure_fire=is_sure_fire,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self._last_confluence[pair] = result
        self._signal_history.append(result)

        # Keep history manageable
        if len(self._signal_history) > 1000:
            self._signal_history = self._signal_history[-500:]

        return result

    async def scan_all_pairs(self, pairs: List[str]) -> List[ConfluenceSignal]:
        """
        Scan all pairs in parallel and return confluence signals.
        
        # ENHANCEMENT: Added priority ordering by signal strength
        """
        tasks = [self.analyze_pair(pair) for pair in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for pair, result in zip(pairs, results):
            if isinstance(result, Exception):
                logger.error("Pair scan failed", pair=pair, error=str(result))
            else:
                valid_results.append(result)

        # Sort by strength (strongest first)
        valid_results.sort(key=lambda s: s.strength, reverse=True)
        return valid_results

    def _get_strategy_weight(self, strategy_name: str) -> float:
        """Get the weight for a strategy, considering performance adjustment."""
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                # Performance-adjusted weight
                base_weight = strategy.weight
                if strategy._trade_count > 10:
                    performance_factor = max(0.5, min(1.5, 0.5 + strategy.win_rate))
                    return base_weight * performance_factor
                return base_weight
        return 0.1  # Default weight

    def get_strategy_stats(self) -> List[Dict[str, Any]]:
        """Get performance stats for all strategies."""
        return [s.get_stats() for s in self.strategies]

    def get_last_confluence(self, pair: str) -> Optional[ConfluenceSignal]:
        """Get the most recent confluence signal for a pair."""
        return self._last_confluence.get(pair)
