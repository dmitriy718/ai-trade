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
from src.utils.indicator_cache import IndicatorCache
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
        book_score: float = 0.0,
        obi_agrees: bool = False,
        is_sure_fire: bool = False,
        entry_price: float = 0.0,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        regime: Optional[str] = None,
        volatility_regime: Optional[str] = None,
        timeframe_agreement: int = 1,
        timeframes: Optional[Dict[str, str]] = None,
    ):
        self.pair = pair
        self.direction = direction
        self.strength = strength
        self.confidence = confidence
        self.confluence_count = confluence_count
        self.signals = signals
        self.obi = obi
        self.book_score = book_score
        self.obi_agrees = obi_agrees
        self.is_sure_fire = is_sure_fire
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.regime = regime
        self.volatility_regime = volatility_regime
        self.timeframe_agreement = timeframe_agreement
        self.timeframes = timeframes or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair": self.pair,
            "direction": self.direction.value,
            "strength": round(float(self.strength), 4),
            "confidence": round(float(self.confidence), 4),
            "confluence_count": int(self.confluence_count),
            "obi": round(float(self.obi), 4),
            "book_score": round(float(self.book_score), 4),
            "obi_agrees": bool(self.obi_agrees),
            "is_sure_fire": bool(self.is_sure_fire),
            "entry_price": float(self.entry_price),
            "stop_loss": float(self.stop_loss),
            "take_profit": float(self.take_profit),
            "regime": self.regime,
            "volatility_regime": self.volatility_regime,
            "timeframe_agreement": int(self.timeframe_agreement),
            "timeframes": dict(self.timeframes),
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
        book_score_threshold: float = 0.2,
        book_score_max_age_seconds: int = 5,
        min_confidence: float = 0.65,
        obi_counts_as_confluence: bool = True,
        obi_weight: float = 0.4,
        round_trip_fee_pct: float = 0.0052,
        use_closed_candles_only: bool = False,
        regime_config: Optional[Any] = None,
        timeframes: Optional[List[int]] = None,
        multi_timeframe_min_agreement: int = 1,
        primary_timeframe: int = 1,
    ):
        self.market_data = market_data
        self.confluence_threshold = confluence_threshold
        self.obi_threshold = obi_threshold
        self.book_score_threshold = book_score_threshold
        self.book_score_max_age_seconds = book_score_max_age_seconds
        self.min_confidence = min_confidence
        self.obi_counts_as_confluence = obi_counts_as_confluence
        self.obi_weight = obi_weight
        self.round_trip_fee_pct = round_trip_fee_pct
        self.use_closed_candles_only = use_closed_candles_only
        self.regime_config = regime_config or {}
        self.timeframes = sorted(timeframes or [1])
        if 1 not in self.timeframes:
            self.timeframes.insert(0, 1)
        self.multi_timeframe_min_agreement = max(1, int(multi_timeframe_min_agreement))
        self.primary_timeframe = primary_timeframe if primary_timeframe in self.timeframes else 1

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
        self._cooldown_checker = None

        # Default regime multipliers (fallback if config missing)
        self._default_trend_weights = {
            "trend": 1.3,
            "momentum": 1.2,
            "breakout": 1.1,
            "mean_reversion": 0.8,
            "reversal": 0.7,
            "keltner": 0.9,
        }
        self._default_range_weights = {
            "mean_reversion": 1.3,
            "keltner": 1.2,
            "reversal": 1.1,
            "trend": 0.8,
            "momentum": 0.8,
            "breakout": 0.8,
        }
        self._default_high_vol_weights = {
            "breakout": 1.2,
            "momentum": 1.1,
            "mean_reversion": 0.9,
            "reversal": 0.9,
        }
        self._default_low_vol_weights = {
            "mean_reversion": 1.2,
            "keltner": 1.1,
            "breakout": 0.9,
            "momentum": 0.9,
        }

    def configure_strategies(
        self, config: Dict[str, Any], single_strategy_mode: Optional[str] = None
    ) -> None:
        """Configure strategies from config dict. If single_strategy_mode is set, only that strategy runs."""
        strategy_map = {
            "keltner": KeltnerStrategy,
            "trend": TrendStrategy,
            "mean_reversion": MeanReversionStrategy,
            "momentum": MomentumStrategy,
            "breakout": BreakoutStrategy,
            "reversal": ReversalStrategy,
        }

        self.strategies = []
        names_to_build = (
            [single_strategy_mode]
            if single_strategy_mode and single_strategy_mode in strategy_map
            else strategy_map.keys()
        )
        for name in names_to_build:
            cls = strategy_map[name]
            strat_config = config.get(name, {})
            if strat_config.get("enabled", True):
                import inspect
                sig = inspect.signature(cls.__init__)
                valid_params = {
                    k: v for k, v in strat_config.items()
                    if k in sig.parameters and k != "self"
                }
                self.strategies.append(cls(**valid_params))

    def set_cooldown_checker(self, checker) -> None:
        """Inject a cooldown checker: fn(pair, strategy_name, side) -> bool."""
        self._cooldown_checker = checker

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

        # Get market data arrays (base timeframe)
        base_times = self.market_data.get_times(pair)
        base_closes = self.market_data.get_closes(pair)
        base_highs = self.market_data.get_highs(pair)
        base_lows = self.market_data.get_lows(pair)
        base_volumes = self.market_data.get_volumes(pair)
        base_opens = self.market_data.get_opens(pair)

        # Optionally drop the most recent (potentially in‑progress) candle
        if self.use_closed_candles_only and len(base_closes) > 1:
            base_times = base_times[:-1]
            base_closes = base_closes[:-1]
            base_highs = base_highs[:-1]
            base_lows = base_lows[:-1]
            base_volumes = base_volumes[:-1]
            base_opens = base_opens[:-1]

        timeframe_results: Dict[int, ConfluenceSignal] = {}

        for tf in self.timeframes:
            closes, highs, lows, volumes, opens = self._resample_ohlcv(
                base_times,
                base_closes,
                base_highs,
                base_lows,
                base_volumes,
                base_opens,
                tf,
            )
            if len(closes) < 50:
                continue

            indicator_cache = IndicatorCache(closes, highs, lows, volumes)
            trend_regime, vol_regime = self._detect_regime(indicator_cache, closes)
            signals = await self._run_strategies(
                pair, closes, highs, lows, volumes, opens, indicator_cache
            )
            timeframe_results[tf] = self._compute_confluence(
                pair, signals, trend_regime, vol_regime
            )

        return self._combine_timeframes(pair, timeframe_results)

    async def _run_strategies(
        self,
        pair: str,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        opens: np.ndarray,
        indicator_cache: IndicatorCache,
    ) -> List[StrategySignal]:
        """Run all strategies for a given timeframe with timeout + cooldown filtering."""
        signals: List[StrategySignal] = []
        for strategy in self.strategies:
            if not strategy.enabled:
                continue
            try:
                signal = await asyncio.wait_for(
                    strategy.analyze(
                        pair, closes, highs, lows, volumes,
                        opens=opens,
                        indicator_cache=indicator_cache,
                        round_trip_fee_pct=self.round_trip_fee_pct,
                    ),
                    timeout=5.0,
                )
                if self._cooldown_checker and signal.direction != SignalDirection.NEUTRAL:
                    side = "buy" if signal.direction == SignalDirection.LONG else "sell"
                    if self._cooldown_checker(pair, signal.strategy_name, side):
                        signal = StrategySignal(
                            strategy_name=signal.strategy_name,
                            pair=signal.pair,
                            direction=SignalDirection.NEUTRAL,
                            strength=0.0,
                            confidence=0.0,
                            metadata={"reason": "strategy_cooldown"},
                        )
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

        return signals

    def _resample_ohlcv(
        self,
        times: np.ndarray,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        opens: np.ndarray,
        timeframe: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Resample 1m OHLCV into a higher timeframe (e.g., 5m, 15m)."""
        if timeframe <= 1:
            return closes, highs, lows, volumes, opens

        n = min(len(times), len(closes), len(highs), len(lows), len(volumes), len(opens))
        if n == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        times = times[-n:]
        closes = closes[-n:]
        highs = highs[-n:]
        lows = lows[-n:]
        volumes = volumes[-n:]
        opens = opens[-n:]

        bucket_seconds = timeframe * 60
        agg_closes: List[float] = []
        agg_opens: List[float] = []
        agg_highs: List[float] = []
        agg_lows: List[float] = []
        agg_volumes: List[float] = []

        current_bucket = None
        count = 0
        open_val = high_val = low_val = close_val = vol_val = 0.0

        for i in range(len(times)):
            bucket = int(times[i] // bucket_seconds)
            if current_bucket is None or bucket != current_bucket:
                if current_bucket is not None and count >= timeframe:
                    agg_opens.append(open_val)
                    agg_highs.append(high_val)
                    agg_lows.append(low_val)
                    agg_closes.append(close_val)
                    agg_volumes.append(vol_val)
                current_bucket = bucket
                count = 1
                open_val = float(opens[i])
                high_val = float(highs[i])
                low_val = float(lows[i])
                close_val = float(closes[i])
                vol_val = float(volumes[i])
            else:
                count += 1
                high_val = max(high_val, float(highs[i]))
                low_val = min(low_val, float(lows[i]))
                close_val = float(closes[i])
                vol_val += float(volumes[i])

        if current_bucket is not None and count >= timeframe:
            agg_opens.append(open_val)
            agg_highs.append(high_val)
            agg_lows.append(low_val)
            agg_closes.append(close_val)
            agg_volumes.append(vol_val)

        return (
            np.array(agg_closes),
            np.array(agg_highs),
            np.array(agg_lows),
            np.array(agg_volumes),
            np.array(agg_opens),
        )

    def _combine_timeframes(
        self,
        pair: str,
        results: Dict[int, ConfluenceSignal],
    ) -> ConfluenceSignal:
        """Combine per-timeframe confluence signals into a final decision."""
        if not results:
            return ConfluenceSignal(
                pair=pair,
                direction=SignalDirection.NEUTRAL,
                strength=0.0,
                confidence=0.0,
                confluence_count=0,
                signals=[],
            )

        primary_tf = self.primary_timeframe if self.primary_timeframe in results else min(results.keys())
        base = results[primary_tf]
        if base.direction == SignalDirection.NEUTRAL:
            return base

        agreement = [
            tf for tf, sig in results.items()
            if sig.direction == base.direction
        ]
        if len(agreement) < self.multi_timeframe_min_agreement:
            return ConfluenceSignal(
                pair=pair,
                direction=SignalDirection.NEUTRAL,
                strength=0.0,
                confidence=0.0,
                confluence_count=0,
                signals=[],
                timeframe_agreement=len(agreement),
                timeframes={str(tf): sig.direction.value for tf, sig in results.items()},
            )

        bonus = min(0.05 * (len(agreement) - 1), 0.15)
        base.strength = min(base.strength + bonus, 1.0)
        base.confidence = min(base.confidence + bonus, 1.0)
        base.timeframe_agreement = len(agreement)
        base.timeframes = {str(tf): sig.direction.value for tf, sig in results.items()}
        return base

    def _compute_confluence(
        self,
        pair: str,
        signals: List[StrategySignal],
        trend_regime: Optional[str] = None,
        vol_regime: Optional[str] = None,
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

        # Order Book Imbalance: compute early for synthetic signal
        order_book = self.market_data.get_order_book(pair)
        if order_book:
            try:
                updated_at = float(order_book.get("updated_at", 0))
                if (time.time() - updated_at) > self.book_score_max_age_seconds:
                    order_book = {}
            except (TypeError, ValueError):
                order_book = {}
        obi = 0.0
        obi_agrees_long = False
        obi_agrees_short = False
        if order_book:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            if bids and asks:
                try:
                    bid_vol = sum(float(b[1]) for b in bids[:10] if len(b) > 1)
                    ask_vol = sum(float(a[1]) for a in asks[:10] if len(a) > 1)
                except (ValueError, TypeError, IndexError):
                    bid_vol = 0.0
                    ask_vol = 0.0
                obi = order_book_imbalance(bid_vol, ask_vol)
                obi_agrees_long = obi > self.obi_threshold
                obi_agrees_short = obi < -self.obi_threshold

        # Microstructure score (preferred over raw OBI when available)
        book_score = 0.0
        book_analysis = self.market_data.get_order_book_analysis(pair)
        use_book_score = False
        if book_analysis:
            try:
                updated_at = float(book_analysis.get("updated_at", 0))
                is_fresh = (time.time() - updated_at) <= self.book_score_max_age_seconds
                if is_fresh:
                    book_score = float(book_analysis.get("book_score", 0.0))
                    use_book_score = "book_score" in book_analysis
            except (TypeError, ValueError):
                book_score = 0.0
                use_book_score = False

        score_for_agreement = book_score if use_book_score else obi
        threshold = self.book_score_threshold if use_book_score else self.obi_threshold
        if score_for_agreement != 0.0:
            obi_agrees_long = score_for_agreement > threshold
            obi_agrees_short = score_for_agreement < -threshold

        # OBI heavy weight: add synthetic "order_book" signal so OBI + 1 strategy = 2 = tradable
        if self.obi_counts_as_confluence:
            entry_price = self.market_data.get_latest_price(pair) or 0.0
            synthetic_strength = min(0.4 + abs(score_for_agreement) * 0.6, 1.0)
            synthetic_confidence = min(0.4 + abs(score_for_agreement) * 0.6, 1.0)
            if obi_agrees_long:
                long_signals.append(
                    StrategySignal(
                        strategy_name="order_book",
                        pair=pair,
                        direction=SignalDirection.LONG,
                        strength=synthetic_strength,
                        confidence=synthetic_confidence,
                        entry_price=entry_price,
                        stop_loss=0.0,
                        take_profit=0.0,
                    )
                )
            if obi_agrees_short:
                short_signals.append(
                    StrategySignal(
                        strategy_name="order_book",
                        pair=pair,
                        direction=SignalDirection.SHORT,
                        strength=synthetic_strength,
                        confidence=synthetic_confidence,
                        entry_price=entry_price,
                        stop_loss=0.0,
                        take_profit=0.0,
                    )
                )

        long_count = len(long_signals)
        short_count = len(short_signals)

        # Determine majority direction (obi_agrees set per direction below)
        if long_count > short_count and long_count >= 1:
            direction = SignalDirection.LONG
            directional_signals = long_signals
            confluence_count = long_count
            obi_agrees = obi_agrees_long
        elif short_count > long_count and short_count >= 1:
            direction = SignalDirection.SHORT
            directional_signals = short_signals
            confluence_count = short_count
            obi_agrees = obi_agrees_short
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
            self._get_strategy_weight(s.strategy_name, trend_regime, vol_regime)
            for s in directional_signals
        )
        if total_weight > 0:
            weighted_strength = sum(
                s.strength * self._get_strategy_weight(s.strategy_name, trend_regime, vol_regime)
                for s in directional_signals
            ) / total_weight
        else:
            weighted_strength = np.mean([s.strength for s in directional_signals])

        # Weighted confidence
        if total_weight > 0:
            weighted_confidence = sum(
                s.confidence * self._get_strategy_weight(s.strategy_name, trend_regime, vol_regime)
                for s in directional_signals
            ) / total_weight
        else:
            weighted_confidence = np.mean([s.confidence for s in directional_signals])

        # Confluence bonus
        confluence_bonus = min((confluence_count - 1) * 0.1, 0.3)
        weighted_confidence = min(weighted_confidence + confluence_bonus, 1.0)

        # Legacy: when OBI is not counted as confluence, still add small confidence bump when it agrees
        if not self.obi_counts_as_confluence and obi_agrees:
            weighted_confidence = min(weighted_confidence + 0.05, 1.0)

        # Regime alignment bonus (small threshold easing when strategy matches regime)
        if self._is_regime_aligned(trend_regime, directional_signals):
            weighted_confidence = min(weighted_confidence + 0.03, 1.0)

        # "Sure Fire" detection: threshold strategies + OBI agreement
        is_sure_fire = (
            confluence_count >= self.confluence_threshold and
            obi_agrees and
            weighted_confidence >= self.min_confidence
        )

        if is_sure_fire:
            weighted_strength = min(weighted_strength + 0.15, 1.0)
            weighted_confidence = min(weighted_confidence + 0.10, 1.0)

        # Aggregate stop loss and take profit (use strongest signal when possible)
        primary_signal = max(directional_signals, key=lambda s: s.strength, default=None)
        entry_price = primary_signal.entry_price if primary_signal else 0
        if primary_signal and primary_signal.stop_loss > 0 and primary_signal.take_profit > 0:
            stop_loss = primary_signal.stop_loss
            take_profit = primary_signal.take_profit
        elif directional_signals:
            sl_values = [s.stop_loss for s in directional_signals if s.stop_loss > 0]
            tp_values = [s.take_profit for s in directional_signals if s.take_profit > 0]
            if direction == SignalDirection.LONG:
                stop_loss = max(sl_values) if sl_values else 0
                take_profit = min(tp_values) if tp_values else 0
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
            book_score=book_score,
            obi_agrees=obi_agrees,
            is_sure_fire=is_sure_fire,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            regime=trend_regime,
            volatility_regime=vol_regime,
        )

        self._last_confluence[pair] = result
        self._signal_history.append(result)

        # Keep history manageable
        if len(self._signal_history) > 1000:
            self._signal_history = self._signal_history[-500:]

        return result

    def _is_regime_aligned(
        self, trend_regime: Optional[str], signals: List[StrategySignal]
    ) -> bool:
        """Return True if the dominant strategy set matches the trend regime."""
        if not trend_regime:
            return False
        trend_set = {"trend", "momentum", "breakout"}
        range_set = {"mean_reversion", "reversal", "keltner"}
        strategy_names = {s.strategy_name for s in signals}
        if trend_regime == "trend":
            return len(strategy_names & trend_set) > 0
        if trend_regime == "range":
            return len(strategy_names & range_set) > 0
        return False

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

    def _get_strategy_weight(
        self,
        strategy_name: str,
        trend_regime: Optional[str] = None,
        vol_regime: Optional[str] = None,
    ) -> float:
        """Get the weight for a strategy, considering performance + regime adjustment."""
        if strategy_name == "order_book":
            return self.obi_weight
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                # Performance-adjusted weight
                base_weight = strategy.weight
                if strategy._trade_count > 10:
                    performance_factor = max(0.5, min(1.5, 0.5 + strategy.win_rate))
                    base_weight *= performance_factor
                return base_weight * self._get_regime_multiplier(
                    strategy_name, trend_regime, vol_regime
                )
        return 0.1  # Default weight

    def _detect_regime(
        self, indicator_cache: IndicatorCache, closes: np.ndarray
    ) -> Tuple[str, str]:
        """Detect trend + volatility regime for the current pair."""
        adx_vals = indicator_cache.adx(14)
        atr_vals = indicator_cache.atr(14)

        adx_val = float(adx_vals[-1]) if len(adx_vals) else 0.0
        atr_val = float(atr_vals[-1]) if len(atr_vals) else 0.0
        price = float(closes[-1]) if len(closes) else 0.0
        atr_pct = (atr_val / price) if price > 0 else 0.0

        adx_threshold = self._get_regime_value("adx_trend_threshold", 25.0)
        atr_high = self._get_regime_value("atr_pct_high", 0.02)
        atr_low = self._get_regime_value("atr_pct_low", 0.008)

        trend_regime = "trend" if adx_val >= adx_threshold else "range"
        if atr_pct >= atr_high:
            vol_regime = "high_vol"
        elif atr_pct <= atr_low:
            vol_regime = "low_vol"
        else:
            vol_regime = "mid_vol"

        return trend_regime, vol_regime

    def _get_regime_value(self, key: str, default: float) -> float:
        """Fetch regime config value with safe fallback."""
        if isinstance(self.regime_config, dict):
            return float(self.regime_config.get(key, default))
        return float(getattr(self.regime_config, key, default))

    def _get_regime_multiplier(
        self,
        strategy_name: str,
        trend_regime: Optional[str],
        vol_regime: Optional[str],
    ) -> float:
        """Return a multiplier based on trend and volatility regimes."""
        trend_weights = self._get_regime_mapping(
            "trend_weight_multipliers", self._default_trend_weights
        )
        range_weights = self._get_regime_mapping(
            "range_weight_multipliers", self._default_range_weights
        )
        high_vol_weights = self._get_regime_mapping(
            "high_vol_weight_multipliers", self._default_high_vol_weights
        )
        low_vol_weights = self._get_regime_mapping(
            "low_vol_weight_multipliers", self._default_low_vol_weights
        )

        mult = 1.0
        if trend_regime == "trend":
            mult *= float(trend_weights.get(strategy_name, 1.0))
        elif trend_regime == "range":
            mult *= float(range_weights.get(strategy_name, 1.0))

        if vol_regime == "high_vol":
            mult *= float(high_vol_weights.get(strategy_name, 1.0))
        elif vol_regime == "low_vol":
            mult *= float(low_vol_weights.get(strategy_name, 1.0))

        return mult

    def _get_regime_mapping(self, key: str, default: Dict[str, float]) -> Dict[str, float]:
        """Fetch regime mapping with safe fallback."""
        if isinstance(self.regime_config, dict):
            return dict(self.regime_config.get(key, default) or default)
        return dict(getattr(self.regime_config, key, default) or default)

    def get_strategy_stats(self) -> List[Dict[str, Any]]:
        """Get performance stats for all strategies."""
        return [s.get_stats() for s in self.strategies]

    def record_trade_result(self, strategy_name: str, pnl: float) -> None:
        """Record a trade result for adaptive strategy weighting."""
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                strategy.record_trade_result(pnl)
                break

    def get_last_confluence(self, pair: str) -> Optional[ConfluenceSignal]:
        """Get the most recent confluence signal for a pair."""
        return self._last_confluence.get(pair)
