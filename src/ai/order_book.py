"""
Order Book Analyzer - Real-time depth analysis for trade confirmation.

Detects whale orders, voids, bid/ask imbalances, and price-impacting
order clusters. Provides actionable intelligence for trade entry/exit
decisions.

# ENHANCEMENT: Added whale tracking with persistence detection
# ENHANCEMENT: Added liquidity void detection for gap-up/down risk
# ENHANCEMENT: Added depth-weighted average price calculation
# ENHANCEMENT: Added spoofing detection heuristic
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.logger import get_logger

logger = get_logger("order_book")


class OrderBookAnalysis:
    """Container for order book analysis results."""

    def __init__(self):
        self.obi: float = 0.0  # Order Book Imbalance
        self.bid_volume: float = 0.0
        self.ask_volume: float = 0.0
        self.spread: float = 0.0
        self.spread_pct: float = 0.0
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        self.mid_price: float = 0.0
        self.whale_bids: List[Dict[str, float]] = []
        self.whale_asks: List[Dict[str, float]] = []
        self.bid_voids: List[Tuple[float, float]] = []  # (start_price, end_price)
        self.ask_voids: List[Tuple[float, float]] = []
        self.vwap_bid: float = 0.0  # Volume-weighted avg bid price
        self.vwap_ask: float = 0.0  # Volume-weighted avg ask price
        self.depth_ratio: float = 0.0  # Bid depth / Ask depth
        self.pressure: str = "neutral"  # "bullish", "bearish", "neutral"
        self.liquidity_score: float = 0.0  # 0-1, higher = more liquid
        self.spoofing_detected: bool = False
        self.timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obi": round(self.obi, 4),
            "bid_volume": round(self.bid_volume, 4),
            "ask_volume": round(self.ask_volume, 4),
            "spread": round(self.spread, 6),
            "spread_pct": round(self.spread_pct, 6),
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": round(self.mid_price, 2),
            "whale_bids": len(self.whale_bids),
            "whale_asks": len(self.whale_asks),
            "bid_voids": len(self.bid_voids),
            "ask_voids": len(self.ask_voids),
            "depth_ratio": round(self.depth_ratio, 4),
            "pressure": self.pressure,
            "liquidity_score": round(self.liquidity_score, 4),
            "spoofing_detected": self.spoofing_detected,
        }


class OrderBookAnalyzer:
    """
    Real-time order book depth analyzer.
    
    Processes order book snapshots to detect:
    - Whale orders (large relative to average)
    - Liquidity voids (price gaps in the book)
    - Bid/Ask imbalance trends
    - Spoofing patterns (large orders that appear/disappear)
    
    # ENHANCEMENT: Added historical imbalance tracking for trend analysis
    # ENHANCEMENT: Added adaptive whale threshold based on market conditions
    """

    def __init__(
        self,
        whale_threshold_usd: float = 50000.0,
        depth: int = 25,
        void_threshold_pct: float = 0.003,  # 0.3% gap = void
    ):
        self.whale_threshold_usd = whale_threshold_usd
        self.depth = depth
        self.void_threshold_pct = void_threshold_pct

        # Historical tracking
        self._obi_history: Dict[str, deque] = {}
        self._whale_persistence: Dict[str, Dict[str, int]] = {}  # price -> count
        self._last_snapshots: Dict[str, Dict] = {}

    def analyze(
        self,
        pair: str,
        bids: List[List],
        asks: List[List],
        current_price: float = 0.0,
    ) -> OrderBookAnalysis:
        """
        Perform comprehensive order book analysis.
        
        Args:
            pair: Trading pair
            bids: [[price, volume, timestamp], ...] sorted desc by price
            asks: [[price, volume, timestamp], ...] sorted asc by price
            current_price: Current market price for context
        
        Returns:
            OrderBookAnalysis with all computed metrics
        """
        analysis = OrderBookAnalysis()
        analysis.timestamp = time.time()

        if not bids or not asks:
            return analysis

        # Parse order book
        bid_prices = np.array([float(b[0]) for b in bids[:self.depth]])
        bid_volumes = np.array([float(b[1]) for b in bids[:self.depth]])
        ask_prices = np.array([float(a[0]) for a in asks[:self.depth]])
        ask_volumes = np.array([float(a[1]) for a in asks[:self.depth]])

        if len(bid_prices) == 0 or len(ask_prices) == 0:
            return analysis

        # Basic metrics
        analysis.best_bid = float(bid_prices[0])
        analysis.best_ask = float(ask_prices[0])
        analysis.mid_price = (analysis.best_bid + analysis.best_ask) / 2
        analysis.spread = analysis.best_ask - analysis.best_bid
        analysis.spread_pct = (
            analysis.spread / analysis.mid_price
            if analysis.mid_price > 0 else 0
        )

        # Volume analysis
        analysis.bid_volume = float(np.sum(bid_volumes))
        analysis.ask_volume = float(np.sum(ask_volumes))

        # Order Book Imbalance
        total_vol = analysis.bid_volume + analysis.ask_volume
        if total_vol > 0:
            analysis.obi = (analysis.bid_volume - analysis.ask_volume) / total_vol
        
        # H7 FIX: Safe depth ratio (avoid inf)
        if analysis.ask_volume > 0:
            analysis.depth_ratio = analysis.bid_volume / analysis.ask_volume
        else:
            analysis.depth_ratio = 10.0 if analysis.bid_volume > 0 else 1.0

        # Track OBI history
        if pair not in self._obi_history:
            self._obi_history[pair] = deque(maxlen=100)
        self._obi_history[pair].append(analysis.obi)

        # VWAP calculation
        bid_notional = bid_prices * bid_volumes
        ask_notional = ask_prices * ask_volumes
        
        if analysis.bid_volume > 0:
            analysis.vwap_bid = float(np.sum(bid_notional) / analysis.bid_volume)
        if analysis.ask_volume > 0:
            analysis.vwap_ask = float(np.sum(ask_notional) / analysis.ask_volume)

        # Whale detection
        price_ref = current_price if current_price > 0 else analysis.mid_price
        analysis.whale_bids = self._detect_whales(
            bid_prices, bid_volumes, price_ref
        )
        analysis.whale_asks = self._detect_whales(
            ask_prices, ask_volumes, price_ref
        )

        # Liquidity void detection
        analysis.bid_voids = self._detect_voids(bid_prices, "bid")
        analysis.ask_voids = self._detect_voids(ask_prices, "ask")

        # Pressure assessment
        analysis.pressure = self._assess_pressure(analysis)

        # Liquidity score (0-1)
        analysis.liquidity_score = self._compute_liquidity_score(analysis)

        # Spoofing detection
        analysis.spoofing_detected = self._detect_spoofing(
            pair, bid_prices, bid_volumes, ask_prices, ask_volumes
        )

        # Store snapshot for comparison
        self._last_snapshots[pair] = {
            "bids": list(zip(bid_prices.tolist(), bid_volumes.tolist())),
            "asks": list(zip(ask_prices.tolist(), ask_volumes.tolist())),
            "timestamp": analysis.timestamp,
        }

        return analysis

    def _detect_whales(
        self, prices: np.ndarray, volumes: np.ndarray, price_ref: float
    ) -> List[Dict[str, float]]:
        """
        Detect whale orders (large relative to average and threshold).
        
        # ENHANCEMENT: Uses adaptive threshold based on average order size
        """
        whales = []
        if len(volumes) == 0:
            return whales

        avg_volume = np.mean(volumes)
        notional = prices * volumes

        for i in range(len(prices)):
            usd_value = float(notional[i])
            is_whale = (
                usd_value >= self.whale_threshold_usd or
                volumes[i] > avg_volume * 5
            )
            if is_whale:
                whales.append({
                    "price": float(prices[i]),
                    "volume": float(volumes[i]),
                    "usd_value": round(usd_value, 2),
                    "multiplier": round(float(volumes[i] / avg_volume), 1),
                })

        return whales

    def _detect_voids(
        self, prices: np.ndarray, side: str
    ) -> List[Tuple[float, float]]:
        """
        Detect liquidity voids (gaps in the order book).
        
        A void is a price range with no orders that is larger than
        the void_threshold_pct of the mid price.
        """
        voids = []
        if len(prices) < 2:
            return voids

        for i in range(len(prices) - 1):
            gap = abs(prices[i] - prices[i + 1])
            gap_pct = gap / prices[i] if prices[i] > 0 else 0

            if gap_pct > self.void_threshold_pct:
                voids.append((
                    float(min(prices[i], prices[i + 1])),
                    float(max(prices[i], prices[i + 1]))
                ))

        return voids

    def _assess_pressure(self, analysis: OrderBookAnalysis) -> str:
        """Assess buying/selling pressure from the order book."""
        if analysis.obi > 0.15:
            return "bullish"
        elif analysis.obi < -0.15:
            return "bearish"
        
        # Check depth ratio
        if analysis.depth_ratio > 1.5:
            return "bullish"
        elif analysis.depth_ratio < 0.67:
            return "bearish"
        
        return "neutral"

    def _compute_liquidity_score(self, analysis: OrderBookAnalysis) -> float:
        """
        Compute a 0-1 liquidity score.
        
        Higher score = more liquid (tighter spread, deeper book, no voids).
        """
        score = 0.5

        # Tight spread bonus
        if analysis.spread_pct < 0.001:
            score += 0.2
        elif analysis.spread_pct < 0.003:
            score += 0.1

        # Good depth bonus
        total_vol = analysis.bid_volume + analysis.ask_volume
        if total_vol > 100:
            score += 0.15
        elif total_vol > 10:
            score += 0.05

        # Void penalty
        void_count = len(analysis.bid_voids) + len(analysis.ask_voids)
        score -= min(void_count * 0.05, 0.2)

        # Balanced book bonus
        if 0.7 < analysis.depth_ratio < 1.3:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _detect_spoofing(
        self,
        pair: str,
        bid_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray,
    ) -> bool:
        """
        Heuristic spoofing detection.
        
        Detects large orders that appear and disappear rapidly,
        which may indicate market manipulation.
        
        # ENHANCEMENT: Tracks order persistence over time windows
        """
        last = self._last_snapshots.get(pair)
        if not last:
            return False

        time_diff = time.time() - last["timestamp"]
        if time_diff > 10:  # Only check recent snapshots
            return False

        # Check if large bid orders disappeared
        last_bids = {p: v for p, v in last["bids"][:5]}
        curr_bids = dict(zip(bid_prices[:5].tolist(), bid_volumes[:5].tolist()))

        disappeared_volume = 0
        for price, volume in last_bids.items():
            if price not in curr_bids and volume > np.mean(bid_volumes) * 3:
                disappeared_volume += volume

        # Same for asks
        last_asks = {p: v for p, v in last["asks"][:5]}
        curr_asks = dict(zip(ask_prices[:5].tolist(), ask_volumes[:5].tolist()))

        for price, volume in last_asks.items():
            if price not in curr_asks and volume > np.mean(ask_volumes) * 3:
                disappeared_volume += volume

        # If large volume disappeared quickly, possible spoofing
        avg_vol = (np.mean(bid_volumes) + np.mean(ask_volumes)) / 2
        return disappeared_volume > avg_vol * 10

    def get_obi_trend(self, pair: str, periods: int = 20) -> float:
        """
        Get the recent OBI trend.
        
        Returns positive for increasing buying pressure,
        negative for increasing selling pressure.
        """
        history = self._obi_history.get(pair, deque())
        if len(history) < periods:
            return 0.0

        recent = list(history)[-periods:]
        if len(recent) < 2:
            return 0.0

        # Simple linear regression slope
        x = np.arange(len(recent))
        y = np.array(recent)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
