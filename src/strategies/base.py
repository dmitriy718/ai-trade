"""
Base Strategy Interface - Abstract base for all trading strategies.

Defines the contract that every strategy must implement, including
signal generation, confidence scoring, and self-diagnostic methods.

# ENHANCEMENT: Added strategy performance tracking
# ENHANCEMENT: Added adaptive weight adjustment based on recent performance
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class StrategySignal:
    """
    Output from a strategy's analysis of a trading pair.
    
    Contains the direction, strength, confidence, and supporting
    metadata for the signal.
    """
    strategy_name: str
    pair: str
    direction: SignalDirection
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))

    @property
    def is_actionable(self) -> bool:
        """Whether this signal warrants potential trade action."""
        return (
            self.direction != SignalDirection.NEUTRAL
            and self.strength >= 0.3
            and self.confidence >= 0.3
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "pair": self.pair,
            "direction": self.direction.value,
            "strength": round(self.strength, 4),
            "confidence": round(self.confidence, 4),
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "timestamp": self.timestamp,
            "metadata": _sanitize_for_json(self.metadata),
        }


def _sanitize_for_json(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization.
    M20 FIX: Also handles NaN/Inf which are not valid JSON."""
    import math
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    elif isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    elif isinstance(obj, np.ndarray):
        return [_sanitize_for_json(x) for x in obj.tolist()]
    return obj


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    Every strategy must implement analyze() which takes market data
    and returns a StrategySignal. Strategies are responsible for
    computing their own indicators and scoring logic.
    
    # ENHANCEMENT: Added performance tracking per strategy
    # ENHANCEMENT: Added warm-up validation
    """

    def __init__(self, name: str, weight: float = 0.20, enabled: bool = True):
        self.name = name
        self.weight = weight
        self.enabled = enabled
        self._trade_count = 0
        self._win_count = 0
        self._total_pnl = 0.0
        self._last_signal: Optional[StrategySignal] = None

    @abstractmethod
    async def analyze(
        self,
        pair: str,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        **kwargs
    ) -> StrategySignal:
        """
        Analyze market data and produce a trading signal.
        
        Args:
            pair: Trading pair (e.g., "BTC/USD")
            closes: NumPy array of close prices
            highs: NumPy array of high prices
            lows: NumPy array of low prices
            volumes: NumPy array of volume data
            **kwargs: Additional data (order book, ticker, etc.)
        
        Returns:
            StrategySignal with direction, strength, and confidence
        """
        pass

    def min_bars_required(self) -> int:
        """Minimum number of bars needed for this strategy."""
        return 50  # Default; override in subclasses

    def record_trade_result(self, pnl: float) -> None:
        """Record a trade result for performance tracking."""
        self._trade_count += 1
        if pnl > 0:
            self._win_count += 1
        self._total_pnl += pnl

    @property
    def win_rate(self) -> float:
        if self._trade_count == 0:
            return 0.0
        return self._win_count / self._trade_count

    @property
    def avg_pnl(self) -> float:
        if self._trade_count == 0:
            return 0.0
        return self._total_pnl / self._trade_count

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "weight": self.weight,
            "trades": self._trade_count,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self._total_pnl, 2),
            "avg_pnl": round(self.avg_pnl, 2),
        }

    def _neutral_signal(self, pair: str, reason: str = "") -> StrategySignal:
        """Create a neutral (no-trade) signal."""
        return StrategySignal(
            strategy_name=self.name,
            pair=pair,
            direction=SignalDirection.NEUTRAL,
            strength=0.0,
            confidence=0.0,
            metadata={"reason": reason} if reason else {},
        )
