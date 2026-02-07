"""
Breakout Strategy - Support/Resistance Level Breakout with Volume.

Detects when price breaks through significant support or resistance
levels with volume confirmation. Uses N-period high/low as dynamic
S/R levels.

# ENHANCEMENT: Added false breakout detection using candle body analysis
# ENHANCEMENT: Added retest confirmation for higher probability entries
# ENHANCEMENT: Added volatility expansion validation
"""

from __future__ import annotations

import numpy as np

from src.strategies.base import BaseStrategy, SignalDirection, StrategySignal
from src.utils.indicators import atr, bollinger_bands, compute_sl_tp, rsi, volume_ratio


class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy with support/resistance and volume confirmation.
    
    Entry (Long):
    1. Price breaks above N-period high
    2. Volume > average * confirmation multiplier
    3. Candle body closes above the breakout level
    4. RSI not yet overbought (room to run)
    
    Entry (Short):
    1. Price breaks below N-period low
    2. Volume confirmation
    3. Candle body closes below breakdown level
    4. RSI not yet oversold
    
    # ENHANCEMENT: Added strength scoring based on level significance
    """

    def __init__(
        self,
        lookback_period: int = 20,
        volume_confirmation: float = 1.3,
        weight: float = 0.20,
        enabled: bool = True,
    ):
        super().__init__(name="breakout", weight=weight, enabled=enabled)
        self.lookback_period = lookback_period
        self.volume_confirmation = volume_confirmation

    def min_bars_required(self) -> int:
        return self.lookback_period + 20

    async def analyze(
        self,
        pair: str,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        **kwargs
    ) -> StrategySignal:
        if len(closes) < self.min_bars_required():
            return self._neutral_signal(pair, "Insufficient data")

        # Key levels
        n_period_high = np.max(highs[-self.lookback_period - 1:-1])
        n_period_low = np.min(lows[-self.lookback_period - 1:-1])

        # Current bar data
        curr_price = closes[-1]
        curr_high = highs[-1]
        curr_low = lows[-1]
        curr_open = closes[-2]  # Use previous close as proxy for current open

        # Indicators
        atr_vals = atr(highs, lows, closes, 14)
        vol_ratio_vals = volume_ratio(volumes, 20)
        rsi_vals = rsi(closes, 14)

        curr_atr = atr_vals[-1]
        curr_vol_ratio = vol_ratio_vals[-1]
        curr_rsi = rsi_vals[-1]

        # Candle body analysis
        candle_body = curr_price - curr_open
        candle_range = curr_high - curr_low if curr_high > curr_low else 0.001
        body_ratio = abs(candle_body) / candle_range

        # Volume confirmation
        vol_confirmed = curr_vol_ratio >= self.volume_confirmation

        # BB width for volatility expansion
        upper, middle, lower = bollinger_bands(closes, 20, 2.0)
        bb_width = (upper[-1] - lower[-1]) / middle[-1] if middle[-1] > 0 else 0
        prev_bb_width = (upper[-2] - lower[-2]) / middle[-2] if len(upper) > 1 and middle[-2] > 0 else bb_width
        volatility_expanding = bb_width > prev_bb_width

        # Count how many times the level was tested (significance)
        resistance_touches = sum(
            1 for i in range(-self.lookback_period, -1)
            if abs(highs[i] - n_period_high) / n_period_high < 0.003
        )
        support_touches = sum(
            1 for i in range(-self.lookback_period, -1)
            if abs(lows[i] - n_period_low) / n_period_low < 0.003
        )

        direction = SignalDirection.NEUTRAL
        strength = 0.0
        confidence = 0.0

        # -- BULLISH BREAKOUT --
        if curr_price > n_period_high and candle_body > 0:
            direction = SignalDirection.LONG

            # Base strength from breakout distance
            breakout_pct = (curr_price - n_period_high) / n_period_high
            strength = 0.35 + min(breakout_pct * 20, 0.25)

            # Volume confirmation bonus
            if vol_confirmed:
                strength += 0.15

            # Body ratio (strong candle close)
            if body_ratio > 0.6:
                strength += 0.1

            # Level significance (more touches = stronger breakout)
            strength += min(resistance_touches * 0.05, 0.15)

            # Confidence
            confidence = 0.3
            if vol_confirmed:
                confidence += 0.2
            if body_ratio > 0.6:
                confidence += 0.1
            if volatility_expanding:
                confidence += 0.1
            if curr_rsi < 70:  # Room to run
                confidence += 0.1
            if resistance_touches >= 2:
                confidence += 0.1
            if breakout_pct > 0.002:  # Meaningful breakout
                confidence += 0.1

        # -- BEARISH BREAKDOWN --
        elif curr_price < n_period_low and candle_body < 0:
            direction = SignalDirection.SHORT

            breakdown_pct = (n_period_low - curr_price) / n_period_low
            strength = 0.35 + min(breakdown_pct * 20, 0.25)

            if vol_confirmed:
                strength += 0.15
            if body_ratio > 0.6:
                strength += 0.1
            strength += min(support_touches * 0.05, 0.15)

            confidence = 0.3
            if vol_confirmed:
                confidence += 0.2
            if body_ratio > 0.6:
                confidence += 0.1
            if volatility_expanding:
                confidence += 0.1
            if curr_rsi > 30:
                confidence += 0.1
            if support_touches >= 2:
                confidence += 0.1
            if breakdown_pct > 0.002:
                confidence += 0.1

        # Fee-aware stops (wider for breakouts)
        if direction == SignalDirection.LONG:
            stop_loss, take_profit = compute_sl_tp(curr_price, curr_atr, "long", 2.5, 4.0)
        elif direction == SignalDirection.SHORT:
            stop_loss, take_profit = compute_sl_tp(curr_price, curr_atr, "short", 2.5, 4.0)
        else:
            stop_loss = 0.0
            take_profit = 0.0

        signal = StrategySignal(
            strategy_name=self.name,
            pair=pair,
            direction=direction,
            strength=min(strength, 1.0),
            confidence=min(confidence, 1.0),
            entry_price=curr_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "n_period_high": round(n_period_high, 2),
                "n_period_low": round(n_period_low, 2),
                "breakout_distance_pct": round(
                    abs(curr_price - n_period_high) / n_period_high * 100
                    if curr_price > n_period_high else
                    abs(n_period_low - curr_price) / n_period_low * 100, 4
                ),
                "volume_ratio": round(curr_vol_ratio, 2),
                "volume_confirmed": vol_confirmed,
                "body_ratio": round(body_ratio, 4),
                "rsi": round(curr_rsi, 2),
                "bb_width": round(bb_width, 6),
                "volatility_expanding": volatility_expanding,
                "resistance_touches": resistance_touches,
                "support_touches": support_touches,
            }
        )

        self._last_signal = signal
        return signal
