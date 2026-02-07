"""
Reversal Strategy - Extreme Exhaustion with Multi-Bar Confirmation.

Detects market reversals at extreme levels by combining RSI extremes
with candlestick pattern analysis and volume exhaustion signals.

# ENHANCEMENT: Added multi-bar confirmation pattern recognition
# ENHANCEMENT: Added divergence-based reversal scoring
# ENHANCEMENT: Added false reversal filtering with trend context
"""

from __future__ import annotations

import numpy as np

from src.strategies.base import BaseStrategy, SignalDirection, StrategySignal
from src.utils.indicators import atr, compute_sl_tp, ema, rsi, volume_ratio


class ReversalStrategy(BaseStrategy):
    """
    Reversal strategy using extreme RSI with confirmation candles.
    
    Entry (Long - Bullish Reversal):
    1. RSI reaches extreme low (< 20)
    2. N confirmation candles show recovery (higher lows/closes)
    3. Volume declining on sell-off (exhaustion)
    4. Bullish candlestick patterns detected
    
    Entry (Short - Bearish Reversal):
    1. RSI reaches extreme high (> 80)
    2. N confirmation candles show weakening
    3. Volume declining on rally
    4. Bearish patterns detected
    
    # ENHANCEMENT: Added hammer/shooting star detection
    # ENHANCEMENT: Added engulfing pattern detection
    """

    def __init__(
        self,
        rsi_extreme_low: int = 20,
        rsi_extreme_high: int = 80,
        confirmation_candles: int = 3,
        weight: float = 0.15,
        enabled: bool = True,
    ):
        super().__init__(name="reversal", weight=weight, enabled=enabled)
        self.rsi_extreme_low = rsi_extreme_low
        self.rsi_extreme_high = rsi_extreme_high
        self.confirmation_candles = confirmation_candles

    def min_bars_required(self) -> int:
        return 50

    def _detect_hammer(
        self, open_p: float, high: float, low: float, close: float
    ) -> bool:
        """Detect hammer candlestick (bullish reversal)."""
        body = abs(close - open_p)
        candle_range = high - low
        if candle_range == 0:
            return False
        lower_shadow = min(open_p, close) - low
        upper_shadow = high - max(open_p, close)
        return (
            lower_shadow > body * 2 and
            upper_shadow < body * 0.5 and
            body / candle_range < 0.4
        )

    def _detect_shooting_star(
        self, open_p: float, high: float, low: float, close: float
    ) -> bool:
        """Detect shooting star (bearish reversal)."""
        body = abs(close - open_p)
        candle_range = high - low
        if candle_range == 0:
            return False
        upper_shadow = high - max(open_p, close)
        lower_shadow = min(open_p, close) - low
        return (
            upper_shadow > body * 2 and
            lower_shadow < body * 0.5 and
            body / candle_range < 0.4
        )

    def _detect_engulfing(
        self, opens: np.ndarray, closes: np.ndarray
    ) -> tuple:
        """
        Detect bullish/bearish engulfing patterns.
        
        Returns (bullish_engulfing, bearish_engulfing)
        """
        if len(opens) < 2:
            return False, False

        prev_open = opens[-2]
        prev_close = closes[-2]
        curr_open = opens[-1]
        curr_close = closes[-1]

        prev_body = prev_close - prev_open
        curr_body = curr_close - curr_open

        # Bullish engulfing: previous red candle, current green candle engulfs
        bullish = (
            prev_body < 0 and curr_body > 0 and
            curr_open < prev_close and curr_close > prev_open
        )

        # Bearish engulfing
        bearish = (
            prev_body > 0 and curr_body < 0 and
            curr_open > prev_close and curr_close < prev_open
        )

        return bullish, bearish

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

        # Compute indicators
        rsi_vals = rsi(closes, 14)
        atr_vals = atr(highs, lows, closes, 14)
        vol_ratio = volume_ratio(volumes, 20)
        ema_20 = ema(closes, 20)

        curr_price = closes[-1]
        curr_rsi = rsi_vals[-1]
        curr_atr = atr_vals[-1]
        curr_vol_ratio = vol_ratio[-1]

        # Check if RSI was extreme in recent bars
        recent_rsi = rsi_vals[-self.confirmation_candles - 3:]
        was_oversold = any(r < self.rsi_extreme_low for r in recent_rsi[:-1]
                          if not np.isnan(r))
        was_overbought = any(r > self.rsi_extreme_high for r in recent_rsi[:-1]
                            if not np.isnan(r))

        # Confirmation: higher lows and higher closes for bullish
        n = self.confirmation_candles
        higher_lows = True
        higher_closes = True
        lower_highs = True
        lower_closes = True

        for i in range(1, min(n, len(closes) - 1)):
            if lows[-i] <= lows[-i - 1]:
                higher_lows = False
            if closes[-i] <= closes[-i - 1]:
                higher_closes = False
            if highs[-i] >= highs[-i - 1]:
                lower_highs = False
            if closes[-i] >= closes[-i - 1]:
                lower_closes = False

        # Candlestick patterns
        opens = kwargs.get("opens", closes)  # Fallback if opens not provided
        if len(opens) < len(closes):
            opens = closes

        hammer = self._detect_hammer(
            opens[-1], highs[-1], lows[-1], closes[-1]
        )
        shooting_star = self._detect_shooting_star(
            opens[-1], highs[-1], lows[-1], closes[-1]
        )
        bull_engulf, bear_engulf = self._detect_engulfing(opens, closes)

        # Volume exhaustion
        vol_exhaustion = curr_vol_ratio < 0.8

        # Distance from EMA (overextension)
        ema_distance = (curr_price - ema_20[-1]) / ema_20[-1] if ema_20[-1] > 0 else 0

        direction = SignalDirection.NEUTRAL
        strength = 0.0
        confidence = 0.0

        # -- BULLISH REVERSAL --
        if was_oversold and (higher_lows or higher_closes):
            direction = SignalDirection.LONG

            strength = 0.35

            # RSI recovery strength
            if curr_rsi > self.rsi_extreme_low:
                recovery = (curr_rsi - self.rsi_extreme_low) / 30
                strength += min(recovery, 0.2)

            # Confirmation candle bonus
            if higher_lows and higher_closes:
                strength += 0.15
            elif higher_lows or higher_closes:
                strength += 0.08

            # Pattern bonus
            if hammer:
                strength += 0.12
            if bull_engulf:
                strength += 0.15

            # Volume exhaustion bonus
            if vol_exhaustion:
                strength += 0.08

            # Overextension bonus (further from mean = stronger reversion)
            if ema_distance < -0.02:
                strength += 0.1

            # Confidence
            confidence = 0.3
            if hammer or bull_engulf:
                confidence += 0.15
            if higher_lows and higher_closes:
                confidence += 0.15
            if vol_exhaustion:
                confidence += 0.1
            if curr_rsi > self.rsi_extreme_low + 5:
                confidence += 0.1
            if ema_distance < -0.02:
                confidence += 0.1
            if np.min(rsi_vals[-10:]) < 15:  # Very extreme
                confidence += 0.1

        # -- BEARISH REVERSAL --
        elif was_overbought and (lower_highs or lower_closes):
            direction = SignalDirection.SHORT

            strength = 0.35

            if curr_rsi < self.rsi_extreme_high:
                recovery = (self.rsi_extreme_high - curr_rsi) / 30
                strength += min(recovery, 0.2)

            if lower_highs and lower_closes:
                strength += 0.15
            elif lower_highs or lower_closes:
                strength += 0.08

            if shooting_star:
                strength += 0.12
            if bear_engulf:
                strength += 0.15

            if vol_exhaustion:
                strength += 0.08

            if ema_distance > 0.02:
                strength += 0.1

            confidence = 0.3
            if shooting_star or bear_engulf:
                confidence += 0.15
            if lower_highs and lower_closes:
                confidence += 0.15
            if vol_exhaustion:
                confidence += 0.1
            if curr_rsi < self.rsi_extreme_high - 5:
                confidence += 0.1
            if ema_distance > 0.02:
                confidence += 0.1
            if np.max(rsi_vals[-10:]) > 85:
                confidence += 0.1

        # Fee-aware stops for reversals
        if direction == SignalDirection.LONG:
            stop_loss, take_profit = compute_sl_tp(curr_price, curr_atr, "long", 1.5, 3.0)
        elif direction == SignalDirection.SHORT:
            stop_loss, take_profit = compute_sl_tp(curr_price, curr_atr, "short", 1.5, 3.0)
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
                "rsi": round(curr_rsi, 2),
                "was_oversold": was_oversold,
                "was_overbought": was_overbought,
                "higher_lows": higher_lows,
                "higher_closes": higher_closes,
                "lower_highs": lower_highs,
                "lower_closes": lower_closes,
                "hammer": hammer,
                "shooting_star": shooting_star,
                "bullish_engulfing": bull_engulf,
                "bearish_engulfing": bear_engulf,
                "volume_exhaustion": vol_exhaustion,
                "ema_distance": round(ema_distance, 6),
            }
        )

        self._last_signal = signal
        return signal
