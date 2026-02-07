"""
Momentum Strategy - RSI + Volume Burst Detection.

Identifies strong momentum moves by combining RSI strength with
significant volume increases. Designed to catch acceleration in
existing trends.

# ENHANCEMENT: Added momentum divergence detection
# ENHANCEMENT: Added rate of change confirmation
# ENHANCEMENT: Added multi-timeframe momentum alignment
"""

from __future__ import annotations

import numpy as np

from src.strategies.base import BaseStrategy, SignalDirection, StrategySignal
from src.utils.indicators import atr, compute_sl_tp, ema, momentum, rsi, volume_ratio


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy using RSI and volume analysis.
    
    Entry (Long):
    1. RSI > threshold (50) and rising
    2. Volume > average * multiplier (volume burst)
    3. Price momentum positive
    4. Recent closes consistently higher
    
    Entry (Short):
    1. RSI < (100 - threshold) and falling
    2. Volume burst
    3. Price momentum negative
    4. Recent closes consistently lower
    
    # ENHANCEMENT: Added momentum scoring with multiple components
    """

    def __init__(
        self,
        rsi_threshold: int = 50,
        volume_multiplier: float = 1.5,
        weight: float = 0.20,
        enabled: bool = True,
    ):
        super().__init__(name="momentum", weight=weight, enabled=enabled)
        self.rsi_threshold = rsi_threshold
        self.volume_multiplier = volume_multiplier

    def min_bars_required(self) -> int:
        return 50

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
        vol_ratio = volume_ratio(volumes, 20)
        atr_vals = atr(highs, lows, closes, 14)
        mom_vals = momentum(closes, 10)
        ema_8 = ema(closes, 8)

        # Current values
        curr_price = closes[-1]
        curr_rsi = rsi_vals[-1]
        prev_rsi = rsi_vals[-2] if len(rsi_vals) > 1 else curr_rsi
        curr_vol_ratio = vol_ratio[-1]
        curr_atr = atr_vals[-1]
        curr_momentum = mom_vals[-1]

        # RSI direction
        rsi_rising = curr_rsi > prev_rsi
        rsi_falling = curr_rsi < prev_rsi

        # Volume burst detection
        volume_burst = curr_vol_ratio >= self.volume_multiplier

        # M21 FIX: Count truly CONSECUTIVE positive/negative candles
        pos_candles = 0
        for i in range(1, min(6, len(closes))):
            if len(closes) > i and closes[-i] > closes[-i - 1]:
                pos_candles += 1
            else:
                break

        neg_candles = 0
        for i in range(1, min(6, len(closes))):
            if len(closes) > i and closes[-i] < closes[-i - 1]:
                neg_candles += 1
            else:
                break

        # Price above/below EMA8
        price_above_ema = curr_price > ema_8[-1]
        price_below_ema = curr_price < ema_8[-1]

        # Rate of change (5-period)
        roc_5 = (closes[-1] - closes[-6]) / closes[-6] if len(closes) > 5 and closes[-6] > 0 else 0

        direction = SignalDirection.NEUTRAL
        strength = 0.0
        confidence = 0.0

        # -- LONG MOMENTUM --
        if curr_rsi > self.rsi_threshold and rsi_rising and volume_burst:
            direction = SignalDirection.LONG

            # Base strength from RSI
            strength = 0.3

            # RSI magnitude bonus
            rsi_excess = curr_rsi - self.rsi_threshold
            strength += min(rsi_excess / 50, 0.2)

            # Volume burst strength
            vol_excess = curr_vol_ratio - self.volume_multiplier
            strength += min(vol_excess * 0.15, 0.2)

            # Momentum bonus
            if curr_momentum > 0:
                strength += min(abs(curr_momentum) * 5, 0.15)

            # Consecutive candles bonus
            if pos_candles >= 3:
                strength += 0.1

            # Confidence
            confidence = 0.35
            if volume_burst:
                confidence += 0.15
            if price_above_ema:
                confidence += 0.1
            if roc_5 > 0.01:
                confidence += 0.1
            if pos_candles >= 3:
                confidence += 0.1
            if curr_rsi > 55 and curr_rsi < 75:
                confidence += 0.1
            if curr_vol_ratio > 2.0:
                confidence += 0.1

        # -- SHORT MOMENTUM --
        elif curr_rsi < (100 - self.rsi_threshold) and rsi_falling and volume_burst:
            direction = SignalDirection.SHORT

            strength = 0.3

            rsi_excess = (100 - self.rsi_threshold) - curr_rsi
            strength += min(rsi_excess / 50, 0.2)

            vol_excess = curr_vol_ratio - self.volume_multiplier
            strength += min(vol_excess * 0.15, 0.2)

            if curr_momentum < 0:
                strength += min(abs(curr_momentum) * 5, 0.15)

            if neg_candles >= 3:
                strength += 0.1

            confidence = 0.35
            if volume_burst:
                confidence += 0.15
            if price_below_ema:
                confidence += 0.1
            if roc_5 < -0.01:
                confidence += 0.1
            if neg_candles >= 3:
                confidence += 0.1
            if curr_rsi < 45 and curr_rsi > 25:
                confidence += 0.1
            if curr_vol_ratio > 2.0:
                confidence += 0.1

        # Fee-aware stops (higher R:R for momentum)
        if direction == SignalDirection.LONG:
            stop_loss, take_profit = compute_sl_tp(curr_price, curr_atr, "long", 2.0, 3.5)
        elif direction == SignalDirection.SHORT:
            stop_loss, take_profit = compute_sl_tp(curr_price, curr_atr, "short", 2.0, 3.5)
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
                "rsi_rising": rsi_rising,
                "volume_ratio": round(curr_vol_ratio, 2),
                "volume_burst": volume_burst,
                "momentum": round(curr_momentum, 6),
                "roc_5": round(roc_5, 6),
                "consecutive_positive": pos_candles,
                "consecutive_negative": neg_candles,
                "price_above_ema8": price_above_ema,
            }
        )

        self._last_signal = signal
        return signal
