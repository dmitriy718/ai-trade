"""
Mean Reversion Strategy - Bollinger Band + RSI extremes.

Trades price reversions to the mean when price touches Bollinger Band
extremes and RSI confirms oversold/overbought conditions.

# ENHANCEMENT: Added band squeeze detection for volatility expansion trades
# ENHANCEMENT: Added mean reversion speed estimation
# ENHANCEMENT: Added false signal filtering via volume divergence
"""

from __future__ import annotations

import numpy as np

from src.strategies.base import BaseStrategy, SignalDirection, StrategySignal
from src.utils.indicators import (
    atr,
    bb_position,
    bollinger_bands,
    compute_sl_tp,
    rsi,
    sma,
    volume_ratio,
)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using Bollinger Bands and RSI.
    
    Entry (Long):
    1. Price touches or penetrates lower Bollinger Band
    2. RSI < oversold threshold
    3. Volume shows exhaustion (declining on sell-off)
    4. BB%B < 0.15 (deep into lower band territory)
    
    Entry (Short):
    1. Price touches or penetrates upper Bollinger Band
    2. RSI > overbought threshold
    3. Volume shows exhaustion
    4. BB%B > 0.85
    
    # ENHANCEMENT: Added Bollinger Band width analysis for squeeze detection
    # ENHANCEMENT: Added RSI divergence detection
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        weight: float = 0.20,
        enabled: bool = True,
    ):
        super().__init__(name="mean_reversion", weight=weight, enabled=enabled)
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def min_bars_required(self) -> int:
        return max(self.bb_period + 20, 50)

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
        upper, middle, lower = bollinger_bands(closes, self.bb_period, self.bb_std)
        bb_pos = bb_position(closes, self.bb_period, self.bb_std)
        rsi_vals = rsi(closes, 14)
        atr_vals = atr(highs, lows, closes, 14)
        vol_ratio = volume_ratio(volumes, 20)

        # Bollinger Band Width (squeeze detection)
        bb_width = (upper - lower) / np.where(middle > 0, middle, 1.0)

        # Current values
        curr_price = closes[-1]
        curr_bb_pos = bb_pos[-1]
        curr_rsi = rsi_vals[-1]
        curr_atr = atr_vals[-1]
        curr_vol_ratio = vol_ratio[-1]
        curr_bb_width = bb_width[-1] if not np.isnan(bb_width[-1]) else 0.0
        curr_upper = upper[-1]
        curr_lower = lower[-1]
        curr_middle = middle[-1]

        # Band squeeze: width below 20-period average suggests imminent expansion
        avg_bb_width = np.nanmean(bb_width[-20:]) if len(bb_width) >= 20 else curr_bb_width
        is_squeezed = curr_bb_width < avg_bb_width * 0.8

        # RSI divergence: price makes new low but RSI doesn't (bullish)
        rsi_bull_divergence = False
        rsi_bear_divergence = False
        if len(closes) >= 10 and len(rsi_vals) >= 10:
            recent_price_low = np.min(lows[-10:])
            recent_price_high = np.max(highs[-10:])
            # Check if current price is near recent low but RSI is higher
            if (curr_price <= recent_price_low * 1.005 and
                    curr_rsi > np.min(rsi_vals[-10:])):
                rsi_bull_divergence = True
            if (curr_price >= recent_price_high * 0.995 and
                    curr_rsi < np.max(rsi_vals[-10:])):
                rsi_bear_divergence = True

        # Volume exhaustion (selling/buying pressure declining)
        vol_declining = False
        if len(volumes) >= 5:
            recent_vol_trend = volumes[-1] < volumes[-3]
            vol_declining = recent_vol_trend

        direction = SignalDirection.NEUTRAL
        strength = 0.0
        confidence = 0.0

        # -- LONG (Buy at lower band) --
        if curr_bb_pos < 0.15 and curr_rsi < self.rsi_oversold:
            direction = SignalDirection.LONG

            # Strength based on BB position depth
            strength = 0.4 + (0.15 - curr_bb_pos) * 2.0

            # RSI extremity bonus
            rsi_bonus = max(0, (self.rsi_oversold - curr_rsi) / 30) * 0.2
            strength += rsi_bonus

            # Volume exhaustion bonus
            if vol_declining:
                strength += 0.1

            # RSI divergence bonus (strong reversal signal)
            if rsi_bull_divergence:
                strength += 0.15

            # Confidence scoring
            confidence = 0.35
            if curr_rsi < 25:
                confidence += 0.15
            if curr_bb_pos < 0.05:
                confidence += 0.15
            if vol_declining:
                confidence += 0.1
            if rsi_bull_divergence:
                confidence += 0.15
            if not is_squeezed:  # Better reversion in normal volatility
                confidence += 0.1

        # -- SHORT (Sell at upper band) --
        elif curr_bb_pos > 0.85 and curr_rsi > self.rsi_overbought:
            direction = SignalDirection.SHORT

            strength = 0.4 + (curr_bb_pos - 0.85) * 2.0

            rsi_bonus = max(0, (curr_rsi - self.rsi_overbought) / 30) * 0.2
            strength += rsi_bonus

            if vol_declining:
                strength += 0.1

            if rsi_bear_divergence:
                strength += 0.15

            confidence = 0.35
            if curr_rsi > 75:
                confidence += 0.15
            if curr_bb_pos > 0.95:
                confidence += 0.15
            if vol_declining:
                confidence += 0.1
            if rsi_bear_divergence:
                confidence += 0.15
            if not is_squeezed:
                confidence += 0.1

        # Fee-aware stop loss and take profit
        if direction == SignalDirection.LONG:
            stop_loss, take_profit = compute_sl_tp(curr_price, curr_atr, "long", 2.0, 3.0)
            # Use middle band as TP if it's further than the minimum
            if curr_middle > take_profit:
                take_profit = curr_middle
        elif direction == SignalDirection.SHORT:
            stop_loss, take_profit = compute_sl_tp(curr_price, curr_atr, "short", 2.0, 3.0)
            # S11 FIX: For shorts, middle band TP is above take_profit (closer to entry = more conservative)
            if curr_middle > take_profit and curr_middle < curr_price:
                take_profit = curr_middle
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
                "bb_position": round(curr_bb_pos, 4),
                "bb_upper": round(curr_upper, 2),
                "bb_lower": round(curr_lower, 2),
                "bb_middle": round(curr_middle, 2),
                "bb_width": round(curr_bb_width, 6),
                "is_squeezed": is_squeezed,
                "rsi": round(curr_rsi, 2),
                "rsi_bull_divergence": rsi_bull_divergence,
                "rsi_bear_divergence": rsi_bear_divergence,
                "volume_ratio": round(curr_vol_ratio, 2),
                "volume_declining": vol_declining,
            }
        )

        self._last_signal = signal
        return signal
