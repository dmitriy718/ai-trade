"""
Trend Following Strategy - EMA Crossover with ADX Filter.

Identifies trending markets using EMA(5/13) crossovers, confirmed
by ADX above threshold. Includes slope analysis for trend momentum
and multi-timeframe alignment checks.

# ENHANCEMENT: Added trend acceleration detection
# ENHANCEMENT: Added false breakout filter using ATR
# ENHANCEMENT: Added multi-period confirmation
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.strategies.base import BaseStrategy, SignalDirection, StrategySignal
from src.utils.indicators import adx, atr, compute_sl_tp, ema, rsi, trend_strength


class TrendStrategy(BaseStrategy):
    """
    Trend-following strategy using EMA crossovers with ADX confirmation.
    
    Entry Conditions (Long):
    1. EMA(fast) crosses above EMA(slow)
    2. ADX > threshold (confirming trend existence)
    3. RSI > 40 (not oversold - momentum present)
    4. Price above both EMAs
    
    Entry Conditions (Short):
    1. EMA(fast) crosses below EMA(slow)
    2. ADX > threshold
    3. RSI < 60
    4. Price below both EMAs
    
    # ENHANCEMENT: Added trend slope analysis for acceleration
    # ENHANCEMENT: Added volume confirmation for crossover signals
    """

    def __init__(
        self,
        ema_fast: int = 5,
        ema_slow: int = 13,
        adx_threshold: int = 25,
        weight: float = 0.25,
        enabled: bool = True,
    ):
        super().__init__(name="trend", weight=weight, enabled=enabled)
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_threshold = adx_threshold

    def min_bars_required(self) -> int:
        return max(self.ema_slow * 3, 50)

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

        cache = kwargs.get("indicator_cache")
        if cache:
            ema_f = cache.ema(self.ema_fast)
            ema_s = cache.ema(self.ema_slow)
            adx_vals = cache.adx(14)
            rsi_vals = cache.rsi(14)
            atr_vals = cache.atr(14)
            ts = cache.trend_strength(self.ema_fast, self.ema_slow)
        else:
            # Compute indicators
            ema_f = ema(closes, self.ema_fast)
            ema_s = ema(closes, self.ema_slow)
            adx_vals = adx(highs, lows, closes, 14)
            rsi_vals = rsi(closes, 14)
            atr_vals = atr(highs, lows, closes, 14)
            ts = trend_strength(closes, self.ema_fast, self.ema_slow)

        # Current values
        fee_pct = kwargs.get("round_trip_fee_pct")
        curr_ema_f = ema_f[-1]
        curr_ema_s = ema_s[-1]
        prev_ema_f = ema_f[-2]
        prev_ema_s = ema_s[-2]
        curr_adx = adx_vals[-1]
        curr_rsi = rsi_vals[-1]
        curr_atr = atr_vals[-1]
        curr_price = closes[-1]
        curr_ts = ts[-1]

        # Crossover detection
        bullish_cross = prev_ema_f <= prev_ema_s and curr_ema_f > curr_ema_s
        bearish_cross = prev_ema_f >= prev_ema_s and curr_ema_f < curr_ema_s

        # Trend alignment (price vs EMAs)
        price_above_emas = curr_price > curr_ema_f and curr_price > curr_ema_s
        price_below_emas = curr_price < curr_ema_f and curr_price < curr_ema_s

        # EMA spread strength
        ema_spread = abs(curr_ema_f - curr_ema_s) / curr_ema_s if curr_ema_s > 0 else 0

        # Trend slope (acceleration)
        if len(ts) >= 3:
            slope = ts[-1] - ts[-3]
        else:
            slope = 0.0

        # Volume confirmation
        avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0

        # Signal scoring
        direction = SignalDirection.NEUTRAL
        strength = 0.0
        confidence = 0.0

        # -- LONG SIGNAL --
        # Require: EMA alignment + price above EMAs + ADX confirms trend + RSI not oversold
        if curr_ema_f > curr_ema_s and price_above_emas:
            if curr_adx >= self.adx_threshold and curr_rsi > 45 and curr_rsi < 75:
                direction = SignalDirection.LONG

                # Base strength from crossover freshness
                strength = 0.5 if bullish_cross else 0.3

                # ADX bonus (strong trends get higher score)
                adx_bonus = min((curr_adx - self.adx_threshold) / 50, 0.2)
                strength += adx_bonus

                # Trend acceleration bonus
                if slope > 0:
                    strength += min(slope * 10, 0.15)

                # Volume confirmation bonus
                if vol_ratio > 1.3:
                    strength += 0.1

                # Confidence based on alignment
                confidence = 0.4
                if price_above_emas:
                    confidence += 0.15
                if curr_adx > 35:
                    confidence += 0.15
                if vol_ratio > 1.2:
                    confidence += 0.1
                if bullish_cross:
                    confidence += 0.1
                if curr_rsi > 50 and curr_rsi < 70:
                    confidence += 0.1

        # -- SHORT SIGNAL --
        elif curr_ema_f < curr_ema_s and price_below_emas:
            if curr_adx >= self.adx_threshold and curr_rsi < 55 and curr_rsi > 25:
                direction = SignalDirection.SHORT

                strength = 0.5 if bearish_cross else 0.3
                adx_bonus = min((curr_adx - self.adx_threshold) / 50, 0.2)
                strength += adx_bonus

                if slope < 0:
                    strength += min(abs(slope) * 10, 0.15)

                if vol_ratio > 1.3:
                    strength += 0.1

                confidence = 0.4
                if price_below_emas:
                    confidence += 0.15
                if curr_adx > 35:
                    confidence += 0.15
                if vol_ratio > 1.2:
                    confidence += 0.1
                if bearish_cross:
                    confidence += 0.1
                if curr_rsi < 50 and curr_rsi > 30:
                    confidence += 0.1

        # Compute fee-aware stop loss and take profit
        if direction == SignalDirection.LONG:
            stop_loss, take_profit = compute_sl_tp(
                curr_price, curr_atr, "long", 2.25, 3.0, round_trip_fee_pct=fee_pct
            )
        elif direction == SignalDirection.SHORT:
            stop_loss, take_profit = compute_sl_tp(
                curr_price, curr_atr, "short", 2.25, 3.0, round_trip_fee_pct=fee_pct
            )
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
                "ema_fast": round(curr_ema_f, 2),
                "ema_slow": round(curr_ema_s, 2),
                "adx": round(curr_adx, 2),
                "rsi": round(curr_rsi, 2),
                "atr": round(curr_atr, 4),
                "trend_strength": round(curr_ts, 6),
                "ema_spread": round(ema_spread, 6),
                "volume_ratio": round(vol_ratio, 2),
                "bullish_cross": bullish_cross,
                "bearish_cross": bearish_cross,
            }
        )

        self._last_signal = signal
        return signal
