"""
RSI Mean Reversion Strategy - Fades RSI extremes.

Uses a standard RSI to identify overbought/oversold conditions and
enters in the opposite direction when RSI begins to turn back.
"""

from __future__ import annotations

import numpy as np

from src.strategies.base import BaseStrategy, SignalDirection, StrategySignal
from src.utils.indicators import atr, compute_sl_tp, rsi


class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI mean reversion strategy.

    Entry (Long):
    1. RSI <= oversold threshold
    2. RSI turning upward

    Entry (Short):
    1. RSI >= overbought threshold
    2. RSI turning downward
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        trend_adjust: int = 5,
        range_adjust: int = -5,
        high_vol_adjust: int = 3,
        low_vol_adjust: int = -2,
        weight: float = 0.12,
        enabled: bool = True,
    ):
        super().__init__(name="rsi_mean_reversion", weight=weight, enabled=enabled)
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.trend_adjust = trend_adjust
        self.range_adjust = range_adjust
        self.high_vol_adjust = high_vol_adjust
        self.low_vol_adjust = low_vol_adjust

    def min_bars_required(self) -> int:
        return max(self.rsi_period + 10, 50)

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
            rsi_vals = cache.rsi(self.rsi_period)
            atr_vals = cache.atr(14)
        else:
            rsi_vals = rsi(closes, self.rsi_period)
            atr_vals = atr(highs, lows, closes, 14)

        fee_pct = kwargs.get("round_trip_fee_pct")
        curr_price = float(closes[-1])
        curr_rsi = float(rsi_vals[-1])
        prev_rsi = float(rsi_vals[-2]) if len(rsi_vals) > 1 else curr_rsi
        curr_atr = float(atr_vals[-1]) if len(atr_vals) else 0.0

        trend_regime = kwargs.get("trend_regime")
        vol_regime = kwargs.get("vol_regime")

        oversold = int(self.rsi_oversold)
        overbought = int(self.rsi_overbought)
        if trend_regime == "trend":
            oversold = max(5, oversold - self.trend_adjust)
            overbought = min(95, overbought + self.trend_adjust)
        elif trend_regime == "range":
            oversold = min(45, oversold + self.range_adjust)
            overbought = max(55, overbought - self.range_adjust)
        if vol_regime == "high_vol":
            oversold = max(5, oversold - self.high_vol_adjust)
            overbought = min(95, overbought + self.high_vol_adjust)
        elif vol_regime == "low_vol":
            oversold = min(45, oversold + self.low_vol_adjust)
            overbought = max(55, overbought - self.low_vol_adjust)

        oversold = max(5, min(45, oversold))
        overbought = min(95, max(55, overbought))

        rsi_rising = curr_rsi > prev_rsi
        rsi_falling = curr_rsi < prev_rsi

        direction = SignalDirection.NEUTRAL
        strength = 0.0
        confidence = 0.0

        if curr_rsi <= oversold and rsi_rising:
            direction = SignalDirection.LONG
            depth = (oversold - curr_rsi) / 30 if oversold > 0 else 0
            strength = 0.35 + min(max(depth, 0), 0.3)
            confidence = 0.35
            if curr_rsi < 25:
                confidence += 0.15
            if rsi_rising:
                confidence += 0.1
            if curr_rsi < 20:
                strength += 0.1

        elif curr_rsi >= overbought and rsi_falling:
            direction = SignalDirection.SHORT
            depth = (curr_rsi - overbought) / 30 if overbought < 100 else 0
            strength = 0.35 + min(max(depth, 0), 0.3)
            confidence = 0.35
            if curr_rsi > 75:
                confidence += 0.15
            if rsi_falling:
                confidence += 0.1
            if curr_rsi > 80:
                strength += 0.1

        if direction == SignalDirection.LONG:
            stop_loss, take_profit = compute_sl_tp(
                curr_price, curr_atr, "long", 2.0, 2.6, round_trip_fee_pct=fee_pct
            )
        elif direction == SignalDirection.SHORT:
            stop_loss, take_profit = compute_sl_tp(
                curr_price, curr_atr, "short", 2.0, 2.6, round_trip_fee_pct=fee_pct
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
                "rsi": round(curr_rsi, 2),
                "rsi_rising": rsi_rising,
                "rsi_falling": rsi_falling,
                "oversold": oversold,
                "overbought": overbought,
                "trend_regime": trend_regime,
                "vol_regime": vol_regime,
            },
        )

        self._last_signal = signal
        return signal
