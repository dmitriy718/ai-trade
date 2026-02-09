"""
Indicator Cache - Shared, lazy indicator computation for strategies.

Creates a per-scan cache so common indicators are computed once and
reused across strategies.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.utils.indicators import (
    adx,
    atr,
    bb_position,
    bollinger_bands,
    ema,
    keltner_channels,
    keltner_position,
    macd,
    momentum,
    rsi,
    trend_strength,
    volume_ratio,
)


class IndicatorCache:
    """Lazy indicator cache bound to a specific OHLCV snapshot."""

    def __init__(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
    ):
        self.closes = closes
        self.highs = highs
        self.lows = lows
        self.volumes = volumes

        self._ema: Dict[int, np.ndarray] = {}
        self._rsi: Dict[int, np.ndarray] = {}
        self._atr: Dict[int, np.ndarray] = {}
        self._adx: Dict[int, np.ndarray] = {}
        self._vol_ratio: Dict[int, np.ndarray] = {}
        self._momentum: Dict[int, np.ndarray] = {}
        self._trend_strength: Dict[Tuple[int, int], np.ndarray] = {}
        self._bb: Dict[Tuple[int, float], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._bb_pos: Dict[Tuple[int, float], np.ndarray] = {}
        self._macd: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._keltner: Dict[Tuple[int, int, float], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._keltner_pos: Dict[Tuple[int, int, float], np.ndarray] = {}

    def ema(self, period: int) -> np.ndarray:
        if period not in self._ema:
            self._ema[period] = ema(self.closes, period)
        return self._ema[period]

    def rsi(self, period: int = 14) -> np.ndarray:
        if period not in self._rsi:
            self._rsi[period] = rsi(self.closes, period)
        return self._rsi[period]

    def atr(self, period: int = 14) -> np.ndarray:
        if period not in self._atr:
            self._atr[period] = atr(self.highs, self.lows, self.closes, period)
        return self._atr[period]

    def adx(self, period: int = 14) -> np.ndarray:
        if period not in self._adx:
            self._adx[period] = adx(self.highs, self.lows, self.closes, period)
        return self._adx[period]

    def volume_ratio(self, period: int = 20) -> np.ndarray:
        if period not in self._vol_ratio:
            self._vol_ratio[period] = volume_ratio(self.volumes, period)
        return self._vol_ratio[period]

    def momentum(self, period: int = 10) -> np.ndarray:
        if period not in self._momentum:
            self._momentum[period] = momentum(self.closes, period)
        return self._momentum[period]

    def trend_strength(self, fast: int, slow: int) -> np.ndarray:
        key = (fast, slow)
        if key not in self._trend_strength:
            fast_ema = self.ema(fast)
            slow_ema = self.ema(slow)
            safe_slow = np.where(slow_ema > 0, slow_ema, 1.0)
            self._trend_strength[key] = (fast_ema - slow_ema) / safe_slow
        return self._trend_strength[key]

    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (period, std_dev)
        if key not in self._bb:
            self._bb[key] = bollinger_bands(self.closes, period, std_dev)
        return self._bb[key]

    def bb_position(self, period: int = 20, std_dev: float = 2.0) -> np.ndarray:
        key = (period, std_dev)
        if key not in self._bb_pos:
            upper, middle, lower = self.bollinger_bands(period, std_dev)
            band_width = upper - lower
            safe_width = np.where(band_width > 0, band_width, 1.0)
            position = (self.closes - lower) / safe_width
            position = np.where(band_width > 0, position, 0.5)
            self._bb_pos[key] = position
        return self._bb_pos[key]

    def macd(
        self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (fast_period, slow_period, signal_period)
        if key not in self._macd:
            self._macd[key] = macd(self.closes, fast_period, slow_period, signal_period)
        return self._macd[key]

    def keltner_channels(
        self, ema_period: int = 20, atr_period: int = 14, multiplier: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (ema_period, atr_period, multiplier)
        if key not in self._keltner:
            self._keltner[key] = keltner_channels(
                self.highs, self.lows, self.closes, ema_period, atr_period, multiplier
            )
        return self._keltner[key]

    def keltner_position(
        self, ema_period: int = 20, atr_period: int = 14, multiplier: float = 1.5
    ) -> np.ndarray:
        key = (ema_period, atr_period, multiplier)
        if key not in self._keltner_pos:
            self._keltner_pos[key] = keltner_position(
                self.closes, self.highs, self.lows, ema_period, atr_period, multiplier
            )
        return self._keltner_pos[key]
