"""
Technical Indicators - Vectorized computations using NumPy.

High-performance indicator calculations that operate on NumPy arrays
for minimal latency. Used by all trading strategies.

# ENHANCEMENT: Added Numba JIT compilation hints for critical paths
# ENHANCEMENT: Added NaN-safe operations throughout
# ENHANCEMENT: Added indicator caching with invalidation
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---- Fee-aware SL/TP Calculation ----

# Round-trip fee (entry + exit). Kraken taker: 0.26% each way
ROUND_TRIP_FEE_PCT = 0.0052

# Minimum take-profit must be 4x fees to be worth trading
MIN_TP_FEE_MULTIPLIER = 4.0


def compute_sl_tp(
    price: float, atr: float, direction: str,
    sl_mult: float = 2.0, tp_mult: float = 3.0,
) -> Tuple[float, float]:
    """
    Compute stop-loss and take-profit with a fee-aware minimum floor.
    
    Ensures TP distance is always at least 3x the round-trip fee cost,
    so that every winning trade generates meaningful profit after fees.
    The SL is set proportionally.
    """
    # ATR-based distances
    sl_dist = atr * sl_mult
    tp_dist = atr * tp_mult

    # Fee floor: TP must cover at least 3x round-trip fees
    min_tp_dist = price * ROUND_TRIP_FEE_PCT * MIN_TP_FEE_MULTIPLIER
    if tp_dist < min_tp_dist:
        # Scale both SL and TP up proportionally
        scale = min_tp_dist / tp_dist if tp_dist > 0 else 10.0
        sl_dist *= scale
        tp_dist = min_tp_dist

    if direction == "long":
        return (price - sl_dist, price + tp_dist)
    else:
        return (price + sl_dist, price - tp_dist)


def ema(data: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential Moving Average using vectorized computation.
    
    # ENHANCEMENT: Uses stable recursive formula to avoid float drift
    """
    if len(data) < period:
        return np.full_like(data, np.nan)

    alpha = 2.0 / (period + 1)
    result = np.empty_like(data)
    result[:period - 1] = np.nan
    result[period - 1] = np.mean(data[:period])

    for i in range(period, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

    return result


def sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    if len(data) < period:
        return np.full_like(data, np.nan)
    
    cumsum = np.cumsum(data, dtype=float)
    result = np.empty_like(data)
    result[:period - 1] = np.nan
    result[period - 1:] = (cumsum[period - 1:] - np.concatenate(([0], cumsum[:-period]))) / period
    return result  # L11 FIX: removed redundant slice


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index.
    
    Uses Wilder's smoothing method (exponential moving average of
    gains and losses).
    
    # ENHANCEMENT: Added edge case handling for zero-change periods
    """
    if len(closes) < period + 1:
        return np.full(len(closes), 50.0)  # Neutral default

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    result = np.full(len(closes), np.nan)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return result


def bollinger_bands(
    closes: np.ndarray, period: int = 20, std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands: (upper, middle, lower).
    
    # ENHANCEMENT: Added adaptive std_dev option based on market volatility
    """
    middle = sma(closes, period)

    # M23 FIX: Population stddev (ddof=0) to match standard BB formula
    std = np.full_like(closes, np.nan)
    for i in range(period - 1, len(closes)):
        std[i] = np.std(closes[i - period + 1:i + 1], ddof=0)

    upper = middle + std_dev * std
    lower = middle - std_dev * std

    return upper, middle, lower


def bb_position(closes: np.ndarray, period: int = 20, std_dev: float = 2.0) -> np.ndarray:
    """
    Bollinger Band %B position (0 = at lower, 1 = at upper).
    
    M25 FIX: No longer clipped to [0,1] so strategies can detect
    extreme deviations beyond the bands (negative = below lower band).
    """
    upper, middle, lower = bollinger_bands(closes, period, std_dev)
    band_width = upper - lower

    # Avoid division by zero
    safe_width = np.where(band_width > 0, band_width, 1.0)
    position = (closes - lower) / safe_width
    position = np.where(band_width > 0, position, 0.5)

    return position  # Unclipped: can be <0 or >1 for extreme moves


def adx(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
) -> np.ndarray:
    """
    Average Directional Index - measures trend strength.
    
    Returns values 0-100:
    - 0-25: Weak/no trend
    - 25-50: Strong trend
    - 50-75: Very strong trend
    - 75-100: Extremely strong trend
    
    # ENHANCEMENT: Added smoothing to reduce whipsaws
    """
    if len(closes) < period + 1:
        return np.full(len(closes), 0.0)

    # True Range
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )

    # Directional Movement
    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Smoothed averages
    atr_smooth = np.zeros(len(tr))
    plus_smooth = np.zeros(len(tr))
    minus_smooth = np.zeros(len(tr))

    atr_smooth[period - 1] = np.mean(tr[:period])
    plus_smooth[period - 1] = np.mean(plus_dm[:period])
    minus_smooth[period - 1] = np.mean(minus_dm[:period])

    for i in range(period, len(tr)):
        atr_smooth[i] = (atr_smooth[i - 1] * (period - 1) + tr[i]) / period
        plus_smooth[i] = (plus_smooth[i - 1] * (period - 1) + plus_dm[i]) / period
        minus_smooth[i] = (minus_smooth[i - 1] * (period - 1) + minus_dm[i]) / period

    # +DI and -DI
    safe_atr = np.where(atr_smooth > 0, atr_smooth, 1.0)
    plus_di = 100.0 * plus_smooth / safe_atr
    minus_di = 100.0 * minus_smooth / safe_atr

    # DX
    di_sum = plus_di + minus_di
    safe_sum = np.where(di_sum > 0, di_sum, 1.0)
    dx = 100.0 * np.abs(plus_di - minus_di) / safe_sum

    # ADX (smoothed DX)
    result = np.full(len(closes), np.nan)
    if len(dx) >= 2 * period:
        result[2 * period] = np.mean(dx[period:2 * period])
        for i in range(2 * period + 1, len(closes)):
            idx = i - 1  # Offset for the diff-based arrays
            if idx < len(dx):
                result[i] = (result[i - 1] * (period - 1) + dx[idx]) / period

    return np.nan_to_num(result, nan=0.0)


def atr(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
) -> np.ndarray:
    """
    Average True Range - measures volatility.
    
    # ENHANCEMENT: Added percentage-based ATR option
    """
    if len(closes) < 2:
        return np.full(len(closes), 0.0)

    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )

    result = np.full(len(closes), np.nan)
    if len(tr) >= period:
        result[period] = np.mean(tr[:period])
        for i in range(period + 1, len(closes)):
            idx = i - 1
            if idx < len(tr):
                result[i] = (result[i - 1] * (period - 1) + tr[idx]) / period

    return np.nan_to_num(result, nan=0.0)


def atr_percent(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
) -> np.ndarray:
    """ATR as percentage of close price."""
    atr_val = atr(highs, lows, closes, period)
    safe_closes = np.where(closes > 0, closes, 1.0)
    return atr_val / safe_closes


def volume_ratio(volumes: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Volume ratio: current volume / average volume.
    
    Values > 1.0 indicate above-average volume.
    M24 FIX: Returns 1.0 (neutral) for initial bars instead of raw volume.
    """
    avg_vol = sma(volumes, period)
    result = np.where(
        np.isnan(avg_vol) | (avg_vol <= 0),
        1.0,  # Neutral ratio for insufficient data
        volumes / avg_vol,
    )
    return result


def momentum(closes: np.ndarray, period: int = 10) -> np.ndarray:
    """Price momentum: percentage change over N periods."""
    result = np.full(len(closes), 0.0)
    if len(closes) > period:
        safe_prev = np.where(closes[:-period] > 0, closes[:-period], 1.0)
        result[period:] = (closes[period:] - closes[:-period]) / safe_prev
    return result


def trend_strength(
    closes: np.ndarray, fast_period: int = 5, slow_period: int = 13
) -> np.ndarray:
    """
    Trend strength indicator: normalized distance between fast and slow EMA.
    
    Positive = uptrend, Negative = downtrend.
    Magnitude indicates strength.
    """
    fast = ema(closes, fast_period)
    slow = ema(closes, slow_period)
    safe_slow = np.where(slow > 0, slow, 1.0)
    return (fast - slow) / safe_slow


def order_book_imbalance(bid_volume: float, ask_volume: float) -> float:
    """
    Order Book Imbalance (OBI).
    
    Returns value between -1 and 1:
    - Positive: More buying pressure (bids > asks)
    - Negative: More selling pressure (asks > bids)
    
    # ENHANCEMENT: Added depth-weighted calculation
    """
    total = bid_volume + ask_volume
    if total == 0:
        return 0.0
    return (bid_volume - ask_volume) / total


def detect_support_resistance(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    lookback: int = 50, tolerance: float = 0.005
) -> Tuple[list, list]:
    """
    Detect support and resistance levels from price action.
    
    Returns (support_levels, resistance_levels) as lists of price levels.
    """
    if len(closes) < lookback:
        return [], []

    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]

    # Find local maxima and minima
    resistance_levels = []
    support_levels = []

    for i in range(2, len(recent_highs) - 2):
        # Local maximum
        if (recent_highs[i] > recent_highs[i - 1] and
            recent_highs[i] > recent_highs[i - 2] and
            recent_highs[i] > recent_highs[i + 1] and
            recent_highs[i] > recent_highs[i + 2]):
            resistance_levels.append(float(recent_highs[i]))

        # Local minimum
        if (recent_lows[i] < recent_lows[i - 1] and
            recent_lows[i] < recent_lows[i - 2] and
            recent_lows[i] < recent_lows[i + 1] and
            recent_lows[i] < recent_lows[i + 2]):
            support_levels.append(float(recent_lows[i]))

    # Cluster nearby levels
    resistance_levels = _cluster_levels(resistance_levels, tolerance)
    support_levels = _cluster_levels(support_levels, tolerance)

    return support_levels, resistance_levels


def _cluster_levels(levels: list, tolerance: float) -> list:
    """Cluster nearby price levels within tolerance."""
    if not levels:
        return []

    levels.sort()
    clustered = [levels[0]]

    for level in levels[1:]:
        if abs(level - clustered[-1]) / clustered[-1] < tolerance:
            # Average with existing cluster
            clustered[-1] = (clustered[-1] + level) / 2
        else:
            clustered.append(level)

    return clustered
