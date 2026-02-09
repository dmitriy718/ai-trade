"""
VWAP Momentum Alpha Strategy - Volume-Weighted pullback entries.

Uses rolling VWAP with volume-weighted standard deviation bands to
identify pullbacks in the direction of the VWAP slope. This is a
momentum-style entry that buys/sells dips back toward VWAP in trend.
"""

from __future__ import annotations

import numpy as np

from src.strategies.base import BaseStrategy, SignalDirection, StrategySignal
from src.utils.indicators import atr, compute_sl_tp, momentum, volume_ratio


def _rolling_vwap_and_std(
    closes: np.ndarray,
    volumes: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling VWAP and volume-weighted std dev.

    Returns arrays aligned to closes length with NaN for warmup bars.
    """
    n = len(closes)
    vwap = np.full(n, np.nan, dtype=float)
    vwap_std = np.full(n, np.nan, dtype=float)
    if n == 0 or window <= 1:
        return vwap, vwap_std

    vol = volumes.astype(float)
    price = closes.astype(float)
    pv = price * vol
    p2v = (price ** 2) * vol

    kernel = np.ones(window)
    sum_vol = np.convolve(vol, kernel, mode="valid")
    sum_pv = np.convolve(pv, kernel, mode="valid")
    sum_p2v = np.convolve(p2v, kernel, mode="valid")

    with np.errstate(divide="ignore", invalid="ignore"):
        vwap_valid = np.where(sum_vol > 0, sum_pv / sum_vol, np.nan)
        var_valid = np.where(sum_vol > 0, (sum_p2v / sum_vol) - (vwap_valid ** 2), np.nan)
    vwap_std_valid = np.sqrt(np.maximum(var_valid, 0.0))

    vwap[window - 1:] = vwap_valid
    vwap_std[window - 1:] = vwap_std_valid
    return vwap, vwap_std


class VWAPMomentumAlphaStrategy(BaseStrategy):
    """
    VWAP momentum strategy focused on pullbacks.

    Entry (Long):
    1. VWAP slope positive (trend up)
    2. Price pulls back to/below VWAP (negative z-score)
    3. Short-term momentum improving

    Entry (Short):
    1. VWAP slope negative (trend down)
    2. Price pulls back to/above VWAP (positive z-score)
    3. Short-term momentum weakening
    """

    def __init__(
        self,
        vwap_window: int = 20,
        band_std: float = 1.5,
        pullback_z: float = 0.6,
        slope_period: int = 5,
        volume_multiplier: float = 1.0,
        pullback_z_trend_adjust: float = -0.12,
        pullback_z_range_adjust: float = 0.12,
        pullback_z_high_vol_adjust: float = 0.10,
        pullback_z_low_vol_adjust: float = -0.05,
        slope_min_pct: float = 0.0005,
        weight: float = 0.12,
        enabled: bool = True,
    ):
        super().__init__(name="vwap_momentum_alpha", weight=weight, enabled=enabled)
        self.vwap_window = vwap_window
        self.band_std = band_std
        self.pullback_z = pullback_z
        self.slope_period = slope_period
        self.volume_multiplier = volume_multiplier
        self.pullback_z_trend_adjust = pullback_z_trend_adjust
        self.pullback_z_range_adjust = pullback_z_range_adjust
        self.pullback_z_high_vol_adjust = pullback_z_high_vol_adjust
        self.pullback_z_low_vol_adjust = pullback_z_low_vol_adjust
        self.slope_min_pct = slope_min_pct

    def min_bars_required(self) -> int:
        return max(self.vwap_window + self.slope_period + 5, 50)

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
            atr_vals = cache.atr(14)
            vol_ratio = cache.volume_ratio(20)
            mom_vals = cache.momentum(5)
        else:
            atr_vals = atr(highs, lows, closes, 14)
            vol_ratio = volume_ratio(volumes, 20)
            mom_vals = momentum(closes, 5)

        vwap, vwap_std = _rolling_vwap_and_std(closes, volumes, self.vwap_window)
        if np.isnan(vwap[-1]) or np.isnan(vwap_std[-1]) or vwap_std[-1] <= 0:
            return self._neutral_signal(pair, "VWAP unavailable")

        fee_pct = kwargs.get("round_trip_fee_pct")
        curr_price = float(closes[-1])
        curr_vwap = float(vwap[-1])
        curr_std = float(vwap_std[-1])
        curr_atr = float(atr_vals[-1]) if len(atr_vals) else 0.0
        curr_vol_ratio = float(vol_ratio[-1]) if len(vol_ratio) else 0.0
        curr_mom = float(mom_vals[-1]) if len(mom_vals) else 0.0
        prev_mom = float(mom_vals[-2]) if len(mom_vals) > 1 else curr_mom

        zscore = (curr_price - curr_vwap) / curr_std if curr_std > 0 else 0.0

        slope_idx = max(1, self.slope_period)
        if len(vwap) > slope_idx and not np.isnan(vwap[-1 - slope_idx]):
            vwap_slope = curr_vwap - float(vwap[-1 - slope_idx])
        else:
            vwap_slope = curr_vwap - float(vwap[-2])

        slope_pct = vwap_slope / curr_vwap if curr_vwap > 0 else 0.0
        min_slope = float(self.slope_min_pct)

        trend_up = vwap_slope > 0 and slope_pct >= min_slope
        trend_down = vwap_slope < 0 and abs(slope_pct) >= min_slope
        mom_improving = curr_mom > prev_mom
        mom_weakening = curr_mom < prev_mom

        pullback_z = float(self.pullback_z)
        trend_regime = kwargs.get("trend_regime")
        vol_regime = kwargs.get("vol_regime")
        if trend_regime == "trend":
            pullback_z += self.pullback_z_trend_adjust
        elif trend_regime == "range":
            pullback_z += self.pullback_z_range_adjust
        if vol_regime == "high_vol":
            pullback_z += self.pullback_z_high_vol_adjust
        elif vol_regime == "low_vol":
            pullback_z += self.pullback_z_low_vol_adjust
        pullback_z = max(0.2, pullback_z)

        pullback_long = trend_up and zscore <= -pullback_z
        pullback_short = trend_down and zscore >= pullback_z

        direction = SignalDirection.NEUTRAL
        strength = 0.0
        confidence = 0.0

        if pullback_long:
            direction = SignalDirection.LONG
            strength = 0.35 + min(abs(zscore) / max(self.band_std, 1e-6), 0.3)
            if curr_vol_ratio >= self.volume_multiplier:
                strength += 0.1
            if mom_improving:
                strength += 0.1

            confidence = 0.35
            if curr_vol_ratio >= self.volume_multiplier:
                confidence += 0.1
            if mom_improving:
                confidence += 0.1
            if vwap_slope / curr_vwap > 0.001:
                confidence += 0.1
            if zscore < -1.0:
                confidence += 0.1

        elif pullback_short:
            direction = SignalDirection.SHORT
            strength = 0.35 + min(abs(zscore) / max(self.band_std, 1e-6), 0.3)
            if curr_vol_ratio >= self.volume_multiplier:
                strength += 0.1
            if mom_weakening:
                strength += 0.1

            confidence = 0.35
            if curr_vol_ratio >= self.volume_multiplier:
                confidence += 0.1
            if mom_weakening:
                confidence += 0.1
            if abs(vwap_slope / curr_vwap) > 0.001:
                confidence += 0.1
            if zscore > 1.0:
                confidence += 0.1

        if direction == SignalDirection.LONG:
            stop_loss, take_profit = compute_sl_tp(
                curr_price, curr_atr, "long", 2.0, 3.0, round_trip_fee_pct=fee_pct
            )
        elif direction == SignalDirection.SHORT:
            stop_loss, take_profit = compute_sl_tp(
                curr_price, curr_atr, "short", 2.0, 3.0, round_trip_fee_pct=fee_pct
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
                "vwap": round(curr_vwap, 6),
                "vwap_std": round(curr_std, 6),
                "zscore": round(zscore, 4),
                "vwap_slope": round(vwap_slope, 6),
                "slope_pct": round(slope_pct, 6),
                "volume_ratio": round(curr_vol_ratio, 2),
                "momentum": round(curr_mom, 6),
                "pullback_z": round(pullback_z, 4),
                "trend_regime": trend_regime,
                "vol_regime": vol_regime,
            },
        )

        self._last_signal = signal
        return signal
