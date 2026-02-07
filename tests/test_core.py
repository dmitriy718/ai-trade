"""
Core component tests for the AI Trading Bot.

Validates configuration loading, indicator calculations,
strategy signal generation, risk management, and database operations.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import BotConfig, ConfigManager
from src.core.database import DatabaseManager
from src.execution.risk_manager import RiskManager
from src.strategies.base import SignalDirection, StrategySignal
from src.strategies.breakout import BreakoutStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.reversal import ReversalStrategy
from src.strategies.trend import TrendStrategy
from src.utils.indicators import (
    adx,
    atr,
    bb_position,
    bollinger_bands,
    ema,
    momentum,
    order_book_imbalance,
    rsi,
    sma,
    trend_strength,
    volume_ratio,
)


# ---- Indicator Tests ----

class TestIndicators:
    """Test technical indicator calculations."""

    def test_ema_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = ema(data, 3)
        assert len(result) == len(data)
        assert np.isnan(result[0])
        assert not np.isnan(result[2])
        assert result[-1] > result[5]  # Uptrend

    def test_sma_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(data, 3)
        assert len(result) == len(data)
        assert np.isnan(result[0])
        assert abs(result[2] - 2.0) < 0.01  # (1+2+3)/3
        assert abs(result[4] - 4.0) < 0.01  # (3+4+5)/3

    def test_rsi_range(self):
        data = np.random.uniform(90, 110, 100)
        result = rsi(data, 14)
        assert len(result) == len(data)
        # RSI should be between 0 and 100
        valid = result[~np.isnan(result)]
        assert all(0 <= v <= 100 for v in valid)

    def test_rsi_extremes(self):
        # All up moves -> RSI should be high
        up_data = np.cumsum(np.ones(50)) + 100
        result = rsi(up_data, 14)
        assert result[-1] > 70

    def test_bollinger_bands(self):
        data = np.random.normal(100, 5, 50)
        upper, middle, lower = bollinger_bands(data, 20, 2.0)
        assert len(upper) == len(data)
        # Upper > middle > lower (for non-NaN values)
        valid_idx = ~np.isnan(upper)
        assert all(upper[valid_idx] >= middle[valid_idx])
        assert all(middle[valid_idx] >= lower[valid_idx])

    def test_bb_position_range(self):
        # M25: bb_position is now unclipped, but should be near [0,1] for normal data
        data = np.random.normal(100, 5, 50)
        result = bb_position(data, 20, 2.0)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        # Most values should be between -0.5 and 1.5 for normal distribution
        assert all(-1.0 <= v <= 2.0 for v in valid)

    def test_atr_positive(self):
        highs = np.random.uniform(101, 110, 50)
        lows = np.random.uniform(90, 99, 50)
        closes = (highs + lows) / 2
        result = atr(highs, lows, closes, 14)
        valid = result[result > 0]
        assert len(valid) > 0
        assert all(v >= 0 for v in valid)

    def test_adx_range(self):
        highs = np.cumsum(np.random.uniform(0, 2, 100)) + 100
        lows = highs - np.random.uniform(1, 3, 100)
        closes = (highs + lows) / 2
        result = adx(highs, lows, closes, 14)
        valid = result[~np.isnan(result)]
        assert all(v >= 0 for v in valid)

    def test_order_book_imbalance(self):
        assert order_book_imbalance(100, 50) > 0  # More bids = positive
        assert order_book_imbalance(50, 100) < 0  # More asks = negative
        assert order_book_imbalance(100, 100) == 0  # Equal = neutral
        assert order_book_imbalance(0, 0) == 0  # Empty = neutral

    def test_volume_ratio(self):
        volumes = np.array([100, 100, 100, 100, 100, 200], dtype=float)
        result = volume_ratio(volumes, 5)
        assert result[-1] > 1.0  # Last bar has above-average volume

    def test_momentum_calculation(self):
        data = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110], dtype=float)
        result = momentum(data, 5)
        assert result[-1] > 0  # Upward momentum

    def test_trend_strength(self):
        uptrend = np.cumsum(np.ones(50)) + 100
        result = trend_strength(uptrend, 5, 13)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert valid[-1] > 0  # Positive trend


# ---- Strategy Tests ----

class TestStrategies:
    """Test strategy signal generation."""

    @staticmethod
    def _generate_uptrend(n=200):
        noise = np.random.normal(0, 0.5, n)
        trend = np.cumsum(np.random.uniform(0.1, 0.5, n))
        closes = 100 + trend + noise
        highs = closes + np.random.uniform(0.2, 1.0, n)
        lows = closes - np.random.uniform(0.2, 1.0, n)
        volumes = np.random.uniform(80, 120, n)
        return closes, highs, lows, volumes

    @staticmethod
    def _generate_ranging(n=200):
        closes = 100 + np.sin(np.linspace(0, 10, n)) * 5 + np.random.normal(0, 0.5, n)
        highs = closes + np.random.uniform(0.2, 1.0, n)
        lows = closes - np.random.uniform(0.2, 1.0, n)
        volumes = np.random.uniform(80, 120, n)
        return closes, highs, lows, volumes

    @pytest.mark.asyncio
    async def test_trend_strategy_returns_signal(self):
        strategy = TrendStrategy()
        closes, highs, lows, volumes = self._generate_uptrend()
        signal = await strategy.analyze("BTC/USD", closes, highs, lows, volumes)
        assert isinstance(signal, StrategySignal)
        assert signal.strategy_name == "trend"
        assert signal.pair == "BTC/USD"

    @pytest.mark.asyncio
    async def test_mean_reversion_returns_signal(self):
        strategy = MeanReversionStrategy()
        closes, highs, lows, volumes = self._generate_ranging()
        signal = await strategy.analyze("ETH/USD", closes, highs, lows, volumes)
        assert isinstance(signal, StrategySignal)
        assert signal.strategy_name == "mean_reversion"

    @pytest.mark.asyncio
    async def test_momentum_returns_signal(self):
        strategy = MomentumStrategy()
        closes, highs, lows, volumes = self._generate_uptrend()
        signal = await strategy.analyze("BTC/USD", closes, highs, lows, volumes)
        assert isinstance(signal, StrategySignal)

    @pytest.mark.asyncio
    async def test_breakout_returns_signal(self):
        strategy = BreakoutStrategy()
        closes, highs, lows, volumes = self._generate_uptrend()
        signal = await strategy.analyze("BTC/USD", closes, highs, lows, volumes)
        assert isinstance(signal, StrategySignal)

    @pytest.mark.asyncio
    async def test_reversal_returns_signal(self):
        strategy = ReversalStrategy()
        closes, highs, lows, volumes = self._generate_ranging()
        signal = await strategy.analyze("BTC/USD", closes, highs, lows, volumes)
        assert isinstance(signal, StrategySignal)

    @pytest.mark.asyncio
    async def test_strategy_insufficient_data(self):
        strategy = TrendStrategy()
        closes = np.array([100.0, 101.0, 102.0])
        highs = closes + 1
        lows = closes - 1
        volumes = np.array([100.0, 100.0, 100.0])
        signal = await strategy.analyze("BTC/USD", closes, highs, lows, volumes)
        assert signal.direction == SignalDirection.NEUTRAL


# ---- Risk Manager Tests ----

class TestRiskManager:
    """Test risk management logic."""

    def test_position_sizing_basic(self):
        rm = RiskManager(initial_bankroll=10000)
        result = rm.calculate_position_size(
            pair="BTC/USD",
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000,
            win_rate=0.55,
            avg_win_loss_ratio=2.0,
            confidence=0.7,
        )
        assert result.allowed
        assert result.size_usd > 0
        assert result.size_usd <= 500  # Max position cap
        assert result.risk_reward_ratio >= 1.0

    def test_position_sizing_respects_daily_limit(self):
        rm = RiskManager(initial_bankroll=10000, max_daily_loss=0.05)
        # Simulate daily loss exceeding 5% of bankroll
        from datetime import datetime, timezone
        rm._daily_pnl = -600  # Over 5% of 10000
        rm._daily_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        result = rm.calculate_position_size(
            pair="BTC/USD",
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000,
        )
        assert not result.allowed
        assert "Daily loss limit" in result.reason

    def test_stop_loss_tracking(self):
        rm = RiskManager()
        state = rm.initialize_stop_loss("T-123", 50000, 49000, "buy")
        assert state.current_sl == 49000
        assert not state.breakeven_activated

    def test_trailing_stop_activation(self):
        rm = RiskManager(
            breakeven_activation_pct=0.01,
            trailing_activation_pct=0.015,
            trailing_step_pct=0.005,
        )
        rm.initialize_stop_loss("T-123", 50000, 49000, "buy")

        # Price moves up 2% -> should activate trailing
        state = rm.update_stop_loss("T-123", 51000, 50000, "buy")
        assert state.breakeven_activated  # 2% > 1% breakeven threshold
        assert state.current_sl >= 50000  # At least breakeven

    def test_risk_of_ruin_zero_for_no_history(self):
        rm = RiskManager()
        assert rm.calculate_risk_of_ruin() == 0.0

    def test_drawdown_factor_scaling(self):
        rm = RiskManager(initial_bankroll=10000)
        rm._peak_bankroll = 10000

        # No drawdown
        rm.current_bankroll = 10000
        assert rm._get_drawdown_factor() == 1.0

        # 3% drawdown -> 0.75
        rm.current_bankroll = 9700
        assert rm._get_drawdown_factor() == 0.75

        # 7% drawdown -> 0.5
        rm.current_bankroll = 9300
        assert rm._get_drawdown_factor() == 0.5


# ---- Database Tests ----

class TestDatabase:
    """Test database operations."""

    @pytest.mark.asyncio
    async def test_database_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = DatabaseManager(db_path)
            await db.initialize()

            # Insert a trade
            trade_id = await db.insert_trade({
                "trade_id": "T-test-001",
                "pair": "BTC/USD",
                "side": "buy",
                "entry_price": 50000,
                "quantity": 0.001,
                "strategy": "trend",
            })
            assert trade_id > 0

            # Get open trades
            open_trades = await db.get_open_trades()
            assert len(open_trades) == 1
            assert open_trades[0]["trade_id"] == "T-test-001"

            # Close the trade
            await db.close_trade("T-test-001", 51000, 1.0, 0.02)

            # Verify closed
            open_trades = await db.get_open_trades()
            assert len(open_trades) == 0

            history = await db.get_trade_history()
            assert len(history) == 1
            assert history[0]["pnl"] == 1.0

            await db.close()

    @pytest.mark.asyncio
    async def test_thought_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = DatabaseManager(db_path)
            await db.initialize()

            await db.log_thought("test", "Test thought", "info")
            thoughts = await db.get_thoughts(limit=10)
            assert len(thoughts) == 1
            assert thoughts[0]["message"] == "Test thought"

            await db.close()

    @pytest.mark.asyncio
    async def test_performance_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = DatabaseManager(db_path)
            await db.initialize()

            stats = await db.get_performance_stats()
            assert stats["total_trades"] == 0
            assert stats["total_pnl"] == 0.0

            await db.close()


# ---- Config Tests ----

class TestConfig:
    """Test configuration loading."""

    def test_default_config(self):
        config = BotConfig()
        assert config.app.mode == "paper"
        assert config.risk.max_risk_per_trade == 0.02
        assert config.ai.confluence_threshold == 3
        assert len(config.trading.pairs) > 0

    def test_risk_validation(self):
        """Test that invalid risk values are rejected."""
        with pytest.raises(Exception):
            from src.core.config import RiskConfig
            RiskConfig(max_risk_per_trade=0.50)  # Over 10% should fail
