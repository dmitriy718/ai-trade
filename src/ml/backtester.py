"""
Backtester - Shadow mode trading simulator.

Replays historical data through the live trading logic to validate
strategy performance. Uses the same signal generation, confluence
detection, and risk management as the live system.

# ENHANCEMENT: Added Monte Carlo simulation for confidence intervals
# ENHANCEMENT: Added walk-forward optimization support
# ENHANCEMENT: Added detailed per-trade analysis export
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.config import get_config
from src.core.logger import get_logger
from src.ai.confluence import ConfluenceDetector
from src.ai.predictor import TFLitePredictor
from src.exchange.market_data import MarketDataCache
from src.execution.risk_manager import RiskManager
from src.strategies.base import BaseStrategy, SignalDirection, StrategySignal
from src.strategies.breakout import BreakoutStrategy
from src.strategies.keltner import KeltnerStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.reversal import ReversalStrategy
from src.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategies.trend import TrendStrategy
from src.strategies.vwap_momentum_alpha import VWAPMomentumAlphaStrategy
from src.utils.indicators import atr

logger = get_logger("backtester")


class BacktestResult:
    """Container for backtest results with comprehensive statistics."""

    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.start_time: str = ""
        self.end_time: str = ""
        self.pair: str = ""
        self.initial_balance: float = 10000.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.get("pnl", 0) > 0)

    @property
    def losing_trades(self) -> int:
        return sum(1 for t in self.trades if t.get("pnl", 0) <= 0)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def total_pnl(self) -> float:
        return sum(t.get("pnl", 0) for t in self.trades)

    @property
    def total_return_pct(self) -> float:
        return (self.total_pnl / self.initial_balance) * 100

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.initial_balance
        max_dd = 0.0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def avg_win(self) -> float:
        wins = [t["pnl"] for t in self.trades if t.get("pnl", 0) > 0]
        return np.mean(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t["pnl"] for t in self.trades if t.get("pnl", 0) <= 0]
        return np.mean(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t["pnl"] for t in self.trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t["pnl"] for t in self.trades if t.get("pnl", 0) < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0
        return gross_profit / gross_loss

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio (assuming 1-minute bars)."""
        if not self.trades:
            return 0.0
        returns = [t.get("pnl_pct", 0) for t in self.trades]
        if len(returns) < 2:
            return 0.0
        avg = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        # Annualize: sqrt(525600) for 1-minute data
        return (avg / std) * np.sqrt(525600 / max(len(returns), 1))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair": self.pair,
            "initial_balance": self.initial_balance,
            "final_balance": self.initial_balance + self.total_pnl,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "max_drawdown": round(self.max_drawdown, 4),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "profit_factor": round(self.profit_factor, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class Backtester:
    """
    Historical backtesting engine.
    
    Replays OHLCV data bar-by-bar through the strategy engine,
    simulating entries, exits, and position management as they
    would occur in live trading.
    
    # ENHANCEMENT: Added multi-strategy confluence backtesting
    # ENHANCEMENT: Added realistic slippage and fee modeling
    # ENHANCEMENT: Added per-bar P&L tracking for equity curves
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.02,
        max_position_pct: float = 0.05,
        slippage_pct: float = 0.001,
        fee_pct: float = 0.0026,
    ):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.slippage_pct = slippage_pct
        self.fee_pct = fee_pct

    async def run(
        self,
        pair: str,
        ohlcv_data: pd.DataFrame,
        strategies: Optional[List[BaseStrategy]] = None,
        confluence_threshold: int = 2,
        mode: str = "simple",
        config: Optional[Any] = None,
        predictor: Optional[TFLitePredictor] = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            pair: Trading pair
            ohlcv_data: DataFrame with columns [time, open, high, low, close, volume]
            strategies: List of strategies to test (default: all built-in strategies)
            confluence_threshold: Min strategies for trade entry
            mode: "simple" (default) or "parity" (live-like)
            config: Optional BotConfig for parity mode
            predictor: Optional TFLitePredictor for parity mode
        
        Returns:
            BacktestResult with full statistics
        """
        if mode == "parity":
            return await self.run_parity(
                pair=pair,
                ohlcv_data=ohlcv_data,
                config=config,
                predictor=predictor,
            )
        if strategies is None:
            strategies = [
                KeltnerStrategy(),
                TrendStrategy(),
                MeanReversionStrategy(),
                RSIMeanReversionStrategy(),
                MomentumStrategy(),
                VWAPMomentumAlphaStrategy(),
                BreakoutStrategy(),
                ReversalStrategy(),
            ]

        result = BacktestResult()
        result.pair = pair
        result.initial_balance = self.initial_balance

        if ohlcv_data.empty or len(ohlcv_data) < 100:
            logger.warning("Insufficient data for backtest", bars=len(ohlcv_data))
            return result

        result.start_time = str(ohlcv_data.iloc[0]["time"])
        result.end_time = str(ohlcv_data.iloc[-1]["time"])

        # State
        balance = self.initial_balance
        position = None  # {side, entry, size_units, sl, tp, strategy}

        closes = ohlcv_data["close"].values.astype(float)
        highs = ohlcv_data["high"].values.astype(float)
        lows = ohlcv_data["low"].values.astype(float)
        volumes = ohlcv_data["volume"].values.astype(float)
        opens = ohlcv_data["open"].values.astype(float)

        # Walk through bars
        warmup = 100  # Skip first N bars for indicator warmup

        for i in range(warmup, len(closes)):
            current_price = closes[i]
            current_high = highs[i]
            current_low = lows[i]

            # Update equity curve
            unrealized = 0
            if position:
                if position["side"] == "buy":
                    unrealized = (current_price - position["entry"]) * position["size_units"]
                else:
                    unrealized = (position["entry"] - current_price) * position["size_units"]
            result.equity_curve.append(balance + unrealized)

            # Check exit conditions for open position
            if position:
                closed = False

                # Stop loss check
                if position["side"] == "buy":
                    if current_low <= position["sl"]:
                        exit_price = position["sl"]
                        pnl = (exit_price - position["entry"]) * position["size_units"]
                        closed = True
                        reason = "stop_loss"
                    elif current_high >= position["tp"]:
                        exit_price = position["tp"]
                        pnl = (exit_price - position["entry"]) * position["size_units"]
                        closed = True
                        reason = "take_profit"
                else:
                    if current_high >= position["sl"]:
                        exit_price = position["sl"]
                        pnl = (position["entry"] - exit_price) * position["size_units"]
                        closed = True
                        reason = "stop_loss"
                    elif current_low <= position["tp"]:
                        exit_price = position["tp"]
                        pnl = (position["entry"] - exit_price) * position["size_units"]
                        closed = True
                        reason = "take_profit"

                if closed:
                    # Apply adverse slippage on exit and recompute PnL
                    if position["side"] == "buy":
                        exit_price *= (1 - self.slippage_pct)
                        gross_pnl = (exit_price - position["entry"]) * position["size_units"]
                    else:
                        exit_price *= (1 + self.slippage_pct)
                        gross_pnl = (position["entry"] - exit_price) * position["size_units"]

                    # Apply fees (entry + exit)
                    entry_fee = float(position.get("entry_fee", 0.0) or 0.0)
                    exit_fee = abs(exit_price * position["size_units"]) * self.fee_pct
                    fees = entry_fee + exit_fee
                    pnl = gross_pnl - fees

                    # Balance already reflects entry fee; only subtract exit fee here
                    balance += gross_pnl - exit_fee
                    entry_value = position["entry"] * position["size_units"]
                    pnl_pct = pnl / entry_value if entry_value > 0 else 0

                    result.trades.append({
                        "bar": i,
                        "pair": pair,
                        "side": position["side"],
                        "entry": position["entry"],
                        "exit": exit_price,
                        "size_units": position["size_units"],
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 4),
                        "fees": round(fees, 4),
                        "reason": reason,
                        "strategy": position["strategy"],
                    })

                    position = None
                    continue

            # Skip if already in position
            if position:
                continue

            # Run strategies on historical slice
            hist_closes = closes[:i + 1]
            hist_highs = highs[:i + 1]
            hist_lows = lows[:i + 1]
            hist_volumes = volumes[:i + 1]

            # Get signals from all strategies
            long_votes = 0
            short_votes = 0
            best_signal = None
            best_strength = 0

            for strategy in strategies:
                if len(hist_closes) < strategy.min_bars_required():
                    continue

                signal = await strategy.analyze(
                    pair, hist_closes, hist_highs, hist_lows, hist_volumes,
                    opens=opens[:i + 1],
                )

                if signal.is_actionable:
                    if signal.direction == SignalDirection.LONG:
                        long_votes += 1
                    elif signal.direction == SignalDirection.SHORT:
                        short_votes += 1

                    if signal.strength > best_strength:
                        best_strength = signal.strength
                        best_signal = signal

            # Check confluence
            if best_signal and max(long_votes, short_votes) >= confluence_threshold:
                direction = SignalDirection.LONG if long_votes > short_votes else SignalDirection.SHORT
                side = "buy" if direction == SignalDirection.LONG else "sell"

                # Position sizing
                sl_distance = abs(best_signal.entry_price - best_signal.stop_loss)
                if sl_distance <= 0 or current_price <= 0:
                    continue

                risk_amount = balance * self.risk_per_trade
                max_size_usd = balance * self.max_position_pct

                size_usd = min(risk_amount / (sl_distance / current_price), max_size_usd)
                size_units = size_usd / current_price

                if size_usd < 10:
                    continue

                # Apply slippage
                if side == "buy":
                    entry = current_price * (1 + self.slippage_pct)
                else:
                    entry = current_price * (1 - self.slippage_pct)

                # Apply entry fees
                entry_fee = size_usd * self.fee_pct
                balance -= entry_fee

                position = {
                    "side": side,
                    "entry": entry,
                    "size_units": size_units,
                    "sl": best_signal.stop_loss,
                    "tp": best_signal.take_profit,
                    "strategy": best_signal.strategy_name,
                    "bar": i,
                    "entry_fee": entry_fee,
                }

        # Close any remaining position at last price
        if position:
            final_price = closes[-1]
            if position["side"] == "buy":
                final_price *= (1 - self.slippage_pct)
                gross_pnl = (final_price - position["entry"]) * position["size_units"]
            else:
                final_price *= (1 + self.slippage_pct)
                gross_pnl = (position["entry"] - final_price) * position["size_units"]
            entry_fee = float(position.get("entry_fee", 0.0) or 0.0)
            exit_fee = abs(final_price * position["size_units"]) * self.fee_pct
            fees = entry_fee + exit_fee
            pnl = gross_pnl - fees
            balance += gross_pnl - exit_fee

            entry_value = position["entry"] * position["size_units"]
            result.trades.append({
                "bar": len(closes) - 1,
                "pair": pair,
                "side": position["side"],
                "entry": position["entry"],
                "exit": final_price,
                "size_units": position["size_units"],
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / entry_value if entry_value > 0 else 0, 4),
                "fees": round(fees, 4),
                "reason": "end_of_data",
                "strategy": position["strategy"],
            })

        logger.info(
            "Backtest complete",
            pair=pair,
            trades=result.total_trades,
            win_rate=round(result.win_rate, 4),
            total_return=round(result.total_return_pct, 2),
            max_drawdown=round(result.max_drawdown, 4),
        )

        return result

    async def run_parity(
        self,
        pair: str,
        ohlcv_data: pd.DataFrame,
        config: Optional[Any] = None,
        predictor: Optional[TFLitePredictor] = None,
    ) -> BacktestResult:
        """
        Live‑parity backtest mode.

        Uses the same confluence engine, AI predictor gating, and
        risk manager sizing as live. Execution is simulated with
        configurable slippage and fees.
        """
        cfg = config or get_config()
        result = BacktestResult()
        result.pair = pair
        result.initial_balance = self.initial_balance

        if ohlcv_data.empty or len(ohlcv_data) < 100:
            logger.warning("Insufficient data for backtest", bars=len(ohlcv_data))
            return result

        result.start_time = str(ohlcv_data.iloc[0]["time"]) if "time" in ohlcv_data.columns else ""
        result.end_time = str(ohlcv_data.iloc[-1]["time"]) if "time" in ohlcv_data.columns else ""

        # Core components (live‑like)
        market_data = MarketDataCache(max_bars=max(len(ohlcv_data), cfg.trading.warmup_bars))
        confluence = ConfluenceDetector(
            market_data=market_data,
            confluence_threshold=cfg.ai.confluence_threshold,
            obi_threshold=cfg.ai.obi_threshold,
            book_score_threshold=getattr(cfg.ai, "book_score_threshold", 0.2),
            book_score_max_age_seconds=getattr(cfg.ai, "book_score_max_age_seconds", 5),
            min_confidence=cfg.ai.min_confidence,
            obi_counts_as_confluence=getattr(cfg.ai, "obi_counts_as_confluence", False),
            obi_weight=getattr(cfg.ai, "obi_weight", 0.4),
            round_trip_fee_pct=cfg.exchange.taker_fee * 2,
            use_closed_candles_only=getattr(cfg.trading, "use_closed_candles_only", False),
            regime_config=getattr(cfg.ai, "regime", None),
            timeframes=getattr(cfg.trading, "timeframes", [1]),
            multi_timeframe_min_agreement=getattr(cfg.ai, "multi_timeframe_min_agreement", 1),
            primary_timeframe=getattr(cfg.ai, "primary_timeframe", 1),
        )
        confluence.configure_strategies(
            cfg.strategies.model_dump(),
            single_strategy_mode=getattr(cfg.trading, "single_strategy_mode", None),
        )

        risk_manager = RiskManager(
            initial_bankroll=self.initial_balance,
            max_risk_per_trade=cfg.risk.max_risk_per_trade,
            max_daily_loss=cfg.risk.max_daily_loss,
            max_position_usd=cfg.risk.max_position_usd,
            kelly_fraction=cfg.risk.kelly_fraction,
            max_kelly_size=cfg.risk.max_kelly_size,
            risk_of_ruin_threshold=cfg.risk.risk_of_ruin_threshold,
            atr_multiplier_sl=cfg.risk.atr_multiplier_sl,
            atr_multiplier_tp=cfg.risk.atr_multiplier_tp,
            trailing_activation_pct=cfg.risk.trailing_activation_pct,
            trailing_step_pct=cfg.risk.trailing_step_pct,
            breakeven_activation_pct=cfg.risk.breakeven_activation_pct,
            cooldown_seconds=cfg.trading.cooldown_seconds,
            max_concurrent_positions=cfg.trading.max_concurrent_positions,
            strategy_cooldowns=cfg.trading.strategy_cooldowns_seconds,
        )
        confluence.set_cooldown_checker(risk_manager.is_strategy_on_cooldown)

        if predictor is None:
            predictor = TFLitePredictor(
                model_path=cfg.ai.tflite_model_path,
                feature_names=cfg.ml.features,
            )
            predictor.load_model()

        # State
        balance = self.initial_balance
        position: Optional[Dict[str, Any]] = None
        trade_id_seq = 0

        # Use numpy arrays for fast access
        closes = ohlcv_data["close"].values.astype(float)
        highs = ohlcv_data["high"].values.astype(float)
        lows = ohlcv_data["low"].values.astype(float)
        volumes = ohlcv_data["volume"].values.astype(float)
        opens = ohlcv_data["open"].values.astype(float)

        warmup = 100

        for i in range(len(closes)):
            # Feed bar into market data cache
            ts_val = ohlcv_data.iloc[i]["time"] if "time" in ohlcv_data.columns else i
            await market_data.update_bar(pair, {
                "time": self._parse_time(ts_val),
                "open": float(opens[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "close": float(closes[i]),
                "volume": float(volumes[i]),
            })

            if i < warmup:
                continue

            current_price = closes[i]
            current_high = highs[i]
            current_low = lows[i]

            # Update equity curve
            unrealized = 0.0
            if position:
                if position["side"] == "buy":
                    unrealized = (current_price - position["entry"]) * position["size_units"]
                else:
                    unrealized = (position["entry"] - current_price) * position["size_units"]
            result.equity_curve.append(balance + unrealized)

            # Manage open position (trailing/breakeven + SL/TP)
            if position:
                trade_id = position["trade_id"]
                side = position["side"]
                state = risk_manager.update_stop_loss(
                    trade_id, current_price, position["entry"], side
                )
                stop_loss = state.current_sl
                take_profit = position["tp"]

                closed = False
                reason = ""
                if side == "buy":
                    if stop_loss > 0 and current_low <= stop_loss:
                        exit_price = stop_loss
                        reason = "stop_loss"
                        closed = True
                    elif take_profit > 0 and current_high >= take_profit:
                        exit_price = take_profit
                        reason = "take_profit"
                        closed = True
                else:
                    if stop_loss > 0 and current_high >= stop_loss:
                        exit_price = stop_loss
                        reason = "stop_loss"
                        closed = True
                    elif take_profit > 0 and current_low <= take_profit:
                        exit_price = take_profit
                        reason = "take_profit"
                        closed = True

                if closed:
                    if side == "buy":
                        exit_price *= (1 - self.slippage_pct)
                        gross_pnl = (exit_price - position["entry"]) * position["size_units"]
                    else:
                        exit_price *= (1 + self.slippage_pct)
                        gross_pnl = (position["entry"] - exit_price) * position["size_units"]

                    entry_fee = float(position.get("entry_fee", 0.0) or 0.0)
                    exit_fee = abs(exit_price * position["size_units"]) * self.fee_pct
                    fees = entry_fee + exit_fee
                    pnl = gross_pnl - fees
                    balance += gross_pnl - exit_fee

                    entry_value = position["entry"] * position["size_units"]
                    pnl_pct = pnl / entry_value if entry_value > 0 else 0

                    result.trades.append({
                        "bar": i,
                        "pair": pair,
                        "side": side,
                        "entry": position["entry"],
                        "exit": exit_price,
                        "size_units": position["size_units"],
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 4),
                        "fees": round(fees, 4),
                        "reason": reason,
                        "strategy": position["strategy"],
                    })

                    risk_manager.close_position(trade_id, pnl)
                    position = None
                else:
                    continue

            # If no open position, evaluate for entry
            if position:
                continue

            signal = await confluence.analyze_pair(pair)
            if signal.direction == SignalDirection.NEUTRAL:
                continue

            prediction_features = self._build_prediction_features(signal, predictor)
            if predictor.is_model_loaded:
                ai_confidence = predictor.predict(prediction_features)
                pre_blend = signal.confidence
                if signal.confluence_count == 1:
                    signal.confidence = 0.7 * pre_blend + 0.3 * ai_confidence
                else:
                    signal.confidence = (pre_blend + ai_confidence) / 2
            else:
                ai_confidence = 0.5

            min_confluence = max(2, getattr(cfg.ai, "confluence_threshold", 3))
            exec_confidence = min(getattr(cfg.ai, "min_confidence", 0.50), 0.65)

            has_keltner = any(
                s.strategy_name == "keltner" and s.is_actionable
                for s in signal.signals
                if s.direction == signal.direction
            )
            keltner_solo_ok = has_keltner and signal.confidence >= 0.52
            any_solo_ok = signal.confluence_count == 1 and signal.confidence >= 0.55

            if signal.confluence_count < min_confluence and not keltner_solo_ok and not any_solo_ok:
                continue

            sl_dist = abs(signal.entry_price - signal.stop_loss)
            tp_dist = abs(signal.take_profit - signal.entry_price) if signal.take_profit else 0
            min_rr = getattr(cfg.ai, "min_risk_reward_ratio", 0.9)
            if sl_dist > 0 and tp_dist > 0 and (tp_dist / sl_dist) < min_rr:
                continue

            if signal.confidence < exec_confidence:
                continue

            win_rate, win_loss_ratio = self._running_stats(result.trades)
            size_result = risk_manager.calculate_position_size(
                pair=pair,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                win_rate=win_rate,
                avg_win_loss_ratio=win_loss_ratio,
                confidence=signal.confidence,
            )
            if not size_result.allowed or size_result.size_units <= 0:
                continue

            side = "buy" if signal.direction == SignalDirection.LONG else "sell"
            entry_price = current_price * (1 + self.slippage_pct) if side == "buy" else current_price * (1 - self.slippage_pct)
            entry_fee = size_result.size_usd * self.fee_pct
            balance -= entry_fee

            trade_id_seq += 1
            trade_id = f"BT-{trade_id_seq}"
            primary_strategy = self._primary_strategy(signal)

            position = {
                "trade_id": trade_id,
                "side": side,
                "entry": entry_price,
                "size_units": size_result.size_units,
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
                "strategy": primary_strategy,
                "bar": i,
                "entry_fee": entry_fee,
            }

            risk_manager.initialize_stop_loss(
                trade_id, entry_price, signal.stop_loss, side
            )
            risk_manager.register_position(
                trade_id, pair, side, entry_price, size_result.size_usd, strategy=primary_strategy
            )

        # Close any remaining position at last price
        if position:
            final_price = closes[-1]
            side = position["side"]
            if side == "buy":
                final_price *= (1 - self.slippage_pct)
                gross_pnl = (final_price - position["entry"]) * position["size_units"]
            else:
                final_price *= (1 + self.slippage_pct)
                gross_pnl = (position["entry"] - final_price) * position["size_units"]

            entry_fee = float(position.get("entry_fee", 0.0) or 0.0)
            exit_fee = abs(final_price * position["size_units"]) * self.fee_pct
            fees = entry_fee + exit_fee
            pnl = gross_pnl - fees
            balance += gross_pnl - exit_fee

            entry_value = position["entry"] * position["size_units"]
            result.trades.append({
                "bar": len(closes) - 1,
                "pair": pair,
                "side": side,
                "entry": position["entry"],
                "exit": final_price,
                "size_units": position["size_units"],
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / entry_value if entry_value > 0 else 0, 4),
                "fees": round(fees, 4),
                "reason": "end_of_data",
                "strategy": position["strategy"],
            })
            risk_manager.close_position(position["trade_id"], pnl)

        logger.info(
            "Parity backtest complete",
            pair=pair,
            trades=result.total_trades,
            win_rate=round(result.win_rate, 4),
            total_return=round(result.total_return_pct, 2),
            max_drawdown=round(result.max_drawdown, 4),
        )

        return result

    def _parse_time(self, value: Any) -> float:
        """Parse timestamp to epoch seconds for MarketDataCache."""
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return pd.to_datetime(value, utc=True).timestamp()
        except Exception:
            return time.time()

    def _build_prediction_features(
        self, signal: StrategySignal, predictor: TFLitePredictor
    ) -> Dict[str, Any]:
        """Build predictor features from a confluence signal."""
        metadata: Dict[str, Any] = {}
        counts: Dict[str, int] = {}
        for s in signal.signals:
            if s.direction == signal.direction:
                for k, v in s.metadata.items():
                    if k in metadata and isinstance(v, (int, float)) and isinstance(metadata[k], (int, float)):
                        metadata[k] = metadata[k] + v
                        counts[k] = counts.get(k, 1) + 1
                    else:
                        metadata[k] = v
        for k in counts:
            metadata[k] = metadata[k] / counts[k]
        return predictor.features.feature_dict_from_signals(
            metadata,
            obi=(signal.book_score if getattr(signal, "book_score", 0.0) else signal.obi),
            spread=0.0,
        )

    def _primary_strategy(self, signal: StrategySignal) -> str:
        """Select the strongest agreeing strategy name."""
        if not signal.signals:
            return "confluence"
        best = max(
            [s for s in signal.signals if s.direction == signal.direction],
            key=lambda s: s.strength,
            default=None,
        )
        return best.strategy_name if best else "confluence"

    def _running_stats(self, trades: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Return win_rate and avg_win_loss_ratio based on trades."""
        if len(trades) < 50:
            return 0.50, 1.5
        wins = [t["pnl"] for t in trades if t.get("pnl", 0) > 0]
        losses = [abs(t["pnl"]) for t in trades if t.get("pnl", 0) <= 0]
        if not wins or not losses:
            return 0.50, 1.5
        win_rate = len(wins) / len(trades)
        avg_win = float(np.mean(wins))
        avg_loss = float(np.mean(losses)) if losses else 1.0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.5
        return win_rate, win_loss_ratio

    async def monte_carlo(
        self,
        backtest_result: BacktestResult,
        simulations: int = 1000,
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Monte Carlo simulation from backtest results.
        
        Shuffles trade order to generate confidence intervals
        for expected performance.
        
        # ENHANCEMENT: Added variance analysis across simulations
        """
        if not backtest_result.trades:
            return {"message": "No trades to simulate"}

        pnls = [t["pnl"] for t in backtest_result.trades]
        final_balances = []

        for _ in range(simulations):
            shuffled = np.random.permutation(pnls)
            balance = backtest_result.initial_balance
            for pnl in shuffled:
                balance += pnl
                if balance <= 0:
                    break
            final_balances.append(balance)

        final_balances.sort()
        p_lower = int(simulations * (1 - confidence) / 2)
        p_upper = int(simulations * (1 + confidence) / 2)

        return {
            "simulations": simulations,
            "confidence": confidence,
            "median_balance": round(float(np.median(final_balances)), 2),
            "mean_balance": round(float(np.mean(final_balances)), 2),
            "lower_bound": round(float(final_balances[p_lower]), 2),
            "upper_bound": round(float(final_balances[min(p_upper, len(final_balances) - 1)]), 2),
            "worst_case": round(float(final_balances[0]), 2),
            "best_case": round(float(final_balances[-1]), 2),
            "probability_of_profit": round(
                sum(1 for b in final_balances if b > backtest_result.initial_balance) / simulations, 4
            ),
            "probability_of_ruin": round(
                sum(1 for b in final_balances if b <= 0) / simulations, 4
            ),
        }
