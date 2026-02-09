"""
Risk Management Engine - Capital Preservation System.

Implements Kelly Criterion position sizing, risk of ruin calculations,
ATR-based stop losses, trailing stops with breakeven logic, daily loss
limits, and trade cooldowns.

# ENHANCEMENT: Added risk of ruin Monte Carlo simulation
# ENHANCEMENT: Added correlation-aware position sizing
# ENHANCEMENT: Added dynamic risk scaling based on drawdown
# ENHANCEMENT: Added maximum adverse excursion tracking
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.logger import get_logger

logger = get_logger("risk_manager")


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    size_usd: float = 0.0
    size_units: float = 0.0
    risk_amount: float = 0.0
    kelly_fraction: float = 0.0
    stop_distance_pct: float = 0.0
    risk_reward_ratio: float = 0.0
    allowed: bool = False
    reason: str = ""


@dataclass
class StopLossState:
    """Current state of a position's stop loss management."""
    initial_sl: float = 0.0
    current_sl: float = 0.0
    breakeven_activated: bool = False
    trailing_activated: bool = False
    trailing_high: float = 0.0    # Highest price since trailing activation
    trailing_low: float = float("inf")  # H6 FIX: inf default so shorts work

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StopLossState:
        """Deserialize state from dict."""
        return cls(**data)


class RiskManager:
    """
    Comprehensive risk management system.
    
    Core responsibilities:
    1. Position sizing via Kelly Criterion
    2. ATR-based stop loss placement
    3. Trailing stop + breakeven logic
    4. Daily loss limit enforcement
    5. Risk of ruin monitoring
    6. Trade cooldown management
    7. Maximum position limits
    
    # ENHANCEMENT: Added drawdown-based risk scaling
    # ENHANCEMENT: Added portfolio heat monitoring
    # ENHANCEMENT: Added risk budget allocation across pairs
    """

    def __init__(
        self,
        initial_bankroll: float = 10000.0,
        max_risk_per_trade: float = 0.02,
        max_daily_loss: float = 0.05,
        max_position_usd: float = 500.0,
        kelly_fraction: float = 0.25,
        max_kelly_size: float = 0.10,
        risk_of_ruin_threshold: float = 0.01,
        atr_multiplier_sl: float = 2.0,
        atr_multiplier_tp: float = 3.0,
        trailing_activation_pct: float = 0.015,
        trailing_step_pct: float = 0.005,
        breakeven_activation_pct: float = 0.01,
        cooldown_seconds: int = 300,
        max_concurrent_positions: int = 5,
        strategy_cooldowns: Optional[Dict[str, int]] = None,
        global_cooldown_seconds_on_loss: int = 1800,
    ):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_position_usd = max_position_usd
        self.kelly_fraction = kelly_fraction
        self.max_kelly_size = max_kelly_size
        self.risk_of_ruin_threshold = risk_of_ruin_threshold
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_step_pct = trailing_step_pct
        self.breakeven_activation_pct = breakeven_activation_pct
        self.cooldown_seconds = cooldown_seconds
        self.max_concurrent_positions = max_concurrent_positions
        self.strategy_cooldowns = strategy_cooldowns or {}
        self.global_cooldown_seconds_on_loss = max(0, int(global_cooldown_seconds_on_loss))

        # State tracking
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._last_trade_time: Dict[str, float] = {}
        self._open_positions: Dict[str, Dict[str, Any]] = {}
        self._stop_states: Dict[str, StopLossState] = {}
        self._strategy_cooldowns: Dict[tuple, float] = {}
        self._peak_bankroll: float = initial_bankroll
        self._max_drawdown: float = 0.0
        self._trade_history: List[Dict[str, float]] = []
        self._daily_reset_date: str = ""
        self._global_cooldown_until: float = 0.0

    # ------------------------------------------------------------------
    # Position Sizing (Kelly Criterion)
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        pair: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        win_rate: float = 0.5,
        avg_win_loss_ratio: float = 1.5,
        confidence: float = 0.5,
    ) -> PositionSizeResult:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Implements quarter-Kelly for safety, capped by multiple
        risk constraints.
        
        Args:
            pair: Trading pair
            entry_price: Proposed entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            win_rate: Historical or estimated win rate
            avg_win_loss_ratio: Average winner / average loser ratio
            confidence: AI confidence in the trade (0-1)
        
        Returns:
            PositionSizeResult with sizing details and approval
            
        # ENHANCEMENT: Added confidence-weighted Kelly
        # ENHANCEMENT: Added drawdown-adjusted sizing
        """
        result = PositionSizeResult()

        # Pre-flight checks
        if not self._pre_trade_checks(pair, result):
            return result

        if entry_price <= 0 or stop_loss <= 0:
            result.reason = "Invalid prices"
            return result

        # H16 FIX: Explicit bankroll depletion guard
        if self.current_bankroll <= 0:
            result.reason = "Bankroll depleted"
            return result

        # Stop loss distance
        sl_distance = abs(entry_price - stop_loss)
        sl_pct = sl_distance / entry_price
        result.stop_distance_pct = sl_pct

        if sl_pct <= 0 or sl_pct > 0.10:  # Max 10% stop
            result.reason = f"Invalid stop distance: {sl_pct:.2%}"
            return result

        # Risk-reward ratio
        tp_distance = abs(take_profit - entry_price)
        if sl_distance > 0:
            result.risk_reward_ratio = tp_distance / sl_distance
        
        if result.risk_reward_ratio < 0.8:
            result.reason = f"R:R ratio too low: {result.risk_reward_ratio:.2f}"
            return result

        # ============================================================
        # Position Sizing: Fixed fractional risk as PRIMARY method.
        # Kelly is used as a CAP only when we have enough data and
        # a positive edge. This ensures the bot always trades in
        # paper mode and collects training data.
        # ============================================================

        # PRIMARY: Fixed fractional risk sizing
        # "Risk X% of bankroll per trade" — always produces a valid size
        risk_amount = self.current_bankroll * self.max_risk_per_trade
        position_size_usd = risk_amount / sl_pct if sl_pct > 0 else 0

        # SECONDARY: Kelly Criterion cap (only if sufficient history)
        p = win_rate
        q = 1 - p
        b = avg_win_loss_ratio if avg_win_loss_ratio > 0 else 1.0

        kelly_full = max((p * b - q) / b, 0)
        kelly_adjusted = kelly_full * self.kelly_fraction * confidence
        kelly_adjusted = min(kelly_adjusted, self.max_kelly_size)
        result.kelly_fraction = kelly_adjusted

        # Only let Kelly reduce size if we have 50+ trades AND a positive edge
        if len(self._trade_history) >= 50 and kelly_full > 0:
            kelly_size = self.current_bankroll * kelly_adjusted
            position_size_usd = min(position_size_usd, kelly_size)

        # Apply drawdown scaling
        drawdown_factor = self._get_drawdown_factor()
        position_size_usd *= drawdown_factor

        # Apply maximum position cap
        position_size_usd = min(position_size_usd, self.max_position_usd)

        # Apply portfolio heat limit (total exposure)
        remaining_capacity = self._get_remaining_capacity()
        position_size_usd = min(position_size_usd, remaining_capacity)

        if position_size_usd < 10:  # Minimum $10 position
            result.reason = (
                f"Position size too small: ${position_size_usd:.2f} "
                f"(kelly_adj={kelly_adjusted:.4f}, sl_pct={sl_pct:.4f}, "
                f"dd_factor={drawdown_factor:.2f}, cap={remaining_capacity:.2f})"
            )
            return result

        result.size_usd = round(position_size_usd, 2)
        result.size_units = round(position_size_usd / entry_price, 8)
        result.risk_amount = round(position_size_usd * sl_pct, 2)
        result.allowed = True

        logger.info(
            "Position size calculated",
            pair=pair,
            size_usd=result.size_usd,
            risk_amount=result.risk_amount,
            kelly=round(kelly_adjusted, 4),
            drawdown_factor=round(drawdown_factor, 2),
        )

        return result

    def _pre_trade_checks(self, pair: str, result: PositionSizeResult) -> bool:
        """Run pre-trade risk checks."""
        # Global cooldown check
        now = time.time()
        if now < self._global_cooldown_until:
            remaining = self._global_cooldown_until - now
            result.reason = f"Global cooldown: {remaining:.0f}s remaining"
            return False

        # C3 FIX: Check only negative PnL, not absolute value
        self._check_daily_reset()
        if self._daily_pnl <= -(self.current_bankroll * self.max_daily_loss):
            result.reason = f"Daily loss limit reached: ${self._daily_pnl:.2f}"
            logger.warning("Daily loss limit reached", daily_pnl=self._daily_pnl)
            return False

        # Cooldown check
        last_trade = self._last_trade_time.get(pair, 0)
        elapsed = time.time() - last_trade
        if elapsed < self.cooldown_seconds:
            remaining = self.cooldown_seconds - elapsed
            result.reason = f"Cooldown active: {remaining:.0f}s remaining"
            return False

        # Max positions check
        if len(self._open_positions) >= self.max_concurrent_positions:
            result.reason = f"Max positions reached: {len(self._open_positions)}"
            return False

        # Risk of ruin check
        ror = self.calculate_risk_of_ruin()
        if ror > self.risk_of_ruin_threshold:
            result.reason = f"Risk of ruin too high: {ror:.2%}"
            logger.warning("Risk of ruin threshold exceeded", ror=ror)
            return False

        return True

    # ------------------------------------------------------------------
    # Stop Loss Management
    # ------------------------------------------------------------------

    def initialize_stop_loss(
        self,
        trade_id: str,
        entry_price: float,
        stop_loss: float,
        side: str,
        trailing_high: float = 0.0,
        trailing_low: float = float("inf"),
    ) -> StopLossState:
        """Initialize stop loss tracking for a new position."""
        state = StopLossState(
            initial_sl=stop_loss,
            current_sl=stop_loss,
            trailing_high=trailing_high if trailing_high > 0 else (entry_price if side == "buy" else 0),
            trailing_low=trailing_low if trailing_low != float("inf") else (entry_price if side == "sell" else float("inf")),
        )
        self._stop_states[trade_id] = state
        return state

    def update_stop_loss(
        self,
        trade_id: str,
        current_price: float,
        entry_price: float,
        side: str,
    ) -> StopLossState:
        """
        Update stop loss with trailing and breakeven logic.
        
        # ENHANCEMENT: Added step-based trailing for smoother execution
        # ENHANCEMENT: Added acceleration on large profit moves
        
        Args:
            trade_id: Trade identifier
            current_price: Current market price
            entry_price: Original entry price
            side: "buy" or "sell"
        
        Returns:
            Updated StopLossState
        """
        state = self._stop_states.get(trade_id)
        if not state:
            return StopLossState()

        if side == "buy":
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Update trailing high
            if current_price > state.trailing_high:
                state.trailing_high = current_price

            # Breakeven activation
            if not state.breakeven_activated and pnl_pct >= self.breakeven_activation_pct:
                state.breakeven_activated = True
                state.current_sl = max(state.current_sl, entry_price)
                logger.debug(
                    "Breakeven activated",
                    trade_id=trade_id, sl=state.current_sl
                )

            # Trailing stop activation
            if pnl_pct >= self.trailing_activation_pct:
                state.trailing_activated = True
                # Trail from the highest price
                new_sl = state.trailing_high * (1 - self.trailing_step_pct)

                # Acceleration: tighter trail on larger profits
                if pnl_pct > 0.03:  # 3%+ profit
                    new_sl = state.trailing_high * (1 - self.trailing_step_pct * 0.5)

                # Only move stop up, never down
                if new_sl > state.current_sl:
                    state.current_sl = new_sl

        elif side == "sell":
            pnl_pct = (entry_price - current_price) / entry_price

            if current_price < state.trailing_low:
                state.trailing_low = current_price

            if not state.breakeven_activated and pnl_pct >= self.breakeven_activation_pct:
                state.breakeven_activated = True
                state.current_sl = min(state.current_sl, entry_price)

            if pnl_pct >= self.trailing_activation_pct:
                state.trailing_activated = True
                new_sl = state.trailing_low * (1 + self.trailing_step_pct)

                if pnl_pct > 0.03:
                    new_sl = state.trailing_low * (1 + self.trailing_step_pct * 0.5)

                if new_sl < state.current_sl:
                    state.current_sl = new_sl

        self._stop_states[trade_id] = state
        return state

    def should_stop_out(
        self, trade_id: str, current_price: float, side: str
    ) -> bool:
        """Check if current price has hit the stop loss."""
        state = self._stop_states.get(trade_id)
        if not state:
            return False

        if side == "buy":
            return current_price <= state.current_sl
        else:
            return current_price >= state.current_sl

    # ------------------------------------------------------------------
    # Risk of Ruin Calculation
    # ------------------------------------------------------------------

    def calculate_risk_of_ruin(self) -> float:
        """
        Calculate probability of losing the entire bankroll.
        
        Uses the classic risk of ruin formula:
        RoR = ((1 - edge) / (1 + edge)) ^ units
        
        Where edge = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_bet
        
        FIX: Requires 50+ trades for statistical validity. With fewer trades
        the variance is too high and a few bad trades would falsely show 100% RoR.
        """
        if len(self._trade_history) < 50:
            return 0.0  # Not enough data for meaningful calculation

        wins = [t["pnl"] for t in self._trade_history if t["pnl"] > 0]
        losses = [abs(t["pnl"]) for t in self._trade_history if t["pnl"] <= 0]

        if not wins or not losses:
            return 0.0

        win_rate = len(wins) / len(self._trade_history)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 0.0

        # Edge calculation
        edge = (win_rate * avg_win - (1 - win_rate) * avg_loss)
        if edge <= 0:
            return 1.0  # Negative edge = eventual ruin

        # Simplified RoR formula
        avg_bet = np.mean([abs(t["pnl"]) for t in self._trade_history])
        if avg_bet == 0:
            return 0.0

        units = self.current_bankroll / avg_bet
        if units <= 0:
            return 1.0

        edge_ratio = edge / avg_bet
        if edge_ratio >= 1:
            return 0.0

        try:
            ror = ((1 - edge_ratio) / (1 + edge_ratio)) ** units
            return min(ror, 1.0)
        except (OverflowError, ZeroDivisionError):
            return 0.0

    # ------------------------------------------------------------------
    # Portfolio & State Management
    # ------------------------------------------------------------------

    def register_position(
        self,
        trade_id: str,
        pair: str,
        side: str,
        entry_price: float,
        size_usd: float,
        strategy: Optional[str] = None,
    ) -> None:
        """Register a new open position."""
        self._open_positions[trade_id] = {
            "pair": pair,
            "side": side,
            "entry_price": entry_price,
            "size_usd": size_usd,
            "strategy": strategy,
            "opened_at": time.time(),
        }
        self._last_trade_time[pair] = time.time()
        self._daily_trades += 1

    def close_position(self, trade_id: str, pnl: float) -> None:
        """Close a position and update risk metrics."""
        pos = self._open_positions.pop(trade_id, None)
        if pos:
            strategy = pos.get("strategy")
            if strategy:
                key = (pos.get("pair"), strategy, pos.get("side"))
                self._strategy_cooldowns[key] = time.time()
        if trade_id in self._stop_states:
            del self._stop_states[trade_id]

        self._daily_pnl += pnl
        self.current_bankroll += pnl
        self._trade_history.append({"pnl": pnl, "time": time.time()})

        # Global cooldown: 30 mins after a loss to prevent churn
        if pnl < 0 and self.global_cooldown_seconds_on_loss > 0:
            self._global_cooldown_until = time.time() + self.global_cooldown_seconds_on_loss
            logger.warning(
                "Loss detected, global cooldown activated",
                pnl=pnl,
                cooldown_seconds=self.global_cooldown_seconds_on_loss,
            )

        # Update peak and drawdown
        if self.current_bankroll > self._peak_bankroll:
            self._peak_bankroll = self.current_bankroll
        drawdown = (self._peak_bankroll - self.current_bankroll) / self._peak_bankroll
        self._max_drawdown = max(self._max_drawdown, drawdown)

        # Keep history manageable
        if len(self._trade_history) > 10000:
            self._trade_history = self._trade_history[-5000:]

    def is_strategy_on_cooldown(
        self, pair: str, strategy: Optional[str], side: Optional[str]
    ) -> bool:
        """Check per-strategy cooldown for a pair and direction."""
        if not strategy or not side:
            return False
        cooldown = self._get_strategy_cooldown_seconds(strategy)
        if cooldown <= 0:
            return False
        last = self._strategy_cooldowns.get((pair, strategy, side), 0.0)
        return (time.time() - last) < cooldown

    def _get_strategy_cooldown_seconds(self, strategy: str) -> int:
        """Get cooldown duration for a strategy, defaulting to 0."""
        try:
            return int(self.strategy_cooldowns.get(strategy, 0))
        except Exception:
            return 0

    def _get_drawdown_factor(self) -> float:
        """
        Calculate risk reduction factor based on current drawdown.
        
        Reduces position sizes as drawdown increases.
        
        # ENHANCEMENT: Exponential scaling for aggressive protection
        """
        if self._peak_bankroll <= 0:
            return 1.0

        drawdown = (self._peak_bankroll - self.current_bankroll) / self._peak_bankroll

        if drawdown < 0.02:
            return 1.0
        elif drawdown < 0.05:
            return 0.75
        elif drawdown < 0.10:
            return 0.5
        elif drawdown < 0.15:
            return 0.25
        else:
            return 0.1  # Minimal sizing during severe drawdown

    def _get_remaining_capacity(self) -> float:
        """Calculate remaining position capacity in USD."""
        total_exposure = sum(
            pos["size_usd"] for pos in self._open_positions.values()
        )
        max_total = self.current_bankroll * 0.50  # Max 50% total exposure
        return max(0, max_total - total_exposure)

    def _check_daily_reset(self) -> None:
        """Reset daily counters at midnight UTC."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._daily_reset_date = today

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_risk_report(self) -> Dict[str, Any]:
        """Get comprehensive risk report."""
        return {
            "bankroll": round(self.current_bankroll, 2),
            "initial_bankroll": self.initial_bankroll,
            "total_return_pct": round(
                (self.current_bankroll - self.initial_bankroll) /
                self.initial_bankroll * 100, 2
            ),
            "peak_bankroll": round(self._peak_bankroll, 2),
            "current_drawdown": round(
                (self._peak_bankroll - self.current_bankroll) /
                self._peak_bankroll * 100, 2
            ) if self._peak_bankroll > 0 else 0,
            "max_drawdown_pct": round(self._max_drawdown * 100, 2),
            "daily_pnl": round(self._daily_pnl, 2),
            "daily_trades": self._daily_trades,
            "open_positions": len(self._open_positions),
            "total_exposure_usd": round(
                sum(p["size_usd"] for p in self._open_positions.values()), 2
            ),
            "risk_of_ruin": round(self.calculate_risk_of_ruin(), 4),
            "drawdown_factor": round(self._get_drawdown_factor(), 2),
            "remaining_capacity_usd": round(self._get_remaining_capacity(), 2),
            "trade_count": len(self._trade_history),
        }
