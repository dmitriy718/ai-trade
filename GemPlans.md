# GemPlans - Optimization & Upgrade Roadmap

This document outlines the proposed technical upgrades for the AI Trading Bot v2.0.0 to improve performance, execution efficiency, and architectural robustness.

## Status: Work in Progress (Session 1 Complete)

---

## 1. High-Priority Refactors (Stability & Correctness)

### ✅ Isolate ML Training (Process Decoupling)
- **Status:** **Completed.**
- **Implementation:** Refactored `src/ml/trainer.py` to use `concurrent.futures.ProcessPoolExecutor` for the TensorFlow training loop (`model.fit`). This prevents the main asyncio event loop from freezing during the ~5-10 minute training/retraining cycles.

### ✅ Robust Position State Recovery
- **Status:** **Completed.**
- **Implementation:**
    - Updated `src/execution/risk_manager.py` to support serialization of `StopLossState` (including `trailing_high/low`).
    - Updated `src/execution/executor.py` to persist this state to the database `metadata` field on every update.
    - Implemented `reinitialize_positions` in `TradeExecutor` to fully restore the risk state from the DB on bot restart, preventing "phantom" stop-outs due to zeroed SL values.

---

## 2. Performance Optimizations

### ✅ NumPy RingBuffer for Market Data
- **Status:** **Completed.**
- **Implementation:**
    - Created `src/core/structures.py` with a high-performance `RingBuffer` class using pre-allocated NumPy arrays.
    - Refactored `src/exchange/market_data.py` to use `RingBuffer` for all OHLCV columns. This eliminates the O(N) memory allocation overhead of `np.append` on every tick, replacing it with O(1) pointer updates.

### ⏳ Parallel Strategy Execution
- **Status:** Pending.
- **Plan:** Implement a `SharedIndicatorCache` to compute common indicators (ATR, RSI) once per scan.

---

## 3. Trading Edge Upgrades

### ✅ Transition to Limit Orders (Maker/Taker Control)
- **Status:** **Completed.**
- **Implementation:**
    - Updated `src/execution/executor.py` to support `limit` orders.
    - `execute_signal` now fetches the current real-time `Ask` (for Buys) or `Bid` (for Sells) and places a Limit Order at that price instead of a Market order.
    - This prevents negative slippage beyond the spread and lays the groundwork for "Maker" strategies (post-only).

### ⏳ Dynamic Volatility Scaling (ATR-Based)
- **Status:** Pending.
- **Plan:** Adjust strategy thresholds based on ATR regime.

---

## 4. UI/UX Enhancements

### ⏳ Real-Time Candlestick Charts
- **Status:** Pending.
- **Plan:** Integrate `Lightweight Charts` into the dashboard.

### ⏳ Telegram Notification Engine
- **Status:** Pending.
- **Plan:** Implement `utils/telegram.py` for instant alerts.

---

## 5. Architectural Upgrade: Event-Driven Core
- **Status:** Pending.
- **Concept:** Move from polling to event-driven execution.

---

## Summary of Work (Session 1)
Crucial stability and performance fixes have been applied. The bot is now:
1.  **Resilient:** It won't crash-loop or lose position safety (stop losses) on restarts.
2.  **Responsive:** It won't freeze during ML training.
3.  **Efficient:** Market data ingestion is O(1) constant time/memory.
4.  **Cost-Effective:** It attempts to execute at specific prices (Limit) rather than blindly accepting Market fills.

**Next Steps:**
- Monitor the new Limit Order logic in live trading to ensure fills occur reliably.
- Proceed with "Parallel Strategy Execution" and "Dynamic Volatility Scaling" in the next session.