# GPT Review 3 — Profit‑Path Follow‑Up (Post‑Fix)

Date: 2026-02-08

Scope: Trading algorithms, execution, risk, and live/backtest parity after the recent fixes. Includes a brief review of adjacent systems that influence trading reliability.

## Summary
Core profit‑path issues from GPTRev2 are resolved: cooldown wiring, live‑parity backtest mode, symmetric fee/slippage accounting, order size precision, limit‑order chase/fallback, adaptive strategy weighting, and spread gating. Remaining risks are mostly around data freshness in execution decisions and continued mismatch between simulated fills and live limit order behavior.

## Profit‑Critical Findings (Remaining)

### Medium
1. **Limit‑order chase uses ticker without freshness checks**
   - File: `src/execution/executor.py`
   - Issue: `_best_limit_price()` uses cached ticker without validating `updated_at`.
   - Profit impact: In fast markets or stale data, chase can reprice incorrectly or chase a stale quote.
   - Suggested fix: Require ticker freshness (e.g., < 2s) or fall back to last close with a configurable offset.

2. **Parity backtest still assumes fills with slippage (no “no‑fill” modeling)**
   - File: `src/ml/backtester.py`
   - Issue: Even in parity mode, entries/exits assume fills with slippage; limit order rejection/timeout behavior isn’t modeled.
   - Profit impact: Live fill rates may be lower; expected returns still optimistic.
   - Suggested fix: Simulate fill probability based on spread/volatility and use limit‑timeout‑then‑fallback logic in backtests.

### Low
3. **Spread filter relies on order book availability**
   - File: `src/core/engine.py`
   - Issue: If no order book is present, spread check is skipped (spread=0).
   - Profit impact: Trades may occur in wide‑spread conditions if book isn’t available.
   - Suggested fix: Gate spread filter on ticker best bid/ask or skip trading when order book is missing in live mode.

4. **Fee‑aware SL/TP uses taker fee by default**
   - File: `src/ai/confluence.py`, `src/utils/indicators.py`
   - Issue: `round_trip_fee_pct` is set from taker fees; if maker/post‑only is dominant, TP/SL can be wider than necessary.
   - Profit impact: Slight reduction in fill rate for tight strategies; not a correctness bug.
   - Suggested fix: Use a blended fee rate or choose based on post‑only config.

## Brief Review of Adjacent Systems (Trading‑Impacting)
- **Market data integrity**: `MarketDataCache` uses per‑pair locks and in‑place updates; stable for live. However, in‑progress candle usage is still a config choice (`use_closed_candles_only`), which can trade responsiveness for noise reduction.
- **WebSocket health**: The WS loop has reconnection handling; the health monitor restarts the WS task if dead. This is good for minimizing stale signals.
- **Risk manager**: Drawdown scaling and max exposure are conservative; global cooldown is now configurable. Behavior aligns with capital preservation but can reduce trade count under volatility.
- **Database writes**: Trades and thought logs are async and relatively lightweight. No immediate performance bottlenecks observed in write paths.

## Overall Status
Trading pipeline is now materially closer to live reality. Remaining risks are manageable and mostly about data freshness and fill realism. If you want, I can implement ticker freshness checks and a no‑fill model for parity backtests next.
