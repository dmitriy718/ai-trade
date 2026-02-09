# GPT Review 2 — Profit‑Critical Code & Performance Review

Date: 2026-02-08

Scope: Algorithms, confluence logic, strategy signals, execution, risk controls, and backtest/simulation paths that directly affect realized P&L.

## Summary
The trading stack is well-structured and the core strategy logic is coherent, but there are several profit‑impacting issues: a startup‑time bug that prevents strategy cooldowns from being wired, a large mismatch between backtesting and live execution logic, and execution reliability gaps (order sizing/precision and limit‑only fill logic). The biggest performance risk is that many metrics and tuning decisions are based on a backtester that does not mirror live behavior, which can systematically overstate expected returns.

## Strengths
- Indicator cache avoids recomputation inside each strategy per scan.
- Strategy implementations are explicit about entry conditions and include fee‑aware SL/TP via `compute_sl_tp`.
- Confluence layer includes regime‑based weighting and order‑book gating (when fresh).
- Risk manager has drawdown‑based scaling and conservative position caps.

## Findings

### Critical
1. **Cooldown checker wired before RiskManager exists**
   - File: `src/core/engine.py`
   - Issue: `self.confluence.set_cooldown_checker(self.risk_manager.is_strategy_on_cooldown)` is called before `self.risk_manager` is created. This will throw at startup or leave cooldowns unset.
   - Profit impact: Trading may not start at all, or strategy cooldowns never apply (increasing churn and fees).
   - Fix plan: Create `RiskManager` before `ConfluenceDetector`, or call `set_cooldown_checker` after risk manager initialization.

### High
2. **Backtester does not reflect live signal gating or execution**
   - Files: `src/ml/backtester.py`, `src/core/engine.py`, `src/ai/confluence.py`, `src/ai/predictor.py`
   - Issue: Backtester ignores AI predictor gating, OBI/book score, regime weighting, confluence confidence blending, and multi‑timeframe logic. It also doesn’t simulate live execution behaviors like limit order non‑fills.
   - Profit impact: Parameter tuning and strategy evaluation are biased toward results that won’t replicate live.
   - Fix plan: Add a “live‑parity” mode that uses the same confluence engine, AI predictor, and execution sizing logic; simulate limit fill probability and partials.

3. **Order sizing does not honor exchange minimums or precision**
   - Files: `src/execution/executor.py`, `src/exchange/kraken_rest.py`
   - Issue: Live orders are sent with raw sizes. There is no rounding to pair decimals or minimum size enforcement.
   - Profit impact: Frequent order rejections in live mode → missed fills, lost edge.
   - Fix plan: Use `get_min_order_size()` and `get_pair_decimals()` to clamp and round volume/price before submit.

4. **Limit‑only entry with no repricing or market fallback**
   - File: `src/execution/executor.py`
   - Issue: Live entries use limit at best bid/ask and wait up to 30s; if price runs, orders can remain unfilled and are canceled.
   - Profit impact: Signal edge decays; missed fills in fast markets; backtest overestimates fill rate.
   - Fix plan: Add configurable “chase” (reprice X times), or market fallback after timeout, or use IOC/IOC‑like logic.

5. **Backtester fee/slippage accounting is asymmetric and optimistic**
   - File: `src/ml/backtester.py`
   - Issues:
     - Entry fee is subtracted from balance but not added to per‑trade PnL or fee totals.
     - Slippage is applied only on entry, not on exit.
     - SL/TP exits assume perfect fills at stop/limit prices.
   - Profit impact: Metrics (profit factor, avg win/loss, Sharpe) are inflated relative to live.
   - Fix plan: Apply symmetric slippage and fees at entry and exit; include entry fees in per‑trade PnL/fees.

### Medium
6. **Strategy performance weighting never updates**
   - Files: `src/strategies/base.py`, `src/ai/confluence.py`, `src/execution/executor.py`
   - Issue: `record_trade_result()` is never called, so win_rate and `_trade_count` stay at 0; the performance‑adjusted weights in confluence never change.
   - Profit impact: Intended adaptive weighting doesn’t happen; strategy mix stays static.
   - Fix plan: Call `record_trade_result()` on the strategy used for each closed trade.

7. **OBI fallback can use stale order book**
   - Files: `src/ai/confluence.py`, `src/exchange/market_data.py`
   - Issue: `book_score` has freshness checks, but raw OBI uses `order_book` without checking `updated_at`.
   - Profit impact: Stale OBI can bias entries, especially in low‑liquidity periods.
   - Fix plan: Apply freshness checks to `order_book` before using OBI.

8. **Confluence chooses tightest SL + most conservative TP across strategies**
   - File: `src/ai/confluence.py`
   - Issue: Longs use max(SL) and min(TP); shorts use min(SL) and max(TP). This often tightens SL and narrows TP.
   - Profit impact: Higher stop‑out rate, smaller position sizes, and lower expected value.
   - Fix plan: Use weighted median or primary‑strategy SL/TP, or apply ATR‑based harmonization instead of min/max.

9. **AI confidence blending can distort signals when model is absent**
   - Files: `src/ai/predictor.py`, `src/core/engine.py`
   - Issue: Heuristic predictor is direction‑agnostic yet blended into signal confidence. This can dampen good signals or boost weak ones.
   - Profit impact: Fewer trades when the model is missing; inconsistent gating by confidence.
   - Fix plan: If model isn’t loaded, avoid blending or use a strategy‑specific heuristic.

10. **Global cooldown is hard‑coded and may be too blunt**
    - File: `src/execution/risk_manager.py`
    - Issue: Any loss triggers a 30‑minute global cooldown.
    - Profit impact: Missed opportunity during recovery periods; may reduce trade count and edge.
    - Fix plan: Make cooldown configurable and conditional (e.g., only after >X% loss or consecutive losses).

### Low
11. **Signal generation uses partially formed candles**
    - Files: `src/exchange/market_data.py`, `src/core/engine.py`, `src/ai/confluence.py`
    - Issue: Ticker updates mutate the current candle’s close and high/low. Strategies often use the most recent bar.
    - Profit impact: Higher noise and whipsaws; more false signals in fast markets.
    - Fix plan: Optionally analyze only closed candles for higher timeframes; allow per‑strategy choice.

12. **Async strategy execution is CPU‑bound**
    - Files: `src/ai/confluence.py`
    - Issue: Each strategy runs in an asyncio task but doesn’t yield; CPU work still executes serially with overhead.
    - Profit impact: Longer scan cycles under load; slower reaction to signals.
    - Fix plan: Execute sequentially or move to thread/process pool for true parallelism if needed.

13. **Fee model in `compute_sl_tp` is fixed, not config‑driven**
    - File: `src/utils/indicators.py`
    - Issue: SL/TP fee floor assumes a static 0.52% round‑trip taker fee.
    - Profit impact: TP/SL too wide for maker/post‑only or lower fee tiers; may reduce fills.
    - Fix plan: Inject config fee rates into SL/TP calculation or adjust per mode.

14. **Spread not used as a trade filter**
    - Files: `src/exchange/market_data.py`, `src/core/engine.py`
    - Issue: Spread is computed but not used for entry gating.
    - Profit impact: Trades in wide‑spread markets can get slippage‑heavy fills.
    - Fix plan: Add a max spread threshold per pair or volatility regime.

## Performance Observations
- Multi‑timeframe resampling runs per scan and per pair; consider maintaining per‑timeframe ring buffers in `MarketDataCache` for O(1) retrieval.
- The confluence pipeline is relatively heavy when scanning many pairs; scan time can increase beyond `scan_interval`, reducing responsiveness and increasing stale signals.

## Recommended Next Steps (Highest ROI)
1. Fix initialization order for cooldown checker (critical correctness).
2. Add live‑parity backtest mode and update fee/slippage modeling to match execution.
3. Implement order sizing rounding and minimum size checks before submission.
4. Add limit order chase/fallback logic and spread‑based entry gating.
5. Wire strategy performance updates to enable adaptive weighting.

## Suggested Validation
- Run a short live‑sim (paper) session with verbose execution logging and compare signal counts, fill rates, and PnL to live‑parity backtests.
- Add a regression report that compares backtest equity curves against a replay using the live execution path.

