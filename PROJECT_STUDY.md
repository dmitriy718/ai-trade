# Project Study: AI Crypto Trading Bot v2.0

**Purpose:** Deep study of every line of code and markdown in this repo — architecture, intent, and new ways to solve the same problem.

---

## 1. The Problem This Project Solves

**Core problem:** *Automate crypto trading on Kraken with an edge: high win rate, controlled risk, and self-improvement over time.*

Concretely:
- **Decision:** When to open/close positions across multiple pairs, without human intervention.
- **Edge:** Combine multiple technical strategies (trend, mean reversion, momentum, breakout, reversal, Keltner) so that only high-confluence, AI-verified setups get executed.
- **Risk:** Preserve capital via Kelly sizing, daily loss limits, ATR stops, trailing/breakeven logic, and one position per pair.
- **Observability:** Live dashboard (WebSocket), AI thought feed, scanner, risk shield — so the operator sees what the bot is doing and why.
- **Resilience:** Paper-first, graceful shutdown, health monitor, WS auto-reconnect, stress test that observes without interfering.

**Stakeholder:** A single operator (or small team) running a Kraken account in paper then live, aiming for longevity and a target win rate (e.g. 65–80%) with wins larger than losses.

---

## 2. Architecture (Visualized)

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                      main.py                              │
                    │  preflight → ConfigManager → setup_logging → run_bot()   │
                    │  run_bot: engine.initialize() → warmup() → uvicorn +     │
                    │           _main_scan_loop, _ws_data_loop, health,        │
                    │           cleanup, retrainer (all asyncio tasks)          │
                    └───────────────────────────┬───────────────────────────────┘
                                                │
        ┌───────────────────────────────────────┼───────────────────────────────────────┐
        │                          BotEngine (src/core/engine.py)                         │
        │  • initialize(): db, rest, ws, market_data, confluence, predictor,              │
        │                  order_book_analyzer, risk_manager, executor, ml_trainer,       │
        │                  retrainer, executor.reinitialize_positions(), dashboard       │
        │  • warmup(): historical OHLC per pair → market_data.warmup()                    │
        │  • _main_scan_loop(): manage positions → scan_all_pairs → AI blend → execute    │
        │  • _ws_data_loop(): on_ticker/ohlc/book/trade → connect → subscribe            │
        │  • _health_monitor(): WS alive?, stale pairs → REST refresh                     │
        │  • _cleanup_loop(): hourly DB cleanup                                           │
        │  • Handlers: _handle_ticker (update_latest_close only), _handle_ohlc, _handle_book│
        └───┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬───────┘
            │             │             │             │             │             │
    ┌───────▼───────┐ ┌───▼────┐ ┌──────▼──────┐ ┌────▼────┐ ┌─────▼─────┐ ┌─────▼─────┐
    │ MarketDataCache│ │Kraken  │ │Confluence   │ │TFLite   │ │RiskManager│ │Trade      │
    │ (exchange/    │ │REST/WS │ │Detector     │ │Predictor│ │(execution/│ │Executor   │
    │  market_data) │ │        │ │(ai/         │ │(ai/     │ │ risk_     │ │(execution/│
    │               │ │        │ │ confluence) │ │predictor)│ │ manager)  │ │ executor) │
    │ RingBuffer    │ │REST:   │ │             │ │         │ │           │ │           │
    │ per pair      │ │OHLC,   │ │ 6 strategies│ │ Blend   │ │ Kelly +   │ │ execute_  │
    │ COL_TIME..    │ │order,  │ │ in parallel │ │ 70/30   │ │ fixed %   │ │ signal →  │
    │ COL_VOLUME    │ │balance │ │ → weighted  │ │ solo,   │ │ daily     │ │ size →    │
    │ update_bar,   │ │WS:     │ │ strength/   │ │ 50/50   │ │ limit,    │ │ limit @   │
    │ update_       │ │ticker, │ │ confidence  │ │ multi   │ │ RoR,      │ │ ask/bid   │
    │ latest_close  │ │ohlc,   │ │ OBI bonus   │ │ heuristic│ │ trailing  │ │ paper/live│
    │ is_stale      │ │book    │ │ Sure Fire   │ │ if no   │ │ breakeven │ │ manage_   │
    └───────┬───────┘ └───┬────┘ └──────┬──────┘ │ model   │ └─────┬─────┘ │ positions │
            │             │             │        └────┬────┘       │       └─────┬─────┘
            │             │             │             │            │             │
    ┌───────▼─────────────▼─────────────▼─────────────▼────────────▼─────────────▼───────┐
    │  Strategies (strategies/*): Keltner, Trend, MeanReversion, Momentum, Breakout,    │
    │  Reversal. Each: analyze(pair, closes, highs, lows, volumes, opens) → StrategySignal│
    │  Indicators (utils/indicators.py): ema, sma, rsi, bollinger, adx, atr, macd,       │
    │  keltner_channels, volume_ratio, compute_sl_tp, order_book_imbalance, etc.        │
    └───────────────────────────────────────────────────────────────────────────────────┘
            │
    ┌───────▼───────────────────────────────────────────────────────────────────────────┐
    │  DatabaseManager (core/database.py): SQLite WAL, trades, signals, metrics,         │
    │  thought_log, ml_features, system_state, daily_summary. Async lock, whitelist     │
    │  updates.                                                                          │
    └───────────────────────────────────────────────────────────────────────────────────┘
            │
    ┌───────▼───────────────────────────────────────────────────────────────────────────┐
    │  DashboardServer (api/server.py): FastAPI, /api/v1/*, /ws/live, static.            │
    │  _build_ws_update() once/sec → performance, positions, thoughts, scanner, risk.   │
    │  Control: pause/resume/close_all behind X-API-Key (DASHBOARD_SECRET_KEY).         │
    └───────────────────────────────────────────────────────────────────────────────────┘
```

**Data flow (one scan cycle):**
1. Engine calls `executor.manage_open_positions()` (stops, trailing, TP).
2. Engine calls `confluence.scan_all_pairs(pairs)` → per pair: stale check → get OHLCV/opens → run 6 strategies in parallel (with timeout) → `_compute_confluence` (direction = majority of actionable signals, weighted strength/confidence, OBI, Sure Fire).
3. For each non-neutral ConfluenceSignal: build prediction features → `predictor.predict()` → blend confidence (solo 70/30, else 50/50) → log thought → gate: confluence≥2 or Keltner solo ≥0.52 or any solo ≥0.55 → if confidence ≥ 0.50 → `executor.execute_signal(signal)`.
4. Executor: validate → no open trade on same pair → risk_manager.calculate_position_size() → paper: _paper_fill at ask/bid; live: limit order + _wait_for_fill → insert_trade, init stop, register position.

---

## 3. Component-by-Component Summary

| Component | Role | Key implementation details |
|-----------|------|----------------------------|
| **main.py** | Entry; lifecycle owned here. No engine.start(); uvicorn signal handlers disabled; custom shutdown_event. | preflight_checks (dirs, .env warn); asyncio.run(run_bot()). |
| **Config** | YAML + env overlay, Pydantic. | ConfigManager singleton; env_mappings for risk/dashboard/ml; strategy configs (Keltner, trend, mean_reversion, …). |
| **Database** | Async SQLite WAL, one lock. | TRADE_UPDATE_COLUMNS whitelist; _ensure_ready; _ts() UTC; get_performance_stats, get_thoughts, cleanup_old_data. |
| **Engine** | Orchestrator. | initialize all deps; warmup parallel; scan loop (min_confluence=2, exec_confidence=0.50, solo rules); WS handlers only update_latest_close for ticker; health restarts WS task if dead. |
| **Kraken REST** | Auth, rate limit, retries. | OrderedDict for recent order IDs (FIFO); headers on GET; retry loop outside semaphore. |
| **Kraken WS** | v2, reconnect, subscribe. | callbacks: ticker, ohlc, book, trade; _resubscribe on reconnect. |
| **MarketDataCache** | RingBuffer per column per pair. | warmup writes valid bars only; update_bar timestamp tolerance; update_latest_close in-place; is_stale(180s in confluence). |
| **Strategies** | Each returns StrategySignal. | base: is_actionable = direction≠neutral, strength≥0.3, confidence≥0.3; _sanitize_for_json for metadata. |
| **Confluence** | Parallel strategies → one ConfluenceSignal per pair. | Stale + !is_warmed_up → neutral; weighted strength/confidence; OBI ±0.05; Sure Fire = threshold + OBI + min_confidence; SL/TP = max/min of agreeing. |
| **Predictor** | TFLite or heuristic. | feature_dict_from_signals; heuristic direction-agnostic (no OBI/trend/momentum boost); cache by rounded state. |
| **RiskManager** | Sizing and stops. | Fixed fractional primary; Kelly cap only if ≥50 trades and kelly_full>0; daily loss check is _daily_pnl <= -limit; trailing_high/low; drawdown factor. |
| **Executor** | Execute and manage. | One position per pair; limit at ask/bid; paper fill with tiny slippage; live 3-retry exit; PnL minus entry+exit fees. |
| **API/Dashboard** | REST + WS. | CORS localhost; control auth via header; WS payload built once/sec; thoughts, performance, positions, scanner, risk. |
| **ML** | Trainer + AutoRetrainer. | ProcessPoolExecutor for model.fit (no event-loop block); train on ml_features; save TFLite. |
| **Backtester** | Replay OHLC. | Bar loop, run strategies, simulate fills; Sharpe annualized for 1m bars. |
| **Indicators** | NumPy. | Fee-aware compute_sl_tp; NaN-safe ema; keltner_channels, macd, bb_position unclipped; volume_ratio 1.0 for warmup. |
| **Stress test** | Observation only. | API health, data freshness, WS, trading activity; no kill/restart. |

---

## 4. Intended Purposes (From Code + Docs)

- **Trade selectively:** Confluence + AI confidence so that only “Sure Fire” or high-conviction solo (e.g. Keltner) trades run.
- **Preserve capital:** Kelly, daily cap, RoR, trailing/breakeven, max position and cooldowns.
- **Stay correct:** No fake bars from ticker; no trading on stale data; fee-aware PnL; one position per pair.
- **Observe and control:** Dashboard with live PnL (realized + unrealized), thoughts, scanner, pause/resume/close_all.
- **Improve over time:** ML pipeline and backtester to tune and retrain the predictor from closed trades.
- **Run reliably:** Graceful shutdown, health monitor, WS restart, non-disruptive stress test.

---

## 5. New Ways to Solve the Same Problem

### A. **Event-driven execution (instead of fixed 30s scan)**

- **Idea:** Emit “signal events” from strategies when conditions cross thresholds (e.g. RSI crosses 30, or price touches Keltner lower band), then a small event bus decides whether to run confluence and execute.
- **Benefit:** React on candle close or on tick clusters (e.g. 1m bar closed) instead of waiting up to 30s; can reduce latency and align entries with bar boundaries.
- **Change:** Replace single `scan_all_pairs` loop with: (1) bar-close or tick handlers that enqueue “pair + reason” events, (2) deduplicated confluence run per pair per bar, (3) same execution pipeline.

### B. **Shared indicator cache (compute once per scan)**

- **Idea:** In ConfluenceDetector, before calling each strategy, compute common series once (e.g. RSI(14), ATR(14), EMA(20), EMA(50)) and pass them into strategies as optional kwargs.
- **Benefit:** Fewer repeated EMA/RSI/ATR calculations across Keltner, Trend, Mean Reversion, etc.; faster scan and lower CPU.
- **Change:** Add `SharedIndicatorCache` in confluence: one compute per (pair, bar_count), then strategies accept `cached_indicators` and use them when present.

### C. **Regime-aware thresholds (volatility / trend regime)**

- **Idea:** Classify “regime” per pair (e.g. high ATR vs low ATR, ADX > 25 vs < 25) and adjust strategy thresholds or weights (e.g. in chop, favor Keltner/mean reversion; in trend, favor trend/momentum).
- **Change:** In engine or confluence, compute regime once per pair (e.g. ADX, ATR percentile); pass to confluence; ConfluenceDetector or strategy factory chooses weights/thresholds from config per regime.

### D. **Multi-timeframe confluence**

- **Idea:** Run the same (or simplified) strategies on 1m, 5m, 15m; require agreement across timeframes for entry (e.g. Keltner long on 5m and 1m).
- **Benefit:** Fewer false breakouts; alignment with higher-timeframe structure.
- **Change:** Market data and warmup per timeframe; separate ConfluenceSignal per (pair, timeframe); engine rule: e.g. only execute if 1m and 5m agree and 1m confidence > threshold.

### E. **Explicit “no trade” and cooldown per setup type**

- **Idea:** After a Keltner long is closed (win or loss), avoid another Keltner long on that pair for N bars or M minutes; same for other strategy types.
- **Benefit:** Reduces re-entry on the same setup in a short window; can backtest per-setup cooldowns.
- **Change:** DB or in-memory store: last exit time per (pair, strategy_id); risk or engine skips that strategy for that pair until cooldown elapsed.

### F. **Reinforcement learning for position sizing and “trade or skip”**

- **Idea:** Keep signal generation as-is but add a small RL agent (e.g. DQN or policy gradient) that learns to scale size or skip trade based on recent PnL and current features.
- **Benefit:** Could learn to reduce size after losses or in certain regimes without hand-coded drawdown rules only.
- **Change:** Predictor or a new module outputs “scale factor” or “skip”; executor applies it; training loop from closed trades and equity curve.

### G. **Order book as first-class signal (not only OBI)**

- **Idea:** Use order book levels and flow (e.g. bid/ask volume deltas over 10s) as inputs to a dedicated “microstructure” score and combine with technical confluence.
- **Benefit:** Earlier detection of imbalance or exhaustion; can improve entry timing.
- **Change:** OrderBookAnalyzer already exists; add a small model or rules that output a “book_score” per pair; feed into confluence weight or into predictor features.

### H. **Backtest-driven parameter tuning (systematic)**

- **Idea:** Use the existing backtester in a loop: grid or Bayesian search over strategy params (e.g. Keltner RSI bands, trend ADX threshold) and pick the set that maximizes Sharpe or win rate under a drawdown constraint.
- **Benefit:** Reduces manual tuning; can re-run periodically on rolling windows.
- **Change:** Script that loads historical OHLCV, runs Backtester for each param set, records metrics; selects best; optionally writes config or model.

### I. **Simpler “single strategy” mode for testing**

- **Idea:** Config flag: e.g. `trading.single_strategy: "keltner"` so that only Keltner runs and no confluence (or trivial confluence of 1).
- **Benefit:** Isolate strategy performance and debug one algo without interaction with others.
- **Change:** ConfluenceDetector or engine: if single_strategy set, run only that strategy and treat as confluence_count=1 with same confidence/solo rules.

### J. **Telegram as primary control plane**

- **Idea:** Extend Telegram bot (utils/telegram.py) so that all critical actions (pause, resume, close_all, and optionally “allow next trade” or “size scale”) go through Telegram with chat_id verification; dashboard becomes read-only or optional.
- **Benefit:** Secure remote control from a single device; audit trail in chat.
- **Change:** Enforce chat_id allowlist; require Telegram for destructive actions; optional: 2FA or code for close_all.

---

## 6. Summary

- **What it is:** A production-style, multi-strategy, AI-checked crypto trading bot for Kraken with strict risk, live dashboard, and self-improvement (ML + backtester).
- **What it does well:** Clear separation of data → strategies → confluence → AI → risk → execution; resilience (stale check, WS restart, fee-aware PnL, one position per pair); observability and safe control.
- **Where it can go next:** Event-driven scans, shared indicators, regime-aware params, multi-timeframe confluence, per-setup cooldowns, RL for size/skip, richer order-book signals, systematic backtest tuning, single-strategy mode, and Telegram-first control.

This document can serve as the single place to “see” the whole system and to pick the next improvement (e.g. start with B + C for quick wins, then A or D for structural change).
