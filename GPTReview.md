# GPT Review — Comprehensive Code Review

Date: 2026-02-08

Scope: Full repo after all requested feature implementations. Focus on correctness, security, reliability, and operational risk. Includes new multi-timeframe, event-driven scanning, tenant scoping, execution enhancements, and parameter tuning.

## Summary
The system is architecturally strong and modular, but several issues remain around tenant isolation, WebSocket caching, execution fill handling, and configuration hygiene. The highest risk items are cross-tenant WebSocket caching and reliance on a default dashboard secret. There are also correctness risks in fill handling (partial fills) and regime/timeframe alignment that can lead to incorrect trading decisions.

## Findings
1. **Critical — Cross‑Tenant Data Leak via WebSocket Cache**  
File: `src/api/server.py`  
Issue: `_ws_cache` is global. When multiple tenants connect, the last tenant’s cache can be reused and sent to other tenants. This breaks tenant isolation.  
Fix plan: Key the cache by tenant_id (e.g., `_ws_cache_by_tenant: Dict[str, (payload, time)]`), and build per-tenant payloads.  
Improve: Add tenant-aware WS channels (`/ws/live?tenant_id=...`) with per-tenant rate limits and explicit access tokens.

2. **High — Dashboard Control Key Defaults to Insecure Value**  
File: `src/api/server.py`  
Issue: `_api_secret` defaults to `"change_this_to_a_random_string"` if env is missing. This is unsafe in production.  
Fix plan: Fail fast on startup if `DASHBOARD_SECRET_KEY` is missing in live mode, or generate a secure key on first run and persist it.  
Improve: Move secrets into `SecureVault` and rotate periodically with explicit admin workflow.

3. **High — Tenant Scoping Is Read‑Only (Writes Always Default)**  
Files: `src/core/engine.py`, `src/core/database.py`, `src/execution/executor.py`  
Issue: Trades, thoughts, and ML features are written without tenant context. Read endpoints are scoped, but writes always land in `tenant_id = 'default'`.  
Fix plan: Propagate tenant_id through execution and logging paths or create per-tenant engines.  
Improve: Introduce a tenant-aware service layer that passes tenant_id through all DB operations and isolates state in memory.

4. **High — Partial Fill Handling Still Loses Some Executions**  
File: `src/execution/executor.py`  
Issue: `_wait_for_fill` relies on `cost` from open orders; if `cost` is zero but `vol_exec > 0`, partial fills can be missed and orders canceled without recording actual filled volume.  
Fix plan: Use `price`/`avg_price` or compute from trade history when `vol_exec > 0`.  
Improve: Subscribe to Kraken trade WS stream and reconcile fills in near-real time, including fee data.

5. **Medium — Maker/Taker Fee Assumption Is Heuristic**  
File: `src/execution/executor.py`  
Issue: Fee rate is inferred from `post_only`. Actual fills can still be taker or maker depending on order behavior. P&L may be biased.  
Fix plan: Retrieve per-order fee data from exchange and store in DB.  
Improve: Add execution quality metrics (maker/taker ratio, average fee, slippage distribution).

6. **Medium — Multi‑Timeframe Resampling Ignores Candle Boundaries**  
File: `src/ai/confluence.py`  
Issue: `_resample_ohlcv` groups last N bars by array position, not by actual timestamps. This can misalign 5m/15m candles if data starts mid‑interval.  
Fix plan: Use timestamps from `MarketDataCache` and align to true timeframe boundaries.  
Improve: Maintain dedicated per-timeframe ring buffers in `MarketDataCache` to avoid resampling in hot paths.

7. **Medium — Event‑Driven Scan Loop Can Delay Stop Management**  
File: `src/core/engine.py`  
Issue: Position management runs once per scan loop. Under event-driven conditions with large `scan_interval`, stop management can be delayed.  
Fix plan: Split stop management into its own periodic task (e.g., every 1–5 seconds) independent of scan cadence.  
Improve: Trigger stop checks on tick/price updates per pair for tighter risk control.

8. **Medium — Order‑Book Score Staleness Not Checked**  
Files: `src/ai/confluence.py`, `src/exchange/market_data.py`  
Issue: `book_score` is used without verifying freshness. If WS order book stalls, stale score can influence trades.  
Fix plan: Store `updated_at` from `update_order_book_analysis` and require freshness before using `book_score`; fallback to neutral or OBI when stale.  
Improve: Add health metrics for order book freshness and auto-disable microstructure gating on stale data.

9. **Medium — Env Overrides Still Reference Non‑existent Config Fields**  
File: `src/core/config.py`  
Issue: `INITIAL_BANKROLL` and `DB_PATH` are mapped to `risk.initial_bankroll` and `app.db_path`, but those fields do not exist in the models. Overrides are silently ignored.  
Fix plan: Add the missing fields to Pydantic models or remove mappings.  
Improve: Add explicit validation errors when env overrides target unknown fields.

10. **Low — Parameter Tuning Overwrites Config Formatting**  
File: `param_tune.py`  
Issue: Writing with `yaml.safe_dump` removes comments and ordering. This can lose important annotations in `config.yaml`.  
Fix plan: Use `ruamel.yaml` to preserve formatting.  
Improve: Write to a separate overlay file and merge at runtime.

11. **Low — Parameter Tuning Is Non‑Deterministic**  
File: `param_tune.py`  
Issue: No fixed random seed for backtests, making results non‑reproducible.  
Fix plan: Add `--seed` and set numpy/random seeds.  
Improve: Store tuning runs and top‑N results to a report file.

12. **Low — Race Risk in `update_latest_close`**  
File: `src/exchange/market_data.py`  
Issue: `update_latest_close` writes buffers without acquiring the per‑pair lock, risking race with `update_bar`.  
Fix plan: Use the same lock or an atomic update method in `MarketDataCache`.  
Improve: Route all data updates through a single producer queue per pair.

## Notes
If you want, I can fix the Critical/High items immediately and follow with a focused test plan. For the multi-tenant architecture, the largest win is to isolate per-tenant state rather than only scoping reads.
