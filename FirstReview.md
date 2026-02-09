# First Comprehensive Code Review

**Date:** 2026-02-06
**Reviewer:** AI Code Review Agent
**Scope:** Full codebase (25 Python files, ~9200 LOC)

---

## CRITICAL (7 issues)

| # | File | Issue | Description |
|---|------|-------|-------------|
| C1 | `database.py` | SQL injection in `update_trade` | Column names from dict keys are interpolated into SQL without whitelist validation |
| C2 | `market_data.py` | Volume/VWAP columns swapped | `COL_VOLUME=5` but data stores vwap at idx 5 and volume at idx 6 — ALL volume-based signals across ALL 5 strategies are broken |
| C3 | `risk_manager.py` | `abs()` on daily PnL check | `abs(daily_pnl)` blocks trading when bot is PROFITABLE (5%+ daily gain stops trading) |
| C4 | `server.py` | No auth on control endpoints | `/close_all`, `/pause`, `/resume` have zero authentication — anyone on the network can close all positions |
| C5 | `telegram.py` | No chat_id verification | Any Telegram user can send `/kill` or `/close_all` |
| C6 | `executor.py` | Live close failure leaves ghost positions | If exit order fails in live mode, position stays open in DB/risk manager — retries at bad prices |
| C7 | `executor.py` | Entry fees never deducted from PnL | Only exit fees subtracted; every trade P&L is systematically overstated |

## HIGH (16 issues)

| # | File | Issue | Description |
|---|------|-------|-------------|
| H1 | `database.py` | No init guard on methods | Any call before `initialize()` crashes with NoneType error |
| H2 | `kraken_rest.py` | Auth headers dropped on GET | Authenticated GET requests don't pass `headers` to httpx |
| H3 | `kraken_rest.py` | Semaphore deadlock on retry | Recursive retries hold semaphore during backoff sleep; can deadlock under load |
| H4 | `reversal.py` | Candlestick patterns non-functional | Falls back to `closes` as `opens`; hammer/engulfing detection always fails |
| H5 | `engine.py` | Duplicate lifecycle management | `main.py` and `engine.start()` both init/warmup; conflicting signal handlers |
| H6 | `risk_manager.py` | Default StopLossState invalid for shorts | `trailing_low=0.0` means trailing stop never activates for sell positions |
| H7 | `order_book.py` | Division by zero on depth_ratio | `bid_volume / ask_volume` when asks=0 produces inf; breaks JSON serialization |
| H8 | `executor.py` | Zero fill price accepted as valid | `_wait_for_fill` returns 0.0 instead of None for missing price |
| H9 | `trainer.py` | Data leakage in normalization | Stats computed on full dataset before train/val split; inflates accuracy |
| H10 | `trainer.py` | `model.fit()` blocks event loop | Synchronous training freezes entire bot (positions unmanaged, WS dead) |
| H11 | `trainer.py` | Inconsistent feature ordering | Per-sample `sorted(keys)` produces different column meanings across samples |
| H12 | `server.py` | CORS allows all origins + credentials | Security anti-pattern; any origin can make authenticated requests |
| H13 | `server.py` | N+1 DB queries per WebSocket client | Each client triggers independent DB queries every second; doesn't scale |
| H14 | `backtester.py` | O(n^2) memory from array slicing | `closes[:i+1]` copies grow quadratically; catastrophic for large datasets |
| H15 | `stress_test.py` | `subprocess.run()` blocks event loop | Synchronous subprocess calls block async loop for up to 60s |
| H16 | `risk_manager.py` | Negative bankroll not explicitly blocked | Fragile chain of checks accidentally works; no explicit guard |

## MEDIUM (30 issues)

| # | File | Issue | Description |
|---|------|-------|-------------|
| M1 | `config.py` | Singleton not thread-safe | TOCTOU race in `__new__` |
| M2 | `config.py` | Silent env var conversion failures | Bad `MAX_RISK_PER_TRADE` silently ignored |
| M3 | `config.py` | Env maps reference non-existent fields | `initial_bankroll` and `db_path` don't exist in models |
| M4 | `database.py` | Timestamp format mismatch | Python ISO vs SQLite format; metrics queries return wrong data |
| M5 | `database.py` | No connection timeout | `aiosqlite.connect()` can hang on locked DB |
| M6 | `database.py` | Reads lack locking | Potential inconsistent reads mixed with locked writes |
| M7 | `logger.py` | No log rotation | `FileHandler` instead of `RotatingFileHandler`; disk fills up |
| M8 | `vault.py` | Checksum written but never verified | `_load()` ignores the checksum field |
| M9 | `vault.py` | Salt not cached; re-read from file on every save | FileNotFoundError if file deleted between init and save |
| M10 | `vault.py` | Vault file has permissive permissions | Default 0o644; should be 0o600 |
| M11 | `vault.py` | Docstring says AES-256 but Fernet uses AES-128 | Misleading security claims |
| M12 | `kraken_rest.py` | Dedup set eviction is random | `set` has no order; evicts arbitrary IDs, not oldest |
| M13 | `kraken_rest.py` | `close()` doesn't null client ref | Subsequent calls use closed client |
| M14 | `kraken_ws.py` | `unsubscribe()` lacks error handling | Subscription removed from map even if send fails |
| M15 | `kraken_ws.py` | Max reconnect reached with no notification | Bot silently stops receiving data |
| M16 | `market_data.py` | Warmup counts failed bars in size | Zero-filled rows corrupt indicators |
| M17 | `market_data.py` | Float timestamp compared with `==` | Floating-point equality fails; duplicate bars appended |
| M18 | `market_data.py` | Accessors return mutable array views | External code can corrupt internal cache |
| M19 | `market_data.py` | 20% outlier rejection too aggressive for crypto | Flash crashes silently rejected |
| M20 | `base.py` | `_sanitize_for_json` doesn't handle NaN/Inf | JSON serialization crashes |
| M21 | `momentum.py` | "Consecutive" candle counting is actually total | Inflates momentum signal |
| M22 | `breakout.py` | Uses previous close as current open | Inaccurate candle body analysis on gap opens |
| M23 | `indicators.py` | BB uses sample stddev instead of population | Bands 2.6% wider than standard |
| M24 | `indicators.py` | `volume_ratio` returns raw volumes for first N bars | Enormous false volume bursts |
| M25 | `indicators.py` | `bb_position` clips to [0,1] | Extreme signals indistinguishable from band touches |
| M26 | `confluence.py` | OBI parsing has no error handling | Malformed order book data crashes analysis |
| M27 | `predictor.py` | `_cache_key` throws on non-serializable values | TypeError on numpy arrays in state |
| M28 | `executor.py` | PnL% calculated before fee deduction | Stored percentage doesn't reflect net PnL |
| M29 | `executor.py` | Hardcoded fee rate in two places | 0.0026 duplicated; no maker/taker distinction |
| M30 | `server.py` | Internal errors leaked to WS clients | Exception messages expose implementation details |

## LOW (20 issues)

| # | File | Issue | Description |
|---|------|-------|-------------|
| L1 | `config.py` | Strategy weights not validated to sum to 1.0 | |
| L2 | `engine.py` | `asyncio.get_event_loop()` deprecated | Use `get_running_loop()` |
| L3 | `engine.py` | `_handle_trade` is a no-op | Trade stream events silently dropped |
| L4 | `main.py` | `preflight_checks` always returns True | Guard is dead code |
| L5 | `main.py` | `import traceback` inside except block | Should be at module level |
| L6 | `kraken_rest.py` | `if since:` skips `since=0` | Should be `if since is not None` |
| L7 | `kraken_rest.py` | Data dict mutated in-place | Caller's dict gets unexpected `nonce` key |
| L8 | `kraken_ws.py` | `_message_queue` is dead code | Created but never consumed |
| L9 | `kraken_ws.py` | `_latency_samples` never populated | `latency_ms` always returns 0.0 |
| L10 | `kraken_ws.py` | `_tasks` list never populated | Cancellation loop is dead code |
| L11 | `indicators.py` | SMA has redundant return slice | `result[:len(data)]` is same length |
| L12 | `indicators.py` | ADX returns 0.0 instead of NaN for unconverged | Indistinguishable from "no trend" |
| L13 | `reversal.py` | Confirmation candle off-by-one | 3 candles = 2 transitions |
| L14 | `confluence.py` | Accesses private `_trade_count` | Should use public property |
| L15 | `predictor.py` | MD5 for cache keys | Code smell; use faster hash |
| L16 | `backtester.py` | Equal votes defaults to SHORT | Asymmetric bias |
| L17 | `backtester.py` | Sharpe annualization assumes minute bars | Overstated for infrequent traders |
| L18 | `server.py` | `_cache` and `_cache_ttl` are dead code | Defined but never used |
| L19 | `telegram.py` | `_message_queue` is dead code | |
| L20 | `stress_test.py` | Container name hardcoded | Should be configurable |

---

**Total: 73 issues (7 Critical, 16 High, 30 Medium, 20 Low)**
