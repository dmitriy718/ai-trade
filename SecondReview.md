# Second Comprehensive Code Review

**Date:** 2026-02-07
**Scope:** Full codebase post-first-review fixes (17 new issues found)

## CRITICAL (2)

| # | File | Issue | Status |
|---|------|-------|--------|
| S1 | `engine.py` | Ticker handler injects fake OHLC bars — corrupts ALL indicators (ATR→0, volume→0, bands→0). Bot stops trading within minutes. | FIXED |
| S2 | `risk_manager.py` | NameError: `kelly_size` and `fixed_risk_size` undefined in error message when position <$10 | FIXED |

## HIGH (4)

| # | File | Issue | Status |
|---|------|-------|--------|
| S3 | `confluence.py` | No staleness check — trades on stale data when WS disconnects | FIXED |
| S4 | `predictor.py` | Heuristic is direction-agnostic — boosts wrong signals (OBI for longs on bearish OBI) | FIXED |
| S5 | `breakout.py` | Ignores available opens data, uses inaccurate `closes[-2]` proxy | FIXED |
| S6 | `executor.py` | Market order fill price may be 0 on Kraken — live mode fills fail silently | FIXED |

## MEDIUM (6)

| # | File | Issue | Status |
|---|------|-------|--------|
| S7 | `engine.py` | Metadata key collision from multiple strategies overwrites RSI/volume non-deterministically | FIXED |
| S8 | `predictor.py` | `atr_pct` feature always defaults to 0.02 — no real signal for AI | FIXED |
| S9 | `engine.py` | Health monitor doesn't restart WS task after max retries exhausted | FIXED |
| S10 | `engine.py` | Shutdown race — DB closed while tasks still running | FIXED |
| S11 | `mean_reversion.py` | Short TP override makes target MORE aggressive instead of conservative | FIXED |
| S12 | `server.py` | Unrealized P&L doesn't deduct estimated exit fees | FIXED |

## LOW (5)

| # | File | Issue | Status |
|---|------|-------|--------|
| S13 | `kraken_rest.py` | `get_trade_history_public` drops `since=0` | FIXED |
| S14 | `engine.py` | Dead `start()` method with deprecated API | FIXED |
| S15 | `market_data.py` | Warmup returns attempted count not actual written count | FIXED |
| S16 | `server.py` | Equity calculation inconsistency between REST and WS | FIXED |
| S17 | `dashboard.js` | Console spam from fetch/WS errors when server is down | FIXED |

---

**Total: 17 issues found, 17 fixed**
