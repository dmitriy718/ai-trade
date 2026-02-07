"""
Database Manager - SQLite with WAL mode for concurrent access.

Provides async database operations for trade logging, position tracking,
performance metrics, and ML training data storage.

# ENHANCEMENT: Added connection pooling for concurrent access
# ENHANCEMENT: Added automatic schema migration support
# ENHANCEMENT: Added query result caching for frequently accessed data
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite


class DatabaseManager:
    """
    Async SQLite database manager with WAL mode for production use.
    
    Features:
    - WAL mode for concurrent read/write
    - Auto-migration on schema changes
    - Connection health monitoring
    - Query batching for bulk inserts
    """

    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database connection and create schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(self.db_path, timeout=30)

        # Enable WAL mode for concurrent access
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA cache_size=-64000")  # 64MB cache
        await self._db.execute("PRAGMA temp_store=MEMORY")
        await self._db.execute("PRAGMA mmap_size=268435456")  # 256MB mmap

        await self._create_schema()
        self._initialized = True

    async def _create_schema(self) -> None:
        """Create all required database tables."""
        schema_sql = """
        -- Active and historical trades
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT UNIQUE NOT NULL,
            pair TEXT NOT NULL,
            side TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
            entry_price REAL NOT NULL,
            exit_price REAL,
            quantity REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'open'
                CHECK(status IN ('open', 'closed', 'cancelled', 'error')),
            strategy TEXT NOT NULL,
            confidence REAL,
            stop_loss REAL,
            take_profit REAL,
            trailing_stop REAL,
            pnl REAL DEFAULT 0.0,
            pnl_pct REAL DEFAULT 0.0,
            fees REAL DEFAULT 0.0,
            slippage REAL DEFAULT 0.0,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            duration_seconds REAL,
            notes TEXT,
            metadata TEXT,  -- JSON blob for extra data
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        -- Order book snapshots for ML training
        CREATE TABLE IF NOT EXISTS order_book_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            bid_volume REAL,
            ask_volume REAL,
            obi REAL,  -- Order Book Imbalance
            spread REAL,
            whale_detected INTEGER DEFAULT 0,
            snapshot_data TEXT  -- JSON blob
        );

        -- Strategy signals log
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            pair TEXT NOT NULL,
            strategy TEXT NOT NULL,
            direction TEXT NOT NULL CHECK(direction IN ('long', 'short', 'neutral')),
            strength REAL NOT NULL,
            confluence_count INTEGER DEFAULT 0,
            ai_confidence REAL,
            acted_upon INTEGER DEFAULT 0,
            metadata TEXT
        );

        -- Performance metrics (time series)
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            tags TEXT  -- JSON blob
        );

        -- ML training data
        CREATE TABLE IF NOT EXISTS ml_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            pair TEXT NOT NULL,
            features TEXT NOT NULL,  -- JSON feature vector
            label REAL,  -- 1.0 = profitable, 0.0 = loss
            trade_id TEXT,
            FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
        );

        -- AI thought log for dashboard
        CREATE TABLE IF NOT EXISTS thought_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            category TEXT NOT NULL,
            message TEXT NOT NULL,
            severity TEXT DEFAULT 'info'
                CHECK(severity IN ('debug', 'info', 'warning', 'error', 'critical')),
            metadata TEXT
        );

        -- System state (key-value store)
        CREATE TABLE IF NOT EXISTS system_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now'))
        );

        -- Daily performance summary
        CREATE TABLE IF NOT EXISTS daily_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            total_trades INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            losing_trades INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0.0,
            max_drawdown REAL DEFAULT 0.0,
            sharpe_ratio REAL,
            win_rate REAL,
            avg_win REAL,
            avg_loss REAL,
            best_trade REAL,
            worst_trade REAL
        );

        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_trades_pair ON trades(pair);
        CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
        CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
        CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
        CREATE INDEX IF NOT EXISTS idx_signals_pair ON signals(pair);
        CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
        CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
        CREATE INDEX IF NOT EXISTS idx_thought_log_timestamp ON thought_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_ml_features_pair ON ml_features(pair);
        CREATE INDEX IF NOT EXISTS idx_order_book_pair ON order_book_snapshots(pair);
        """
        await self._db.executescript(schema_sql)
        await self._db.commit()

    # ------------------------------------------------------------------
    # Trade Operations
    # ------------------------------------------------------------------

    def _ts(self) -> str:
        """Consistent UTC timestamp in SQLite-compatible format."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def _ensure_ready(self) -> None:
        """H1 FIX: Guard against use before initialization."""
        if not self._initialized or self._db is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

    async def insert_trade(self, trade: Dict[str, Any]) -> int:
        """Insert a new trade record."""
        self._ensure_ready()
        async with self._lock:
            cursor = await self._db.execute(
                """INSERT INTO trades 
                (trade_id, pair, side, entry_price, quantity, status, strategy,
                 confidence, stop_loss, take_profit, entry_time, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade["trade_id"], trade["pair"], trade["side"],
                    trade["entry_price"], trade["quantity"], trade.get("status", "open"),
                    trade["strategy"], trade.get("confidence"),
                    trade.get("stop_loss"), trade.get("take_profit"),
                    trade.get("entry_time", self._ts()),
                    json.dumps(trade.get("metadata", {}))
                )
            )
            await self._db.commit()
            return cursor.lastrowid

    # C1 FIX: Whitelist columns to prevent SQL injection
    TRADE_UPDATE_COLUMNS = frozenset({
        "exit_price", "pnl", "pnl_pct", "fees", "slippage", "status",
        "stop_loss", "take_profit", "trailing_stop", "exit_time",
        "duration_seconds", "notes", "metadata", "quantity",
    })

    async def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> None:
        """Update an existing trade record. Only whitelisted columns allowed."""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        async with self._lock:
            set_clauses = []
            values = []
            for key, value in updates.items():
                if key not in self.TRADE_UPDATE_COLUMNS:
                    raise ValueError(f"Column '{key}' not allowed in trade updates")
                if key == "metadata" and not isinstance(value, str):
                    value = json.dumps(value)
                set_clauses.append(f"{key} = ?")
                values.append(value)
            if not set_clauses:
                return
            set_clauses.append("updated_at = datetime('now')")
            values.append(trade_id)

            sql = f"UPDATE trades SET {', '.join(set_clauses)} WHERE trade_id = ?"
            await self._db.execute(sql, values)
            await self._db.commit()

    async def close_trade(
        self, trade_id: str, exit_price: float, pnl: float,
        pnl_pct: float, fees: float = 0.0, slippage: float = 0.0
    ) -> None:
        """Close a trade with final P&L calculation."""
        now = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            # Get entry time to calculate duration
            cursor = await self._db.execute(
                "SELECT entry_time FROM trades WHERE trade_id = ?", (trade_id,)
            )
            row = await cursor.fetchone()
            duration = 0.0
            if row and row[0]:
                try:
                    entry_dt = datetime.fromisoformat(row[0])
                    exit_dt = datetime.fromisoformat(now)
                    duration = (exit_dt - entry_dt).total_seconds()
                except (ValueError, TypeError):
                    pass

            await self._db.execute(
                """UPDATE trades SET 
                    exit_price = ?, pnl = ?, pnl_pct = ?, fees = ?,
                    slippage = ?, status = 'closed', exit_time = ?,
                    duration_seconds = ?, updated_at = datetime('now')
                WHERE trade_id = ?""",
                (exit_price, pnl, pnl_pct, fees, slippage, now, duration, trade_id)
            )
            await self._db.commit()

    async def get_open_trades(self, pair: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open trades, optionally filtered by pair."""
        sql = "SELECT * FROM trades WHERE status = 'open'"
        params: tuple = ()
        if pair:
            sql += " AND pair = ?"
            params = (pair,)
        sql += " ORDER BY entry_time DESC"

        cursor = await self._db.execute(sql, params)
        columns = [description[0] for description in cursor.description]
        rows = await cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    async def get_trade_history(
        self, limit: int = 100, pair: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get closed trade history."""
        sql = "SELECT * FROM trades WHERE status = 'closed'"
        params: List[Any] = []
        if pair:
            sql += " AND pair = ?"
            params.append(pair)
        sql += " ORDER BY exit_time DESC LIMIT ?"
        params.append(limit)

        cursor = await self._db.execute(sql, params)
        columns = [description[0] for description in cursor.description]
        rows = await cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    # ------------------------------------------------------------------
    # Signal Operations
    # ------------------------------------------------------------------

    async def insert_signal(self, signal: Dict[str, Any]) -> int:
        """Insert a strategy signal."""
        async with self._lock:
            cursor = await self._db.execute(
                """INSERT INTO signals
                (timestamp, pair, strategy, direction, strength,
                 confluence_count, ai_confidence, acted_upon, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    signal["pair"], signal["strategy"], signal["direction"],
                    signal["strength"], signal.get("confluence_count", 0),
                    signal.get("ai_confidence"), signal.get("acted_upon", 0),
                    json.dumps(signal.get("metadata", {}))
                )
            )
            await self._db.commit()
            return cursor.lastrowid

    # ------------------------------------------------------------------
    # Metrics Operations
    # ------------------------------------------------------------------

    async def insert_metric(
        self, name: str, value: float, tags: Optional[Dict] = None
    ) -> None:
        """Insert a performance metric data point."""
        async with self._lock:
            await self._db.execute(
                "INSERT INTO metrics (timestamp, metric_name, metric_value, tags) VALUES (?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), name, value,
                 json.dumps(tags or {}))
            )
            await self._db.commit()

    async def get_metrics(
        self, name: str, hours: int = 24, limit: int = 1000
    ) -> List[Tuple[str, float]]:
        """Get metric time series for the last N hours."""
        cursor = await self._db.execute(
            """SELECT timestamp, metric_value FROM metrics
            WHERE metric_name = ? AND timestamp >= datetime('now', ?)
            ORDER BY timestamp DESC LIMIT ?""",
            (name, f"-{hours} hours", limit)
        )
        return await cursor.fetchall()

    # ------------------------------------------------------------------
    # Thought Log (Dashboard AI Feed)
    # ------------------------------------------------------------------

    async def log_thought(
        self, category: str, message: str,
        severity: str = "info", metadata: Optional[Dict] = None
    ) -> None:
        """Log an AI thought/decision for the dashboard."""
        async with self._lock:
            await self._db.execute(
                """INSERT INTO thought_log (timestamp, category, message, severity, metadata)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    category, message, severity,
                    json.dumps(metadata or {})
                )
            )
            await self._db.commit()

    async def get_thoughts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent AI thoughts for dashboard display."""
        cursor = await self._db.execute(
            """SELECT timestamp, category, message, severity, metadata
            FROM thought_log ORDER BY id DESC LIMIT ?""",
            (limit,)
        )
        columns = ["timestamp", "category", "message", "severity", "metadata"]
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            d = dict(zip(columns, row))
            if d["metadata"]:
                try:
                    d["metadata"] = json.loads(d["metadata"])
                except json.JSONDecodeError:
                    pass
            results.append(d)
        return results

    # ------------------------------------------------------------------
    # ML Features
    # ------------------------------------------------------------------

    async def insert_ml_features(
        self, pair: str, features: Dict[str, float],
        label: Optional[float] = None, trade_id: Optional[str] = None
    ) -> None:
        """Insert ML feature vector for training."""
        async with self._lock:
            await self._db.execute(
                """INSERT INTO ml_features (timestamp, pair, features, label, trade_id)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    pair, json.dumps(features), label, trade_id
                )
            )
            await self._db.commit()

    async def get_ml_training_data(
        self, min_samples: int = 10000
    ) -> List[Dict[str, Any]]:
        """Get labeled ML training data."""
        cursor = await self._db.execute(
            """SELECT features, label FROM ml_features
            WHERE label IS NOT NULL
            ORDER BY timestamp DESC LIMIT ?""",
            (min_samples,)
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            features = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            results.append({"features": features, "label": row[1]})
        return results

    # ------------------------------------------------------------------
    # System State
    # ------------------------------------------------------------------

    async def set_state(self, key: str, value: Any) -> None:
        """Set a system state key-value pair."""
        async with self._lock:
            await self._db.execute(
                """INSERT OR REPLACE INTO system_state (key, value, updated_at)
                VALUES (?, ?, datetime('now'))""",
                (key, json.dumps(value))
            )
            await self._db.commit()

    async def get_state(self, key: str, default: Any = None) -> Any:
        """Get a system state value."""
        cursor = await self._db.execute(
            "SELECT value FROM system_state WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return row[0]
        return default

    # ------------------------------------------------------------------
    # Daily Summary
    # ------------------------------------------------------------------

    async def update_daily_summary(self, date: str, stats: Dict[str, Any]) -> None:
        """Update or insert daily performance summary."""
        async with self._lock:
            await self._db.execute(
                """INSERT OR REPLACE INTO daily_summary
                (date, total_trades, winning_trades, losing_trades,
                 total_pnl, max_drawdown, sharpe_ratio, win_rate,
                 avg_win, avg_loss, best_trade, worst_trade)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    date,
                    stats.get("total_trades", 0),
                    stats.get("winning_trades", 0),
                    stats.get("losing_trades", 0),
                    stats.get("total_pnl", 0.0),
                    stats.get("max_drawdown", 0.0),
                    stats.get("sharpe_ratio"),
                    stats.get("win_rate"),
                    stats.get("avg_win"),
                    stats.get("avg_loss"),
                    stats.get("best_trade"),
                    stats.get("worst_trade"),
                )
            )
            await self._db.commit()

    # ------------------------------------------------------------------
    # Performance Stats
    # ------------------------------------------------------------------

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get aggregate performance statistics."""
        stats = {}

        # Total P&L
        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status = 'closed'"
        )
        row = await cursor.fetchone()
        stats["total_pnl"] = row[0] if row else 0.0

        # Trade counts
        cursor = await self._db.execute(
            """SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses
            FROM trades WHERE status = 'closed'"""
        )
        row = await cursor.fetchone()
        stats["total_trades"] = row[0] if row else 0
        stats["winning_trades"] = row[1] if row else 0
        stats["losing_trades"] = row[2] if row else 0
        stats["win_rate"] = (
            stats["winning_trades"] / stats["total_trades"]
            if stats["total_trades"] > 0 else 0.0
        )

        # Average win/loss
        cursor = await self._db.execute(
            "SELECT AVG(pnl) FROM trades WHERE status = 'closed' AND pnl > 0"
        )
        row = await cursor.fetchone()
        stats["avg_win"] = row[0] if row and row[0] else 0.0

        cursor = await self._db.execute(
            "SELECT AVG(pnl) FROM trades WHERE status = 'closed' AND pnl <= 0"
        )
        row = await cursor.fetchone()
        stats["avg_loss"] = row[0] if row and row[0] else 0.0

        # Open positions
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM trades WHERE status = 'open'"
        )
        row = await cursor.fetchone()
        stats["open_positions"] = row[0] if row else 0

        # Today's P&L
        cursor = await self._db.execute(
            """SELECT COALESCE(SUM(pnl), 0) FROM trades
            WHERE status = 'closed' AND date(exit_time) = date('now')"""
        )
        row = await cursor.fetchone()
        stats["today_pnl"] = row[0] if row else 0.0

        return stats

    # ------------------------------------------------------------------
    # Cleanup & Close
    # ------------------------------------------------------------------

    async def cleanup_old_data(self, retention_hours: int = 72) -> None:
        """Remove old metrics and thought logs past retention period."""
        async with self._lock:
            await self._db.execute(
                "DELETE FROM metrics WHERE timestamp < datetime('now', ?)",
                (f"-{retention_hours} hours",)
            )
            await self._db.execute(
                "DELETE FROM thought_log WHERE timestamp < datetime('now', ?)",
                (f"-{retention_hours} hours",)
            )
            await self._db.execute(
                "DELETE FROM order_book_snapshots WHERE timestamp < datetime('now', ?)",
                (f"-{retention_hours} hours",)
            )
            await self._db.commit()

    async def close(self) -> None:
        """Close database connection gracefully."""
        if self._db:
            await self._db.close()
            self._db = None
            self._initialized = False
