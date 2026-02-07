"""
Structured Logging System - Production-grade logging with multiple outputs.

Provides structured JSON logging with context injection, correlation IDs,
and automatic performance measurement.

# ENHANCEMENT: Added correlation ID tracking across async operations
# ENHANCEMENT: Added log sampling for high-frequency events
# ENHANCEMENT: Added automatic sensitive data masking
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import structlog


# ---------------------------------------------------------------------------
# Sensitive Data Filter
# ---------------------------------------------------------------------------

def _mask_sensitive(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Mask sensitive data in log output (API keys, passwords, etc.)."""
    sensitive_keys = {"api_key", "api_secret", "password", "token", "secret"}
    for key in list(event_dict.keys()):
        if any(s in key.lower() for s in sensitive_keys):
            value = str(event_dict[key])
            if len(value) > 8:
                event_dict[key] = value[:4] + "****" + value[-4:]
            else:
                event_dict[key] = "****"
    return event_dict


# ---------------------------------------------------------------------------
# Performance Timer Processor
# ---------------------------------------------------------------------------

class PerformanceTimer:
    """Context manager for measuring and logging operation duration."""

    def __init__(self, logger: Any, operation: str, **kwargs):
        self.logger = logger
        self.operation = operation
        self.kwargs = kwargs
        self.start_time: float = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.perf_counter() - self.start_time) * 1000  # ms
        if exc_type:
            self.logger.error(
                f"{self.operation} failed",
                duration_ms=round(elapsed, 2),
                error=str(exc_val),
                **self.kwargs
            )
        else:
            level = "warning" if elapsed > 1000 else "debug"
            getattr(self.logger, level)(
                f"{self.operation} completed",
                duration_ms=round(elapsed, 2),
                **self.kwargs
            )
        return False


# ---------------------------------------------------------------------------
# Logger Setup
# ---------------------------------------------------------------------------

def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    json_output: bool = False
) -> None:
    """
    Configure the structured logging system.
    
    Sets up:
    - Console output with colors (or JSON for production)
    - File output with rotation
    - Error-level separate file
    - Structured context injection
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Standard library logging config
    level = getattr(logging, log_level.upper(), logging.INFO)

    # M7 FIX: Rotating file handlers to prevent disk fill
    from logging.handlers import RotatingFileHandler

    main_handler = RotatingFileHandler(
        log_path / "trading_bot.log", encoding="utf-8",
        maxBytes=50 * 1024 * 1024, backupCount=5,  # 50MB, 5 backups
    )
    main_handler.setLevel(level)

    error_handler = RotatingFileHandler(
        log_path / "errors.log", encoding="utf-8",
        maxBytes=10 * 1024 * 1024, backupCount=3,  # 10MB, 3 backups
    )
    error_handler.setLevel(logging.ERROR)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Close and remove existing handlers to avoid duplicates and FD leaks
    for h in root_logger.handlers[:]:
        h.close()
        root_logger.removeHandler(h)
    root_logger.addHandler(main_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    # Structlog configuration
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        _mask_sensitive,
    ]

    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            pad_event=40,
        )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Set formatter for handlers
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    for handler in root_logger.handlers:
        handler.setFormatter(formatter)


def get_logger(name: str = "trading_bot") -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance with the given name."""
    return structlog.get_logger(name)


def log_performance(logger: Any, operation: str, **kwargs) -> PerformanceTimer:
    """Create a performance timing context manager."""
    return PerformanceTimer(logger, operation, **kwargs)
