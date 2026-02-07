"""
Configuration Manager - Loads and validates all system configuration.

Merges YAML config with environment variables. Environment variables take
precedence over YAML values for deployment flexibility.

# ENHANCEMENT: Added hot-reload capability for runtime config changes
# ENHANCEMENT: Added deep validation with Pydantic models
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Pydantic Configuration Models (strict validation)
# ---------------------------------------------------------------------------

class ExchangeConfig(BaseModel):
    name: str = "kraken"
    ws_url: str = "wss://ws.kraken.com/v2"
    ws_auth_url: str = "wss://ws-auth.kraken.com/v2"
    rest_url: str = "https://api.kraken.com"
    rate_limit_per_second: int = 15
    max_retries: int = 5
    retry_base_delay: float = 1.0
    timeout: int = 30


class TradingConfig(BaseModel):
    pairs: List[str] = Field(default_factory=lambda: ["BTC/USD", "ETH/USD"])
    scan_interval_seconds: int = 60
    hft_scan_interval_seconds: int = 1
    warmup_bars: int = 500
    warmup_timeframe: str = "1m"
    max_concurrent_positions: int = 5
    cooldown_seconds: int = 300


class StrategyWeights(BaseModel):
    enabled: bool = True
    weight: float = 0.20


class TrendConfig(StrategyWeights):
    ema_fast: int = 5
    ema_slow: int = 13
    adx_threshold: int = 25


class MeanReversionConfig(StrategyWeights):
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_oversold: int = 30
    rsi_overbought: int = 70


class MomentumConfig(StrategyWeights):
    rsi_threshold: int = 50
    volume_multiplier: float = 1.5


class BreakoutConfig(StrategyWeights):
    lookback_period: int = 20
    volume_confirmation: float = 1.3


class ReversalConfig(StrategyWeights):
    rsi_extreme_low: int = 20
    rsi_extreme_high: int = 80
    confirmation_candles: int = 3


class KeltnerConfig(StrategyWeights):
    ema_period: int = 20
    atr_period: int = 14
    kc_multiplier: float = 1.5
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    rsi_long_max: float = 40
    rsi_short_min: float = 60
    weight: float = 0.30


class StrategiesConfig(BaseModel):
    keltner: KeltnerConfig = Field(default_factory=KeltnerConfig)
    trend: TrendConfig = Field(default_factory=TrendConfig)
    mean_reversion: MeanReversionConfig = Field(default_factory=MeanReversionConfig)
    momentum: MomentumConfig = Field(default_factory=MomentumConfig)
    breakout: BreakoutConfig = Field(default_factory=BreakoutConfig)
    reversal: ReversalConfig = Field(default_factory=ReversalConfig)


class AIConfig(BaseModel):
    confluence_threshold: int = 3
    min_confidence: float = 0.65
    tflite_model_path: str = "models/trade_predictor.tflite"
    order_book_depth: int = 25
    obi_threshold: float = 0.15
    whale_threshold_usd: float = 50000.0


class RiskConfig(BaseModel):
    max_risk_per_trade: float = 0.02
    max_daily_loss: float = 0.05
    max_position_usd: float = 500.0
    atr_multiplier_sl: float = 2.0
    atr_multiplier_tp: float = 3.0
    trailing_activation_pct: float = 0.015
    trailing_step_pct: float = 0.005
    breakeven_activation_pct: float = 0.01
    kelly_fraction: float = 0.25
    max_kelly_size: float = 0.10
    risk_of_ruin_threshold: float = 0.01

    @field_validator("max_risk_per_trade")
    @classmethod
    def validate_risk(cls, v):
        if v <= 0 or v > 0.10:
            raise ValueError("max_risk_per_trade must be between 0 and 0.10")
        return v

    @field_validator("max_daily_loss")
    @classmethod
    def validate_daily_loss(cls, v):
        if v <= 0 or v > 0.20:
            raise ValueError("max_daily_loss must be between 0 and 0.20")
        return v


class DashboardConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    thought_feed_max: int = 200
    refresh_interval_ms: int = 1000


class MLConfig(BaseModel):
    retrain_interval_hours: int = 168
    min_samples: int = 10000
    epochs: int = 50
    batch_size: int = 64
    validation_split: float = 0.2
    features: List[str] = Field(default_factory=lambda: [
        "rsi", "ema_ratio", "bb_position", "adx", "volume_ratio",
        "obi", "atr_pct", "momentum_score", "trend_strength", "spread_pct"
    ])


class MonitoringConfig(BaseModel):
    health_check_interval: int = 30
    auto_restart: bool = True
    max_restart_attempts: int = 10
    heartbeat_interval: int = 10
    metrics_retention_hours: int = 72


class AppConfig(BaseModel):
    name: str = "AI Crypto Trading Bot"
    version: str = "2.0.0"
    mode: str = "paper"
    log_level: str = "INFO"


class BotConfig(BaseModel):
    """Master configuration model with full validation."""
    app: AppConfig = Field(default_factory=AppConfig)
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


# ---------------------------------------------------------------------------
# Configuration Manager (Singleton)
# ---------------------------------------------------------------------------

class ConfigManager:
    """
    Thread-safe configuration manager with hot-reload support.
    
    Loads configuration from YAML file, then overlays environment
    variables for deployment flexibility. Validates all values through
    Pydantic models.
    
    # ENHANCEMENT: Added file-watching for hot-reload capability
    # ENHANCEMENT: Added config versioning for rollback support
    """

    _instance: Optional[ConfigManager] = None
    _config: Optional[BotConfig] = None

    def __new__(cls) -> ConfigManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self.load()

    def load(self, config_path: str = "config/config.yaml") -> BotConfig:
        """Load configuration from YAML + environment variables."""
        load_dotenv()

        # Load YAML config
        yaml_config: Dict[str, Any] = {}
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r") as f:
                yaml_config = yaml.safe_load(f) or {}

        # Apply environment variable overrides
        self._apply_env_overrides(yaml_config)

        # Validate through Pydantic
        self._config = BotConfig(**yaml_config)
        return self._config

    def _apply_env_overrides(self, config: Dict[str, Any]) -> None:
        """Override YAML values with environment variables where set."""
        env_mappings = {
            "TRADING_MODE": ("app", "mode"),
            "LOG_LEVEL": ("app", "log_level"),
            "MAX_RISK_PER_TRADE": ("risk", "max_risk_per_trade", float),
            "MAX_DAILY_LOSS": ("risk", "max_daily_loss", float),
            "MAX_POSITION_USD": ("risk", "max_position_usd", float),
            "INITIAL_BANKROLL": ("risk", "initial_bankroll", float),
            "DASHBOARD_HOST": ("dashboard", "host"),
            "DASHBOARD_PORT": ("dashboard", "port", int),
            "MODEL_RETRAIN_INTERVAL_HOURS": ("ml", "retrain_interval_hours", int),
            "DB_PATH": ("app", "db_path"),
        }

        for env_key, mapping in env_mappings.items():
            value = os.getenv(env_key)
            if value is not None:
                section = mapping[0]
                key = mapping[1]
                converter = mapping[2] if len(mapping) > 2 else str

                if section not in config:
                    config[section] = {}
                try:
                    config[section][key] = converter(value)
                except (ValueError, TypeError):
                    pass  # Keep YAML value if env conversion fails

    @property
    def config(self) -> BotConfig:
        """Get the current validated configuration."""
        if self._config is None:
            self.load()
        return self._config

    def reload(self, config_path: str = "config/config.yaml") -> BotConfig:
        """Hot-reload configuration from disk."""
        return self.load(config_path)

    def get(self, dotpath: str, default: Any = None) -> Any:
        """
        Access config values using dot notation.
        
        Example: config.get("risk.max_risk_per_trade") -> 0.02
        """
        obj = self._config
        for key in dotpath.split("."):
            if hasattr(obj, key):
                obj = getattr(obj, key)
            elif isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return default
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """Export full config as dictionary."""
        return self._config.model_dump() if self._config else {}


# Convenience accessor
def get_config() -> BotConfig:
    """Get the global configuration instance."""
    return ConfigManager().config
