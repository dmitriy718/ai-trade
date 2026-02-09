# AI Crypto Trading Bot v2.0.0

**Multi-strategy, AI-powered crypto trading bot for Kraken exchange with real-time intelligence, self-improving ML pipeline, and production-grade resilience.**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Trading Bot v2.0                       │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│ Market   │ Strategy │    AI    │Execution │   Dashboard    │
│  Data    │  Engine  │ Engine   │  Engine  │    (FastAPI)   │
│          │          │          │          │                │
│ • Kraken │ • Trend  │• TFLite  │• Kelly   │• WebSocket     │
│   WS/REST│ • Mean   │  Predict │  Sizing  │  Live Feed     │
│ • NumPy  │   Rev    │• Order   │• ATR     │• AI Thought    │
│   Cache  │ • Mom    │  Book    │  Stops   │  Feed          │
│ • 500+   │ • Break  │  Analysis│• Trail + │• Cyberpunk     │
│   Bars   │ • Rev    │• Conflu- │  BE logic│  Glass UI      │
│          │          │  ence    │• Risk of │• REST API      │
│          │          │  Detect  │  Ruin    │                │
├──────────┴──────────┴──────────┴──────────┴────────────────┤
│                     Core Infrastructure                     │
│  SQLite WAL • Structured Logging • AES-256 Vault • Config  │
├─────────────────────────────────────────────────────────────┤
│                     ML Pipeline                             │
│  Auto-Retrain • Backtester • Monte Carlo • Feature Eng     │
├─────────────────────────────────────────────────────────────┤
│                   Docker + Resilience                       │
│  ARM64 Optimized • Auto-Restart • Health Checks • Stress   │
└─────────────────────────────────────────────────────────────┘
```

## Features

### Market Scanning Engine
- **Parallel scanning** of 8+ trading pairs every 60 seconds
- **5 strategies** running in parallel: Trend (EMA crossover + ADX), Mean Reversion (Bollinger + RSI), Momentum (RSI + Volume burst), Breakout (S/R + Volume), Reversal (Extreme RSI + Candle patterns)
- **500+ bar** NumPy RAM cache per pair for fast indicator computation

### AI Intelligence Engine
- **Multi-Algorithm Confluence**: 5 strategies scored and weighted simultaneously
- **"Sure Fire" Setup Detector**: Triggers when 3+ strategies align AND Order Book Imbalance confirms
- **TFLite Verification Model**: Neural network predicts trade success probability
- **Order Book Analysis**: Whale detection, liquidity void mapping, spoofing detection

### Execution & Risk Engine
- **Kelly Criterion** position sizing with quarter-Kelly safety factor
- **ATR-based** trailing stop loss with breakeven logic
- **Risk of Ruin** monitoring with automatic position scaling
- **Daily loss limit**: Max 5% bankroll drawdown per day
- **Trade cooldowns** and max concurrent position limits
- **Drawdown-scaled sizing**: Automatically reduces exposure in drawdowns

### Self-Improving ML Pipeline
- **Auto-Retraining**: Weekly TFLite model updates from historical data
- **Backtester**: Shadow mode with same logic as live trading
- **Monte Carlo Simulation**: Confidence intervals for expected performance
- **Feature Engineering**: 10-dimensional normalized feature vectors

### Dashboard
- **Glassmorphism cyberpunk UI** with real-time WebSocket updates
- **AI Thought Feed**: Live stream of bot reasoning and decisions
- **Algo Pulse**: Strategy performance monitor with visual bars
- **Ticker Scanner**: All pairs with sparkline price charts
- **Risk Shield**: Real-time risk metrics with gauge visualizations
- **HUD**: Live P&L, positions, uptime, and system status

### Telegram Command Center
- `/status`, `/pnl`, `/positions`, `/risk` — monitoring
- `/pause`, `/resume` — trade control
- `/close_all`, `/kill` — emergency actions

## Quick Start

### 1. Clone and Configure

```bash
git clone <your-repo-url>
cd aitradercursor2

# Copy environment template
cp .env.example .env

# Edit .env with your Kraken API credentials
nano .env
```

### 2. Run with Docker (Recommended)

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f trading-bot

# Access dashboard
open http://localhost:8080
```

### 3. Run Locally

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the bot
python main.py
```

### 4. Run Stress Tests

```bash
# Quick smoke test (5 minutes)
python stress_test.py --quick

# Full stress test (custom duration)
python stress_test.py --hours 24 --interval 10
```

## Configuration

All configuration is in `config/config.yaml`. Environment variables in `.env` override YAML values.

### Key Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `app.mode` | `paper` | Trading mode: `paper` or `live` |
| `trading.pairs` | 8 pairs | Trading pairs to monitor |
| `trading.scan_interval_seconds` | 60 | Main scan cycle interval |
| `risk.max_risk_per_trade` | 0.02 | Max 2% bankroll per trade |
| `risk.max_daily_loss` | 0.05 | Max 5% daily loss |
| `risk.kelly_fraction` | 0.25 | Quarter-Kelly for safety |
| `ai.confluence_threshold` | 3 | Min strategies to agree |
| `ai.min_confidence` | 0.65 | Min AI confidence to trade |

## Project Structure

```
aitradercursor2/
├── main.py                  # Entry point
├── config/
│   └── config.yaml          # Master configuration
├── src/
│   ├── core/
│   │   ├── config.py        # Configuration manager
│   │   ├── database.py      # SQLite + WAL database
│   │   ├── engine.py        # Main bot orchestrator
│   │   ├── logger.py        # Structured logging
│   │   └── vault.py         # AES-256 encrypted vault
│   ├── exchange/
│   │   ├── kraken_rest.py   # Kraken REST API client
│   │   ├── kraken_ws.py     # Kraken WebSocket client
│   │   └── market_data.py   # NumPy price cache
│   ├── strategies/
│   │   ├── base.py          # Strategy base class
│   │   ├── trend.py         # EMA crossover + ADX
│   │   ├── mean_reversion.py # Bollinger + RSI
│   │   ├── momentum.py      # RSI + Volume burst
│   │   ├── breakout.py      # S/R breakout
│   │   └── reversal.py      # Extreme reversal
│   ├── ai/
│   │   ├── confluence.py    # Multi-strategy detector
│   │   ├── predictor.py     # TFLite prediction model
│   │   └── order_book.py    # Order book analyzer
│   ├── execution/
│   │   ├── executor.py      # Trade execution engine
│   │   └── risk_manager.py  # Risk management system
│   ├── api/
│   │   └── server.py        # FastAPI dashboard server
│   ├── ml/
│   │   ├── trainer.py       # ML model training pipeline
│   │   └── backtester.py    # Historical backtester
│   └── utils/
│       ├── indicators.py    # Technical indicators (NumPy)
│       └── telegram.py      # Telegram command center
├── static/
│   ├── index.html           # Dashboard HTML
│   ├── css/dashboard.css    # Cyberpunk glassmorphism CSS
│   └── js/dashboard.js      # Dashboard client JS
├── stress_test.py           # 72-hour stress tester
├── Dockerfile               # Multi-stage ARM64 build
├── docker-compose.yml       # Production compose
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
└── .gitignore
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/status` | System status |
| GET | `/api/v1/trades` | Trade history |
| GET | `/api/v1/positions` | Open positions |
| GET | `/api/v1/performance` | Performance metrics |
| GET | `/api/v1/strategies` | Strategy stats |
| GET | `/api/v1/risk` | Risk report |
| GET | `/api/v1/thoughts` | AI thought feed |
| GET | `/api/v1/scanner` | Market scanner |
| POST | `/api/v1/control/close_all` | Close all positions |
| POST | `/api/v1/control/pause` | Pause trading |
| POST | `/api/v1/control/resume` | Resume trading |
| WS | `/ws/live` | Real-time data stream |

## Safety Features

- **Paper mode by default** - No real money at risk until explicitly enabled
- **Risk of Ruin monitoring** - Automatic position reduction when RoR exceeds threshold
- **Daily loss limit** - Stops trading at 5% bankroll drawdown per day
- **Kelly Criterion sizing** - Mathematically optimal position sizing with safety factor
- **Trailing stops + breakeven** - Automated profit protection
- **Order deduplication** - Prevents double orders from network issues
- **Graceful shutdown** - Preserves state and open positions on restart

## License

Private. All rights reserved.
