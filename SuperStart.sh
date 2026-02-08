#!/bin/bash

# ==============================================================================
# AI TRADING BOT - SUPER START SCRIPT (Raspberry Pi 5 Optimized)
# ==============================================================================

# Text formatting
BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}${BOLD}"
echo "    ╔══════════════════════════════════════════════╗"
echo "    ║        AI CRYPTO TRADING BOT v2.0.0          ║"
echo "    ║          Raspberry Pi 5 Deployer             ║"
echo "    ╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# 1. PREREQUISITES CHECK
echo -e "${BOLD}[1/5] Checking prerequisites...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3 is not installed. Please install it with: sudo apt install python3 python3-venv${NC}"
    exit 1
fi

# 2. CONFIGURATION
echo -e "
${BOLD}[2/5] Configuring environment...${NC}"

# Prompt for Kraken Keys
read -p "Enter Kraken API Key: " KRAKEN_KEY
read -p "Enter Kraken API Secret: " KRAKEN_SECRET

# Prompt for Trading Mode
echo -e "
${YELLOW}Trading Mode:${NC}"
echo "1) Paper Mode (Simulated, safe)"
echo "2) Live Mode (REAL MONEY)"
read -p "Select mode [1/2]: " MODE_CHOICE

if [ "$MODE_CHOICE" == "2" ]; then
    TRADING_MODE="live"
    echo -e "${RED}${BOLD}WARNING: LIVE MODE SELECTED. Real funds will be used.${NC}"
else
    TRADING_MODE="paper"
    echo -e "${GREEN}Paper Mode selected.${NC}"
fi

# Prompt for Bankroll
read -p "Enter initial bankroll (USD) [Default 10000]: " BANKROLL
BANKROLL=${BANKROLL:-10000}

# Create .env file
cat <<EOF > .env
# Created by SuperStart.sh
KRAKEN_API_KEY=$KRAKEN_KEY
KRAKEN_API_SECRET=$KRAKEN_SECRET
TRADING_MODE=$TRADING_MODE
INITIAL_BANKROLL=$BANKROLL
MAX_RISK_PER_TRADE=0.02
MAX_DAILY_LOSS=0.05
MAX_POSITION_USD=500.0
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
DASHBOARD_SECRET_KEY=$(openssl rand -hex 16)
VAULT_PASSWORD=$(openssl rand -hex 12)
LOG_LEVEL=INFO
MODEL_RETRAIN_INTERVAL_HOURS=168
MIN_TRAINING_SAMPLES=10000
DB_PATH=data/trading.db
EOF

echo -e "${GREEN}.env configuration created successfully.${NC}"

# 3. BUILD
echo -e "
${BOLD}[3/5] Building environment and dependencies...${NC}"

# Create directories
mkdir -p data logs models config

# Setup Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install
echo "Installing dependencies (this may take a few minutes on Pi)..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. START BOT
echo -e "
${BOLD}[4/5] Starting the bot...${NC}"

# Kill any existing process
pkill -f "python main.py" &> /dev/null

# Start in background
nohup python main.py > logs/bot_output.log 2>&1 &

echo -e "${GREEN}Bot is now running in the background! (PID: $!)${NC}"
echo "Logs are available in logs/trading_bot.log"

# 5. LAUNCH DASHBOARD
echo -e "
${BOLD}[5/5] Launching dashboard...${NC}"
echo "Waiting for server to initialize..."
sleep 5

# Detect OS and open browser
if command -v chromium-browser &> /dev/null; then
    chromium-browser http://localhost:8080 &
elif command -v firefox &> /dev/null; then
    firefox http://localhost:8080 &
elif [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:8080
else
    echo -e "${YELLOW}Could not detect browser. Please visit: http://localhost:8080${NC}"
fi

echo -e "
${CYAN}${BOLD}DEPLOYMENT COMPLETE!${NC}"
echo -e "Your bot is now hunting for confluence setups."
echo -e "Type 'tail -f logs/trading_bot.log' to watch live activity."
