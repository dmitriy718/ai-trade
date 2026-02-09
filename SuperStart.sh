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
DIM='\033[2m'
NC='\033[0m' # No Color

# Timestamp prefix for every message
ts() { echo -n "[$(date +%H:%M:%S)] "; }

# Spinner: run in background; pass PID to stop_spinner to kill it
spinner_pid=""
start_spinner() {
  local chars='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
  while true; do
    for i in $(seq 0 7); do
      echo -ne "\r  ${CYAN}${chars:$i:1}${NC} $1          "
      sleep 0.08
    done
  done
}
stop_spinner() {
  [ -n "$1" ] && kill "$1" 2>/dev/null
  echo -ne "\r  ${GREEN}✓${NC} $2\n"
}

run_with_spinner() {
  local msg="$1"
  shift
  start_spinner "$msg" &
  spinner_pid=$!
  "$@" > /tmp/superstart_out 2>&1
  local ret=$?
  stop_spinner "$spinner_pid" "$msg"
  [ $ret -ne 0 ] && cat /tmp/superstart_out
  return $ret
}

echo -e "${CYAN}${BOLD}"
echo "    ╔══════════════════════════════════════════════╗"
echo "    ║        AI CRYPTO TRADING BOT v2.0.0          ║"
echo "    ║          Raspberry Pi 5 Deployer             ║"
echo "    ╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# 0. UPDATE CODE (skip if FAST_SETUP=1)
echo -e "\n$(ts)${BOLD}[0/5] Code update${NC}"
if [ "${FAST_SETUP}" != "1" ]; then
  echo -e "  $(ts)${DIM}Fetching latest from GitHub (origin main)...${NC}"
  git pull origin main
  echo -e "  $(ts)${GREEN}✓ Repository up to date.${NC}"
else
  echo -e "  $(ts)${YELLOW}FAST_SETUP=1: skipping git pull.${NC}"
fi

# 1. PREREQUISITES CHECK
echo -e "\n$(ts)${BOLD}[1/5] Prerequisites${NC}"
echo -e "  $(ts)${DIM}Checking for Python3...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "  $(ts)${RED}✗ Python3 not found. Install: sudo apt install python3 python3-venv${NC}"
    exit 1
fi
PYVER=$(python3 -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo "?")
echo -e "  $(ts)${GREEN}✓ Python3 found: ${PYVER}${NC}"

# 2. CONFIGURATION
echo -e "\n$(ts)${BOLD}[2/5] Configuration${NC}"

if [ -f ".env" ]; then
    echo -e "  $(ts)${YELLOW}.env file present.${NC}"
    read -p "  Use existing .env? [Y/n]: " KEEP_ENV
    KEEP_ENV=${KEEP_ENV:-Y}
else
    KEEP_ENV="n"
fi

if [[ "$KEEP_ENV" =~ ^[Yy]$ ]]; then
    echo -e "  $(ts)${GREEN}✓ Using existing .env${NC}"
else
    # Prompt for Kraken Keys
    read -p "Enter Kraken API Key: " KRAKEN_KEY
    read -p "Enter Kraken API Secret: " KRAKEN_SECRET

    # Prompt for Trading Mode
    echo -e "\n${YELLOW}Trading Mode:${NC}"
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
KRAKEN_API_KEY=
KRAKEN_API_SECRET=
TRADING_MODE=$TRADING_MODE
INITIAL_BANKROLL=$BANKROLL
MAX_RISK_PER_TRADE=0.005
MAX_DAILY_LOSS=0.05
MAX_POSITION_USD=200.0
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
DASHBOARD_SECRET_KEY=
VAULT_PASSWORD=
LOG_LEVEL=INFO
MODEL_RETRAIN_INTERVAL_HOURS=168
MIN_TRAINING_SAMPLES=10000
DB_PATH=data/trading.db
EOF
    echo -e "${GREEN}.env configuration created successfully.${NC}"
fi

# 3. BUILD
echo -e "\n$(ts)${BOLD}[3/5] Build & dependencies${NC}"
echo -e "  $(ts)${DIM}Creating directories: data, logs, models, config${NC}"
mkdir -p data logs models config
echo -e "  $(ts)${GREEN}✓ Directories ready.${NC}"

if [ ! -d "venv" ]; then
    echo -e "  $(ts)${DIM}Creating virtual environment (first time only)...${NC}"
    python3 -m venv venv
    echo -e "  $(ts)${GREEN}✓ Virtual environment created.${NC}"
else
    echo -e "  $(ts)${GREEN}✓ Virtual environment already exists.${NC}"
fi

echo -e "  $(ts)${DIM}Activating venv and preparing pip...${NC}"
source venv/bin/activate
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
echo -e "  $(ts)${DIM}Pip cache: ${PIP_CACHE_DIR}${NC}"

if [ "${FAST_SETUP}" != "1" ]; then
  echo -e "  $(ts)${DIM}Upgrading pip...${NC}"
  pip install --upgrade pip -q
  echo -e "  $(ts)${GREEN}✓ Pip upgraded.${NC}"
fi

if [ -f "requirements-pi.txt" ]; then
  echo -e "  $(ts)${DIM}Installing from requirements-pi.txt (no TensorFlow; faster on Pi)...${NC}"
  echo -e "  $(ts)${YELLOW}This may take several minutes. Each package will be listed below.${NC}"
  pip install --prefer-binary -r requirements-pi.txt 2>&1 | while IFS= read -r line; do
    echo -e "    ${DIM}[pip]${NC} $line"
  done
  echo -e "  $(ts)${GREEN}✓ Dependencies installed (requirements-pi.txt).${NC}"
else
  echo -e "  $(ts)${DIM}Installing from requirements.txt...${NC}"
  pip install --prefer-binary -r requirements.txt 2>&1 | while IFS= read -r line; do
    echo -e "    ${DIM}[pip]${NC} $line"
  done
  echo -e "  $(ts)${GREEN}✓ Dependencies installed.${NC}"
fi

# 4. START BOT
echo -e "\n$(ts)${BOLD}[4/5] Starting the bot${NC}"
echo -e "  $(ts)${DIM}Stopping any existing bot process...${NC}"
pkill -f "python main.py" &> /dev/null
sleep 1
echo -e "  $(ts)${DIM}Starting main.py in background (logs → logs/bot_output.log, logs/trading_bot.log)...${NC}"
nohup venv/bin/python main.py >> logs/bot_output.log 2>&1 &
BOT_PID=$!
echo -e "  $(ts)${GREEN}✓ Bot running (PID: ${BOT_PID}).${NC}"
echo -e "  $(ts)${DIM}Watch live: tail -f logs/trading_bot.log${NC}"

# 5. LAUNCH DASHBOARD
echo -e "\n$(ts)${BOLD}[5/5] Dashboard${NC}"
echo -e "  $(ts)${DIM}Waiting 5s for server to bind...${NC}"
sleep 5
echo -e "  $(ts)${GREEN}✓ Ready.${NC}"

if [ -n "$DISPLAY" ]; then
    echo -e "  $(ts)${DIM}Opening dashboard in browser...${NC}"
    if command -v chromium-browser &> /dev/null; then
        chromium-browser http://localhost:8080 &
    elif command -v firefox &> /dev/null; then
        firefox http://localhost:8080 &
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        open http://localhost:8080
    fi
else
    echo -e "  $(ts)${YELLOW}Headless: open http://$(hostname).local:8080 in your browser.${NC}"
fi

echo -e "\n$(ts)${CYAN}${BOLD}DEPLOYMENT COMPLETE${NC}"
echo -e "  Bot is hunting for confluence setups. Logs: logs/trading_bot.log"
echo -e "  $(ts)${DIM}Run with FAST_SETUP=1 for a quicker restart (skips git pull and pip upgrade).${NC}"