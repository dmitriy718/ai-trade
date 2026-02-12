#!/usr/bin/env bash
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
SECRETS="$BASE/.secrets"
DB="$BASE/data/trading.db"
STATE="$BASE/.health_check.state"
OUT="$BASE/logs/health_check.log"

BANKROLL="${INITIAL_BANKROLL:-}"
if [ -z "$BANKROLL" ] && [ -f "$BASE/.env" ]; then
  BANKROLL=$(grep -E '^INITIAL_BANKROLL=' "$BASE/.env" | tail -n 1 | cut -d'=' -f2- | tr -d '[:space:]')
fi
if [ -z "$BANKROLL" ]; then
  BANKROLL="10000"
fi

TOKEN="${TELEGRAM_BOT_TOKEN:-}"
CHAT_ID="${TELEGRAM_CHAT_ID:-}"
if [ -z "$TOKEN" ] && [ -f "$SECRETS/telegram_token" ]; then
  TOKEN=$(cat "$SECRETS/telegram_token" | tr -d '[:space:]')
fi
if [ -z "$CHAT_ID" ] && [ -f "$SECRETS/telegram_chat_id" ]; then
  CHAT_ID=$(cat "$SECRETS/telegram_chat_id" | tr -d '[:space:]')
fi

if [ -z "$TOKEN" ] || [ -z "$CHAT_ID" ]; then
  exit 0
fi

pid=$(pgrep -fa "venv/bin/python main.py" | awk 'NR==1{print $1}')
status="RUNNING"
if [ -z "${pid:-}" ]; then
  status="DOWN"
fi
status_emoji="✅"
if [ "$status" = "DOWN" ]; then
  status_emoji="🔴"
fi

read -r total_trades max_id uptime_s open_positions win_rate drawdown_pnl drawdown_pct streak_label <<EOF
$(BOT_PID="${pid:-0}" DB="$DB" INITIAL_BANKROLL="${BANKROLL}" /usr/bin/python3 - <<'PY'
import os
import sqlite3

pid = int(os.environ.get("BOT_PID", "0") or 0)
DB = os.environ.get("DB")
initial_bankroll = float(os.environ.get("INITIAL_BANKROLL", "10000") or 10000)

total = 0
max_id = 0
open_positions = 0
win_rate = 0.0
drawdown_pnl = 0.0
drawdown_pct = 0.0
streak_label = "N/A"

if DB and os.path.exists(DB):
    conn = sqlite3.connect(DB, timeout=5)
    cur = conn.cursor()
    cur.execute("select count(*) from trades")
    total = cur.fetchone()[0] or 0
    cur.execute("select max(id) from trades")
    max_id = cur.fetchone()[0] or 0
    cur.execute("select count(*) from trades where status='open'")
    open_positions = cur.fetchone()[0] or 0
    cur.execute("select pnl from trades where status='closed' order by id asc")
    rows = cur.fetchall()
    if rows:
        wins = sum(1 for (p,) in rows if (p or 0) > 0)
        win_rate = wins / max(len(rows), 1)
        cum = 0.0
        peak = float(initial_bankroll)
        for (p,) in rows:
            cum += float(p or 0)
            equity = initial_bankroll + cum
            if equity > peak:
                peak = equity
        equity_now = initial_bankroll + cum
        drawdown_pnl = max(0.0, peak - equity_now)
        drawdown_pct = (drawdown_pnl / peak * 100.0) if peak > 0 else 0.0
        # streak (wins/losses)
        streak = 0
        streak_type = ''
        last = None
        for (p,) in rows:
            win = (float(p or 0) > 0)
            if last is None:
                streak = 1
                streak_type = 'W' if win else 'L'
            else:
                if win and streak_type == 'W':
                    streak += 1
                elif (not win) and streak_type == 'L':
                    streak += 1
                else:
                    streak = 1
                    streak_type = 'W' if win else 'L'
            last = win
        if streak_type:
            streak_label = f"{streak_type}{streak}"
    conn.close()

uptime_s = ""
if pid > 0 and os.path.exists(f"/proc/{pid}/stat"):
    with open("/proc/uptime", "r") as f:
        uptime_sys = float(f.read().split()[0])
    with open(f"/proc/{pid}/stat", "r") as f:
        fields = f.read().split()
    start_ticks = int(fields[21])
    hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    proc_uptime = uptime_sys - (start_ticks / hz)
    uptime_s = max(proc_uptime, 0.0)

print(f"{total} {max_id} {uptime_s} {open_positions} {win_rate} {drawdown_pnl} {drawdown_pct} {streak_label}")
PY
)
EOF

last_id=0
if [ -f "$STATE" ]; then
  last_id=$(cat "$STATE" || echo 0)
fi
new_trades=0
if [ -n "$max_id" ] && [ "$max_id" -ge "$last_id" ]; then
  new_trades=$((max_id - last_id))
fi

format_duration() {
  local s="$1"
  if [ -z "$s" ]; then
    echo "--:--:--"
    return
  fi
  local total=${s%.*}
  local h=$((total / 3600))
  local m=$(((total % 3600) / 60))
  local sec=$((total % 60))
  printf "%02d:%02d:%02d" "$h" "$m" "$sec"
}

uptime_fmt=$(format_duration "${uptime_s:-}")

now_iso=$(date -Is)
win_rate_pct=$(awk "BEGIN { printf \"%.1f%%\", ${win_rate} * 100 }")
drawdown_line=$(printf "\\$%.2f (%.2f%%)" "${drawdown_pnl}" "${drawdown_pct}")
msg=$(printf "✨ *All Systems Check* ✨\n\n*Host:* %s\n*Status:* %s %s\n*Uptime:* %s\n*Open Positions:* %s\n*Win Rate:* %s\n*Drawdown:* %s\n*Streak:* %s\n*Total Trades:* %s\n*New Trades:* %s\n*Time:* %s" \
  "${HOSTNAME}" "${status_emoji}" "${status}" "${uptime_fmt}" "${open_positions}" "${win_rate_pct}" "${drawdown_line}" "${streak_label}" "${total_trades}" "${new_trades}" "${now_iso}")

# Local log (for diagnostics)
{
  echo "---- $(date -Is) ----"
  echo "$msg"
} >> "$OUT"

curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
  --data-urlencode "chat_id=${CHAT_ID}" \
  --data-urlencode "parse_mode=Markdown" \
  --data-urlencode "text=${msg}" >/dev/null || true

echo "$max_id" > "$STATE"
