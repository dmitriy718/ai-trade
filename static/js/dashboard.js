/**
 * AI Trading Bot - Dashboard Client v3
 *
 * Zero console spam: WS-first with graceful degradation.
 * No fetch() calls until server is confirmed reachable.
 */

// ---- State ----
let ws = null;
let wsAttempts = 0;
let isPaused = false;
let priceHistory = {};
let lastThoughtIds = [];
let userScrolledThoughts = false;
let scrollResetTimer = null;
let serverReachable = false;   // gate: no fetches until true
let reconnectTimer = null;

// ---- WebSocket (sole data channel when healthy) ----

function connectWebSocket() {
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${protocol}://${window.location.host}/ws/live`;

    // Suppress the console error: only create WS if we believe server is up
    // On first load we try regardless; after that we gate on serverReachable
    if (wsAttempts > 0 && !serverReachable) {
        // Silently probe with a HEAD request first (no console error on fail)
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), 3000);
        fetch('/api/v1/health', { signal: ctrl.signal, method: 'HEAD' })
            .then(r => { clearTimeout(t); if (r.ok) { serverReachable = true; doConnect(wsUrl); } else { scheduleReconnect(); } })
            .catch(() => { clearTimeout(t); scheduleReconnect(); });
        return;
    }

    doConnect(wsUrl);
}

function doConnect(wsUrl) {
    try { ws = new WebSocket(wsUrl); } catch { scheduleReconnect(); return; }

    ws.onopen = () => {
        wsAttempts = 0;
        serverReachable = true;
        updateStatus('ONLINE', false);
        loadSettings();
    };

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'update') renderUpdate(msg.data);
        } catch {}
    };

    ws.onclose = () => {
        ws = null;
        serverReachable = false;
        updateStatus('RECONNECTING...', true);
        scheduleReconnect();
    };

    ws.onerror = () => { /* onclose will fire next */ };
}

function scheduleReconnect() {
    if (reconnectTimer) return; // already scheduled
    wsAttempts++;
    // Exponential backoff: 2s, 4s, 8s, 16s, max 30s
    const delay = Math.min(2000 * Math.pow(2, Math.min(wsAttempts - 1, 4)), 30000);
    reconnectTimer = setTimeout(() => { reconnectTimer = null; connectWebSocket(); }, delay);
}

function updateStatus(text, isError) {
    const el = document.getElementById('statusIndicator');
    const st = document.getElementById('statusText');
    if (st) st.textContent = text;
    if (el) el.classList.toggle('error', isError);
}

// ---- Render Orchestrator ----

function renderUpdate(data) {
    if (!data) return;
    renderStatus(data.status);
    renderPerformance(data.performance, data.risk);
    renderPositions(data.positions);
    renderThoughts(data.thoughts);
    renderScanner(data.scanner);
    renderRisk(data.risk);
}

// ---- Status / HUD ----

function renderStatus(status) {
    if (!status) return;
    setText('tradingMode', (status.mode || '--').toUpperCase());
    setText('uptime', formatUptime(status.uptime || 0));
    if (status.scan_count !== undefined) setText('scanCount', status.scan_count.toLocaleString());

    const label = status.running ? (status.paused ? 'PAUSED' : 'ONLINE') : 'STOPPED';
    updateStatus(label, !status.running || status.paused);

    isPaused = status.paused || false;
    const btn = document.getElementById('pauseBtn');
    if (btn) btn.textContent = isPaused ? '▶ RESUME' : '⏸ PAUSE';
}

// ---- Portfolio / Performance ----

function renderPerformance(perf, risk) {
    if (!perf) return;
    const realized = perf.total_pnl || 0;
    const unrealized = perf.unrealized_pnl || 0;
    const totalPnl = realized + unrealized;
    const el = document.getElementById('totalPnl');
    if (el) {
        el.textContent = formatMoney(totalPnl);
        el.className = 'pnl-value' + (totalPnl < 0 ? ' negative' : '');
    }

    const uEl = document.getElementById('unrealizedPnl');
    if (uEl) {
        uEl.textContent = formatMoney(unrealized);
        uEl.style.color = unrealized >= 0 ? 'var(--neon-green)' : 'var(--red)';
    }

    const todayEl = document.getElementById('todayPnl');
    if (todayEl) {
        todayEl.textContent = formatMoney(realized);
        todayEl.style.color = realized >= 0 ? 'var(--neon-green)' : 'var(--red)';
    }

    if (risk) setText('bankroll', formatMoney(risk.bankroll || 0));
    setText('winRate', ((perf.win_rate || 0) * 100).toFixed(1) + '%');
    setText('totalTrades', perf.total_trades || 0);
    setText('openPositions', perf.open_positions || 0);
    if (risk) setText('drawdown', (risk.current_drawdown || 0).toFixed(1) + '%');
}

// ---- Positions Table ----

function renderPositions(positions) {
    const tbody = document.getElementById('positionsBody');
    if (!tbody) return;
    if (!positions || positions.length === 0) {
        tbody.innerHTML = '<tr class="empty-row"><td colspan="6">No open positions</td></tr>';
        return;
    }
    let html = '';
    for (const pos of positions) {
        const pnl = pos.unrealized_pnl || 0;
        const pnlPct = pos.unrealized_pnl_pct || 0;
        html += `<tr>
            <td>${pos.pair}</td>
            <td class="${pos.side === 'buy' ? 'side-buy' : 'side-sell'}">${pos.side.toUpperCase()}</td>
            <td>${formatPrice(pos.entry_price)}</td>
            <td>${formatPrice(pos.current_price || 0)}</td>
            <td class="${pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">${formatMoney(pnl)} (${(pnlPct * 100).toFixed(2)}%)</td>
            <td>${formatPrice(pos.stop_loss || 0)}</td>
        </tr>`;
    }
    tbody.innerHTML = html;
}

// ---- AI Thought Feed ----

function renderThoughts(thoughts) {
    const feed = document.getElementById('thoughtFeed');
    const countBadge = document.getElementById('thoughtCount');
    if (!feed || !thoughts) return;
    if (countBadge) countBadge.textContent = thoughts.length;

    const newKeys = thoughts.map(t => t.timestamp + '|' + t.message);
    if (newKeys.length === lastThoughtIds.length && newKeys.every((k, i) => k === lastThoughtIds[i])) return;
    lastThoughtIds = newKeys;

    const existingMap = new Map();
    feed.querySelectorAll('.thought-item[data-key]').forEach(n => existingMap.set(n.getAttribute('data-key'), n));

    const frag = document.createDocumentFragment();
    for (const thought of thoughts) {
        const key = thought.timestamp + '|' + thought.message;
        const existing = existingMap.get(key);
        if (existing) { frag.appendChild(existing); continue; }

        const div = document.createElement('div');
        div.className = 'thought-item thought-new';
        div.setAttribute('data-key', key);
        div.innerHTML =
            `<span class="thought-time">${formatTime(thought.timestamp)}</span>` +
            `<span class="thought-category ${thought.category || 'system'}">${escHtml((thought.category || 'system').toUpperCase())}</span>` +
            `<span class="thought-msg">${escHtml(thought.message || '')}</span>`;
        frag.appendChild(div);
    }

    feed.innerHTML = '';
    feed.appendChild(frag);
    if (!userScrolledThoughts) feed.scrollTop = 0;

    requestAnimationFrame(() => {
        feed.querySelectorAll('.thought-new').forEach(el => {
            el.addEventListener('animationend', () => el.classList.remove('thought-new'), { once: true });
        });
    });
}

function setupThoughtScroll() {
    const feed = document.getElementById('thoughtFeed');
    if (!feed) return;
    feed.addEventListener('scroll', () => {
        userScrolledThoughts = feed.scrollTop > 40;
        clearTimeout(scrollResetTimer);
        if (userScrolledThoughts) {
            scrollResetTimer = setTimeout(() => {
                userScrolledThoughts = false;
                feed.scrollTo({ top: 0, behavior: 'smooth' });
            }, 15000);
        }
    });
}

// ---- Ticker Scanner ----

function renderScanner(scanner) {
    const grid = document.getElementById('scannerGrid');
    if (!grid || !scanner) return;
    const entries = Object.entries(scanner);
    const existing = grid.querySelectorAll('.scanner-item');

    if (existing.length === entries.length) {
        entries.forEach(([pair, data], i) => {
            const item = existing[i];
            const price = data.price || 0;
            if (!priceHistory[pair]) priceHistory[pair] = [];
            priceHistory[pair].push(price);
            if (priceHistory[pair].length > 20) priceHistory[pair].shift();
            item.querySelector('.scanner-pair').textContent = pair;
            item.querySelector('.scanner-price').textContent = formatPrice(price);
            item.querySelector('.scanner-bars').textContent = (data.bars || 0) + ' bars';
            item.classList.toggle('stale', !!data.stale);
            const spark = item.querySelector('.sparkline');
            if (spark) spark.innerHTML = generateSparkline(priceHistory[pair]);
        });
        return;
    }

    let html = '';
    for (const [pair, data] of entries) {
        const price = data.price || 0;
        if (!priceHistory[pair]) priceHistory[pair] = [];
        priceHistory[pair].push(price);
        if (priceHistory[pair].length > 20) priceHistory[pair].shift();
        html += `<div class="scanner-item${data.stale ? ' stale' : ''}">
            <span class="scanner-pair">${pair}</span>
            <span class="scanner-price">${formatPrice(price)}</span>
            <div class="sparkline">${generateSparkline(priceHistory[pair])}</div>
            <span class="scanner-bars">${data.bars || 0} bars</span>
        </div>`;
    }
    grid.innerHTML = html || '<div class="scanner-item"><span class="scanner-pair">Waiting...</span></div>';
}

// ---- Risk Monitor ----

function renderRisk(risk) {
    if (!risk) return;
    setGauge('rorGauge', 'rorValue', (risk.risk_of_ruin || 0) * 100, v => v.toFixed(2) + '%');
    const dl = Math.abs(risk.daily_pnl || 0);
    const maxDl = (risk.bankroll || 10000) * 0.05;
    setGauge('dailyLossGauge', 'dailyLossValue', (dl / maxDl) * 100, () => formatMoney(risk.daily_pnl || 0));
    const exp = risk.total_exposure_usd || 0;
    const maxExp = (risk.bankroll || 10000) * 0.5;
    setGauge('exposureGauge', 'exposureValue', (exp / maxExp) * 100, () => formatMoney(exp));
    setText('ddFactor', (risk.drawdown_factor || 1.0).toFixed(2) + 'x');
}

function setGauge(barId, valId, pct, fmt) {
    const bar = document.getElementById(barId);
    const val = document.getElementById(valId);
    if (bar) bar.style.width = Math.min(pct, 100) + '%';
    if (val) val.textContent = fmt(pct);
}

// ---- Sparkline ----

function generateSparkline(data) {
    if (!data || data.length < 2) return '';
    const min = Math.min(...data), max = Math.max(...data), range = max - min || 1;
    return data.map(v => `<div class="spark-bar" style="height:${Math.max(2, ((v - min) / range) * 18)}px"></div>`).join('');
}

// ---- Settings ----

const API_KEY = 'change_this_to_a_random_string';

async function loadSettings() {
    if (!serverReachable) return;
    try {
        const r = await fetch('/api/v1/settings');
        const d = await r.json();
        const cb = document.getElementById('weightedOrderBook');
        if (cb && typeof d.weighted_order_book === 'boolean') cb.checked = d.weighted_order_book;
    } catch {}
}

async function saveWeightedOrderBook(checked) {
    if (!serverReachable) return;
    try {
        await fetch('/api/v1/settings', {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
            body: JSON.stringify({ weighted_order_book: checked })
        });
    } catch {}
}

function setupSettings() {
    const cb = document.getElementById('weightedOrderBook');
    if (!cb) return;
    cb.addEventListener('change', () => saveWeightedOrderBook(cb.checked));
}

// ---- Controls ----

async function togglePause() {
    if (!serverReachable) return;
    const ep = isPaused ? '/api/v1/control/resume' : '/api/v1/control/pause';
    try { await fetch(ep, { method: 'POST', headers: { 'X-API-Key': API_KEY } }); } catch {}
}

async function closeAll() {
    if (!serverReachable) return;
    if (!confirm('EMERGENCY: Close ALL open positions?')) return;
    try {
        const r = await fetch('/api/v1/control/close_all', { method: 'POST', headers: { 'X-API-Key': API_KEY } });
        const d = await r.json();
        alert(`Closed ${d.closed} positions`);
    } catch {}
}

// ---- Strategy Stats (only when server is reachable) ----

async function fetchStrategyStats() {
    if (!serverReachable) return;
    try {
        const strategies = await (await fetch('/api/v1/strategies')).json();
        const barMap = { trend:'trendBar', mean_reversion:'meanrevBar', momentum:'momentumBar', breakout:'breakoutBar', reversal:'reversalBar' };
        const statMap = { trend:'trendStat', mean_reversion:'meanrevStat', momentum:'momentumStat', breakout:'breakoutStat', reversal:'reversalStat' };
        for (const s of strategies) {
            const bar = document.getElementById(barMap[s.name]);
            const stat = document.getElementById(statMap[s.name]);
            if (bar) bar.style.width = ((s.win_rate || 0) * 100) + '%';
            if (stat) stat.textContent = s.trades > 0 ? `${(s.win_rate*100).toFixed(0)}% (${s.trades})` : '--';
        }
    } catch {}
}

// ---- Utilities ----

function setText(id, val) {
    const el = document.getElementById(id);
    if (el && el.textContent !== String(val)) el.textContent = val;
}

function formatMoney(v) {
    const n = Number(v) || 0;
    return (n >= 0 ? '+$' : '-$') + Math.abs(n).toFixed(2);
}

function formatPrice(p) {
    p = Number(p) || 0;
    if (p >= 1000) return '$' + p.toFixed(2);
    if (p >= 1) return '$' + p.toFixed(4);
    return '$' + p.toFixed(6);
}

function formatUptime(s) {
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
    return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;
}

function formatTime(iso) {
    if (!iso) return '--:--:--';
    try { return new Date(iso).toLocaleTimeString('en-US', { hour12: false }); } catch { return '--:--:--'; }
}

function escHtml(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }

// ---- Init ----

document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    setupThoughtScroll();
    setupSettings();
    // Load settings when server is reachable (after first WS connect)
    setInterval(loadSettings, 15000);
    loadSettings();
    // Only fetch strategies when server is reachable (gated inside the function)
    setInterval(fetchStrategyStats, 5000);
    // NO fallback polling — WebSocket is the sole data channel.
    // Reconnection handles recovery. This eliminates all console spam.
});
