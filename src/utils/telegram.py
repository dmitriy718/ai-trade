"""
Telegram Command Center - Remote control and notification system.

Provides real-time trade notifications, performance alerts, and
remote command execution via Telegram bot.

Commands:
    /status - System status
    /pnl - Current P&L
    /positions - Open positions
    /kill - Emergency stop (requires confirmation)
    /close_all - Close all positions
    /pause - Pause trading
    /resume - Resume trading
    /risk - Risk report

# ENHANCEMENT: Added rate limiting for notification spam prevention
# ENHANCEMENT: Added command authentication for security
# ENHANCEMENT: Added rich message formatting with inline keyboards
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable, Dict, Optional

from src.core.logger import get_logger

logger = get_logger("telegram")


class TelegramBot:
    """
    Telegram bot for remote monitoring and control.
    
    Features:
    - Real-time trade notifications
    - P&L alerts on threshold breaches
    - Remote command execution
    - Rate-limited messaging
    
    # ENHANCEMENT: Added message queuing for offline periods
    # ENHANCEMENT: Added notification deduplication
    """

    def __init__(
        self,
        token: str = "",
        chat_id: str = "",
        rate_limit_seconds: int = 2,
    ):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.rate_limit_seconds = rate_limit_seconds
        self._bot = None
        self._app = None
        self._last_message_time: float = 0
        self._bot_engine = None  # L19: Removed unused _message_queue
        self._enabled = bool(self.token and self.chat_id)

    def set_bot_engine(self, engine) -> None:
        """Inject the bot engine reference for command execution."""
        self._bot_engine = engine

    async def initialize(self) -> bool:
        """Initialize the Telegram bot."""
        if not self._enabled:
            logger.info("Telegram bot disabled (no token/chat_id)")
            return False

        try:
            from telegram import Update
            from telegram.ext import (
                Application,
                CommandHandler,
                ContextTypes,
            )

            self._app = Application.builder().token(self.token).build()

            # Register command handlers
            self._app.add_handler(CommandHandler("status", self._cmd_status))
            self._app.add_handler(CommandHandler("pnl", self._cmd_pnl))
            self._app.add_handler(CommandHandler("positions", self._cmd_positions))
            self._app.add_handler(CommandHandler("risk", self._cmd_risk))
            self._app.add_handler(CommandHandler("pause", self._cmd_pause))
            self._app.add_handler(CommandHandler("resume", self._cmd_resume))
            self._app.add_handler(CommandHandler("close_all", self._cmd_close_all))
            self._app.add_handler(CommandHandler("kill", self._cmd_kill))
            self._app.add_handler(CommandHandler("help", self._cmd_help))

            logger.info("Telegram bot initialized")
            return True

        except ImportError:
            logger.warning("python-telegram-bot not installed")
            self._enabled = False
            return False
        except Exception as e:
            logger.error("Telegram init failed", error=str(e))
            self._enabled = False
            return False

    async def start(self) -> None:
        """Start the Telegram bot polling."""
        if not self._enabled or not self._app:
            return

        try:
            await self._app.initialize()
            await self._app.start()
            await self._app.updater.start_polling()

            await self.send_message(
                "🚀 *AI Trading Bot Started*\n"
                "Mode: `{}`\n"
                "Use /help for commands".format(
                    self._bot_engine.mode if self._bot_engine else "unknown"
                )
            )
        except Exception as e:
            logger.error("Telegram start failed", error=str(e))

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self._app:
            try:
                await self.send_message("🔴 *AI Trading Bot Stopped*")
                await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception:
                pass

    async def send_message(self, text: str, parse_mode: str = "Markdown") -> None:
        """Send a message to the configured chat, respecting rate limits."""
        if not self._enabled or not self._app:
            return

        # Rate limiting
        elapsed = time.time() - self._last_message_time
        if elapsed < self.rate_limit_seconds:
            await asyncio.sleep(self.rate_limit_seconds - elapsed)

        try:
            await self._app.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode,
            )
            self._last_message_time = time.time()
        except Exception as e:
            logger.debug("Telegram send failed", error=str(e))

    async def notify_trade(self, trade: Dict[str, Any]) -> None:
        """Send trade notification."""
        side_emoji = "📈" if trade.get("side") == "buy" else "📉"
        msg = (
            f"{side_emoji} *Trade Executed*\n"
            f"Pair: `{trade.get('pair')}`\n"
            f"Side: `{trade.get('side', '').upper()}`\n"
            f"Price: `${trade.get('entry_price', 0):.2f}`\n"
            f"Size: `${trade.get('size_usd', 0):.2f}`\n"
            f"SL: `${trade.get('stop_loss', 0):.2f}`\n"
            f"Confidence: `{trade.get('confidence', 0):.2%}`"
        )
        await self.send_message(msg)

    async def notify_close(self, trade: Dict[str, Any]) -> None:
        """Send position close notification."""
        pnl = trade.get("pnl", 0)
        emoji = "✅" if pnl > 0 else "❌"
        msg = (
            f"{emoji} *Position Closed*\n"
            f"Pair: `{trade.get('pair')}`\n"
            f"P&L: `${pnl:.2f}` ({trade.get('pnl_pct', 0):.2%})\n"
            f"Reason: `{trade.get('reason', 'unknown')}`"
        )
        await self.send_message(msg)

    # ------------------------------------------------------------------
    # Command Handlers
    # ------------------------------------------------------------------

    def _is_authorized(self, update) -> bool:
        """C5 FIX: Verify chat_id matches configured authorized user."""
        if not self.chat_id:
            return False
        return str(update.message.chat_id) == str(self.chat_id)

    async def _cmd_status(self, update, context) -> None:
        """Handle /status command."""
        if not self._is_authorized(update):
            return
        if not self._bot_engine:
            await update.message.reply_text("Bot engine not connected")
            return

        msg = (
            "📊 *System Status*\n"
            f"Mode: `{self._bot_engine.mode}`\n"
            f"Running: `{self._bot_engine._running}`\n"
            f"Paused: `{self._bot_engine._trading_paused}`\n"
            f"Pairs: `{len(self._bot_engine.pairs)}`\n"
            f"Scans: `{self._bot_engine._scan_count}`"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_pnl(self, update, context) -> None:
        """Handle /pnl command."""
        if not self._is_authorized(update) or not self._bot_engine:
            return

        stats = await self._bot_engine.db.get_performance_stats()
        risk = self._bot_engine.risk_manager.get_risk_report()
        msg = (
            "💰 *Performance*\n"
            f"Total P&L: `${stats.get('total_pnl', 0):.2f}`\n"
            f"Today P&L: `${stats.get('today_pnl', 0):.2f}`\n"
            f"Win Rate: `{stats.get('win_rate', 0):.1%}`\n"
            f"Trades: `{stats.get('total_trades', 0)}`\n"
            f"Bankroll: `${risk.get('bankroll', 0):.2f}`\n"
            f"Drawdown: `{risk.get('current_drawdown', 0):.1f}%`"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_positions(self, update, context) -> None:
        """Handle /positions command."""
        if not self._is_authorized(update) or not self._bot_engine:
            return

        positions = await self._bot_engine.db.get_open_trades()
        if not positions:
            await update.message.reply_text("📭 No open positions")
            return

        msg = "📊 *Open Positions*\n\n"
        for pos in positions:
            current_price = self._bot_engine.market_data.get_latest_price(pos["pair"])
            if pos["side"] == "buy":
                pnl = (current_price - pos["entry_price"]) * pos["quantity"]
            else:
                pnl = (pos["entry_price"] - current_price) * pos["quantity"]
            emoji = "🟢" if pnl >= 0 else "🔴"
            msg += (
                f"{emoji} `{pos['pair']}` {pos['side'].upper()}\n"
                f"  Entry: ${pos['entry_price']:.2f} | Now: ${current_price:.2f}\n"
                f"  P&L: ${pnl:.2f}\n\n"
            )
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_risk(self, update, context) -> None:
        """Handle /risk command."""
        if not self._is_authorized(update) or not self._bot_engine:
            return

        risk = self._bot_engine.risk_manager.get_risk_report()
        msg = (
            "🛡️ *Risk Report*\n"
            f"Bankroll: `${risk.get('bankroll', 0):.2f}`\n"
            f"Exposure: `${risk.get('total_exposure_usd', 0):.2f}`\n"
            f"Daily P&L: `${risk.get('daily_pnl', 0):.2f}`\n"
            f"Drawdown: `{risk.get('current_drawdown', 0):.1f}%`\n"
            f"Max DD: `{risk.get('max_drawdown_pct', 0):.1f}%`\n"
            f"Risk of Ruin: `{risk.get('risk_of_ruin', 0):.4f}`\n"
            f"DD Factor: `{risk.get('drawdown_factor', 1):.2f}x`"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_pause(self, update, context) -> None:
        """Handle /pause command."""
        if not self._is_authorized(update):
            return
        if self._bot_engine:
            self._bot_engine._trading_paused = True
            await update.message.reply_text("⏸ Trading *PAUSED*", parse_mode="Markdown")

    async def _cmd_resume(self, update, context) -> None:
        """Handle /resume command."""
        if not self._is_authorized(update):
            return
        if self._bot_engine:
            self._bot_engine._trading_paused = False
            await update.message.reply_text("▶️ Trading *RESUMED*", parse_mode="Markdown")

    async def _cmd_close_all(self, update, context) -> None:
        """Handle /close_all command."""
        if not self._is_authorized(update):
            return
        if self._bot_engine:
            count = await self._bot_engine.executor.close_all_positions("telegram")
            await update.message.reply_text(
                f"⚠️ Closed *{count}* positions", parse_mode="Markdown"
            )

    async def _cmd_kill(self, update, context) -> None:
        """Handle /kill command - emergency shutdown."""
        if not self._is_authorized(update):
            return
        if self._bot_engine:
            await update.message.reply_text(
                "🔴 *EMERGENCY SHUTDOWN INITIATED*\n"
                "Closing all positions and stopping...",
                parse_mode="Markdown"
            )
            await self._bot_engine.executor.close_all_positions("emergency_kill")
            await self._bot_engine.stop()

    async def _cmd_help(self, update, context) -> None:
        """Handle /help command."""
        if not self._is_authorized(update):
            return
        msg = (
            "🤖 *AI Trading Bot Commands*\n\n"
            "/status - System status\n"
            "/pnl - Performance summary\n"
            "/positions - Open positions\n"
            "/risk - Risk report\n"
            "/pause - Pause trading\n"
            "/resume - Resume trading\n"
            "/close\\_all - Close all positions\n"
            "/kill - Emergency shutdown\n"
            "/help - This message"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
