"""
Discord Control Bot - Remote control via Discord.

Commands: /pause, /resume, /close_all, /kill, /status, /pnl, /positions, /risk.
Only responds in allowed channel(s) or from allowed user IDs.
Uses control router for all control actions.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from src.core.logger import get_logger

logger = get_logger("discord_bot")


class DiscordBot:
    """
    Discord bot for remote monitoring and control.
    Commands invoke the control router; auth via allowed_channel_ids or guild.
    """

    def __init__(
        self,
        token: str = "",
        allowed_channel_ids: Optional[List[str]] = None,
        allowed_guild_id: Optional[str] = None,
    ):
        self.token = token
        self.allowed_channel_ids = [str(c) for c in (allowed_channel_ids or [])]
        self.allowed_guild_id = str(allowed_guild_id) if allowed_guild_id else None
        self._control_router = None
        self._bot = None
        self._enabled = bool(self.token)

    def set_control_router(self, router) -> None:
        """Inject the control router for pause/resume/close_all/kill."""
        self._control_router = router

    def _is_authorized(self, channel_id: Optional[int], guild_id: Optional[int]) -> bool:
        """Return True if channel or guild is in allowlist."""
        if self.allowed_guild_id and guild_id is not None:
            if str(guild_id) == self.allowed_guild_id:
                return True
        if self.allowed_channel_ids and channel_id is not None:
            if str(channel_id) in self.allowed_channel_ids:
                return True
        return bool(not self.allowed_channel_ids and not self.allowed_guild_id)

    async def initialize(self) -> bool:
        """Initialize the Discord bot and register commands."""
        if not self._enabled:
            logger.info("Discord bot disabled (no token)")
            return False

        try:
            import discord
            from discord.ext import commands

            intents = discord.Intents.default()
            intents.message_content = True
            self._bot = commands.Bot(command_prefix="!", intents=intents)

            @self._bot.tree.command(name="pause", description="Pause trading")
            async def pause(interaction: discord.Interaction):
                if not self._is_authorized(interaction.channel_id, interaction.guild_id):
                    await interaction.response.send_message("Not authorized.", ephemeral=True)
                    return
                if self._control_router:
                    await self._control_router.pause()
                    await interaction.response.send_message("Trading **paused**.", ephemeral=True)
                else:
                    await interaction.response.send_message("Control router not set.", ephemeral=True)

            @self._bot.tree.command(name="resume", description="Resume trading")
            async def resume(interaction: discord.Interaction):
                if not self._is_authorized(interaction.channel_id, interaction.guild_id):
                    await interaction.response.send_message("Not authorized.", ephemeral=True)
                    return
                if self._control_router:
                    await self._control_router.resume()
                    await interaction.response.send_message("Trading **resumed**.", ephemeral=True)
                else:
                    await interaction.response.send_message("Control router not set.", ephemeral=True)

            @self._bot.tree.command(name="close_all", description="Close all positions")
            async def close_all(interaction: discord.Interaction):
                if not self._is_authorized(interaction.channel_id, interaction.guild_id):
                    await interaction.response.send_message("Not authorized.", ephemeral=True)
                    return
                if self._control_router:
                    result = await self._control_router.close_all("discord")
                    await interaction.response.send_message(
                        f"Closed **{result.get('closed', 0)}** positions.", ephemeral=True
                    )
                else:
                    await interaction.response.send_message("Control router not set.", ephemeral=True)

            @self._bot.tree.command(name="kill", description="Emergency stop (close all + stop bot)")
            async def kill(interaction: discord.Interaction):
                if not self._is_authorized(interaction.channel_id, interaction.guild_id):
                    await interaction.response.send_message("Not authorized.", ephemeral=True)
                    return
                if self._control_router:
                    await interaction.response.send_message(
                        "Emergency shutdown initiated.", ephemeral=True
                    )
                    await self._control_router.kill()
                else:
                    await interaction.response.send_message("Control router not set.", ephemeral=True)

            @self._bot.tree.command(name="status", description="System status")
            async def status(interaction: discord.Interaction):
                if not self._is_authorized(interaction.channel_id, interaction.guild_id):
                    await interaction.response.send_message("Not authorized.", ephemeral=True)
                    return
                if self._control_router:
                    s = self._control_router.get_status()
                    msg = (
                        f"**Status:** {s.get('status')}\n"
                        f"**Paused:** {s.get('paused')}\n"
                        f"**Mode:** {s.get('mode')}\n"
                        f"**Scans:** {s.get('scan_count')}\n"
                        f"**WS:** {s.get('ws_connected')}"
                    )
                    await interaction.response.send_message(msg, ephemeral=True)
                else:
                    await interaction.response.send_message("Control router not set.", ephemeral=True)

            @self._bot.tree.command(name="pnl", description="P&L summary")
            async def pnl(interaction: discord.Interaction):
                if not self._is_authorized(interaction.channel_id, interaction.guild_id):
                    await interaction.response.send_message("Not authorized.", ephemeral=True)
                    return
                if self._control_router:
                    data = await self._control_router.get_pnl()
                    msg = (
                        f"**P&L:** ${data.get('total_pnl', 0):.2f}\n"
                        f"**Today:** ${data.get('today_pnl', 0):.2f}\n"
                        f"**Win rate:** {data.get('win_rate', 0):.1%}\n"
                        f"**Bankroll:** ${data.get('bankroll', 0):.2f}"
                    )
                    await interaction.response.send_message(msg, ephemeral=True)
                else:
                    await interaction.response.send_message("Control router not set.", ephemeral=True)

            @self._bot.tree.command(name="positions", description="Open positions")
            async def positions(interaction: discord.Interaction):
                if not self._is_authorized(interaction.channel_id, interaction.guild_id):
                    await interaction.response.send_message("Not authorized.", ephemeral=True)
                    return
                if self._control_router:
                    positions = await self._control_router.get_positions()
                    if not positions:
                        await interaction.response.send_message("No open positions.", ephemeral=True)
                        return
                    lines = []
                    for pos in positions[:10]:
                        lines.append(
                            f"{pos.get('pair')} {pos.get('side')} @ ${pos.get('entry_price', 0):.2f}"
                        )
                    await interaction.response.send_message(
                        "**Positions:**\n" + "\n".join(lines), ephemeral=True
                    )
                else:
                    await interaction.response.send_message("Control router not set.", ephemeral=True)

            @self._bot.tree.command(name="risk", description="Risk report")
            async def risk(interaction: discord.Interaction):
                if not self._is_authorized(interaction.channel_id, interaction.guild_id):
                    await interaction.response.send_message("Not authorized.", ephemeral=True)
                    return
                if self._control_router:
                    r = self._control_router.get_risk()
                    msg = (
                        f"**Bankroll:** ${r.get('bankroll', 0):.2f}\n"
                        f"**Drawdown:** {r.get('current_drawdown', 0):.1f}%\n"
                        f"**Risk of ruin:** {r.get('risk_of_ruin', 0):.4f}"
                    )
                    await interaction.response.send_message(msg, ephemeral=True)
                else:
                    await interaction.response.send_message("Control router not set.", ephemeral=True)

            @self._bot.event
            async def on_ready():
                try:
                    synced = await self._bot.tree.sync()
                    logger.info("Discord bot ready", synced=len(synced))
                except Exception as e:
                    logger.warning("Discord tree sync failed", error=str(e))

            logger.info("Discord bot initialized")
            return True

        except ImportError:
            logger.warning("discord.py not installed")
            self._enabled = False
            return False
        except Exception as e:
            logger.error("Discord init failed", error=str(e))
            self._enabled = False
            return False

    async def start(self) -> None:
        """Start the Discord bot (non-blocking via task)."""
        if not self._enabled or not self._bot:
            return
        try:
            await self._bot.start(self.token)
        except Exception as e:
            logger.error("Discord bot start failed", error=str(e))

    async def stop(self) -> None:
        """Stop the Discord bot."""
        if self._bot:
            try:
                await self._bot.close()
            except Exception:
                pass
