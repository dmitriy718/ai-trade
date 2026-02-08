"""
Slack Control Bot - Remote control via Slack.

Slash commands: /trading-pause, /trading-resume, /trading-close-all,
/trading-kill, /trading-status, /trading-pnl, /trading-positions, /trading-risk.
Verify request with Slack signing secret; restrict to allowed channel.
Uses control router for all control actions.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from src.core.logger import get_logger

logger = get_logger("slack_bot")


class SlackBot:
    """
    Slack bot for remote monitoring and control.
    Uses Socket Mode (app_token) so no public URL is required.
    Commands invoke the control router; auth via allowed_channel_id.
    """

    def __init__(
        self,
        token: str = "",
        signing_secret: str = "",
        app_token: Optional[str] = None,
        allowed_channel_id: Optional[str] = None,
    ):
        self.token = token  # bot token xoxb-...
        self.signing_secret = signing_secret
        self.app_token = app_token or ""  # xapp-... for Socket Mode
        self.allowed_channel_id = str(allowed_channel_id) if allowed_channel_id else None
        self._control_router = None
        self._app = None
        self._handler = None
        self._enabled = bool(self.token and self.app_token)

    def set_control_router(self, router) -> None:
        """Inject the control router for pause/resume/close_all/kill."""
        self._control_router = router

    def _is_authorized(self, channel_id: Optional[str]) -> bool:
        """Return True if channel is in allowlist or no restriction."""
        if not self.allowed_channel_id:
            return True
        return channel_id == self.allowed_channel_id

    async def initialize(self) -> bool:
        """Initialize the Slack app and register slash commands."""
        if not self._enabled:
            logger.info("Slack bot disabled (no token or app_token)")
            return False

        try:
            from slack_bolt import App
            from slack_bolt.adapter.socket_mode import SocketModeHandler

            self._app = App(token=self.token, signing_secret=self.signing_secret or " ")
            router = self._control_router

            @self._app.command("/trading-pause")
            async def pause(ack, command, say):
                ack()
                if not self._is_authorized(command.get("channel_id")):
                    say(text="Not authorized.", channel=command["channel_id"])
                    return
                if router:
                    await router.pause()
                    say(text="Trading *paused*.", channel=command["channel_id"])
                else:
                    say(text="Control router not set.", channel=command["channel_id"])

            @self._app.command("/trading-resume")
            async def resume(ack, command, say):
                ack()
                if not self._is_authorized(command.get("channel_id")):
                    say(text="Not authorized.", channel=command["channel_id"])
                    return
                if router:
                    await router.resume()
                    say(text="Trading *resumed*.", channel=command["channel_id"])
                else:
                    say(text="Control router not set.", channel=command["channel_id"])

            @self._app.command("/trading-close-all")
            async def close_all(ack, command, say):
                ack()
                if not self._is_authorized(command.get("channel_id")):
                    say(text="Not authorized.", channel=command["channel_id"])
                    return
                if router:
                    result = await router.close_all("slack")
                    say(
                        text=f"Closed *{result.get('closed', 0)}* positions.",
                        channel=command["channel_id"],
                    )
                else:
                    say(text="Control router not set.", channel=command["channel_id"])

            @self._app.command("/trading-kill")
            async def kill(ack, command, say):
                ack()
                if not self._is_authorized(command.get("channel_id")):
                    say(text="Not authorized.", channel=command["channel_id"])
                    return
                if router:
                    say(text="Emergency shutdown initiated.", channel=command["channel_id"])
                    await router.kill()
                else:
                    say(text="Control router not set.", channel=command["channel_id"])

            @self._app.command("/trading-status")
            async def status(ack, command, say):
                ack()
                if not self._is_authorized(command.get("channel_id")):
                    say(text="Not authorized.", channel=command["channel_id"])
                    return
                if router:
                    s = router.get_status()
                    msg = (
                        f"*Status:* {s.get('status')}\n"
                        f"*Paused:* {s.get('paused')}\n"
                        f"*Mode:* {s.get('mode')}\n"
                        f"*Scans:* {s.get('scan_count')}\n"
                        f"*WS:* {s.get('ws_connected')}"
                    )
                    say(text=msg, channel=command["channel_id"])
                else:
                    say(text="Control router not set.", channel=command["channel_id"])

            @self._app.command("/trading-pnl")
            async def pnl(ack, command, say):
                ack()
                if not self._is_authorized(command.get("channel_id")):
                    say(text="Not authorized.", channel=command["channel_id"])
                    return
                if router:
                    data = await router.get_pnl()
                    msg = (
                        f"*P&L:* ${data.get('total_pnl', 0):.2f}\n"
                        f"*Today:* ${data.get('today_pnl', 0):.2f}\n"
                        f"*Win rate:* {data.get('win_rate', 0):.1%}\n"
                        f"*Bankroll:* ${data.get('bankroll', 0):.2f}"
                    )
                    say(text=msg, channel=command["channel_id"])
                else:
                    say(text="Control router not set.", channel=command["channel_id"])

            @self._app.command("/trading-positions")
            async def positions(ack, command, say):
                ack()
                if not self._is_authorized(command.get("channel_id")):
                    say(text="Not authorized.", channel=command["channel_id"])
                    return
                if router:
                    positions_list = await router.get_positions()
                    if not positions_list:
                        say(text="No open positions.", channel=command["channel_id"])
                        return
                    lines = [
                        f"{p.get('pair')} {p.get('side')} @ ${p.get('entry_price', 0):.2f}"
                        for p in positions_list[:10]
                    ]
                    say(
                        text="*Positions:*\n" + "\n".join(lines),
                        channel=command["channel_id"],
                    )
                else:
                    say(text="Control router not set.", channel=command["channel_id"])

            @self._app.command("/trading-risk")
            async def risk_cmd(ack, command, say):
                ack()
                if not self._is_authorized(command.get("channel_id")):
                    say(text="Not authorized.", channel=command["channel_id"])
                    return
                if router:
                    r = router.get_risk()
                    msg = (
                        f"*Bankroll:* ${r.get('bankroll', 0):.2f}\n"
                        f"*Drawdown:* {r.get('current_drawdown', 0):.1f}%\n"
                        f"*Risk of ruin:* {r.get('risk_of_ruin', 0):.4f}"
                    )
                    say(text=msg, channel=command["channel_id"])
                else:
                    say(text="Control router not set.", channel=command["channel_id"])

            # SocketModeHandler runs in a thread; we run it in an executor
            self._handler = SocketModeHandler(self._app, self.app_token)
            logger.info("Slack bot initialized")
            return True

        except ImportError:
            logger.warning("slack-bolt not installed")
            self._enabled = False
            return False
        except Exception as e:
            logger.error("Slack init failed", error=str(e))
            self._enabled = False
            return False

    async def start(self) -> None:
        """Start the Slack bot (SocketModeHandler runs in thread)."""
        if not self._enabled or not self._handler:
            return
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._handler.start)
        except Exception as e:
            logger.error("Slack bot start failed", error=str(e))

    async def stop(self) -> None:
        """Stop the Slack bot."""
        if self._handler and hasattr(self._handler, "close"):
            try:
                self._handler.close()
            except Exception:
                pass
