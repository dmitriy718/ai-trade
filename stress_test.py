#!/usr/bin/env python3
"""
72-Hour Stress Tester — Non-disruptive observation-only resilience monitor.

Periodically checks that the bot is alive, API responds, data is fresh,
and WebSocket streams work. Does NOT restart containers, kill processes,
or interfere with the running bot in any way.
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


class StressTester:
    DASHBOARD_URL = "http://localhost:8080"

    def __init__(self, duration_hours: float = 72, interval_minutes: int = 5):
        self.duration_hours = duration_hours
        self.interval_minutes = interval_minutes
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.events = []
        self._running = True

    async def run(self):
        print(f"\n{'='*50}")
        print(f"  STRESS MONITOR — {self.duration_hours}h, every {self.interval_minutes}m")
        print(f"  Non-disruptive: observe only, no interference")
        print(f"{'='*50}\n")

        end_time = time.time() + self.duration_hours * 3600

        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, self._stop)
        loop.add_signal_handler(signal.SIGTERM, self._stop)

        iteration = 0
        while time.time() < end_time and self._running:
            iteration += 1
            elapsed_h = (time.time() - (end_time - self.duration_hours * 3600)) / 3600
            print(f"\n--- Iteration #{iteration} ({elapsed_h:.1f}h / {self.duration_hours}h) ---")

            await self._test_api_health()
            await self._test_data_freshness()
            await self._test_websocket()
            await self._test_trading_activity()

            # Sleep in small chunks for responsive shutdown
            sleep_secs = self.interval_minutes * 60
            wake = time.time() + sleep_secs
            while time.time() < wake and self._running:
                await asyncio.sleep(min(5, wake - time.time()))

        self._report()

    def _stop(self):
        self._running = False

    async def _test_api_health(self):
        """Check all API endpoints respond 200."""
        self.tests_run += 1
        endpoints = [
            "/api/v1/status", "/api/v1/performance", "/api/v1/positions",
            "/api/v1/strategies", "/api/v1/risk", "/api/v1/thoughts",
            "/api/v1/scanner",
        ]
        try:
            import httpx
            all_ok = True
            async with httpx.AsyncClient(timeout=5) as client:
                for ep in endpoints:
                    try:
                        r = await client.get(f"{self.DASHBOARD_URL}{ep}")
                        if r.status_code != 200:
                            print(f"  FAIL {ep}: HTTP {r.status_code}")
                            all_ok = False
                    except Exception as e:
                        print(f"  FAIL {ep}: {e}")
                        all_ok = False

            if all_ok:
                self.tests_passed += 1
                print(f"  PASS: All {len(endpoints)} API endpoints OK")
            else:
                self.tests_failed += 1
                self._log("api_health", "Some endpoints failed", False)
        except ImportError:
            print("  SKIP: httpx not installed")
        except Exception as e:
            self.tests_failed += 1
            print(f"  FAIL: API health check error: {e}")

    async def _test_data_freshness(self):
        """Check that market data is not stale."""
        self.tests_run += 1
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.DASHBOARD_URL}/api/v1/scanner")
                scanner = r.json()
                stale = [p for p, d in scanner.items() if d.get("stale")]
                fresh = [p for p, d in scanner.items() if not d.get("stale")]

                if len(fresh) >= len(scanner) / 2:
                    self.tests_passed += 1
                    print(f"  PASS: {len(fresh)}/{len(scanner)} pairs fresh, {len(stale)} stale")
                else:
                    self.tests_failed += 1
                    print(f"  FAIL: Too many stale pairs: {stale}")
        except Exception as e:
            self.tests_failed += 1
            print(f"  FAIL: Data freshness error: {e}")

    async def _test_websocket(self):
        """Check WebSocket connects and receives data."""
        self.tests_run += 1
        try:
            import websockets
            ws_url = self.DASHBOARD_URL.replace("http", "ws") + "/ws/live"
            async with websockets.connect(ws_url) as ws:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(msg)
                if data.get("type") == "update":
                    self.tests_passed += 1
                    print(f"  PASS: WebSocket received update")
                else:
                    self.tests_failed += 1
                    print(f"  FAIL: Unexpected WS message type: {data.get('type')}")
        except Exception as e:
            self.tests_failed += 1
            print(f"  FAIL: WebSocket error: {e}")

    async def _test_trading_activity(self):
        """Check the bot is actually scanning and trading."""
        self.tests_run += 1
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.DASHBOARD_URL}/api/v1/status")
                status = r.json()
                scan = status.get("scan_count", 0)
                ws = status.get("ws_connected", False)

                r2 = await client.get(f"{self.DASHBOARD_URL}/api/v1/performance")
                perf = r2.json()
                trades = perf.get("total_trades", 0)
                open_pos = perf.get("open_positions", 0)
                equity = perf.get("total_equity", 0)

                print(f"  INFO: Scan#{scan} | WS:{ws} | Trades:{trades} | Open:{open_pos} | Equity:${equity:.2f}")

                if scan > 0 and ws:
                    self.tests_passed += 1
                    print(f"  PASS: Bot is active")
                else:
                    self.tests_failed += 1
                    print(f"  FAIL: Bot appears inactive (scan={scan}, ws={ws})")
        except Exception as e:
            self.tests_failed += 1
            print(f"  FAIL: Activity check error: {e}")

    def _log(self, test: str, msg: str, passed: bool):
        self.events.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "test": test, "message": msg, "passed": passed,
        })

    def _report(self):
        total = self.tests_run
        rate = self.tests_passed / total * 100 if total > 0 else 0
        report = {
            "tests_run": total,
            "passed": self.tests_passed,
            "failed": self.tests_failed,
            "pass_rate": round(rate, 1),
        }
        Path("data").mkdir(exist_ok=True)
        Path("data/stress_test_report.json").write_text(json.dumps(report, indent=2))

        print(f"\n{'='*50}")
        print(f"  STRESS TEST REPORT")
        print(f"  Tests: {total} | Pass: {self.tests_passed} | Fail: {self.tests_failed} | Rate: {rate:.1f}%")
        print(f"  Report saved to data/stress_test_report.json")
        print(f"{'='*50}\n")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=72.0)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.hours = 5 / 60
        args.interval = 1

    t = StressTester(args.hours, args.interval)
    await t.run()
    sys.exit(0 if t.tests_failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
