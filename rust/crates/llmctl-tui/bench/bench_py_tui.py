"""Headless latency of the Python TUI (llmctl tui), same metrics as
`llmctl-next tui --bench`: key press -> frame idle, and dashboard data
refresh (request -> rendered). Read-only: screen switches and refreshes.

Run: ~/AI/.venv/bin/python -I bench_py_tui.py ROUNDS
"""
import asyncio
import math
import sys
import time

from llmctl.tui.app import MissionControlApp
from llmctl.tui.screens_dashboard import DashboardScreen

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
KEYS = ["d", "g", "l", "s", "u"]  # Dashboard, GPUs, Events, Sessions, Units


def pct(xs):
    xs = sorted(xs)
    rank = lambda p: xs[max(1, math.ceil(p * len(xs))) - 1]
    return rank(0.50), rank(0.95), xs[-1]


async def main():
    rendered = asyncio.Event()
    orig = DashboardScreen.render_data

    def spy(self, data):
        orig(self, data)
        rendered.set()

    DashboardScreen.render_data = spy
    app = MissionControlApp()
    async with app.run_test(size=(160, 50)) as pilot:
        # First dashboard load.
        await asyncio.wait_for(rendered.wait(), 30)
        key_ms = []
        for _ in range(ROUNDS):
            for k in KEYS:
                t = time.perf_counter()
                await pilot.press(k)
                await pilot.pause()
                key_ms.append((time.perf_counter() - t) * 1000)
                await asyncio.sleep(0.04)
        await pilot.press("d")
        await pilot.pause()
        refresh_ms = []
        for _ in range(min(ROUNDS, 20)):
            rendered.clear()
            t = time.perf_counter()
            await pilot.press("r")
            await asyncio.wait_for(rendered.wait(), 30)
            await pilot.pause()
            refresh_ms.append((time.perf_counter() - t) * 1000)
    k50, k95, kmax = pct(key_ms)
    r50, r95, rmax = pct(refresh_ms)
    print(f"key to frame   n={len(key_ms):<4} p50 {k50:.2f} ms  p95 {k95:.2f} ms  max {kmax:.2f} ms")
    print(f"dash refresh   n={len(refresh_ms):<4} p50 {r50:.2f} ms  p95 {r95:.2f} ms  max {rmax:.2f} ms")


asyncio.run(main())
