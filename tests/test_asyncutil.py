"""run_sync works with and without a running loop (FIXPLAN Phase 3)."""

from __future__ import annotations

import asyncio

from llmctl._asyncutil import run_sync


async def _answer() -> int:
    await asyncio.sleep(0)
    return 42


def test_run_sync_without_a_loop() -> None:
    assert run_sync(_answer()) == 42


def test_run_sync_inside_a_running_loop() -> None:
    async def caller() -> int:
        # A bare asyncio.run here raises "cannot be called from a running loop".
        return run_sync(_answer())

    assert asyncio.run(caller()) == 42
