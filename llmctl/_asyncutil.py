"""Run a coroutine from sync code, whether or not an event loop is running.

Service methods (health probes, registry scan, runtime inspection, session
start/stop) are sync and drive async adapters. A bare ``asyncio.run`` raises
``RuntimeError`` when the caller is already inside an event loop -- an
``async def`` API route or a Textual handler on the UI thread. :func:`run_sync`
uses ``asyncio.run`` normally and, only when a loop is already running on this
thread, runs the coroutine to completion on a short-lived worker thread.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import Any


def run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` to completion and return its result."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
