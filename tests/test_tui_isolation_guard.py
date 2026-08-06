"""The TUI isolation helper must actually keep tests off the live stack.

Without this guard the isolation in ``llmctl._testing`` is a convention that
decays silently: a helper that stops covering a new network call still looks
like it works, because the test passes either way -- against whatever happens
to be running on the developer's machine.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import pytest

from llmctl._testing import isolate_tui
from llmctl.config import load_settings
from llmctl.services.backends import probe_openai_v1_models
from llmctl.tui import _data


def _live_db_url() -> str:
    """The URL an unisolated test would open."""
    return load_settings().database_url


def test_isolate_tui_redirects_the_database(tmp_path, monkeypatch) -> None:
    """``get_overview`` must open the temp file, never the real registry."""
    live = _live_db_url()
    opened: list[str] = []
    real_get_engine = _data.get_engine

    def spy(url: str, *args, **kwargs):
        opened.append(url)
        return real_get_engine(url, *args, **kwargs)

    isolate_tui(monkeypatch, tmp_path)
    monkeypatch.setattr(_data, "get_engine", spy, raising=False)

    _data.get_overview()

    assert opened, "get_overview opened no database at all"
    assert live not in opened, f"test touched the live registry: {live}"
    assert all(str(tmp_path) in url for url in opened), opened


def test_isolate_tui_writes_nothing_to_the_live_database(tmp_path, monkeypatch) -> None:
    """A belt-and-braces check on the real file's mtime across a TUI read."""
    live_path = Path(_live_db_url().replace("sqlite:///", ""))
    before = live_path.stat().st_mtime_ns if live_path.exists() else None

    isolate_tui(monkeypatch, tmp_path)
    _data.get_overview()

    after = live_path.stat().st_mtime_ns if live_path.exists() else None
    assert before == after, f"the live registry was written during a TUI test: {live_path}"


def test_isolate_tui_pins_the_network_probes(tmp_path, monkeypatch) -> None:
    """Gateway/GPU/backend probes must not reach off-process."""
    isolate_tui(monkeypatch, tmp_path)

    overview = _data.get_overview()

    assert overview["router"]["running"] is False
    assert overview["gpu_count"] == 0
    assert overview["nvml_available"] is False


def test_isolate_tui_blocks_every_http_seam(tmp_path, monkeypatch) -> None:
    """The transport is closed, not just the callers that were known in 2026-08.

    llmctl has five separate ``_default_http_get`` seams; ``screens_units``
    reached :8003 through its own copy until ``urlopen`` itself was pinned.
    Asserting on the transport keeps a sixth seam from reopening the hole.
    """
    isolate_tui(monkeypatch, tmp_path)

    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen("http://127.0.0.1:8003/v1/models", timeout=0.1)

    # And the units screen's private copy resolves to the same blocked call.
    assert probe_openai_v1_models("http://127.0.0.1:8003", 0.1) is None
