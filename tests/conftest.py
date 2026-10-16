"""Keep ``llmctl run`` tests away from the live machine's lanes and plugins.

``llmctl run`` lists the lanes the installed plugins describe and STOPS
active lane units (``systemctl --user stop``) before starting vLLM. A test
that mocks ``start_vllm_tp`` but not that step would stop the real
FreeToken/llama.cpp servers on a dev box. Disabling plugin discovery gives
every test an empty lane list and no companions; tests that exercise either
build their own registry. Load-time records go to a temp dir for the same
reason.
"""

from __future__ import annotations

import os

import pytest

# Done at import time, before any test module imports llmctl.cli: the module-
# level Rich consoles read FORCE_COLOR when they are constructed, so a fixture
# would be too late for them.
os.environ.pop("FORCE_COLOR", None)
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"


@pytest.fixture(autouse=True)
def _plain_terminal(monkeypatch) -> None:
    """Rich/Typer must print plain text: tests assert on substrings of CLI
    output, and `FORCE_COLOR` (set on some shells and CI runners) fills it
    with escape codes that split the words apart."""
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("_TYPER_FORCE_DISABLE_TERMINAL", "1")


@pytest.fixture(autouse=True)
def _no_live_lanes(monkeypatch, tmp_path) -> None:
    from llmctl import plugins

    monkeypatch.setenv("LLMCTL_PLUGINS", "none")
    plugins.reset()
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


@pytest.fixture(autouse=True)
def _no_live_registry(monkeypatch, tmp_path) -> None:
    """The registry database lives in the data dir (``~/.local/share/llmctl``).
    CLI tests such as ``scan --import`` and ``sessions`` write to it, so
    without this every test run imported scanned models into the developer's
    real registry, and failed where that directory is read-only (a sandbox).
    Tests that need a specific database still set ``LLMCTL_DB_URL``."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.delenv("LLMCTL_DB_URL", raising=False)
