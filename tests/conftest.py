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


@pytest.fixture(autouse=True)
def _no_live_settings(monkeypatch, tmp_path) -> None:
    """Every test starts from default settings, not the developer's
    ``~/.config/llmctl/settings.yaml``. With ``scheduler.require_auth_token``
    on there, each ``create_app()`` test got 503 "no token configured" on the
    workstation while CI (no settings file) stayed green. Only the default
    location moves (``XDG_CONFIG_HOME``), so tests that pass a path or set
    ``LLMCTL_CONFIG_DIR`` themselves still get theirs."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("LLMCTL_API_TOKEN", raising=False)


@pytest.fixture(autouse=True)
def _isolated_op_lock(monkeypatch, tmp_path) -> None:
    """Point the host-wide GPU operation lock at a per-test tmp path.

    Mutating paths (llmctl run/stop/restart, swaps, API mutations) take an
    flock on this file. Without the override every test would share (and
    could block on) the developer's real lock at
    ``~/.local/share/llmctl/operations.lock``."""
    monkeypatch.setenv("LLMCTL_OP_LOCK", str(tmp_path / "operations.lock"))


@pytest.fixture(autouse=True)
def _no_live_router(monkeypatch) -> None:
    """Never reach the live llm-router, or read its key file, from the suite.

    ``llmctl run``/``vllm`` ask llm-router to swap (``POST /admin/switch``)
    when it answers and a key is found. Only ``test_cli_swap`` stubbed that,
    so any other test that ran those commands on a machine with the router up
    read the real key and could send a real, forced model switch (2026-10-20:
    20 such calls; 2026-11-11: found again in ``test_cli_vllm_readiness``).
    Router-path tests re-enable both names explicitly."""
    from llmctl import cli_swap

    monkeypatch.delenv("ROUTER_API_KEY", raising=False)
    monkeypatch.setattr(cli_swap, "router_key", lambda env_file: None)
    monkeypatch.setattr(cli_swap, "router_up", lambda url: False)
