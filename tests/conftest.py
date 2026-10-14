"""Keep ``llmctl swap`` tests away from the live machine's lanes and plugins.

``llmctl swap`` lists the lanes the installed plugins describe and STOPS
active lane units (``systemctl --user stop``) before starting vLLM. A test
that mocks ``start_vllm_tp`` but not that step would stop the real
FreeToken/llama.cpp servers on a dev box. Disabling plugin discovery gives
every test an empty lane list and no companions; tests that exercise either
build their own registry. Load-time records go to a temp dir for the same
reason.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_live_lanes(monkeypatch, tmp_path) -> None:
    from llmctl import plugins

    monkeypatch.setenv("LLMCTL_PLUGINS", "none")
    plugins.reset()
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
