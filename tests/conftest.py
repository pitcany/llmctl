"""Keep ``llmctl swap`` tests away from the live machine's lanes.

``llmctl swap`` reads llm-router's router.yaml and STOPS active lane units
(``systemctl --user stop``) before starting vLLM. A test that mocks
``start_vllm_tp`` but not that step would stop the real FreeToken/llama.cpp
servers on a dev box. Pointing the router config at a path that does not
exist gives every test an empty lane list; tests that exercise lanes build
their own. Load-time records go to a temp dir for the same reason.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_live_lanes(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLMCTL_ROUTER_CONFIG", str(tmp_path / "no-router.yaml"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
