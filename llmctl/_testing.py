"""Shared isolation for the Textual TUI tests.

``MissionControlApp`` boots straight onto the Dashboard, whose fetch calls
``_data.get_overview()``. Unisolated, that opens the developer's real registry
at ``~/.local/share/llmctl/llmctl.sqlite3`` and issues live probes to Ollama,
LM Studio, vLLM and the gateway -- so a TUI test reads whatever happens to be
running on the machine, and CI is green for the wrong reason (see commit
7869c398, which fixed one instance of exactly that).

This is a plain helper rather than a ``conftest.py`` fixture on purpose: the
package has no ``conftest.py``, and every test file wires its own dependencies
explicitly. Call :func:`isolate_tui` first in a TUI test, then override any
individual ``_data`` helper the test is actually about -- a later
``monkeypatch.setattr`` wins over the defaults installed here.

It lives *in the package* rather than under ``tests/`` because the monorepo has
its own top-level ``tests/`` directory. Making ``packages/llmctl/tests`` an
importable package (so a sibling helper could be imported from it) gave two
different directories the module name ``tests``, and every file importing the
helper failed to collect with ``No module named 'tests._tui_isolation'`` as
soon as pytest ran from the monorepo root -- which is how CI runs it. An
absolute import from the package has no such collision and works under both
pytest import modes. Nothing here imports pytest; ``monkeypatch`` is supplied
by the caller.

Only the *boundaries* are stubbed: the database is redirected to a throwaway
file and the network/hardware probes are pinned to inert values. Aggregation
logic such as ``get_overview`` still runs for real against the temp database,
so isolating a test does not stop it from exercising the code under test.
"""

from __future__ import annotations

import urllib.error
from pathlib import Path
from typing import Any

from llmctl.config import load_settings
from llmctl.tui import _data

#: Repo configs, so ``load_settings`` never reads the developer's own file.
CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


class _StubHealthService:
    """Stand-in for ``HealthService`` that reports a quiet, healthy stack.

    The real one probes every runtime endpoint over HTTP.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get_health(self) -> dict[str, Any]:
        return {
            "state": "ok",
            "safe_mode": False,
            "gpu_count": 0,
            "nvml_available": False,
            "runtimes": {},
        }


def isolate_tui(monkeypatch, tmp_path, *, db_name: str = "tui.db"):
    """Point the TUI data layer at throwaway state instead of the live stack.

    Returns the ``Settings`` object in use so a caller can adjust it further
    (it is a deep copy; mutating it cannot affect the developer's config).
    """
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(CONFIGS_DIR))
    settings = load_settings().model_copy(deep=True)
    settings.database.url = f"sqlite:///{tmp_path / db_name}"
    monkeypatch.setattr(_data, "load_settings", lambda: settings, raising=False)

    # Network and hardware boundaries. Each of these reaches off-process in
    # production; pinned here so a test never depends on what is running.
    #
    # Backend *detection* is deliberately left real: several tests are about
    # exactly that logic. Only the socket underneath it is closed, so
    # ``detect_backends``/``missing_backends`` still run their PATH checks
    # while the vLLM HTTP probe fails the way it does on a machine with
    # nothing serving.
    #
    # ``urlopen`` is patched rather than the five per-module ``_default_http_get``
    # seams (services.backends, services.validate, adapters.vllm,
    # adapters.vllm_systemd, tui.screens_units): naming them individually leaks
    # the moment a sixth appears, and screens_units already proved that by
    # reaching :8003 through its own copy.
    def _no_http(*args: Any, **kwargs: Any):
        raise urllib.error.URLError("test: HTTP probes are disabled")

    monkeypatch.setattr("urllib.request.urlopen", _no_http, raising=False)
    monkeypatch.setattr(_data, "get_gpu_info", lambda *a, **k: [], raising=False)
    monkeypatch.setattr(_data, "nvml_available", lambda: False, raising=False)
    monkeypatch.setattr(_data, "_probe_gateway", lambda *a, **k: False, raising=False)
    monkeypatch.setattr(_data, "HealthService", _StubHealthService, raising=False)
    monkeypatch.setattr(_data, "get_served_on_tp_unit", lambda: None, raising=False)
    return settings
