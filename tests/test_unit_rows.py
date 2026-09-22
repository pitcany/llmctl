"""`llmctl status` and the TUI Units screen must describe a unit identically.

They were separate implementations of the same table and had already drifted in
two ways:

* **Probe tri-state collapsed.** `services.backends.probe_openai_v1_models`
  returns ``list[str] | None`` — ``[]`` means "answered, nothing loaded" and
  ``None`` means "could not be reached", which are not the same fact. The Units
  screen carried a private copy that returned ``[]`` for both, so a slow-but-
  healthy unit rendered ``starting?`` in the TUI and ``no`` in `status`, while a
  genuinely idle unit *also* rendered ``starting?`` where `status` said
  ``yes (empty list)``.

* **Phantom env paths.** `resolve_env_file()` always returns a path, even for a
  role llmctl never writes an env file for. `status` was fixed to hide paths
  that do not exist (commit 10d44ac1); the Units screen still showed them, so
  operators edited files nothing reads.

One producer, `services.units.unit_rows`, now feeds both.
"""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from pathlib import Path

from llmctl.config import ManagedUnitConfig, Settings
from llmctl.services.units import unit_rows


class _Resp(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http(payload: object | None = None, *, fail: bool = False):
    """Fake transport: a JSON body, or an unreachable endpoint."""

    def get(url: str, timeout: float):
        if fail:
            raise urllib.error.URLError("refused")
        return _Resp(json.dumps(payload).encode())

    return get


def _settings(tmp_path: Path, env_name: str = "vllm-tp.env") -> Settings:
    s = Settings()
    s.managed_units.vllm_tp = ManagedUnitConfig(
        unit_name="vllm-tp",
        default_port=8003,
        env_file_path=tmp_path / env_name,
    )
    return s


def test_served_models_are_reported(tmp_path) -> None:
    rows = unit_rows(
        _settings(tmp_path), http_get=_http({"data": [{"id": "qwen3.6-27b"}]})
    )
    assert rows[0].served == ["qwen3.6-27b"]


def test_reachable_but_idle_is_an_empty_list_not_none(tmp_path) -> None:
    """The unit answered. It has nothing loaded. That is not 'unreachable'."""
    rows = unit_rows(_settings(tmp_path), http_get=_http({"data": []}))
    assert rows[0].served == []
    assert rows[0].served is not None


def test_unreachable_is_none_not_an_empty_list(tmp_path) -> None:
    """Nobody answered. That is not 'running with nothing loaded'."""
    rows = unit_rows(_settings(tmp_path), http_get=_http(fail=True))
    assert rows[0].served is None


def test_env_file_absence_is_reported(tmp_path) -> None:
    """resolve_env_file() always returns a path; only some of them exist."""
    rows = unit_rows(_settings(tmp_path), http_get=_http(fail=True))
    assert rows[0].env_file_exists is False


def test_env_file_presence_is_reported(tmp_path) -> None:
    (tmp_path / "vllm-tp.env").write_text("VLLM_MODEL=x\n")
    rows = unit_rows(_settings(tmp_path), http_get=_http(fail=True))
    assert rows[0].env_file_exists is True


def test_row_carries_the_identifying_fields(tmp_path) -> None:
    rows = unit_rows(_settings(tmp_path), http_get=_http(fail=True))
    row = rows[0]
    assert row.role == "vllm-tp"
    assert row.unit_name == "vllm-tp"
    assert row.port == 8003


# --- lanes that declare their own unit in router.yaml -------------------------

_LANE_YAML = """
backends:
  - name: glm53
    served_model: glm-5.3-flash
    upstream: http://127.0.0.1:8006
    start_cmd: ["/bin/glm53-start"]
    unit: glm-5.3-flash
    runtime: llama_cpp
  - name: onvllmport
    served_model: shadow
    upstream: http://127.0.0.1:8003
    start_cmd: ["/bin/shadow-start"]
    unit: shadow
  - name: nounit
    served_model: anon
    upstream: http://127.0.0.1:9999
    start_cmd: ["/bin/anon-start"]
"""


def test_a_lane_with_its_own_unit_needs_no_settings_row(tmp_path, monkeypatch) -> None:
    """FreeToken and GLM-5.3 were invisible to `status` for want of that row."""
    router = tmp_path / "router.yaml"
    router.write_text(_LANE_YAML)
    monkeypatch.setenv("LLMCTL_ROUTER_CONFIG", str(router))

    rows = {row.role: row for row in unit_rows(_settings(tmp_path), http_get=_http({"data": []}))}

    assert rows["glm53"].unit_name == "glm-5.3-flash"
    assert rows["glm53"].port == 8006
    # A lane on a port a managed_units row already owns does not duplicate it,
    # and a lane that names no unit cannot be derived at all.
    assert "onvllmport" not in rows
    assert "nounit" not in rows
    assert rows["vllm-tp"].unit_name == "vllm-tp"
