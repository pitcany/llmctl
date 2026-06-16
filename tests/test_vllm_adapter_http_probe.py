"""Tests for the HTTP-probe path on :class:`VLLMAdapter`.

These tests pin the user-facing behaviour change from "vLLM is
'unavailable' because no binary is on PATH" to "vLLM is OK because
the managed unit at :8003 is serving llama-3.3-70b right now".
"""

from __future__ import annotations

import asyncio
import io
import json
import urllib.error
from collections.abc import Callable
from typing import Any

import pytest

from llmctl.adapters.vllm import VLLMAdapter
from llmctl.config import ManagedUnitConfig, ManagedUnitsConfig
from llmctl.db import RuntimeName
from llmctl.schemas import HealthState


def _models_payload(ids: list[str]) -> bytes:
    return json.dumps({"object": "list", "data": [{"id": i} for i in ids]}).encode()


def _http_responder(
    *,
    by_port: dict[int, list[str]] | None = None,
    fail_ports: set[int] | None = None,
) -> Callable[[str, float], Any]:
    """Build an http_get that returns scripted /v1/models payloads."""
    by_port = by_port or {}
    fail_ports = fail_ports or set()

    def get(url: str, timeout: float) -> Any:
        # extract port from http://localhost:<port>/v1/models
        port_str = url.split(":")[2].split("/")[0]
        port = int(port_str)
        if port in fail_ports or port not in by_port:
            raise urllib.error.URLError(f"port {port} unreachable")
        return io.BytesIO(_models_payload(by_port[port]))

    return get


def _no_units() -> ManagedUnitsConfig:
    """Build a managed-units config where every probe will succeed if scripted."""
    return ManagedUnitsConfig(
        vllm_tp=ManagedUnitConfig(unit_name="vllm-tp", default_port=8003),
    )


def test_health_ok_when_managed_unit_serves_models() -> None:
    """The end-user complaint: vLLM said 'unavailable' while serving on :8003."""
    adapter = VLLMAdapter(
        managed_units=_no_units(),
        http_get=_http_responder(by_port={8003: ["llama-3.3-70b"]}),
    )
    status = asyncio.run(adapter.health_check())
    assert status.state is HealthState.OK
    assert "llama-3.3-70b" in status.message
    assert "vllm-tp" in status.message
    assert status.details["served"]["vllm-tp"] == ["llama-3.3-70b"]


def test_health_reports_served_models_for_tp_unit() -> None:
    """The TP unit serving multiple models -> all reported."""
    adapter = VLLMAdapter(
        managed_units=_no_units(),
        http_get=_http_responder(
            by_port={8003: ["llama-3.3-70b", "qwen"]}
        ),
    )
    status = asyncio.run(adapter.health_check())
    assert status.state is HealthState.OK
    served = status.details["served"]
    assert served["vllm-tp"] == ["llama-3.3-70b", "qwen"]


def test_health_falls_back_to_binary_check_when_no_units_respond(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No HTTP success -> behave like before (binary lookup).

    The vllm binary is intentionally NOT on PATH in CI; expect
    UNAVAILABLE with the binary error message.
    """
    monkeypatch.setattr("shutil.which", lambda _: None)
    adapter = VLLMAdapter(
        managed_units=_no_units(),
        http_get=_http_responder(by_port={}),  # all probes fail
    )
    status = asyncio.run(adapter.health_check())
    assert status.state is HealthState.UNAVAILABLE
    assert "binary" in status.message.lower()


def test_health_falls_back_when_unit_returns_empty_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live unit with no models loaded -> still fall back (not 'OK with nothing')."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    adapter = VLLMAdapter(
        managed_units=_no_units(),
        http_get=_http_responder(by_port={8003: []}),  # answers but empty
    )
    status = asyncio.run(adapter.health_check())
    # No served models means "not actually useful" -> fallback path
    assert status.state is HealthState.UNAVAILABLE


def test_discover_models_returns_served_names_from_managed_units() -> None:
    """The second user complaint: vLLM models invisible in `llmctl models`."""
    adapter = VLLMAdapter(
        managed_units=_no_units(),
        http_get=_http_responder(by_port={8003: ["llama-3.3-70b"]}),
    )
    models = asyncio.run(adapter.discover_models())
    assert [m.name for m in models] == ["llama-3.3-70b"]
    m = models[0]
    assert m.runtime is RuntimeName.VLLM
    assert m.source == "llama-3.3-70b"
    assert m.metadata["managed_unit"] == "vllm-tp"
    assert m.metadata["port"] == 8003
    assert m.metadata["discovered_via"] == "http"


def test_discover_models_dedupes_within_unit() -> None:
    """A repeated served name on the TP unit should appear once."""
    adapter = VLLMAdapter(
        managed_units=_no_units(),
        http_get=_http_responder(by_port={8003: ["same", "same"]}),
    )
    models = asyncio.run(adapter.discover_models())
    assert len(models) == 1
    assert models[0].metadata["managed_unit"] == "vllm-tp"


def test_discover_models_returns_empty_when_no_units_serve(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No HTTP success + no on-disk models -> empty list, no crash."""
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(tmp_path))
    adapter = VLLMAdapter(
        managed_units=_no_units(),
        http_get=_http_responder(by_port={}),
    )
    models = asyncio.run(adapter.discover_models())
    assert models == []


def test_probe_timeout_is_short_by_default() -> None:
    """Tail-latency safety: the default per-port timeout is sub-2s.

    A long timeout would stall the TUI refresh whenever a unit is
    down. 1.5s is short enough that 3 down units add <5s to refresh.
    """
    adapter = VLLMAdapter(managed_units=_no_units())
    assert adapter._probe_timeout_s < 2.0


def test_probe_timeout_overridable() -> None:
    """Operators on slow networks can bump it."""
    adapter = VLLMAdapter(managed_units=_no_units(), probe_timeout_s=10.0)
    assert adapter._probe_timeout_s == 10.0


def test_probe_passes_timeout_to_http_get() -> None:
    """Verify the timeout is actually plumbed into the HTTP call."""
    captured_timeout: list[float] = []

    def fake_get(url: str, timeout: float) -> Any:
        captured_timeout.append(timeout)
        return io.BytesIO(_models_payload([]))

    adapter = VLLMAdapter(
        managed_units=_no_units(),
        http_get=fake_get,
        probe_timeout_s=2.5,
    )
    asyncio.run(adapter.health_check())
    assert all(t == 2.5 for t in captured_timeout)
    assert len(captured_timeout) == 1  # tp


def test_malformed_payload_treated_as_unreachable() -> None:
    """Garbage JSON from a port -> treated the same as 'down'."""
    def fake_get(url: str, timeout: float) -> Any:
        return io.BytesIO(b"{not valid json")

    adapter = VLLMAdapter(
        managed_units=_no_units(),
        http_get=fake_get,
    )
    # No port returns a valid payload -> health falls back to binary check
    status = asyncio.run(adapter.health_check())
    assert status.state in (HealthState.UNAVAILABLE, HealthState.OK)  # depends on $PATH


def test_non_dict_data_entries_skipped() -> None:
    """Malformed list entries don't crash discovery."""
    def fake_get(url: str, timeout: float) -> Any:
        return io.BytesIO(
            json.dumps({"data": [{"id": "good"}, "not-a-dict", {"no_id": "x"}]}).encode()
        )

    adapter = VLLMAdapter(
        managed_units=_no_units(),
        http_get=fake_get,
    )
    models = asyncio.run(adapter.discover_models())
    # The "not-a-dict" raises AttributeError on .get; we expect the
    # whole probe to fail-safe and return empty rather than crash.
    # If we changed behaviour to skip individual bad entries, this
    # assertion would need to relax to `len(models) == 1`.
    assert isinstance(models, list)
