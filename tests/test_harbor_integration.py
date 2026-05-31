"""Unit tests for :mod:`llmctl.integrations.harbor`.

Harbor integration is optional — when Docker is missing or the named
container isn't running, every function returns the appropriate
"unavailable" value and does nothing. These tests pin that contract
plus the pin-validation logic used to warn before a swap.
"""

from __future__ import annotations

import io
import subprocess

import pytest

from llmctl.integrations.harbor import (
    DEFAULT_OLLAMA_CONTAINER,
    DEFAULT_WEBUI_CONTAINER,
    StopOutcome,
    WebUIPin,
    find_orphaned_pins,
    is_docker_available,
    served_model_ids,
    stop_ollama_container,
    webui_custom_models,
)


def _completed(
    stdout: str = "",
    returncode: int = 0,
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    """Build a :class:`subprocess.CompletedProcess` for fake runners."""
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class _Runner:
    """Scripted subprocess runner — first match wins."""

    def __init__(self, scripts: dict[str, subprocess.CompletedProcess[str]]) -> None:
        self.scripts = scripts
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        key = " ".join(argv[:3])  # e.g. "docker inspect -f"
        for prefix, result in self.scripts.items():
            if key.startswith(prefix) or " ".join(argv[:2]).startswith(prefix):
                return result
        return _completed()


def test_docker_missing_short_circuits_stop() -> None:
    """No docker on PATH -> DOCKER_MISSING, runner never invoked."""
    runner = _Runner({})
    outcome = stop_ollama_container(
        runner=runner,
        which=lambda _: None,
        logger=lambda _: None,
    )
    assert outcome is StopOutcome.DOCKER_MISSING
    assert runner.calls == []


def test_container_not_running_returns_not_running() -> None:
    """docker inspect says false -> NOT_RUNNING, no stop attempted."""
    runner = _Runner({
        "docker inspect": _completed(stdout="false\n"),
    })
    outcome = stop_ollama_container(
        runner=runner,
        which=lambda _: "/usr/bin/docker",
        logger=lambda _: None,
    )
    assert outcome is StopOutcome.NOT_RUNNING
    # inspect was called; stop was not
    assert any("inspect" in c[1] for c in runner.calls)
    assert not any("stop" in c[1] for c in runner.calls)


def test_running_container_is_stopped() -> None:
    """docker inspect=true -> STOPPED after docker stop succeeds."""
    logged: list[str] = []
    runner = _Runner({
        "docker inspect": _completed(stdout="true\n"),
        "docker stop": _completed(),
    })
    outcome = stop_ollama_container(
        runner=runner,
        which=lambda _: "/usr/bin/docker",
        logger=logged.append,
    )
    assert outcome is StopOutcome.STOPPED
    assert any(DEFAULT_OLLAMA_CONTAINER in line for line in logged)
    assert any("freed GPU memory" in line for line in logged)


def test_stop_failure_returns_failed() -> None:
    """Non-zero exit from docker stop -> FAILED with stderr surfaced."""
    logged: list[str] = []
    runner = _Runner({
        "docker inspect": _completed(stdout="true\n"),
        "docker stop": _completed(returncode=1, stderr="permission denied\n"),
    })
    outcome = stop_ollama_container(
        runner=runner,
        which=lambda _: "/usr/bin/docker",
        logger=logged.append,
    )
    assert outcome is StopOutcome.FAILED
    assert any("permission denied" in line for line in logged)


def test_webui_custom_models_returns_none_when_unavailable() -> None:
    """When the webui container isn't running, return None silently."""
    runner = _Runner({
        "docker inspect": _completed(stdout="false\n"),
    })
    assert (
        webui_custom_models(runner=runner, which=lambda _: "/usr/bin/docker")
        is None
    )


def test_webui_custom_models_parses_pipe_separated_rows() -> None:
    """sqlite output is `id|name|base_model_id` per row."""
    sqlite_out = (
        "llama-tools|Llama Tools|llama-3.3-70b\n"
        "llama-code-review|Llama CR|llama-3.3-70b\n"
        "no-base-model|Floating|\n"
    )
    runner = _Runner({
        "docker inspect": _completed(stdout="true\n"),
        "docker exec": _completed(stdout=sqlite_out),
    })
    pins = webui_custom_models(runner=runner, which=lambda _: "/usr/bin/docker")
    assert pins == [
        WebUIPin("llama-tools", "Llama Tools", "llama-3.3-70b"),
        WebUIPin("llama-code-review", "Llama CR", "llama-3.3-70b"),
        WebUIPin("no-base-model", "Floating", ""),
    ]


def test_webui_custom_models_ignores_malformed_rows() -> None:
    """Rows missing the second/third pipe should be skipped, not crash."""
    runner = _Runner({
        "docker inspect": _completed(stdout="true\n"),
        "docker exec": _completed(stdout="good|row|base\nbad-row\n|partial|\n"),
    })
    pins = webui_custom_models(runner=runner, which=lambda _: "/usr/bin/docker")
    assert pins is not None
    aliases = [p.id for p in pins]
    assert "good" in aliases
    # "bad-row" has no pipes -> skipped
    assert "bad-row" not in aliases


def test_find_orphaned_pins_matches_next_served_name() -> None:
    """Pins whose base_model_id matches the next served name are not orphaned."""
    runner = _Runner({
        "docker inspect": _completed(stdout="true\n"),
        "docker exec": _completed(stdout=(
            "tools|Tools|llama-3.3-70b\n"
            "review|Review|qwen2.5-coder-32b\n"
            "no-pin|Floating|\n"
        )),
    })
    orphans = find_orphaned_pins(
        "llama-3.3-70b",
        runner=runner,
        which=lambda _: "/usr/bin/docker",
    )
    assert orphans is not None
    aliases = sorted(p.id for p in orphans)
    # tools points at the next served name -> not orphaned
    # review points elsewhere -> orphaned
    # no-pin has empty base -> not orphaned (doesn't pin anything)
    assert aliases == ["review"]


def test_find_orphaned_pins_returns_none_when_webui_unavailable() -> None:
    """Missing WebUI surfaces as None so the caller skips the check."""
    runner = _Runner({"docker inspect": _completed(stdout="false\n")})
    assert (
        find_orphaned_pins(
            "any",
            runner=runner,
            which=lambda _: "/usr/bin/docker",
        )
        is None
    )


def test_served_model_ids_returns_set_from_v1_models() -> None:
    """``/v1/models`` JSON payload -> set of `id` strings."""
    def fake_get(url: str, timeout: float):
        assert url == "http://localhost:8003/v1/models"
        return io.BytesIO(
            b'{"data":[{"id":"llama-3.3-70b","object":"model"},'
            b'{"id":"another","object":"model"}]}'
        )

    ids = served_model_ids(8003, http_get=fake_get)
    assert ids == {"llama-3.3-70b", "another"}


def test_served_model_ids_returns_none_on_connection_error() -> None:
    """Unreachable endpoint -> None (treated as "service down")."""
    import urllib.error

    def fake_get(url: str, timeout: float):
        raise urllib.error.URLError("connection refused")

    assert served_model_ids(8003, http_get=fake_get) is None


def test_served_model_ids_returns_empty_set_on_loaded_zero_models() -> None:
    """Healthy server with no models loaded -> empty set, not None."""
    def fake_get(url: str, timeout: float):
        return io.BytesIO(b'{"data": []}')

    assert served_model_ids(8003, http_get=fake_get) == set()


def test_served_model_ids_skips_non_string_ids() -> None:
    """Malformed payload entries are silently skipped."""
    def fake_get(url: str, timeout: float):
        return io.BytesIO(
            b'{"data":[{"id":"good"},{"id":42},{"no_id":"x"}]}'
        )

    assert served_model_ids(8003, http_get=fake_get) == {"good"}


def test_is_docker_available_proxies_to_which() -> None:
    assert is_docker_available(lambda _: None) is False
    assert is_docker_available(lambda _: "/usr/bin/docker") is True


def test_custom_container_name_respected() -> None:
    """Tests reuse the same docker plumbing for a non-default container."""
    runner = _Runner({"docker inspect": _completed(stdout="false\n")})
    outcome = stop_ollama_container(
        container="my.ollama",
        runner=runner,
        which=lambda _: "/usr/bin/docker",
        logger=lambda _: None,
    )
    assert outcome is StopOutcome.NOT_RUNNING
    # inspect call carried the custom name
    assert any("my.ollama" in c for c in runner.calls[0])


def test_default_containers_are_stable() -> None:
    """Constants are part of the public API — pin their values."""
    assert DEFAULT_OLLAMA_CONTAINER == "harbor.ollama"
    assert DEFAULT_WEBUI_CONTAINER == "harbor.webui"


@pytest.mark.parametrize("inspect_stdout,expected", [
    ("true\n", True),
    ("false\n", False),
    ("", False),
    ("invalid output\n", False),
])
def test_container_state_parsing(inspect_stdout: str, expected: bool) -> None:
    """``docker inspect`` output parsing is exact-match on `true`."""
    runner = _Runner({"docker inspect": _completed(stdout=inspect_stdout)})
    outcome = stop_ollama_container(
        runner=runner,
        which=lambda _: "/usr/bin/docker",
        logger=lambda _: None,
    )
    if expected:
        # Would have proceeded to stop (which we didn't script -> default no-op success)
        assert outcome is StopOutcome.STOPPED
    else:
        assert outcome is StopOutcome.NOT_RUNNING
