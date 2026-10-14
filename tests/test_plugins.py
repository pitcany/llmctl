"""The plugin seam: discovery, isolation from a broken plugin, the built-in
systemd companion, and the orchestrator picking up what plugins registered."""

from __future__ import annotations

from typing import Any

import pytest

from llmctl import plugins
from llmctl.plugins import PluginRegistry, SystemdCompanion, load_plugins


@pytest.fixture(autouse=True)
def _fresh_registry() -> Any:
    plugins.reset()
    yield
    plugins.reset()


class _EntryPoint:
    def __init__(self, name: str, target: Any) -> None:
        self.name = name
        self._target = target

    def load(self) -> Any:
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


def test_plugins_register_into_every_extension_point() -> None:
    class Companion:
        name = "box"
        start_hint = "docker start box"

        def stop(self, *, logger: Any) -> bool | None:
            return True

        def start(self, *, logger: Any) -> bool:
            return True

        def describe(self) -> str:
            return "stops box"

    def register(reg: PluginRegistry) -> None:
        reg.companions.append(Companion())
        reg.post_start_checks.append(type("C", (), {"name": "c"})())
        reg.lane_sources.append(lambda units: [])
        reg.commands.append(lambda app: None)

    reg = load_plugins(eps=[_EntryPoint("stack", register)])

    assert reg.loaded == ["stack"]
    assert [c.name for c in reg.companions] == ["box"]
    assert len(reg.post_start_checks) == len(reg.lane_sources) == len(reg.commands) == 1
    assert plugins.registry() is reg  # cached


def test_a_broken_plugin_is_skipped_and_reported_not_fatal() -> None:
    reported: list[str] = []

    def good(reg: PluginRegistry) -> None:
        reg.commands.append(lambda app: None)

    reg = load_plugins(
        eps=[
            _EntryPoint("bad", ImportError("no module named nothing")),
            _EntryPoint("good", good),
        ],
        stderr=reported.append,
    )

    assert reg.loaded == ["good"]
    assert reg.failed == [("bad", "ImportError: no module named nothing")]
    assert len(reg.commands) == 1
    assert reported and "bad" in reported[0]


def test_llmctl_plugins_none_disables_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMCTL_PLUGINS", "none")
    calls: list[str] = []
    reg = load_plugins(eps=[_EntryPoint("stack", lambda r: calls.append("loaded"))])
    assert calls == [] and reg.loaded == []


class _Systemctl:
    def __init__(self, active: bool) -> None:
        self.active = active
        self.calls: list[tuple[str, str, bool]] = []

    def is_active(self, unit: str, *, user: bool = False) -> bool:
        self.calls.append(("is-active", unit, user))
        return self.active

    def stop(self, unit: str, *, user: bool = False) -> Any:
        self.calls.append(("stop", unit, user))
        return type("R", (), {"ok": True, "stdout": "", "stderr": ""})()

    def start(self, unit: str, *, user: bool = False) -> Any:
        self.calls.append(("start", unit, user))
        return type("R", (), {"ok": True})()


def test_systemd_companion_stops_only_an_active_unit() -> None:
    idle = _Systemctl(active=False)
    assert SystemdCompanion("speech", systemctl=idle).stop(logger=lambda _: None) is None
    assert ("stop", "speech", False) not in idle.calls

    busy = _Systemctl(active=True)
    companion = SystemdCompanion("speech", user=True, systemctl=busy)
    assert companion.stop(logger=lambda _: None) is True
    assert ("stop", "speech", True) in busy.calls
    assert companion.start(logger=lambda _: None) is True
    assert companion.start_hint == "systemctl --user start speech"
    assert SystemdCompanion("speech").start_hint == "sudo systemctl start speech"


def test_orchestrator_defaults_to_the_registered_companions(tmp_path: Any) -> None:
    """``Dependencies()`` without an explicit list uses whatever plugins gave."""
    from llmctl.services.vllm_orchestrator import Dependencies, _registered_companions

    class Companion:
        name = "box"
        start_hint = "docker start box"

        def stop(self, *, logger: Any) -> bool | None:
            return None

        def start(self, *, logger: Any) -> bool:
            return True

        def describe(self) -> str:
            return "stops box"

    load_plugins(eps=[_EntryPoint("stack", lambda r: r.companions.append(Companion()))])
    assert Dependencies().companions is None
    assert [c.name for c in _registered_companions()] == ["box"]
