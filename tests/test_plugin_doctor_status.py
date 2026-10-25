"""Plugin doctor checks and status sections.

A host-specific plugin (a macOS llama-swap one, say) has its own health checks
and its own picture of what is loaded. They join ``llmctl doctor`` and
``llmctl status`` so the operator keeps one checkup and one status view.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from llmctl import plugins
from llmctl.plugins import PluginRegistry, load_plugins

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(CONFIGS))
    monkeypatch.setenv("LLMCTL_DB_URL", f"sqlite:///{tmp_path / 'cli.sqlite3'}")
    monkeypatch.delenv("LLMCTL_PLUGINS", raising=False)
    monkeypatch.setattr("llmctl.services.units.probe_openai_v1_models", lambda *_a, **_k: None)
    plugins.reset()
    yield
    plugins.reset()


class _EntryPoint:
    def __init__(self, name: str, target: Any) -> None:
        self.name = name
        self._target = target

    def load(self) -> Any:
        return self._target


def _install(register: Any) -> None:
    # Importing llmctl.cli loads (and caches) the registry for plugin commands.
    plugins.reset()
    load_plugins(eps=[_EntryPoint("mac", register)])


def _all_checks(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in report["passed"] + report["warnings"] + report["failures"]}


def test_plugin_doctor_checks_join_the_report() -> None:
    from llmctl.config import load_settings
    from llmctl.services.doctor import DoctorCheck, run_doctor

    seen: list[Any] = []

    def router_check(settings: Any) -> list[DoctorCheck]:
        seen.append(settings)
        return [DoctorCheck("llama-swap", "warn", "router not answering", "start it")]

    _install(lambda reg: reg.doctor_checks.append(router_check))
    settings = load_settings()
    report = run_doctor(settings)

    check = _all_checks(report)["llama-swap"]
    assert check["verdict"] == "warn"
    assert check["remediation"] == "start it"
    assert "database" in _all_checks(report)  # the built-in checks still run
    assert seen == [settings]


def test_a_crashing_plugin_check_is_a_failure_not_a_crash() -> None:
    from llmctl.config import load_settings
    from llmctl.services.doctor import run_doctor

    def broken(settings: Any) -> list[Any]:
        raise RuntimeError("bad registry file")

    _install(lambda reg: reg.doctor_checks.append(broken))
    report = run_doctor(load_settings())

    failure = _all_checks(report)["doctor:broken#0"]
    assert failure["verdict"] == "fail"
    assert "bad registry file" in failure["detail"]


def test_a_plugin_check_returning_the_wrong_type_is_a_failure() -> None:
    from llmctl.config import load_settings
    from llmctl.services.doctor import run_doctor

    def register(reg: PluginRegistry) -> None:
        reg.doctor_checks.append(lambda settings: [{"name": "x", "verdict": "pass"}])
        reg.doctor_checks.append(lambda settings: 1 / 0)

    _install(register)
    report = run_doctor(load_settings())

    checks = _all_checks(report)
    assert "not DoctorCheck" in checks["doctor:<lambda>#0"]["detail"]
    assert checks["doctor:<lambda>#1"]["verdict"] == "fail"  # two lambdas, two entries


def test_plugin_status_sections_print_after_the_units_table() -> None:
    from llmctl.cli import app

    _install(lambda reg: reg.status_sections.append(lambda: "llama-swap: 2 lanes loaded"))
    out = CliRunner().invoke(app, ["status"])

    assert out.exit_code == 0, out.output
    assert out.output.index("Managed units") < out.output.index("llama-swap: 2 lanes loaded")


def test_plugin_status_sections_stay_out_of_the_json() -> None:
    from llmctl.cli import app

    _install(lambda reg: reg.status_sections.append(lambda: "llama-swap: 2 lanes loaded"))
    out = CliRunner().invoke(app, ["status", "--json"])

    assert out.exit_code == 0, out.output
    assert isinstance(json.loads(out.output), list)
    assert "llama-swap" not in out.output


def test_a_crashing_status_section_is_reported_not_fatal() -> None:
    from llmctl.cli import app

    def broken() -> str:
        raise RuntimeError("router down")

    def register(reg: PluginRegistry) -> None:
        reg.status_sections.append(broken)
        reg.status_sections.append(lambda: "second section")

    _install(register)
    out = CliRunner().invoke(app, ["status"])

    assert out.exit_code == 0, out.output
    assert "router down" in out.output
    assert "second section" in out.output
