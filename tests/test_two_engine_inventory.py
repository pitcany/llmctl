"""Both configured vLLM engines must appear in existing operator views."""

from __future__ import annotations

import io
import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from llmctl.adapters.vllm import VLLMAdapter
from llmctl.config import ManagedUnitConfig, ManagedUnitsConfig, Settings
from llmctl.integrations.systemctl import SystemctlRunner
from llmctl.schemas import HealthState
from llmctl.services.engine_load import endpoints
from llmctl.services.systemd import install_unit, render_vllm_unit
from llmctl.services.validate import check_managed_unit_ports


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.managed_units.vllm_engine.env_file_path = tmp_path / "main.env"
    settings.managed_units.vllm_engine2 = ManagedUnitConfig(
        unit_name="vllm-engine2", default_port=8010, user_scope=True,
        env_file_path=tmp_path / "second.env",
    )
    settings.managed_units.units["other-engine"] = ManagedUnitConfig(
        unit_name="llama-server", default_port=8100, runtime="llama_cpp",
    )
    return settings


def _http_get(url: str, timeout: float) -> io.BytesIO:
    port = urlsplit(url).port
    if port == 8003:
        raise OSError("main engine is stopped")
    assert port == 8010, "vLLM inventory must not probe another engine's unit"
    body = {"version": "test-version"} if url.endswith("/version") else {
        "data": [{"id": "second-model"}],
    }
    return io.BytesIO(json.dumps(body).encode())


def test_status_includes_second_engine_and_load_inventory_has_no_duplicates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import llmctl.cli as cli

    settings = _settings(tmp_path)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(
        "llmctl.services.units.probe_openai_v1_models",
        lambda url, *_: ["second-model"] if ":8010" in url else None,
    )
    result = CliRunner().invoke(cli.app, ["status", "--json"])
    assert result.exit_code == 0, result.output
    rows = {row["role"]: row for row in json.loads(result.output)}
    assert rows["vllm-engine2"]["served_models"] == ["second-model"]
    assert rows["vllm-engine2"]["port"] == 8010
    assert sum(row.upstream.endswith(":8010") for row in endpoints(settings, [])) == 1


@pytest.mark.parametrize("reserved", ["vllm-engine2", "vllm_engine2"])
def test_second_engine_role_cannot_be_shadowed(reserved: str) -> None:
    with pytest.raises(ValidationError, match="reserved role"):
        ManagedUnitsConfig(units={reserved: ManagedUnitConfig()})


async def test_second_engine_health_and_loaded_models_when_main_is_down(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    adapter = VLLMAdapter(managed_units=settings.managed_units, http_get=_http_get)
    health = await adapter.health_check()
    assert health.state == HealthState.OK
    assert health.details["served"] == {"vllm-engine2": ["second-model"]}
    loaded = await adapter.list_loaded_models()
    assert loaded is not None
    assert [(model.name, model.metadata["port"]) for model in loaded] == [
        ("second-model", 8010),
    ]
    assert await adapter.version() == "test-version"


async def test_discovery_includes_both_vllm_engines(tmp_path: Path, monkeypatch) -> None:
    from llmctl.adapters._common import ProcessRuntimeAdapter

    async def no_filesystem_models(self):
        return []

    monkeypatch.setattr(ProcessRuntimeAdapter, "discover_models", no_filesystem_models)

    def get(url: str, timeout: float) -> io.BytesIO:
        port = urlsplit(url).port
        assert port in {8003, 8010}
        return io.BytesIO(json.dumps({"data": [{"id": f"model-{port}"}]}).encode())

    adapter = VLLMAdapter(managed_units=_settings(tmp_path).managed_units, http_get=get)
    models = await adapter.discover_models()
    assert {model.name for model in models} == {"model-8003", "model-8010"}


def test_port_validation_uses_user_manager_for_second_engine(tmp_path: Path) -> None:
    from types import SimpleNamespace

    calls: list[list[str]] = []

    def run(argv: list[str]) -> SimpleNamespace:
        calls.append(argv)
        return SimpleNamespace(
            returncode=0, stdout="active\n" if "--user" in argv else "inactive\n", stderr="",
        )

    runner = SystemctlRunner(runner=run)
    runner.available = lambda: True
    unit = _settings(tmp_path).managed_units.vllm_engine2
    assert unit is not None

    def unreachable(url: str, timeout: float) -> io.BytesIO:
        raise OSError("second engine has lost its endpoint")

    findings = check_managed_unit_ports([unit], systemctl=runner, http_get=unreachable)
    assert [finding.target for finding in findings] == ["vllm-engine2"]
    assert all("--user" in call for call in calls)


def test_second_engine_unit_preview_preserves_user_scope(tmp_path: Path) -> None:
    unit_config = _settings(tmp_path).managed_units.vllm_engine2
    assert unit_config is not None
    unit = render_vllm_unit(unit_config, launcher=tmp_path / "vllm-launcher.sh")
    assert unit.user is True
    assert "\nUser=" not in unit.content
    assert "WantedBy=default.target" in unit.content
    report = install_unit(unit, dry_run=True)
    assert "/.config/systemd/user/" in report.unit_path
    assert all("sudo" not in action for action in report.actions)
