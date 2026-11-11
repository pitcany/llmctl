"""`model add SOURCE` and `import-registry` keep the fields they are given."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import Session
from typer.testing import CliRunner

from llmctl.adapters.base import AdapterStatus
from llmctl.cli import app
from llmctl.db import RuntimeName, get_engine, init_db
from llmctl.schemas import HealthState, Model, RegistryExport
from llmctl.services import router as router_mod
from llmctl.services.registry import RegistryService

runner = CliRunner()


class _FakeOllama:
    def __init__(self) -> None:
        self.pulled: list[str] = []

    async def pull_model(self, name, *, on_progress=None):  # noqa: ANN001, ANN201
        self.pulled.append(name)
        return AdapterStatus(runtime=RuntimeName.OLLAMA, state=HealthState.OK, message="ok")

    async def discover_models(self):  # noqa: ANN201
        return [
            Model(name=n, runtime=RuntimeName.OLLAMA, source=n, size_bytes=123)
            for n in self.pulled
        ]


@pytest.fixture
def db_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    url = f"sqlite:///{tmp_path / 'cli.sqlite3'}"
    monkeypatch.setenv("LLMCTL_DB_URL", url)
    init_db(url)
    return url


def _models(url: str) -> list[Model]:
    with Session(get_engine(url)) as db:
        return RegistryService(db).list_models(include_inactive=True)


def test_model_add_source_applies_the_metadata_flags(
    db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeOllama()
    monkeypatch.setattr(router_mod.RuntimeRouter, "get_adapter", lambda self, r: fake)
    result = runner.invoke(
        app,
        [
            "model", "add", "qwen3:8b", "--yes",
            "--notes", "prod only",
            "--tags", "chat, small",
            "--quantization", "Q4_K_M",
            "--max-context", "8192",
            "--estimated-vram", "6.5",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    (model,) = _models(db_url)
    assert model.notes == "prod only"
    assert model.tags == ["chat", "small"]
    assert model.quantization == "Q4_K_M"
    assert model.max_context == 8192
    assert model.estimated_vram_gb == 6.5


def test_model_add_source_rejects_path_before_downloading(
    db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeOllama()
    monkeypatch.setattr(router_mod.RuntimeRouter, "get_adapter", lambda self, r: fake)
    result = runner.invoke(app, ["model", "add", "qwen3:8b", "--yes", "--path", "/models/x"])
    assert result.exit_code == 2, result.output
    assert fake.pulled == []
    assert _models(db_url) == []


def test_import_registry_keeps_host_and_inactive(db_url: str, tmp_path: Path) -> None:
    bundle = RegistryExport(
        models=[
            Model(
                id="m1",
                name="remote",
                runtime=RuntimeName.OLLAMA,
                source="remote:latest",
                host="http://10.0.0.9:11434",
                active=False,
            ),
            Model(id="m2", name="local", runtime=RuntimeName.OLLAMA, source="local:latest"),
        ],
        profiles=[],
    )
    path = tmp_path / "bundle.json"
    path.write_text(bundle.model_dump_json())
    result = runner.invoke(app, ["import-registry", str(path)])
    assert result.exit_code == 0, result.output
    by_name = {m.name: m for m in _models(db_url)}
    assert by_name["remote"].host == "http://10.0.0.9:11434"
    assert by_name["remote"].active is False
    assert by_name["local"].active is True
