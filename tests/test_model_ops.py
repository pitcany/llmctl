"""Engine-agnostic add/remove (``llmctl.services.model_ops``).

The operator types one SOURCE and gets the engine's own fetch; removing takes
one flag and does the engine's own delete. These tests pin the source grammar,
that add registers (and revives) exactly one row, and that purge removes the
model itself, refuses what it must not touch, and never lies about it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlmodel import Session
from textual.app import App
from textual.widgets import Input, Static
from typer.testing import CliRunner

from llmctl.adapters.base import AdapterStatus
from llmctl.db import (
    ModelStatus,
    RuntimeName,
    SessionRecord,
    SessionStatus,
    get_engine,
    init_db,
)
from llmctl.schemas import HealthState, Model, ModelCreate
from llmctl.services import model_ops
from llmctl.services.model_ops import (
    AddPlan,
    ModelOps,
    ModelOpsError,
    plan_add,
    plan_remove,
)
from llmctl.services.registry import RegistryService
from llmctl.tui._modals_registry import AddModelModal, AddRequest

# --------------------------------------------------------------------------
# Source grammar
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "engine", "want_engine", "want_method", "want_command"),
    [
        ("qwen3:8b", None, "ollama", "ollama-pull", "ollama pull qwen3:8b"),
        ("qwen3", None, "ollama", "ollama-pull", "ollama pull qwen3:latest"),
        (
            "ollama:huihui_ai/qwen3-abliterated",
            None,
            "ollama",
            "ollama-pull",
            "ollama pull huihui_ai/qwen3-abliterated:latest",
        ),
        (
            "hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M",
            None,
            "ollama",
            "ollama-pull",
            "ollama pull hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M",
        ),
        (
            "hf:Qwen/Qwen3-8B",
            "ollama",
            "ollama",
            "ollama-pull",
            "ollama pull hf.co/Qwen/Qwen3-8B:latest",
        ),
        ("Qwen/Qwen3-8B", None, "vllm", "hf-download", "hf download Qwen/Qwen3-8B"),
        ("Qwen/Qwen3-8B", "sglang", "sglang", "hf-download", "hf download Qwen/Qwen3-8B"),
        (
            "unsloth/Qwen3-8B-GGUF:Q4_K_M",
            None,
            "llama_cpp",
            "hf-download",
            "hf download unsloth/Qwen3-8B-GGUF --include '*Q4_K_M*.gguf'",
        ),
        (
            "https://huggingface.co/unsloth/Qwen3-8B-GGUF/blob/main/Qwen3-8B-Q4_K_M.gguf",
            None,
            "llama_cpp",
            "hf-download",
            "hf download unsloth/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf",
        ),
        ("lmstudio:qwen/qwen3-8b", None, "lmstudio", "lmstudio-get", "lms get qwen/qwen3-8b --yes"),
        ("qwen/qwen3-8b", "lms", "lmstudio", "lmstudio-get", "lms get qwen/qwen3-8b --yes"),
    ],
)
def test_plan_add_picks_the_engine_from_the_source(
    source, engine, want_engine, want_method, want_command
) -> None:
    plan = plan_add(source, engine)
    assert plan.engine.value == want_engine
    assert plan.method == want_method
    assert plan.command == want_command


@pytest.mark.parametrize(
    ("source", "engine", "fragment"),
    [
        ("", None, "Give a source"),
        ("Qwen/Qwen3-8B:Q4_K_M", "vllm", "llama_cpp's job"),
        ("x", "openai", "adopted"),
        ("x", "warp-drive", "Unknown engine"),
        ("/definitely/not/here", None, "No such file"),
        ("ollama:qwen3", "vllm", "conflicts"),
        ("org/strata-model", "strata", "local config JSON"),
        ("hf:org/strata-model", "strata", "local config JSON"),
    ],
)
def test_plan_add_refuses_with_a_reason(source, engine, fragment) -> None:
    with pytest.raises(ModelOpsError, match=fragment):
        plan_add(source, engine)


def test_plan_add_local_paths(tmp_path: Path) -> None:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    assert plan_add(str(gguf)).engine == RuntimeName.LLAMA_CPP

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    plan = plan_add(str(ckpt))
    assert (plan.engine, plan.method) == (RuntimeName.VLLM, "register")
    assert plan_add(str(ckpt), "sglang").engine == RuntimeName.SGLANG

    shards = tmp_path / "split"
    shards.mkdir()
    for i in (1, 2):
        (shards / f"big-Q4_K_M-0000{i}-of-00002.gguf").write_bytes(b"x")
    plan = plan_add(str(shards))
    assert plan.source.endswith("big-Q4_K_M-00001-of-00002.gguf")

    with pytest.raises(ModelOpsError, match="ollama create"):
        plan_add(str(gguf), "ollama")


# --------------------------------------------------------------------------
# Fixtures: a real registry DB, fake engines
# --------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'ops.db'}"
    init_db(url)
    with Session(get_engine(url)) as session:
        yield session


class _FakeOllama:
    def __init__(self, *, delete_ok: bool = True) -> None:
        self.pulled: list[str] = []
        self.deleted: list[str] = []
        self.delete_ok = delete_ok

    async def pull_model(self, name, *, on_progress=None):
        self.pulled.append(name)
        if on_progress:
            on_progress("pulling manifest", None, None)
            on_progress("downloading", 5, 10)
        return AdapterStatus(runtime=RuntimeName.OLLAMA, state=HealthState.OK, message="ok")

    async def discover_models(self):
        return [
            Model(name=n, runtime=RuntimeName.OLLAMA, source=n, size_bytes=123)
            for n in self.pulled
        ]

    async def delete_model(self, model):
        state = HealthState.OK if self.delete_ok else HealthState.DEGRADED
        if self.delete_ok:
            self.deleted.append(model.source)
        return AdapterStatus(runtime=RuntimeName.OLLAMA, state=state, message="daemon said no")


def _router(adapter):
    return SimpleNamespace(get_adapter=lambda runtime: adapter)


# --------------------------------------------------------------------------
# Add
# --------------------------------------------------------------------------


def test_add_ollama_pulls_and_registers_one_row(db) -> None:
    fake = _FakeOllama()
    ops = ModelOps(db, router=_router(fake))
    messages: list[str] = []
    model = ops.add(plan_add("qwen3"), on_progress=messages.append).model
    assert fake.pulled == ["qwen3:latest"]
    assert (model.runtime, model.source, model.size_bytes) == (
        RuntimeName.OLLAMA,
        "qwen3:latest",
        123,
    )
    assert any("50%" in m for m in messages)
    # Adding again (or a scan) refreshes the same row, never a twin.
    ops.add(plan_add("qwen3:latest"))
    rows = [m for m in RegistryService(db).list_models() if m.source == "qwen3:latest"]
    assert len(rows) == 1


def test_add_revives_a_removed_row(db) -> None:
    fake = _FakeOllama()
    ops = ModelOps(db, router=_router(fake))
    first = ops.add(plan_add("qwen3:8b")).model
    ops.remove(first.id)
    assert RegistryService(db).get_model(first.id) is None
    again = ops.add(plan_add("qwen3:8b")).model
    assert again.id == first.id
    assert again.status == ModelStatus.DISCOVERED


def _fake_hf(files: dict[str, int]):
    siblings = [SimpleNamespace(rfilename=f, size=s) for f, s in files.items()]
    info = SimpleNamespace(siblings=siblings)
    return SimpleNamespace(model_info=lambda repo, files_metadata: info)


def test_add_gguf_downloads_only_the_chosen_quant(db, tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    calls: list[dict] = []

    def download(**kwargs):
        calls.append(kwargs)
        snap = hub / "models--org--M-GGUF" / "snapshots" / "abc"
        snap.mkdir(parents=True)
        for f in kwargs["allow_patterns"]:
            (snap / f).write_bytes(b"x")
        return str(snap)

    ops = ModelOps(
        db,
        hf_api=_fake_hf(
            {"M-Q4_K_M.gguf": 5, "M-Q8_0.gguf": 9, "mmproj-F16.gguf": 1, "README.md": 1}
        ),
        download=download,
        disk_free=lambda p: 10**15,
        hub_cache=hub,
    )
    model = ops.add(plan_add("org/M-GGUF:q4_k_m")).model
    assert calls[0]["allow_patterns"] == ["M-Q4_K_M.gguf"]
    assert model.runtime == RuntimeName.LLAMA_CPP
    assert model.path.endswith("M-Q4_K_M.gguf")
    assert model.quantization == "q4_k_m"


def test_add_gguf_without_quant_lists_the_choices(db) -> None:
    ops = ModelOps(db, hf_api=_fake_hf({"M-Q4_K_M.gguf": 5, "M-Q8_0.gguf": 9}))
    with pytest.raises(ModelOpsError, match=r"Available: Q4_K_M, Q8_0"):
        ops.resolve_hf_files(plan_add("org/M-GGUF"))


def test_add_refuses_when_the_disk_cannot_hold_it(db, tmp_path: Path) -> None:
    ops = ModelOps(
        db,
        hf_api=_fake_hf({"model.safetensors": 50 * 10**9, "config.json": 1}),
        download=lambda **kw: pytest.fail("must not download"),
        disk_free=lambda p: 20 * 10**9,
        hub_cache=tmp_path,
    )
    with pytest.raises(ModelOpsError, match="Not enough disk"):
        ops.add(plan_add("org/Big"))


def test_add_safetensors_skips_other_formats(db) -> None:
    ops = ModelOps(
        db,
        hf_api=_fake_hf(
            {"model.safetensors": 1, "config.json": 1, "onnx/model.onnx": 1, "x.gguf": 1}
        ),
    )
    files, _ = ops.resolve_hf_files(plan_add("org/M"))
    assert files == ["config.json", "model.safetensors"]


# --------------------------------------------------------------------------
# Remove
# --------------------------------------------------------------------------


def test_remove_without_purge_leaves_the_engine_alone(db) -> None:
    fake = _FakeOllama()
    ops = ModelOps(db, router=_router(fake))
    model = ops.add(plan_add("qwen3:8b")).model
    result = ops.remove(model.id)
    assert result.removed_row and not result.purged
    assert fake.deleted == []
    assert "untouched" in result.message


def test_purge_ollama_runs_ollama_rm(db) -> None:
    fake = _FakeOllama()
    ops = ModelOps(db, router=_router(fake))
    model = ops.add(plan_add("qwen3:8b")).model
    plan = plan_remove(model, db=db)
    assert plan.can_purge and "ollama rm qwen3:8b" in plan.purge_label
    result = ops.remove(model.id, purge=True)
    assert fake.deleted == ["qwen3:8b"]
    assert result.purged and "removed from Ollama" in result.message


def test_failed_purge_keeps_the_row(db) -> None:
    fake = _FakeOllama(delete_ok=False)
    ops = ModelOps(db, router=_router(fake))
    model = ops.add(plan_add("qwen3:8b")).model
    with pytest.raises(ModelOpsError, match="daemon said no"):
        ops.remove(model.id, purge=True)
    assert RegistryService(db).get_model(model.id) is not None


def _file_model(db, path: Path, runtime=RuntimeName.LLAMA_CPP) -> Model:
    return RegistryService(db).add_model(
        ModelCreate(name=path.name, runtime=runtime, source=str(path), path=str(path))
    )


def test_purge_hf_checkpoint_deletes_the_whole_cache_entry(db, tmp_path, monkeypatch) -> None:
    hub = tmp_path / "hub"
    repo = hub / "models--org--M"
    blob = repo / "blobs" / "sha"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"x" * 1000)
    snap = repo / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(blob)
    monkeypatch.setattr(model_ops, "_model_roots", lambda: [hub.resolve()])
    model = _file_model(db, snap, RuntimeName.VLLM)

    plan = plan_remove(model, hub_cache=hub, roots=[hub])
    assert plan.targets == (repo,)
    assert plan.size_bytes == 1000

    result = ModelOps(db, hub_cache=hub).remove(model.id, purge=True)
    assert result.purged and not repo.exists()


def test_purge_vllm_row_keyed_by_repo_id_finds_the_cache(db, tmp_path) -> None:
    hub = tmp_path / "hub"
    (hub / "models--Qwen--Qwen3-8B" / "blobs").mkdir(parents=True)
    model = Model(id="x", name="Qwen/Qwen3-8B", runtime=RuntimeName.VLLM, path="Qwen/Qwen3-8B")
    plan = plan_remove(model, hub_cache=hub, roots=[hub])
    assert plan.can_purge
    assert plan.targets == (hub / "models--Qwen--Qwen3-8B",)


def test_purge_one_quant_keeps_sibling_quants(db, tmp_path, monkeypatch) -> None:
    hub = tmp_path / "hub"
    repo = hub / "models--org--M-GGUF"
    snap = repo / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (repo / "blobs").mkdir()
    for name in ("M-Q4_K_M.gguf", "M-Q8_0.gguf"):
        blob = repo / "blobs" / name.lower()
        blob.write_bytes(b"x")
        (snap / name).symlink_to(blob)
    monkeypatch.setattr(model_ops, "_model_roots", lambda: [hub.resolve()])
    model = _file_model(db, snap / "M-Q4_K_M.gguf")
    ModelOps(db, hub_cache=hub).remove(model.id, purge=True)
    assert not (snap / "M-Q4_K_M.gguf").exists()
    assert not (repo / "blobs" / "m-q4_k_m.gguf").exists()
    assert (snap / "M-Q8_0.gguf").exists()


def test_purge_refuses_paths_outside_the_model_roots(db, tmp_path) -> None:
    stray = tmp_path / "elsewhere" / "m.gguf"
    stray.parent.mkdir()
    stray.write_bytes(b"x")
    model = _file_model(db, stray)
    plan = plan_remove(model, roots=[tmp_path / "roots"], hub_cache=tmp_path / "hub")
    assert not plan.can_purge and "outside" in plan.purge_summary
    assert stray.exists()


def test_purge_of_a_catalog_symlink_removes_only_the_link(db, tmp_path) -> None:
    real = tmp_path / "real" / "ckpt"
    real.mkdir(parents=True)
    (real / "config.json").write_text("{}")
    link = tmp_path / "store" / "ckpt"
    link.parent.mkdir()
    link.symlink_to(real)
    model = _file_model(db, link, RuntimeName.VLLM)
    plan = plan_remove(model, roots=[tmp_path], hub_cache=tmp_path / "hub")
    assert plan.can_purge and "link" in plan.purge_label and "stay" in plan.purge_summary


def test_purge_is_blocked_while_a_session_runs_it(db, tmp_path, monkeypatch) -> None:
    root = tmp_path / "models"
    root.mkdir()
    f = root / "m.gguf"
    f.write_bytes(b"x")
    monkeypatch.setattr(model_ops, "_model_roots", lambda: [root.resolve()])
    model = _file_model(db, f)
    db.add(
        SessionRecord(
            model_id=model.id, runtime=RuntimeName.LLAMA_CPP, status=SessionStatus.RUNNING
        )
    )
    db.commit()
    with pytest.raises(ModelOpsError, match="stop it first"):
        ModelOps(db, hub_cache=tmp_path / "hub").remove(model.id, purge=True)
    assert f.exists()


def test_lmstudio_purge_uses_the_folder_lms_reports(db, tmp_path) -> None:
    root = tmp_path / "lms"
    folder = root / "pub" / "m"
    folder.mkdir(parents=True)
    (folder / "m-Q4.gguf").write_bytes(b"x")
    listing = '[{"modelKey": "pub/m", "path": "pub/m/m-Q4.gguf"}]'
    runner = lambda *a, **k: SimpleNamespace(stdout=listing, returncode=0)  # noqa: E731
    model = Model(id="x", name="pub/m", runtime=RuntimeName.LMSTUDIO, source="pub/m")
    plan = plan_remove(model, roots=[root], hub_cache=tmp_path / "hub", runner=runner)
    assert plan.can_purge and plan.targets == (folder,)


# --------------------------------------------------------------------------
# CLI + TUI surfaces
# --------------------------------------------------------------------------


def test_cli_model_add_source_and_rm_purge(tmp_path, monkeypatch) -> None:
    from llmctl.cli import app
    from llmctl.services import router as router_mod

    monkeypatch.setenv("LLMCTL_DB_URL", f"sqlite:///{tmp_path / 'cli.sqlite3'}")
    fake = _FakeOllama()
    monkeypatch.setattr(router_mod.RuntimeRouter, "get_adapter", lambda self, r: fake)
    runner = CliRunner()

    result = runner.invoke(app, ["model", "add", "qwen3:8b", "--yes"])
    assert result.exit_code == 0, result.output
    assert "same as: ollama pull qwen3:8b" in result.output
    assert "Added qwen3:8b" in result.output

    result = runner.invoke(app, ["model", "rm", "qwen3:8b"])
    assert result.exit_code == 0, result.output
    assert "--purge" in result.output  # tells the operator how to go further
    assert fake.deleted == []

    runner.invoke(app, ["model", "add", "qwen3:8b", "--yes"])
    result = runner.invoke(app, ["model", "rm", "qwen3:8b", "--purge", "--yes"])
    assert result.exit_code == 0, result.output
    assert fake.deleted == ["qwen3:8b"]


class _Host(App[None]):
    def __init__(self, modal) -> None:
        super().__init__()
        self._modal = modal
        self.result: object = "unset"

    def on_mount(self) -> None:
        self.push_screen(self._modal, lambda r: setattr(self, "result", r))


def test_add_modal_previews_the_plan_and_returns_it() -> None:
    async def _main() -> None:
        app = _Host(AddModelModal())
        async with app.run_test() as pilot:
            await pilot.pause()
            app.screen.query_one("#add-source", Input).value = "unsloth/Qwen3-8B-GGUF:Q4_K_M"
            await pilot.pause()
            preview = str(app.screen.query_one("#add-preview", Static).content)
            assert "llama.cpp" in preview and "hf download" in preview
            app.screen.query_one("#add-submit").press()
            await pilot.pause()
        assert isinstance(app.result, AddRequest)
        assert isinstance(app.result.plan, AddPlan)
        assert app.result.plan.engine == RuntimeName.LLAMA_CPP

    asyncio.run(_main())


def test_add_modal_refuses_a_source_that_does_not_plan() -> None:
    async def _main() -> None:
        app = _Host(AddModelModal())
        async with app.run_test() as pilot:
            await pilot.pause()
            app.screen.query_one("#add-source", Input).value = "Qwen/Qwen3-8B:Q4"
            app.screen.query_one("#add-engine").value = "vllm"
            await pilot.pause()
            assert "llama_cpp" in str(app.screen.query_one("#add-preview", Static).content)
            app.screen.query_one("#add-submit").press()
            await pilot.pause()
            assert isinstance(app.screen, AddModelModal)  # still open
        assert app.result == "unset"

    asyncio.run(_main())


# --------------------------------------------------------------------------
# Guards found against the live registry
# --------------------------------------------------------------------------


def test_purge_refuses_a_model_on_another_machine(tmp_path) -> None:
    model = Model(id="x", name="qwen3:8b", runtime=RuntimeName.OLLAMA, host="some-mac.lan")
    plan = plan_remove(model, roots=[tmp_path], hub_cache=tmp_path)
    assert not plan.can_purge and "some-mac.lan" in plan.purge_summary


def test_purge_is_blocked_while_a_preset_names_the_weights(db, tmp_path, monkeypatch) -> None:
    """The deployed generator is a preset; purging its weights kills the lane."""
    import llmctl.presets as presets_mod
    import llmctl.services.preset_loader as loader_mod

    hub = tmp_path / "hub"
    repo = hub / "models--Qwen--Qwen3-8B"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "b").write_bytes(b"x")
    record = SimpleNamespace(model=SimpleNamespace(model_id="Qwen/Qwen3-8B"))
    monkeypatch.setattr(presets_mod, "load_all_records", lambda **kw: {"qwen3-8b": record})
    monkeypatch.setattr(loader_mod, "resolve_preset_link", lambda m, models: (None, "unlinked"))
    monkeypatch.setattr(model_ops, "_model_roots", lambda: [hub.resolve()])
    model = Model(id="x", name="other-name", runtime=RuntimeName.VLLM, path=str(repo / "snapshots"))
    (repo / "snapshots").mkdir()

    plan = plan_remove(model, hub_cache=hub, roots=[hub])
    assert plan.targets == (repo,)
    assert any("preset qwen3-8b" in b for b in plan.blockers)


def test_hard_linked_bytes_are_not_reported_as_freed(tmp_path) -> None:
    root = tmp_path / "lms"
    folder = root / "pub" / "m"
    folder.mkdir(parents=True)
    f = folder / "m.gguf"
    f.write_bytes(b"x" * 500)
    (tmp_path / "store.gguf").hardlink_to(f)
    model = Model(id="x", name="m", runtime=RuntimeName.LLAMA_CPP, path=str(folder))
    plan = plan_remove(model, roots=[root], hub_cache=tmp_path / "hub")
    assert plan.size_bytes == 0
    assert "hard-linked elsewhere" in plan.purge_summary


def test_lmstudio_view_of_the_ollama_library_is_not_purged(tmp_path) -> None:
    listing = (
        '[{"modelKey": "llama3.2", "path": "ollama-library/llama3.2/x.gguf"},'
        ' {"modelKey": "llama3.2", "deviceIdentifier": "mac", "path": "pub/llama3.2/x.gguf"}]'
    )
    runner = lambda *a, **k: SimpleNamespace(stdout=listing, returncode=0)  # noqa: E731
    model = Model(id="x", name="llama3.2", runtime=RuntimeName.LMSTUDIO, source="llama3.2")
    plan = plan_remove(model, roots=[tmp_path], hub_cache=tmp_path, runner=runner)
    assert not plan.can_purge and "Ollama" in plan.purge_summary
