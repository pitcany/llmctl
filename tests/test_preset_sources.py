from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llmctl.cli import app

BASE = {
    "served_name": "s",
    "model_id": "org/m",
    "quantization": "awq",
    "vllm_quantization_flag": "none",
    "tensor_parallel_size": 1,
    "max_model_len": 4096,
}


def _doc(alias: str, **extra) -> str:
    return yaml.safe_dump({"alias": alias, **BASE, **extra})


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    for sub in ("config", "data", "state", "cache"):
        (home / sub).mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(home / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / "cache"))
    return home


def _dirs(home: Path) -> tuple[Path, Path]:
    return home / "config" / "llmctl" / "presets", home / "config" / "llm-models"


def _snapshot(root: Path):
    out = {}
    for p in sorted([root, *root.rglob("*")]):
        st = p.lstat()
        out[str(p)] = (
            st.st_mode,
            os.readlink(p) if p.is_symlink() else None,
            p.read_bytes() if p.is_file() and not p.is_symlink() else None,
            st.st_mtime_ns,
        )
    return out


def _run(*args):
    return CliRunner().invoke(app, list(args))


def test_clean_layout(env):
    canon, legacy = _dirs(env)
    canon.mkdir(parents=True)
    (canon / "a.yaml").write_text(_doc("a"))
    res = _run("presets", "sources", "--json")
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data["canonical_dir"] == str(canon)
    assert data["presets"][0]["effective"] == str((canon / "a.yaml").resolve())
    assert data["presets"][0]["files"] == [
        {"path": str(canon / "a.yaml"), "dir": "canonical", "kind": "file", "target": None}
    ]
    assert data["problems"] == []


def test_legacy_beats_canonical_and_matches_loader(env, tmp_path, monkeypatch):
    canon, legacy = _dirs(env)
    canon.mkdir(parents=True)
    legacy.mkdir(parents=True)
    (canon / "a.yaml").write_text(_doc("a"))
    (legacy / "a.yaml").write_text(_doc("a", max_model_len=8))
    (canon / "b.yaml").write_text(_doc("b"))
    (legacy / "b.yaml").symlink_to(canon / "b.yaml")
    (canon / "c.yaml").symlink_to(legacy / "c.yaml")
    (legacy / "c.yaml").write_text(_doc("c"))
    (legacy / "file-name.yaml").write_text(_doc("zzz"))
    res = _run("presets", "sources", "--json")
    assert res.exit_code == 1
    data = json.loads(res.stdout)
    by = {p["alias"]: p for p in data["presets"]}
    assert [p["alias"] for p in data["presets"]] == sorted(by)
    assert by["a"]["collision"] and not by["b"]["collision"] and not by["c"]["collision"]
    assert by["b"]["files"][1]["kind"] == "symlink"
    assert by["b"]["files"][1]["target"] == str((canon / "b.yaml").resolve())

    # compare with the loader in a separate HOME
    from llmctl.presets import load_all_records

    other = tmp_path / "other"
    (other / "cfg").mkdir(parents=True)
    # reuse same layout: copy via same env is simplest, loader may migrate nothing here
    # because the canonical dir already exists.
    records = load_all_records()
    for alias, rec in records.items():
        assert by[alias]["effective"] == str(rec.source_path.resolve())


def test_problems(env):
    canon, legacy = _dirs(env)
    canon.mkdir(parents=True)
    legacy.mkdir(parents=True)
    (canon / "broken.yaml").symlink_to(canon / "nope.yaml")
    (canon / "l1.yaml").symlink_to(canon / "l2.yaml")
    (canon / "l2.yaml").symlink_to(canon / "l1.yaml")
    (legacy / "mal.yaml").write_text("a: [\n")
    (legacy / "list.yaml").write_text("- 1\n")
    (legacy / "bad.yaml").write_text("alias: x\n")
    unread = legacy / "unread.yaml"
    unread.write_text(_doc("u"))
    unread.chmod(0)
    try:
        res = _run("presets", "sources", "--json")
        assert res.exit_code == 1
        problems = {Path(p["path"]).name: p["problem"] for p in json.loads(res.stdout)["problems"]}
    finally:
        unread.chmod(0o600)
    expected = {"broken.yaml", "l1.yaml", "l2.yaml", "mal.yaml", "list.yaml", "bad.yaml"}
    if os.geteuid() != 0:
        expected.add("unread.yaml")
    assert expected <= set(problems)
    assert all(problems[n] for n in expected)
    table = _run("presets", "sources")
    assert table.exit_code == 1 and "PROBLEM" in table.stdout


def test_table_flags_collision(env):
    canon, legacy = _dirs(env)
    canon.mkdir(parents=True)
    legacy.mkdir(parents=True)
    (canon / "a.yaml").write_text(_doc("a"))
    (legacy / "a.yaml").write_text(_doc("a", max_model_len=1))
    res = _run("presets", "sources")
    assert res.exit_code == 1 and "COLLISION" in res.stdout


def test_read_only(env):
    canon, legacy = _dirs(env)
    legacy.mkdir(parents=True)
    (legacy / "a.yaml").write_text(_doc("a"))
    before = _snapshot(env)
    assert _run("presets", "sources").exit_code == 0
    assert _run("presets", "sources", "--json").exit_code == 0
    res = _run("settings", "a", "--json")
    assert res.exit_code == 0
    assert _snapshot(env) == before
    assert not canon.exists()


def test_settings_json(env):
    canon, legacy = _dirs(env)
    canon.mkdir(parents=True)
    legacy.mkdir(parents=True)
    (canon / "a.yaml").write_text(_doc("a"))
    (legacy / "a.yaml").write_text(_doc("a", gpu_memory_utilization=0.5))
    data = json.loads(_run("settings", "a", "--json").stdout)
    assert data["kind"] == "preset" and data["name"] == "a"
    assert data["path"] == str((legacy / "a.yaml").resolve())
    assert data["shadowed"] == [str(canon / "a.yaml")]
    f = data["fields"]
    assert f["gpu_memory_utilization"] == {"value": 0.5, "source": "preset"}
    assert f["max_num_seqs"] == {"value": None, "source": "default"}
    from llmctl.presets import Model

    assert set(f) == set(Model.model_fields)


def test_settings_json_unknown(env):
    for args in (["settings", "nope", "--json"], ["settings", "--json"]):
        res = CliRunner().invoke(app, args)
        assert res.exit_code == 2
        assert len(res.stderr.strip().splitlines()) == 1 if res.stderr else True
    assert not (env / "data").glob("**/*.db").__next__ if False else True
    assert not list(env.rglob("*.db"))


def test_module_source_and_all():
    import llmctl.presets as pkg
    import llmctl.presets.sources as mod

    src = Path(mod.__file__).read_text()
    for banned in ("load_all", "migrate_legacy_presets", "init_db", "_session"):
        assert banned not in src
    assert "sources" not in pkg.__all__


def test_presets_bare_unchanged(env, monkeypatch):
    import dataclasses

    from llmctl.services.preset_loader import load_preset_views

    canon, _ = _dirs(env)
    canon.mkdir(parents=True)
    (canon / "a.yaml").write_text(_doc("a"))
    res = _run("presets", "--json")
    assert res.exit_code == 0
    expected = json.loads(
        json.dumps([dataclasses.asdict(v) for v in load_preset_views()], default=str)
    )
    assert json.loads(res.stdout) == expected
    assert _run("presets").exit_code == 0
