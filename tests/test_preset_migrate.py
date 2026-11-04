from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llmctl.cli import app
from llmctl.presets.store import load_all, save_preset

BASE = {
    "served_name": "s",
    "model_id": "org/m",
    "quantization": "awq",
    "vllm_quantization_flag": "none",
    "tensor_parallel_size": 1,
    "max_model_len": 4096,
}


def _doc(alias: str, **extra) -> str:
    return "# keep me\n" + yaml.safe_dump({"alias": alias, **BASE, **extra})


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    home = tmp_path / "home"
    for sub in ("config", "data", "state", "cache"):
        (home / sub).mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(home / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / "cache"))
    legacy = home / "config" / "llm-models"
    canon = home / "config" / "llmctl" / "presets"
    legacy.mkdir(parents=True)
    return legacy, canon


def _snap(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root))
        if p.is_symlink():
            out[rel] = ("link", os.readlink(p))
        elif p.is_dir():
            out[rel] = ("dir",)
        else:
            out[rel] = ("file", p.read_bytes(), stat.S_IMODE(p.stat().st_mode))
    return out


def run(*args):
    return CliRunner().invoke(app, ["presets", "migrate", *args])


def test_dry_run_and_apply(dirs, tmp_path):
    legacy, canon = dirs
    (legacy / "a.yaml").write_text(_doc("a"))
    (legacy / "a.yaml").chmod(0o640)
    (legacy / "bad.yaml").write_text("- nope\n")
    (legacy / "_skip.yaml").write_text("x: 1\n")
    root = tmp_path / "home"
    before = _snap(root)
    r = run("--json")
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert data["applied"] is False and data["manifest"] is None
    assert {a["alias"]: a["action"] for a in data["actions"]} == {"a": "move", "bad": "invalid"}
    assert _snap(root) == before

    canon.mkdir(parents=True)  # stops load_all's implicit auto-link migration
    models = load_all()
    legacy_bytes = (legacy / "a.yaml").read_bytes()
    r = run("--apply", "--json")
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    manifest = Path(data["manifest"])
    assert manifest.parent == canon / ".migrations"
    c = canon / "a.yaml"
    assert c.is_file() and not c.is_symlink()
    assert c.read_bytes() == legacy_bytes
    assert stat.S_IMODE(c.stat().st_mode) == 0o640
    link = os.readlink(legacy / "a.yaml")
    assert not os.path.isabs(link)
    assert (legacy / "a.yaml").resolve() == c.resolve()
    assert (legacy / "a.yaml").read_bytes() == legacy_bytes
    assert load_all() == models
    rec = json.loads(manifest.read_text())["changes"][0]
    assert rec["canonical_before"]["type"] == "absent" and rec["action"] == "move"
    assert rec["sha256"]

    src = json.loads(CliRunner().invoke(app, ["presets", "sources", "--json"]).stdout)
    assert src["presets"][0]["collision"] is False
    assert src["presets"][0]["effective"] == str(c.resolve())

    snap2 = _snap(root)
    r = run("--apply", "--json")
    d2 = json.loads(r.stdout)
    assert d2["manifest"] is None
    assert all(a["action"] not in ("move", "identical") for a in d2["actions"])
    assert _snap(root) == snap2

    # edit then rollback keeps the edit
    m = models["a"].model_copy(update={"max_model_len": 8192})
    save_preset(m)
    assert "max_model_len: 8192" in (legacy / "a.yaml").read_text()
    r = run("--rollback", str(manifest))
    assert r.exit_code == 0, r.output
    assert not (legacy / "a.yaml").is_symlink()
    assert "max_model_len: 8192" in (legacy / "a.yaml").read_text()
    assert stat.S_IMODE((legacy / "a.yaml").stat().st_mode) == 0o640
    assert not c.exists()
    assert load_all()["a"].max_model_len == 8192


def test_collision_identical_exit_codes(dirs):
    legacy, canon = dirs
    canon.mkdir(parents=True)
    (legacy / "a.yaml").write_text(_doc("a"))
    (canon / "a.yaml").write_text(_doc("a", max_model_len=1))
    (legacy / "b.yaml").write_text(_doc("b"))
    (canon / "b.yaml").write_text(_doc("b"))
    (legacy / "c.yaml").write_text(_doc("c"))
    ca = (canon / "a.yaml").read_bytes()
    assert run("--json").exit_code == 1
    r = run("--apply", "--json")
    assert r.exit_code == 1
    acts = {a["alias"]: a["action"] for a in json.loads(r.stdout)["actions"]}
    assert acts == {"a": "collision", "b": "identical", "c": "move"}
    assert not (legacy / "a.yaml").is_symlink()
    assert (canon / "a.yaml").read_bytes() == ca
    assert (legacy / "b.yaml").is_symlink() and (legacy / "c.yaml").is_symlink()
    manifest = json.loads(r.stdout)["manifest"]
    assert run("--rollback", manifest).exit_code == 0
    assert (canon / "b.yaml").exists()  # identical keeps pre-existing
    assert not (canon / "c.yaml").exists()
    assert not (legacy / "b.yaml").is_symlink()


def test_canonical_symlink_to_legacy(dirs):
    legacy, canon = dirs
    canon.mkdir(parents=True)
    (legacy / "a.yaml").write_text(_doc("a"))
    (canon / "a.yaml").symlink_to(os.path.relpath(legacy / "a.yaml", canon))
    models = load_all()
    assert json.loads(run("--json").stdout)["actions"][0]["action"] == "move"
    r = run("--apply", "--json")
    assert r.exit_code == 0
    assert not (canon / "a.yaml").is_symlink()
    assert (legacy / "a.yaml").is_symlink()
    assert (legacy / "a.yaml").resolve() == (canon / "a.yaml").resolve()
    assert load_all() == models
    assert run("--rollback", json.loads(r.stdout)["manifest"]).exit_code == 0
    assert (canon / "a.yaml").is_symlink()
    assert not (legacy / "a.yaml").is_symlink()
    assert load_all() == models


def test_rollback_bad_manifest(dirs, tmp_path):
    legacy, _ = dirs
    (legacy / "a.yaml").write_text(_doc("a"))
    before = _snap(tmp_path / "home")
    assert run("--rollback", str(tmp_path / "nope.json")).exit_code != 0
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    before = _snap(tmp_path / "home")
    r = run("--rollback", str(bad))
    assert r.exit_code != 0
    assert _snap(tmp_path / "home") == before


def test_broken_canonical_symlink_is_move(dirs):
    legacy, canon = dirs
    canon.mkdir(parents=True)
    (legacy / "a.yaml").write_text(_doc("a"))
    (canon / "a.yaml").symlink_to("does-not-exist.yaml")
    r = run("--json")
    assert [a["action"] for a in json.loads(r.stdout)["actions"]] == ["move"]
    assert r.exit_code == 0
    r = run("--apply", "--json")
    assert r.exit_code == 0
    assert not (canon / "a.yaml").is_symlink()
    assert run("--rollback", json.loads(r.stdout)["manifest"]).exit_code == 0
    assert os.readlink(canon / "a.yaml") == "does-not-exist.yaml"
