"""`llmctl setup ENGINE` installs an engine; `run` then finds it without config edits."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from llmctl.config import Settings
from llmctl.services import engines as eng


def _settings(tmp_path: Path) -> Settings:
    return Settings(paths={"config_dir": tmp_path / "cfg", "data_dir": tmp_path / "data"})


def _deps(**over):
    base = dict(
        which=lambda b: {"uv": "/usr/bin/uv", "git": "/usr/bin/git"}.get(b),
        fetch_json=lambda url: [],
        download=lambda url, dest: None,
        driver_cuda=lambda: "13.2",
        machine=lambda: "x86_64",
        log=lambda m: None,
        run=lambda argv: subprocess.CompletedProcess(argv, 0, "", ""),
        stream=lambda argv: 0,
    )
    base.update(over)
    return eng.SetupDeps(**base)


def test_engine_names_are_canonicalised() -> None:
    assert eng.canonical_engine("llama.cpp") == "llama_cpp"
    assert eng.canonical_engine("FT") == "freetoken"
    with pytest.raises(eng.SetupError, match="Unknown engine"):
        eng.canonical_engine("warp")


@pytest.mark.parametrize("engine,requirement,binary", [
    ("vllm", "vllm", "vllm"),
    ("sglang", "sglang[all]", "python"),
    ("freetoken", "freetoken[accel]", "ft"),
])
def test_python_engines_get_their_own_venv(tmp_path, engine, requirement, binary) -> None:
    plan = eng.plan_setup(engine, settings=_settings(tmp_path), deps=_deps())
    venv = tmp_path / "data" / "engines" / engine / ".venv"
    assert plan.steps[0].startswith(f"uv venv {venv}")
    assert requirement in plan.steps[1]
    assert plan.record["binary"] == str(venv / "bin" / binary)
    assert plan.record["python_root"] == str(venv)


def test_without_uv_the_stdlib_venv_is_used(tmp_path) -> None:
    plan = eng.plan_setup("vllm", settings=_settings(tmp_path), deps=_deps(which=lambda b: None))
    assert " -m venv " in plan.steps[0] and "pip install" in plan.steps[1]


def _asset(name: str) -> dict:
    return {"name": name, "browser_download_url": f"u/{name}"}


_ASSETS = [
    _asset("llama-b11624-bin-ubuntu-x64.tar.gz"),
    _asset("llama-b11624-bin-ubuntu-cuda-12.8-x64.tar.gz"),
    _asset("cudart-llama-b11624-bin-ubuntu-cuda-12.8-x64.tar.gz"),
    _asset("llama-b11624-bin-ubuntu-cuda-13.4-x64.tar.gz"),
]


def test_llama_cpp_picks_the_newest_cuda_build_the_driver_supports(tmp_path) -> None:
    releases = [{"tag_name": "b11624", "assets": _ASSETS}]
    plan = eng.plan_setup(
        "llama_cpp", settings=_settings(tmp_path), deps=_deps(fetch_json=lambda url: releases)
    )
    assert [n for _, n in plan.downloads] == [
        "llama-b11624-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "cudart-llama-b11624-bin-ubuntu-cuda-12.8-x64.tar.gz",
    ]
    assert "CUDA 12.8" in plan.steps[0]
    assert plan.record["binary"].endswith("engines/llama_cpp/bin/llama-server")


def test_llama_cpp_falls_back_to_cpu_without_a_driver(tmp_path) -> None:
    releases = [{"tag_name": "b11624", "assets": _ASSETS}]
    plan = eng.plan_setup(
        "llama_cpp",
        settings=_settings(tmp_path),
        deps=_deps(fetch_json=lambda url: releases, driver_cuda=lambda: None),
    )
    assert [n for _, n in plan.downloads] == ["llama-b11624-bin-ubuntu-x64.tar.gz"]


def test_llama_cpp_without_a_matching_build_is_a_problem_not_a_crash(tmp_path) -> None:
    plan = eng.plan_setup(
        "llama_cpp",
        settings=_settings(tmp_path),
        deps=_deps(
            fetch_json=lambda url: [{"tag_name": "x", "assets": _ASSETS}],
            machine=lambda: "aarch64",  # the release above has no arm64 build
        ),
    )
    assert plan.problem and "no Ubuntu build" in plan.problem


def test_strata_is_cloned_and_set_up_with_its_own_installer(tmp_path) -> None:
    plan = eng.plan_setup(
        "strata", settings=_settings(tmp_path), deps=_deps(), extra=["--family", "coder"]
    )
    target = tmp_path / "data" / "engines" / "strata"
    assert plan.steps[0] == f"git clone {eng.STRATA_REPO} {target}"
    assert "--yes --no-start --family coder --data-dir" in plan.steps[1]
    assert plan.record["binary"] == str(target / ".venv" / "bin" / "python")
    assert plan.record["server"] == str(target / "serve" / "server.py")


def test_ollama_uses_its_installer_and_needs_sudo(tmp_path) -> None:
    plan = eng.plan_setup("ollama", settings=_settings(tmp_path), deps=_deps())
    assert plan.needs_sudo and "install.sh" in plan.steps[0]
    found = eng.plan_setup(
        "ollama", settings=_settings(tmp_path), deps=_deps(which=lambda b: "/usr/local/bin/ollama")
    )
    assert "already installed" in found.steps[0] and not found.steps[1:]


def test_run_setup_records_the_engine_and_runtime_config_picks_it_up(tmp_path) -> None:
    settings = _settings(tmp_path)
    plan = eng.plan_setup("vllm", settings=settings, deps=_deps())
    calls: list[list[str]] = []

    def run(argv):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "0.30.0", "")

    def stream(argv):
        calls.append(list(argv))
        Path(plan.record["binary"]).parent.mkdir(parents=True, exist_ok=True)
        Path(plan.record["binary"]).write_text("#!/bin/sh\n")
        return 0

    record = eng.run_setup(plan, settings=settings, deps=_deps(run=run, stream=stream))

    assert record["version"] == "0.30.0"
    assert calls[0][:2] == ["/usr/bin/uv", "venv"]
    written = yaml.safe_load((tmp_path / "cfg" / "engines.yaml").read_text())
    assert written["vllm"]["binary"] == plan.record["binary"]
    # No settings.yaml edit needed: the scheduler sees the installed binary.
    assert settings.runtime_config("vllm").binary == plan.record["binary"]
    # ... and an explicit settings.yaml entry still wins.
    settings.runtimes["vllm"] = type(settings.runtime_config("vllm"))(binary="/custom/vllm")
    assert settings.runtime_config("vllm").binary == "/custom/vllm"
    statuses = {s.engine: s for s in eng.engine_statuses(settings)}
    assert statuses["vllm"].source == "llmctl" and statuses["vllm"].version == "0.30.0"


def test_a_failing_step_is_a_setup_error_with_the_output(tmp_path) -> None:
    settings = _settings(tmp_path)
    plan = eng.plan_setup("vllm", settings=settings, deps=_deps())
    deps = _deps(run=lambda argv: subprocess.CompletedProcess(argv, 1, "", "no python 3.12"))
    with pytest.raises(eng.SetupError, match="no python 3.12"):
        eng.run_setup(plan, settings=settings, deps=deps)
    assert not (tmp_path / "cfg" / "engines.yaml").exists()


def test_cli_setup_lists_engines_and_dry_runs(tmp_path, monkeypatch) -> None:
    from llmctl.cli import app

    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LLMCTL_DATA_DIR", str(tmp_path / "data"))
    runner = CliRunner()
    listing = runner.invoke(app, ["setup"])
    assert listing.exit_code == 0, listing.output
    for engine in eng.ENGINE_NAMES:
        assert engine in listing.output
    dry = runner.invoke(app, ["setup", "vllm", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "uv venv" in dry.output or "-m venv" in dry.output
    assert "nothing was done" in dry.output.lower()
    assert not (tmp_path / "data" / "engines").exists()


def test_launcher_python_root_falls_back_to_the_vllm_env_setup_made(tmp_path, monkeypatch) -> None:
    from llmctl.integrations.launcher_env import resolve_python_root

    for var in ("LLMCTL_PYTHON_ROOT", "GPU_MODELS_PYTHON_ROOT", "CONDA_PREFIX", "VIRTUAL_ENV"):
        monkeypatch.delenv(var, raising=False)
    venv = tmp_path / "venv"
    venv.mkdir()
    monkeypatch.setattr(
        eng, "load_engines", lambda path=None: {"vllm": {"python_root": str(venv)}}
    )
    assert resolve_python_root() == str(venv)


def test_doctor_reports_engines(tmp_path, monkeypatch) -> None:
    from llmctl.services.doctor import _check_engines

    settings = _settings(tmp_path)
    monkeypatch.setattr(eng, "engine_statuses", lambda cfg: [
        SimpleNamespace(engine="vllm", source="llmctl", binary="/x/vllm", version="0.30.0"),
        SimpleNamespace(engine="ollama", source="missing", binary=None, version=None),
    ])
    checks = {c.name: c for c in _check_engines(settings)}
    assert checks["engine:vllm"].verdict == "pass" and "0.30.0" in checks["engine:vllm"].detail
    assert checks["engine:ollama"].verdict == "warn"
    assert "llmctl setup ollama" in (checks["engine:ollama"].remediation or "")


def test_a_binary_that_cannot_run_is_not_recorded(tmp_path) -> None:
    """The prebuilt llama.cpp needs glibc 2.38; on an older box say so, record nothing."""
    settings = _settings(tmp_path)
    releases = [{"tag_name": "b1", "assets": _ASSETS}]
    plan = eng.plan_setup(
        "llama_cpp", settings=settings, deps=_deps(fetch_json=lambda url: releases)
    )
    binary = Path(plan.record["binary"])

    def download(url, dest):
        import tarfile

        src = tmp_path / "pkg" / "build" / "bin"
        src.mkdir(parents=True, exist_ok=True)
        (src / "llama-server").write_text("#!/bin/sh\n")
        with tarfile.open(dest, "w:gz") as tar:
            tar.add(src.parent, arcname="build")

    glibc = "llama-server: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found"
    deps = _deps(
        download=download,
        run=lambda argv: subprocess.CompletedProcess(argv, 1, "", glibc),
    )
    with pytest.raises(eng.SetupError, match="newer glibc"):
        eng.run_setup(plan, settings=settings, deps=deps)
    assert binary.exists()  # unpacked, so the operator can inspect it
    assert not (tmp_path / "cfg" / "engines.yaml").exists()
