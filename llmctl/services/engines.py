"""``llmctl setup ENGINE``: install an inference engine so ``llmctl run`` can use it.

Each engine installs into its own place under llmctl's data directory
(``<data_dir>/engines/<engine>/``) and is recorded in ``engines.yaml`` next
to ``settings.yaml``::

    vllm:
      binary: /home/me/.local/share/llmctl/engines/vllm/.venv/bin/vllm
      python_root: /home/me/.local/share/llmctl/engines/vllm/.venv
      installed_at: "2026-10-14T21:03:11"
      version: "0.30.0"

:meth:`llmctl.config.Settings.runtime_config` reads that record, so an
engine installed here is found without any ``settings.yaml`` edit, and a
``runtimes.<engine>.binary`` the operator wrote still wins. The vLLM
launcher's interpreter (``LLMCTL_PYTHON_ROOT`` and friends) falls back to
the recorded ``python_root`` the same way.

How each engine is installed:

* vLLM, SGLang, FreeToken: a Python venv (``uv venv``, or ``python -m venv``
  when uv is missing) with the engine's wheel in it.
* llama.cpp: the prebuilt ``llama-server`` from the latest GitHub release,
  CUDA build matched to the driver when there is an NVIDIA GPU, CPU build
  otherwise. Those builds need glibc 2.38 (Ubuntu 24.04); on an older
  system, or with ``--from-source``, the same release is cloned and built
  with cmake instead (CUDA for this machine's GPUs when nvcc is present).
* Strata: ``git clone`` of github.com/Niko1221/Strata plus its own
  ``setup.sh --yes --no-start`` (builds the engine, downloads the model it
  picks for the machine's RAM — 70-120 GB, an hour or more). The config
  JSON setup writes is registered as a model, so ``llmctl run`` can serve it.
* Ollama: its official installer script (``curl -fsSL
  https://ollama.com/install.sh | sh``), which uses sudo; llmctl shows the
  command and runs it only after confirmation.

Everything that touches the network or the filesystem goes through
:class:`SetupDeps` so the planning and the tests run without either.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from llmctl.config import Settings, load_settings

ENGINES_FILE = "engines.yaml"

#: Engines ``llmctl setup`` knows, in the order ``llmctl setup`` lists them.
ENGINE_NAMES: tuple[str, ...] = ("vllm", "sglang", "freetoken", "strata", "llama_cpp", "ollama")

_ALIASES = {
    "llama": "llama_cpp",
    "llamacpp": "llama_cpp",
    "llama.cpp": "llama_cpp",
    "ft": "freetoken",
}

STRATA_REPO = "https://github.com/Niko1221/Strata.git"

#: pip requirement and console binary of the Python engines.
_PYTHON_ENGINES: dict[str, tuple[str, str]] = {
    "vllm": ("vllm", "vllm"),
    "sglang": ("sglang[all]", "python"),  # launched as `python -m sglang.launch_server`
    "freetoken": ("freetoken[accel]", "ft"),
}

LLAMA_CPP_REPO = "ggml-org/llama.cpp"
#: The glibc the prebuilt llama.cpp Ubuntu builds link against (Ubuntu 24.04).
PREBUILT_MIN_GLIBC = (2, 38)
OLLAMA_INSTALL_URL = "https://ollama.com/install.sh"


class SetupError(RuntimeError):
    """Raised when an engine cannot be installed; the message says why."""


def canonical_engine(name: str) -> str:
    """``llama.cpp`` -> ``llama_cpp``, ``ft`` -> ``freetoken``; raises on unknown."""
    value = name.strip().lower().replace("-", "_")
    value = _ALIASES.get(value, _ALIASES.get(name.strip().lower(), value))
    if value not in ENGINE_NAMES:
        raise SetupError(f"Unknown engine {name!r}; one of: {', '.join(ENGINE_NAMES)}.")
    return value


# --- the record -------------------------------------------------------------------


def engines_file(settings: Settings | None = None) -> Path:
    cfg = settings or load_settings()
    return cfg.config_dir / ENGINES_FILE


def load_engines(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """The ``engines.yaml`` mapping, or ``{}`` when absent or unreadable."""
    target = path or engines_file()
    try:
        data = yaml.safe_load(target.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def record_engine(engine: str, info: dict[str, Any], path: Path | None = None) -> Path:
    target = path or engines_file()
    data = load_engines(target)
    data[engine] = info
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, sort_keys=True))
    return target


def forget_engine(engine: str, path: Path | None = None) -> None:
    target = path or engines_file()
    data = load_engines(target)
    if data.pop(engine, None) is not None:
        target.write_text(yaml.safe_dump(data, sort_keys=True))


# --- host facts and side effects, injectable --------------------------------------


@dataclass
class SetupDeps:
    """Seams for everything that leaves the process."""

    run: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = lambda argv: (
        subprocess.run(list(argv), capture_output=True, text=True, check=False)  # noqa: S603
    )
    stream: Callable[..., int] = lambda argv, env=None: subprocess.run(  # noqa: S603
        list(argv), check=False, env=env
    ).returncode
    which: Callable[[str], str | None] = shutil.which
    fetch_json: Callable[[str], Any] = lambda url: json.load(
        urllib.request.urlopen(url, timeout=30)  # noqa: S310 - fixed https hosts
    )
    download: Callable[[str, Path], None] = lambda url, dest: urllib.request.urlretrieve(  # noqa: S310
        url, dest
    )
    driver_cuda: Callable[[], str | None] = lambda: _driver_cuda_version()
    machine: Callable[[], str] = platform.machine
    log: Callable[[str], None] = print
    glibc: Callable[[], tuple[int, int] | None] = lambda: _glibc_version()
    nvcc: Callable[[], str | None] = lambda: _find_nvcc()
    #: Compute capabilities of the local GPUs as CMake wants them (``120``).
    cuda_archs: Callable[[], list[str]] = lambda: _cuda_archs()
    cpus: Callable[[], int] = lambda: os.cpu_count() or 4


def _driver_cuda_version() -> str | None:
    """``13.2`` from ``nvidia-smi``, or None without an NVIDIA driver."""
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv
            ["nvidia-smi"], capture_output=True, text=True, timeout=10, check=False
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out)
    return match.group(1) if match else None


def _glibc_version() -> tuple[int, int] | None:
    """``(2, 35)`` on glibc systems, None elsewhere (musl, macOS)."""
    lib, version = platform.libc_ver()
    match = re.match(r"^(\d+)\.(\d+)", version or "")
    if lib != "glibc" or not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _find_nvcc() -> str | None:
    """nvcc on PATH, or the toolkit's default location (often not on PATH)."""
    found = shutil.which("nvcc")
    if found:
        return found
    default = Path("/usr/local/cuda/bin/nvcc")
    return str(default) if default.is_file() else None


def _cuda_archs() -> list[str]:
    """``["120"]`` for RTX 50-series cards; empty without nvidia-smi."""
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    archs = {line.strip().replace(".", "") for line in out.splitlines() if line.strip()}
    return sorted(a for a in archs if a.isdigit())


# --- plans ---------------------------------------------------------------------------


@dataclass
class SetupPlan:
    """What ``setup ENGINE`` will do, shown before it does it."""

    engine: str
    steps: list[str] = field(default_factory=list)
    #: Where the engine lands; None for a system installer (Ollama).
    install_dir: Path | None = None
    needs_sudo: bool = False
    #: The record written on success (binary, python_root, ...).
    record: dict[str, Any] = field(default_factory=dict)
    #: The downloads, for the llama.cpp plan: (url, filename).
    downloads: list[tuple[str, str]] = field(default_factory=list)
    #: The commands of a from-source build, run in order.
    commands: list[list[str]] = field(default_factory=list)
    #: Environment overrides for those commands (merged over the caller's).
    env: dict[str, str] = field(default_factory=dict)
    problem: str | None = None


def engine_dir(engine: str, settings: Settings | None = None) -> Path:
    cfg = settings or load_settings()
    return cfg.data_dir / "engines" / engine


def plan_setup(
    engine: str,
    *,
    settings: Settings | None = None,
    deps: SetupDeps | None = None,
    python: str | None = None,
    extra: Sequence[str] = (),
    from_source: bool = False,
) -> SetupPlan:
    """Decide the steps for ``engine`` without touching anything.

    ``extra`` is for engines with their own installer flags (Strata).
    ``from_source`` builds llama.cpp with cmake instead of downloading it.
    """
    deps = deps or SetupDeps()
    engine = canonical_engine(engine)
    if from_source and engine != "llama_cpp":
        raise SetupError("--from-source applies to llama.cpp only.")
    target = engine_dir(engine, settings)
    if engine in _PYTHON_ENGINES:
        return _plan_python_engine(engine, target, deps, python)
    if engine == "llama_cpp":
        glibc = deps.glibc()
        if from_source:
            return _plan_llama_cpp_source(target, deps, reason=None)
        if sys.platform == "linux" and glibc is not None and glibc < PREBUILT_MIN_GLIBC:
            have = ".".join(map(str, glibc))
            need = ".".join(map(str, PREBUILT_MIN_GLIBC))
            return _plan_llama_cpp_source(
                target,
                deps,
                reason=f"this system has glibc {have}; the prebuilt builds need {need}, "
                "so llama.cpp is built from source",
            )
        return _plan_llama_cpp(target, deps)
    if engine == "strata":
        return _plan_strata(target, deps, extra)
    return _plan_ollama(deps)


def _plan_strata(target: Path, deps: SetupDeps, extra: Sequence[str]) -> SetupPlan:
    """Clone Strata and run its own setup non-interactively.

    ``extra`` is passed to ``setup.sh`` verbatim (``--family coder``,
    ``--model IQ2_XS``, ``--gpu 1``, ...); with none, Strata picks the model
    size for the machine's RAM. ``--data-dir`` keeps the 70-120 GB of weights
    under llmctl's data directory unless the caller gave one.
    """
    plan = SetupPlan(engine="strata", install_dir=target)
    if sys.platform != "linux":
        plan.problem = "llmctl drives Strata's Linux setup only; on Windows run START-HERE.bat."
        return plan
    if not deps.which("git"):
        plan.problem = "git is not installed; Strata is fetched with git clone."
        return plan
    args = list(extra)
    if "--data-dir" not in args:
        args += ["--data-dir", str(target.parent / "strata-data")]
    if (target / "setup.sh").exists():
        plan.steps.append(f"Strata is already checked out at {target}; `git pull` it")
    else:
        plan.steps.append(f"git clone {STRATA_REPO} {target}")
    plan.steps.append(
        f"{target}/setup.sh --yes --no-start {' '.join(args)}   (builds the engine and "
        "downloads the model: 70-120 GB, an hour or more; resumable)"
    )
    plan.steps.append("register the config JSON setup wrote as a model (`llmctl ls` lists it)")
    plan.record = {
        "binary": str(target / ".venv" / "bin" / "python"),
        "python_root": str(target / ".venv"),
        "server": str(target / "serve" / "server.py"),
        "setup_args": args,
    }
    return plan


def _plan_python_engine(
    engine: str, target: Path, deps: SetupDeps, python: str | None
) -> SetupPlan:
    requirement, binary = _PYTHON_ENGINES[engine]
    venv = target / ".venv"
    uv = deps.which("uv")
    plan = SetupPlan(engine=engine, install_dir=target)
    if uv:
        py = ["--python", python] if python else []
        plan.steps.append(f"uv venv {venv} {' '.join(py)}".rstrip())
        plan.steps.append(f"uv pip install --python {venv}/bin/python '{requirement}'")
    else:
        base = python or sys.executable
        plan.steps.append(f"{base} -m venv {venv}")
        plan.steps.append(f"{venv}/bin/python -m pip install '{requirement}'")
    plan.record = {
        "binary": str(venv / "bin" / binary),
        "python_root": str(venv),
        "requirement": requirement,
    }
    return plan


def _llama_cpp_assets(
    assets: list[dict[str, Any]], driver_cuda: str | None, machine: str
) -> tuple[list[dict[str, Any]], str]:
    """Pick the Ubuntu llama.cpp build for this host; returns ``(assets, flavour)``.

    CUDA builds need a driver at least as new as the toolkit they were built
    with, so the newest CUDA build not above the driver's version wins; with
    no NVIDIA driver the CPU build is used. The matching ``cudart-`` archive
    (the CUDA runtime libraries) rides along so the binary runs on a host
    without a CUDA toolkit.
    """
    arch = "arm64" if machine in ("aarch64", "arm64") else "x64"
    names = {a["name"]: a for a in assets if isinstance(a.get("name"), str)}
    if driver_cuda:
        driver = tuple(int(x) for x in driver_cuda.split("."))
        candidates = []
        for name in names:
            m = re.match(rf"^llama-(.+)-bin-ubuntu-cuda-([0-9]+\.[0-9]+)-{arch}\.tar\.gz$", name)
            if m:
                version = tuple(int(x) for x in m.group(2).split("."))
                if version <= driver:
                    candidates.append((version, name, m.group(2)))
        if candidates:
            _, name, cuda = max(candidates)
            picked = [names[name]]
            cudart = names.get(f"cudart-{name}")
            if cudart:
                picked.append(cudart)
            return picked, f"CUDA {cuda}"
    cpu = [a for n, a in names.items() if re.match(rf"^llama-(.+)-bin-ubuntu-{arch}\.tar\.gz$", n)]
    return cpu[:1], "CPU"


def _plan_llama_cpp(target: Path, deps: SetupDeps) -> SetupPlan:
    plan = SetupPlan(engine="llama_cpp", install_dir=target)
    if sys.platform != "linux":
        plan.problem = (
            "Prebuilt llama.cpp downloads are Linux-only here; install llama-server yourself."
        )
        return plan
    try:
        releases = deps.fetch_json(f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases?per_page=10")
    except Exception as exc:  # noqa: BLE001 - network
        plan.problem = f"Could not list llama.cpp releases: {exc}"
        return plan
    release = _latest_release(releases)
    if release is None:
        plan.problem = "No published llama.cpp release was found."
        return plan
    if not release.get("assets"):
        plan.problem = (
            f"llama.cpp {release.get('tag_name')} has no downloadable builds yet; try again "
            "later, or build it: llmctl setup llama.cpp --from-source"
        )
        return plan
    picked, flavour = _llama_cpp_assets(release["assets"], deps.driver_cuda(), deps.machine())
    if not picked:
        plan.problem = f"Release {release.get('tag_name')} has no Ubuntu build for this host."
        return plan
    plan.downloads = [(a["browser_download_url"], a["name"]) for a in picked]
    tag = release.get("tag_name", "?")
    plan.steps.append(f"download llama.cpp {tag} ({flavour}) from GitHub: " + ", ".join(
        name for _, name in plan.downloads
    ))
    plan.steps.append(f"unpack into {target}")
    plan.record = {
        "binary": str(target / "bin" / "llama-server"),
        "version": tag,
        "flavour": flavour,
    }
    return plan


_SRC = "src"
_STAGE = "src.new"


def _latest_release(releases: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The newest published (non-draft) release.

    Both install paths use it, so the prebuilt download and a source build
    always name the same release. Prereleases are kept on purpose: llama.cpp
    marks every ``bNNNNN`` build as a prerelease and attaches the platform
    builds only to those, while its few stable releases carry none.
    """
    return next((r for r in releases if r.get("tag_name") and not r.get("draft")), None)


def _plan_llama_cpp_source(target: Path, deps: SetupDeps, *, reason: str | None) -> SetupPlan:
    """Clone the latest llama.cpp release and build ``llama-server`` with cmake."""
    plan = SetupPlan(engine="llama_cpp", install_dir=target)
    missing = [tool for tool in ("git", "cmake") if not deps.which(tool)]
    if not (deps.which("c++") or deps.which("g++")):
        missing.append("a C++ compiler")
    if missing:
        plan.problem = (
            f"Building llama.cpp needs {', '.join(missing)}. On Ubuntu: "
            "sudo apt install git cmake build-essential, then run setup again."
        )
        return plan
    cuda = deps.driver_cuda()
    nvcc = deps.nvcc() if cuda else None
    if cuda and not nvcc:
        plan.problem = (
            "This machine has an NVIDIA driver but no CUDA toolkit (nvcc), so a GPU build "
            "is not possible. Install the toolkit (https://developer.nvidia.com/cuda-downloads) "
            "and run setup again."
        )
        return plan
    try:
        releases = deps.fetch_json(f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases?per_page=10")
    except Exception as exc:  # noqa: BLE001 - network
        plan.problem = f"Could not list llama.cpp releases: {exc}"
        return plan
    release = _latest_release(releases)
    tag = release.get("tag_name") if release else None
    if tag is None:
        plan.problem = "No published llama.cpp release was found to build."
        return plan
    # Built in a staging folder and moved over the installed one only once the
    # new binary runs, so a failed rebuild leaves the working install alone.
    # $ORIGIN RPATHs keep the binary finding its libggml after that move.
    src = target / _SRC
    stage = target / _STAGE
    build = stage / "build"
    configure = [
        "cmake", "-S", str(stage), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF",
        "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON",
    ]
    if nvcc:
        configure += ["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_COMPILER={nvcc}"]
        archs = deps.cuda_archs()
        if archs:
            configure.append(f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(archs)}")
        flavour = "source, CUDA" + (f" sm {', '.join(archs)}" if archs else "")
    else:
        flavour = "source, CPU"
    plan.commands = [
        ["git", "clone", "--depth", "1", "--branch", tag,
         f"https://github.com/{LLAMA_CPP_REPO}.git", str(stage)],
        configure,
        ["cmake", "--build", str(build), "--config", "Release",
         "-j", str(deps.cpus()), "--target", "llama-server"],
    ]
    if reason:
        plan.steps.append(reason)
    plan.steps.append(
        f"build llama.cpp {tag} ({flavour}) in {stage}, then move it to {src}; "
        "a CUDA build takes 10-30 minutes"
    )
    # gcc finds its linker on PATH. A second toolchain earlier on PATH (Homebrew's
    # binutils, say) links against that toolchain's libraries, which can need a
    # newer glibc than this system has: put the compiler's own folder first.
    cxx = deps.which("c++") or deps.which("g++")
    ld = deps.which("ld")
    if cxx and ld and Path(ld).parent != Path(cxx).parent:
        first = str(Path(cxx).parent)
        plan.env = {"PATH": f"{first}{os.pathsep}{os.environ.get('PATH', '')}"}
        plan.steps.append(
            f"put {first} first on PATH for the build, so {cxx} links with its own "
            f"linker instead of {ld}"
        )
    plan.steps.extend(" ".join(cmd) for cmd in plan.commands)
    plan.record = {
        "binary": str(src / "build" / "bin" / "llama-server"),
        "version": tag,
        "flavour": flavour,
    }
    return plan


def _plan_ollama(deps: SetupDeps) -> SetupPlan:
    plan = SetupPlan(engine="ollama", needs_sudo=True)
    found = deps.which("ollama")
    if found:
        plan.steps.append(f"ollama is already installed at {found}; nothing to do")
        plan.record = {"binary": found}
        return plan
    plan.steps.append(f"curl -fsSL {OLLAMA_INSTALL_URL} | sh   (Ollama's installer; uses sudo)")
    plan.record = {"binary": "ollama"}
    return plan


# --- doing it ------------------------------------------------------------------------


def run_setup(
    plan: SetupPlan,
    *,
    settings: Settings | None = None,
    deps: SetupDeps | None = None,
    record_path: Path | None = None,
) -> dict[str, Any]:
    """Execute ``plan``; returns the record written to ``engines.yaml``.

    Raises :class:`SetupError` with the failing step's output on any error.
    """
    deps = deps or SetupDeps()
    if plan.problem:
        raise SetupError(plan.problem)
    if plan.engine in _PYTHON_ENGINES:
        _install_python_engine(plan, deps)
    elif plan.engine == "llama_cpp":
        _install_llama_cpp(plan, deps)
    elif plan.engine == "strata":
        _install_strata(plan, deps)
    else:
        _install_ollama(plan, deps)
    record = dict(plan.record)
    record["installed_at"] = datetime.now().isoformat(timespec="seconds")
    if plan.install_dir is not None:
        record["install_dir"] = str(plan.install_dir)
    version, failure = _version_of(plan.engine, record.get("binary"), deps)
    if failure:
        # Recording a binary that cannot run would make `run` fail later with
        # a worse message; say now what is wrong and leave engines.yaml alone.
        raise SetupError(_explain_binary_failure(plan.engine, failure))
    if version:
        record["version"] = version
    path = record_path or engines_file(settings)
    record_engine(plan.engine, record, path)
    return record


def _install_python_engine(plan: SetupPlan, deps: SetupDeps) -> None:
    assert plan.install_dir is not None
    plan.install_dir.mkdir(parents=True, exist_ok=True)
    venv = plan.install_dir / ".venv"
    requirement = plan.record["requirement"]
    uv = deps.which("uv")
    if uv:
        python = plan.steps[0].split("--python ", 1)[1] if "--python " in plan.steps[0] else None
        argv = [uv, "venv", str(venv)] + (["--python", python] if python else [])
        _must(deps, argv)
        _must_stream(
            deps, [uv, "pip", "install", "--python", str(venv / "bin" / "python"), requirement]
        )
    else:
        base = plan.steps[0].split(" -m venv ", 1)[0]
        _must(deps, [base, "-m", "venv", str(venv)])
        _must_stream(deps, [str(venv / "bin" / "python"), "-m", "pip", "install", requirement])
    binary = Path(plan.record["binary"])
    if not binary.exists():
        raise SetupError(f"{plan.engine} installed but {binary} is missing; the wheel changed?")


def _install_llama_cpp(plan: SetupPlan, deps: SetupDeps) -> None:
    assert plan.install_dir is not None
    if plan.commands:
        _build_llama_cpp(plan, deps)
        return
    target = plan.install_dir
    target.mkdir(parents=True, exist_ok=True)
    downloads = target / "downloads"
    downloads.mkdir(exist_ok=True)
    for url, name in plan.downloads:
        archive = downloads / name
        deps.log(f"  downloading {name} ...")
        deps.download(url, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(target, filter="data")
    binary = _find_file(target, "llama-server")
    if binary is None:
        raise SetupError(f"llama-server was not in the archive(s) unpacked under {target}.")
    bin_dir = target / "bin"
    bin_dir.mkdir(exist_ok=True)
    final = bin_dir / "llama-server"
    if binary.resolve() != final.resolve():
        # Keep the libraries next to the binary: the release layout puts both
        # in one folder and llama-server finds libggml via RPATH ($ORIGIN).
        for item in binary.parent.iterdir():
            dest = bin_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
    final.chmod(final.stat().st_mode | 0o111)
    shutil.rmtree(downloads, ignore_errors=True)


def _build_llama_cpp(plan: SetupPlan, deps: SetupDeps) -> None:
    assert plan.install_dir is not None
    plan.install_dir.mkdir(parents=True, exist_ok=True)
    src = plan.install_dir / _SRC
    stage = plan.install_dir / _STAGE
    if stage.exists():
        # Left by an earlier failed build: `git clone` refuses a non-empty folder.
        shutil.rmtree(stage)
    env = {**os.environ, **plan.env} if plan.env else None
    for argv in plan.commands:
        _must_stream(deps, argv, env=env)
    staged = stage / "build" / "bin" / "llama-server"
    if not staged.is_file():
        raise SetupError(f"The build finished but {staged} is missing.")
    check = deps.run([str(staged), "--version"])
    if check.returncode != 0:
        detail = (check.stderr or check.stdout or "").strip()[-400:]
        raise SetupError(
            f"The new build does not run ({detail}); the installed one was left in place."
        )
    # Swap by renames: the old tree is set aside, not deleted, until the new
    # one is in place, so no moment leaves engines.yaml pointing at nothing.
    backup = plan.install_dir / "src.old"
    if backup.exists():
        shutil.rmtree(backup)
    if src.exists():
        src.rename(backup)
    try:
        stage.rename(src)
    except OSError:
        if backup.exists():
            backup.rename(src)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _install_strata(plan: SetupPlan, deps: SetupDeps) -> None:
    assert plan.install_dir is not None
    target = plan.install_dir
    git = deps.which("git") or "git"
    if (target / "setup.sh").exists():
        _must(deps, [git, "-C", str(target), "pull", "--ff-only"])
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        _must_stream(deps, [git, "clone", STRATA_REPO, str(target)])
    _must_stream(
        deps, [str(target / "setup.sh"), "--yes", "--no-start", *plan.record["setup_args"]]
    )
    python = Path(plan.record["binary"])
    if not python.exists():
        raise SetupError(f"Strata's setup finished but {python} is missing.")
    configs = sorted(target.glob("strata-*.json"), key=lambda p: p.stat().st_mtime)
    if configs:
        plan.record["config"] = str(configs[-1])


def strata_configs(install_dir: Path) -> list[Path]:
    """The server configs (one per installed model) in a Strata checkout."""
    return sorted(p for p in install_dir.glob("strata-*.json") if p.is_file())


def _install_ollama(plan: SetupPlan, deps: SetupDeps) -> None:
    if deps.which("ollama"):
        return
    rc = deps.stream(["sh", "-c", f"curl -fsSL {OLLAMA_INSTALL_URL} | sh"])
    if rc != 0:
        raise SetupError(f"Ollama's installer exited {rc}.")
    found = deps.which("ollama")
    if not found:
        raise SetupError("The installer finished but `ollama` is not on PATH; open a new shell.")
    plan.record["binary"] = found


def _find_file(root: Path, name: str) -> Path | None:
    for path in root.rglob(name):
        if path.is_file():
            return path
    return None


def _must(deps: SetupDeps, argv: Sequence[str]) -> None:
    proc = deps.run(argv)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise SetupError(f"`{' '.join(argv)}` failed (exit {proc.returncode}): {detail}")


def _must_stream(deps: SetupDeps, argv: Sequence[str], env: dict[str, str] | None = None) -> None:
    deps.log(f"  $ {' '.join(argv)}")
    rc = deps.stream(argv, env) if env else deps.stream(argv)
    if rc != 0:
        raise SetupError(f"`{' '.join(argv)}` failed (exit {rc}).")


def _explain_binary_failure(engine: str, failure: str) -> str:
    if "GLIBC_" in failure:
        return (
            f"{engine} installed, but its prebuilt binary needs a newer glibc than this "
            f"system has ({failure.strip().splitlines()[0][:160]}). The GitHub builds "
            "target Ubuntu 24.04. Build it from source instead: "
            "llmctl setup llama.cpp --from-source"
        )
    return f"{engine} installed, but its binary does not run: {failure.strip()[:400]}"


def _version_of(
    engine: str, binary: str | None, deps: SetupDeps
) -> tuple[str | None, str | None]:
    """``(version, failure)``: the version string, or why the binary did not run."""
    if not binary:
        return None, None
    argv = {
        "vllm": [binary, "--version"],
        "freetoken": [binary, "--version"],
        "llama_cpp": [binary, "--version"],
        "ollama": [binary, "--version"],
        "sglang": [binary, "-c", "import sglang; print(sglang.__version__)"],
        "strata": ["git", "-C", str(Path(binary).parent.parent.parent), "describe", "--tags"],
    }.get(engine)
    if not argv:
        return None, None
    try:
        proc = deps.run(argv)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    text = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        return None, text or f"exit {proc.returncode}"
    match = re.search(r"\d+\.\d+(?:\.\d+)?(?:[a-z0-9.+-]*)", text)
    return (match.group(0) if match else None), None


# --- status ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineStatus:
    engine: str
    #: ``llmctl`` (installed by setup), ``path`` (found on PATH / settings),
    #: ``lane`` (a router lane serves it; ``binary`` is the lane name), or ``missing``.
    source: str
    binary: str | None
    version: str | None = None


#: Seconds a version probe of an engine found on PATH may take.
VERSION_TIMEOUT_S = 5.0


def _run_with_timeout(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv from _version_of
        list(argv), capture_output=True, text=True, check=False, timeout=VERSION_TIMEOUT_S
    )


def _path_version(
    engine: str,
    binary: str,
    run: Callable[[Sequence[str]], subprocess.CompletedProcess[str]],
) -> str | None:
    """Version of an engine found on PATH, cheap enough for every ``setup``/``doctor``.

    ``vllm --version`` imports torch (about 9 s on the workstation); the
    python next to it reads the installed distribution's metadata in
    milliseconds.
    """
    python = Path(binary).with_name("python")
    if engine == "vllm" and python.exists():
        argv = [str(python), "-c", "from importlib.metadata import version; print(version('vllm'))"]
        try:
            proc = run(argv)
        except Exception:  # noqa: BLE001 - a version is optional
            return None
        lines = (proc.stdout or "").strip().splitlines()
        return lines[-1].strip() if proc.returncode == 0 and lines else None
    return _version_of(engine, binary, SetupDeps(run=run))[0]


def engine_statuses(
    settings: Settings | None = None,
    *,
    which: Callable[[str], str | None] = shutil.which,
    engines_path: Path | None = None,
    lanes: Sequence[Any] | None = None,
    run: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run_with_timeout,
) -> list[EngineStatus]:
    """Where each engine comes from today, for ``llmctl setup`` and ``doctor``.

    An engine with no install of its own counts as present when a router
    lane serves it (``lanes``; default: what the plugin lane sources load).
    An engine found on PATH gets its version from a short probe (``run``);
    a probe that fails or times out leaves the version empty.
    """
    cfg = settings or load_settings()
    recorded = load_engines(engines_path or engines_file(cfg))
    if lanes is None:
        try:
            from llmctl.services.lanes import load_lanes

            lanes = load_lanes(cfg.managed_units)
        except Exception:  # noqa: BLE001 - lanes are advisory here
            lanes = []
    rows: list[EngineStatus] = []
    for engine in ENGINE_NAMES:
        info = recorded.get(engine) or {}
        binary = info.get("binary")
        # Executable, not just present: doctor's backend check leaves a missing
        # engine to this row, so a broken install must not read as installed.
        # SGLang and Strata record a venv python, which outlives the engine.
        if binary and os.access(str(binary), os.X_OK) and _really_installed(engine, str(binary)):
            rows.append(EngineStatus(engine, "llmctl", str(binary), info.get("version")))
            continue
        default = cfg.runtime_config(engine).binary or _default_binary(engine)
        found = which(default) if default else None
        if found and not _really_installed(engine, found):
            found = None  # a python on PATH is not an engine
        lane = next((ln for ln in lanes if getattr(ln, "runtime", None) == engine), None)
        if found:
            rows.append(EngineStatus(engine, "path", found, _path_version(engine, found, run)))
        elif lane is not None:
            rows.append(EngineStatus(engine, "lane", lane.name))
        else:
            rows.append(EngineStatus(engine, "missing", None))
    return rows


def _really_installed(engine: str, binary: str) -> bool:
    """SGLang and Strata launch through a python: check the engine is behind it."""
    from llmctl.services.scheduler import _python_has_module, _strata_server_for

    if engine == "sglang":
        return _python_has_module(binary, "sglang")
    if engine == "strata":
        return _strata_server_for(binary) is not None
    return True


def _default_binary(engine: str) -> str:
    return {"ollama": "ollama", "llama_cpp": "llama-server", "strata": "", "sglang": ""}.get(
        engine, engine
    )


def recorded_binary(engine: str, engines_path: Path | None = None) -> str | None:
    """The binary ``llmctl setup`` recorded for ``engine``, if it still exists."""
    info = load_engines(engines_path).get(engine) or {}
    binary = info.get("binary")
    return str(binary) if binary and Path(str(binary)).exists() else None


def recorded_python_root(engine: str, engines_path: Path | None = None) -> str | None:
    info = load_engines(engines_path).get(engine) or {}
    root = info.get("python_root")
    return str(root) if root and Path(str(root)).exists() else None


__all__ = [
    "ENGINE_NAMES",
    "EngineStatus",
    "SetupDeps",
    "SetupError",
    "SetupPlan",
    "canonical_engine",
    "engine_statuses",
    "engines_file",
    "load_engines",
    "plan_setup",
    "recorded_binary",
    "recorded_python_root",
    "run_setup",
    "strata_configs",
]
