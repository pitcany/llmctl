"""Resolve launcher environment values at unit-generation time.

The vLLM systemd EnvironmentFile body needs three host-specific values:

* the Python prefix where the vLLM interpreter lives (for
  ``LD_LIBRARY_PATH`` and ``PATH``),
* the CUDA toolkit root,
* ``HF_HOME`` for the HuggingFace cache.

These are sourced from the caller's environment so llmctl can ship to
hosts other than the author's box.

Resolution rules
----------------
Python root (where ``bin/python`` and ``lib/`` live):
    1. ``$LLMCTL_PYTHON_ROOT`` (explicit opt-in)
    2. ``$GPU_MODELS_PYTHON_ROOT`` (back-compat with gpu-models)
    3. ``$CONDA_PREFIX`` (active conda env)
    4. ``$VIRTUAL_ENV`` (active venv)
    5. raise :class:`LauncherEnvError`

CUDA root: ``$LLMCTL_CUDA_ROOT`` (falls back to ``$GPU_MODELS_CUDA_ROOT``,
default ``/usr/local/cuda``).

HF_HOME: ``$HF_HOME`` (default ``~/.cache/huggingface``).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "LauncherEnvError",
    "build_child_env",
    "launcher_env_lines",
    "parse_environment_file",
    "resolve_cuda_root",
    "resolve_hf_home",
    "resolve_python_root",
]


_DEFAULT_CUDA_ROOT = "/usr/local/cuda"
_DEFAULT_HF_CACHE = ".cache/huggingface"


class LauncherEnvError(RuntimeError):
    """Raised when no Python interpreter root can be resolved."""


def resolve_python_root() -> str:
    """Return the Python prefix for the vLLM launcher.

    Checks ``$LLMCTL_PYTHON_ROOT``, ``$GPU_MODELS_PYTHON_ROOT``,
    ``$CONDA_PREFIX``, then ``$VIRTUAL_ENV``. Raises
    :class:`LauncherEnvError` when none is set so the caller fails fast
    with an actionable message (instead of the launcher crashing later
    when ``ld`` can't find ``libcuda.so``).
    """
    for var in (
        "LLMCTL_PYTHON_ROOT",
        "GPU_MODELS_PYTHON_ROOT",
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
    ):
        val = os.environ.get(var)
        if val:
            return val
    raise LauncherEnvError(
        "No Python environment detected. Activate a conda env "
        "(sets $CONDA_PREFIX), a venv (sets $VIRTUAL_ENV), or set "
        "$LLMCTL_PYTHON_ROOT explicitly. The vLLM launcher needs this "
        "to construct LD_LIBRARY_PATH and PATH for the systemd unit."
    )


def resolve_cuda_root() -> str:
    """Return the CUDA toolkit root. Default ``/usr/local/cuda``."""
    return os.environ.get(
        "LLMCTL_CUDA_ROOT",
        os.environ.get("GPU_MODELS_CUDA_ROOT", _DEFAULT_CUDA_ROOT),
    )


def resolve_hf_home() -> str:
    """Return ``HF_HOME``. Default ``~/.cache/huggingface``."""
    val = os.environ.get("HF_HOME")
    if val:
        return val
    return str(Path.home() / _DEFAULT_HF_CACHE)


def launcher_env_lines(python_root: str | None = None) -> list[str]:
    """Return ``LD_LIBRARY_PATH`` / ``PATH`` / ``HF_HOME`` lines.

    Resolved from the running environment at call time so the same
    rendering function can be used across hosts. The line ordering and
    value composition match ``gpu_models._launcher_env.launcher_env_lines``
    byte-for-byte — both produce the EnvironmentFile body that
    ``scripts/vllm-launcher.sh`` reads.

    ``python_root`` overrides that resolution for presets that must run
    under a different interpreter than the caller's (see
    :class:`~llmctl.integrations.vllm_env.VLLMLaunchSpec`). Because the
    launcher ``exec``s the interpreter directly, this is also what puts
    that env's ``bin/`` first on ``PATH`` — flashinfer's JIT shells out
    to ``ninja`` and fails the engine if it isn't there.
    """
    python_root = python_root or resolve_python_root()
    cuda_root = resolve_cuda_root()
    hf_home = resolve_hf_home()
    return [
        f"LD_LIBRARY_PATH={python_root}/lib:{cuda_root}/lib64",
        f"PATH={python_root}/bin:{cuda_root}/bin:/usr/local/bin:/usr/bin:/bin",
        f"HF_HOME={hf_home}",
    ]


def parse_environment_file(path: Path) -> dict[str, str]:
    """Parse a systemd ``EnvironmentFile`` body into a ``dict``.

    Restrictive format: no quoting, no ``$VAR`` interpolation, no
    ``export``. Lines split on the first ``=``; blank lines and
    ``#``-prefixed comments are skipped. systemd trims surrounding
    whitespace from unquoted values, which we mirror.
    """
    env: dict[str, str] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        env[key.strip()] = value.strip()
    return env


def build_child_env(env_file_path: Path) -> dict[str, str]:
    """Merge ``os.environ`` with a parsed EnvironmentFile body.

    EnvironmentFile values win — they're the launcher-required settings
    the systemd unit would inject via ``EnvironmentFile=``. Inheriting
    ``os.environ`` first keeps PATH / HOME / TERM available so the
    spawned subprocess behaves like a normal interactive launch.
    """
    env = dict(os.environ)
    env.update(parse_environment_file(env_file_path))
    return env
