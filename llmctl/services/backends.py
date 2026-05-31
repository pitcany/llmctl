"""Backend binary detection for the ``doctor`` command.

Resolves the executables each runtime needs so the control plane can report,
up front, which backends are actually usable on the current host. Detection is
read-only and never launches anything.
"""

from __future__ import annotations

import shutil
import sys

from llmctl.config import Settings, load_settings

#: Logical backend -> default executable name probed on PATH.
_BACKEND_BINARIES: dict[str, str] = {
    "vllm": "vllm",
    "llama_cpp": "llama-server",
    "lmstudio": "lms",
    "ollama": "ollama",
    "python": "python",
}


def detect_backends(settings: Settings | None = None) -> list[dict[str, object]]:
    """Return availability info for every known backend executable.

    The ``python`` backend always resolves to the running interpreter. Other
    binaries honor a ``runtimes.<name>.binary`` override from settings when set.
    """
    cfg = settings or load_settings()
    results: list[dict[str, object]] = []
    for backend, default_binary in _BACKEND_BINARIES.items():
        if backend == "python":
            results.append(
                {"backend": backend, "binary": "python", "path": sys.executable, "available": True}
            )
            continue
        override = cfg.runtime_config(backend).binary if backend in {"vllm", "llama_cpp"} else None
        binary = override or default_binary
        path = shutil.which(binary)
        results.append(
            {
                "backend": backend,
                "binary": binary,
                "path": path,
                "available": path is not None,
            }
        )
    return results


def missing_backends(settings: Settings | None = None) -> list[str]:
    """Return the names of backends whose binary could not be resolved."""
    return [b["backend"] for b in detect_backends(settings) if not b["available"]]
